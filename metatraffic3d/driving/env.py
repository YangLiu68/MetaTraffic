"""End-to-end, camera-only driving environment (Gymnasium API).

Observation (what the policy sees):
    images   uint8 (n_cameras, H, W, 3) from the Tesla-like rig (default 4 of the 8 cameras)
    speed    m/s, rpm / 1000, gear one-hot (R N 1-5), current steering angle
    command  one-hot: follow / left / right / straight / uturn / arrive, plus distance to it
No lane positions, maps or object lists: perception is up to the policy.

Action (direct control, no speed set-points):
    {"control": Box([-1, 0, 0], [1, 1, 1]) = steer, throttle, brake,  "gear": Discrete(7) = R N 1 2 3 4 5}

Each step = 0.1 s: traffic, pedestrians and incidents advance, then 50 MuJoCo physics steps of
the ego car (drivetrain, tyres, slopes, weather friction). Episodes end on arrival, a
collision, leaving the road for 2 s, getting stuck for 60 s, or the time limit.
``info["privileged"]`` carries the expert's action and the infractions for training/eval.
"""

from __future__ import annotations

import math
import tempfile

import mujoco
import numpy as np
import shapely

from ..assets import AssetLibrary
from ..backends.mujoco.runtime import WEATHERS, MujocoCity
from ..backends.mujoco.scene import VehicleSlots
from ..backends.mujoco.vehicle import GEARS, EgoCar
from ..config import SimConfig
from ..traffic.demand import demand_weights
from ..traffic.events import IncidentManager
from ..traffic.sim import TrafficSim
from ..world import World, build_world
from .cameras import RIG, cameras_xml, fovy
from .ego import EgoTracker
from .expert import ExpertDriver

COMMANDS = ("follow", "left", "right", "straight", "uturn", "arrive")
DEFAULT_CAMERAS = ("front_main", "pillar_left", "pillar_right", "rear")
WEATHER_P = {"clear": 0.4, "overcast": 0.15, "dusk": 0.12, "night": 0.13, "rain": 0.12, "fog": 0.08}


class DrivingEnv:
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: SimConfig | None = None, world: World | None = None, cameras=DEFAULT_CAMERAS,
                 image_size: tuple[int, int] = (160, 96), weather: str | None = None, incidents: bool = True,
                 max_seconds: float = 240.0, route_m: tuple[float, float] = (500.0, 1400.0),
                 out_dir: str | None = None, assets: AssetLibrary | None = None, seed: int = 0,
                 shadows: bool = False) -> None:  # fmt: skip
        self.cfg = cfg or SimConfig()
        self.world = world or build_world(self.cfg)
        self.cameras = tuple(cameras)
        self.image_size = image_size
        self.fixed_weather = weather
        self.max_seconds = max_seconds
        self.shadows = shadows
        self.route_m = route_m
        self.dt = 0.1
        self.substeps = 50
        self.sim = TrafficSim(self.world.net, self.cfg, demand_weights(self.world), seed=seed)
        self.sim.height = self.world.terrain.height
        self.incidents = IncidentManager(self.sim, seed) if incidents else None
        self.sim.incidents = self.incidents
        self.tracker = EgoTracker(self.sim)
        road = self.world.roads
        shapely.prepare(road)
        self.tracker.road = road
        out_dir = out_dir or tempfile.mkdtemp(prefix="metatraffic3d_")
        slots = VehicleSlots(parked=True, pedestrians=self.cfg.traffic.num_pedestrians, props=True, ego=True,
                             ego_cameras=cameras_xml("ego"), spot_types=self.sim.spot_type)  # fmt: skip
        self.city = MujocoCity(self.world, out_dir, npc=self.cfg.traffic.num_npc + 10, slots=slots, assets=assets)
        self.model, self.data = self.city.model, self.city.data
        self.car = EgoCar(self.model, self.data)
        self.expert = ExpertDriver(self.tracker)
        w, h = image_size
        self.renderer = mujoco.Renderer(self.model, h, w, max_geom=60000)
        for name, spec in RIG.items():
            self.model.cam_fovy[self.model.camera(f"ego_{name}").id] = fovy(spec.hfov, w / h)
        self.rng = np.random.default_rng(seed)
        self._geom_kind = self._classify_geoms()
        self.episode = 0

    # ------------------------------------------------------------------ spaces
    @property
    def observation_space(self):
        from gymnasium import spaces

        w, h = self.image_size
        return spaces.Dict({
            "images": spaces.Box(0, 255, (len(self.cameras), h, w, 3), np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (12,), np.float32),
            "command": spaces.Box(0, 1, (len(COMMANDS) + 1,), np.float32),
        })  # fmt: skip

    @property
    def action_space(self):
        from gymnasium import spaces

        return spaces.Dict({"control": spaces.Box(np.array([-1, 0, 0], np.float32), np.array([1, 1, 1], np.float32)),
                            "gear": spaces.Discrete(len(GEARS))})  # fmt: skip

    def _classify_geoms(self) -> dict[int, str]:
        kinds = {}
        m = self.model
        for i in range(m.ngeom):
            n = m.geom(i).name
            if n.startswith("npc_"):
                kinds[i] = "vehicle"
            elif n.startswith("ped"):
                kinds[i] = "pedestrian"
            elif n.startswith("parked_"):
                kinds[i] = "parked_car"
            elif n.startswith("prop_"):
                kinds[i] = "construction"
        return kinds

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        seed = int(self.rng.integers(2**31))
        sim = self.sim
        sim.reset(seed=seed)
        if self.incidents is not None:
            self.incidents.reset(seed)
        weather = (options or {}).get("weather") or self.fixed_weather or str(
            self.rng.choice(list(WEATHER_P), p=list(WEATHER_P.values())))  # fmt: skip
        sim.set_weather(weather)
        self.city.set_weather(weather)
        flags = self.renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_FOG] = WEATHERS[weather].fog is not None
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = weather not in ("night",)
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.shadows and weather in ("clear", "dusk")
        for _ in range(int((options or {}).get("warmup", 20))):  # let traffic settle a little
            sim.step(1.0)
        lane_id, s, dest = self._pick_route()
        v = self.tracker.spawn(lane_id, s, dest)
        x, y, h = self.tracker.pose_on_lane(lane_id, s)
        mujoco.mj_resetData(self.model, self.data)
        self.city.sync(sim, incidents=self.incidents)
        z = float(self.world.terrain.height(x, y)) + self.world.net.lane_z(lane_id, max(s - 2.3, 0.0))
        self.car.place(x, y, z, h, speed=0.0)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100):  # settle the suspension
            self.car.apply(0.002)
            mujoco.mj_step(self.model, self.data)
        self.expert.reset()
        self._steps = 0
        self._expert_cache = None
        self.t0 = sim.time
        self.offroad_time = 0.0
        self.weather = weather
        self.episode += 1
        self._prev_remaining = self._remaining()
        self.ego_vehicle = v
        return self._observe(), self._info(None)

    def _pick_route(self):
        sim, net = self.sim, self.world.net
        for _ in range(200):
            de = net.dedges[int(self.rng.choice(sim.router.ids))]
            lane = net.lanes[de.drive[0]]
            if lane.length < 45 or de.id in sim.closed_dedges or lane.id in sim.static_blocks:
                continue
            s = float(self.rng.uniform(12.0, lane.length - 20.0))
            if not sim._gap_for_pullout(lane.id, s - 2.3, 4.6):
                continue
            x, y, _ = lane.path.pose_at(s)
            for _ in range(20):
                dest = sim.random_kerb_point(self.rng)
                d = math.dist((x, y), net.lanes[net.dedges[dest[0]].drive[0]].path.point_at(dest[1]))
                if self.route_m[0] * 0.6 <= d <= self.route_m[1] and dest[0] != de.id:
                    return lane.id, s, dest
        raise RuntimeError("could not find an ego route")

    def _remaining(self) -> float:
        """Route length left (metres) along the planned streets."""
        v, net, r = self.tracker.veh, self.world.net, self.sim.router
        if v is None or v.dest is None:
            return 0.0
        if v.on == 0:
            de, s = net.lanes[v.elem].dedge, v.s
        else:
            de, s = net.lanes[net.connectors[v.elem].to_lane].dedge, 0.0
        dest_de, dest_s = v.dest
        if de == dest_de and s <= dest_s:
            return dest_s - s
        ctg = self.sim.ctg(v)
        path = r.path(de, ctg, dest_de, max_len=200)
        if path[-1] != dest_de:
            return 1e4
        total = r.length[r.index[de]] - s + sum(r.length[r.index[d]] for d in path[1:-1]) + dest_s
        return float(total)

    # ------------------------------------------------------------------ step
    def expert_action(self) -> dict:
        """The expert's action for the current state (computed once per step, then cached)."""
        if self._expert_cache is not None and self._expert_cache[0] == self._steps:
            return self._expert_cache[1]
        st = self.car.state(self.dt)
        steer, throttle, brake, gear = self.expert.act(st, self.dt)
        act = {"control": np.array([steer, throttle, brake], np.float32), "gear": gear}
        self._expert_cache = (self._steps, act)
        return act

    def step(self, action: dict):
        sim = self.sim
        expert = self.expert_action()  # label for imitation, computed before acting
        self._steps += 1
        sim.step(self.dt)
        st0 = self.car.state(self.dt)
        self.city.focus = (st0.x, st0.y)
        self.city.sync(sim, incidents=self.incidents)
        steer, throttle, brake = np.asarray(action["control"], float)
        self.car.act(steer, throttle, brake, int(action["gear"]))
        hits = set()
        for _ in range(self.substeps):
            self.car.apply(0.002)
            mujoco.mj_step(self.model, self.data)
            for g in self.car.collisions():
                if g in self._geom_kind:
                    hits.add(self._geom_kind[g])
        st = self.car.state(self.dt)
        z_rel = st.z - float(self.world.terrain.height(st.x, st.y))
        self.tracker.update(st.x, st.y, st.yaw, st.speed, self.dt, z_rel)
        tr = self.tracker
        for kind in hits:
            tr.inf.add("collision", sim.time, with_=kind)
        remaining = self._remaining()
        progress = float(np.clip(self._prev_remaining - remaining, -5.0, 5.0))
        self._prev_remaining = remaining
        self.offroad_time = self.offroad_time + self.dt if getattr(tr, "offroad", False) else 0.0
        reward = 0.1 * progress - 0.001 * abs(st.accel) - 0.002 * abs(st.yaw_rate)
        penalties = {"collision": 10.0, "red_light": 3.0, "stop_sign": 1.5, "wrong_way": 0.5, "off_road": 0.3,
                     "speeding": 0.05, "route_deviation": 0.5}  # fmt: skip
        recent = [e for e in tr.inf.events if e["t"] >= sim.time - self.dt + 1e-6]
        reward -= sum(penalties.get(e["kind"], 0.0) for e in recent)
        terminated, truncated, reason = False, False, None
        if tr.ctrl.arrived:
            terminated, reason = True, "arrived"
            reward += 10.0
        elif hits & {"vehicle", "pedestrian", "parked_car"}:
            terminated, reason = True, "collision"
        elif self.offroad_time >= 2.0:
            terminated, reason = True, "off_road"
        elif tr.veh.still > 60.0:
            truncated, reason = True, "stuck"
        elif sim.time - self.t0 >= self.max_seconds:
            truncated, reason = True, "timeout"
        info = self._info(expert)
        info["reason"] = reason
        return self._observe(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ observation
    def render_cameras(self, names=None, size: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
        names = names or self.cameras
        r = self.renderer
        if size is not None and size != self.image_size:
            r = mujoco.Renderer(self.model, size[1], size[0], max_geom=60000)
            r.scene.flags[:] = self.renderer.scene.flags
        out = {}
        for n in names:
            r.update_scene(self.data, camera=f"ego_{n}")
            out[n] = r.render().copy()
        return out

    def _observe(self) -> dict:
        st = self.car.state(self.dt)
        imgs = self.render_cameras()
        state = np.zeros(12, np.float32)
        state[0] = st.speed / 15.0
        state[1] = st.rpm / 6000.0
        state[2 + GEARS.index(st.gear)] = 1.0
        state[9] = st.steer / 0.6
        state[10] = math.sin(st.pitch)
        state[11] = float(st.shifting)
        cmd, dist = self.tracker.command()
        command = np.zeros(len(COMMANDS) + 1, np.float32)
        command[COMMANDS.index(cmd)] = 1.0
        command[-1] = min(dist / 100.0, 1.0)
        return {"images": np.stack([imgs[n] for n in self.cameras]), "state": state, "command": command}

    def _info(self, expert) -> dict:
        st = self.car.state(self.dt)
        return {"privileged": {"expert": expert, "infractions": list(self.tracker.inf.events),
                               "remaining_m": self._prev_remaining if hasattr(self, "_prev_remaining") else None,
                               "pose": (st.x, st.y, st.yaw), "speed": st.speed, "weather": getattr(self, "weather", None),
                               "time": self.sim.time}}  # fmt: skip

    def close(self) -> None:
        self.renderer.close()
