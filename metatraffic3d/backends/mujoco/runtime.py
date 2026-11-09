"""Mirror the traffic simulation in MuJoCo: cars, people, lights, props, weather.

``MujocoCity`` owns the compiled model. ``sync`` copies the simulation state into it each
frame (mocap poses on the terrain, parked cars, lamp materials, flashing light bars, cones);
``set_weather`` changes the light, fog and road friction. The ego car is simulated, not synced.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass

import mujoco
import numpy as np

from ...assets import AssetLibrary
from ...traffic.sim import PARKED
from ...traffic.vehicles import TYPES
from ...world import World
from .scene import LIT_LAMPS, PARKED_POOL, PED_POOL, PROP_POOLS, VehicleSlots, build_mjcf, pool_sizes, ypr_quat

HIDDEN = np.array([0.0, 0.0, -60.0])
G_OFF = 4  # geom group that is never drawn (pooled bodies not in use)


# No headlight spot lights: in MuJoCo's renderer an active spot light leaves the rest of the
# scene unlit (found 2026-11-08, also with castshadow off). Night is a dim, street-lit city:
# low sun and ambient light, glowing traffic lamps, a dark sky.
@dataclass(frozen=True)
class Weather:
    name: str
    sun: tuple[float, float, float]
    ambient: tuple[float, float, float]
    fog: tuple[float, float, float] | None  # fog colour, or None for clear air
    fog_range: tuple[float, float]  # start/end in units of the model extent (100 m)
    friction: float
    headlights: bool


WEATHERS = {
    "clear": Weather("clear", (0.75, 0.73, 0.68), (0.42, 0.42, 0.45), None, (15, 60), 1.0, False),
    "overcast": Weather("overcast", (0.40, 0.41, 0.43), (0.50, 0.50, 0.52), (0.70, 0.72, 0.75), (6, 30), 1.0, False),
    "dusk": Weather("dusk", (0.75, 0.45, 0.25), (0.25, 0.22, 0.25), (0.55, 0.42, 0.40), (6, 30), 1.0, False),
    "night": Weather("night", (0.10, 0.10, 0.15), (0.20, 0.20, 0.26), (0.04, 0.05, 0.08), (2, 12), 0.95, False),
    "rain": Weather("rain", (0.30, 0.31, 0.33), (0.36, 0.37, 0.40), (0.55, 0.57, 0.60), (1.5, 10), 0.65, False),
    "fog": Weather("fog", (0.35, 0.35, 0.36), (0.55, 0.55, 0.56), (0.78, 0.79, 0.80), (0.2, 2.2), 0.85, False),
}


class MujocoCity:
    def __init__(self, world: World, out_dir: str, npc: int, slots: VehicleSlots | None = None,
                 assets: AssetLibrary | None = None) -> None:  # fmt: skip
        self.world = world
        self.assets = assets or AssetLibrary()
        self.slots = slots or VehicleSlots(npc=npc)
        self.slots.npc = npc
        self.xml_path = build_mjcf(world, out_dir, self.assets, self.slots)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        m = self.model
        mid = lambda name: m.body_mocapid[m.body(name).id]  # noqa: E731
        self.parked_pool: dict[str, list[tuple[int, int]]] = {}  # type -> [(mocap id, paint geom)]
        if self.slots.parked:
            for t, n in PARKED_POOL.items():
                self.parked_pool[t] = [(mid(f"parked_{t}_{i}"), m.geom(f"parked_{t}_{i}_paint").id
                                        if self._has_geom(f"parked_{t}_{i}_paint") else -1) for i in range(n)]  # fmt: skip
        self.spot_xy = np.array([s.pose[:2] for s in world.net.spots]) if world.net.spots else np.zeros((0, 2))
        self.parked_radius = 110.0
        self.focus = None  # (x, y) the parked-car pool is filled around; None = map centre
        self.pools: dict[str, list[int]] = {}
        self.paint: dict[int, int] = {}  # mocap id -> paint geom
        self.bars: dict[int, tuple[int, int]] = {}
        if npc:
            for t, n in pool_sizes(npc).items():
                ids = []
                for i in range(n):
                    name = f"npc_{t}_{i}"
                    k = mid(name)
                    ids.append(k)
                    try:
                        self.paint[k] = m.geom(f"{name}_paint").id
                    except KeyError:
                        pass
                    try:
                        self.bars[k] = (m.geom(f"{name}_bar_l").id, m.geom(f"{name}_bar_r").id)
                    except KeyError:
                        pass
                self.pools[t] = ids
        n_ped = min(self.slots.pedestrians, PED_POOL)
        self.ped_ids = [mid(f"ped{i}") for i in range(n_ped)]
        self.ped_paint = [m.geom(f"ped{i}_shirt").id if self._has_geom(f"ped{i}_shirt") else -1 for i in range(n_ped)]
        self.pax_ids = [mid(f"pax{i}") for i in range(self.slots.passengers)]
        self.taxi_ids = [mid(f"taxi{i}") for i in range(self.slots.taxis)]
        self.prop_pools = {a: [mid(f"prop_{a}_{i}") for i in range(n)] for a, n in PROP_POOLS.items()} if self.slots.props else {}
        self.prop_cols = {k: m.geom(f"prop_{a}_{i}_col").id for a, n in PROP_POOLS.items() for i, k in
                          enumerate(self.prop_pools.get(a, []))}  # fmt: skip
        self.lit = [(mid(f"lit{i}"), m.geom(f"lit{i}").id) for i in range(LIT_LAMPS)] if world.signals else []
        self.head_xy = np.array([sh.lamps[0][:2] for sh in world.signals]) if world.signals else np.zeros((0, 2))
        # geoms of every mocap body, to hide pooled bodies that are not in use
        self._geoms_of: dict[int, tuple[int, np.ndarray]] = {}
        for b in range(m.nbody):
            k = m.body_mocapid[b]
            if k >= 0:
                adr, num = m.body_geomadr[b], m.body_geomnum[b]
                self._geoms_of[k] = (adr, m.geom_group[adr:adr + num].copy())
        self._shown: dict[int, bool] = {}
        self.mat = {name: m.material(name).id for name in
                    ("red_on", "red_off", "yellow_on", "yellow_off", "green_on", "green_off", "bar_red", "bar_blue", "bar_dark")}  # fmt: skip
        self.ground = m.geom("ground_collision").id
        if m.nhfield:  # fill the terrain height field (rows run along +y, as in Terrain.grid)
            _, _, hgt = world.terrain.grid(8.0)
            top = m.hfield_size[0, 2]
            m.hfield_data[:] = (np.clip(hgt, 0, None) / top).ravel()
        self.sun = m.light("sun").id
        self.head = []  # headlight spot lights removed (see WEATHERS note)
        self._slot_of: dict[int, int] = {}
        self._free: dict[str, list[int]] = {t: list(ids) for t, ids in self.pools.items()}
        self._last_parked_key = None
        self._props_version = -1
        self.weather = WEATHERS["clear"]
        self._pending: list[tuple] = []
        for k in self._geoms_of:
            self._hide(k)

    def _hide(self, k: int) -> None:
        if self._shown.get(k, True):
            adr, groups = self._geoms_of[k]
            self.model.geom_group[adr:adr + len(groups)] = G_OFF
            self.data.mocap_pos[k] = HIDDEN
            self._shown[k] = False

    def _show(self, k: int) -> None:
        if not self._shown.get(k, False):
            adr, groups = self._geoms_of[k]
            self.model.geom_group[adr:adr + len(groups)] = groups
            self._shown[k] = True

    def _has_geom(self, name: str) -> bool:
        try:
            self.model.geom(name)
            return True
        except KeyError:
            return False

    # ------------------------------------------------------------------ weather
    def set_weather(self, name: str) -> Weather:
        w = WEATHERS[name]
        m = self.model
        m.light_diffuse[self.sun] = w.sun
        m.vis.headlight.ambient[:] = w.ambient
        m.vis.headlight.diffuse[:] = np.array(w.sun) * 0.45
        if w.fog is not None:
            m.vis.rgba.fog[:] = (*w.fog, 1.0)
            m.vis.rgba.haze[:] = (*w.fog, 1.0)
        m.vis.map.fogstart, m.vis.map.fogend = w.fog_range
        m.geom_friction[self.ground, 0] = w.friction
        for h in self.head:
            m.light_active[h] = w.headlights
        self.weather = w
        return w

    def scene_option(self) -> mujoco.MjvOption:
        return mujoco.MjvOption()

    # ------------------------------------------------------------------ helpers
    def _place(self, mocap: int, x: float, y: float, yaw: float, length: float = 4.5, width: float = 1.8,
               z_off: float = 0.0) -> None:  # fmt: skip
        if mocap in self._geoms_of:
            self._show(mocap)
        self._pending.append((mocap, x, y, yaw, length, width, z_off))

    def _flush(self) -> None:
        """Set all queued mocap poses at once, standing on the terrain (pitch and roll from slope)."""
        if not self._pending:
            return
        a = np.array(self._pending, float)
        self._pending = []
        k = a[:, 0].astype(int)
        x, y, yaw, L, W, zoff = a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5], a[:, 6]
        H = self.world.terrain
        if H.cfg.amplitude > 0:
            c, s = np.cos(yaw), np.sin(yaw)
            px = np.concatenate([x + c * L / 2, x - c * L / 2, x - s * W / 2, x + s * W / 2])
            py = np.concatenate([y + s * L / 2, y - s * L / 2, y + c * W / 2, y - c * W / 2])
            zf, zb, zl, zr = np.split(H.fast_height(px, py), 4)
            z = (zf + zb + zl + zr) / 4
            pitch = np.arctan2(zb - zf, L)
            roll = np.arctan2(zl - zr, W)
        else:
            z = pitch = roll = np.zeros(len(k))
        self.data.mocap_pos[k] = np.column_stack([x, y, z + zoff])
        for i in range(len(k)):
            self.data.mocap_quat[k[i]] = ypr_quat(yaw[i], pitch[i], roll[i])

    # ------------------------------------------------------------------ syncing
    def sync(self, sim, requests=(), taxi_vehicles=(), incidents=None) -> None:
        m, d = self.model, self.data
        t = sim.time
        # parked cars near the focus point
        if self.parked_pool and len(self.spot_xy):
            fx, fy = self.focus if self.focus is not None else self.spot_xy.mean(0)
            key = (round(fx / 10), round(fy / 10), int(sim.time // 5))
            if key != self._last_parked_key:
                self._last_parked_key = key
                d2 = (self.spot_xy[:, 0] - fx) ** 2 + (self.spot_xy[:, 1] - fy) ** 2
                near = np.flatnonzero((d2 < self.parked_radius**2) & (sim.spot_state == PARKED))
                near = near[np.argsort(d2[near])]
                used = {t: 0 for t in self.parked_pool}
                for sid in near:
                    ptype = sim.spot_type[sid]
                    pool = self.parked_pool.get(ptype)
                    if pool is None or used[ptype] >= len(pool):
                        continue
                    k, paint = pool[used[ptype]]
                    used[ptype] += 1
                    x, y, h = sim.net.spots[sid].pose
                    vt = TYPES[ptype]
                    self._place(k, x, y, h, vt.length, vt.width)
                    if paint >= 0:
                        m.geom_rgba[paint, :3] = sim.spot_color[sid]
                for ptype, pool in self.parked_pool.items():
                    for k, _ in pool[used[ptype]:]:
                        self._hide(k)
        # background traffic
        live = {v.id: v for v in sim.vehicles.values() if v.kind == "npc"}
        for vid in list(self._slot_of):
            if vid not in live:
                k = self._slot_of.pop(vid)
                self._hide(k)
                for t_name, ids in self.pools.items():
                    if k in ids:
                        self._free[t_name].append(k)
        flash = int(t * 3) % 2 == 0
        for vid, v in live.items():
            if vid not in self._slot_of:
                pool = self._free.get(v.vtype) or self._free.get("sedan")
                if not pool:
                    continue
                k = pool.pop()
                self._slot_of[vid] = k
                if k in self.paint:
                    m.geom_rgba[self.paint[k], :3] = v.color
            k = self._slot_of[vid]
            x, y, h = sim.pose(v)
            vt = TYPES.get(v.vtype)
            self._place(k, x, y, h, vt.length if vt else 4.5, vt.width if vt else 1.8, sim.elevation(v))
            if k in self.bars:
                left, right = self.bars[k]
                on = v.emergency
                m.geom_matid[left] = self.mat["bar_red"] if on and flash else self.mat["bar_dark"]
                m.geom_matid[right] = self.mat["bar_blue"] if on and not flash else self.mat["bar_dark"]
        for i, v in enumerate(taxi_vehicles):
            if i < len(self.taxi_ids):
                x, y, h = sim.pose(v)
                self._place(self.taxi_ids[i], x, y, h, 4.6, 1.85)
        # pedestrians (the nearest ones)
        fx, fy = self.focus if self.focus is not None else (self.spot_xy.mean(0) if len(self.spot_xy) else (0.0, 0.0))
        pos = [(sim.peds.position(p), p) for p in sim.peds.peds]
        pos.sort(key=lambda q: (q[0][0] - fx) ** 2 + (q[0][1] - fy) ** 2)
        for i, k in enumerate(self.ped_ids):
            if i < len(pos):
                (x, y, h), p = pos[i]
                on_road = p.state == 2
                self._place(k, x, y, h, 0.4, 0.4, 0.0 if on_road else 0.2)
                if self.ped_paint[i] >= 0:
                    m.geom_rgba[self.ped_paint[i], :3] = p.shirt
            else:
                self._hide(k)
        # waiting passengers (taxi env)
        waiting = [r for r in requests if getattr(r, "picked_at", None) is None][: len(self.pax_ids)]
        for i, k in enumerate(self.pax_ids):
            if i < len(waiting):
                r = waiting[i]
                lane = sim.net.lanes[sim.net.dedges[r.origin[0]].drive[0]]
                x, y, h = lane.path.pose_at(r.origin[1])
                off = sim.net.dedges[r.origin[0]].cls.half_width - lane.offset + 1.2
                self._place(k, x + math.sin(h) * off, y - math.cos(h) * off, h, 0.4, 0.4, 0.2)
            else:
                self._hide(k)
        # construction / crash props
        if incidents is not None and incidents.version != self._props_version:
            self._props_version = incidents.version
            used = {a: 0 for a in self.prop_pools}
            for asset, x, y, yaw in incidents.props:
                pool = self.prop_pools.get(asset)
                if pool is None or used[asset] >= len(pool):
                    continue
                k = pool[used[asset]]
                used[asset] += 1
                self._place(k, x, y, yaw, 0.5, 0.5)
            for asset, pool in self.prop_pools.items():
                for k in pool[used[asset]:]:
                    self._hide(k)
        # traffic lights: light the lamp of the current state on the nearest signal heads
        if self.lit:
            net = sim.net
            blink = int(t * 1.5) % 2 == 0
            d2 = (self.head_xy[:, 0] - fx) ** 2 + (self.head_xy[:, 1] - fy) ** 2
            order = np.argsort(d2)[: len(self.lit)]
            used = 0
            for hi in order:
                sh = self.world.signals[hi]
                ctl = net.nodes[sh.node].control
                if ctl.outage:
                    if not blink:
                        continue
                    idx, mat = 0, "red_on"
                else:
                    st = ctl.state(sh.de_in, t)
                    idx, mat = {"R": (0, "red_on"), "Y": (1, "yellow_on"), "G": (2, "green_on")}[st]
                k, g = self.lit[used]
                used += 1
                self._show(k)
                d.mocap_pos[k] = sh.lamps[idx]
                m.geom_matid[g] = self.mat[mat]
            for k, _ in self.lit[used:]:
                self._hide(k)
        self._flush()
        mujoco.mj_kinematics(m, d)


class VideoWriter:
    """Pipe RGB frames into ffmpeg (H.264 mp4); falls back to PNG frames without ffmpeg."""

    def __init__(self, path: str, width: int, height: int, fps: int = 20) -> None:
        self.path = path
        self.ffmpeg = shutil.which("ffmpeg")
        self.n = 0
        if self.ffmpeg:
            self.proc = subprocess.Popen(
                [self.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                 "yuv420p", "-crf", "22", path],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        else:
            self.dir = os.path.splitext(path)[0] + "_frames"
            os.makedirs(self.dir, exist_ok=True)

    def add(self, frame: np.ndarray) -> None:
        if self.ffmpeg:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            from PIL import Image

            Image.fromarray(frame).save(os.path.join(self.dir, f"{self.n:05d}.png"))
        self.n += 1

    def close(self) -> None:
        if self.ffmpeg:
            self.proc.stdin.close()
            self.proc.wait()


class ChaseCamera:
    """Smoothed chase camera behind a car, or a fixed view."""

    def __init__(self, distance: float = 28.0, elevation: float = -22.0, smooth: float = 0.15) -> None:
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.distance, self.elevation, self.smooth = distance, elevation, smooth
        self._az: float | None = None
        self._look: np.ndarray | None = None

    def follow(self, x: float, y: float, yaw: float, z: float = 0.0) -> mujoco.MjvCamera:
        target_az = math.degrees(yaw)
        if self._az is None:
            self._az, self._look = target_az, np.array([x, y, z + 1.0])
        diff = (target_az - self._az + 180) % 360 - 180
        self._az += diff * self.smooth
        self._look += (np.array([x, y, z + 1.0]) - self._look) * min(1.0, self.smooth * 3)
        self.cam.lookat[:] = self._look
        self.cam.azimuth, self.cam.elevation, self.cam.distance = self._az, self.elevation, self.distance
        return self.cam

    def fixed(self, look, distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
        self.cam.lookat[:] = look
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        return self.cam
