"""Record a MuJoCo video of the taxi fleet driving through the 3D city.

    python -m metatraffic3d video --seconds 120 --out demo.mp4 [--checkpoint runs/x/policy.pt]
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ...assets import AssetLibrary
from ...config import SimConfig
from ...marl.baselines import POLICIES
from ...marl.taxi_env import IDLE, NONE, REPOSITION, TO_DROPOFF, TO_PICKUP, TaxiEnv
from ...world import build_world
from .runtime import ChaseCamera, MujocoCity, VideoWriter
from .scene import VehicleSlots

STATUS = {IDLE: "waiting for a fare", TO_PICKUP: "driving to pick-up", TO_DROPOFF: "carrying a passenger",
          REPOSITION: "looking for a kerb spot"}  # fmt: skip


def _overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    h = 16 + 22 * len(lines)
    d.rectangle([10, 10, 430, 10 + h], fill=(15, 18, 24, 170))
    for i, line in enumerate(lines):
        d.text((22, 20 + 22 * i), line, fill=(255, 255, 255, 255))
    return np.asarray(img)


def record(out: str, seconds: float = 90.0, cfg: SimConfig | None = None, policy: str = "greedy_feasible",
           weights=None, follow: int = 0, width: int = 1280, height: int = 720, fps: int = 20,
           dt: float = 0.1, warmup: float = 120.0, assets: AssetLibrary | None = None,
           seed: int = 1, intro: float = 4.0) -> dict:  # fmt: skip
    cfg = cfg or SimConfig()
    cfg.traffic.dt = dt
    cfg.taxi.episode_seconds = warmup + seconds + 60.0
    assets = assets or AssetLibrary("auto")
    world = build_world(cfg, assets=assets)
    env = TaxiEnv(cfg, world, seed=seed)
    env.sim.height = world.terrain.height
    slots = VehicleSlots(taxis=cfg.taxi.num_taxis, passengers=30, pedestrians=cfg.traffic.num_pedestrians,
                         props=True, spot_types=env.sim.spot_type)  # fmt: skip
    city = MujocoCity(world, os.path.splitext(out)[0] + "_mjcf", npc=cfg.traffic.num_npc + 20, slots=slots,
                      assets=assets)  # fmt: skip
    renderer = mujoco.Renderer(city.model, height, width, max_geom=60000)
    opt = mujoco.MjvOption()
    cam = ChaseCamera()
    writer = VideoWriter(out, width, height, fps)
    rng = np.random.default_rng(seed)
    frames_per_tick = max(1, round(1.0 / (fps * dt))) if fps * dt < 1 else 1
    state = {"tick": 0, "t0": None}
    xmin, ymin, xmax, ymax = world.bounds
    tall = max(world.buildings, key=lambda b: b.height).footprint.centroid

    def on_tick(e: TaxiEnv) -> None:
        if e.sim.time < warmup:
            return
        state["tick"] += 1
        if state["tick"] % frames_per_tick:
            return
        if state["t0"] is None:
            state["t0"] = e.sim.time
        tt = e.sim.time - state["t0"]
        taxi = e.taxis[follow]
        x, y, h = e.sim.pose(taxi.veh)
        city.focus = (x, y)
        city.sync(e.sim, e.requests.values(), [t.veh for t in e.taxis])
        z = float(world.terrain.height(x, y))
        if tt < intro:  # opening shot: sweep over downtown toward the taxi
            f = tt / intro
            camera = cam.fixed([tall.x * (1 - f) + x * f, tall.y * (1 - f) + y * f, z], 900 * (1 - f) + 60 * f,
                               -60 + 80 * f, -45 + 20 * f)  # fmt: skip
        else:
            camera = cam.follow(x, y, h, z)
        renderer.update_scene(city.data, camera, opt)
        frame = renderer.render()
        delivered = sum(t.delivered for t in e.taxis)
        waiting = sum(1 for r in e.requests.values() if r.picked_at is None)
        lines = [f"MetaTraffic 3D  |  MuJoCo  |  t = {e.sim.time - warmup:5.0f} s",
                 f"taxi {follow}: {STATUS.get(taxi.status, '')}, {taxi.veh.v * 3.6:4.0f} km/h",
                 f"fleet of {e.n}: {delivered} delivered, {waiting} passengers waiting",
                 f"{sum(1 for v in e.sim.vehicles.values() if v.kind == 'npc')} cars in traffic, "
                 f"{int((e.sim.spot_state == 1).sum())} parked"]  # fmt: skip
        writer.add(_overlay(frame, lines))

    env.on_tick = on_tick
    obs, _ = env.reset(seed=seed)
    act = POLICIES.get(policy)
    while env.agents and env.sim.time < warmup + seconds:
        actions = {}
        for a, o in obs.items():
            if o["kind"] == NONE:
                actions[a] = 0
            elif weights is not None:
                from ...marl.mappo import np_policy

                actions[a] = np_policy(weights, o, rng, greedy=True)[0]
            else:
                actions[a] = act(o, rng)
        obs, *_ = env.step(actions)
    writer.close()
    return {"frames": writer.n, "path": out, "delivered": sum(t.delivered for t in env.taxis),
            "xml": city.xml_path}  # fmt: skip
