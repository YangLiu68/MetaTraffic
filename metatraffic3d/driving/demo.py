"""Record a multi-camera driving video: the eight rig cameras around a chase view, with a HUD.

    python -m metatraffic3d drive-demo --seed 4 --seconds 60 --weather clear --out demo.mp4
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image, ImageDraw

from ..assets import AssetLibrary
from ..backends.mujoco.runtime import ChaseCamera, VideoWriter
from ..config import SimConfig
from ..world import build_world
from .cameras import RIG, fovy
from .env import DrivingEnv

LAYOUT = (  # 3 x 3 mosaic, Tesla-style
    ("pillar_left", "front_main", "pillar_right"),
    ("repeater_left", "CHASE", "repeater_right"),
    ("front_wide", "rear", "front_narrow"),
)
LABEL = {"front_main": "MAIN", "front_wide": "WIDE", "front_narrow": "NARROW", "pillar_left": "LEFT PILLAR",
         "pillar_right": "RIGHT PILLAR", "repeater_left": "LEFT REPEATER", "repeater_right": "RIGHT REPEATER",
         "rear": "REAR", "CHASE": "CHASE (not an input)"}  # fmt: skip


def _bar(d: ImageDraw.ImageDraw, x: int, y: int, w: int, value: float, color, label: str) -> None:
    d.rectangle([x, y, x + w, y + 10], outline=(200, 200, 200, 255))
    d.rectangle([x, y, x + int(w * max(0.0, min(1.0, value))), y + 10], fill=color)
    d.text((x + w + 6, y - 2), label, fill=(255, 255, 255, 255))


def record(out: str, seed: int = 4, seconds: float = 60.0, weather: str | None = "clear", tile: tuple[int, int] = (480, 270),
           cfg: SimConfig | None = None, episode_seed: int = 1, terrain: str | None = None, fps: int = 10,
           assets: AssetLibrary | None = None, log=print) -> dict:  # fmt: skip
    cfg = cfg or SimConfig()
    cfg.map.seed = seed
    if terrain:
        cfg.profile.terrain = terrain
    assets = assets or AssetLibrary("auto")
    t0 = time.time()
    world = build_world(cfg, assets=assets)
    env = DrivingEnv(cfg, world, cameras=("front_main",), image_size=tile, weather=weather, assets=assets,
                     max_seconds=seconds + 5, seed=episode_seed, shadows=True)  # fmt: skip
    log(f"world + env {time.time() - t0:.0f} s; profile {world.profile.as_dict()}")
    m = env.model
    tw, th = tile
    for name, spec in RIG.items():
        m.cam_fovy[m.camera(f"ego_{name}").id] = fovy(spec.hfov, tw / th)
    renderer = env.renderer
    chase = ChaseCamera(distance=16.0, elevation=-14.0, smooth=0.2)
    W, H = tw * 3, th * 3
    writer = VideoWriter(out, W, H, fps)
    obs, info = env.reset(seed=episode_seed)
    reasons, frames = None, 0
    steps = int(seconds / env.dt)
    every = max(1, round(1.0 / (fps * env.dt)))
    action = env.expert_action()
    for k in range(steps):
        action = env.expert_action()
        obs, r, term, trunc, info = env.step(action)
        if k % every == 0:
            st = env.car.state(env.dt)
            canvas = Image.new("RGB", (W, H))
            for row, names in enumerate(LAYOUT):
                for col, name in enumerate(names):
                    if name == "CHASE":
                        cam = chase.follow(st.x, st.y, st.yaw, st.z)
                        renderer.update_scene(env.data, cam)
                    else:
                        renderer.update_scene(env.data, camera=f"ego_{name}")
                    img = Image.fromarray(renderer.render())
                    dd = ImageDraw.Draw(img, "RGBA")
                    dd.rectangle([0, 0, 8 + 7 * len(LABEL[name]), 16], fill=(0, 0, 0, 140))
                    dd.text((4, 2), LABEL[name], fill=(255, 255, 255, 255))
                    canvas.paste(img, (col * tw, row * th))
            d = ImageDraw.Draw(canvas, "RGBA")
            x0, y0 = tw + 8, th + th - 92
            d.rectangle([x0 - 4, y0 - 4, x0 + tw - 12, y0 + 86], fill=(10, 12, 16, 170))
            cmd, dist = env.tracker.command()
            inf = [e["kind"] for e in info["privileged"]["infractions"]][-2:]
            d.text((x0, y0), f"{st.speed * 3.6:5.1f} km/h  gear {st.gear}  {st.rpm:4.0f} rpm  t={env.sim.time - env.t0:5.1f}s",
                   fill=(255, 255, 255, 255))  # fmt: skip
            d.text((x0, y0 + 16), f"route: {cmd} in {dist:4.0f} m | {env._prev_remaining:5.0f} m to go | {env.weather}",
                   fill=(255, 255, 255, 255))  # fmt: skip
            ctl = action["control"]
            _bar(d, x0, y0 + 36, 90, (ctl[0] + 1) / 2, (90, 160, 255, 255), "steer")
            _bar(d, x0 + 160, y0 + 36, 90, ctl[1], (80, 220, 120, 255), "throttle")
            _bar(d, x0 + 320, y0 + 36, 60, ctl[2], (240, 80, 70, 255), "brake")
            d.text((x0, y0 + 56), "infractions: " + (", ".join(inf) if inf else "none") +
                   f" | incidents active: {len(env.incidents.active) if env.incidents else 0}", fill=(255, 220, 160, 255))  # fmt: skip
            writer.add(np.asarray(canvas))
            frames += 1
        if term or trunc:
            reasons = info["reason"]
            break
    writer.close()
    out_info = {"frames": frames, "reason": reasons, "seconds": round(env.sim.time - env.t0, 1),
                "infractions": [e["kind"] for e in info["privileged"]["infractions"]], "weather": env.weather,
                "profile": world.profile.as_dict(), "wall_s": round(time.time() - t0, 1), "path": out}  # fmt: skip
    log(str(out_info))
    return out_info
