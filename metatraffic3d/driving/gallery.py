"""Render README images: aerial views of several seeds, one street in every weather, the highway.

    python -m metatraffic3d.driving.gallery --out docs/images
"""

from __future__ import annotations

import argparse
import math
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ..assets import AssetLibrary
from ..backends.mujoco.runtime import WEATHERS
from ..config import SimConfig
from ..world import build_world
from .env import DrivingEnv


def _label(img: np.ndarray, text: str) -> Image.Image:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, 10 + 7 * len(text), 20], fill=(0, 0, 0, 150))
    d.text((5, 4), text, fill=(255, 255, 255, 255))
    return im


def aerial(env: DrivingEnv, size=(960, 540)) -> np.ndarray:
    w = env.world
    r = mujoco.Renderer(env.model, size[1], size[0], max_geom=20000)
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    cam = mujoco.MjvCamera()
    cen = max(w.buildings, key=lambda b: b.height).footprint.centroid
    cam.lookat[:] = [cen.x, cen.y, float(w.terrain.height(cen.x, cen.y))]
    cam.distance, cam.azimuth, cam.elevation = 900.0, -55.0, -32.0
    r.update_scene(env.data, cam)
    img = r.render()
    r.close()
    return img


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/images")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 5, 7, 13])
    ap.add_argument("--street-seed", type=int, default=7)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    tiles = []
    for seed in a.seeds:
        cfg = SimConfig()
        cfg.map.seed = seed
        lib = AssetLibrary("auto")
        world = build_world(cfg, assets=lib)
        env = DrivingEnv(cfg, world, assets=lib, weather="clear", image_size=(480, 270), seed=seed, shadows=True)
        env.reset(seed=1, options={"warmup": 30})
        p = world.profile
        text = f"seed {seed}: {p.terrain}, {'coast ' if p.coast else ''}{'river ' if p.river else ''}" \
               f"{'highway' if world.net.highway else ''}".strip()  # fmt: skip
        tiles.append(_label(aerial(env), text))
        if seed == a.street_seed:  # one street, every weather, through the main camera
            row = []
            for wname in WEATHERS:
                env.reset(seed=3, options={"weather": wname, "warmup": 30})
                for _ in range(40):
                    env.step(env.expert_action())
                img = env.render_cameras(["front_main"], size=(480, 270))["front_main"]
                row.append(_label(img, wname))
            strip = Image.new("RGB", (480 * 3, 270 * 2))
            for i, im in enumerate(row):
                strip.paste(im, ((i % 3) * 480, (i // 3) * 270))
            strip.save(os.path.join(a.out, "weather.jpg"), quality=88)
        if world.net.highway:
            e = world.net.edges[world.net.highway[0]]
            r = mujoco.Renderer(env.model, 540, 960, max_geom=20000)
            cam = mujoco.MjvCamera()
            x, y, h = e.center.pose_at(e.center.length * 0.3)
            cam.lookat[:] = [x, y, float(world.terrain.height(x, y)) + 6]
            cam.distance, cam.azimuth, cam.elevation = 140.0, math.degrees(h) + 60, -18.0
            r.update_scene(env.data, cam)
            _label(r.render(), f"seed {seed}: elevated highway (experimental)").save(os.path.join(a.out, "highway.jpg"), quality=88)
            r.close()
        env.close()
    sheet = Image.new("RGB", (960 * 2, 540 * ((len(tiles) + 1) // 2)))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 2) * 960, (i // 2) * 540))
    sheet.save(os.path.join(a.out, "cities.jpg"), quality=88)
    print("wrote", os.listdir(a.out))


if __name__ == "__main__":
    main()
