"""MetaTraffic 3D command line.

    python -m metatraffic3d build      --seed 4 [--terrain hilly] [--png map.png]
    python -m metatraffic3d mjcf       --seed 4 --out scenes/seed4        # open with: python -m mujoco.viewer --mjcf scenes/seed4/scene.xml
    python -m metatraffic3d usd        --seed 4 --out city.usd            # for Isaac Sim
    python -m metatraffic3d drive-demo --seed 4 --weather clear --seconds 60 --out demo.mp4
    python -m metatraffic3d collect | train-bc | evaluate | rl ...     # end-to-end driving pipeline
    python -m metatraffic3d train-taxi | eval-taxi ...                 # multi-agent taxi fleet
    python -m metatraffic3d assets fetch | credits                     # Poly Pizza models
"""

from __future__ import annotations

import argparse
import json
import sys


def _cfg(a):
    from .config import SimConfig

    cfg = SimConfig()
    cfg.map.seed = a.seed
    if getattr(a, "terrain", None):
        cfg.profile.terrain = a.terrain
    if getattr(a, "width", None):
        cfg.map.width, cfg.map.height = a.width, a.height
    return cfg


def _map_png(world, path: str, scale: float = 0.35) -> None:
    from PIL import Image, ImageDraw

    from .world import COLORS

    xmin, ymin, xmax, ymax = world.bounds
    w, h = int((xmax - xmin) * scale), int((ymax - ymin) * scale)
    img = Image.new("RGB", (w, h), tuple(int(c * 255) for c in COLORS["ground"][:3]))
    d = ImageDraw.Draw(img)

    def px(pts):
        return [((x - xmin) * scale, (ymax - y) * scale) for x, y in pts]

    zone_rgb = {"cbd": (90, 110, 150), "commercial": (200, 150, 110), "residential": (225, 205, 175), "industrial": (150, 150, 150)}
    for g, col in ((world.sea, (70, 120, 180)), (world.river, (70, 120, 180))):
        if g is not None:
            for p in getattr(g, "geoms", [g]):
                if p.geom_type == "Polygon":
                    d.polygon(px(p.exterior.coords), fill=col)
    for p in world.parks:
        d.polygon(px(p.exterior.coords), fill=(120, 170, 100))
    for b in world.buildings:
        d.polygon(px(b.footprint.exterior.coords), fill=zone_rgb.get(b.zone, (200, 200, 200)))
    for e in world.net.edges:
        d.line(px(e.center.pts), fill=(60, 60, 64), width=max(1, int(e.cls.half_width * 2 * scale)))
    for n in world.net.nodes:
        ctl = n.control
        if ctl is not None and ctl.kind != "none":
            x, y = px([n.pos])[0]
            col = {"signal": (230, 60, 50), "all_stop": (240, 200, 40), "two_stop": (250, 140, 30)}[ctl.kind]
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)
    img.save(path)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    passthrough = {
        "collect": "metatraffic3d.driving.collect", "train-bc": "metatraffic3d.driving.train_bc",
        "evaluate": "metatraffic3d.driving.evaluate", "rl": "metatraffic3d.driving.rl",
        "train-taxi": "metatraffic3d.marl.train", "eval-taxi": "metatraffic3d.marl.evaluate",
        "assets": "metatraffic3d.assets.polypizza",
    }  # fmt: skip
    if cmd in passthrough:
        import importlib

        mod = importlib.import_module(passthrough[cmd])
        return mod.main(rest) or 0
    ap = argparse.ArgumentParser(prog=f"metatraffic3d {cmd}")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--terrain", choices=["flat", "rolling", "hilly", "steep"], default=None)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--height", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--weather", default="clear")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--policy", default=None, help="drive-demo: record an agent checkpoint instead of the autopilot, on evaluation route --episode")
    a = ap.parse_args(rest)
    from .assets import AssetLibrary
    from .world import build_world

    if cmd == "build":
        w = build_world(_cfg(a), with_meshes=a.png is None or True)
        s = w.summary()
        s["network"] = {k: v for k, v in s["network"].items() if k != "node_degrees"}
        print(json.dumps(s, indent=1, default=str))
        if a.png:
            _map_png(w, a.png)
            print("map ->", a.png)
    elif cmd == "mjcf":
        from .backends.mujoco.scene import VehicleSlots, build_mjcf

        assets = AssetLibrary("auto")
        w = build_world(_cfg(a), assets=assets)
        path = build_mjcf(w, a.out or f"scenes/seed{a.seed}", assets, VehicleSlots(npc=0, parked=False, props=False))
        print("MJCF ->", path)
    elif cmd == "usd":
        from .backends.isaac.usd_export import export_usd

        assets = AssetLibrary("auto")
        w = build_world(_cfg(a), assets=assets)
        print(json.dumps(export_usd(w, a.out or f"city_seed{a.seed}.usd", assets), indent=1, default=str))
    elif cmd == "drive-demo":
        from .driving.demo import record, record_policy

        if a.policy:
            record_policy(a.out or f"policy_seed{a.seed}_ep{a.episode}.mp4", a.policy, seed=a.seed,
                          episode=a.episode, seconds=a.seconds)
            return 0
        record(a.out or f"demo_seed{a.seed}.mp4", seed=a.seed, weather=a.weather, seconds=a.seconds,
               episode_seed=a.episode, terrain=a.terrain)  # fmt: skip
    else:
        print(f"unknown command {cmd!r}\n{__doc__}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
