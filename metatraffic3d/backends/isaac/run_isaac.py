"""Drive an exported MetaTraffic city inside NVIDIA Isaac Sim and record the ego camera rig.

    # with Isaac Sim's Python (needs shapely + pillow: ./python.sh -m pip install shapely pillow)
    ./python.sh metatraffic3d/backends/isaac/run_isaac.py --usd city.usd --seed 4 --seconds 60 \
        --headless --out frames/ --cameras front_main,pillar_left,pillar_right,rear

UNTESTED: written against the public Isaac Sim 4.x / 5.x Python API (``isaacsim.SimulationApp``,
``omni.usd``, ``omni.replicator.core``) on a Mac that cannot run Isaac Sim. The parts that do not
need Isaac (rebuilding the road network from the stage, the vehicle / pedestrian / lamp mirror,
the kinematic ego) live in ``mirror.py`` and are tested with usd-core; this file only adds the
app, stage loading, render products and the frame loop. Expect small API fixes on first run.

What happens:
1. ``SimulationApp`` starts (headless or with a window), the USD stage is opened.
2. The road network is rebuilt from the ``SimConfig`` stored in the stage (no meshes), and
   ``TrafficSim`` + ``IncidentManager`` run on it (``--seed`` seeds the traffic).
3. ``StageMirror`` creates pools of instanceable prims in the session layer (the city file is
   not modified) and writes poses, colours and lamp visibility each 0.1 s step.
4. The ego is kinematic: ``KinematicEgo`` rides a background car and carries the camera rig
   (PhysX vehicle dynamics are future work). A chase camera follows it for the viewport.
5. Replicator renders the chosen rig cameras every ``--every`` steps; annotator output is
   saved under ``--out/<camera>/`` (rgb as PNG, other annotators as .npy + .json).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--usd", required=True, help="stage written by usd_export.export_usd")
    ap.add_argument("--seed", type=int, default=4, help="traffic / incident seed")
    ap.add_argument("--map-seed", type=int, default=None, help="only for stages without stored config")
    ap.add_argument("--seconds", type=float, default=60.0, help="simulated seconds to run")
    ap.add_argument("--dt", type=float, default=0.1, help="traffic step (s)")
    ap.add_argument("--warmup", type=float, default=20.0, help="seconds of traffic before rendering")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out", default=None, help="directory for camera frames (omit: no capture)")
    ap.add_argument("--cameras", default="front_main,pillar_left,pillar_right,rear",
                    help="rig cameras to render (names from driving/cameras.py), plus 'chase'")  # fmt: skip
    ap.add_argument("--annotators", default="rgb", help="comma list, e.g. rgb,semantic_segmentation,"
                    "bounding_box_2d_tight,distance_to_image_plane")  # fmt: skip
    ap.add_argument("--size", default="960x540", help="render size WxH")
    ap.add_argument("--every", type=int, default=1, help="capture every N traffic steps")
    ap.add_argument("--subframes", type=int, default=1, help="RTX subframes per capture (more = less noise)")
    ap.add_argument("--weather", default="clear", choices=("clear", "overcast", "dusk", "night", "rain", "fog"))
    ap.add_argument("--npc", type=int, default=None, help="background cars (default: from the config)")
    ap.add_argument("--no-incidents", action="store_true")
    ap.add_argument("--renderer", default="RayTracedLighting", choices=("RayTracedLighting", "PathTracing"))
    return ap.parse_args(argv)


def start_app(headless: bool, width: int, height: int, renderer: str):
    """Isaac Sim's app object. Must exist before any ``omni.*`` import."""
    try:
        from isaacsim import SimulationApp  # Isaac Sim 4.x / 5.x
    except ImportError:
        from omni.isaac.kit import SimulationApp  # Isaac Sim 2022.x / 2023.x
    return SimulationApp({"headless": headless, "width": width, "height": height, "renderer": renderer})


def open_stage(app, path: str):
    import omni.usd

    ctx = omni.usd.get_context()
    if not ctx.open_stage(os.path.abspath(path)):
        raise RuntimeError(f"Isaac Sim could not open {path}")
    for _ in range(1000):  # wait for payloads / textures (same test as isaacsim.core's is_stage_loading)
        app.update()
        if ctx.get_stage_loading_status()[2] <= 0:
            break
    return ctx.get_stage()


class ChaseCam:
    """A camera prim behind and above the ego, for the viewport and an optional 'chase' capture."""

    def __init__(self, stage, layer, path: str = "/World/ChaseCam", back: float = 11.0, up: float = 4.5,
                 pitch: float = -14.0) -> None:  # fmt: skip
        from pxr import Gf, Sdf, Usd, UsdGeom

        from metatraffic3d.backends.isaac.usd_export import APERTURE_MM, focal_length

        self.back, self.up, self.pitch, self.path = back, up, pitch, path
        with Usd.EditContext(stage, layer):
            cam = UsdGeom.Camera.Define(stage, path)
            cam.CreateFocalLengthAttr(focal_length(75.0))
            cam.CreateHorizontalApertureAttr(APERTURE_MM)
            cam.CreateVerticalApertureAttr(APERTURE_MM * 9 / 16)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 3000.0))
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0, 0, 50))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0))
        sp = Sdf.Path(path)
        self._t = layer.GetAttributeAtPath(sp.AppendProperty("xformOp:translate"))
        self._o = layer.GetAttributeAtPath(sp.AppendProperty("xformOp:orient"))
        self._yaw = None

    def update(self, mirror, pose, terrain) -> None:
        from pxr import Gf

        from metatraffic3d.backends.isaac.usd_export import camera_rotation, quat_from_matrix

        if pose is None:
            return
        x, y, yaw = pose
        if self._yaw is None:
            self._yaw = yaw
        self._yaw += (math.remainder(yaw - self._yaw, 2 * math.pi)) * 0.15  # smooth turns
        cx, cy = x - math.cos(self._yaw) * self.back, y - math.sin(self._yaw) * self.back
        cz = float(terrain.height(x, y)) + self.up
        q = quat_from_matrix(camera_rotation(math.degrees(self._yaw), self.pitch))
        mirror._set(self._t, Gf.Vec3d(cx, cy, cz))
        mirror._set(self._o, Gf.Quatf(*q))


class FrameSaver:
    """Replicator render products + annotators for a set of camera prims."""

    def __init__(self, cameras: dict[str, str], size: tuple[int, int], annotators: list[str], out: str) -> None:
        import omni.replicator.core as rep

        self.rep, self.out = rep, out
        try:
            rep.orchestrator.set_capture_on_play(False)
        except AttributeError:
            pass
        self.items = []  # (camera name, annotator name, annotator)
        for name, prim_path in cameras.items():
            rp = rep.create.render_product(prim_path, size)
            os.makedirs(os.path.join(out, name), exist_ok=True)
            for a in annotators:
                annot = rep.AnnotatorRegistry.get_annotator(a)
                annot.attach([rp])
                self.items.append((name, a, annot))
        self.n = 0

    def capture(self, subframes: int = 1) -> None:
        import numpy as np
        from PIL import Image

        try:
            self.rep.orchestrator.step(rt_subframes=subframes, pause_timeline=False)
        except TypeError:  # older Replicator
            self.rep.orchestrator.step(rt_subframes=subframes)
        for cam, a, annot in self.items:
            data = annot.get_data()
            arr, info = (data.get("data"), data.get("info")) if isinstance(data, dict) else (data, None)
            stem = os.path.join(self.out, cam, f"{a}_{self.n:05d}")
            if a in ("rgb", "LdrColor") and arr is not None and np.asarray(arr).ndim == 3:
                Image.fromarray(np.asarray(arr)[..., :3].astype(np.uint8)).save(stem + ".png")
            else:
                np.save(stem + ".npy", np.asarray(arr))
                if info:
                    with open(stem + ".json", "w") as fh:
                        json.dump(info, fh, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        self.n += 1


def main(argv=None) -> int:
    args = parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))
    app = start_app(args.headless, w, h, args.renderer)
    try:
        return run(app, args, (w, h))
    finally:
        app.close()


def run(app, args, size) -> int:
    # Everything below may import pxr / omni, so only after the app exists.
    try:
        import metatraffic3d  # noqa: F401
    except ImportError:  # run as a file from a source checkout
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    import numpy as np
    from pxr import Usd

    from metatraffic3d.backends.isaac.mirror import KinematicEgo, StageMirror, set_lighting
    from metatraffic3d.backends.isaac.usd_export import stage_config
    from metatraffic3d.config import SimConfig
    from metatraffic3d.traffic.demand import demand_weights
    from metatraffic3d.traffic.events import IncidentManager
    from metatraffic3d.traffic.sim import TrafficSim
    from metatraffic3d.world import build_world

    stage = open_stage(app, args.usd)
    cfg = stage_config(stage)
    if cfg is None:
        cfg = SimConfig()
        cfg.map.seed = args.map_seed if args.map_seed is not None else 4
        print(f"[metatraffic] no stored config in {args.usd}; rebuilding map seed {cfg.map.seed}")
    if args.npc is not None:
        cfg.traffic.num_npc = args.npc
    world = build_world(cfg, with_meshes=False)  # same road network as the exported meshes
    sim = TrafficSim(world.net, cfg, demand_weights(world), seed=args.seed)
    sim.height = world.terrain.height
    incidents = None if args.no_incidents else IncidentManager(sim, args.seed)
    sim.incidents = incidents
    sim.set_weather(args.weather)
    layer = stage.GetSessionLayer()  # runtime edits never touch the city file
    with Usd.EditContext(stage, layer):
        set_lighting(stage, args.weather)
    mirror = StageMirror(stage, world, npc=cfg.traffic.num_npc + 10, pedestrians=cfg.traffic.num_pedestrians,
                         layer=layer)  # fmt: skip
    ego = KinematicEgo(stage, mirror, rng=np.random.default_rng(args.seed))
    chase = ChaseCam(stage, layer)
    for _ in range(int(args.warmup)):
        sim.step(1.0)
    pose = ego.update(sim)
    chase.update(mirror, pose, world.terrain)
    mirror.sync(sim, incidents)
    app.update()

    if not args.headless:  # look through the chase camera in the GUI
        try:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().camera_path = chase.path
        except Exception as exc:  # viewport API differs between Kit versions
            print(f"[metatraffic] could not set the viewport camera: {exc}")

    saver = None
    if args.out:
        names = [n.strip() for n in args.cameras.split(",") if n.strip()]
        cams = {n: (chase.path if n == "chase" else ego.cameras[n]) for n in names}
        saver = FrameSaver(cams, size, [a.strip() for a in args.annotators.split(",") if a.strip()], args.out)

    steps = int(round(args.seconds / args.dt))
    for k in range(steps):
        if not app.is_running():
            break
        sim.step(args.dt)
        pose = ego.update(sim)
        chase.update(mirror, pose, world.terrain)
        counts = mirror.sync(sim, incidents)
        if saver is not None and k % args.every == 0:
            saver.capture(args.subframes)
        else:
            app.update()
        if k % int(round(10 / args.dt)) == 0:
            print(f"[metatraffic] t={sim.time:7.1f}s {counts} ego={ego.vid}")
    if saver is not None:
        print(f"[metatraffic] wrote {saver.n} frames per camera to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
