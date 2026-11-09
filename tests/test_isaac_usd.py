"""USD export for Isaac Sim, checked with usd-core (Isaac Sim itself is not needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from metatraffic3d.assets import AssetLibrary  # noqa: E402
from metatraffic3d.backends.isaac.mirror import KinematicEgo, StageMirror, place_pose, set_lighting  # noqa: E402
from metatraffic3d.backends.isaac.usd_export import (  # noqa: E402
    PROTOTYPES,
    config_from_json,
    config_to_json,
    export_usd,
    stage_config,
)
from metatraffic3d.config import SimConfig  # noqa: E402
from metatraffic3d.driving.cameras import TESLA_RIG  # noqa: E402
from metatraffic3d.terrain import Terrain, TerrainConfig  # noqa: E402
from metatraffic3d.world import build_world  # noqa: E402

# Seed 1 at 300x200: flat, with a river (water, bridges) and the elevated highway. On this map
# size the hilly seeds (2, 3, 6, 7, 9) fail in world.py's triangulation and seeds 4, 5, 13 in
# roads.py's corner trimming, so the hilly path is tested by swapping in a hilly Terrain below.
SEED = 1


def small_cfg() -> SimConfig:
    cfg = SimConfig()
    cfg.map.width, cfg.map.height = 300, 200
    cfg.map.seed = SEED
    return cfg


@pytest.fixture(scope="module")
def assets():
    return AssetLibrary("procedural")


@pytest.fixture(scope="module")
def world(assets):
    return build_world(small_cfg(), assets=assets)


@pytest.fixture(scope="module")
def exported(world, assets, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("usd") / "city.usda")
    stats = export_usd(world, path, assets)
    return path, stats


def _mesh_faces(prim) -> int:
    return len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or [])


def test_stage_metadata_and_city_meshes(world, exported):
    path, stats = exported
    stage = Usd.Stage.Open(path)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetDefaultPrim().GetPath() == "/World"
    meshes = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/City")) if p.IsA(UsdGeom.Mesh)]
    expected = [m for m in world.scene.meshes if len(m.faces)]
    assert len(meshes) == len(expected) == stats["city_meshes"]
    faces = sum(_mesh_faces(p) for p in meshes)
    assert faces == world.scene.stats()["faces"] == stats["city_faces"]
    assert sum(len(UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in meshes) == sum(len(m.vertices) for m in expected)
    groups = {p.GetName() for p in stage.GetPrimAtPath("/World/City").GetChildren()}
    assert groups == {m.group for m in expected}
    for p in meshes:  # every mesh has a colour and a shared material of that colour
        mesh = UsdGeom.Mesh(p)
        assert mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none
        mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        assert mat and str(mat.GetPath()).startswith("/World/Looks/c_")
        rgb = mesh.GetDisplayColorPrimvar().Get()[0]
        shader = UsdShade.Shader(stage.GetPrimAtPath(f"{mat.GetPath()}/Surface"))
        assert np.allclose(shader.GetInput("diffuseColor").Get(), rgb, atol=1 / 255)
    n_colors = len({tuple(int(round(c * 255)) for c in m.rgba[:3]) for m in expected})
    looks = [p for p in stage.GetPrimAtPath("/World/Looks").GetChildren() if p.GetName().startswith("c_")]
    assert len(looks) >= n_colors  # shared: one per colour (plus prototype colours), not one per mesh


def test_bounding_box_matches_world(world, exported):
    stage = Usd.Stage.Open(exported[0])
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    box = cache.ComputeWorldBound(stage.GetPrimAtPath("/World")).ComputeAlignedRange()
    xmin, ymin, xmax, ymax = world.bounds
    tol = 0.05 * max(xmax - xmin, ymax - ymin)  # roads and kerbs run a little past the map edge
    lo, hi = box.GetMin(), box.GetMax()
    for a, b in ((lo[0], xmin), (lo[1], ymin), (hi[0], xmax), (hi[1], ymax)):
        assert abs(a - b) < tol, (lo, hi, world.bounds)
    assert -10 < lo[2] and hi[2] < 400  # nothing far below the sea or above the tallest tower


def test_props_are_point_instancers(world, exported):
    stage = Usd.Stage.Open(exported[0])
    by_asset = {}
    for p in world.props:
        by_asset.setdefault(p.asset, []).append(p)
    total = 0
    for asset, props in by_asset.items():
        inst = UsdGeom.PointInstancer(stage.GetPrimAtPath(f"/World/Props/{asset}"))
        assert inst, asset
        protos = inst.GetPrototypesRel().GetTargets()
        assert len(protos) == 1 and stage.GetPrimAtPath(protos[0]).IsValid()
        pos = np.array(inst.GetPositionsAttr().Get())
        assert len(pos) == len(props) == len(inst.GetProtoIndicesAttr().Get())
        z = world.terrain.height(np.array([p.x for p in props]), np.array([p.y for p in props]))
        assert np.allclose(pos[:, 2], np.asarray(z).reshape(-1) + [p.z for p in props], atol=1e-3)
        xf = inst.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default())
        for k in (0, len(props) // 2, len(props) - 1):  # local x axis points along the prop's yaw
            m = Gf.Matrix4d(xf[k])
            d = m.TransformDir(Gf.Vec3d(1, 0, 0))
            assert abs(math.remainder(math.atan2(d[1], d[0]) - props[k].yaw, 2 * math.pi)) < 2e-3  # quath precision
            assert abs(m.ExtractTranslation()[0] - props[k].x) < 1e-2
            assert abs(m.TransformDir(Gf.Vec3d(1, 0, 0)).GetLength() - props[k].scale) < 1e-2
        total += len(props)
    assert total == len(world.props) == exported[1]["prop_instances"]


def test_signals_lamps(world, exported):
    stage = Usd.Stage.Open(exported[0])
    heads = [p for p in stage.GetPrimAtPath("/World/Signals").GetChildren() if p.GetName().startswith("head_")]
    assert len(heads) == len(world.signals) > 0
    for prim, sh in zip(sorted(heads, key=lambda p: int(p.GetName()[5:])), world.signals):
        assert prim.GetAttribute("metatraffic:node").Get() == sh.node
        assert prim.GetAttribute("metatraffic:deIn").Get() == sh.de_in
        for name, xyz in zip(("red", "yellow", "green"), sh.lamps):
            lamp = prim.GetChild(name)
            assert lamp.IsA(UsdGeom.Sphere)
            assert UsdGeom.Imageable(lamp).ComputeVisibility() == UsdGeom.Tokens.invisible
            assert np.allclose(lamp.GetAttribute("xformOp:translate").Get(), xyz, atol=1e-6)
            mat, _ = UsdShade.MaterialBindingAPI(lamp).ComputeBoundMaterial()
            emis = UsdShade.Shader(stage.GetPrimAtPath(f"{mat.GetPath()}/Surface")).GetInput("emissiveColor").Get()
            assert max(emis) > 1.0


def test_prototypes_and_camera_rig(exported, assets):
    stage = Usd.Stage.Open(exported[0])
    for name in PROTOTYPES:
        prim = stage.GetPrimAtPath(f"/World/Prototypes/{name}")
        assert prim.IsValid() and prim.IsInstanceable(), name
        assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
        faces = sum(_mesh_faces(p) for p in Usd.PrimRange(prim) if p.IsA(UsdGeom.Mesh) and not p.GetName().startswith("bar"))
        assert faces == sum(len(p.faces) for p in assets.parts(name, "high")), name
    paint = stage.GetPrimAtPath("/World/Prototypes/sedan/paint0")
    mat, _ = UsdShade.MaterialBindingAPI(paint).ComputeBoundMaterial()
    reader = UsdShade.Shader(stage.GetPrimAtPath(f"{mat.GetPath()}/paintColorReader"))
    assert reader.GetIdAttr().Get() == "UsdPrimvarReader_float3"
    cams = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Prototypes/ego")) if p.IsA(UsdGeom.Camera)]
    assert [c.GetName() for c in cams] == [c.name for c in TESLA_RIG]
    for prim, spec in zip(cams, TESLA_RIG):
        gcam = UsdGeom.Camera(prim).GetCamera()
        assert abs(gcam.GetFieldOfView(Gf.Camera.FOVHorizontal) - spec.hfov) < 1e-3
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        assert np.allclose(m.ExtractTranslation(), spec.pos, atol=1e-6)
        fwd = m.TransformDir(Gf.Vec3d(0, 0, -1))  # USD cameras look down -Z
        yaw, pitch = math.radians(spec.yaw), math.radians(spec.pitch)
        want = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
        assert np.allclose(fwd, want, atol=1e-5), spec.name
        up = m.TransformDir(Gf.Vec3d(0, 1, 0))
        assert up[2] > 0.9  # no roll


def test_physics(exported, world):
    stage = Usd.Stage.Open(exported[0])
    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/Physics/Scene"))
    assert scene and tuple(scene.GetGravityDirectionAttr().Get()) == (0, 0, -1)
    assert abs(scene.GetGravityMagnitudeAttr().Get() - 9.81) < 1e-6
    ground = stage.GetPrimAtPath("/World/Physics/Ground")
    assert ground.HasAPI(UsdPhysics.CollisionAPI) and ground.HasAPI(UsdPhysics.MeshCollisionAPI)
    mat, _ = UsdShade.MaterialBindingAPI(ground).ComputeBoundMaterial("physics")
    assert UsdPhysics.MaterialAPI(mat.GetPrim()).GetStaticFrictionAttr().Get() > 0.5
    pts = np.array(UsdGeom.Mesh(ground).GetPointsAttr().Get())
    xmin, ymin, xmax, ymax = world.bounds
    assert pts[:, 0].min() <= xmin and pts[:, 0].max() >= xmax and pts[:, 1].min() <= ymin and pts[:, 1].max() >= ymax
    assert stage_config(stage).map.seed == SEED


def test_hilly_terrain_and_binary(world, assets, tmp_path):
    """Swap in a hilly terrain: props follow it and the ground collider becomes a height grid."""
    flat = world.terrain
    world.terrain = Terrain(world.bounds, None, 1, TerrainConfig(amplitude=25.0))
    try:
        stats = export_usd(world, str(tmp_path / "hilly.usd"), assets, ground_spacing=20.0)
        stage = Usd.Stage.Open(str(tmp_path / "hilly.usd"))
        with open(tmp_path / "hilly.usd", "rb") as fh:
            assert fh.read(8) == b"PXR-USDC"  # binary crate, not text
        assert stats["ground_kind"] == "terrain_grid"
        pts = np.array(UsdGeom.Mesh(stage.GetPrimAtPath("/World/Physics/Ground")).GetPointsAttr().Get())
        assert np.allclose(pts[:, 2], world.terrain.height(pts[:, 0], pts[:, 1]), atol=1e-2)
        assert pts[:, 2].max() > 5.0
        props = [p for p in world.props if p.asset == "streetlight"]
        pos = np.array(UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/Props/streetlight")).GetPositionsAttr().Get())
        z = world.terrain.height(np.array([p.x for p in props]), np.array([p.y for p in props])) + [p.z for p in props]
        assert np.allclose(pos[:, 2], z, atol=1e-2)
    finally:
        world.terrain = flat
    assert stats["file_mb"] > 0 and stats["prims"] > 100


def test_instancing_off_bakes_props(world, assets, tmp_path):
    stats = export_usd(world, str(tmp_path / "baked.usda"), assets, instancing=False)
    stage = Usd.Stage.Open(str(tmp_path / "baked.usda"))
    assert not any(p.IsA(UsdGeom.PointInstancer) for p in stage.Traverse())
    assert not stage.GetPrimAtPath("/World/Prototypes/sedan").IsInstanceable()
    assert stats["prop_instancers"] == 0 and stats["prop_instances"] == len(world.props)


def test_config_round_trip():
    cfg = SimConfig()
    cfg.map.seed, cfg.parking.occupancy, cfg.profile.terrain = 11, (0.2, 0.3), "hilly"
    back = config_from_json(config_to_json(cfg))
    assert back == cfg and isinstance(back.parking.occupancy, tuple)


def test_mirror_moves_traffic_on_the_stage(world, exported):
    """The runner's Isaac-free half: pools, poses, colours, lamps and the ego on a real stage."""
    from metatraffic3d.traffic.demand import demand_weights
    from metatraffic3d.traffic.events import IncidentManager
    from metatraffic3d.traffic.sim import TrafficSim

    stage = Usd.Stage.Open(exported[0])
    root = stage.GetRootLayer()
    cfg = stage_config(stage)
    sim = TrafficSim(world.net, cfg, demand_weights(world), seed=2)
    sim.height = world.terrain.height
    incidents = IncidentManager(sim, 2)
    sim.incidents = incidents
    mirror = StageMirror(stage, world, npc=cfg.traffic.num_npc + 10, pedestrians=cfg.traffic.num_pedestrians,
                         layer=stage.GetSessionLayer())  # fmt: skip
    ego = KinematicEgo(stage, mirror)
    for _ in range(15):
        sim.step(1.0)
    for _ in range(5):
        sim.step(0.1)
        ego.update(sim)
        counts = mirror.sync(sim, incidents)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        set_lighting(stage, "night")
    assert stage.GetPrimAtPath("/World/Environment/Sun").GetAttribute("inputs:intensity").Get() < 100
    assert not root.dirty  # runtime prims and lighting live in the session layer only
    n_npc = sum(1 for v in sim.vehicles.values() if v.kind == "npc" and v.id not in mirror.exclude)
    assert counts["npc"] == n_npc > 0 and counts["parked"] > 0 and counts["peds"] > 0
    for vid, slot in list(mirror._slot_of.items())[:20]:  # drawn exactly where the sim has them
        v = sim.vehicles[vid]
        prim = stage.GetPrimAtPath(slot.path)
        assert prim.IsInstance() and UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.inherited
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        x, y, yaw = sim.pose(v)
        pos, _ = place_pose(world.terrain, x, y, yaw)
        assert np.allclose(m.ExtractTranslation(), pos, atol=1e-6)
        d = m.TransformDir(Gf.Vec3d(1, 0, 0))
        assert abs(math.remainder(math.atan2(d[1], d[0]) - yaw, 2 * math.pi)) < 1e-4
        assert np.allclose(prim.GetAttribute("primvars:paintColor").Get(), v.color, atol=1e-6)
    shown = 0
    for prim in stage.GetPrimAtPath("/World/Signals").GetChildren():
        if prim.GetName().startswith("head_"):
            vis = [UsdGeom.Imageable(prim.GetChild(c)).ComputeVisibility() == UsdGeom.Tokens.inherited
                   for c in ("red", "yellow", "green")]  # fmt: skip
            assert sum(vis) <= 1
            shown += sum(vis)
    assert shown == counts["lamps"] > 0
    # ego rides a car; its cameras are real (non-instance-proxy) prims for render products
    x, y, yaw = ego.pose
    cam = stage.GetPrimAtPath(ego.cameras["front_main"])
    assert cam.IsA(UsdGeom.Camera) and not cam.IsInstanceProxy()
    m = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    fwd = m.TransformDir(Gf.Vec3d(0, 0, -1))
    assert abs(math.remainder(math.atan2(fwd[1], fwd[0]) - yaw, 2 * math.pi)) < 1e-3
    assert ego.vid in mirror.exclude and ego.vid not in mirror._slot_of


def test_run_isaac_imports_without_isaac():
    from metatraffic3d.backends.isaac import run_isaac

    args = run_isaac.parse_args(["--usd", "city.usd", "--seed", "4", "--seconds", "60"])
    assert args.seconds == 60 and args.seed == 4
