"""Write the 3D city as a USD stage for NVIDIA Isaac Sim (or any USD viewer).

Only ``pxr`` (``pip install usd-core``) is needed, so this runs anywhere; Isaac Sim is not.
Stage layout (metres, Z up, ``/World`` is the default prim)::

    /World/Looks/c_RRGGBB              UsdPreviewSurface materials, one per colour, shared
    /World/City/<group>/<mesh>         static city meshes (displayColor + bound material)
    /World/Props/<asset>               one PointInstancer per prop asset (trees, street lights, signs)
    /World/Signals/lenses              dark lenses of every signal head (one merged mesh)
    /World/Signals/head_<i>/{red,yellow,green}
                                       lit lamps (emissive spheres), invisible until the runner shows them;
                                       ``metatraffic:node`` / ``metatraffic:deIn`` say which approach they face
    /World/Prototypes/<name>           invisible, instanceable models the runner references
                                       (vehicle types, ``person``, incident props)
    /World/Prototypes/ego              ego car carrying the Tesla-like camera rig (not instanceable)
    /World/Physics/Scene               UsdPhysics scene (gravity -Z 9.81)
    /World/Physics/Ground              invisible collision mesh of the terrain (a coarse plane when flat)
    /World/Environment/{Sun,Sky}       distant light + dome light

Per-instance colours: paint and shirt parts read the ``paintColor`` primvar (falling back to the
model's own colour), emergency light bars read ``barLeft`` / ``barRight``. The runner authors
these as constant primvars on each instance root, where they are inherited by the model.

The world's ``SimConfig`` is stored in the root layer's ``customLayerData["metatraffic"]`` so the
runner can rebuild the identical road network without the meshes.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time

import numpy as np
from pxr import Gf, Kind, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from ...assets import AssetLibrary
from ...config import SimConfig
from ...driving.cameras import TESLA_RIG, CameraSpec
from ...meshes import Part, box, icosphere, merge
from ...traffic.vehicles import TYPES
from ...world import World, prop_meshes

INCIDENT_PROPS = ("traffic_cone", "barrier", "road_work_sign", "excavator")
PROTOTYPES = (*TYPES, "person", *INCIDENT_PROPS)
EGO_ASSET, EGO_PAINT = "sedan", (0.92, 0.93, 0.95, 1.0)
LAMP_COLORS = {"red": (1.0, 0.12, 0.08), "yellow": (1.0, 0.72, 0.05), "green": (0.15, 1.0, 0.35)}
LAMP_RADIUS, LENS_RADIUS = 0.15, 0.13
EMISSIVE_GAIN = 4.0  # UsdPreviewSurface emissiveColor multiplier (RTX needs > 1 to read as "lit" in daylight)
BAR_COLORS = {"barLeft": (0.9, 0.05, 0.05), "barRight": (0.1, 0.2, 1.0)}
APERTURE_MM = 20.955  # Isaac Sim's default horizontal aperture; only the ratio to focalLength matters
CAMERA_ASPECT = 16 / 9
GROUND_FRICTION = (1.0, 0.9)  # static, dynamic
# semantic classes for Replicator annotators (labels are inherited by descendants)
GROUP_CLASS = {"ground": "terrain", "park": "vegetation", "tree": "vegetation", "road": "road", "marking": "road_marking",
               "sidewalk": "sidewalk", "building": "building", "water": "water", "bridge": "bridge", "prop": "pole"}  # fmt: skip
PROP_CLASS = {"tree_round": "vegetation", "tree_cone": "vegetation", "streetlight": "pole", "stop_sign": "traffic_sign",
              "traffic_cone": "construction", "barrier": "construction", "road_work_sign": "construction",
              "excavator": "construction", "person": "pedestrian"}  # fmt: skip
SURFACE_GROUPS = {"ground", "park", "road", "marking", "water", "sidewalk"}  # smooth normals are right for these


# ---------------------------------------------------------------------------- config round trip
def config_to_json(cfg: SimConfig) -> str:
    return json.dumps(dataclasses.asdict(cfg), sort_keys=True)


def config_from_json(text: str) -> SimConfig:
    """Inverse of ``config_to_json`` (lists become tuples where the default is a tuple)."""

    def fill(obj, data: dict):
        for f in dataclasses.fields(obj):
            if f.name not in data:
                continue
            cur, val = getattr(obj, f.name), data[f.name]
            if dataclasses.is_dataclass(cur) and isinstance(val, dict):
                fill(cur, val)
            elif isinstance(cur, tuple) and isinstance(val, list):
                setattr(obj, f.name, tuple(val))
            else:
                setattr(obj, f.name, val)
        return obj

    return fill(SimConfig(), json.loads(text))


def stage_config(stage: Usd.Stage) -> SimConfig | None:
    """The ``SimConfig`` an exported stage was built from (None if it was not written by us)."""
    meta = stage.GetRootLayer().customLayerData.get("metatraffic")
    return config_from_json(meta["config"]) if meta and "config" in meta else None


# ---------------------------------------------------------------------------- small math helpers
def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def quat_from_matrix(r: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix whose columns are the rotated x, y, z axes."""
    m = np.asarray(r, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    q /= np.linalg.norm(q)
    return tuple(float(c) for c in (q if q[0] >= 0 else -q))


def camera_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Columns = camera x (right), y (up), z (back) in the car frame. USD cameras look down -Z."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.cross(right, fwd)
    return np.column_stack([right, up, -fwd])


def focal_length(hfov_deg: float, aperture: float = APERTURE_MM) -> float:
    return aperture / (2 * math.tan(math.radians(hfov_deg) / 2))


def face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    out = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), np.array([0.0, 0.0, 1.0]))
    return out.astype(np.float32)


def _ident(name: str) -> str:
    return Tf.MakeValidIdentifier(name) or "_"


def _rgb_hex(rgb) -> str:
    return "".join(f"{int(round(min(max(c, 0.0), 1.0) * 255)):02x}" for c in rgb[:3])


# ---------------------------------------------------------------------------- authoring helpers
class _Looks:
    """Shared UsdPreviewSurface materials under /World/Looks, created on first use."""

    def __init__(self, stage: Usd.Stage, root: str = "/World/Looks") -> None:
        self.stage, self.root = stage, root
        UsdGeom.Scope.Define(stage, root)
        self.cache: dict[tuple, UsdShade.Material] = {}

    def _surface(self, name: str):
        mat = UsdShade.Material.Define(self.stage, f"{self.root}/{name}")
        sh = UsdShade.Shader.Define(self.stage, f"{self.root}/{name}/Surface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat, sh

    def _reader(self, mat_path: str, primvar: str, fallback) -> UsdShade.Shader:
        rd = UsdShade.Shader.Define(self.stage, f"{mat_path}/{primvar}Reader")
        rd.CreateIdAttr("UsdPrimvarReader_float3")
        rd.CreateInput("varname", Sdf.ValueTypeNames.String).Set(primvar)
        rd.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(*fallback))
        rd.CreateOutput("result", Sdf.ValueTypeNames.Float3)
        return rd

    def color(self, rgba) -> UsdShade.Material:
        """Plain colour (alpha < 1 becomes opacity)."""
        a = float(rgba[3]) if len(rgba) > 3 else 1.0
        key = ("c", _rgb_hex(rgba), round(a, 2))
        if key not in self.cache:
            name = f"c_{key[1]}" + (f"_a{int(round(a * 100)):02d}" if a < 0.995 else "")
            mat, sh = self._surface(name)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgba[:3])))
            if a < 0.995:
                sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(a)
            self.cache[key] = mat
        return self.cache[key]

    def paint(self, rgba) -> UsdShade.Material:
        """Diffuse colour from the ``paintColor`` primvar (per instance), default ``rgba``."""
        key = ("paint", _rgb_hex(rgba))
        if key not in self.cache:
            mat, sh = self._surface(f"paint_{key[1]}")
            rd = self._reader(str(mat.GetPath()), "paintColor", tuple(map(float, rgba[:3])))
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
            self.cache[key] = mat
        return self.cache[key]

    def emissive(self, name: str, rgb, primvar: str | None = None) -> UsdShade.Material:
        """Glowing material; with ``primvar`` the glow colour comes from it (default dark)."""
        key = ("emissive", name)
        if key not in self.cache:
            mat, sh = self._surface(name)
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c * 0.6 for c in rgb)))
            emis = sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
            if primvar:
                rd = self._reader(str(mat.GetPath()), primvar, (0.0, 0.0, 0.0))
                emis.ConnectToSource(rd.ConnectableAPI(), "result")
            else:
                emis.Set(Gf.Vec3f(*(c * EMISSIVE_GAIN for c in rgb)))
            self.cache[key] = mat
        return self.cache[key]


def _bind(prim: Usd.Prim, mat: UsdShade.Material, purpose: str | None = None) -> None:
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    if purpose:
        api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, purpose)
    else:
        api.Bind(mat)


def label(prim: Usd.Prim, cls: str) -> None:
    """Semantic class for Replicator (both the legacy Isaac ``SemanticsAPI`` and ``UsdSemantics`` spellings)."""
    prim.AddAppliedSchema("SemanticsAPI:Semantics")
    prim.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String, custom=False).Set("class")
    prim.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String, custom=False).Set(cls)
    prim.AddAppliedSchema("SemanticsLabelsAPI:class")
    prim.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray, custom=False).Set([cls])


def define_mesh(stage: Usd.Stage, path: str, v, f, rgba, mat: UsdShade.Material | None,
                normals: bool = True, double_sided: bool = True) -> UsdGeom.Mesh:  # fmt: skip
    v32 = np.ascontiguousarray(v, dtype=np.float32)
    f32 = np.ascontiguousarray(f, dtype=np.int32)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v32))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f32), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f32.ravel()))
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v32.min(0))), Gf.Vec3f(*map(float, v32.max(0)))]))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(double_sided)
    if normals:  # flat shading: one normal per face corner
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(face_normals(v32, f32), 3, axis=0)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    m.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, rgba[:3]))]))
    if len(rgba) > 3 and rgba[3] < 0.995:
        m.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.constant).Set(Vt.FloatArray([float(rgba[3])]))
    if mat is not None:
        _bind(m.GetPrim(), mat)
    return m


def _set_pose(prim: Usd.Prim, pos, quat=None) -> None:
    """translate (double3) + orient (quatf) ops, the layout Isaac Sim's XForm helpers expect."""
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, pos)))
    if quat is not None:
        w, x, y, z = quat
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(float(w), float(x), float(y), float(z)))


def build_model(stage: Usd.Stage, path: str, parts: list[Part], looks: _Looks, kind: str = "component",
                paint_override=None) -> int:  # fmt: skip
    """An Xform with one mesh per paint part and one per remaining colour. Returns its face count."""
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    Usd.ModelAPI(root).SetKind(kind)
    faces = 0
    by_color: dict[tuple, list] = {}
    k = 0
    for p in parts:
        if len(p.faces) == 0:
            continue
        if p.tintable:
            rgba = paint_override or p.rgba
            mat = looks.color(rgba) if paint_override else looks.paint(rgba)
            define_mesh(stage, f"{path}/paint{k}", p.vertices, p.faces, rgba, mat)
            k += 1
        else:
            by_color.setdefault(tuple(round(float(c), 4) for c in p.rgba), []).append((p.vertices, p.faces))
        faces += len(p.faces)
    for i, (rgba, vf) in enumerate(by_color.items()):
        v, f = merge(vf)
        define_mesh(stage, f"{path}/part{i}", v, f, rgba, looks.color(rgba))
    return faces


# ---------------------------------------------------------------------------- the exporter
def export_usd(world: World, path: str, assets: AssetLibrary | None = None, instancing: bool = True,
               ground_spacing: float = 8.0, rig: tuple[CameraSpec, ...] = TESLA_RIG) -> dict:  # fmt: skip
    """Write ``world`` to ``path`` (``.usd``/``.usdc`` = binary crate, ``.usda`` = text). Returns stats.

    ``instancing=False`` bakes the props into merged meshes (one per asset part) instead of
    PointInstancers, and leaves the prototypes non-instanceable; use it for viewers without
    instancing support. ``ground_spacing`` is the collision grid step on hilly terrain.
    """
    t0 = time.perf_counter()
    assets = assets or AssetLibrary()
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".usd", ".usda", ".usdc"):
        raise ValueError(f"unsupported extension {ext!r} (use .usd, .usdc or .usda)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = Sdf.Layer.Find(path)
    if old is not None:  # same file exported earlier in this process: start from an empty layer
        old.Clear()
        layer = old
    else:
        if os.path.exists(path):
            os.remove(path)
        layer = Sdf.Layer.CreateNew(path, args={"format": "usdc"} if ext == ".usd" else {})
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world_prim)
    Usd.ModelAPI(world_prim).SetKind(Kind.Tokens.assembly)
    xmin, ymin, xmax, ymax = (float(b) for b in world.bounds)
    layer.customLayerData = {"metatraffic": {
        "exporter": "metatraffic3d.backends.isaac.usd_export", "format_version": 1,
        "config": config_to_json(world.cfg), "map_seed": int(world.cfg.map.seed),
        "bounds": Vt.DoubleArray([xmin, ymin, xmax, ymax]), "terrain_amplitude": float(world.terrain.cfg.amplitude),
    }}  # fmt: skip
    looks = _Looks(stage)
    stats: dict = {"path": path, "format": "usda" if ext == ".usda" else "usdc"}

    # ---- static city: /World/City/<group>/<mesh>
    UsdGeom.Xform.Define(stage, "/World/City")
    used: dict[str, set[str]] = {}
    city_faces = city_verts = city_meshes = 0
    for m in world.scene.meshes:
        if len(m.faces) == 0:
            continue
        g = _ident(m.group)
        if g not in used:
            used[g] = set()
            gp = UsdGeom.Xform.Define(stage, f"/World/City/{g}").GetPrim()
            label(gp, GROUP_CLASS.get(m.group, m.group))
        name, k = _ident(m.name), 1
        while name in used[g]:
            k += 1
            name = f"{_ident(m.name)}_{k}"
        used[g].add(name)
        define_mesh(stage, f"/World/City/{g}/{name}", m.vertices, m.faces, m.rgba, looks.color(m.rgba),
                    normals=m.group not in SURFACE_GROUPS)  # fmt: skip
        city_faces += len(m.faces)
        city_verts += len(m.vertices)
        city_meshes += 1
    stats.update(city_meshes=city_meshes, city_faces=city_faces, city_vertices=city_verts)

    # ---- props: one PointInstancer per asset, standing on the terrain
    UsdGeom.Xform.Define(stage, "/World/Props")
    by_asset: dict[str, list] = {}
    for p in world.props:
        by_asset.setdefault(p.asset, []).append(p)
    prop_counts: dict[str, int] = {}
    proto_faces = 0
    if instancing:
        for asset, props in sorted(by_asset.items()):
            ipath = f"/World/Props/{_ident(asset)}"
            inst = UsdGeom.PointInstancer.Define(stage, ipath)
            label(inst.GetPrim(), PROP_CLASS.get(asset, "prop"))
            UsdGeom.Scope.Define(stage, f"{ipath}/Prototypes")
            lod = "low" if asset.startswith("tree") else "high"
            proto_faces += build_model(stage, f"{ipath}/Prototypes/{_ident(asset)}", assets.parts(asset, lod), looks)
            xs = np.array([p.x for p in props])
            ys = np.array([p.y for p in props])
            zs = np.asarray(world.terrain.height(xs, ys), float).reshape(-1) + np.array([p.z for p in props])
            pos = np.column_stack([xs, ys, zs]).astype(np.float32)
            sc = np.repeat(np.array([p.scale for p in props], np.float32)[:, None], 3, axis=1)
            inst.CreatePrototypesRel().SetTargets([Sdf.Path(f"{ipath}/Prototypes/{_ident(asset)}")])
            inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(np.zeros(len(props), np.int32)))
            inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(pos))
            inst.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(*yaw_quat(p.yaw)) for p in props]))
            inst.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(sc))
            ext_ = inst.ComputeExtentAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default())
            if ext_:
                inst.CreateExtentAttr(ext_)
            prop_counts[asset] = len(props)
    else:
        for m in prop_meshes(world.props, assets, world.terrain):
            define_mesh(stage, f"/World/Props/{_ident(m.name)}", m.vertices, m.faces, m.rgba, looks.color(m.rgba))
            proto_faces += len(m.faces)
        prop_counts = {a: len(ps) for a, ps in by_asset.items()}
    stats.update(prop_instancers=len(prop_counts) if instancing else 0, prop_instances=sum(prop_counts.values()),
                 props_by_asset=prop_counts)  # fmt: skip

    # ---- traffic-light lamps
    n_lamps = 0
    if world.signals:
        sig = UsdGeom.Xform.Define(stage, "/World/Signals").GetPrim()
        label(sig, "traffic_light")
        lens_v, lens_f = merge([icosphere(LENS_RADIUS, center=xyz) for sh in world.signals for xyz in sh.lamps])
        define_mesh(stage, "/World/Signals/lenses", lens_v, lens_f, (0.12, 0.12, 0.12, 1.0),
                    looks.color((0.12, 0.12, 0.12, 1.0)), normals=False)  # fmt: skip
        lamp_mats = {c: looks.emissive(f"lamp_{c}", rgb) for c, rgb in LAMP_COLORS.items()}
        for i, sh in enumerate(world.signals):
            head = UsdGeom.Xform.Define(stage, f"/World/Signals/head_{i}").GetPrim()
            head.CreateAttribute("metatraffic:node", Sdf.ValueTypeNames.Int).Set(int(sh.node))
            head.CreateAttribute("metatraffic:deIn", Sdf.ValueTypeNames.Int).Set(int(sh.de_in))
            head.CreateAttribute("metatraffic:yaw", Sdf.ValueTypeNames.Float).Set(float(sh.yaw))
            for color, xyz in zip(LAMP_COLORS, sh.lamps):
                s = UsdGeom.Sphere.Define(stage, f"/World/Signals/head_{i}/{color}")
                s.CreateRadiusAttr(LAMP_RADIUS)
                s.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(-LAMP_RADIUS), Gf.Vec3f(LAMP_RADIUS)]))
                s.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                _set_pose(s.GetPrim(), xyz)
                _bind(s.GetPrim(), lamp_mats[color])
                n_lamps += 1
    stats.update(signal_heads=len(world.signals), lamps=n_lamps)

    # ---- prototypes for the runner (vehicles, people, incident props, ego with cameras)
    UsdGeom.Scope.Define(stage, "/World/Prototypes")
    n_protos = 0
    for name in PROTOTYPES:
        try:
            parts = assets.parts(name, "high")
        except (KeyError, FileNotFoundError):
            continue
        ppath = f"/World/Prototypes/{_ident(name)}"
        proto_faces += build_model(stage, ppath, parts, looks)
        prim = stage.GetPrimAtPath(ppath)
        vt = TYPES.get(name)
        if vt is not None:
            prim.CreateAttribute("metatraffic:length", Sdf.ValueTypeNames.Float).Set(vt.length)
            prim.CreateAttribute("metatraffic:width", Sdf.ValueTypeNames.Float).Set(vt.width)
            if vt.emergency or name == "police":  # light bars, flashed by the runner via primvars
                h = 3.05 if vt.length > 6 else 1.75
                for side, (key, rgb) in zip((1, -1), BAR_COLORS.items()):
                    bv, bf = box(0.4, 0.6, 0.16, center=(vt.length / 2 - 1.2, side * 0.32, h))
                    define_mesh(stage, f"{ppath}/{key}", bv, bf, (*rgb, 1.0), looks.emissive(key, rgb, primvar=key))
                    proto_faces += len(bf)
        label(prim, "vehicle" if vt is not None else PROP_CLASS.get(name, "prop"))
        UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        if instancing:
            prim.SetInstanceable(True)
        n_protos += 1
    ego = "/World/Prototypes/ego"
    UsdGeom.Xform.Define(stage, ego)
    ego_prim = stage.GetPrimAtPath(ego)
    UsdGeom.Imageable(ego_prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    label(ego_prim, "ego")
    proto_faces += build_model(stage, f"{ego}/body", assets.parts(EGO_ASSET, "high"), looks, paint_override=EGO_PAINT)
    UsdGeom.Xform.Define(stage, f"{ego}/cameras")
    for c in rig:
        cam = UsdGeom.Camera.Define(stage, f"{ego}/cameras/{_ident(c.name)}")
        cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
        cam.CreateFocalLengthAttr(focal_length(c.hfov))
        cam.CreateHorizontalApertureAttr(APERTURE_MM)
        cam.CreateVerticalApertureAttr(APERTURE_MM / CAMERA_ASPECT)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
        cam.GetPrim().CreateAttribute("metatraffic:hfov", Sdf.ValueTypeNames.Float).Set(c.hfov)
        _set_pose(cam.GetPrim(), c.pos, quat_from_matrix(camera_rotation(c.yaw, c.pitch)))
    stats.update(prototypes=n_protos, prototype_faces=proto_faces, cameras=len(rig))

    # ---- physics: scene, ground collider with friction
    UsdGeom.Scope.Define(stage, "/World/Physics")
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics/Scene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    margin = 200.0
    hilly = world.terrain.cfg.amplitude > 0
    if hilly:
        xs, ys, hgt = world.terrain.grid(ground_spacing)
    else:  # flat city: a coarse plane reaching past the map edge
        xs = np.linspace(xmin - margin, xmax + margin, max(2, int((xmax - xmin + 2 * margin) // 100) + 1))
        ys = np.linspace(ymin - margin, ymax + margin, max(2, int((ymax - ymin + 2 * margin) // 100) + 1))
        hgt = np.zeros((len(ys), len(xs)))
    gx, gy = np.meshgrid(xs, ys)
    gv = np.column_stack([gx.ravel(), gy.ravel(), np.asarray(hgt, float).ravel()])
    nx, ny = len(xs), len(ys)
    ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1))
    a = (jj * nx + ii).ravel()
    gf = np.vstack([np.column_stack([a, a + 1, a + nx + 1]), np.column_stack([a, a + nx + 1, a + nx])])
    ground = define_mesh(stage, "/World/Physics/Ground", gv, gf, (0.5, 0.5, 0.5, 1.0), None, normals=False,
                         double_sided=False)  # fmt: skip
    ground.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    gp = ground.GetPrim()
    UsdPhysics.CollisionAPI.Apply(gp)
    UsdPhysics.MeshCollisionAPI.Apply(gp).CreateApproximationAttr(UsdPhysics.Tokens.none)
    pmat = UsdShade.Material.Define(stage, "/World/Physics/GroundMaterial")
    pm = UsdPhysics.MaterialAPI.Apply(pmat.GetPrim())
    pm.CreateStaticFrictionAttr(GROUND_FRICTION[0])
    pm.CreateDynamicFrictionAttr(GROUND_FRICTION[1])
    pm.CreateRestitutionAttr(0.0)
    _bind(gp, pmat, purpose="physics")
    physx = False
    try:  # only inside Isaac Sim / Omniverse (usd-core has no PhysxSchema)
        from pxr import PhysxSchema

        PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(gp).CreateContactOffsetAttr(0.02)
        physx = True
    except ImportError:
        pass
    stats.update(ground_faces=len(gf), ground_kind="terrain_grid" if hilly else "plane", physx_schema=physx)

    # ---- lights
    UsdGeom.Xform.Define(stage, "/World/Environment")
    sun = UsdLux.DistantLight.Define(stage, "/World/Environment/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.9))
    d = np.array([-0.35, 0.25, -1.0])  # direction the sunlight travels (as in the MuJoCo scene)
    zc = -d / np.linalg.norm(d)  # distant lights shine along their local -Z
    xc = np.cross([0.0, 0.0, 1.0], zc)
    xc /= np.linalg.norm(xc)
    _set_pose(sun.GetPrim(), (0.0, 0.0, 0.0), quat_from_matrix(np.column_stack([xc, np.cross(zc, xc), zc])))
    sky = UsdLux.DomeLight.Define(stage, "/World/Environment/Sky")
    sky.CreateIntensityAttr(1000.0)
    sky.CreateColorAttr(Gf.Vec3f(0.78, 0.85, 0.95))

    stage.GetRootLayer().Save()
    stats["materials"] = len(looks.cache)
    stats["prims"] = sum(1 for _ in stage.TraverseAll())
    stats["meshes"] = sum(1 for p in stage.TraverseAll() if p.IsA(UsdGeom.Mesh))
    stats["file_mb"] = round(os.path.getsize(path) / 1e6, 2)
    stats["seconds"] = round(time.perf_counter() - t0, 2)
    return stats


def main(argv=None) -> None:
    import argparse

    from ...world import build_world

    ap = argparse.ArgumentParser(description="Export a MetaTraffic 3D city to USD for Isaac Sim.")
    ap.add_argument("--seed", type=int, default=4, help="map seed")
    ap.add_argument("--out", default="city.usd", help=".usd (binary) or .usda (text)")
    ap.add_argument("--assets", default="auto", choices=("auto", "procedural"))
    ap.add_argument("--no-instancing", action="store_true")
    args = ap.parse_args(argv)
    cfg = SimConfig()
    cfg.map.seed = args.seed
    assets = AssetLibrary(args.assets)
    world = build_world(cfg, assets=assets)
    print(json.dumps(export_usd(world, args.out, assets, instancing=not args.no_instancing), indent=1))


if __name__ == "__main__":
    main()
