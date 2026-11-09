"""Mirror the traffic simulation onto an exported USD stage (the Isaac runtime, minus Isaac).

Pure ``pxr`` + numpy, so it runs and is tested with ``usd-core`` on any machine;
``run_isaac.py`` drives it inside Isaac Sim. Like the MuJoCo runtime it keeps pools of prims
(one Xform per slot, referencing ``/World/Prototypes/<type>``) and each frame writes:

* background cars: pose on the terrain (pitch/roll from the slope), paint colour, flashing
  light bars on emergency vehicles;
* parked cars at the occupied kerb spots nearest the focus point (the ego);
* pedestrians nearest the focus (shirt colour), incident props (cones, barriers, ...);
* traffic lights: which of the red / yellow / green lamps of every head is visible.

Everything is authored in one layer (``layer``; the runner passes the session layer so the
city file is never modified). Per-frame writes go straight to that layer's attribute specs
inside one ``Sdf.ChangeBlock`` (one change notice per frame instead of one per attribute),
and only for values that changed.
"""

from __future__ import annotations

import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from ...traffic.sim import PARKED
from ...traffic.vehicles import TYPES
from ...world import World
from .usd_export import BAR_COLORS, EMISSIVE_GAIN

PARKED_POOL = {"sedan": 160, "suv": 100, "hatchback": 70, "van": 35, "pickup": 30}
PED_POOL = 120
PROP_POOLS = {"traffic_cone": 260, "barrier": 60, "road_work_sign": 12, "excavator": 8}
HIDDEN = (0.0, 0.0, -60.0)
INHERITED, INVISIBLE = UsdGeom.Tokens.inherited, UsdGeom.Tokens.invisible
LAMP_INDEX = {"R": 0, "Y": 1, "G": 2}
LAMP_NAMES = ("red", "yellow", "green")
# sun (rgb, intensity), sky (rgb, intensity), ground friction; same names as the MuJoCo weathers
LIGHTING = {
    "clear": ((1.0, 0.97, 0.9), 3000.0, (0.78, 0.85, 0.95), 1000.0, 1.0),
    "overcast": ((0.9, 0.92, 0.95), 700.0, (0.72, 0.74, 0.77), 1100.0, 1.0),
    "dusk": ((1.0, 0.6, 0.35), 1200.0, (0.55, 0.42, 0.40), 300.0, 1.0),
    "night": ((0.6, 0.65, 0.9), 15.0, (0.04, 0.05, 0.08), 20.0, 0.95),
    "rain": ((0.85, 0.87, 0.9), 450.0, (0.55, 0.57, 0.60), 700.0, 0.65),
    "fog": ((0.9, 0.9, 0.9), 350.0, (0.78, 0.79, 0.80), 900.0, 0.85),
}


def pool_sizes(n_npc: int) -> dict[str, int]:
    """Background-car slots per vehicle type (as ``backends/mujoco/scene.pool_sizes``)."""
    sizes = {name: int(math.ceil(t.share * n_npc * 1.6)) + 2 for name, t in TYPES.items() if t.share > 0}
    sizes.update({"ambulance": 3, "fire_truck": 3})
    return sizes


def ypr_quat(yaw: float, pitch: float, roll: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) for z-y-x Euler angles; pitch + = nose down."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)  # fmt: skip


def place_pose(terrain, x: float, y: float, yaw: float, length: float = 4.5, width: float = 1.8,
               z_off: float = 0.0) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:  # fmt: skip
    """Position and orientation of a body resting on the terrain (four-corner fit, as in MuJoCo)."""
    if terrain.cfg.amplitude > 0:
        c, s = math.cos(yaw), math.sin(yaw)
        zf = float(terrain.height(x + c * length / 2, y + s * length / 2))
        zb = float(terrain.height(x - c * length / 2, y - s * length / 2))
        zl = float(terrain.height(x - s * width / 2, y + c * width / 2))
        zr = float(terrain.height(x + s * width / 2, y - c * width / 2))
        z = (zf + zb + zl + zr) / 4
        pitch, roll = math.atan2(zb - zf, length), math.atan2(zl - zr, width)
    else:
        z, pitch, roll = 0.0, 0.0, 0.0
    return (float(x), float(y), z + z_off), ypr_quat(yaw, pitch, roll)


def set_lighting(stage: Usd.Stage, name: str) -> None:
    """Sun, sky and ground friction for a weather name (fog/rain effects are not modelled)."""
    sun_rgb, sun_i, sky_rgb, sky_i, friction = LIGHTING[name]
    for path, rgb, inten in (("/World/Environment/Sun", sun_rgb, sun_i), ("/World/Environment/Sky", sky_rgb, sky_i)):
        prim = stage.GetPrimAtPath(path)
        if prim:
            prim.GetAttribute("inputs:intensity").Set(inten)
            prim.GetAttribute("inputs:color").Set(Gf.Vec3f(*rgb))
    mat = stage.GetPrimAtPath("/World/Physics/GroundMaterial")
    if mat:
        mat.GetAttribute("physics:dynamicFriction").Set(friction * 0.9)
        mat.GetAttribute("physics:staticFriction").Set(friction)


class _Slot:
    """One pooled prim and the layer specs written each frame."""

    __slots__ = ("path", "proto", "translate", "orient", "visibility", "paint", "bars", "shown", "pose", "color")

    def __init__(self, path: str, proto: str) -> None:
        self.path, self.proto = path, proto
        self.paint = None
        self.bars = None
        self.shown = False
        self.pose = None
        self.color = None


class StageMirror:
    def __init__(self, stage: Usd.Stage, world: World, npc: int = 0, pedestrians: int = 0, parked: bool = True,
                 props: bool = True, root: str = "/World/Traffic", instanceable: bool = True,
                 layer: Sdf.Layer | None = None) -> None:  # fmt: skip
        self.stage, self.world, self.root = stage, world, root
        self.layer = layer or stage.GetEditTarget().GetLayer()
        self.instanceable = instanceable
        self._updates: list[tuple[Sdf.AttributeSpec, object]] = []
        with Usd.EditContext(stage, self.layer):
            self._build(npc, pedestrians, parked, props)

    def _build(self, npc: int, pedestrians: int, parked: bool, props: bool) -> None:
        world = self.world
        UsdGeom.Scope.Define(self.stage, self.root)
        self.pools: dict[str, list[_Slot]] = {}
        if npc:
            for t, n in pool_sizes(npc).items():
                self.pools[t] = [self._slot(f"npc_{t}_{i}", t) for i in range(n)]
        self._free = {t: list(reversed(slots)) for t, slots in self.pools.items()}
        self._slot_of: dict[int, _Slot] = {}
        self.parked_pool = {t: [self._slot(f"parked_{t}_{i}", t) for i in range(n)]
                            for t, n in PARKED_POOL.items()} if parked else {}  # fmt: skip
        self.ped_slots = [self._slot(f"ped{i}", "person") for i in range(min(pedestrians, PED_POOL))]
        self.prop_pools = {a: [self._slot(f"prop_{a}_{i}", a, tint=False) for i in range(n)]
                           for a, n in PROP_POOLS.items()} if props else {}  # fmt: skip
        net = world.net
        self.spot_xy = np.array([s.pose[:2] for s in net.spots]) if net.spots else np.zeros((0, 2))
        self.parked_radius = 200.0
        self.focus: tuple[float, float] | None = None  # parked cars / pedestrians are drawn near here
        self.exclude: set[int] = set()  # sim vehicle ids not drawn (the one the ego camera rides)
        self._last_parked_key = None
        self._props_version = -1
        self.heads = self._read_heads()
        self._head_state: list[int | None] = [None] * len(self.heads)  # lit lamp index, -1 = dark

    # ------------------------------------------------------------------ setup
    def _slot(self, name: str, proto: str, tint: bool = True) -> _Slot:
        path = Sdf.Path(f"{self.root}/{name}")
        prim = self.stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddInternalReference(Sdf.Path(f"/World/Prototypes/{proto}"))
        prim.SetInstanceable(self.instanceable)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*HIDDEN))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0))
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(INVISIBLE)
        pv = UsdGeom.PrimvarsAPI(prim)
        s = _Slot(str(path), proto)
        spec = self.layer.GetAttributeAtPath
        s.translate = spec(path.AppendProperty("xformOp:translate"))
        s.orient = spec(path.AppendProperty("xformOp:orient"))
        s.visibility = spec(path.AppendProperty("visibility"))
        if tint:
            pv.CreatePrimvar("paintColor", Sdf.ValueTypeNames.Color3f, UsdGeom.Tokens.constant)
            s.paint = spec(path.AppendProperty("primvars:paintColor"))
        vt = TYPES.get(proto)
        if vt is not None and (vt.emergency or proto == "police"):
            for key in BAR_COLORS:
                pv.CreatePrimvar(key, Sdf.ValueTypeNames.Color3f, UsdGeom.Tokens.constant).Set(Gf.Vec3f(0.0))
            s.bars = tuple(spec(path.AppendProperty(f"primvars:{key}")) for key in BAR_COLORS)
        return s

    def _read_heads(self) -> list[dict]:
        heads = []
        sig = self.stage.GetPrimAtPath("/World/Signals")
        if not sig:
            return heads
        for prim in sig.GetChildren():
            node = prim.GetAttribute("metatraffic:node")
            if not node:
                continue
            for c in LAMP_NAMES:  # make sure our layer holds a visibility spec we can write to
                UsdGeom.Imageable(prim.GetChild(c)).CreateVisibilityAttr().Set(INVISIBLE)
            lamps = [self.layer.GetAttributeAtPath(prim.GetPath().AppendChild(c).AppendProperty("visibility"))
                     for c in LAMP_NAMES]  # fmt: skip
            pos = prim.GetChild("red").GetAttribute("xformOp:translate").Get()
            heads.append({"node": int(node.Get()), "de_in": int(prim.GetAttribute("metatraffic:deIn").Get()),
                          "lamps": lamps, "xy": (pos[0], pos[1])})  # fmt: skip
        return heads

    # ------------------------------------------------------------------ writing
    def _set(self, spec: Sdf.AttributeSpec, value) -> None:
        self._updates.append((spec, value))

    def _place(self, s: _Slot, x: float, y: float, yaw: float, length: float = 4.5, width: float = 1.8,
               z_off: float = 0.0) -> None:  # fmt: skip
        pos, q = place_pose(self.world.terrain, x, y, yaw, length, width, z_off)
        if s.pose != (pos, q):
            s.pose = (pos, q)
            self._set(s.translate, Gf.Vec3d(*pos))
            self._set(s.orient, Gf.Quatf(*q))
        if not s.shown:
            s.shown = True
            self._set(s.visibility, INHERITED)

    def _hide(self, s: _Slot) -> None:
        if s.shown:
            s.shown = False
            s.pose = None
            self._set(s.visibility, INVISIBLE)
            self._set(s.translate, Gf.Vec3d(*HIDDEN))

    def _tint(self, s: _Slot, rgb) -> None:
        rgb = tuple(float(c) for c in rgb[:3])
        if s.paint is not None and s.color != rgb:
            s.color = rgb
            self._set(s.paint, Gf.Vec3f(*rgb))

    def flush(self) -> int:
        """Apply the queued writes in one change block; returns how many were written."""
        n = len(self._updates)
        if n:
            with Sdf.ChangeBlock():
                for spec, value in self._updates:
                    spec.default = value
            self._updates.clear()
        return n

    # ------------------------------------------------------------------ syncing
    def sync(self, sim, incidents=None) -> dict:
        t = sim.time
        net = sim.net
        if self.focus is not None:
            fx, fy = self.focus
        elif len(self.spot_xy):
            fx, fy = self.spot_xy.mean(0)
        else:
            fx, fy = 0.0, 0.0
        counts = {"npc": 0, "parked": 0, "peds": 0, "props": 0, "lamps": 0}
        # parked cars near the focus (refreshed when the focus moves or every 5 s)
        if self.parked_pool and len(self.spot_xy):
            key = (round(fx / 10), round(fy / 10), int(t // 5))
            if key != self._last_parked_key:
                self._last_parked_key = key
                d2 = (self.spot_xy[:, 0] - fx) ** 2 + (self.spot_xy[:, 1] - fy) ** 2
                near = np.flatnonzero((d2 < self.parked_radius**2) & (sim.spot_state == PARKED))
                near = near[np.argsort(d2[near])]
                used = dict.fromkeys(self.parked_pool, 0)
                for sid in near:
                    ptype = sim.spot_type[sid]
                    pool = self.parked_pool.get(ptype)
                    if pool is None or used[ptype] >= len(pool):
                        continue
                    s = pool[used[ptype]]
                    used[ptype] += 1
                    x, y, h = net.spots[sid].pose
                    vt = TYPES[ptype]
                    self._place(s, x, y, h, vt.length, vt.width)
                    self._tint(s, sim.spot_color[sid])
                for ptype, pool in self.parked_pool.items():
                    for s in pool[used[ptype]:]:
                        self._hide(s)
            counts["parked"] = sum(s.shown for pool in self.parked_pool.values() for s in pool)
        # background traffic
        live = {v.id: v for v in sim.vehicles.values() if v.kind == "npc" and v.id not in self.exclude}
        for vid in list(self._slot_of):
            if vid not in live:
                s = self._slot_of.pop(vid)
                self._hide(s)
                self._free[s.proto].append(s)
        flash = int(t * 3) % 2 == 0
        for vid, v in live.items():
            s = self._slot_of.get(vid)
            if s is None:
                pool = self._free.get(v.vtype) or self._free.get("sedan")
                if not pool:
                    continue
                s = pool.pop()
                self._slot_of[vid] = s
                self._tint(s, v.color)
            x, y, h = sim.pose(v)
            vt = TYPES.get(v.vtype)
            self._place(s, x, y, h, vt.length if vt else 4.5, vt.width if vt else 1.8)
            if s.bars is not None:
                on = bool(v.emergency)
                left = BAR_COLORS["barLeft"] if on and flash else (0.0, 0.0, 0.0)
                right = BAR_COLORS["barRight"] if on and not flash else (0.0, 0.0, 0.0)
                self._set(s.bars[0], Gf.Vec3f(*(c * EMISSIVE_GAIN for c in left)))
                self._set(s.bars[1], Gf.Vec3f(*(c * EMISSIVE_GAIN for c in right)))
            counts["npc"] += 1
        # pedestrians nearest the focus
        if self.ped_slots:
            pos = [(sim.peds.position(p), p) for p in sim.peds.peds]
            pos.sort(key=lambda q: (q[0][0] - fx) ** 2 + (q[0][1] - fy) ** 2)
            for i, s in enumerate(self.ped_slots):
                if i < len(pos):
                    (x, y, h), p = pos[i]
                    self._place(s, x, y, h, 0.4, 0.4, 0.0 if p.state == 2 else 0.2)  # 2 = crossing the road
                    self._tint(s, p.shirt)
                    counts["peds"] += 1
                else:
                    self._hide(s)
        # construction / crash props
        if incidents is not None and self.prop_pools and incidents.version != self._props_version:
            self._props_version = incidents.version
            used = dict.fromkeys(self.prop_pools, 0)
            for asset, x, y, yaw in incidents.props:
                pool = self.prop_pools.get(asset)
                if pool is None or used[asset] >= len(pool):
                    continue
                self._place(pool[used[asset]], x, y, yaw, 0.5, 0.5)
                used[asset] += 1
            for asset, pool in self.prop_pools.items():
                for s in pool[used[asset]:]:
                    self._hide(s)
        counts["props"] = sum(s.shown for pool in self.prop_pools.values() for s in pool)
        # traffic lights: every head shows the lamp of its current state (outage = blinking red)
        blink = int(t * 1.5) % 2 == 0
        for i, hd in enumerate(self.heads):
            ctl = net.nodes[hd["node"]].control
            if ctl is None or ctl.kind != "signal":
                lit = -1
            elif ctl.outage:
                lit = 0 if blink else -1
            else:
                lit = LAMP_INDEX[ctl.state(hd["de_in"], t)]
            prev = self._head_state[i]
            if prev != lit:
                for k, spec in enumerate(hd["lamps"]):
                    if prev is None or k in (prev, lit):
                        self._set(spec, INHERITED if k == lit else INVISIBLE)
                self._head_state[i] = lit
            counts["lamps"] += lit >= 0
        counts["writes"] = self.flush()
        return counts


class KinematicEgo:
    """The ego car rides along a TrafficSim vehicle (kinematic: its pose comes from the sim).

    It references ``/World/Prototypes/ego`` (body + camera rig) at ``path``, hides the followed
    vehicle in the mirror's pools, and picks a new vehicle when the old one parks or leaves.
    A PhysX-driven ego (``PhysxVehicleAPI``, or an articulation with wheel joints) is future work.
    """

    def __init__(self, stage: Usd.Stage, mirror: StageMirror, path: str = "/World/Ego",
                 rng: np.random.Generator | None = None) -> None:  # fmt: skip
        self.stage, self.mirror, self.path = stage, mirror, path
        self.rng = rng or np.random.default_rng(0)
        with Usd.EditContext(stage, mirror.layer):
            prim = stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddInternalReference(Sdf.Path("/World/Prototypes/ego"))
            prim.SetInstanceable(False)  # cameras must be real prims for render products
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(INHERITED)
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*HIDDEN))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0))
        sp = Sdf.Path(path)
        self._t = mirror.layer.GetAttributeAtPath(sp.AppendProperty("xformOp:translate"))
        self._o = mirror.layer.GetAttributeAtPath(sp.AppendProperty("xformOp:orient"))
        self.vid: int | None = None
        self.pose: tuple[float, float, float] | None = None
        cams = stage.GetPrimAtPath(f"{path}/cameras")
        self.cameras = {c.GetName(): str(c.GetPath()) for c in cams.GetChildren() if c.IsA(UsdGeom.Camera)} if cams else {}

    def _pick(self, sim):
        cand = [v for v in sim.vehicles.values() if v.kind == "npc" and v.state == "drive" and v.dest is not None
                and v.vtype in ("sedan", "suv", "hatchback", "taxi", "sports") and not v.emergency]  # fmt: skip
        if not cand:
            cand = [v for v in sim.vehicles.values() if v.kind == "npc" and v.state == "drive"]
        return cand[int(self.rng.integers(len(cand)))] if cand else None

    def update(self, sim) -> tuple[float, float, float] | None:
        """Follow the vehicle (queued in the mirror; written by the next ``mirror.sync``)."""
        v = sim.vehicles.get(self.vid) if self.vid is not None else None
        if v is None or v.state in ("parked", "wait_out", "dwell"):  # trip over: ride another car
            self.mirror.exclude.discard(self.vid)
            v = self._pick(sim)
            self.vid = v.id if v is not None else None
            if v is None:
                return None
            self.mirror.exclude.add(v.id)
        x, y, h = sim.pose(v)
        pos, q = place_pose(self.mirror.world.terrain, x, y, h, 4.5, 1.8)
        self.mirror._set(self._t, Gf.Vec3d(*pos))
        self.mirror._set(self._o, Gf.Quatf(*q))
        self.mirror.focus = (x, y)
        self.pose = (x, y, h)
        return self.pose


def bound_material(prim: Usd.Prim) -> str | None:
    """Path of the material bound to a prim (helper for tests and debugging)."""
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    return str(mat.GetPath()) if mat else None
