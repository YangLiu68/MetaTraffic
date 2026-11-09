"""Write the 3D city as a MuJoCo model (scene.xml + OBJ meshes).

* Static city geometry is visual only. A hidden plane (flat cities) or height field (hilly
  ones) carries the cars, so gravity and slopes act on the ego car.
* Traffic-light lamps are small geoms whose material is switched at run time (lit / dark).
* Pools of mocap bodies stand in for background vehicles (one pool per vehicle type),
  pedestrians, passengers and construction props; unused ones wait under the ground.
* Every kerb spot has a parked-car geom that is shown or hidden at run time.
* The ego car is a physical car (see ``vehicle.py``) carrying the camera rig and headlights.

Collision bits: ground 1/1, ego tyres 1/1 (ground only), ego body 2/2, traffic, people and
props 2/0 (they only touch the ego).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from ...assets import AssetLibrary
from ...meshes import Palette, merge, write_obj
from ...traffic.vehicles import PARKED_TYPES, TYPES
from ...world import World, prop_meshes
from .vehicle import ego_actuators, ego_xml

G_CITY, G_PROPS, G_VEHICLES, G_HIDDEN = 0, 1, 2, 3
PROP_POOLS = {"traffic_cone": 260, "barrier": 60, "road_work_sign": 12, "excavator": 8}
PARKED_POOL = {"sedan": 60, "suv": 36, "hatchback": 26, "van": 12, "pickup": 10}
PED_POOL = 120  # pedestrians drawn (the nearest ones)
LIT_LAMPS = 160  # lit traffic-light lamps drawn (the nearest signal heads)
LAMP_COLORS = {"red": (1.0, 0.12, 0.08), "yellow": (1.0, 0.72, 0.05), "green": (0.15, 1.0, 0.35)}


@dataclass
class VehicleSlots:
    npc: int = 0  # total background cars (split into per-type pools)
    taxis: int = 0
    physics_taxis: bool = False
    parked: bool = True
    passengers: int = 0
    pedestrians: int = 0
    props: bool = True
    ego: bool = False
    ego_cameras: str = ""
    spot_types: list[str] = field(default_factory=list)


def _fmt(a) -> str:
    return " ".join(f"{x:.4g}" for x in np.ravel(a))


def _rgba(c) -> str:
    return _fmt(c if len(c) == 4 else (*c, 1.0))


def pool_sizes(n_npc: int) -> dict[str, int]:
    sizes = {name: int(math.ceil(t.share * n_npc * 1.6)) + 2 for name, t in TYPES.items() if t.share > 0}
    sizes.update({"ambulance": 3, "fire_truck": 3})
    return sizes


def build_mjcf(world: World, out_dir: str, assets: AssetLibrary | None = None,
               slots: VehicleSlots | None = None) -> str:  # fmt: skip
    assets = assets or AssetLibrary()
    slots = slots or VehicleSlots()
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    xmin, ymin, xmax, ymax = world.bounds
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    size = max(xmax - xmin, ymax - ymin)
    asset_xml: list[str] = []
    body_xml: list[str] = []
    written: set[str] = set()
    palette = Palette(32)

    def mesh_asset(name: str, v, f, uv=None) -> None:
        if name in written:
            return
        written.add(name)
        write_obj(os.path.join(mesh_dir, f"{name}.obj"), v, f, uv)
        asset_xml.append(f'<mesh name="{name}" file="{name}.obj" inertia="shell"/>')

    # ---- static city
    for m in list(world.scene.meshes) + prop_meshes(world.props, assets, world.terrain):
        mesh_asset(m.name, m.vertices, m.faces)
        group = G_PROPS if m.group in ("tree", "prop", "marking") else G_CITY
        body_xml.append(f'<geom name="{m.name}" type="mesh" mesh="{m.name}" rgba="{_rgba(m.rgba)}" contype="0" '
                        f'conaffinity="0" group="{group}" material="matte"/>')  # fmt: skip

    # ---- ground for physics
    terrain = world.terrain
    if terrain.cfg.amplitude > 0:
        xs, ys, hgt = terrain.grid(8.0)
        top = float(hgt.max()) + 1e-3
        asset_xml.append(f'<hfield name="terrain" nrow="{len(ys)}" ncol="{len(xs)}" '
                         f'size="{(xs[-1] - xs[0]) / 2:.2f} {(ys[-1] - ys[0]) / 2:.2f} {top:.3f} 2"/>')  # fmt: skip
        hx, hy = (xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2
        body_xml.append(f'<geom name="ground_collision" type="hfield" hfield="terrain" pos="{hx:.2f} {hy:.2f} 0" '
                        f'rgba="0 0 0 0" group="{G_HIDDEN}" contype="1" conaffinity="1" friction="1 0.005 0.0001"/>')  # fmt: skip
    else:
        body_xml.append(f'<geom name="ground_collision" type="plane" size="{size:.0f} {size:.0f} 1" '
                        f'pos="{cx:.1f} {cy:.1f} 0" rgba="0 0 0 0" group="{G_HIDDEN}" contype="1" conaffinity="1" '
                        f'friction="1 0.005 0.0001"/>')  # fmt: skip

    # ---- elevated highway deck: boxes the ego car can drive on (experimental)
    for eid in world.net.highway:
        e = world.net.edges[eid]
        hw = e.cls.half_width + 0.3
        for s0 in np.arange(0.0, e.center.length - 1e-3, 8.0):
            s1 = min(s0 + 8.0, e.center.length)
            (x0, y0), (x1, y1) = e.center.point_at(s0), e.center.point_at(s1)
            z0 = float(terrain.height(x0, y0)) + e.z(s0)
            z1 = float(terrain.height(x1, y1)) + e.z(s1)
            if max(z0, z1) - min(float(terrain.height(x0, y0)), float(terrain.height(x1, y1))) < 0.3:
                continue  # at street level the ground carries the car
            ds = math.hypot(x1 - x0, y1 - y0)
            yaw = math.atan2(y1 - y0, x1 - x0)
            pitch = -math.atan2(z1 - z0, ds)
            q = ypr_quat(yaw, pitch, 0.0)
            body_xml.append(f'<geom type="box" size="{ds / 2 + 0.3:.2f} {hw:.2f} 0.4" pos="{(x0 + x1) / 2:.2f} {(y0 + y1) / 2:.2f} '
                            f'{(z0 + z1) / 2 + 0.06 - 0.4:.2f}" quat="{_fmt(q)}" rgba="0 0 0 0" group="{G_HIDDEN}" '
                            f'contype="1" conaffinity="1" friction="1 0.005 0.0001"/>')  # fmt: skip

    # ---- traffic-light lamps: dark lenses baked into one mesh, a pool of lit lamps placed at run time
    if world.signals:
        from ...meshes import icosphere

        lens = [icosphere(0.13, center=xyz) for sh in world.signals for xyz in sh.lamps]
        v, f = merge(lens)
        mesh_asset("signal_lenses", v, f)
        body_xml.append(f'<geom name="signal_lenses" type="mesh" mesh="signal_lenses" rgba="0.12 0.12 0.12 1" '
                        f'contype="0" conaffinity="0" group="{G_PROPS}"/>')  # fmt: skip
        for i in range(LIT_LAMPS):
            body_xml.append(f'<body name="lit{i}" mocap="true" pos="{cx:.1f} {cy:.1f} -60"><geom name="lit{i}" type="sphere" '
                            f'size="0.15" material="red_on" contype="0" conaffinity="0" group="{G_PROPS}"/></body>')  # fmt: skip

    # ---- asset helper for dynamic things
    def parts_of(name: str, lod: str = "high"):
        """(mesh, rgba, tintable, textured) per geom: the paint part alone, everything else merged."""
        parts = assets.parts(name, lod)
        paint = [p for p in parts if p.tintable]
        rest = [p for p in parts if not p.tintable]
        out = []
        for i, p in enumerate(paint):
            mname = f"a_{name}_{lod}_paint{i}"
            mesh_asset(mname, p.vertices, p.faces)
            out.append((mname, p.rgba, True, False))
        if rest:
            mname = f"a_{name}_{lod}_body"
            v, f, uv = palette.merged(rest)
            mesh_asset(mname, v, f, uv)
            out.append((mname, (1, 1, 1, 1), False, True))
        return out

    def mocap(name: str, asset: str, collide: tuple[float, float, float] | None, extra: str = "",
              paint_name: str | None = None, lod: str = "high") -> str:  # fmt: skip
        geoms = []
        for mn, c, tint, textured in parts_of(asset, lod):
            nm = f' name="{paint_name}"' if (tint and paint_name) else ""
            mat = ' material="palette"' if textured else ""
            col = "" if textured else f' rgba="{_rgba(c)}"'
            geoms.append(f'<geom{nm} type="mesh" mesh="{mn}"{col}{mat} contype="0" conaffinity="0" group="{G_VEHICLES}"/>')
        if collide is not None:
            sx, sy, sz = collide
            geoms.append(f'<geom name="{name}_col" type="box" size="{sx / 2:.2f} {sy / 2:.2f} {sz / 2:.2f}" pos="0 0 {sz / 2:.2f}" '
                         f'rgba="0 0 0 0" contype="2" conaffinity="0" group="{G_HIDDEN}"/>')  # fmt: skip
        return f'<body name="{name}" mocap="true" pos="{cx:.1f} {cy:.1f} -60">{"".join(geoms)}{extra}</body>'

    # ---- parked cars: a pool per car type, filled with the occupied kerb spots nearest the camera
    if slots.parked:
        for t in PARKED_TYPES:
            vt = TYPES[t]
            for i in range(PARKED_POOL[t]):
                name = f"parked_{t}_{i}"
                body_xml.append(mocap(name, t, (vt.length, vt.width, 1.6), paint_name=f"{name}_paint"))

    # ---- background traffic pools
    if slots.npc:
        for t, n in pool_sizes(slots.npc).items():
            vt = TYPES[t]
            bar = ""
            if vt.emergency or t == "police":
                h = 3.05 if vt.length > 6 else 1.75
                bar = (f'<geom name="{{n}}_bar_l" type="box" size="0.2 0.3 0.08" pos="{vt.length / 2 - 1.2:.2f} 0.32 {h}" material="bar_red" contype="0" conaffinity="0" group="{G_VEHICLES}"/>'
                       f'<geom name="{{n}}_bar_r" type="box" size="0.2 0.3 0.08" pos="{vt.length / 2 - 1.2:.2f} -0.32 {h}" material="bar_blue" contype="0" conaffinity="0" group="{G_VEHICLES}"/>')  # fmt: skip
            for i in range(n):
                name = f"npc_{t}_{i}"
                body_xml.append(mocap(name, t, (vt.length, vt.width, 1.6 if vt.length < 6 else 3.0),
                                      bar.replace("{n}", name), paint_name=f"{name}_paint"))  # fmt: skip
    for i in range(slots.taxis):
        body_xml.append(mocap(f"taxi{i}", "taxi", (4.6, 1.85, 1.6)))
    for i in range(slots.passengers):
        body_xml.append(mocap(f"pax{i}", "person", None))
    for i in range(min(slots.pedestrians, PED_POOL)):
        body_xml.append(mocap(f"ped{i}", "person", (0.5, 0.5, 1.7), paint_name=f"ped{i}_shirt"))
    if slots.props:
        for asset, n in PROP_POOLS.items():
            for i in range(n):
                body_xml.append(mocap(f"prop_{asset}_{i}", asset, (0.4, 0.4, 0.7) if asset == "traffic_cone" else (0.5, 1.8, 1.0)))

    # ---- ego car
    actuators = ""
    if slots.ego:
        vis = "".join(f'<geom type="mesh" mesh="{mn}" {"material=" + chr(34) + "palette" + chr(34) if tex else "rgba=" + chr(34) + _rgba(c if not tint else (0.92, 0.93, 0.95, 1)) + chr(34)} '
                      f'contype="0" conaffinity="0" mass="0" group="{G_VEHICLES}"/>' for mn, c, tint, tex in parts_of("sedan"))  # fmt: skip
        lights = ""  # no spot lights: in MuJoCo's renderer an active spot light darkens the rest of the scene
        body_xml.append(ego_xml("ego", vis, slots.ego_cameras, lights))
        actuators = f"<actuator>{ego_actuators('ego')}</actuator>"

    palette.save(os.path.join(mesh_dir, "palette.png"))
    lamp_mats = "".join(
        f'<material name="{c}_on" rgba="{_fmt(rgb)} 1" emission="1.5"/><material name="{c}_off" rgba="{_fmt(np.array(rgb) * 0.18)} 1"/>'
        for c, rgb in LAMP_COLORS.items())  # fmt: skip
    xml = f"""<mujoco model="metatraffic3d">
  <compiler angle="radian" meshdir="meshes" texturedir="meshes"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <size memory="1G"/>
  <statistic center="{cx:.1f} {cy:.1f} 0" extent="100"/>
  <visual>
    <global offwidth="1920" offheight="1080" azimuth="-60" elevation="-35"/>
    <map znear="0.003" zfar="70" fogstart="15" fogend="60"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.42 0.42 0.45" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <rgba fog="0.78 0.84 0.92 1" haze="0.78 0.84 0.92 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.70 0.90" rgb2="0.92 0.95 1.0" width="256" height="256"/>
    <material name="matte" specular="0.05" shininess="0.1"/>
    <texture name="palette" type="2d" file="palette.png"/>
    <material name="palette" texture="palette" specular="0.15" shininess="0.3"/>
    <material name="bar_red" rgba="0.9 0.05 0.05 1" emission="1"/><material name="bar_blue" rgba="0.1 0.2 1 1" emission="1"/>
    <material name="bar_dark" rgba="0.15 0.15 0.18 1"/>
    {lamp_mats}
    {chr(10).join("    " + a for a in asset_xml).strip()}
  </asset>
  <worldbody>
    <light name="sun" directional="true" castshadow="true" pos="{cx:.1f} {cy:.1f} 800" dir="-0.35 0.25 -1" diffuse="0.75 0.73 0.68" specular="0.1 0.1 0.1"/>
    {chr(10).join("    " + b for b in body_xml).strip()}
  </worldbody>
  {actuators}
</mujoco>
"""
    path = os.path.join(out_dir, "scene.xml")
    with open(path, "w") as fh:
        fh.write(xml)
    return path


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def ypr_quat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for z-y-x Euler angles; pitch + = nose down (MuJoCo y axis)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])  # fmt: skip
