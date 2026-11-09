"""2D mapgen city -> 3D world: terrain, districts, roads with lanes and kerbs, junction control,
buildings, parks, water and bridges.

Everything is procedural and seeded. The city's global character (terrain style, coast,
river, district mix, traffic level) comes from ``profile.sample_profile`` unless pinned in the
config. The world is backend-neutral: triangle meshes (``Scene``), props (trees, lights, signs)
and signal heads whose lamps the backends light up. Vehicles and people live in ``traffic``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import shapely
import shapely.affinity
from shapely import LineString, Point, Polygon
from shapely import box as shp_box
from shapely.prepared import prep

from mapgen import CityConfig, CityMap, generate_city

from .assets import AssetLibrary
from .config import RAMP_LENGTH, SimConfig
from .geometry import Polyline, offset_points
from .heights import Building, draw_buildings
from .meshes import Mesh, Scene, box, cylinder, extrude, merge, ribbon, transform, triangulate, walls
from .profile import TERRAIN_STYLES, CityProfile, sample_profile
from .roads import RoadNetwork, build_network, to_metres
from .terrain import Terrain, TerrainConfig, drape
from .traffic.control import assign_controls
from .zoning import Zoning

COLORS = {
    "ground": (0.72, 0.72, 0.67, 1.0),
    "asphalt": (0.23, 0.24, 0.26, 1.0),
    "parking": (0.31, 0.32, 0.34, 1.0),
    "sidewalk": (0.68, 0.67, 0.64, 1.0),
    "grass": (0.43, 0.61, 0.33, 1.0),
    "white": (0.93, 0.93, 0.91, 1.0),
    "yellow": (0.93, 0.76, 0.20, 1.0),
    "water": (0.22, 0.42, 0.62, 1.0),
    "embankment": (0.55, 0.52, 0.48, 1.0),
    "bridge": (0.62, 0.60, 0.57, 1.0),
    "railing": (0.40, 0.42, 0.45, 1.0),
    "pole": (0.25, 0.27, 0.29, 1.0),
    "signal_housing": (0.12, 0.13, 0.14, 1.0),
    "stop_red": (0.78, 0.08, 0.08, 1.0),
}
STYLE_COLORS = {
    "house": [(0.93, 0.91, 0.86), (0.86, 0.80, 0.68), (0.74, 0.82, 0.88), (0.92, 0.86, 0.66), (0.70, 0.45, 0.36)],
    "roof": [(0.55, 0.22, 0.18), (0.40, 0.30, 0.24), (0.33, 0.35, 0.38), (0.25, 0.32, 0.27)],
    "apartment": [(0.82, 0.78, 0.70), (0.72, 0.66, 0.60), (0.80, 0.80, 0.80), (0.66, 0.52, 0.44)],
    "shop": [(0.88, 0.84, 0.76), (0.80, 0.74, 0.66), (0.78, 0.60, 0.48), (0.86, 0.86, 0.86)],
    "office": [(0.74, 0.75, 0.77), (0.66, 0.68, 0.72), (0.80, 0.78, 0.73)],
    "tower": [(0.50, 0.60, 0.70), (0.60, 0.66, 0.73), (0.42, 0.49, 0.58), (0.70, 0.72, 0.74)],
    "warehouse": [(0.75, 0.77, 0.78), (0.62, 0.68, 0.74), (0.80, 0.76, 0.66)],
    "awning": [(0.75, 0.18, 0.16), (0.16, 0.42, 0.30), (0.18, 0.30, 0.62), (0.85, 0.60, 0.15)],
}
GLASS = (0.20, 0.27, 0.34)
Z_ROAD, Z_PARK, Z_MARK, KERB = 0.06, 0.065, 0.09, 0.2
MASS_PROPS = ("streetlight", "stop_sign")  # thousands of copies: use the light procedural models


@dataclass
class Prop:
    asset: str
    x: float
    y: float
    yaw: float
    scale: float = 1.0
    z: float = 0.0  # height above the ground at (x, y)


@dataclass
class SignalHead:
    node: int
    de_in: int  # the approach this head faces
    lamps: list[tuple[float, float, float]]  # red, yellow, green lamp centres (world)
    yaw: float  # direction the lamps face


@dataclass
class World:
    cfg: SimConfig
    profile: CityProfile
    city: CityMap
    net: RoadNetwork
    terrain: Terrain
    zoning: Zoning
    buildings: list[Building]
    land: shapely.Geometry
    sea: shapely.Geometry | None
    river: shapely.Geometry | None
    parks: list[Polygon]
    roads: shapely.Geometry  # paved carriageway footprint
    sidewalks: shapely.Geometry
    scene: Scene
    props: list[Prop] = field(default_factory=list)
    signals: list[SignalHead] = field(default_factory=list)
    controls: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.scene.bounds

    def height(self, x, y):
        return self.terrain.height(x, y)

    def summary(self) -> dict:
        zones: dict[str, int] = {}
        styles: dict[str, int] = {}
        for b in self.buildings:
            zones[b.zone] = zones.get(b.zone, 0) + 1
            styles[b.style] = styles.get(b.style, 0) + 1
        return {"seed": self.cfg.map.seed, "profile": self.profile.as_dict(), "terrain": self.terrain.stats(),
                "network": self.net.summary(), "junction_control": self.controls, "zones": zones,
                "building_styles": styles, "props": len(self.props), "signal_heads": len(self.signals),
                "scene": self.scene.stats(), "timings": self.timings}  # fmt: skip


def generate_map(cfg: SimConfig, profile: CityProfile) -> CityMap:
    mc = cfg.map
    return generate_city(CityConfig(seed=mc.seed, width=mc.width, height=mc.height, zoom=mc.zoom, water=mc.water,
                                    coast=profile.coast, river=profile.river))  # fmt: skip


def build_world(cfg: SimConfig | None = None, city: CityMap | None = None, with_meshes: bool = True,
                assets: AssetLibrary | None = None) -> World:  # fmt: skip
    """Build the city from ``cfg.map.seed``. ``with_meshes=False`` skips 3D geometry (training).

    ``assets`` (default ``AssetLibrary("auto")``) supplies Poly Pizza houses and shops for lots they
    fit; without downloaded models every building is procedural.
    """
    cfg = cfg or SimConfig()
    assets = assets or AssetLibrary("auto")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal t0
        now = time.perf_counter()
        timings[name] = round(now - t0, 3)
        t0 = now

    seed = cfg.map.seed
    profile = sample_profile(seed, cfg.profile)
    city = city or generate_map(cfg, profile)
    lap("mapgen")
    m = cfg.map.metres_per_unit
    net = build_network(city, m, cfg.network, cfg.parking, cfg.traffic, highway=profile.highway)
    lap("network")

    ox, oy = city.domain_origin
    dw, dh = city.domain_dims
    domain = shp_box(ox * m, -(oy + dh) * m, (ox + dw) * m, -oy * m)
    river = net.river.intersection(domain) if net.river is not None else None
    if river is not None and river.is_empty:
        river = None
    sea = None
    if len(city.sea) >= 3:
        sea = Polygon(to_metres(city.sea, m)).buffer(0)
        if len(city.coastline) >= 2:
            sea = sea.difference(LineString(to_metres(city.coastline, m)).buffer(cfg.world.coast_strip * m))
        sea = sea.intersection(domain)
        if sea.is_empty:
            sea = None
    water = shapely.union_all([g for g in (sea, river) if g is not None]) if (sea or river) else None
    land = domain.difference(water) if water is not None else domain

    amp, density = TERRAIN_STYLES[profile.terrain]
    terrain = Terrain(domain.bounds, water, seed, TerrainConfig(amplitude=amp, hills_per_km2=density))
    lap("terrain")

    # ---- districts and junction control
    main_lines = [LineString(e.center.pts) for e in net.edges if e.cls.name == "main"]
    centres = [tuple(to_metres([c], m)[0]) for c in city.radial_centres]
    zoning = Zoning(domain.bounds, centres, main_lines, water, seed, profile.zone_mix)
    node_zone, _ = zoning.classify(np.array([n.pos for n in net.nodes]))
    controls = assign_controls(net, {n.id: z for n, z in zip(net.nodes, node_zone)}, seed)
    lap("zoning")

    # ---- road surfaces (2D footprints)
    drive_parts, full_parts, walk_parts = [], [], []
    for e in net.edges:
        if e.deck > 0:
            continue  # the elevated highway is drawn separately
        line = LineString(e.center.pts)
        c = e.cls
        drive_parts.append(line.buffer(c.median / 2 + c.drive_lanes * c.lane_width, cap_style="flat", join_style="round"))
        full_parts.append(line.buffer(c.half_width, cap_style="flat", join_style="round"))
        walk_parts.append(line.buffer(c.half_width + c.sidewalk, cap_style="round", join_style="round"))
    discs = []
    for node in net.nodes:
        if node.degree >= 2:
            hw = max(net.edges[i].cls.half_width for i in node.edges if net.edges[i].deck <= 0)
            discs.append(Point(node.pos).buffer(hw * 1.15, quad_segs=6))
    asphalt = shapely.union_all(drive_parts + discs)
    roads = shapely.union_all([asphalt] + full_parts)
    parking = roads.difference(asphalt)
    sidewalks = shapely.union_all(walk_parts).difference(roads)
    if sea is not None:
        sidewalks, roads, asphalt, parking = (g.difference(sea) for g in (sidewalks, roads, asphalt, parking))
    parks_m = []
    for p in city.parks:
        if len(p) >= 3:
            poly = Polygon(to_metres(p, m)).buffer(0)
            if not poly.is_empty:
                parks_m.append(poly)
    park_area = shapely.union_all(parks_m).difference(roads.union(sidewalks)) if parks_m else None
    if park_area is not None and water is not None:
        park_area = park_area.difference(water)
    paved = shapely.union_all([roads, sidewalks] + ([park_area] if park_area is not None else []))
    lap("road_polygons")

    # ---- buildings
    lots = _lots(city, m, water, paved)
    cents = shapely.get_coordinates(shapely.centroid(np.asarray(lots, dtype=object))) if lots else np.zeros((0, 2))
    lot_zone, fields = zoning.classify(cents) if lots else (np.array([]), {"intensity": np.array([])})
    buildings = draw_buildings(lots, lot_zone, fields["intensity"], seed, cfg.buildings)
    lap("buildings")

    meshes: list[Mesh] = []
    props: list[Prop] = []
    signals: list[SignalHead] = []
    if with_meshes:
        H = terrain

        def add(name, vf, color, group, lift=True):
            v, f = vf
            if len(f):
                v = drape(np.asarray(v, float), H) if lift else np.asarray(v, float)
                meshes.append(Mesh(name, v, np.asarray(f, np.int64), color, group))

        hilly = terrain.cfg.amplitude > 0

        def dense(g):  # extra vertices only where the ground is not flat
            return shapely.segmentize(g, 6.0) if hilly else g

        add("asphalt", triangulate(dense(asphalt), Z_ROAD), COLORS["asphalt"], "road")
        add("parking_strip", triangulate(dense(parking), Z_PARK), COLORS["parking"], "road")
        sw = dense(sidewalks)
        add("sidewalk", merge([triangulate(sw, KERB), walls(sw, 0.0, KERB)]), COLORS["sidewalk"], "sidewalk")
        whites, yellows = _markings(net, hilly)
        add("markings_white", whites, COLORS["white"], "marking")
        add("markings_yellow", yellows, COLORS["yellow"], "marking")
        lap("road_meshes")
        if hilly:
            ground, grass = _terrain_meshes(terrain, land, park_area)
            add("ground", ground, COLORS["ground"], "ground", lift=False)
            add("grass", grass, COLORS["grass"], "park", lift=False)
        else:
            add("ground", triangulate(land.difference(paved), 0.0), COLORS["ground"], "ground", lift=False)
            if park_area is not None:
                add("grass", triangulate(park_area, 0.0), COLORS["grass"], "park", lift=False)
        lap("ground")
        if sea is not None:
            add("sea", triangulate(sea, -cfg.world.sea_depth), COLORS["water"], "water", lift=False)
            add("sea_wall", walls(shapely.segmentize(sea, 10.0), -cfg.world.sea_depth - 0.3, 0.0, inward=True),
                COLORS["embankment"], "water")  # fmt: skip
        if river is not None:
            add("river", triangulate(river, -cfg.world.river_depth), COLORS["water"], "water", lift=False)
            add("river_wall", walls(shapely.segmentize(river, 10.0), -cfg.world.river_depth - 0.3, 0.0, inward=True),
                COLORS["embankment"], "water")  # fmt: skip
            deck, rails, piers = _bridges(net, river, roads, sidewalks, cfg)
            add("bridge_deck", deck, COLORS["bridge"], "bridge")
            add("bridge_piers", piers, COLORS["bridge"], "bridge", lift=False)
            add("bridge_railings", rails, COLORS["railing"], "bridge")
        lap("water")
        meshes.extend(_building_meshes(buildings, seed, terrain, assets, net))
        lap("building_meshes")
        props = _park_trees(park_area, cfg, seed) if park_area is not None else []
        props += _street_props(net, cfg, water)
        for name, vf, color in _highway_meshes(net, terrain):
            add(name, vf, color, "bridge" if "deck" in name or "pier" in name else "road", lift=False)
        sig_meshes, signals, stop_props = _junction_furniture(net, terrain)
        props += stop_props
        for name, vf, color in sig_meshes:
            add(name, vf, color, "prop", lift=False)
        lap("props")

    scene = Scene(meshes=meshes, bounds=domain.bounds)
    return World(cfg, profile, city, net, terrain, zoning, buildings, land, sea, river, parks_m, roads, sidewalks,
                 scene, props, signals, controls, timings)  # fmt: skip


# ---------------------------------------------------------------------- lots
def _lots(city: CityMap, m: float, water, paved) -> list[Polygon]:
    lots = []
    paved_clear = paved.buffer(1.0)
    blocked = prep(paved_clear)
    wet = prep(water) if water is not None else None
    for lot in city.lots:
        if len(lot) < 3:
            continue
        poly = Polygon(to_metres(lot, m))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 20:
            continue
        if wet is not None and wet.intersects(poly):
            continue
        if blocked.intersects(poly):  # wide roads eat into the lot: keep what is left
            area0 = poly.area
            poly = poly.difference(paved_clear)
            if poly.is_empty:
                continue
            if poly.geom_type != "Polygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            if poly.area < max(40.0, 0.35 * area0):
                continue
        lots.append(poly if poly.geom_type == "Polygon" else max(poly.geoms, key=lambda g: g.area))
    return lots


# ---------------------------------------------------------------------- terrain mesh
def _terrain_meshes(terrain: Terrain, land, park_area, spacing: float = 10.0):
    """Ground and grass triangles on a regular grid, clipped to land at the water's edge."""
    xs, ys, hgt = terrain.grid(spacing)
    nx, ny = len(xs), len(ys)
    gx, gy = np.meshgrid(xs, ys)
    inside = shapely.contains_xy(land, gx, gy)
    vid = np.arange(nx * ny).reshape(ny, nx)
    verts = np.column_stack([gx.ravel(), gy.ravel(), hgt.ravel()])
    a, b, c, d = vid[:-1, :-1], vid[:-1, 1:], vid[1:, 1:], vid[1:, :-1]
    full = inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, 1:] & inside[1:, :-1]
    tris = np.vstack([np.column_stack([a[full], b[full], c[full]]), np.column_stack([a[full], c[full], d[full]])])
    cen = verts[tris].mean(1)
    grass_mask = shapely.contains_xy(park_area, cen[:, 0], cen[:, 1]) if park_area is not None else np.zeros(len(tris), bool)
    ground = [(verts, tris[~grass_mask])]
    grass = [(verts, tris[grass_mask])]
    partial = (inside[:-1, :-1] | inside[:-1, 1:] | inside[1:, 1:] | inside[1:, :-1]) & ~full
    for j, i in zip(*np.nonzero(partial)):
        cell = shp_box(xs[i], ys[j], xs[i + 1], ys[j + 1]).intersection(land)
        if cell.is_empty or cell.area < 0.5:
            continue
        v, f = triangulate(cell, 0.0)
        if len(f):
            v[:, 2] = terrain.height(v[:, 0], v[:, 1])
            ground.append((v, f))
    return merge(ground), merge(grass)


# ---------------------------------------------------------------------- markings
def _densify(pts: np.ndarray, step: float = 5.0) -> np.ndarray:
    pl = Polyline(pts)
    if pl.length <= step:
        return pl.pts
    return pl.resample(step)


def _dashes(pts: np.ndarray, dash: float, gap: float) -> list[np.ndarray]:
    pl = Polyline(pts)
    out, s = [], gap / 2
    while s + dash <= pl.length:
        out.append(pl.sub(s, s + dash))
        s += dash + gap
    return out


def _markings(net: RoadNetwork, hilly: bool = False):
    white, yellow = [], []
    for e in net.edges:
        if e.deck > 0:
            continue
        c = e.cls
        de = net.dedges[2 * e.id]
        if de.length < 2.0:
            continue
        densify = _densify if hilly else (lambda p: p)
        centre = densify(de.center.sub(de.s0, de.s1))
        if c.name == "main":
            for off in (0.18 + c.median / 2 - 0.3, -(0.18 + c.median / 2 - 0.3)):
                yellow.append(ribbon(offset_points(centre, off), 0.14, Z_MARK))
        else:
            for piece in _dashes(centre, 3.0, 4.0):
                yellow.append(ribbon(piece, 0.14, Z_MARK))
        for d in (0, 1):
            ded = net.dedges[2 * e.id + d]
            ctr = densify(ded.center.sub(ded.s0, ded.s1))
            for k in range(1, c.drive_lanes):  # dashed lane dividers
                for piece in _dashes(offset_points(ctr, c.median / 2 + k * c.lane_width), 3.0, 6.0):
                    white.append(ribbon(piece, 0.12, Z_MARK))
            if c.parking:  # solid line between the driving lane and the kerb parking lane
                white.append(ribbon(offset_points(ctr, c.median / 2 + c.drive_lanes * c.lane_width), 0.1, Z_MARK))
    for node in net.nodes:
        if node.degree < 3:
            continue
        ctl = node.control
        for de_id in node.out_dedges:  # zebra crossings on every leg
            de = net.dedges[de_id]
            c = de.cls
            if c.sidewalk <= 0:
                continue
            x, y, h = de.center.pose_at(max(de.s0 - 2.5, 0.5))
            nx, ny = math.sin(h), -math.cos(h)
            for off in np.arange(-c.half_width + 0.6, c.half_width - 0.3, 1.0):
                px, py = x + nx * off, y + ny * off
                a = (px - math.cos(h) * 1.5, py - math.sin(h) * 1.5)
                b = (px + math.cos(h) * 1.5, py + math.sin(h) * 1.5)
                white.append(ribbon(np.array([a, b]), 0.5, Z_MARK))
        if ctl is not None and ctl.kind != "none":
            for de_id in node.in_dedges:  # stop lines across the approach lanes
                if ctl.kind == "two_stop" and de_id not in ctl.stop_approaches:
                    continue
                de = net.dedges[de_id]
                c = de.cls
                x, y, h = de.center.pose_at(max(de.s1 - 0.4, 0.0))
                nx, ny = math.sin(h), -math.cos(h)
                inner, outer = c.median / 2 + 0.2, c.median / 2 + c.drive_lanes * c.lane_width
                white.append(ribbon(np.array([(x + nx * inner, y + ny * inner), (x + nx * outer, y + ny * outer)]), 0.45, Z_MARK))
    return merge(white), merge(yellow)


# ---------------------------------------------------------------------- elevated highway (experimental)
def _strip(left: np.ndarray, right: np.ndarray):
    """Quads between two 3D polylines of equal length (left/right edges of a surface)."""
    n = len(left)
    v = np.vstack([left, right])
    i = np.arange(n - 1)
    f = np.vstack([np.column_stack([i, i + n, i + n + 1]), np.column_stack([i, i + n + 1, i + 1])])
    return v, f


def _highway_meshes(net: RoadNetwork, terrain: Terrain):
    out_deck, out_side, out_mark, out_barrier, out_pier = [], [], [], [], []
    for eid in net.highway:
        e = net.edges[eid]
        c = e.cls
        s = np.arange(0.0, e.center.length, 4.0)
        s = np.append(s, e.center.length)
        pts = np.array([e.center.point_at(x) for x in s])
        hd = np.array([e.center.heading_at(x) for x in s])
        nx, ny = np.sin(hd), -np.cos(hd)  # right normal
        z = terrain.height(pts[:, 0], pts[:, 1]) + np.array([e.z(x) for x in s])

        def edge(off, dz=0.0, pts=pts, nx=nx, ny=ny, z=z):
            return np.column_stack([pts[:, 0] + nx * off, pts[:, 1] + ny * off, z + dz])

        hw = c.half_width
        out_deck.append(_strip(edge(-hw, 0.06), edge(hw, 0.06)))  # driving surface
        out_side.append(_strip(edge(hw + 0.5, -1.1), edge(-hw - 0.5, -1.1)))  # underside
        for off in (-hw - 0.5, hw + 0.5):
            out_side.append(_strip(edge(off, -1.1), edge(off, 0.05)) if off > 0 else _strip(edge(off, 0.05), edge(off, -1.1)))
            inner = off - 0.35 if off > 0 else off + 0.35
            out_barrier.append(_strip(edge(inner, 0.95), edge(off, 0.95)))  # jersey barrier top
            out_barrier.append(_strip(edge(inner, 0.05), edge(inner, 0.95)) if off < 0 else _strip(edge(inner, 0.95), edge(inner, 0.05)))
        for off in (-c.median / 2 + 0.2, c.median / 2 - 0.2):
            out_mark.append(_strip(edge(off - 0.07, 0.09), edge(off + 0.07, 0.09)))
        for k in range(1, c.drive_lanes):
            for side in (-1, 1):
                off = side * (c.median / 2 + k * c.lane_width)
                dash = (np.arange(len(s)) // 2) % 2 == 0  # 8 m on, 8 m off
                lft, rgt = edge(off - 0.07, 0.09), edge(off + 0.07, 0.09)
                for i in np.flatnonzero(dash[:-1]):
                    out_mark.append(_strip(lft[i:i + 2], rgt[i:i + 2]))
        for x in np.arange(RAMP_LENGTH * 0.6, e.center.length - RAMP_LENGTH * 0.6, 36.0):
            px, py, h = e.center.pose_at(x)
            ground = float(terrain.height(px, py))
            top = ground + e.z(x) - 1.1
            if top - ground < 2.0:
                continue
            v, f = box(1.8, hw * 1.2, top - ground, center=(0, 0, (top - ground) / 2))
            out_pier.append((transform(v, px, py, ground, h), f))
    meshes = []
    if out_deck:
        meshes += [("highway_deck_surface", merge(out_deck), COLORS["asphalt"]),
                   ("highway_deck_structure", merge(out_side), COLORS["bridge"]),
                   ("highway_markings", merge(out_mark), COLORS["white"]),
                   ("highway_barriers", merge(out_barrier), (0.72, 0.72, 0.70, 1.0)),
                   ("highway_piers", merge(out_pier), COLORS["bridge"])]  # fmt: skip
    return meshes


# ---------------------------------------------------------------------- junction furniture
def _junction_furniture(net: RoadNetwork, terrain: Terrain):
    """Signal poles with heads (lamps lit by the backend) and stop signs, at the kerb by each stop line."""
    poles, housings = [], []
    signals: list[SignalHead] = []
    stop_props: list[Prop] = []
    for node in net.nodes:
        ctl = node.control
        if ctl is None or ctl.kind == "none":
            continue
        for de_id in node.in_dedges:
            de = net.dedges[de_id]
            c = de.cls
            x, y, h = de.center.pose_at(max(de.s1 - 1.0, 0.0))
            nx, ny = math.sin(h), -math.cos(h)  # right normal
            px, py = x + nx * (c.half_width + 0.9), y + ny * (c.half_width + 0.9)
            z0 = float(terrain.height(px, py)) + KERB
            if ctl.kind == "signal":
                arm = min(c.half_width * 0.7, 9.0) if c.drive_lanes > 1 else 0.0
                pole_h = 6.2 if arm else 3.6
                poles.append(cylinder(0.13, pole_h, n=8, center=(px, py, z0 + pole_h / 2)))
                hx, hy = px - nx * arm, py - ny * arm  # head hangs over the lanes on a mast arm
                if arm:
                    length = math.hypot(hx - px, hy - py)
                    v, f = box(length, 0.16, 0.16)
                    poles.append((transform(v, (px + hx) / 2, (py + hy) / 2, z0 + pole_h - 0.3, math.atan2(hy - py, hx - px)), f))
                hz = z0 + (pole_h - 1.1 if arm else pole_h - 0.2)
                v, f = box(0.32, 0.42, 1.15)
                housings.append((transform(v, hx, hy, hz, h), f))
                face = (-math.cos(h) * 0.2, -math.sin(h) * 0.2)  # lamps on the side facing traffic
                lamps = [(hx + face[0], hy + face[1], hz + dz) for dz in (0.35, 0.0, -0.35)]
                signals.append(SignalHead(node.id, de_id, lamps, h + math.pi))
            elif ctl.must_stop(de_id):
                stop_props.append(Prop("stop_sign", px, py, h + math.pi, 1.0, KERB))
    meshes = []
    if poles:
        meshes.append(("signal_poles", merge(poles), COLORS["pole"]))
    if housings:
        meshes.append(("signal_housings", merge(housings), COLORS["signal_housing"]))
    return meshes, signals, stop_props


# ---------------------------------------------------------------------- bridges
def _bridges(net: RoadNetwork, river, roads, sidewalks, cfg: SimConfig):
    deck_area = shapely.union_all([roads, sidewalks]).intersection(river)
    if deck_area.is_empty:
        empty = (np.zeros((0, 3)), np.zeros((0, 3), np.int64))
        return empty, empty, empty
    deck = walls(deck_area, -1.2, 0.0)
    edge_lines = deck_area.boundary.intersection(river.buffer(-0.5))
    rails = extrude(edge_lines.buffer(0.15, cap_style="flat"), KERB, 1.25)
    piers = []
    for e in net.edges:
        for s0, s1 in e.bridge:
            for s in np.arange(s0 + 18.0, s1 - 10.0, 30.0):
                x, y, h = e.center.pose_at(s)
                depth = cfg.world.river_depth + 0.5 - 1.2
                v, f = box(2.0, (e.cls.half_width * 2 + 2 * e.cls.sidewalk) * 0.8, depth, center=(0, 0, -1.2 - depth / 2))
                piers.append((transform(v, x, y, 0.0, h), f))
    return deck, rails, merge(piers)


# ---------------------------------------------------------------------- buildings
def _gabled_roof(rect: Polygon, z_eave: float, rise: float):
    """Pitched roof over a rectangle, ridge along its long side."""
    pts = np.asarray(rect.exterior.coords)[:4]
    e0, e1 = pts[1] - pts[0], pts[2] - pts[1]
    if np.hypot(*e0) < np.hypot(*e1):
        pts = np.roll(pts, -1, axis=0)
    p0, p1, p2, p3 = pts  # p0->p1 long side
    m03, m12 = (p0 + p3) / 2, (p1 + p2) / 2
    v = np.array([[*p0, z_eave], [*p1, z_eave], [*p2, z_eave], [*p3, z_eave],
                  [*m03, z_eave + rise], [*m12, z_eave + rise]])  # fmt: skip
    f = np.array([[0, 1, 5], [0, 5, 4], [2, 3, 4], [2, 4, 5], [1, 2, 5], [3, 0, 4]])
    # make every face point outward/upward
    cen = v.mean(0)
    for i, (a, b, c) in enumerate(f):
        n = np.cross(v[b] - v[a], v[c] - v[a])
        if np.dot(n, v[[a, b, c]].mean(0) - cen) < 0:
            f[i] = [a, c, b]
    return v, f


def _bands(fp, z0: float, z1: float, floor_h: float, start: float):
    """Dark window bands, one per floor, slightly proud of the facade."""
    ring = fp.buffer(0.06, join_style="mitre")
    out = []
    z = z0 + start
    while z + 1.6 < z1:
        out.append(walls(ring, z + 0.8, z + 2.3))
        z += floor_h
    return out


MODEL_STYLES = {"house": ("house_a", "house_b", "house_c"), "shop": ("shop_a", "shop_b")}


def _street_facing(net: RoadNetwork):
    tree = shapely.STRtree([LineString(e.center.pts) for e in net.edges])
    lines = np.asarray([LineString(e.center.pts) for e in net.edges], dtype=object)

    def facing(pt) -> float:
        line = lines[int(tree.nearest(pt))]
        q = line.interpolate(line.project(pt))
        return math.atan2(q.y - pt.y, q.x - pt.x)

    return facing


def _place_model(b: Building, assets: AssetLibrary, rng, facing, terrain: Terrain):
    """Put a Poly Pizza house/shop on the lot if one fits, facing the street; else None."""
    names = [n for n in MODEL_STYLES.get(b.style, ()) if assets.used.get(n) != "procedural"]
    if not names:
        return None
    name = names[int(rng.integers(len(names)))]
    try:
        parts = assets.parts(name)
    except (KeyError, FileNotFoundError):
        return None
    if assets.used.get(name) != "polypizza":
        return None
    allv = np.vstack([p.vertices for p in parts])
    mx, my = np.ptp(allv[:, 0]), np.ptp(allv[:, 1])
    fp = b.footprint
    scale = min(1.25, 0.9 * math.sqrt(fp.area / max(mx * my, 1e-3)))
    if scale < 0.7:
        return None
    c = fp.centroid
    yaw = facing(c)
    box_fp = shapely.affinity.rotate(shp_box(-mx / 2 * scale, -my / 2 * scale, mx / 2 * scale, my / 2 * scale),
                                     yaw, use_radians=True, origin=(0, 0))  # fmt: skip
    box_fp = shapely.affinity.translate(box_fp, c.x, c.y)
    if not fp.buffer(1.5).contains(box_fp):
        return None
    xy = np.asarray(box_fp.exterior.coords)
    z = float(np.min(terrain.height(xy[:, 0], xy[:, 1]))) - 0.05
    return [(transform(p.vertices, c.x, c.y, z, yaw, scale), p.faces, p.rgba) for p in parts]


def _building_meshes(buildings: list[Building], seed: int, terrain: Terrain, assets: AssetLibrary | None = None,
                     net: RoadNetwork | None = None) -> list[Mesh]:  # fmt: skip
    rng = np.random.default_rng(seed + 7)
    buckets: dict[tuple, list] = {}
    facing = _street_facing(net) if (assets is not None and net is not None) else None

    def put(key, color, vf):
        buckets.setdefault((key, tuple(color)), []).append(vf)

    for b in buildings:
        fp = b.footprint
        xy = np.asarray(fp.exterior.coords)
        zs = terrain.height(xy[:, 0], xy[:, 1])
        base = float(np.min(zs)) - 0.6
        top = float(np.max(zs)) + b.height
        pal = STYLE_COLORS[b.style]
        color = pal[int(rng.integers(len(pal)))]
        if facing is not None and b.style in MODEL_STYLES and rng.random() < 0.85:
            placed = _place_model(b, assets, rng, facing, terrain)
            if placed is not None:
                for v, f, rgba in placed:
                    put("model", rgba[:3], (v, f))
                continue
        if b.style == "house":
            rect = fp.minimum_rotated_rectangle
            body = rect if rect.area < fp.area * 1.25 and fp.buffer(0.5).contains(rect) else fp
            body = body.buffer(-1.2, join_style="mitre")
            if body.is_empty or body.area < 25:
                body = fp
            eave = top - 0.0
            put("house", color, extrude(body, base, eave))
            r = body.minimum_rotated_rectangle
            if r.geom_type == "Polygon":
                roof_c = STYLE_COLORS["roof"][int(rng.integers(len(STYLE_COLORS["roof"])))]
                put("roof", roof_c, _gabled_roof(r.buffer(0.4, join_style="mitre"), eave, float(rng.uniform(1.8, 3.2))))
        elif b.style == "shop":
            put("shop", color, extrude(fp, base, top))
            put("glass", GLASS, walls(fp.buffer(0.05, join_style="mitre"), base + 0.6 + (top - b.height - base), top - b.height + 3.3))
            aw = STYLE_COLORS["awning"][int(rng.integers(len(STYLE_COLORS["awning"])))]
            put("awning", aw, extrude(fp.buffer(0.9, join_style="mitre").difference(fp), top - b.height + 3.3, top - b.height + 3.6))
            for vf in _bands(fp, top - b.height + 4.5, top, 3.2, 0.0):
                put("glass", GLASS, vf)
        elif b.style in ("office", "apartment"):
            put(b.style, color, extrude(fp, base, top))
            for vf in _bands(fp, top - b.height, top, 3.2 if b.style == "office" else 6.4, 0.0):
                put("glass", GLASS, vf)
            inset = fp.buffer(-max(1.0, 0.15 * math.sqrt(fp.area)), join_style="mitre")
            if not inset.is_empty and inset.area > 8:
                put("roofunit", (0.55, 0.56, 0.58), extrude(inset, top, top + 1.2))
        elif b.style == "tower":
            podium_top = top - b.height + 4.5 + 2 * 3.2
            put("tower", color, extrude(fp, base, podium_top))
            for vf in _bands(fp, top - b.height, podium_top, 3.2, 0.0):
                put("glass", GLASS, vf)
            tower = fp.buffer(-min(6.0, 0.18 * math.sqrt(fp.area)), join_style="mitre")
            if tower.is_empty or tower.area < 60:
                tower = fp
            if tower.geom_type != "Polygon":
                tower = max(tower.geoms, key=lambda g: g.area)
            put("tower", color, extrude(tower, podium_top, top))
            for vf in _bands(tower, podium_top, top, 3.6, 0.4):
                put("glass", GLASS, vf)
            crown = tower.buffer(-max(1.5, 0.25 * math.sqrt(tower.area)), join_style="mitre")
            if not crown.is_empty and crown.area > 10:
                put("roofunit", (0.45, 0.47, 0.50), extrude(crown, top, top + 3.0))
        else:  # warehouse
            put("warehouse", color, extrude(fp, base, top))
            put("roofunit", (0.58, 0.60, 0.62), extrude(fp.buffer(-0.4, join_style="mitre"), top, top + 0.4))
    out = []
    for i, ((key, color), parts) in enumerate(sorted(buckets.items())):
        v, f = merge(parts)
        out.append(Mesh(f"bld_{key}_{i}", v, f, (*color, 1.0), "building"))
    return out


# ---------------------------------------------------------------------- props
def _park_trees(park_area, cfg: SimConfig, seed: int) -> list[Prop]:
    rng = np.random.default_rng(seed + 13)
    density = cfg.world.trees_per_100m2 / 100.0
    if density <= 0 or park_area.is_empty:
        return []
    spacing = 1.0 / math.sqrt(density)
    xmin, ymin, xmax, ymax = park_area.bounds
    gx, gy = np.meshgrid(np.arange(xmin, xmax, spacing), np.arange(ymin, ymax, spacing))
    pts = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(0, spacing, (gx.size, 2))
    inner = park_area.buffer(-cfg.world.tree_clearance)
    keep = shapely.contains_xy(inner, pts[:, 0], pts[:, 1]) if pts.size else []
    props = []
    for x, y in pts[np.asarray(keep, bool)]:
        kind = "tree_round" if rng.random() < 0.7 else "tree_cone"
        props.append(Prop(kind, float(x), float(y), float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(0.75, 1.3))))
    return props


def _street_props(net: RoadNetwork, cfg: SimConfig, water) -> list[Prop]:
    rng = np.random.default_rng(cfg.map.seed + 17)
    wet = prep(water.buffer(1.0)) if water is not None else None
    props = []
    for e in net.edges:
        c = e.cls
        for d in (0, 1):
            de = net.dedges[2 * e.id + d]
            if de.length < 20:
                continue
            off = c.half_width + c.sidewalk * 0.55
            if c.name == "main" and cfg.world.street_trees:
                for s in np.arange(de.s0 + 6.0, de.s1 - 6.0, cfg.world.street_tree_spacing):
                    x, y, h = de.center.pose_at(s)
                    px, py = x + math.sin(h) * off, y - math.cos(h) * off
                    if wet is None or not wet.contains(Point(px, py)):
                        props.append(Prop("tree_round", px, py, float(rng.uniform(0, 6.28)), float(rng.uniform(0.6, 0.85)), KERB))
            if c.name in ("main", "major") and d == 0:
                for s in np.arange(de.s0 + 10.0, de.s1 - 5.0, 36.0):
                    x, y, h = de.center.pose_at(s)
                    side = 1 if int(s / 36.0) % 2 else -1
                    lo = c.half_width + 0.4
                    px, py = x + math.sin(h) * lo * side, y - math.cos(h) * lo * side
                    yaw = h + (math.pi / 2 if side > 0 else -math.pi / 2)
                    props.append(Prop("streetlight", px, py, yaw, 1.0, KERB))
    return props


def prop_meshes(props: list[Prop], assets: AssetLibrary, terrain: Terrain | None = None) -> list[Mesh]:
    """Bake all props into a few merged meshes (one per asset part colour), standing on the ground."""
    groups: dict[tuple, list] = {}
    zs = terrain.height(np.array([p.x for p in props]), np.array([p.y for p in props])) if (terrain and props) else None
    for k, p in enumerate(props):
        z = p.z + (float(zs[k]) if zs is not None else 0.0)
        lod = "low" if p.asset in MASS_PROPS or p.asset.startswith("tree") else "high"
        for i, part in enumerate(assets.parts(p.asset, lod)):
            v = transform(part.vertices, p.x, p.y, z, p.yaw, p.scale)
            groups.setdefault((p.asset, i, part.rgba), []).append((v, part.faces))
    out = []
    for (asset, i, rgba), parts in groups.items():
        v, f = merge(parts)
        group = "tree" if asset.startswith("tree") else "prop"
        out.append(Mesh(f"{asset}_{i}", v, f, rgba, group))
    return out
