"""Lane-level road network built from a mapgen city.

Pipeline: mapgen polylines (map units, y down) -> metres (y up) -> noded at crossings ->
nearby intersections snapped -> stubs and degree-2 joints cleaned -> largest connected part ->
directed edges with driving/parking lanes -> turn connectors with conflict sets -> kerb spots.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import LineString, Point, Polygon, STRtree
from shapely.prepared import prep

from .config import (
    DECK_HEIGHT,
    HIGHWAY,
    RAMP_LENGTH,
    ROAD_CLASSES,
    NetworkConfig,
    ParkingConfig,
    RoadClass,
    TrafficConfig,
)
from .geometry import Polyline, hermite, min_turn_radius, offset_points, wrap_angle

Vec = tuple[float, float]


def to_metres(pts, m: float) -> np.ndarray:
    a = np.asarray(pts, dtype=float).reshape(-1, 2)
    return np.column_stack([a[:, 0] * m, -a[:, 1] * m])


@dataclass
class Node:
    id: int
    pos: Vec
    edges: list[int] = field(default_factory=list)
    in_dedges: list[int] = field(default_factory=list)
    out_dedges: list[int] = field(default_factory=list)
    connectors: list[int] = field(default_factory=list)
    control: object = None  # traffic.control.JunctionControl, set by the world builder

    @property
    def degree(self) -> int:
        return len(self.edges)


@dataclass
class Edge:
    id: int
    u: int
    v: int
    cls: RoadClass
    center: Polyline  # u -> v
    trim_u: float = 0.0
    trim_v: float = 0.0
    bridge: list[tuple[float, float]] = field(default_factory=list)  # arc-length spans over water
    deck: float = 0.0  # elevated highway: deck height (0 = at street level)

    def z(self, s: float) -> float:
        """Height of the road above the ground at arc length ``s`` (ramps at both ends)."""
        if self.deck <= 0:
            return 0.0
        t = min(max(min(s, self.center.length - s) / RAMP_LENGTH, 0.0), 1.0)
        return self.deck * t * t * (3 - 2 * t)


@dataclass
class DirEdge:
    id: int  # 2 * edge + (0 forward u->v, 1 backward)
    edge: int
    start: int
    end: int
    cls: RoadClass
    center: Polyline  # untrimmed, in travel direction
    s0: float  # trimmed start / end on ``center``
    s1: float
    drive: list[int] = field(default_factory=list)  # lane ids, index 0 = rightmost
    park: int | None = None
    succ: dict[int, str] = field(default_factory=dict)  # next dedge -> movement S/L/R/U

    @property
    def length(self) -> float:
        return self.s1 - self.s0

    @property
    def reverse(self) -> int:
        return self.id ^ 1


@dataclass
class Lane:
    id: int
    dedge: int
    index: int  # drive lanes: 0 = rightmost; parking lane: -1
    kind: str  # "drive" | "park"
    path: Polyline
    offset: float  # distance right of the road centre line
    width: float
    speed_limit: float
    out_conns: dict[int, list[int]] = field(default_factory=dict)  # next dedge -> connectors
    spots: list[int] = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.path.length


@dataclass
class Connector:
    id: int
    node: int
    from_lane: int
    to_lane: int
    movement: str
    path: Polyline
    speed_limit: float
    conflicts: frozenset[int] = frozenset()

    @property
    def length(self) -> float:
        return self.path.length


@dataclass
class Spot:
    """A kerb-side parking spot on a parking lane."""

    id: int
    lane: int  # parking lane
    s: float  # centre along the parking lane
    drive_lane: int  # adjacent driving lane
    s_drive: float  # matching arc length on the driving lane
    pose: tuple[float, float, float]
    length: float


class RoadNetwork:
    def __init__(self) -> None:
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.dedges: dict[int, DirEdge] = {}
        self.lanes: list[Lane] = []
        self.connectors: list[Connector] = []
        self.spots: list[Spot] = []
        self.river: Polygon | None = None
        self.bounds: tuple[float, float, float, float] = (0, 0, 0, 0)
        self.fixed_blocks: dict[int, list[tuple[float, float]]] = {}  # permanent lane closures
        self.highway: list[int] = []  # edge ids of the elevated highway (experimental)

    # ------------------------------------------------------------------ queries
    def lane_z(self, lane_id: int, s: float) -> float:
        """Height above ground of a point on a driving lane (non-zero only on the highway)."""
        lane = self.lanes[lane_id]
        de = self.dedges[lane.dedge]
        e = self.edges[de.edge]
        if e.deck <= 0:
            return 0.0
        sc = de.s0 + s * de.length / max(lane.length, 1e-6)
        return e.z(sc if de.id % 2 == 0 else e.center.length - sc)

    def lane_of(self, de: int, index: int = 0) -> Lane:
        return self.lanes[self.dedges[de].drive[index]]

    def node_of_lane_end(self, lane_id: int) -> int:
        return self.dedges[self.lanes[lane_id].dedge].end

    def summary(self) -> dict:
        by_cls: dict[str, float] = defaultdict(float)
        for e in self.edges:
            by_cls[e.cls.name] += e.center.length / 1000
        deg = np.bincount([n.degree for n in self.nodes])
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "km_by_class": {k: round(v, 1) for k, v in sorted(by_cls.items())},
            "drive_lanes": sum(1 for lane in self.lanes if lane.kind == "drive"),
            "parking_lanes": sum(1 for lane in self.lanes if lane.kind == "park"),
            "connectors": len(self.connectors),
            "parking_spots": len(self.spots),
            "bridges": sum(1 for e in self.edges if e.bridge),
            "node_degrees": {i: int(c) for i, c in enumerate(deg) if c},
        }


# ====================================================================== building
def build_network(
    city,
    metres_per_unit: float,
    net_cfg: NetworkConfig | None = None,
    park_cfg: ParkingConfig | None = None,
    traffic_cfg: TrafficConfig | None = None,
    classes: dict[str, RoadClass] | None = None,
    highway: bool = False,
) -> RoadNetwork:
    net_cfg = net_cfg or NetworkConfig()
    park_cfg = park_cfg or ParkingConfig()
    traffic_cfg = traffic_cfg or TrafficConfig()
    classes = classes or ROAD_CLASSES
    m = metres_per_unit

    sources: list[tuple[np.ndarray, RoadClass]] = []
    for group, cname in (
        (city.main_roads, "main"),
        (city.coastline_roads, "main"),
        (city.major_roads, "major"),
        ([city.secondary_river], "major"),
        (city.minor_roads, "minor"),
    ):
        for line in group:
            if len(line) >= 2:
                sources.append((to_metres(line, m), classes[cname]))

    pieces = _node_lines(sources)
    nodes_pos, edge_list = _graph_from_pieces(pieces)
    nodes_pos, edge_list = _snap_nodes(nodes_pos, edge_list, net_cfg.snap_distance)
    edge_list = _drop_duplicates(edge_list)
    edge_list = _prune_dead_ends(edge_list, net_cfg.min_dead_end)
    edge_list = _merge_degree_two(edge_list)
    edge_list = _largest_component(edge_list)
    hw = _highway_path(edge_list) if highway else None
    if hw is not None:
        u, v, pts, path = hw
        edge_list.append((u, v, pts, HIGHWAY))

    net = RoadNetwork()
    if len(city.river) >= 3:
        river = Polygon(to_metres(city.river, m))
        net.river = river if river.is_valid else river.buffer(0)

    used = sorted({e[0] for e in edge_list} | {e[1] for e in edge_list})
    remap = {old: i for i, old in enumerate(used)}
    net.nodes = [Node(i, tuple(map(float, nodes_pos[old]))) for i, old in enumerate(used)]
    for i, (u, v, pts, cls) in enumerate(edge_list):
        e = Edge(i, remap[u], remap[v], cls, Polyline(pts))
        net.edges.append(e)
        net.nodes[e.u].edges.append(i)
        if e.v != e.u:
            net.nodes[e.v].edges.append(i)

    for e in net.edges:
        if e.cls is HIGHWAY:
            e.deck = DECK_HEIGHT
            net.highway.append(e.id)
    _compute_trims(net, net_cfg)
    _find_bridges(net)
    _build_lanes(net)
    _build_connectors(net, traffic_cfg)
    _build_spots(net, park_cfg)
    if hw is not None:
        _close_under_ramps(net, [(remap[a], remap[b]) for a, b in hw[3]])
    allp = np.vstack([e.center.pts for e in net.edges])
    net.bounds = (*allp.min(0), *allp.max(0))
    return net


def _node_lines(sources: list[tuple[np.ndarray, RoadClass]]) -> list[tuple[np.ndarray, RoadClass]]:
    """Split all lines at their crossings; give each piece the best class it lies on."""
    geoms = [LineString(p) for p, _ in sources]
    noded = shapely.get_parts(shapely.union_all(geoms))
    tree = STRtree(geoms)
    out = []
    for piece in noded:
        if piece.length < 1e-6:
            continue
        mid = piece.interpolate(0.5, normalized=True)
        cand = tree.query(mid, predicate="dwithin", distance=0.05)
        if len(cand) == 0:
            cand = [int(tree.nearest(mid))]
        cls = max((sources[i][1] for i in cand), key=lambda c: c.priority)
        out.append((np.asarray(piece.coords)[:, :2], cls))
    return out


def _highway_path(edges, min_len: float = 1200.0):
    """Longest nearly straight run of main roads: (start node, end node, polyline, [(u, v) edges])."""
    main = [(u, v, pts) for u, v, pts, cls in edges if cls.name == "main"]
    if not main:
        return None
    out: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for u, v, pts in main:
        out[u].append((v, pts))
        out[v].append((u, pts[::-1]))

    def heading(p, end: bool) -> float:
        a, b = (p[-2], p[-1]) if end else (p[0], p[1])
        return math.atan2(b[1] - a[1], b[0] - a[0])

    best = None
    for start in list(out):
        for nxt, pts in out[start]:  # greedy straight-ahead walk
            chain, cur, seen, total = [pts], nxt, {start, nxt}, Polyline(pts).length
            nodes = [(start, nxt)]
            while True:
                h = heading(chain[-1], True)
                cand = [(n, p) for n, p in out[cur] if n not in seen and abs(wrap_angle(heading(p, False) - h)) < 0.35]
                if not cand:
                    break
                n, p = min(cand, key=lambda c: abs(wrap_angle(heading(c[1], False) - h)))
                chain.append(p)
                nodes.append((cur, n))
                seen.add(n)
                total += Polyline(p).length
                cur = n
            if total >= min_len and (best is None or total > best[0]):
                best = (total, start, cur, chain, nodes)
    if best is None:
        return None
    _, a, b, chain, nodes = best
    pts = np.vstack([chain[0]] + [c[1:] for c in chain[1:]])
    smooth = Polyline(pts).resample(10.0)
    k = np.ones(7) / 7  # light smoothing so the viaduct has no kinks; ends stay on the nodes
    for dim in (0, 1):
        inner = np.convolve(smooth[:, dim], k, mode="same")
        smooth[3:-3, dim] = inner[3:-3]
    return a, b, smooth, nodes


def _close_under_ramps(net: RoadNetwork, path_nodes: list[tuple[int, int]]) -> None:
    """Close the arterial's median-side lanes under both highway ramps (the ramps sit there)."""
    def close_from(node_seq):
        left = RAMP_LENGTH + 15.0
        for u, v in node_seq:
            e = next((net.edges[i] for i in net.nodes[u].edges if {net.edges[i].u, net.edges[i].v} == {u, v}
                      and net.edges[i].cls.name == "main"), None)  # fmt: skip
            if e is None:
                return
            for d in (0, 1):
                de = net.dedges[2 * e.id + d]
                for lane_id in de.drive[max(len(de.drive) - 2, 1):]:
                    L = net.lanes[lane_id].length
                    span = (0.0, min(left, L)) if de.start == u else (max(L - left, 0.0), L)
                    net.fixed_blocks.setdefault(lane_id, []).append(span)
            left -= e.center.length
            if left <= 0:
                return

    close_from(path_nodes)
    close_from([(b, a) for a, b in reversed(path_nodes)])


def _graph_from_pieces(pieces):
    key_to_id: dict[tuple[float, float], int] = {}
    pos: list[np.ndarray] = []

    def node(p) -> int:
        k = (round(float(p[0]), 3), round(float(p[1]), 3))
        if k not in key_to_id:
            key_to_id[k] = len(pos)
            pos.append(np.array(k))
        return key_to_id[k]

    edges = []
    for pts, cls in pieces:
        edges.append((node(pts[0]), node(pts[-1]), pts, cls))
    return np.array(pos), edges


def _snap_nodes(pos: np.ndarray, edges, dist: float):
    """Union nodes closer than ``dist``; move them to the cluster centroid."""
    n = len(pos)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cell = defaultdict(list)
    for i, p in enumerate(pos):
        cell[(int(p[0] // dist), int(p[1] // dist))].append(i)
    for (cx, cy), members in cell.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell.get((cx + dx, cy + dy), ()):
                    for i in members:
                        if i < j and np.hypot(*(pos[i] - pos[j])) < dist:
                            parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    new_pos = pos.copy()
    for _root, members in groups.items():
        c = pos[members].mean(0)
        for i in members:
            new_pos[i] = c
    out = []
    for u, v, pts, cls in edges:
        ru, rv = find(u), find(v)
        pts = pts.copy()
        pu, pv = new_pos[ru], new_pos[rv]
        # drop interior points that now sit inside the snapped junction, then pin the ends
        inner = pts[1:-1]
        if len(inner):
            far = (np.hypot(*(inner - pu).T) > dist * 0.5) & (np.hypot(*(inner - pv).T) > dist * 0.5)
            inner = inner[far]
        pts = np.vstack([pu, inner, pv]) if len(inner) else np.vstack([pu, pv])
        length = np.hypot(*np.diff(pts, axis=0).T).sum()
        if ru == rv:
            if length < 4 * dist:
                continue  # tiny loop left over from snapping
            # split a long loop into two edges through a new mid node
            mid_i = len(pts) // 2
            if mid_i == 0 or mid_i == len(pts) - 1:
                continue
            new_pos = np.vstack([new_pos, pts[mid_i]])
            mid = len(new_pos) - 1
            parent.append(mid)
            out.append((ru, mid, pts[: mid_i + 1], cls))
            out.append((mid, rv, pts[mid_i:], cls))
            continue
        if length < 1e-3:
            continue
        out.append((ru, rv, pts, cls))
    return new_pos, out


def _drop_duplicates(edges):
    """Keep one of several edges joining the same two nodes along (nearly) the same path."""
    best: dict[tuple[int, int], list] = defaultdict(list)
    for e in edges:
        best[(min(e[0], e[1]), max(e[0], e[1]))].append(e)
    out = []
    for group in best.values():
        kept: list = []
        for e in sorted(group, key=lambda e: -e[3].priority):
            line = LineString(e[2])
            if any(line.hausdorff_distance(LineString(k[2])) < 6.0 for k in kept):
                continue
            kept.append(e)
        out.extend(kept)
    return out


def _degrees(edges) -> dict[int, int]:
    deg: dict[int, int] = defaultdict(int)
    for u, v, _, _ in edges:
        deg[u] += 1
        deg[v] += 1
    return deg


def _prune_dead_ends(edges, min_len: float):
    while True:
        deg = _degrees(edges)
        keep = []
        removed = False
        for e in edges:
            u, v, pts, _ = e
            length = np.hypot(*np.diff(pts, axis=0).T).sum()
            if (deg[u] == 1 or deg[v] == 1) and length < min_len:
                removed = True
                continue
            keep.append(e)
        edges = keep
        if not removed:
            return edges


def _merge_degree_two(edges):
    """Join two edges meeting at a node with nothing else there (same class only)."""
    edges = list(edges)
    changed = True
    while changed:
        changed = False
        inc: dict[int, list[int]] = defaultdict(list)
        for i, (u, v, _, _) in enumerate(edges):
            inc[u].append(i)
            inc[v].append(i)
        dead = set()
        for n, ids in inc.items():
            if len(ids) != 2 or ids[0] == ids[1] or ids[0] in dead or ids[1] in dead:
                continue
            a, b = edges[ids[0]], edges[ids[1]]
            if a[3] is not b[3]:
                continue
            pa = a[2] if a[1] == n else a[2][::-1]
            pb = b[2] if b[0] == n else b[2][::-1]
            ua = a[0] if a[1] == n else a[1]
            vb = b[1] if b[0] == n else b[0]
            if ua == vb:
                continue  # would close a ring into a self loop
            edges[ids[0]] = (ua, vb, np.vstack([pa, pb[1:]]), a[3])
            dead.add(ids[1])
            changed = True
        edges = [e for i, e in enumerate(edges) if i not in dead]
    return edges


def _largest_component(edges):
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v, _, _ in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen: set[int] = set()
    best: set[int] = set()
    for start in adj:
        if start in seen:
            continue
        comp = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in comp:
                    comp.add(y)
                    stack.append(y)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    return [e for e in edges if e[0] in best]


def _end_heading(edge: Edge, at_u: bool) -> float:
    """Heading of the edge leaving the given end node."""
    p = edge.center
    if at_u:
        q = p.point_at(min(8.0, p.length * 0.3))
        a = p.pts[0]
    else:
        q = p.point_at(max(p.length - 8.0, p.length * 0.7))
        a = p.pts[-1]
    return math.atan2(q[1] - a[1], q[0] - a[0])


def _compute_trims(net: RoadNetwork, cfg: NetworkConfig) -> None:
    """How far each lane stops short of the junction centre so crossing roads do not overlap."""
    for node in net.nodes:
        ends = []  # (heading, half width, edge id, at_u)
        for eid in node.edges:
            e = net.edges[eid]
            if e.u == node.id:
                ends.append((_end_heading(e, True), e.cls.half_width, eid, True))
            if e.v == node.id:
                ends.append((_end_heading(e, False), e.cls.half_width, eid, False))
        if len(ends) == 1:
            _, _, eid, at_u = ends[0]
            _set_trim(net.edges[eid], at_u, cfg.dead_end_trim)
            continue
        ends.sort(key=lambda t: t[0])
        k = len(ends)
        req = [0.0] * k
        for i in range(k):
            j = (i + 1) % k
            gap = (ends[j][0] - ends[i][0]) % (2 * math.pi)
            if k == 1:
                gap = 2 * math.pi
            hw = max(ends[i][1], ends[j][1])
            if gap < 1e-3:
                need = cfg.max_trim  # two roads leave in the same direction (e.g. a highway ramp)
            elif gap < math.pi - 1e-3:
                need = hw / math.tan(gap / 2)
            else:
                need = 0.0
            need = max(need, 0.0)
            req[i] = max(req[i], need)
            req[j] = max(req[j], need)
        for i, (_, hw, eid, at_u) in enumerate(ends):
            base = 0.0 if k == 2 else hw
            t = max(req[i], base) + cfg.corner_margin
            _set_trim(net.edges[eid], at_u, min(t, cfg.max_trim))
    for e in net.edges:  # never trim more than most of a short edge
        total = e.trim_u + e.trim_v
        lim = e.center.length - 2.0
        if total > lim:
            f = max(lim, 0.0) / total
            e.trim_u *= f
            e.trim_v *= f


def _set_trim(e: Edge, at_u: bool, t: float) -> None:
    if at_u:
        e.trim_u = max(e.trim_u, t)
    else:
        e.trim_v = max(e.trim_v, t)


def _find_bridges(net: RoadNetwork) -> None:
    if net.river is None:
        return
    river = net.river
    rp = prep(river)
    for e in net.edges:
        if e.deck > 0:
            continue  # the viaduct crosses water on its own piers
        line = LineString(e.center.pts)
        if not rp.intersects(line):
            continue
        spans = []
        for part in shapely.get_parts(line.intersection(river)):
            if part.geom_type != "LineString" or part.length < 1.0:
                continue
            a = line.project(Point(part.coords[0]))
            b = line.project(Point(part.coords[-1]))
            spans.append((min(a, b), max(a, b)))
        e.bridge = sorted(spans)


def _build_lanes(net: RoadNetwork) -> None:
    for e in net.edges:
        for d in (0, 1):
            center = e.center if d == 0 else e.center.reversed()
            start, end = (e.u, e.v) if d == 0 else (e.v, e.u)
            t_start, t_end = (e.trim_u, e.trim_v) if d == 0 else (e.trim_v, e.trim_u)
            de = DirEdge(2 * e.id + d, e.id, start, end, e.cls, center, t_start, center.length - t_end)
            piece = center.sub(de.s0, de.s1)
            if len(piece) < 2 or np.hypot(*(piece[-1] - piece[0])) < 1e-6:
                piece = np.vstack([center.point_at(de.s0), center.point_at(de.s0 + 0.5)])
            c = e.cls
            for k in range(c.drive_lanes):
                off = c.lane_offset(k)
                lane = Lane(len(net.lanes), de.id, k, "drive", Polyline(offset_points(piece, off)),
                            off, c.lane_width, c.speed_limit)  # fmt: skip
                net.lanes.append(lane)
                de.drive.append(lane.id)
            if c.parking:
                off = c.park_offset
                lane = Lane(len(net.lanes), de.id, -1, "park", Polyline(offset_points(piece, off)),
                            off, c.park_width, 0.0)  # fmt: skip
                net.lanes.append(lane)
                de.park = lane.id
            net.dedges[de.id] = de
            net.nodes[start].out_dedges.append(de.id)
            net.nodes[end].in_dedges.append(de.id)


def _movement(h_in: float, h_out: float, uturn: bool) -> str:
    if uturn:
        return "U"
    a = wrap_angle(h_out - h_in)
    if abs(a) < math.radians(35):
        return "S"
    return "L" if a > 0 else "R"


def _build_connectors(net: RoadNetwork, tcfg: TrafficConfig) -> None:
    for node in net.nodes:
        dead_end = node.degree == 1
        made: list[Connector] = []
        for din_id in node.in_dedges:
            din = net.dedges[din_id]
            in_lanes = [net.lanes[i] for i in din.drive]
            n_in = len(in_lanes)
            moves = []  # (out dedge, movement)
            for dout_id in node.out_dedges:
                dout = net.dedges[dout_id]
                uturn = dout.edge == din.edge
                if uturn and not dead_end:
                    continue
                turn = abs(wrap_angle(net.lanes[dout.drive[0]].path.start_heading - in_lanes[0].path.end_heading))
                if not uturn and turn > 2.6:
                    continue  # hairpin back along a parallel road (e.g. off the highway ramp)
                moves.append((dout_id, _movement(in_lanes[0].path.end_heading,
                                                 net.lanes[dout.drive[0]].path.start_heading, uturn)))  # fmt: skip
            pairs = {d: _lane_pairs(mv, n_in, len(net.dedges[d].drive)) for d, mv in moves}
            covered = {k for ps in pairs.values() for k, _ in ps}
            for k in range(n_in):  # every lane needs a way out (e.g. middle lanes at a T)
                if k in covered or not moves:
                    continue
                order = ("R", "S", "L", "U") if k < n_in / 2 else ("L", "S", "R", "U")
                d, mv = min(moves, key=lambda m: order.index(m[1]))
                n_out = len(net.dedges[d].drive)
                j = 0 if mv == "R" else (n_out - 1 if mv in ("L", "U") else min(k, n_out - 1))
                pairs[d].append((k, j))
            for dout_id, mv in moves:
                dout = net.dedges[dout_id]
                uturn = mv == "U"
                for k, j in pairs[dout_id]:
                    lin = in_lanes[k]
                    lout = net.lanes[dout.drive[j]]
                    p0, p1 = lin.path.pts[-1], lout.path.pts[0]
                    if uturn:
                        pts = _uturn_path(p0, lin.path.end_heading, p1)
                    else:
                        pts = hermite(p0, lin.path.end_heading, p1, lout.path.start_heading, n=12)
                    r = min_turn_radius(pts)
                    v = min(lin.speed_limit, lout.speed_limit, math.sqrt(tcfg.lateral_accel * r))
                    c = Connector(len(net.connectors), node.id, lin.id, lout.id, mv, Polyline(pts), max(v, 2.5))
                    net.connectors.append(c)
                    made.append(c)
                    lin.out_conns.setdefault(dout_id, []).append(c.id)
                    din.succ[dout_id] = mv
        node.connectors = [c.id for c in made]
        _set_conflicts(net, made)


def _lane_pairs(mv: str, n_in: int, n_out: int) -> list[tuple[int, int]]:
    """Which entry lane may feed which exit lane for a movement (lane 0 = kerb side).

    Right turns use the kerb lane (into the two kerb-side exit lanes on wide roads), left turns
    the median lane (two of them on 4-lane carriageways), U-turns the median lane, and straight
    traffic keeps its lane (spread or merged when the lane count changes).
    """
    if mv == "R":
        return [(0, j) for j in range(min(2, n_out) if n_out >= 3 else 1)]
    if mv == "U":
        return [(n_in - 1, n_out - 1)]
    if mv == "L":
        ins = [n_in - 1] + ([n_in - 2] if n_in >= 4 else [])
        return [(k, max(n_out - 1 - (n_in - 1 - k), 0)) for k in ins]
    if n_in == n_out:
        return [(k, k) for k in range(n_in)]
    pairs = []
    for k in range(n_in):
        lo = (k * n_out) // n_in
        hi = max(lo, -(-(k + 1) * n_out // n_in) - 1)
        pairs.extend((k, j) for j in range(lo, hi + 1))
    return pairs


def _uturn_path(p0, h0: float, p1) -> np.ndarray:
    """Half-loop from the end of one lane round to the start of the opposite lane."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    fwd = np.array([math.cos(h0), math.sin(h0)])
    mid = (p0 + p1) / 2
    r = max(np.hypot(*(p1 - p0)) / 2, 1.0)
    side = p0 - mid
    t = np.linspace(0, math.pi, 14)[:, None]
    return mid + side * np.cos(t) + fwd * r * np.sin(t) * 1.4


def _set_conflicts(net: RoadNetwork, conns: list[Connector]) -> None:
    if not conns:
        return
    geoms = [LineString(c.path.pts) for c in conns]
    tree = STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    conf: dict[int, set[int]] = defaultdict(set)
    for i, j in zip(left, right):
        if i >= j:
            continue
        a, b = conns[i], conns[j]
        if a.from_lane == b.from_lane:
            continue  # same entry lane: vehicles follow each other, no crossing
        conf[a.id].add(b.id)
        conf[b.id].add(a.id)
    by_out: dict[int, list[Connector]] = defaultdict(list)
    for c in conns:
        by_out[c.to_lane].append(c)
    for group in by_out.values():  # merging into the same lane always conflicts
        for a in group:
            for b in group:
                if a.id != b.id and a.from_lane != b.from_lane:
                    conf[a.id].add(b.id)
    for c in conns:
        c.conflicts = frozenset(conf[c.id])


def _build_spots(net: RoadNetwork, cfg: ParkingConfig) -> None:
    water = prep(net.river.buffer(4.0)) if net.river is not None else None
    for de in net.dedges.values():
        if de.park is None:
            continue
        lane = net.lanes[de.park]
        drive = net.lanes[de.drive[0]]
        usable = lane.length - 2 * cfg.end_clearance
        n = int(usable // cfg.spot_length)
        if n <= 0:
            continue
        start = cfg.end_clearance + (usable - n * cfg.spot_length) / 2
        for i in range(n):
            s = start + (i + 0.5) * cfg.spot_length
            x, y, h = lane.path.pose_at(s)
            if water is not None and water.contains(Point(x, y)):
                continue  # no parking on bridges
            s_drive, _ = drive.path.project((x, y))
            spot = Spot(len(net.spots), lane.id, s, drive.id, s_drive, (x, y, h), cfg.spot_length)
            net.spots.append(spot)
            lane.spots.append(spot.id)
