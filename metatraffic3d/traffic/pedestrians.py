"""Pedestrians: walk the sidewalks, turn corners, cross at zebra crossings, sometimes jaywalk.

A pedestrian is on one side of a street (``side`` +1 = right of the edge's u->v direction),
at arc length ``s`` along the edge centre line, walking toward ``u`` or ``v``. At a junction
end it either turns the corner onto the neighbouring street or crosses the street it is on.

* Zebra crossings at junctions are handled by the intersection manager: while anyone is on a
  crossing, no car may enter from or turn into that leg, and pedestrians only start crossing
  when no car is in those turn paths (and, at lights, when that leg's traffic has red).
* Mid-block jaywalkers put a moving obstacle on the lanes they cross; cars brake for it.
  Most wait for a gap; a few (``reckless``) step out anyway - one of the incident types.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..roads import RoadNetwork

WALK, WAIT, CROSS = 0, 1, 2


@dataclass(eq=False)
class Pedestrian:
    id: int
    edge: int
    side: int  # +1 right / -1 left of the edge direction u->v
    s: float
    direction: int  # +1 toward v, -1 toward u
    speed: float
    state: int = WALK
    shirt: tuple = (0.2, 0.4, 0.8)
    # crossing
    cross_node: int | None = None  # junction crossing (None = mid-block)
    cross_t: float = 0.0
    cross_len: float = 0.0
    wait: float = 0.0
    reckless: bool = False
    ttl: float = 600.0


class PedestrianSim:
    def __init__(self, net: RoadNetwork, n: int, rng: np.random.Generator, jaywalk_rate: float = 0.04,
                 reckless_share: float = 0.15) -> None:  # fmt: skip
        self.net = net
        self.n = n
        self.rng = rng
        self.jaywalk_rate = jaywalk_rate  # chance per block walked that someone crosses mid-block
        self.reckless_share = reckless_share
        self.peds: list[Pedestrian] = []
        self._id = 0
        self.cross_busy: dict[tuple[int, int], int] = {}  # (node, edge) -> people on that zebra
        self.lane_blocks: dict[int, list[tuple[float, float, Pedestrian]]] = {}
        self.stats = {"crossings": 0, "jaywalks": 0, "reckless": 0}
        # neighbour edges around each node, sorted by outgoing heading (counter-clockwise)
        self.order: dict[int, list[int]] = {}
        for node in net.nodes:
            hs = []
            for eid in node.edges:
                e = net.edges[eid]
                if e.cls.sidewalk <= 0:
                    continue  # no footpath (elevated highway)
                p = e.center
                if e.u == node.id:
                    q = p.point_at(min(8.0, p.length / 2))
                else:
                    q = p.point_at(max(p.length - 8.0, p.length / 2))
                hs.append((math.atan2(q[1] - node.pos[1], q[0] - node.pos[0]), eid))
            self.order[node.id] = [e for _, e in sorted(hs)]
        self.walk_edges = [e.id for e in net.edges if e.center.length > 30 and e.cls.sidewalk > 0]

    # ------------------------------------------------------------------ geometry
    def _offset(self, e) -> float:
        return e.cls.half_width + e.cls.sidewalk * 0.5

    def _limits(self, e) -> tuple[float, float]:
        return e.trim_u, e.center.length - e.trim_v

    def position(self, p: Pedestrian) -> tuple[float, float, float]:
        e = self.net.edges[p.edge]
        x, y, h = e.center.pose_at(p.s)
        off = self._offset(e)
        if p.state == CROSS:
            f = min(max(p.cross_t / max(p.cross_len, 1e-3), 0.0), 1.0)
            lat = p.side * off * (1 - 2 * f)
            heading = h - math.pi / 2 * p.side
        else:
            lat = p.side * off
            heading = h if p.direction > 0 else h + math.pi
        return x + math.sin(h) * lat, y - math.cos(h) * lat, heading

    # ------------------------------------------------------------------ lifecycle
    def spawn(self) -> None:
        if not self.walk_edges:
            return
        eid = int(self.rng.choice(self.walk_edges))
        e = self.net.edges[eid]
        lo, hi = self._limits(e)
        if hi - lo < 4:
            return
        shirt = tuple(self.rng.uniform(0.1, 0.9, 3))
        p = Pedestrian(self._id, eid, int(self.rng.choice([-1, 1])), float(self.rng.uniform(lo, hi)),
                       int(self.rng.choice([-1, 1])), float(self.rng.uniform(1.0, 1.6)), shirt=shirt,
                       ttl=float(self.rng.uniform(200, 900)))  # fmt: skip
        self._id += 1
        self.peds.append(p)

    def step(self, dt: float, sim) -> None:
        while len(self.peds) < self.n:
            self.spawn()
        self.lane_blocks = {}
        keep = []
        for p in self.peds:
            p.ttl -= dt
            if p.state == WALK:
                self._walk(p, dt, sim)
            elif p.state == WAIT:
                self._try_cross(p, dt, sim)
            if p.state == CROSS:
                self._cross(p, dt)
            if p.ttl > 0 or p.state == CROSS:
                keep.append(p)
        self.peds = keep

    def _walk(self, p: Pedestrian, dt: float, sim) -> None:
        e = self.net.edges[p.edge]
        lo, hi = self._limits(e)
        p.s += p.direction * p.speed * dt
        # occasional jaywalk mid-block
        if lo + 8 < p.s < hi - 8 and self.rng.random() < self.jaywalk_rate * p.speed * dt / max(hi - lo, 1.0):
            p.state, p.cross_node, p.wait = WAIT, None, 0.0
            p.reckless = self.rng.random() < self.reckless_share
            return
        if p.s >= hi or p.s <= lo:
            p.s = min(max(p.s, lo), hi)
            node = e.v if p.direction > 0 else e.u
            if self.net.nodes[node].degree >= 3 and self.rng.random() < 0.5:
                p.state, p.cross_node, p.wait = WAIT, node, 0.0
            else:
                self._turn_corner(p, node)

    def _turn_corner(self, p: Pedestrian, node: int) -> None:
        e = self.net.edges[p.edge]
        order = self.order[node]
        if len(order) <= 1:  # dead end: turn around
            p.direction *= -1
            return
        # side seen looking outward from the node along this edge
        out_side = p.side if e.u == node else -p.side
        i = order.index(p.edge)
        nxt = order[(i + 1) % len(order)] if out_side < 0 else order[(i - 1) % len(order)]
        n2 = self.net.edges[nxt]
        new_out_side = -out_side  # the corner is shared: left of one is right of the next
        p.edge = nxt
        if n2.u == node:
            p.s, p.direction, p.side = n2.trim_u, 1, new_out_side
        else:
            p.s, p.direction, p.side = n2.center.length - n2.trim_v, -1, -new_out_side

    # ------------------------------------------------------------------ crossings
    def _junction_legs_clear(self, sim, node: int, edge: int) -> bool:
        ctl = getattr(self.net.nodes[node], "control", None)
        for d in (2 * edge, 2 * edge + 1):
            de = self.net.dedges[d]
            if de.end == node and ctl is not None and ctl.kind == "signal" and not ctl.outage:
                if ctl.state(d, sim.time) != "R":
                    return False
        return not sim.leg_busy(node, edge)

    def _try_cross(self, p: Pedestrian, dt: float, sim) -> None:
        p.wait += dt
        e = self.net.edges[p.edge]
        if p.cross_node is not None:
            if self._junction_legs_clear(sim, p.cross_node, p.edge) or p.wait > 90:
                key = (p.cross_node, p.edge)
                self.cross_busy[key] = self.cross_busy.get(key, 0) + 1
                self._start_cross(p, e)
                self.stats["crossings"] += 1
            return
        # mid-block: wait for a gap in traffic (unless reckless)
        if p.reckless or sim.gap_clear(p.edge, p.s, 35.0) or p.wait > 60:
            self._start_cross(p, e)
            self.stats["jaywalks"] += 1
            self.stats["reckless"] += int(p.reckless)

    def _start_cross(self, p: Pedestrian, e) -> None:
        p.state = CROSS
        p.cross_t = 0.0
        p.cross_len = 2 * self._offset(e)
        if p.cross_node is not None:  # cross on the zebra just outside the junction box
            near_u = p.cross_node == e.u
            p.s = max(e.trim_u - 2.5, 0.5) if near_u else min(e.center.length - e.trim_v + 2.5, e.center.length - 0.5)

    def _cross(self, p: Pedestrian, dt: float) -> None:
        p.cross_t += p.speed * dt
        e = self.net.edges[p.edge]
        if p.cross_node is None:  # block every driving lane we are about to walk over
            for d in (2 * e.id, 2 * e.id + 1):
                de = self.net.dedges[d]
                s_lane = (p.s - de.s0) if d % 2 == 0 else (e.center.length - p.s - de.s0)
                for lane in de.drive:
                    self.lane_blocks.setdefault(lane, []).append((s_lane - 1.5, s_lane + 1.5, p))
        if p.cross_t >= p.cross_len:
            if p.cross_node is not None:
                key = (p.cross_node, p.edge)
                self.cross_busy[key] = max(self.cross_busy.get(key, 1) - 1, 0)
                node = p.cross_node
                p.side = -p.side
                p.state = WALK
                self._turn_corner(p, node)
            else:
                p.side = -p.side
                p.state = WALK
            p.cross_node = None
