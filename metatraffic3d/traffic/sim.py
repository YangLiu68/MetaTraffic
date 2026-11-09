"""Microscopic traffic simulation on the lane network (backend-neutral, pure Python).

* Longitudinal motion: Intelligent Driver Model (IDM) per vehicle, ballistic update.
* Junctions: each turn path (connector) is granted by an intersection manager. A grant needs
  no busy conflicting connector, room on the exit lane (no blocking the box), and respects
  arrival order with a bonus for main roads; long waits block newer conflicting requests.
* Kerbs: sub-roads have a parking lane per direction. Parked cars fill at most
  ``max_occupancy`` of each kerb, so there is always somewhere to pull in. Cars pull in and
  out (waiting for a gap); with no free spot a taxi double-parks and blocks its lane.
* Background traffic: NPC trips between kerbs weighted by nearby building floor area; they
  route by current travel times, and park (or vanish) at the end. Stuck NPCs are removed.

Vehicle ``s`` is the position of the FRONT bumper along the current lane or connector.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..config import SimConfig
from ..roads import RoadNetwork
from .pedestrians import PedestrianSim
from .router import Router
from .vehicles import TYPES, sample_parked_type, sample_type

LANE, CONN = 0, 1
FREE, PARKED, RESERVED, TAXI, CLOSED = 0, 1, 2, 3, 4
WEATHER = {  # driver behaviour per weather: (headway x, max accel x, speed x)
    "clear": (1.0, 1.0, 1.0), "overcast": (1.0, 1.0, 1.0), "dusk": (1.1, 1.0, 0.97),
    "night": (1.15, 0.95, 0.93), "rain": (1.35, 0.8, 0.88), "fog": (1.5, 0.85, 0.8),
}
PALETTE = [(0.82, 0.10, 0.12), (0.12, 0.25, 0.65), (0.92, 0.92, 0.94), (0.15, 0.15, 0.17),
           (0.55, 0.57, 0.60), (0.10, 0.45, 0.25), (0.75, 0.70, 0.62), (0.45, 0.10, 0.12),
           (0.20, 0.55, 0.80), (0.95, 0.55, 0.10)]  # fmt: skip


class Controller(Protocol):
    def choose_next(self, veh: Vehicle, options: list[tuple[int, float, str]]) -> int | None: ...
    def plan_stop(self, veh: Vehicle) -> float: ...
    def on_arrive(self, veh: Vehicle) -> None: ...
    def on_dwell_done(self, veh: Vehicle) -> None: ...


@dataclass(eq=False)
class Vehicle:
    id: int
    kind: str  # "npc" | "taxi"
    length: float = 4.5
    color: tuple = (0.6, 0.6, 0.6)
    state: str = "drive"  # drive | maneuver_in | parked | dwell | wait_out
    on: int = LANE
    elem: int = -1
    s: float = 0.0
    v: float = 0.0
    a: float = 0.0
    dest: tuple[int, float] | None = None  # (directed edge, s on its lane 0)
    next_de: int | None = None
    conn: int | None = None
    requested: bool = False
    granted: bool = False
    req_time: float = 0.0
    stop_s: float | None = None
    spot: int | None = None
    double_parked: bool = False
    timer: float = 0.0
    dwell_time: float = 0.0  # how long to stay after pulling in
    still: float = 0.0
    clearing: int | None = None
    prev_path: object | None = None  # previous lane/connector path (for drawing the tail)
    lc_offset: float = 0.0  # lateral offset left from a lane change (drawn, decays to 0)
    lc_cooldown: float = 0.0
    at_line: bool = False  # came to a full stop at a stop line (stop signs)
    reckless: bool = False  # runs red lights and stop signs (incident)
    hard_brake: float = 0.0  # seconds left of an emergency stop (incident)
    stalled: bool = False  # broken down in its lane (incident)
    external: bool = False  # position comes from outside (the ego car)
    vtype: str = "sedan"
    emergency: bool = False  # siren on: priority at junctions, others yield
    aggressive: bool = False  # accepts small gaps when changing lanes (cut-ins)
    aside_for: int | None = None  # pulled to the kerb to let an emergency vehicle pass
    avoid_lane: int | None = None  # move out of this lane (a siren is coming)
    leave_after_maneuver: bool = False
    controller: Controller | None = None
    trip_start: float = 0.0
    dist: float = 0.0  # odometer
    removed: bool = False


@dataclass
class SimStats:
    spawned: int = 0
    arrived: int = 0
    parked: int = 0
    despawned: int = 0
    teleported: int = 0
    pullouts: int = 0
    double_parks: int = 0
    grants: int = 0
    lane_changes: int = 0
    trip_times: list[float] = field(default_factory=list)


class TrafficSim:
    def __init__(self, net: RoadNetwork, cfg: SimConfig, demand: np.ndarray | None = None,
                 seed: int = 0) -> None:  # fmt: skip
        self.net = net
        self.cfg = cfg
        self.tc = cfg.traffic
        self.pc = cfg.parking
        self.router = Router(net)
        self.rng = np.random.default_rng(seed)
        n_de = len(self.router.ids)
        if demand is None:
            demand = np.ones(n_de)
        self.demand = np.asarray(demand, float)
        self.lane_in_conns: list[list[int]] = [[] for _ in net.lanes]
        for c in net.connectors:
            self.lane_in_conns[c.to_lane].append(c.id)
        self.lane_out_list: list[list[int]] = [[] for _ in net.lanes]
        for lane in net.lanes:
            for cids in lane.out_conns.values():
                self.lane_out_list[lane.id].extend(cids)
        self.lane_priority = np.array([net.dedges[lane.dedge].cls.priority for lane in net.lanes])
        # which turn paths touch each junction leg (for zebra crossings)
        self.leg_conns: dict[tuple[int, int], list[int]] = {}
        for c in net.connectors:
            for lane_id in (c.from_lane, c.to_lane):
                e = net.dedges[net.lanes[lane_id].dedge].edge
                self.leg_conns.setdefault((c.node, e), []).append(c.id)
        self.conn_legs = [(c.node, net.dedges[net.lanes[c.from_lane].dedge].edge,
                           net.dedges[net.lanes[c.to_lane].dedge].edge) for c in net.connectors]  # fmt: skip
        self.height = None  # optional terrain height function h(x, y) for slopes
        self.incidents = None  # optional events.IncidentManager
        # kerb (parking lane) bookkeeping
        self.kerb_spots = {lane.id: lane.spots for lane in net.lanes if lane.kind == "park" and lane.spots}
        trng = np.random.default_rng(cfg.map.seed + 55)  # parked-car models are part of the city
        self.spot_type = [sample_parked_type(trng) for _ in net.spots]
        self.dest_dedges = np.array([d for d in self.router.ids if net.dedges[d].park is not None
                                     and net.lanes[net.dedges[d].park].spots], dtype=int)  # fmt: skip
        self.reset()

    # ================================================================ lifecycle
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        net = self.net
        self.time = 0.0
        self.vehicles: dict[int, Vehicle] = {}
        self._next_id = 0
        self.lane_veh: list[list[Vehicle]] = [[] for _ in net.lanes]
        self.conn_veh: list[list[Vehicle]] = [[] for _ in net.connectors]
        self.conn_busy = np.zeros(len(net.connectors), dtype=np.int32)
        self.requests: dict[int, list[Vehicle]] = defaultdict(list)
        self.spot_state = np.zeros(len(net.spots), dtype=np.int8)
        self.spot_color = np.zeros((len(net.spots), 3))
        self.spot_holder: dict[int, Vehicle] = {}
        self.stats = SimStats()
        self.set_weather(getattr(self, "weather", "clear"))
        self.peds = PedestrianSim(net, self.tc.num_pedestrians, self.rng, self.tc.jaywalk_rate,
                                  self.tc.reckless_share)  # fmt: skip
        self.static_blocks: dict[int, list[tuple[float, float]]] = {
            k: list(v) for k, v in net.fixed_blocks.items()}  # lane -> closed spans (works, ramps)  # fmt: skip
        self.closed_dedges: set[int] = set()
        self.router.set_costs(self.router.free_time.copy())
        self.speed_ema = self.router.speed_limit.copy()
        self.count_on = np.zeros(len(self.router.ids))
        self._last_cost_update = 0.0
        lo, hi = self.pc.occupancy
        for _lane_id, spots in self.kerb_spots.items():
            rho = min(self.rng.uniform(lo, hi), self.pc.max_occupancy)
            k = int(math.floor(rho * len(spots)))
            for sid in self.rng.choice(spots, size=k, replace=False):
                self._park_static(int(sid))
        for _ in range(self.tc.num_npc):
            self.spawn_npc(initial=True)

    def _park_static(self, sid: int, color=None) -> None:
        self.spot_state[sid] = PARKED
        self.spot_color[sid] = color if color is not None else PALETTE[int(self.rng.integers(len(PALETTE)))]

    # ================================================================ parking helpers
    def kerb_occupancy(self, park_lane: int) -> float:
        spots = self.kerb_spots.get(park_lane, [])
        if not spots:
            return 1.0
        return float(np.count_nonzero(self.spot_state[spots] != FREE)) / len(spots)

    def free_spot_near(self, de: int, s_drive: float, radius: float, ahead_only: bool = False) -> int | None:
        dedge = self.net.dedges[de]
        if dedge.park is None:
            return None
        best, best_d = None, math.inf
        for sid in self.net.lanes[dedge.park].spots:
            if self.spot_state[sid] != FREE:
                continue
            d = self.net.spots[sid].s_drive - s_drive
            if ahead_only and d < 2.0:
                continue
            if abs(d) <= radius and abs(d) < best_d:
                best, best_d = sid, abs(d)
        return best

    def random_kerb_point(self, rng=None, weighted: bool = True) -> tuple[int, float]:
        """A destination: a directed sub-road edge (by demand) and a spot-aligned point on it."""
        rng = rng or self.rng
        cand = self.dest_dedges
        if weighted:
            w = self.demand[[self.router.index[d] for d in cand]] ** self.tc.downtown_bias
            de = int(rng.choice(cand, p=w / w.sum()))
        else:
            de = int(rng.choice(cand))
        spots = self.net.lanes[self.net.dedges[de].park].spots
        sid = spots[int(rng.integers(len(spots)))]
        return de, float(self.net.spots[sid].s_drive)

    # ================================================================ spawning / removal
    def _new_vehicle(self, kind: str, **kw) -> Vehicle:
        v = Vehicle(self._next_id, kind, **kw)
        self._next_id += 1
        self.vehicles[v.id] = v
        return v

    def set_weather(self, kind: str) -> None:
        """Drivers keep longer gaps and go slower in rain, fog and at night."""
        self.weather = kind
        self._wx = WEATHER.get(kind, WEATHER["clear"])

    def spawn_npc(self, initial: bool = False, vtype: str | None = None, dest=None) -> Vehicle | None:
        """Start an NPC trip, preferably by pulling a parked car out of its spot."""
        dest = dest or self.random_kerb_point()
        parked = np.flatnonzero(self.spot_state == PARKED)
        t = TYPES[vtype] if vtype else sample_type(self.rng)
        if vtype is None and t.parks and len(parked) and self.rng.random() < 0.6:
            sid = int(self.rng.choice(parked))
            color = tuple(self.spot_color[sid])
            pt = TYPES[self.spot_type[sid]]
            v = self._new_vehicle("npc", color=color, length=pt.length, vtype=pt.name)
            self.spot_state[sid] = RESERVED  # held by the departing car until it pulls out
            self.spot_holder[sid] = v
            v.state, v.spot, v.dest = "wait_out", sid, dest
            v.trip_start = self.time
            self.stats.spawned += 1
            return v
        # otherwise join from a driveway somewhere along a random street
        for _ in range(10):
            de, _ = self.random_kerb_point(weighted=False)
            lane = self.net.lanes[self.net.dedges[de].drive[0]]
            if lane.length < 20:
                continue
            s = float(self.rng.uniform(8.0, lane.length - 8.0))
            if self._gap_for_pullout(lane.id, s - t.length / 2, t.length):
                v = self._new_vehicle("npc", color=PALETTE[int(self.rng.integers(len(PALETTE)))], length=t.length,
                                      vtype=t.name, emergency=t.emergency)  # fmt: skip
                v.dest = dest
                v.trip_start = self.time
                v.v = lane.speed_limit * 0.5 if initial else 0.0
                self._insert_lane(v, lane.id, s)
                self.stats.spawned += 1
                return v
        return None

    def add_taxi(self, controller: Controller, sid: int | None = None) -> Vehicle:
        """Place a taxi stopped in a free kerb spot."""
        if sid is None:
            free = np.flatnonzero(self.spot_state == FREE)
            sid = int(self.rng.choice(free))
        v = self._new_vehicle("taxi", color=(0.98, 0.78, 0.10), length=4.6, controller=controller)
        self.spot_state[sid] = TAXI
        self.spot_holder[sid] = v
        v.state, v.spot = "parked", sid
        return v

    def _insert_lane(self, v: Vehicle, lane_id: int, s: float) -> None:
        lst = self.lane_veh[lane_id]
        keys = [-o.s for o in lst]
        lst.insert(bisect.bisect_right(keys, -s), v)
        v.on, v.elem, v.s, v.state = LANE, lane_id, s, "drive"
        v.next_de, v.conn, v.requested, v.granted, v.stop_s = None, None, False, False, None

    @staticmethod
    def in_traffic(v: Vehicle) -> bool:
        """True while the car occupies a driving lane or connector."""
        return v.elem >= 0 and (v.state in ("drive", "maneuver_in") or (v.state == "dwell" and v.double_parked))

    def _drop_junction(self, v: Vehicle) -> None:
        """Cancel a pending request or an unused grant for the next junction."""
        if v.requested and not v.granted and v.conn is not None:
            node = self.net.connectors[v.conn].node
            if v in self.requests[node]:
                self.requests[node].remove(v)
        if v.granted and v.conn is not None and v.on == LANE:
            self.conn_busy[v.conn] -= 1  # granted but never entered
        v.requested = v.granted = False
        v.conn = None
        v.next_de = None

    def _detach(self, v: Vehicle) -> None:
        if self.in_traffic(v):
            lst = self.lane_veh[v.elem] if v.on == LANE else self.conn_veh[v.elem]
            if v in lst:
                lst.remove(v)
        self._drop_junction(v)
        if v.on == CONN and v.elem >= 0 and v.state == "drive":
            self.conn_busy[v.elem] -= 1
        if v.clearing is not None:
            self.conn_busy[v.clearing] -= 1
            v.clearing = None

    def remove(self, v: Vehicle) -> None:
        self._detach(v)
        if v.spot is not None and self.spot_holder.get(v.spot) is v:
            self.spot_state[v.spot] = FREE
            del self.spot_holder[v.spot]
        v.removed = True
        self.vehicles.pop(v.id, None)

    # ================================================================ targets (used by taxis)
    def set_destination(self, v: Vehicle, de: int, s: float) -> None:
        """Retarget a vehicle; cancels an ungranted junction request."""
        v.dest = (de, s)
        if v.on == LANE and not v.granted:
            self._drop_junction(v)
        v.stop_s = None
        if v.state == "drive" and v.spot is not None:
            self.release_spot(v)  # the old kerb reservation is no longer needed

    def depart(self, v: Vehicle) -> None:
        """A parked/dwelling vehicle wants to rejoin traffic."""
        if v.state == "maneuver_in":
            v.leave_after_maneuver = True  # finish pulling in, then pull straight out again
            return
        if v.state == "dwell" and v.double_parked:
            v.state = "drive"
            v.double_parked = False
            v.stop_s = None
        elif v.state in ("parked", "dwell"):
            v.state = "wait_out"

    # ================================================================ main step
    def step(self, dt: float | None = None) -> None:
        dt = dt or self.tc.dt
        if self.incidents is not None:
            self.incidents.step(dt)
        self.peds.step(dt, self)
        self._plan_turns()
        self._lane_changes(dt)
        self._handle_requests()
        self._grant()
        self._pullouts()
        self._move(dt)
        self._timers(dt)
        self._yield_to_emergency()
        self._update_congestion(dt)
        if self.time - self._last_cost_update >= 30.0:
            self._refresh_costs()
        n_npc = sum(1 for v in self.vehicles.values() if v.kind == "npc")
        for _ in range(min(3, self.tc.num_npc - n_npc)):
            self.spawn_npc()
        self.time += dt

    # ---------------------------------------------------------------- junction requests
    def _request_distance(self, v: Vehicle) -> float:
        b = self.tc.comfort_decel
        return max(30.0, v.v * self.tc.time_headway + v.v * v.v / (2 * b) + 25.0)

    def _handle_requests(self) -> None:
        net = self.net
        for lane_id, lst in enumerate(self.lane_veh):
            if not lst:
                continue
            lane = net.lanes[lane_id]
            for v in lst:
                if v.granted:
                    continue
                if v.state != "drive" or v.requested or v.stalled:
                    break
                if lane.length - v.s > self._request_distance(v):
                    break
                if self._destination_ahead(v):
                    break
                if v.next_de is None:
                    v.next_de = self._choose_next(v, lane, from_lane=True)
                    if v.next_de is None:
                        break  # waiting for a decision
                if v.next_de in self.closed_dedges:
                    v.next_de = self._choose_next(v, lane, from_lane=True, ask=False)
                    if v.next_de is None:
                        break
                if v.next_de not in lane.out_conns:
                    if lane.length - v.s > 15.0 and v.still < 4.0:
                        break  # still trying to reach a lane for that turn
                    v.next_de = self._choose_next(v, lane, from_lane=True, ask=False)  # missed it
                    if v.next_de is None:
                        break
                if v.conn is None:
                    v.conn = self._choose_connector(v, lane)
                    if v.conn is None:
                        v.next_de = None
                        break
                v.requested = True
                v.req_time = self.time
                self.requests[net.connectors[v.conn].node].append(v)
                break

    def _destination_ahead(self, v: Vehicle) -> bool:
        if v.dest is None or v.on != LANE:
            return False
        lane = self.net.lanes[v.elem]
        return lane.dedge == v.dest[0] and v.s <= v.dest[1] + 0.5

    def ctg(self, v: Vehicle) -> np.ndarray:
        de, s = v.dest
        length = max(self.net.lanes[self.net.dedges[de].drive[0]].length, 1e-3)
        return self.router.cost_to_go(de, min(s / length, 1.0))

    def _choose_next(self, v: Vehicle, lane, from_lane: bool = False, ask: bool = True) -> int | None:
        """Next street after this one: from any lane (planned on entry) or from the current lane."""
        if from_lane:
            allowed = list(lane.out_conns.keys())
        else:
            allowed = list(self.net.dedges[lane.dedge].succ.keys())
        if not allowed:
            return None
        if v.dest is None:
            return int(self.rng.choice(allowed))
        opts = self.router.options(lane.dedge, allowed, self.ctg(v))
        if not opts:
            return int(self.rng.choice(allowed))
        if v.controller is not None and ask:
            return v.controller.choose_next(v, opts)
        return opts[0][0]

    def _plan_turns(self) -> None:
        """Vehicles decide where to turn as soon as they enter a street (so they can change lanes)."""
        for lane_id, lst in enumerate(self.lane_veh):
            if not lst:
                continue
            lane = self.net.lanes[lane_id]
            for v in lst:
                if v.next_de is None and v.state == "drive" and not v.requested and not self._destination_ahead(v):
                    v.next_de = self._choose_next(v, lane)

    # ---------------------------------------------------------------- lane changes
    def _lane_changes(self, dt: float) -> None:
        """Mandatory changes toward a lane that allows the planned turn, plus overtaking slow cars."""
        net = self.net
        moves = []
        for de in net.dedges.values():
            if len(de.drive) < 2:
                continue
            for lane_id in de.drive:
                lst = self.lane_veh[lane_id]
                for i, v in enumerate(lst):
                    if v.state != "drive" or v.requested or v.granted or v.lc_cooldown > 0 or v.external or v.stalled:
                        continue
                    lane = net.lanes[lane_id]
                    rem = lane.length - v.s
                    if rem < 8.0 or v.s < v.length:
                        continue
                    k = lane.index
                    targets = [li for li in de.drive if v.next_de is None or v.next_de in net.lanes[li].out_conns]
                    open_ = [li for li in targets if self._blocks_ahead(li, v.s) > 120.0] if self.static_blocks else targets
                    avoid, v.avoid_lane = v.avoid_lane, None
                    if avoid is not None:
                        open_ = [li for li in open_ if li != avoid] or [li for li in de.drive if li != avoid]
                    if open_:
                        targets = open_
                    want = None
                    if lane_id not in targets and targets:
                        tk = min((net.lanes[t].index for t in targets), key=lambda t: abs(t - k))
                        want = k + (1 if tk > k else -1)
                    elif i > 0 and rem > 60.0:
                        ld = lst[i - 1]
                        gap = ld.s - ld.length - v.s
                        if gap < 25.0 and ld.v < v.v + 0.5 and ld.v < 0.8 * lane.speed_limit:
                            for nk in (k + 1, k - 1):  # prefer passing on the left
                                if 0 <= nk < len(de.drive) and de.drive[nk] in targets:
                                    if self._free_ahead(de.drive[nk], v.s * net.lanes[de.drive[nk]].length / lane.length) > gap + 15.0:
                                        want = nk
                                        break
                    if want is not None and 0 <= want < len(de.drive):
                        moves.append((v, lane_id, de.drive[want]))
        for v, src, dst in moves:
            s_new = v.s * self.net.lanes[dst].length / max(self.net.lanes[src].length, 1e-3)
            if not self._gap_for_change(dst, s_new, v):
                continue
            self.lane_veh[src].remove(v)
            self._append_lane(v, dst, s_new)
            off = self.net.lanes[dst].offset - self.net.lanes[src].offset
            v.lc_offset = -off  # drawn from the old lane, sliding across
            v.lc_cooldown = 4.0
            self.stats.lane_changes += 1

    def _free_ahead(self, lane_id: int, s: float) -> float:
        best = math.inf
        for o in self.lane_veh[lane_id]:
            if o.s > s:
                best = min(best, o.s - o.length - s)
        return best

    def _gap_for_change(self, lane_id: int, s: float, v: Vehicle) -> bool:
        k = 0.35 if v.aggressive else 1.0  # aggressive drivers cut in
        for o in self.lane_veh[lane_id]:
            if o.s >= s:
                if o.s - o.length - s < max(1.5, v.v * 0.6 * k):
                    return False
            elif s - v.length - o.s < max(2.0, o.v * 1.0 * k + max(o.v - v.v, 0.0) * 1.5 * k):
                return False
        if s < v.length + 3.0:
            for cid in self.lane_in_conns[lane_id]:
                if self.conn_veh[cid]:
                    return False
        return True

    def _choose_connector(self, v: Vehicle, lane) -> int | None:
        cids = lane.out_conns.get(v.next_de)
        if not cids:
            return None
        if len(cids) == 1:
            return cids[0]
        net = self.net
        # pick the exit lane that allows the following turn; then the emptier one
        want = None
        if v.dest is not None and v.next_de != v.dest[0]:
            ctg = self.ctg(v)
            out_lanes = net.dedges[v.next_de].drive
            opts = self.router.options(v.next_de, list(net.lanes[out_lanes[0]].out_conns.keys() |
                                                        net.lanes[out_lanes[-1]].out_conns.keys()), ctg)  # fmt: skip
            if opts:
                nxt = opts[0][0]
                ok = [li for li in out_lanes if nxt in net.lanes[li].out_conns]
                want = set(ok) if ok else None
        elif v.dest is not None:
            want = {net.dedges[v.next_de].drive[0]}  # destination kerb is on the right
        best, best_q = None, math.inf
        for cid in cids:
            to_lane = net.connectors[cid].to_lane
            q = len(self.lane_veh[to_lane]) + (0 if want is None or to_lane in want else 1000)
            if q < best_q:
                best, best_q = cid, q
        return best

    def leg_busy(self, node: int, edge: int) -> bool:
        """Any car in (or cleared into) a turn path that uses this junction leg."""
        for cid in self.leg_conns.get((node, edge), ()):
            if self.conn_busy[cid]:
                return True
        for de in (2 * edge, 2 * edge + 1):  # cars waiting right at the line also block
            dedge = self.net.dedges[de]
            if dedge.end == node:
                for lane in dedge.drive:
                    lst = self.lane_veh[lane]
                    if lst and self.net.lanes[lane].length - lst[0].s < 2.0 and lst[0].granted:
                        return True
        return False

    def gap_clear(self, edge: int, s_center: float, dist: float) -> bool:
        """No car within ``dist`` upstream of a point on either side of a street."""
        e = self.net.edges[edge]
        for d in (2 * edge, 2 * edge + 1):
            de = self.net.dedges[d]
            s_lane = (s_center - de.s0) if d % 2 == 0 else (e.center.length - s_center - de.s0)
            for lane in de.drive:
                for o in self.lane_veh[lane]:
                    if -3.0 < s_lane - o.s < dist + o.v * 2:
                        return False
        return True

    def _blocks_ahead(self, lane_id: int, s: float) -> float:
        """Distance to the nearest closed span or crossing pedestrian ahead on a lane."""
        best = math.inf
        for lo, hi in self.static_blocks.get(lane_id, ()):
            if hi > s and lo - s < best:
                best = max(lo - s, 0.0)
        for lo, _hi, _ in self.peds.lane_blocks.get(lane_id, ()):
            if lo > s - 0.5 and lo - s < best:
                best = lo - s
        return best

    def _exit_room(self, lane_id: int) -> float:
        lst = self.lane_veh[lane_id]
        room = self.net.lanes[lane_id].length if not lst else lst[-1].s - lst[-1].length
        for cid in self.lane_in_conns[lane_id]:
            room -= self.conn_busy[cid] * 6.5
        return room

    def _grant(self) -> None:
        net = self.net
        for node, reqs in self.requests.items():
            if not reqs:
                continue
            ctl = net.nodes[node].control
            signal = ctl is not None and ctl.kind == "signal" and not ctl.outage
            reqs.sort(key=lambda v: v.req_time - 6.0 * self.lane_priority[v.elem] - (1e4 if v.emergency else 0.0))
            held: set[int] = set()
            granted = []
            dropped = []
            for v in reqs:
                c = net.connectors[v.conn]
                de_in = net.lanes[v.elem].dedge
                if ctl is not None and not v.reckless and not v.emergency:
                    if signal:
                        st = ctl.state(de_in, self.time)
                        dist = net.lanes[v.elem].length - v.s
                        if st == "R" or (st == "Y" and dist > max(v.v * 1.2, 4.0)):
                            continue
                    elif ctl.must_stop(de_in) and not v.at_line:
                        continue
                _, e_in, e_out = self.conn_legs[c.id]
                if self.peds.cross_busy.get((node, e_in)) or self.peds.cross_busy.get((node, e_out)):
                    continue  # people on the zebra
                blocked = c.id in held or any(self.conn_busy[x] for x in c.conflicts)
                if blocked:
                    if not signal and self.time - v.req_time > self.tc.starvation_time:
                        held |= c.conflicts
                    continue
                if any(lo <= 1.0 for lo, _ in self.static_blocks.get(c.to_lane, ())):
                    v.requested, v.next_de, v.conn = False, None, None  # street closed: plan again
                    dropped.append(v)
                    continue
                need = min(v.length + self.tc.min_gap, net.lanes[c.to_lane].length * 0.9)
                if self._exit_room(c.to_lane) < need:
                    continue
                v.granted = True
                self.conn_busy[c.id] += 1
                granted.append(v)
                self.stats.grants += 1
            if granted or dropped:
                self.requests[node] = [v for v in reqs if not v.granted and v not in dropped]

    # ---------------------------------------------------------------- motion
    def _idm(self, v: Vehicle, v0: float, gap: float, dv: float) -> float:
        tc = self.tc
        t = TYPES.get(v.vtype)
        hw, ax, sx = self._wx
        a_max = tc.max_accel * ax * (t.accel if t else 1.0)
        v0 = v0 * sx * (t.speed if t else 1.0)
        s_star = tc.min_gap + max(0.0, v.v * tc.time_headway * hw + v.v * dv / (2 * math.sqrt(a_max * tc.comfort_decel)))
        free = 1.0 - (v.v / max(v0, 0.1)) ** tc.accel_exponent
        inter = (s_star / max(gap, 0.1)) ** 2 if gap < 1e5 else 0.0
        return max(a_max * (free - inter), -9.0)

    def _gap_ahead(self, v: Vehicle, idx: int) -> tuple[float, float, float]:
        """(gap, speed difference, speed limit) for vehicle ``v`` at position ``idx`` of its list."""
        net = self.net
        gap, dv = math.inf, 0.0
        if v.on == LANE:
            lane = net.lanes[v.elem]
            limit = lane.speed_limit
            lst = self.lane_veh[v.elem]
            if idx > 0:
                ld = lst[idx - 1]
                gap, dv = ld.s - ld.length - v.s, v.v - ld.v
            else:
                rem = lane.length - v.s
                if v.granted and v.conn is not None:
                    c = net.connectors[v.conn]
                    for cid in self.lane_out_list[lane.id]:
                        cl = self.conn_veh[cid]
                        if cl:
                            ld = cl[-1]
                            g = rem + ld.s - ld.length
                            if g < gap:
                                gap, dv = g, v.v - ld.v
                    if not self.conn_veh[c.id]:
                        ol = self.lane_veh[c.to_lane]
                        if ol:
                            ld = ol[-1]
                            g = rem + c.length + ld.s - ld.length
                            if g < gap:
                                gap, dv = g, v.v - ld.v
                    limit = min(limit, c.speed_limit if rem < 25 else limit)
                elif not self._destination_ahead(v) or v.dest is None:
                    gap, dv = rem - 0.5, v.v  # stop line
            if v.stop_s is not None and v.stop_s >= v.s - 0.5:
                g = v.stop_s - v.s
                if g < gap:
                    gap, dv = g + self.tc.min_gap, v.v
            if lane.id in self.static_blocks or lane.id in self.peds.lane_blocks:
                g = self._blocks_ahead(lane.id, v.s)
                if g < gap:
                    gap, dv = g, v.v
        else:
            c = net.connectors[v.elem]
            limit = c.speed_limit
            lst = self.conn_veh[v.elem]
            if idx > 0:
                ld = lst[idx - 1]
                gap, dv = ld.s - ld.length - v.s, v.v - ld.v
            else:
                ol = self.lane_veh[c.to_lane]
                if ol:
                    ld = ol[-1]
                    gap, dv = c.length - v.s + ld.s - ld.length, v.v - ld.v
        return gap, dv, limit

    def _move(self, dt: float) -> None:
        net = self.net
        # 1) accelerations from the start-of-step state
        moves: list[tuple[Vehicle, float]] = []
        for lst in self.lane_veh:
            lane_len = self.net.lanes[lst[0].elem].length if lst else 0.0
            for i, v in enumerate(lst):
                if v.state != "drive":
                    continue
                if v.stop_s is None and v.dest is not None and self._destination_ahead(v):
                    v.stop_s = self._plan_stop(v)
                    v.dest = (v.dest[0], v.stop_s)  # keep "destination ahead" true until the stop
                if i == 0 and not v.at_line and v.v < 0.3 and lane_len - v.s < 4.0:
                    v.at_line = True
                    v.req_time = self.time if v.requested else v.req_time
                gap, dv, limit = self._gap_ahead(v, i)
                a = self._idm(v, limit, gap, dv)
                if v.stalled:
                    a = -6.0
                elif v.hard_brake > 0:
                    v.hard_brake -= dt
                    a = -7.0
                elif self.height is not None and self.tc.grade_effect and not v.external:
                    a -= 9.81 * self.tc.grade_effect * self._grade(v)
                if v.external:
                    v.a = a  # advice only; the ego's motion comes from physics
                    continue
                moves.append((v, a))
        for lst in self.conn_veh:
            for i, v in enumerate(lst):
                gap, dv, limit = self._gap_ahead(v, i)
                if v.external:
                    v.a = self._idm(v, limit, gap, dv)
                    continue
                moves.append((v, self._idm(v, limit, gap, dv)))
        # 2) integrate and hand over between lanes and connectors
        for v, a in moves:
            v.a = a
            nv = v.v + a * dt
            if nv < 0:
                ds = -v.v * v.v / (2 * a) if a < 0 else 0.0
                nv = 0.0
            else:
                ds = (v.v + nv) / 2 * dt
            v.v = nv
            v.s += ds
            v.dist += ds
            v.still = v.still + dt if nv < 0.2 else 0.0
            if v.on == LANE:
                lane = net.lanes[v.elem]
                if v.clearing is not None and v.s >= v.length:
                    self.conn_busy[v.clearing] -= 1
                    v.clearing = None
                if v.s > lane.length:
                    if v.granted and v.conn is not None:
                        over = v.s - lane.length
                        self.lane_veh[lane.id].remove(v)
                        if v.clearing is not None:
                            self.conn_busy[v.clearing] -= 1
                            v.clearing = None
                        v.prev_path = lane.path
                        v.on, v.elem, v.s = CONN, v.conn, over
                        self.conn_veh[v.conn].append(v)
                        v.requested = v.granted = False
                        v.stop_s = None  # any stop point belonged to the street we just left
                    else:
                        v.s = lane.length
                        v.v = 0.0
            if v.on == CONN:
                c = net.connectors[v.elem]
                if v.s > c.length:
                    over = v.s - c.length
                    self.conn_veh[c.id].remove(v)
                    v.prev_path = c.path
                    v.clearing = c.id  # the tail is still in the junction box
                    out = net.lanes[c.to_lane]
                    self._append_lane(v, out.id, over)
                    v.next_de, v.conn = None, None
                    v.at_line = False
        # 3) arrivals
        for v, _ in moves:
            if v.removed or v.on != LANE or v.state != "drive" or v.stop_s is None:
                continue
            if v.dest is None or net.lanes[v.elem].dedge != v.dest[0]:
                v.stop_s = None  # stale stop point from another street
                continue
            if v.v < 0.3 and abs(v.stop_s - v.s) < 2.5:
                self._arrive(v)

    def _grade(self, v: Vehicle) -> float:
        """sin(slope) along the car's heading, from the terrain height function."""
        x, y, h = self.pose(v)
        c, s_ = math.cos(h), math.sin(h)
        dz = self.height(x + 3 * c, y + 3 * s_) - self.height(x - 3 * c, y - 3 * s_)
        return dz / 6.0

    def _append_lane(self, v: Vehicle, lane_id: int, s: float) -> None:
        lst = self.lane_veh[lane_id]
        if lst and lst[-1].s < s:
            keys = [-o.s for o in lst]
            lst.insert(bisect.bisect_right(keys, -s), v)
        else:
            lst.append(v)
        v.on, v.elem, v.s = LANE, lane_id, s

    def _plan_stop(self, v: Vehicle) -> float:
        """Where to stop on the destination street; NPCs reserve a kerb spot if one is allowed."""
        if v.controller is not None:
            return v.controller.plan_stop(v)
        de, s = v.dest
        park = self.net.dedges[de].park
        t = TYPES.get(v.vtype)
        if park is not None and v.spot is None and (t is None or t.parks):
            n = len(self.kerb_spots.get(park, ()))
            if n and self.kerb_occupancy(park) + 1.0 / n <= self.pc.max_occupancy + 1e-9:
                sid = self.free_spot_near(de, max(s, v.s + 12.0), 60.0)
                if sid is not None and self.net.spots[sid].s_drive > v.s + 8.0:
                    self.reserve(v, sid)
                    return self.net.spots[sid].s_drive
        return max(s, v.s + 1.0)

    def reserve(self, v: Vehicle, sid: int) -> None:
        if v.spot is not None and v.spot != sid:
            self.release_spot(v)
        self.spot_state[sid] = TAXI if v.kind == "taxi" else RESERVED
        self.spot_holder[sid] = v
        v.spot = sid

    def release_spot(self, v: Vehicle) -> None:
        if v.spot is not None and self.spot_holder.get(v.spot) is v:
            self.spot_state[v.spot] = FREE
            del self.spot_holder[v.spot]
        v.spot = None

    def _arrive(self, v: Vehicle) -> None:
        v.stop_s = None
        if v.controller is not None:
            v.controller.on_arrive(v)
            return
        # NPC: pull into its reserved spot, or vanish (trip ends off-map / in a garage)
        self.stats.arrived += 1
        self.stats.trip_times.append(self.time - v.trip_start)
        if v.spot is not None:
            self.begin_pull_in(v, v.spot)
            return
        self.stats.despawned += 1
        self.remove(v)

    def begin_pull_in(self, v: Vehicle, sid: int, dwell: float = 0.0) -> None:
        """Pull into spot ``sid`` from alongside it (the car blocks its lane for a moment)."""
        if v.spot != sid:
            self.release_spot(v)
            self.reserve(v, sid)
        self._drop_junction(v)
        v.state = "maneuver_in"
        v.timer = 4.0
        v.dwell_time = dwell
        v.v = 0.0

    def stop_here(self, v: Vehicle, seconds: float) -> None:
        """Double-park: dwell in the driving lane (blocks traffic)."""
        self._drop_junction(v)
        v.state = "dwell"
        v.double_parked = True
        v.timer = seconds
        v.v = 0.0
        self.stats.double_parks += 1

    # ---------------------------------------------------------------- timers and kerbs
    def _timers(self, dt: float) -> None:
        for v in list(self.vehicles.values()):
            if v.removed:
                continue  # removed earlier in this loop (e.g. cleared from in front of a taxi)
            if v.lc_cooldown > 0:
                v.lc_cooldown = max(0.0, v.lc_cooldown - dt)
            if v.state == "maneuver_in":
                v.timer -= dt
                if v.timer <= 0:
                    self.lane_veh[v.elem].remove(v)
                    if v.clearing is not None:
                        self.conn_busy[v.clearing] -= 1
                        v.clearing = None
                    if v.kind == "npc":  # becomes an ordinary parked car
                        self._park_static(v.spot, v.color)
                        self.spot_holder.pop(v.spot, None)
                        v.spot = None
                        self.stats.parked += 1
                        self.vehicles.pop(v.id, None)
                        v.removed = True
                    elif v.leave_after_maneuver:
                        v.leave_after_maneuver = False
                        v.state = "wait_out"
                    else:
                        v.state = "dwell"
                        v.timer = v.dwell_time
            elif v.state == "dwell":
                v.timer -= dt
                if v.timer <= 0 and v.controller is None and v.double_parked:
                    v.state, v.double_parked, v.stop_s = "drive", False, None  # delivery done
                    continue
                if v.timer <= 0:
                    if v.double_parked:
                        v.timer = math.inf  # keeps blocking until told to depart
                    else:
                        v.state = "parked"
                    if v.controller is not None:
                        v.controller.on_dwell_done(v)
            elif v.state == "drive" and v.still > self.tc.stuck_timeout and not v.external and not v.stalled:
                if v.kind == "npc":
                    self.stats.teleported += 1
                    self.remove(v)
                else:
                    self._clear_blockers(v)
                    v.still = 0.0

    def _yield_to_emergency(self) -> None:
        """Cars ahead of a siren move over: change lane on wide roads, pull aside on narrow ones."""
        sirens = [v for v in self.vehicles.values() if v.emergency and v.on == LANE and v.state == "drive"]
        for e in sirens:
            lst = self.lane_veh[e.elem]
            de = self.net.dedges[self.net.lanes[e.elem].dedge]
            for o in list(lst):
                if o is e or o.s <= e.s or o.s - e.s > 90.0 or o.external or o.emergency or o.state != "drive":
                    continue
                if o.granted or o.requested:
                    continue
                if len(de.drive) > 1:
                    o.lc_cooldown = 0.0
                    o.avoid_lane = e.elem  # the lane-change step moves it off the siren's lane
                else:
                    self._drop_junction(o)
                    lst.remove(o)
                    o.state, o.aside_for, o.v = "aside", e.id, 0.0
        for v in list(self.vehicles.values()):  # rejoin once the siren is well past
            if v.state != "aside":
                continue
            e = self.vehicles.get(v.aside_for)
            if e is None or e.elem != v.elem or e.on != LANE or e.s > v.s + 12.0:
                if self._gap_for_pullout(v.elem, v.s - v.length / 2, v.length):
                    v.aside_for = None
                    self._append_lane(v, v.elem, v.s)
                    v.state = "drive"

    def _clear_blockers(self, v: Vehicle) -> None:
        """A taxi stuck for too long: remove the NPCs queued right in front of it."""
        if v.on == LANE:
            lst = self.lane_veh[v.elem]
            i = lst.index(v)
            for o in lst[max(0, i - 3):i]:
                if o.kind == "npc":
                    self.stats.teleported += 1
                    self.remove(o)

    def _pullouts(self) -> None:
        for v in list(self.vehicles.values()):
            if v.state != "wait_out":
                continue
            spot = self.net.spots[v.spot]
            if not self._gap_for_pullout(spot.drive_lane, spot.s_drive, v.length):
                continue
            sid = v.spot
            self.spot_state[sid] = FREE
            self.spot_holder.pop(sid, None)
            v.spot = None
            self._insert_lane(v, spot.drive_lane, spot.s_drive + v.length / 2)
            v.v = 0.0
            self.stats.pullouts += 1

    def _gap_for_pullout(self, lane_id: int, s: float, length: float) -> bool:
        for o in self.lane_veh[lane_id]:
            if o.s > s:
                if o.s - o.length - (s + length / 2) < 3.0:
                    return False
            elif (s - length / 2) - o.s < max(8.0, o.v * 2.5):
                return False
        for cid in self.lane_in_conns[lane_id]:
            c = self.net.connectors[cid]
            for o in self.conn_veh[cid]:
                if (s - length / 2) - (o.s - c.length) < max(8.0, o.v * 2.5):
                    return False
        return True

    # ---------------------------------------------------------------- congestion
    def _update_congestion(self, dt: float) -> None:
        idx = self.router.index
        n = len(self.router.ids)
        speed_sum = np.zeros(n)
        count = np.zeros(n)
        for lane_id, lst in enumerate(self.lane_veh):
            if not lst:
                continue
            i = idx[self.net.lanes[lane_id].dedge]
            for v in lst:
                if v.state == "drive":
                    speed_sum[i] += v.v
                    count[i] += 1
        alpha = self.tc.congestion_ema * dt
        has = count > 0
        obs = np.where(has, speed_sum / np.maximum(count, 1), self.router.speed_limit)
        self.speed_ema += alpha * (obs - self.speed_ema)
        self.count_on = count

    def _refresh_costs(self) -> None:
        r = self.router
        v = np.clip(self.speed_ema, 0.5, r.speed_limit)
        queue_delay = self.count_on * 1.5  # vehicles ahead each take ~1.5 s to clear a junction
        costs = r.length / v + queue_delay
        for d in self.closed_dedges:
            costs[r.index[d]] = 1e6
        r.set_costs(costs)
        self._last_cost_update = self.time

    def congestion(self, de: int) -> float:
        """Current speed as a fraction of the limit (1 = free flow)."""
        i = self.router.index[de]
        return float(self.speed_ema[i] / self.router.speed_limit[i])

    # ---------------------------------------------------------------- drawing
    def pose(self, v: Vehicle) -> tuple[float, float, float]:
        """Centre of the car (x, y, yaw)."""
        if v.spot is not None and (v.state in ("parked", "wait_out") or (v.state == "dwell" and not v.double_parked)):
            return self.net.spots[v.spot].pose
        path = self.net.lanes[v.elem].path if v.on == LANE else self.net.connectors[v.elem].path
        s = v.s - v.length / 2
        if s < 0 and v.prev_path is not None:
            pp = v.prev_path
            x, y, h = pp.pose_at(pp.length + s)
        else:
            x, y, h = path.pose_at(max(s, 0.0))
        if v.state == "aside":
            lane = self.net.lanes[v.elem]
            x, y, h = lane.path.pose_at(max(v.s - v.length / 2, 0.0))
            off = lane.width * 0.75
            return x + math.sin(h) * off, y - math.cos(h) * off, h
        if v.lc_offset and v.on == LANE:
            off = v.lc_offset * max(v.lc_cooldown - 1.0, 0.0) / 3.0
            x, y = x + math.sin(h) * off, y - math.cos(h) * off
        if v.state == "maneuver_in" and v.spot is not None:
            sx, sy, sh = self.net.spots[v.spot].pose
            f = min(max(1.0 - v.timer / 4.0, 0.0), 1.0)
            x, y = x + (sx - x) * f, y + (sy - y) * f
        return x, y, h

    def elevation(self, v: Vehicle) -> float:
        """Height above the ground of a car (only non-zero on the elevated highway)."""
        if v.elem < 0 or v.state not in ("drive", "maneuver_in", "dwell"):
            return 0.0
        if v.on == LANE:
            return self.net.lane_z(v.elem, max(v.s - v.length / 2, 0.0))
        c = self.net.connectors[v.elem]
        frac = v.s / max(c.length, 1e-6)
        return (1 - frac) * self.net.lane_z(c.from_lane, self.net.lanes[c.from_lane].length) + frac * self.net.lane_z(c.to_lane, 0.0)

    def snapshot(self) -> dict:
        moving = [v for v in self.vehicles.values() if self.in_traffic(v)]
        speeds = [v.v for v in moving if v.state == "drive"]
        return {
            "time": self.time,
            "vehicles": len(self.vehicles),
            "moving": len(moving),
            "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
            "stopped_share": float(np.mean([s < 0.5 for s in speeds])) if speeds else 0.0,
            "parked": int(np.count_nonzero(self.spot_state == PARKED)),
            "spots": len(self.spot_state),
            "max_kerb_occupancy": max((self.kerb_occupancy(k) for k in self.kerb_spots), default=0.0),
        }
