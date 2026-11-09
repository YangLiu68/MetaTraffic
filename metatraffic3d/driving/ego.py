"""Keep the physical ego car in the traffic simulation and judge how it drives.

The ego is an *external* vehicle in ``TrafficSim``: other cars, junctions and pedestrians see
it in its lane, but its position comes from MuJoCo. Each control step ``EgoTracker.update``
projects the car onto the lane or turn path it is physically on (re-matching it when it
changes lanes or turns somewhere unexpected), and records infractions:

    collision, off_road, wrong_way, red_light, stop_sign, speeding, route_deviation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely import STRtree

from ..geometry import wrap_angle
from ..traffic.sim import CONN, LANE, TrafficSim, Vehicle


@dataclass
class Infractions:
    events: list[dict] = field(default_factory=list)

    def add(self, kind: str, t: float, **info) -> None:
        self.events.append({"kind": kind, "t": round(t, 2), **info})

    def count(self, kind: str | None = None) -> int:
        return sum(1 for e in self.events if kind is None or e["kind"] == kind)


class EgoController:
    """Sim controller hooks for the ego: stop at the destination kerb, report arrival."""

    def __init__(self) -> None:
        self.arrived = False

    def choose_next(self, veh, options):
        return options[0][0]

    def plan_stop(self, veh):
        return veh.dest[1]

    def on_arrive(self, veh):
        self.arrived = True

    def on_dwell_done(self, veh):
        pass


class EgoTracker:
    def __init__(self, sim: TrafficSim, length: float = 4.6) -> None:
        self.sim = sim
        self.net = sim.net
        self.length = length
        lanes = [lane for lane in self.net.lanes if lane.kind == "drive"]
        self._lane_ids = np.array([lane.id for lane in lanes])
        self._lane_tree = STRtree([shapely.LineString(lane.path.pts) for lane in lanes])
        self.road = None  # paved area for off-road checks (set by the env)
        self.veh: Vehicle | None = None
        self.ctrl = EgoController()
        self.inf = Infractions()

    # ------------------------------------------------------------------ setup
    def spawn(self, lane_id: int, s: float, dest: tuple[int, float]) -> Vehicle:
        sim = self.sim
        v = sim._new_vehicle("ego", color=(0.9, 0.9, 0.95), length=self.length, controller=self.ctrl)
        v.external = True
        v.dest = dest
        sim._insert_lane(v, lane_id, s)
        self.veh = v
        self.ctrl.arrived = False
        self.inf = Infractions()
        self._last_flags: dict[str, float] = {}
        self.route_start = sim.time
        return v

    def pose_on_lane(self, lane_id: int, s: float) -> tuple[float, float, float]:
        """Centre pose of a car whose front bumper is at ``s`` on a lane."""
        return self.net.lanes[lane_id].path.pose_at(max(s - self.length / 2, 0.0))

    # ------------------------------------------------------------------ update from physics
    def update(self, x: float, y: float, yaw: float, speed: float, dt: float = 0.1, z_rel: float = 0.0) -> None:
        """``z_rel``: height of the car above the ground (tells the highway deck from the street below)."""
        self.z_rel = z_rel
        self.offroad = False  # set again below if the car is off the road this step
        sim, net, v = self.sim, self.net, self.veh
        if v is None or v.removed or v.state != "drive":
            return
        fx, fy = x + math.cos(yaw) * self.length / 2, y + math.sin(yaw) * self.length / 2
        v.v = max(speed, 0.0)
        v.still = v.still + dt if v.v < 0.2 else 0.0
        if v.on == LANE:
            lane = net.lanes[v.elem]
            s, lat = lane.path.project((fx, fy))
            if abs(lat) > lane.width * 0.6 or abs(net.lane_z(lane.id, max(s, 0.0)) - z_rel) > 3.0:
                self._rematch(fx, fy, yaw)
                return
            if s >= lane.length - 0.05:
                self._enter_junction(fx, fy, yaw)
                return
            self._set_s(v, s)
            self._check_lane(lane, yaw, v.v)
            if v.dest is not None and lane.dedge == v.dest[0] and abs(v.dest[1] - s) < 4.0 and v.v < 1.0:
                self.ctrl.arrived = True
        else:
            c = net.connectors[v.elem]
            s, lat = c.path.project((fx, fy))
            if abs(lat) > 3.5:  # turning somewhere else: pick the turn path that fits best
                best = self._best_connector(c.node, net.connectors[v.elem].from_lane, fx, fy)
                if best is not None and best != c.id:
                    self._move_conn(v, best)
                    c = net.connectors[best]
                    s, lat = c.path.project((fx, fy))
            if s >= c.length - 0.05:
                out = net.lanes[c.to_lane]
                s_out, _ = out.path.project((fx, fy))
                sim.conn_veh[c.id].remove(v)
                v.prev_path = c.path
                v.clearing = c.id
                sim._append_lane(v, out.id, max(s_out, 0.0))
                v.next_de, v.conn, v.at_line = None, None, False
                self._resort(out.id)
                return
            v.s = max(s, 0.0)
        self._check_road(x, y)

    def _set_s(self, v: Vehicle, s: float) -> None:
        v.s = max(s, 0.0)
        self._resort(v.elem)
        if v.clearing is not None and v.s >= v.length:
            self.sim.conn_busy[v.clearing] -= 1
            v.clearing = None

    def _resort(self, lane_id: int) -> None:
        self.sim.lane_veh[lane_id].sort(key=lambda o: -o.s)

    def _check_lane(self, lane, yaw: float, speed: float) -> None:
        t = self.sim.time
        if abs(wrap_angle(yaw - lane.path.heading_at(self.veh.s))) > 2.2 and speed > 1.0:
            self._flag("wrong_way", t)
        if speed > lane.speed_limit * 1.15 + 1.0:
            self._flag("speeding", t, limit=round(lane.speed_limit * 3.6))

    def _check_road(self, x: float, y: float) -> None:
        if self.road is not None and not shapely.contains_xy(self.road, x, y):
            self._flag("off_road", self.sim.time)

    def _flag(self, kind: str, t: float, cooldown: float = 3.0, **info) -> None:
        """Record an infraction (at most once per ``cooldown`` seconds while it continues)."""
        if kind == "off_road":
            self.offroad = True
        if t - self._last_flags.get(kind, -1e9) >= cooldown:
            self.inf.add(kind, t, **info)
            self._last_flags[kind] = t

    def _rematch(self, fx: float, fy: float, yaw: float) -> None:
        """Associate the ego with the nearest driving lane heading the same way."""
        sim, net, v = self.sim, self.net, self.veh
        cand = self._lane_tree.query(shapely.Point(fx, fy).buffer(4.0))
        best, best_d = None, math.inf
        for i in cand:
            lane = net.lanes[int(self._lane_ids[i])]
            s, lat = lane.path.project((fx, fy))
            if abs(wrap_angle(yaw - lane.path.heading_at(s))) > 1.2:
                continue
            if abs(net.lane_z(lane.id, max(s, 0.0)) - getattr(self, "z_rel", 0.0)) > 3.0:
                continue  # a road above or below us
            if abs(lat) < best_d:
                best, best_d = (lane.id, s), abs(lat)
        if best is None or best_d > 3.0:
            self._flag("off_road", sim.time)
            return
        lane_id, s = best
        old = net.lanes[v.elem]
        if lane_id == v.elem:
            self._set_s(v, s)
            return
        sim._drop_junction(v)
        sim.lane_veh[v.elem].remove(v)
        sim._append_lane(v, lane_id, max(s, 0.0))
        self._resort(lane_id)
        new = net.lanes[lane_id]
        if new.dedge != old.dedge:
            self._flag("route_deviation", sim.time, cooldown=0.0)
            v.next_de = None

    def _best_connector(self, node: int, from_lane: int, fx: float, fy: float) -> int | None:
        net = self.net
        lane = net.lanes[from_lane]
        cands = [cid for cids in lane.out_conns.values() for cid in cids]
        if not cands:
            cands = [cid for cid in net.nodes[node].connectors]
        best, best_d = None, math.inf
        for cid in cands:
            _, lat = net.connectors[cid].path.project((fx, fy))
            if abs(lat) < best_d:
                best, best_d = cid, abs(lat)
        return best

    def _move_conn(self, v: Vehicle, cid: int) -> None:
        sim = self.sim
        if v in sim.conn_veh[v.elem]:
            sim.conn_veh[v.elem].remove(v)
        sim.conn_busy[v.elem] -= 1
        sim.conn_busy[cid] += 1
        v.elem = cid
        sim.conn_veh[cid].append(v)
        sim.conn_veh[cid].sort(key=lambda o: -o.s)

    def _enter_junction(self, fx: float, fy: float, yaw: float) -> None:
        """The ego crossed the stop line: move it into a turn path, judging whether it was allowed."""
        sim, net, v = self.sim, self.net, self.veh
        lane = net.lanes[v.elem]
        if not lane.out_conns:
            self._flag("off_road", sim.time)
            return
        node = net.dedges[lane.dedge].end
        ctl = net.nodes[node].control
        if v.granted and v.conn is not None:
            cid = v.conn
        else:
            ahead = (fx + math.cos(yaw) * 4.0, fy + math.sin(yaw) * 4.0)
            cid = self._best_connector(node, lane.id, *ahead)
            if ctl is not None and ctl.kind == "signal" and not ctl.outage:
                if ctl.state(lane.dedge, sim.time) == "R":
                    self.inf.add("red_light", sim.time, node=node)
            elif ctl is not None and ctl.must_stop(lane.dedge) and not v.at_line:
                self.inf.add("stop_sign", sim.time, node=node)
            sim._drop_junction(v)
            sim.conn_busy[cid] += 1
        sim.lane_veh[lane.id].remove(v)
        if v.clearing is not None:
            sim.conn_busy[v.clearing] -= 1
            v.clearing = None
        v.prev_path = lane.path
        v.on, v.elem, v.s = CONN, cid, 0.0
        v.requested = v.granted = False
        sim.conn_veh[cid].append(v)

    # ------------------------------------------------------------------ guidance
    def command(self) -> tuple[str, float]:
        """High-level route command for the next junction and the distance to it.

        One of: follow, left, right, straight, uturn, arrive (conditional imitation learning).
        """
        v, net = self.veh, self.net
        if v is None:
            return "follow", 0.0
        if v.on == CONN:
            return {"L": "left", "R": "right", "S": "straight", "U": "uturn"}[net.connectors[v.elem].movement], 0.0
        lane = net.lanes[v.elem]
        dist = lane.length - v.s
        if v.dest is not None and lane.dedge == v.dest[0] and v.s <= v.dest[1] + 0.5:
            return "arrive", max(v.dest[1] - v.s, 0.0)
        nxt = v.next_de
        if nxt is None:
            return "follow", dist
        mv = net.dedges[lane.dedge].succ.get(nxt, "S")
        name = {"L": "left", "R": "right", "S": "straight", "U": "uturn"}[mv]
        return (name if dist < 60.0 else "follow"), dist

    def path_ahead(self, length: float = 60.0) -> np.ndarray:
        """Centre line the ego should follow next (current lane, planned turn, next lane)."""
        v, net, sim = self.veh, self.net, self.sim
        pts = []
        if v.on == LANE:
            lane = net.lanes[v.elem]
            pts.append(lane.path.sub(max(v.s - self.length, 0.0), lane.length))
            cid = v.conn
            if cid is None and v.next_de is not None:
                cids = lane.out_conns.get(v.next_de)
                cid = cids[0] if cids else None
            if cid is not None:
                c = net.connectors[cid]
                pts.append(c.path.pts)
                pts.append(net.lanes[c.to_lane].path.sub(0, length))
        else:
            c = net.connectors[v.elem]
            pts.append(c.path.sub(max(v.s - self.length, 0.0), c.length))
            pts.append(net.lanes[c.to_lane].path.sub(0, length))
        del sim
        return np.vstack(pts)
