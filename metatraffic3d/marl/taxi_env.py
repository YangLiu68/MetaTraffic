"""Multi-agent taxi environment (PettingZoo ``ParallelEnv`` API, no PettingZoo dependency).

Every taxi is an agent. Passengers appear at kerbs (more often near tall buildings) and wait
a few minutes. Agents make two kinds of decisions, only when one is due:

* dispatch: an idle taxi picks one of the nearest open requests, or waits;
* route: approaching a junction with several sensible ways on, the taxi picks the next street.

All other agents get a single "continue" option in that step. Observations list the options
as feature rows, so one scoring network handles both decision kinds (see ``mappo.py``).

Rewards: fare on delivery, minus driving time, minus passenger waiting at pickup, minus
penalties for abandoned pickups and for double-parking (no free kerb spot).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import SimConfig
from ..traffic.demand import demand_weights
from ..traffic.sim import LANE, TrafficSim, Vehicle
from ..world import World, build_world

NONE, DISPATCH, ROUTE = 0, 1, 2
IDLE, TO_PICKUP, TO_DROPOFF, REPOSITION = 0, 1, 2, 3
TURNS = ("S", "L", "R", "U")

# option feature layout
F_KIND = 0  # 3 one-hot: wait/continue, request, route
F_REQ = 3  # 10 request features
F_ROUTE = 13  # 10 route features
N_OPT_FEATS = 23
N_SELF_FEATS = 14


@dataclass
class Request:
    id: int
    origin: tuple[int, float]
    dest: tuple[int, float]
    t_request: float
    patience: float
    trip_km: float
    trip_free: float  # free-flow trip time, seconds
    claimed_by: int | None = None
    picked_at: float | None = None
    delivered_at: float | None = None
    xy: tuple[float, float] = (0.0, 0.0)


@dataclass
class TaxiState:
    idx: int
    veh: Vehicle | None = None
    status: int = IDLE
    request: Request | None = None
    decision: int = NONE
    options: list = field(default_factory=list)  # raw option payloads for the pending decision
    route_answer: tuple[int, int] | None = None  # (lane id, chosen next edge)
    wait_until: float = 0.0
    drive_since: float = 0.0
    reward: float = 0.0
    last_decision_time: float = 0.0
    # episode metrics
    delivered: int = 0
    abandoned: int = 0
    double_parks: int = 0
    drive_seconds: float = 0.0
    earned: float = 0.0


class TaxiController:
    """Plugs one taxi into the traffic simulation (see ``traffic.sim.Controller``)."""

    def __init__(self, env: TaxiEnv, taxi: TaxiState) -> None:
        self.env = env
        self.taxi = taxi

    def choose_next(self, veh, options):
        return self.env._route_choice(self.taxi, veh, options)

    def plan_stop(self, veh):
        return self.env._plan_stop(self.taxi, veh)

    def on_arrive(self, veh):
        self.env._on_arrive(self.taxi, veh)

    def on_dwell_done(self, veh):
        self.env._on_dwell_done(self.taxi, veh)


class TaxiEnv:
    metadata = {"name": "metatraffic_taxi_v0", "render_modes": []}

    def __init__(self, cfg: SimConfig | None = None, world: World | None = None, seed: int = 0) -> None:
        self.cfg = cfg or SimConfig()
        self.world = world or build_world(self.cfg)
        self.tx = self.cfg.taxi
        self.sim = TrafficSim(self.world.net, self.cfg, demand_weights(self.world), seed=seed)
        self.n = self.tx.num_taxis
        self.possible_agents = [f"taxi_{i}" for i in range(self.n)]
        self.agents = list(self.possible_agents)
        self.k_opts = max(self.tx.max_request_options + 1, self.tx.max_route_options)
        xmin, ymin, xmax, ymax = self.world.bounds
        self._origin = np.array([xmin, ymin])
        self._scale = np.array([xmax - xmin, ymax - ymin])
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.taxis: list[TaxiState] = []
        self.on_tick = None  # optional callback after every simulation tick (rendering)

    # ------------------------------------------------------------------ spaces
    def observation_space(self, agent: str):
        from gymnasium import spaces

        k = self.k_opts
        return spaces.Dict({
            "options": spaces.Box(-np.inf, np.inf, (k, N_OPT_FEATS), np.float32),
            "mask": spaces.MultiBinary(k),
            "self": spaces.Box(-np.inf, np.inf, (N_SELF_FEATS,), np.float32),
            "kind": spaces.Discrete(3),
        })  # fmt: skip

    def action_space(self, agent: str):
        from gymnasium import spaces

        return spaces.Discrete(self.k_opts)

    # ------------------------------------------------------------------ episode
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self.sim.reset(seed=self._seed)
        self.agents = list(self.possible_agents)
        self.requests: dict[int, Request] = {}
        self._req_id = 0
        self.finished: list[Request] = []
        self.expired_unclaimed = 0
        self.taxis = []
        for i in range(self.n):
            t = TaxiState(i)
            t.veh = self.sim.add_taxi(TaxiController(self, t))
            self.taxis.append(t)
        for _ in range(int(self.tx.request_rate_per_min * 3)):  # a few passengers already waiting
            self._spawn_request()
        for t in self.taxis:
            self._ask_dispatch(t)
        return self._observe(), self._infos()

    def step(self, actions: dict[str, int]):
        for t in self.taxis:
            a = actions.get(self.possible_agents[t.idx], 0)
            if t.decision != NONE:
                self._apply(t, int(a))
        end = self.tx.episode_seconds
        while self.sim.time < end and not any(t.decision != NONE for t in self.taxis):
            self._tick()
        done = self.sim.time >= end
        rewards = {}
        raw = np.array([t.reward for t in self.taxis])
        mix = self.tx.team_reward
        for t in self.taxis:
            rewards[self.possible_agents[t.idx]] = float((1 - mix) * raw[t.idx] + mix * raw.mean())
            t.reward = 0.0
        obs = self._observe()
        terms = {a: False for a in self.agents}
        truncs = {a: done for a in self.agents}
        infos = self._infos()
        if done:
            for a in self.agents:
                infos[a]["episode"] = self.metrics()
            self.agents = []
        return obs, rewards, terms, truncs, infos

    def _tick(self) -> None:
        dt = self.cfg.traffic.dt
        tx = self.tx
        if self.rng.random() < tx.request_rate_per_min * dt / 60.0:
            self._spawn_request()
        self.sim.step(dt)
        for t in self.taxis:
            if t.status in (TO_PICKUP, TO_DROPOFF, REPOSITION) and t.veh.state in ("drive", "maneuver_in"):
                t.reward -= tx.drive_cost_per_min * dt / 60.0
                t.drive_seconds += dt
            if t.status == IDLE and t.decision == NONE and self.sim.time >= t.wait_until and t.veh.state == "parked":
                self._ask_dispatch(t)
        self._expire_requests()
        if self.on_tick is not None:
            self.on_tick(self)

    # ------------------------------------------------------------------ requests
    def _spawn_request(self) -> None:
        sim = self.sim
        for _ in range(20):
            o = sim.random_kerb_point(self.rng)
            d = sim.random_kerb_point(self.rng)
            po = self._xy(*o)
            pd = self._xy(*d)
            if math.dist(po, pd) >= self.tx.min_trip:
                break
        else:
            return
        router = sim.router
        ctg = router.cost_to_go(d[0], self._frac(*d), congested=False)
        free = router.eta(o[0], self._frac(*o), ctg, d[0], self._frac(*d), congested=False)
        if not math.isfinite(free):
            return
        path = router.path(o[0], ctg, d[0])
        if path[-1] != d[0] or (o[0] == d[0] and o[1] > d[1]):
            return  # unusual geometry (a loop back onto the same street): skip this request
        metres = sum(router.length[router.index[x]] for x in path) - o[1] - (router.length[router.index[d[0]]] - d[1])
        km = max(metres, 0.0) / 1000.0
        lo, hi = self.tx.patience
        r = Request(self._req_id, o, d, sim.time, float(self.rng.uniform(lo, hi)), km, free, xy=po)
        self._req_id += 1
        self.requests[r.id] = r

    def _expire_requests(self) -> None:
        now = self.sim.time
        for r in list(self.requests.values()):
            if r.picked_at is None and now - r.t_request > r.patience:
                del self.requests[r.id]
                if r.claimed_by is not None:
                    t = self.taxis[r.claimed_by]
                    t.reward -= self.tx.abandon_penalty
                    t.abandoned += 1
                    t.request = None
                    self._become_idle(t)
                else:
                    self.expired_unclaimed += 1

    # ------------------------------------------------------------------ decisions
    def _ask_dispatch(self, t: TaxiState) -> None:
        t.decision = DISPATCH
        t.options = [("wait", None)]
        loc = self._location(t.veh)
        open_reqs = [r for r in self.requests.values() if r.claimed_by is None]
        scored = []
        for r in open_reqs:
            eta = self._eta(loc, r.origin, True)
            if math.isfinite(eta):
                scored.append((eta, r))
        scored.sort(key=lambda x: x[0])
        for eta, r in scored[: self.k_opts - 1]:
            t.options.append(("request", r, eta))
        t.last_decision_time = self.sim.time

    def _route_choice(self, t: TaxiState, veh: Vehicle, options) -> int | None:
        if t.route_answer is not None:
            lane, choice = t.route_answer
            t.route_answer = None
            if lane == veh.elem and any(o[0] == choice for o in options):
                return choice
        best = options[0][1]
        keep = [o for o in options if o[1] <= best * self.tx.route_slack + 60.0][: self.tx.max_route_options]
        if len(keep) == 1 or t.status in (IDLE, REPOSITION):
            return keep[0][0]
        if t.decision == ROUTE:
            return None  # this taxi's route choice is already waiting for the agent
        if t.decision != NONE:  # a dispatch is pending; just take the best street
            return keep[0][0]
        t.decision = ROUTE
        t.options = [("route", o) for o in keep]
        t.last_decision_time = self.sim.time
        return None

    def _apply(self, t: TaxiState, a: int) -> None:
        kind, opts = t.decision, t.options
        t.decision, t.options = NONE, []
        a = a if 0 <= a < len(opts) else 0
        if kind == ROUTE:
            choice = opts[a][1][0]
            t.route_answer = (t.veh.elem, choice)
            return
        # dispatch
        if opts[a][0] == "request" and opts[a][1].claimed_by is None and opts[a][1].id in self.requests:
            r = opts[a][1]
            r.claimed_by = t.idx
            t.request = r
            t.status = TO_PICKUP
            self.sim.set_destination(t.veh, *r.origin)
            self._leave_kerb(t)
            return
        # wait (or the chosen passenger was just taken by another taxi)
        if t.veh.state == "parked":
            t.wait_until = self.sim.time + self.tx.wait_action_seconds
        else:
            self._reposition(t)  # cannot wait in a driving lane: go find a kerb spot

    def _leave_kerb(self, t: TaxiState) -> None:
        if t.veh.state in ("parked", "dwell"):
            self.sim.depart(t.veh)
        t.drive_since = self.sim.time

    def _reposition(self, t: TaxiState) -> None:
        sim = self.sim
        de, s = self._location(t.veh)
        target = None
        for radius in (150.0, 400.0):
            sid = sim.free_spot_near(de, s, radius, ahead_only=True)
            if sid is not None:
                target = (de, sim.net.spots[sid].s_drive)
                break
        if target is None:
            target = sim.random_kerb_point(self.rng, weighted=False)
        t.status = REPOSITION
        sim.set_destination(t.veh, *target)
        self._leave_kerb(t)

    def _become_idle(self, t: TaxiState) -> None:
        t.status = IDLE
        t.request = None
        t.wait_until = self.sim.time
        if t.decision == ROUTE:  # drop a pending route choice; the dispatch comes first
            t.decision, t.options = NONE, []
        if t.decision == NONE:
            self._ask_dispatch(t)

    # ------------------------------------------------------------------ kerb stops
    def _plan_stop(self, t: TaxiState, veh: Vehicle) -> float:
        de, s = veh.dest
        sim = self.sim
        sid = sim.free_spot_near(de, s, self.tx.pickup_search)
        if sid is not None and sim.net.spots[sid].s_drive > veh.s + 6.0:
            sim.reserve(veh, sid)
            return sim.net.spots[sid].s_drive
        sid = sim.free_spot_near(de, veh.s + 10.0, self.tx.pickup_search, ahead_only=True)
        if sid is not None:
            sim.reserve(veh, sid)
            return sim.net.spots[sid].s_drive
        return max(s, veh.s + 1.0)

    def _on_arrive(self, t: TaxiState, veh: Vehicle) -> None:
        tx = self.tx
        dwell = {TO_PICKUP: tx.boarding_time, TO_DROPOFF: tx.alighting_time}.get(t.status, 0.0)
        if veh.spot is not None:
            self.sim.begin_pull_in(veh, veh.spot, dwell)
        else:
            self.sim.stop_here(veh, dwell)
            t.reward -= tx.double_park_penalty
            t.double_parks += 1

    def _on_dwell_done(self, t: TaxiState, veh: Vehicle) -> None:
        now = self.sim.time
        if t.status in (TO_PICKUP, TO_DROPOFF) and t.request is not None:
            tgt = t.request.origin if t.status == TO_PICKUP else t.request.dest
            if math.dist(self._xy(*self._location(veh)), self._xy(*tgt)) > 60.0:
                self.sim.set_destination(veh, *tgt)  # stopped somewhere else: keep going
                self._leave_kerb(t)
                return
        if t.status == TO_PICKUP and t.request is not None:
            r = t.request
            r.picked_at = now
            t.reward -= self.tx.wait_penalty_per_min * (now - r.t_request) / 60.0
            t.status = TO_DROPOFF
            self.sim.set_destination(veh, *r.dest)
            self._leave_kerb(t)
        elif t.status == TO_DROPOFF and t.request is not None:
            r = t.request
            fare = self.tx.fare_base + self.tx.fare_per_km * r.trip_km
            t.reward += fare
            t.earned += fare
            t.delivered += 1
            r.delivered_at = now
            self.finished.append(r)
            self.requests.pop(r.id, None)
            self._become_idle(t)
        else:  # parked after repositioning (or a stale target)
            self._become_idle(t)

    # ------------------------------------------------------------------ helpers
    def _location(self, veh: Vehicle) -> tuple[int, float]:
        net = self.sim.net
        if veh.spot is not None and veh.state in ("parked", "dwell", "wait_out") and not veh.double_parked:
            sp = net.spots[veh.spot]
            return net.lanes[sp.drive_lane].dedge, sp.s_drive
        if veh.on == LANE:
            return net.lanes[veh.elem].dedge, veh.s
        c = net.connectors[veh.elem]
        return net.lanes[c.to_lane].dedge, 0.0

    def _frac(self, de: int, s: float) -> float:
        lane = self.sim.net.lanes[self.sim.net.dedges[de].drive[0]]
        return min(max(s / max(lane.length, 1e-3), 0.0), 1.0)

    def _xy(self, de: int, s: float) -> tuple[float, float]:
        lane = self.sim.net.lanes[self.sim.net.dedges[de].drive[0]]
        return lane.path.point_at(s)

    def _eta(self, loc, target, congested: bool) -> float:
        r = self.sim.router
        ctg = r.cost_to_go(target[0], self._frac(*target), congested)
        return r.eta(loc[0], self._frac(*loc), ctg, target[0], self._frac(*target), congested)

    # ------------------------------------------------------------------ observations
    def _self_features(self, t: TaxiState) -> np.ndarray:
        f = np.zeros(N_SELF_FEATS, np.float32)
        f[t.status] = 1.0
        x, y, _ = self.sim.pose(t.veh)
        f[4:6] = (np.array([x, y]) - self._origin) / self._scale
        f[6] = t.veh.v / 14.0
        f[7] = self.sim.time / self.tx.episode_seconds
        f[8] = len([r for r in self.requests.values() if r.claimed_by is None]) / 10.0
        f[9] = sum(1 for o in self.taxis if o.status == IDLE) / self.n
        if t.request is not None:
            tgt = t.request.dest if t.status == TO_DROPOFF else t.request.origin
            f[10] = min(self._eta(self._location(t.veh), tgt, True) / 600.0, 3.0)
        de, _ = self._location(t.veh)
        f[11] = self.sim.congestion(de)
        f[12] = float(t.veh.double_parked)
        f[13] = (self.sim.time - t.drive_since) / 600.0 if t.status != IDLE else 0.0
        return f

    def _option_features(self, t: TaxiState) -> tuple[np.ndarray, np.ndarray]:
        k = self.k_opts
        feats = np.zeros((k, N_OPT_FEATS), np.float32)
        mask = np.zeros(k, np.int8)
        if t.decision == NONE:
            feats[0, F_KIND] = 1.0
            mask[0] = 1
            return feats, mask
        loc = self._location(t.veh)
        now = self.sim.time
        for i, opt in enumerate(t.options[:k]):
            mask[i] = 1
            if opt[0] == "wait":
                feats[i, F_KIND] = 1.0
            elif opt[0] == "request":
                r, eta = opt[1], opt[2]
                feats[i, F_KIND + 1] = 1.0
                free = self._eta(loc, r.origin, False)
                closer = 0
                for o in self.taxis:
                    if o is not t and o.status == IDLE:
                        if self._eta(self._location(o.veh), r.origin, True) < eta:
                            closer += 1
                feats[i, F_REQ:F_REQ + 10] = [
                    eta / 600.0, free / 600.0, r.trip_km / 5.0, r.trip_free / 600.0,
                    (self.tx.fare_base + self.tx.fare_per_km * r.trip_km) / 3.0,
                    (now - r.t_request) / 600.0, (r.patience - (now - r.t_request) - eta) / 600.0,
                    closer / max(self.n - 1, 1), self.sim.congestion(r.origin[0]),
                    self.sim.demand[self.sim.router.index[r.dest[0]]] / (self.sim.demand.mean() * 5),
                ]  # fmt: skip
            else:
                de2, eta, mv = opt[1]
                best = t.options[0][1][1]
                feats[i, F_KIND + 2] = 1.0
                tgt = t.veh.dest
                ctg_free = self.sim.router.cost_to_go(tgt[0], self._frac(*tgt), congested=False)
                j = self.sim.router.index[de2]
                eta_free = ctg_free[j]
                lane0 = self.sim.net.dedges[de2].drive[0]
                feats[i, F_ROUTE:F_ROUTE + 10] = [
                    eta / 600.0, min(eta_free, 1e4) / 600.0, self.sim.router.length[j] / 500.0,
                    self.sim.congestion(de2), len(self.sim.lane_veh[lane0]) / 10.0,
                    *(1.0 if mv == m else 0.0 for m in TURNS), (eta - best) / 60.0,
                ]  # fmt: skip
        return feats, mask

    def _observe(self) -> dict:
        obs = {}
        for t in self.taxis:
            feats, mask = self._option_features(t)
            obs[self.possible_agents[t.idx]] = {"options": feats, "mask": mask,
                                                "self": self._self_features(t), "kind": t.decision}  # fmt: skip
        return obs

    def state(self) -> np.ndarray:
        """Global state for a centralised critic."""
        parts = [self._self_features(t) for t in self.taxis]
        snap = self.sim.snapshot()
        extra = np.array([len(self.requests) / 10.0, snap["mean_speed"] / 14.0, snap["stopped_share"],
                          self.sim.time / self.tx.episode_seconds], np.float32)  # fmt: skip
        return np.concatenate(parts + [extra]).astype(np.float32)

    def _infos(self) -> dict:
        return {self.possible_agents[t.idx]: {"decision": t.decision, "time": self.sim.time} for t in self.taxis}

    # ------------------------------------------------------------------ metrics
    def metrics(self) -> dict:
        done = [r for r in self.finished if r.delivered_at is not None]
        waits = [r.picked_at - r.t_request for r in done]
        rides = [r.delivered_at - r.picked_at for r in done]
        ratio = [(r.delivered_at - r.picked_at) / max(r.trip_free, 1.0) for r in done]
        hours = self.sim.time / 3600.0
        return {
            "delivered": len(done),
            "delivered_per_taxi_hour": len(done) / max(self.n * hours, 1e-9),
            "mean_wait_s": float(np.mean(waits)) if waits else float("nan"),
            "mean_ride_s": float(np.mean(rides)) if rides else float("nan"),
            "ride_vs_free_flow": float(np.mean(ratio)) if ratio else float("nan"),
            "abandoned": sum(t.abandoned for t in self.taxis),
            "expired_unclaimed": self.expired_unclaimed,
            "double_parks": sum(t.double_parks for t in self.taxis),
            "drive_share": sum(t.drive_seconds for t in self.taxis) / max(self.n * self.sim.time, 1e-9),
            "earned": sum(t.earned for t in self.taxis),
            "npc_teleports": self.sim.stats.teleported,
            "requests": self._req_id,
        }

    def close(self) -> None:
        pass
