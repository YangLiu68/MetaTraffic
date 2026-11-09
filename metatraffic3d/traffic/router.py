"""Travel-time routing over directed road segments.

``cost_to_go(dest)`` runs one backward Dijkstra and returns, for every directed edge, the
time from its start to the destination. Vehicles pick their next edge greedily from that
table at every junction, so routes adapt to congestion as the table is refreshed.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from ..roads import RoadNetwork

TURN_COST = {"S": 1.0, "R": 3.0, "L": 6.0, "U": 20.0}


class Router:
    def __init__(self, net: RoadNetwork) -> None:
        self.net = net
        self.ids = sorted(net.dedges)
        self.index = {de: i for i, de in enumerate(self.ids)}
        n = len(self.ids)
        self.length = np.array([net.dedges[d].length for d in self.ids])
        self.speed_limit = np.array([net.dedges[d].cls.speed_limit for d in self.ids])
        # junction crossing length is folded into the turn cost of the next edge
        self.free_time = self.length / self.speed_limit
        self.pred: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        self.succ: list[list[tuple[int, str]]] = [[] for _ in range(n)]
        for d in self.ids:
            i = self.index[d]
            for d2, mv in net.dedges[d].succ.items():
                j = self.index[d2]
                self.succ[i].append((j, mv))
                self.pred[j].append((i, TURN_COST[mv]))
        self.costs = self.free_time.copy()
        self.version = 0
        self._cache: dict[tuple[int, int, bool], np.ndarray] = {}

    def set_costs(self, costs: np.ndarray) -> None:
        self.costs = np.maximum(costs, 1e-3)
        self.version += 1
        self._cache.clear()

    def edge_time(self, de: int, congested: bool = True) -> float:
        i = self.index[de]
        return float(self.costs[i] if congested else self.free_time[i])

    def cost_to_go(self, dest_de: int, dest_frac: float, congested: bool = True) -> np.ndarray:
        """Seconds from the START of each directed edge to a point ``dest_frac`` along ``dest_de``."""
        di = self.index[dest_de]
        key = (di, int(round(dest_frac * 20)), congested)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        cost = self.costs if congested else self.free_time
        dist = np.full(len(self.ids), math.inf)
        dist[di] = cost[di] * dest_frac
        heap = [(dist[di], di)]
        while heap:
            d, j = heapq.heappop(heap)
            if d > dist[j]:
                continue
            for i, turn in self.pred[j]:
                nd = d + turn + cost[i]
                if nd < dist[i]:
                    dist[i] = nd
                    heapq.heappush(heap, (nd, i))
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = dist
        return dist

    def options(self, de: int, allowed: list[int], ctg: np.ndarray) -> list[tuple[int, float, str]]:
        """Successors of ``de`` reachable from the vehicle's lane, with ETA via each (sorted)."""
        out = []
        succ = self.net.dedges[de].succ
        for d2 in allowed:
            j = self.index[d2]
            eta = TURN_COST[succ[d2]] + ctg[j]
            if eta < 1e5:  # closed streets cost 1e6
                out.append((d2, eta, succ[d2]))
        out.sort(key=lambda t: t[1])
        return out

    def eta(self, de: int, frac_done: float, ctg: np.ndarray, dest_de: int, dest_frac: float,
            congested: bool = True) -> float:  # fmt: skip
        """Time from a point on ``de`` to the destination."""
        i = self.index[de]
        cost = self.costs if congested else self.free_time
        if de == dest_de and frac_done <= dest_frac:
            return float(cost[i] * (dest_frac - frac_done))
        best = math.inf
        for j, mv in self.succ[i]:
            best = min(best, TURN_COST[mv] + ctg[j])
        return float(cost[i] * (1.0 - frac_done) + best)

    def path(self, de: int, ctg: np.ndarray, dest_de: int, max_len: int = 400) -> list[int]:
        """Greedy successor chain from ``de`` to the destination (for drawing and features)."""
        out = [de]
        seen = {de}
        cur = de
        while cur != dest_de and len(out) < max_len:
            i = self.index[cur]
            best, nxt = math.inf, None
            for j, mv in self.succ[i]:
                c = TURN_COST[mv] + ctg[j]
                if c < best:
                    best, nxt = c, self.ids[j]
            if nxt is None or nxt in seen:
                break
            out.append(nxt)
            seen.add(nxt)
            cur = nxt
        return out
