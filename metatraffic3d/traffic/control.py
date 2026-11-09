"""Junction control: traffic lights, all-way stops, two-way stops.

Rules (like most North-American cities):
* where two arterials (main or major roads) cross, or an arterial meets a full crossroads,
  there are traffic lights;
* a side street meeting an arterial at a T stops (two-way stop); the arterial has priority;
* minor x minor junctions are all-way stops in residential and industrial districts, and
  about a third of them get lights in the CBD and commercial districts;
* bends and dead ends have no control.

Signals run fixed-time plans with two phases (one per road axis), yellow 3 s and all-red 2 s,
and a random offset per junction. Pedestrians walk across the legs of the red axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..geometry import wrap_angle
from ..roads import RoadNetwork

YELLOW, ALL_RED = 3.0, 2.0


@dataclass
class JunctionControl:
    kind: str = "none"  # none | signal | all_stop | two_stop
    phases: list[list[int]] = field(default_factory=list)  # in-dedge groups, one per phase
    green: list[float] = field(default_factory=list)
    offset: float = 0.0
    stop_approaches: set[int] = field(default_factory=set)  # in-dedges that must stop
    approach_phase: dict[int, int] = field(default_factory=dict)
    outage: bool = False  # signal failed: behaves as an all-way stop (flashing red)

    @property
    def cycle(self) -> float:
        return sum(self.green) + len(self.green) * (YELLOW + ALL_RED)

    def state(self, de_in: int, t: float) -> str:
        """'G', 'Y' or 'R' for an approach (signals only)."""
        if self.kind != "signal" or self.outage:
            return "R"
        p = (t + self.offset) % self.cycle
        for i, g in enumerate(self.green):
            if p < g:
                cur, sub = i, "G"
                break
            p -= g
            if p < YELLOW:
                cur, sub = i, "Y"
                break
            p -= YELLOW
            if p < ALL_RED:
                cur, sub = -1, "R"
                break
            p -= ALL_RED
        else:
            cur, sub = -1, "R"
        return sub if self.approach_phase.get(de_in, -2) == cur else "R"

    def must_stop(self, de_in: int) -> bool:
        if self.kind == "signal":
            return self.outage
        return self.kind == "all_stop" or de_in in self.stop_approaches


def _axis_groups(net: RoadNetwork, node) -> list[list[int]]:
    """Split the approaches of a junction into two groups by road axis."""
    ins = list(node.in_dedges)
    if not ins:
        return []
    heads = {d: net.lanes[net.dedges[d].drive[0]].path.end_heading for d in ins}
    # reference axis: the approach on the most important road
    ref = max(ins, key=lambda d: (net.dedges[d].cls.priority, net.dedges[d].length))
    a, b = [], []
    for d in ins:
        diff = abs(wrap_angle(2 * (heads[d] - heads[ref]))) / 2  # angle between axes, 0..pi/2
        (a if diff < math.pi / 4 else b).append(d)
    return [g for g in (a, b) if g]


def assign_controls(net: RoadNetwork, node_zone: dict[int, str] | None = None, seed: int = 0) -> dict[str, int]:
    rng = np.random.default_rng(seed + 31)
    node_zone = node_zone or {}
    counts: dict[str, int] = {}
    for node in net.nodes:
        ctl = JunctionControl()
        if node.degree >= 3:
            n = {0: 0, 1: 0, 2: 0}
            for e in node.edges:
                n[min(net.edges[e].cls.priority, 2)] += 1  # highway ramps count as arterials
            zone = node_zone.get(node.id, "residential")
            top = 2 if n[2] else (1 if n[1] else 0)
            through = n[top]  # legs of the most important road class
            side = node.degree - through
            crossing_arterials = through >= 3 or (top == 2 and n[1] > 0)
            crossroads = through >= 2 and side >= 2
            if top > 0 and (crossing_arterials or (crossroads and (top == 2 or rng.random() < 0.5))):
                ctl.kind = "signal"  # two arterials cross, or a full crossroads on an arterial
            elif top > 0 and side >= 1:
                ctl.kind = "two_stop"  # side street meets an arterial: the side street stops
                ctl.stop_approaches = {d for d in node.in_dedges if min(net.dedges[d].cls.priority, 2) < top}
            elif zone in ("cbd", "commercial") and rng.random() < 0.35:
                ctl.kind = "signal"
            else:
                ctl.kind = "all_stop"
            if ctl.kind == "signal":
                ctl.phases = _axis_groups(net, node)
                if len(ctl.phases) < 2:
                    ctl.kind = "all_stop"
                else:
                    for i, grp in enumerate(ctl.phases):
                        main = max(net.dedges[d].cls.priority for d in grp)
                        ctl.green.append(32.0 if main == 2 else 22.0)
                        for d in grp:
                            ctl.approach_phase[d] = i
                    ctl.offset = float(rng.uniform(0, ctl.cycle))
        node.control = ctl
        counts[ctl.kind] = counts.get(ctl.kind, 0) + 1
    return counts
