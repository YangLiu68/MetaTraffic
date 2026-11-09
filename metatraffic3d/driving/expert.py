"""Privileged expert driver: the teacher that produces demonstrations for pre-training.

It sees the simulator state (lane geometry, the planned route, the junction manager's
permission and the IDM advice that already accounts for lights, stop signs, pedestrians,
closures and the car ahead) and outputs the same direct controls the learned policy must
produce: steer, throttle, brake and gear. Like the CARLA autopilot used by LBC / Roach /
TransFuser, it is a rule-based driver; the camera policy learns to imitate it, then RL
improves on it.
"""

from __future__ import annotations

import math

import numpy as np

from ..backends.mujoco.vehicle import GEARS, STEER_MAX, WHEEL_BASE, EgoState, auto_gear
from ..geometry import Polyline, offset_points
from ..traffic.sim import LANE
from .ego import EgoTracker


class ExpertDriver:
    def __init__(self, tracker: EgoTracker, noise: float = 0.0, rng: np.random.Generator | None = None) -> None:
        self.tr = tracker
        self.noise = noise  # DART-style action noise for data collection (labels stay clean)
        self.rng = rng or np.random.default_rng(0)
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self.gear = "1"
        self.lc_lane: int | None = None

    # ---------------------------------------------------------------- lateral
    def _target_path(self) -> np.ndarray:
        tr, v, net = self.tr, self.tr.veh, self.tr.net
        if v.on == LANE:
            lane = net.lanes[v.elem]
            de = net.dedges[lane.dedge]
            # lane change needed for the planned turn?
            if len(de.drive) > 1 and v.next_de is not None and v.next_de not in lane.out_conns and not v.granted:
                ok = [li for li in de.drive if v.next_de in net.lanes[li].out_conns]
                if ok:
                    k = lane.index
                    tk = min((net.lanes[li].index for li in ok), key=lambda t: abs(t - k))
                    nxt = de.drive[k + (1 if tk > k else -1)]
                    s_new = v.s * net.lanes[nxt].length / max(lane.length, 1e-3)
                    if self.lc_lane == nxt or tr.sim._gap_for_change(nxt, s_new, v):
                        self.lc_lane = nxt
                        target = net.lanes[nxt]
                        return target.path.sub(max(s_new - 5, 0), min(s_new + 60, target.length))
            self.lc_lane = None
        return tr.path_ahead(80.0)

    def _steer(self, st: EgoState, path: np.ndarray) -> float:
        pl = Polyline(path)
        rx = st.x - math.cos(st.yaw) * WHEEL_BASE / 2
        ry = st.y - math.sin(st.yaw) * WHEEL_BASE / 2
        s0, _ = pl.project((rx, ry))
        ld = float(np.clip(4.0 + 0.55 * abs(st.speed), 5.0, 22.0))
        tx, ty = pl.point_at(s0 + ld)
        if s0 + ld > pl.length:  # extend straight past the end of the known path
            h = pl.end_heading
            extra = s0 + ld - pl.length
            tx, ty = tx + math.cos(h) * extra, ty + math.sin(h) * extra
        alpha = math.atan2(ty - ry, tx - rx) - st.yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        delta = math.atan2(2 * WHEEL_BASE * math.sin(alpha), ld)
        return float(np.clip(delta / STEER_MAX, -1.0, 1.0))

    # ---------------------------------------------------------------- longitudinal
    def _pedals(self, st: EgoState, dt: float) -> tuple[float, float]:
        v = self.tr.veh
        lane = self.tr.net.lanes[v.elem] if v.on == LANE else None
        limit = lane.speed_limit if lane is not None else self.tr.net.connectors[v.elem].speed_limit
        a_adv = float(np.clip(v.a, -8.0, 2.5))
        cmd, dist = self.tr.command()
        v_ref = float(np.clip(st.speed + a_adv * 0.6, 0.0, limit))
        if cmd in ("left", "right", "uturn") and dist < 25:
            v_ref = min(v_ref, 6.0)
        err = v_ref - st.speed
        grade = 9.81 * math.sin(-st.pitch)  # uphill needs more throttle
        if v_ref < 0.2 and st.speed < 0.6:
            self.integral = 0.0
            return 0.0, 0.45  # hold at standstill
        if err < -0.4 or a_adv < -1.5:
            self.integral = 0.0
            return 0.0, float(np.clip(-a_adv / 7.0 + max(-err, 0.0) * 0.15, 0.05, 1.0))
        self.integral = float(np.clip(self.integral + err * dt, -3.0, 3.0))
        throttle = 0.12 * err + 0.05 * self.integral + 0.03 * grade + 0.04 * v_ref / 10.0
        return float(np.clip(throttle, 0.0, 1.0)), 0.0

    # ---------------------------------------------------------------- action
    def act(self, st: EgoState, dt: float = 0.1) -> tuple[float, float, float, int]:
        steer = self._steer(st, self._target_path())
        throttle, brake = self._pedals(st, dt)
        self.gear = auto_gear(self.gear if not st.shifting else self.gear, st.speed, throttle)
        return steer, throttle, brake, GEARS.index(self.gear)

    def noisy(self, action: tuple[float, float, float, int]) -> tuple[float, float, float, int]:
        """Perturbed action actually executed while recording (DART); the label stays ``action``."""
        if self.noise <= 0:
            return action
        steer, throttle, brake, gear = action
        steer = float(np.clip(steer + self.rng.normal(0, 0.25 * self.noise), -1, 1))
        if brake == 0.0:
            throttle = float(np.clip(throttle + self.rng.normal(0, 0.2 * self.noise), 0, 1))
        return steer, throttle, brake, gear


__all__ = ["ExpertDriver", "offset_points"]
