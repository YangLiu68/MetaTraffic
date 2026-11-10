import math
from dataclasses import dataclass
from typing import Tuple, List
from .constants import (
    CAR_LEN, CAR_WID, NOSE_DOT,
    WHEELBASE, MAX_SPEED_FWD, MAX_SPEED_REV,
    ACCEL, BRAKE, DRAG,
    MAX_STEER_DEG, STEER_RATE_DEG, STEER_RETURN_DEG
)
from .utils import clamp, rad

@dataclass
class Car:
    pos: Tuple[float, float]
    yaw: float
    v: float
    delta: float
    road_type: str
    color: Tuple[int,int,int] = (255,80,80)

    @classmethod
    def spawn(cls, pos, heading, road_type: str):
        yaw = math.atan2(heading[1], heading[0])
        return cls(pos=(pos[0], pos[1]), yaw=yaw, v=0.0, delta=0.0, road_type=road_type)

    def physics_input(self, dt: float, throttle: bool, brake: bool, steer_left: bool, steer_right: bool):
        v = self.v
        if throttle: v += ACCEL * dt
        if brake:    v -= BRAKE * dt
        if (not throttle) and (not brake):
            if abs(v) > 1e-4:
                drag = DRAG * dt * (1 if v > 0 else -1)
                if abs(drag) > abs(v): v = 0.0
                else: v -= drag
        self.v = clamp(v, MAX_SPEED_REV, MAX_SPEED_FWD)

        steer_rate = rad(STEER_RATE_DEG)
        if steer_left:  self.delta += steer_rate * dt
        if steer_right: self.delta -= steer_rate * dt

        if (not steer_left) and (not steer_right):
            ret = rad(STEER_RETURN_DEG) * dt
            if self.delta > 0: self.delta = max(0.0, self.delta - ret)
            elif self.delta < 0: self.delta = min(0.0, self.delta + ret)

        self.delta = clamp(self.delta, -rad(MAX_STEER_DEG), rad(MAX_STEER_DEG))

    def physics_step(self, dt: float):
        if abs(self.v) < 1e-6:
            return
        beta = math.tan(self.delta)
        self.pos = (self.pos[0] + math.cos(self.yaw) * (self.v * dt),
                    self.pos[1] + math.sin(self.yaw) * (self.v * dt))
        self.yaw -= (self.v / WHEELBASE) * beta * dt

    def body_points(self) -> tuple[list[tuple[float,float]], list[tuple[float,float]]]:
        d = (math.cos(self.yaw), math.sin(self.yaw))
        r = (-d[1], d[0])
        L = CAR_LEN
        W = CAR_WID
        c = self.pos
        body = [
            (c[0] + d[0]*(L/2) + r[0]*(W/2), c[1] + d[1]*(L/2) + r[1]*(W/2)),
            (c[0] + d[0]*(L/2) - r[0]*(W/2), c[1] + d[1]*(L/2) - r[1]*(W/2)),
            (c[0] - d[0]*(L/2) - r[0]*(W/2), c[1] - d[1]*(L/2) - r[1]*(W/2)),
            (c[0] - d[0]*(L/2) + r[0]*(W/2), c[1] - d[1]*(L/2) + r[1]*(W/2)),
        ]
        nose_c = (c[0] + d[0]*(L/2 - NOSE_DOT/2), c[1] + d[1]*(L/2 - NOSE_DOT/2))
        h = NOSE_DOT/2
        nose = [
            (nose_c[0] + r[0]*h + d[0]*h, nose_c[1] + r[1]*h + d[1]*h),
            (nose_c[0] - r[0]*h + d[0]*h, nose_c[1] - r[1]*h + d[1]*h),
            (nose_c[0] - r[0]*h - d[0]*h, nose_c[1] - r[1]*h - d[1]*h),
            (nose_c[0] + r[0]*h - d[0]*h, nose_c[1] + r[1]*h - d[1]*h),
        ]
        return body, nose
