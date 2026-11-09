"""The ego car: a MuJoCo multibody car driven by direct controls.

Action (what an end-to-end policy outputs):
    steer    in [-1, 1]  road-wheel angle target (full lock = 0.6 rad), rate-limited
    throttle in [0, 1]   engine torque request
    brake    in [0, 1]   brake torque on all four wheels
    gear     in {R, N, 1, 2, 3, 4, 5}  (index 0..6); shifting takes 0.35 s with no drive torque

Physics: free chassis (1300 kg) on four wheels with tyre contacts (friction set by the
weather), steerable front knuckles, rear-wheel drive. Engine torque follows a torque curve
of engine speed, which follows the driven wheels through the gear and final drive ratios;
below idle the clutch slips. Gravity, slopes and friction come from MuJoCo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

GEARS = ("R", "N", "1", "2", "3", "4", "5")
RATIOS = {"R": -3.3, "N": 0.0, "1": 3.6, "2": 2.15, "3": 1.45, "4": 1.08, "5": 0.84}
FINAL_DRIVE = 3.9
WHEEL_R = 0.33
WHEEL_BASE = 2.75
TRACK = 1.58
IDLE_RPM, PEAK_RPM, REDLINE = 850.0, 4200.0, 6600.0
PEAK_TORQUE = 330.0  # N m
BRAKE_TORQUE = 2600.0  # N m per wheel at full brake
STEER_MAX = 0.6
STEER_RATE = 0.9  # rad/s
SHIFT_TIME = 0.35
FRICTION = {"clear": 1.0, "overcast": 1.0, "dusk": 1.0, "night": 0.95, "rain": 0.65, "fog": 0.85, "snow": 0.35}


def engine_torque(rpm: float) -> float:
    """Full-throttle torque curve (N m)."""
    if rpm < IDLE_RPM:
        return PEAK_TORQUE * 0.55
    if rpm < PEAK_RPM:
        return PEAK_TORQUE * (0.55 + 0.45 * (rpm - IDLE_RPM) / (PEAK_RPM - IDLE_RPM))
    if rpm < REDLINE:
        return PEAK_TORQUE * (1.0 - 0.35 * (rpm - PEAK_RPM) / (REDLINE - PEAK_RPM))
    return 0.0  # rev limiter


def ego_xml(prefix: str = "ego", parts_xml: str = "", cameras_xml: str = "", lights_xml: str = "") -> str:
    """MJCF body for the ego car (chassis collides with traffic; wheels with the ground)."""
    wheels = []
    for name, x, y, steer in (("fl", WHEEL_BASE / 2, TRACK / 2, True), ("fr", WHEEL_BASE / 2, -TRACK / 2, True),
                              ("rl", -WHEEL_BASE / 2, TRACK / 2, False), ("rr", -WHEEL_BASE / 2, -TRACK / 2, False)):  # fmt: skip
        wheel = (f'<body name="{prefix}_{name}" pos="{0 if steer else x} {0 if steer else y} {0 if steer else WHEEL_R}">'
                 f'<joint name="{prefix}_{name}_spin" type="hinge" axis="0 1 0" damping="0.05" armature="0.6"/>'
                 f'<geom name="{prefix}_{name}_tyre" type="cylinder" size="{WHEEL_R} 0.11" euler="1.5708 0 0" mass="20" '
                 f'rgba="0.08 0.08 0.09 1" friction="1.0 0.01 0.0005" condim="6" contype="1" conaffinity="1" group="2" '
                 f'solref="0.004 1" solimp="0.95 0.99 0.001"/></body>')  # fmt: skip
        if steer:
            wheels.append(f'<body name="{prefix}_{name}_knuckle" pos="{x} {y} {WHEEL_R}">'
                          f'<joint name="{prefix}_{name}_steer" type="hinge" axis="0 0 1" range="-0.65 0.65" damping="20" armature="0.5"/>'
                          f'<geom type="sphere" size="0.06" mass="8" contype="0" conaffinity="0" group="3"/>{wheel}</body>')  # fmt: skip
        else:
            wheels.append(wheel)
    return (f'<body name="{prefix}" pos="0 0 0.6"><freejoint name="{prefix}_root"/>'
            f'<geom name="{prefix}_chassis" type="box" size="2.2 0.88 0.32" pos="0 0 0.72" mass="1230" '
            f'rgba="0 0 0 0" contype="2" conaffinity="2" group="3"/>'
            f'<geom name="{prefix}_cabin" type="box" size="1.2 0.8 0.3" pos="-0.2 0 1.25" mass="1" '
            f'rgba="0 0 0 0" contype="2" conaffinity="2" group="3"/>'
            f'<site name="{prefix}_imu" pos="0 0 0.7"/>{parts_xml}{cameras_xml}{lights_xml}{"".join(wheels)}</body>')


def ego_actuators(prefix: str = "ego") -> str:
    acts = [f'<position name="{prefix}_{w}_steer" joint="{prefix}_{w}_steer" kp="9000" kv="300" ctrlrange="-0.65 0.65"/>'
            for w in ("fl", "fr")]  # fmt: skip
    for w in ("fl", "fr", "rl", "rr"):
        acts.append(f'<motor name="{prefix}_{w}_torque" joint="{prefix}_{w}_spin" ctrlrange="-12000 12000"/>')
    return "".join(acts)


@dataclass
class EgoState:
    speed: float  # m/s along the car (negative when reversing)
    rpm: float
    gear: str
    steer: float  # current road-wheel angle, rad
    shifting: bool
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    yaw_rate: float
    accel: float  # longitudinal, m/s^2


class EgoCar:
    """Turns direct controls into actuator torques every physics step."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "ego") -> None:
        self.m, self.d = model, data
        self.body = model.body(prefix).id
        self.root_qpos = model.jnt_qposadr[model.joint(f"{prefix}_root").id]
        self.root_dof = model.jnt_dofadr[model.joint(f"{prefix}_root").id]
        self.steer_act = [model.actuator(f"{prefix}_{w}_steer").id for w in ("fl", "fr")]
        self.torque_act = {w: model.actuator(f"{prefix}_{w}_torque").id for w in ("fl", "fr", "rl", "rr")}
        self.spin_dof = {w: model.jnt_dofadr[model.joint(f"{prefix}_{w}_spin").id] for w in ("fl", "fr", "rl", "rr")}
        self.steer_qpos = [model.jnt_qposadr[model.joint(f"{prefix}_{w}_steer").id] for w in ("fl", "fr")]
        self.tyres = [model.geom(f"{prefix}_{w}_tyre").id for w in ("fl", "fr", "rl", "rr")]
        self.chassis = [model.geom(f"{prefix}_chassis").id, model.geom(f"{prefix}_cabin").id]
        self.reset_controls()
        self._last_v = 0.0

    def reset_controls(self) -> None:
        self.gear = "1"
        self.target_gear = "1"
        self.shift_timer = 0.0
        self.steer_cmd = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.steer_angle = 0.0

    def place(self, x: float, y: float, z: float, yaw: float, speed: float = 0.0) -> None:
        q = self.root_qpos
        self.d.qpos[q:q + 3] = (x, y, z + 0.05)
        self.d.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        dof = self.root_dof
        self.d.qvel[dof:dof + 6] = 0.0
        self.d.qvel[dof] = speed * math.cos(yaw)
        self.d.qvel[dof + 1] = speed * math.sin(yaw)
        for _w, i in self.spin_dof.items():
            self.d.qvel[i] = speed / WHEEL_R
        for q in self.steer_qpos:
            self.d.qpos[q] = 0.0
        self.reset_controls()
        self._last_v = speed

    def set_friction(self, mu: float) -> None:
        for g in self.tyres:
            self.m.geom_friction[g, 0] = mu

    def act(self, steer: float, throttle: float, brake: float, gear: int | str) -> None:
        self.steer_cmd = float(np.clip(steer, -1, 1))
        self.throttle = float(np.clip(throttle, 0, 1))
        self.brake = float(np.clip(brake, 0, 1))
        g = GEARS[int(gear)] if not isinstance(gear, str) else gear
        if g != self.target_gear:
            self.target_gear = g
            self.shift_timer = SHIFT_TIME

    # ---------------------------------------------------------------- per physics step
    def apply(self, dt: float) -> None:
        d = self.d
        if self.shift_timer > 0:
            self.shift_timer -= dt
            if self.shift_timer <= 0:
                self.gear = self.target_gear
        # steering (rate limited), with a little Ackermann
        target = self.steer_cmd * STEER_MAX
        step = STEER_RATE * dt
        self.steer_angle += float(np.clip(target - self.steer_angle, -step, step))
        a = self.steer_angle
        if abs(a) > 1e-4:
            r = WHEEL_BASE / math.tan(abs(a))
            inner = math.atan(WHEEL_BASE / max(r - TRACK / 2, 0.5))
            outer = math.atan(WHEEL_BASE / (r + TRACK / 2))
            left, right = (inner, outer) if a > 0 else (-outer, -inner)
        else:
            left = right = 0.0
        d.ctrl[self.steer_act[0]] = left
        d.ctrl[self.steer_act[1]] = right
        # drivetrain (rear-wheel drive)
        omega = {w: d.qvel[i] for w, i in self.spin_dof.items()}
        w_drive = (omega["rl"] + omega["rr"]) / 2
        ratio = 0.0 if self.shift_timer > 0 else RATIOS[self.gear]
        rpm = abs(w_drive * ratio * FINAL_DRIVE) * 60 / (2 * math.pi)
        self.rpm = max(rpm, IDLE_RPM)
        drive = 0.0
        if ratio != 0.0:
            t_engine = self.throttle * engine_torque(rpm)
            t_engine -= (1 - self.throttle) * 25.0 * min(rpm / 3000.0, 1.0)  # engine braking
            drive = t_engine * ratio * FINAL_DRIVE * 0.9
        for w in ("fl", "fr", "rl", "rr"):
            brake = -self.brake * BRAKE_TORQUE * math.tanh(omega[w] / 0.5)
            torque = brake + (drive / 2 if w in ("rl", "rr") else 0.0)
            d.ctrl[self.torque_act[w]] = torque

    # ---------------------------------------------------------------- state
    def state(self, dt: float = 0.05) -> EgoState:
        d = self.d
        q = self.root_qpos
        x, y, z = d.qpos[q:q + 3]
        qw, qx, qy, qz = d.qpos[q + 3:q + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        dof = self.root_dof
        vx, vy = d.qvel[dof], d.qvel[dof + 1]
        speed = vx * math.cos(yaw) + vy * math.sin(yaw)
        acc = (speed - self._last_v) / max(dt, 1e-3)
        self._last_v = speed
        return EgoState(float(speed), float(getattr(self, "rpm", IDLE_RPM)), self.gear, float(self.steer_angle),
                        self.shift_timer > 0, float(x), float(y), float(z), float(yaw), float(pitch), float(roll),
                        float(d.qvel[dof + 5]), float(acc))  # fmt: skip

    def collisions(self) -> list[int]:
        """Geom ids the chassis is touching (traffic, pedestrians, props)."""
        out = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            if c.geom1 in self.chassis:
                out.append(int(c.geom2))
            elif c.geom2 in self.chassis:
                out.append(int(c.geom1))
        return out


def auto_gear(gear: str, speed: float, throttle: float) -> str:
    """Simple automatic shift schedule used by the expert driver (upshift ~2500-3000 rpm)."""
    if speed < -0.1 or gear == "R":
        return gear
    up = {"1": 5.5, "2": 10.0, "3": 15.5, "4": 22.0}
    down = {"2": 3.5, "3": 7.5, "4": 12.0, "5": 17.5}
    if gear in up and speed > up[gear] * (0.8 + 0.4 * throttle):
        return str(int(gear) + 1)
    if gear in down and speed < down[gear]:
        return str(int(gear) - 1)
    if gear == "N":
        return "1"
    return gear
