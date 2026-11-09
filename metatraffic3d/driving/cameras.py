"""Multi-camera rig on the ego car, laid out like Tesla's eight cameras (approximate).

Pinhole cameras (MuJoCo has no fisheye), placed in the car frame (x forward, y left, z up).
``hfov`` is the horizontal field of view; the vertical one follows the render aspect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraSpec:
    name: str
    pos: tuple[float, float, float]
    yaw: float  # degrees, + = left
    pitch: float  # degrees, + = up
    hfov: float  # degrees


TESLA_RIG = (
    CameraSpec("front_main", (0.55, 0.0, 1.42), 0.0, -2.0, 50.0),
    CameraSpec("front_wide", (0.55, 0.12, 1.42), 0.0, -4.0, 120.0),
    CameraSpec("front_narrow", (0.55, -0.12, 1.42), 0.0, -1.0, 35.0),
    CameraSpec("pillar_left", (0.05, 0.86, 1.28), 60.0, -3.0, 90.0),
    CameraSpec("pillar_right", (0.05, -0.86, 1.28), -60.0, -3.0, 90.0),
    CameraSpec("repeater_left", (1.35, 0.96, 0.98), 140.0, -6.0, 75.0),
    CameraSpec("repeater_right", (1.35, -0.96, 0.98), -140.0, -6.0, 75.0),
    CameraSpec("rear", (-2.28, 0.0, 1.08), 180.0, -10.0, 120.0),
)
RIG = {c.name: c for c in TESLA_RIG}


def xyaxes(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """MuJoCo camera frame (x = right, y = up; it looks along -z) for a viewing direction."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.cross(right, fwd)
    return np.concatenate([right, up])


def fovy(hfov_deg: float, aspect: float) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) / aspect))


def cameras_xml(prefix: str = "ego", rig=TESLA_RIG) -> str:
    out = []
    for c in rig:
        ax = " ".join(f"{a:.5f}" for a in xyaxes(c.yaw, c.pitch))
        out.append(f'<camera name="{prefix}_{c.name}" pos="{c.pos[0]} {c.pos[1]} {c.pos[2]}" xyaxes="{ax}" '
                   f'fovy="{fovy(c.hfov, 16 / 9):.2f}"/>')  # fmt: skip
    return "".join(out)
