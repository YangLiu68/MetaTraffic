"""Vehicle types for background traffic: size, performance and how common they are."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleType:
    name: str
    length: float
    width: float
    share: float  # fraction of background traffic
    accel: float = 1.0  # multiplies max acceleration
    speed: float = 1.0  # multiplies the speed limit the driver aims for
    parks: bool = True  # may end its trip in a kerb spot
    emergency: bool = False


TYPES = {t.name: t for t in (
    VehicleType("sedan", 4.5, 1.8, 0.30),
    VehicleType("suv", 4.8, 1.95, 0.17),
    VehicleType("hatchback", 4.0, 1.75, 0.13),
    VehicleType("pickup", 5.3, 2.0, 0.07),
    VehicleType("van", 5.0, 2.0, 0.07),
    VehicleType("taxi", 4.6, 1.85, 0.04),
    VehicleType("sports", 4.4, 1.85, 0.02, accel=1.5, speed=1.1),
    VehicleType("police", 4.8, 1.9, 0.02),
    VehicleType("delivery", 6.0, 2.1, 0.06, accel=0.8, parks=False),
    VehicleType("truck", 7.5, 2.4, 0.05, accel=0.6, speed=0.9, parks=False),
    VehicleType("bus", 12.0, 2.55, 0.04, accel=0.6, speed=0.9, parks=False),
    VehicleType("garbage_truck", 8.5, 2.5, 0.02, accel=0.5, speed=0.8, parks=False),
    VehicleType("ambulance", 6.0, 2.2, 0.0, accel=1.1, speed=1.3, parks=False, emergency=True),
    VehicleType("fire_truck", 9.0, 2.5, 0.0, accel=0.8, speed=1.2, parks=False, emergency=True),
)}  # fmt: skip
PARKED_TYPES = ("sedan", "suv", "hatchback", "van", "pickup")
_NAMES = [n for n, t in TYPES.items() if t.share > 0]
_P = np.array([TYPES[n].share for n in _NAMES])
_P = _P / _P.sum()


def sample_type(rng: np.random.Generator) -> VehicleType:
    return TYPES[_NAMES[int(rng.choice(len(_NAMES), p=_P))]]


def sample_parked_type(rng: np.random.Generator) -> str:
    return PARKED_TYPES[int(rng.choice(len(PARKED_TYPES), p=[0.42, 0.25, 0.18, 0.08, 0.07]))]
