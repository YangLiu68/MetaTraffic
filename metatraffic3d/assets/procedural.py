"""Low-poly fallback assets built from boxes, cylinders and spheres (no downloads needed).

Local frame: x forward, y left, z up, origin at the centre of the footprint on the ground.
"""

from __future__ import annotations

import numpy as np

from ..meshes import Part, box, cylinder, icosphere, merge

GLASS = (0.18, 0.22, 0.28, 1.0)
TYRE = (0.08, 0.08, 0.09, 1.0)
LIGHT = (0.95, 0.92, 0.75, 1.0)


def _wheels(length: float, width: float, r: float = 0.33):
    parts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(cylinder(r, 0.24, n=10, center=(sx * length * 0.32, sy * (width / 2 - 0.12), r), axis="y"))
    return merge(parts)


def car(length: float = 4.4, width: float = 1.8, paint=(0.7, 0.1, 0.1, 1.0)) -> list[Part]:
    body = box(length, width, 0.7, center=(0, 0, 0.68))
    cabin = box(length * 0.52, width * 0.88, 0.55, center=(-length * 0.05, 0, 1.3))
    lights = merge([box(0.05, 0.3, 0.12, center=(length / 2, sy * width * 0.33, 0.8)) for sy in (-1, 1)])
    return [
        Part(*body, paint, tintable=True),
        Part(*cabin, GLASS),
        Part(*_wheels(length, width), TYRE),
        Part(*lights, LIGHT),
    ]


def taxi() -> list[Part]:
    parts = car(4.6, 1.85, (0.98, 0.78, 0.10, 1.0))
    parts[0].tintable = False
    sign = box(0.6, 0.3, 0.25, center=(-0.25, 0, 1.7))
    parts.append(Part(*sign, (0.95, 0.95, 0.95, 1.0)))
    return parts


def van(paint=(0.9, 0.9, 0.92, 1.0)) -> list[Part]:
    body = box(5.0, 2.0, 1.7, center=(0, 0, 1.2))
    wind = box(0.1, 1.7, 0.6, center=(2.5, 0, 1.55))
    return [Part(*body, paint, tintable=True), Part(*wind, GLASS), Part(*_wheels(5.0, 2.0), TYRE)]


def tree_round(height: float = 7.0) -> list[Part]:
    trunk = cylinder(0.22, height * 0.45, n=6, center=(0, 0, height * 0.225))
    crown = icosphere(height * 0.3, center=(0, 0, height * 0.62), squash=1.1)
    return [Part(*trunk, (0.38, 0.27, 0.17, 1.0)), Part(*crown, (0.27, 0.52, 0.24, 1.0))]


def tree_cone(height: float = 8.0) -> list[Part]:
    trunk = cylinder(0.2, height * 0.3, n=6, center=(0, 0, height * 0.15))
    crown = cylinder(height * 0.22, height * 0.75, n=8, center=(0, 0, height * 0.6), r_top=0.05)
    return [Part(*trunk, (0.35, 0.25, 0.16, 1.0)), Part(*crown, (0.18, 0.42, 0.25, 1.0))]


def streetlight(height: float = 7.0) -> list[Part]:
    pole = cylinder(0.08, height, n=6, center=(0, 0, height / 2))
    arm = box(1.6, 0.12, 0.12, center=(0.8, 0, height - 0.1))
    lamp = box(0.5, 0.25, 0.12, center=(1.5, 0, height - 0.22))
    return [Part(*merge([pole, arm]), (0.3, 0.32, 0.34, 1.0)), Part(*lamp, LIGHT)]


def person(shirt=(0.15, 0.45, 0.85, 1.0)) -> list[Part]:
    legs = cylinder(0.16, 0.85, n=6, center=(0, 0, 0.43))
    torso = cylinder(0.22, 0.65, n=6, center=(0, 0, 1.18))
    head = icosphere(0.13, center=(0, 0, 1.65))
    return [Part(*legs, (0.15, 0.15, 0.2, 1.0)), Part(*torso, shirt), Part(*head, (0.85, 0.68, 0.55, 1.0))]


def stop_sign() -> list[Part]:
    pole = cylinder(0.04, 2.3, n=6, center=(0, 0, 1.15))
    plate = cylinder(0.38, 0.03, n=8, center=(0.03, 0, 2.15), axis="x")
    rim = cylinder(0.42, 0.02, n=8, center=(0.015, 0, 2.15), axis="x")
    return [Part(*pole, (0.55, 0.56, 0.58, 1.0)), Part(*plate, (0.80, 0.06, 0.06, 1.0)),
            Part(*rim, (0.95, 0.95, 0.95, 1.0))]  # fmt: skip


def traffic_cone() -> list[Part]:
    cone = cylinder(0.18, 0.7, n=8, center=(0, 0, 0.38), r_top=0.03)
    base = box(0.42, 0.42, 0.04, center=(0, 0, 0.02))
    band = cylinder(0.12, 0.1, n=8, center=(0, 0, 0.45))
    return [Part(*merge([cone, base]), (0.95, 0.42, 0.06, 1.0)), Part(*band, (0.95, 0.95, 0.95, 1.0))]


def barrier() -> list[Part]:
    legs = merge([box(0.08, 0.5, 0.9, center=(0, sy * 0.8, 0.45)) for sy in (-1, 1)])
    board = box(0.06, 2.0, 0.3, center=(0, 0, 0.8))
    return [Part(*legs, (0.9, 0.9, 0.9, 1.0)), Part(*board, (0.95, 0.35, 0.05, 1.0))]


def truck(paint=(0.85, 0.85, 0.88, 1.0), length: float = 7.5) -> list[Part]:
    cab = box(2.0, 2.3, 2.4, center=(length / 2 - 1.0, 0, 1.6))
    cargo = box(length - 2.2, 2.4, 2.9, center=(-1.1, 0, 1.95))
    wind = box(0.08, 2.0, 0.8, center=(length / 2, 0, 2.2))
    wheels = merge([cylinder(0.48, 0.3, n=10, center=(sx, sy * 1.05, 0.48), axis="y")
                    for sx in (length / 2 - 1.2, -length / 2 + 1.3) for sy in (-1, 1)])  # fmt: skip
    return [Part(*cab, paint, tintable=True), Part(*cargo, (0.92, 0.92, 0.92, 1.0)), Part(*wind, GLASS), Part(*wheels, TYRE)]


def bus() -> list[Part]:
    body = box(12.0, 2.55, 2.9, center=(0, 0, 1.75))
    windows = box(11.2, 2.6, 0.9, center=(0.2, 0, 2.3))
    wheels = merge([cylinder(0.5, 0.3, n=10, center=(sx, sy * 1.1, 0.5), axis="y") for sx in (3.8, -3.6) for sy in (-1, 1)])
    return [Part(*body, (0.15, 0.45, 0.75, 1.0)), Part(*windows, GLASS), Part(*wheels, TYRE)]


def emergency(kind: str = "ambulance") -> list[Part]:
    paint = (0.95, 0.95, 0.95, 1.0) if kind == "ambulance" else (0.75, 0.08, 0.06, 1.0)
    length = 6.0 if kind == "ambulance" else 9.0
    parts = truck(paint, length)
    parts[0].tintable = False
    parts[1] = Part(parts[1].vertices, parts[1].faces, paint)
    bar = box(0.4, 1.4, 0.18, center=(length / 2 - 1.0, 0, 2.9))
    stripe = box(length - 0.4, 2.42, 0.25, center=(-0.2, 0, 1.4))
    stripe_c = (0.85, 0.10, 0.10, 1.0) if kind == "ambulance" else (0.95, 0.95, 0.95, 1.0)
    return parts + [Part(*bar, (0.2, 0.3, 0.95, 1.0)), Part(*stripe, stripe_c)]


def _scaled(parts: list[Part], sz: float) -> list[Part]:
    for p in parts:
        p.vertices = p.vertices * np.array([1.0, 1.0, sz])
    return parts


def road_work_sign() -> list[Part]:
    legs = merge([box(0.05, 0.05, 1.2, center=(0, sy * 0.35, 0.6)) for sy in (-1, 1)])
    plate = cylinder(0.45, 0.03, n=4, center=(0.03, 0, 1.35), axis="x")
    return [Part(*legs, (0.6, 0.6, 0.62, 1.0)), Part(*plate, (0.98, 0.55, 0.05, 1.0))]


def excavator() -> list[Part]:
    tracks = merge([box(3.6, 0.6, 0.8, center=(0, sy * 1.1, 0.4)) for sy in (-1, 1)])
    body = box(2.6, 2.4, 1.4, center=(-0.2, 0, 1.5))
    cab = box(1.0, 1.0, 1.2, center=(0.6, 0.6, 2.8))
    arm = merge([box(2.8, 0.35, 0.35, center=(2.2, -0.4, 2.6)), box(0.35, 0.35, 2.2, center=(3.5, -0.4, 1.6))])
    return [Part(*tracks, (0.12, 0.12, 0.13, 1.0)), Part(*body, (0.95, 0.72, 0.08, 1.0)),
            Part(*cab, GLASS), Part(*arm, (0.95, 0.72, 0.08, 1.0))]  # fmt: skip


PROCEDURAL = {
    "sedan": lambda: car(4.5, 1.8),
    "suv": lambda: _scaled(car(4.8, 1.95), 1.22),
    "hatchback": lambda: car(4.0, 1.75),
    "sports": lambda: _scaled(car(4.4, 1.85), 0.82),
    "police": lambda: car(4.8, 1.9, (0.08, 0.08, 0.1, 1.0)),
    "pickup": lambda: _scaled(truck((0.55, 0.12, 0.10, 1.0), 5.3), 0.72),
    "delivery": lambda: truck((0.95, 0.95, 0.95, 1.0), 6.0),
    "garbage_truck": lambda: truck((0.15, 0.45, 0.25, 1.0), 8.5),
    "road_work_sign": road_work_sign,
    "excavator": excavator,
    "stop_sign": stop_sign,
    "traffic_cone": traffic_cone,
    "barrier": barrier,
    "truck": truck,
    "bus": bus,
    "ambulance": lambda: emergency("ambulance"),
    "fire_truck": lambda: emergency("fire_truck"),
    "car": car,
    "taxi": taxi,
    "van": van,
    "tree_round": tree_round,
    "tree_cone": tree_cone,
    "streetlight": streetlight,
    "person": person,
}
