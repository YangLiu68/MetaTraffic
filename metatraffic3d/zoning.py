"""Functional districts: CBD / financial, commercial, residential, industrial.

Every point gets a zone from smooth fields, so neighbouring lots agree and districts form:

* downtown: Gaussian around mapgen's radial district centres (or the map centre);
* main-road frontage: shops and offices line the arterials;
* waterfront and outskirts: ports, warehouses and depots;
* a seeded low-frequency noise field, so districts are irregular and differ per seed.

The profile's ``zone_mix`` shifts the thresholds, so some cities come out mostly suburban and
others mostly downtown. Nothing is guaranteed to appear.
"""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely import STRtree

ZONES = ("cbd", "commercial", "residential", "industrial")


class Zoning:
    def __init__(self, bounds, centres, main_lines, water, seed: int, mix: dict | None = None) -> None:
        rng = np.random.default_rng(seed + 401)
        xmin, ymin, xmax, ymax = bounds
        self.bounds = bounds
        self.diag = math.hypot(xmax - xmin, ymax - ymin)
        self.centres = np.asarray(centres if centres else [((xmin + xmax) / 2, (ymin + ymax) / 2)], float)
        self.main_tree = STRtree(main_lines) if main_lines else None
        self.main_lines = np.asarray(main_lines, dtype=object) if main_lines else None
        self.water = water if water is not None and not water.is_empty else None
        mix = mix or {"cbd": 0.12, "commercial": 0.2, "residential": 0.55, "industrial": 0.13}
        base = {"cbd": 0.12, "commercial": 0.2, "residential": 0.55, "industrial": 0.13}
        # a zone twice as common as usual gets a lower threshold
        self.shift = {z: 0.35 * math.log(max(mix.get(z, 1e-3), 1e-3) / base[z]) for z in ZONES}
        k = 14
        self.nx = rng.uniform(xmin, xmax, k)
        self.ny = rng.uniform(ymin, ymax, k)
        self.ns = rng.uniform(250, 700, k)
        self.na = rng.normal(0, 1, (3, k))

    def _noise(self, x, y, i):
        d2 = (x[:, None] - self.nx[None]) ** 2 + (y[:, None] - self.ny[None]) ** 2
        return np.tanh((np.exp(-d2 / (2 * self.ns[None] ** 2)) * self.na[i][None]).sum(1))

    def fields(self, pts: np.ndarray) -> dict[str, np.ndarray]:
        pts = np.asarray(pts, float).reshape(-1, 2)
        x, y = pts[:, 0], pts[:, 1]
        d_c = np.min(np.hypot(x[:, None] - self.centres[None, :, 0], y[:, None] - self.centres[None, :, 1]), axis=1)
        downtown = np.exp(-0.5 * (d_c / (0.16 * self.diag)) ** 2)
        geoms = shapely.points(x, y)
        if self.main_tree is not None:
            idx = self.main_tree.nearest(geoms)
            d_main = shapely.distance(geoms, self.main_lines[idx])
        else:
            d_main = np.full(len(x), 1e4)
        frontage = np.exp(-d_main / 70.0)
        d_water = shapely.distance(geoms, self.water) if self.water is not None else np.full(len(x), 1e4)
        waterfront = np.exp(-d_water / 250.0)
        xmin, ymin, xmax, ymax = self.bounds
        edge = np.minimum.reduce([x - xmin, xmax - x, y - ymin, ymax - y])
        outskirts = np.exp(-edge / 350.0)
        return {"downtown": downtown, "frontage": frontage, "waterfront": waterfront, "outskirts": outskirts,
                "intensity": np.clip(0.7 * downtown + 0.3 * frontage, 0, 1)}  # fmt: skip

    def classify(self, pts: np.ndarray) -> tuple[np.ndarray, dict]:
        f = self.fields(pts)
        x, y = np.asarray(pts, float).reshape(-1, 2).T
        n0, n1, n2 = self._noise(x, y, 0), self._noise(x, y, 1), self._noise(x, y, 2)
        cbd = 2.2 * f["downtown"] + 0.4 * f["frontage"] + 0.35 * n0 - 1.45 + self.shift["cbd"]
        com = 1.3 * f["frontage"] + 0.7 * f["downtown"] + 0.35 * n1 - 0.95 + self.shift["commercial"]
        ind = 1.1 * f["waterfront"] + 0.6 * f["outskirts"] - 0.8 * f["downtown"] + 0.45 * n2 - 0.85 + self.shift["industrial"]
        zone = np.full(len(x), "residential", dtype=object)
        zone[ind > 0] = "industrial"
        zone[com > 0] = "commercial"
        zone[cbd > 0] = "cbd"
        return zone, f
