"""Building types and heights by district, all seeded.

* residential: detached houses (1-3 floors, pitched roofs) on small lots, apartment blocks
  (4-10 floors) on larger ones;
* commercial: shops with a glass ground floor, 1-6 floors;
* cbd: offices and towers, height rising toward the downtown centre (log-normal);
* industrial: warehouses, 1-2 tall storeys on big footprints.

Small lots are capped (height <= 4 x sqrt(area)) so nothing is pencil-thin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from .config import BuildingConfig


@dataclass
class Building:
    footprint: shapely.Polygon
    floors: int
    height: float
    intensity: float
    zone: str = "residential"
    style: str = "house"  # house | apartment | shop | office | tower | warehouse

    @property
    def floor_area(self) -> float:
        return self.footprint.area * self.floors


def draw_buildings(lots: list[shapely.Polygon], zones: np.ndarray, intensity: np.ndarray, seed: int,
                   cfg: BuildingConfig | None = None) -> list[Building]:  # fmt: skip
    cfg = cfg or BuildingConfig()
    rng = np.random.default_rng(seed + cfg.seed_offset)
    out = []
    for lot, zone, inten in zip(lots, zones, intensity):
        area = lot.area
        cap = max(1, int((4.0 * np.sqrt(area) - cfg.ground_floor) / cfg.floor_height + 1))
        fh = cfg.floor_height
        if zone == "cbd":
            median = 6.0 + 30.0 * inten**1.5
            floors = int(round(np.exp(rng.normal(np.log(median), 0.45))))
            style = "tower" if floors >= 12 else "office"
        elif zone == "commercial":
            floors = int(round(np.exp(rng.normal(np.log(2.0 + 4.0 * inten), 0.35))))
            floors = min(floors, 8)
            style = "shop" if floors <= 4 else "office"
        elif zone == "industrial":
            floors = int(rng.choice([1, 1, 2]))
            fh = 5.5
            style = "warehouse"
        else:
            if area < 520 and rng.random() < 0.85:
                floors = int(rng.choice([1, 2, 2, 2, 3]))
                style = "house"
            else:
                floors = int(round(np.exp(rng.normal(np.log(5.0), 0.3))))
                floors = min(floors, 10)
                style = "apartment"
        floors = int(np.clip(min(floors, cap), 1, cfg.max_floors))
        height = cfg.ground_floor + (floors - 1) * fh if style != "warehouse" else floors * fh + 1.0
        out.append(Building(lot, floors, float(height), float(inten), str(zone), style))
    return out
