"""City profile: the global character of a city, drawn from its seed.

Nothing is forced. Each factor is sampled from the map seed with the probabilities below,
and any factor can be pinned by setting it in ``ProfileConfig`` (None = sample it).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TERRAIN_STYLES = {  # amplitude (m), hills per km2
    "flat": (0.0, 0.0),
    "rolling": (14.0, 1.2),
    "hilly": (38.0, 1.6),
    "steep": (70.0, 1.9),
}


@dataclass
class ProfileConfig:
    terrain: str | None = None  # flat | rolling | hilly | steep
    coast: bool | None = None
    river: bool | None = None
    zone_mix: dict | None = None  # weights for cbd / commercial / residential / industrial
    traffic_level: float | None = None  # multiplies the number of background cars
    pedestrian_level: float | None = None
    highway: bool | None = None  # experimental elevated expressway
    seed_offset: int = 7


@dataclass
class CityProfile:
    terrain: str
    coast: bool
    river: bool
    zone_mix: dict
    traffic_level: float
    pedestrian_level: float
    highway: bool

    def as_dict(self) -> dict:
        return asdict(self)


def sample_profile(seed: int, cfg: ProfileConfig | None = None) -> CityProfile:
    cfg = cfg or ProfileConfig()
    rng = np.random.default_rng(seed + cfg.seed_offset)
    styles = list(TERRAIN_STYLES)
    terrain = cfg.terrain or str(rng.choice(styles, p=[0.35, 0.3, 0.25, 0.1]))
    coast = cfg.coast if cfg.coast is not None else bool(rng.random() < 0.6)
    river = cfg.river if cfg.river is not None else bool(rng.random() < 0.65)
    if cfg.zone_mix is not None:
        mix = dict(cfg.zone_mix)
    else:  # Dirichlet around a typical mid-size city
        w = rng.dirichlet([2.0, 3.0, 7.0, 1.5])
        mix = dict(zip(("cbd", "commercial", "residential", "industrial"), map(float, w)))
    traffic = cfg.traffic_level if cfg.traffic_level is not None else float(rng.uniform(0.6, 1.4))
    peds = cfg.pedestrian_level if cfg.pedestrian_level is not None else float(rng.uniform(0.5, 1.5))
    highway = cfg.highway if cfg.highway is not None else bool(rng.random() < 0.3)
    return CityProfile(terrain, coast, river, mix, traffic, peds, highway)
