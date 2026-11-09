import pytest

from metatraffic3d.config import SimConfig


def small_cfg(seed: int = 3, **profile) -> SimConfig:
    cfg = SimConfig()
    cfg.map.seed = seed
    cfg.map.width, cfg.map.height = 300.0, 200.0
    cfg.traffic.num_npc = 40
    cfg.traffic.num_pedestrians = 40
    for k, v in profile.items():
        setattr(cfg.profile, k, v)
    return cfg


@pytest.fixture(scope="session")
def small_world():
    from metatraffic3d.assets import AssetLibrary
    from metatraffic3d.world import build_world

    return build_world(small_cfg(coast=True, river=True, terrain="flat"), assets=AssetLibrary("procedural"))
