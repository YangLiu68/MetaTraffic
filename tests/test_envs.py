import numpy as np
import pytest

from metatraffic3d.marl.baselines import POLICIES
from metatraffic3d.marl.taxi_env import TaxiEnv

from .conftest import small_cfg


def test_taxi_env_runs_and_is_deterministic(small_world):
    cfg = small_cfg()
    cfg.taxi.episode_seconds = 300
    cfg.taxi.num_taxis = 3
    out = []
    for _ in range(2):
        env = TaxiEnv(cfg, small_world)
        obs, _ = env.reset(seed=5)
        assert set(obs) == set(env.possible_agents)
        while env.agents:
            acts = {a: POLICIES["greedy_feasible"](o) for a, o in obs.items()}
            for a, o in obs.items():
                assert o["mask"][acts[a]] == 1
            obs, rew, term, trunc, info = env.step(acts)
        out.append(info[env.possible_agents[0]]["episode"])
    assert out[0] == out[1]


@pytest.mark.slow
def test_driving_env_expert_drives(small_world, tmp_path):
    pytest.importorskip("mujoco")
    from metatraffic3d.assets import AssetLibrary
    from metatraffic3d.driving.env import DrivingEnv

    env = DrivingEnv(small_cfg(), small_world, image_size=(96, 64), weather="clear", route_m=(150, 500),
                     out_dir=str(tmp_path), assets=AssetLibrary("procedural"), seed=2)
    obs, info = env.reset(seed=3, options={"warmup": 2})
    assert obs["images"].shape == (4, 64, 96, 3) and obs["images"].dtype == np.uint8
    start = env._prev_remaining
    for _ in range(300):
        obs, r, term, trunc, info = env.step(env.expert_action())
        if term or trunc:
            break
    kinds = [e["kind"] for e in info["privileged"]["infractions"]]
    assert "collision" not in kinds
    assert env._prev_remaining < start  # it made progress along the route
