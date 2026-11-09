"""Record driving demonstrations: camera images + state + command + the expert's action.

    python -m metatraffic3d.driving.collect --out data/demos --episodes 40 --seeds 4 5 6 --noise 0.3

* ``--driver expert``: the expert drives; DART-style noise is added to the executed action so
  the data shows how to recover from small mistakes (the label is always the clean action).
* ``--driver policy --checkpoint ...``: a learned policy drives and the expert labels every
  state (DAgger): the data then covers the states the policy itself reaches.

Each episode is one ``.npz``: JPEG bytes of every camera frame (flattened + offsets), state,
command, expert controls and gear, and per-step metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from ..assets import AssetLibrary
from ..config import SimConfig
from ..world import build_world
from .env import DrivingEnv
from .jpeg import decode, encode  # noqa: F401  (re-exported)


def record_episode(env: DrivingEnv, seed: int, noise: float, policy=None, max_steps: int = 2400) -> dict:
    rng = np.random.default_rng(seed)
    env.expert.noise = noise
    env.expert.rng = rng
    obs, info = env.reset(seed=seed)
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()
    frames, states, commands, controls, gears, meta = [], [], [], [], [], []
    reason = None
    for _ in range(max_steps):
        label = env.expert_action()
        if policy is not None:
            act = policy(obs)
        else:
            c = env.expert.noisy((*label["control"], label["gear"]))
            act = {"control": np.array(c[:3], np.float32), "gear": c[3]}
        frames.append([encode(im) for im in obs["images"]])
        states.append(obs["state"])
        commands.append(obs["command"])
        controls.append(label["control"])
        gears.append(label["gear"])
        obs, r, term, trunc, info = env.step(act)
        meta.append((r, env.car.state().speed))
        if term or trunc:
            reason = info["reason"]
            break
    flat = [b for fr in frames for b in fr]
    offsets = np.cumsum([0] + [len(b) for b in flat])
    return {
        "jpeg": np.frombuffer(b"".join(flat), np.uint8), "offsets": offsets, "n_cams": len(env.cameras),
        "state": np.array(states, np.float32), "command": np.array(commands, np.float32),
        "control": np.array(controls, np.float32), "gear": np.array(gears, np.int64),
        "reward": np.array([m[0] for m in meta], np.float32), "reason": str(reason),
        "weather": env.weather, "infractions": json.dumps(info["privileged"]["infractions"]),
    }  # fmt: skip


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/demos")
    ap.add_argument("--seeds", type=int, nargs="+", default=[4])
    ap.add_argument("--episodes", type=int, default=10, help="per map seed")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--driver", choices=["expert", "policy"], default="expert")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--start", type=int, default=0, help="first episode index (to resume)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    policy = None
    if a.driver == "policy":
        from .policy import load_policy

        policy = load_policy(a.checkpoint)
    for map_seed in a.seeds:
        cfg = SimConfig()
        cfg.map.seed = map_seed
        assets = AssetLibrary("auto")
        world = build_world(cfg, assets=assets)
        env = DrivingEnv(cfg, world, assets=assets, seed=map_seed)
        for ep in range(a.start, a.start + a.episodes):
            path = os.path.join(a.out, f"map{map_seed}_{a.driver}_ep{ep:04d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            rec = record_episode(env, seed=map_seed * 1000 + ep, noise=a.noise if a.driver == "expert" else 0.0,
                                 policy=policy, max_steps=a.max_steps)  # fmt: skip
            np.savez(path, **rec)
            n = len(rec["state"])
            print(f"map {map_seed} ep {ep}: {n} steps, {rec['reason']}, {rec['weather']}, "
                  f"{n / (time.time() - t0):.1f} steps/s, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
