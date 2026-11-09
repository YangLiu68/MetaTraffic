"""Record driving demonstrations: camera images + state + command + the expert's action.

    python -m metatraffic3d.driving.collect --out data/demos --episodes 40 --seeds 4 5 6 --noise 0.3

* ``--driver expert``: the expert drives; DART-style noise is added to the executed action so
  the data shows how to recover from small mistakes (the label is always the clean action).
* ``--driver policy --checkpoint ...``: a learned policy drives and the expert labels every
  state (DAgger): the data then covers the states the policy itself reaches. With ``--beta``
  the expert takes over for random stretches (~``--segment`` seconds) a ``beta`` share of the
  time, so episodes do not all end in the policy's first mistake.

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


def record_episode(env: DrivingEnv, seed: int, noise: float, policy=None, max_steps: int = 2400,
                   beta: float = 0.0, segment: float = 3.0) -> dict:  # fmt: skip
    rng = np.random.default_rng(seed)
    env.expert.noise = noise
    env.expert.rng = rng
    obs, info = env.reset(seed=seed)
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()
    frames, states, commands, controls, gears, meta, by_policy = [], [], [], [], [], [], []
    reason = None
    expert_turn = policy is None or rng.random() < beta
    for _ in range(max_steps):
        label = env.expert_action()
        pol = policy(obs) if policy is not None else None  # called every step to keep its frame history
        if policy is not None and rng.random() < env.dt / segment:  # hand over now and then
            expert_turn = rng.random() < beta
        if expert_turn:
            c = env.expert.noisy((*label["control"], label["gear"]))
            act = {"control": np.array(c[:3], np.float32), "gear": c[3]}
        else:
            act = pol
        by_policy.append(not expert_turn)
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
        "by_policy": np.array(by_policy, bool),
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
    ap.add_argument("--beta", type=float, default=0.0, help="DAgger: share of time the expert drives")
    ap.add_argument("--segment", type=float, default=3.0, help="DAgger: mean seconds between hand-overs")
    ap.add_argument("--noise-policy", type=float, default=0.15, help="DART noise on expert stretches in DAgger")
    ap.add_argument("--tag", default=None, help="file name tag (default: the driver)")
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
            path = os.path.join(a.out, f"map{map_seed}_{a.tag or a.driver}_ep{ep:04d}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            rec = record_episode(env, seed=map_seed * 1000 + ep + (5000 if policy is not None else 0),
                                 noise=a.noise if a.driver == "expert" else a.noise_policy, policy=policy,
                                 max_steps=a.max_steps, beta=a.beta, segment=a.segment)  # fmt: skip
            np.savez(path, **rec)
            n = len(rec["state"])
            print(f"map {map_seed} ep {ep}: {n} steps ({rec['by_policy'].mean():.0%} policy), {rec['reason']}, {rec['weather']}, "
                  f"{n / (time.time() - t0):.1f} steps/s, {os.path.getsize(path) / 1e6:.1f} MB", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
