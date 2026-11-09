"""Compare taxi dispatch/routing policies on held-out demand seeds.

    python -m metatraffic3d.marl.evaluate --checkpoint runs/mappo/policy.pt --seeds 20 --out runs/taxi_eval.json
"""

from __future__ import annotations

import argparse
import json
import os

from ..config import SimConfig
from ..world import build_world
from .mappo import evaluate, load_actor_weights


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--map-seed", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episode-seconds", type=float, default=1800.0)
    ap.add_argument("--policies", nargs="+", default=["random", "greedy_free", "greedy", "greedy_feasible"])
    ap.add_argument("--out", default="runs/taxi_eval.json")
    a = ap.parse_args(argv)
    cfg = SimConfig()
    cfg.map.seed = a.map_seed
    cfg.taxi.episode_seconds = a.episode_seconds
    world = build_world(cfg, with_meshes=False)
    policies = {p: None for p in a.policies}
    if a.checkpoint:
        policies["mappo"] = load_actor_weights(a.checkpoint)
    seeds = [777_000 + i for i in range(a.seeds)]
    res = evaluate(cfg, world, policies, seeds, workers=a.workers)
    keys = ("delivered_per_taxi_hour", "mean_wait_s", "ride_vs_free_flow", "abandoned", "expired_unclaimed",
            "double_parks", "earned")  # fmt: skip
    for name, m in res.items():
        print(f"{name:16s} " + "  ".join(f"{k}={m[k][0]:.2f}±{m[k][1]:.2f}" for k in keys))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
