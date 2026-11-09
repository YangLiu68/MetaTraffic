"""Train the taxi fleet with MAPPO.

    python -m metatraffic3d.marl.train --out runs/mappo --iterations 60
"""

from __future__ import annotations

import argparse
import json
import os
import time

from ..config import SimConfig
from ..world import build_world
from .mappo import PPOConfig, Trainer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/mappo")
    ap.add_argument("--map-seed", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--episodes-per-iter", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episode-seconds", type=float, default=1800.0)
    ap.add_argument("--taxis", type=int, default=5)
    ap.add_argument("--npc", type=int, default=120)
    ap.add_argument("--bc-episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    cfg = SimConfig()
    cfg.map.seed = a.map_seed
    cfg.taxi.episode_seconds = a.episode_seconds
    cfg.taxi.num_taxis = a.taxis
    cfg.traffic.num_npc = a.npc
    world = build_world(cfg, with_meshes=False)
    ppo = PPOConfig(iterations=a.iterations, episodes_per_iter=a.episodes_per_iter, workers=a.workers,
                    bc_episodes=a.bc_episodes, seed=a.seed)  # fmt: skip
    log_path = os.path.join(a.out, "train.log")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    log(f"map seed {a.map_seed}: {world.net.summary()['nodes']} junctions; {a.taxis} taxis, {a.npc} NPCs")
    trainer = Trainer(cfg, world, ppo, log=log)
    try:
        hist = trainer.train(checkpoint=os.path.join(a.out, "policy.pt"))
    finally:
        trainer.close()
    with open(os.path.join(a.out, "history.json"), "w") as fh:
        json.dump(hist, fh, indent=1)


if __name__ == "__main__":
    main()
