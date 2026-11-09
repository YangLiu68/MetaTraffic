"""Closed-loop evaluation: each driver drives the same held-out routes; report driving metrics.

    python -m metatraffic3d.driving.evaluate --drivers expert runs/bc/policy.pt --seeds 6 --episodes 8

Metrics per driver (mean over episodes): arrival rate, route completion, distance, average
speed, collisions, red-light and stop-sign violations, off-road and wrong-way events, and
infractions per km. Same seeds and weather for every driver.
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

KINDS = ("collision", "red_light", "stop_sign", "off_road", "wrong_way", "speeding", "route_deviation")


def run_episode(env: DrivingEnv, driver, seed: int, max_steps: int = 2400) -> dict:
    obs, info = env.reset(seed=seed)
    if hasattr(driver, "reset"):
        driver.reset()
    start = env._prev_remaining
    dist = 0.0
    last = None
    reason = None
    for _ in range(max_steps):
        act = env.expert_action() if driver == "expert" else driver(obs)
        obs, r, term, trunc, info = env.step(act)
        st = env.car.state()
        if last is not None:
            dist += float(np.hypot(st.x - last[0], st.y - last[1]))
        last = (st.x, st.y)
        if term or trunc:
            reason = info["reason"]
            break
    inf = info["privileged"]["infractions"]
    done = max(0.0, min(1.0, 1.0 - env._prev_remaining / max(start, 1.0)))
    out = {"arrived": float(reason == "arrived"), "completion": done if reason != "arrived" else 1.0,
           "km": dist / 1000.0, "speed_kmh": dist / max(env.sim.time - env.t0, 1e-6) * 3.6, "reason": reason,
           "weather": env.weather}  # fmt: skip
    for k in KINDS:
        out[k] = sum(1 for e in inf if e["kind"] == k)
    return out


def summarize(rows: list[dict]) -> dict:
    km = sum(r["km"] for r in rows)
    s = {k: float(np.mean([r[k] for r in rows])) for k in ("arrived", "completion", "km", "speed_kmh")}
    for k in KINDS:
        s[k + "_per_km"] = sum(r[k] for r in rows) / max(km, 1e-6)
    s["episodes"] = len(rows)
    s["reasons"] = {x: sum(1 for r in rows if r["reason"] == x) for x in {r["reason"] for r in rows}}
    return s


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drivers", nargs="+", default=["expert"], help="'expert' or checkpoint paths")
    ap.add_argument("--seeds", type=int, nargs="+", default=[6])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--route", type=float, nargs=2, default=[300.0, 800.0], help="route length range (m)")
    ap.add_argument("--out", default="runs/eval.json")
    ap.add_argument("--two-pedals", action="store_true", help="use raw throttle and brake (no one-pedal rule)")
    a = ap.parse_args(argv)
    from .policy import load_policy

    drivers = {d: ("expert" if d == "expert" else load_policy(d, one_pedal=not a.two_pedals)) for d in a.drivers}
    results: dict[str, list] = {d: [] for d in drivers}
    for map_seed in a.seeds:
        cfg = SimConfig()
        cfg.map.seed = map_seed
        assets = AssetLibrary("auto")
        world = build_world(cfg, assets=assets)
        env = DrivingEnv(cfg, world, assets=assets, seed=map_seed, route_m=tuple(a.route),
                         max_seconds=a.max_steps * 0.1)  # fmt: skip
        for ep in range(a.episodes):
            seed = 900_000 + map_seed * 100 + ep
            for name, drv in drivers.items():
                t0 = time.time()
                row = run_episode(env, drv, seed, a.max_steps)
                row.update(map=map_seed, ep=ep)
                results[name].append(row)
                label = os.path.basename(os.path.dirname(name)) if name.endswith(".pt") else name
                print(f"{label} map {map_seed} ep {ep}: {row['reason']} completion {row['completion']:.2f} "
                      f"{row['km']:.2f} km collisions {row['collision']} red {row['red_light']} "
                      f"[{time.time() - t0:.0f}s]", flush=True)  # fmt: skip
    summary = {name: summarize(rows) for name, rows in results.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"summary": summary, "episodes": results}, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
