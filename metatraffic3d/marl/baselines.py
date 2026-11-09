"""Hand-written policies to compare against: random, greedy (congestion-aware), greedy free-flow.

All read only the observation, like a learned policy would.
"""

from __future__ import annotations

import numpy as np

from .taxi_env import DISPATCH, F_REQ, F_ROUTE, NONE, ROUTE


def random_policy(obs: dict, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(obs["mask"])
    return int(rng.choice(valid))


def greedy_policy(obs: dict, rng=None, congested: bool = True) -> int:
    """Nearest passenger by ETA; fastest next street by ETA (live or free-flow travel times)."""
    kind, feats, mask = obs["kind"], obs["options"], obs["mask"].astype(bool)
    if kind == NONE:
        return 0
    if kind == DISPATCH:
        col = F_REQ + (0 if congested else 1)
        reqs = [i for i in np.flatnonzero(mask) if feats[i, 1] > 0.5]
        if not reqs:
            return 0
        return int(min(reqs, key=lambda i: feats[i, col]))
    if kind == ROUTE:
        col = F_ROUTE + (0 if congested else 1)
        valid = np.flatnonzero(mask)
        return int(min(valid, key=lambda i: feats[i, col]))
    return 0


def greedy_free_policy(obs: dict, rng=None) -> int:
    return greedy_policy(obs, rng, congested=False)


def greedy_feasible_policy(obs: dict, rng=None) -> int:
    """Like ``greedy`` but skips passengers who will have given up before the taxi arrives
    and passengers another idle taxi is closer to."""
    kind, feats, mask = obs["kind"], obs["options"], obs["mask"].astype(bool)
    if kind != DISPATCH:
        return greedy_policy(obs, rng)
    reqs = [i for i in np.flatnonzero(mask) if feats[i, 1] > 0.5]
    good = [i for i in reqs if feats[i, F_REQ + 6] > 0.0 and feats[i, F_REQ + 7] == 0.0]
    if not good:
        good = [i for i in reqs if feats[i, F_REQ + 6] > 0.0]
    if not good:
        return 0
    return int(min(good, key=lambda i: feats[i, F_REQ]))


POLICIES = {"random": random_policy, "greedy": greedy_policy, "greedy_free": greedy_free_policy,
            "greedy_feasible": greedy_feasible_policy}  # fmt: skip
