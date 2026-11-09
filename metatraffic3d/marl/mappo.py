"""MAPPO for the taxi fleet: shared option-scoring actor, centralised critic, semi-MDP GAE.

* Actor: one MLP scores every option row (option features + the taxi's own features); a
  softmax over the valid rows is the policy. The same network handles dispatch and routing.
* Critic: MLP on the global state plus the agent's own features (MAPPO, agent-specific state).
* Decisions are irregular in time, so returns discount by ``gamma ** seconds`` between them.
* Rollout workers run the simulator with a numpy copy of the actor (no torch in workers).
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import asdict, dataclass

import numpy as np

from .baselines import greedy_policy
from .taxi_env import NONE, TaxiEnv


@dataclass
class PPOConfig:
    iterations: int = 60
    episodes_per_iter: int = 4
    workers: int = 4
    gamma_per_s: float = 0.998  # half-life ~5.8 simulated minutes
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 512
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    bc_episodes: int = 8  # behaviour-cloning warm start from the greedy policy (0 = off)
    bc_steps: int = 1500
    seed: int = 0


# ---------------------------------------------------------------------- numpy actor
def np_scores(weights: list[np.ndarray], opts: np.ndarray, self_f: np.ndarray) -> np.ndarray:
    k = opts.shape[0]
    x = np.concatenate([opts, np.repeat(self_f[None, :], k, axis=0)], axis=1)
    n = len(weights) // 2
    for i in range(n):
        x = x @ weights[2 * i].T + weights[2 * i + 1]
        if i < n - 1:
            x = np.tanh(x)
    return x[:, 0]


def np_policy(weights, obs, rng, greedy: bool = False) -> tuple[int, float]:
    logits = np_scores(weights, obs["options"], obs["self"])
    mask = obs["mask"].astype(bool)
    logits = np.where(mask, logits, -1e9)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    a = int(np.argmax(p)) if greedy else int(rng.choice(len(p), p=p))
    return a, float(np.log(p[a] + 1e-12))


# ---------------------------------------------------------------------- rollouts
_ENV: TaxiEnv | None = None


def _init_worker(cfg, world) -> None:
    global _ENV
    _ENV = TaxiEnv(cfg, world)


def run_episode(env: TaxiEnv, seed: int, weights=None, policy: str = "learned", greedy_actions: bool = False):
    """Play one episode. Returns (decision records, episode metrics)."""
    rng = np.random.default_rng(seed + 12345)
    obs, _ = env.reset(seed=seed)
    records: list[dict] = []
    last: dict[str, int] = {}
    while env.agents:
        acts = {}
        state = None
        for agent, o in obs.items():
            if o["kind"] == NONE:
                acts[agent] = 0
                continue
            if state is None:
                state = env.state()
            if policy == "learned":
                a, logp = np_policy(weights, o, rng, greedy=greedy_actions)
            elif policy == "greedy":
                a, logp = greedy_policy(o), 0.0
            else:
                from .baselines import POLICIES

                a, logp = POLICIES[policy](o, rng), 0.0
            acts[agent] = a
            last[agent] = len(records)
            records.append({"agent": int(agent.split("_")[1]), "options": o["options"], "mask": o["mask"],
                            "self": o["self"], "state": state, "action": a, "logp": logp,
                            "t": env.sim.time, "reward": 0.0, "done": False})  # fmt: skip
        obs, rew, _, trunc, info = env.step(acts)
        for agent, r in rew.items():
            if agent in last:
                records[last[agent]]["reward"] += r
    for idx in last.values():
        records[idx]["done"] = True
    end_time = env.sim.time
    metrics = info[next(iter(info))]["episode"]
    for r in records:
        r["t_end"] = end_time
    return records, metrics


def _worker_episode(args):
    seed, weights, policy, greedy_actions = args
    return run_episode(_ENV, seed, weights, policy, greedy_actions)


# ---------------------------------------------------------------------- torch side
def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def make_nets(n_opt: int, n_self: int, n_state: int, hidden: int):
    torch, nn = _torch()
    actor = nn.Sequential(nn.Linear(n_opt + n_self, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                          nn.Linear(hidden, 1))  # fmt: skip
    critic = nn.Sequential(nn.Linear(n_state + n_self, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                           nn.Linear(256, 1))  # fmt: skip
    return actor, critic


def actor_weights(actor) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().astype(np.float64) for p in actor.parameters()]


def _batch(records):
    opts = np.stack([r["options"] for r in records]).astype(np.float32)
    mask = np.stack([r["mask"] for r in records]).astype(bool)
    selfs = np.stack([r["self"] for r in records]).astype(np.float32)
    states = np.stack([np.concatenate([r["state"], r["self"]]) for r in records]).astype(np.float32)
    return opts, mask, selfs, states


def _logits(actor, opts, selfs, mask):
    torch, _ = _torch()
    k = opts.shape[1]
    x = torch.cat([opts, selfs[:, None, :].expand(-1, k, -1)], dim=2)
    lg = actor(x).squeeze(-1)
    return lg.masked_fill(~mask, -1e9)


def gae(records, values: np.ndarray, gamma_per_s: float, lam: float):
    """Per-agent semi-MDP GAE. ``records`` are in time order within one episode."""
    adv = np.zeros(len(records))
    by_agent: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(records):
        by_agent.setdefault((r["episode"], r["agent"]), []).append(i)
    for idxs in by_agent.values():
        nxt_adv, nxt_val = 0.0, 0.0
        for j in reversed(range(len(idxs))):
            i = idxs[j]
            r = records[i]
            if r["done"] or j == len(idxs) - 1:
                dt = r["t_end"] - r["t"]
                disc = 0.0
            else:
                dt = records[idxs[j + 1]]["t"] - r["t"]
                disc = gamma_per_s ** dt
            delta = r["reward"] + disc * nxt_val - values[i]
            nxt_adv = delta + disc * lam * nxt_adv
            adv[i] = nxt_adv
            nxt_val = values[i]
    return adv, adv + values


class Trainer:
    def __init__(self, env_cfg, world, ppo: PPOConfig | None = None, log=print) -> None:
        torch, _ = _torch()
        self.ppo = ppo or PPOConfig()
        self.env_cfg = env_cfg
        self.world = world
        self.log = log
        torch.manual_seed(self.ppo.seed)
        probe = TaxiEnv(env_cfg, world)
        obs, _ = probe.reset(seed=0)
        o = next(iter(obs.values()))
        self.n_opt = o["options"].shape[1]
        self.n_self = o["self"].shape[0]
        self.n_state = probe.state().shape[0]
        self.actor, self.critic = make_nets(self.n_opt, self.n_self, self.n_state, self.ppo.hidden)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.ppo.lr)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.ppo.workers, initializer=_init_worker, initargs=(env_cfg, world))
        self.history: list[dict] = []

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def collect(self, seeds, policy="learned"):
        w = actor_weights(self.actor)
        out = self.pool.map(_worker_episode, [(s, w, policy, False) for s in seeds])
        records, metrics = [], []
        for ep, (recs, m) in enumerate(out):
            for r in recs:
                r["episode"] = ep
            records.extend(recs)
            metrics.append(m)
        return records, metrics

    def behaviour_clone(self) -> None:
        torch, nn = _torch()
        p = self.ppo
        if p.bc_episodes <= 0:
            return
        seeds = [10_000 + i for i in range(p.bc_episodes)]
        records, metrics = self.collect(seeds, policy="greedy")
        opts, mask, selfs, _ = _batch(records)
        acts = torch.as_tensor([r["action"] for r in records])
        Ot, M, S = torch.as_tensor(opts), torch.as_tensor(mask), torch.as_tensor(selfs)
        idx = torch.nonzero(M.sum(1) > 1).squeeze(1)  # only real choices teach anything
        for step in range(p.bc_steps):
            b = idx[torch.randint(len(idx), (min(256, len(idx)),))]
            lg = _logits(self.actor, Ot[b], S[b], M[b])
            loss = nn.functional.cross_entropy(lg, acts[b])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if step in (0, p.bc_steps - 1):
                with torch.no_grad():
                    lg = _logits(self.actor, Ot[idx], S[idx], M[idx])
                    acc = float((lg.argmax(1) == acts[idx]).float().mean())
                self.log(f"bc step {step} loss {float(loss):.4f} accuracy {acc:.3f}")
        self.log(f"bc data: {len(records)} decisions, greedy delivered/taxi-h "
                 f"{np.mean([m['delivered_per_taxi_hour'] for m in metrics]):.2f}")  # fmt: skip

    def update(self, records) -> dict:
        torch, nn = _torch()
        p = self.ppo
        opts, mask, selfs, states = _batch(records)
        Ot, M, S, ST = map(torch.as_tensor, (opts, mask, selfs, states))
        with torch.no_grad():
            values = self.critic(ST).squeeze(-1).numpy()
        adv, ret = gae(records, values, p.gamma_per_s, p.lam)
        A = torch.as_tensor(adv, dtype=torch.float32)
        R = torch.as_tensor(ret, dtype=torch.float32)
        act = torch.as_tensor([r["action"] for r in records])
        old_logp = torch.as_tensor([r["logp"] for r in records], dtype=torch.float32)
        multi = M.sum(1) > 1
        stats = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "n": 0}
        for _ in range(p.epochs):
            perm = torch.randperm(len(act))
            for i in range(0, len(perm), p.minibatch):
                b = perm[i : i + p.minibatch]
                lg = _logits(self.actor, Ot[b], S[b], M[b])
                logp_all = torch.log_softmax(lg, dim=1)
                logp = logp_all.gather(1, act[b][:, None]).squeeze(1)
                ratio = torch.exp(logp - old_logp[b])
                a_b = A[b]
                a_b = (a_b - A.mean()) / (A.std() + 1e-8)
                pg = -torch.min(ratio * a_b, torch.clamp(ratio, 1 - p.clip, 1 + p.clip) * a_b)
                pg = (pg * multi[b]).sum() / multi[b].sum().clamp(min=1)
                probs = logp_all.exp()
                ent = -(probs * logp_all.clamp(min=-30)).sum(1)
                ent = (ent * multi[b]).sum() / multi[b].sum().clamp(min=1)
                v = self.critic(ST[b]).squeeze(-1)
                vloss = ((v - R[b]) ** 2).mean()
                loss = pg + p.value_coef * vloss - p.entropy * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), p.max_grad_norm)
                self.opt.step()
                stats["pi"] += float(pg)
                stats["v"] += float(vloss)
                stats["ent"] += float(ent)
                stats["kl"] += float((old_logp[b] - logp).mean())
                stats["n"] += 1
        n = max(stats.pop("n"), 1)
        return {k: v / n for k, v in stats.items()}

    def train(self, checkpoint: str | None = None) -> list[dict]:
        p = self.ppo
        t0 = time.time()
        self.behaviour_clone()
        for it in range(p.iterations):
            seeds = [p.seed * 100_000 + it * p.episodes_per_iter + j for j in range(p.episodes_per_iter)]
            records, metrics = self.collect(seeds)
            st = self.update(records)
            row = {"iter": it, "decisions": len(records), "wall_s": round(time.time() - t0, 1),
                   "return_per_taxi": float(sum(r["reward"] for r in records)) / (len(metrics) * self.env_cfg.taxi.num_taxis),
                   **{k: float(np.nanmean([m[k] for m in metrics])) for k in metrics[0]}, **st}  # fmt: skip
            self.history.append(row)
            self.log(f"it {it:3d} ret/taxi {row['return_per_taxi']:6.2f} deliv/taxi-h {row['delivered_per_taxi_hour']:5.2f} "
                     f"wait {row['mean_wait_s']:5.0f}s abandon {row['abandoned']:4.1f} ent {st['ent']:.3f} kl {st['kl']:.4f} "
                     f"[{row['wall_s']:.0f}s]")  # fmt: skip
            if checkpoint and (it % 5 == 4 or it == p.iterations - 1):
                self.save(checkpoint)
        return self.history

    def save(self, path: str) -> None:
        torch, _ = _torch()
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "dims": (self.n_opt, self.n_self, self.n_state), "ppo": asdict(self.ppo),
                    "history": self.history}, path)  # fmt: skip


def load_actor_weights(path: str) -> list[np.ndarray]:
    torch, _ = _torch()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_opt, n_self, n_state = ck["dims"]
    actor, _ = make_nets(n_opt, n_self, n_state, ck["ppo"]["hidden"])
    actor.load_state_dict(ck["actor"])
    return actor_weights(actor)


def evaluate(env_cfg, world, policies: dict, seeds, workers: int = 4) -> dict:
    """Mean metrics per policy over the same seeds. ``policies``: name -> weights or None."""
    ctx = mp.get_context("spawn")
    out = {}
    with ctx.Pool(workers, initializer=_init_worker, initargs=(env_cfg, world)) as pool:
        for name, weights in policies.items():
            policy = "learned" if weights is not None else name
            res = pool.map(_worker_episode, [(s, weights, policy, True) for s in seeds])
            ms = [m for _, m in res]
            out[name] = {k: (float(np.nanmean([m[k] for m in ms])), float(np.nanstd([m[k] for m in ms])))
                         for k in ms[0]}  # fmt: skip
            out[name]["_per_seed"] = ms
    return out


__all__ = ["PPOConfig", "Trainer", "evaluate", "load_actor_weights", "run_episode"]
