"""RL fine-tuning of the pre-trained camera policy (PPO, BC-regularised).

    python -m metatraffic3d.driving.rl --init runs/bc/policy.pt --out runs/rl --steps 20000

The backbone stays frozen (features are computed once per step); the fusion head, the command
branches, the value head and the action std are trained with PPO on the environment reward
(route progress, comfort, infraction penalties, collision and arrival terms). A small
behaviour-cloning term on the expert's labels for the visited states (``--bc-coef``) keeps the
policy from forgetting how to drive while RL improves it (as in DAPG / BC-regularised PPO).
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
from .policy import Encoder, bc_loss, device, make_head


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", default="runs/rl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[4, 5, 7])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--rollout", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bc-coef", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.99)
    a = ap.parse_args(argv)
    import torch

    os.makedirs(a.out, exist_ok=True)
    dev = device()
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    head = make_head(ck["n_cams"]).to(dev)
    head.load_state_dict(ck["head"])
    enc = Encoder(pretrained=ck.get("pretrained", True), dev=dev)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)
    envs = []
    for s in a.seeds:
        cfg = SimConfig()
        cfg.map.seed = s
        assets = AssetLibrary("auto")
        envs.append(DrivingEnv(cfg, build_world(cfg, assets=assets), assets=assets, seed=s))
    env_i, ep_seed = 0, 500_000
    env = envs[0]
    obs, info = env.reset(seed=ep_seed)
    ep_ret, ep_len, episodes, log = 0.0, 0, [], []
    t0 = time.time()
    total = 0
    while total < a.steps:
        buf = {k: [] for k in ("feat", "state", "cmd", "act", "gear", "logp", "val", "rew", "done", "term", "boot",
                               "e_ctl", "e_gear")}  # fmt: skip
        head.eval()
        for _ in range(a.rollout):
            feat = enc(obs["images"][None])
            st = torch.as_tensor(obs["state"][None], device=dev)
            cm = torch.as_tensor(obs["command"][None], device=dev)
            with torch.no_grad():
                mean, gl, v = head(feat, st, cm)
                std = head.log_std.exp()
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample()
                gdist = torch.distributions.Categorical(logits=gl)
                gear = gdist.sample()
                logp = dist.log_prob(act).sum(1) + gdist.log_prob(gear)
            a_np = act[0].cpu().numpy()
            ctl = np.array([np.clip(a_np[0], -1, 1), np.clip(a_np[1], 0, 1), np.clip(a_np[2], 0, 1)], np.float32)
            obs2, r, term, trunc, info = env.step({"control": ctl, "gear": int(gear.item())})
            ex = info["privileged"]["expert"]
            boot = 0.0
            if trunc and not term:  # cut off by a time limit: the future still has value
                with torch.no_grad():
                    _, _, vb = head(enc(obs2["images"][None]), torch.as_tensor(obs2["state"][None], device=dev),
                                    torch.as_tensor(obs2["command"][None], device=dev))  # fmt: skip
                boot = float(vb.item())
            for k, val in (("feat", feat[0].half().cpu()), ("state", st[0].cpu()), ("cmd", cm[0].cpu()),
                           ("act", act[0].cpu()), ("gear", gear[0].cpu()), ("logp", logp[0].cpu()), ("val", v[0].cpu()),
                           ("rew", r), ("done", term or trunc), ("term", term), ("boot", boot),
                           ("e_ctl", ex["control"]), ("e_gear", ex["gear"])):  # fmt: skip
                buf[k].append(val)
            ep_ret += r
            ep_len += 1
            total += 1
            obs = obs2
            if term or trunc:
                episodes.append({"return": ep_ret, "len": ep_len, "reason": info["reason"], "map": a.seeds[env_i],
                                 "infractions": len(info["privileged"]["infractions"])})  # fmt: skip
                ep_ret, ep_len = 0.0, 0
                env_i = int(rng.integers(len(envs)))
                env = envs[env_i]
                ep_seed += 1
                obs, info = env.reset(seed=ep_seed)
        # bootstrap and GAE
        with torch.no_grad():
            _, _, v_last = head(enc(obs["images"][None]), torch.as_tensor(obs["state"][None], device=dev),
                                torch.as_tensor(obs["command"][None], device=dev))  # fmt: skip
        vals = torch.stack(buf["val"]).numpy()
        rews = np.array(buf["rew"], np.float32)
        dones = np.array(buf["done"], np.float32)
        terms = np.array(buf["term"], np.float32)
        boots = np.array(buf["boot"], np.float32)
        adv = np.zeros_like(rews)
        last, nv = 0.0, float(v_last.item())
        for t in reversed(range(len(rews))):
            if dones[t]:  # episode ended here: bootstrap only if it was cut off, not finished
                nv, last = (0.0 if terms[t] else boots[t]), 0.0
            delta = rews[t] + a.gamma * nv - vals[t]
            last = delta + a.gamma * 0.95 * last
            adv[t] = last
            nv = vals[t]
        ret = adv + vals
        F = torch.stack(buf["feat"]).float().to(dev)
        S, C = torch.stack(buf["state"]).to(dev), torch.stack(buf["cmd"]).to(dev)
        A, G, LP = torch.stack(buf["act"]).to(dev), torch.stack(buf["gear"]).to(dev), torch.stack(buf["logp"]).to(dev)
        ADV = torch.as_tensor(adv, device=dev)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        RET = torch.as_tensor(ret, device=dev)
        EC = torch.as_tensor(np.array(buf["e_ctl"]), device=dev)
        EG = torch.as_tensor(np.array(buf["e_gear"]), device=dev)
        head.train()
        stats = []
        for _ in range(4):
            perm = torch.randperm(len(rews), device=dev)
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                mean, gl, v = head(F[b], S[b], C[b])
                dist = torch.distributions.Normal(mean, head.log_std.exp())
                gd = torch.distributions.Categorical(logits=gl)
                logp = dist.log_prob(A[b]).sum(1) + gd.log_prob(G[b])
                ratio = (logp - LP[b]).exp()
                pg = -torch.min(ratio * ADV[b], ratio.clamp(0.8, 1.2) * ADV[b]).mean()
                vl = ((v - RET[b]) ** 2).mean()
                bcl, _ = bc_loss(mean, gl, EC[b], EG[b])
                loss = pg + 0.5 * vl + a.bc_coef * bcl - 0.001 * dist.entropy().sum(1).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
                opt.step()
                stats.append((float(pg), float(vl), float(bcl)))
        recent = episodes[-10:]
        row = {"steps": total, "wall_s": round(time.time() - t0), "episodes": len(episodes),
               "mean_return_10": float(np.mean([e["return"] for e in recent])) if recent else None,
               "arrived_10": float(np.mean([e["reason"] == "arrived" for e in recent])) if recent else None,
               "pg": float(np.mean([s[0] for s in stats])), "v": float(np.mean([s[1] for s in stats])),
               "bc": float(np.mean([s[2] for s in stats])), "std": head.log_std.exp().tolist()}  # fmt: skip
        log.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()}, "n_cams": ck["n_cams"],
                    "pretrained": ck.get("pretrained", True), "log": log, "episodes": episodes},
                   os.path.join(a.out, "policy.pt"))  # fmt: skip


if __name__ == "__main__":
    main()
