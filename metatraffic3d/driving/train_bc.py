"""Behaviour cloning on recorded demonstrations (pre-training before RL).

    python -m metatraffic3d.driving.train_bc --data data/demos --out runs/bc

1. Every camera frame goes once through the frozen ImageNet ResNet-18; features are cached.
2. The fusion + command-branch head is trained to imitate the expert's steer / throttle /
   brake / gear, with episode-level train/validation split.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from .collect import decode
from .policy import Encoder, bc_loss, device, make_head


def load_episodes(pattern_dirs: list[str]):
    files = sorted(f for d in pattern_dirs for f in glob.glob(os.path.join(d, "*.npz")))
    return files


def episode_features(path: str, enc: Encoder, cache_dir: str, batch: int = 64):
    """(features (T, N, 512) float16, state, command, control, gear) for one episode (cached)."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    key = os.path.join(cache_dir, f"{folder}__" + os.path.basename(path).replace(".npz", ".feat.npz"))
    z = np.load(path)
    if os.path.exists(key):
        feats = np.load(key)["feats"]
    else:
        jpeg, off, n = z["jpeg"].tobytes(), z["offsets"], int(z["n_cams"])
        frames = [decode(jpeg[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
        t = len(frames) // n
        imgs = np.stack(frames).reshape(t, n, *frames[0].shape)
        out = []
        for i in range(0, t, batch):
            out.append(enc(imgs[i:i + batch]).cpu().numpy().astype(np.float16))
        feats = np.concatenate(out) if out else np.zeros((0, n, 512), np.float16)
        np.savez(key, feats=feats)
    return feats, z["state"], z["command"], z["control"], z["gear"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["data/demos"])
    ap.add_argument("--out", default="runs/bc")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--init", default=None, help="start from this checkpoint (DAgger rounds)")
    ap.add_argument("--no-pretrained", action="store_true", help="ablation: random backbone weights")
    a = ap.parse_args(argv)
    import torch

    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "features" + ("_random" if a.no_pretrained else ""))
    os.makedirs(cache, exist_ok=True)
    dev = device()
    enc = Encoder(pretrained=not a.no_pretrained, dev=dev)
    files = load_episodes(a.data)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    n_val = max(1, int(len(files) * a.val))
    val_files = {files[i] for i in order[:n_val]}
    tr, va = [], []
    t0 = time.time()
    for f in files:
        ep = episode_features(f, enc, cache)
        (va if f in val_files else tr).append(ep)
    print(f"features for {len(files)} episodes in {time.time() - t0:.0f}s on {dev}", flush=True)

    def stack(eps):
        return [torch.as_tensor(np.concatenate([e[i] for e in eps]).astype(np.float32 if i < 4 else np.int64))
                for i in range(5)]  # fmt: skip

    Xtr, Xva = stack(tr), stack(va)
    n_cams = Xtr[0].shape[1]
    head = make_head(n_cams).to(dev)
    if a.init:
        head.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["head"])
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    log = []
    best = (1e9, None)
    for epoch in range(a.epochs):
        head.train()
        perm = torch.randperm(len(Xtr[0]))
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            f, s, c, ctl, g = (x[b].to(dev) for x in Xtr)
            f = f + 0.05 * torch.randn_like(f)  # light feature noise as regularisation
            mean, gl, _ = head(f, s, c)
            loss, _ = bc_loss(mean, gl, ctl, g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        head.eval()
        with torch.no_grad():
            f, s, c, ctl, g = (x.to(dev) for x in Xva)
            mean, gl, _ = head(f, s, c)
            vloss, m = bc_loss(mean, gl, ctl, g)
        row = {"epoch": epoch, "train_loss": tot / len(perm), "val_loss": float(vloss), **m}
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if float(vloss) < best[0]:
            best = (float(vloss), {k: v.detach().cpu() for k, v in head.state_dict().items()})
    torch.save({"head": best[1], "n_cams": n_cams, "pretrained": not a.no_pretrained, "log": log,
                "train_steps": len(Xtr[0]), "val_steps": len(Xva[0]), "files": files}, os.path.join(a.out, "policy.pt"))  # fmt: skip
    print(f"saved {a.out}/policy.pt (best val {best[0]:.4f}); {len(Xtr[0])} train / {len(Xva[0])} val steps")


if __name__ == "__main__":
    main()
