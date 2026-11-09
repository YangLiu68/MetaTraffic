"""Behaviour cloning on recorded demonstrations (pre-training before RL).

    python -m metatraffic3d.driving.train_bc --data runs/data/demos --out runs/bc --arch transformer

1. Every camera frame goes once through the frozen ImageNet ResNet-18; its 3 x 5 grid of
   512-d features is cached on disk (float16, memory-mapped, one file per episode).
2. ``--arch transformer`` trains DriveFormer (``transformer.py``) on 4-frame histories with
   their real time gaps, predicting the action now plus the next second of controls.
   ``--arch mlp`` trains the earlier concatenate-and-branch model (``policy.py``) on the same
   cache (spatially pooled features of the current frame) as a baseline.
Validation is split by episode.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from .jpeg import decode
from .transformer import FEAT, HORIZON, N_TOK, history_index


def cache_features(path: str, enc, cache_dir: str, batch: int = 64) -> np.ndarray:
    """(T, N, 15, 512) float16 memmap of an episode's spatial features (computed once)."""
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    key = os.path.join(cache_dir, f"{folder}__" + os.path.basename(path).replace(".npz", ".tok.npy"))
    if not os.path.exists(key) or os.path.getmtime(key) < os.path.getmtime(path):
        z = np.load(path)
        jpeg, off, n = z["jpeg"].tobytes(), z["offsets"], int(z["n_cams"])
        frames = [decode(jpeg[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
        t = len(frames) // n
        imgs = np.stack(frames).reshape(t, n, *frames[0].shape)
        tmp = key[:-4] + ".tmp.npy"
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(t, n, N_TOK, FEAT))
        for i in range(0, t, batch):
            out[i:i + batch] = enc(imgs[i:i + batch]).cpu().numpy().astype(np.float16)
        out.flush()
        del out
        os.replace(tmp, key)
    return np.load(key, mmap_mode="r")


class Demos:
    def __init__(self, files: list[str], enc, cache_dir: str) -> None:
        self.eps = []
        for f in files:
            z = np.load(f)
            self.eps.append({"feat": cache_features(f, enc, cache_dir), "state": z["state"].astype(np.float32),
                             "command": z["command"].astype(np.float32), "control": z["control"].astype(np.float32),
                             "gear": z["gear"].astype(np.int64)})  # fmt: skip
        self.items = np.array([(e, t) for e, ep in enumerate(self.eps) for t in range(len(ep["state"]))])

    def batch(self, items: np.ndarray, arch: str) -> dict:
        out: dict[str, list] = {k: [] for k in ("img", "ages", "states", "cmd", "ctl", "gear", "chunk", "mask")}
        for e, t in items:
            ep = self.eps[e]
            T = len(ep["state"])
            if arch == "transformer":
                idx, ages = history_index(int(t))
                out["img"].append(np.asarray(ep["feat"][idx], np.float32))
                out["ages"].append(ages.astype(np.float32))
                out["states"].append(ep["state"][idx])
            else:
                out["img"].append(np.asarray(ep["feat"][t], np.float32).mean(axis=1))  # pooled = ResNet avgpool
                out["states"].append(ep["state"][t])
            out["cmd"].append(ep["command"][t])
            out["ctl"].append(ep["control"][t])
            out["gear"].append(ep["gear"][t])
            fut = np.arange(t + 1, t + 1 + HORIZON)
            out["chunk"].append(ep["control"][np.minimum(fut, T - 1)])
            out["mask"].append((fut < T).astype(np.float32))
        return {k: np.stack(v) for k, v in out.items() if v}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", default=["runs/data/demos"])
    ap.add_argument("--out", default="runs/bc")
    ap.add_argument("--arch", choices=["transformer", "mlp"], default="transformer")
    ap.add_argument("--cache", default="runs/features")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--chunk-weight", type=float, default=0.5)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    a = ap.parse_args(argv)
    import torch

    from .policy import bc_loss, device, make_head
    from .transformer import SpatialEncoder, chunk_loss, make_driveformer

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    dev = device()
    enc = SpatialEncoder(dev)
    files = sorted(f for d in a.data for f in glob.glob(os.path.join(d, "*.npz")))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * a.val)))
    val_files = [files[i] for i in sorted(order[:n_val])]
    tr_files = [f for f in files if f not in val_files]
    t0 = time.time()
    tr, va = Demos(tr_files, enc, a.cache), Demos(val_files, enc, a.cache)
    del enc
    print(f"features ready in {time.time() - t0:.0f}s: {len(tr.items)} train / {len(va.items)} val steps "
          f"({len(tr_files)}/{len(val_files)} episodes) on {dev}", flush=True)  # fmt: skip
    n_cams = tr.eps[0]["feat"].shape[1]
    model = (make_driveformer(n_cams) if a.arch == "transformer" else make_head(n_cams)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2 if a.arch == "transformer" else 1e-4)
    steps_per_epoch = len(tr.items) // a.batch
    if a.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, a.max_steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)

    def run(b: dict, train: bool):
        tt = {k: torch.as_tensor(v, device=dev) for k, v in b.items()}
        img = tt["img"] + (0.05 * torch.randn_like(tt["img"]) if train else 0.0)
        if a.arch == "transformer":
            mean, gl, chunk, _ = model(img, tt["ages"], tt["states"], tt["cmd"])
            loss, m = bc_loss(mean, gl, tt["ctl"], tt["gear"])
            lc = chunk_loss(chunk, tt["chunk"], tt["mask"])
            m["chunk_l1"] = float(lc)
            return loss + a.chunk_weight * lc, m
        mean, gl, _ = model(img, tt["states"], tt["cmd"])
        return bc_loss(mean, gl, tt["ctl"], tt["gear"])

    log, best = [], (1e9, None)
    for epoch in range(a.epochs):
        model.train()
        perm = rng.permutation(len(tr.items))[: steps_per_epoch * a.batch]
        tot = 0.0
        for i in range(steps_per_epoch):
            loss, _ = run(tr.batch(tr.items[perm[i * a.batch:(i + 1) * a.batch]], a.arch), True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        model.eval()
        vl, vm, nb = 0.0, {}, 0
        with torch.no_grad():
            for i in range(0, min(len(va.items), 6000), 256):
                loss, m = run(va.batch(va.items[i:i + 256], a.arch), False)
                vl += float(loss)
                for k, v in m.items():
                    vm[k] = vm.get(k, 0.0) + v
                nb += 1
        row = {"epoch": epoch, "train_loss": tot / max(steps_per_epoch, 1), "val_loss": vl / nb,
               **{k: v / nb for k, v in vm.items()}, "wall_s": round(time.time() - t0)}  # fmt: skip
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if row["val_loss"] < best[0]:
            best = (row["val_loss"], {k: v.detach().cpu() for k, v in model.state_dict().items()})
    torch.save({"arch": a.arch, "model": best[1], "n_cams": n_cams, "pretrained": True, "log": log,
                "params": n_params, "train_files": tr_files, "val_files": val_files},
               os.path.join(a.out, "policy.pt"))  # fmt: skip
    print(f"saved {a.out}/policy.pt ({a.arch}, {n_params / 1e6:.2f} M params, best val {best[0]:.4f})")


if __name__ == "__main__":
    main()
