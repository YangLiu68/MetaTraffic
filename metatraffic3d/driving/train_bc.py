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


def peak_memory_gb() -> float:
    """Peak physical footprint of this process so far (includes MPS memory on macOS)."""
    try:
        import ctypes
        import ctypes.util

        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        buf = ctypes.create_string_buffer(512)
        if lib.proc_pid_rusage(os.getpid(), 4, buf) == 0:  # rusage_info_v4.ri_lifetime_max_phys_footprint
            return int.from_bytes(buf.raw[240:248], "little") / 1e9
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def cache_key(path: str, cache_dir: str) -> str:
    folder = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return os.path.join(cache_dir, f"{folder}__" + os.path.basename(path).replace(".npz", ".tok.npy"))


def cache_fresh(path: str, cache_dir: str) -> bool:
    key = cache_key(path, cache_dir)
    return os.path.exists(key) and os.path.getmtime(key) >= os.path.getmtime(path)


def cache_features(path: str, enc, cache_dir: str, batch: int = 64) -> np.ndarray:
    """(T, N, 15, 512) float16 memmap of an episode's spatial features (computed once)."""
    key = cache_key(path, cache_dir)
    if not cache_fresh(path, cache_dir):
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
            self.eps.append({"raw": cache_features(f, enc, cache_dir), "state": z["state"].astype(np.float32),
                             "command": z["command"].astype(np.float32), "control": z["control"].astype(np.float32),
                             "gear": z["gear"].astype(np.int64)})  # fmt: skip
        self.items = np.array([(e, t) for e, ep in enumerate(self.eps) for t in range(len(ep["state"]))])

    def sample_tokens(self, n: int, rng) -> np.ndarray:
        rows = self.items[rng.choice(len(self.items), size=min(n, len(self.items)), replace=False)]
        return np.concatenate([np.asarray(self.eps[e]["raw"][t], np.float32).reshape(-1, FEAT) for e, t in rows])

    def compress(self, pca) -> None:
        """Project every cached feature with ``pca`` and keep the result in RAM (float16)."""
        for ep in self.eps:
            raw = ep.pop("raw")
            out = np.empty(raw.shape[:3] + (pca.comp.shape[0],), np.float16)
            for i in range(0, len(raw), 256):
                out[i:i + 256] = pca(np.asarray(raw[i:i + 256])).astype(np.float16)
            ep["feat"] = out

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
    ap.add_argument("--pca", type=int, default=64, help="compress backbone features to this many dims")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--init", default=None, help="warm start from this checkpoint (keeps its PCA and val episodes)")
    ap.add_argument("--no-keyframes", action="store_true", help="do not up-weight gear changes / braking")
    a = ap.parse_args(argv)
    import torch

    from .policy import bc_loss, device, keyframe_weights, make_head
    from .transformer import PCA, SpatialEncoder, chunk_loss, make_driveformer

    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    dev = device() if a.device == "auto" else torch.device(a.device)
    files = sorted(f for d in a.data for f in glob.glob(os.path.join(d, "*.npz")))
    # the backbone is only needed for episodes whose features are not cached yet
    enc = None if all(cache_fresh(f, a.cache) for f in files) else SpatialEncoder(device())
    rng = np.random.default_rng(0)
    init = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else None
    if init is not None:  # keep the old validation episodes; split only the new ones
        old = set(init["train_files"]) | set(init["val_files"])
        new = [f for f in files if f not in old]
        order = rng.permutation(len(new))
        val_files = sorted([f for f in init["val_files"] if f in files] +
                           [new[i] for i in order[:int(round(len(new) * a.val))]])  # fmt: skip
        a.pca = init["feat_dim"]
    else:
        order = rng.permutation(len(files))
        n_val = max(1, int(round(len(files) * a.val)))
        val_files = [files[i] for i in sorted(order[:n_val])]
    tr_files = [f for f in files if f not in val_files]
    t0 = time.time()
    tr, va = Demos(tr_files, enc, a.cache), Demos(val_files, enc, a.cache)
    if enc is not None:
        del enc
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()  # give back the memory the backbone used for feature extraction
    if init is not None:
        pca = PCA(init["pca_mean"], init["pca_comp"])
        pca.explained = float("nan")
    else:
        pca = PCA.fit(tr.sample_tokens(800, rng), a.pca)
    tr.compress(pca)
    va.compress(pca)
    print(f"features ready in {time.time() - t0:.0f}s: {len(tr.items)} train / {len(va.items)} val steps "
          f"({len(tr_files)}/{len(val_files)} episodes) on {dev}; PCA {FEAT}->{a.pca} keeps "
          f"{pca.explained:.1%} of the variance; peak memory {peak_memory_gb():.2f} GB", flush=True)  # fmt: skip
    n_cams = tr.eps[0]["feat"].shape[1]
    model = (make_driveformer(n_cams, feat_dim=a.pca) if a.arch == "transformer"
             else make_head(n_cams, feat_dim=a.pca)).to(dev)  # fmt: skip
    if init is not None:
        assert init["arch"] == a.arch, f"--init is a {init['arch']} checkpoint"
        model.load_state_dict(init["model"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2 if a.arch == "transformer" else 1e-4)
    steps_per_epoch = len(tr.items) // a.batch
    if a.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, a.max_steps_per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_epoch, pct_start=0.1)

    def run(b: dict, train: bool):
        tt = {k: torch.as_tensor(v, device=dev) for k, v in b.items()}
        img = tt["img"] + (0.05 * torch.randn_like(tt["img"]) if train else 0.0)
        now = tt["states"][:, -1] if a.arch == "transformer" else tt["states"]
        w_ctl, w_gear = (None, None) if a.no_keyframes else keyframe_weights(now, tt["ctl"], tt["gear"])
        if a.arch == "transformer":
            mean, gl, chunk, _ = model(img, tt["ages"], tt["states"], tt["cmd"])
        else:
            mean, gl, _ = model(img, tt["states"], tt["cmd"])
        loss, m = bc_loss(mean, gl, tt["ctl"], tt["gear"], w_ctl, w_gear, pedal_logits=model.last_pedal_logits)
        if not train:  # counts for the rare decisions (turned into rates per epoch)
            shift = tt["gear"] != now[:, 2:9].argmax(1)
            moving_brake = (tt["ctl"][:, 2] > 0.1) & (now[:, 0] * 15.0 > 1.0)
            m["n_shift"] = float(shift.sum())
            m["hit_shift"] = float(((gl.argmax(1) == tt["gear"]) & shift).sum())
            m["n_brake"] = float(moving_brake.sum())
            m["err_brake"] = float(((mean[:, 2] - tt["ctl"][:, 2]).abs() * moving_brake).sum())
        if a.arch == "transformer":
            lc = chunk_loss(chunk, tt["chunk"], tt["mask"])
            m["chunk_l1"] = float(lc)
            loss = loss + a.chunk_weight * lc
        return loss, m

    vidx = np.sort(np.random.default_rng(1).permutation(len(va.items))[:6000])  # fixed spread-out val subset
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
            for i in range(0, len(vidx), a.batch):
                loss, m = run(va.batch(va.items[vidx[i:i + a.batch]], a.arch), False)
                vl += float(loss)
                for k, v in m.items():
                    vm[k] = vm.get(k, 0.0) + v
                nb += 1
        counts = {k: vm.pop(k) for k in ("n_shift", "hit_shift", "n_brake", "err_brake")}
        row = {"epoch": epoch, "train_loss": tot / max(steps_per_epoch, 1), "val_loss": vl / nb,
               **{k: v / nb for k, v in vm.items()},
               "shift_acc": counts["hit_shift"] / max(counts["n_shift"], 1.0), "n_shift": int(counts["n_shift"]),
               "brake_moving_l1": counts["err_brake"] / max(counts["n_brake"], 1.0),
               "wall_s": round(time.time() - t0), "peak_gb": round(peak_memory_gb(), 2)}  # fmt: skip
        if dev.type == "mps":
            torch.mps.empty_cache()
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if row["val_loss"] < best[0]:
            best = (row["val_loss"], {k: v.detach().cpu() for k, v in model.state_dict().items()})
        # save the best weights so far after every epoch, so an interrupted run still leaves a policy
        tmp = os.path.join(a.out, "policy.pt.tmp")
        torch.save({"arch": a.arch, "model": best[1], "n_cams": n_cams, "pretrained": True, "log": log,
                    "feat_dim": a.pca, "pca_mean": pca.mean, "pca_comp": pca.comp, "epochs_done": epoch + 1,
                    "init": a.init,
                    "params": n_params, "train_files": tr_files, "val_files": val_files}, tmp)  # fmt: skip
        os.replace(tmp, os.path.join(a.out, "policy.pt"))
    print(f"saved {a.out}/policy.pt ({a.arch}, {n_params / 1e6:.2f} M params, best val {best[0]:.4f})")


if __name__ == "__main__":
    main()
