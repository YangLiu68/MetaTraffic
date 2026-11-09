"""DriveFormer: a time-aware multi-camera transformer policy (Wayformer-style token fusion).

Every input becomes a token in one sequence:

* image tokens: each camera frame -> frozen ImageNet ResNet-18 -> a 3 x 5 grid of 512-d
  features (15 tokens per camera, so *where* things are in the image is kept);
* history: the last ``HIST`` frames, ``STRIDE`` control steps apart (default 4 frames, 0.2 s
  apart = 0.6 s), for every camera;
* a state token per history frame (speed, rpm, gear, steering, slope, shifting);
* one route-command token (follow / left / right / straight / U-turn / arrive + distance);
* three learned query tokens: action, future-action chunk, value.

Time awareness: image and state tokens get a Fourier embedding of their *age in seconds*
(how long ago the frame was captured; the first frames of an episode repeat the oldest
available frame with its true, smaller age), plus camera, image-position and token-type
embeddings. A pre-norm transformer encoder mixes everything; the query tokens read out

* the action now: steer (tanh), throttle and brake (sigmoid), gear logits (R N 1-5);
* a chunk of the next ``HORIZON`` controls (1 s) - an auxiliary, trajectory-like target that
  makes the policy plan ahead and keeps successive actions consistent;
* a value estimate for RL fine-tuning.
"""

from __future__ import annotations

import math

import numpy as np

HIST = 4
STRIDE = 2
HORIZON = 10
GRID = (3, 5)  # ResNet-18 layer4 grid for 96 x 160 images
N_TOK = GRID[0] * GRID[1]
FEAT = 512
DT = 0.1


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def history_index(t: int, hist: int = HIST, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Frame indices (oldest first) and their ages in seconds for step ``t`` of an episode."""
    idx = np.array([max(t - k * stride, 0) for k in range(hist - 1, -1, -1)])
    return idx, (t - idx) * DT


class SpatialEncoder:
    """Images (B, N, H, W, 3) uint8 -> tokens (B, N, 15, 512) with frozen ImageNet ResNet-18."""

    def __init__(self, dev=None) -> None:
        import torchvision

        torch, nn = _torch()
        from .policy import device

        self.dev = dev or device()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(*list(net.children())[:-2]).to(self.dev).eval()  # up to layer4
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, images: np.ndarray):
        torch, _ = _torch()
        b, n, h, w, _ = images.shape
        x = torch.as_tensor(np.ascontiguousarray(images).reshape(b * n, h, w, 3), device=self.dev)
        x = (x.permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        with torch.no_grad():
            f = self.net(x)  # (b*n, 512, 3, 5)
        return f.flatten(2).transpose(1, 2).reshape(b, n, -1, FEAT)


def make_driveformer(n_cams: int = 4, d: int = 256, layers: int = 4, heads: int = 8, state_dim: int = 12,
                     cmd_dim: int = 7, n_gears: int = 7, feat_dim: int = FEAT):  # fmt: skip
    torch, nn = _torch()

    class DriveFormer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.img_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, d))
            self.cam_emb = nn.Embedding(n_cams, d)
            self.pos_emb = nn.Embedding(N_TOK, d)
            self.type_emb = nn.Embedding(5, d)  # image, state, command, action/chunk query, value query
            self.register_buffer("freqs", torch.tensor([2.0 ** k for k in range(-2, 6)]) * math.pi)
            self.time_proj = nn.Linear(16, d)
            self.state_proj = nn.Linear(state_dim, d)
            self.cmd_proj = nn.Linear(cmd_dim, d)
            self.queries = nn.Parameter(torch.randn(3, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.act_head = nn.Linear(d, 3 + n_gears)
            self.chunk_head = nn.Linear(d, HORIZON * 3)
            self.value_head = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
            self.log_std = nn.Parameter(torch.full((3,), -1.6))
            self.n_cams = n_cams

        def time_embed(self, age):  # age (..., ) seconds -> (..., d)
            a = age[..., None] * self.freqs
            return self.time_proj(torch.cat([a.sin(), a.cos()], dim=-1))

        def forward(self, img, ages, states, command):
            """img (B, T, N, 15, feat_dim); ages (B, T) s; states (B, T, 12); command (B, 7)."""
            b, t, n, k, _ = img.shape
            dev = img.device
            tok = self.img_proj(img)
            tok = tok + self.cam_emb(torch.arange(n, device=dev))[None, None, :, None, :]
            tok = tok + self.pos_emb(torch.arange(k, device=dev))[None, None, None, :, :]
            te = self.time_embed(ages)  # (B, T, d)
            tok = tok + te[:, :, None, None, :] + self.type_emb.weight[0]
            st = self.state_proj(states) + te + self.type_emb.weight[1]
            cm = (self.cmd_proj(command) + self.type_emb.weight[2])[:, None]
            q = self.queries[None].expand(b, -1, -1) + self.type_emb.weight[[3, 3, 4]][None]
            x = torch.cat([tok.reshape(b, t * n * k, -1), st, cm, q], dim=1)
            if self.training and torch.is_grad_enabled():
                # gradient checkpointing: keep only each layer's input and recompute its attention
                # maps in the backward pass (the maps are ~250 tokens squared per head and sample)
                from torch.utils.checkpoint import checkpoint

                for layer in self.encoder.layers:
                    x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = self.encoder(x)
            h = self.norm(x)
            qa, qc, qv = h[:, -3], h[:, -2], h[:, -1]
            out = self.act_head(qa)
            raw = out[:, :3]
            mean = torch.stack([torch.tanh(raw[:, 0]), torch.sigmoid(raw[:, 1]), torch.sigmoid(raw[:, 2])], dim=1)
            ch = self.chunk_head(qc).view(b, HORIZON, 3)
            chunk = torch.stack([torch.tanh(ch[..., 0]), torch.sigmoid(ch[..., 1]), torch.sigmoid(ch[..., 2])], dim=-1)
            return mean, out[:, 3:], chunk, self.value_head(qv).squeeze(-1)

    return DriveFormer()


class PCA:
    """Fixed linear compression of backbone features (fitted on training data, stored in checkpoints)."""

    def __init__(self, mean: np.ndarray, comp: np.ndarray) -> None:
        self.mean = np.asarray(mean, np.float32)
        self.comp = np.asarray(comp, np.float32)  # (k, 512)

    @classmethod
    def fit(cls, x: np.ndarray, k: int) -> PCA:
        """Top-``k`` principal directions of rows ``x`` (n, 512), via the 512 x 512 covariance."""
        x = np.asarray(x, np.float64)
        mean = x.mean(0)
        xc = x - mean
        evals, evecs = np.linalg.eigh(xc.T @ xc)  # ascending
        evals, evecs = evals[::-1].clip(min=0.0), evecs[:, ::-1]
        pca = cls(mean, evecs[:, :k].T)
        pca.explained = float(evals[:k].sum() / max(evals.sum(), 1e-12))
        return pca

    def __call__(self, x):
        """numpy or torch array (..., 512) -> (..., k)."""
        if isinstance(x, np.ndarray):
            return (x.astype(np.float32) - self.mean) @ self.comp.T
        torch, _ = _torch()
        m = torch.as_tensor(self.mean, device=x.device)
        c = torch.as_tensor(self.comp, device=x.device)
        return (x.float() - m) @ c.T


def chunk_loss(chunk, target, mask):
    """L1 over the future controls that exist (mask (B, H) = 1 inside the episode)."""
    err = (chunk - target).abs().sum(-1)
    return (err * mask).sum() / mask.sum().clamp(min=1.0)


class TransformerRunner:
    """Callable policy for the env with its own frame history (call ``reset()`` per episode)."""

    def __init__(self, model, encoder: SpatialEncoder, pca: PCA | None = None) -> None:
        self.model = model.to(encoder.dev).eval()
        self.enc = encoder
        self.pca = pca
        self.reset()

    def reset(self) -> None:
        self.frames: list = []
        self.states: list = []

    def __call__(self, obs: dict) -> dict:
        torch, _ = _torch()
        dev = self.enc.dev
        f = self.enc(obs["images"][None])[0]  # (N, 15, 512)
        self.frames.append(self.pca(f) if self.pca is not None else f)
        self.states.append(torch.as_tensor(obs["state"], device=dev))
        t = len(self.frames) - 1
        idx, ages = history_index(t)
        img = torch.stack([self.frames[i] for i in idx])[None]
        st = torch.stack([self.states[i] for i in idx])[None]
        cm = torch.as_tensor(obs["command"][None], device=dev)
        with torch.no_grad():
            mean, gl, _, _ = self.model(img, torch.as_tensor(ages[None], dtype=torch.float32, device=dev), st, cm)
        keep = HIST * STRIDE + 1
        if len(self.frames) > keep:  # only the history window is needed
            drop = len(self.frames) - keep
            self.frames, self.states = self.frames[drop:], self.states[drop:]
        return {"control": mean[0].cpu().numpy().astype(np.float32), "gear": int(gl[0].argmax().item())}
