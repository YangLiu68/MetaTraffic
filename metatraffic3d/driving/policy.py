"""Camera-only driving policy with a pretrained vision backbone (transfer learning).

* Backbone: torchvision ResNet-18 with ImageNet weights, shared by all cameras, frozen by
  default (its 512-d pooled features are cached for fast imitation learning; the last block
  can be unfrozen for fine-tuning).
* Fusion: per-camera features + a learned camera embedding, concatenated with speed / rpm /
  gear / steering and the route command.
* Heads (conditional imitation, Codevilla et al. 2018): one branch per route command, each
  predicting steer (tanh), throttle and brake (sigmoid) and gear logits (R N 1-5); plus a
  value head and a learned action std for RL fine-tuning.
"""

from __future__ import annotations

import numpy as np

N_COMMANDS = 6
N_GEARS = 7
FEAT = 512


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def device():
    torch, _ = _torch()
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_backbone(pretrained: bool = True):
    import torchvision

    torch, nn = _torch()
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = torchvision.models.resnet18(weights=weights)
    net.fc = nn.Identity()
    return net


class Encoder:
    """Images (B, N, H, W, 3) uint8 -> features (B, N, 512) with the frozen backbone."""

    def __init__(self, pretrained: bool = True, dev=None) -> None:
        torch, _ = _torch()
        self.dev = dev or device()
        self.net = make_backbone(pretrained).to(self.dev).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def __call__(self, images: np.ndarray):
        torch, _ = _torch()
        b, n, h, w, _ = images.shape
        x = torch.as_tensor(images.reshape(b * n, h, w, 3), device=self.dev).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        with torch.no_grad():
            f = self.net(x)
        return f.view(b, n, -1)


def make_head(n_cams: int, state_dim: int = 12, cmd_dim: int = N_COMMANDS + 1):
    torch, nn = _torch()

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.cam = nn.Parameter(torch.zeros(n_cams, FEAT))
            self.proj = nn.Sequential(nn.Linear(FEAT, 128), nn.ReLU())
            self.trunk = nn.Sequential(nn.Linear(n_cams * 128 + state_dim + cmd_dim, 512), nn.ReLU(),
                                       nn.Linear(512, 256), nn.ReLU())  # fmt: skip
            self.branches = nn.Linear(256, N_COMMANDS * (3 + N_GEARS))
            self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
            self.log_std = nn.Parameter(torch.full((3,), -1.6))

        def forward(self, feats, state, command):
            b = feats.shape[0]
            z = self.proj(feats + self.cam).reshape(b, -1)
            h = self.trunk(torch.cat([z, state, command], dim=1))
            out = self.branches(h).view(b, N_COMMANDS, 3 + N_GEARS)
            idx = command[:, :N_COMMANDS].argmax(1)
            sel = out[torch.arange(b), idx]
            raw = sel[:, :3]
            mean = torch.stack([torch.tanh(raw[:, 0]), torch.sigmoid(raw[:, 1]), torch.sigmoid(raw[:, 2])], dim=1)
            return mean, sel[:, 3:], self.value(h).squeeze(-1)

    return Head()


def bc_loss(mean, gear_logits, control, gear):
    torch, nn = _torch()
    l_steer = (mean[:, 0] - control[:, 0]).abs()
    l_pedal = (mean[:, 1:] - control[:, 1:]).abs().sum(1)
    weight = 1.0 + 3.0 * (control[:, 0].abs() > 0.08).float()  # turns are rarer than straight driving
    l_gear = nn.functional.cross_entropy(gear_logits, gear, reduction="none")
    loss = (weight * (2.0 * l_steer + l_pedal) + 0.2 * l_gear).mean()
    acc = (gear_logits.argmax(1) == gear).float().mean()
    return loss, {"steer_l1": float(l_steer.mean()), "pedal_l1": float(l_pedal.mean()), "gear_acc": float(acc)}


class PolicyRunner:
    """Callable policy for the env: obs -> action dict (deterministic means, argmax gear)."""

    def __init__(self, head, encoder: Encoder) -> None:
        self.head = head.to(encoder.dev).eval()
        self.enc = encoder

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> dict:
        torch, _ = _torch()
        dev = self.enc.dev
        feats = self.enc(obs["images"][None])
        st = torch.as_tensor(obs["state"][None], device=dev)
        cm = torch.as_tensor(obs["command"][None], device=dev)
        with torch.no_grad():
            mean, gear_logits, _ = self.head(feats, st, cm)
        m = mean[0].cpu().numpy()
        return {"control": m.astype(np.float32), "gear": int(gear_logits[0].argmax().item())}


def load_policy(path: str):
    """A callable driver from a checkpoint (DriveFormer or the MLP baseline)."""
    torch, _ = _torch()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("arch") == "transformer":
        from .transformer import SpatialEncoder, TransformerRunner, make_driveformer

        model = make_driveformer(ck["n_cams"])
        model.load_state_dict(ck["model"])
        return TransformerRunner(model, SpatialEncoder())
    head = make_head(ck["n_cams"])
    head.load_state_dict(ck.get("model") or ck["head"])
    return PolicyRunner(head, Encoder(pretrained=ck.get("pretrained", True)))
