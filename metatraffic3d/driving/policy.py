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


def make_head(n_cams: int, state_dim: int = 12, cmd_dim: int = N_COMMANDS + 1, feat_dim: int = FEAT):
    torch, nn = _torch()

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.cam = nn.Parameter(torch.zeros(n_cams, feat_dim))
            self.proj = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU())
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
            self.last_pedal_logits = raw[:, 1:3]  # for a loss that never saturates (see bc_loss)
            mean = torch.stack([torch.tanh(raw[:, 0]), torch.sigmoid(raw[:, 1]), torch.sigmoid(raw[:, 2])], dim=1)
            return mean, sel[:, 3:], self.value(h).squeeze(-1)

    return Head()


def keyframe_weights(state, control, gear):
    """Extra weight for the rare moments where the expert *changes* what it is doing.

    Copying the current gear or the current speed trend is right on almost every step, so a
    cloned policy learns to copy and never shifts or brakes on its own (the "copycat" problem of
    behaviour cloning from histories). Up-weighting gear changes, braking while moving and
    pulling away from a stop makes those decisions count. ``state`` is the current (B, 12) state.
    """
    torch, _ = _torch()
    cur_gear = state[:, 2:2 + N_GEARS].argmax(1)
    speed = state[:, 0] * 15.0
    w_gear = 1.0 + 9.0 * (gear != cur_gear).float()
    braking = (control[:, 2] > 0.1) & (speed > 1.0)
    pulling_away = (control[:, 1] > 0.05) & (speed < 0.5)
    w_ctl = 1.0 + 4.0 * (braking | pulling_away).float()
    return w_ctl, w_gear


def bc_loss(mean, gear_logits, control, gear, w_ctl=None, w_gear=None, pedal_logits=None):
    """Imitation loss. With ``pedal_logits`` the throttle and brake use binary cross-entropy on
    the logits: an L1 loss on sigmoid outputs drives a mostly-zero target (the brake) deep into
    the flat end of the sigmoid, where its gradient vanishes and the output can never recover."""
    torch, nn = _torch()
    l_steer = (mean[:, 0] - control[:, 0]).abs()
    l_pedal = (mean[:, 1:] - control[:, 1:]).abs().sum(1)
    if pedal_logits is not None:
        l_fit = nn.functional.binary_cross_entropy_with_logits(pedal_logits, control[:, 1:], reduction="none").sum(1)
    else:
        l_fit = l_pedal
    weight = 1.0 + 3.0 * (control[:, 0].abs() > 0.08).float()  # turns are rarer than straight driving
    if w_ctl is not None:
        weight = weight * w_ctl
    l_gear = nn.functional.cross_entropy(gear_logits, gear, reduction="none")
    if w_gear is not None:
        l_gear = l_gear * w_gear
    loss = (weight * (2.0 * l_steer + l_fit) + 0.2 * l_gear).mean()
    acc = (gear_logits.argmax(1) == gear).float().mean()
    return loss, {"steer_l1": float(l_steer.mean()), "pedal_l1": float(l_pedal.mean()), "gear_acc": float(acc)}


def pick_one_pedal(control: np.ndarray) -> np.ndarray:
    """Keep the larger pedal and release the other, as the expert does (it never presses both).

    Throttle and brake are separate outputs, each trained towards the *average* label. Where the
    expert sometimes waits and sometimes pulls away (standing at a stop line, say) both come out
    half-pressed, and a little brake holds the car against a little throttle - it never moves
    off. Picking one pedal turns that average back into a decision.
    """
    c = np.array(control, np.float32)
    if c[1] >= c[2]:
        c[2] = 0.0
    else:
        c[1] = 0.0
    return c


class PolicyRunner:
    """Callable policy for the env: obs -> action dict (deterministic means, argmax gear)."""

    def __init__(self, head, encoder: Encoder, pca=None, jpeg: bool = True, one_pedal: bool = True) -> None:
        self.head = head.to(encoder.dev).eval()
        self.enc = encoder
        self.pca = pca
        self.jpeg = jpeg  # training frames were JPEG-compressed; match that at run time
        self.one_pedal = one_pedal

    def reset(self) -> None:
        pass

    def __call__(self, obs: dict) -> dict:
        torch, _ = _torch()
        dev = self.enc.dev
        from .jpeg import roundtrip

        feats = self.enc((roundtrip(obs["images"]) if self.jpeg else obs["images"])[None])
        if self.pca is not None:
            feats = self.pca(feats)
        st = torch.as_tensor(obs["state"][None], device=dev)
        cm = torch.as_tensor(obs["command"][None], device=dev)
        with torch.no_grad():
            mean, gear_logits, _ = self.head(feats, st, cm)
        m = mean[0].cpu().numpy().astype(np.float32)
        return {"control": pick_one_pedal(m) if self.one_pedal else m, "gear": int(gear_logits[0].argmax().item())}


def load_policy(path: str, one_pedal: bool = True):
    """A callable driver from a checkpoint (DriveFormer or the MLP baseline); see ``pick_one_pedal``."""
    torch, _ = _torch()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    from .transformer import PCA, SpatialEncoder, TransformerRunner, make_driveformer

    pca = PCA(ck["pca_mean"], ck["pca_comp"]) if "pca_comp" in ck else None
    feat_dim = ck.get("feat_dim", FEAT)
    if ck.get("arch") == "transformer":
        model = make_driveformer(ck["n_cams"], feat_dim=feat_dim)
        model.load_state_dict(ck["model"])
        return TransformerRunner(model, SpatialEncoder(), pca, one_pedal=one_pedal)
    head = make_head(ck["n_cams"], feat_dim=feat_dim)
    head.load_state_dict(ck.get("model") or ck["head"])
    return PolicyRunner(head, Encoder(pretrained=ck.get("pretrained", True)), pca, one_pedal=one_pedal)
