import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 初始化工具 ----------
def orthogonal_init(layer, scale=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=scale)
        nn.init.constant_(layer.bias, 0.0)
    return layer

# ---------- CNN Encoder ----------
class MaskEncoder(nn.Module):
    def __init__(self, patch_size=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),          # 16x16x16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),         # 8x8x32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),         # 4x4x64
            nn.Flatten(),
            nn.Linear(4 * 4 * 64, 256), nn.ReLU(),
        )
        self.output_dim = 256

    def forward(self, x_flat):
        # x_flat shape: [B, 1024 + 5]
        patch = x_flat[:, :1024].reshape(-1, 1, 32, 32)
        return self.conv(patch)

# ---------- Actor ----------
class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.encoder = MaskEncoder()
        self.extra_dim = obs_dim - 1024  # speed + heading + displacement
        self.extra_fc = nn.Sequential(
            nn.LayerNorm(self.extra_dim),
            nn.Linear(self.extra_dim, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim + 64),
            nn.Linear(self.encoder.output_dim + 64, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
        self.apply(lambda m: orthogonal_init(m, scale=1.0))

    def forward(self, obs):
        patch_feat = self.encoder(obs)
        extra_feat = self.extra_fc(obs[:, -self.extra_dim:])
        x = torch.cat([patch_feat, extra_feat], dim=-1)
        return self.trunk(x)

# ---------- Critic ----------
class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.encoder = MaskEncoder()
        self.extra_dim = obs_dim - 1024

        # 分支结构：obs 编码 + action 结合
        self.obs_branch = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim + self.extra_dim),
            nn.Linear(self.encoder.output_dim + self.extra_dim, hidden), nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # 初始化更稳
        self.apply(lambda m: orthogonal_init(m, scale=1.0))
        nn.init.orthogonal_(self.q_head[-1].weight, gain=0.1)
        nn.init.constant_(self.q_head[-1].bias, 0.0)

    def forward(self, obs, act):
        patch_feat = self.encoder(obs)
        extra_feat = obs[:, -self.extra_dim:]
        x = torch.cat([patch_feat, extra_feat], dim=-1)
        x = self.obs_branch(x)
        xa = torch.cat([x, act], dim=-1)
        return self.q_head(xa)
