import os, sys, time, argparse, math, random, signal, csv
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from drive.rl.env import DrivingEnv
from drive.rl.models import Actor, Critic
from drive.rl.replay import ReplayBuffer


# ============================================================
# 环境检测工具
# ============================================================
def in_jupyter() -> bool:
    """检测当前是否运行在 Jupyter / IPython 环境"""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


def select_device(pref: str = "auto") -> torch.device:
    pref = pref.lower()
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


# ============================================================
# 评估函数
# ============================================================
def evaluate_policy(env: DrivingEnv, actor: nn.Module, device: torch.device, n_episodes: int = 5):
    actor.eval()
    rets = []
    succ = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            o = env.reset()
            ep_ret, done = 0.0, False
            while not done:
                ot = to_tensor(o, device).unsqueeze(0)
                a = actor(ot)[0].cpu().numpy().astype(np.float32)
                o, r, done, info = env.step(a)
                ep_ret += r
            rets.append(ep_ret)
            succ += 1 if info.get("success") else 0
    actor.train()
    return float(np.mean(rets)), succ / n_episodes


# ============================================================
# 保存函数
# ============================================================
def save_metrics_csv(metrics: Dict[str, List], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "metrics.csv")
    fields = list(metrics.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in zip(*metrics.values()):
            writer.writerow(row)
    print(f"\n[CSV] Metrics saved to: {csv_path}")


# ============================================================
# 训练函数
# ============================================================
def ddpg_train(
    svg_path: str,
    out_dir: str,
    steps: int = 600_000,
    num_envs: int = 8,
    start_steps: int = 20_000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    lr: float = 5e-5,
    seed: int = 0,
    device: str = "auto",
    log_every: int = 1_000,
    save_every: int = 5_000,
    ema_alpha: float = 0.03,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device(device)
    print(f"[Device] Using {device}, ParallelEnv ×{num_envs}")

    # 加载环境
    with open(svg_path, "r", encoding="utf-8") as f:
        svg_text = f.read()
    test_env = DrivingEnv(svg_text, seed=seed)
    obs_dim, act_dim = test_env.obs_dim, test_env.act_dim

    # 并行环境
    envs = ParallelEnv(svg_text, num_envs=num_envs, base_seed=seed)
    obs = envs.reset()

    # 模型与优化器
    actor, critic = Actor(obs_dim, act_dim).to(device), Critic(obs_dim, act_dim).to(device)
    target_actor, target_critic = Actor(obs_dim, act_dim).to(device), Critic(obs_dim, act_dim).to(device)
    actor_opt, critic_opt = optim.Adam(actor.parameters(), lr=lr), optim.Adam(critic.parameters(), lr=lr)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    # 尝试从之前的 checkpoint 加载
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    global_step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        actor_opt.load_state_dict(ckpt["actor_opt"])
        critic_opt.load_state_dict(ckpt["critic_opt"])
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())
        global_step = ckpt.get("step", 0)
        print(f"[Load] Resumed from {ckpt_path} (step={global_step})")
    else:
        print("[Init] Starting new training run (no checkpoint found)")

    buf = ReplayBuffer(obs_dim, act_dim, size=300_000)

    # 指标与绘图
    metrics = {"step": [], "ep_return": [], "actor_loss": [], "critic_loss": [], "sr_ema": []}
    success_ema = 0.0
    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ln_ret, = axs[0].plot([], [], label="Episode Return")
    ln_aloss, = axs[1].plot([], [], label="Actor Loss")
    ln_closs, = axs[2].plot([], [], label="Critic Loss")
    for ax in axs:
        ax.legend(); ax.grid(True)
    axs[0].set_title("Return"); axs[1].set_title("Actor Loss"); axs[2].set_title("Critic Loss")
    fig.tight_layout()

    interrupted = {"flag": False}
    def handle_sigint(sig, frame):
        interrupted["flag"] = True
        print("\n[Signal] Ctrl+C detected, stopping gracefully…")
    signal.signal(signal.SIGINT, handle_sigint)

    # 主循环
    ep_returns = np.zeros(num_envs, dtype=np.float32)
    last_plot_time = time.time()

    while global_step < steps:
        # 动作选择
        if global_step < start_steps:
            actions = np.random.uniform(-1, 1, size=(num_envs, act_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                actions = actor(to_tensor(obs, device)).cpu().numpy().astype(np.float32)
            noise = np.random.normal(0, 0.2, size=actions.shape)
            actions = np.clip(actions + noise, -1, 1).astype(np.float32)

        # 并行环境步进
        next_obs, rewards, dones, infos = envs.step(actions)
        for i in range(num_envs):
            buf.add(obs[i], actions[i], rewards[i],
                    next_obs[i] if not dones[i] else obs[i],
                    float(dones[i]))
            ep_returns[i] += rewards[i]

            if dones[i]:
                succ = 1.0 if infos[i].get("success") else 0.0
                success_ema = (1 - ema_alpha) * success_ema + ema_alpha * succ
                metrics["step"].append(global_step)
                metrics["ep_return"].append(ep_returns[i])
                metrics["actor_loss"].append(0.0)
                metrics["critic_loss"].append(0.0)
                metrics["sr_ema"].append(success_ema)
                sys.stdout.write(f"\r[Env {i}] step={global_step:7d} ret={ep_returns[i]:8.2f} sr={success_ema:5.2f}\n")
                ep_returns[i] = 0.0

        obs = next_obs
        global_step += num_envs

        # 网络更新
        if buf.size >= start_steps:
            obs_b, act_b, rew_b, obs2_b, done_b = buf.sample(batch_size)
            obs_t, act_t = to_tensor(obs_b, device), to_tensor(act_b, device)
            rew_t, obs2_t = to_tensor(rew_b, device).unsqueeze(-1), to_tensor(obs2_b, device)
            done_t = to_tensor(done_b, device).unsqueeze(-1)

            with torch.no_grad():
                target_a = target_actor(obs2_t)
                target_q = target_critic(obs2_t, target_a)
                y = rew_t + (1 - done_t) * gamma * target_q

            q = critic(obs_t, act_t)
            critic_loss = ((q - y) ** 2).mean()
            critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

            actor_loss = -critic(obs_t, actor(obs_t)).mean()
            actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

            # Polyak 平滑
            with torch.no_grad():
                for p, tp in zip(actor.parameters(), target_actor.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)

            metrics["actor_loss"][-1] = actor_loss.item()
            metrics["critic_loss"][-1] = critic_loss.item()

        # 定期刷新曲线（每 2 秒一次）
        if time.time() - last_plot_time > 2 and len(metrics["step"]) > 3:
            last_plot_time = time.time()
            ln_ret.set_data(metrics["step"], smooth(metrics["ep_return"], 10))
            ln_aloss.set_data(metrics["step"], metrics["actor_loss"])
            ln_closs.set_data(metrics["step"], metrics["critic_loss"])
            for ax in axs:
                ax.relim(); ax.autoscale_view()
            plt.pause(0.001)

        # 定期保存
        if global_step % save_every < num_envs:
            os.makedirs(out_dir, exist_ok=True)
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_opt": actor_opt.state_dict(),
                "critic_opt": critic_opt.state_dict(),
                "step": global_step
            }, ckpt_path)
            print(f"\n[Save] checkpoint at step={global_step}")

        if interrupted["flag"]:
            print("\n[Stop] Interrupted, exiting…")
            break

    envs.close()
    save_metrics_csv(metrics, out_dir)
    plt.ioff(); plt.show()
    print("[Done] Training finished.")



# ============================================================
# 命令行入口
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--svg", required=True, help="Path to map.svg")
    p.add_argument("--out", default="checkpoints", help="Output directory")
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    ddpg_train(svg_path=args.svg, out_dir=args.out, steps=args.steps, seed=args.seed)


if __name__ == "__main__":
    main()
