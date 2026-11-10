import os, sys, time, argparse, math, random, signal, csv
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe

from drive.rl.env import DrivingEnv
from drive.rl.models import Actor, Critic
from drive.rl.replay import ReplayBuffer
import pandas as pd

def smooth(y, window=42):
    return pd.Series(y).rolling(window, min_periods=1).mean().values


# ============================================================
# 环境并行封装
# ============================================================
def worker(remote, parent_remote, svg_text, seed):
    parent_remote.close()
    env = DrivingEnv(svg_text, seed=seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, rew, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, rew, done, info))
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "close":
                remote.close()
                break
    except KeyboardInterrupt:
        pass


class ParallelEnv:
    def __init__(self, svg_text, num_envs=8, base_seed=0):
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            seed = base_seed + i * 97
            p = Process(target=worker, args=(work_remote, remote, svg_text, seed))
            p.daemon = True
            p.start()
            work_remote.close()
            self.ps.append(p)

    def step(self, actions):
        for remote, a in zip(self.remotes, actions):
            remote.send(("step", a))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()


# ============================================================
# 通用工具函数
# ============================================================
def select_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)

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
# 主训练函数（多环境版）
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
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_opt, critic_opt = optim.Adam(actor.parameters(), lr=lr), optim.Adam(critic.parameters(), lr=lr)
    buf = ReplayBuffer(obs_dim, act_dim, size=300_000)
    
    # ============================================================
    # 可选：加载已有模型（如果存在）
    # ============================================================
    actor_path = os.path.join(out_dir, "actor.pt")
    critic_path = os.path.join(out_dir, "critic.pt")

    if os.path.exists(actor_path) and os.path.exists(critic_path):
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())
        print(f"[Load] Loaded pretrained models from {out_dir}")


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
    global_step = 0
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
            #ln_ret.set_data(metrics["step"], metrics["ep_return"])
            ln_ret.set_data(metrics["step"], smooth(metrics["ep_return"], 10))
            ln_aloss.set_data(metrics["step"], metrics["actor_loss"])
            ln_closs.set_data(metrics["step"], metrics["critic_loss"])
            for ax in axs:
                ax.relim(); ax.autoscale_view()
            plt.pause(0.001)

        # 定期保存
        if global_step % save_every < num_envs:
            os.makedirs(out_dir, exist_ok=True)
            torch.save(actor.state_dict(), os.path.join(out_dir, "actor.pt"))
            torch.save(critic.state_dict(), os.path.join(out_dir, "critic.pt"))
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
    p.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    args = p.parse_args()
    ddpg_train(svg_path=args.svg, out_dir=args.out, steps=args.steps,
               seed=args.seed, num_envs=args.num_envs)

if __name__ == "__main__":
    main()
