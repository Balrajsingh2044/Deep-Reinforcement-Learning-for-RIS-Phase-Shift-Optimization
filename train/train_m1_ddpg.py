# --- PROJECT PATH FIX ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.ddpg import Actor, Critic, ReplayBuffer, OUNoise, soft_update
from utils.wandb_logger import init_wandb
from ris_env import RISEnv

REWARD_SCALE = 10.0
STATE_DIM = 1056
ACTION_DIM = 64


def evaluate(actor, env, device, n_eval=200):
    """Deterministic policy evaluation on FRESH channels."""
    actor.eval()
    rates = []
    with torch.no_grad():
        for _ in range(n_eval):
            s = env.reset().to(device)
            a = actor(s.unsqueeze(0)).squeeze(0)
            r, _ = env.step(a)
            rates.append(r.item())
    actor.train()
    return float(np.mean(rates)), float(np.std(rates))


def train_m1(total_steps=200000,
             batch_size=128,
             gamma=0.99,
             tau=0.005,
             lr_actor=1e-4,
             lr_critic=3e-4,
             warmup=1000,
             eval_every=500):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    wandb = init_wandb(project="RIS-M1-DDPG", config={
        "total_steps": total_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "reward_scale": REWARD_SCALE,
        "formulation": "one-shot contextual bandit (done=1)",
    })

    env = RISEnv(N_bs=4, M_ris=64, K_users=4, device=device)
    
    actor = Actor(STATE_DIM, ACTION_DIM).to(device)
    actor_target = Actor(STATE_DIM, ACTION_DIM).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(STATE_DIM, ACTION_DIM).to(device)
    critic_target = Critic(STATE_DIM, ACTION_DIM).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_opt = optim.Adam(critic.parameters(), lr=lr_critic)

    buffer = ReplayBuffer(capacity=100_000)
    noise = OUNoise(ACTION_DIM, theta=0.15, sigma=0.15)

    best_reward = -float("inf")
    critic_loss = torch.tensor(0.0)

    for step in range(1, total_steps + 1):

        # --- ONE decision = ONE fresh channel realization ---
        state = env.reset().to(device)
        noise.reset()

        # Decay exploration over the first half of training
        noise.sigma = 0.15 * max(0.05, 1.0 - step / (0.5 * total_steps))

        with torch.no_grad():
            action = actor(state.unsqueeze(0)).squeeze(0)

        action_np = np.clip(action.detach().cpu().numpy() + noise.sample(), -1.0, 1.0)
        action_t = torch.FloatTensor(action_np).to(device)

        reward, _ = env.step(action_t)
        r_val = reward.item() / REWARD_SCALE

        # ============================================================
        # CRITICAL FIX: this is a TERMINAL decision.
        # No bootstrapping -> Q_target == r -> no 100x inflation.
        # ============================================================
        done = 1.0

        s_np = state.detach().cpu().numpy()
        buffer.push(s_np, action_np, r_val, s_np, done)

        # --- Learning ---
        if step > warmup and len(buffer) >= batch_size:
            s, a, r, ns, d = buffer.sample(batch_size)
            s, a, r, ns, d = (x.to(device) for x in (s, a, r, ns, d))

            with torch.no_grad():
                target_q = r + (1.0 - d) * gamma * critic_target(ns, actor_target(ns))

            current_q = critic(s, a)
            critic_loss = nn.functional.smooth_l1_loss(current_q, target_q)

            if not torch.isfinite(critic_loss):
                print(f"[WARN] Non-finite critic loss at step {step}, skipping.")
                continue

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_opt.step()

            # DIAGNOSTIC: proves the reward scale and Q scale are sane
            if step == warmup + 1:
                print(f"[DIAG] r mean      = {r.mean().item():.4f}")
                print(f"[DIAG] Q mean      = {current_q.mean().item():.4f}")
                print(f"[DIAG] target mean = {target_q.mean().item():.4f}")
                print(f"[DIAG] These three should be CLOSE to each other.")

            # Delayed policy update (every 2 critic updates)
            if step % 2 == 0:
                actor_loss = -critic(s, actor(s)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
                actor_opt.step()

                soft_update(actor_target, actor, tau)
                soft_update(critic_target, critic, tau)

        # --- Periodic honest evaluation ---
        if step % eval_every == 0:
            mean_r, std_r = evaluate(actor, env, device, n_eval=200)
            wandb.log({
                "Step": step,
                "Eval_SumRate": mean_r,
                "Eval_Std": std_r,
                "Critic_Loss": float(critic_loss.item()),
                "Exploration_Sigma": noise.sigma,
            })
            print(f"Step {step:6d} | Eval SumRate {mean_r:6.3f} ± {std_r:5.3f} "
                  f"| CriticLoss {critic_loss.item():.4f}")

            if mean_r > best_reward:
                best_reward = mean_r
                torch.save(actor, "checkpoints/m1_actor_best.pth")
                torch.save(actor, "models/m1_ddpg_actor.pth")   
                print(f"  ↳ New best saved ({best_reward:.3f})")

    wandb.finish()
    print(f"\nTraining complete. Best deterministic sum rate: {best_reward:.3f}")


if __name__ == "__main__":
    train_m1()