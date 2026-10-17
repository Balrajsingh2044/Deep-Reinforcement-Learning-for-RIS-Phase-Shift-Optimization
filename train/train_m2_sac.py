import sys
import os
import torch
import argparse

# Ensure root directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sac_codebook import SACAgent
from utils.wandb_logger import WandBLogger

# TODO: Replace this with Ayansh's actual environment import once pushed!
# from envs.ris_channel import RISChannelEnv

def make_env(seed=0):
    # Placeholder: Replace with actual env instantiation
    # env = RISChannelEnv(seed=seed)
    # return env
    raise NotImplementedError("Waiting for Ayansh's environment push!")

def train(num_episodes=200_000, num_phases=8, seed=0, device="cuda", ablation=False):
    env = make_env(seed=seed)
    state_dim = env.state_dim if hasattr(env, 'state_dim') else 128 # Fallback
    num_elements = env.num_elements if hasattr(env, 'num_elements') else 64

    agent = SACAgent(
        state_dim=state_dim, num_elements=num_elements, num_phases=num_phases, device=device
    )

    run_name = f"M2_{num_phases}phases_seed{seed}"
    logger = WandBLogger(
        project="ris_drl_sprint",
        config={"episodes": num_episodes, "phases": num_phases, "seed": seed},
        group="M2_Ablation" if ablation else "M2_Main",
        name=run_name
    )

    state = env.reset()
    episode_reward = 0.0

    for episode in range(1, num_episodes + 1):
        action = agent.select_action(state, deterministic=False)
        next_state, reward, done, info = env.step(action)

        agent.buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward

        metrics = agent.update()

        if done:
            logger.log({"episode": episode, "reward": episode_reward, **metrics}, step=episode)
            if episode % 1000 == 0:
                print(f"[Ep {episode}] Reward: {episode_reward:.2f} | Alpha: {agent.alpha:.3f}")
            
            state = env.reset()
            episode_reward = 0.0

    logger.finish()
    torch.save(agent.actor.state_dict(), f"m2_actor_{num_phases}phases_seed{seed}.pt")
    print(f"✅ Training complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200000)
    parser.add_argument("--phases", type=int, default=8, help="8=3-bit, 4=2-bit, 2=1-bit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    if args.ablation:
        for phases in [2, 4, 8]:
            print(f"\n{'='*40}\n Starting Ablation: {phases} phases \n{'='*40}")
            train(num_episodes=50000, num_phases=phases, seed=args.seed, ablation=True)
    else:
        train(num_episodes=args.episodes, num_phases=args.phases, seed=args.seed)