import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.ddpg import Actor, Critic, ReplayBuffer, OUNoise, soft_update
from utils.wandb_logger import init_wandb
from env.ris_env import RISEnv 

def train_m1(episodes=200000, batch_size=256, gamma=0.99, tau=0.005, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb = init_wandb(project="RIS-M1-DDPG", config={"episodes": episodes, "lr": lr})
    
    # Initialize Ayansh's real environment
    env = RISEnv(N_bs=4, M_ris=16, K_users=2, device=device)
    
    # CORRECTED DIMENSIONS based on Ayansh's env
    state_dim = 208   # (2*4 + 2*16 + 16*4) * 2 (real+imag)
    action_dim = 16   # M_ris = 16 RIS elements
    
    actor = Actor(state_dim, action_dim).to(device)
    actor_target = Actor(state_dim, action_dim).to(device)
    actor_target.load_state_dict(actor.state_dict())
    
    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(critic.parameters(), lr=lr)
    buffer = ReplayBuffer(capacity=100000)
    noise = OUNoise(action_dim)
    
    for ep in range(episodes):
        # 1. Reset environment (returns PyTorch tensor on device)
        state = env.reset().to(device)
        noise.reset()
        ep_reward = 0
        
        # Ayansh's env doesn't have multi-step episodes
        # Each "episode" is one channel realization with one action
        # We simulate multiple steps by resetting the channel each time
        max_steps = 50
        for step in range(max_steps):
            # 2. Get action from Actor (returns tensor in [-1, 1])
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
            action = actor(state_tensor).squeeze(0)
            
            # 3. Add exploration noise
            action_np = action.detach().cpu().numpy()
            action_noisy = np.clip(action_np + noise.sample(), -1.0, 1.0)
            action_tensor = torch.FloatTensor(action_noisy).to(device)
            
            # 4. Step environment (returns reward and h_eff, NOT next_state)
            reward, h_eff = env.step(action_tensor)
            
            # 5. For RL, we need a "next_state" - use the same state since channel doesn't change
            next_state = state.clone()
            done = (step == max_steps - 1)
            
            # 6. Store in buffer (convert to numpy for buffer)
            buffer.push(
                state.detach().cpu().numpy(),
                action_noisy,
                reward.item() if isinstance(reward, torch.Tensor) else reward,
                next_state.detach().cpu().numpy(),
                float(done)
            )
            
            state = next_state
            ep_reward += reward.item() if isinstance(reward, torch.Tensor) else reward
            
            # 7. Train networks
            if len(buffer) > batch_size:
                s, a, r, ns, d = buffer.sample(batch_size)
                s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)
                
                # Critic Update
                with torch.no_grad():
                    target_q = r + (1 - d) * gamma * critic_target(ns, actor_target(ns))
                critic_loss = nn.MSELoss()(critic(s, a), target_q)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()
                
                # Actor Update
                actor_loss = -critic(s, actor(s)).mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()
                
                # Soft Update
                soft_update(actor_target, actor, tau)
                soft_update(critic_target, critic, tau)
                
        # Log metrics
        wandb.log({"M1_Episode_Reward": ep_reward, "M1_Critic_Loss": critic_loss.item()})
        
        # Save checkpoint
        if ep % 1000 == 0:
            torch.save(actor.state_dict(), f"checkpoints/m1_actor_ep{ep}.pth")
            
    wandb.finish()

if __name__ == "__main__":
    train_m1()