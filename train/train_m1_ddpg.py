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
    env = RISEnv() 
    
    # IMPORTANT: Ensure state_dim matches Ayansh's env output shape. 
    # If his env returns a different shape (e.g., 256), change 128 to that number.
    state_dim = 128 
    action_dim = 64 # M=64 RIS elements
    
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
        # 1. Reset the real environment
        state = env.reset() 
        noise.reset()
        ep_reward = 0
        
        # Run for a maximum of 50 steps per episode (adjust if Ayansh's env has a different limit)
        max_steps = 50 
        for step in range(max_steps):
            # 2. Get action from the Actor network
            state_tensor = torch.FloatTensor(state).to(device)
            action = actor(state_tensor).detach().cpu().numpy()
            
            # 3. Add exploration noise and clip to valid bounds
            action_noise = np.clip(action + noise.sample(), -1.0, 1.0)
            
            # 4. Step the real environment (returns 4 values)
            next_state, reward, done, _ = env.step(action_noise)
            
            # 5. Store transition in replay buffer
            buffer.push(state, action_noise, reward, next_state, done)
            state = next_state
            ep_reward += reward
            
            # 6. Train the networks if we have enough samples in the buffer
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
                
                # Soft Update for target networks
                soft_update(actor_target, actor, tau)
                soft_update(critic_target, critic, tau)
                
            # If the environment signals the episode is over, break early
            if done:
                break
                
        # Log metrics to Weights & Biases
        wandb.log({"M1_Episode_Reward": ep_reward, "M1_Critic_Loss": critic_loss.item()})
        
        # Save checkpoint every 1000 episodes
        if ep % 1000 == 0:
            torch.save(actor.state_dict(), f"checkpoints/m1_actor_ep{ep}.pth")
            
    wandb.finish()

if __name__ == "__main__":
    train_m1()