import torch
import torch.nn as nn
import torch.optim as optim
from models.ddpg import Actor, Critic, ReplayBuffer, OUNoise, soft_update
from utils.wandb_logger import init_wandb
# from env.ris_env import RISEnv # Uncomment when Ayansh provides this

def train_m1(episodes=200000, batch_size=256, gamma=0.99, tau=0.005, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb = init_wandb(project="RIS-M1-DDPG", config={"episodes": episodes, "lr": lr})
    
    # env = RISEnv() # Initialize Ayansh's environment
    state_dim = 128 # Replace with actual state dim from env
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
        # state = env.reset() # Replace with actual env reset
        state = torch.randn(state_dim).numpy() # Dummy state for syntax check
        noise.reset()
        ep_reward = 0
        
        for step in range(50): # Max steps per episode
            # action = actor(torch.FloatTensor(state).to(device)).detach().cpu().numpy()
            action = np.random.uniform(-1, 1, action_dim) # Dummy action
            action_noise = np.clip(action + noise.sample(), -1.0, 1.0)
            
            # next_state, reward, done, _ = env.step(action_noise) # Replace
            next_state, reward, done = np.random.randn(state_dim), 1.0, False # Dummy
            
            buffer.push(state, action_noise, reward, next_state, done)
            state = next_state
            ep_reward += reward
            
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
                
        wandb.log({"M1_Episode_Reward": ep_reward, "M1_Critic_Loss": critic_loss.item()})
        if ep % 1000 == 0:
            torch.save(actor.state_dict(), f"checkpoints/m1_actor_ep{ep}.pth")
    wandb.finish()

if __name__ == "__main__":
    train_m1()