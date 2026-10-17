import torch
from ris_env import RISEnv

def riemannian_gradient_ascent(env, max_iter=50, lr=0.5):
    # Initialize environment
    env.reset()
    
    # Initialize phases randomly
    phases = torch.rand(env.M_ris) * 2 * torch.pi
    phases.requires_grad = True
    optimizer = torch.optim.Adam([phases], lr=lr)
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        # Map to [0, 2pi) for the environment
        mapped_phases = torch.sigmoid(phases) * 2 * torch.pi
        sum_rate, _ = env.step(mapped_phases)
        
        loss = -sum_rate  # We want to maximize rate, so minimize negative rate
        loss.backward()
        optimizer.step()
        
    final_phases = torch.sigmoid(phases.detach()) * 2 * torch.pi
    # Convert to float only at the very end for reporting
    return final_phases, sum_rate.item()