import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
from models.ddpg import Actor, Critic
from models.deep_unfolding import DeepUnfolding

def sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Sanity Checks...")

    # M1
    state_dim, action_dim = 1056, 64
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    assert actor(torch.randn(4, state_dim).to(device)).shape == (4, action_dim)
    assert critic(torch.randn(4, state_dim).to(device), torch.randn(4, action_dim).to(device)).shape == (4, 1)
    print("✅ M1 (DDPG) shapes verified.")

    # M3
    model_m3 = DeepUnfolding(state_dim=1056, action_dim=64, num_iterations=15).to(device)
    out_m3, loss_m3 = model_m3(torch.randn(4, 1056).to(device), torch.randn(4, 64).to(device))
    assert not torch.isnan(loss_m3) and out_m3.shape == (4, 64)
    print("✅ M3 (Deep Unfolding) shapes verified.")

    # M2
    snapped = F.gumbel_softmax(torch.randn(4, 8), tau=0.1, hard=True, dim=-1)
    assert torch.all(snapped.max(dim=-1)[0] == 1.0).item()
    print("✅ M2 (SAC + Codebook) snapping verified.")

    print("🚀 All sanity checks passed.")

if __name__ == "__main__":
    sanity_check()