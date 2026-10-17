import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepUnfolding(nn.Module):
    """Unified Deep Unfolding for real-valued phase angle prediction."""
    def __init__(self, state_dim=1056, action_dim=64, num_iterations=15):
        super().__init__()
        self.num_iterations = num_iterations
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.step_sizes = nn.Parameter(torch.ones(num_iterations) * 0.1)
        self.refine_net = nn.Sequential(
            nn.Linear(action_dim, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, states, ao_target=None):
        phases = self.feature_net(states)
        for i in range(self.num_iterations):
            refinement = self.refine_net(phases)
            phases = phases + self.step_sizes[i] * refinement
        
        if ao_target is not None:
            loss = F.mse_loss(phases, ao_target)
            return phases, loss
        return phases, None