import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepUnfolding(nn.Module):
    """
    Deep Unfolding for real-valued phase angle prediction.
    Input: 1056-dim channel state (real)
    Output: 64-dim phase angles (real, in radians)
    """
    def __init__(self, state_dim=1056, action_dim=64, num_iterations=15):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Feature extractor: maps 1056-dim state to 64-dim initial phases
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Learnable step sizes for each unrolled iteration
        self.step_sizes = nn.Parameter(torch.ones(num_iterations) * 0.1)
        
        # Iteration refinement network
        self.refine_net = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, states, ao_target=None):
        """
        states: (batch, 1056) real-valued channel state
        ao_target: (batch, 64) real-valued optimal phase angles (radians)
        """
        phases = self.feature_net(states)
        
        for i in range(self.num_iterations):
            refinement = self.refine_net(phases)
            phases = phases + self.step_sizes[i] * refinement
        
        if ao_target is not None:
            loss = F.mse_loss(phases, ao_target)
            return phases, loss
        
        return phases, None