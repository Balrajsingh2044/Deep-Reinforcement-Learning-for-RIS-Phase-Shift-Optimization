import numpy as np
import torch

class MockRISEnv:
    def __init__(self, state_dim=128, action_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def reset(self):
        # Returns a normalized state vector (cascaded channel)
        return np.random.randn(self.state_dim).astype(np.float32)
        
    def step(self, action):
        # action shape: (64,)
        # Reward is the Achievable Sum Rate (must be positive, e.g., 30-50 Mbps)
        reward = np.random.uniform(35.0, 48.0) 
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        done = False
        return next_state, reward, done, {}

    def calculate_sum_rate(self, states_tensor, actions_tensor):
        """Batched sum rate calculation for the Evaluation script."""
        batch_size = states_tensor.shape[0]
        # Returns fake sum rates between 35 and 50 Mbps
        return torch.rand(batch_size) * 15.0 + 35.0 