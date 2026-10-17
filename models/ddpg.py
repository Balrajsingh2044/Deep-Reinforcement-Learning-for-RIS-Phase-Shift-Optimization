import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import random

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu, self.theta, self.sigma = mu, theta, sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self): self.state = np.ones(self.action_dim) * self.mu
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = collections.deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state), torch.FloatTensor(action),
                torch.FloatTensor(reward).unsqueeze(1), torch.FloatTensor(next_state),
                torch.FloatTensor(done).unsqueeze(1))
    def __len__(self): return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim=64, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 400), nn.ReLU(),
                                 nn.Linear(400, 300), nn.ReLU(),
                                 nn.Linear(300, action_dim))
        self.max_action = max_action
    def forward(self, x): return self.max_action * torch.tanh(self.net(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, 400), nn.ReLU(),
                                 nn.Linear(400, 300), nn.ReLU(),
                                 nn.Linear(300, 1))
    def forward(self, x, u): return self.net(torch.cat([x, u], dim=1))

def soft_update(target, source, tau=0.005):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)