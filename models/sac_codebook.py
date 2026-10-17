import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
import random
import numpy as np

CODEBOOK_PHASES = torch.tensor([2 * math.pi * i / 8 for i in range(8)], dtype=torch.float32)

class SACActor(nn.Module):
    def __init__(self, state_dim, num_elements=64, num_phases=8):
        super().__init__()
        self.num_elements = num_elements
        self.num_phases = num_phases
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, num_elements * num_phases)
        )

    def forward(self, x):
        batch = x.shape[0]
        return self.net(x).view(batch, self.num_elements, self.num_phases)

class SACCritic(nn.Module):
    def __init__(self, state_dim, num_elements=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + num_elements, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, u):
        return self.net(torch.cat([x, u], dim=1))

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10))
    y = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(y, dim=-1)
    if hard:
        index = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft
    return y_soft

def logits_to_phases(logits, temperature=1.0, hard=False):
    log_probs = F.log_softmax(logits, dim=-1)
    one_hot = gumbel_softmax_sample(logits, temperature, hard)
    phases = (one_hot * CODEBOOK_PHASES.to(one_hot.device)).sum(dim=-1)
    log_prob = (one_hot * log_probs).sum(dim=-1).sum(dim=-1)
    return phases, log_prob

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = collections.deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        # Convert tensors to CPU numpy arrays to save GPU memory and ensure clean batching
        if isinstance(state, torch.Tensor): state = state.cpu().numpy()
        if isinstance(action, torch.Tensor): action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor): next_state = next_state.cpu().numpy()
        
        self.buffer.append((state, action, float(reward), next_state, float(done)))
        
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Stack into batched tensors and move to the correct device
        state = torch.FloatTensor(np.array(states)).to(device)
        action = torch.FloatTensor(np.array(actions)).to( device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(np.array(next_states)).to(device)
        done = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        return state, action, reward, next_state, done
        
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, num_elements=64, num_phases=8, lr=3e-4, gamma=0.99, tau=0.005, batch_size=256, device="cuda"):
        self.device = torch.device(device)
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.num_elements, self.num_phases = num_elements, num_phases

        self.actor = SACActor(state_dim, num_elements, num_phases).to(self.device)
        self.q1 = SACCritic(state_dim, num_elements).to(self.device)
        self.q2 = SACCritic(state_dim, num_elements).to(self.device)
        self.q1_t = SACCritic(state_dim, num_elements).to(self.device)
        self.q2_t = SACCritic(state_dim, num_elements).to(self.device)

        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -num_elements * math.log(num_phases)
        self.log_alpha = torch.tensor(math.log(0.2), requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.buffer = ReplayBuffer(100000)

    @property
    def alpha(self): return self.log_alpha.exp().detach()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.actor(s)
            phases, _ = logits_to_phases(logits, temperature=0.1 if deterministic else 1.0, hard=True)
        return phases.squeeze(0).cpu().numpy()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return {}

        s, a, r, s_next, done = self.buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_logits = self.actor(s_next)
            next_phases, next_log_p = logits_to_phases(next_logits, hard=False)
            q_min_next = torch.min(self.q1_t(s_next, next_phases), self.q2_t(s_next, next_phases))
            target_q = r + (1 - done) * self.gamma * (q_min_next - self.alpha * next_log_p.unsqueeze(1))

        q1_loss = F.mse_loss(self.q1(s, a), target_q)
        q2_loss = F.mse_loss(self.q2(s, a), target_q)
        
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        logits = self.actor(s)
        phases, log_p = logits_to_phases(logits, hard=False)
        actor_loss = (self.alpha * log_p.unsqueeze(1) - torch.min(self.q1(s, phases), self.q2(s, phases))).mean()
        
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_p.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        for p, p_t in zip(self.q1.parameters(), self.q1_t.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)
        for p, p_t in zip(self.q2.parameters(), self.q2_t.parameters()):
            p_t.data.copy_(self.tau * p.data + (1 - self.tau) * p_t.data)

        return {"q_loss": (q1_loss+q2_loss).item(), "actor_loss": actor_loss.item(), "alpha": self.alpha.item()}