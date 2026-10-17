import torch

class RISEnv:
    def __init__(self, N_bs=4, M_ris=64, K_users=4, device='cpu'):
        self.N_bs = N_bs
        self.M_ris = M_ris
        self.K_users = K_users
        self.device = torch.device(device)
        self.reset()
        
    def reset(self):
        self.h_direct = torch.randn(self.K_users, self.N_bs, device=self.device, dtype=torch.complex64)
        self.G = torch.randn(self.K_users, self.M_ris, device=self.device, dtype=torch.complex64)
        self.F = torch.randn(self.M_ris, self.N_bs, device=self.device, dtype=torch.complex64)
        
        state = torch.cat([self.h_direct.real.flatten(), self.h_direct.imag.flatten(),
                           self.G.real.flatten(), self.G.imag.flatten(),
                           self.F.real.flatten(), self.F.imag.flatten()])
        return state

    def step(self, raw_actions):
        if not isinstance(raw_actions, torch.Tensor):
            raw_actions = torch.tensor(raw_actions, dtype=torch.float32, device=self.device)
        else:
            raw_actions = raw_actions.to(self.device)
            
        phases = torch.sigmoid(raw_actions) * 2 * torch.pi
        theta = torch.exp(1j * phases)
        Theta = torch.diag(theta)
        
        h_eff = self.h_direct + self.G @ Theta @ self.F 
        W = h_eff.conj().T 
        W = W / torch.norm(W, dim=0, keepdim=True) 
        
        H_W = h_eff @ W 
        signal_power = torch.abs(torch.diag(H_W))**2
        total_power = torch.sum(torch.abs(H_W)**2, dim=1)
        interference_noise = total_power - signal_power + 1.0 
        
        sinr = signal_power / interference_noise
        sum_rate = torch.sum(torch.log2(1 + sinr)).real
        
        return sum_rate, h_eff


class RISEnvWrapper:
    """Wraps RISEnv to act like a standard 1-step Gym environment."""
    def __init__(self, N_bs=4, M_ris=64, K_users=4, device='cuda'):
        self.env = RISEnv(N_bs=N_bs, M_ris=M_ris, K_users=K_users, device=device)
        self.device = torch.device(device)
        self.M_ris = M_ris
        self.state_dim = 2 * (K_users * N_bs + K_users * M_ris + M_ris * N_bs)
        self.num_elements = M_ris

    def reset(self, seed=None):
        self.state = self.env.reset()
        return self.state

    def step(self, action_phases):
        if not isinstance(action_phases, torch.Tensor):
            action_phases = torch.tensor(action_phases, dtype=torch.float32, device=self.device)
        else:
            action_phases = action_phases.to(self.device)
        
        theta = torch.exp(1j * action_phases)
        Theta = torch.diag(theta)
        
        h_eff = self.env.h_direct + self.env.G @ Theta @ self.env.F
        W = h_eff.conj().T 
        W = W / torch.norm(W, dim=0, keepdim=True) 
        
        H_W = h_eff @ W 
        signal_power = torch.abs(torch.diag(H_W))**2
        total_power = torch.sum(torch.abs(H_W)**2, dim=1)
        interference_noise = total_power - signal_power + 1.0 
        
        sinr = signal_power / interference_noise
        sum_rate = torch.sum(torch.log2(1 + sinr)).real
        
        reward = sum_rate.item() if isinstance(sum_rate, torch.Tensor) else sum_rate
        return self.state, reward, True, {"sum_rate": reward}