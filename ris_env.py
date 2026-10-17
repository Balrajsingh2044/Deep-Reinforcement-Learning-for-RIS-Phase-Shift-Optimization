import torch

class RISEnv:
    def __init__(self, N_bs=4, M_ris=16, K_users=2, device='cpu'):
        self.N_bs = N_bs
        self.M_ris = M_ris
        self.K_users = K_users
        self.device = device
        
    def reset(self):
        self.h_direct = (torch.randn(self.K_users, self.N_bs) + 1j * torch.randn(self.K_users, self.N_bs)) / 1.414
        self.G = (torch.randn(self.K_users, self.M_ris) + 1j * torch.randn(self.K_users, self.M_ris)) / 1.414
        self.F = (torch.randn(self.M_ris, self.N_bs) + 1j * torch.randn(self.M_ris, self.N_bs)) / 1.414
        
        state = torch.cat([self.h_direct.real.flatten(), self.h_direct.imag.flatten(),
                           self.G.real.flatten(), self.G.imag.flatten(),
                           self.F.real.flatten(), self.F.imag.flatten()])
        return state

    def step(self, raw_actions):
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
        
        # Return tensor (not .item()) so it works for both RL and AO
        return sum_rate, h_eff