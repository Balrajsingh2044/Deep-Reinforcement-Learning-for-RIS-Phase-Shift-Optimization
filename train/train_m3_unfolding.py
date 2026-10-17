# --- PROJECT PATH FIX ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.wandb_logger import init_wandb

class DeepUnfoldingReal(nn.Module):
    """
    Simplified Deep Unfolding for real-valued phase angle prediction.
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
        # Initialize phases from state features
        phases = self.feature_net(states)
        
        # Unrolled iterations with learnable step sizes
        for i in range(self.num_iterations):
            # Refinement step (simplified RCG-like update)
            refinement = self.refine_net(phases)
            phases = phases + self.step_sizes[i] * refinement
            
            # Optional: clip to valid phase range [0, 2π] or [-π, π]
            # phases = torch.remainder(phases, 2 * torch.pi)
        
        if ao_target is not None:
            # MSE loss between predicted and target angles
            loss = nn.functional.mse_loss(phases, ao_target)
            return phases, loss
        
        return phases, None


def train_m3(epochs=100, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb = init_wandb(project="RIS-M3-Unfolding", config={"epochs": epochs, "lr": lr})

    # Load Ayansh's dataset
    data = torch.load("data/ao_ground_truth.pt", map_location=device)
    
    states = data['channel_state']      # (10000, 1056) real-valued
    actions = data['ao_target']          # (10000, 64) real-valued phase angles
    
    print(f"Loaded dataset: {states.shape[0]} samples")
    print(f"  States: {states.shape}, range [{states.min():.3f}, {states.max():.3f}]")
    print(f"  Actions: {actions.shape}, range [{actions.min():.3f}, {actions.max():.3f}]")
    
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DeepUnfoldingReal(state_dim=1056, action_dim=64, num_iterations=15).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)
            
            optimizer.zero_grad()
            output, loss = model(states_batch, actions_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        wandb.log({"M3_Epoch": epoch, "M3_MSE_Loss": avg_loss})
        print(f"Epoch {epoch}/{epochs} | Loss {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/m3_unfolding_best.pth")
            torch.save(model.state_dict(), "models/m3_unfolding.pth")
            print(f"  ↳ New best model saved (loss={best_loss:.6f})")

    wandb.finish()
    print(f"\nM3 training complete. Best loss: {best_loss:.6f}")

if __name__ == "__main__":
    train_m3()