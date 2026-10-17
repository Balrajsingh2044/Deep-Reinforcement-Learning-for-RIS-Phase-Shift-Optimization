import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.deep_unfolding import DeepUnfoldingRCG
from utils.wandb_logger import init_wandb

def train_m3(epochs=100, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb = init_wandb(project="RIS-M3-Unfolding", config={"epochs": epochs, "lr": lr})
    
    # Load Ayansh's dataset: assumes dict with 'initial_phase', 'channel_state', 'ao_target'
    # data = torch.load("data/ao_baseline.pt") 
    # For now, using dummy complex data
    dataset = TensorDataset(
        torch.randn(10000, 64, dtype=torch.complex64), # initial_phase
        torch.randn(10000, 64, dtype=torch.complex64), # channel_state
        torch.randn(10000, 64, dtype=torch.complex64)  # ao_target
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DeepUnfoldingRCG(num_elements=64, num_iterations=15).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for init_phase, chan_state, ao_target in dataloader:
            init_phase, chan_state, ao_target = init_phase.to(device), chan_state.to(device), ao_target.to(device)
            
            optimizer.zero_grad()
            output, loss = model(init_phase, chan_state, ao_target)
            loss.backward()
            
            # Gradient clipping to prevent NaNs in complex networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        wandb.log({"M3_Epoch": epoch, "M3_MSE_Loss": avg_loss})
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
        
    torch.save(model.state_dict(), "checkpoints/m3_unfolding_best.pth")
    wandb.finish()

if __name__ == "__main__":
    train_m3()