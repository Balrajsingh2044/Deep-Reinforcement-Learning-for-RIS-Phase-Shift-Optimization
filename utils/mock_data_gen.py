import torch
import os

def generate_mock_ao_data(save_path="data/ao_baseline.pt", num_samples=10000):
    print("Generating mock AO baseline data for M3...")
    os.makedirs("data", exist_ok=True)
    
    # Complex-valued tensors for RIS phases and channels
    initial_phase = torch.randn(num_samples, 64, dtype=torch.complex64)
    channel_state = torch.randn(num_samples, 64, dtype=torch.complex64)
    
    # The "Ground Truth" AO target (must also be on the complex unit circle!)
    ao_target = torch.randn(num_samples, 64, dtype=torch.complex64)
    ao_target = ao_target / (torch.abs(ao_target) + 1e-8) # Normalize to magnitude 1
    
    torch.save({
        'initial_phase': initial_phase,
        'channel_state': channel_state,
        'ao_target': ao_target
    }, save_path)
    print(f"✅ Saved mock data to {save_path}")

if __name__ == "__main__":
    generate_mock_ao_data()