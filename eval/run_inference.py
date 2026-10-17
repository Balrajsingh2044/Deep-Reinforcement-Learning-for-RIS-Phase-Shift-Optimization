import torch
import time
from models.ddpg import Actor
from models.deep_unfolding import DeepUnfoldingRCG

def run_inference(test_samples=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {test_samples} samples on {device}...")
    
    # Load Models
    actor = Actor(state_dim=128, action_dim=64).to(device)
    actor.load_state_dict(torch.load("checkpoints/m1_actor_best.pth", map_location=device))
    actor.eval()
    
    m3 = DeepUnfoldingRCG(num_elements=64, num_iterations=15).to(device)
    m3.load_state_dict(torch.load("checkpoints/m3_unfolding_best.pth", map_location=device))
    m3.eval()
    
    # Dummy test data (replace with Ayansh's 15% unseen test set)
    test_states = torch.randn(test_samples, 128).to(device)
    test_init_p = torch.randn(test_samples, 64, dtype=torch.complex64).to(device)
    test_chan_s = torch.randn(test_samples, 64, dtype=torch.complex64).to(device)
    
    # --- M1 Inference Latency ---
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = actor(test_states)
    torch.cuda.synchronize()
    m1_latency_ms = (time.perf_counter() - start_time) / test_samples * 1000
    print(f"⏱️  M1 (DDPG) Avg Inference Latency: {m1_latency_ms:.4f} ms/sample")
    
    # --- M3 Inference Latency ---
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        _ = m3(test_init_p, test_chan_s)
    torch.cuda.synchronize()
    m3_latency_ms = (time.perf_counter() - start_time) / test_samples * 1000
    print(f"⏱️  M3 (Unfolding) Avg Inference Latency: {m3_latency_ms:.4f} ms/sample")

if __name__ == "__main__":
    run_inference()