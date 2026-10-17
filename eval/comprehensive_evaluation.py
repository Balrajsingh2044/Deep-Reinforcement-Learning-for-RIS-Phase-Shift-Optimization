import torch
import time
import numpy as np
from models.ddpg import Actor
from models.deep_unfolding import DeepUnfoldingRCG

def evaluate_model(env, model, states, init_phases, chan_states, device, num_samples=1500):
    """Runs inference and calculates Sum Rate and Latency."""
    model.eval()
    
    # 1. Measure Exact CUDA Latency
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        if isinstance(model, Actor):
            actions = model(states.to(device))
            # Calculate sum rate using Ayansh's environment function
            # sum_rates = env.calculate_sum_rate(states, actions) 
            sum_rates = torch.randn(num_samples) * 5 + 40 # DUMMY: Replace with env calculation
        else: # DeepUnfoldingRCG
            actions, _ = model(init_phases.to(device), chan_states.to(device))
            # sum_rates = env.calculate_sum_rate(states, actions)
            sum_rates = torch.randn(num_samples) * 5 + 42 # DUMMY: Replace with env calculation
            
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency_ms = ((end_time - start_time) / num_samples) * 1000
    return sum_rates.cpu().numpy(), latency_ms

def run_full_evaluation(num_seeds=5, test_samples=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Comprehensive Evaluation over {num_seeds} seeds...")
    
    # Metrics storage
    metrics = {
        "M1_DDPG": {"rate": [], "gap": [], "latency": []},
        "M3_Unfolding": {"rate": [], "gap": [], "latency": []},
        "AO_Baseline": {"rate": []} # Ground truth
    }
    
    # Dummy data (Replace with Ayansh's 15% unseen test set: 1500 samples)
    # test_states = torch.load("data/test_states.pt")
    # test_init_p = torch.load("data/test_init_p.pt")
    # test_chan_s = torch.load("data/test_chan_s.pt")
    # ao_rates = torch.load("data/ao_rates.pt")
    
    test_states = torch.randn(test_samples, 128)
    test_init_p = torch.randn(test_samples, 64, dtype=torch.complex64)
    test_chan_s = torch.randn(test_samples, 64, dtype=torch.complex64)
    ao_rates = np.random.normal(45.0, 2.0, test_samples) # DUMMY AO baseline rates
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"  Evaluating Seed {seed+1}/{num_seeds}...")
        
        # --- Evaluate M1 (DDPG) ---
        actor = Actor(state_dim=128, action_dim=64).to(device)
        # actor.load_state_dict(torch.load(f"checkpoints/m1_actor_seed{seed}.pth", map_location=device))
        m1_rates, m1_latency = evaluate_model(None, actor, test_states, None, None, device, test_samples)
        
        # --- Evaluate M3 (Deep Unfolding) ---
        m3 = DeepUnfoldingRCG(num_elements=64, num_iterations=15).to(device)
        # m3.load_state_dict(torch.load(f"checkpoints/m3_unfolding_seed{seed}.pth", map_location=device))
        m3_rates, m3_latency = evaluate_model(None, m3, test_states, test_init_p, test_chan_s, device, test_samples)
        
        # --- Calculate % Gap to AO Bound ---
        # Formula: (AO_Rate - DRL_Rate) / AO_Rate * 100
        m1_gap = ((ao_rates - m1_rates) / ao_rates) * 100
        m3_gap = ((ao_rates - m3_rates) / ao_rates) * 100
        
        metrics["M1_DDPG"]["rate"].append(np.mean(m1_rates))
        metrics["M1_DDPG"]["gap"].append(np.mean(m1_gap))
        metrics["M1_DDPG"]["latency"].append(m1_latency)
        
        metrics["M3_Unfolding"]["rate"].append(np.mean(m3_rates))
        metrics["M3_Unfolding"]["gap"].append(np.mean(m3_gap))
        metrics["M3_Unfolding"]["latency"].append(m3_latency)
        
        metrics["AO_Baseline"]["rate"].append(np.mean(ao_rates))

    # --- Print Final Report for Table 2 ---
    print("\n" + "="*60)
    print("📊 FINAL METRICS FOR MANUSCRIPT (Mean ± Std Dev)")
    print("="*60)
    
    for model_name in ["M1_DDPG", "M3_Unfolding"]:
        rate_mean, rate_std = np.mean(metrics[model_name]["rate"]), np.std(metrics[model_name]["rate"])
        gap_mean, gap_std = np.mean(metrics[model_name]["gap"]), np.std(metrics[model_name]["gap"])
        lat_mean, lat_std = np.mean(metrics[model_name]["latency"]), np.std(metrics[model_name]["latency"])
        
        print(f"\n{model_name}:")
        print(f"  Achievable Sum Rate : {rate_mean:.2f} ± {rate_std:.2f} Mbps")
        print(f"  % Gap to AO Bound   : {gap_mean:.2f}% ± {gap_std:.2f}%")
        print(f"  Inference Latency   : {lat_mean:.4f} ± {lat_std:.4f} ms/sample")
        
    ao_mean, ao_std = np.mean(metrics["AO_Baseline"]["rate"]), np.std(metrics["AO_Baseline"]["rate"])
    print(f"\nAO Baseline (Upper Bound): {ao_mean:.2f} ± {ao_std:.2f} Mbps")
    print("="*60)

if __name__ == "__main__":
    run_full_evaluation(num_seeds=5, test_samples=1500)