# --- PROJECT PATH FIX ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------

import torch
import time
import numpy as np
from models.ddpg import Actor
from models.deep_unfolding import DeepUnfoldingRCG
from ris_env import RISEnv

STATE_DIM, ACTION_DIM = 1056, 64

def evaluate_model(env, model, states, init_phases, chan_states, device, num_samples=1500, is_m3=False):
    model.eval()
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()

    with torch.no_grad():
        if is_m3:
            # For M3, we pass complex inputs and get complex outputs
            actions, _ = model(init_phases.to(device), chan_states.to(device))
            # Convert complex phase shifts to real-valued actions for env.step if needed
            # Or just use the magnitude/angle depending on how env expects it. 
            # Assuming env.step expects raw phase angles or similar, but here we just measure latency/shape.
            # Note: The current env.step expects a tensor of size 64.
            
            # If M3 output is complex unit circle, we might need to extract angle for env
            # But for latency measurement, this is sufficient.
            sum_rates = torch.zeros(num_samples).to(device) # Placeholder for rate calc in eval
            
        else:
            actions = model(states.to(device))
            # Calculate sum rate using environment
            # Note: env.step returns (sum_rate, h_eff). We only need sum_rate here.
            # However, env.step expects action shape (64,).
            
            # To get actual rates for evaluation, we should loop or batch if env supports it.
            # Since current env is single-sample, we'll do a quick loop for accuracy or 
            # assume batched processing if implemented. For now, let's fix the shape mismatch.
            
            # Correct way for single-sample env in batch eval:
            rates_list = []
            for i in range(num_samples):
                r, _ = env.step(actions[i])
                rates_list.append(r.item())
            sum_rates = torch.tensor(rates_list).to(device)

    torch.cuda.synchronize() if device.type == "cuda" else None
    latency_ms = (time.perf_counter() - start) / num_samples * 1000
    
    # Return numpy array for metrics calculation
    if is_m3:
        return sum_rates.cpu().numpy(), latency_ms
    else:
        return sum_rates.cpu().numpy(), latency_ms

def run_full_evaluation(num_seeds=5, test_samples=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Updated to match Ayansh's locked spec: K=4, M=64, N=4
    env = RISEnv(N_bs=4, M_ris=64, K_users=4, device=device)
    print(f"🚀 Evaluating over {num_seeds} seeds on {device}...")

    metrics = {
        "M1_DDPG":      {"rate": [], "gap": [], "latency": []},
        "M3_Unfolding": {"rate": [], "gap": [], "latency": []},
        "AO_Baseline":  {"rate": []},
    }

    torch.manual_seed(0)
    # Test data shapes updated to 1056 and 64
    test_states = torch.randn(test_samples, STATE_DIM)
    test_init_p = torch.randn(test_samples, ACTION_DIM, dtype=torch.complex64)
    test_chan_s = torch.randn(test_samples, ACTION_DIM, dtype=torch.complex64)
    
    # Placeholder for AO rates - REPLACE with Ayansh's actual AO test rates later
    ao_rates = np.random.normal(45.0, 2.0, test_samples) 

    for seed in range(num_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"  Seed {seed+1}/{num_seeds}...")

        # M1 Evaluation
        actor = Actor(STATE_DIM, ACTION_DIM).to(device)
        # actor.load_state_dict(torch.load(f"checkpoints/m1_actor_seed{seed}.pth", map_location=device))
        actor.eval()
        
        # Get actions from M1
        with torch.no_grad():
            m1_actions = actor(test_states.to(device))
        
        # Calculate rates for M1
        m1_rates_list = []
        start_time = time.perf_counter()
        for i in range(test_samples):
            r, _ = env.step(m1_actions[i])
            m1_rates_list.append(r.item())
        end_time = time.perf_counter()
        m1_lat = ((end_time - start_time) / test_samples) * 1000
        m1_rates = np.array(m1_rates_list)

        # M3 Evaluation
        from train.train_m3_unfolding import DeepUnfoldingReal
        m3 = DeepUnfoldingReal(state_dim=1056, action_dim=64, num_iterations=15).to(device)
        # m3.load_state_dict(torch.load(f"checkpoints/m3_unfolding_seed{seed}.pth", map_location=device))
        m3.eval()

        with torch.no_grad():
            m3_actions, _ = m3(test_states.to(device))  # Use same test_states as M1

        m3_rates_list = []
        start_time = time.perf_counter()
        for i in range(test_samples):
            r, _ = env.step(m3_actions[i])
            m3_rates_list.append(r.item())
        end_time = time.perf_counter()
        m3_lat = ((end_time - start_time) / test_samples) * 1000
        m3_rates = np.array(m3_rates_list)
            
        m3_rates_list = []
        start_time = time.perf_counter()
        for i in range(test_samples):
            # Env might expect real values or complex. Adjust based on env.step implementation.
            # If env.step expects raw phase shifts (real), use m3_angles.
            r, _ = env.step(m3_angles[i])
            m3_rates_list.append(r.item())
        end_time = time.perf_counter()
        m3_lat = ((end_time - start_time) / test_samples) * 1000
        m3_rates = np.array(m3_rates_list)

        # Calculate Gap
        m1_gap = ((ao_rates - m1_rates) / ao_rates) * 100
        m3_gap = ((ao_rates - m3_rates) / ao_rates) * 100

        metrics["M1_DDPG"]["rate"].append(np.mean(m1_rates))
        metrics["M1_DDPG"]["gap"].append(np.mean(m1_gap))
        metrics["M1_DDPG"]["latency"].append(m1_lat)

        metrics["M3_Unfolding"]["rate"].append(np.mean(m3_rates))
        metrics["M3_Unfolding"]["gap"].append(np.mean(m3_gap))
        metrics["M3_Unfolding"]["latency"].append(m3_lat)

        metrics["AO_Baseline"]["rate"].append(np.mean(ao_rates))

    print("\n" + "=" * 60)
    print("📊 FINAL METRICS (Mean ± Std Dev, 5 seeds)")
    print("=" * 60)
    for name in ["M1_DDPG", "M3_Unfolding"]:
        r = metrics[name]["rate"]; g = metrics[name]["gap"]; l = metrics[name]["latency"]
        print(f"\n{name}:")
        print(f"  Sum Rate        : {np.mean(r):.2f} ± {np.std(r):.2f} bits/s/Hz")
        print(f"  % Gap to AO     : {np.mean(g):.2f}% ± {np.std(g):.2f}%")
        print(f"  Latency         : {np.mean(l):.4f} ± {np.std(l):.4f} ms/sample")
    ao = metrics["AO_Baseline"]["rate"]
    print(f"\nAO Baseline     : {np.mean(ao):.2f} ± {np.std(ao):.2f} bits/s/Hz")
    print("=" * 60)

if __name__ == "__main__":
    run_full_evaluation(num_seeds=5, test_samples=1500)