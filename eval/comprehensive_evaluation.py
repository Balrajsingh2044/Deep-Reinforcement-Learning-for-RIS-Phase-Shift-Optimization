# --- PROJECT PATH FIX ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------

import torch
import time
import numpy as np
from models.ddpg import Actor
from models.deep_unfolding import DeepUnfolding
from ris_env import RISEnv

STATE_DIM, ACTION_DIM = 1056, 64

def run_full_evaluation(num_seeds=5, test_samples=1500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RISEnv(N_bs=4, M_ris=64, K_users=4, device=device)
    print(f"🚀 Evaluating over {num_seeds} seeds on {device}...")

    metrics = {
        "M1_DDPG":      {"rate": [], "gap": [], "latency": []},
        "M3_Unfolding": {"rate": [], "gap": [], "latency": []},
        "AO_Baseline":  {"rate": []},
    }

    torch.manual_seed(0)
    test_states = torch.randn(test_samples, STATE_DIM)
    ao_rates = np.random.normal(45.0, 2.0, test_samples)  # REPLACE with Ayansh's actual AO test rates

    for seed in range(num_seeds):
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"  Seed {seed+1}/{num_seeds}...")

        # --- M1 Evaluation ---
        actor = Actor(STATE_DIM, ACTION_DIM).to(device)
        # actor.load_state_dict(torch.load("models/m1_ddpg_actor.pth", map_location=device))
        actor.eval()
        
        m1_rates_list = []
        start_time = time.perf_counter()
        with torch.no_grad():
            m1_actions = actor(test_states.to(device))
        for i in range(test_samples):
            r, _ = env.step(m1_actions[i])
            m1_rates_list.append(r.item())
        end_time = time.perf_counter()
        m1_lat = ((end_time - start_time) / test_samples) * 1000
        m1_rates = np.array(m1_rates_list)

        # --- M3 Evaluation ---
        m3 = DeepUnfolding(state_dim=1056, action_dim=64, num_iterations=15).to(device)
        # m3.load_state_dict(torch.load("models/m3_unfolding.pth", map_location=device))
        m3.eval()
        
        m3_rates_list = []
        start_time = time.perf_counter()
        with torch.no_grad():
            m3_actions = m3(test_states.to(device))[0]  # Returns (phases, loss)
        for i in range(test_samples):
            r, _ = env.step(m3_actions[i])
            m3_rates_list.append(r.item())
        end_time = time.perf_counter()
        m3_lat = ((end_time - start_time) / test_samples) * 1000
        m3_rates = np.array(m3_rates_list)

        # --- Calculate Gap ---
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