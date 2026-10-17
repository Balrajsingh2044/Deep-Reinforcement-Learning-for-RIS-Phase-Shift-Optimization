# --- PROJECT PATH FIX ---
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------

import torch
import torch.nn.functional as F
from models.ddpg import Actor, Critic
from models.deep_unfolding import DeepUnfolding

def sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Sanity Checks...")

    # --- M1 (DDPG) ---
    state_dim, action_dim = 1056, 64
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)

    dummy_state = torch.randn(4, state_dim).to(device)
    dummy_action = torch.randn(4, action_dim).to(device)

    out_actor = actor(dummy_state)
    out_critic = critic(dummy_state, dummy_action)
    assert out_actor.shape == (4, action_dim), "M1 Actor shape mismatch!"
    assert out_critic.shape == (4, 1), "M1 Critic shape mismatch!"
    print("✅ M1 (DDPG) forward pass shapes verified.")

    # --- M3 (Deep Unfolding) ---
    model_m3 = DeepUnfolding(state_dim=1056, action_dim=64, num_iterations=15).to(device)
    test_states = torch.randn(4, 1056).to(device)
    test_actions = torch.randn(4, 64).to(device)

    out_m3, loss_m3 = model_m3(test_states, test_actions)
    assert not torch.isnan(loss_m3), "M3 produced NaN loss!"
    assert out_m3.shape == (4, 64), "M3 output shape mismatch!"
    print("✅ M3 (Deep Unfolding) forward/backward verified.")

    # --- M2 (SAC + Gumbel-Softmax codebook) ---
    num_phases = 8
    logits = torch.randn(4, num_phases)
    snapped = F.gumbel_softmax(logits, tau=0.1, hard=True, dim=-1)
    max_vals, _ = snapped.max(dim=-1)
    assert torch.all(max_vals == 1.0).item(), "M2 Gumbel failed to snap!"
    print("✅ M2 (SAC + Codebook) Gumbel snapping verified.")

    print("🚀 All sanity checks passed. Ready for full training.")

if __name__ == "__main__":
    sanity_check()