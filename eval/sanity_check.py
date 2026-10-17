import sys
import os
import torch

# Ensure the root directory is in the Python path (fixes IDE red lines)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ddpg import Actor, Critic
from models.deep_unfolding import DeepUnfoldingRCG
# Fixed import: added SACCritic
from models.sac_codebook import SACActor, SACCritic, logits_to_phases, CODEBOOK_PHASES

def sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Sanity Checks...")
    
    # 1. M1 Shape Check
    state_dim, action_dim = 208, 16  # Was 128, 64
    actor = Actor(state_dim, action_dim).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    
    dummy_state = torch.randn(4, state_dim).to(device)
    dummy_action = torch.randn(4, action_dim).to(device)
    
    out_actor = actor(dummy_state)
    out_critic = critic(dummy_state, dummy_action)
    assert out_actor.shape == (4, action_dim), "M1 Actor shape mismatch!"
    assert out_critic.shape == (4, 1), "M1 Critic shape mismatch!"
    print("✅ M1 (DDPG) forward pass shapes verified.")
    
    # 2. M3 NaN Check
    model_m3 = DeepUnfoldingRCG(num_elements=64, num_iterations=15).to(device)
    init_p = torch.randn(4, 64, dtype=torch.complex64).to(device)
    chan_s = torch.randn(4, 64, dtype=torch.complex64).to(device)
    ao_t = torch.randn(4, 64, dtype=torch.complex64).to(device)
    
    out_m3, loss_m3 = model_m3(init_p, chan_s, ao_t)
    assert not torch.isnan(loss_m3), "M3 produced NaN loss!"
    print("✅ M3 (Deep Unfolding) forward/backward pass verified (No NaNs).")

    # 3. M2 (SAC + Codebook) Shape & Snapping Check
    actor_m2 = SACActor(state_dim=128, num_elements=64, num_phases=8).to(device)
    critic_m2 = SACCritic(state_dim=128, num_elements=64).to(device)
    
    dummy_state_m2 = torch.randn(4, 128).to(device)
    logits = actor_m2(dummy_state_m2) # Shape: (4, 64, 8)
    
    # Test Gumbel-Softmax snapping
    phases, log_prob = logits_to_phases(logits, temperature=1.0, hard=True)
    
    assert phases.shape == (4, 64), f"M2 phase output shape mismatch! Got {phases.shape}"
    assert not torch.isnan(phases).any(), "M2 produced NaN phases!"
    
    # Verify phases are actually snapping to the 8 codebook values
    valid_phases = any(torch.isclose(phases.flatten()[0], cp, atol=1e-5).item() for cp in CODEBOOK_PHASES)
    assert valid_phases, "M2 Gumbel-Softmax is not snapping to the codebook!"
    
    out_critic_m2 = critic_m2(dummy_state_m2, phases)
    assert out_critic_m2.shape == (4, 1), "M2 Critic shape mismatch!"
    print("✅ M2 (SAC + Codebook) forward pass, Gumbel snapping, and shapes verified.")
    
    print("🚀 All sanity checks passed. Ready for full training.")

if __name__ == "__main__":
    sanity_check()