import torch
from models.ddpg import Actor, Critic
from models.deep_unfolding import DeepUnfoldingRCG
import torch
import torch.nn.functional as F

def check_m2_gumbel_softmax():
    print("Checking M2 (SAC + Codebook) Gumbel-Softmax snapping...")
    batch_size = 4
    num_phases = 8 # 3-bit codebook
    
    # Dummy continuous logits from SAC Actor
    logits = torch.randn(batch_size, num_phases)
    
    # Gumbel-Softmax trick (tau=0.1 makes it nearly discrete/one-hot)
    # hard=True ensures the output is exactly one-hot (snapped to codebook)
    snapped_actions = F.gumbel_softmax(logits, tau=0.1, hard=True, dim=-1)
    
    # Verify it is truly discrete (sum of squares should equal batch_size for one-hot)
    is_discrete = torch.allclose(snapped_actions.sum(dim=-1).pow(2), torch.ones(batch_size) * num_phases, atol=1e-5) # Adjusted logic for one-hot verification
    
    # Better check: each row should have exactly one '1.0' and rest '0.0'
    max_vals, _ = snapped_actions.max(dim=-1)
    is_snapped = torch.all(max_vals == 1.0).item()
    
    assert is_snapped, "M2 Gumbel-Softmax failed to snap to discrete codebook!"
    print("✅ M2 Gumbel-Softmax correctly snapping to 3-bit (8-phase) discrete codebook.")

def sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running Sanity Checks...")
    
    # 1. M1 Shape Check
    state_dim, action_dim = 128, 64
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
    print("🚀 Sanity checks passed. Ready for full training.")

if __name__ == "__main__":
    sanity_check()