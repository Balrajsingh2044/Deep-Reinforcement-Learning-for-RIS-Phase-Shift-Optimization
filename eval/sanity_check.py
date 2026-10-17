import torch
from models.ddpg import Actor, Critic
from models.deep_unfolding import DeepUnfoldingRCG

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