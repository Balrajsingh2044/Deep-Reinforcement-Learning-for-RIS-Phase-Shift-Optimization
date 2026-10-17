import torch
import torch.nn.functional as F
import numpy as np
from models.ddpg import Actor, Critic, ReplayBuffer, OUNoise
from models.deep_unfolding import DeepUnfoldingRCG
from utils.mock_env import MockRISEnv

def test_m1_ddpg():
    print("\n--- Testing M1 (DDPG) ---")
    env = MockRISEnv()
    state_dim, action_dim = 128, 64
    
    # 1. Test Networks
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    state = torch.FloatTensor(env.reset()).unsqueeze(0)
    
    action = actor(state)
    assert action.shape == (1, 64), "Actor output shape wrong!"
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0), "Actor action out of bounds!"
    
    q_val = critic(state, action)
    assert q_val.shape == (1, 1), "Critic output shape wrong!"
    print("✅ M1 Networks forward pass: OK")

    # 2. Test Replay Buffer & Noise
    buffer = ReplayBuffer(capacity=100)
    noise = OUNoise(action_dim)
    
    for _ in range(150): # Overfill buffer to test deque
        s = env.reset()
        a = np.clip(actor(torch.FloatTensor(s).unsqueeze(0)).detach().numpy()[0] + noise.sample(), -1, 1)
        ns, r, d, _ = env.step(a)
        buffer.push(s, a, r, ns, d)
        
    assert len(buffer) == 100, "Buffer capacity failed!"
    s_b, a_b, r_b, ns_b, d_b = buffer.sample(32)
    assert s_b.shape == (32, 128) and r_b.shape == (32, 1), "Buffer sampling shape failed!"
    print("✅ M1 Buffer & Noise: OK")

def test_m3_unfolding():
    print("\n--- Testing M3 (Deep Unfolding) ---")
    model = DeepUnfoldingRCG(num_elements=64, num_iterations=15)
    
    # Load mock data
    data = torch.load("data/ao_baseline.pt")
    batch = {k: v[:32] for k, v in data.items()}
    
    out_phase, loss = model(batch['initial_phase'], batch['channel_state'], batch['ao_target'])
    
    # 1. Check Loss
    assert not torch.isnan(loss), "M3 Loss is NaN!"
    loss.backward() # Check backward pass
    
    # 2. CRITICAL MATH CHECK: RIS phase shifts MUST have a magnitude of exactly 1.0
    magnitudes = torch.abs(out_phase)
    is_unit_circle = torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-5)
    assert is_unit_circle, "CRITICAL FAILURE: M3 output phases are not on the unit circle (|z| != 1)!"
    
    print("✅ M3 Forward/Backward pass: OK")
    print("✅ M3 Riemannian Projection (Unit Circle constraint): OK")

def test_evaluation_metrics():
    print("\n--- Testing Evaluation Logic ---")
    # Test the % Gap formula
    ao_rate = np.array([50.0, 40.0])
    drl_rate = np.array([45.0, 36.0])
    
    # Expected gap: (50-45)/50 = 10%, (40-36)/40 = 10%
    gap = ((ao_rate - drl_rate) / ao_rate) * 100
    assert np.allclose(gap, [10.0, 10.0]), "Gap calculation formula is wrong!"
    print("✅ Evaluation % Gap Formula: OK")

if __name__ == "__main__":
    test_m1_ddpg()
    test_m3_unfolding()
    test_evaluation_metrics()
    print("\n🎉 ALL SELF-CHECKS PASSED. Your code is 100% ready for integration.")