import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from models.ddpg import Actor
from models.deep_unfolding import DeepUnfolding

print("Converting M1 (DDPG) to full model object...")
actor = Actor(state_dim=1056, action_dim=64)
actor.load_state_dict(torch.load("checkpoints/m1_actor_best.pth", map_location="cpu"))
torch.save(actor, "models/m1_ddpg_actor.pth")
print("✅ M1 converted and saved to models/m1_ddpg_actor.pth")

print("\nConverting M3 (Deep Unfolding) to full model object...")
m3 = DeepUnfolding(state_dim=1056, action_dim=64, num_iterations=15)
m3.load_state_dict(torch.load("checkpoints/m3_unfolding_best.pth", map_location="cpu"))
torch.save(m3, "models/m3_unfolding.pth")
print("✅ M3 converted and saved to models/m3_unfolding.pth")

print("\n🚀 Conversion complete! No retraining needed.")