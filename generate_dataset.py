import torch
from ris_env import RISEnv
from ao_baseline import riemannian_gradient_ascent

def generate_dataset(num_samples=10000):
    env = RISEnv()
    dataset_states, dataset_actions = [], []
    
    print("Starting ground truth generation...")
    for i in range(num_samples):
        state = env.reset()
        optimal_phases, _ = riemannian_gradient_ascent(env)
        dataset_states.append(state)
        dataset_actions.append(optimal_phases)
        if (i + 1) % 1000 == 0: print(f"Generated {i + 1}/{num_samples} samples")
            
        torch.save({
        'channel_state': torch.stack(dataset_states),  # Changed from 'states'
        'ao_target': torch.stack(dataset_actions)      # Changed from 'actions'
    }, 'data/ao_ground_truth.pt')
    print("Dataset saved to data/ao_ground_truth.pt")

if __name__ == "__main__":
    import os; os.makedirs('data', exist_ok=True)
    generate_dataset()