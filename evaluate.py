import torch, time, os
from ris_env import RISEnv

def load_model(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found!")
        return None
    # weights_only=False is required to load custom classes like SACActor
    model = torch.load(path, map_location='cpu', weights_only=False)
    model.eval()
    return model

def process_action(action_batch, model_name):
    # 1. Unpack if the model returns a tuple (e.g., action, log_prob) or dict
    if isinstance(action_batch, (tuple, list)):
        action_batch = action_batch[0]
    elif isinstance(action_batch, dict):
        action_batch = list(action_batch.values())[0]
        
    # 2. Remove the batch dimension -> shape becomes [X]
    action = action_batch.squeeze(0)
    
    # 3. SPECIAL HANDLING FOR M2_SAC
    if model_name == "M2_SAC":
        if action.numel() == 512:
            # Reshape 512 logits into [64 elements, 8 phases]
            action = action.view(64, 8)
            # Pick the phase with the highest logit for each of the 64 elements
            phase_indices = torch.argmax(action, dim=1) # shape: [64]
            # Convert indices (0-7) to actual phase shifts (0 to 2*pi)
            action = phase_indices.float() * (2 * torch.pi / 8.0)
        elif action.dim() == 2 and action.shape == (64, 8):
            phase_indices = torch.argmax(action, dim=1)
            action = phase_indices.float() * (2 * torch.pi / 8.0)
        elif action.dim() == 2 and action.shape == (64, 1):
            action = action.squeeze(1)
            
    return action

def evaluate_model(model, env, model_name, num_episodes=500, csi_noise_std=0.0):
    total_rate, total_time = 0.0, 0.0
    with torch.no_grad():
        for _ in range(num_episodes):
            state = env.reset()
            if csi_noise_std > 0: 
                state = state + torch.randn_like(state) * csi_noise_std
            
            # Add batch dimension for the neural network -> [1, 1056]
            state_batch = state.unsqueeze(0) 
            
            start_time = time.perf_counter()
            action_batch = model(state_batch) 
            total_time += (time.perf_counter() - start_time)
            
            # Process the output to ensure it's exactly [64] for the environment
            action = process_action(action_batch, model_name)
            
            rate, _ = env.step(action)
            total_rate += rate
            
    avg_rate = total_rate / num_episodes
    avg_latency_ms = (total_time / num_episodes) * 1000 
    return avg_rate, avg_latency_ms

if __name__ == "__main__":
    env = RISEnv() 
    state_dim = env.reset().shape[0]
    
    models = {
        "M1_DDPG": "models/m1_ddpg_actor.pth",
        "M2_SAC": "models/m2_sac_actor.pth",
        "M3_Unfold": "models/m3_unfolding.pth"
    }
    
    print("Starting Final Joint Evaluation...")
    print(f"{'Model':<12} | {'CSI Noise':<10} | {'Sum Rate (bps/Hz)':<18} | {'Latency (ms)':<12}")
    print("-" * 65)
    
    results_data = []
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"{name}: SKIPPED (File not found)")
            continue
            
        try:
            model = load_model(path)
            print(f"\n--- Evaluating {name} ---")
            for noise in [0.0, 0.5, 1.0, 2.0]: 
                rate, latency = evaluate_model(model, env, name, num_episodes=500, csi_noise_std=noise)
                print(f"CSI Noise: {noise:<5.1f} | Sum Rate: {rate:<10.2f} | Latency: {latency:<12.4f} ms")
                results_data.append(f"{name},{noise},{rate},{latency}")
        except Exception as e:
            print(f"ERROR evaluating {name}: {e}")
            
    os.makedirs('results', exist_ok=True)
    with open('results/raw_eval_data.csv', 'w') as f:
        f.write("Model,CSI_Noise,Sum_Rate,Latency_ms\n")
        for row in results_data:
            f.write(row + "\n")
            
    print("\nEvaluation complete. Raw data saved to results/raw_eval_data.csv")