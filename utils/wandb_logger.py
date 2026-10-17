# utils/wandb_logger.py

class MockWandB:
    def __init__(self, project, config):
        self.project = project
        print(f"[MockWandB] Initialized project: {project}")
        print(f"[MockWandB] Config: {config}")
    
    def log(self, metrics):
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                   for k, v in metrics.items()])
        print(f"[{self.project}] {metrics_str}")
    
    def finish(self):
        print(f"[MockWandB] Training complete for {self.project}")

def init_wandb(project, config):
    return MockWandB(project, config)