# utils/wandb_logger.py
import wandb

class WandBLogger:
    def __init__(self, project="ris_drl_sprint", config=None, group=None, name=None):
        self.run = wandb.init(project=project, config=config or {}, group=group, name=name)

    def log(self, metrics, step=None):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()