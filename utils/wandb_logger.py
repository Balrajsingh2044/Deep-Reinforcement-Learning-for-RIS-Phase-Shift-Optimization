import wandb

def init_wandb(project, config):
    return wandb.init(project=project, config=config)