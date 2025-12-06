"""
Offline RL trainer for diffusion policy.
"""

import os
import numpy as np
from omegaconf import OmegaConf
import torch
import pickle
import hydra
import logging
import wandb
import random

log = logging.getLogger(__name__)

DEVICE = "cuda:0"


def to_device(x, device=DEVICE):
    if torch.is_tensor(x):
        return x.to(device)
    elif type(x) is dict:
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        print(f"Unrecognized type in `to_device`: {type(x)}")


def batch_to_device(batch, device="cuda:0"):
    vals = [to_device(getattr(batch, field), device) for field in batch._fields]
    return type(batch)(*vals)

class TrainAgent:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Batch size for gradient update
        self.batch_size: int = cfg.train.batch_size

        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.max_grad_norm = cfg.train.get("max_grad_norm", None)

        # Logging, rendering, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)


        # Build dataset
        self.dataset_expert_train = hydra.utils.instantiate(self.cfg.offline_expert_dataset)
        self.dataloader_expert_train = torch.utils.data.DataLoader(
            self.dataset_expert_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_expert_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_expert_train.device == "cpu" else False,
        )

        self.dataset_failure_train = hydra.utils.instantiate(self.cfg.offline_failure_dataset)
        self.dataloader_failure_train = torch.utils.data.DataLoader(
            self.dataset_failure_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_failure_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_failure_train.device == "cpu" else False,
        )

    def run(self):
        pass
        