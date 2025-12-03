"""
Offline RL trainer for diffusion policy.
"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
from copy import deepcopy
from tqdm import tqdm

log = logging.getLogger(__name__)
from util.timer import Timer
from collections import deque
from agent.offline.train_agent import TrainAgent, batch_to_device
from util.scheduler import CosineAnnealingWarmupRestarts

class TrainIDQLDiffusionAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Wwarm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # Optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_q_optimizer = torch.optim.AdamW(
            self.model.critic_q.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_v_optimizer = torch.optim.AdamW(
            self.model.critic_v.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_v_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_v_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_q_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_q_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        # Updates
        self.critic_tau = cfg.train.critic_tau

    def run(self):
        # Start training loop
        timer = Timer()
        for itr in tqdm(range(self.n_epochs)):
            running_actor_loss = []
            running_critic_q_loss = []
            running_critic_v_loss = []
            for batch in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch = batch_to_device(batch)
                # Sample batch
                obs_b = batch.conditions["state"]
                next_obs_b = batch.conditions["next_state"]
                actions_b = batch.actions
                reward_b = batch.rewards
                terminated_b = batch.dones

                # update critic value function
                critic_loss_v = self.model.loss_critic_v(
                    {"state": obs_b}, actions_b
                )
                running_critic_v_loss.append(critic_loss_v.item())
                self.critic_v_optimizer.zero_grad()
                critic_loss_v.backward()
                self.critic_v_optimizer.step()

                # update critic q function
                critic_loss_q = self.model.loss_critic_q(
                    {"state": obs_b},
                    {"state": next_obs_b},
                    actions_b,
                    reward_b,
                    terminated_b,
                    self.gamma,
                )
                running_critic_q_loss.append(critic_loss_q.item())
                self.critic_q_optimizer.zero_grad()
                critic_loss_q.backward()
                self.critic_q_optimizer.step()

                # update target q function
                self.model.update_target_critic(self.critic_tau)

                # update actor
                actor_loss = self.model.loss(
                    actions_b,
                    {"state": obs_b},
                )
                running_actor_loss.append(actor_loss.item())
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.actor.parameters(),
                            self.max_grad_norm,
                        )
                    self.actor_optimizer.step()

            # log
            if itr % self.log_freq == 0:
                log.info(
                        f"Iteration {itr}, Actor Loss: {np.mean(running_actor_loss)}, Critic Q Loss: {np.mean(running_critic_q_loss)}, Critic V Loss: {np.mean(running_critic_v_loss)}"
                    )
                wandb.log(
                    {
                        "actor_loss": np.mean(running_actor_loss),
                        "critic_q_loss": np.mean(running_critic_q_loss),
                        "critic_v_loss": np.mean(running_critic_v_loss),
                    },
                    step=itr,
                    commit=False,
                )

            # save model
            if itr % self.save_model_freq == 0:
                data = {
                    "epoch": itr,
                    "model": self.model.state_dict(),
                }
                savepath = os.path.join(self.cfg.logdir, f"checkpoint/state_{itr}.pt")
                torch.save(data, savepath)
                log.info(f"Saved model to {savepath}")

            # update lr scheduler
            self.actor_lr_scheduler.step()
            self.critic_q_lr_scheduler.step()
            self.critic_v_lr_scheduler.step()