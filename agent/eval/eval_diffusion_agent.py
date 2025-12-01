"""
Evaluate pre-trained/DPPO-fine-tuned diffusion policy.

"""

import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent


class EvalDiffusionAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):

        # Start training loop
        timer = Timer()

        # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        # Reset env before iteration starts
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        if self.save_full_observations:  # state-only
            obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            obs_full_trajs = np.vstack(
                (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
            )

        # ==========================================================
        #   COLLECT ROLLOUTS
        # ==========================================================
        actions_log = []   # save ALL actions for later extraction

        for step in range(self.n_steps):
            if step % 50 == 0:
                print(f"Processed step {step} of {self.n_steps}")

            # Select action
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                samples = self.model(cond=cond, deterministic=True)
                output_venv = (
                    samples.trajectories.cpu().numpy()
                )  # n_env x horizon x act
            action_venv = output_venv[:, : self.act_steps]

            actions_log.append(action_venv.copy())

            # Env step
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = terminated_venv | truncated_venv
            if self.save_full_observations:  # state-only
                obs_full_venv = np.array(
                    [info["full_obs"]["state"] for info in info_venv]
                )  # n_envs x act_steps x obs_dim
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                )

            prev_obs_venv = obs_venv

        actions_log = np.array(actions_log)           # shape (n_steps, n_envs, act_steps, act_dim)
        # ==========================================================
        #   COMPUTE EPISODE BOUNDARIES AND REWARDS
        # ==========================================================
        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )

            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            avg_episode_reward = np.mean(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
        else:
            episode_reward = np.array([])
            num_episode_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
            log.info("[WARNING] No episode completed within the iteration!")

        # ==========================================================
        #   COLLECT FAILED ROLLOUTS INTO npz file
        # ==========================================================
        failed_trajs = []
        failed_traj_lengths = []
        total_states = []
        total_actions = []
        for idx, (env_ind, start, end) in enumerate(episodes_start_end):
            ep_reward = episode_reward[idx]

            if self.furniture_sparse_reward:
                success_flag = (ep_reward >= self.best_reward_threshold_for_success)
            else:
                success_flag = (
                    episode_best_reward[idx] >= self.best_reward_threshold_for_success
                )

            if success_flag:
                continue

            # Extract obs, actions
            obs_seq = obs_full_trajs[start*self.act_steps:(end+1)*self.act_steps, env_ind]
            act_seq = actions_log[start:end+1, env_ind]
            act_seq = act_seq.reshape(-1, act_seq.shape[-1])
            failed_traj_lengths.append(len(act_seq))
            total_states.append(obs_seq)
            total_actions.append(act_seq)

        failed_traj_lengths = np.array(failed_traj_lengths)
        total_states = np.vstack(total_states)
        total_actions = np.vstack(total_actions)

        # Save to npz if any failed rollouts exist
        if len(failed_traj_lengths) > 0:
            save_path = os.path.join(self.logdir, "failed_rollouts.npz")
            print(f"[Eval] Saving FAILED rollouts to {save_path}")
            print(f"[Eval] Num failed rollouts: {len(failed_traj_lengths)}")
            np.savez(save_path, states=total_states, actions=total_actions, traj_lengths=failed_traj_lengths)

        # ==========================================================
        #   PLOT TRAJECTORIES (unchanged)
        # ==========================================================
        if self.traj_plotter is not None:
            self.traj_plotter(
                obs_full_trajs=obs_full_trajs,
                n_render=self.n_render,
                max_episode_steps=self.max_episode_steps,
                render_dir=self.render_dir,
                itr=0,
            )

        # Log loss and save metrics
        time = timer()
        log.info(
            f"eval: num episode {num_episode_finished:4d} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            time=time,
        )
