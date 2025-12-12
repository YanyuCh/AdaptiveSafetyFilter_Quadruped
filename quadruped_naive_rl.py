# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for training SAC.
"""

from typing import Optional, Callable, Dict, Union, List
import time
import sys
import os
import numpy as np
import torch
import wandb

from quadruped_base_training import BaseTraining
from obstacle_avoidance_navigation_env import ObstacleAvoidanceNavigation

# Add ISAACS-main directory to path for importing agent modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ISAACS'))
from agent.sac import SAC


class NaiveRL(BaseTraining):
  policy: SAC

  def __init__(
      self, cfg_agent, cfg_train, cfg_arch, cfg_env, verbose: bool = True
  ):
    super().__init__(cfg_agent, cfg_env)

    print("= Constructing policy agent")
    self.policy = SAC(cfg_train, cfg_arch, self.rng)
    self.policy.build_network(verbose=verbose)

    self.warmup_action_range = np.array(
        cfg_agent.warmup_action_range, dtype=np.float32
    )

    # alias
    self.module_all = [self.policy]

  @property
  def has_backup(self):
    return False

  def learn(
      self, env: ObstacleAvoidanceNavigation, reset_kwargs: Optional[Dict] = None,
      action_kwargs: Optional[Dict] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None,
      visualize_callback: Optional[Callable] = None
  ):
    env = self.init_learn(env)

    start_learning = time.time()
    if reset_kwargs is None:
      reset_kwargs = {}
    obs_all = env.reset_multiple(torch.arange(self.n_envs, device=env.device), mode='train')
    while self.cnt_step <= self.max_steps:
      # Selects action.
      with torch.no_grad():
        if self.cnt_step < self.min_steps_b4_opt:
          action_all = self.rng.uniform(
              low=self.warmup_action_range[:, 0],
              high=self.warmup_action_range[:, 1],
              size=(self.n_envs, self.warmup_action_range.shape[0])
          )
          action_all = torch.FloatTensor(action_all).to(self.device)
        else:
          # Get physical parameters for all envs
          friction_all = env.friction_coeffs[:, 0]  # Shape: (n_envs,)
          payload_all = env.payloads  # Shape: (n_envs,)
          append_all = torch.stack([friction_all, payload_all], dim=1)  # Shape: (n_envs, 2)

          action_all, _ = self.policy.actor.net.sample(
              obs_all.float().to(self.device),
              append=append_all.to(self.device),
              latent=None
          )

      # Interacts with the env.
      obs_nxt_all, r_all, done_all, info_all = env.step(action_all)
      # Store all transitions
      for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        self.store_transition(
            obs_all[[env_idx]],
            {self.policy.actor.net_name: action_all[[env_idx]]},
            r_all[env_idx], obs_nxt_all[[env_idx]], done, info
        )

      # Batch process terminations and resets
      done_indices = torch.where(done_all)[0]
      if len(done_indices) > 0:
        # Batch reset all done environments
        new_obs = env.reset_multiple(done_indices, mode='train')
        obs_nxt_all[done_indices] = new_obs

        # Batch count safety violations and episodes
        g_x_values = torch.tensor([info_all[i]['g_x'] for i in done_indices], device=self.device)
        num_violations = torch.sum(g_x_values > 0).item()
        self.cnt_safety_violation += num_violations
        self.cnt_num_episode += len(done_indices)

      self.violation_record.append(self.cnt_safety_violation)
      self.episode_record.append(self.cnt_num_episode)
      obs_all = obs_nxt_all

      # Optimizes NNs and checks performance.
      if (
          self.cnt_step >= self.min_steps_b4_opt
          and self.cnt_opt_period >= self.opt_freq
      ):
        print(f"Updates at sample step {self.cnt_step}")
        self.cnt_opt_period = 0
        loss_critic_dict, loss_actor_dict = self.update(
            self.num_update_per_opt
        )
        loss = [
            loss_critic_dict['central'], loss_actor_dict['ctrl'][0],
            loss_actor_dict['ctrl'][1], loss_actor_dict['ctrl'][2]
        ]
        self.train_record.append(loss)
        self.cnt_opt += 1  # Counts number of optimization.
        if self.cfg_agent.use_wandb:
          log_dict = {
              "loss/critic": loss[0],
              "loss/policy": loss[1],
              "loss/entropy": loss[2],
              "loss/alpha": loss[3],
              "metrics/cnt_safety_violation": self.cnt_safety_violation,
              "metrics/cnt_num_episode": self.cnt_num_episode,
              "hyper_parameters/alpha": self.policy.actor.alpha,
              "hyper_parameters/gamma": self.policy.critic.gamma,
          }
          wandb.log(log_dict, step=self.cnt_step, commit=False)

        # Checks after fixed number of gradient updates.
        if self.cnt_opt % self.check_opt_freq == 0 or self.first_update:
          # Updates the high-level policy in the environment with the newest policy.
          env.hl_policy.update_policy(self.policy.actor.net)

          self.check(
              env=env, action_kwargs=action_kwargs,
              rollout_step_callback=rollout_step_callback,
              rollout_episode_callback=rollout_episode_callback,
              visualize_callback=visualize_callback
          )

          # Resets anyway.
          obs_all = env.reset_multiple(torch.arange(self.n_envs, device=env.device), mode='train')

      # Updates counter.
      self.cnt_step += self.n_envs
      self.cnt_opt_period += self.n_envs

      # Updates gamma, lr, etc.
      for _ in range(self.n_envs):
        self.policy.update_hyper_param()

    self.save(env, force_save=True)
    end_learning = time.time()
    time_learning = end_learning - start_learning
    print('\nLearning: {:.1f}'.format(time_learning))

    train_record = np.array(self.train_record)
    train_progress = np.array(self.train_progress)
    violation_record = np.array(self.violation_record)
    episode_record = np.array(self.episode_record)
    return (
        train_record, train_progress, violation_record, episode_record,
        self.pq_top_k
    )

  def restore(self, step: int, model_folder: str):
    super().restore(step, model_folder, "agent")

  def save(
      self, env: ObstacleAvoidanceNavigation, force_save: bool = False,
      reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      action_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
      rollout_step_callback: Optional[Callable] = None,
      rollout_episode_callback: Optional[Callable] = None
  ) -> Dict:
    if force_save:
      info = {}
      metric = 0.
    else:
      # Select representative evaluation environments using grid strategy
      selected_env_ids, _, _ = env.select_eval_envs(
          strategy='grid',
          num_f_points=getattr(self.cfg_agent, 'num_eval_f_points', 5),
          num_m_points=getattr(self.cfg_agent, 'num_eval_m_points', 21)
      )

      # Get evaluation parameters from config
      num_trajectories_per_env = getattr(self.cfg_agent, 'num_trajectories_per_env', 10)
      T_rollout_s = getattr(self.cfg_agent, 'eval_timeout_s', 3.0)
      rollout_end_criterion: str = self.cfg_agent.rollout_end_criterion

      # Simulate trajectories on selected environments
      _, results, length = env.simulate_trajectories(
          selected_env_ids=selected_env_ids,
          num_trajectories_per_env=num_trajectories_per_env,
          T_rollout_s=T_rollout_s,
          end_criterion=rollout_end_criterion,
          eval_ranges=None,
          action_kwargs=action_kwargs_list,
          rollout_step_callback=rollout_step_callback,
          rollout_episode_callback=rollout_episode_callback,
          use_tqdm=True
      )

      # Calculate total number of trajectories evaluated
      total_num_eval = len(selected_env_ids) * num_trajectories_per_env

      # Compute metrics
      safe_rate = np.sum(results != -1) / total_num_eval
      if rollout_end_criterion == "reach-avoid":
        success_rate = np.sum(results == 1) / total_num_eval
        metric = success_rate
        info = dict(
            safe_rate=safe_rate, ep_length=np.mean(length), metric=metric,
            num_eval_envs=len(selected_env_ids),
            total_trajectories=total_num_eval
        )
      else:
        metric = safe_rate
        info = dict(
            ep_length=np.mean(length), metric=metric,
            num_eval_envs=len(selected_env_ids),
            total_trajectories=total_num_eval
        )

    if self.policy.actor_type == 'max':
      # Maximizes cost -> minimizes safe/success rate.
      super()._save(metric=1 - metric, force_save=force_save)
    else:
      super()._save(metric=metric, force_save=force_save)
    return info
