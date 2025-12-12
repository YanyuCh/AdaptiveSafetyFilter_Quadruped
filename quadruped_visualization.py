# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Callable, Optional, Union, List, Any, Tuple
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import torch

from obstacle_avoidance_navigation_env import ObstacleAvoidanceNavigation
from shapely.geometry import Polygon, MultiPoint
from matplotlib import pyplot as plt


def get_values(
    env: ObstacleAvoidanceNavigation,
    critic: Callable[[np.ndarray, Optional[Union[np.ndarray, torch.Tensor]]], np.ndarray],
    xs: np.ndarray, 
    ys: np.ndarray,
    heading: float,
    v_forward: float,              # Forward velocity in body frame (m/s)
    w_yaw: float,                  # Yaw rate in body frame (rad/s)
    default_joint_pos_dev: np.ndarray,  # Shape: (12,) - deviation from default
    default_joint_vel: np.ndarray,      # Shape: (12,)
    friction_coeff: float,
    mass_payload: float,
    batch_size: int,
    fail_value: float = 1.
):
    """
    Compute value function over a 2D grid of (x, y) positions using batched operations.
    
    Args:
        env: ObstacleAvoidanceNavigation environment
        critic: Value function that takes (obs_batch, append) -> values
        xs: 1D array of x coordinates in LOCAL frame (unnormalized), shape (nx,)
        ys: 1D array of y coordinates in LOCAL frame (unnormalized), shape (ny,)
        heading: Heading angle in radians (unnormalized)
        v_forward: Forward velocity in body frame (unnormalized)
        w_yaw: Yaw rate in body frame (unnormalized)
        default_joint_pos_dev: Joint position deviations from default, shape (12,)
        default_joint_vel: Joint velocities, shape (12,)
        friction_coeff: Friction coefficient (f)
        mass_payload: Payload mass (m)
        batch_size: Batch size for critic evaluation
        fail_value: Value for constraint-violating states
        
    Returns:
        values: 2D array (len(xs), len(ys)) of value function values
    """
    nx = len(xs)
    ny = len(ys)
    
    # ============================================================
    # STEP 1: Create meshgrid of all (x, y) points
    # ============================================================
    X, Y = np.meshgrid(xs, ys, indexing='ij')  # Shape: (nx, ny)
    X_flat = X.flatten()  # Shape: (nx * ny,)
    Y_flat = Y.flatten()  # Shape: (nx * ny,)
    num_points = len(X_flat)
    
    # ============================================================
    # STEP 2: BATCHED CONSTRAINT EVALUATION
    # ============================================================
    # Construct state tensor for ALL points
    # Shape: (num_points, 1, 3) for (num_envs, num_steps, num_states)
    # Using [x, y, heading] - though constraint only uses x, y
    states_constraint = torch.stack([
        torch.from_numpy(X_flat).float(),
        torch.from_numpy(Y_flat).float(),
        torch.full_like(torch.from_numpy(X_flat), heading).float()
    ], dim=1).unsqueeze(1).to(env.device)  # Shape: (num_points, 1, 3)

    # Dummy controls: shape (num_points, 1, 2)
    dummy_controls = torch.zeros(num_points, 1, 2, dtype=torch.float32, device=env.device)

    # Dummy time indices: shape (num_points, 1)
    dummy_time_indices = torch.zeros(num_points, 1, dtype=torch.int32, device=env.device)
    
    # Compute constraints for ALL points in one batch
    # Output shape: (num_points, num_steps) = (num_points, 1)
    g_x_all = env.hl_constraint.get_cost(
        states_constraint,
        dummy_controls,
        dummy_time_indices
    )  # Shape: (num_points, 1)

    # Squeeze to get shape (num_points,)
    if g_x_all.dim() == 2:
        if g_x_all.shape[1] == 1:
            g_x_all = g_x_all.squeeze(1)  # (num_points, 1) -> (num_points,)
        else:
            g_x_all = g_x_all[:, -1]      # (num_points, num_steps) -> take last step
      
    # Create safety mask: True for safe points (g_x < failure_thr)
    safe_mask = g_x_all < env.failure_thr  # Shape: (num_points,) - boolean tensor
    
    # Get indices of safe points
    safe_indices = torch.nonzero(safe_mask, as_tuple=False).squeeze(1)  # Shape: (num_safe,)
    num_safe = len(safe_indices)
    
    # ============================================================
    # STEP 3: CONSTRUCT OBSERVATIONS FOR SAFE POINTS ONLY
    # ============================================================
    if num_safe == 0:
        # No safe points - return all fail_value
        return np.full((nx, ny), fill_value=fail_value)
    
    # Extract safe x, y coordinates
    X_safe = X_flat[safe_indices.cpu().numpy()]  # Shape: (num_safe,)
    Y_safe = Y_flat[safe_indices.cpu().numpy()]  # Shape: (num_safe,)
    
    # Normalize x, y positions
    X_norm = 2 * (X_safe - env.x_lowerbound_local) / \
             (env.x_upperbound_local - env.x_lowerbound_local) - 1  # Shape: (num_safe,)
    Y_norm = 2 * (Y_safe - env.y_lowerbound_local) / \
             (env.y_upperbound_local - env.y_lowerbound_local) - 1  # Shape: (num_safe,)
    
    # Base velocities (body frame) - same for all points
    base_lin_vel = np.array([v_forward, 0.0, 0.0])
    base_ang_vel = np.array([0.0, 0.0, w_yaw])
    
    # Scale velocities
    base_lin_vel_scaled = base_lin_vel * env.obs_scales.lin_vel
    base_ang_vel_scaled = base_ang_vel * env.obs_scales.ang_vel
    
    # Scale joint states
    joint_pos_scaled = default_joint_pos_dev * env.obs_scales.dof_pos  # Shape: (12,)
    joint_vel_scaled = default_joint_vel * env.obs_scales.dof_vel      # Shape: (12,)
    
    # Assemble observations for all safe points
    # Create base observation (constant parts)
    base_obs = np.concatenate([
        [heading],                  # 1D: heading (same for all)
        base_lin_vel_scaled,        # 3D: scaled linear velocity
        base_ang_vel_scaled,        # 3D: scaled angular velocity
        joint_pos_scaled,           # 12D: scaled joint positions
        joint_vel_scaled            # 12D: scaled joint velocities
    ])  # Shape: (31,)
    
    # Broadcast base_obs to all safe points
    obs_all = np.zeros((num_safe, env.num_hl_obs))  # Shape: (num_safe, 33)
    obs_all[:, 0] = X_norm      # x normalized
    obs_all[:, 1] = Y_norm      # y normalized
    obs_all[:, 2:] = base_obs   # Rest of observation (constant)
    
    # ============================================================
    # STEP 4: BATCHED CRITIC EVALUATION
    # ============================================================
    # Prepare append: physical parameters [f, m] for all safe points
    append = np.tile([friction_coeff, mass_payload], (num_safe, 1))  # Shape: (num_safe, 2)
    
    # Evaluate critic in batches to avoid memory issues
    values_safe = np.zeros(num_safe)
    
    for start_idx in range(0, num_safe, batch_size):
        end_idx = min(start_idx + batch_size, num_safe)
        obs_batch = obs_all[start_idx:end_idx]
        append_batch = append[start_idx:end_idx]
        
        # Query critic
        v_batch = critic(obs_batch, append=append_batch)
        values_safe[start_idx:end_idx] = v_batch.flatten()
    
    # ============================================================
    # STEP 5: ASSEMBLE FINAL VALUE GRID
    # ============================================================
    # Initialize with fail_value
    values_flat = np.full(num_points, fill_value=fail_value)
    
    # Fill in values for safe points
    values_flat[safe_indices.cpu().numpy()] = values_safe
    
    # Reshape to 2D grid
    values = values_flat.reshape(nx, ny)
    
    return values


def plot_traj(
    ax, trajectory: np.ndarray, result: int, c: str = 'b', lw: float = 2.,
    zorder: int = 1, vel_scatter: bool = False, s: int = 40
):
  traj_x = trajectory[:, 0]
  traj_y = trajectory[:, 1]

  if vel_scatter:
    vel = trajectory[:, 2]
    ax.scatter(
        traj_x[0], traj_y[0], s=s, c=vel[0], cmap=cm.copper, vmin=0, vmax=2.,
        edgecolor='none', marker='s', zorder=zorder
    )
    ax.scatter(
        traj_x[1:-1], traj_y[1:-1], s=s - 12, c=vel[1:-1], cmap=cm.copper,
        vmin=0, vmax=2., edgecolor='none', marker='o', zorder=zorder
    )
    if result == -1:
      marker_final = 'X'
      edgecolor_final = 'r'
    elif result == 1:
      marker_final = '*'
      edgecolor_final = 'g'
    else:
      marker_final = '^'
      edgecolor_final = 'y'
    ax.scatter(
        traj_x[-1], traj_y[-1], s=s, c=vel[-1], cmap=cm.copper, vmin=0,
        vmax=2., edgecolor=edgecolor_final, marker=marker_final, zorder=zorder
    )
  else:
    ax.scatter(traj_x[0], traj_y[0], s=s, c=c, zorder=zorder)
    ax.plot(traj_x, traj_y, c=c, ls='-', lw=lw, zorder=zorder)

    if result == -1:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='x', c='r', zorder=zorder)
    elif result == 1:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='*', c='g', zorder=zorder)
    else:
      ax.scatter(traj_x[-1], traj_y[-1], s=s, marker='^', c='y', zorder=zorder)
