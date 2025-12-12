# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import List
import os
import sys
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import wandb
import argparse
from functools import partial
from shutil import copyfile
import jax
from omegaconf import OmegaConf
import pickle as pkl
import torch
import glob

from quadruped_naive_rl import NaiveRL
from obstacle_avoidance_navigation_env import ObstacleAvoidanceNavigation
from quadruped_visualization import plot_traj, get_values

# Add paths for ISAACS imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ISAACS'))

from agent.sac import SAC
from simulators import PrintLogger, save_obj

# Add paths for observation-conditioned-reachability imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability', 'utils'))

# Observation-conditioned-reachability/utils imports
from dynamics import Dubins3D
from navigation_task import NavigationTask

# Add paths for walk-these-ways imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability', 'libraries', 'walk-these-ways'))

# Walk-These-Ways imports
from go1_gym.envs.base.legged_robot_config import Cfg

jax.config.update('jax_platform_name', 'cpu')


# ================================================================
# Helper Functions for Loading WTW Policy and Config
# ================================================================

def load_policy(logdir):
    """
    Load pretrained walk-these-ways policy as a low-level controller.

    Args:
        logdir: Directory containing the pretrained checkpoints

    Returns:
        policy: A callable policy function
    """
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_wtw_config_and_policy(label, wtw_runs_root):
    """
    Load Walk-These-Ways config and pretrained policy.

    Args:
        label: Label for the pretrained run (e.g., "gait-conditioned-agility/pretrain-v0/train")
        wtw_runs_root: Root directory for walk-these-ways runs

    Returns:
        cfg_wtw: Loaded and modified Cfg object
        ll_policy: Loaded low-level policy
        logdir: Path to the loaded run directory
    """
    # Find the run directory
    dirs = glob.glob(f"{wtw_runs_root}/{label}/*")
    if len(dirs) == 0:
        raise ValueError(f"No runs found at {wtw_runs_root}/{label}/*")
    logdir = sorted(dirs)[0]

    print(f"Loading from: {logdir}")

    # Load policy
    ll_policy = load_policy(logdir)
    print("Loaded low-level policy")

    # Load config
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg_dict = pkl_cfg["Cfg"]

        # Apply config to Cfg class
        for key, value in cfg_dict.items():
            if hasattr(Cfg, key) and key != 'command_ranges':
                for key2, value2 in cfg_dict[key].items():
                    setattr(getattr(Cfg, key), key2, value2)
                    
    # turn off DR and early termination for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False
    Cfg.domain_rand.randomize_rigids_after_start = False
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_roll_pitch = False
    
    # redefine env related parameter values
    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1000
    Cfg.env.episode_length_s = 3
    Cfg.env.env_spacing = 16.       # Important to avoid overlap!! (env max length = 12-(-2) = 14 in x)
    
    # redefine terrain related parameter values
    Cfg.terrain.mesh_type = 'plane'
    Cfg.terrain.x_init_range = 0
    Cfg.terrain.y_init_range = 0
    Cfg.terrain.yaw_init_range = 0
    Cfg.terrain.x_init_offset = 0
    Cfg.terrain.y_init_offset = 0

    # lag and control
    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    # set payload
    Cfg.domain_rand.randomize_base_mass = True
    Cfg.domain_rand.added_mass_range = [-1., 1.]   

    # set friction
    Cfg.domain_rand.randomize_friction = True
    Cfg.domain_rand.friction_range = [0.1, 0.5]

    cfg_wtw = Cfg
    print("Loaded Walk-These-Ways config")

    return cfg_wtw, ll_policy, logdir


# region: local functions
def visualize(
    env: ObstacleAvoidanceNavigation,
    policy: SAC,
    fig_path: str,
    friction_list: List[float],
    payload_list: List[float],
    subfigsz_x: float = 4.5,
    subfigsz_y: float = 3.5,
    nx: int = 150,
    ny: int = 150,
    batch_size: int = 512,
    cmap: str = 'seismic',
    vmin: float = -.25,
    vmax: float = .25,
    alpha: float = 0.5,
    fontsize: int = 16
):
    """
    Create a 5x5 grid of value function plots for different friction and payload combinations.

    Args:
        env: ObstacleAvoidanceNavigation environment
        policy: SAC policy with value function
        fig_path: Path to save figure
        friction_list: List of friction coefficients [0.1, 0.2, 0.3, 0.4, 0.5]
        payload_list: List of payload masses [-1.0, -0.5, 0.0, 0.5, 1.0]
        subfigsz_x, subfigsz_y: Subplot size
        nx, ny: Grid resolution for value function
        batch_size: Batch size for critic evaluation
        cmap: Colormap for value function
        vmin, vmax: Value function color limits
        alpha: Transparency for value function
        fontsize: Font size for labels
    """
    # Create 5x5 grid (payload x friction)
    n_row = len(payload_list)  # 5 rows for different payloads
    n_col = len(friction_list)  # 5 columns for different frictions
    figsize = (subfigsz_x * n_col, subfigsz_y * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=True, sharey=True)

    vmin_label = vmin
    vmax_label = vmax
    vmean_label = 0

    # Extended bounds to include boundary obstacles
    x_min_plot = -2.5
    x_max_plot = 12.5
    y_min_plot = -5.5
    y_max_plot = 5.5

    # Create x-y grid for value function in LOCAL frame (extended bounds)
    xs = np.linspace(x_min_plot, x_max_plot, nx)
    ys = np.linspace(y_min_plot, y_max_plot, ny)

    # Fixed high-level observation parameters for value function
    # All in body frame: heading, base velocities, joint states
    heading = 0.0  # 0 radians
    v_forward = 1.0  # 1 m/s forward velocity
    w_yaw = 0.0  # No yaw rate
    default_joint_pos_dev = np.zeros(12)  # No joint deviation
    default_joint_vel = np.zeros(12)  # No joint velocity

    # Plot each subplot
    for i, friction in enumerate(friction_list):
        # Set column title
        axes[0][i].set_title(f"Friction: {friction:.2f}", fontsize=fontsize)

        for j, payload in enumerate(payload_list):
            ax = axes[j][i]

            # Set row label
            if i == 0:
                ax.set_ylabel(f"Payload: {payload:.2f} kg", fontsize=fontsize)

            # Compute value function for current (f, m) combination
            values = get_values(
                env=env,
                critic=policy.value,
                xs=xs,
                ys=ys,
                heading=heading,
                v_forward=v_forward,
                w_yaw=w_yaw,
                default_joint_pos_dev=default_joint_pos_dev,
                default_joint_vel=default_joint_vel,
                friction_coeff=friction,
                mass_payload=payload,
                batch_size=batch_size,
                fail_value=vmax
            )

            # Plot value function as heatmap
            extent = [x_min_plot, x_max_plot, y_min_plot, y_max_plot]
            im = ax.imshow(
                values.T, interpolation='none', extent=extent,
                origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                zorder=-1, alpha=alpha
            )

            # Plot obstacles if environment has them
            if hasattr(env, 'task') and env.task is not None:
                if hasattr(env.task, 'environment') and hasattr(env.task.environment, 'obstacles'):
                    for obstacle in env.task.environment.obstacles:
                        # Obstacles are in local frame
                        if hasattr(obstacle, 'center') and hasattr(obstacle, 'radius'):
                            # CircularObstacle - draw only outline (no fill)
                            circle = Circle(
                                obstacle.center, obstacle.radius,
                                facecolor='none',  # Transparent fill
                                edgecolor='black',
                                linewidth=2.,
                                zorder=3  # Draw on top of value function
                            )
                            ax.add_patch(circle)
                        elif hasattr(obstacle, 'center') and hasattr(obstacle, 'length'):
                            # BoxObstacle - draw only outline (no fill)
                            cx, cy = obstacle.center[0], obstacle.center[1]
                            length = obstacle.length
                            width = obstacle.width
                            angle = obstacle.angle

                            # Corners in obstacle frame (before rotation)
                            corners_local = np.array([
                                [-length/2, -width/2],
                                [length/2, -width/2],
                                [length/2, width/2],
                                [-length/2, width/2]
                            ])

                            # Rotation matrix
                            cos_a = np.cos(angle)
                            sin_a = np.sin(angle)
                            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

                            # Rotate and translate
                            corners_global = corners_local @ rot_matrix.T + np.array([cx, cy])

                            polygon = Polygon(
                                corners_global, closed=True,
                                facecolor='none',  # Transparent fill
                                edgecolor='black',
                                linewidth=2.,
                                zorder=3  # Draw on top of value function
                            )
                            ax.add_patch(polygon)

            # Set axis properties
            ax.axis(extent)
            ax.set_xticks(np.linspace(extent[0], extent[1], 5))
            ax.set_yticks(np.linspace(extent[2], extent[3], 5))
            ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)
            ax.set_aspect('equal', adjustable='box')

    # Add colorbar
    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.08, hspace=0.12
    )
    cbar_ax = fig.add_axes([0.96, 0.05, 0.015, 0.90])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[vmin, 0, vmax])
    v_ticklabels = np.around(np.array([vmin_label, vmean_label, vmax_label]), 2)
    cbar.ax.set_yticklabels(labels=v_ticklabels, fontsize=fontsize - 4)
    cbar.set_label('Value Function', fontsize=fontsize, rotation=270, labelpad=20)

    # Save figure
    print(f"Saving visualization to {fig_path}")
    fig.savefig(fig_path, dpi=400, bbox_inches='tight')
    plt.close('all')


# endregion


def main(
    wtw_label: str,
    isaacs_config_file: str,
    env_id: str,
    wtw_runs_root: str = None,
    env_dir: str = None
):
    """
    Main training function.

    Args:
        wtw_label: Label for walk-these-ways pretrained run (e.g., "gait-conditioned-agility/pretrain-v0/train")
        isaacs_config_file: Path to ISAACS config YAML file
        env_id: Environment ID for loading environment.pickle (e.g., "0", "1", ..., "99")
        wtw_runs_root: Root directory for walk-these-ways runs (optional)
        env_dir: Directory containing environment_{env_id} folders (optional)
    """

    # ================================================================
    # STEP 1: Set Default Paths
    # ================================================================

    # Set default WTW runs root if not provided
    if wtw_runs_root is None:
        wtw_runs_root = os.path.join(
            os.path.dirname(__file__),
            '..',
            'observation-conditioned-reachability',
            'libraries',
            'walk-these-ways',
            'runs'
        )

    # Set default environment directory if not provided
    if env_dir is None:
        env_dir = os.path.join(
            os.path.dirname(__file__),
            '..',
            'observation-conditioned-reachability',
            'data',
            'environments',
            'simulation'
        )

    # ================================================================
    # STEP 2: Load ISAACS Config
    # ================================================================

    print("\n" + "="*70)
    print("STEP 1: Loading ISAACS Config")
    print("="*70)

    cfg_isaacs = OmegaConf.load(isaacs_config_file)
    cfg_isaacs.train.device = cfg_isaacs.solver.device

    os.makedirs(cfg_isaacs.solver.out_folder, exist_ok=True)
    copyfile(isaacs_config_file, os.path.join(cfg_isaacs.solver.out_folder, 'config.yaml'))
    log_path = os.path.join(cfg_isaacs.solver.out_folder, 'log.txt')
    if os.path.exists(log_path):
        os.remove(log_path)
    sys.stdout = PrintLogger(log_path)
    sys.stderr = PrintLogger(log_path)

    print(f"Loaded ISAACS config from: {isaacs_config_file}")
    print(f"Output folder: {cfg_isaacs.solver.out_folder}")
    print(f"Device: {cfg_isaacs.train.device}")

    # ================================================================
    # STEP 3: Load Walk-These-Ways Config and LL Policy
    # ================================================================

    print("\n" + "="*70)
    print("STEP 2: Loading Walk-These-Ways Config and LL Policy")
    print("="*70)

    cfg_wtw, ll_policy, logdir = load_wtw_config_and_policy(wtw_label, wtw_runs_root)

    print(f"Successfully loaded from: {logdir}")

    # ================================================================
    # STEP 4: Load Environment for NavigationTask
    # ================================================================

    print("\n" + "="*70)
    print("STEP 3: Loading Environment for NavigationTask")
    print("="*70)

    # Load environment from pickle file (matching run_sims.py structure)
    env_pickle_path = os.path.join(env_dir, env_id, 'environment.pickle')

    if not os.path.exists(env_pickle_path):
        raise FileNotFoundError(f"Environment file not found: {env_pickle_path}")

    with open(env_pickle_path, 'rb') as f:
        env_for_task = pkl.load(f)

    print(f"Loaded environment from: {env_pickle_path}")

    # ================================================================
    # STEP 5: Create NavigationTask
    # ================================================================

    print("\n" + "="*70)
    print("STEP 4: Creating NavigationTask")
    print("="*70)

    dynamics = Dubins3D()

    task = NavigationTask(
        robot_radius=0.35,  # Fixed value from run_sims.py
        goal_position=np.array([10.0, 0.0]),  # Fixed value from run_sims.py
        goal_radius=0.5,  # Fixed value from run_sims.py
        environment=env_for_task,
        dynamics=dynamics
    )

    print(f"Created NavigationTask:")
    print(f"  - Robot radius: {task.robot_radius}")
    print(f"  - Goal position: {task.goal_position}")
    print(f"  - Goal radius: {task.goal_radius}")
    print(f"  - Number of obstacles: {len(task.environment.obstacles)}")

    # ================================================================
    # STEP 6: Create ObstacleAvoidanceNavigation Environment
    # ================================================================

    print("\n" + "="*70)
    print("STEP 5: Creating ObstacleAvoidanceNavigation Environment")
    print("="*70)

    env = ObstacleAvoidanceNavigation(
        sim_device=cfg_isaacs.train.device,
        headless=True,  # Always headless for training
        num_envs=None,  # Will be set from cfg_wtw.env.num_envs
        cfg=cfg_wtw,  # Walk-these-ways config
        eval_cfg=None,  # No eval config
        task=task,  # NavigationTask we just created
        ll_policy=ll_policy,  # Pretrained LL policy
        cfg_cost=cfg_isaacs.cost,  # From ISAACS config
        cfg_arch=cfg_isaacs.arch,  # From ISAACS config
        cfg_env=cfg_isaacs.environment  # From ISAACS config (NOT cfg.solver!)
    )
    env.step_keep_constraints = False

    print(f"Environment created successfully:")
    print(f"  - Number of environments: {env.num_envs}")
    print(f"  - High-level observation dim: {env.num_hl_obs}")
    print(f"  - High-level action dim: {env.num_hl_actions}")
    print(f"  - Device: {env.device}")

    # ================================================================
    # STEP 7: Setup WandB (if enabled)
    # ================================================================

    if cfg_isaacs.solver.use_wandb:
        print("\n" + "="*70)
        print("STEP 6: Setting up Weights & Biases")
        print("="*70)

        wandb.init(
            entity='CassieC',
            project=cfg_isaacs.solver.project_name,
            name=cfg_isaacs.solver.name
        )
        tmp_cfg = {
            'environment': OmegaConf.to_container(cfg_isaacs.environment),
            'solver': OmegaConf.to_container(cfg_isaacs.solver),
            'arch': OmegaConf.to_container(cfg_isaacs.arch),
            'train': OmegaConf.to_container(cfg_isaacs.train)
        }
        wandb.config.update(tmp_cfg)
        print("WandB initialized")

    # ================================================================
    # STEP 8: Construct Solver
    # ================================================================

    print("\n" + "="*70)
    print("STEP 7: Constructing Solver")
    print("="*70)

    solver = NaiveRL(
        cfg_isaacs.solver,
        cfg_isaacs.train,
        cfg_isaacs.arch,
        cfg_isaacs.environment
    )
    policy = solver.policy

    # Initialize high-level policy in environment
    env.init_hl_policy(
        cfg_hl_policy=SimpleNamespace(device=policy.device),
        actor=policy.actor.net
    )

    print(
        f'\nTotal parameters in actor: {sum(p.numel() for p in policy.actor.net.parameters() if p.requires_grad)}'
    )
    print(f"Training device: {cfg_isaacs.train.device}")
    print(f"Policy device: {policy.device}")
    print(f"Critic using CUDA: {next(policy.critic.net.parameters()).is_cuda}")

    # ================================================================
    # STEP 9: Training
    # ================================================================

    print("\n" + "="*70)
    print("STEP 8: Starting Training")
    print("="*70)

    # Define friction and payload combinations for visualization
    friction_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    payload_list = [-1.0, -0.5, 0.0, 0.5, 1.0]

    visualize_callback = partial(
        visualize,
        friction_list=friction_list,
        payload_list=payload_list,
        nx=cfg_isaacs.solver.get('cmap_res_x', 150),
        ny=cfg_isaacs.solver.get('cmap_res_y', 150),
        subfigsz_x=cfg_isaacs.solver.get('fig_size_x', 4.5),
        subfigsz_y=cfg_isaacs.solver.get('fig_size_y', 3.5),
        vmin=-cfg_isaacs.environment.g_x_fail,
        vmax=cfg_isaacs.environment.g_x_fail
    )

    train_record, train_progress, violation_record, episode_record, pq_top_k = (
        solver.learn(env, visualize_callback=visualize_callback)
    )

    # ================================================================
    # STEP 10: Save Training Results
    # ================================================================

    print("\n" + "="*70)
    print("STEP 9: Saving Training Results")
    print("="*70)

    train_dict = {}
    train_dict['train_record'] = train_record
    train_dict['train_progress'] = train_progress
    train_dict['violation_record'] = violation_record
    train_dict['episode_record'] = episode_record
    train_dict['pq_top_k'] = list(pq_top_k.queue)
    save_obj(train_dict, os.path.join(cfg_isaacs.solver.out_folder, 'train'))

    print(f"Training results saved to: {cfg_isaacs.solver.out_folder}")
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RARL on Quadruped Navigation with Adaptive Safety Filter"
    )

    # Walk-These-Ways arguments
    parser.add_argument(
        "--wtw_label",
        type=str,
        required=True,
        help="Label for walk-these-ways pretrained run (e.g., 'gait-conditioned-agility/pretrain-v0/train')"
    )
    parser.add_argument(
        "--wtw_runs_root",
        type=str,
        default=None,
        help="Root directory for walk-these-ways runs (optional, uses default if not specified)"
    )

    # ISAACS config argument
    parser.add_argument(
        "-cf", "--config_file",
        help="ISAACS config file path",
        type=str,
        default=os.path.join("config", "sac.yaml")
    )

    # Environment/Task arguments
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="Environment ID for loading environment.pickle (e.g., '0', '1', ..., '99')"
    )
    parser.add_argument(
        "--env_dir",
        type=str,
        default=None,
        help="Directory containing environment_{env_id} folders (optional, uses default if not specified)"
    )

    args = parser.parse_args()

    main(
        wtw_label=args.wtw_label,
        isaacs_config_file=args.config_file,
        env_id=args.env_id,
        wtw_runs_root=args.wtw_runs_root,
        env_dir=args.env_dir
    )
