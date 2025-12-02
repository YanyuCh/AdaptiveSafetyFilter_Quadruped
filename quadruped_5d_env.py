import glob
import torch
import pickle as pkl
import numpy as np
import sys
import os

# Add paths for external dependencies
# Path to observation-conditioned-reachability for obstacle classes
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability'))

# Path to walk-these-ways for go1_gym
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability', 'libraries', 'walk-these-ways'))

# Isaac Gym related imports
# These work directly once Isaac Gym is installed in conda/virtual environment
from isaacgym import gymtorch, gymapi, gymutil, torch_utils
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis

# walk-these-ways related imports
# These require the path append above to locate go1_gym
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

# ISAACS related imports (from current directory)
from dubins3d_cost import Dubins3d_Cost, Dubins3d_Constraint

# Obstacle imports (from observation-conditioned-reachability)
from utils.simulation_utils.obstacle import CircularObstacle, BoxObstacle
from utils.hj_reachability_utils.obstacle import CylindricalObstacle2D

VIS_LINE_Z = 0.01  # slightly above ground to ensure no obscuring issues


class Quadruped5DEnv:
    """
    Isaac Gym-based quadruped environment compatible with ISAACS framework.

    This class wraps the VelocityTrackingEasyEnv with a Dubins3D interface
    and provides policy attachment capabilities for ISAACS training/evaluation.

    The environment manages:
    - Parallel Isaac Gym simulation environments
    - Pretrained walk-these-ways low-level controller
    - Dubins3D state representation and control interface
    - Visualization of trajectories, goals, and obstacles
    - Policy attachment for high-level learning
    """

    def __init__(self, label, task, payload_range, friction_range,
                 headless=False, body_color=(1, 1, 1)):
        """
        Initialize the Quadruped5DEnv.

        Args:
            label: string, pretrained walk-these-ways policy name
            task: object NavigationTask, contains:
                  - robot_radius: float
                  - goal_position: array-like
                  - goal_radius: float
                  - environment: contains obstacle info
                  - dynamics: Dubins3D (also not used in Isaac Gym?)
            payload_range: List [min, max] defining range for payload randomization
            friction_range: List [min, max] defining range for friction randomization
            headless: bool, True -> No render, False -> render
            body_color: tuple (r, g, b), default color of the quadruped
        """
        self.label = label
        self.task = task
        self.payload_range = payload_range
        self.friction_range = friction_range
        self.headless = headless
        self.body_color = body_color

        # Load the pretrained policy
        self._load_policy()

        # Initialize the base environment
        self._initialize_base_env()

        # Setup visualization
        self._setup_visualization()

        # Initialize policy placeholder for ISAACS
        self.high_level_policy = None

    def _load_policy(self):
        """Load the pretrained walk-these-ways policy as a low-level controller."""
        dirs = glob.glob(f"/observation-conditioned-reachability/libraries/walk-these-ways/runs/{self.label}/*")
        self.logdir = sorted(dirs)[0]

        # Load JIT-compiled policy modules
        body = torch.jit.load(self.logdir + '/checkpoints/body_latest.jit')
        adaptation_module = torch.jit.load(self.logdir + '/checkpoints/adaptation_module_latest.jit')

        # Define policy function
        def policy(obs, info={}):
            latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
            action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
            info['latent'] = latent
            return action

        self.low_level_policy = policy

    def _initialize_base_env(self):
        """Initialize the base Isaac Gym environment with proper configuration."""
        # Load configuration
        with open(self.logdir + "/parameters.pkl", 'rb') as file:
            pkl_cfg = pkl.load(file)
            cfg = pkl_cfg["Cfg"]

            for key, value in cfg.items():
                if hasattr(Cfg, key) and key != 'command_ranges':
                    for key2, value2 in cfg[key].items():
                        setattr(getattr(Cfg, key), key2, value2)

        # Turn off domain randomization for evaluation
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

        # Redefine environment parameters
        Cfg.env.num_recording_envs = 1
        Cfg.env.num_envs = 1000
        Cfg.env.episode_length_s = 60

        # Redefine terrain parameters
        Cfg.terrain.mesh_type = 'plane'
        Cfg.terrain.x_init_range = 0
        Cfg.terrain.y_init_range = 0
        Cfg.terrain.yaw_init_range = 0
        Cfg.terrain.x_init_offset = 0
        Cfg.terrain.y_init_offset = 0

        # Lag and control
        Cfg.domain_rand.lag_timesteps = 6
        Cfg.domain_rand.randomize_lag_timesteps = True
        Cfg.control.control_type = "actuator_net"

        # Set payload randomization
        Cfg.domain_rand.randomize_base_mass = True
        Cfg.domain_rand.added_mass_range = self.payload_range

        # Set friction randomization
        Cfg.domain_rand.randomize_friction = True
        Cfg.domain_rand.friction_range = self.friction_range

        # Create base environment
        self.env = VelocityTrackingEasyEnv(
            sim_device=f'cuda:{torch.cuda.current_device()}',
            headless=self.headless,
            cfg=Cfg,
            task=self.task
        )
        self.env = HistoryWrapper(self.env)

        # Set body color
        body_rigid_shape_indices = [0]
        for i in body_rigid_shape_indices:
            self.env.gym.set_rigid_body_color(
                self.env.envs[0],
                self.env.actor_handles[0],
                i,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(*self.body_color)
            )

        self.body_rigid_shape_indices = body_rigid_shape_indices

        # Expose commonly used attributes
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.gym = self.env.gym
        self.viewer = self.env.viewer
        self.sim = self.env.sim

    def _setup_visualization(self):
        """Setup visualization lines and shapes for goals and obstacles."""
        # Initialize line storage
        self.perm_lines = []
        self.temp_lines = []

        # Add goal position
        self._add_perm_circ(
            self.task.goal_position[0],
            self.task.goal_position[1],
            self.task.goal_radius,
            100,
            (0, 1, 0)
        )

        # Add obstacles
        obstacle_color = (1, 0, 0)
        for obstacle in self.task.environment.obstacles:
            if isinstance(obstacle, (CylindricalObstacle2D, CircularObstacle)):
                self._add_perm_circ(
                    obstacle.center[0],
                    obstacle.center[1],
                    obstacle.radius,
                    100,
                    obstacle_color
                )
            elif isinstance(obstacle, BoxObstacle):
                self._add_perm_box(
                    *obstacle.center,
                    obstacle.angle,
                    obstacle.length,
                    obstacle.width,
                    obstacle_color
                )
            else:
                raise NotImplementedError(f"Obstacle type {type(obstacle)} not supported")

        # Refresh drawings
        self.refresh_drawings()

    # ==================== Visualization Helper Methods ====================

    def _add_line_to_list(self, line_list, x1, y1, x2, y2, c):
        """Add a line segment to a list of lines."""
        line_list.append(((x1, y1, VIS_LINE_Z), (x2, y2, VIS_LINE_Z), c))

    def _add_perm_line(self, x1, y1, x2, y2, c):
        """Add a permanent line segment."""
        self._add_line_to_list(self.perm_lines, x1, y1, x2, y2, c)

    def add_temp_line(self, x1, y1, x2, y2, c):
        """Add a temporary line segment (public method)."""
        self._add_line_to_list(self.temp_lines, x1, y1, x2, y2, c)

    def _add_circ_to_list(self, line_list, x, y, r, n, c):
        """Add a circle approximated by line segments to a list."""
        ts = np.linspace(0, 2*np.pi, num=n)
        xs = r*np.cos(ts) + x
        ys = r*np.sin(ts) + y
        for i in range(n-1):
            self._add_line_to_list(line_list, xs[i], ys[i], xs[i+1], ys[i+1], c)

    def _add_perm_circ(self, x, y, r, n, c):
        """Add a permanent circle."""
        self._add_circ_to_list(self.perm_lines, x, y, r, n, c)

    def add_temp_circ(self, x, y, r, n, c):
        """Add a temporary circle (public method)."""
        self._add_circ_to_list(self.temp_lines, x, y, r, n, c)

    def _add_box_to_list(self, line_list, x, y, th, l, w, c):
        """Add a box approximated by line segments to a list."""
        ego_corners = np.array([
            [-l/2, -w/2],
            [-l/2, +w/2],
            [+l/2, -w/2],
            [+l/2, +w/2],
        ])
        cth, sth = np.cos(th), np.sin(th)
        rot = np.array([
            [cth, sth],
            [-sth, cth],
        ])
        world_corners = np.matmul(rot.T, ego_corners[:, :, np.newaxis]).squeeze(axis=-1) + np.array([x, y])
        self._add_line_to_list(line_list, *world_corners[0], *world_corners[1], c)
        self._add_line_to_list(line_list, *world_corners[0], *world_corners[2], c)
        self._add_line_to_list(line_list, *world_corners[3], *world_corners[1], c)
        self._add_line_to_list(line_list, *world_corners[3], *world_corners[2], c)

    def _add_perm_box(self, x, y, th, l, w, c):
        """Add a permanent box."""
        self._add_box_to_list(self.perm_lines, x, y, th, l, w, c)

    def add_temp_box(self, x, y, th, l, w, c):
        """Add a temporary box (public method)."""
        self._add_box_to_list(self.temp_lines, x, y, th, l, w, c)

    def remove_temp_lines(self):
        """Remove all temporary visualization lines."""
        self.temp_lines = []

    def refresh_drawings(self):
        """Refresh all visualization drawings."""
        self.gym.clear_lines(self.viewer)

        def draw_lines(lines):
            if len(lines) == 0:
                return
            ls = np.empty((len(lines), 2), gymapi.Vec3.dtype)
            cs = np.empty(len(lines), gymapi.Vec3.dtype)
            for i in range(len(lines)):
                ls[i][0] = lines[i][0]
                ls[i][1] = lines[i][1]
                cs[i] = lines[i][2]
            self.gym.add_lines(self.viewer, self.env.envs[0], len(lines), ls, cs)

        draw_lines(self.perm_lines)
        draw_lines(self.temp_lines)

    # ==================== Core Environment Methods ====================

    def set_robot_state(self, x, y, th):
        """
        Set robot state for ALL environments.

        Args:
            x: float or tensor of shape (num_envs,) - x positions
            y: float or tensor of shape (num_envs,) - y positions
            th: float or tensor of shape (num_envs,) - orientations (yaw angles in rad)
        """
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        self.env.root_states[env_ids] = self.env.base_init_state
        self.env.root_states[env_ids, :3] += self.env.env_origins[env_ids]

        # Convert to tensors if float provided
        if not isinstance(x, torch.Tensor):
            x = torch.full((self.env.num_envs,), x, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.full((self.env.num_envs,), y, device=self.device)
        if not isinstance(th, torch.Tensor):
            th = torch.full((self.env.num_envs,), th, device=self.device)

        # Set positions (batched)
        self.env.root_states[env_ids, 0] = x
        self.env.root_states[env_ids, 1] = y

        # Base yaws (batched)
        init_yaws = th.reshape(-1, 1)  # shape (num_envs, 1)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.env.root_states[env_ids, 3:7] = quat

        # Base velocities (set to zero)
        self.env.root_states[env_ids, 7:13] = torch_rand_float(
            0, 0, (len(env_ids), 6), device=self.device
        )  # [7:10]: lin vel, [10:13]: ang vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.env.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def step_dubins3d(self, v, w, obs, env_id=0, add_line=True,
                     line_color=(0, 0, 1), body_color=(1, 1, 1)):
        """
        Step environment using Dubins3D action.

        Args:
            v: float or tensor of shape (num_envs,) - forward velocities
            w: float or tensor of shape (num_envs,) - angular velocities
            obs: observation dictionary
            env_id: which environment to visualize (default 0)
            add_line: whether to add visualization line
            line_color: color for trajectory line
            body_color: color for robot body

        Returns:
            obs: observation dictionary
        """
        # Store previous position for visualization (only for specified env by env_id)
        prev_x, prev_y = self.env.base_pos[env_id, 0].item(), self.env.base_pos[env_id, 1].item()

        gaits = {
            "pronking": [0, 0, 0],
            "trotting": [0.5, 0, 0],
            "bounding": [0, 0.5, 0],
            "pacing": [0, 0, 0.5]
        }

        y_vel_cmd = 0.0
        body_height_cmd = 0.0
        step_frequency_cmd = 3.0
        gait = torch.tensor(gaits["trotting"])
        footswing_height_cmd = 0.08
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25

        with torch.no_grad():
            actions = self.low_level_policy(obs)

        # Convert to tensors if floats provided
        if not isinstance(v, torch.Tensor):
            v = torch.full((self.env.num_envs,), v, device=self.device)
        if not isinstance(w, torch.Tensor):
            w = torch.full((self.env.num_envs,), w, device=self.device)

        # Clamp controls to limits: v in [0, 2], w in [-2, 2]
        v = torch.clamp(v, min=0.0, max=2.0)
        w = torch.clamp(w, min=-2.0, max=2.0)

        # Set commands (returned in obs which the next step action tries to track)
        self.env.commands[:, 0] = v
        self.env.commands[:, 1] = y_vel_cmd
        self.env.commands[:, 2] = w
        self.env.commands[:, 3] = body_height_cmd
        self.env.commands[:, 4] = step_frequency_cmd
        self.env.commands[:, 5:8] = gait
        self.env.commands[:, 8] = 0.5
        self.env.commands[:, 9] = footswing_height_cmd
        self.env.commands[:, 10] = pitch_cmd
        self.env.commands[:, 11] = roll_cmd
        self.env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = self.env.step(actions)

        # Visualization (only for specified environment env_id)
        if add_line:
            curr_x, curr_y = self.env.base_pos[env_id, 0].item(), self.env.base_pos[env_id, 1].item()
            self._add_perm_line(prev_x, prev_y, curr_x, curr_y, line_color)

        for i in self.body_rigid_shape_indices:
            self.gym.set_rigid_body_color(
                self.env.envs[env_id],
                self.env.actor_handles[env_id],
                i,
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(*body_color)
            )

        return obs

    def current_dubins3d_state(self, env_id=None):
        """
        Get current Dubins3D state.

        Args:
            env_id: int or None
                - If int: return state of that specific environment
                - If None: return states of ALL environments (batched)

        Returns:
            state: torch.Tensor:
                - If env_id is int: shape (3,) containing [x, y, theta]
                - If env_id is None: shape (num_envs, 3) containing [[x, y, theta], ...]
        """
        if env_id is not None:
            # Single environment
            x = self.env.base_pos[env_id, 0]
            y = self.env.base_pos[env_id, 1]
            theta = torch_utils.get_euler_xyz(self.env.base_quat)[2][env_id]
            state = torch.stack([x, y, theta])  # shape: (3,)
            state = self.wrap_heading(state.reshape(1, -1))  # wrapped state (heading in [-pi, pi])
            return state.squeeze(0)  # Shape: (3,)
        else:
            # All environments (batched)
            x = self.env.base_pos[:, 0]  # Shape: (num_envs,)
            y = self.env.base_pos[:, 1]  # Shape: (num_envs,)
            theta = torch_utils.get_euler_xyz(self.env.base_quat)[2]  # Shape: (num_envs,)
            state = torch.stack([x, y, theta], dim=1)  # Shape: (num_envs, 3)
            state = self.wrap_heading(state)  # wrapped state (heading in [-pi, pi])
            return state  # Shape: (num_envs, 3)

    def in_collision(self):
        """Check if any environment is in collision with obstacles."""
        return torch.any(
            torch.linalg.norm(
                gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim))[-len(self.task.environment.obstacles):],
                dim=-1
            ) > 0
        )

    def wrap_heading(self, state):
        """
        Wrap heading to [-pi, pi].

        Args:
            state: tensor of shape (num_envs, num_states or state_dim)

        Returns:
            wrapped_state: tensor of shape (num_envs, num_states or state_dim)
        """
        # Assume state = [x, y, th, ...] where heading th is at dim = 2
        heading_dim = 2
        wrapped_state = state.clone()

        # Extract heading values from all environments
        heading = wrapped_state[:, heading_dim]

        # Wrap heading to [-pi, pi] using modulo operation
        # Formula: (heading + pi) % (2*pi) - pi
        wrapped_heading = (heading + torch.pi) % (2 * torch.pi) - torch.pi

        # Update the heading dimension with wrapped values
        wrapped_state[:, heading_dim] = wrapped_heading

        return wrapped_state

    # ==================== ISAACS Compatibility Methods ====================

    def attach_policy(self, policy):
        """
        Attach a high-level policy for ISAACS training/evaluation.

        Args:
            policy: A policy object with a get_action method that takes state
                   and returns Dubins3D action (v, w)
        """
        self.high_level_policy = policy

    def get_action(self, state, policy=None):
        """
        Get action from the attached policy for a given state.

        This method is compatible with ISAACS evaluation that requires
        calling get_action from the policy for specific states during rollout.

        Args:
            state: torch.Tensor of shape (state_dim,) or (num_envs, state_dim)
                   State for which to get action
            policy: Optional policy to use. If None, uses self.high_level_policy

        Returns:
            action: torch.Tensor of shape (action_dim,) or (num_envs, action_dim)
                   Action from the policy
        """
        if policy is None:
            policy = self.high_level_policy

        if policy is None:
            raise ValueError("No policy attached. Use attach_policy() or pass policy argument.")

        return policy.get_action(state)

    def reset(self, env_ids=None):
        """
        Reset environments.

        Args:
            env_ids: Optional tensor of environment IDs to reset.
                    If None, resets all environments.

        Returns:
            obs: Observation dictionary
        """
        if env_ids is None:
            return self.env.reset()
        else:
            return self.env.reset_idx(env_ids)

    def step(self, actions):
        """
        Step the environment with low-level actions.

        Args:
            actions: torch.Tensor of shape (num_envs, action_dim)
                    Low-level motor actions

        Returns:
            obs: observation dictionary
            rew: reward tensor
            done: done tensor
            info: info dictionary
        """
        return self.env.step(actions)

    # ==================== Property Accessors ====================

    @property
    def base_pos(self):
        """Get base positions from the wrapped environment."""
        return self.env.base_pos

    @property
    def base_quat(self):
        """Get base quaternions from the wrapped environment."""
        return self.env.base_quat

    @property
    def root_states(self):
        """Get root states from the wrapped environment."""
        return self.env.root_states

    @property
    def commands(self):
        """Get commands from the wrapped environment."""
        return self.env.commands

    @property
    def envs(self):
        """Get environment handles from the wrapped environment."""
        return self.env.envs

    @property
    def actor_handles(self):
        """Get actor handles from the wrapped environment."""
        return self.env.actor_handles
