"""
Standalone Velocity Tracking Environment for Quadruped
Based on VelocityTrackingEasyEnv from walk-these-ways.
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, Tuple, List
from scipy.spatial.transform import Rotation

from isaacgym import gymutil, gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *

# Import configuration and utilities from the original codebase
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../observation-conditioned-reachability/libraries/walk-these-ways'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../observation-conditioned-reachability'))

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from go1_gym.utils.terrain import Terrain
from go1_gym.envs.rewards.corl_rewards import CoRLRewards
from go1_gym.envs.base.curriculum import RewardThresholdCurriculum

from utils.simulation_utils.environment import CustomGroundEnvironment
from utils.simulation_utils.obstacle import CircularObstacle, BoxObstacle


class VelocityTrackingEnv:
    """
    Standalone Velocity Tracking Environment for Quadruped Locomotion.

    This class implements a complete RL environment for training quadruped robots
    to track velocity commands using Isaac Gym. It includes:
    - Physics simulation setup
    - Observation and reward computation
    - Command curriculum
    - Domain randomization
    - Terrain handling
    """

    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None,
                 physics_engine="SIM_PHYSX", task=None):
        """
        Initialize the velocity tracking environment.

        Args:
            sim_device (str): Device for simulation ('cuda:0' or 'cpu')
            headless (bool): Run without rendering
            num_envs (int): Number of parallel environments
            prone (bool): Start robot in prone position
            deploy (bool): Deployment mode
            cfg (Cfg): Training configuration
            eval_cfg (Cfg): Evaluation configuration (optional)
            initial_dynamics_dict (dict): Initial dynamics parameters
            physics_engine (str): Physics engine type
            task: Task object (optional)
        """
        # Store configuration
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.initial_dynamics_dict = initial_dynamics_dict
        self.task = task
        self.headless = headless

        # Update num_envs if specified
        if num_envs is not None:
            cfg.env.num_envs = num_envs

        # Parse configurations
        if eval_cfg is not None:
            self._parse_cfg(eval_cfg)
        self._parse_cfg(self.cfg)

        # Initialize Isaac Gym
        self.gym = gymapi.acquire_gym()

        # Parse physics engine
        if isinstance(physics_engine, str) and physics_engine == "SIM_PHYSX":
            self.physics_engine = gymapi.SIM_PHYSX
        else:
            self.physics_engine = physics_engine

        # Setup simulation parameters
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        self.sim_params = sim_params

        # Device setup
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True
        if sim_device_type == 'cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # Graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = self.sim_device_id

        # Environment dimensions
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        if eval_cfg is not None:
            self.num_eval_envs = eval_cfg.env.num_envs
            self.num_train_envs = cfg.env.num_envs
            self.num_envs = self.num_eval_envs + self.num_train_envs
        else:
            self.num_eval_envs = 0
            self.num_train_envs = cfg.env.num_envs
            self.num_envs = cfg.env.num_envs

        # PyTorch optimization flags
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Initialize basic buffers (more buffers initialized in _init_buffers)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs,
                                             device=self.device, dtype=torch.float)

        self.extras = {}

        # Initialize environment state
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.record_now = False
        self.record_eval_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0
        self.common_step_counter = 0

        # Create simulation, terrain and environments
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # Viewer setup
        self.enable_viewer_sync = True
        self.viewer = None

        if self.headless == False:
            # Create viewer and subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        # Initialize command distribution
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        # Initialize all simulation buffers
        self._init_buffers()

        # Prepare reward functions
        self._prepare_reward_function()

        # Mark initialization complete
        self.init_done = True

    def _parse_cfg(self, cfg):
        """
        Parse configuration to convert time values to timesteps.

        Args:
            cfg: Configuration object
        """
        # Compute timestep
        self.dt = self.cfg.control.decimation * self.cfg.sim.dt

        # Convert time to timesteps
        self.cfg.env.max_episode_length = int(self.cfg.env.episode_length_s / self.dt)

        # Domain randomization intervals
        self.cfg.domain_rand.rand_interval = int(self.cfg.domain_rand.rand_interval_s / self.dt)
        self.cfg.domain_rand.push_interval = int(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.gravity_rand_interval = int(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.domain_rand.gravity_rand_duration = int(
            self.cfg.domain_rand.gravity_rand_duration * self.cfg.domain_rand.gravity_rand_interval)

        # Observation scaling
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = dict()
        for key in dir(self.cfg.reward_scales):
            if not key.startswith('__'):
                self.reward_scales[key] = getattr(self.cfg.reward_scales, key)

    def create_sim(self):
        """
        Creates simulation, terrain and environments.
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                      self.physics_engine, self.sim_params)

        # Create terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            if self.eval_cfg is not None:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs,
                                      self.eval_cfg.terrain, self.num_eval_envs)
            else:
                self.terrain = Terrain(self.cfg.terrain, self.num_train_envs)

        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")

        self._create_envs()

    def _create_ground_plane(self):
        """Create a ground plane in the simulation."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """Create heightfield terrain."""
        self.gym.add_heightfield(self.sim, self.terrain.vertices, self.terrain.triangles)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """Create triangle mesh terrain."""
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'),
                                   gymapi.TriangleMeshParams())
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """
        Creates environments with robots.
        """
        asset_path = self.cfg.asset.file.format(MINI_GYM_ROOT_DIR=MINI_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_dofs = self.num_dof
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        self.num_actuated_dof = self.num_actions

        # Get DOF names
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        self.dof_names = [self.gym.get_asset_dof_name(self.robot_asset, i) for i in range(self.num_dof)]

        # Get rigid body names
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)
        body_names = [self.gym.get_asset_rigid_body_name(self.robot_asset, i) for i in range(self.num_bodies)]

        # Find feet indices
        self.feet_indices = torch.zeros(4, dtype=torch.long, device=self.device, requires_grad=False)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        for i, name in enumerate(feet_names):
            self.feet_indices[i] = body_names.index(name)

        # Find penalized contact and termination contact indices
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long,
                                                     device=self.device, requires_grad=False)
        for i, name in enumerate(penalized_contact_names):
            self.penalized_contact_indices[i] = body_names.index(name)

        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i, name in enumerate(termination_contact_names):
            self.termination_contact_indices[i] = body_names.index(name)

        # Base init state
        pos = self.cfg.init_state.pos
        rot = self.cfg.init_state.rot
        self.base_init_state = pos + rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)

        # Default joint positions
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i, name in enumerate(self.dof_names):
            angle = self.cfg.init_state.default_joint_angles.get(name, 0)
            self.default_dof_pos[i] = angle

        # PD gains
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i, name in enumerate(self.dof_names):
            self.p_gains[i] = self.cfg.control.stiffness.get(name, 0.)
            self.d_gains[i] = self.cfg.control.damping.get(name, 0.)

        # Environment spacing
        env_spacing = self.cfg.env.env_spacing
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

        # Create environments
        self.envs = []
        self.actor_handles = []

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        for i in range(self.num_envs):
            # Create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # Create actor
            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "robot", i,
                                                 self.cfg.asset.self_collisions, 0)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        # Get environment origins
        self.env_origins = self._get_env_origins(torch.arange(self.num_envs, device=self.device), self.cfg)

    def _get_env_origins(self, env_ids, cfg):
        """
        Get environment origins based on terrain curriculum.

        Args:
            env_ids: Environment IDs
            cfg: Configuration object

        Returns:
            torch.Tensor: Environment origins [num_envs, 3]
        """
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # Simplified curriculum - place envs on terrain
            self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.terrain_types = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            for i in range(self.num_envs):
                env_origins[i] = self.terrain_origins[i % len(self.terrain_origins)]
            self.custom_origins = True
        else:
            # For plane terrain, no special origins
            env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.custom_origins = False

        return env_origins

    def _init_buffers(self):
        """
        Initialize all torch buffers for simulation state.
        """
        # Get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Create state tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # Initialize derived state buffers
        self.base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.projected_gravity = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.foot_velocities = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_positions = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # Previous states
        self.prev_base_pos = torch.zeros_like(self.base_pos)
        self.prev_base_quat = torch.zeros_like(self.base_quat)
        self.prev_base_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.prev_foot_velocities = torch.zeros_like(self.foot_velocities)

        # Action and control buffers
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_joint_pos_target = torch.zeros_like(self.joint_pos_target)
        self.last_last_joint_pos_target = torch.zeros_like(self.joint_pos_target)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)

        # Command buffers
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_value = torch.zeros_like(self.commands)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                          device=self.device, requires_grad=False)
        if self.cfg.commands.num_commands > 3:
            # Extended command scaling for gait parameters
            extra_scales = []
            if self.cfg.commands.num_commands > 3:
                extra_scales.append(1.0)  # Body height
            if self.cfg.commands.num_commands > 4:
                extra_scales.append(self.obs_scales.gait_freq_cmd)  # Frequency
            if self.cfg.commands.num_commands > 5:
                extra_scales += [self.obs_scales.gait_phase_cmd] * 3  # Phase offsets
            if self.cfg.commands.num_commands > 8:
                extra_scales.append(1.0)  # Duration
            if self.cfg.commands.num_commands > 9:
                extra_scales.append(self.obs_scales.footswing_height_cmd)  # Footswing height
            self.commands_scale = torch.cat([self.commands_scale,
                                            torch.tensor(extra_scales, device=self.device, requires_grad=False)])

        # Gait and contact buffers
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False)

        # Terrain height buffers
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
            self.measured_heights = torch.zeros(self.num_envs, self.cfg.env.num_height_points,
                                               dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.measured_heights = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        # Domain randomization buffers
        self._init_custom_buffers()

        # Gravity
        self.gravity_vec = torch.tensor([0., 0., -1.], device=self.device, requires_grad=False).repeat(self.num_envs, 1)
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat(self.num_envs, 1)

        # Lag buffer for action delay randomization
        self.lag_buffer = [torch.zeros_like(self.actions) for _ in range(int(self.cfg.domain_rand.lag_timesteps) + 1)]

        # Actuator network buffers (if using actuator network)
        if self.cfg.control.control_type == "actuator_net":
            self.joint_pos_err_last_last = torch.zeros_like(self.dof_pos)
            self.joint_pos_err_last = torch.zeros_like(self.dof_pos)
            self.joint_vel_last_last = torch.zeros_like(self.dof_vel)
            self.joint_vel_last = torch.zeros_like(self.dof_vel)
            # Load actuator network
            # self.actuator_network = torch.jit.load(self.cfg.control.actuator_net_file).to(self.device)

        # Noise scale vector
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.add_noise = self.cfg.noise.add_noise

    def _init_custom_buffers(self):
        """
        Initialize domain randomization buffers.
        """
        # Friction
        self.default_friction = self.cfg.terrain.static_friction
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                         requires_grad=False) * self.default_friction

        # Restitution
        self.default_restitution = self.cfg.terrain.restitution
        self.restitutions = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                      requires_grad=False) * self.default_restitution

        # Payload (added mass)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # COM displacement
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # Motor properties
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # PD gain factors
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # Gravity randomization
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        # Ground friction (for terrain)
        self.ground_friction_coeffs = torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False) * self.default_friction

        # Initialize randomization
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_rigid_body_props(env_ids, self.cfg)
        self._randomize_dof_props(env_ids, self.cfg)
        self._randomize_gravity()

    def _get_noise_scale_vec(self, cfg):
        """
        Get observation noise scale vector.

        Args:
            cfg: Configuration object

        Returns:
            torch.Tensor: Noise scale vector matching observation dimensions
        """
        noise_vec = torch.zeros(self.num_obs, device=self.device, requires_grad=False)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise_scales
        noise_level = self.cfg.noise.noise_level

        # This is simplified - full implementation would match observation structure exactly
        # For now, return uniform noise scaling
        noise_vec[:] = noise_level

        return noise_vec

    def _init_height_points(self, env_ids, cfg):
        """
        Initialize height measurement points around the robot.

        Args:
            env_ids: Environment IDs
            cfg: Configuration

        Returns:
            torch.Tensor: Height measurement points [num_envs, num_points, 3]
        """
        # Grid of measurement points
        y = torch.tensor(cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        x = torch.tensor(cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        num_points = grid_x.numel()
        points = torch.zeros(self.num_envs, num_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()

        return points

    def _init_command_distribution(self, env_ids):
        """
        Initialize command curriculum for sampling velocity commands.

        Args:
            env_ids: Environment IDs to initialize
        """
        # Setup curriculum categories
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            self.category_names = ['pronk', 'trot', 'pace', 'bound']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            # Create curriculum for each gait category
            self.curricula = []
            for category in self.category_names:
                curriculum = RewardThresholdCurriculum(
                    seed=self.cfg.commands.curriculum_seed + len(self.curricula),
                    x_vel=(self.cfg.commands.limit_vel_x[0],
                           self.cfg.commands.limit_vel_x[1],
                           self.cfg.commands.num_bins_vel_x),
                    y_vel=(self.cfg.commands.limit_vel_y[0],
                           self.cfg.commands.limit_vel_y[1],
                           self.cfg.commands.num_bins_vel_y),
                    yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                             self.cfg.commands.limit_vel_yaw[1],
                             self.cfg.commands.num_bins_vel_yaw),
                    body_height=(self.cfg.commands.limit_body_height[0],
                                 self.cfg.commands.limit_body_height[1],
                                 self.cfg.commands.num_bins_body_height),
                    gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                    self.cfg.commands.limit_gait_frequency[1],
                                    self.cfg.commands.num_bins_gait_frequency),
                    gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                self.cfg.commands.limit_gait_phase[1],
                                self.cfg.commands.num_bins_gait_phase),
                    gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                 self.cfg.commands.limit_gait_offset[1],
                                 self.cfg.commands.num_bins_gait_offset),
                    gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                 self.cfg.commands.limit_gait_bound[1],
                                 self.cfg.commands.num_bins_gait_bound),
                    gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                   self.cfg.commands.limit_gait_duration[1],
                                   self.cfg.commands.num_bins_gait_duration),
                    footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                      self.cfg.commands.limit_footswing_height[1],
                                      self.cfg.commands.num_bins_footswing_height),
                    body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                self.cfg.commands.limit_body_pitch[1],
                                self.cfg.commands.num_bins_body_pitch),
                    body_roll=(self.cfg.commands.limit_body_roll[0],
                               self.cfg.commands.limit_body_roll[1],
                               self.cfg.commands.num_bins_body_roll),
                    stance_width=(self.cfg.commands.limit_stance_width[0],
                                  self.cfg.commands.limit_stance_width[1],
                                  self.cfg.commands.num_bins_stance_width),
                    stance_length=(self.cfg.commands.limit_stance_length[0],
                                   self.cfg.commands.limit_stance_length[1],
                                   self.cfg.commands.num_bins_stance_length),
                    aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                     self.cfg.commands.limit_aux_reward_coef[1],
                                     self.cfg.commands.num_bins_aux_reward_coef),
                )
                self.curricula.append(curriculum)

        # Handle LipschitzCurriculum if needed
        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                     binary_phases=self.cfg.commands.binary_phases)

        # Environment command tracking
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int32)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int32)

        # CRITICAL: Initialize curriculum weights for valid command range
        # This sets weights to 1.0 for all command bins within the valid range
        low = np.array([
            self.cfg.commands.lin_vel_x[0],
            self.cfg.commands.lin_vel_y[0],
            self.cfg.commands.ang_vel_yaw[0],
            self.cfg.commands.body_height_cmd[0],
            self.cfg.commands.gait_frequency_cmd_range[0],
            self.cfg.commands.gait_phase_cmd_range[0],
            self.cfg.commands.gait_offset_cmd_range[0],
            self.cfg.commands.gait_bound_cmd_range[0],
            self.cfg.commands.gait_duration_cmd_range[0],
            self.cfg.commands.footswing_height_range[0],
            self.cfg.commands.body_pitch_range[0],
            self.cfg.commands.body_roll_range[0],
            self.cfg.commands.stance_width_range[0],
            self.cfg.commands.stance_length_range[0],
            self.cfg.commands.aux_reward_coef_range[0],
        ])
        high = np.array([
            self.cfg.commands.lin_vel_x[1],
            self.cfg.commands.lin_vel_y[1],
            self.cfg.commands.ang_vel_yaw[1],
            self.cfg.commands.body_height_cmd[1],
            self.cfg.commands.gait_frequency_cmd_range[1],
            self.cfg.commands.gait_phase_cmd_range[1],
            self.cfg.commands.gait_offset_cmd_range[1],
            self.cfg.commands.gait_bound_cmd_range[1],
            self.cfg.commands.gait_duration_cmd_range[1],
            self.cfg.commands.footswing_height_range[1],
            self.cfg.commands.body_pitch_range[1],
            self.cfg.commands.body_roll_range[1],
            self.cfg.commands.stance_width_range[1],
            self.cfg.commands.stance_length_range[1],
            self.cfg.commands.aux_reward_coef_range[1],
        ])

        # Initialize curriculum weights - this is ESSENTIAL to avoid division by zero
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

        # Curriculum thresholds for different reward components
        self.curriculum_thresholds = {
            "tracking_lin_vel": 0.8,
            "tracking_ang_vel": 0.5,
            "tracking_contacts_shaped_force": 0.9,
            "tracking_contacts_shaped_vel": 0.9
        }

        # Initialize episode sums for reward tracking
        self.episode_sums = {}
        self.command_sums = {}
        for key in self.reward_scales.keys():
            self.episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            self.command_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Additional command sums for tracking
        self.command_sums["lin_vel_raw"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_sums["ang_vel_raw"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_sums["lin_vel_residual"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_sums["ang_vel_residual"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.command_sums["ep_timesteps"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Evaluation episode sums
        self.episode_sums_eval = {}
        for key in self.episode_sums.keys():
            self.episode_sums_eval[key] = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * -1

    def _prepare_reward_function(self):
        """
        Prepare reward function container and build list of active reward functions.
        """
        # Create reward container
        self.reward_container = CoRLRewards(self)

        # Build list of reward functions with non-zero scales
        self.reward_functions = []
        self.reward_names = []

        for name, scale in self.reward_scales.items():
            if scale != 0 and hasattr(self.reward_container, '_reward_' + name):
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))
                self.reward_names.append(name)

        # Create reward buffer components
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.rew_buf_neg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def step(self, actions):
        """
        Perform one simulation step.

        Args:
            actions (torch.Tensor): Actions to apply [num_envs, num_actions]

        Returns:
            tuple: (observations, privileged_obs, rewards, dones, extras)
        """
        # Clip and store actions
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # Store previous states
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        # Render GUI
        self.render_gui()

        # Physics stepping
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # Post-physics step processing
        self.post_physics_step()

        # Update foot positions for extras
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        # Clip observations
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # Update extras with additional information
        self.extras.update({
            "privileged_obs": self.privileged_obs_buf,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:3],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """
        Process physics step: check terminations, compute observations and rewards.
        """
        # Refresh gym state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        # Update counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Update state quantities
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        # Post-physics callbacks
        self._post_physics_step_callback()

        # Compute terminations, rewards, observations
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        # Update action history
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # Debug visualization
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self._render_headless()

    def _post_physics_step_callback(self):
        """
        Callback for common computations after physics step.
        """
        # Teleport robots to prevent falling off edge
        self._call_train_eval(self._teleport_robots, torch.arange(self.num_envs, device=self.device))

        # Resample commands periodically
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        self._step_contact_targets()

        # Measure terrain heights
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights(torch.arange(self.num_envs, device=self.device), self.cfg)

        # Push robots randomly
        self._call_train_eval(self._push_robots, torch.arange(self.num_envs, device=self.device))

        # Randomize DOF properties periodically
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(as_tuple=False).flatten()
        self._call_train_eval(self._randomize_dof_props, env_ids)

        # Randomize gravity
        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))

        # Randomize rigid body properties
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

    def check_termination(self):
        """
        Check which environments need to be reset.
        """
        # Contact termination
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # Timeout termination
        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf |= self.time_out_buf

        # Body height termination
        if self.cfg.rewards.use_terminal_body_height:
            self.body_height_buf = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1) \
                                   < self.cfg.rewards.terminal_body_height
            self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

    def reset_idx(self, env_ids):
        """
        Reset specified environments.

        Args:
            env_ids (torch.Tensor): Environment IDs to reset
        """
        if len(env_ids) == 0:
            return

        # Resample commands
        self._resample_commands(env_ids)

        # Randomize properties
        self._call_train_eval(self._randomize_dof_props, env_ids)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._call_train_eval(self._randomize_rigid_body_props, env_ids)
            self._call_train_eval(self.refresh_actor_rigid_shape_props, env_ids)

        # Reset robot states
        self._call_train_eval(self._reset_dofs, env_ids)
        self._call_train_eval(self._reset_root_states, env_ids)

        # Reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Log episode statistics
        train_env_ids = env_ids[env_ids < self.num_train_envs]
        if len(train_env_ids) > 0:
            self.extras["train/episode"] = {}
            for key in self.episode_sums.keys():
                self.extras["train/episode"]['rew_' + key] = torch.mean(self.episode_sums[key][train_env_ids])
                self.episode_sums[key][train_env_ids] = 0.

        eval_env_ids = env_ids[env_ids >= self.num_train_envs]
        if len(eval_env_ids) > 0:
            self.extras["eval/episode"] = {}
            for key in self.episode_sums.keys():
                unset_eval_envs = eval_env_ids[self.episode_sums_eval[key][eval_env_ids] == -1]
                self.episode_sums_eval[key][unset_eval_envs] = self.episode_sums[key][unset_eval_envs]
                self.episode_sums[key][eval_env_ids] = 0.

        # Log curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["train/episode"]["terrain_level"] = torch.mean(self.terrain_levels[:self.num_train_envs].float())

        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf[:self.num_train_envs]

        # Reset gait indices
        self.gait_indices[env_ids] = 0

        # Reset lag buffer
        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0

    def compute_reward(self):
        """
        Compute rewards by calling all active reward functions.
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew

            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew

            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew

        # Apply reward shaping
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style:
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)

        self.episode_sums["total"] += self.rew_buf

        # Add termination reward
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

        # Track velocity tracking performance
        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        """
        Compute observations for all environments.
        """
        # Base observation: gravity + joint state + actions
        self.obs_buf = torch.cat((
            self.projected_gravity,
            (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:, :self.num_actuated_dof]) * self.obs_scales.dof_pos,
            self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
            self.actions
        ), dim=-1)

        # Add command observation
        if self.cfg.env.observe_command:
            self.obs_buf = torch.cat((
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos[:, :self.num_actuated_dof] - self.default_dof_pos[:, :self.num_actuated_dof]) * self.obs_scales.dof_pos,
                self.dof_vel[:, :self.num_actuated_dof] * self.obs_scales.dof_vel,
                self.actions
            ), dim=-1)

        # Add previous actions
        if self.cfg.env.observe_two_prev_actions:
            self.obs_buf = torch.cat((self.obs_buf, self.last_actions), dim=-1)

        # Add timing parameter
        if self.cfg.env.observe_timing_parameter:
            self.obs_buf = torch.cat((self.obs_buf, self.gait_indices.unsqueeze(1)), dim=-1)

        # Add clock inputs
        if self.cfg.env.observe_clock_inputs:
            self.obs_buf = torch.cat((self.obs_buf, self.clock_inputs), dim=-1)

        # Add velocity observations
        if self.cfg.env.observe_vel:
            if self.cfg.commands.global_reference:
                self.obs_buf = torch.cat((
                    self.root_states[:self.num_envs, 7:10] * self.obs_scales.lin_vel,
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    self.obs_buf
                ), dim=-1)
            else:
                self.obs_buf = torch.cat((
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.base_ang_vel * self.obs_scales.ang_vel,
                    self.obs_buf
                ), dim=-1)

        if self.cfg.env.observe_only_ang_vel:
            self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.obs_buf), dim=-1)

        if self.cfg.env.observe_only_lin_vel:
            self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf), dim=-1)

        # Add yaw
        if self.cfg.env.observe_yaw:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
            self.obs_buf = torch.cat((self.obs_buf, heading), dim=-1)

        # Add contact states
        if self.cfg.env.observe_contact_states:
            self.obs_buf = torch.cat((
                self.obs_buf,
                (self.contact_forces[:, self.feet_indices, 2] > 1.).view(self.num_envs, -1) * 1.0
            ), dim=1)

        # Add noise
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # Build privileged observations
        self.privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)
        self.next_privileged_obs_buf = torch.empty(self.num_envs, 0).to(self.device)

        if self.cfg.env.priv_observe_friction:
            friction_coeffs_scale, friction_coeffs_shift = get_scale_shift(self.cfg.normalization.friction_range)
            self.privileged_obs_buf = torch.cat((
                self.privileged_obs_buf,
                (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale
            ), dim=1)
            self.next_privileged_obs_buf = torch.cat((
                self.next_privileged_obs_buf,
                (self.friction_coeffs[:, 0].unsqueeze(1) - friction_coeffs_shift) * friction_coeffs_scale
            ), dim=1)

        if self.cfg.env.priv_observe_restitution:
            restitutions_scale, restitutions_shift = get_scale_shift(self.cfg.normalization.restitution_range)
            self.privileged_obs_buf = torch.cat((
                self.privileged_obs_buf,
                (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale
            ), dim=1)
            self.next_privileged_obs_buf = torch.cat((
                self.next_privileged_obs_buf,
                (self.restitutions[:, 0].unsqueeze(1) - restitutions_shift) * restitutions_scale
            ), dim=1)

        if self.cfg.env.priv_observe_base_mass:
            payloads_scale, payloads_shift = get_scale_shift(self.cfg.normalization.added_mass_range)
            self.privileged_obs_buf = torch.cat((
                self.privileged_obs_buf,
                (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale
            ), dim=1)
            self.next_privileged_obs_buf = torch.cat((
                self.next_privileged_obs_buf,
                (self.payloads.unsqueeze(1) - payloads_shift) * payloads_scale
            ), dim=1)

        # Additional privileged observations can be added here following the same pattern

    def _compute_torques(self, actions):
        """
        Compute torques from actions using PD controller or actuator network.

        Args:
            actions (torch.Tensor): Actions [num_envs, num_actions]

        Returns:
            torch.Tensor: Torques [num_envs, num_dof]
        """
        # Scale actions
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction

        # Handle action lag
        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type

        if control_type == "actuator_net":
            # Actuator network control (requires loading network)
            self.joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(
                self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                self.joint_vel, self.joint_vel_last, self.joint_vel_last_last
            )
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            # PD controller
            torques = self.p_gains * self.Kp_factors * (
                self.joint_pos_target - self.dof_pos + self.motor_offsets
            ) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        # Apply motor strength and clip
        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids, cfg):
        """
        Reset DOF positions and velocities for specified environments.

        Args:
            env_ids (torch.Tensor): Environment IDs
            cfg: Configuration object
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids, cfg):
        """
        Reset root states (base position and velocity) for specified environments.

        Args:
            env_ids (torch.Tensor): Environment IDs
            cfg: Configuration object
        """
        # Base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0:1] += torch_rand_float(
                -cfg.terrain.x_init_range, cfg.terrain.x_init_range,
                (len(env_ids), 1), device=self.device
            )
            self.root_states[env_ids, 1:2] += torch_rand_float(
                -cfg.terrain.y_init_range, cfg.terrain.y_init_range,
                (len(env_ids), 1), device=self.device
            )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :2] += torch_rand_float(
                -cfg.terrain.x_init_range, cfg.terrain.x_init_range,
                (len(env_ids), 2), device=self.device
            )

        # Random yaw
        self.root_states[env_ids, 2] += cfg.terrain.x_init_offset
        yaw_random = torch_rand_float(
            -cfg.terrain.yaw_init_range, cfg.terrain.yaw_init_range,
            (len(env_ids), 1), device=self.device
        ).squeeze(1)
        quat_random = quat_from_angle_axis(yaw_random, to_torch([0., 0., 1.], device=self.device))
        self.root_states[env_ids, 3:7] = quat_random

        # Random velocities
        self.root_states[env_ids, 7:10] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 3), device=self.device
        )
        self.root_states[env_ids, 10:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 3), device=self.device
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _resample_commands(self, env_ids):
        """
        Resample velocity and gait commands for specified environments.

        Args:
            env_ids (torch.Tensor): Environment IDs to resample
        """
        if len(env_ids) == 0:
            return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)

        # Update curricula based on episode performance
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_contacts_shaped_force", "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(
                    old_bins, task_rewards, success_thresholds,
                    local_range=np.array([0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
                )

        # Assign environments to categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [
            env_ids[torch.logical_and(
                probability_per_category * i <= random_env_floats,
                random_env_floats < probability_per_category * (i + 1)
            )]
            for i in range(len(self.category_names))
        ]

        # Sample commands from curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
            zip(self.category_names, category_env_ids, self.curricula)
        ):
            batch_size = len(env_ids_in_category)
            if batch_size == 0:
                continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(
                new_commands[:, :self.cfg.commands.num_commands]
            ).to(self.device)

        # Apply gait-specific phase adjustments
        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

        # Zero out small commands
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # Reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.

    def _step_contact_targets(self):
        """
        Update desired contact states based on gait parameters.
        """
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 4]
            phases = self.commands[:, 5]
            offsets = self.commands[:, 6]
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            if self.cfg.commands.pacing_offset:
                foot_indices = [
                    self.gait_indices + phases + offsets + bounds,
                    self.gait_indices + bounds,
                    self.gait_indices + offsets,
                    self.gait_indices + phases
                ]
            else:
                foot_indices = [
                    self.gait_indices + phases + offsets + bounds,
                    self.gait_indices + offsets,
                    self.gait_indices + bounds,
                    self.gait_indices + phases
                ]

            self.foot_indices = torch.remainder(
                torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0
            )

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                    0.5 / (1 - durations[swing_idxs])
                )

            # Update clock inputs
            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
            self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
            self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

            self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
            self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
            self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
            self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

            self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
            self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
            self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
            self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

            # Compute desired contact states with smoothing
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

            for i, foot_idx in enumerate(foot_indices):
                smoothing_multiplier = (
                    smoothing_cdf_start(torch.remainder(foot_idx, 1.0)) *
                    (1 - smoothing_cdf_start(torch.remainder(foot_idx, 1.0) - 0.5)) +
                    smoothing_cdf_start(torch.remainder(foot_idx, 1.0) - 1) *
                    (1 - smoothing_cdf_start(torch.remainder(foot_idx, 1.0) - 0.5 - 1))
                )
                self.desired_contact_states[:, i] = smoothing_multiplier

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _get_heights(self, env_ids, cfg):
        """
        Get terrain heights at measurement points around robot.

        Args:
            env_ids (torch.Tensor): Environment IDs
            cfg: Configuration object

        Returns:
            torch.Tensor: Heights at measurement points
        """
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), self.cfg.env.num_height_points,
                             device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            # Transform measurement points to world frame
            points = self.height_points[env_ids].clone()
            points += self.base_pos[env_ids, :3].unsqueeze(1)

            # Sample heights from terrain
            points_x = ((points[:, :, 0] - self.terrain.border_size) / self.terrain.horizontal_scale).long()
            points_y = ((points[:, :, 1] - self.terrain.border_size) / self.terrain.horizontal_scale).long()

            points_x = torch.clip(points_x, 0, self.height_samples.shape[0] - 2)
            points_y = torch.clip(points_y, 0, self.height_samples.shape[1] - 2)

            heights = self.height_samples[points_x, points_y] * self.terrain.vertical_scale

            return heights
        else:
            return torch.zeros(len(env_ids), self.cfg.env.num_height_points,
                             device=self.device, requires_grad=False)

    def _call_train_eval(self, func, env_ids):
        """
        Call function separately for train and eval environments.

        Args:
            func: Function to call
            env_ids (torch.Tensor): Environment IDs

        Returns:
            Combined return value from train and eval calls
        """
        env_ids_train = env_ids[env_ids < self.num_train_envs]
        env_ids_eval = env_ids[env_ids >= self.num_train_envs]

        ret, ret_eval = None, None

        if len(env_ids_train) > 0:
            ret = func(env_ids_train, self.cfg)
        if len(env_ids_eval) > 0:
            ret_eval = func(env_ids_eval, self.eval_cfg)
            if ret is not None and ret_eval is not None:
                ret = torch.cat((ret, ret_eval), axis=-1)

        return ret

    def _push_robots(self, env_ids, cfg):
        """
        Apply random pushes to robots.

        Args:
            env_ids (torch.Tensor): Environment IDs
            cfg: Configuration object
        """
        if cfg.domain_rand.push_robots:
            push_interval = cfg.domain_rand.push_interval
            env_ids_to_push = env_ids[self.episode_length_buf[env_ids] % push_interval == 0]

            if len(env_ids_to_push) > 0:
                max_vel = cfg.domain_rand.max_push_vel_xy
                self.root_states[env_ids_to_push, 7:9] = torch_rand_float(
                    -max_vel, max_vel, (len(env_ids_to_push), 2), device=self.device
                )

                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robots(self, env_ids, cfg):
        """
        Teleport robots back if they fall off terrain edges.

        Args:
            env_ids (torch.Tensor): Environment IDs
            cfg: Configuration object
        """
        if cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            # Check if robot fell off terrain
            # Simplified implementation - full version would check terrain boundaries
            pass

    def _randomize_gravity(self, external_force=None):
        """
        Randomize gravity vector.

        Args:
            external_force (torch.Tensor): External force to apply (optional)
        """
        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device, requires_grad=False) * \
                           (max_gravity - min_gravity) + min_gravity
            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _randomize_rigid_body_props(self, env_ids, cfg):
        """
        Randomize rigid body properties (mass, COM, friction, restitution).

        Args:
            env_ids (torch.Tensor): Environment IDs
            cfg: Configuration object
        """
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                               requires_grad=False) * (max_payload - min_payload) + min_payload

        if cfg.domain_rand.randomize_com_displacement:
            min_com, max_com = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float,
                                                           device=self.device, requires_grad=False) * \
                                                (max_com - min_com) + min_com

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float,
                                                         device=self.device, requires_grad=False) * \
                                              (max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float,
                                                   device=self.device, requires_grad=False) * \
                                        (max_restitution - min_restitution) + min_restitution

    def _randomize_dof_props(self, env_ids, cfg):
        """
        Randomize DOF properties (motor strength, offset, PD gains).

        Args:
            env_ids (torch.Tensor): Environment IDs
            cfg: Configuration object
        """
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float,
                                                         device=self.device, requires_grad=False).unsqueeze(1) * \
                                              (max_strength - min_strength) + min_strength

        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                       device=self.device, requires_grad=False) * \
                                            (max_offset - min_offset) + min_offset

        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp, max_Kp = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float,
                                                    device=self.device, requires_grad=False).unsqueeze(1) * \
                                         (max_Kp - min_Kp) + min_Kp

        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd, max_Kd = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float,
                                                    device=self.device, requires_grad=False).unsqueeze(1) * \
                                         (max_Kd - min_Kd) + min_Kd

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        """
        Refresh rigid shape properties (friction, restitution) for actors.

        Args:
            env_ids (torch.Tensor): Environment IDs
            cfg: Configuration object
        """
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_rigid_shape_props(self, props, env_id):
        """
        Process rigid shape properties during environment creation.

        Args:
            props: Rigid shape properties
            env_id (int): Environment ID

        Returns:
            Modified properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props

    def _process_dof_props(self, props, env_id):
        """
        Process DOF properties during environment creation.

        Args:
            props: DOF properties
            env_id (int): Environment ID

        Returns:
            Modified properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float,
                                             device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float,
                                             device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float,
                                            device=self.device, requires_grad=False)

            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                # Apply soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _process_rigid_body_props(self, props, env_id):
        """
        Process rigid body properties during environment creation.

        Args:
            props: Rigid body properties
            env_id (int): Environment ID

        Returns:
            Modified properties
        """
        if env_id == 0:
            self.default_body_mass = props[0].mass

        props[0].mass = self.default_body_mass + self.payloads[env_id]
        props[0].com = gymapi.Vec3(
            self.com_displacements[env_id, 0],
            self.com_displacements[env_id, 1],
            self.com_displacements[env_id, 2]
        )

        return props

    def _draw_debug_vis(self):
        """
        Draw debug visualization (height measurement points).
        """
        if self.cfg.terrain.measure_heights:
            # Draw spheres at height measurement points
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 0, 0))
            for i in range(self.num_envs):
                base_pos = self.base_pos[i].cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = self.height_points[i].cpu().numpy()

                for j, (point, height) in enumerate(zip(height_points, heights)):
                    x = base_pos[0] + point[0]
                    y = base_pos[1] + point[1]
                    z = height
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _render_headless(self):
        """
        Handle headless rendering for camera sensors.
        """
        if self.record_now or self.record_eval_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

    def render_gui(self, sync_frame_time=True):
        """
        Render GUI and handle viewer events.

        Args:
            sync_frame_time (bool): Synchronize frame time
        """
        if self.viewer:
            # Check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # Check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # Fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # Step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def set_camera(self, position, lookat):
        """
        Set camera position and lookat point.

        Args:
            position (list): Camera position [x, y, z]
            lookat (list): Lookat point [x, y, z]
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def reset(self):
        """
        Reset all environments.

        Returns:
            torch.Tensor: Initial observations
        """
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs

    def close(self):
        """
        Clean up and close the environment.
        """
        if self.headless == False:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


# Example usage
if __name__ == "__main__":
    # This would require proper configuration setup
    # See the original codebase for configuration examples
    print("Standalone Velocity Tracking Environment created successfully!")
    print("Import this class and instantiate with proper Cfg configuration.")
