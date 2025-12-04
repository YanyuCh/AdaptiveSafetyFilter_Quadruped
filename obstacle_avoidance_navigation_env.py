import os
import sys
import numpy as np
import torch
import glob
from typing import Dict, Tuple, List, Optional
from scipy.spatial.transform import Rotation

from isaacgym import gymutil, gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *

# Import configuration and utilities from the original codebase
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../observation-conditioned-reachability/libraries/walk-these-ways'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../observation-conditioned-reachability'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../ISAACS/simulators/policy'))

from dubins3d_cost import Dubins3d_Cost, Dubins3d_Constraint

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.base.legged_robot import LeggedRobot
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from go1_gym.utils.terrain import Terrain
from go1_gym.envs.rewards.corl_rewards import CoRLRewards
from go1_gym.envs.base.curriculum import RewardThresholdCurriculum

from utils.simulation_utils.environment import CustomGroundEnvironment
from utils.simulation_utils.obstacle import CircularObstacle, BoxObstacle

from nn_policy import NeuralNetworkControlSystem

class ObstacleAvoidanceNavigation(LeggedRobot):
    def __init__(self, sim_device, headless, num_envs=None, prone=False, deploy=False,
                 cfg: Cfg = None, eval_cfg: Cfg = None, initial_dynamics_dict=None, physics_engine="SIM_PHYSX", task=None,
                 ll_policy = None, cfg_cost = None, cfg_arch = None, cfg_env = None):
        '''
        Initialize a high-level navigation env using a pretrained low-level locomotion policy defined in ll_policy
        
        Required Args:
            sim_device: device for simulation (cpu or cuda)
            headless: True if not render
            cfg: configuration of the env
            task: navigation task (contain goal info, obstacle info etc.)
            ll_policy: a pretrained velocity-tracking low-level locomotion policy
            cfg_cost: ISAACS cost configuration
            cfg_arch: ISAACS architecture configuration for high-level policy and value network
                      cfg_arch.actor_0: control network
                      cfg_arch.critic_0: value network
                      (cfg_arch.actor_1: disturbance network)
            cfg_env: ISAACS env configuration
        '''

        if num_envs is not None:
            cfg.env.num_envs = num_envs
        sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(vars(cfg.sim), sim_params)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, eval_cfg, initial_dynamics_dict, task)
        
        # custom env initialization
        # load ISSACS cost, arch, and env configuration
        self.cfg_cost = cfg_cost
        self.cfg_arch = cfg_arch
        self.cfg_env = cfg_env
        self.append_dim = self.cfg_arch.actor_0.append_dim  # = 2 (condition on physical parameters f and m (SAME for both actor and critic))
        self.end_criterion = self.cfg_env.end_criterion     # termination criterion for an episode ('failure', 'timeout', 'reach-avoid')
        self.g_x_fail = float(self.cfg_env.g_x_fail)        # value to assign to g_x when failure occurs
        self.condition_on_physical_params = getattr(self.cfg_env, "condition_on_physical_params", True)  # whether to include physical parameters [f, m] in info dict
        # initialize high-level navigation policy
        self.hl_policy = None       # set later via init_policy()
        self.num_hl_actions = 2     # [v, w] (forward velocity and yaw rate)
        self.v_limits = [0., 2.]    # hard high-level control limits
        self.w_limits = [-2., 2.]
        # load pretrained low-level locomotion policy
        self.ll_policy = ll_policy
        # track both high-level and low-level observations
        self.num_hl_obs = 33    # (x, y, heading, 6D linear+angular vel, 12D joint pos, 12D join vel)
        self.hl_obs_buf = torch.zeros(self.num_envs, self.num_hl_obs, device = self.device, dtype = torch.float32)
        # low-level observations already tracked in LeggedRobot with self.obs_buf
        # DEFAULT: OBS - LOW-LEVEL, HL_OBS - HIGH-LEVEL
        
        # integrate history wrapper for low-level observations
        self.obs_history_length = self.cfg.env.num_observation_history
        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_history, dtype = torch.float32, device = self.device, requires_grad = False)
        
        # ISAACS-related initialization
        self.hl_cost = Dubins3d_Cost(cfg = self.cfg_cost, task = self.task)
        self.hl_constraint = Dubins3d_Constraint(cfg = self.cfg_cost, task = self.task)
        self.x_lowerbound_local = -2.
        self.x_upperbound_local = 12.
        self.y_lowerbound_local = -5.
        self.y_upperbound_local = 5.
        self.failure_thr = 0.
        self.step_keep_constraints = False  # whether to keep constraints in info dict
        self.step_keep_targets = False  # whether to keep targets in info dict
        
    # High-level navigation policy related functions
    def init_hl_policy(self, cfg_hl_policy, **kwargs):
        '''
        Initialize the high-level navigation policy
        
        Required Args:
            cfg: configuration, can be SimpleNamespace(device)
            actor: actor net (in **kwargs)
        '''
        self.hl_policy = NeuralNetworkControlSystem(id = 'ego', cfg = cfg_hl_policy, **kwargs)
    def get_action(self, hl_obs, **kwargs):
        '''
        Get the action to execute with the current hl_obs from the current hl_policy
        
        Required Args:
            hl_obs: current high-level observation
        
        Returns:
            action: the action to be executed 
            solver_info: a dict containing info for the solver, e.g. processing time, status etc.
        '''
        agents_action = None    # no other agents' actions observable
        action, solver_info = self.hl_policy.get_action(obs = hl_obs, agents_action = agents_action, **kwargs)
        return action, solver_info
    
    # Integrate history wrapper for low-level observation (Required for low-level locomotion policy!)
    def get_observations(self):
        # update obs_history: delete the oldest obs and add the newest obs
        obs = self.obs_buf
        privileged_obs = self.privileged_obs_buf
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs), dim = -1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}
    
    # Track high-level navigation state
    def compute_hl_observations(self):
        '''
        Compute high-level navigation state in LOCAL frame

        Keep NORMALIZATION for stable training!
        '''
        # Convert global base_pos to local frame by subtracting env_origins
        local_base_pos = self.base_pos - self.env_origins  # Shape: (num_envs, 3)

        # normalized 3D Dubins Car state using LOCAL positions
        x = 2 * (local_base_pos[:, 0] - self.x_lowerbound_local) / (self.x_upperbound_local - self.x_lowerbound_local) - 1
        y = 2 * (local_base_pos[:, 1] - self.y_lowerbound_local) / (self.y_upperbound_local - self.y_lowerbound_local) - 1
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])  # world frame, range: [-pi, pi]

        # ALL Normalized! (except f and m)
        self.hl_obs_buf = torch.cat([x.unsqueeze(1), y.unsqueeze(1), heading.unsqueeze(1),  # 3D Dubins Car state
                                     self.base_lin_vel * self.obs_scales.lin_vel,       # 3D base linear velocity
                                     self.base_ang_vel * self.obs_scales.ang_vel,       # 3D base angular velocity
                                     (self.dof_pos[:, :self.num_actuated_dof] -
                                      self.default_dof_pos[:, :self.num_actuated_dof])
                                     * self.obs_scales.dof_pos,                         # 12D joint positions
                                     self.dof_vel[:, :self.num_actuated_dof]
                                     * self.obs_scales.dof_vel,                         # 12D joint velocities
                                     #self.friction_coeffs[:, 0],                       # 1D robot friction
                                     #self.payloads[:]                                  # 1D payload
                                     ],
                                     dim = -1) 
    
    # ISAACS-style cost, constraint, done, and info for the high-level policy
    def get_hl_cost(self, current_state, action, next_state):
        '''
        Compute ISAACS-style cost

        Use local_base_pos as state for simplicity since cost computation only require LOCAL x and y position

        Args:
            current_state: tensor of shape (num_envs, num_states=3), LOCAL base_pos at the current time step t for each env
            action: tensor of shape (num_envs, num_hl_actions=2), high-level ctrl [v, w] executed at the current time step t for each env
            next_state: tensor of shape (num_envs, num_states=3), LOCAL base_pos at the next time step t+1 for each env

        Returns:
            cost: tensor of shape (num_envs,),
                  total cost for one transition = cost(current_state, action) + cost(next_state, dummy action [0, 0])
        '''
        # combine current_state and next_state into states with shape (num_envs, num_steps=2, num_states=3)
        # Step 1: current_state, Step 2: next_state
        states = torch.stack([current_state, next_state], dim=1)  # (num_envs, 2, 3)

        # create dummy action [0, 0] for step 2
        dummy_action = torch.zeros(self.num_envs, self.num_hl_actions, device=self.device)  # (num_envs, 2)

        # combine action and dummy_action into actions with shape (num_envs, num_steps=2, num_hl_actions=2)
        # Step 1: action, Step 2: dummy_action [0, 0]
        actions = torch.stack([action, dummy_action], dim=1)  # (num_envs, 2, 2)
        
        # create dummy_time_indices with shape (num_envs, num_steps=2)
        dummy_time_indices = torch.zeros(self.num_envs, 2, device = self.device)
        
        cost = torch.sum(self.hl_cost.get_cost(states, actions, dummy_time_indices), dim = -1)
        return cost
    
    def get_hl_reward(self, current_state, action, next_state):
        '''
        Compute ISAACS-style reward = -cost, tensor of shape (num_envs, 1)
        '''
        cost = self.get_hl_cost(current_state, action, next_state)
        reward = -cost.unsqueeze(dim = 1).float()
        return reward
    
    def get_hl_constraints(self, current_state, action, next_state):
        '''
        Compute ISAACS-style constraints

        Use local_base_pos as state for simplicity since constraints computation only require LOCAL x and y position

        Args:
            current_state: tensor of shape (num_envs, num_states=3), LOCAL base_pos at the current time step t for each env
            action: tensor of shape (num_envs, num_hl_actions=2), high-level ctrl [v, w] executed at the current time step t for each env
            next_state: tensor of shape (num_envs, num_states=3), LOCAL base_pos at the next time step t+1 for each env

        Returns:
            cons_dict: dictionary with individual constraint values at each time step for each environment
                       cons_dict = {'cons_type1': tensor of shape (num_envs, num_steps),
                                    'cons_type2': tensor of shape (num_envs, num_steps),
                                    ...}
        '''
        # combine current_state and next_state into states with shape (num_envs, num_steps=2, num_states=3)
        # Step 1: current_state, Step 2: next_state
        states = torch.stack([current_state, next_state], dim=1)  # (num_envs, 2, 3)

        # create dummy action [0, 0] for step 2
        dummy_action = torch.zeros(self.num_envs, self.num_hl_actions, device=self.device)  # (num_envs, 2)

        # combine action and dummy_action into actions with shape (num_envs, num_steps=2, num_hl_actions=2)
        # Step 1: action, Step 2: dummy_action [0, 0]
        actions = torch.stack([action, dummy_action], dim=1)  # (num_envs, 2, 2)

        # create dummy_time_indices with shape (num_envs, num_steps=2)
        dummy_time_indices = torch.zeros(self.num_envs, 2, device=self.device)

        # get constraint dictionary from hl_constraint
        cons_dict = self.hl_constraint.get_cost_dict(states, actions, dummy_time_indices)

        return cons_dict
    
    def get_hl_target_margin(self, current_state, action, next_state):
        '''
        Compute ISAACS-style targets

        Placeholder for REACH-avoid cases. NOT used here
        '''
        return None
    
    def get_hl_done_and_info(self, next_state, constraints,
                             targets = None, final_only: bool = True, end_criterion = None):
        '''
        Get the done flag and a dictionary to provide additional information of the step function given current state, constraints, and targets

        Args:
            next_state: tensor of shape (num_envs, num_states=3), LOCAL base_pos at the next time step t+1 for each env
                        NOT used, for interface compatibility
            constraints: a dictionary where each (key, value) pair is the name and values of constraint function for all envs
            targets: a dictionary where each (key, value) pair is the name and values of a target margin function for all envs
                     only used for REACH-avoid case, NOT used here

        Returns:
            done: tensor of shape (num_envs,), each element is a bool (True if the episode terminates)
            info: a tuple pf num_envs info dicts
                  each dict contains info for each env, and all numerical values are float/np.ndarrays
        '''
        if end_criterion is None:
            end_criterion = self.end_criterion

        # Initialize output tensors
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        info_list = []

        # Process each environment individually
        for env_idx in range(self.num_envs):
            # Extract constraints for this single env: dict with values of shape (num_steps,)
            env_constraints = {key: value[env_idx] for key, value in constraints.items()}

            # Extract targets for this single env if provided
            env_targets = None
            if targets is not None:
                env_targets = {key: value[env_idx] for key, value in targets.items()}

            # Compute done flag and info dict for this env
            env_done, env_info = self.get_single_env_done_info(
                env_idx, env_constraints, env_targets, final_only, end_criterion
            )

            done[env_idx] = env_done
            info_list.append(env_info)

        return done, tuple(info_list)

    def get_single_env_done_info(self, env_idx, constraints, targets, final_only, end_criterion):
        '''
        Helper function to compute done flag and info dict for a single environment

        Args:
            env_idx: int, index of the environment
            constraints: dict where each value is a tensor of shape (num_steps,) for this env
            targets: dict where each value is a tensor of shape (num_steps,) for this env, or None
            final_only: bool, whether to only consider the final step
            end_criterion: str, termination criterion ('failure', 'timeout', 'reach-avoid')

        Returns:
            done: bool, True if the episode terminates
            info: dict containing info for this env with float/np.ndarray values
        '''
        done = False
        done_type = "not_raised"

        # Check timeout using episode_length_buf instead of self.cnt
        if self.episode_length_buf[env_idx] >= self.cfg.env.max_episode_length:
            done = True
            done_type = "timeout"

        # Retrieve constraint values and compute g_x_list
        # Concatenate all constraint values: stack them to get shape (num_constraint_types, num_steps)
        constraint_values_list = []
        num_pts = None
        for key, value in constraints.items():
            if num_pts is None:
                num_pts = value.shape[0]  # num_steps
            else:
                assert num_pts == value.shape[0], (
                    f"The length of constraint ({key}) do not match"
                )
            constraint_values_list.append(value)

        # Stack to get shape (num_constraint_types, num_steps)
        constraint_values = torch.stack(constraint_values_list, dim=0)

        # Take max over constraint types to get g_x_list of shape (num_steps,)
        g_x_list = torch.max(constraint_values, dim=0)[0]

        # Retrieve target values and compute l_x_list
        if targets is not None:
            target_values_list = []
            for key, value in targets.items():
                assert num_pts == value.shape[0], (
                    f"The length of target ({key}) do not match"
                )
                target_values_list.append(value)

            # Stack to get shape (num_target_types, num_steps)
            target_values = torch.stack(target_values_list, dim=0)

            # Take max over target types to get l_x_list of shape (num_steps,)
            l_x_list = torch.max(target_values, dim=0)[0]
        else:
            l_x_list = torch.full((num_pts,), fill_value=float('inf'), device=self.device)

        # Get g_x, l_x, and binary_cost based on final_only flag
        if final_only:
            g_x = float(g_x_list[-1].item())
            l_x = float(l_x_list[-1].item())
            binary_cost = 1.0 if g_x > self.failure_thr else 0.0
        else:
            g_x = g_x_list.cpu().numpy()
            l_x = l_x_list.cpu().numpy()
            binary_cost = 1.0 if torch.any(g_x_list > self.failure_thr).item() else 0.0

        # Determine done flag based on end_criterion
        if end_criterion == 'failure':
            if final_only:
                failure = torch.any(constraint_values[:, -1] > self.failure_thr).item()
            else:
                failure = torch.any(constraint_values > self.failure_thr).item()
            if failure:
                done = True
                done_type = "failure"
                g_x = self.g_x_fail

        elif end_criterion == 'reach-avoid':
            if final_only:
                failure = g_x > self.failure_thr
                success = not failure and l_x <= 0.0
            else:
                # Compute value function backward in time
                v_x_list = torch.empty(num_pts, device=self.device)
                v_x_list[num_pts - 1] = torch.max(l_x_list[num_pts - 1], g_x_list[num_pts - 1])
                for i in range(num_pts - 2, -1, -1):
                    v_x_list[i] = torch.max(g_x_list[i], torch.min(l_x_list[i], v_x_list[i + 1]))

                inst = torch.argmin(v_x_list).item()
                failure = torch.any(constraint_values[:, :inst + 1] > self.failure_thr).item()
                success = not failure and (v_x_list[inst].item() <= 0.0)

            if success:
                done = True
                done_type = "success"
            elif failure:
                done = True
                done_type = "failure"
                g_x = self.g_x_fail

        elif end_criterion == 'timeout':
            pass
        else:
            raise ValueError(f"End criterion '{end_criterion}' not supported!")

        # Build info dict with float/np.ndarray values
        info = {
            "done_type": done_type,
            "g_x": g_x,
            "l_x": l_x,
            "binary_cost": binary_cost
        }

        # Optionally include physical parameters for ISAACS network conditioning
        if self.condition_on_physical_params:
            friction = self.friction_coeffs[env_idx, 0].item()
            payload = self.payloads[env_idx].item()
            append = np.array([friction, payload], dtype=np.float32)
            info['append'] = append  # physical parameters [f, m] for current state
            info['append_nxt'] = append  # same as append since physical parameters don't change

        # Optionally include constraints and targets in info
        if self.step_keep_constraints:
            # Convert constraint tensors to numpy arrays
            info['constraints'] = {key: value.cpu().numpy() for key, value in constraints.items()}
        if self.step_keep_targets and targets is not None:
            # Convert target tensors to numpy arrays
            info['targets'] = {key: value.cpu().numpy() for key, value in targets.items()}

        return done, info
    
    # Core functions: step() interacts with the envs
    #                 reset() resets any terminated envs
    def step(self, hl_actions):
        '''
        Step all parallel environments one control-step forward simultaneously,
        Using high-level navigation actions paired with pretrained low-level locomotion policy

        Args:
            hl_actions: tensor of shape (num_envs, num_hl_actions), high-level control actions [v, w] for all envs

        Returns:
            obs: tensor of shape (num_envs, num_hl_obs), high-level navigation states at next realized step for all envs
            reward: tensor of shape (num_envs, 1), ISAACS-style reward computed from this single transition for all envs
            done: tensor of shape (num_envs, ), contains all bools, True if terminated
            info: a tuple of num_envs info dicts where each dict contains info for each env
                  all numerical values are float/np.ndarrays
        '''

        # ==================== SAVE CURRENT STATE (for ISAACS computation) ====================
        # Use LOCAL positions for cost, constraints, done and info computation by subtracting env_origins
        current_state = self.base_pos.clone() - self.env_origins  # shape: (num_envs, 3) [x, y, z] in local frame

        # ==================== HIGH-LEVEL TO LOW-LEVEL COMMAND INTERFACE ====================
        # Default commands for low-level locomotion policy
        gaits = {"pronking": [0, 0, 0],
                 "trotting": [0.5, 0, 0],
                 "bounding": [0, 0.5, 0],
                 "pacing": [0, 0, 0.5]}
        y_vel_cmd = 0.0
        body_height_cmd = 0.0
        step_frequency_cmd = 3.0
        gait = torch.tensor(gaits["trotting"])
        footswing_height_cmd = 0.08
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25

        # Set commands for next step using hl_actions (BEFORE getting low-level actions)
        v = hl_actions[:, 0]    # tensor of shape (num_envs,)
        w = hl_actions[:, 1]    # tensor of shape (num_envs,)
        # Clip to v_limits and w_limits
        v = torch.clamp(v, min=self.v_limits[0], max=self.v_limits[1])
        w = torch.clamp(w, min=self.w_limits[0], max=self.w_limits[1])

        # Update commands buffer for low-level policy
        self.commands[:, 0] = v
        self.commands[:, 1] = y_vel_cmd
        self.commands[:, 2] = w
        self.commands[:, 3] = body_height_cmd
        self.commands[:, 4] = step_frequency_cmd
        self.commands[:, 5:8] = gait
        self.commands[:, 8] = 0.5
        self.commands[:, 9] = footswing_height_cmd
        self.commands[:, 10] = pitch_cmd
        self.commands[:, 11] = roll_cmd
        self.commands[:, 12] = stance_width_cmd

        # ==================== COMPUTE LOW-LEVEL OBSERVATIONS ====================
        # Must call compute_observations() to update self.obs_buf
        self.compute_observations()

        # ==================== GET LOW-LEVEL ACTIONS FROM PRETRAINED POLICY ====================
        # Get observations (UNCLIPPED from get_observations)
        obs_dict = self.get_observations()

        # CLIP observations for low-level policy (as done during training)
        clip_obs = self.cfg.normalization.clip_observations
        obs_dict['obs'] = torch.clip(obs_dict['obs'], -clip_obs, clip_obs)
        #obs_dict['obs_history'] = torch.clip(obs_dict['obs_history'], -clip_obs, clip_obs)
        if obs_dict['privileged_obs'] is not None:
            obs_dict['privileged_obs'] = torch.clip(obs_dict['privileged_obs'], -clip_obs, clip_obs)

        # Get low-level actions
        with torch.no_grad():
            actions = self.ll_policy(obs_dict)

        # ==================== APPLY LOW-LEVEL ACTIONS TO ENVIRONMENT ====================
        # Apply actions (low-level) to interact with the env (same as LeggedRobot.step())
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # Save previous states for tracking (as in LeggedRobot.step())
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        # Render GUI if needed
        self.render_gui()

        # Step physics for decimation steps
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # ==================== POST PHYSICS STEP COMPUTATIONS ====================
        # Refresh state tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        # Increment counters
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Prepare quantities (as in LeggedRobot.post_physics_step())
        self.base_pos[:] = self.root_states[:self.num_envs, 0:3]
        self.base_quat[:] = self.root_states[:self.num_envs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:self.num_envs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]

        # Call _post_physics_step_callback() for common computations
        self._post_physics_step_callback()

        # ==================== COMPUTE ISAACS-STYLE REWARD, DONE, AND INFO ====================
        # Get next state in LOCAL frame for ISAACS computations
        next_state = self.base_pos.clone() - self.env_origins  # shape: (num_envs, 3) [x, y, z] in local frame

        # Compute ISAACS-style reward using custom get_hl_reward()
        hl_reward = self.get_hl_reward(current_state, hl_actions, next_state)  # shape: (num_envs, 1)

        # Compute constraints for done and info
        hl_constraints = self.get_hl_constraints(current_state, hl_actions, next_state)

        # Compute targets (None for avoid-only case)
        hl_targets = self.get_hl_target_margin(current_state, hl_actions, next_state)

        # Compute done and info using custom get_hl_done_and_info()
        hl_done, hl_info = self.get_hl_done_and_info(next_state, hl_constraints, hl_targets, final_only=True)

        # ==================== DO NOT AUTO-RESET ====================
        # ISAACS framework handles reset externally in the training loop.
        # Reset will be called manually by venv.reset_one() when done=True.
        # DO NOT call self.reset_idx(env_ids) here!

        # ==================== COMPUTE HIGH-LEVEL OBSERVATIONS ====================
        # Compute high-level observations for next step
        self.compute_hl_observations()
        hl_obs = self.hl_obs_buf.clone()  # shape: (num_envs, num_hl_obs)

        # ==================== UPDATE TRACKING BUFFERS ====================
        # Update last actions and other tracking buffers (as in LeggedRobot.post_physics_step())
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        # ========================= RENDER ==========================
        # Draw debug visualization if needed
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self._render_headless()

        # ========================= POST-PROCESS IN VelocityTrackingEasyEnv ==========================
        # Update foot_positions
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        # Update extras dict (important for low-level policy tracking)
        self.extras.update({
            "privileged_obs": self.privileged_obs_buf.cpu().numpy() if self.privileged_obs_buf is not None else None,
            "joint_pos": self.dof_pos.cpu().numpy(),
            "joint_vel": self.dof_vel.cpu().numpy(),
            "joint_pos_target": self.joint_pos_target.cpu().detach().numpy(),
            "joint_vel_target": torch.zeros(12).cpu().numpy(),
            "body_linear_vel": self.base_lin_vel.cpu().detach().numpy(),
            "body_angular_vel": self.base_ang_vel.cpu().detach().numpy(),
            "body_linear_vel_cmd": self.commands.cpu().numpy()[:, 0:2],
            "body_angular_vel_cmd": self.commands.cpu().numpy()[:, 2:3],
            "contact_states": (self.contact_forces[:, self.feet_indices, 2] > 1.).detach().cpu().numpy().copy(),
            "foot_positions": (self.foot_positions).detach().cpu().numpy().copy(),
            "body_pos": self.root_states[:, 0:3].detach().cpu().numpy(),
            "torques": self.torques.detach().cpu().numpy()
        })

        return hl_obs, hl_reward, hl_done, hl_info

    # Reset functions for ISAACS compatibility
    def reset_one(self, env_id: int, mode: str = 'train',
                  custom_state: Optional[torch.Tensor] = None,
                  eval_ranges: Optional[Dict[str, List[float]]] = None):
        """
        Reset a single environment specified by env_id.

        This function is compatible with ISAACS's training loop which calls:
            obs = venv.reset_one(index=env_idx)

        Args:
            env_id (int): index of the environment to reset (0 to num_envs-1)
            mode (str): reset mode - 'train', 'eval', or 'custom'
            custom_state (torch.Tensor, optional): custom RAW high-level state (33D)
                                                   [local_x, local_y, heading, base_lin_vel(3),
                                                    base_ang_vel(3), dof_pos(12), dof_vel(12)]
            eval_ranges (Dict, optional): custom ranges for evaluation mode (LOCAL frame for x,y, RAW for others)
                                         Keys: 'x', 'y', 'yaw'

        Returns:
            hl_obs (torch.Tensor): high-level observation for the reset environment,
                                   shape (num_hl_obs,), dtype float32
        """
        env_ids_tensor = torch.tensor([env_id], dtype=torch.long, device=self.device)

        # Convert custom_state to batch format if provided
        custom_states = None
        if custom_state is not None:
            if custom_state.dim() == 1:
                custom_states = custom_state.unsqueeze(0)  # Add batch dimension
            else:
                custom_states = custom_state

        self._reset_envs(env_ids_tensor, mode=mode, custom_states=custom_states, eval_ranges=eval_ranges)

        # Return the hl_obs for this single environment as torch.Tensor
        return self.hl_obs_buf[env_id].clone()

    def reset_multiple(self, env_ids: torch.Tensor, mode: str = 'train',
                       custom_states: Optional[torch.Tensor] = None,
                       eval_ranges: Optional[Dict[str, List[float]]] = None):
        """
        Reset multiple environments specified by env_ids.

        Useful for:
        - Initial reset of all environments: reset_multiple(torch.arange(num_envs))
        - Batch processing multiple terminated environments together

        Args:
            env_ids (torch.Tensor): tensor of environment indices to reset,
                                    shape (num_envs_to_reset,), dtype long
            mode (str): reset mode - 'train', 'eval', or 'custom'
            custom_states (torch.Tensor, optional): custom RAW high-level states (num_resets, 33)
                                                    [local_x, local_y, heading, base_lin_vel(3),
                                                     base_ang_vel(3), dof_pos(12), dof_vel(12)]
            eval_ranges (Dict, optional): custom ranges for evaluation mode (LOCAL frame for x,y, RAW for others)
                                         Keys: 'x', 'y', 'yaw'

        Returns:
            hl_obs (torch.Tensor): high-level observations for all reset environments,
                                   shape (num_envs_to_reset, num_hl_obs), dtype float32
        """
        self._reset_envs(env_ids, mode=mode, custom_states=custom_states, eval_ranges=eval_ranges)

        # Return hl_obs for all reset environments
        return self.hl_obs_buf[env_ids].clone()

    def _reset_envs(self, env_ids: torch.Tensor,
                    mode: str = 'train',
                    custom_states: Optional[torch.Tensor] = None,
                    eval_ranges: Optional[Dict[str, List[float]]] = None):
        """
        Internal function to reset specified environments with rejection sampling.

        This replaces the problematic reset_idx() from LeggedRobot with proper logic for:
        1. Random initial states in LOCAL frame (converted to GLOBAL for simulation)
        2. Rejection sampling to ensure initial states are SAFE (no constraint violations)
        3. Default commands (v=0, w=0) without curriculum resampling
        4. Cleared low-level policy buffers and tracking variables

        Args:
            env_ids (torch.Tensor): tensor of environment indices to reset
            mode (str): reset mode - 'train', 'eval', or 'custom'
                       - 'train': use default training ranges with rejection sampling
                       - 'eval': use tighter evaluation ranges with rejection sampling
                       - 'custom': use provided custom_states directly (no sampling)
            custom_states (torch.Tensor, optional): custom RAW high-level states for initialization
                                                    shape: (num_resets, 33)
                                                    state format: [local_x, local_y, heading,
                                                                  base_lin_vel(3) in BODY frame,
                                                                  base_ang_vel(3) in BODY frame,
                                                                  dof_pos(12) absolute values,
                                                                  dof_vel(12)]
                                                    NOTE: x, y are in LOCAL frame, velocities in BODY frame
                                                    Only used when mode='custom'
            eval_ranges (Dict, optional): custom ranges for evaluation mode
                                         If None, use default eval ranges
                                         Keys: 'x', 'y', 'yaw' (x, y in LOCAL frame)
        """
        num_resets = len(env_ids)

        # ==================== HANDLE CUSTOM STATE INITIALIZATION ====================
        if mode == 'custom':
            if custom_states is None:
                raise ValueError("custom_states must be provided when mode='custom'")
            if custom_states.shape[0] != num_resets:
                raise ValueError(f"custom_states batch size {custom_states.shape[0]} doesn't match num_resets {num_resets}")
            if custom_states.shape[1] != self.num_hl_obs:  # Should be 33
                raise ValueError(f"custom_states dimension {custom_states.shape[1]} doesn't match num_hl_obs {self.num_hl_obs}")

            # Extract RAW states from custom_states
            # custom_states format: [local_x, local_y, heading, base_lin_vel(3), base_ang_vel(3), dof_pos(12), dof_vel(12)]
            # NOTE: x, y are in LOCAL frame, velocities are in BODY frame
            local_x = custom_states[:, 0]
            local_y = custom_states[:, 1]
            heading = custom_states[:, 2]

            # Extract base velocities (RAW values in BODY frame)
            base_lin_vel = custom_states[:, 3:6]  # (num_resets, 3)
            base_ang_vel = custom_states[:, 6:9]  # (num_resets, 3)

            # Extract joint positions and velocities (RAW absolute values)
            dof_pos = custom_states[:, 9:21]  # (num_resets, 12)
            dof_vel = custom_states[:, 21:33]  # (num_resets, 12)

            # Set yaw from heading
            yaw = heading

            # Skip rejection sampling
            skip_sampling = True

        else:
            skip_sampling = False

            # ==================== DEFINE SAMPLING RANGES BASED ON MODE ====================
            if mode == 'train':
                # Training ranges - wider coverage for learning
                x_range = [-2.0, 12.0]            # LOCAL frame
                y_range = [-5.0, 5.0]             # LOCAL frame
                yaw_range = [-np.pi, np.pi]       # RAW range

            elif mode == 'eval':
                # Evaluation ranges - tighter, more realistic
                if eval_ranges is not None:
                    x_range = eval_ranges.get('x', [3.0, 8.0])
                    y_range = eval_ranges.get('y', [-3.0, 3.0])
                    yaw_range = eval_ranges.get('yaw', [-np.pi/2, np.pi/2])
                else:
                    # Default eval ranges
                    x_range = [3.0, 8.0]              # LOCAL frame
                    y_range = [-3.0, 3.0]             # LOCAL frame
                    yaw_range = [-np.pi/2, np.pi/2]   # RAW range
            else:
                raise ValueError(f"mode must be 'train', 'eval', or 'custom', got {mode}")

        # ==================== REJECTION SAMPLING FOR SAFE INITIAL STATES ====================
        # NOTE: Constraints only depend on (x, y) positions in LOCAL frame, NOT on yaw or z!
        # Robot is approximated as a circle (radius 0.35), so orientation doesn't affect collision.
        # ALL obstacles and boundaries are defined in x-y plane, so z position doesn't affect collision either.
        # We only need to rejection sample (x, y), then sample yaw independently, z should be kept at default for plane terrain.

        if not skip_sampling:
            # Keep track of which environments still need valid initial positions
            envs_to_sample = torch.ones(num_resets, dtype=torch.bool, device=self.device)

            # Preallocate tensors for sampled positions
            local_x = torch.zeros(num_resets, device=self.device)
            local_y = torch.zeros(num_resets, device=self.device)

            # Rejection sampling loop - vectorized for efficiency
            iteration = 0
            warn_threshold = 100  # Warn if taking too long

            while envs_to_sample.any():
                num_to_sample = envs_to_sample.sum().item()

                # Periodic warning for debugging
                if iteration == warn_threshold:
                    print(f"WARNING: Rejection sampling taking longer than expected ({warn_threshold} iterations).")
                    print(f"Still searching for valid states for {num_to_sample}/{num_resets} environments.")
                    print("This may indicate obstacle configuration issues.")

                # Sample candidate (x, y) positions in LOCAL frame using mode-specific ranges
                candidate_x = torch.rand(num_to_sample, device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
                candidate_y = torch.rand(num_to_sample, device=self.device) * (y_range[1] - y_range[0]) + y_range[0]

                # Use dummy z=0 for constraint checking (constraints don't depend on z)
                candidate_z = torch.zeros(num_to_sample, device=self.device)

                # Stack into state tensor for constraint checking: [x, y, z] in LOCAL frame
                candidate_states = torch.stack([candidate_x, candidate_y, candidate_z], dim=1)  # (num_to_sample, 3)

                # Create dummy action [0, 0] for constraint evaluation
                dummy_actions = torch.zeros(num_to_sample, 2, device=self.device)

                # Compute constraints for candidate states
                # get_hl_constraints expects (current_state, action, next_state) all in LOCAL frame
                # For initial state check, we use same state for both current and next
                constraints = self.get_hl_constraints(
                    candidate_states,
                    dummy_actions,
                    candidate_states
                )

                # Check if any constraint is violated (constraint > 0 means violation)
                # constraints is a dict with values of shape (num_to_sample, num_steps=2)
                # We check the first step (initial state)
                constraint_violations = torch.zeros(num_to_sample, dtype=torch.bool, device=self.device)
                for key, value in constraints.items():
                    # value shape: (num_to_sample, 2), check first step [:, 0]
                    constraint_violations |= (value[:, 0] > 0.0)

                # Accept states that have NO violations
                valid_states = ~constraint_violations

                # Assign valid (x, y) positions to their corresponding environments
                envs_to_sample_indices = torch.where(envs_to_sample)[0]
                valid_indices = envs_to_sample_indices[valid_states]

                local_x[valid_indices] = candidate_x[valid_states]
                local_y[valid_indices] = candidate_y[valid_states]

                # Mark these environments as successfully sampled
                envs_to_sample[valid_indices] = False

                iteration += 1

            # Log successful completion
            if iteration > 1:
                print(f"Rejection sampling completed in {iteration} iterations for {num_resets} environments.")

            # Sample yaw independently using mode-specific ranges (no constraints on orientation)
            yaw = torch.rand(num_resets, device=self.device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]

        # If using custom mode, local_x, local_y, and yaw are already set above

        # ==================== RESET ROOT STATES (position, orientation, velocities) ====================
        # Z position: use nominal height from config + terrain height
        local_z = torch.ones(num_resets, device=self.device) * self.base_init_state[2]

        # Convert LOCAL to GLOBAL by adding env_origins
        global_x = local_x + self.env_origins[env_ids, 0]
        global_y = local_y + self.env_origins[env_ids, 1]
        global_z = local_z + self.env_origins[env_ids, 2]

        # Set positions in root_states
        self.root_states[env_ids, 0] = global_x
        self.root_states[env_ids, 1] = global_y
        self.root_states[env_ids, 2] = global_z

        # Set orientation from sampled/provided yaw
        quat = quat_from_euler_xyz(
            torch.zeros(num_resets, device=self.device),  # roll = 0
            torch.zeros(num_resets, device=self.device),  # pitch = 0
            yaw
        )
        self.root_states[env_ids, 3:7] = quat

        # Set velocities based on mode
        if mode == 'custom':
            # Use provided RAW velocities from custom_states (in BASE/BODY frame)
            # Need to convert to WORLD frame for root_states
            world_lin_vel = quat_rotate(quat, base_lin_vel)
            world_ang_vel = quat_rotate(quat, base_ang_vel)

            self.root_states[env_ids, 7:10] = world_lin_vel
            self.root_states[env_ids, 10:13] = world_ang_vel
        else:
            # Randomize initial velocities for train and eval modes (SAME ranges)
            # Sample in BASE/BODY frame, then convert to WORLD frame

            # Linear velocity in BASE frame: broader range than just control limits for value function learning
            # x (forward): [-0.5, 2.5] m/s (covers backward drift to fast forward)
            # y (lateral): [-0.25, 0.25] m/s (small lateral movement)
            # z (vertical): [-0.25, 0.25] m/s (small vertical oscillation)
            base_lin_vel_x = torch.rand(num_resets, device=self.device) * 3.0 - 0.5  # [-0.5, 2.5]
            base_lin_vel_y = (torch.rand(num_resets, device=self.device) - 0.5) * 0.5  # [-0.25, 0.25]
            base_lin_vel_z = (torch.rand(num_resets, device=self.device) - 0.5) * 0.5  # [-0.25, 0.25]
            base_lin_vel = torch.stack([base_lin_vel_x, base_lin_vel_y, base_lin_vel_z], dim=1)  # (num_resets, 3)

            # Angular velocity in BASE frame: broader range to cover all reachable states
            # x,y (roll/pitch rate): [-0.25, 0.25] rad/s (small perturbations)
            # z (yaw rate): [-2.5, 2.5] rad/s (covers w_limits + overshoot)
            base_ang_vel_x = (torch.rand(num_resets, device=self.device) - 0.5) * 0.5  # [-0.25, 0.25]
            base_ang_vel_y = (torch.rand(num_resets, device=self.device) - 0.5) * 0.5  # [-0.25, 0.25]
            base_ang_vel_z = (torch.rand(num_resets, device=self.device) - 0.5) * 5.0  # [-2.5, 2.5]
            base_ang_vel = torch.stack([base_ang_vel_x, base_ang_vel_y, base_ang_vel_z], dim=1)  # (num_resets, 3)

            # Convert BASE frame velocities to WORLD frame
            world_lin_vel = quat_rotate(quat, base_lin_vel)
            world_ang_vel = quat_rotate(quat, base_ang_vel)

            self.root_states[env_ids, 7:10] = world_lin_vel
            self.root_states[env_ids, 10:13] = world_ang_vel

        # ==================== RESET DOF STATES (joint positions and velocities) ====================
        if mode == 'custom':
            # Use provided RAW joint states from custom_states
            self.dof_pos[env_ids, :self.num_actuated_dof] = dof_pos
            self.dof_vel[env_ids, :self.num_actuated_dof] = dof_vel

            # Set non-actuated DOFs to default if any exist
            if self.num_dof > self.num_actuated_dof:
                self.dof_pos[env_ids, self.num_actuated_dof:] = self.default_dof_pos[env_ids, self.num_actuated_dof:]
                self.dof_vel[env_ids, self.num_actuated_dof:] = 0.0
        else:
            # Randomize joint positions around default for train and eval modes (SAME ranges)
            # This provides wide coverage of joint configuration space
            self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * torch_rand_float(
                0.5, 1.5, (num_resets, self.num_dof), device=self.device
            )

            # Joint velocities set to zero (clean initial state, avoids instability)
            self.dof_vel[env_ids] = 0.0

        # ==================== SET DEFAULT COMMANDS (NO CURRICULUM!) ====================
        # Set commands to default values (v=0, w=0, etc.)
        # DO NOT call self.resample_commands() which uses curriculum!
        self.commands[env_ids, 0] = 0.0   # x velocity (forward)
        self.commands[env_ids, 1] = 0.0   # y velocity (lateral)
        self.commands[env_ids, 2] = 0.0   # yaw rate
        self.commands[env_ids, 3] = 0.0   # body height
        self.commands[env_ids, 4] = 3.0   # step frequency (default)
        self.commands[env_ids, 5:8] = torch.tensor([0.5, 0, 0], device=self.device)  # gait (trotting)
        self.commands[env_ids, 8] = 0.5   # phase offset
        self.commands[env_ids, 9] = 0.08  # footswing height
        self.commands[env_ids, 10] = 0.0  # pitch
        self.commands[env_ids, 11] = 0.0  # roll
        self.commands[env_ids, 12] = 0.25 # stance width

        # ==================== RESET LOW-LEVEL POLICY BUFFERS ====================
        # Clear observation history for low-level policy
        self.obs_history[env_ids] = 0.0

        # Reset tracking buffers
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0
        if hasattr(self, 'last_joint_pos_target'):
            self.last_joint_pos_target[env_ids] = self.default_dof_pos[env_ids]
            self.last_last_joint_pos_target[env_ids] = self.default_dof_pos[env_ids]

        # Reset gait phase tracking (if using adaptive gait)
        if hasattr(self, 'gait_indices'):
            self.gait_indices[env_ids] = 0.0

        # Reset lag buffer (actuator delay simulation)
        if hasattr(self, 'lag_buffer'):
            for i in range(len(self.lag_buffer)):
                self.lag_buffer[i][env_ids] = 0.0

        # ==================== RESET COUNTERS ====================
        self.episode_length_buf[env_ids] = 0

        # ==================== APPLY RESET TO SIMULATION ====================
        # Convert env_ids to int32 for Isaac Gym API
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Set actor root states in simulation
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # Set DOF states in simulation
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

        # ==================== REFRESH AND UPDATE INTERNAL STATE REPRESENTATIONS ====================
        # CRITICAL: Must refresh and update internal states BEFORE computing observations!
        # Otherwise, compute_hl_observations() will use STALE values from terminated episode.

        # Refresh state tensors from simulation
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Update internal state tracking variables for reset environments
        self.base_pos[env_ids] = self.root_states[env_ids, 0:3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids],
            self.root_states[env_ids, 7:10]
        )
        self.base_ang_vel[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids],
            self.root_states[env_ids, 10:13]
        )
        self.projected_gravity[env_ids] = quat_rotate_inverse(
            self.base_quat[env_ids],
            self.gravity_vec[env_ids]
        )

        # Update foot positions (needed for low-level policy)
        self.foot_positions[env_ids] = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[env_ids, self.feet_indices, 0:3]

        # ==================== COMPUTE OBSERVATIONS FOR RESET ENVIRONMENTS ====================
        # Compute low-level observations (needed for tracking and buffers)
        self.compute_observations()

        # Compute high-level observations (what we return to ISAACS)
        self.compute_hl_observations()

    # Evaluation environment selection functions
    def select_eval_envs(self, strategy: str = 'grid',
                         num_f_points: int = 5, num_m_points: int = 21,
                         f_range: Optional[List[float]] = None, m_range: Optional[List[float]] = None, exclude_middle_m: bool = False):
        """
        Select representative environments for evaluation based on physical parameters (friction, payload).

        This function implements stratified sampling across the (f, m) parameter space to ensure
        comprehensive evaluation coverage while avoiding redundant testing of similar conditions.

        Args:
            strategy (str): Selection strategy
                - 'stratified': Use bin centers for systematic grid sampling 
                - 'grid': Use uniform grid points (bin edges)
                - 'bins': Random selection within each bin
            num_f_points (int): Number of friction values to sample (default: 5 for 'grid')
            num_m_points (int): Number of payload values to sample (default: 21 for 'grid')
            f_range (List[float], optional): [f_min, f_max] friction range
                If None, auto-detect from self.friction_coeffs
            m_range (List[float], optional): [m_min, m_max] payload range
                If None, auto-detect from self.payloads
            exclude_middle_m (bool): If True, only select envs with |m| >= 0.5
                Useful for testing extreme conditions only (default: False)

        Returns:
            selected_env_ids (torch.Tensor): Selected environment IDs, shape (num_selected,), dtype long
            grid_mapping (Dict): Maps (f_rep, m_rep) -> env_id for each grid point
            metadata (Dict): Contains:
                - 'f_values': List of representative friction values
                - 'm_values': List of representative payload values
                - 'strategy': Selection strategy used
                - 'num_selected': Number of environments selected
                - 'f_range': Actual friction range used
                - 'm_range': Actual payload range used

        Notes:
            - Requires self.friction_coeffs and self.payloads to be initialized
            - Uses Euclidean distance with scaling to handle different parameter ranges
            - Ensures no environment is selected twice (unique selection)
            - If a grid point has no nearby environments, it's skipped with a warning
        """
        # Auto-detect parameter ranges if not provided
        if f_range is None:
            f_min = self.friction_coeffs[:, 0].min().item()
            f_max = self.friction_coeffs[:, 0].max().item()
            f_range = [f_min, f_max]

        if m_range is None:
            m_min = self.payloads.min().item()
            m_max = self.payloads.max().item()
            m_range = [m_min, m_max]

        # Generate representative values based on strategy
        if strategy == 'stratified':
            # Use bin centers for more representative sampling
            f_bin_width = (f_range[1] - f_range[0]) / num_f_points
            f_values = [f_range[0] + f_bin_width * (i + 0.5) for i in range(num_f_points)]

            m_bin_width = (m_range[1] - m_range[0]) / num_m_points
            m_values = [m_range[0] + m_bin_width * (i + 0.5) for i in range(num_m_points)]

        elif strategy == 'grid':
            # Use uniform grid points (edges)
            f_values = [f_range[0] + (f_range[1] - f_range[0]) * i / (num_f_points - 1)
                       for i in range(num_f_points)]
            m_values = [m_range[0] + (m_range[1] - m_range[0]) * i / (num_m_points - 1)
                       for i in range(num_m_points)]

        elif strategy == 'bins':
            # For bins strategy, we'll create bin edges and select randomly within each
            f_values = [f_range[0] + (f_range[1] - f_range[0]) * (i + 0.5) / num_f_points
                       for i in range(num_f_points)]
            m_values = [m_range[0] + (m_range[1] - m_range[0]) * (i + 0.5) / num_m_points
                       for i in range(num_m_points)]

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Must be 'stratified', 'grid', or 'bins'")

        # Filter m_values if excluding middle range
        if exclude_middle_m:
            m_values = [m for m in m_values if abs(m) >= 0.5]

        # Prepare environment data for selection
        # Convert to CPU numpy for easier manipulation
        friction_np = self.friction_coeffs[:, 0].cpu().numpy()  # Shape: (num_envs,)
        payload_np = self.payloads.cpu().numpy()  # Shape: (num_envs,)

        # Build list of (env_id, f, m) tuples
        all_envs = [(i, friction_np[i], payload_np[i]) for i in range(self.num_envs)]

        # Select environments closest to each grid point
        selected_env_ids = []
        grid_mapping = {}

        # Calculate scaling factor for distance metric
        # Scale m by (f_range_width / m_range_width) to give equal importance
        f_range_width = f_range[1] - f_range[0]
        m_range_width = m_range[1] - m_range[0]
        m_scale = f_range_width / m_range_width if m_range_width > 0 else 1.0

        for f_rep in f_values:
            for m_rep in m_values:
                # Find closest environment to (f_rep, m_rep) that hasn't been selected
                best_env_id = None
                best_dist = float('inf')

                for env_id, f, m in all_envs:
                    # Skip if already selected
                    if env_id in selected_env_ids:
                        continue

                    # Compute weighted Euclidean distance
                    dist = np.sqrt((f - f_rep)**2 + ((m - m_rep) * m_scale)**2)

                    if dist < best_dist:
                        best_dist = dist
                        best_env_id = env_id

                # Add to selection if found
                if best_env_id is not None:
                    selected_env_ids.append(best_env_id)
                    grid_mapping[(f_rep, m_rep)] = best_env_id
                else:
                    # Warn if no environment found (shouldn't happen unless num_envs < num_grid_points)
                    print(f"Warning: No available environment found for (f={f_rep:.3f}, m={m_rep:.3f})")

        # Convert to torch tensor
        selected_env_ids_tensor = torch.tensor(selected_env_ids, dtype=torch.long, device=self.device)

        # Build metadata dict
        metadata = {
            'f_values': f_values,
            'm_values': m_values,
            'strategy': strategy,
            'num_selected': len(selected_env_ids),
            'f_range': f_range,
            'm_range': m_range,
            'exclude_middle_m': exclude_middle_m,
            'num_f_points': num_f_points,
            'num_m_points': num_m_points
        }

        # Print summary
        '''print("=" * 70)
        print("Evaluation Environment Selection Summary")
        print("=" * 70)
        print(f"Strategy: {strategy}")
        print(f"Friction range: [{f_range[0]:.3f}, {f_range[1]:.3f}]")
        print(f"Payload range: [{m_range[0]:.3f}, {m_range[1]:.3f}] kg")
        print(f"Friction points: {num_f_points} → {f_values}")
        print(f"Payload points: {len(m_values)} → {m_values[:5]}... (showing first 5)")
        print(f"Exclude middle m (|m| < 0.5): {exclude_middle_m}")
        print(f"Total grid points: {num_f_points} × {len(m_values)} = {num_f_points * len(m_values)}")
        print(f"Environments selected: {len(selected_env_ids)}")
        print(f"Available environments: {self.num_envs}")
        print("=" * 70)'''

        return selected_env_ids_tensor, grid_mapping, metadata

    # Trajectory simulation functions for evaluation and visualization
    def _get_state_for_trajectory(self, env_id: int) -> torch.Tensor:
        """
        Extract RAW state in LOCAL frame for trajectory logging and visualization.

        This function extracts the physical state (not normalized observations) needed for:
        - Trajectory visualization (plotting x-y paths)
        - Evaluation metrics

        Args:
            env_id (int): environment index

        Returns:
            torch.Tensor: state vector of shape (3,) containing [x, y, heading] in LOCAL frame
                         - x, y: position in meters (RAW, not normalized)
                         - heading: yaw angle in radians [-π, π] (RAW)
        """
        # Get LOCAL position in meters (RAW, not normalized)
        local_pos = self.base_pos[env_id] - self.env_origins[env_id]  # (3,) [x, y, z]

        # Get heading angle in radians (RAW)
        forward = quat_apply(self.base_quat[env_id:env_id+1], self.forward_vec[env_id:env_id+1])
        heading = torch.atan2(forward[0, 1], forward[0, 0])  # scalar, range: [-π, π]

        # Return [x_raw, y_raw, heading_raw] for visualization
        return torch.tensor([local_pos[0], local_pos[1], heading], device=self.device, dtype=torch.float32)

    def simulate_one_trajectory(
        self, selected_env_ids: torch.Tensor, T_rollout_steps: int,
        end_criterion: str = 'failure',
        action_kwargs: Optional[Dict] = None,
        rollout_step_callback: Optional[Callable] = None,
        rollout_episode_callback: Optional[Callable] = None,
        return_info: bool = True
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Optional[List[Dict]]]:
        """
        Simulate one trajectory for EACH selected environment simultaneously.

        This function steps all environments in parallel (Isaac Gym requirement), but only
        tracks and logs data for the selected environments. Each selected environment may
        terminate at different times (different trajectory lengths).

        Args:
            selected_env_ids (torch.Tensor): environment indices to simulate, shape (num_selected,)
            T_rollout_steps (int): maximum rollout horizon in control steps
            end_criterion (str): termination criterion ('failure', 'timeout', 'reach-avoid')
                                 For quadruped: use 'failure' (terminate on constraint violation OR timeout)
            action_kwargs (Dict, optional): keyword arguments for get_action() function
            rollout_step_callback (Callable, optional): callback function executed after each step
            rollout_episode_callback (Callable, optional): callback function executed after each episode
            return_info (bool): whether to build and return info dictionaries (default: True)
                               Set to False to skip building info dicts for faster evaluation

        Returns:
            state_hists (List[List[torch.Tensor]]): length num_selected
                                              Each element is a list of state tensors representing ONE trajectory
                                              Inner list length: T_i+1 (varies per env based on termination)
                                              Each state tensor shape: (3,) containing RAW [x, y, heading] in LOCAL frame
                                              Structure: [[tensor(3,), tensor(3,), ...], [tensor(3,), ...], ...]
            results (torch.Tensor): shape (num_selected,), dtype long
                                   Result codes: 1 = success, -1 = failure, 0 = timeout
            lengths (torch.Tensor): shape (num_selected,), dtype long
                                   Length of each trajectory (number of states = steps + 1)
            infos (List[Dict] or None): length num_selected, info dictionaries for each selected env
                               Each dict contains: obs_hist, action_hist, reward_hist, step_hist
                               Returns None if return_info=False
        """
        if action_kwargs is None:
            action_kwargs = {}

        num_selected = len(selected_env_ids)

        # Track which selected envs are still active (not yet terminated)
        active_mask = torch.ones(num_selected, dtype=torch.bool, device=self.device)

        # Storage for each selected env
        state_hists = [[] for _ in range(num_selected)]
        obs_hists = [[] for _ in range(num_selected)]
        action_hists = [[] for _ in range(num_selected)]
        reward_hists = [[] for _ in range(num_selected)]
        step_hists = [[] for _ in range(num_selected)]

        results = torch.zeros(num_selected, dtype=torch.long, device=self.device)
        lengths = torch.zeros(num_selected, dtype=torch.long, device=self.device)

        # Store INITIAL state for all selected envs (before any actions)
        for i, env_id in enumerate(selected_env_ids):
            state_hists[i].append(self._get_state_for_trajectory(env_id.item()))

        # Main simulation loop
        for step_idx in range(T_rollout_steps):
            # ============================================================
            # VECTORIZED ACTION COMPUTATION
            # Get observations for ALL selected envs in one batch
            # ============================================================
            hl_obs_batch = self.hl_obs_buf[selected_env_ids]  # Shape: (num_selected, num_hl_obs)

            # Prepare action_kwargs with append if conditioning on physical parameters
            if self.condition_on_physical_params:
                # Get physical parameters for all selected envs as tensors
                friction_batch = self.friction_coeffs[selected_env_ids, 0]  # (num_selected,)
                payload_batch = self.payloads[selected_env_ids]  # (num_selected,)
                append_batch = torch.stack([friction_batch, payload_batch], dim=1)  # (num_selected, 2)

                # Create action_kwargs with batched append
                action_kwargs_batch = action_kwargs.copy() if action_kwargs else {}
                action_kwargs_batch['append'] = append_batch
            else:
                action_kwargs_batch = action_kwargs if action_kwargs else {}

            # Single batched forward pass through policy network for all selected envs
            with torch.no_grad():
                actions_batch, solver_info = self.get_action(hl_obs_batch, **action_kwargs_batch)
                # actions_batch shape: (num_selected, num_hl_actions)

            # Prepare actions for ALL envs (Isaac Gym requirement)
            hl_actions = torch.zeros(self.num_envs, self.num_hl_actions, device=self.device)

            # Assign computed actions to selected envs, masked by active_mask
            # Zero out actions for terminated envs (active_mask[i] == False)
            active_actions = actions_batch * active_mask.unsqueeze(1).float()  # Broadcasting: (num_selected, num_hl_actions)
            hl_actions[selected_env_ids] = active_actions

            # Step ALL environments simultaneously (Isaac Gym parallelization)
            obs_all, rewards_all, dones_all, infos_all = self.step(hl_actions)

            # Process results for ACTIVE selected envs only
            for i, env_id in enumerate(selected_env_ids):
                env_id_item = env_id.item()

                if active_mask[i]:
                    # Log data for this active selected env
                    state_hists[i].append(self._get_state_for_trajectory(env_id_item))
                    obs_hists[i].append(obs_all[env_id_item].clone().cpu())
                    action_hists[i].append(hl_actions[env_id_item].clone().cpu())
                    reward_hists[i].append(rewards_all[env_id_item].item())
                    step_hists[i].append(infos_all[env_id_item])

                    # Check if this env terminated (either timeout OR constraint violation)
                    if dones_all[env_id_item]:
                        active_mask[i] = False
                        lengths[i] = len(state_hists[i])  # Number of states (includes initial state)

                        # Determine result based on done_type
                        done_type = infos_all[env_id_item]['done_type']
                        if done_type == 'failure':
                            results[i] = -1  # Failure (constraint violated)
                        elif done_type == 'success':
                            results[i] = 1   # Success (reached goal)
                        else:  # 'timeout' or 'not_raised'
                            results[i] = 0   # Timeout

            # Execute step callback if provided
            if rollout_step_callback is not None:
                rollout_step_callback(
                    self, state_hists, action_hists, None, step_hists, time_idx=step_idx
                )

            # Early exit if ALL selected envs have terminated
            if not active_mask.any():
                break

        # Handle any envs that are still active after loop (should not happen if T_rollout_steps == max_episode_length)
        for i in range(num_selected):
            if active_mask[i]:
                # Should have timed out, but didn't get marked
                results[i] = 0  # Timeout
                lengths[i] = len(state_hists[i])

        # Execute episode callback if provided
        if rollout_episode_callback is not None:
            rollout_episode_callback(self, state_hists, action_hists, None, step_hists)

        # Build info dictionaries for each selected env (only if requested)
        if return_info:
            infos = []
            for i in range(num_selected):
                info_dict = {
                    'obs_hist': torch.stack(obs_hists[i]).numpy() if len(obs_hists[i]) > 0 else np.array([]),
                    'action_hist': torch.stack(action_hists[i]).numpy() if len(action_hists[i]) > 0 else np.array([]),
                    'reward_hist': np.array(reward_hists[i]),
                    'step_hist': step_hists[i],
                    'plan_hist': [],  # Not used for quadruped (no MPC-style planning)
                    'shield_ind': []  # Not used for quadruped (no shield in this context)
                }
                infos.append(info_dict)
        else:
            infos = None

        return state_hists, results, lengths, infos

    def simulate_trajectories(
        self, selected_env_ids: torch.Tensor, num_trajectories_per_env: int,
        T_rollout_s: float = 3.0, end_criterion: str = 'failure',
        eval_ranges: Optional[Dict[str, List[float]]] = None,
        action_kwargs: Optional[Dict] = None,
        rollout_step_callback: Optional[Callable] = None,
        rollout_episode_callback: Optional[Callable] = None,
        return_info: bool = False,
        use_tqdm: bool = False
    ):
        """
        Simulate multiple trajectories for selected evaluation environments.

        This is the main interface for evaluation and visualization. It simulates
        num_trajectories_per_env trajectories for each selected environment, with
        each trajectory starting from a newly sampled initial condition.

        Args:
            selected_env_ids (torch.Tensor): environment indices to evaluate, shape (num_selected,)
            num_trajectories_per_env (int): number of trajectories to simulate per selected env
            T_rollout_s (float): maximum trajectory duration in seconds (default: 3.0)
                                Should match episode_length_s from training config
            end_criterion (str): termination criterion (default: 'failure')
                                - 'failure': terminate on constraint violation OR timeout
                                - 'timeout': only terminate on timeout
                                - 'reach-avoid': terminate on success, failure, OR timeout
            eval_ranges (Dict, optional): custom ranges for sampling initial states
                                         Keys: 'x', 'y', 'yaw' (x, y in LOCAL frame, yaw in radians)
                                         If None, use default eval ranges from reset function
            action_kwargs (Dict, optional): keyword arguments for get_action() function
            rollout_step_callback (Callable, optional): callback after each step
            rollout_episode_callback (Callable, optional): callback after each episode
            return_info (bool): whether to return info dictionaries (default: False)
            use_tqdm (bool): whether to show progress bar (default: False)

        Returns:
            Without return_info (default):
                trajectories (List[np.ndarray]): length num_selected * num_trajectories_per_env
                                                Each element shape (T_i+1, 3) with [x, y, heading]
                results (np.ndarray): shape (num_selected * num_trajectories_per_env,)
                                     Result codes: 1 = success, -1 = failure, 0 = timeout
                lengths (np.ndarray): shape (num_selected * num_trajectories_per_env,)
                                     Length of each trajectory (number of states)

            With return_info=True:
                trajectories, results, lengths, info_list
                info_list (List[Dict]): length num_selected * num_trajectories_per_env
                                       Each dict contains obs_hist, action_hist, reward_hist, etc.

        Example usage:
            # Select representative envs
            selected_ids, _, _ = env.select_eval_envs(strategy='grid', num_f_points=5, num_m_points=21)

            # Simulate 10 trajectories per selected env
            trajectories, results, lengths = env.simulate_trajectories(
                selected_env_ids=selected_ids,
                num_trajectories_per_env=10,
                T_rollout_s=3.0,
                end_criterion='failure',
                eval_ranges={'x': [3, 7], 'y': [-3, 3], 'yaw': [-np.pi/2, np.pi/2]}
            )

            # Compute metrics
            safe_rate = np.sum(results != -1) / len(results)
            avg_length = np.mean(lengths)
        """
        # Convert time to steps
        # Assumes self.dt is the control timestep in seconds
        # dt = decimation * sim_dt (e.g., 10 * 0.002 = 0.02s = 50Hz control frequency)
        dt = self.cfg.control.decimation * self.sim_params.dt
        T_rollout_steps = int(T_rollout_s / dt)

        num_selected = len(selected_env_ids)
        total_num_trajs = num_selected * num_trajectories_per_env

        # Storage for all trajectories
        all_trajectories = []
        all_results = []
        all_lengths = []
        all_infos = []

        # Optional progress bar
        if use_tqdm:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(num_trajectories_per_env), desc="Simulating trajectories")
            except ImportError:
                iterator = range(num_trajectories_per_env)
                print("tqdm not available, proceeding without progress bar")
        else:
            iterator = range(num_trajectories_per_env)

        # Loop through trajectories: each iteration simulates ONE trajectory for EACH selected env
        for traj_idx in iterator:
            # Reset ALL selected envs to NEW initial conditions sampled from eval_ranges
            self.reset_multiple(selected_env_ids, mode='eval', eval_ranges=eval_ranges)

            # Simulate one trajectory for each selected env simultaneously
            state_hists, results, lengths, infos = self.simulate_one_trajectory(
                selected_env_ids=selected_env_ids,
                T_rollout_steps=T_rollout_steps,
                end_criterion=end_criterion,
                action_kwargs=action_kwargs,
                rollout_step_callback=rollout_step_callback,
                rollout_episode_callback=rollout_episode_callback,
                return_info=return_info
            )

            # Accumulate results from this batch
            all_trajectories.extend(state_hists)
            all_results.append(results)
            all_lengths.append(lengths)
            if return_info:
                all_infos.extend(infos)

        # Concatenate results across all iterations
        results_tensor = torch.cat(all_results)  # (total_num_trajs,)
        lengths_tensor = torch.cat(all_lengths)  # (total_num_trajs,)

        # Convert trajectories from List[List[torch.Tensor]] to List[np.ndarray]
        # all_trajectories structure after extend:
        #   - Type: List[List[torch.Tensor]]
        #   - Length: num_selected * num_trajectories_per_env (total number of trajectories)
        #   - Each element is ONE trajectory stored as a list of state tensors
        #   - Inner list contains T_i+1 state tensors, each of shape (3,)
        trajectories_np = []
        for state_timesteps in all_trajectories:
            # state_timesteps is ONE trajectory: [tensor(3,), tensor(3,), ..., tensor(3,)]
            # Stack individual state tensors into a single trajectory array
            traj_tensor = torch.stack(state_timesteps)  # (T_i+1, 3)
            trajectories_np.append(traj_tensor.cpu().numpy())

        # Convert results and lengths to numpy
        results_np = results_tensor.cpu().numpy()
        lengths_np = lengths_tensor.cpu().numpy()

        # Return based on return_info flag
        if return_info:
            return trajectories_np, results_np, lengths_np, all_infos
        else:
            return trajectories_np, results_np, lengths_np

    def _create_envs(self):
        """
        Override parent _create_envs to create obstacles with proper global positioning for each environment.
        Each environment has the same LOCAL geometry but different GLOBAL positions based on env_origins.

        This is identical to the parent implementation except for obstacle creation (lines marked with MODIFIED).
        """
        from go1_gym.utils.helpers import to_torch
        from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis
        from utils.simulation_utils.obstacle import CircularObstacle, BoxObstacle
        from utils.simulation_utils.environment import CustomGroundEnvironment

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
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._call_train_eval(self._get_env_origins, torch.arange(self.num_envs, device=self.device))
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._call_train_eval(self._randomize_rigid_body_props, torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "anymal", i,
                                                  self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            # create custom ground (if applicable)
            if self.task is not None and isinstance(self.task.environment, CustomGroundEnvironment):
                def add_custom_ground(x, y, l, w, h, f, c):
                    asset_options = gymapi.AssetOptions()
                    asset_options.disable_gravity = True
                    asset_options.fix_base_link = True
                    asset_options.replace_cylinder_with_capsule = True
                    asset_rigid_shape_properties = gymapi.RigidShapeProperties()
                    asset_rigid_shape_properties.friction = f-1
                    pose = gymapi.Transform()
                    pose.p = gymapi.Vec3(x, y, -h/2)
                    asset_ground = self.gym.create_box(self.sim, l, w, h, asset_options)
                    self.gym.set_asset_rigid_shape_properties(asset_ground, [asset_rigid_shape_properties])
                    rotation = Rotation.from_euler(seq='zyx', angles=[0, 0, 0], degrees=False)
                    pose.r = gymapi.Quat(*rotation.as_quat())
                    ground_handle = self.gym.create_actor(self.envs[i], asset_ground, pose, 'ground', i, 0, 1)
                    self.gym.set_rigid_body_color(env_handle, ground_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(*c))
                for gpd in self.task.environment.ground_parameter_dicts:
                    # MODIFIED: Convert local ground position to global
                    global_x = gpd['x'] + self.env_origins[i, 0].item()
                    global_y = gpd['y'] + self.env_origins[i, 1].item()
                    add_custom_ground(global_x, global_y, gpd['length'], gpd['width'], 0.1, gpd['friction'], gpd['color'])

            # create obstacles
            # MODIFIED: Convert local obstacle positions to global positions for each environment
            if self.task is not None:
                for obstacle in self.task.environment.obstacles:
                    asset_options = gymapi.AssetOptions()
                    asset_options.disable_gravity = True
                    asset_options.fix_base_link = True
                    asset_options.replace_cylinder_with_capsule = True
                    pose = gymapi.Transform()

                    # MODIFIED: Convert local obstacle position to global by adding env_origin
                    # obstacle.center is in local frame, add env_origin[i] to get global position
                    global_x = obstacle.center[0] + self.env_origins[i, 0].item()
                    global_y = obstacle.center[1] + self.env_origins[i, 1].item()
                    pose.p = gymapi.Vec3(global_x, global_y, obstacle.height/2)

                    if isinstance(obstacle, CircularObstacle):
                        asset_obstacle = self.gym.create_capsule(self.sim, obstacle.radius, obstacle.height, asset_options)
                        pose.r = gymapi.Quat(0.5, 0.5, -0.5, 0.5)
                    elif isinstance(obstacle, BoxObstacle):
                        asset_obstacle = self.gym.create_box(self.sim, obstacle.length, obstacle.width, obstacle.height, asset_options)
                        rotation = Rotation.from_euler(seq='zyx', angles=[obstacle.angle, 0, 0], degrees=False)
                        pose.r = gymapi.Quat(*rotation.as_quat())
                    else:
                        raise NotImplementedError
                    obstacle_handle = self.gym.create_actor(self.envs[i], asset_obstacle, pose, 'obstacle', i, 0, 1)
                    self.gym.set_rigid_body_color(env_handle, obstacle_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.5, 0.5, 0.5))

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])
        # if recording video, set up camera
        if self.cfg.env.record_video:
            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 360
            self.camera_props.height = 240
            self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
            self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1.5, 1, 3.0),
                                         gymapi.Vec3(0, 0, 0))
            if self.eval_cfg is not None:
                self.rendering_camera_eval = self.gym.create_camera_sensor(self.envs[self.num_train_envs],
                                                                           self.camera_props)
                self.gym.set_camera_location(self.rendering_camera_eval, self.envs[self.num_train_envs],
                                             gymapi.Vec3(1.5, 1, 3.0),
                                             gymapi.Vec3(0, 0, 0))
        self.video_writer = None
        self.video_frames = []
        self.video_frames_eval = []
        self.complete_video_frames = []
        self.complete_video_frames_eval = []

