from typing import Optional, List, Dict
from functools import partial
import numpy as np
import torch
from abc import ABC, abstractmethod
from functools import partial
import sys
import os

# Add the path to import obstacle classes from observation-conditioned-reachability
# __file__ is AdaptiveSafetyFilter_Quadruped/dubins3d_cost.py
# Going up one level (..) gives us /home/cassie/Quadruped
# Then we append observation-conditioned-reachability to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability'))
from utils.simulation_utils.obstacle import CircularObstacle, BoxObstacle

# custom helper function to transform a cost into an exponential barrier cost
def apply_barrier(cost, q1, q2, clip_min, clip_max):
    '''
    barrier_cost = q1 * exp(q2 * clip(cost, clip_min, clip_max))

    Args:
        cost: tensor of shape (num_steps, )
        q1: scalar, scaling factor (controls overall magnitude)
        q2: scalar, barrier steepness (higher = steeper penalty growth)
        clip_min, clip_max: scalars, bounds on single cost value

    Returns:
        barrier_cost: tensor of shape (num_steps, )
    '''
    clipped_cost = torch.clamp(cost, clip_min, clip_max)
    barrier_cost = q1 * torch.exp(q2 * clipped_cost)
    return barrier_cost

# compute cost for a SINGLE circular obstacle for ALL envs
def compute_circular_obstacle_cost(state, obs_center, obs_radius, robot_radius, buffer = 0.0):
    '''
    Signed distance cost for a circular robot vs a circular obstacle
    - cost > 0: collision (penetration depth)
    - cost = 0: touching
    - cost < 0: safe (clearance distance)

    Args:
        state: tensor of shape (num_envs, num_states or state_dim)
        obs_center: numpy array of shape (2,), obstacle position [x, y]
        obs_radius: float, obstacle radius
        robot_radius: float, robot radius
        buffer: float, additional safety margin, default = 0.0
        
    Returns:
        cost: tensor of shape (num_envs,)
    '''
    # Extract robot x, y positions from state (assumes x at index 0, y at index 1)
    # state shape: (num_envs, state_dim) -> robot_pos shape: (num_envs, 2)
    robot_pos = state[:, :2]  # [x, y] for all environments

    # Convert obs_center to torch tensor on the same device as state
    obs_center_tensor = torch.tensor(obs_center, dtype=state.dtype, device=state.device)

    # Compute distance from each robot to the obstacle center
    # robot_pos shape: (num_envs, 2), obs_center_tensor shape: (2,)
    # distance shape: (num_envs,)
    distance = torch.linalg.norm(robot_pos - obs_center_tensor, dim=1)

    # Signed distance constraint:
    # safe_distance = robot_radius + obs_radius + buffer
    # cost = safe_distance - actual_distance
    # When actual_distance < safe_distance: cost > 0 (violation/collision)
    # When actual_distance >= safe_distance: cost <= 0 (safe)
    # cost shape: (num_envs,)
    cost = (robot_radius + obs_radius + buffer) - distance

    return cost

# compute cost for a SINGLE working area upperbound for ALL envs
def compute_upperbound_cost(state, bound, axis, robot_radius, buffer = 0.0):
    '''
    Signed distance cost for a circular robot vs an upperbound
    - cost > 0: out of bound (the specified state axis value > upperbound)
    - cost = 0: at edge
    - cost < 0: in bound (the specified state axis value < upperbound)

    Args:
        state: tensor of shape (num_envs, num_states or state_dim)
        bound: float, bound value
        axis: string, 'x' or 'y', axis where the bound is applied
              assume x at index 0 in state, y at index 1 in state
        robot_radius: float, robot radius
        buffer: float, additional safety margin, default = 0.0

    Returns:
        cost: tensor of shape (num_envs,)
    '''
    # Determine which dimension to use based on axis
    if axis == 'x':
        dim = 0
    elif axis == 'y':
        dim = 1
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")

    # Extract the relevant state dimension for all environments
    # state shape: (num_envs, state_dim) -> state_value shape: (num_envs,)
    state_value = state[:, dim]

    # For upperbound constraint with circular robot:
    # The robot violates the bound when: state_value + robot_radius + buffer > bound
    # Rearranging: cost = (state_value + robot_radius + buffer) - bound
    # cost > 0: violation (robot exceeds upperbound)
    # cost = 0: robot is exactly at the edge
    # cost < 0: safe (robot is within bound)
    cost = (state_value + robot_radius + buffer) - bound

    return cost

# compute cost for a SINGLE working area lowerbound for ALL envs
def compute_lowerbound_cost(state, bound, axis, robot_radius, buffer = 0.0):
    '''
    Signed distance cost for a circular robot vs a lowerbound
    - cost > 0: out of bound (the specified state axis value < lowerbound)
    - cost = 0: at edge
    - cost < 0: in bound (the specified state axis value > lowerbound)

    Args:
        state: tensor of shape (num_envs, num_states or state_dim)
        bound: float, bound value
        axis: string, 'x' or 'y', axis where the bound is applied
              assume x at index 0 in state, y at index 1 in state
        robot_radius: float, robot radius
        buffer: float, additional safety margin, default = 0.0

    Returns:
        cost: tensor of shape (num_envs,)
    '''
    # Determine which dimension to use based on axis
    if axis == 'x':
        dim = 0
    elif axis == 'y':
        dim = 1
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis}")

    # Extract the relevant state dimension for all environments
    # state shape: (num_envs, state_dim) -> state_value shape: (num_envs,)
    state_value = state[:, dim]

    # For lowerbound constraint with circular robot:
    # The robot violates the bound when: state_value - robot_radius - buffer < bound
    # Rearranging: cost = bound - (state_value - robot_radius - buffer)
    # cost > 0: violation (robot goes below lowerbound)
    # cost = 0: robot is exactly at the edge
    # cost < 0: safe (robot is within bound)
    cost = (bound + robot_radius + buffer) - state_value

    return cost


class Cost(ABC):
    
    def __init__(self):
        super().__init__()
        
    def get_cost(self, state, ctrl, time_indices):
        raise NotImplementedError
    
    
class Dubins3d_Cost(Cost):
    def __init__(self, cfg, task):
        super().__init__()
        # cost related parameters
        # robot geometry
        self.robot_radius = 0.35  # m
        # navigation task cost parameters
        self.w_control: float = cfg.w_control
        self.w_goal: float = cfg.w_goal
        self.w_velocity: float = cfg.w_velocity
        # soft constraint parameters
        self.q1_obs: float = cfg.q1_obs
        self.q2_obs: float = cfg.q2_obs
        self.q1_bounds: float = cfg.q1_bounds
        self.q2_bounds: float = cfg.q2_bounds
        self.barrier_clip_min: float = cfg.barrier_clip_min
        self.barrier_clip_max: float = cfg.barrier_clip_max
        self.buffer: float = getattr(cfg, "buffer", 0.)
        # task
        self.task = task
        # workspace bounds
        self.x_lowerbound = getattr(cfg, "x_lowerbound", -2 + 0.1/2)
        self.x_upperbound = getattr(cfg, "x_upperbound", 12 - 0.1/2)
        self.y_lowerbound = getattr(cfg, "y_lowerbound", -5 + 0.1/2)
        self.y_upperbound = getattr(cfg, "y_upperbound",  5 - 0.1/2)
    def get_cost(self, state, ctrl, time_indices):
        '''
        Compute cost for each time_index

        Args:
            state: tensor of shape (num_envs, num_steps, num_states or state_dim), states at each time index for each environment
            ctrl: tensor of shape (num_envs, num_steps, num_controls or ctrl_dim), controls at each time index for each environment
            time_indices: tensor of shape (num_envs, num_steps), time indices for each environment

        Returns:
            cost: tensor of shape (num_envs, num_steps), cost computed individually at each time index for each environment
                  (sum over all cost components)
        '''
        num_envs, num_steps, state_dim = state.shape

        # Initialize total cost to zero for all environments and time steps
        total_cost = torch.zeros(num_envs, num_steps, dtype=state.dtype, device=state.device)

        # ============================
        # 1. GOAL COST (weighted squared distance to goal)
        # ============================
        # Extract robot position (x, y) from state
        robot_pos = state[:, :, :2]  # shape: (num_envs, num_steps, 2)
        goal_pos = torch.tensor(self.task.goal_position, dtype=state.dtype, device=state.device)  # shape: (2,)

        # Compute squared distance to goal
        goal_dist_sq = torch.sum((robot_pos - goal_pos)**2, dim=2)  # shape: (num_envs, num_steps)
        goal_cost = self.w_goal * goal_dist_sq
        total_cost += goal_cost

        # ============================
        # 2. CONTROL EFFORT COST (weighted squared control cost)
        # ============================
        # Compute squared control cost
        control_cost = self.w_control * torch.sum(ctrl**2, dim=2)  # shape: (num_envs, num_steps)
        total_cost += control_cost

        # ============================
        # 3. OBSTACLE COSTS (barrier-wrapped, circular obstacles only)
        # ============================
        # Only account for circular obstacles from self.task.environment.obstacles
        for obstacle in self.task.environment.obstacles:
            # Check if obstacle is a CircularObstacle (skip BoxObstacle)
            if isinstance(obstacle, CircularObstacle):
                # This is a circular obstacle
                obs_center = obstacle.center  # numpy array of shape (2,)
                obs_radius = obstacle.radius  # float

                # Compute circular obstacle cost for all time steps
                # Need to reshape state to (num_envs * num_steps, state_dim) for the function
                state_reshaped = state.reshape(num_envs * num_steps, state_dim)

                # Compute circular obstacle cost using the local function
                obs_cost_raw = compute_circular_obstacle_cost(
                    state=state_reshaped,
                    obs_center=obs_center,
                    obs_radius=obs_radius,
                    robot_radius=self.robot_radius,
                    buffer=self.buffer
                )  # shape: (num_envs * num_steps,)

                # Reshape back to (num_envs, num_steps)
                obs_cost_raw = obs_cost_raw.reshape(num_envs, num_steps)

                # Apply barrier function to obstacle cost
                obs_cost_barrier = apply_barrier(
                    cost=obs_cost_raw,
                    q1=self.q1_obs,
                    q2=self.q2_obs,
                    clip_min=self.barrier_clip_min,
                    clip_max=self.barrier_clip_max
                )  # shape: (num_envs, num_steps)

                total_cost += obs_cost_barrier

        # ============================
        # 4. WORKSPACE BOUNDARY COSTS (barrier-wrapped)
        # ============================
        # Use the workspace bounds from self.__init__

        # Need to reshape state to (num_envs * num_steps, state_dim) for the bound functions
        state_reshaped = state.reshape(num_envs * num_steps, state_dim)

        # X-axis upper bound
        x_upper_cost_raw = compute_upperbound_cost(
            state=state_reshaped,
            bound=self.x_upperbound,
            axis='x',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        )  # shape: (num_envs * num_steps,)
        x_upper_cost_raw = x_upper_cost_raw.reshape(num_envs, num_steps)
        x_upper_cost_barrier = apply_barrier(
            cost=x_upper_cost_raw,
            q1=self.q1_bounds,
            q2=self.q2_bounds,
            clip_min=self.barrier_clip_min,
            clip_max=self.barrier_clip_max
        )
        total_cost += x_upper_cost_barrier

        # X-axis lower bound
        x_lower_cost_raw = compute_lowerbound_cost(
            state=state_reshaped,
            bound=self.x_lowerbound,
            axis='x',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        )  # shape: (num_envs * num_steps,)
        x_lower_cost_raw = x_lower_cost_raw.reshape(num_envs, num_steps)
        x_lower_cost_barrier = apply_barrier(
            cost=x_lower_cost_raw,
            q1=self.q1_bounds,
            q2=self.q2_bounds,
            clip_min=self.barrier_clip_min,
            clip_max=self.barrier_clip_max
        )
        total_cost += x_lower_cost_barrier

        # Y-axis upper bound
        y_upper_cost_raw = compute_upperbound_cost(
            state=state_reshaped,
            bound=self.y_upperbound,
            axis='y',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        )  # shape: (num_envs * num_steps,)
        y_upper_cost_raw = y_upper_cost_raw.reshape(num_envs, num_steps)
        y_upper_cost_barrier = apply_barrier(
            cost=y_upper_cost_raw,
            q1=self.q1_bounds,
            q2=self.q2_bounds,
            clip_min=self.barrier_clip_min,
            clip_max=self.barrier_clip_max
        )
        total_cost += y_upper_cost_barrier

        # Y-axis lower bound
        y_lower_cost_raw = compute_lowerbound_cost(
            state=state_reshaped,
            bound=self.y_lowerbound,
            axis='y',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        )  # shape: (num_envs * num_steps,)
        y_lower_cost_raw = y_lower_cost_raw.reshape(num_envs, num_steps)
        y_lower_cost_barrier = apply_barrier(
            cost=y_lower_cost_raw,
            q1=self.q1_bounds,
            q2=self.q2_bounds,
            clip_min=self.barrier_clip_min,
            clip_max=self.barrier_clip_max
        )
        total_cost += y_lower_cost_barrier

        return total_cost
    

class Dubins3d_Constraint(Cost):
    def __init__(self, cfg, task):
        super().__init__()
        # Robot geometry
        self.robot_radius = 0.35  # m
        # Buffer for constraints
        self.buffer: float = getattr(cfg, "buffer", 0.)
        # Task
        self.task = task
        # Workspace bounds
        self.x_lowerbound = getattr(cfg, "x_lowerbound", -2 + 0.1/2)
        self.x_upperbound = getattr(cfg, "x_upperbound", 12 - 0.1/2)
        self.y_lowerbound = getattr(cfg, "y_lowerbound", -5 + 0.1/2)
        self.y_upperbound = getattr(cfg, "y_upperbound",  5 - 0.1/2)

    def get_cost(self, state, ctrl, time_indices):
        '''
        Compute constraint for each time_index

        Args:
            state: tensor of shape (num_envs, num_steps, num_states or state_dim), states at each time index for each environment
            ctrl: tensor of shape (num_envs, num_steps, num_controls or ctrl_dim), controls at each time index for each environment
            time_indices: tensor of shape (num_envs, num_steps), time indices for each environment

        Returns:
            constraint: tensor of shape (num_envs, num_steps), constraint computed individually at each time index for each environment
                        (max over all constraint components)
        '''
        num_envs, num_steps, state_dim = state.shape

        # Initialize constraint to -inf (all satisfied) for all environments and time steps
        max_constraint = torch.full((num_envs, num_steps), float('-inf'), dtype=state.dtype, device=state.device)

        # ============================
        # 1. OBSTACLE CONSTRAINTS (circular obstacles only, no barrier)
        # ============================
        # Only account for circular obstacles from self.task.environment.obstacles
        for obstacle in self.task.environment.obstacles:
            # Check if obstacle is a CircularObstacle (skip BoxObstacle)
            if isinstance(obstacle, CircularObstacle):
                # This is a circular obstacle
                obs_center = obstacle.center  # numpy array of shape (2,)
                obs_radius = obstacle.radius  # float

                # Compute circular obstacle constraint for all time steps
                # Need to reshape state to (num_envs * num_steps, state_dim) for the function
                state_reshaped = state.reshape(num_envs * num_steps, state_dim)

                # Compute circular obstacle constraint using the local function
                obs_constraint = compute_circular_obstacle_cost(
                    state=state_reshaped,
                    obs_center=obs_center,
                    obs_radius=obs_radius,
                    robot_radius=self.robot_radius,
                    buffer=self.buffer
                )  # shape: (num_envs * num_steps,)

                # Reshape back to (num_envs, num_steps)
                obs_constraint = obs_constraint.reshape(num_envs, num_steps)

                # Take maximum over all obstacle constraints
                max_constraint = torch.maximum(max_constraint, obs_constraint)

        # ============================
        # 2. WORKSPACE BOUNDARY CONSTRAINTS (no barrier)
        # ============================
        # Use the workspace bounds from self.__init__

        # Need to reshape state to (num_envs * num_steps, state_dim) for the bound functions
        state_reshaped = state.reshape(num_envs * num_steps, state_dim)

        # X-axis upper bound constraint
        x_upper_constraint = compute_upperbound_cost(
            state=state_reshaped,
            bound=self.x_upperbound,
            axis='x',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        )  # shape: (num_envs * num_steps,)
        x_upper_constraint = x_upper_constraint.reshape(num_envs, num_steps)
        max_constraint = torch.maximum(max_constraint, x_upper_constraint)

        # X-axis lower bound constraint
        x_lower_constraint = compute_lowerbound_cost(
            state=state_reshaped,
            bound=self.x_lowerbound,
            axis='x',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        )  # shape: (num_envs * num_steps,)
        x_lower_constraint = x_lower_constraint.reshape(num_envs, num_steps)
        max_constraint = torch.maximum(max_constraint, x_lower_constraint)

        # Y-axis upper bound constraint
        y_upper_constraint = compute_upperbound_cost(
            state=state_reshaped,
            bound=self.y_upperbound,
            axis='y',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        )  # shape: (num_envs * num_steps,)
        y_upper_constraint = y_upper_constraint.reshape(num_envs, num_steps)
        max_constraint = torch.maximum(max_constraint, y_upper_constraint)

        # Y-axis lower bound constraint
        y_lower_constraint = compute_lowerbound_cost(
            state=state_reshaped,
            bound=self.y_lowerbound,
            axis='y',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        )  # shape: (num_envs * num_steps,)
        y_lower_constraint = y_lower_constraint.reshape(num_envs, num_steps)
        max_constraint = torch.maximum(max_constraint, y_lower_constraint)

        return max_constraint

    def get_cost_dict(self, state, ctrl, time_indices):
        '''
        Get individual constraint values as dictionary for each time_index

        Args:
            state: tensor of shape (num_envs, num_steps, num_states or state_dim), states at each time index for each environment
            ctrl: tensor of shape (num_envs, num_steps, num_controls or ctrl_dim), controls at each time index for each environment
            time_indices: tensor of shape (num_envs, num_steps), time indices for each environment

        Returns:
            cons_dict: dictionary with individual constraint values at each time index for each environment
                       cons_dict = {'cons_type1': tensor of shape (num_envs, num_steps),
                                    'cons_type2': tensor of shape (num_envs, num_steps),
                                    ...}
        '''
        num_envs, num_steps, state_dim = state.shape

        # Initialize constraint dictionary
        cons_dict = {}

        # ============================
        # 1. OBSTACLE CONSTRAINTS (circular obstacles only)
        # ============================
        # Compute maximum obstacle constraint over all circular obstacles
        obs_cons = torch.full((num_envs, num_steps), float('-inf'), dtype=state.dtype, device=state.device)

        for obstacle in self.task.environment.obstacles:
            if isinstance(obstacle, CircularObstacle):
                obs_center = obstacle.center
                obs_radius = obstacle.radius

                # Reshape state for computation
                state_reshaped = state.reshape(num_envs * num_steps, state_dim)

                # Compute circular obstacle constraint
                obs_constraint = compute_circular_obstacle_cost(
                    state=state_reshaped,
                    obs_center=obs_center,
                    obs_radius=obs_radius,
                    robot_radius=self.robot_radius,
                    buffer=self.buffer
                )  # shape: (num_envs * num_steps,)

                # Reshape back
                obs_constraint = obs_constraint.reshape(num_envs, num_steps)

                # Take maximum over all obstacles
                obs_cons = torch.maximum(obs_cons, obs_constraint)

        cons_dict['obs_cons'] = obs_cons

        # ============================
        # 2. WORKSPACE BOUNDARY CONSTRAINTS
        # ============================
        # Reshape state for computation
        state_reshaped = state.reshape(num_envs * num_steps, state_dim)

        # X-axis upper bound
        x_upper_cons = compute_upperbound_cost(
            state=state_reshaped,
            bound=self.x_upperbound,
            axis='x',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        ).reshape(num_envs, num_steps)
        cons_dict['x_upper_cons'] = x_upper_cons

        # X-axis lower bound
        x_lower_cons = compute_lowerbound_cost(
            state=state_reshaped,
            bound=self.x_lowerbound,
            axis='x',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        ).reshape(num_envs, num_steps)
        cons_dict['x_lower_cons'] = x_lower_cons

        # Y-axis upper bound
        y_upper_cons = compute_upperbound_cost(
            state=state_reshaped,
            bound=self.y_upperbound,
            axis='y',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        ).reshape(num_envs, num_steps)
        cons_dict['y_upper_cons'] = y_upper_cons

        # Y-axis lower bound
        y_lower_cons = compute_lowerbound_cost(
            state=state_reshaped,
            bound=self.y_lowerbound,
            axis='y',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        ).reshape(num_envs, num_steps)
        cons_dict['y_lower_cons'] = y_lower_cons

        return cons_dict
    
    def get_cost_dict(self, state, ctrl, time_indices):
        '''
        Get individual constraint values as dictionary for each time_index

        Args:
            state: tensor of shape (num_envs, num_steps, num_states or state_dim), states at each time index for each environment
            ctrl: tensor of shape (num_envs, num_steps, num_controls or ctrl_dim), controls at each time index for each environment
            time_indices: tensor of shape (num_envs, num_steps), time indices for each environment

        Returns:
            cons_dict: a tuple of length num_envs, 
                       where each element if a dictionary containing individual 
            dictionary with individual constraint values at each time index for each environment
                       cons_dict = {'cons_type1': tensor of shape (num_envs, num_steps),
                                    'cons_type2': tensor of shape (num_envs, num_steps),
                                    ...}
        '''
        num_envs, num_steps, state_dim = state.shape

        # Initialize constraint dictionary
        cons_dict = {}

        # ============================
        # 1. OBSTACLE CONSTRAINTS (circular obstacles only)
        # ============================
        # Compute maximum obstacle constraint over all circular obstacles
        obs_cons = torch.full((num_envs, num_steps), float('-inf'), dtype=state.dtype, device=state.device)

        for obstacle in self.task.environment.obstacles:
            if isinstance(obstacle, CircularObstacle):
                obs_center = obstacle.center
                obs_radius = obstacle.radius

                # Reshape state for computation
                state_reshaped = state.reshape(num_envs * num_steps, state_dim)

                # Compute circular obstacle constraint
                obs_constraint = compute_circular_obstacle_cost(
                    state=state_reshaped,
                    obs_center=obs_center,
                    obs_radius=obs_radius,
                    robot_radius=self.robot_radius,
                    buffer=self.buffer
                )  # shape: (num_envs * num_steps,)

                # Reshape back
                obs_constraint = obs_constraint.reshape(num_envs, num_steps)

                # Take maximum over all obstacles
                obs_cons = torch.maximum(obs_cons, obs_constraint)

        cons_dict['obs_cons'] = obs_cons

        # ============================
        # 2. WORKSPACE BOUNDARY CONSTRAINTS
        # ============================
        # Reshape state for computation
        state_reshaped = state.reshape(num_envs * num_steps, state_dim)

        # X-axis upper bound
        x_upper_cons = compute_upperbound_cost(
            state=state_reshaped,
            bound=self.x_upperbound,
            axis='x',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        ).reshape(num_envs, num_steps)
        cons_dict['x_upper_cons'] = x_upper_cons

        # X-axis lower bound
        x_lower_cons = compute_lowerbound_cost(
            state=state_reshaped,
            bound=self.x_lowerbound,
            axis='x',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        ).reshape(num_envs, num_steps)
        cons_dict['x_lower_cons'] = x_lower_cons

        # Y-axis upper bound
        y_upper_cons = compute_upperbound_cost(
            state=state_reshaped,
            bound=self.y_upperbound,
            axis='y',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        ).reshape(num_envs, num_steps)
        cons_dict['y_upper_cons'] = y_upper_cons

        # Y-axis lower bound
        y_lower_cons = compute_lowerbound_cost(
            state=state_reshaped,
            bound=self.y_lowerbound,
            axis='y',
            robot_radius=self.robot_radius,
            buffer=self.buffer
        ).reshape(num_envs, num_steps)
        cons_dict['y_lower_cons'] = y_lower_cons

        return cons_dict
    