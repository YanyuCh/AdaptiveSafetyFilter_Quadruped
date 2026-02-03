# Adaptive Safety Filter for Quadruped Navigation: Comprehensive Documentation

This document provides a detailed explanation of the method used to build and train the reachability value network (critic) and best-effort safe policy (actor) for quadruped obstacle avoidance navigation.

---

## Table of Contents

1. [Environments](#1-environments)
2. [Interaction with Environments](#2-interaction-with-environments)
3. [Network Structure](#3-network-structure)
4. [Training Pipeline](#4-training-pipeline)
5. [Evaluation and Logging](#5-evaluation-and-logging)
6. [Interpreting Training Results](#6-interpreting-training-results)
7. [Suggested Modifications](#7-suggested-modifications)
8. [Test Script Construction](#8-test-script-construction)

---

## 1. Environments

### 1.1 Environment Overview

The training uses a **hierarchical control architecture** with two levels:
- **High-Level (HL)**: Navigation policy that outputs velocity commands (v, w)
- **Low-Level (LL)**: Pre-trained locomotion policy from Walk-These-Ways that tracks velocity commands

The environment is implemented in `obstacle_avoidance_navigation_env.py` as class `ObstacleAvoidanceNavigation`, which extends `LeggedRobot` from the Walk-These-Ways framework.

### 1.2 Total Number of Environments

```yaml
# From quadruped_sac.yaml
solver:
  num_envs: 1000  # 1000 parallel environments
```

All 1000 environments run **simultaneously on GPU** using NVIDIA Isaac Gym's vectorized simulation.

### 1.3 Environment/Simulation Type

- **Physics Engine**: NVIDIA Isaac Gym with PhysX backend (`SIM_PHYSX`)
- **Simulation Type**: GPU-accelerated parallel rigid body simulation
- **Terrain Type**: Flat plane (`mesh_type: 'plane'`)
- **Control Frequency**: 50 Hz (control decimation = 10, sim dt = 0.002s)
- **Episode Length**: 3.0 seconds (150 control steps)

### 1.4 Environment Layout/Arrangement

Each environment occupies a **16m x 16m** space (via `env_spacing: 16.0`) to prevent overlap. The local coordinate frame for each environment is:

```
Workspace Bounds (LOCAL frame):
- X: [-2.0, 12.0] meters
- Y: [-5.0, 5.0] meters

Goal Position: (10.0, 0.0)
Goal Radius: 0.5 meters
Robot Radius: 0.35 meters (used for collision checking)
```

Obstacles are loaded from a pickle file (`EnvironmentData/{env_id}/environment.pickle`) containing `CircularObstacle` objects with:
- `center`: numpy array (2,) - position [x, y]
- `radius`: float - obstacle radius

### 1.5 Actors and Objects

**Robot (GO1 Quadruped)**:
- 12 DOF (3 joints per leg x 4 legs): hip, thigh, calf
- Mass: ~12 kg nominal
- Base dimensions: approximately 0.4m x 0.3m x 0.2m

**Static Objects**:
- Circular obstacles (variable count, loaded from pickle)
- Workspace boundaries (implemented as constraints, not physical objects)

### 1.6 Domain Randomization

Domain randomization is configured via the Walk-These-Ways config (`Cfg`):

```python
# From train_rarl_quadruped.py - DISABLED during training for this project:
Cfg.domain_rand.push_robots = False
Cfg.domain_rand.randomize_friction = True      # ENABLED
Cfg.domain_rand.randomize_gravity = False
Cfg.domain_rand.randomize_restitution = False
Cfg.domain_rand.randomize_motor_offset = False
Cfg.domain_rand.randomize_motor_strength = False
Cfg.domain_rand.randomize_base_mass = True     # ENABLED
Cfg.domain_rand.randomize_Kd_factor = False
Cfg.domain_rand.randomize_Kp_factor = False
Cfg.domain_rand.randomize_joint_friction = False
Cfg.domain_rand.randomize_com_displacement = False

# Lag simulation (ENABLED)
Cfg.domain_rand.lag_timesteps = 6
Cfg.domain_rand.randomize_lag_timesteps = True
```

### 1.7 Key Environment Parameters

**Friction Coefficient**:
```python
Cfg.domain_rand.friction_range = [0.1, 0.5]  # Coefficient of friction
```

**Payload/Added Mass**:
```python
Cfg.domain_rand.added_mass_range = [-1.0, 1.0]  # kg added to base
```

**Control Parameters**:
```yaml
# High-level action limits
action_range:
  - [0.0, 2.0]   # Forward velocity v (m/s)
  - [-2.0, 2.0]  # Angular velocity w (rad/s)
```

**Cost/Constraint Parameters** (from `quadruped_sac.yaml`):
```yaml
cost:
  q1_obs: 1.0           # Obstacle barrier scaling
  q2_obs: 10.0          # Obstacle barrier steepness
  q1_bounds: 1.0        # Boundary barrier scaling
  q2_bounds: 10.0       # Boundary barrier steepness
  w_control: 0.01       # Control effort weight
  w_goal: 0.4           # Goal distance weight
  w_velocity: 0.1       # Velocity tracking weight
  barrier_clip_min: -0.25
  barrier_clip_max: 5.0
  buffer: 0.0           # Safety buffer around obstacles

environment:
  g_x_fail: 0.2         # Constraint value assigned on failure
  end_criterion: failure # Episode terminates on constraint violation
```

---

## 2. Interaction with Environments

### 2.1 Environment Initialization

Initialization occurs in `train_rarl_quadruped.py`:

```python
# 1. Load Walk-These-Ways config and pretrained LL policy
cfg_wtw, ll_policy, logdir = load_wtw_config_and_policy(wtw_label, wtw_runs_root)

# 2. Load navigation task from pickle (obstacles, goal)
with open(env_pickle_path, 'rb') as f:
    env_for_task = pkl.load(f)

task = NavigationTask(
    robot_radius=0.35,
    goal_position=np.array([10.0, 0.0]),
    goal_radius=0.5,
    environment=env_for_task,
    dynamics=Dubins3D()
)

# 3. Create environment
env = ObstacleAvoidanceNavigation(
    sim_device=cfg_isaacs.train.device,  # "cuda:0"
    headless=True,
    num_envs=None,  # Uses cfg_wtw.env.num_envs = 1000
    cfg=cfg_wtw,
    task=task,
    ll_policy=ll_policy,
    cfg_cost=cfg_isaacs.cost,
    cfg_arch=cfg_isaacs.arch,
    cfg_env=cfg_isaacs.environment
)
```

### 2.2 Episode Termination Criteria

An episode terminates when ANY of the following occurs:

1. **Failure (Constraint Violation)**: `g_x > failure_thr` (default `failure_thr = 0`)
   - Collision with circular obstacle
   - Exiting workspace boundaries

2. **Timeout**: `episode_length_buf >= max_episode_length` (150 steps = 3 seconds)

```python
# From get_single_env_done_info():
if end_criterion == 'failure':
    failure = torch.any(constraint_values[:, -1] > self.failure_thr).item()
    if failure:
        done = True
        done_type = "failure"
        g_x = self.g_x_fail  # = 0.2
```

### 2.3 Environment Reset

Reset is handled by `_reset_envs()` in `obstacle_avoidance_navigation_env.py`:

**Reset Modes**:
- `'train'`: Wide coverage sampling with rejection sampling
- `'eval'`: Tighter ranges for evaluation
- `'custom'`: Use provided states directly

**Parameter Reset Ranges (Train Mode)**:

| Parameter | Range | Description |
|-----------|-------|-------------|
| Local X | [-2.0, 12.0] m | Full workspace |
| Local Y | [-5.0, 5.0] m | Full workspace |
| Yaw | [-π, π] rad | Full rotation |
| Forward velocity | [-0.5, 2.5] m/s | Includes backward drift |
| Lateral velocity | [-0.25, 0.25] m/s | Small lateral movement |
| Vertical velocity | [-0.25, 0.25] m/s | Small oscillation |
| Yaw rate | [-2.5, 2.5] rad/s | Beyond control limits |
| Roll/Pitch rate | [-0.25, 0.25] rad/s | Small perturbations |
| Joint positions | 0.5x to 1.5x default | Random scaling |
| Joint velocities | 0.0 | Zero initial |

**Rejection Sampling**: Initial (x, y) positions are sampled with rejection to ensure no constraint violations at t=0:

```python
while envs_to_sample.any():
    # Sample candidate positions
    candidate_x = torch.rand(...) * (x_range[1] - x_range[0]) + x_range[0]
    candidate_y = torch.rand(...) * (y_range[1] - y_range[0]) + y_range[0]

    # Check constraints
    constraints = self.get_hl_constraints(candidate_states, dummy_actions, candidate_states)
    constraint_violations = (value[:, 0] > 0.0)  # Any constraint > 0 is violation

    # Accept valid states
    valid_states = ~constraint_violations
    envs_to_sample[valid_indices] = False
```

### 2.4 Step Function Interface

The `step()` function takes high-level actions and returns:

```python
hl_obs, hl_reward, hl_done, hl_info = env.step(hl_actions)
```

**Input**:
- `hl_actions`: Tensor of shape `(num_envs, 2)` containing `[v, w]` commands

**Processing Flow**:
1. Save current state (LOCAL frame)
2. Convert HL actions to LL commands (trotting gait, default body height, etc.)
3. Compute LL observations and clip
4. Execute LL policy to get joint position targets
5. Step physics for `decimation` iterations (10 steps)
6. Refresh state tensors
7. Compute ISAACS cost, constraints, done flags
8. Compute HL observations

**Output**:
- `hl_obs`: Tensor `(num_envs, 33)` - normalized high-level observations
- `hl_reward`: Tensor `(num_envs, 1)` - negative cost (ISAACS-style)
- `hl_done`: Tensor `(num_envs,)` - boolean termination flags
- `hl_info`: Tuple of dicts with `g_x`, `l_x`, `binary_cost`, `append`, etc.

### 2.5 High-Level Observation Format (33D)

```
Index  | Dimension | Description                    | Normalization
-------|-----------|--------------------------------|---------------
0      | 1         | x position (LOCAL)             | [-1, 1] mapped from [-2, 12]
1      | 1         | y position (LOCAL)             | [-1, 1] mapped from [-5, 5]
2      | 1         | heading angle                  | [-π, π] (raw)
3-5    | 3         | base linear velocity (BODY)    | scaled by obs_scales.lin_vel
6-8    | 3         | base angular velocity (BODY)   | scaled by obs_scales.ang_vel
9-20   | 12        | joint position deviations      | scaled by obs_scales.dof_pos
21-32  | 12        | joint velocities               | scaled by obs_scales.dof_vel
```

---

## 3. Network Structure

### 3.1 Actor Network (Best-Effort Safe Policy)

**Class**: `GaussianPolicy` (from `libraries/ISAACS/agent/model.py`)

**Architecture**:
```
Input Layer:
  - obs_dim (33) + append_dim (2) = 35 neurons

Hidden Layers:
  - Layer 1: 35 -> 256 (ReLU)
  - Layer 2: 256 -> 256 (ReLU)
  - Layer 3: 256 -> 256 (ReLU)

Output Layer (Mean):
  - 256 -> 2 (Identity activation, then tanh squashing)

Output Layer (Log Std):
  - 256 -> 2 (Identity activation, clamped to [-10, 1])
```

**Input Format**:
- `obs`: Tensor of shape `(batch_size, 33)` - normalized HL observations
- `append`: Tensor of shape `(batch_size, 2)` - physical parameters `[friction, payload]`

**Output Format**:
- `action`: Tensor of shape `(batch_size, 2)` - `[v, w]` scaled to action ranges
- `log_prob`: Tensor of shape `(batch_size, 1)` - log probability for entropy

**Action Scaling**:
```python
# From GaussianPolicy.sample():
y = torch.tanh(x)  # Squash to [-1, 1]
action = y * self.scale + self.bias
# scale = (a_max - a_min) / 2 = [1.0, 2.0]
# bias = (a_max + a_min) / 2 = [1.0, 0.0]
# Final action: v in [0, 2], w in [-2, 2]
```

### 3.2 Critic Network (Reachability Value Network)

**Class**: `TwinnedQNetwork` (from `libraries/ISAACS/agent/model.py`)

**Architecture** (Twin Q-networks Q1 and Q2):
```
Input Layer:
  - obs_dim (33) + action_dim (2) + append_dim (2) = 37 neurons

Hidden Layers:
  - Layer 1: 37 -> 128 (ReLU)
  - Layer 2: 128 -> 128 (ReLU)
  - Layer 3: 128 -> 128 (ReLU)

Output Layer:
  - 128 -> 1 (scalar Q-value)
```

**Input Format**:
- `state`: Tensor of shape `(batch_size, 33)` - normalized HL observations
- `action`: Tensor of shape `(batch_size, 2)` - actions `[v, w]`
- `append`: Tensor of shape `(batch_size, 2)` - physical parameters `[friction, payload]`

**Output Format**:
- `Q1, Q2`: Tensors of shape `(batch_size, 1)` - twin Q-value estimates

**Value Interpretation**:
- The critic encodes **cost-to-go** (not reward-to-go)
- Lower values indicate safer states
- Positive values indicate states likely to violate constraints
- The threshold `g_x_fail = 0.2` defines the safety boundary

### 3.3 Network Configuration Summary

```yaml
arch:
  actor_0:
    mlp_dim: [256, 256, 256]
    activation: ReLU
    append_dim: 2      # [friction, payload]
    latent_dim: 0
    obs_dim: 33
    action_dim: 2
    action_range: [[0., 2.0], [-2.0, 2.0]]

  critic_0:
    mlp_dim: [128, 128, 128]
    activation: ReLU
    append_dim: 2
    latent_dim: 0
    obs_dim: 33
    action_dim: 2
```

---

## 4. Training Pipeline

### 4.1 Algorithm Overview

The training uses **Soft Actor-Critic (SAC)** adapted for safety-constrained reinforcement learning from the ISAACS framework. Key differences from standard SAC:

1. **Cost Minimization** (not reward maximization): Actor minimizes Q-value
2. **Safety Mode**: Uses `max(Q1, Q2)` instead of `min(Q1, Q2)` for pessimistic safety
3. **Exponential Barrier Costs**: Constraints are encoded as soft penalties
4. **Physical Parameter Conditioning**: Networks are conditioned on `[friction, payload]`

### 4.2 Training Hyperparameters

```yaml
solver:
  max_steps: 4,000,000      # Total environment interactions
  memory_capacity: 50,000   # Replay buffer size
  min_steps_b4_opt: 50,000  # Warmup steps (random actions)
  opt_freq: 2,000           # Steps between optimization phases
  update_per_opt: 200       # Gradient updates per optimization phase
  batch_size: 128           # Batch size for updates

train:
  critic_0:
    lr: 0.0001              # Critic learning rate
    gamma: 0.999            # Discount factor
    tau: 0.005              # Target network soft update rate
    mode: safety            # Cost minimization mode
    terminal_type: max      # Use max Q for terminal states
    update_target_period: 2 # Update target every 2 gradient steps

  actor_0:
    lr: 0.0001              # Actor learning rate
    lr_al: 0.00001          # Alpha (entropy) learning rate
    alpha: 0.005            # Initial entropy coefficient
    learn_alpha: true       # Enable automatic entropy tuning
    actor_type: min         # Minimize Q-value (cost)
    update_period: 2        # Update actor every 2 gradient steps
```

### 4.3 Training Loop (from `quadruped_naive_rl.py`)

```python
def learn(self, env, ...):
    # Initialize all environments
    obs_all = env.reset_multiple(torch.arange(self.n_envs), mode='train')

    while self.cnt_step <= self.max_steps:  # 4,000,000 steps

        # 1. SELECT ACTION
        if self.cnt_step < self.min_steps_b4_opt:  # < 50,000
            # Warmup: Random uniform actions
            action_all = random_uniform(warmup_action_range)
        else:
            # Policy: Sample from actor with physical parameters
            friction_all = env.friction_coeffs[:, 0]
            payload_all = env.payloads
            append_all = torch.stack([friction_all, payload_all], dim=1)
            action_all, _ = self.policy.actor.net.sample(obs_all, append=append_all)

        # 2. STEP ENVIRONMENT (all 1000 envs simultaneously)
        obs_nxt_all, r_all, done_all, info_all = env.step(action_all)

        # 3. STORE TRANSITIONS
        for env_idx in range(num_envs):
            self.store_transition(
                obs_all[env_idx],
                {'ctrl': action_all[env_idx]},
                r_all[env_idx],
                obs_nxt_all[env_idx],
                done_all[env_idx],
                info_all[env_idx]  # Contains g_x, append, etc.
            )

        # 4. HANDLE TERMINATIONS
        done_indices = torch.where(done_all)[0]
        if len(done_indices) > 0:
            new_obs = env.reset_multiple(done_indices, mode='train')
            obs_nxt_all[done_indices] = new_obs
            # Track violations
            g_x_values = [info_all[i]['g_x'] for i in done_indices]
            self.cnt_safety_violation += sum(g_x > 0 for g_x in g_x_values)

        # 5. OPTIMIZATION (every opt_freq=2000 steps after warmup)
        if (self.cnt_step >= min_steps_b4_opt and
            self.cnt_opt_period >= opt_freq):

            for _ in range(update_per_opt):  # 200 updates
                batch = self.sample_batch(batch_size=128)

                # Update critic (TD learning)
                loss_critic = self.policy.critic.update(batch, ...)

                # Update actor (policy gradient)
                if timer % update_period == 0:
                    loss_actor = self.policy.actor.update(batch, critic)

                # Update target network
                if timer % update_target_period == 0:
                    self.policy.critic.update_target()

        # 6. PERIODIC EVALUATION
        if self.cnt_opt % check_opt_freq == 0:
            self.check(env, visualize_callback=visualize_callback)

        self.cnt_step += self.n_envs  # Increment by 1000
```

### 4.4 Data Collection per Step

For each of the 1000 environments, one transition is stored:

```python
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])

# s:    Current observation tensor (33,)
# a:    Action dict {'ctrl': tensor(2,)}
# r:    Scalar reward (float)
# s_:   Next observation tensor (33,)
# done: Boolean termination flag
# info: Dict containing:
#       - 'g_x': Constraint value (float, positive = violation)
#       - 'l_x': Target margin (float, for reach-avoid)
#       - 'binary_cost': 1.0 if violated, 0.0 otherwise
#       - 'append': [friction, payload] array (2,)
#       - 'append_nxt': Same as append (physical params don't change)
```

### 4.5 Network Update Details

**Critic Update** (from `base_block.py`):

```python
def update(self, batch, action_nxt_dict, entropy_motives_dict):
    # Get current Q-values
    q1, q2 = self.net(batch.state, action, append=append)

    # Get target Q-values for next states
    next_q1, next_q2 = self.target(batch.non_final_state_nxt, action_nxt, ...)

    # Bellman target (safety mode)
    y = get_bellman_update(
        mode='safety',         # Use max for pessimistic safety
        gamma=0.999,
        terminal_type='max',   # max(g_x, gamma * V_next) at terminal
        ...
    )

    # MSE loss for both Q-networks
    loss_q1 = mse_loss(q1, y)
    loss_q2 = mse_loss(q2, y)
    loss_q = loss_q1 + loss_q2

    # Backpropagate
    self.optimizer.zero_grad()
    loss_q.backward()
    self.optimizer.step()
```

**Actor Update** (from `base_block.py`):

```python
def update(self, batch, critic):
    # Sample actions from current policy
    action_sample, log_prob = self.net.sample(state, append=append)

    # Get Q-values for sampled actions
    q_pi_1, q_pi_2 = critic.net(batch.state, action_sample, append=append)

    # Take MAX for pessimistic safety (actor_type='min')
    q_pi = torch.max(q_pi_1, q_pi_2)

    # Policy loss: minimize Q + entropy
    loss_entropy = self.alpha * log_prob.mean()
    loss_q_eval = q_pi.mean()  # Want to minimize cost
    loss_pi = loss_q_eval + loss_entropy

    # Backpropagate
    self.optimizer.zero_grad()
    loss_pi.backward()
    self.optimizer.step()

    # Update entropy coefficient (alpha)
    loss_alpha = (self.alpha * (-log_prob - target_entropy)).mean()
    if self.learn_alpha:
        self.log_alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.log_alpha_optimizer.step()
```

### 4.6 Training Statistics

| Metric | Value |
|--------|-------|
| Total environment steps | 4,000,000 |
| Parallel environments | 1,000 |
| Effective training iterations | 4,000 |
| Warmup steps | 50,000 (50 iterations) |
| Optimization phases | ~1,975 |
| Gradient updates per phase | 200 |
| Total gradient updates | ~395,000 |
| Evaluations | ~79 (every 25 opt phases) |

---

## 5. Evaluation and Logging

### 5.1 Evaluation Environment Selection

During evaluation, a **grid-based stratified sampling** strategy selects representative environments:

```python
# From quadruped_sac.yaml
num_eval_f_points: 5    # Friction grid points
num_eval_m_points: 21   # Payload grid points

# Results in 5 x 21 = 105 representative environments
```

The `select_eval_envs()` function:
1. Creates a grid over (friction, payload) parameter space
2. Finds the closest actual environment to each grid point
3. Ensures no environment is selected twice

### 5.2 Trajectory Simulation

For each selected environment, multiple trajectories are simulated:

```python
num_trajectories_per_env: 10
eval_timeout_s: 3.0  # Same as training episode length

# Total evaluation trajectories: 105 envs x 10 trajs = 1,050
```

### 5.3 Evaluation Metrics

**Safe Rate**:
```python
safe_rate = np.sum(results != -1) / total_num_eval
# Fraction of trajectories that did NOT violate constraints
```

**Success Rate** (for reach-avoid mode):
```python
success_rate = np.sum(results == 1) / total_num_eval
# Fraction of trajectories that reached the goal safely
```

**Episode Length**:
```python
avg_length = np.mean(lengths)
# Average trajectory length in steps
```

### 5.4 Checkpoint Management

The training maintains a **top-k priority queue** of checkpoints:

```yaml
save_top_k: 10
save_metric: safety  # Use safe_rate as metric
```

```python
# From _save():
if metric > self.pq_top_k.queue[0][0]:
    # Remove worst checkpoint
    _, step_remove = self.pq_top_k.get()
    module.remove(step_remove, module_folder)
    # Add new checkpoint
    self.pq_top_k.put((metric, self.cnt_step))
    module.save(self.cnt_step, module_folder)
```

### 5.5 Output Files

Training outputs are saved to `experiments/quadruped_sac/v1/`:

```
experiments/quadruped_sac/v1/
├── config.yaml              # Copy of training configuration
├── log.txt                  # Training log (stdout + stderr)
├── train_details            # Torch tensor with training metrics
├── model/
│   └── agent/
│       ├── ctrl/            # Actor checkpoints
│       │   ├── ctrl-50000.pth
│       │   ├── ctrl-100000.pth
│       │   └── ...
│       └── central/         # Critic checkpoints
│           ├── central-50000.pth
│           ├── central-100000.pth
│           └── ...
└── figure/
    ├── 0.png                # Initial value function visualization
    ├── 50000.png
    ├── 100000.png
    └── ...                  # 5x5 grid (friction x payload) value plots
```

### 5.6 WandB Logging

If enabled (`use_wandb: true`), the following metrics are logged:

**Per Optimization Phase**:
- `loss/critic`: Critic TD loss
- `loss/policy`: Actor policy gradient loss
- `loss/entropy`: Entropy loss component
- `loss/alpha`: Entropy coefficient
- `metrics/cnt_safety_violation`: Cumulative violations
- `metrics/cnt_num_episode`: Cumulative episodes
- `hyper_parameters/alpha`: Current entropy coefficient
- `hyper_parameters/gamma`: Current discount factor

**Per Evaluation**:
- `metrics/safe_rate`: Fraction of safe trajectories
- `metrics/ep_length`: Average episode length
- `metrics/num_eval_envs`: Number of evaluation environments
- `metrics/total_trajectories`: Total trajectories evaluated

### 5.7 Visualization

The `visualize()` function creates a **5x5 grid of value function heatmaps**:

- Rows: 5 payload values [-1.0, -0.5, 0.0, 0.5, 1.0] kg
- Columns: 5 friction values [0.1, 0.2, 0.3, 0.4, 0.5]
- Each subplot shows V(x, y) for fixed heading=0, v=1 m/s, w=0
- Color scale: Blue (safe, V < 0) to Red (unsafe, V > 0)
- Obstacle outlines shown in black

---

## 6. Interpreting Training Results

### 6.1 Loss Curves

**Critic Loss (`loss/critic`)**:
- Should decrease and stabilize over training
- High initial values are normal (random policy generates diverse states)
- Spikes may indicate distribution shift after policy updates
- Target: Stable low values (< 1.0 typical)

**Actor Loss (`loss/policy`)**:
- Represents the expected cost under current policy
- Should decrease as policy learns to avoid unsafe states
- May fluctuate as exploration continues

**Entropy Loss (`loss/entropy`)**:
- Should decrease initially (policy becomes more deterministic)
- Stabilizes as alpha adapts

**Alpha (`hyper_parameters/alpha`)**:
- Starts at 0.005
- Should decrease as policy improves
- Lower alpha = more deterministic policy

### 6.2 Safety Metrics

**Safe Rate**:
- Primary metric for safety filter training
- Target: > 0.95 for robust safety
- < 0.8 indicates significant constraint violations

**Safety Violations (`cnt_safety_violation`)**:
- Cumulative count of constraint violations during training
- Should grow slower as training progresses
- Ratio: violations/episodes indicates failure rate

### 6.3 Value Function Visualization

**Interpreting Heatmaps**:
- **Blue regions** (V < 0): Safe states with margin to constraints
- **White regions** (V ≈ 0): Near safety boundary
- **Red regions** (V > 0): Unsafe states (will likely violate)
- **Black outlines**: Obstacle boundaries

**Expected Patterns**:
- Blue "corridors" between obstacles indicating safe paths
- Red regions inside/near obstacles
- Red regions near workspace boundaries
- Higher friction → larger safe regions
- Heavier payload → smaller safe regions (slower response)

### 6.4 Checkpoint Quality

The top-k checkpoints are ranked by safe rate. To select the best model:

```python
# Load from pq_top_k
best_metric, best_step = max(pq_top_k.queue)
print(f"Best checkpoint: step {best_step} with safe_rate {best_metric}")
```

---

## 7. Suggested Modifications

### 7.1 For Fixed Environment with Varying Physical Parameters

Since your goal is to train for a **fixed obstacle layout** but **varying friction and payload**, consider:

**A. Environment Data**:
- Use a single `environment.pickle` with your fixed obstacle configuration
- All 1000 parallel environments will have the same obstacles but different (friction, payload)

**B. Physical Parameter Distribution**:
The current configuration already supports this:
```python
Cfg.domain_rand.friction_range = [0.1, 0.5]
Cfg.domain_rand.added_mass_range = [-1.0, 1.0]
```

**C. Potential Config Modifications**:

```yaml
# Increase conditioning emphasis
arch:
  actor_0:
    append_dim: 2  # Keep [friction, payload] conditioning
  critic_0:
    append_dim: 2

# Potentially increase network capacity for better generalization
arch:
  actor_0:
    mlp_dim: [512, 512, 256]  # Larger for better generalization
  critic_0:
    mlp_dim: [256, 256, 128]

# Adjust evaluation grid for your specific ranges
solver:
  num_eval_f_points: 5   # Or increase for finer coverage
  num_eval_m_points: 11  # Adjust based on payload range importance
```

### 7.2 Network Architecture Suggestions

If the current architecture underfits:
```yaml
arch:
  actor_0:
    mlp_dim: [512, 512, 256]
    activation: ReLU  # Or try ELU for smoother gradients
  critic_0:
    mlp_dim: [256, 256, 256]
```

### 7.3 Training Hyperparameter Suggestions

For more robust learning:
```yaml
solver:
  memory_capacity: 100000   # Larger buffer for more diverse experience
  min_steps_b4_opt: 100000  # Longer warmup for better initial coverage
  update_per_opt: 400       # More updates per phase

train:
  critic_0:
    gamma: 0.995            # Slightly lower for faster propagation
    tau: 0.01               # Faster target updates
  actor_0:
    alpha: 0.01             # Higher initial entropy for exploration
```

### 7.4 Cost Function Modifications

If constraints are too soft:
```yaml
cost:
  q1_obs: 2.0      # Higher penalty magnitude
  q2_obs: 15.0     # Steeper barrier near obstacles
  q1_bounds: 2.0
  q2_bounds: 15.0
  buffer: 0.05     # Add 5cm safety buffer
```

---

## 8. Test Script Construction

Below is a template test script based on ISAACS's `test_safety_filter.py` adapted for the quadruped:

```python
#!/usr/bin/env python3
"""
Test script for learned safety filter on quadruped navigation.

Usage:
    python test_safety_filter_quadruped.py \
        --config_file config/test_safety_filter.yaml \
        --model_step 3500000 \
        --friction 0.3 \
        --payload 0.0
"""

import os
import sys
import copy
import argparse
import pickle
import time
from types import SimpleNamespace
import numpy as np
from omegaconf import OmegaConf
import torch

# Import local modules (must come before torch for IsaacGym)
from obstacle_avoidance_navigation_env import ObstacleAvoidanceNavigation
from quadruped_visualization import plot_traj, get_values

# Walk-These-Ways imports
from go1_gym.envs.base.legged_robot_config import Cfg

# OCR imports
from libraries.OCR.utils.dynamics import Dubins3D
from libraries.OCR.utils.navigation_task import NavigationTask

# ISAACS imports
from libraries.ISAACS.agent.sac import SAC
from libraries.ISAACS.simulators.policy.nn_policy import NeuralNetworkControlSystem

# For nominal MPC policy (from OCR)
# Uncomment and adapt based on your MPC implementation
# from libraries.OCR.utils.mpc import SamplingBasedMPC


def load_wtw_config_and_policy(label, wtw_runs_root):
    """Load Walk-These-Ways config and pretrained policy."""
    import glob
    import pickle as pkl

    dirs = glob.glob(f"{wtw_runs_root}/{label}/*")
    logdir = sorted(dirs)[0]

    # Load policy
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        return action

    # Load config
    with open(logdir + "/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        cfg_dict = pkl_cfg["Cfg"]
        for key, value in cfg_dict.items():
            if hasattr(Cfg, key) and key != 'command_ranges':
                for key2, value2 in cfg_dict[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # Disable domain randomization
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_roll_pitch = False

    # Set environment parameters
    Cfg.env.num_envs = 1  # Single env for testing
    Cfg.env.episode_length_s = 20.0  # Longer for full navigation
    Cfg.terrain.mesh_type = 'plane'

    return Cfg, policy, logdir


class SafetyFilter:
    """
    Safety filter that monitors value function and overrides nominal policy
    when predicted to violate constraints.
    """
    def __init__(self, base_policy, safety_policy, critic, value_threshold=0.0):
        """
        Args:
            base_policy: Nominal task policy (e.g., MPC)
            safety_policy: Learned safe policy (actor)
            critic: Learned value function (critic)
            value_threshold: Override when V > threshold
        """
        self.base_policy = base_policy
        self.safety_policy = safety_policy
        self.critic = critic
        self.value_threshold = value_threshold

    def get_action(self, obs, append):
        """
        Get action using safety filter.

        Args:
            obs: Observation tensor (33,)
            append: Physical parameters [friction, payload] tensor (2,)

        Returns:
            action: Action tensor (2,)
            info: Dict with 'shielded' flag
        """
        # Get nominal action from base policy
        nominal_action = self.base_policy.get_action(obs, append)

        # Evaluate value of next state under nominal action
        with torch.no_grad():
            # Get Q-value for nominal action
            q1, q2 = self.critic.net(
                obs.unsqueeze(0),
                nominal_action.unsqueeze(0),
                append=append.unsqueeze(0)
            )
            value_nominal = torch.max(q1, q2).item()

        # If value exceeds threshold, use safety policy
        if value_nominal > self.value_threshold:
            safety_action, _ = self.safety_policy.net.sample(
                obs.unsqueeze(0),
                append=append.unsqueeze(0)
            )
            return safety_action.squeeze(0), {'shielded': True, 'value': value_nominal}
        else:
            return nominal_action, {'shielded': False, 'value': value_nominal}


def main(args):
    # ================================================================
    # SETUP
    # ================================================================

    # Load training config
    cfg = OmegaConf.load(args.config_file)
    device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.out_folder, exist_ok=True)

    print("=" * 70)
    print("QUADRUPED SAFETY FILTER TEST")
    print("=" * 70)
    print(f"Friction: {args.friction}")
    print(f"Payload: {args.payload}")
    print(f"Model step: {args.model_step}")
    print(f"Start position: (0, 0)")
    print(f"Goal position: (10, 0)")

    # ================================================================
    # LOAD ENVIRONMENT
    # ================================================================

    print("\nLoading Walk-These-Ways config and policy...")
    cfg_wtw, ll_policy, _ = load_wtw_config_and_policy(
        args.wtw_label, args.wtw_runs_root
    )

    # Override friction and payload for testing
    cfg_wtw.domain_rand.randomize_friction = False
    cfg_wtw.domain_rand.randomize_base_mass = False

    print("\nLoading navigation task...")
    with open(args.env_pickle, 'rb') as f:
        env_data = pickle.load(f)

    task = NavigationTask(
        robot_radius=0.35,
        goal_position=np.array([10.0, 0.0]),
        goal_radius=0.5,
        environment=env_data,
        dynamics=Dubins3D()
    )

    print(f"Number of obstacles: {len(task.environment.obstacles)}")

    # Create environment
    print("\nCreating environment...")
    env = ObstacleAvoidanceNavigation(
        sim_device=args.device,
        headless=not args.render,
        num_envs=1,
        cfg=cfg_wtw,
        task=task,
        ll_policy=ll_policy,
        cfg_cost=cfg.cost,
        cfg_arch=cfg.arch,
        cfg_env=cfg.environment
    )

    # Manually set friction and payload for the test environment
    env.friction_coeffs[0, 0] = args.friction
    env.payloads[0] = args.payload

    # ================================================================
    # LOAD TRAINED NETWORKS
    # ================================================================

    print("\nLoading trained networks...")
    rng = np.random.default_rng(seed=0)

    # Build SAC policy structure
    sac = SAC(cfg.train, cfg.arch, rng)
    sac.build_network(verbose=False)

    # Load trained weights
    model_folder = os.path.join(args.model_folder, "model", "agent")
    sac.actor.restore(args.model_step, model_folder, verbose=True)
    sac.critic.restore(args.model_step, model_folder, verbose=True)

    # Initialize HL policy in environment
    env.init_hl_policy(
        cfg_hl_policy=SimpleNamespace(device=device),
        actor=sac.actor.net
    )

    print(f"Loaded actor from step {args.model_step}")
    print(f"Loaded critic from step {args.model_step}")

    # ================================================================
    # SETUP POLICIES
    # ================================================================

    print("\nSetting up policies...")

    # Safety policy (learned actor)
    safety_policy = NeuralNetworkControlSystem(
        id='ego',
        actor=sac.actor.net,
        cfg=SimpleNamespace(device=device)
    )

    # Nominal policy placeholder
    # Replace with your sampling-based MPC implementation
    class DummyNominalPolicy:
        """Placeholder for MPC - replace with actual implementation."""
        def __init__(self, goal_position):
            self.goal = goal_position

        def get_action(self, obs, append):
            # Simple proportional controller towards goal
            # Replace with your MPC implementation
            x_norm = obs[0].item()
            y_norm = obs[1].item()
            heading = obs[2].item()

            # Unnormalize position
            x = (x_norm + 1) / 2 * 14 - 2  # [-2, 12]
            y = (y_norm + 1) / 2 * 10 - 5  # [-5, 5]

            # Direction to goal
            dx = self.goal[0] - x
            dy = self.goal[1] - y
            angle_to_goal = np.arctan2(dy, dx)

            # Desired angular velocity
            angle_error = angle_to_goal - heading
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

            # Simple P controller
            v = min(1.5, np.sqrt(dx**2 + dy**2) * 0.5)
            w = np.clip(2.0 * angle_error, -2.0, 2.0)

            return torch.tensor([v, w], device=append.device)

    nominal_policy = DummyNominalPolicy(goal_position=np.array([10.0, 0.0]))

    # Safety filter
    safety_filter = SafetyFilter(
        base_policy=nominal_policy,
        safety_policy=safety_policy,
        critic=sac.critic,
        value_threshold=args.value_threshold
    )

    # ================================================================
    # RUN TEST TRAJECTORIES
    # ================================================================

    print("\n" + "=" * 70)
    print("RUNNING TEST TRAJECTORIES")
    print("=" * 70)

    results_list = []
    trajectories = []
    shield_counts = []

    for traj_idx in range(args.num_trajectories):
        print(f"\nTrajectory {traj_idx + 1}/{args.num_trajectories}")

        # Reset to starting position (0, 0)
        # Create custom state: [x=0, y=0, heading=0, ...]
        custom_state = torch.zeros(33, device=device)
        custom_state[0] = 2 * (0 - (-2)) / (12 - (-2)) - 1  # x=0 normalized
        custom_state[1] = 2 * (0 - (-5)) / (5 - (-5)) - 1   # y=0 normalized
        custom_state[2] = 0.0  # heading = 0
        # Rest are zeros (default velocities and joint states)

        obs = env.reset_one(0, mode='custom', custom_state=custom_state)

        trajectory = []
        shield_log = []
        done = False
        step = 0
        max_steps = int(args.timeout_s / 0.02)  # 20s at 50Hz

        # Get physical parameters
        friction = env.friction_coeffs[0, 0]
        payload = env.payloads[0]
        append = torch.tensor([friction, payload], device=device)

        while not done and step < max_steps:
            # Get current position for logging
            local_pos = env.base_pos[0] - env.env_origins[0]
            trajectory.append(local_pos.cpu().numpy().copy())

            # Get action from safety filter
            action, filter_info = safety_filter.get_action(obs, append)
            shield_log.append(filter_info['shielded'])

            if filter_info['shielded']:
                print(f"  Step {step}: SHIELDED (V={filter_info['value']:.3f})")

            # Step environment
            obs_nxt, reward, done_tensor, info = env.step(action.unsqueeze(0))
            obs = obs_nxt[0]
            done = done_tensor[0].item()

            # Check if reached goal
            x = local_pos[0].item()
            y = local_pos[1].item()
            dist_to_goal = np.sqrt((x - 10)**2 + y**2)
            if dist_to_goal < 0.5:
                print(f"  GOAL REACHED at step {step}!")
                break

            step += 1

        # Final position
        local_pos = env.base_pos[0] - env.env_origins[0]
        trajectory.append(local_pos.cpu().numpy().copy())

        # Determine result
        g_x = info[0]['g_x']
        if g_x > 0:
            result = -1  # Failure
            print(f"  FAILED (constraint violation)")
        elif dist_to_goal < 0.5:
            result = 1   # Success
        else:
            result = 0   # Timeout
            print(f"  TIMEOUT")

        results_list.append(result)
        trajectories.append(np.array(trajectory))
        shield_counts.append(np.sum(shield_log))

        print(f"  Final position: ({local_pos[0].item():.2f}, {local_pos[1].item():.2f})")
        print(f"  Shield interventions: {np.sum(shield_log)}/{len(shield_log)}")

    # ================================================================
    # RESULTS SUMMARY
    # ================================================================

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results_arr = np.array(results_list)
    success_rate = np.mean(results_arr == 1)
    failure_rate = np.mean(results_arr == -1)
    timeout_rate = np.mean(results_arr == 0)
    avg_shield_rate = np.mean([sc / len(traj) for sc, traj in zip(shield_counts, trajectories)])

    print(f"Success rate: {success_rate:.1%}")
    print(f"Failure rate: {failure_rate:.1%}")
    print(f"Timeout rate: {timeout_rate:.1%}")
    print(f"Average shield rate: {avg_shield_rate:.1%}")

    # Save results
    results_dict = {
        'friction': args.friction,
        'payload': args.payload,
        'trajectories': trajectories,
        'results': results_arr,
        'shield_counts': shield_counts,
        'success_rate': success_rate,
        'failure_rate': failure_rate
    }

    save_path = os.path.join(args.out_folder, f'results_f{args.friction}_m{args.payload}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f"\nResults saved to: {save_path}")

    # ================================================================
    # VISUALIZATION
    # ================================================================

    if args.visualize:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot obstacles
        for obstacle in task.environment.obstacles:
            if hasattr(obstacle, 'center') and hasattr(obstacle, 'radius'):
                circle = Circle(
                    obstacle.center, obstacle.radius,
                    facecolor='gray', edgecolor='black', alpha=0.5
                )
                ax.add_patch(circle)

        # Plot trajectories
        for traj, result in zip(trajectories, results_arr):
            color = 'g' if result == 1 else ('r' if result == -1 else 'orange')
            ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7)
            ax.scatter(traj[0, 0], traj[0, 1], c='blue', s=50, marker='o')
            ax.scatter(traj[-1, 0], traj[-1, 1], c=color, s=50,
                      marker='*' if result == 1 else ('x' if result == -1 else '^'))

        # Plot goal
        goal_circle = Circle((10, 0), 0.5, facecolor='green', alpha=0.3)
        ax.add_patch(goal_circle)

        ax.set_xlim(-3, 13)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Safety Filter Test (f={args.friction}, m={args.payload})')
        ax.grid(True, alpha=0.3)

        fig_path = os.path.join(args.out_folder, f'trajectories_f{args.friction}_m{args.payload}.png')
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test quadruped safety filter")

    # Required arguments
    parser.add_argument("--config_file", type=str, required=True,
                        help="Path to training config YAML")
    parser.add_argument("--model_folder", type=str, required=True,
                        help="Path to folder containing trained models")
    parser.add_argument("--model_step", type=int, required=True,
                        help="Training step of model to load")

    # Test parameters
    parser.add_argument("--friction", type=float, default=0.3,
                        help="Friction coefficient for testing")
    parser.add_argument("--payload", type=float, default=0.0,
                        help="Payload mass for testing (kg)")
    parser.add_argument("--value_threshold", type=float, default=0.0,
                        help="Value threshold for safety filter")
    parser.add_argument("--num_trajectories", type=int, default=10,
                        help="Number of test trajectories")
    parser.add_argument("--timeout_s", type=float, default=20.0,
                        help="Maximum trajectory duration (seconds)")

    # Paths
    parser.add_argument("--wtw_label", type=str,
                        default="gait-conditioned-agility/pretrain-v0/train",
                        help="Walk-These-Ways model label")
    parser.add_argument("--wtw_runs_root", type=str,
                        default="libraries/OCR/libraries/walk-these-ways/runs",
                        help="Root directory for WTW runs")
    parser.add_argument("--env_pickle", type=str,
                        default="EnvironmentData/6/environment.pickle",
                        help="Path to environment pickle file")
    parser.add_argument("--out_folder", type=str,
                        default="experiments/test_safety_filter",
                        help="Output folder for results")

    # Runtime options
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for computation")
    parser.add_argument("--render", action="store_true",
                        help="Enable rendering")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate trajectory visualization")

    args = parser.parse_args()
    main(args)
```

### 8.1 Test Script Usage

```bash
# Basic usage with default parameters
python test_safety_filter_quadruped.py \
    --config_file quadruped_sac.yaml \
    --model_folder experiments/quadruped_sac/v1 \
    --model_step 3500000 \
    --friction 0.3 \
    --payload 0.0 \
    --visualize

# Test with different physical parameters
python test_safety_filter_quadruped.py \
    --config_file quadruped_sac.yaml \
    --model_folder experiments/quadruped_sac/v1 \
    --model_step 3500000 \
    --friction 0.15 \
    --payload -0.8 \
    --value_threshold 0.05 \
    --num_trajectories 20 \
    --visualize
```

### 8.2 Integrating with Sampling-Based MPC

To replace the dummy nominal policy with your MPC:

```python
# Import your MPC implementation
from libraries.OCR.utils.mpc import SamplingBasedMPC  # Adjust path

# Create MPC instance
nominal_policy = SamplingBasedMPC(
    dynamics=Dubins3D(),
    goal_position=np.array([10.0, 0.0]),
    obstacles=task.environment.obstacles,
    horizon=10,
    num_samples=1000,
    # ... other MPC parameters
)

# The MPC should implement get_action(obs, append) -> torch.Tensor
```

### 8.3 Key Test Script Features

1. **Physical Parameter Control**: Manually set friction and payload
2. **Custom Initial State**: Start at (0, 0) with heading towards goal
3. **Safety Filter Integration**: Monitors value function and overrides when unsafe
4. **Comprehensive Logging**: Track shield interventions and trajectory outcomes
5. **Visualization**: Plot trajectories with obstacle overlay

---

## Summary

This documentation covers the complete training pipeline for an adaptive safety filter on quadruped navigation:

1. **Environment**: 1000 parallel IsaacGym environments with circular obstacles, domain randomization over friction and payload
2. **Interaction**: Hierarchical control with pretrained LL policy, rejection-sampled resets, ISAACS-style rewards
3. **Networks**: Gaussian policy actor (256x3) + Twin Q-network critic (128x3), conditioned on physical parameters
4. **Training**: SAC with cost minimization, 4M steps, 50k warmup, 200 updates every 2k steps
5. **Evaluation**: Grid-stratified sampling over (friction, payload), safe rate as primary metric
6. **Interpretation**: Monitor loss curves, safe rate, and value function visualizations
7. **Modifications**: Suggestions for fixed environment with varying physical parameters
8. **Testing**: Template script for safety filter evaluation with MPC integration

For questions or issues, refer to the source files:
- Training: `train_rarl_quadruped.py`, `quadruped_naive_rl.py`
- Environment: `obstacle_avoidance_navigation_env.py`
- Networks: `libraries/ISAACS/agent/base_block.py`, `libraries/ISAACS/agent/model.py`
- Visualization: `quadruped_visualization.py`
