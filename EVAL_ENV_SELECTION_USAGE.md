# Evaluation Environment Selection - Usage Guide

This document explains how to use the `select_eval_envs()` method implemented in `ObstacleAvoidanceNavigation` for systematic evaluation across physical parameter space.

## Quick Reference

### One-Line Usage
```python
eval_env_ids, grid_mapping, metadata = env.select_eval_envs()
```

### Common Configurations
```python
# Standard (Recommended): 84 envs (4 friction × 21 payload)
eval_env_ids, grid_mapping, metadata = env.select_eval_envs(
    strategy='stratified', num_f_points=4, num_m_points=21
)

# Fast: 60 envs (5 friction × 12 payload)
eval_env_ids, grid_mapping, metadata = env.select_eval_envs(
    strategy='stratified', num_f_points=5, num_m_points=12
)

# Extreme conditions only: ~40 envs (|m| >= 0.5)
eval_env_ids, grid_mapping, metadata = env.select_eval_envs(
    strategy='stratified', num_f_points=4, num_m_points=21, exclude_middle_m=True
)
```

### Testing
```bash
# Test with mock data (no Isaac Gym needed)
python3 AdaptiveSafetyFilter_Quadruped/test_eval_env_selection.py
```

---

## Purpose

The `select_eval_envs()` function selects representative environments from your parallel Isaac Gym environments based on their physical parameters (friction `f` and payload `m`). This ensures comprehensive evaluation coverage while avoiding redundant testing.

## Basic Usage

```python
# After creating your environment
env = ObstacleAvoidanceNavigation(...)

# Select 84 representative environments (4 friction × 21 payload values)
eval_env_ids, grid_mapping, metadata = env.select_eval_envs(
    strategy='stratified',  # Use bin centers (recommended)
    num_f_points=4,         # 4 friction values
    num_m_points=21         # 21 payload values
)

print(f"Selected {len(eval_env_ids)} environments for evaluation")
print(f"Friction values: {metadata['f_values']}")
print(f"Payload values: {metadata['m_values']}")
```

## Parameters

### `strategy` (str, default='stratified')
- **'stratified'**: Use bin centers for systematic grid sampling **(RECOMMENDED)**
  - Example: With f_range=[0.1, 0.5] and num_f_points=4
  - Generates: [0.15, 0.25, 0.35, 0.45] (bin centers)
  - More representative than edge values

- **'grid'**: Use uniform grid points (bin edges)
  - Example: With same settings
  - Generates: [0.1, 0.2, 0.3, 0.4, 0.5] (if num_f_points=5)

- **'bins'**: Random selection within each bin
  - Similar to stratified but could be extended for within-bin randomization

### `num_f_points` (int, default=4)
Number of friction values to sample. With f_range=[0.1, 0.5]:
- 4 points → 0.1 width bins
- 5 points → 0.08 width bins

### `num_m_points` (int, default=21)
Number of payload values to sample. With m_range=[-1.0, 1.0]:
- 21 points → 0.1 kg width bins
- 11 points → 0.2 kg width bins

### `f_range` (List[float], optional)
Friction range [f_min, f_max]. If None, auto-detects from `self.friction_coeffs`.

### `m_range` (List[float], optional)
Payload range [m_min, m_max]. If None, auto-detects from `self.payloads`.

### `exclude_middle_m` (bool, default=False)
If True, only selects environments with |m| >= 0.5 (extreme payloads only).
- False: Tests all payload values (full coverage)
- True: Tests only extreme payloads (harder conditions)

## Return Values

### `selected_env_ids` (torch.Tensor)
- Shape: (num_selected,)
- dtype: torch.long
- Device: Same as environment
- Contains environment IDs to use for evaluation

### `grid_mapping` (Dict)
- Keys: (f_rep, m_rep) tuples
- Values: env_id (int)
- Maps each representative grid point to the selected environment

Example:
```python
# Find which environment was selected for f=0.15, m=-0.95
env_id = grid_mapping[(0.15, -0.95)]
print(f"Environment {env_id} represents (f=0.15, m=-0.95)")
```

### `metadata` (Dict)
Contains evaluation configuration details:
```python
{
    'f_values': [0.15, 0.25, 0.35, 0.45],  # Sampled friction values
    'm_values': [-0.95, -0.85, ..., 0.95],  # Sampled payload values
    'strategy': 'stratified',
    'num_selected': 84,
    'f_range': [0.1, 0.5],
    'm_range': [-1.0, 1.0],
    'exclude_middle_m': False,
    'num_f_points': 4,
    'num_m_points': 21
}
```

## Usage Examples

### Example 1: Standard Evaluation (84 environments)
```python
# Full coverage with 4 friction × 21 payload values
eval_env_ids, grid_mapping, metadata = env.select_eval_envs(
    strategy='stratified',
    num_f_points=4,
    num_m_points=21
)
# Result: 84 environments covering full (f, m) space
```

### Example 2: Extreme Conditions Only (40 environments)
```python
# Only test extreme payloads (|m| >= 0.5)
# Useful for testing robustness to large payload variations
eval_env_ids, grid_mapping, metadata = env.select_eval_envs(
    strategy='stratified',
    num_f_points=4,
    num_m_points=21,
    exclude_middle_m=True  # Only |m| >= 0.5
)
# Result: ~40 environments (4 f × ~10 extreme m values)
```

### Example 3: Coarser Grid (60 environments)
```python
# Fewer samples for faster evaluation
eval_env_ids, grid_mapping, metadata = env.select_eval_envs(
    strategy='stratified',
    num_f_points=5,
    num_m_points=12
)
# Result: 60 environments with coarser coverage
```

### Example 4: Custom Ranges
```python
# Evaluate only a subset of the training range
eval_env_ids, grid_mapping, metadata = env.select_eval_envs(
    strategy='stratified',
    num_f_points=3,
    num_m_points=11,
    f_range=[0.2, 0.4],  # Only middle friction values
    m_range=[-0.5, 0.5]  # Only moderate payloads
)
# Result: 33 environments in restricted parameter space
```

## Integration with Training/Evaluation

### In Training Script (naive_rl.py or similar)

```python
# During initialization or first evaluation
if not hasattr(self, 'eval_env_ids'):
    # Select evaluation environments once
    self.eval_env_ids, self.eval_grid_mapping, self.eval_metadata = \
        env.select_eval_envs(
            strategy='stratified',
            num_f_points=4,
            num_m_points=21
        )

    # Save to config for reproducibility
    self.num_eval_envs = len(self.eval_env_ids)
    self.num_traj_per_eval_env = 10  # 10 trajectories per env
```

### In simulate_trajectories()

```python
def simulate_trajectories(self, eval_env_ids=None, num_traj_per_env=10, ...):
    """
    Simulate trajectories using selected evaluation environments.

    Args:
        eval_env_ids: Selected environment IDs from select_eval_envs()
        num_traj_per_env: Number of trajectories per environment
    """
    if eval_env_ids is None:
        # Use all environments
        eval_env_ids = torch.arange(self.num_envs, device=self.device)

    num_eval_envs = len(eval_env_ids)
    total_trajectories = num_eval_envs * num_traj_per_env

    # Run num_traj_per_env trajectories for each selected env
    # ...
```

## Expected Output

When you call `select_eval_envs()`, you'll see output like:

```
======================================================================
Evaluation Environment Selection Summary
======================================================================
Strategy: stratified
Friction range: [0.100, 0.500]
Payload range: [-1.000, 1.000] kg
Friction points: 4 → [0.15, 0.25, 0.35, 0.45]
Payload points: 21 → [-0.95, -0.85, -0.75, -0.65, -0.55]... (showing first 5)
Exclude middle m (|m| < 0.5): False
Total grid points: 4 × 21 = 84
Environments selected: 84
Available environments: 100
======================================================================
```

## Tips and Best Practices

1. **Number of Environments**: Ensure `num_envs >= num_f_points × num_m_points`
   - If you request 84 grid points but only have 50 environments, some won't be selected

2. **Stratified vs Grid**: Use 'stratified' for most cases
   - Bin centers are more representative than edges
   - Example: 0.15 (center) is more typical than 0.1 (edge)

3. **Exclude Middle M**: Use sparingly
   - Excluding middle payloads (|m| < 0.5) reduces coverage
   - Only use if you specifically want to test extreme conditions

4. **Reproducibility**: Save `eval_env_ids` and `metadata` for reproducible evaluation
   ```python
   torch.save({
       'eval_env_ids': eval_env_ids,
       'grid_mapping': grid_mapping,
       'metadata': metadata
   }, 'eval_env_selection.pt')
   ```

5. **Per-Environment Metrics**: Use `grid_mapping` to track which (f, m) combination each result corresponds to
   ```python
   for (f_rep, m_rep), env_id in grid_mapping.items():
       env_results = results[env_id]
       print(f"(f={f_rep:.2f}, m={m_rep:.2f}): safe_rate={safe_rate:.2%}")
   ```

## Common Issues

### Issue: "Warning: No available environment found"
**Cause**: Not enough environments to cover all grid points, or environments cluster in certain regions.

**Solution**:
- Increase `num_envs` in your training config
- Reduce `num_f_points` or `num_m_points`
- Check that your domain randomization actually varies (f, m)

### Issue: Multiple grid points select the same environment
**Cause**: Similar grid points are too close together relative to environment distribution.

**Solution**:
- The algorithm prevents this by checking `if env_id in selected_env_ids`
- If this happens, it means you don't have enough diverse environments

### Issue: Selected environments don't match expected (f, m) values exactly
**Explanation**: This is by design! The function finds the *closest* environment to each grid point.
- With random sampling, you're unlikely to have environments at exact grid values
- The closest-match approach ensures coverage while using available environments

---

## Configuration Comparison Table

| Setup | # Envs | Traj/env | Total traj | Eval steps | Overhead | Eval time | Coverage | Statistics |
|-------|--------|----------|------------|------------|----------|-----------|----------|------------|
| **Recommended** | **84** | **10** | **840** | **252k** | **25%** | **~5 min** | **★★★★★** | **★★★☆☆** |
| Fast | 84 | 5 | 420 | 126k | 12.6% | ~2.5 min | ★★★★☆ | ★★☆☆☆ |
| Thorough | 105 | 10 | 1,050 | 315k | 31.5% | ~6.3 min | ★★★★★ | ★★★☆☆ |
| Extreme only | 40 | 10 | 400 | 120k | 12% | ~2.4 min | ★★★☆☆ | ★★★☆☆ |

**Key Numbers (Recommended Setup):**
- **Environments**: 84 (4f × 21m)
- **Trajectories per env**: 10
- **Total trajectories**: 840
- **Rollout length**: 300 steps (6 sec at 50 Hz)
- **Total eval steps**: 252,000
- **Overhead**: 25% (eval_steps / training_steps_between_evals)
- **Eval time**: ~5 minutes per evaluation cycle
- **Evaluation frequency**: Every 100,000 training steps (50 opt cycles)

---

## Files and Resources

- **Implementation**: [obstacle_avoidance_navigation_env.py:1027-1200](obstacle_avoidance_navigation_env.py)
- **Summary**: [EVAL_ENV_SELECTION_SUMMARY.md](EVAL_ENV_SELECTION_SUMMARY.md)
- **Test Script**: [test_eval_env_selection.py](test_eval_env_selection.py)

## See Also

- `simulate_trajectories()`: Use selected environments for evaluation rollouts
- `reset_multiple()`: Reset selected environments with eval-specific ranges
- Evaluation visualization: Will be implemented in `ISAACS/utils/eval_visualization.py`
