# Training Script Examples for train_rarl_quadruped.py

This document provides example commands for running the main training script.

---

## Script Arguments

The training script [train_rarl_quadruped.py](train_rarl_quadruped.py:591-640) accepts the following arguments:

### Required Arguments

1. **`--wtw_label`** (required)
   - Label for walk-these-ways pretrained run
   - Format: `"subdirectory/run-name/train"`
   - Example: `"gait-conditioned-agility/pretrain-v0/train"`

2. **`--env_id`** (required)
   - Environment ID for loading environment.pickle
   - Available IDs: `"0"` through `"99"` (100 environments total)
   - Example: `"0"`, `"42"`, `"99"`

### Optional Arguments

3. **`--config_file`** or **`-cf`** (optional)
   - Path to ISAACS config YAML file
   - Default: `"config/sac.yaml"`
   - Your config: `"quadruped_sac.yaml"`

4. **`--wtw_runs_root`** (optional)
   - Root directory for walk-these-ways runs
   - Default: `../observation-conditioned-reachability/libraries/walk-these-ways/runs`
   - Only specify if your WTW runs are in a custom location

5. **`--env_dir`** (optional)
   - Directory containing environment_{env_id} folders
   - Default: `../observation-conditioned-reachability/data/environments/simulation`
   - Only specify if your environment data is in a custom location

---

python train_rarl_quadruped.py --wtw_runs_root ../libraries/walk-these-ways/runs --wtw_label gait-conditioned-agility/pretrain-v0/train --env_dir ../libraries/observation_conditioned_reachability/observation_conditioned_reachability/data/environments/simulation --env_id 6 --config_file quadruped_sac.yaml

## Example Commands

### Example 1: Basic Training (Minimal Arguments)

Train using environment 0 with default paths:

```bash
cd /home/cassie/Quadruped/AdaptiveSafetyFilter_Quadruped

python train_rarl_quadruped.py \
    --wtw_label "gait-conditioned-agility/pretrain-v0/train" \
    --config_file "quadruped_sac.yaml" \
    --env_id "0"
```

**What this does:**
- Uses Walk-These-Ways policy from: `../observation-conditioned-reachability/libraries/walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/*/`
- Loads environment from: `../observation-conditioned-reachability/data/environments/simulation/0/environment.pickle`
- Uses config: `quadruped_sac.yaml`
- Saves results to: `experiments/quadruped_sac/v1/` (defined in config)

---

### Example 2: Different Environment

Train on environment 42:

```bash
python train_rarl_quadruped.py \
    --wtw_label "gait-conditioned-agility/pretrain-v0/train" \
    --config_file "quadruped_sac.yaml" \
    --env_id "42"
```

---

### Example 3: Custom WTW Runs Location

If your Walk-These-Ways runs are in a custom location:

```bash
python train_rarl_quadruped.py \
    --wtw_label "gait-conditioned-agility/pretrain-v0/train" \
    --config_file "quadruped_sac.yaml" \
    --env_id "0" \
    --wtw_runs_root "/custom/path/to/walk-these-ways/runs"
```

---

### Example 4: Custom Environment Data Location

If your environment data is in a custom location:

```bash
python train_rarl_quadruped.py \
    --wtw_label "gait-conditioned-agility/pretrain-v0/train" \
    --config_file "quadruped_sac.yaml" \
    --env_id "0" \
    --env_dir "/custom/path/to/environments"
```

**Note:** The script will look for: `/custom/path/to/environments/0/environment.pickle`

---

### Example 5: All Custom Paths

Full control over all paths:

```bash
python train_rarl_quadruped.py \
    --wtw_label "my-custom-training/experiment-1/train" \
    --config_file "quadruped_sac.yaml" \
    --env_id "10" \
    --wtw_runs_root "/path/to/wtw/runs" \
    --env_dir "/path/to/envs"
```

---

## Expected Directory Structure

For the script to find everything with default paths:

```
/home/cassie/Quadruped/
├── AdaptiveSafetyFilter_Quadruped/
│   ├── train_rarl_quadruped.py          ← Your training script
│   ├── quadruped_sac.yaml               ← Your config file
│   ├── obstacle_avoidance_navigation_env.py
│   ├── quadruped_naive_rl.py
│   └── ... (other files)
│
├── observation-conditioned-reachability/
│   ├── libraries/
│   │   └── walk-these-ways/
│   │       └── runs/
│   │           └── gait-conditioned-agility/    ← WTW runs
│   │               └── pretrain-v0/
│   │                   └── train/
│   │                       └── <timestamp>/     ← Actual run folder
│   │                           ├── checkpoints/
│   │                           │   ├── body_latest.jit
│   │                           │   └── adaptation_module_latest.jit
│   │                           └── parameters.pkl
│   └── data/
│       └── environments/
│           └── simulation/
│               ├── 0/
│               │   └── environment.pickle       ← Environment 0
│               ├── 1/
│               │   └── environment.pickle       ← Environment 1
│               ⋮
│               └── 99/
│                   └── environment.pickle       ← Environment 99
│
├── ISAACS/                                      ← ISAACS framework
│   ├── agent/
│   └── simulators/
```

---

## Training Output

The training script will create the following output structure:

```
experiments/quadruped_sac/v1/              ← From config: solver.out_folder
├── config.yaml                             ← Copy of your config
├── log.txt                                 ← Training log
├── train_details                           ← Training metrics (PyTorch format)
├── train                                   ← Training results (pickle format)
├── model/                                  ← Saved model checkpoints
│   ├── agent/
│   │   ├── actor/
│   │   │   └── ctrl_<step>.pth
│   │   └── critic/
│   │       └── central_<step>.pth
└── figure/                                 ← Visualization plots
    ├── 0.png                               ← Initial visualization
    ├── <step1>.png
    ├── <step2>.png
    └── ...
```

---

## Training Progress Monitoring

### Console Output

The script prints detailed progress information:

```
======================================================================
STEP 1: Loading ISAACS Config
======================================================================
Loaded ISAACS config from: quadruped_sac.yaml
Output folder: experiments/quadruped_sac/v1
Device: cuda:0

======================================================================
STEP 2: Loading Walk-These-Ways Config and LL Policy
======================================================================
Loading from: ../observation-conditioned-reachability/libraries/walk-these-ways/runs/...
Loaded low-level policy
Loaded Walk-These-Ways config

... (more steps) ...

======================================================================
STEP 8: Starting Training
======================================================================
Updates at sample step 50000
Checks at sample step 50000:
  => Success rate: 0.45
  => Safety violations: 120

... (training continues) ...
```

### Weights & Biases (WandB)

If `solver.use_wandb: true` in your config:

1. Training metrics are logged to WandB
2. Project: `quadruped_sac` (from config)
3. Run name: `v1` (from config)
4. Metrics tracked:
   - loss/critic
   - loss/policy
   - loss/entropy
   - loss/alpha
   - metrics/success_rate (or safe_rate)
   - metrics/cnt_safety_violation
   - metrics/cnt_num_episode
   - hyper_parameters/alpha
   - hyper_parameters/gamma

To view training progress:
```bash
# In your browser, go to:
# https://wandb.ai/CassieC/quadruped_sac
```

---

## Common Issues and Solutions

### Issue 1: Walk-These-Ways Policy Not Found

**Error:**
```
ValueError: No runs found at /path/to/wtw/runs/gait-conditioned-agility/pretrain-v0/train/*
```

**Solution:**
- Check that your WTW policy exists at the expected location
- Verify the `--wtw_label` matches your actual directory structure
- Use `--wtw_runs_root` if your WTW runs are in a custom location

---

### Issue 2: Environment File Not Found

**Error:**
```
FileNotFoundError: Environment file not found: /path/to/environments/0/environment.pickle
```

**Solution:**
- Verify the environment ID exists (0-99)
- Check that `environment.pickle` exists in the folder
- Use `--env_dir` if your environment data is in a custom location

---

### Issue 3: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce `solver.num_envs` in `quadruped_sac.yaml` (try 500 or 250)
- Reduce `solver.batch_size` (try 64)
- Use a GPU with more memory

---

### Issue 4: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'omegaconf'
```

**Solution:**
- Install missing packages (see [DEPENDENCY_REPORT.md](DEPENDENCY_REPORT.md))
```bash
pip install omegaconf shapely jaxlib
```

---

## Training Time Estimates

Based on config settings in `quadruped_sac.yaml`:

- **Max training steps:** 4,000,000
- **Parallel environments:** 1,000
- **Update frequency:** Every 2,000 environment steps
- **Total updates:** ~2,000 gradient updates

**Estimated time (rough):**
- With GPU (CUDA): 4-8 hours
- Without GPU: Not recommended (too slow)

**Checkpoints saved:**
- Every `check_opt_freq * opt_freq` steps
- `check_opt_freq = 25`, `opt_freq = 2000`
- Checkpoint every 50,000 steps
- Total checkpoints: ~80 during training
- Top 10 models saved (based on success/safe rate)

---

## Quick Start Checklist

Before running training:

- [ ] Install missing Python packages (see DEPENDENCY_REPORT.md)
- [ ] Verify Walk-These-Ways policy exists and is accessible
- [ ] Verify environment data exists (check one: `ls ../observation-conditioned-reachability/data/environments/simulation/0/`)
- [ ] Check GPU availability: `nvidia-smi`
- [ ] Review config file: `quadruped_sac.yaml`
- [ ] (Optional) Set up WandB: `wandb login`
- [ ] Run dependency checker: `python check_dependencies.py`

---

## Running Multiple Training Runs

To train on multiple environments in parallel:

```bash
# Terminal 1: Train on environment 0
CUDA_VISIBLE_DEVICES=0 python train_rarl_quadruped.py \
    --wtw_label "gait-conditioned-agility/pretrain-v0/train" \
    --config_file "quadruped_sac.yaml" \
    --env_id "0"

# Terminal 2: Train on environment 1 (different GPU)
CUDA_VISIBLE_DEVICES=1 python train_rarl_quadruped.py \
    --wtw_label "gait-conditioned-agility/pretrain-v0/train" \
    --config_file "quadruped_sac.yaml" \
    --env_id "1"

# ... and so on
```

**Note:** Make sure to use different output folders in the config or they will overwrite each other!

---

## Advanced: Creating Custom Configs

To create a new config for a different experiment:

```bash
# Copy the base config
cp quadruped_sac.yaml quadruped_sac_v2.yaml

# Edit the new config
nano quadruped_sac_v2.yaml

# Update these fields:
# - solver.name: "v2"
# - solver.out_folder: "experiments/quadruped_sac/v2"

# Run with new config
python train_rarl_quadruped.py \
    --wtw_label "gait-conditioned-agility/pretrain-v0/train" \
    --config_file "quadruped_sac_v2.yaml" \
    --env_id "0"
```

---

## Questions?

If you encounter any issues:

1. Check [DEPENDENCY_REPORT.md](DEPENDENCY_REPORT.md) for dependency issues
2. Review the console output for specific error messages
3. Verify all paths are correct
4. Check that your GPU has sufficient memory
