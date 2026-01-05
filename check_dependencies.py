#!/usr/bin/env python3
"""
Dependency checker for AdaptiveSafetyFilter_Quadruped training code.
This script verifies all imports and paths are correct.
"""

import sys
import os

# Add paths just like in the actual training script
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ISAACS'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability', 'libraries', 'walk-these-ways'))

def check_import(module_name, from_module=None):
    """Try to import a module and report success/failure."""
    try:
        if from_module:
            exec(f"from {from_module} import {module_name}")
            print(f"✓ Successfully imported {module_name} from {from_module}")
        else:
            __import__(module_name)
            print(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}" + (f" from {from_module}" if from_module else ""))
        print(f"  Error: {e}")
        return False
    except Exception as e:
        print(f"⚠ Warning when importing {module_name}" + (f" from {from_module}" if from_module else ""))
        print(f"  Error: {e}")
        return False

def main():
    print("="*70)
    print("DEPENDENCY CHECK FOR QUADRUPED TRAINING")
    print("="*70)

    all_good = True

    # Core Python packages
    print("\n[1/8] Checking core Python packages...")
    all_good &= check_import("numpy")
    all_good &= check_import("torch")
    all_good &= check_import("matplotlib")
    all_good &= check_import("omegaconf")
    all_good &= check_import("jax")
    all_good &= check_import("wandb")
    all_good &= check_import("scipy")
    all_good &= check_import("shapely")

    # IsaacGym (may not be available without proper installation)
    print("\n[2/8] Checking IsaacGym...")
    isaacgym_available = check_import("isaacgym")
    if not isaacgym_available:
        print("  ⚠ IsaacGym not available - this is required for training!")
    all_good &= isaacgym_available

    # Local imports from AdaptiveSafetyFilter_Quadruped
    print("\n[3/8] Checking local AdaptiveSafetyFilter_Quadruped imports...")
    all_good &= check_import("quadruped_naive_rl", "quadruped_naive_rl")
    all_good &= check_import("ObstacleAvoidanceNavigation", "obstacle_avoidance_navigation_env")
    all_good &= check_import("plot_traj", "quadruped_visualization")
    all_good &= check_import("get_values", "quadruped_visualization")
    all_good &= check_import("Dubins3d_Cost", "dubins3d_cost")
    all_good &= check_import("Dubins3d_Constraint", "dubins3d_cost")

    # ISAACS imports
    print("\n[4/8] Checking ISAACS imports...")
    all_good &= check_import("SAC", "agent.sac")
    all_good &= check_import("PrintLogger", "simulators")
    all_good &= check_import("save_obj", "simulators")

    # observation-conditioned-reachability/utils imports
    print("\n[5/8] Checking observation-conditioned-reachability/utils imports...")
    all_good &= check_import("Dubins3D", "dynamics")
    all_good &= check_import("NavigationTask", "navigation_task")
    all_good &= check_import("CircularObstacle", "simulation_utils.obstacle")
    all_good &= check_import("BoxObstacle", "simulation_utils.obstacle")

    # Walk-These-Ways imports
    print("\n[6/8] Checking Walk-These-Ways imports...")
    all_good &= check_import("Cfg", "go1_gym.envs.base.legged_robot_config")

    # Check file paths
    print("\n[7/8] Checking critical file paths...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    critical_paths = [
        ("../ISAACS", "ISAACS directory"),
        ("../observation-conditioned-reachability", "observation-conditioned-reachability directory"),
        ("../observation-conditioned-reachability/utils", "OCR utils directory"),
        ("../observation-conditioned-reachability/libraries/walk-these-ways", "walk-these-ways directory"),
        ("quadruped_sac.yaml", "Config file"),
    ]

    for rel_path, description in critical_paths:
        abs_path = os.path.join(base_dir, rel_path)
        if os.path.exists(abs_path):
            print(f"✓ Found {description}: {abs_path}")
        else:
            print(f"✗ Missing {description}: {abs_path}")
            all_good = False

    # Check environment data directory
    print("\n[8/8] Checking environment data directory...")
    env_data_dir = os.path.join(
        base_dir, '..', 'observation-conditioned-reachability',
        'data', 'environments', 'simulation'
    )
    if os.path.exists(env_data_dir):
        env_dirs = [d for d in os.listdir(env_data_dir) if os.path.isdir(os.path.join(env_data_dir, d))]
        print(f"✓ Found environment data directory with {len(env_dirs)} environments")
        print(f"  Available environment IDs: {sorted(env_dirs)[:10]}..." if len(env_dirs) > 10 else f"  Available environment IDs: {sorted(env_dirs)}")
    else:
        print(f"✗ Missing environment data directory: {env_data_dir}")
        all_good = False

    # Summary
    print("\n" + "="*70)
    if all_good:
        print("✓ ALL CHECKS PASSED!")
        print("Your environment is ready for training.")
        return 0
    else:
        print("✗ SOME CHECKS FAILED!")
        print("Please fix the issues above before running training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
