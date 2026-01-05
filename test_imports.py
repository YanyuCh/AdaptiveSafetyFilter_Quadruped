#!/usr/bin/env python
"""Test that all critical imports work for training."""

import sys
import os

# Add paths like train_rarl_quadruped.py does
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ISAACS'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability'))  # For utils.dynamics
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability', 'libraries', 'walk-these-ways'))

print("Testing critical imports...")

# Test ISAACS imports
from agent.sac import SAC
print("✓ ISAACS SAC imported")

# Test OCR utils imports
from dynamics import Dubins3D
from navigation_task import NavigationTask
print("✓ OCR dynamics and navigation task imported")

# Test walk-these-ways
from go1_gym.envs.base.legged_robot_config import Cfg
print("✓ walk-these-ways config imported")

print("\n" + "="*70)
print("SUCCESS! All critical imports work!")
print("Your environment is ready for training!")
print("="*70)
