#!/usr/bin/env python3
"""Test script to verify utils.simulation_utils imports work correctly."""

import sys
import os

# Add the observation-conditioned-reachability path (same as in dubins3d_cost.py)
# Use insert(0, ...) to ensure this path takes precedence over other 'utils' modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability'))

print("Testing import of utils.simulation_utils.obstacle...")
print(f"Added to sys.path: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'observation-conditioned-reachability'))}")
print()

try:
    from utils.simulation_utils.obstacle import CircularObstacle, BoxObstacle
    print("✅ SUCCESS: Import worked!")
    print(f"   CircularObstacle: {CircularObstacle}")
    print(f"   BoxObstacle: {BoxObstacle}")
except ModuleNotFoundError as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
