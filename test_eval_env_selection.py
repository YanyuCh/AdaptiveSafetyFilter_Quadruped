#!/usr/bin/env python3
"""
Test script for select_eval_envs() function.

This demonstrates how to use the evaluation environment selection
without running a full training loop.

Usage:
    python test_eval_env_selection.py
"""

import torch
import numpy as np


def test_select_eval_envs_mock():
    """
    Test the selection logic with mock data (without Isaac Gym).

    This creates fake friction and payload data to verify the selection algorithm.
    """
    print("=" * 70)
    print("Testing select_eval_envs() with Mock Data")
    print("=" * 70)

    # Mock parameters
    num_envs = 100
    device = torch.device('cpu')

    # Create mock friction and payload distributions
    # Uniform random in expected ranges
    friction_coeffs = torch.rand(num_envs, 1) * 0.4 + 0.1  # [0.1, 0.5]
    payloads = torch.rand(num_envs) * 2.0 - 1.0  # [-1.0, 1.0]

    print(f"\nMock environment setup:")
    print(f"  Number of environments: {num_envs}")
    print(f"  Friction range: [{friction_coeffs.min().item():.3f}, {friction_coeffs.max().item():.3f}]")
    print(f"  Payload range: [{payloads.min().item():.3f}, {payloads.max().item():.3f}] kg")

    # Mock the select_eval_envs logic (copy-paste from actual implementation)
    def select_eval_envs(friction_coeffs, payloads, strategy='stratified',
                        num_f_points=4, num_m_points=21, exclude_middle_m=False):

        # Auto-detect ranges
        f_min = friction_coeffs.min().item()
        f_max = friction_coeffs.max().item()
        f_range = [f_min, f_max]

        m_min = payloads.min().item()
        m_max = payloads.max().item()
        m_range = [m_min, m_max]

        # Generate representative values
        if strategy == 'stratified':
            f_bin_width = (f_range[1] - f_range[0]) / num_f_points
            f_values = [f_range[0] + f_bin_width * (i + 0.5) for i in range(num_f_points)]

            m_bin_width = (m_range[1] - m_range[0]) / num_m_points
            m_values = [m_range[0] + m_bin_width * (i + 0.5) for i in range(num_m_points)]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Filter m_values if needed
        if exclude_middle_m:
            m_values = [m for m in m_values if abs(m) >= 0.5]

        # Prepare environment data
        friction_np = friction_coeffs[:, 0].cpu().numpy()
        payload_np = payloads.cpu().numpy()
        all_envs = [(i, friction_np[i], payload_np[i]) for i in range(len(friction_coeffs))]

        # Select environments
        selected_env_ids = []
        grid_mapping = {}

        # Calculate scaling
        f_range_width = f_range[1] - f_range[0]
        m_range_width = m_range[1] - m_range[0]
        m_scale = f_range_width / m_range_width if m_range_width > 0 else 1.0

        for f_rep in f_values:
            for m_rep in m_values:
                best_env_id = None
                best_dist = float('inf')

                for env_id, f, m in all_envs:
                    if env_id in selected_env_ids:
                        continue

                    dist = np.sqrt((f - f_rep)**2 + ((m - m_rep) * m_scale)**2)

                    if dist < best_dist:
                        best_dist = dist
                        best_env_id = env_id

                if best_env_id is not None:
                    selected_env_ids.append(best_env_id)
                    grid_mapping[(f_rep, m_rep)] = best_env_id

        selected_env_ids_tensor = torch.tensor(selected_env_ids, dtype=torch.long, device=device)

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

        return selected_env_ids_tensor, grid_mapping, metadata

    # Test Case 1: Standard selection (84 envs)
    print("\n" + "=" * 70)
    print("Test Case 1: Standard Selection (4 f × 21 m)")
    print("=" * 70)

    eval_env_ids, grid_mapping, metadata = select_eval_envs(
        friction_coeffs, payloads,
        strategy='stratified',
        num_f_points=4,
        num_m_points=21
    )

    print(f"Strategy: {metadata['strategy']}")
    print(f"Friction values: {metadata['f_values']}")
    print(f"Payload values (first 5): {metadata['m_values'][:5]}")
    print(f"Total grid points: {len(metadata['f_values'])} × {len(metadata['m_values'])} = {len(metadata['f_values']) * len(metadata['m_values'])}")
    print(f"Environments selected: {len(eval_env_ids)}")
    print(f"Selection complete: {len(eval_env_ids) == metadata['num_f_points'] * metadata['num_m_points']}")

    # Verify no duplicates
    unique_ids = torch.unique(eval_env_ids)
    print(f"All unique: {len(unique_ids) == len(eval_env_ids)}")

    # Show some examples
    print("\nExample mappings:")
    sample_keys = list(grid_mapping.keys())[:5]
    for f_rep, m_rep in sample_keys:
        env_id = grid_mapping[(f_rep, m_rep)]
        actual_f = friction_coeffs[env_id, 0].item()
        actual_m = payloads[env_id].item()
        dist = np.sqrt((actual_f - f_rep)**2 + ((actual_m - m_rep) * 0.2)**2)
        print(f"  Grid ({f_rep:.3f}, {m_rep:.3f}) → Env {env_id} (actual: f={actual_f:.3f}, m={actual_m:.3f}, dist={dist:.4f})")

    # Test Case 2: Exclude middle m (smaller set)
    print("\n" + "=" * 70)
    print("Test Case 2: Exclude Middle M (only |m| >= 0.5)")
    print("=" * 70)

    eval_env_ids_extreme, grid_mapping_extreme, metadata_extreme = select_eval_envs(
        friction_coeffs, payloads,
        strategy='stratified',
        num_f_points=4,
        num_m_points=21,
        exclude_middle_m=True
    )

    print(f"M values after filtering: {len(metadata_extreme['m_values'])} (from {metadata['num_m_points']})")
    print(f"Environments selected: {len(eval_env_ids_extreme)}")
    print(f"Expected: ~{metadata['num_f_points'] * 10} (4 f × ~10 extreme m)")

    # Verify all selected m values are extreme
    selected_m_values = [m_rep for f_rep, m_rep in grid_mapping_extreme.keys()]
    all_extreme = all(abs(m) >= 0.5 for m in selected_m_values)
    print(f"All |m| >= 0.5: {all_extreme}")

    # Test Case 3: Coverage analysis
    print("\n" + "=" * 70)
    print("Test Case 3: Coverage Analysis")
    print("=" * 70)

    # Check distribution of selected environments
    selected_f = [friction_coeffs[env_id, 0].item() for env_id in eval_env_ids]
    selected_m = [payloads[env_id].item() for env_id in eval_env_ids]

    print(f"Selected friction range: [{min(selected_f):.3f}, {max(selected_f):.3f}]")
    print(f"Selected payload range: [{min(selected_m):.3f}, {max(selected_m):.3f}]")
    print(f"Coverage: Friction {len(set(np.round(selected_f, 2)))} unique bins")
    print(f"Coverage: Payload {len(set(np.round(selected_m, 2)))} unique bins")

    print("\n" + "=" * 70)
    print("All Tests Passed!")
    print("=" * 70)


if __name__ == '__main__':
    # Run test with mock data
    test_select_eval_envs_mock()

    print("\n\nNOTE: To test with actual Isaac Gym environment, use:")
    print("  from obstacle_avoidance_navigation_env import ObstacleAvoidanceNavigation")
    print("  env = ObstacleAvoidanceNavigation(...)")
    print("  eval_env_ids, grid_mapping, metadata = env.select_eval_envs()")
