"""
Simple usage example for DTW calculator.

This script demonstrates how to calculate DTW distances between input waves
and representative waves stored in the working directory.
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from process_representative_waves import calculate_dtw_distances


def simple_dtw_test():
    """Simple test of DTW functionality."""

    print("=== DTW Calculator Test ===")

    # Create realistic test waves
    length = 1000
    t = np.linspace(0, 10, length)

    # Simulate accelerometer data
    np.random.seed(42)
    test_waves = {
        "acc_x_gravity_free_wave": 0.5 * np.sin(2 * np.pi * 1.5 * t)
        + 0.2 * np.random.randn(length),
        "acc_y_gravity_free_wave": 0.3 * np.cos(2 * np.pi * 1.2 * t)
        + 0.15 * np.random.randn(length),
        "acc_z_gravity_free_wave": 0.4 * np.sin(2 * np.pi * 0.8 * t)
        + 0.25 * np.random.randn(length),
        "acc_mag_gravity_free_wave": np.ones(length) * 0.7
        + 0.1 * np.random.randn(length),
    }

    print(f"Input waves created with {length} points each")

    # Calculate DTW for specific behaviors (faster than all behaviors)
    target_behaviors = ["Neck___pinch_skin", "Forehead___scratch"]

    print(f"Calculating DTW for behaviors: {target_behaviors}")

    dtw_results = calculate_dtw_distances(
        test_waves,
        target_behaviors=target_behaviors,
        method="sliding_window",
        window_size=30,
    )

    # Display results
    print("\\nResults:")
    for behavior_name, wave_results in dtw_results.items():
        print(f"\\n{behavior_name}:")
        behavior_avg = 0
        for wave_type, distances in wave_results.items():
            avg_dist = distances.mean()
            behavior_avg += avg_dist
            print(f"  {wave_type}: avg={avg_dist:.4f}, std={distances.std():.4f}")

        behavior_avg /= len(wave_results)
        print(f"  Overall average: {behavior_avg:.4f}")

    print("\\n=== Test completed successfully! ===")
    return dtw_results


if __name__ == "__main__":
    simple_dtw_test()
