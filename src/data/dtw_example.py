"""
Example usage of the RepresentativeWaveDTWCalculator.
"""

import matplotlib.pyplot as plt
import numpy as np
from process_representative_waves import calculate_dtw_distances


def create_sample_input_waves(length: int = 1000) -> dict:
    """Create sample input waves for testing."""
    np.random.seed(42)

    # Create more realistic sensor-like data
    t = np.linspace(0, 10, length)

    input_waves = {
        "acc_x_gravity_free_wave": (
            0.5 * np.sin(2 * np.pi * 1.5 * t)
            + 0.2 * np.random.randn(length)
            + 0.1 * np.sin(2 * np.pi * 10 * t)
        ),
        "acc_y_gravity_free_wave": (
            0.3 * np.cos(2 * np.pi * 1.2 * t)
            + 0.15 * np.random.randn(length)
            + 0.05 * np.sin(2 * np.pi * 15 * t)
        ),
        "acc_z_gravity_free_wave": (
            0.4 * np.sin(2 * np.pi * 0.8 * t) + 0.25 * np.random.randn(length)
        ),
        "acc_mag_gravity_free_wave": (
            np.sqrt(0.5**2 + 0.3**2 + 0.4**2) + 0.1 * np.random.randn(length)
        ),
    }

    return input_waves


def main():
    """Run DTW calculation example."""
    print("Creating sample input waves...")
    input_waves = create_sample_input_waves(1000)

    print("Input wave statistics:")
    for wave_type, wave_data in input_waves.items():
        print(f"  {wave_type}:")
        print(f"    Shape: {wave_data.shape}")
        print(f"    Mean: {wave_data.mean():.4f}")
        print(f"    Std: {wave_data.std():.4f}")
        print(f"    Min: {wave_data.min():.4f}")
        print(f"    Max: {wave_data.max():.4f}")

    print("\nCalculating DTW distances for all behaviors...")
    try:
        # Calculate DTW for all behaviors
        dtw_results = calculate_dtw_distances(
            input_waves, method="sliding_window", window_size=50
        )

        print("\nDTW calculation completed!")
        print(f"Number of behaviors processed: {len(dtw_results)}")

        # Summary statistics
        print("\nDTW Distance Summary:")
        for behavior_name, wave_distances in dtw_results.items():
            print(f"\n{behavior_name}:")
            for wave_type, distances in wave_distances.items():
                print(f"  {wave_type}:")
                print(f"    Mean DTW: {distances.mean():.4f}")
                print(f"    Std DTW: {distances.std():.4f}")
                print(f"    Min DTW: {distances.min():.4f}")
                print(f"    Max DTW: {distances.max():.4f}")

        # Find behavior with minimum average DTW distance
        behavior_avg_distances = {}
        for behavior_name, wave_distances in dtw_results.items():
            total_avg = np.mean(
                [distances.mean() for distances in wave_distances.values()]
            )
            behavior_avg_distances[behavior_name] = total_avg

        best_match_behavior = min(behavior_avg_distances.items(), key=lambda x: x[1])

        print(f"\nBest matching behavior: {best_match_behavior[0]}")
        print(f"Average DTW distance: {best_match_behavior[1]:.4f}")

        # Show top 3 best matches
        sorted_behaviors = sorted(behavior_avg_distances.items(), key=lambda x: x[1])

        print("\nTop 3 best matches:")
        for i, (behavior_name, avg_distance) in enumerate(sorted_behaviors[:3]):
            print(f"{i + 1}. {behavior_name}: {avg_distance:.4f}")

        return dtw_results

    except Exception as e:
        print(f"Error during DTW calculation: {e}")
        import traceback

        traceback.print_exc()
        return None


def plot_dtw_comparison(
    dtw_results: dict, input_waves: dict, behavior_name: str, wave_type: str
):
    """Plot input wave and corresponding DTW distances."""
    if behavior_name not in dtw_results or wave_type not in dtw_results[behavior_name]:
        print(f"Data not available for {behavior_name}/{wave_type}")
        return

    input_wave = input_waves[wave_type]
    dtw_distances = dtw_results[behavior_name][wave_type]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot input wave
    ax1.plot(input_wave, "b-", linewidth=1, label="Input Wave")
    ax1.set_title(f"Input Wave: {wave_type}")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot DTW distances
    ax2.plot(dtw_distances, "r-", linewidth=1, label="DTW Distance")
    ax2.set_title(f"DTW Distances vs {behavior_name}")
    ax2.set_xlabel("Time Point")
    ax2.set_ylabel("DTW Distance")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f"/kaggle/working/dtw_comparison_{behavior_name}_{wave_type}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    results = main()

    if results is not None:
        # Example of plotting results
        input_waves = create_sample_input_waves(1000)

        # Plot for the first behavior and wave type
        first_behavior = list(results.keys())[0]
        first_wave_type = list(results[first_behavior].keys())[0]

        print(f"\nCreating plot for {first_behavior}/{first_wave_type}...")
        plot_dtw_comparison(results, input_waves, first_behavior, first_wave_type)
