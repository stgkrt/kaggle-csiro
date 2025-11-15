"""
Representative waves DTW calculator module.

This module provides functionality to calculate Dynamic Time Warping (DTW)
between input waveforms and representative wave files stored in numpy format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dtaidistance import dtw

logger = logging.getLogger(__name__)


class RepresentativeWaveDTWCalculator:
    """
    Calculate DTW between input waveforms and representative waves.

    This class loads representative wave files and calculates DTW distances
    for each sample in the input waveform, returning the same shape as input.
    """

    def __init__(
        self, representative_waves_dir: str = "/kaggle/working/representative_waves"
    ):
        """
        Initialize the DTW calculator.

        Args:
            representative_waves_dir: Path to directory containing
                representative waves
        """
        self.representative_waves_dir = Path(representative_waves_dir)
        self.representative_waves = {}
        self._load_representative_waves()

    def _load_representative_waves(self):
        """Load all representative wave files from the directory."""
        if not self.representative_waves_dir.exists():
            raise FileNotFoundError(
                f"Representative waves directory not found: "
                f"{self.representative_waves_dir}"
            )

        # Get all behavior categories
        behavior_dirs = [
            d for d in self.representative_waves_dir.iterdir() if d.is_dir()
        ]

        for behavior_dir in behavior_dirs:
            behavior_name = behavior_dir.name
            self.representative_waves[behavior_name] = {}

            # Load all npy files in each behavior directory
            npy_files = list(behavior_dir.glob("*.npy"))

            for npy_file in npy_files:
                wave_type = npy_file.stem  # e.g., 'acc_mag_gravity_free_wave'
                try:
                    wave_data = np.load(npy_file)
                    self.representative_waves[behavior_name][wave_type] = wave_data
                    logger.info(
                        f"Loaded {behavior_name}/{wave_type}: shape {wave_data.shape}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load {npy_file}: {e}")

        logger.info(
            f"Loaded representative waves for "
            f"{len(self.representative_waves)} behaviors"
        )

    def calculate_dtw_pointwise(
        self,
        input_wave: np.ndarray,
        representative_wave: np.ndarray,
        window_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Calculate DTW distance for each point in the input wave.

        Args:
            input_wave: Input waveform (1D array)
            representative_wave: Representative waveform (1D array)
            window_size: Window size for DTW calculation (None for no
                constraint)

        Returns:
            DTW distances with same length as input_wave
        """
        input_len = len(input_wave)

        # Get the optimal warping path
        if window_size is not None:
            path = dtw.warping_path(input_wave, representative_wave, window=window_size)
        else:
            path = dtw.warping_path(input_wave, representative_wave)

        # Initialize output array
        pointwise_distances = np.full(input_len, np.inf)

        # For each point in the input wave, find the minimum distance
        # along the warping path
        for i in range(input_len):
            min_dist = np.inf
            for path_i, path_j in path:
                if path_i == i:
                    dist = abs(input_wave[i] - representative_wave[path_j])
                    min_dist = min(min_dist, dist)
            pointwise_distances[i] = min_dist

        return pointwise_distances

    def calculate_dtw_sliding_window(
        self,
        input_wave: np.ndarray,
        representative_wave: np.ndarray,
        window_size: int = 50,
    ) -> np.ndarray:
        """
        Calculate DTW using sliding window approach for pointwise distances.

        Args:
            input_wave: Input waveform (1D array)
            representative_wave: Representative waveform (1D array)
            window_size: Size of sliding window

        Returns:
            DTW distances with same length as input_wave
        """
        input_len = len(input_wave)
        pointwise_distances = np.zeros(input_len)

        half_window = window_size // 2

        for i in range(input_len):
            # Define window around current position
            start_idx = max(0, i - half_window)
            end_idx = min(input_len, i + half_window + 1)

            # Extract window from input
            input_window = input_wave[start_idx:end_idx]

            # Calculate DTW distance between window and representative wave
            distance = dtw.distance(input_window, representative_wave)

            # Normalize by window length
            pointwise_distances[i] = distance / len(input_window)

        return pointwise_distances

    def calculate_dtw_for_behavior(
        self,
        input_waves: Dict[str, np.ndarray],
        behavior_name: str,
        method: str = "sliding_window",
        window_size: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Calculate DTW distances for all wave types of a specific behavior.

        Args:
            input_waves: Dictionary with wave type as key and wave data as value
            behavior_name: Name of the behavior to compare against
            method: DTW calculation method ("pointwise" or "sliding_window")
            window_size: Window size for sliding window method

        Returns:
            Dictionary with wave type as key and DTW distances as value
        """
        if behavior_name not in self.representative_waves:
            raise ValueError(
                f"Behavior '{behavior_name}' not found in representative waves"
            )

        dtw_distances = {}
        representative_behavior = self.representative_waves[behavior_name]

        for wave_type, input_wave in input_waves.items():
            if wave_type not in representative_behavior:
                logger.warning(
                    f"Wave type '{wave_type}' not found for behavior '{behavior_name}'"
                )
                continue

            representative_wave = representative_behavior[wave_type]

            if method == "pointwise":
                distances = self.calculate_dtw_pointwise(
                    input_wave, representative_wave, window_size
                )
            elif method == "sliding_window":
                distances = self.calculate_dtw_sliding_window(
                    input_wave, representative_wave, window_size
                )
            else:
                raise ValueError(f"Unknown method: {method}")

            dtw_distances[wave_type] = distances

        return dtw_distances

    def calculate_dtw_for_all_behaviors(
        self,
        input_waves: Dict[str, np.ndarray],
        method: str = "sliding_window",
        window_size: int = 50,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate DTW distances for all behaviors and wave types.

        Args:
            input_waves: Dictionary with wave type as key and wave data as value
            method: DTW calculation method ("pointwise" or "sliding_window")
            window_size: Window size for sliding window method

        Returns:
            Nested dictionary: {behavior_name: {wave_type: dtw_distances}}
        """
        all_dtw_distances = {}

        for behavior_name in self.representative_waves.keys():
            try:
                behavior_distances = self.calculate_dtw_for_behavior(
                    input_waves, behavior_name, method, window_size
                )
                all_dtw_distances[behavior_name] = behavior_distances
                logger.info(f"Calculated DTW for behavior: {behavior_name}")
            except Exception as e:
                logger.error(
                    f"Failed to calculate DTW for behavior {behavior_name}: {e}"
                )

        return all_dtw_distances

    def get_available_behaviors(self) -> List[str]:
        """Get list of available behavior names."""
        return list(self.representative_waves.keys())

    def get_available_wave_types(self, behavior_name: str) -> List[str]:
        """Get list of available wave types for a specific behavior."""
        if behavior_name not in self.representative_waves:
            return []
        return list(self.representative_waves[behavior_name].keys())


def calculate_dtw_distances(
    input_waves: Dict[str, np.ndarray],
    representative_waves_dir: str = "/kaggle/working/representative_waves",
    method: str = "sliding_window",
    window_size: int = 50,
    target_behaviors: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convenience function to calculate DTW distances.

    Args:
        input_waves: Dictionary with wave type as key and wave data as value
        representative_waves_dir: Path to representative waves directory
        method: DTW calculation method ("pointwise" or "sliding_window")
        window_size: Window size for calculation
        target_behaviors: List of specific behaviors to calculate (None for all)

    Returns:
        Nested dictionary: {behavior_name: {wave_type: dtw_distances}}
    """
    calculator = RepresentativeWaveDTWCalculator(representative_waves_dir)

    if target_behaviors is None:
        return calculator.calculate_dtw_for_all_behaviors(
            input_waves, method, window_size
        )
    else:
        result = {}
        for behavior_name in target_behaviors:
            if behavior_name in calculator.get_available_behaviors():
                result[behavior_name] = calculator.calculate_dtw_for_behavior(
                    input_waves, behavior_name, method, window_size
                )
        return result


# Example usage
if __name__ == "__main__":
    # Example of how to use the DTW calculator

    # Create sample input waves
    import pandas as pd

    df_path = "/kaggle/working/processed_diff01_cumsum_swaphandness4/processed_df.csv"
    df = pd.read_csv(df_path)
    sequence_ids = df["sequence_id"].unique()
    use_id = sequence_ids[0]
    sequence_df = df[df["sequence_id"] == use_id]
    input_waves = {
        "acc_x_gravity_free_wave": sequence_df["acc_x_gravity_free_wave"].values,
        "acc_y_gravity_free_wave": sequence_df["acc_y_gravity_free_wave"].values,
        "acc_z_gravity_free_wave": sequence_df["acc_z_gravity_free_wave"].values,
        "acc_mag_gravity_free_wave": sequence_df["acc_mag_gravity_free_wave"].values,
    }

    # Calculate DTW distances
    try:
        dtw_results = calculate_dtw_distances(input_waves)

        print("DTW calculation completed!")
        print(f"Number of behaviors processed: {len(dtw_results)}")

        for behavior_name, wave_distances in dtw_results.items():
            print(f"\nBehavior: {behavior_name}")
            for wave_type, distances in wave_distances.items():
                print(
                    f"  {wave_type}: shape {distances.shape}, mean={distances.mean():.4f}"
                )

    except Exception as e:
        print(f"Error: {e}")
