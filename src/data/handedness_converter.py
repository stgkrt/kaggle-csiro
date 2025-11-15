"""
Handedness-based data conversion utility for sensor data.

This module provides functions to convert sensor data for left-handed subjects:

NumPy Array Functions:
1. convert_imu_features: Basic conversion - flips acc_x and rot_x
2. convert_quaternion_to_left_handed: Coordinate system conversion - flips rot_y and rot_z
3. convert_imu_features_full: Complete conversion - combines both transformations

DataFrame Functions:
1. convert_dataframe: Basic conversion - flips acc_x and rot_x for left-handed subjects
2. convert_dataframe_quaternion: Coordinate system conversion - flips rot_y and rot_z
3. convert_dataframe_full: Complete conversion - combines both transformations

For left-handed subjects (handedness=0), the transformations are:
- acc_x → -acc_x (handedness adjustment)
- rot_x → -rot_x (handedness adjustment)
- rot_y → -rot_y (coordinate system conversion)
- rot_z → -rot_z (coordinate system conversion)
- rot_w remains unchanged (quaternion scalar part)
"""

from pathlib import Path
from typing import Dict, Set

import numpy as np
import pandas as pd


class HandednessConverter:
    """
    Converts sensor data based on subject handedness.
    For left-handed subjects (handedness=0), flips the X-axis values of accelerometer and gyroscope data.
    """

    def __init__(
        self,
        demographics_csv_path: str = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv",
    ):
        """
        Initialize the converter with demographics data.

        Args:
            demographics_csv_path: Path to the demographics CSV file
        """
        self.demographics_df = pd.read_csv(demographics_csv_path)
        self.left_handed_subjects = self._get_left_handed_subjects()

    def _get_left_handed_subjects(self) -> Set[str]:
        """
        Get set of left-handed subjects (handedness=0).

        Returns:
            Set of subject IDs who are left-handed
        """
        left_handed = self.demographics_df[self.demographics_df["handedness"] == 0][
            "subject"
        ].tolist()
        return set(left_handed)

    def is_left_handed(self, subject_id: str) -> bool:
        """
        Check if a subject is left-handed.

        Args:
            subject_id: Subject ID to check

        Returns:
            True if subject is left-handed, False otherwise
        """
        return subject_id in self.left_handed_subjects

    def convert_imu_features(
        self, imu_features: np.ndarray, subject_id: str, imu_cols: list
    ) -> np.ndarray:
        """
        Convert IMU features based on handedness.
        Flips acc_x and rot_x values for left-handed subjects.

        Args:
            imu_features: IMU feature array of shape (sequence_length, num_features)
            subject_id: Subject ID
            imu_cols: List of IMU column names

        Returns:
            Converted IMU features
        """
        if not self.is_left_handed(subject_id):
            return imu_features

        # Create a copy to avoid modifying the original data
        converted_features = imu_features.copy()

        # Find indices of acc_x and rot_x columns
        acc_x_idx = None
        rot_x_idx = None

        for i, col in enumerate(imu_cols):
            if col == "acc_x":
                acc_x_idx = i
            elif col == "rot_x":
                rot_x_idx = i

        # Flip acc_x values if column exists
        if acc_x_idx is not None:
            converted_features[:, acc_x_idx] = -converted_features[:, acc_x_idx]

        # Flip rot_x values if column exists
        if rot_x_idx is not None:
            converted_features[:, rot_x_idx] = -converted_features[:, rot_x_idx]

        return converted_features

    def convert_quaternion_to_left_handed(
        self, imu_features: np.ndarray, subject_id: str, imu_cols: list
    ) -> np.ndarray:
        """
        Convert quaternion from right-handed to left-handed coordinate system.
        For left-handed subjects, converts the quaternion representation.

        In right-handed to left-handed conversion:
        - rot_x remains the same (x-axis flip is handled separately)
        - rot_y becomes -rot_y (y-axis inversion)
        - rot_z becomes -rot_z (z-axis inversion)
        - rot_w remains the same (scalar part)

        Args:
            imu_features: IMU feature array of shape (sequence_length, num_features)
            subject_id: Subject ID
            imu_cols: List of IMU column names

        Returns:
            Converted IMU features with quaternion coordinate system conversion
        """
        if not self.is_left_handed(subject_id):
            return imu_features

        # Create a copy to avoid modifying the original data
        converted_features = imu_features.copy()

        # Find indices of quaternion components
        rot_w_idx = None
        rot_x_idx = None
        rot_y_idx = None
        rot_z_idx = None

        for i, col in enumerate(imu_cols):
            if col == "rot_w":
                rot_w_idx = i
            elif col == "rot_x":
                rot_x_idx = i
            elif col == "rot_y":
                rot_y_idx = i
            elif col == "rot_z":
                rot_z_idx = i

        # Convert quaternion from right-handed to left-handed coordinate system
        # rot_w stays the same (scalar part)
        # rot_x stays the same (x-axis component)
        # rot_y becomes -rot_y (y-axis inversion)
        # rot_z becomes -rot_z (z-axis inversion)

        if rot_y_idx is not None:
            converted_features[:, rot_y_idx] = -converted_features[:, rot_y_idx]

        if rot_z_idx is not None:
            converted_features[:, rot_z_idx] = -converted_features[:, rot_z_idx]

        return converted_features

    def convert_imu_features_full(
        self, imu_features: np.ndarray, subject_id: str, imu_cols: list
    ) -> np.ndarray:
        """
        Complete IMU features conversion for left-handed subjects.
        Combines both coordinate system conversion and axis flipping.

        Args:
            imu_features: IMU feature array of shape (sequence_length, num_features)
            subject_id: Subject ID
            imu_cols: List of IMU column names

        Returns:
            Fully converted IMU features
        """
        if not self.is_left_handed(subject_id):
            return imu_features

        # Create a copy to avoid modifying the original data
        converted_features = imu_features.copy()

        # Find indices of all relevant columns
        acc_x_idx = None
        rot_w_idx = None
        rot_x_idx = None
        rot_y_idx = None
        rot_z_idx = None

        for i, col in enumerate(imu_cols):
            if col == "acc_x":
                acc_x_idx = i
            elif col == "rot_w":
                rot_w_idx = i
            elif col == "rot_x":
                rot_x_idx = i
            elif col == "rot_y":
                rot_y_idx = i
            elif col == "rot_z":
                rot_z_idx = i

        # Step 1: Flip acc_x for handedness
        if acc_x_idx is not None:
            converted_features[:, acc_x_idx] = -converted_features[:, acc_x_idx]

        # Step 2: Convert quaternion coordinate system (right-handed to left-handed)
        # rot_w stays the same (scalar part)
        # rot_x stays the same initially, but we may need to flip it for handedness
        # rot_y becomes -rot_y (y-axis inversion for coordinate system)
        # rot_z becomes -rot_z (z-axis inversion for coordinate system)

        if rot_y_idx is not None:
            converted_features[:, rot_y_idx] = -converted_features[:, rot_y_idx]

        if rot_z_idx is not None:
            converted_features[:, rot_z_idx] = -converted_features[:, rot_z_idx]

        # Step 3: Additional rot_x flip for handedness (after coordinate conversion)
        if rot_x_idx is not None:
            converted_features[:, rot_x_idx] = -converted_features[:, rot_x_idx]

        return converted_features

    def convert_dataframe(
        self, df: pd.DataFrame, acc_x_col: str = "acc_x", rot_x_col: str = "rot_x"
    ) -> pd.DataFrame:
        """
        Convert a dataframe by flipping acc_x and rot_x for left-handed subjects.

        Args:
            df: DataFrame containing sensor data with 'subject' column
            acc_x_col: Name of accelerometer X column
            rot_x_col: Name of gyroscope/rotation X column

        Returns:
            DataFrame with converted values
        """
        converted_df = df.copy()

        # Check if required columns exist
        if "subject" not in df.columns:
            raise ValueError("DataFrame must contain 'subject' column")

        for col in [acc_x_col, rot_x_col]:
            if col in df.columns:
                # Create mask for left-handed subjects
                left_handed_mask = converted_df["subject"].isin(
                    self.left_handed_subjects
                )
                # Flip values for left-handed subjects
                converted_df.loc[left_handed_mask, col] = -converted_df.loc[
                    left_handed_mask, col
                ]

        return converted_df

    def convert_dataframe_quaternion(
        self,
        df: pd.DataFrame,
        rot_w_col: str = "rot_w",
        rot_x_col: str = "rot_x",
        rot_y_col: str = "rot_y",
        rot_z_col: str = "rot_z",
    ) -> pd.DataFrame:
        """
        Convert quaternion in dataframe from right-handed to left-handed coordinate system
        for left-handed subjects.

        Args:
            df: DataFrame containing sensor data with 'subject' column
            rot_w_col: Name of quaternion w component column
            rot_x_col: Name of quaternion x component column
            rot_y_col: Name of quaternion y component column
            rot_z_col: Name of quaternion z component column

        Returns:
            DataFrame with converted quaternion values
        """
        converted_df = df.copy()

        # Check if required columns exist
        if "subject" not in df.columns:
            raise ValueError("DataFrame must contain 'subject' column")

        # Create mask for left-handed subjects
        left_handed_mask = converted_df["subject"].isin(self.left_handed_subjects)

        # Convert quaternion coordinate system for left-handed subjects
        # rot_w stays the same (scalar part)
        # rot_x stays the same (x-axis component)
        # rot_y becomes -rot_y (y-axis inversion)
        # rot_z becomes -rot_z (z-axis inversion)

        if rot_y_col in df.columns:
            converted_df.loc[left_handed_mask, rot_y_col] = -converted_df.loc[
                left_handed_mask, rot_y_col
            ]

        if rot_z_col in df.columns:
            converted_df.loc[left_handed_mask, rot_z_col] = -converted_df.loc[
                left_handed_mask, rot_z_col
            ]

        return converted_df

    def convert_dataframe_full(
        self,
        df: pd.DataFrame,
        acc_x_col: str = "acc_x",
        rot_w_col: str = "rot_w",
        rot_x_col: str = "rot_x",
        rot_y_col: str = "rot_y",
        rot_z_col: str = "rot_z",
    ) -> pd.DataFrame:
        """
        Complete dataframe conversion for left-handed subjects.
        Combines both coordinate system conversion and axis flipping.

        Args:
            df: DataFrame containing sensor data with 'subject' column
            acc_x_col: Name of accelerometer X column
            rot_w_col: Name of quaternion w component column
            rot_x_col: Name of quaternion x component column
            rot_y_col: Name of quaternion y component column
            rot_z_col: Name of quaternion z component column

        Returns:
            DataFrame with fully converted values
        """
        converted_df = df.copy()

        # Check if required columns exist
        if "subject" not in df.columns:
            raise ValueError("DataFrame must contain 'subject' column")

        # Create mask for left-handed subjects
        left_handed_mask = converted_df["subject"].isin(self.left_handed_subjects)

        # Step 1: Flip acc_x for handedness
        if acc_x_col in df.columns:
            converted_df.loc[left_handed_mask, acc_x_col] = -converted_df.loc[
                left_handed_mask, acc_x_col
            ]

        # Step 2: Convert quaternion coordinate system (right-handed to left-handed)
        # rot_w stays the same (scalar part)
        # rot_x stays the same initially, but we flip it in step 3
        # rot_y becomes -rot_y (y-axis inversion for coordinate system)
        # rot_z becomes -rot_z (z-axis inversion for coordinate system)

        if rot_y_col in df.columns:
            converted_df.loc[left_handed_mask, rot_y_col] = -converted_df.loc[
                left_handed_mask, rot_y_col
            ]

        if rot_z_col in df.columns:
            converted_df.loc[left_handed_mask, rot_z_col] = -converted_df.loc[
                left_handed_mask, rot_z_col
            ]

        # Step 3: Additional rot_x flip for handedness (after coordinate conversion)
        if rot_x_col in df.columns:
            converted_df.loc[left_handed_mask, rot_x_col] = -converted_df.loc[
                left_handed_mask, rot_x_col
            ]

        return converted_df

    def get_left_handed_subjects_list(self) -> list:
        """
        Get list of left-handed subject IDs.

        Returns:
            List of left-handed subject IDs
        """
        return list(self.left_handed_subjects)

    def print_handedness_info(self):
        """Print information about handedness distribution in the dataset."""
        total_subjects = len(self.demographics_df)
        left_handed_count = len(self.left_handed_subjects)
        right_handed_count = total_subjects - left_handed_count

        print("Handedness Distribution:")
        print(f"Total subjects: {total_subjects}")
        print(f"Right-handed (handedness=1): {right_handed_count}")
        print(f"Left-handed (handedness=0): {left_handed_count}")
        print(f"Left-handed subjects: {sorted(list(self.left_handed_subjects))}")


if __name__ == "__main__":
    # Example usage
    converter = HandednessConverter()
    converter.print_handedness_info()

    # Test with sample data
    sample_imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
    sample_features = np.array(
        [[1.0, 2.0, 3.0, 0.5, -0.5, 0.3, 0.2], [1.5, 2.5, 3.5, 0.6, -0.6, 0.4, 0.3]]
    )

    # Test with left-handed subject
    left_handed_subject = "SUBJ_002923"

    print(f"\n=== Testing {left_handed_subject} (Left-handed) ===")
    print("Original features:")
    print("acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z")
    print(sample_features)

    # Test basic conversion (only acc_x and rot_x flip)
    converted_basic = converter.convert_imu_features(
        sample_features, left_handed_subject, sample_imu_cols
    )
    print("\nBasic conversion (acc_x and rot_x flipped):")
    print(converted_basic)

    # Test quaternion coordinate system conversion
    converted_quat = converter.convert_quaternion_to_left_handed(
        sample_features, left_handed_subject, sample_imu_cols
    )
    print("\nQuaternion coordinate system conversion (rot_y and rot_z flipped):")
    print(converted_quat)

    # Test full conversion (combines both)
    converted_full = converter.convert_imu_features_full(
        sample_features, left_handed_subject, sample_imu_cols
    )
    print("\nFull conversion (acc_x flip + quaternion coord system + rot_x flip):")
    print(converted_full)

    # Test with right-handed subject
    right_handed_subject = "SUBJ_000206"
    print(f"\n=== Testing {right_handed_subject} (Right-handed) ===")
    print("Original features:")
    print(sample_features)

    converted_rh_full = converter.convert_imu_features_full(
        sample_features, right_handed_subject, sample_imu_cols
    )
    print("Full conversion (no change expected):")
    print(converted_rh_full)

    # Test DataFrame conversion functions
    print("\n=== Testing DataFrame Conversions ===")

    # Create sample DataFrame
    sample_df = pd.DataFrame(
        {
            "subject": [
                left_handed_subject,
                left_handed_subject,
                right_handed_subject,
                right_handed_subject,
            ],
            "acc_x": [1.0, 1.5, 2.0, 2.5],
            "acc_y": [2.0, 2.5, 3.0, 3.5],
            "acc_z": [3.0, 3.5, 4.0, 4.5],
            "rot_w": [0.5, 0.6, 0.7, 0.8],
            "rot_x": [-0.5, -0.6, -0.7, -0.8],
            "rot_y": [0.3, 0.4, 0.5, 0.6],
            "rot_z": [0.2, 0.3, 0.4, 0.5],
        }
    )

    print("Original DataFrame:")
    print(sample_df)

    # Test basic DataFrame conversion
    df_basic = converter.convert_dataframe(sample_df)
    print("\nBasic DataFrame conversion (acc_x and rot_x flipped for left-handed):")
    print(df_basic)

    # Test quaternion DataFrame conversion
    df_quat = converter.convert_dataframe_quaternion(sample_df)
    print(
        "\nQuaternion DataFrame conversion (rot_y and rot_z flipped for left-handed):"
    )
    print(df_quat)

    # Test full DataFrame conversion
    df_full = converter.convert_dataframe_full(sample_df)
    print("\nFull DataFrame conversion (complete transformation for left-handed):")
    print(df_full)

    # Show the differences more clearly
    print("\n=== Summary of Transformations for Left-handed Subjects ===")
    print("Original:     [acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z]")
    print("Basic:        [-acc_x, acc_y, acc_z, rot_w, -rot_x, rot_y, rot_z]")
    print("Quaternion:   [acc_x, acc_y, acc_z, rot_w, rot_x, -rot_y, -rot_z]")
    print("Full:         [-acc_x, acc_y, acc_z, rot_w, -rot_x, -rot_y, -rot_z]")
