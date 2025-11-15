import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml  # type: ignore
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.data.augmentations import TimeSeriesAugmentation
from src.data.dataset_process import (
    get_features_and_labels_valid_filter,
    get_input_features,
    get_labels,
    get_max_min_by_group,
)
from src.data.process_representative_waves import RepresentativeWaveDTWCalculator


class AugmentedValidRepAuxDataset(Dataset):
    """
    A basic dataset with augmentation that can be extended for custom datasets.

    This dataset includes DTW (Dynamic Time Warping) features calculated from
    representative waves, which are added to the IMU features to enhance
    gesture recognition performance.
    """

    def __init__(
        self,
        df,
        imu_cols: list[str],
        thm_cols: list[str],
        tof_cols: list[str],
        transforms: TimeSeriesAugmentation,
        pad_percentile: int = 95,
        signal_cut_rate: float = 0.8,
        phase: str = "fit",
        use_dtw_features: bool = True,
        representative_waves_dir: str = "/kaggle/working/representative_waves",
        dtw_window_size: int = 10,
        use_float16: bool = True,
    ):
        self.df = df
        self.transforms = transforms
        self.phase = phase
        self.signal_cut_rate = signal_cut_rate
        self.use_dtw_features = use_dtw_features
        self.dtw_window_size = dtw_window_size
        self.use_float16 = use_float16
        self.dtype = np.float16 if use_float16 else np.float32

        # DTW calculator initialization
        if self.use_dtw_features:
            try:
                self.dtw_calculator = RepresentativeWaveDTWCalculator(
                    representative_waves_dir
                )
                available_behaviors = len(self.dtw_calculator.get_available_behaviors())
                print(
                    f"DTW calculator initialized with {available_behaviors} behaviors"
                )
            except Exception as e:
                print(f"Failed to initialize DTW calculator: {e}")
                self.use_dtw_features = False
                self.dtw_calculator = None
        else:
            self.dtw_calculator = None

        # 引数で受け取った列名情報を使用
        self.imu_cols = imu_cols
        self.thm_cols = thm_cols
        self.tof_agg_cols = tof_cols
        self.features_cols = imu_cols + thm_cols + tof_cols
        feature_scaler_path = Path("/kaggle/working/features/feature_scaler.yaml")
        with open(feature_scaler_path, "r") as f:
            self.feature_scaler = yaml.safe_load(f)

        print(f"Number of IMU features: {imu_cols}")
        print(f"Number of thermal features: {thm_cols}")
        print(f"Number of ToF features: {tof_cols}")
        (
            self.imu_features_list,
            self.thm_features_list,
            self.tof_features_list,
            self.labels,
            self.orient_labels,
            self.behavior_labels,
            data_len_list,
        ) = get_features_and_labels_valid_filter(
            df,
            imu_cols=imu_cols,
            thm_cols=thm_cols,
            tof_cols=tof_cols,
            features_cols=self.features_cols,
            label_col="gesture_le",
            orient_label_col="orientation_le",
            behavior_label_col="behavior_le",
        )
        # self.pad_len = int(np.percentile(data_len_list, pad_percentile))
        self.pad_len = 127
        self.fillna_value = 0.0
        self.dtw_featurenames_list = [
            "acc_x_gravity_free",
            "acc_y_gravity_free",
            "acc_z_gravity_free",
        ]
        self.dtw_featureidx_list = [
            self.imu_cols.index(col) for col in self.dtw_featurenames_list
        ]

    def _scale_features(self, features, features_col_list):
        for i, col in enumerate(features_col_list):
            max_val = self.feature_scaler[col]["max"]
            min_val = self.feature_scaler[col]["min"]
            if max_val == min_val:
                features[:, i] = 0.0
            else:
                features[:, i] = (features[:, i] - min_val) / (max_val - min_val + 1e-8)
        return features

    def _generate_wave_data_from_acc(self, dtw_features):
        """Generate wave data from acceleration features for DTW calculation."""
        # Convert to appropriate dtype for processing
        dtw_features = dtw_features.astype(self.dtype)

        # Assuming the first 3 columns are acc_x, acc_y, acc_z
        # Check if gravity-free acceleration exists in the columns
        # Based on the grep results, we might have acc_x_gravity_free etc.
        wave_data = {
            "acc_x_gravity_free_wave": dtw_features[:, 0],
            "acc_y_gravity_free_wave": dtw_features[:, 1],
            "acc_z_gravity_free_wave": dtw_features[:, 2],
        }

        return wave_data

    def _calculate_dtw_features(self, dtw_features):
        """Calculate DTW features and add them to imu_features."""
        if not self.use_dtw_features or self.dtw_calculator is None:
            return np.zeros_like(dtw_features, dtype=self.dtype)

        try:
            # Generate wave data from acceleration features
            wave_data = self._generate_wave_data_from_acc(dtw_features)
            # Calculate DTW distances for all behaviors
            dtw_results = self.dtw_calculator.calculate_dtw_for_all_behaviors(
                wave_data, method="sliding_window", window_size=self.dtw_window_size
            )

            # Flatten DTW results into features
            dtw_features_list = []
            for _behavior_name, wave_distances in dtw_results.items():
                for _wave_type, distances in wave_distances.items():
                    # Ensure distances have the same length as input features
                    if len(distances) == len(dtw_features):
                        # Convert to specified dtype for memory efficiency
                        distances_dtype = distances.astype(self.dtype)
                        dtw_features_list.append(distances_dtype.reshape(-1, 1))

            if dtw_features_list:
                dtw_features_result = np.concatenate(dtw_features_list, axis=1)
                return dtw_features_result.astype(self.dtype)
            else:
                return np.zeros_like(dtw_features, dtype=self.dtype)

        except Exception:
            return np.zeros_like(dtw_features, dtype=self.dtype)

    def __len__(self):
        return len(self.imu_features_list)

    def _get_random_cut_features(self, features):
        # padded_features = np.zeros((self.pad_len, features.shape[1]))
        padded_features = self.fillna_value * np.ones(
            (self.pad_len, features.shape[1]), dtype=self.dtype
        )
        feature_length = len(features)
        signal_cut_length_base = min(self.pad_len, feature_length)
        cut_length = np.random.randint(
            int(signal_cut_length_base * self.signal_cut_rate),
            signal_cut_length_base + 1,
        )
        # 信号の後ろからランダムな位置で切り出す
        cut_start = np.random.randint(0, feature_length - cut_length + 1)
        cut_end = cut_start + cut_length
        random_signal_start_idx = np.random.randint(0, self.pad_len - cut_length + 1)
        padded_features[
            random_signal_start_idx : random_signal_start_idx + cut_length, :
        ] = features[cut_start:cut_end, :].astype(self.dtype)
        return padded_features

    def _get_post_cut_features(self, features):
        feature_length = len(features)
        padded_features = np.zeros((self.pad_len, features.shape[1]), dtype=self.dtype)
        if feature_length < self.pad_len:
            padded_features[-feature_length:, :] = features.astype(self.dtype)
        else:
            padded_features = features[-self.pad_len :].astype(self.dtype)
        return padded_features

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        imu_features = self.imu_features_list[idx]
        thm_features = self.thm_features_list[idx]
        tof_features = self.tof_features_list[idx]

        # rotだけ残して他のchはscaleしてもよさそう(効かなかった)
        # imu_features[:, -4:] = self._scale_features(
        #     imu_features[:, -4:], self.imu_cols[-4:]
        # )
        thm_features = self._scale_features(thm_features, self.thm_cols)
        tof_features = self._scale_features(tof_features, self.tof_agg_cols)
        # Apply augmentations if available and in training phase
        if self.transforms is not None and self.phase == "fit":
            # Convert to numpy for augmentations
            inputs_np = {
                "imu_features": imu_features,
                "thm_features": thm_features,
                "tof_features": tof_features,
            }
            # Apply augmentations
            inputs_np = self.transforms(inputs_np)
        else:
            inputs_np = {
                "imu_features": imu_features,
                "thm_features": thm_features,
                "tof_features": tof_features,
            }

        if self.phase == "fit":
            imu_features = self._get_random_cut_features(inputs_np["imu_features"])
            thm_features = self._get_random_cut_features(inputs_np["thm_features"])
            tof_features = self._get_random_cut_features(inputs_np["tof_features"])
        elif self.phase == "valid":
            imu_features = self._get_post_cut_features(inputs_np["imu_features"])
            thm_features = self._get_post_cut_features(inputs_np["thm_features"])
            tof_features = self._get_post_cut_features(inputs_np["tof_features"])
        else:
            raise ValueError(f"Phase {self.phase} is not supported.")

        # calculate dtw
        dtw_features = self._calculate_dtw_features(
            imu_features[:, self.dtw_featureidx_list]
        )
        rot_features = imu_features[:, -4:].astype(self.dtype)
        other_features = imu_features[:, :-4].astype(self.dtype)
        imu_features = np.concatenate(
            [other_features, dtw_features, rot_features], axis=1
        ).astype(self.dtype)

        # Convert to appropriate tensor dtype
        tensor_dtype = torch.float16 if self.use_float16 else torch.float32

        inputs = {
            "imu_features": torch.tensor(imu_features, dtype=tensor_dtype),
            "thm_features": torch.tensor(thm_features, dtype=tensor_dtype),
            "tof_features": torch.tensor(tof_features, dtype=tensor_dtype),
        }
        labels = {
            "labels": torch.tensor(self.labels[idx], dtype=tensor_dtype),
            "orientation": torch.tensor(self.orient_labels[idx], dtype=tensor_dtype),
            "behavior": torch.tensor(self.behavior_labels[idx], dtype=tensor_dtype),
        }
        return inputs, labels


if __name__ == "__main__":
    # df_path = "/kaggle/working/processed_rot_orient_behavior/processed_df.csv"
    # df_path = "/kaggle/working/processed_tof_region/processed_df.csv"
    # df_path = "/kaggle/working/processed_roll16/processed_df.csv"

    # df_path = "/kaggle/working/processed_diff_cumsum/processed_df.csv"
    # df_path = (
    #     "/kaggle/working/processed_diff01_cumsum_swaphandness_height/processed_df.csv"
    # )
    df_path = (
        "/kaggle/working/processed_diff01_cumsum_swaphandness_elbow/processed_df.csv"
    )
    df = pd.read_csv(df_path)
    print(len(df))
    print(df.columns)
    # Load feature columns for testing
    features_dir = Path("/kaggle/working/features")
    with open(features_dir / "imu_cols.yaml", "r") as f:
        imu_cols = yaml.safe_load(f)["imu_cols"]
    with open(features_dir / "thm_cols.yaml", "r") as f:
        thm_cols = yaml.safe_load(f)["thm_cols"]
    with open(features_dir / "tof_agg_cols.yaml", "r") as f:
        tof_cols = yaml.safe_load(f)["tof_agg_cols"]

    transforms = TimeSeriesAugmentation(
        time_stretch_range=(0.8, 1.2),
        time_shift_range=0.1,
        magnitude_scale_range=(0.8, 1.2),
        rotation_angle_range=np.pi / 6,
        mask_ratio=0.1,
        freq_filter_range=(0.1, 0.9),
        aug_prob=0.5,
    )
    dataset = AugmentedValidRepAuxDataset(
        df,
        imu_cols=imu_cols,
        thm_cols=thm_cols,
        tof_cols=tof_cols,
        transforms=transforms,
        pad_percentile=95,
        phase="valid",
        use_dtw_features=True,  # Enable DTW features
        representative_waves_dir="/kaggle/working/representative_waves",
        dtw_window_size=50,
        use_float16=True,  # Enable float16 for faster processing
    )
    print(f"Dataset length: {len(dataset)}")
    import os

    train_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True,
    )
    for inputs, labels in train_loader:
        start_time = time.time()

        print("Original IMU features shape:", inputs["imu_features"].shape)
        print("Thermal features shape:", inputs["thm_features"].shape)
        print("ToF features shape:", inputs["tof_features"].shape)
        print("Data type:", inputs["imu_features"].dtype)

        print("Labels shape:", labels["labels"].shape)
        print("Orientation labels shape:", labels["orientation"].shape)
        print("Behavior labels shape:", labels["behavior"].shape)

        # Check if DTW features were added
        original_imu_cols_count = len(imu_cols)
        actual_imu_features = inputs["imu_features"].shape[-1]
        if actual_imu_features > original_imu_cols_count:
            dtw_features_count = actual_imu_features - original_imu_cols_count
            print(f"DTW features added: {dtw_features_count} features")
        else:
            print("No DTW features added")

        end_time = time.time()
        print(f"Batch processing time: {end_time - start_time:.4f} seconds")

        # Memory usage check
        imu_memory = (
            inputs["imu_features"].element_size() * inputs["imu_features"].numel()
        )
        print(f"IMU features memory usage: {imu_memory / 1024 / 1024:.2f} MB")

        break
