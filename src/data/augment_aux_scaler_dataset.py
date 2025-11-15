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
    get_features_and_labels,
    get_input_features,
    get_labels,
    get_max_min_by_group,
)


class AugmentedAuxScalerDataset(Dataset):
    """A basic dataset with augmentation that can be extended for custom datasets."""

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
    ):
        self.df = df
        self.transforms = transforms
        self.phase = phase
        self.signal_cut_rate = signal_cut_rate

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
        ) = get_features_and_labels(
            df,
            imu_cols=imu_cols,
            thm_cols=thm_cols,
            tof_cols=tof_cols,
            features_cols=self.features_cols,
            label_col="gesture_le",
            orient_label_col="orientation_le",
            behavior_label_col="behavior_le",
        )
        self.pad_len = int(np.percentile(data_len_list, pad_percentile))
        self.imu_features_list, self.imu_scaler = self.standard_scale(
            self.imu_features_list
        )
        self.thm_features_list, self.thm_scaler = self.standard_scale(
            self.thm_features_list
        )
        self.tof_features_list, self.tof_scaler = self.standard_scale(
            self.tof_features_list
        )
        with open("/kaggle/working/encoders/imu_scaler.pkl", "wb") as f:
            joblib.dump(self.imu_scaler, f)
        with open("/kaggle/working/encoders/thm_scaler.pkl", "wb") as f:
            joblib.dump(self.thm_scaler, f)
        with open("/kaggle/working/encoders/tof_scaler.pkl", "wb") as f:
            joblib.dump(self.tof_scaler, f)

    def _scale_features(self, features, features_col_list):
        for i, col in enumerate(features_col_list):
            max_val = self.feature_scaler[col]["max"]
            min_val = self.feature_scaler[col]["min"]
            if max_val == min_val:
                features[:, i] = 0.0
            else:
                features[:, i] = (features[:, i] - min_val) / (max_val - min_val + 1e-8)
        return features

    def standard_scale(self, data_unscaled):
        scaler_function = StandardScaler()
        scaler = scaler_function.fit(np.concatenate(data_unscaled, axis=0))
        return [scaler.transform(x) for x in data_unscaled], scaler

    def __len__(self):
        return len(self.imu_features_list)

    def _get_random_cut_features(self, features):
        padded_features = np.zeros((self.pad_len, features.shape[1]))
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
        ] = features[cut_start:cut_end, :]
        return padded_features

    def _get_post_cut_features(self, features):
        feature_length = len(features)
        padded_features = np.zeros((self.pad_len, features.shape[1]))
        if feature_length < self.pad_len:
            padded_features[-feature_length:, :] = features
        else:
            padded_features = features[-self.pad_len :]
        return padded_features

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        imu_features = self.imu_features_list[idx]
        thm_features = self.thm_features_list[idx]
        tof_features = self.tof_features_list[idx]
        # imu_features = self._scale_features(imu_features, self.imu_cols)
        # thm_features = self._scale_features(thm_features, self.thm_cols)
        # tof_features = self._scale_features(tof_features, self.tof_agg_cols)
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
        inputs = {
            "imu_features": torch.Tensor(imu_features),
            "thm_features": torch.Tensor(thm_features),
            "tof_features": torch.Tensor(tof_features),
        }
        labels = {
            "labels": torch.Tensor(self.labels[idx]),
            "orientation": torch.Tensor(self.orient_labels[idx]),
            "behavior": torch.Tensor(self.behavior_labels[idx]),
        }
        return inputs, labels


if __name__ == "__main__":
    # df_path = "/kaggle/working/processed_rot_orient_behavior/processed_df.csv"
    # df_path = "/kaggle/working/processed_tof_region/processed_df.csv"
    df_path = "/kaggle/working/processed_roll16/processed_df.csv"
    df = pd.read_csv(df_path)
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
    dataset = AugmentedAuxScalerDataset(
        df,
        imu_cols=imu_cols,
        thm_cols=thm_cols,
        tof_cols=tof_cols,
        transforms=transforms,
        pad_percentile=95,
        phase="valid",
    )
    print(f"Dataset length: {len(dataset)}")

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    for inputs, labels in train_loader:
        print(inputs["imu_features"].shape)
        print(inputs["thm_features"].shape)
        print(inputs["tof_features"].shape)

        print(labels["labels"].shape)
        print(labels["orientation"].shape)
        print(labels["behavior"].shape)
