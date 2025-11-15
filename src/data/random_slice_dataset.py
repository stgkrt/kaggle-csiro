from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml  # type: ignore
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.data.dataset_process import (
    get_features_list,
    get_input_features,
    get_labels,
    get_max_min_by_group,
)


class RandomSliceDataset(Dataset):
    """A basic dataset class that can be extended for custom datasets."""

    def __init__(
        self,
        df,
        pad_percentile: int = 80,
        phase: str = "fit",
        mixup_prob: float = 0.2,
        mixup_alpha: float = 0.2,
        mixup_max_len_rate: float = 0.15,
    ):
        self.df = df
        self.output_dir = Path("/kaggle/working/encoders")
        self.phase = phase
        self.mixup_prob = mixup_prob
        self.scaling_each = False
        colnames = df.columns.tolist()
        imu_cols = [col for col in colnames if col.startswith(("acc_", "rot_"))]
        imu_cols += [col for col in colnames if col.startswith(("rot_matrix_"))]
        thm_cols = [col for col in colnames if col.startswith("thm_")]
        tof_agg_cols = []
        for i in range(1, 6):
            tof_agg_cols.extend(
                [
                    f"tof_{i}_mean",
                    f"tof_{i}_std",
                    f"tof_{i}_max",
                    # f"tof_{i}_min",
                ]
            )
        print(f"Number of IMU features: {len(imu_cols)}")
        # print(f"Number of position/velocity features: {len(pos_vel_cols)}")
        print(f"Number of thermal features: {len(thm_cols)}")
        print(f"Number of aggregated ToF features: {len(tof_agg_cols)}")
        # self.features_cols = imu_cols + thm_cols + tof_agg_cols
        # self.features_cols = imu_cols + pos_vel_cols + thm_cols + tof_agg_cols
        self.features_cols = imu_cols + thm_cols + tof_agg_cols

        self.features_list, data_len_list, self.group_list = get_features_list(
            df, self.features_cols
        )
        self.pad_len = int(np.percentile(data_len_list, pad_percentile))

        self.max_min_df = get_max_min_by_group(
            df, self.features_cols, group_col="subject"
        )

        if self.phase == "fit" or self.phase == "valid":
            self.labels = get_labels(df)

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        if self.phase == "fit":
            features = self.features_list[idx]
            feature_length = len(features)
            if feature_length < self.pad_len:
                features = np.pad(
                    features,
                    ((0, self.pad_len - feature_length), (0, 0)),
                    mode="constant",
                )
            else:
                slice_idx = np.random.randint(0, feature_length - self.pad_len + 1)
                slice_idx = max(slice_idx, 0)
                features = features[slice_idx : slice_idx + self.pad_len]
        elif self.phase == "valid":
            features = self.features_list[idx]
            feature_length = len(features)
            if feature_length < self.pad_len:
                features = np.pad(
                    features,
                    ((0, self.pad_len - feature_length), (0, 0)),
                    mode="constant",
                )
            else:
                features = features[-self.pad_len :]
        else:
            print(f"Phase {self.phase} is not supported.")

        inputs = {"features": torch.Tensor(features)}
        if self.phase == "fit" or self.phase == "valid":
            labels = {
                "labels": torch.Tensor(self.labels[idx]),
            }
            return inputs, labels
        else:
            return inputs


if __name__ == "__main__":
    # Example usage

    # df_path = "/kaggle/working/processed/processed_df.csv"
    df_path = "/kaggle/working/processed/processed_with_homotrans_df.csv"
    # df_path = "/kaggle/working/processed_rolling_mean/processed_with_homotrans_df.csv"
    df = pd.read_csv(df_path)

    dataset = RandomSliceDataset(
        df,
        pad_percentile=95,
        phase="fit",
        mixup_prob=0.2,
        mixup_alpha=0.2,
        mixup_max_len_rate=0.15,
    )
    print(f"Dataset length: {len(dataset)}")

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    for inputs, labels in train_loader:
        print(inputs["features"].shape)
        if "labels" in labels:
            print(labels["labels"].shape)
        break
