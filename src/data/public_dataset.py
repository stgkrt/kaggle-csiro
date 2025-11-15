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
    get_features_list,
    get_input_features,
    get_labels,
    get_max_min_by_group,
)


class PublicDataset(Dataset):
    """A basic dataset class that can be extended for custom datasets."""

    def __init__(
        self,
        df,
        imu_cols: list[str],
        thm_cols: list[str],
        tof_cols: list[str],
        pad_percentile: int = 95,
        phase: str = "fit",
        mixup_prob: float = 0.2,
        mixup_alpha: float = 0.2,
        mixup_max_len_rate: float = 0.15,
        transforms: TimeSeriesAugmentation | None = None,
    ):
        self.df = df
        self.output_dir = Path("/kaggle/working/encoders")
        self.phase = phase
        self.mixup_prob = mixup_prob
        self.scaling_each = False
        self.transforms = transforms

        # 引数で受け取った列名情報を使用
        self.imu_cols = imu_cols
        self.thm_cols = thm_cols
        self.tof_agg_cols = tof_cols

        print(f"Number of IMU features: {len(imu_cols)}")
        print(f"Number of thermal features: {len(thm_cols)}")
        print(f"Number of ToF features: {len(tof_cols)}")
        print("features_cols:", imu_cols + thm_cols + tof_cols)
        # Use IMU features only for now (as per the original implementation)
        self.features_cols = self.imu_cols
        print("features_cols:", self.features_cols)

        features_list, data_len_list, self.group_list = get_features_list(
            df, self.features_cols
        )
        self.pad_len = int(np.percentile(data_len_list, pad_percentile))

        self.input_features, self.pad_len = get_input_features(
            features_list, self.pad_len, self.features_cols, scaling=False
        )
        self.max_min_df = get_max_min_by_group(
            df, self.features_cols, group_col="subject"
        )

        if phase == "fit":
            self.labels = get_labels(df)
            # labels = np.array(labels_list, dtype=np.int64)
            # labels_ohe = F.one_hot(
            #     torch.tensor(labels), num_classes=len(df["gesture_le"].unique())
            # ).numpy()
            # self.labels = labels_ohe

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        if self.scaling_each:
            features = self.input_features[idx]
            for i, col in enumerate(self.features_cols):
                max_val = self.max_min_df.loc[self.group_list[idx], f"{col}_max"]
                min_val = self.max_min_df.loc[self.group_list[idx], f"{col}_min"]
                if max_val == min_val:
                    features[:, i] = 0.0
                else:
                    features[:, i] = (features[:, i] - min_val) / (
                        max_val - min_val + 1e-8
                    )

            # features = self.input_features[idx]
            # for i in range(features.shape[1]):
            #     max_val = np.max(features[:, i])
            #     min_val = np.min(features[:, i])
            #     if max_val == min_val:
            #         features[:, i] = 0.0
            #     else:
            #         features[:, i] = (features[:, i] - min_val) / (max_val - min_val)
            self.input_features[idx] = features
        inputs = {"features": torch.Tensor(self.input_features[idx])}
        if self.phase == "fit":
            if np.random.rand() < self.mixup_prob:
                inputs["features"] = self.mixup.update(torch.Tensor(inputs["features"]))
            labels = {
                "labels": torch.Tensor(self.labels[idx]),
            }
            return inputs, labels
        else:
            return inputs


if __name__ == "__main__":
    # Example usage
    df_path = "/kaggle/working/processed_rolling_mean/processed_with_homotrans_df.csv"
    df = pd.read_csv(df_path)

    # Load feature columns for testing
    features_dir = Path("/kaggle/working/features")
    with open(features_dir / "imu_cols.yaml", "r") as f:
        imu_cols = yaml.safe_load(f)["imu_cols"]
    with open(features_dir / "thm_cols.yaml", "r") as f:
        thm_cols = yaml.safe_load(f)["thm_cols"]
    with open(features_dir / "tof_agg_cols.yaml", "r") as f:
        tof_cols = yaml.safe_load(f)["tof_agg_cols"]

    dataset = PublicDataset(
        df,
        imu_cols=imu_cols,
        thm_cols=thm_cols,
        tof_cols=tof_cols,
        pad_percentile=95,
        phase="fit",
        mixup_prob=0.2,
        mixup_alpha=0.2,
        mixup_max_len_rate=0.15,
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Input features shape: {dataset.input_features.shape}")
    print(f"Labels shape: {dataset.labels.shape}")

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
