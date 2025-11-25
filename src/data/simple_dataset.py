from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from albumentations.core.composition import Compose
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    """A basic dataset class that can be extended for custom datasets."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root_dir: Path,
        phase: str = "fit",
        transforms: Optional[Compose] = None,
    ):
        self.df = df
        self.image_path_list = df["image_path"].tolist()

        self.image_path_list = [
            data_root_dir / Path(image_path) for image_path in self.image_path_list
        ]
        self.target_cols = [
            "Dry_Clover_g",
            "Dry_Dead_g",
            "Dry_Green_g",
            "Dry_Total_g",
            "GDM_g",
        ]
        self.phase = phase
        self.transforms = transforms

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        image_path = self.image_path_list[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        row = self.df[
            self.df["image_path"] == str(image_path.relative_to(image_path.parents[1]))
        ]

        if self.phase == "fit" and self.transforms is not None:
            image = self.transforms(image=image)["image"]

        target_values = [
            row[row["target_name"] == target_name]["target"].values[0]
            for target_name in self.target_cols
        ]
        inputs = {
            "image": torch.Tensor(image),
        }
        labels = {
            "labels": torch.Tensor(target_values),
        }
        return inputs, labels


if __name__ == "__main__":
    df_path = Path("/kaggle/input/csiro-biomass/train.csv")
    data_root_dir = Path("/kaggle/input/csiro-biomass")
    df = pd.read_csv(df_path)

    from src.configs import AugmentationConfig
    from src.data.augmentations import get_train_transforms

    aug_config = AugmentationConfig()
    train_transforms = get_train_transforms(aug_config)
    # print(df.head())
    dataset = SimpleDataset(df, data_root_dir, phase="fit", transforms=train_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    for batch in dataloader:
        inputs, labels = batch
        print(inputs["image"].shape)
        print(labels["labels"].shape)
        break
