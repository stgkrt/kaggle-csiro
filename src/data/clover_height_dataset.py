from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from albumentations.core.composition import Compose
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CloverHeightDataset(Dataset):
    """A basic dataset class that can be extended for custom datasets."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root_dir: Path,
        target_cols: list[str],
        phase: str = "fit",
        transforms: Optional[Compose] = None,
    ):
        self.df = df
        self.image_path_list = df["image_path"].tolist()

        self.image_path_list = [
            data_root_dir / Path(image_path) for image_path in self.image_path_list
        ]
        # Preserve order while removing duplicates using dict.fromkeys()
        self.image_path_list = list(dict.fromkeys(self.image_path_list))
        self.target_cols = target_cols
        self.pure_clover = [
            "Clover",
            "SubcloverDalkeith",
            "SubcloverLosa",
            "WhiteClover",
        ]
        self.mix_clover = [
            "Phalaris_BarleyGrass_SilverGrass_SpearGrass_Clover_Capeweed",
            "Phalaris_Clover",
            "Phalaris_Clover_Ryegrass_Barleygrass_Bromegrass",
            "Phalaris_Ryegrass_Clover",
            "Ryegrass_Clover",
        ]
        self.include_clover_species = self.pure_clover + self.mix_clover
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
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        inputs = {
            "image": torch.Tensor(image),
        }
        if self.phase == "test":
            return inputs
        else:
            target_values = [
                row[row["target_name"] == target_name]["target"].values[0]
                for target_name in self.target_cols
            ]
            # target_values[0]が0かそうでないかで分類するラベルを追加
            # include_clover_label = 1.0 if target_values[0] > 0 else 0.0
            include_clover_label = (
                1.0 if row["Species"].values[0] in self.include_clover_species else 0.0
            )
            height = row["Height_Ave_cm"].values[0]
            labels = {
                "labels": torch.Tensor(target_values),
                "include_clover_label": torch.Tensor([include_clover_label]),
                "height": torch.Tensor([height]),
            }
            return inputs, labels


if __name__ == "__main__":
    from src.configs import AugmentationConfig, DatasetConfig
    from src.data.augmentations import get_train_transforms

    dataset_config = DatasetConfig()
    aug_config = AugmentationConfig()
    train_transforms = get_train_transforms(aug_config)

    df = pd.read_csv(dataset_config.df_path)
    # print(df.head())
    dataset = CloverHeightDataset(
        df,
        data_root_dir=dataset_config.data_root_dir,
        target_cols=dataset_config.target_cols,
        phase="fit",
        transforms=train_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    for batch in dataloader:
        inputs, labels = batch
        print(inputs["image"].shape)
        print(labels["labels"].shape)
        print(labels["include_clover_label"].shape)
        break
