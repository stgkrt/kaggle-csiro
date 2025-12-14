from pathlib import Path
from typing import Optional

import pandas as pd
from albumentations.core.composition import Compose
from data.clover_dataset import CloverDataset
from data.clover_height_dataset import CloverHeightDataset
from data.height_dataset import HeightDataset
from data.height_gshh_dataset import HeightGSHHDataset
from data.simple_dataset import SimpleDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DataModule(LightningDataModule):
    """`LightningDataModule` for dataset.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset_name: str = "public",
        df_path: Path = Path("/kaggle/input/csiro-biomass/train.csv"),
        data_root_dir: Path = Path("/kaggle/input/csiro-biomass/"),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        target_cols: Optional[list[str]] = None,
        train_ids: list[str] = [""],
        valid_ids: list[str] = [""],
        train_transforms: Compose | None = None,
        valid_transforms: Compose | None = None,
        mixup_prob: float = 0.2,
        mixup_alpha: float = 0.2,
    ) -> None:
        """
        Args:
            data_dir (str): The data directory. Defaults to `"data/"`.
            train_num (int): The number of training samples. Defaults to `55000`.
            valid_num (int): The number of validation samples. Defaults to `5000`.
            test_num (int): The number of test samples. Defaults to `10000`.
            batch_size (int): The batch size. Defaults to `64`.
            num_workers (int): The number of workers. Defaults to `0`.
            pin_memory (bool): Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None
        # from config
        self.dataset_name = dataset_name
        self.df = pd.read_csv(df_path)
        self.target_cols = target_cols
        self.batch_size = batch_size
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ids = train_ids
        self.valid_ids = valid_ids

        self.data_root_dir = data_root_dir
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,  # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,  # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,  # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def setup(self, stage: str | None = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible"
                    + f"by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        train_df = self.df[self.df["sample_id"].isin(self.train_ids)]
        valid_df = self.df[self.df["sample_id"].isin(self.valid_ids)]
        if self.dataset_name == "simple":
            self.data_train = SimpleDataset(
                df=train_df,
                data_root_dir=self.data_root_dir,
                transforms=self.train_transforms,
                target_cols=self.target_cols,
            )
            self.data_val = SimpleDataset(
                df=valid_df,
                data_root_dir=self.data_root_dir,
                transforms=self.valid_transforms,
                target_cols=self.target_cols,
            )
        elif self.dataset_name == "height":
            self.data_train = HeightDataset(
                df=train_df,
                data_root_dir=self.data_root_dir,
                phase="fit",
                transforms=self.train_transforms,
                target_cols=self.target_cols,
            )
            self.data_val = HeightDataset(
                df=valid_df,
                data_root_dir=self.data_root_dir,
                phase="validate",
                transforms=self.valid_transforms,
                target_cols=self.target_cols,
            )
        elif self.dataset_name == "height_gshh":
            self.data_train = HeightGSHHDataset(
                df=train_df,
                data_root_dir=self.data_root_dir,
                phase="fit",
                transforms=self.train_transforms,
                target_cols=self.target_cols,
            )
            self.data_val = HeightGSHHDataset(
                df=valid_df,
                data_root_dir=self.data_root_dir,
                phase="validate",
                transforms=self.valid_transforms,
                target_cols=self.target_cols,
            )
        elif self.dataset_name == "clover":
            self.data_train = CloverDataset(
                df=train_df,
                data_root_dir=self.data_root_dir,
                phase="fit",
                transforms=self.train_transforms,
                target_cols=self.target_cols,
            )
            self.data_val = CloverDataset(
                df=valid_df,
                data_root_dir=self.data_root_dir,
                phase="validate",
                transforms=self.valid_transforms,
                target_cols=self.target_cols,
            )
        elif self.dataset_name == "clover_height":
            self.data_train = CloverHeightDataset(
                df=train_df,
                data_root_dir=self.data_root_dir,
                phase="fit",
                transforms=self.train_transforms,
                target_cols=self.target_cols,
            )
            self.data_val = CloverHeightDataset(
                df=valid_df,
                data_root_dir=self.data_root_dir,
                phase="validate",
                transforms=self.valid_transforms,
                target_cols=self.target_cols,
            )
        else:
            raise NotImplementedError(
                f"Dataset {self.dataset_name} is not implemented."
            )


if __name__ == "__main__":
    dataset_name = "simple"
    dataset_name = "height_gshh"
    from src.configs import DatasetConfig

    data_config = DatasetConfig()
    df = pd.read_csv(data_config.df_path)
    train_ids = df.iloc[:100]["sample_id"].tolist()
    valid_ids = df.iloc[:-50]["sample_id"].tolist()

    data_module = DataModule(
        dataset_name=dataset_name,
        df_path=data_config.df_path,
        data_root_dir=Path("/kaggle/input/csiro-biomass/"),
        target_cols=data_config.target_cols,
        num_workers=0,
        batch_size=8,
        pin_memory=True,
        train_ids=train_ids,
        valid_ids=valid_ids,
        train_transforms=None,
        valid_transforms=None,
        mixup_prob=0.5,
        mixup_alpha=0.2,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()

    for input, target in train_loader:
        for key, value in input.items():
            print(f"Input {key}: {value.shape}")
        for key, value in target.items():
            print(f"Target {key}: {value.shape}")
