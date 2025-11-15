import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.config_dataclass import SplitConfig
from src.data.augment_aux_dataset import AugmentedAuxDataset
from src.data.augment_aux_mixup_dataset import AugmentedAuxMixDataset
from src.data.augment_aux_scaler_dataset import AugmentedAuxScalerDataset
from src.data.augment_dataset import AugmentedDataset
from src.data.augment_validfilter_aux_dataset import AugmentedValidAuxDataset
from src.data.augment_validfilter_rep_aux_dataset import AugmentedValidRepAuxDataset
from src.data.augmentations import TimeSeriesAugmentation
from src.data.aux_meta_dataset import AuxMetaDataset
from src.data.basic_dataset import BasicDataset
from src.data.public_dataset import PublicDataset
from src.data.random_cut_dataset import RandomCutDataset
from src.data.random_slice_dataset import RandomSliceDataset


class DataModule(LightningDataModule):
    """`LightningDataModule` for dataset.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset_name: str = "public",
        df_path: Path = Path(
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
        ),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        splits: SplitConfig = SplitConfig(),
        train_transforms: TimeSeriesAugmentation | None = None,
        valid_transforms: TimeSeriesAugmentation | None = None,
        mixup_prob: float = 0.2,
        mixup_alpha: float = 0.2,
        mixup_max_len_rate: float = 0.15,
        max_epoch: int = 100,
        imu_cols: list[str] | None = None,
        thm_cols: list[str] | None = None,
        tof_cols: list[str] | None = None,
    ) -> None:
        """Initialize a `MNISTDataModule`.

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

        self.batch_size = batch_size
        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_epoch = max_epoch

        self.train_ids = splits.train_ids
        self.valid_ids = splits.valid_ids

        # Feature columns
        self.imu_cols = imu_cols or []
        self.thm_cols = thm_cols or []
        self.tof_cols = tof_cols or []

        # Mixup parameters
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.mixup_max_len_rate = mixup_max_len_rate
        # data transformations
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if len(self.data_train) % self.batch_size_per_device <= 2:  # type: ignore
            drop_last = True
        else:
            drop_last = False
        return DataLoader(
            dataset=self.data_train,  # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if len(self.data_val) % self.batch_size_per_device <= 2:  # type: ignore
            drop_last = True
        else:
            drop_last = False
        return DataLoader(
            dataset=self.data_val,  # type: ignore
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=drop_last,
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

        train_df = self.df[self.df["sequence_id"].isin(self.train_ids)]
        valid_df = self.df[self.df["sequence_id"].isin(self.valid_ids)]
        self.train_df = train_df.copy()
        if self.dataset_name == "public":
            self.data_train = PublicDataset(
                df=train_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                mixup_prob=self.mixup_prob,
                mixup_alpha=self.mixup_alpha,
                mixup_max_len_rate=self.mixup_max_len_rate,
                transforms=None,
            )
            self.data_val = PublicDataset(
                df=valid_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                mixup_prob=self.mixup_prob,
                mixup_alpha=self.mixup_alpha,
                mixup_max_len_rate=self.mixup_max_len_rate,
                transforms=None,
            )
        elif self.dataset_name == "random_slice":
            self.data_train = RandomSliceDataset(
                df=train_df,
                pad_percentile=80,
                mixup_prob=self.mixup_prob,
                mixup_alpha=self.mixup_alpha,
                mixup_max_len_rate=self.mixup_max_len_rate,
            )
            self.data_val = RandomSliceDataset(
                df=valid_df,
                pad_percentile=80,
                phase="valid",
                mixup_prob=self.mixup_prob,
                mixup_alpha=self.mixup_alpha,
                mixup_max_len_rate=self.mixup_max_len_rate,
            )
        elif self.dataset_name == "random_cut":
            self.data_train = RandomCutDataset(
                df=train_df,
                pad_percentile=95,
                mixup_prob=self.mixup_prob,
                mixup_alpha=self.mixup_alpha,
                mixup_max_len_rate=self.mixup_max_len_rate,
            )
            self.data_val = RandomCutDataset(
                df=valid_df,
                pad_percentile=95,
                phase="valid",
                mixup_prob=self.mixup_prob,
                mixup_alpha=self.mixup_alpha,
                mixup_max_len_rate=self.mixup_max_len_rate,
            )
        elif self.dataset_name == "basic":
            self.data_train = BasicDataset(
                df=train_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="fit",
                transforms=None,
            )
            self.data_val = BasicDataset(
                df=valid_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="valid",
                transforms=None,
            )
        elif self.dataset_name == "augmented_basic":
            self.data_train = AugmentedDataset(
                df=train_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="fit",
                transforms=self.train_transforms,  # type: ignore
            )
            self.data_val = AugmentedDataset(
                df=valid_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="valid",
                transforms=self.valid_transforms,  # type: ignore
            )
        elif self.dataset_name == "augmented_aux":
            self.data_train = AugmentedAuxDataset(
                df=train_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="fit",
                transforms=self.train_transforms,  # type: ignore
            )
            self.data_val = AugmentedAuxDataset(
                df=valid_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="valid",
                transforms=self.valid_transforms,  # type: ignore
            )
        elif self.dataset_name == "augmented_validfilter_aux":
            self.data_train = AugmentedValidAuxDataset(
                df=train_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="fit",
                transforms=self.train_transforms,  # type: ignore
            )
            self.data_val = AugmentedValidAuxDataset(
                df=valid_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="valid",
                transforms=self.valid_transforms,  # type: ignore
            )
        elif self.dataset_name == "augmented_validfilter_rep_aux":
            self.data_train = AugmentedValidRepAuxDataset(
                df=train_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="fit",
                transforms=self.train_transforms,  # type: ignore
            )
            self.data_val = AugmentedAuxScalerDataset(
                df=valid_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="valid",
                transforms=self.valid_transforms,  # type: ignore
            )
        elif self.dataset_name == "augmented_aux_meta":
            meta_df = pd.read_csv(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
            )
            meta_df["age"] = (meta_df["age"] / 80.0).astype(np.float32)
            meta_df["height_cm"] = (meta_df["height_cm"] / 200.0).astype(np.float32)
            meta_df["shoulder_to_wrist_cm"] = (
                meta_df["shoulder_to_wrist_cm"] / 80.0
            ).astype(np.float32)
            meta_df["elbow_to_wrist_cm"] = (meta_df["elbow_to_wrist_cm"] / 50.0).astype(
                np.float32
            )
            meta_cols = [
                "adult_child",
                "age",
                "sex",
                "handedness",
                "height_cm",
                "shoulder_to_wrist_cm",
                "elbow_to_wrist_cm",
            ]
            self.data_train = AuxMetaDataset(
                df=train_df,
                meta_df=meta_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                meta_cols=meta_cols,
                pad_percentile=95,
                phase="fit",
                transforms=self.train_transforms,  # type: ignore
            )
            self.data_val = AuxMetaDataset(
                df=valid_df,
                meta_df=meta_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                meta_cols=meta_cols,
                pad_percentile=95,
                phase="valid",
                transforms=self.valid_transforms,  # type: ignore
            )
        elif self.dataset_name == "augmented_aux_mixup":
            # train_ids = train_df["sequence_id"].unique()[:1000]
            # valid_ids = valid_df["sequence_id"].unique()[:1000]
            # # 100種類のidだけ使う
            # train_df = train_df[train_df["sequence_id"].isin(train_ids)]
            # valid_df = valid_df[valid_df["sequence_id"].isin(valid_ids)]
            # print("sample!!!!!!!!!!!!!!")
            # print("\n\n\n\n\n\n")
            self.train_df = train_df.copy()
            self.data_train = AugmentedAuxMixDataset(
                df=train_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="fit",
                mixup_prob=self.mixup_prob,
                mixup_switch_epoch=int(self.max_epoch * 0.5),
                transforms=self.train_transforms,  # type: ignore
            )
            self.data_val = AugmentedAuxMixDataset(
                df=valid_df,
                imu_cols=self.imu_cols,
                thm_cols=self.thm_cols,
                tof_cols=self.tof_cols,
                pad_percentile=95,
                phase="valid",
                transforms=self.valid_transforms,  # type: ignore
            )
        else:
            raise NotImplementedError(
                f"Dataset {self.dataset_name} is not implemented."
            )


if __name__ == "__main__":
    import joblib
    import yaml  # type: ignore

    from src.config_dataclass import DatasetConfig, SplitConfig

    fold = 0
    split_config = SplitConfig()
    split_config.split_dir = Path("/kaggle/working/splits")
    split_config.fold = fold
    # get splits
    with open(split_config.split_dir / f"fold_{fold}" / "train.yaml", "r") as f:
        split_config.train_ids = yaml.safe_load(f)
    with open(split_config.split_dir / f"fold_{fold}" / "valid.yaml", "r") as f:
        split_config.valid_ids = yaml.safe_load(f)

    # Load feature columns for testing
    features_dir = Path("/kaggle/working/features")
    with open(features_dir / "imu_cols.yaml", "r") as f:
        imu_cols = yaml.safe_load(f)["imu_cols"]
    with open(features_dir / "thm_cols.yaml", "r") as f:
        thm_cols = yaml.safe_load(f)["thm_cols"]
    with open(features_dir / "tof_agg_cols.yaml", "r") as f:
        tof_cols = yaml.safe_load(f)["tof_agg_cols"]

    config = DatasetConfig(
        # dataset_name="random_slice",
        # dataset_name="random_cut",
        # dataset_name="basic",
        # dataset_name="augmented_basic",
        # dataset_name="augmented_aux",
        # dataset_name="augmented_aux_meta",
        # dataset_name="augmented_aux_mixup",
        dataset_name="augmented_validfilter_rep_aux",
        df_path=Path(
            # "/kaggle/working/processed_rotations_2/processed_with_rots_df.csv"
            # "/kaggle/working/processed_rot_orient_behavior/processed_df.csv"
            "/kaggle/working/processed_diff01_cumsum_swaphandness4/processed_df.csv"
        ),
        batch_size=64,
        num_workers=2,
        pin_memory=False,
    )
    data_module = DataModule(
        dataset_name=config.dataset_name,
        df_path=config.df_path,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        pin_memory=config.pin_memory,
        splits=split_config,
        train_transforms=None,
        valid_transforms=None,
        imu_cols=imu_cols,
        thm_cols=thm_cols,
        tof_cols=tof_cols,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    print(train_loader.dataset)
    for input, target in train_loader:
        for key, value in input.items():
            print(f"Input {key}: {value.shape}")
        for key, value in target.items():
            print(f"Target {key}: {value.shape}")
        # break
        print(target["labels"].unique())
