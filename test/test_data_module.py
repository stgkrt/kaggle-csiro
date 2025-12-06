"""Tests for src/data/data_module.py"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.configs import AugmentationConfig
from src.data.augmentations import get_train_transforms, get_valid_transforms
from src.data.data_module import DataModule


class TestDataModule:
    """Test DataModule class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        temp_dir = tempfile.mkdtemp()
        data_root = Path(temp_dir)

        # Create sample images
        image_dir = data_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(image_dir / f"sample_{i}.jpg")

        # Create sample dataframe
        target_cols = [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g",
        ]
        data = []

        for i in range(10):
            for target_name in target_cols:
                data.append(
                    {
                        "sample_id": f"sample_{i}_{target_name}",
                        "image_path": f"images/sample_{i}.jpg",
                        "target_name": target_name,
                        "target": float(np.random.rand() * 100),
                    }
                )

        df = pd.DataFrame(data)
        df_path = data_root / "train.csv"
        df.to_csv(df_path, index=False)

        # Split IDs
        all_ids = [f"sample_{i}" for i in range(10)]
        train_ids = all_ids[:7]
        valid_ids = all_ids[7:]

        return {
            "df_path": df_path,
            "data_root_dir": data_root,
            "target_cols": target_cols,
            "train_ids": train_ids,
            "valid_ids": valid_ids,
        }

    def test_initialization(self, sample_data):
        """Test DataModule initialization"""
        aug_config = AugmentationConfig()
        train_transforms = get_train_transforms(aug_config)
        valid_transforms = get_valid_transforms(aug_config)

        datamodule = DataModule(
            dataset_name="simple",
            df_path=sample_data["df_path"],
            data_root_dir=sample_data["data_root_dir"],
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            target_cols=sample_data["target_cols"],
            train_ids=sample_data["train_ids"],
            valid_ids=sample_data["valid_ids"],
            train_transforms=train_transforms,
            valid_transforms=valid_transforms,
        )

        assert datamodule is not None
        assert isinstance(datamodule, LightningDataModule)

    def test_inherits_lightning_datamodule(self, sample_data):
        """Test that DataModule inherits from LightningDataModule"""
        aug_config = AugmentationConfig()
        train_transforms = get_train_transforms(aug_config)
        valid_transforms = get_valid_transforms(aug_config)

        datamodule = DataModule(
            df_path=sample_data["df_path"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            train_ids=sample_data["train_ids"],
            valid_ids=sample_data["valid_ids"],
            train_transforms=train_transforms,
            valid_transforms=valid_transforms,
        )

        assert isinstance(datamodule, LightningDataModule)

    def test_hparams_saved(self, sample_data):
        """Test that hyperparameters are saved"""
        aug_config = AugmentationConfig()
        train_transforms = get_train_transforms(aug_config)
        valid_transforms = get_valid_transforms(aug_config)

        datamodule = DataModule(
            df_path=sample_data["df_path"],
            data_root_dir=sample_data["data_root_dir"],
            batch_size=16,
            num_workers=2,
            target_cols=sample_data["target_cols"],
            train_ids=sample_data["train_ids"],
            valid_ids=sample_data["valid_ids"],
            train_transforms=train_transforms,
            valid_transforms=valid_transforms,
        )

        assert hasattr(datamodule, "hparams")
        assert datamodule.hparams.batch_size == 16
        assert datamodule.hparams.num_workers == 2

    def test_default_parameters(self):
        """Test default parameters"""
        datamodule = DataModule()

        assert datamodule.dataset_name == "public"
        assert datamodule.batch_size == 64
        assert datamodule.num_workers == 0
        assert datamodule.pin_memory is True

    def test_batch_size_setting(self, sample_data):
        """Test that batch size is correctly set"""
        aug_config = AugmentationConfig()
        train_transforms = get_train_transforms(aug_config)
        valid_transforms = get_valid_transforms(aug_config)

        for batch_size in [4, 8, 16, 32]:
            datamodule = DataModule(
                df_path=sample_data["df_path"],
                data_root_dir=sample_data["data_root_dir"],
                batch_size=batch_size,
                target_cols=sample_data["target_cols"],
                train_ids=sample_data["train_ids"],
                valid_ids=sample_data["valid_ids"],
                train_transforms=train_transforms,
                valid_transforms=valid_transforms,
            )

            assert datamodule.batch_size == batch_size
            assert datamodule.batch_size_per_device == batch_size


class TestDataModuleDataLoaders:
    """Test DataModule dataloader methods"""

    @pytest.fixture
    def setup_datamodule(self):
        """Setup DataModule with sample data"""
        temp_dir = tempfile.mkdtemp()
        data_root = Path(temp_dir)

        # Create sample images
        image_dir = data_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(image_dir / f"sample_{i}.jpg")

        # Create sample dataframe
        target_cols = [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g",
        ]
        data = []

        for i in range(10):
            for target_name in target_cols:
                data.append(
                    {
                        "sample_id": f"sample_{i}_{target_name}",
                        "image_path": f"images/sample_{i}.jpg",
                        "target_name": target_name,
                        "target": float(np.random.rand() * 100),
                    }
                )

        df = pd.DataFrame(data)
        df_path = data_root / "train.csv"
        df.to_csv(df_path, index=False)

        all_ids = [f"sample_{i}" for i in range(10)]
        train_ids = all_ids[:7]
        valid_ids = all_ids[7:]

        aug_config = AugmentationConfig()
        train_transforms = get_train_transforms(aug_config)
        valid_transforms = get_valid_transforms(aug_config)

        datamodule = DataModule(
            dataset_name="simple",
            df_path=df_path,
            data_root_dir=data_root,
            batch_size=4,
            num_workers=0,
            pin_memory=False,
            target_cols=target_cols,
            train_ids=train_ids,
            valid_ids=valid_ids,
            train_transforms=train_transforms,
            valid_transforms=valid_transforms,
        )

        return datamodule

    def test_train_dataloader_exists(self, setup_datamodule):
        """Test that train_dataloader method exists"""
        datamodule = setup_datamodule
        assert hasattr(datamodule, "train_dataloader")
        assert callable(datamodule.train_dataloader)

    def test_val_dataloader_exists(self, setup_datamodule):
        """Test that val_dataloader method exists"""
        datamodule = setup_datamodule
        assert hasattr(datamodule, "val_dataloader")
        assert callable(datamodule.val_dataloader)


class TestDataModuleIntegration:
    """Integration tests for DataModule"""

    def test_datamodule_attributes(self):
        """Test that DataModule has required attributes"""
        datamodule = DataModule()

        required_attrs = [
            "dataset_name",
            "batch_size",
            "num_workers",
            "pin_memory",
            "train_ids",
            "valid_ids",
            "data_root_dir",
            "train_transforms",
            "valid_transforms",
        ]

        for attr in required_attrs:
            assert hasattr(datamodule, attr), f"Missing attribute: {attr}"

    def test_different_dataset_names(self):
        """Test with different dataset names"""
        dataset_names = ["simple", "height", "clover"]

        for dataset_name in dataset_names:
            datamodule = DataModule(dataset_name=dataset_name)
            assert datamodule.dataset_name == dataset_name

    def test_mixup_parameters(self):
        """Test mixup parameters are stored"""
        datamodule = DataModule(mixup_prob=0.5, mixup_alpha=0.3)

        assert datamodule.mixup_prob == 0.5
        assert datamodule.mixup_alpha == 0.3

    def test_empty_train_valid_ids(self):
        """Test with empty train/valid ID lists"""
        datamodule = DataModule(train_ids=[], valid_ids=[])

        assert datamodule.train_ids == []
        assert datamodule.valid_ids == []

    def test_pin_memory_setting(self):
        """Test pin_memory setting"""
        datamodule_true = DataModule(pin_memory=True)
        datamodule_false = DataModule(pin_memory=False)

        assert datamodule_true.pin_memory is True
        assert datamodule_false.pin_memory is False

    def test_num_workers_setting(self):
        """Test num_workers setting"""
        for num_workers in [0, 2, 4]:
            datamodule = DataModule(num_workers=num_workers)
            assert datamodule.num_workers == num_workers


class TestDataModuleEdgeCases:
    """Test edge cases for DataModule"""

    def test_none_transforms(self):
        """Test with None transforms"""
        datamodule = DataModule(train_transforms=None, valid_transforms=None)

        assert datamodule.train_transforms is None
        assert datamodule.valid_transforms is None

    def test_large_batch_size(self):
        """Test with large batch size"""
        datamodule = DataModule(batch_size=1024)
        assert datamodule.batch_size == 1024

    def test_zero_num_workers(self):
        """Test with zero num_workers (single-process loading)"""
        datamodule = DataModule(num_workers=0)
        assert datamodule.num_workers == 0

    def test_target_cols_none(self):
        """Test with None target_cols"""
        datamodule = DataModule(target_cols=None)
        assert datamodule.target_cols is None

    def test_custom_target_cols(self):
        """Test with custom target columns"""
        custom_cols = ["target1", "target2", "target3"]
        datamodule = DataModule(target_cols=custom_cols)
        assert datamodule.target_cols == custom_cols


if __name__ == "__main__":
    # Simple test runner
    print("Testing DataModule initialization...")
    datamodule = DataModule()
    assert datamodule is not None
    print("✓ DataModule initialized successfully")

    print("\nTesting DataModule attributes...")
    test = TestDataModuleIntegration()
    test.test_datamodule_attributes()
    test.test_different_dataset_names()
    test.test_mixup_parameters()
    print("✓ DataModule attribute tests passed")

    print("\nTesting edge cases...")
    test = TestDataModuleEdgeCases()
    test.test_none_transforms()
    test.test_large_batch_size()
    test.test_zero_num_workers()
    print("✓ Edge case tests passed")

    print("\nAll DataModule tests passed!")
    print("Note: Some tests require pytest with fixtures for full functionality.")
