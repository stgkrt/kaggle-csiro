"""Tests for src/data/simple_dataset.py"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.configs import AugmentationConfig
from src.data.augmentations import get_train_transforms, get_valid_transforms
from src.data.simple_dataset import SimpleDataset


class TestSimpleDataset:
    """Test SimpleDataset class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        # Create temporary directory and sample images
        temp_dir = tempfile.mkdtemp()
        data_root = Path(temp_dir)

        # Create sample images
        image_dir = data_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Create 3 sample images
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(image_dir / f"sample_{i}.jpg")

        # Create sample dataframe (long format)
        data = []
        target_cols = [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g",
        ]

        for i in range(3):
            for target_name in target_cols:
                data.append(
                    {
                        "image_path": f"images/sample_{i}.jpg",
                        "target_name": target_name,
                        "target": float(np.random.rand() * 100),
                    }
                )

        df = pd.DataFrame(data)

        return {
            "df": df,
            "data_root_dir": data_root,
            "target_cols": target_cols,
        }

    def test_initialization(self, sample_data):
        """Test dataset initialization"""
        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
            transforms=None,
        )

        assert dataset is not None
        assert len(dataset) == 3  # 3 unique images

    def test_len(self, sample_data):
        """Test __len__ method"""
        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
        )

        assert len(dataset) == 3

    def test_getitem_fit_phase(self, sample_data):
        """Test __getitem__ in fit phase"""
        aug_config = AugmentationConfig(resize=False)
        transforms = get_valid_transforms(aug_config)

        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
            transforms=transforms,
        )

        inputs, labels = dataset[0]

        # Check inputs
        assert "image" in inputs
        assert isinstance(inputs["image"], torch.Tensor)
        assert inputs["image"].shape[0] == 3  # RGB channels

        # Check labels
        assert "labels" in labels
        assert isinstance(labels["labels"], torch.Tensor)
        assert labels["labels"].shape[0] == 5  # 5 targets

    def test_getitem_test_phase(self, sample_data):
        """Test __getitem__ in test phase"""
        aug_config = AugmentationConfig(resize=False)
        transforms = get_valid_transforms(aug_config)

        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="test",
            transforms=transforms,
        )

        inputs = dataset[0]

        # In test phase, only inputs are returned
        assert "image" in inputs
        assert isinstance(inputs["image"], torch.Tensor)
        assert inputs["image"].shape[0] == 3

    def test_with_train_transforms(self, sample_data):
        """Test dataset with training transforms"""
        aug_config = AugmentationConfig(
            resize=True, resize_img_height=256, resize_img_width=256
        )
        transforms = get_train_transforms(aug_config)

        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
            transforms=transforms,
        )

        inputs, labels = dataset[0]

        # Check that resize was applied
        assert inputs["image"].shape[1] == 256
        assert inputs["image"].shape[2] == 256

    def test_without_transforms(self, sample_data):
        """Test dataset without transforms"""
        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
            transforms=None,
        )

        inputs, labels = dataset[0]

        # Without transforms, image should still be a Tensor
        assert isinstance(inputs["image"], torch.Tensor)

    def test_duplicate_removal(self, sample_data):
        """Test that duplicate image paths are removed"""
        # Add duplicate entries in dataframe
        df_with_dupes = pd.concat(
            [sample_data["df"], sample_data["df"]], ignore_index=True
        )

        dataset = SimpleDataset(
            df=df_with_dupes,
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
        )

        # Should still have only 3 unique images
        assert len(dataset) == 3

    def test_target_values_correctness(self, sample_data):
        """Test that target values are correctly extracted"""
        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
            transforms=None,
        )

        inputs, labels = dataset[0]

        # Check that we have correct number of targets
        assert labels["labels"].shape[0] == len(sample_data["target_cols"])

        # Check that target values are floats
        assert labels["labels"].dtype in [torch.float32, torch.float64]


class TestSimpleDatasetEdgeCases:
    """Test edge cases for SimpleDataset"""

    def test_single_image(self):
        """Test with single image"""
        temp_dir = tempfile.mkdtemp()
        data_root = Path(temp_dir)

        # Create single image
        image_dir = data_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(image_dir / "sample.jpg")

        # Create dataframe
        target_cols = [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g",
        ]
        data = []
        for target_name in target_cols:
            data.append(
                {
                    "image_path": "images/sample.jpg",
                    "target_name": target_name,
                    "target": 1.0,
                }
            )

        df = pd.DataFrame(data)

        dataset = SimpleDataset(
            df=df,
            data_root_dir=data_root,
            target_cols=target_cols,
            phase="fit",
        )

        assert len(dataset) == 1

    def test_multiple_target_columns(self):
        """Test with different numbers of target columns"""
        temp_dir = tempfile.mkdtemp()
        data_root = Path(temp_dir)

        image_dir = data_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(image_dir / "sample.jpg")

        # Test with 3 targets instead of 5
        target_cols = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"]
        data = []
        for target_name in target_cols:
            data.append(
                {
                    "image_path": "images/sample.jpg",
                    "target_name": target_name,
                    "target": 1.0,
                }
            )

        df = pd.DataFrame(data)

        dataset = SimpleDataset(
            df=df,
            data_root_dir=data_root,
            target_cols=target_cols,
            phase="fit",
        )

        inputs, labels = dataset[0]
        assert labels["labels"].shape[0] == 3


class TestSimpleDatasetIntegration:
    """Integration tests for SimpleDataset"""

    def test_with_dataloader(self, sample_data):
        """Test dataset with PyTorch DataLoader"""
        from torch.utils.data import DataLoader

        aug_config = AugmentationConfig()
        transforms = get_valid_transforms(aug_config)

        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
            transforms=transforms,
        )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Test iteration
        for batch in dataloader:
            inputs, labels = batch
            assert inputs["image"].shape[0] <= 2  # batch size
            assert labels["labels"].shape[0] <= 2
            break

    def test_consistent_output_shape(self, sample_data):
        """Test that output shapes are consistent across batches"""
        aug_config = AugmentationConfig(
            resize=True, resize_img_height=256, resize_img_width=256
        )
        transforms = get_train_transforms(aug_config)

        dataset = SimpleDataset(
            df=sample_data["df"],
            data_root_dir=sample_data["data_root_dir"],
            target_cols=sample_data["target_cols"],
            phase="fit",
            transforms=transforms,
        )

        shapes = []
        for i in range(len(dataset)):
            inputs, labels = dataset[i]
            shapes.append(inputs["image"].shape)

        # All shapes should be the same
        assert all(s == shapes[0] for s in shapes)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        temp_dir = tempfile.mkdtemp()
        data_root = Path(temp_dir)

        image_dir = data_root / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(image_dir / f"sample_{i}.jpg")

        target_cols = [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g",
        ]
        data = []

        for i in range(3):
            for target_name in target_cols:
                data.append(
                    {
                        "image_path": f"images/sample_{i}.jpg",
                        "target_name": target_name,
                        "target": float(np.random.rand() * 100),
                    }
                )

        df = pd.DataFrame(data)

        return {
            "df": df,
            "data_root_dir": data_root,
            "target_cols": target_cols,
        }


if __name__ == "__main__":
    # Simple test runner
    print("Testing SimpleDataset initialization...")
    # Note: This requires actual test data, so we'll just verify imports
    print("✓ Imports successful")

    print("\nSimpleDataset tests require pytest with fixtures to run properly.")
    print("Run with: pytest test_simple_dataset.py")
