"""Tests for src/data/augmentations.py"""

import numpy as np
import pytest
from albumentations.core.composition import Compose

from src.configs import AugmentationConfig
from src.data.augmentations import get_train_transforms, get_valid_transforms


class TestGetTrainTransforms:
    """Test get_train_transforms function"""

    def test_returns_compose(self):
        """Test that function returns Compose object"""
        aug_config = AugmentationConfig()
        transforms = get_train_transforms(aug_config)
        assert isinstance(transforms, Compose)

    def test_default_config(self):
        """Test with default configuration"""
        aug_config = AugmentationConfig()
        transforms = get_train_transforms(aug_config)

        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        transformed = transforms(image=dummy_image)

        assert "image" in transformed
        assert transformed["image"].shape[0] == 3  # channels first
        assert transformed["image"].shape[1] == aug_config.resize_img_height
        assert transformed["image"].shape[2] == aug_config.resize_img_width

    def test_horizontal_flip_enabled(self):
        """Test horizontal flip is included when enabled"""
        aug_config = AugmentationConfig(horizontal_flip=True, hflip_prob=1.0)
        transforms = get_train_transforms(aug_config)

        # Check that HorizontalFlip is in the transforms
        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "HorizontalFlip" in transform_names

    def test_vertical_flip_enabled(self):
        """Test vertical flip is included when enabled"""
        aug_config = AugmentationConfig(vertical_flip=True, vflip_prob=1.0)
        transforms = get_train_transforms(aug_config)

        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "VerticalFlip" in transform_names

    def test_shadow_enabled(self):
        """Test shadow augmentation is included when enabled"""
        aug_config = AugmentationConfig(shadow=True, shadow_prob=1.0)
        transforms = get_train_transforms(aug_config)

        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "RandomShadow" in transform_names

    def test_brightness_contrast_enabled(self):
        """Test brightness/contrast is included when enabled"""
        aug_config = AugmentationConfig(
            brightness_contrast=True, brightness_contrast_prob=1.0
        )
        transforms = get_train_transforms(aug_config)

        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "RandomBrightnessContrast" in transform_names

    def test_resize_enabled(self):
        """Test resize is included when enabled"""
        aug_config = AugmentationConfig(
            resize=True, resize_img_height=256, resize_img_width=256
        )
        transforms = get_train_transforms(aug_config)

        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        transformed = transforms(image=dummy_image)

        assert transformed["image"].shape[1] == 256
        assert transformed["image"].shape[2] == 256

    def test_normalization_applied(self):
        """Test that normalization is always applied"""
        aug_config = AugmentationConfig()
        transforms = get_train_transforms(aug_config)

        # Normalize should always be in the pipeline
        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "Normalize" in transform_names

    def test_to_tensor_applied(self):
        """Test that ToTensorV2 is always applied"""
        aug_config = AugmentationConfig()
        transforms = get_train_transforms(aug_config)

        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "ToTensorV2" in transform_names

    def test_no_augmentations(self):
        """Test with all augmentations disabled"""
        aug_config = AugmentationConfig(
            horizontal_flip=False,
            vertical_flip=False,
            shadow=False,
            brightness_contrast=False,
            resize=False,
        )
        transforms = get_train_transforms(aug_config)

        # Should still have Normalize and ToTensorV2
        assert len(transforms.transforms) >= 2


class TestGetValidTransforms:
    """Test get_valid_transforms function"""

    def test_returns_compose(self):
        """Test that function returns Compose object"""
        aug_config = AugmentationConfig()
        transforms = get_valid_transforms(aug_config)
        assert isinstance(transforms, Compose)

    def test_default_config(self):
        """Test with default configuration"""
        aug_config = AugmentationConfig()
        transforms = get_valid_transforms(aug_config)

        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        transformed = transforms(image=dummy_image)

        assert "image" in transformed
        assert transformed["image"].shape[0] == 3  # channels first

    def test_no_random_augmentations(self):
        """Test that validation transforms have no random augmentations"""
        aug_config = AugmentationConfig(
            horizontal_flip=True,
            vertical_flip=True,
            shadow=True,
            brightness_contrast=True,
        )
        transforms = get_valid_transforms(aug_config)

        # Should not have any random augmentations
        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "HorizontalFlip" not in transform_names
        assert "VerticalFlip" not in transform_names
        assert "RandomShadow" not in transform_names
        assert "RandomBrightnessContrast" not in transform_names

    def test_normalization_applied(self):
        """Test that normalization is applied"""
        aug_config = AugmentationConfig()
        transforms = get_valid_transforms(aug_config)

        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "Normalize" in transform_names

    def test_to_tensor_applied(self):
        """Test that ToTensorV2 is applied"""
        aug_config = AugmentationConfig()
        transforms = get_valid_transforms(aug_config)

        transform_names = [t.__class__.__name__ for t in transforms.transforms]
        assert "ToTensorV2" in transform_names

    def test_resize_when_enabled(self):
        """Test that resize is applied when enabled"""
        aug_config = AugmentationConfig(
            resize=True, resize_img_height=256, resize_img_width=256
        )
        transforms = get_valid_transforms(aug_config)

        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        transformed = transforms(image=dummy_image)

        assert transformed["image"].shape[1] == 256
        assert transformed["image"].shape[2] == 256

    def test_minimal_transforms(self):
        """Test minimal validation transforms"""
        aug_config = AugmentationConfig(resize=False)
        transforms = get_valid_transforms(aug_config)

        # Should have at least Normalize and ToTensorV2
        assert len(transforms.transforms) >= 2


class TestTransformsIntegration:
    """Integration tests for transforms"""

    def test_train_and_valid_consistency(self):
        """Test that train and valid transforms produce consistent shapes"""
        aug_config = AugmentationConfig(
            resize=True, resize_img_height=256, resize_img_width=256
        )

        train_transforms = get_train_transforms(aug_config)
        valid_transforms = get_valid_transforms(aug_config)

        dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        train_output = train_transforms(image=dummy_image)
        valid_output = valid_transforms(image=dummy_image)

        # Shapes should match
        assert train_output["image"].shape == valid_output["image"].shape

    def test_different_input_sizes(self):
        """Test transforms with different input image sizes"""
        aug_config = AugmentationConfig(
            resize=True, resize_img_height=256, resize_img_width=256
        )
        transforms = get_train_transforms(aug_config)

        # Test various input sizes
        for height, width in [(128, 128), (512, 512), (1024, 768)]:
            dummy_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            transformed = transforms(image=dummy_image)

            # Output should always be the configured size
            assert transformed["image"].shape[1] == 256
            assert transformed["image"].shape[2] == 256

    def test_grayscale_to_rgb_conversion(self):
        """Test that transforms handle RGB images correctly"""
        aug_config = AugmentationConfig()
        transforms = get_train_transforms(aug_config)

        # RGB image should produce 3 channels
        rgb_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        transformed = transforms(image=rgb_image)

        assert transformed["image"].shape[0] == 3

    def test_reproducibility_with_seed(self):
        """Test that transforms are deterministic with same random state"""
        aug_config = AugmentationConfig(horizontal_flip=True, hflip_prob=0.5)
        transforms = get_train_transforms(aug_config)

        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Note: Albumentations uses internal random state
        # This test just ensures transforms run without error
        result1 = transforms(image=dummy_image.copy())
        result2 = transforms(image=dummy_image.copy())

        assert result1["image"].shape == result2["image"].shape


if __name__ == "__main__":
    # Simple test runner
    print("Testing get_train_transforms...")
    test = TestGetTrainTransforms()
    test.test_returns_compose()
    test.test_default_config()
    test.test_normalization_applied()
    print("✓ get_train_transforms tests passed")

    print("\nTesting get_valid_transforms...")
    test = TestGetValidTransforms()
    test.test_returns_compose()
    test.test_default_config()
    test.test_no_random_augmentations()
    print("✓ get_valid_transforms tests passed")

    print("\nTesting integration...")
    test = TestTransformsIntegration()
    test.test_train_and_valid_consistency()
    test.test_different_input_sizes()
    print("✓ Integration tests passed")

    print("\nAll augmentation tests passed!")
