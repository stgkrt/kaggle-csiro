import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.configs import AugmentationConfig


def get_train_transforms(aug_config: AugmentationConfig):
    """Create training data augmentations based on the given configuration."""
    transforms = []
    if aug_config.horizontal_flip:
        transforms.append(A.HorizontalFlip(p=aug_config.hflip_prob))
    if aug_config.vertical_flip:
        transforms.append(A.VerticalFlip(p=aug_config.vflip_prob))
    if aug_config.shadow:
        transforms.append(
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(
                    aug_config.num_shadows_lower,
                    aug_config.num_shadows_upper,
                ),
                shadow_dimension=aug_config.shadow_dimension,
                p=aug_config.shadow_prob,
            )
        )
    if aug_config.brightness_contrast:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(
                    -aug_config.brightness_limit,
                    aug_config.brightness_limit,
                ),
                contrast_limit=(
                    -aug_config.contrast_limit,
                    aug_config.contrast_limit,
                ),
                p=aug_config.brightness_contrast_prob,
            )
        )
    if aug_config.resize:
        transforms.append(
            A.Resize(
                height=aug_config.resize_img_height,
                width=aug_config.resize_img_width,
                p=1.0,
            )
        )

    transforms.append(
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def get_valid_transforms(aug_config: AugmentationConfig):
    """Create validation data augmentations based on the given configuration."""
    transforms = []
    transforms.append(
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    if aug_config.resize:
        transforms.append(
            A.Resize(
                height=aug_config.resize_img_height,
                width=aug_config.resize_img_width,
                p=1.0,
            )
        )
    transforms.append(ToTensorV2())
    return A.Compose(transforms)
