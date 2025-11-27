from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainerConfig(BaseSettings):
    default_root_dir: Path = Path("kaggle/working")
    min_epochs: int = 1
    max_epochs: int = 10
    accelerator: str = "gpu"
    gradient_clip_val: float = 2.0
    gradient_clip_algorithm: str = "norm"
    check_val_every_n_epoch: int = 1
    compile: bool = True
    devices: int = 1
    deterministic: bool = True
    # optimizer scheduler params
    lr: float = 1e-3
    weight_decay: float = 1e-3
    scheduler_eta_min: float = 1e-9
    scheduler_t_mult: int = 1
    ema_decay: float = 0.998  # EMA decay rate
    ema_enable: bool = True  # Whether to enable EMA


class ModelConfig(BaseSettings):
    model_name: str = "simple_model"
    backbone_name: str = "tf_efficientnet_b0"
    pretrained: bool = True

    in_channels: int = 3
    n_classes: int = 5


class LossConfig(BaseSettings):
    loss_name: str = "mse_loss"
    pos_weight: Optional[float] = 10.0


class AugmentationConfig(BaseSettings):
    random_crop: bool = True
    crop_size: int = 128
    crop_prob: float = 0.5
    horizontal_flip: bool = True
    hflip_prob: float = 0.5
    vertical_flip: bool = False
    vflip_prob: float = 0.5
    resize: bool = True
    # resize_img_height: int = 384
    # resize_img_width: int = 384 * 2
    # resize_img_height: int = 256
    # resize_img_width: int = 256
    resize_img_height: int = 512
    resize_img_width: int = 512
    shadow: bool = True
    shadow_roi_start: float = 0.0
    shadow_roi_end: float = 0.5
    num_shadows_lower: int = 1
    num_shadows_upper: int = 2
    shadow_dimension: int = 50
    shadow_prob: float = 0.3
    brightness_contrast: bool = True
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    brightness_contrast_prob: float = 0.3


class DatasetConfig(BaseSettings):
    dataset_name: str = "simple"
    data_root_dir: Path = Path("/kaggle/input/csiro-biomass/")
    df_path: Path = Path("/kaggle/input/csiro-biomass/train.csv")
    # target_cols: List[str] = [
    #     "Dry_Green_g",
    #     "Dry_Dead_g",
    #     "Dry_Clover_g",
    #     "GDM_g",
    #     "Dry_Total_g",
    # ]
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True

    # Mixup configuration
    mixup_prob: float = 0.2
    mixup_alpha: float = 0.2
    mixup_max_len_rate: float = 0.15

    # Augmentation configuration
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)


class SplitConfig(BaseSettings):
    fold: int = 0
    split_dir: Path = Path("/kaggle/working/splits")
    train_ids: List[str] = Field(default_factory=list)
    valid_ids: List[str] = Field(default_factory=list)


class LoggerConfig(BaseSettings):
    project: str = "CSIRO"
    name: Optional[str] = None
    offline: bool = True
    save_dir: Path = Path(".")


class CallbacksConfig(BaseSettings):
    model_checkpoint: bool = True
    early_stopping: bool = False
    lr_monitor: bool = True
    progress_bar: bool = True


class Config(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)
    # Basic settings
    competition_name: str = "CSIRO"
    notes: Optional[str] = None
    seed: int = 42
    exp_name: str = "debug"
    fold: int = 0

    ckpt_path: Optional[Path] = None
    tags: str = "public 0.0"

    # Sub-configurations
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    callbacks: CallbacksConfig = Field(default_factory=CallbacksConfig)


def create_config_from_args() -> Config:
    """Create configuration from command line arguments using pydantic_argparse"""
    # config = ArgumentParser().parse_args()
    train_config = TrainerConfig()
    model_config = ModelConfig()
    loss_config = LossConfig()
    aug_config = AugmentationConfig()
    dataset_config = DatasetConfig()
    split_config = SplitConfig()
    logger_config = LoggerConfig()
    callbacks_config = CallbacksConfig()
    config = Config(
        trainer=train_config,
        model=model_config,
        loss=loss_config,
        augmentation=aug_config,
        dataset=dataset_config,
        split=split_config,
        logger=logger_config,
        callbacks=callbacks_config,
    )
    return config


if __name__ == "__main__":
    config = create_config_from_args()
    print(config)
    print("----- Trainer Config -----")
    print(config.trainer)
    print("----- Model Config -----")
    print(config.model)
    print("----- Dataset Config -----")
    print(config.dataset)
    print("----- Loss Config -----")
    print(config.loss)
    print("----- Augmentation Config -----")
    print(config.augmentation)
    print("----- Split Config -----")
    print(config.split)
    print("----- Logger Config -----")
    print(config.logger)
    print("----- Callbacks Config -----")
    print(config.callbacks)
