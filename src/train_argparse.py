import json
import os
import subprocess
from pathlib import Path
from typing import Any, List

import pytorch_lightning as L
import torch.optim as optim
import yaml  # type: ignore
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.config_dataclass import Config, create_config_from_args
from src.configs import LossConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from src.data.augmentations import TimeSeriesAugmentation
from src.data.data_module import DataModule
from src.log_utils.pylogger import RankedLogger
from src.metrics.competition_metrics import CompetitionMetrics
from src.model.architectures.model_architectures import ModelArchitectures
from src.model.losses import LossModule
from src.model.model_module import ModelModule

log = RankedLogger(__name__, rank_zero_only=True)


def load_feature_columns() -> dict[str, list[str]]:
    """Load feature column names from YAML files"""
    features_dir = Path("/kaggle/working/features")

    # Load IMU columns
    with open(features_dir / "imu_cols.yaml", "r") as f:
        imu_cols = yaml.safe_load(f)["imu_cols"]

    # Load thermal columns
    with open(features_dir / "thm_cols.yaml", "r") as f:
        thm_cols = yaml.safe_load(f)["thm_cols"]

    # Load ToF columns
    with open(features_dir / "tof_agg_cols.yaml", "r") as f:
        tof_cols = yaml.safe_load(f)["tof_agg_cols"]

    feature_columns = {
        "imu_cols": imu_cols,
        "thm_cols": thm_cols,
        "tof_cols": tof_cols,
    }

    log.info("Loaded feature columns:")
    log.info(f"  IMU features: {len(imu_cols)}")
    log.info(f"  Thermal features: {len(thm_cols)}")
    log.info(f"  ToF features: {len(tof_cols)}")

    return feature_columns


def create_augmentations(
    config: Config,
) -> tuple[TimeSeriesAugmentation | None, TimeSeriesAugmentation | None]:
    """Create augmentations for training and validation"""
    train_augmentations = None
    valid_augmentations = None

    if config.dataset.augmentation.enable_augmentations:
        log.info("Creating augmentations...")
        time_stretch_range = (
            1.0 - config.dataset.augmentation.time_stretch_range,
            1.0 + config.dataset.augmentation.time_stretch_range,
        )
        magnitude_scale_range = (
            1.0 - config.dataset.augmentation.magnitude_scale_range,
            1.0 + config.dataset.augmentation.magnitude_scale_range,
        )
        freq_filter_range = (
            config.dataset.augmentation.freq_filter_range_low,
            config.dataset.augmentation.freq_filter_range_high,
        )
        train_augmentations = TimeSeriesAugmentation(
            time_stretch_range=time_stretch_range,
            time_shift_range=config.dataset.augmentation.time_shift_range,
            noise_std=config.dataset.augmentation.noise_std,
            magnitude_scale_range=magnitude_scale_range,
            rotation_angle_range=config.dataset.augmentation.rotation_angle_range,
            mask_ratio=config.dataset.augmentation.mask_ratio,
            freq_filter_range=freq_filter_range,
            dropout_ratio=0.3,
            aug_prob=config.dataset.augmentation.aug_prob,
            aug_dropout_prob=0.2,
        )
        log.info(
            f"Training augmentations created with "
            f"aug_prob={config.dataset.augmentation.aug_prob}"
        )
        log.info(
            f"Augmentation settings: "
            f"noise_std={config.dataset.augmentation.noise_std}, "
            f"time_shift_range={config.dataset.augmentation.time_shift_range}, "
            f"mask_ratio={config.dataset.augmentation.mask_ratio}"
        )

        # Validation augmentations (usually no augmentation or milder)
        valid_augmentations = None
        log.info("Validation augmentations: None (no augmentation during validation)")
    else:
        log.info("Augmentations disabled")

    return train_augmentations, valid_augmentations


def setup_directories(config: Config) -> None:
    """Setup output directories"""
    # Create experiment directory
    exp_dir = Path(f"/kaggle/working/{config.exp_name}/fold_{config.fold}")

    os.makedirs(exp_dir, exist_ok=True)
    config.trainer.default_root_dir = exp_dir

    # Create config directory
    config_dir = exp_dir / "configs"
    os.makedirs(config_dir, exist_ok=True)

    log.info(f"Experiment directory: {exp_dir}")
    log.info(f"Config directory: {config_dir}")


def save_config(config: Config, feature_columns: dict[str, list[str]]) -> str:
    """Save configuration to files, separated by dataclass"""
    config_dir = Path(config.trainer.default_root_dir)

    # Create subdirectories for each config type
    dataclass_dir = config_dir / "dataclasses"
    os.makedirs(dataclass_dir, exist_ok=True)

    # Helper function to convert dataclass to serializable dict
    def to_serializable_dict(obj):
        if hasattr(obj, "__dict__"):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, Path):
                    result[key] = str(value)
                elif hasattr(value, "__dict__"):
                    result[key] = to_serializable_dict(value)
                else:
                    result[key] = value
            return result
        else:
            return str(obj)

    # Save each dataclass separately
    saved_files = []

    # 1. Main Config (basic settings only)
    main_config = {
        "competition_name": config.competition_name,
        "notes": config.notes,
        "seed": config.seed,
        "exp_name": config.exp_name,
        "fold": config.fold,
        "preprocessed_img_size": config.preprocessed_img_size,
        "img_size": config.img_size,
        "in_channels": config.in_channels,
        "ckpt_path": str(config.ckpt_path) if config.ckpt_path else None,
        "tags": config.tags,
    }
    main_path = dataclass_dir / "main_config.json"
    with open(main_path, "w") as f:
        json.dump(main_config, f, indent=2, default=str)
    saved_files.append(f"Main config: {main_path}")

    # 2. Trainer Config
    trainer_config = to_serializable_dict(config.trainer)
    trainer_path = dataclass_dir / "trainer_config.json"
    with open(trainer_path, "w") as f:
        json.dump(trainer_config, f, indent=2, default=str)
    saved_files.append(f"Trainer config: {trainer_path}")

    # 3. Model Config
    model_config = to_serializable_dict(config.model)
    model_path = dataclass_dir / "model_config.json"
    with open(model_path, "w") as f:
        json.dump(model_config, f, indent=2, default=str)
    saved_files.append(f"Model config: {model_path}")

    # 4. Dataset Config
    dataset_config = to_serializable_dict(config.dataset)
    dataset_path = dataclass_dir / "dataset_config.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset_config, f, indent=2, default=str)
    saved_files.append(f"Dataset config: {dataset_path}")

    # 5. Split Config
    split_config = to_serializable_dict(config.split)
    split_path = dataclass_dir / "split_config.json"
    with open(split_path, "w") as f:
        json.dump(split_config, f, indent=2, default=str)
    saved_files.append(f"Split config: {split_path}")

    # 6. Logger Config
    logger_config = to_serializable_dict(config.logger)
    logger_path = dataclass_dir / "logger_config.json"
    with open(logger_path, "w") as f:
        json.dump(logger_config, f, indent=2, default=str)
    saved_files.append(f"Logger config: {logger_path}")

    # 7. Callbacks Config
    callbacks_config = to_serializable_dict(config.callbacks)
    callbacks_path = dataclass_dir / "callbacks_config.json"
    with open(callbacks_path, "w") as f:
        json.dump(callbacks_config, f, indent=2, default=str)
    saved_files.append(f"Callbacks config: {callbacks_path}")

    # 8. Feature Columns
    feature_columns_path = dataclass_dir / "feature_columns.json"
    with open(feature_columns_path, "w") as f:
        json.dump(feature_columns, f, indent=2, default=str)
    saved_files.append(f"Feature columns: {feature_columns_path}")

    # Also save complete config as before for backward compatibility
    serializable_config = to_serializable_dict(config)

    # Save complete config as YAML
    complete_yaml_path = config_dir / "config.yaml"
    with open(complete_yaml_path, "w") as f:
        yaml.dump(serializable_config, f, default_flow_style=False, allow_unicode=True)
    saved_files.append(f"Complete config (YAML): {complete_yaml_path}")

    # Also save feature columns as YAML for easy reading
    feature_columns_yaml_path = config_dir / "feature_columns.yaml"
    with open(feature_columns_yaml_path, "w") as f:
        yaml.dump(feature_columns, f, default_flow_style=False, allow_unicode=True)
    saved_files.append(f"Feature columns (YAML): {feature_columns_yaml_path}")

    # Create summary file with all paths
    summary_path = config_dir / "config_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Configuration files saved\n")
        f.write("=" * 30 + "\n\n")
        for file_info in saved_files:
            f.write(f"{file_info}\n")

    log.info("Configuration saved by dataclass:")
    for file_info in saved_files:
        log.info(f"  {file_info}")
    log.info(f"Summary: {summary_path}")

    return str(complete_yaml_path)


def setup_callbacks(config: Config) -> list[Any]:
    """Setup training callbacks"""
    callbacks = []

    if config.callbacks.model_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{config.trainer.default_root_dir}/checkpoints",
            filename="best-{epoch}-{val_loss:.3f}",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    if config.callbacks.early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            verbose=True,
        )
        callbacks.append(early_stopping)

    if config.callbacks.lr_monitor:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    if config.callbacks.progress_bar:
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)

    return callbacks


def setup_logger(config: Config) -> list[WandbLogger, CSVLogger]:  # type: ignore
    """Setup wandb logger"""
    run_name = f"{config.exp_name}_fold{config.fold}"
    logger = [
        WandbLogger(
            project=config.logger.project,
            name=run_name,
            offline=config.logger.offline,
            tags=config.tags.split(),
            group=config.exp_name,
        ),
        CSVLogger(
            save_dir=config.trainer.default_root_dir,
            name=f"{config.exp_name}_fold{config.fold}",
        ),
    ]
    return logger


def create_model(config: Config) -> ModelModule:
    """Create model instance"""
    # Create loss config
    loss_config = LossConfig(
        loss_name=config.model.loss_name,
        pos_weight=config.model.pos_weight,
        target_gesture_dict_path=config.model.target_gesture_dict_path,
        aux_weight=config.model.aux_weight,
    )

    # Create optimizer config
    optimizer_config = OptimizerConfig(
        lr=config.model.lr,
        weight_decay=config.model.weight_decay,
        num_warmup_steps=0,
    )

    # Create scheduler config
    scheduler_config = SchedulerConfig(
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Create model config using existing ModelConfig dataclass
    model_config = ModelConfig(
        _target_="",  # Not used in this context
        model_name=config.model.model_name,
        pad_len=config.model.pad_len,
        imu_dim=config.model.imu_dim,
        tof_dim=config.model.tof_dim,
        thm_dim=config.model.thm_dim,
        n_classes=config.model.n_classes,
        loss_config=loss_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        default_emb_dim=config.model.default_emb_dim,
        layer_num=config.model.layer_num,
    )

    # Create model architectures with proper ModelConfig object
    model_architectures = ModelArchitectures(model_config)

    # Create loss module
    criterion = LossModule(loss_config)  # type: ignore

    # Create metrics
    metrics = CompetitionMetrics(
        inverse_gesture_dict_path=config.model.inverse_gesture_dict_path  # type: ignore
    )
    if config.exp_name.startswith("debug"):
        config.trainer.max_epochs = 3

    # Create model
    model = ModelModule(
        model_architectures=model_architectures,
        criterion=criterion,
        metrics=metrics,
        compile=False,
        oof_dir=config.trainer.default_root_dir,
        lr=config.model.lr,
        weight_decay=config.model.weight_decay,
        max_epochs=config.trainer.max_epochs,
        scheduler_t_0=config.trainer.max_epochs,
        scheduler_t_mult=config.model.scheduler_t_mult,
        scheduler_eta_min=config.model.scheduler_eta_min,
        ema_decay=config.model.ema_decay,
        ema_enable=config.model.ema_enable,
    )
    return model


def create_datamodule(
    config: Config,
    feature_columns: dict[str, list[str]],
    train_augmentations: TimeSeriesAugmentation | None = None,
    valid_augmentations: TimeSeriesAugmentation | None = None,
) -> DataModule:
    """Create data module instance"""
    fold = config.fold
    if fold is not None:
        log.info(f"Using fold: {fold}")
    else:
        log.info("No fold specified, using default split")
    # get splits
    with open(config.split.split_dir / f"fold_{fold}" / "train.yaml", "r") as f:
        config.split.train_ids = yaml.safe_load(f)
    with open(config.split.split_dir / f"fold_{fold}" / "valid.yaml", "r") as f:
        config.split.valid_ids = yaml.safe_load(f)
    print(f"train fold path: {config.split.split_dir / f'fold_{fold}' / 'train.yaml'}")
    print(f"valid fold path: {config.split.split_dir / f'fold_{fold}' / 'valid.yaml'}")
    datamodule = DataModule(
        dataset_name=config.dataset.dataset_name,
        df_path=config.dataset.df_path,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        splits=config.split,
        train_transforms=train_augmentations,
        valid_transforms=valid_augmentations,
        mixup_prob=config.dataset.mixup_prob,
        mixup_alpha=config.dataset.mixup_alpha,
        mixup_max_len_rate=config.dataset.mixup_max_len_rate,
        imu_cols=feature_columns["imu_cols"],
        thm_cols=feature_columns["thm_cols"],
        tof_cols=feature_columns["tof_cols"],
        max_epoch=config.trainer.max_epochs,
    )
    return datamodule


def create_trainer(config: Config, callbacks: list, logger: WandbLogger) -> L.Trainer:
    """Create trainer instance"""
    trainer = L.Trainer(
        default_root_dir=config.trainer.default_root_dir,
        min_epochs=config.trainer.min_epochs,
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        gradient_clip_val=config.trainer.gradient_clip_val,
        gradient_clip_algorithm=config.trainer.gradient_clip_algorithm,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        deterministic=config.trainer.deterministic,
        callbacks=callbacks,
        logger=logger,
        reload_dataloaders_every_n_epochs=1,
    )
    return trainer


def run_train(config: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run training"""
    log.info("Starting training")

    # Set seed
    log.info(f"Setting seed: {config.seed}")
    L.seed_everything(config.seed)

    # Load feature columns
    log.info("Loading feature columns...")
    feature_columns = load_feature_columns()

    # Create augmentations
    train_augmentations, valid_augmentations = create_augmentations(config)

    # Setup directories
    setup_directories(config)

    # Save configuration
    config_path = save_config(config, feature_columns)

    # Create components
    log.info("Creating model...")
    model = create_model(config)

    log.info("Creating data module...")
    datamodule = create_datamodule(
        config, feature_columns, train_augmentations, valid_augmentations
    )

    log.info("Setting up callbacks...")
    callbacks = setup_callbacks(config)

    log.info("Setting up logger...")
    logger = setup_logger(config)

    log.info("Creating trainer...")
    trainer = create_trainer(config, callbacks, logger)

    # Log hyperparameters
    hyperparams = config.__dict__.copy()
    hyperparams["feature_columns"] = feature_columns
    for logger_ in logger:
        logger_.log_hyperparams(hyperparams)

    # Start training
    log.info("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    # Prepare return values
    object_dict = {
        "cfg": config,
        "model": model,
        "trainer": trainer,
        "config_path": config_path,
        "feature_columns": feature_columns,
    }

    metrics_dict = {**train_metrics}
    return metrics_dict, object_dict


def main() -> None:
    """Main function"""
    # Create config from command line arguments
    config = create_config_from_args()

    log.info(f"Configuration: {config}")

    # Run training
    metrics_dict, object_dict = run_train(config)

    log.info(f"Training completed. Metrics: {metrics_dict}")

    # Get config path from object_dict
    config_path = object_dict.get("config_path", "Unknown")
    log.info(f"Configuration saved to {config_path}")

    # Sync wandb logs
    try:
        subprocess.run(["wandb", "sync", "--sync-all"], check=False)
    except Exception as e:
        log.warning(f"Failed to sync wandb logs: {e}")


if __name__ == "__main__":
    main()
