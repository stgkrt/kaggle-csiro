import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytorch_lightning as L
import yaml  # type: ignore
from albumentations.core.composition import Compose
from configs import Config, create_config_from_args
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from src.data.augmentations import (
    get_train_transforms,
    get_valid_transforms,
)
from src.data.data_module import DataModule
from src.log_utils.pylogger import RankedLogger
from src.metrics.competition_metrics import CompetitionMetrics
from src.model.architectures.model_architectures import ModelArchitectures
from src.model.losses import LossModule
from src.model.model_module import ModelModule

log = RankedLogger(__name__, rank_zero_only=True)


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


def save_config(config: Config) -> str:
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

    # Also save complete config as before for backward compatibility
    serializable_config = to_serializable_dict(config)

    # Save complete config as YAML
    complete_yaml_path = config_dir / "config.yaml"
    with open(complete_yaml_path, "w") as f:
        yaml.dump(serializable_config, f, default_flow_style=False, allow_unicode=True)
    saved_files.append(f"Complete config (YAML): {complete_yaml_path}")

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
    callbacks: list[Callback] = []

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
    config_dict = config.__dict__.copy()
    logger = [
        WandbLogger(
            project=config.logger.project,
            name=run_name,
            offline=config.logger.offline,
            tags=config.tags.split(),
            notes=config.notes,
            group=config.exp_name,
            config=config_dict,
            entity=config.logger.entity,  # team名を入れる
        ),
        CSVLogger(
            save_dir=config.trainer.default_root_dir,
            name=f"{config.exp_name}_fold{config.fold}",
        ),
    ]
    return logger


def create_model(config: Config, valid_df: pd.DataFrame) -> ModelModule:
    """Create model instance"""
    # Create loss config
    # Create model architectures with proper ModelConfig object
    model_architectures = ModelArchitectures(config.model)

    # Create loss module
    criterion = LossModule(config.loss)

    # Create metrics
    metrics = CompetitionMetrics()
    if config.exp_name.startswith("debug"):
        config.trainer.max_epochs = 3

    # Create model
    model = ModelModule(
        model_architectures=model_architectures,
        criterion=criterion,
        metrics=metrics,
        target_cols=config.dataset.target_cols,
        compile=config.trainer.compile,
        valid_df=valid_df,
        oof_dir=config.trainer.default_root_dir,
        lr=config.trainer.lr,
        weight_decay=config.trainer.weight_decay,
        max_epochs=config.trainer.max_epochs,
        scheduler_t_0=config.trainer.max_epochs,
        scheduler_t_mult=config.trainer.scheduler_t_mult,
        scheduler_eta_min=config.trainer.scheduler_eta_min,
        ema_decay=config.trainer.ema_decay,
        ema_enable=config.trainer.ema_enable,
    )
    return model


def create_datamodule(
    config: Config,
    train_augmentations: Compose | None = None,
    valid_augmentations: Compose | None = None,
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
        data_root_dir=config.dataset.data_root_dir,
        target_cols=config.dataset.target_cols,
        num_workers=config.dataset.num_workers,
        batch_size=config.dataset.batch_size,
        pin_memory=config.dataset.pin_memory,
        train_ids=config.split.train_ids,
        valid_ids=config.split.valid_ids,
        train_transforms=train_augmentations,
        valid_transforms=valid_augmentations,
        mixup_prob=config.dataset.mixup_prob,
        mixup_alpha=config.dataset.mixup_alpha,
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
        log_every_n_steps=10,
    )
    return trainer


def load_valid_df(config: Config) -> pd.DataFrame:
    """Load validation dataframe"""
    df = pd.read_csv(config.dataset.df_path)
    fold = config.fold
    splits_dir = config.split.split_dir
    valid_ids_path = splits_dir / f"fold_{fold}" / "valid.yaml"
    with open(valid_ids_path, "r") as f:
        valid_ids = yaml.safe_load(f)
    valid_df = df[df["sample_id"].isin(valid_ids)].reset_index(drop=True)
    return valid_df


def run_train(config: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run training"""
    log.info("Starting training")

    # Set seed
    log.info(f"Setting seed: {config.seed}")
    L.seed_everything(config.seed, workers=True)

    # Setup directories
    setup_directories(config)

    # Create components
    log.info("Creating model...")
    valid_df = load_valid_df(config)
    model = create_model(config, valid_df=valid_df)

    log.info("Creating data module...")
    train_augmentations = get_train_transforms(config.dataset.augmentation)
    valid_augmentations = get_valid_transforms(config.dataset.augmentation)
    datamodule = create_datamodule(config, train_augmentations, valid_augmentations)

    log.info("Setting up callbacks...")
    callbacks = setup_callbacks(config)

    log.info("Setting up logger...")
    logger = setup_logger(config)

    log.info("Creating trainer...")
    trainer = create_trainer(config, callbacks, logger)

    # Log hyperparameters
    hyperparams = config.__dict__.copy()
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
    }

    metrics_dict = {**train_metrics}
    return metrics_dict, object_dict


def main() -> None:
    """Main function"""
    # Create config from command line arguments
    config = create_config_from_args()
    # configをyamlとして保存
    setup_directories(config)
    save_config(config)

    log.info(f"Configuration: {config}")

    # Run training
    metrics_dict, object_dict = run_train(config)

    log.info(f"Training completed. Metrics: {metrics_dict}")

    # Sync wandb logs
    try:
        subprocess.run(["wandb", "sync", "--sync-all"], check=False)
        log.info("Wandb logs synced successfully.")
    except Exception as e:
        log.warning(f"Failed to sync wandb logs: {e}")


if __name__ == "__main__":
    main()
