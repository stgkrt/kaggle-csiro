"""
Dataclass-based configuration for training
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore


@dataclass
class TrainerConfig:
    """Trainer configuration"""

    min_epochs: int = 1
    max_epochs: int = 100
    accelerator: str = "gpu"
    gradient_clip_val: float = 2.0
    gradient_clip_algorithm: str = "norm"
    check_val_every_n_epoch: int = 1
    devices: int = 1
    deterministic: bool = True
    default_root_dir: Path = Path("kaggle/working")


@dataclass
class ArgParseModelConfig:
    """Model configuration"""

    model_name: str = "public_model"
    pad_len: int = 127
    imu_dim: int = 11
    tof_dim: int = 25
    thm_dim: int = 5
    n_classes: int = 18

    # Simple CNN Model specific parameters
    default_emb_dim: int = 32
    layer_num: int = 5

    # Optimizer configuration
    lr: float = 2e-4
    weight_decay: float = 1e-3

    # Scheduler configuration
    scheduler_t_0: int = 100  # T_0 for CosineAnnealingWarmRestarts
    scheduler_t_mult: int = 1  # T_mult for CosineAnnealingWarmRestarts
    scheduler_eta_min: float = 1e-9  # eta_min for CosineAnnealingWarmRestarts

    # EMA configuration
    ema_decay: float = 0.9999  # EMA decay rate
    ema_enable: bool = True  # Whether to enable EMA

    # Loss configuration
    loss_name: str = "weighted_cross_entropy"
    pos_weight: float | None = 10.0
    target_gesture_dict_path: Path | None = Path(
        "/kaggle/working/encoders/inverse_gesture_dict.pkl"
    )
    aux_weight: float | None = 0.2
    # Metrics configuration
    inverse_gesture_dict_path: Path | None = Path(
        "/kaggle/working/encoders/inverse_gesture_dict.pkl"
    )
    smoothing: float | None = 0.1  # Smoothing factor for SoftTargetBCELoss


@dataclass
class AugmentationConfig:
    """Augmentation configuration"""

    time_stretch_range: float = 0.2
    time_shift_range: float = 0.1
    noise_std: float = 0.02
    magnitude_scale_range: float = 0.1
    rotation_angle_range: float = 0.1
    mask_ratio: float = 0.1
    freq_filter_range_low: float = 0.1
    freq_filter_range_high: float = 0.9
    aug_prob: float = 0.8
    enable_augmentations: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration"""

    dataset_name: str = "public"
    df_path: Path = Path("/kaggle/working/processed/processed_df.csv")
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True

    # Mixup configuration
    mixup_prob: float = 0.2
    mixup_alpha: float = 0.2
    mixup_max_len_rate: float = 0.15

    # Augmentation configuration
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class SplitConfig:
    """Split configuration for cross-validation"""

    fold: int = 0
    # split_dir: Path = Path("/kaggle/working/splits_sequence_group")
    # split_dir: Path = Path("/kaggle/working/splits")
    split_dir: Path = Path("/kaggle/working/splits_public")
    train_ids: List[str] = field(default_factory=list)
    valid_ids: List[str] = field(default_factory=list)


@dataclass
class LoggerConfig:
    """Logger configuration"""

    project: str = "CMI2025"
    name: Optional[str] = None
    offline: bool = True
    save_dir: Path = Path(".")


@dataclass
class CallbacksConfig:
    """Callbacks configuration"""

    model_checkpoint: bool = True
    early_stopping: bool = False
    lr_monitor: bool = True
    progress_bar: bool = True


@dataclass
class Config:
    """Main configuration class"""

    # Basic settings
    competition_name: str = "CMI2025"
    notes: Optional[str] = None
    seed: int = 42
    exp_name: str = "default"
    fold: int = 0
    preprocessed_img_size: int = 128
    img_size: int = 128
    in_channels: int = 1
    ckpt_path: Optional[Path] = None
    tags: str = "public 0.0"

    # Sub-configurations
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: ArgParseModelConfig = field(default_factory=ArgParseModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataclass_configs(config_dir: str) -> Config:
    """Load configuration from separate dataclass files"""
    config_dir_path = Path(config_dir) / "dataclasses"

    # Load each dataclass config
    main_config_path = config_dir_path / "main_config.json"
    trainer_config_path = config_dir_path / "trainer_config.json"
    model_config_path = config_dir_path / "model_config.json"
    dataset_config_path = config_dir_path / "dataset_config.json"
    split_config_path = config_dir_path / "split_config.json"
    logger_config_path = config_dir_path / "logger_config.json"
    callbacks_config_path = config_dir_path / "callbacks_config.json"

    # Load main config
    with open(main_config_path, "r") as f:
        main_data = json.load(f)

    # Load trainer config
    with open(trainer_config_path, "r") as f:
        trainer_data = json.load(f)

    # Load model config
    with open(model_config_path, "r") as f:
        model_data = json.load(f)

    # Load dataset config
    with open(dataset_config_path, "r") as f:
        dataset_data = json.load(f)

    # Load split config
    with open(split_config_path, "r") as f:
        split_data = json.load(f)

    # Load logger config
    with open(logger_config_path, "r") as f:
        logger_data = json.load(f)

    # Load callbacks config
    with open(callbacks_config_path, "r") as f:
        callbacks_data = json.load(f)

    # Create config objects
    config = Config()

    # Update main config
    config.competition_name = main_data.get("competition_name", config.competition_name)
    config.notes = main_data.get("notes", config.notes)
    config.seed = main_data.get("seed", config.seed)
    config.exp_name = main_data.get("exp_name", config.exp_name)
    config.fold = main_data.get("fold", config.fold)
    config.preprocessed_img_size = main_data.get(
        "preprocessed_img_size", config.preprocessed_img_size
    )
    config.img_size = main_data.get("img_size", config.img_size)
    config.in_channels = main_data.get("in_channels", config.in_channels)
    config.ckpt_path = (
        Path(main_data["ckpt_path"]) if main_data.get("ckpt_path") else None
    )
    config.tags = main_data.get("tags", config.tags)

    # Update trainer config
    for key, value in trainer_data.items():
        if hasattr(config.trainer, key):
            if key == "default_root_dir":
                setattr(config.trainer, key, Path(value))
            else:
                setattr(config.trainer, key, value)

    # Update model config
    for key, value in model_data.items():
        if hasattr(config.model, key):
            if key in ["target_gesture_dict_path", "inverse_gesture_dict_path"]:
                setattr(config.model, key, Path(value))
            else:
                setattr(config.model, key, value)

    # Update dataset config
    for key, value in dataset_data.items():
        if hasattr(config.dataset, key):
            if key == "df_path":
                setattr(config.dataset, key, Path(value))
            else:
                setattr(config.dataset, key, value)

    # Update split config
    for key, value in split_data.items():
        if hasattr(config.split, key):
            if key == "split_dir":
                setattr(config.split, key, Path(value))
            else:
                setattr(config.split, key, value)

    # Update logger config
    for key, value in logger_data.items():
        if hasattr(config.logger, key):
            if key == "save_dir":
                setattr(config.logger, key, Path(value))
            else:
                setattr(config.logger, key, value)

    # Update callbacks config
    for key, value in callbacks_data.items():
        if hasattr(config.callbacks, key):
            setattr(config.callbacks, key, value)

    return config


def save_dataclass_config(dataclass_obj, config_dir: str, config_name: str) -> str:
    """Save a single dataclass configuration to JSON file"""
    import json
    from pathlib import Path

    config_dir_path = Path(config_dir) / "dataclasses"
    config_dir_path.mkdir(parents=True, exist_ok=True)

    # Convert to serializable dict
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

    serializable_data = to_serializable_dict(dataclass_obj)

    config_path = config_dir_path / f"{config_name}.json"
    with open(config_path, "w") as f:
        json.dump(serializable_data, f, indent=2, default=str)

    return str(config_path)


def create_config_from_args() -> Config:
    """Create configuration from command line arguments"""
    parser = argparse.ArgumentParser(description="Training script")

    # Basic arguments
    parser.add_argument(
        "--exp_name", type=str, default="default", help="Experiment name"
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Notes for the experiment (optional, for logging)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="public 0.0",
        help="Tags for the experiment (space-separated)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")

    # Trainer arguments
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument(
        "--min_epochs", type=int, default=1, help="Minimum number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")

    # Scheduler arguments
    parser.add_argument(
        "--scheduler_t_mult",
        type=int,
        default=1,
        help="T_mult parameter for CosineAnnealingWarmRestarts",
    )
    parser.add_argument(
        "--scheduler_eta_min",
        type=float,
        default=1e-9,
        help="eta_min parameter for CosineAnnealingWarmRestarts",
    )

    # EMA arguments
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.998,
        help="EMA decay rate",
    )
    parser.add_argument(
        "--ema_enable",
        action="store_true",
        default=True,
        help="Enable EMA (default: True)",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Disable EMA",
    )

    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="Accelerator type"
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")

    # Model arguments
    parser.add_argument(
        "--model_name", type=str, default="public_model", help="Model name"
    )
    parser.add_argument("--n_classes", type=int, default=18, help="Number of classes")
    parser.add_argument(
        "--imu_dim", type=int, default=29, help="Number of IMU features"
    )
    parser.add_argument(
        "--tof_dim", type=int, default=25, help="Number of ToF features"
    )
    parser.add_argument(
        "--thm_dim", type=int, default=5, help="Number of thermal features"
    )
    parser.add_argument(
        "--pad_len", type=int, default=127, help="Padding length for sequences"
    )

    # Simple CNN Model arguments
    parser.add_argument(
        "--default_emb_dim",
        type=int,
        default=32,
        help="Default embedding dimension for simple CNN model",
    )
    parser.add_argument(
        "--layer_num", type=int, default=5, help="Number of layers for simple CNN model"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="public",
        help="Dataset name",
    )
    parser.add_argument(
        "--df_path",
        type=str,
        default="/kaggle/working/processed/processed_df.csv",
        help="Dataset path",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for dataloader"
    )

    # Loss arguments
    parser.add_argument(
        "--loss_name",
        type=str,
        default="weighted_cross_entropy",
        help="Loss function name",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Positive weight for loss function",
    )
    parser.add_argument(
        "--aux_weight",
        type=float,
        default=0.2,
        help="Auxiliary loss weight",
    )
    parser.add_argument(
        "--target_gesture_dict_path",
        type=str,
        default="/kaggle/working/encoders/inverse_gesture_dict.pkl",
        help="Path to target gesture dictionary",
    )
    parser.add_argument(
        "--inverse_gesture_dict_path",
        type=str,
        default="/kaggle/working/encoders/inverse_gesture_dict.pkl",
        help="Path to inverse gesture dictionary",
    )

    # Mixup arguments
    parser.add_argument(
        "--mixup_prob", type=float, default=0.2, help="Probability of applying mixup"
    )
    parser.add_argument(
        "--mixup_alpha", type=float, default=0.2, help="Alpha parameter for mixup"
    )
    parser.add_argument(
        "--mixup_max_len_rate",
        type=float,
        default=0.3,
        help="Maximum length rate for mixup",
    )

    # Augmentation arguments
    parser.add_argument(
        "--enable_augmentations",
        action="store_true",
        default=True,
        help="Enable data augmentations (default: True)",
    )
    parser.add_argument(
        "--no_augmentations",
        action="store_true",
        help="Disable data augmentations",
    )
    parser.add_argument(
        "--aug_prob",
        type=float,
        default=0.5,
        help="Overall augmentation probability",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.02,
        help="Standard deviation for noise augmentation",
    )
    parser.add_argument(
        "--time_shift_range",
        type=float,
        default=0.1,
        help="Time shift range for time shifting augmentation",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.1,
        help="Mask ratio for time masking augmentation",
    )

    # Logger arguments
    parser.add_argument(
        "--project", type=str, default="CMI2025", help="Project name for logging"
    )
    parser.add_argument("--offline", action="store_true", help="Use offline logging")

    args = parser.parse_args()

    # Create config object
    config = Config()

    # Update basic settings
    config.exp_name = args.exp_name
    config.notes = args.notes
    config.tags = args.tags
    config.seed = args.seed
    config.fold = args.fold

    # Update trainer config
    config.trainer.max_epochs = args.epochs
    config.trainer.min_epochs = args.min_epochs
    config.trainer.accelerator = args.accelerator
    config.trainer.devices = args.devices

    # Update model config
    config.model.model_name = args.model_name
    config.model.n_classes = args.n_classes
    config.model.imu_dim = args.imu_dim
    config.model.tof_dim = args.tof_dim
    config.model.thm_dim = args.thm_dim
    config.model.pad_len = args.pad_len
    config.model.default_emb_dim = args.default_emb_dim
    config.model.layer_num = args.layer_num
    config.model.lr = args.lr
    config.model.weight_decay = args.weight_decay
    config.model.scheduler_t_0 = args.epochs
    config.model.scheduler_t_mult = args.scheduler_t_mult
    config.model.scheduler_eta_min = args.scheduler_eta_min
    config.model.ema_decay = args.ema_decay
    config.model.ema_enable = args.ema_enable and not args.no_ema
    config.model.loss_name = args.loss_name
    config.model.pos_weight = args.pos_weight
    config.model.aux_weight = args.aux_weight
    config.model.target_gesture_dict_path = Path(args.target_gesture_dict_path)
    config.model.inverse_gesture_dict_path = Path(args.inverse_gesture_dict_path)

    # Update dataset config
    config.dataset.dataset_name = args.dataset_name
    config.dataset.df_path = Path(args.df_path)
    config.dataset.batch_size = args.batch_size
    config.dataset.num_workers = args.num_workers
    config.dataset.mixup_prob = args.mixup_prob
    config.dataset.mixup_alpha = args.mixup_alpha
    config.dataset.mixup_max_len_rate = args.mixup_max_len_rate

    # Update augmentation config
    config.dataset.augmentation.enable_augmentations = (
        args.enable_augmentations and not args.no_augmentations
    )
    config.dataset.augmentation.aug_prob = args.aug_prob
    config.dataset.augmentation.noise_std = args.noise_std
    config.dataset.augmentation.time_shift_range = args.time_shift_range
    config.dataset.augmentation.mask_ratio = args.mask_ratio

    # Update logger config
    config.logger.project = args.project
    config.logger.offline = args.offline
    config.logger.name = args.exp_name

    return config


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """Merge configuration with override dictionary"""
    # Simple implementation - can be extended for nested updates
    for key, value in override_dict.items():
        if hasattr(base_config, key):
            if isinstance(getattr(base_config, key), TrainerConfig):
                for sub_key, sub_value in value.items():
                    setattr(getattr(base_config, key), sub_key, sub_value)
            elif isinstance(getattr(base_config, key), ArgParseModelConfig):
                for sub_key, sub_value in value.items():
                    setattr(getattr(base_config, key), sub_key, sub_value)
            elif isinstance(getattr(base_config, key), DatasetConfig):
                for sub_key, sub_value in value.items():
                    setattr(getattr(base_config, key), sub_key, sub_value)
            else:
                setattr(base_config, key, value)
    return base_config
