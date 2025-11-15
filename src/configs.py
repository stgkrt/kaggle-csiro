from dataclasses import dataclass
from pathlib import Path


@dataclass
class DirConfig:
    data_dir: Path
    processed_dir: Path
    output_dir: Path


@dataclass
class AugConfig:
    _target_: str
    p: float


@dataclass
class SplitConfig:
    train_num: int
    valid_num: int
    test_num: int


@dataclass
class DatasetConfig:
    _target_: str
    dataset_name: str
    df_path: Path
    batch_size: int
    num_workers: int
    pin_memory: bool
    splits: SplitConfig | None
    train_transforms: list[AugConfig] | None
    valid_transforms: list[AugConfig] | None


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float
    num_warmup_steps: int


@dataclass
class SchedulerConfig:
    mode: str
    factor: float
    patience: int


@dataclass
class LossConfig:
    loss_name: str
    pos_weight: float | None
    target_gesture_dict_path: Path | None
    aux_weight: float | None


@dataclass
class ModelConfig:
    _target_: str
    model_name: str
    pad_len: int
    imu_dim: int
    tof_dim: int
    thm_dim: int
    n_classes: int
    loss_config: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    default_emb_dim: int = 32
    layer_num: int = 5


@dataclass
class EarlyStoppingConfig:
    _target_: str
    monitor: str
    min_delta: float
    patience: int
    mode: str
    strict: bool
    check_finite: bool
    stopping_threshold: float
    divergence_threshold: float
    check_on_train_epoch_end: bool


@dataclass
class ModelcheckpointConfig:
    _target_: str
    dirpath: str
    filename: str
    monitor: str
    verbose: bool
    save_last: bool
    save_top_k: int
    mode: str
    auto_insert_metric_name: bool
    every_n_val_epochs: int
    train_time_interval: int
    every_n_epochs: int
    save_on_train_epoch_end: bool


@dataclass
class ModelSummaryConfig:
    _target_: str
    max_depth: int


@dataclass
class RichProgressbar:
    _target_: str


@dataclass
class Callbacks:
    early_stopping: EarlyStoppingConfig
    modelcheckpoint: ModelcheckpointConfig
    model_summary: ModelSummaryConfig
    rich_progressbar: RichProgressbar


@dataclass
class LoggerConfig:
    _target_: str


@dataclass
class TrainerConfig:
    epochs: int
    accelerator: str
    use_amp: bool
    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int


@dataclass
class TrainConfig:
    exp_name: str
    seed: int
    ckpt_path: str
    dir: DirConfig
    model: ModelConfig
    dataset: DatasetConfig
    trainer: TrainerConfig
    callbacks: Callbacks
    logger: LoggerConfig
