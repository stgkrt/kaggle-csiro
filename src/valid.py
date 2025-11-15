import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import yaml  # type: ignore
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.configs import ModelConfig
from src.data.augment_dataset import AugmentedDataset
from src.data.dataset_process import get_labels
from src.metrics.competition_metrics import CompetitionMetrics
from src.model.architectures.model_architectures import ModelArchitectures


def get_nan_ids(valid_ids, set_nan_rate) -> list:
    # valid_idからsequence_idをランダムに抽出
    nan_num = int(len(valid_ids) * set_nan_rate)
    return np.random.choice(valid_ids, size=nan_num, replace=False).tolist()


def get_valid_features_list(
    df: pd.DataFrame,
    imu_cols: list = [],
    thm_cols: list = [],
    tof_cols: list = [],
    set_nan_ids: list = [],
):
    imu_features_list, thm_features_list, tof_features_list = [], [], []
    sequence_ids = []  # 順序を保持するためのリスト
    fillnan_value = 0.0
    # sequence_idでソートしてから処理することで順序を保証
    for sequence_id, group in df.groupby("sequence_id", sort=True):
        sequence_ids.append(sequence_id)
        imu_features = (
            group[imu_cols]
            .ffill()
            .bfill()
            .fillna(fillnan_value)
            .values.astype("float32")
        )
        thm_features = (
            group[thm_cols]
            .ffill()
            .bfill()
            .fillna(fillnan_value)
            .values.astype("float32")
        )
        tof_features = (
            group[tof_cols]
            .ffill()
            .bfill()
            .fillna(fillnan_value)
            .values.astype("float32")
        )
        if sequence_id in set_nan_ids:
            thm_features = fillnan_value * np.ones_like(thm_features)
            tof_features = fillnan_value * np.ones_like(tof_features)
        imu_features_list.append(imu_features)
        thm_features_list.append(thm_features)
        tof_features_list.append(tof_features)

    return (
        imu_features_list,
        thm_features_list,
        tof_features_list,
        sequence_ids,  # 順序を返す
    )


class ValidDataset(Dataset):
    """A basic dataset with augmentation that can be extended for custom datasets."""

    def __init__(
        self,
        df,
        imu_cols: list[str],
        thm_cols: list[str],
        tof_cols: list[str],
        pad_len: int = 512,
        set_nan_ids: list = [],
    ):
        self.df = df

        # 引数で受け取った列名情報を使用
        self.imu_cols = imu_cols
        self.thm_cols = thm_cols
        self.tof_agg_cols = tof_cols
        self.features_cols = imu_cols + thm_cols + tof_cols
        feature_scaler_path = Path("/kaggle/working/features/feature_scaler.yaml")
        with open(feature_scaler_path, "r") as f:
            self.feature_scaler = yaml.safe_load(f)

        (
            self.imu_features_list,
            self.thm_features_list,
            self.tof_features_list,
            self.sequence_ids,  # 順序を保持
        ) = get_valid_features_list(
            df,
            imu_cols=self.imu_cols,
            thm_cols=self.thm_cols,
            tof_cols=self.tof_agg_cols,
            set_nan_ids=set_nan_ids,
        )
        self.pad_len = pad_len
        # 同じ順序でラベルを取得
        self.labels = self._get_labels_in_order(df)

    def _get_labels_in_order(self, df):
        """sequence_idsの順序に合わせてラベルを取得"""
        labels_list = []
        for sequence_id in self.sequence_ids:
            group = df[df["sequence_id"] == sequence_id]
            label = group["gesture_le"].iloc[0]
            labels_list.append(label)
        labels = np.array(labels_list, dtype=np.int64)
        labels_ohe = torch.nn.functional.one_hot(
            torch.tensor(labels), num_classes=len(df["gesture_le"].unique())
        ).numpy()
        return labels_ohe

    def _scale_features(self, features, features_col_list):
        for i, col in enumerate(features_col_list):
            max_val = self.feature_scaler[col]["max"]
            min_val = self.feature_scaler[col]["min"]
            if max_val == min_val:
                features[:, i] = 0.0
            else:
                features[:, i] = (features[:, i] - min_val) / (max_val - min_val + 1e-8)
        return features

    def __len__(self):
        return len(self.imu_features_list)

    def _get_post_cut_features(self, features):
        feature_length = len(features)
        padded_features = np.zeros((self.pad_len, features.shape[1]))
        if feature_length < self.pad_len:
            padded_features[-feature_length:, :] = features
        else:
            padded_features = features[-self.pad_len :]
        return padded_features

    def __getitem__(self, idx):
        """Return a single item from the dataset."""
        imu_features = self.imu_features_list[idx]
        thm_features = self.thm_features_list[idx]
        tof_features = self.tof_features_list[idx]
        # imu_features = self._scale_features(imu_features, self.imu_cols)
        thm_features = self._scale_features(thm_features, self.thm_cols)
        tof_features = self._scale_features(tof_features, self.tof_agg_cols)
        # Apply augmentations if available and in training phase
        inputs_np = {
            "imu_features": imu_features,
            "thm_features": thm_features,
            "tof_features": tof_features,
        }

        imu_features = self._get_post_cut_features(inputs_np["imu_features"])
        thm_features = self._get_post_cut_features(inputs_np["thm_features"])
        tof_features = self._get_post_cut_features(inputs_np["tof_features"])
        inputs = {
            "imu_features": torch.Tensor(imu_features),
            "thm_features": torch.Tensor(thm_features),
            "tof_features": torch.Tensor(tof_features),
        }
        labels = {
            "labels": torch.Tensor(self.labels[idx]),
        }
        return inputs, labels


def run_validation(
    df_path: Path,
    exp_dir: Path,
    splits_dir: Path,
    fold: int,
    inverse_gesture_dict_path: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    output_dir: Optional[Path] = None,
    save_predictions: bool = True,
) -> Dict[str, Any]:
    """
    指定したモデルとデータセットを使ってvalidationのみを実行する関数

    Args:
        model: 評価するPyTorchモデル
        valid_dataset: バリデーション用のデータセット
        batch_size: バッチサイズ
        num_workers: データローダーのワーカー数
        pin_memory: データローダーのpin_memory
        output_dir: 出力ディレクトリ
        save_predictions: 予測結果を保存するかどうか
        device: 使用するデバイス ("auto", "cpu", "cuda", "mps")
        metrics_calculator: メトリクス計算器

    Returns:
        validation結果のメトリクス辞書
    """
    logger_path = exp_dir / "validation.log"
    logger = logging.getLogger(logger_path.stem)

    df = pd.read_csv(df_path)
    valid_ids_path = splits_dir / f"fold_{fold}" / "valid.yaml"
    with open(valid_ids_path, "r") as f:
        valid_ids = yaml.safe_load(f)
    valid_df = df[df["sequence_id"].isin(valid_ids)].reset_index(drop=True)

    # デバイスの設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    config_path = exp_dir / f"fold_{fold}" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = ModelConfig(
        _target_="",
        model_name=config["model"]["model_name"],
        pad_len=config["model"]["pad_len"],
        imu_dim=config["model"]["imu_dim"],
        tof_dim=config["model"]["tof_dim"],
        thm_dim=config["model"]["thm_dim"],
        n_classes=config["model"]["n_classes"],
        default_emb_dim=config["model"]["default_emb_dim"],
        layer_num=config["model"]["layer_num"],
        loss_config=None,  # type: ignore
        optimizer=None,  # type: ignore
        scheduler=None,  # type: ignore
    )
    model_architectures = ModelArchitectures(model_config)
    model = model_architectures.model
    model.load_state_dict(
        torch.load(exp_dir / f"fold_{fold}" / "final_weights.pth", map_location=device)
    )
    # model.load_state_dict(
    #     torch.load(exp_dir / f"fold_{fold}" / "best_weights.pth", map_location=device)
    # )
    model.to(device)
    model.eval()

    set_nan_ids_path = splits_dir / f"fold_{fold}" / "valid_setnan_ids.yaml"
    with open(set_nan_ids_path, "r") as f:
        set_nan_ids_list = yaml.safe_load(f)["set_nan_ids"]
    features_config_path = exp_dir / f"fold_{fold}" / "feature_columns.yaml"
    with open(features_config_path, "r") as f:
        features_config = yaml.safe_load(f)

    valid_dataset = ValidDataset(
        df=valid_df,
        imu_cols=features_config["imu_cols"],
        thm_cols=features_config["thm_cols"],
        tof_cols=features_config["tof_cols"],
        pad_len=config["model"]["pad_len"],
        set_nan_ids=set_nan_ids_list,
    )
    # valid_dataset = AugmentedDataset(
    #     valid_df,
    #     transforms=None,  # type: ignore
    #     phase="valid",
    #     imu_cols=features_config["imu_cols"],
    #     thm_cols=features_config["thm_cols"],
    #     tof_cols=features_config["tof_cols"],
    #     pad_percentile=95,
    #     signal_cut_rate=0.8,
    # )

    dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    metrics_calculator = CompetitionMetrics(
        inverse_gesture_dict_path=inverse_gesture_dict_path
    )

    # 予測結果を保存するリスト
    all_predictions = torch.Tensor().to(device)
    all_labels = torch.Tensor().to(device)
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            outputs = model(inputs)
            predictions = outputs["logits"]
            labels = targets["labels"]
            all_predictions = torch.cat((all_predictions, predictions), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    predictions_np = all_predictions.detach().cpu().numpy()
    labels_np = all_labels.detach().cpu().numpy()

    metrics_value = metrics_calculator(
        y_true=all_labels,
        y_pred=all_predictions,
    )

    # 予測結果の保存
    if save_predictions and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 予測結果の保存
        pred_cols = [f"pred_{i}" for i in range(predictions_np.shape[-1])]  # type: ignore
        target_cols = [f"target_{i}" for i in range(labels_np.shape[-1])]  # type: ignore
        # oofの保存
        oof = pl.DataFrame(
            {
                **{col: predictions_np[:, i] for i, col in enumerate(pred_cols)},
                **{col: labels_np[:, i] for i, col in enumerate(target_cols)},
            }
        )
        oof_path = output_dir / "oof.csv"
        oof.write_csv(str(oof_path))
        print(f"Validation results saved to: {oof_path}")

    # メトリクスの保存
    metrics_dict: Dict[str, Any] = {}
    metrics_dict["f1_macro"] = metrics_calculator.macro_f1
    metrics_dict["f1_binary"] = metrics_calculator.binary_f1
    metrics_dict["competition_score"] = metrics_value
    for key, value in metrics_dict.items():
        logger.info(f"{key}: {value}")

    return metrics_dict


if __name__ == "__main__":
    is_making_nan_ids = False
    # df_path = Path("/kaggle/working/processed_rotations_2/processed_with_rots_df.csv")
    # df_path = Path(
    #     "/kaggle/working/processed_diff01_cumsum_swaphandness4/processed_df.csv"
    # )
    df_path = "/kaggle/working/processed_diff01_swaphandness_means/processed_df.csv"
    inverse_gesture_dict_path = Path(
        "/kaggle/working/encoders/inverse_gesture_dict.pkl"
    )

    batch_size = 64
    num_workers = 2
    pin_memory = True
    save_predictions = True
    # exp_name = "exp_013_5_004_eachbranch_cnn_model"
    # exp_name = "exp_013_5_005_eachbranch_cnn_model"
    # exp_name = "exp_013_6_007_eachbranch_trans_model"
    # exp_name = "exp_017_8_009_orient_behavior_aux"
    # exp_name = "exp_018_8_002_splitseq"
    # exp_name = "exp_018_8_006_splitseq"
    # exp_name = "exp_019_8_006_splitseq_bugfixed"
    # exp_name = "exp_035_9_004_splitpublic_valid_swap_metascale_scaleaug"
    # exp_name = "exp_037_10_001_splitpublic_valid_swap_Nanaug01_cnn"
    # exp_name = "exp_039_10_002_splitpublic_valid_swap_timestrech_cnn"

    exp_name = "exp_039_10_005_splitpublic_valid_swap_drop02timestrech_cnn"
    # exp_name = "exp_039_11_006_splitpublic_valid_swap_timestrech_trans"
    # exp_name = "exp_039_10_007_splitpublic_valid_swap_timestrech_lstmhead"
    # exp_name = "exp_041_10_001_splitpublic_valid_swap_drop05timestrech_cnn"
    # exp_name = "exp_041_10_002_splitpublic_valid_swap_drop07timestrech_cnn"
    # exp_name = "exp_041_10_003_splitpublic_valid_swap_drop08timestrech_nan-1or0_cnn"
    # exp_name = "exp_041_10_004_splitpublic_valid_swap_drop03timestrech_nan-1or0_cnn"

    # exp_name = "exp_041_10_004_splitpublic_valid_swap_drop03timestrech_nan-1or0_cnn"
    # exp_name = "exp_041_10_005_splitpublic_valid_swap_drop02timestrech_nan-1or0_cnn"
    # exp_name = "exp_043_9_002_splitpublic_tofthmmean_cnn"

    # exp_name = "exp_044_9_009_splitpublic_cnn"

    # exp_name = "exp_048_9_001_splitpublic_cnn"

    exp_name = "exp_048_9_001_splitpublic_cnn"
    exp_dir = Path(f"/kaggle/working/{exp_name}")
    nan_rate = 1.0

    # splits_dir = Path("/kaggle/working/splits_sequence_group")
    splits_dir = Path("/kaggle/working/splits_public")
    metrics_list = []
    for fold in range(5):
        output_dir = exp_dir / f"fold_{fold}_valid"
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_making_nan_ids:
            fold_valid_ids_path = splits_dir / f"fold_{fold}" / "valid.yaml"
            with open(fold_valid_ids_path, "r") as f:
                fold_valid_ids = yaml.safe_load(f)
            set_nan_ids_list = get_nan_ids(fold_valid_ids, nan_rate)

            save_path = splits_dir / f"fold_{fold}" / "valid_setnan_ids.yaml"
            with open(save_path, "w") as f:
                yaml.dump({"set_nan_ids": set_nan_ids_list}, f)
            print(f"Set NaN IDs for fold {fold} created and saved.")

        metrics_dict = run_validation(
            df_path=df_path,
            inverse_gesture_dict_path=inverse_gesture_dict_path,
            exp_dir=exp_dir,
            splits_dir=splits_dir,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            output_dir=output_dir,
            save_predictions=save_predictions,
        )
        print(f"Metrics for fold {fold}: {metrics_dict}")
        metrics_list.append(metrics_dict["competition_score"])
    for fold, metrics in enumerate(metrics_list):
        print(f"Fold {fold} competition score: {metrics}")
    print(f"Average competition score across folds: {np.mean(metrics_list)}")
