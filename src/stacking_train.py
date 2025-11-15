import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml  # type:ignore
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))
from src.metrics.competition_metrics import CompetitionMetrics


# numpyでsoftmax
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def setup_logging(output_dir: Path):
    """ログ設定を行う：output_dirへのファイル出力と標準出力の両方を設定"""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "stacking_train.log"

    # 既存のハンドラーをクリア
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # フォーマッターを作成
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # ファイルハンドラーを作成
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # コンソールハンドラーを作成
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ルートロガーにハンドラーを追加
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # モジュールロガーを取得（propagateを有効にしてルートに委譲）
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True

    return logger


# 初期設定（後でsetup_loggingで上書きされる）
logger = logging.getLogger(__name__)


def load_demographics(demographics_path: Path) -> pd.DataFrame:
    """人口統計学的データを読み込み"""
    demographics = pd.read_csv(demographics_path)
    logger.info(f"Demographics data loaded: {len(demographics)} subjects")
    train_df = pd.read_csv(
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
    )
    subject_seqid_df = train_df[["subject", "sequence_id"]].drop_duplicates()
    demographics = demographics.merge(subject_seqid_df, on="subject", how="left")
    return demographics


def load_oof_predictions(
    exp_dir: Path, fold: int
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """指定された実験・フォルドのOOF予測を読み込み"""
    # oof_path = exp_dir / f"fold_{fold}" / "oof.csv"
    oof_path = exp_dir / f"fold_{fold}" / "best_oof.csv"
    if not oof_path.exists():
        logger.warning(f"OOF file not found: {oof_path}")
        return None

    oof_df = pd.read_csv(oof_path)
    pred_cols = [col for col in oof_df.columns if col.startswith("pred_")]

    logger.info(
        f"Loaded OOF from {oof_path}: {len(oof_df)} samples, "
        f"{len(pred_cols)} predictions"
    )
    config_path = exp_dir / f"fold_{fold}" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    split_path = Path(config["split"]["split_dir"])
    valid_ids_path = split_path / f"fold_{fold}" / "valid.yaml"
    with open(valid_ids_path, "r") as f:
        valid_seq_ids = yaml.safe_load(f)
    oof_df["sequence_id"] = valid_seq_ids[: len(oof_df)]
    return oof_df, pred_cols


def get_sequence_subject_mapping(
    df_path: Path, valid_seq_ids: List[str]
) -> pd.DataFrame:
    """validation sequenceに対応するsubject情報を取得"""
    # データが大きいので、必要な列だけ読み込み
    df = pd.read_csv(df_path, usecols=["sequence_id", "subject"])

    # validation sequenceのみをフィルタ
    valid_df = df[df["sequence_id"].isin(valid_seq_ids)]

    # sequence_idごとにsubjectを取得（1つのsequenceは1つのsubjectに対応）
    seq_subject_mapping = (
        valid_df.groupby("sequence_id")["subject"].first().reset_index()
    )

    logger.info(
        f"Created sequence-subject mapping: {len(seq_subject_mapping)} sequences"
    )
    return seq_subject_mapping


def collect_oof_features(
    working_dir: Path,
    exp_name: str,
    df_path: Path,
    splits_dir: Path,
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    指定された実験のOOF予測を全foldから収集し、メタデータと結合

    Returns:
        features_df: 特徴量データフレーム
        targets_df: ターゲットデータフレーム
    """
    logger.info("--- Start Collect OOF Files ---")

    oof_all_fold = pd.DataFrame()
    for fold in range(n_folds):
        logger.info(f"Processing fold {fold}...")

        # OOF予測を読み込み
        exp_dir = working_dir / exp_name
        oof_result = load_oof_predictions(exp_dir, fold)
        if oof_result is None:
            continue

        oof_df, pred_cols = oof_result

        # sequence_idを追加
        oof_df = oof_df.copy()
        oof_df["fold"] = fold

        # oof_df[pred_cols] = softmax(oof_df[pred_cols].values, axis=1)
        # 予測列にfold情報を追加
        feature_cols = []
        for col in pred_cols:
            new_col = f"{col}_{exp_name}"
            oof_df = oof_df.rename(columns={col: new_col})
            feature_cols.append(new_col)

        # 特徴量とターゲットを分離
        target_cols = [col for col in oof_df.columns if col.startswith("target_")]

        if oof_all_fold.empty:
            oof_all_fold = oof_df.copy()
        else:
            oof_all_fold = pd.concat([oof_all_fold, oof_df], axis=0)

        logger.info(f"Fold {fold} processed: {len(oof_df)} samples")

    if oof_all_fold.empty:
        raise ValueError("No valid OOF data found")
    logger.info(f"all fold processed: {len(oof_all_fold)} samples")
    return oof_all_fold, feature_cols, target_cols


def prepare_stacking_data(
    working_dir: Path,
    exp_names: List[str],
    df_path: Path,
    splits_dir: Path,
    demographics_path: Path,
    n_folds: int = 5,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    複数の実験からOOF予測を収集し、スタッキング用データを準備
    """
    logger.info("=== Start Preparing Data ===")
    # メタデータを読み込み
    demographics = load_demographics(demographics_path)
    all_feature_cols = [
        "adult_child",
        "age",
        "sex",
        "handedness",
        "height_cm",
        "shoulder_to_wrist_cm",
        "elbow_to_wrist_cm",
    ]
    oof_all_exp_df = pd.DataFrame()
    for exp_name in exp_names:
        logger.info(f"Processing experiment: {exp_name}")

        try:
            oof_exp_df, feature_cols, target_cols = collect_oof_features(
                working_dir, exp_name, df_path, splits_dir, n_folds
            )
            oof_feature_exp_df = oof_exp_df[
                ["sequence_id", "fold"] + feature_cols
            ].copy()
            if oof_feature_exp_df.empty:
                logger.warning(f"Experiment {exp_name} produced no valid OOF data")
                continue

            if oof_all_exp_df.empty:
                oof_all_exp_df = oof_exp_df.copy()
            else:
                oof_all_exp_df = oof_all_exp_df.merge(
                    oof_feature_exp_df, on=["sequence_id", "fold"], how="left"
                )
            logger.info(f"Experiment {exp_name} processed: {len(oof_exp_df)} samples")
            logger.info(f"All Experiments processed: {len(oof_all_exp_df)} samples")
            all_feature_cols.extend(feature_cols)

        except Exception as e:
            logger.error(f"Failed to process experiment {exp_name}: {e}")
            continue

    # メタデータを追加
    oof_all_exp_df = oof_all_exp_df.merge(demographics, on="sequence_id", how="left")
    logger.info(f"Final stacking dataset: {len(oof_all_exp_df)} samples")

    return oof_all_exp_df, all_feature_cols, target_cols


def calculate_class_weights(
    inverse_gesture_dict_path: Path,
    n_classes: int,
    target_weight: float = 3.0,
    non_target_weight: float = 1.0,
) -> Dict[int, float]:
    """
    target gestureのクラスに高い重みを設定したclass_weightを計算

    Args:
        inverse_gesture_dict_path: ジェスチャー辞書のパス
        n_classes: クラス数
        target_weight: target gestureの重み
        non_target_weight: non-target gestureの重み

    Returns:
        クラスIDと重みの辞書
    """
    # Target gestures定義
    target_gestures = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Neck - pinch skin",
        "Neck - scratch",
    ]

    try:
        # inverse_gesture_dictを読み込み
        import joblib

        inverse_gesture_dict = joblib.load(inverse_gesture_dict_path)

        # クラス重みを設定
        class_weights = {}
        for class_id in range(n_classes):
            gesture_name = inverse_gesture_dict.get(class_id, "unknown")
            if gesture_name in target_gestures:
                class_weights[class_id] = target_weight
                logger.info(
                    f"Target gesture '{gesture_name}' (ID: {class_id}) "
                    f"weight: {target_weight}"
                )
            else:
                class_weights[class_id] = non_target_weight

        target_count = len([w for w in class_weights.values() if w == target_weight])
        non_target_count = len(
            [w for w in class_weights.values() if w == non_target_weight]
        )
        logger.info(
            f"Class weights configured: {target_count} target classes, "
            f"{non_target_count} non-target classes"
        )

        return class_weights

    except Exception as e:
        logger.warning(f"Failed to load gesture dictionary: {e}")
        # フォールバック: 全クラス同じ重み
        return {i: 1.0 for i in range(n_classes)}


def convert_targets_to_class_labels(targets_df: pd.DataFrame) -> pd.Series:
    """One-hot encodedターゲットをクラスラベルに変換"""
    target_cols = [col for col in targets_df.columns if col.startswith("target_")]
    target_matrix = targets_df[target_cols].values
    class_labels = np.argmax(target_matrix, axis=1)
    return pd.Series(class_labels, index=targets_df.index)


def evaluate_with_competition_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    inverse_gesture_dict_path: Path,
    return_detailed: bool = True,
) -> Dict[str, float]:
    """
    競技用メトリクスを含む詳細な評価を実行

    Args:
        y_true: 真のクラスラベル
        y_pred: 予測されたクラス確率分布
        inverse_gesture_dict_path: ジェスチャー辞書のパス
        return_detailed: 詳細メトリクスを返すかどうか

    Returns:
        評価メトリクスの辞書
    """
    # クラスラベルを取得
    y_true_labels = y_true
    y_pred_labels = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

    # 競技用メトリクス
    try:
        # 確率分布をone-hot形式に変換（CompetitionMetricsはone-hot形式を期待）
        n_classes = y_pred.shape[1] if y_pred.ndim > 1 else int(np.max(y_pred)) + 1
        y_true_onehot = np.zeros((len(y_true_labels), n_classes))
        y_pred_onehot = np.zeros((len(y_pred_labels), n_classes))

        for i, label in enumerate(y_true_labels):
            if 0 <= label < n_classes:
                y_true_onehot[i, int(label)] = 1

        for i, label in enumerate(y_pred_labels):
            if 0 <= label < n_classes:
                y_pred_onehot[i, int(label)] = 1

        # PyTorchテンソルに変換
        y_true_tensor = torch.tensor(y_true_onehot, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred_onehot, dtype=torch.float32)

        # 競技用メトリクス計算
        competition_metrics = CompetitionMetrics(
            inverse_gesture_dict_path=inverse_gesture_dict_path
        )
        competition_score = competition_metrics(y_true_tensor, y_pred_tensor)

        results = {
            "competition_score": competition_score,
            "competition_binary_f1": competition_metrics.binary_f1,
            "competition_macro_f1": competition_metrics.macro_f1,
        }

    except Exception as e:
        logger.warning(f"Competition metrics calculation failed: {e}")
        results = {
            "competition_score": 0.0,
            "competition_binary_f1": 0.0,
            "competition_macro_f1": 0.0,
        }

    if not return_detailed:
        return {"competition_score": results["competition_score"]}

    return results


def train_lightgbm_stacking(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    output_dir: Path,
    inverse_gesture_dict_path: Path,
    n_folds: int = 5,
    target_weight: float = 3.0,
    non_target_weight: float = 1.0,
    lgb_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LightGBMを使用してスタッキングモデルを学習

    Args:
        features_df: 特徴量データフレーム
        targets_df: ターゲットデータフレーム
        output_dir: 出力ディレクトリ
        inverse_gesture_dict_path: ジェスチャー辞書のパス
        n_folds: CVのfold数
        target_weight: target gestureクラスの重み
        non_target_weight: non-target gestureクラスの重み

    Returns:
        評価結果の辞書
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_scores = []
    feature_importance_list = []
    models = []

    n_classes = 18
    oof_preds = np.zeros((len(df), n_classes))
    oof_targets = np.zeros((len(df),))

    for fold_idx in range(n_folds):
        logger.info(f"Training fold {fold_idx}...")
        val_idx = df[df["fold"] == fold_idx].index
        train_df = df[df["fold"] != fold_idx].reset_index(drop=True)
        val_df = df[df["fold"] == fold_idx].reset_index(drop=True)

        X_train = train_df[feature_cols]
        y_train = train_df[target_cols]
        X_val = val_df[feature_cols]
        y_val = val_df[target_cols]

        # y_train, y_valはone_hotからclass_idxに変換
        y_train = np.argmax(y_train, axis=1)
        y_val = np.argmax(y_val, axis=1)

        logger.info(
            f"Training LightGBM {len(X_train)} samples {len(feature_cols)} features"
        )
        logger.info(f"features cols is -> {feature_cols}")

        # クラス重みを計算
        class_weights = calculate_class_weights(
            inverse_gesture_dict_path, n_classes, target_weight, non_target_weight
        )
        sample_weights = [class_weights[value] for value in y_train]

        # データセットの作成
        train_dataset = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

        # モデルの学習
        model = lgb.train(
            lgb_params,
            train_dataset,
            valid_sets=[val_dataset],
            num_boost_round=10000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        # 予測
        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds
        oof_targets[val_idx] = y_val

        # スコア計算（競技用メトリクスを含む）
        val_metrics = evaluate_with_competition_metrics(
            y_true=y_val,
            y_pred=val_preds,
            inverse_gesture_dict_path=inverse_gesture_dict_path,
            return_detailed=True,
        )

        cv_scores.append(val_metrics["competition_score"])

        logger.info(
            f"Fold {fold_idx} Competition Score: {val_metrics['competition_score']:.4f}"
        )
        logger.info(
            f"Fold {fold_idx} Binary F1: {val_metrics['competition_binary_f1']:.4f}"
        )
        logger.info(
            f"Fold {fold_idx} Macro F1: {val_metrics['competition_macro_f1']:.4f}"
        )

        # 特徴量重要度を保存
        feature_importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importance(),
                "fold": fold_idx,
            }
        )
        feature_importance_list.append(feature_importance)

        # モデルを保存
        model_path = output_dir / f"lgb_model_fold_{fold_idx}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        models.append(model)

    # CV結果の集約
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)

    logger.info(f"CV Competition Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

    # OOF予測の評価（競技用メトリクスを含む）
    oof_metrics = evaluate_with_competition_metrics(
        y_true=oof_targets,
        y_pred=oof_preds,
        inverse_gesture_dict_path=inverse_gesture_dict_path,
        return_detailed=True,
    )

    logger.info(f"OOF Competition Score: {oof_metrics['competition_score']:.4f}")
    logger.info(f"OOF Binary F1: {oof_metrics['competition_binary_f1']:.4f}")
    logger.info(f"OOF Macro F1: {oof_metrics['competition_macro_f1']:.4f}")

    # 結果の保存
    results = {
        "cv_scores": cv_scores,
        "mean_cv_score": mean_cv_score,
        "std_cv_score": std_cv_score,
        "oof_competition_score": oof_metrics["competition_score"],
        "oof_binary_f1": oof_metrics["competition_binary_f1"],
        "oof_macro_f1": oof_metrics["competition_macro_f1"],
        "n_features": len(feature_cols),
        "n_samples": len(df),
        "n_classes": n_classes,
    }

    # OOF予測を保存
    oof_df = df[["sequence_id", "subject", "fold"]].copy()
    for i in range(n_classes):
        oof_df[f"oof_pred_{i}"] = oof_preds[:, i]
    oof_df["oof_pred_label"] = np.argmax(oof_preds, axis=1)
    oof_df["true_label"] = oof_targets
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    # 特徴量重要度を保存
    feature_importance_df = pd.concat(feature_importance_list, ignore_index=True)
    feature_importance_summary = (
        feature_importance_df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
    )
    feature_importance_summary = feature_importance_summary.sort_values(
        "mean", ascending=False
    )
    feature_importance_summary.to_csv(
        output_dir / "feature_importance.csv", index=False
    )

    # 結果を保存
    with open(output_dir / "stacking_results.yaml", "w") as f:
        yaml.dump(results, f)

    logger.info(f"Stacking training completed. Results saved to {output_dir}")

    return results


def run_train(
    exp_names,
    working_dir,
    df_path,
    splits_dir,
    demographics_path,
    inverse_gesture_dict_path,
    output_dir,
    n_folds,
    random_state,
    target_weight=3.0,
    non_target_weight=1.0,
    lgb_params=None,
):
    # パスの準備
    working_dir = Path(working_dir)
    df_path = Path(df_path)
    splits_dir = Path(splits_dir)
    demographics_path = Path(demographics_path)
    inverse_gesture_dict_path = Path(inverse_gesture_dict_path)
    output_dir = Path(output_dir)

    logger.info(f"Starting stacking training with experiments: {exp_names}")

    try:
        # スタッキングデータの準備
        stacking_df, feature_cols, target_cols = prepare_stacking_data(
            working_dir=working_dir,
            exp_names=exp_names,
            df_path=df_path,
            splits_dir=splits_dir,
            demographics_path=demographics_path,
            n_folds=n_folds,
        )

        # LightGBMスタッキングモデルの学習
        results = train_lightgbm_stacking(
            df=stacking_df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            output_dir=output_dir,
            inverse_gesture_dict_path=inverse_gesture_dict_path,
            n_folds=n_folds,
            target_weight=target_weight,
            non_target_weight=non_target_weight,
            lgb_params=lgb_params,
        )

        logger.info("Stacking training completed successfully")
        logger.info(
            f"CV Score: {results['mean_cv_score']:.4f} ± {results['std_cv_score']:.4f}"
        )

    except Exception as e:
        logger.error(f"Stacking training failed: {e}")
        raise


if __name__ == "__main__":
    # マニュアル入力
    exp_name = "exp039_base"
    exp_names = [
        # "exp_022_8_004_splitold_layer_dim_num",
        # "exp_032_9_014_splitold_validfilter_thmscaler",
        "exp_039_10_002_splitpublic_valid_swap_timestrech_cnn",
        "exp_039_11_003_splitpublic_valid_swap_timestrech_trans",
        "exp_039_10_003_splitpublic_valid_swap_timestrech_lstmhead",
    ]
    target_weight = 1.5
    working_dir = "/kaggle/working"
    df_path = "/kaggle/working/processed_diff01_cumsum/processed_df.csv"
    splits_dir = "/kaggle/working/splits_public"
    demographics_path = (
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
    )
    inverse_gesture_dict_path = "/kaggle/working/encoders/inverse_gesture_dict.pkl"
    output_dir = f"/kaggle/working/stacking_models_{exp_name}_w{target_weight}"
    n_folds = 5
    random_state = 42

    non_target_weight = 1.0

    params = {
        "objective": "multiclass",
        "num_class": 18,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 100,
        "learning_rate": 0.001,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
    }

    # 早期にログ設定を初期化
    output_dir_path = Path(output_dir)
    logger = setup_logging(output_dir_path)

    logger.info("=== Stacking Training Started ===")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Experiments: {exp_names}")

    run_train(
        exp_names=exp_names,
        working_dir=working_dir,
        df_path=df_path,
        splits_dir=splits_dir,
        demographics_path=demographics_path,
        inverse_gesture_dict_path=inverse_gesture_dict_path,
        output_dir=output_dir,
        n_folds=n_folds,
        random_state=random_state,
        target_weight=target_weight,
        non_target_weight=non_target_weight,
        lgb_params=params,
    )
