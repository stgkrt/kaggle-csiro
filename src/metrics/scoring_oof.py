from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # type: ignore

from src.metrics.competition_metrics import weighted_r2_score


def load_fold_oof(exp_dir: Path, fold: int) -> pd.DataFrame:
    """指定されたfoldのOOFファイルを読み込む"""
    oof_path = exp_dir / f"fold_{fold}" / "oof.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF file not found: {oof_path}")
    return pd.read_csv(oof_path)


def load_fold_valid_indices(splits_dir: Path, fold: int) -> list:
    """指定されたfoldの検証用sample_idを読み込む"""
    valid_path = splits_dir / f"fold_{fold}" / "valid.yaml"
    if not valid_path.exists():
        raise FileNotFoundError(f"Valid indices file not found: {valid_path}")

    with open(valid_path, "r") as f:
        valid_indices = yaml.safe_load(f)
    return valid_indices


def load_train_data(train_csv_path: Path) -> pd.DataFrame:
    """train.csvを読み込む"""
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")
    return pd.read_csv(train_csv_path)


def calculate_overall_oof_score(
    exp_dir: Path, train_csv_path: Path, splits_dir: Path, n_folds: int = 5
) -> dict:
    """
    全Foldを通したOverall OOF Scoreを計算

    Args:
        exp_dir: 実験ディレクトリのパス
        train_csv_path: train.csvのパス
        splits_dir: fold分割ファイルが格納されているディレクトリのパス
        n_folds: Fold数

    Returns:
        dict: スコア情報を含む辞書
    """
    # train.csvを読み込み
    train_df = load_train_data(train_csv_path)
    print(f"Loaded train.csv: {len(train_df)} rows")

    # 全foldのOOF予測を収集
    all_predictions = []
    all_targets = []
    all_sample_ids = []

    for fold in range(n_folds):
        print(f"\nProcessing fold {fold}...")

        # OOFファイルを読み込み
        oof_df = load_fold_oof(exp_dir, fold)
        print(f"  Loaded OOF: {len(oof_df)} rows")

        # 検証用インデックスを読み込み
        valid_indices = load_fold_valid_indices(splits_dir, fold)
        print(f"  Valid indices: {len(valid_indices)} samples")

        # 予測値と真値を取得
        pred_cols = [f"pred_{i}" for i in range(5)]
        target_cols = [f"target_{i}" for i in range(5)]

        predictions = oof_df[pred_cols].values
        targets = oof_df[target_cols].values

        all_predictions.append(predictions)
        all_targets.append(targets)
        all_sample_ids.extend(valid_indices)

        # 各foldのスコアを計算
        fold_score, fold_r2_scores = weighted_r2_score(targets, predictions)
        print(f"  Fold {fold} Score: {fold_score:.6f}")
        print(f"  Individual R2: {fold_r2_scores}")

    # 全foldを結合
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    print(f"\n{'=' * 60}")
    print(f"Total samples: {len(all_predictions)}")
    print(f"Unique sample IDs: {len(set(all_sample_ids))}")

    # Overall OOF Scoreを計算
    overall_score, overall_r2_scores = weighted_r2_score(all_targets, all_predictions)

    print(f"\n{'=' * 60}")
    print(f"OVERALL OOF SCORE: {overall_score:.6f}")
    print(f"{'=' * 60}")
    print("\nIndividual R2 Scores:")
    target_names = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
    weights = [0.1, 0.1, 0.1, 0.2, 0.5]

    for _, (name, r2, weight) in enumerate(
        zip(target_names, overall_r2_scores, weights, strict=True)
    ):
        print(f"  {name:15s}: R2={r2:.6f} (weight={weight})")

    return {
        "overall_score": overall_score,
        "individual_r2_scores": overall_r2_scores.tolist(),
        "target_names": target_names,
        "weights": weights,
        "n_samples": len(all_predictions),
    }


if __name__ == "__main__":
    exp_dir = Path("/kaggle/working/exp_003_000")
    train_csv_path = Path("/kaggle/input/csiro-biomass/train.csv")
    splits_dir = Path("/kaggle/working/splits")
    n_folds = 5

    # スコアを計算
    results = calculate_overall_oof_score(
        exp_dir=exp_dir,
        train_csv_path=train_csv_path,
        splits_dir=splits_dir,
        n_folds=n_folds,
    )

    # 結果を保存
    output_path = exp_dir / "overall_oof_score.yaml"
    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"\nResults saved to: {output_path}")
