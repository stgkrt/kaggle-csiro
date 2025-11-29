from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # type: ignore

from src.metrics.competition_metrics import weighted_r2_score


def load_train_data(train_csv_path: Path) -> pd.DataFrame:
    """train.csvを読み込む"""
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")
    return pd.read_csv(train_csv_path)


def calculate_overall_oof_score(train_csv_path: Path, sub_csv_path: Path) -> dict:
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
    sub_df = pd.read_csv(sub_csv_path)
    sub_df = sub_df.rename(columns={"target": "pred"})
    scoring_df = train_df.merge(
        sub_df[["sample_id", "pred"]], on="sample_id", how="left"
    )
    print(f"\n{'=' * 60}")
    print(f"Total samples: {len(scoring_df)}")
    print(f"Unique sample IDs: {scoring_df['sample_id'].nunique()}")
    # 全foldの予測とターゲットを収集
    all_targets = scoring_df.sort_values("sample_id")["target"].values.reshape(-1, 5)
    all_predictions = scoring_df.sort_values("sample_id")["pred"].values.reshape(-1, 5)
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
    train_csv_path = Path("/kaggle/input/csiro-biomass/train.csv")
    sub_csv_path = Path("/kaggle/working/submission.csv")

    # スコアを計算
    results = calculate_overall_oof_score(
        train_csv_path=train_csv_path,
        sub_csv_path=sub_csv_path,
    )
