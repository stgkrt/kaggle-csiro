"""
実験結果フォルダからOOF予測を読み込んでスコアを計算するスクリプト

Usage:
    python src/check_oof_score.py --exp_dir /kaggle/working/exp_005_cloverclass_003
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# from src.metrics.competition_metrics import weighted_r2_score
from src.metrics.competition_metrics import weighted_r2_score_v2 as weighted_r2_score


def load_all_oof_predictions(
    exp_dir: Path, n_folds: int = 5
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """全foldのOOF予測を読み込んで結合する"""
    oof_dfs = []
    fold_dfs = {}

    for fold in range(n_folds):
        oof_path = exp_dir / f"fold_{fold}" / "oof.csv"

        if not oof_path.exists():
            print(f"Warning: OOF file not found: {oof_path}")
            continue

        oof_df = pd.read_csv(oof_path)
        oof_df["fold"] = fold  # fold番号を追加
        print(f"Loaded fold_{fold}: {len(oof_df)} rows")
        oof_dfs.append(oof_df)
        fold_dfs[fold] = oof_df

    if not oof_dfs:
        raise FileNotFoundError(f"No OOF files found in {exp_dir}")

    # 全foldを結合
    all_oof_df = pd.concat(oof_dfs, ignore_index=True)
    print(f"\nTotal OOF predictions: {len(all_oof_df)} rows")

    return all_oof_df, fold_dfs


def calculate_oof_score(
    oof_df: pd.DataFrame, train_df: pd.DataFrame, fold_name: str = "Overall"
) -> dict:
    """
    OOF予測とtrain.csvからスコアを計算

    Args:
        oof_df: OOF予測データ (columns: sample_id, pred, target)
        train_df: 訓練データ (columns: sample_id, target, ...)
        fold_name: 表示用のfold名

    Returns:
        dict: スコア情報
    """
    # sample_idでソート
    oof_df = oof_df.sort_values("sample_id").reset_index(drop=True)

    # 画像IDを抽出 (sample_idから "__Dry_*" を除去)
    oof_df["image_id"] = oof_df["sample_id"].str.split("__").str[0]
    unique_images = oof_df["image_id"].unique()

    print(f"Unique images in OOF: {len(unique_images)}")
    print(f"Unique sample_ids: {oof_df['sample_id'].nunique()}")

    # 各画像ごとに5つのターゲット値を並べる
    predictions_list = []
    targets_list = []
    valid_image_ids = []

    # ターゲット名の順序を定義
    target_order = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]

    for image_id in unique_images:
        # 各画像に対する5つの予測を取得
        image_oof = oof_df[oof_df["image_id"] == image_id].copy()

        # sample_idからターゲット名を抽出
        image_oof["target_name"] = image_oof["sample_id"].str.split("__").str[1]

        # ターゲット順にソート
        image_oof["target_order"] = image_oof["target_name"].map(
            {name: i for i, name in enumerate(target_order)}
        )
        image_oof = image_oof.sort_values("target_order")

        # 5つすべてが揃っているかチェック
        if len(image_oof) != 5:
            print(
                f"Warning: Image {image_id} has {len(image_oof)} targets (expected 5)"
            )
            continue

        predictions_list.append(image_oof["pred"].values)
        targets_list.append(image_oof["target"].values)
        valid_image_ids.append(image_id)

    if not predictions_list:
        raise ValueError("No valid predictions found")

    # numpy配列に変換 (shape: [n_images, 5])
    predictions = np.array(predictions_list)
    targets = np.array(targets_list)

    print(f"\nValid images for scoring: {len(valid_image_ids)}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")

    # スコア計算
    overall_score, r2_scores = weighted_r2_score(targets, predictions)

    # 結果を表示
    print(f"\n{'=' * 60}")
    print(f"{fold_name} OOF SCORE: {overall_score:.6f}")
    print(f"{'=' * 60}")
    print("\nIndividual R2 Scores:")

    weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    for name, r2, weight in zip(target_order, r2_scores, weights):
        print(f"  {name:15s}: R2={r2:.6f} (weight={weight})")

    return {
        "overall_score": float(overall_score),
        "individual_r2_scores": r2_scores,
        "target_names": target_order,
        "weights": weights,
        "n_images": len(valid_image_ids),
        "n_samples": len(predictions) * 5,
    }


def postprocess_predictions(oof_df: pd.DataFrame) -> pd.DataFrame:
    dry_dead_mean = 12.04
    oof_df.loc[oof_df["sample_id"].str.endswith("_Dry_Dead_g"), "pred"] = (
        dry_dead_mean * 0.5
        + oof_df.loc[oof_df["sample_id"].str.endswith("_Dry_Dead_g"), "pred"] * 0.5
    )

    return oof_df


def main():
    parser = argparse.ArgumentParser(
        description="Calculate OOF score from experiment results"
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="/kaggle/working/exp_005_cloverclass_003",
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="/kaggle/input/csiro-biomass/train.csv",
        help="Path to train.csv",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds",
    )

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    train_csv_path = Path(args.train_csv)

    # ディレクトリの存在チェック
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")

    print(f"Experiment directory: {exp_dir}")
    print(f"Train CSV: {train_csv_path}")
    print(f"Number of folds: {args.n_folds}")
    print(f"\n{'=' * 60}\n")

    # OOF予測を読み込み
    all_oof_df, fold_dfs = load_all_oof_predictions(exp_dir, n_folds=args.n_folds)
    # all_oof_df = postprocess_predictions(all_oof_df)

    # train.csvを読み込み (実際にはOOF内のtarget列を使用するため参照のみ)
    train_df = pd.read_csv(train_csv_path)
    print(f"\nTrain CSV loaded: {len(train_df)} rows")

    # 各foldごとのスコアを計算
    fold_results = {}
    for fold_idx, fold_df in sorted(fold_dfs.items()):
        print(f"\n{'#' * 60}")
        print(f"# Fold {fold_idx}")
        print(f"{'#' * 60}")
        # fold_df_processed = postprocess_predictions(fold_df.copy())
        fold_df_processed = fold_df.copy()

        fold_result = calculate_oof_score(
            fold_df_processed, train_df, fold_name=f"Fold {fold_idx}"
        )
        fold_results[fold_idx] = fold_result

    # 全体のスコア計算
    print(f"\n{'#' * 60}")
    print("# Overall (All Folds)")
    print(f"{'#' * 60}")
    overall_results = calculate_oof_score(all_oof_df, train_df, fold_name="Overall")

    # 結果を保存
    output_path = exp_dir / "oof_score_summary.txt"
    with open(output_path, "w") as f:
        f.write(f"Experiment: {exp_dir.name}\n")
        f.write(f"{'=' * 60}\n\n")

        # 各foldの結果
        for fold_idx, result in sorted(fold_results.items()):
            f.write(f"Fold {fold_idx} Score: {result['overall_score']:.6f}\n")
            for name, r2, weight in zip(
                result["target_names"],
                result["individual_r2_scores"],
                result["weights"],
            ):
                f.write(f"  {name:15s}: R2={r2:.6f} (weight={weight})\n")
            f.write(
                f"  Images: {result['n_images']}, Samples: {result['n_samples']}\n\n"
            )

        # 全体の結果
        f.write(f"{'=' * 60}\n")
        f.write(f"Overall OOF Score: {overall_results['overall_score']:.6f}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write("Individual R2 Scores:\n")
        for name, r2, weight in zip(
            overall_results["target_names"],
            overall_results["individual_r2_scores"],
            overall_results["weights"],
        ):
            f.write(f"  {name:15s}: R2={r2:.6f} (weight={weight})\n")
        f.write(f"\nNumber of images: {overall_results['n_images']}\n")
        f.write(f"Number of samples: {overall_results['n_samples']}\n")

    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
