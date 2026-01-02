"""
後処理スクリプト - OOF予測値の分布を訓練データの分布に一致させる
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import QuantileTransformer


@dataclass
class PostProcessConfig:
    """後処理の設定クラス"""

    # 基本設定
    exp_dir: str = "/kaggle/working/exp_011_003"
    train_path: str = "/kaggle/input/csiro-biomass/train.csv"
    output_path: Optional[str] = None

    # 分布マッチング設定
    method: str = "histogram"  # "quantile", "histogram", "mode_max", or "min_max"
    targets: Optional[List[str]] = None  # Noneの場合は全ターゲット
    log_transform: bool = False

    # 可視化設定
    visualize: bool = False
    viz_dir: Optional[str] = None

    def __post_init__(self):
        """post_initでパスをPathオブジェクトに変換"""
        self.exp_dir = Path(self.exp_dir)
        self.train_path = Path(self.train_path)

        if self.output_path is None:
            self.output_path = self.exp_dir / "oof_processed.csv"
        else:
            self.output_path = Path(self.output_path)

        if self.viz_dir is not None:
            self.viz_dir = Path(self.viz_dir)
        else:
            self.viz_dir = self.exp_dir / "visualizations"

    @classmethod
    def from_args(cls) -> "PostProcessConfig":
        """コマンドライン引数から設定を作成"""
        parser = argparse.ArgumentParser(
            description="OOF予測値の分布を訓練データの分布に一致させる"
        )
        parser.add_argument(
            "--exp_dir",
            type=str,
            default="/kaggle/working/exp_011_003",
            help="実験ディレクトリのパス",
        )
        parser.add_argument(
            "--train_path",
            type=str,
            default="/kaggle/input/csiro-biomass/train.csv",
            help="train.csvのパス",
        )
        parser.add_argument(
            "--output_path",
            type=str,
            default=None,
            help="後処理されたOOF予測の出力パス（デフォルト: exp_dir/oof_processed.csv）",
        )
        parser.add_argument(
            "--method",
            type=str,
            default="histogram",
            choices=["quantile", "histogram", "mode_max", "min_max"],
            help="分布マッチング手法（quantile, histogram, mode_max, or min_max）",
        )
        parser.add_argument(
            "--targets",
            type=str,
            nargs="+",
            default=None,
            help="分布補正するターゲットのリスト（例: --targets Dry_Clover_g Dry_Dead_g）。指定しない場合は全ターゲットを処理",
        )
        parser.add_argument(
            "--visualize",
            action="store_true",
            help="予測値と真の値、分布の可視化を生成",
        )
        parser.add_argument(
            "--viz_dir",
            type=str,
            default=None,
            help="可視化結果の保存先ディレクトリ（デフォルト: exp_dir/visualizations）",
        )
        parser.add_argument(
            "--log_transform",
            action="store_true",
            help="分布調整前に対数変換を適用する（対数空間で分布を合わせる）",
        )

        args = parser.parse_args()

        return cls(
            exp_dir=args.exp_dir,
            train_path=args.train_path,
            output_path=args.output_path,
            method=args.method,
            targets=args.targets,
            visualize=args.visualize,
            viz_dir=args.viz_dir,
            log_transform=args.log_transform,
        )

    def print_config(self):
        """設定内容を表示"""
        print("=" * 80)
        print("Configuration")
        print("=" * 80)
        print(f"Experiment directory: {self.exp_dir}")
        print(f"Train data path: {self.train_path}")
        print(f"Output path: {self.output_path}")
        print(f"Method: {self.method}")
        print(f"Targets: {self.targets if self.targets else 'All'}")
        print(f"Log transform: {self.log_transform}")
        print(f"Visualize: {self.visualize}")
        if self.visualize:
            print(f"Visualization directory: {self.viz_dir}")
        print("=" * 80)


def load_oof_predictions(exp_dir: Path) -> pd.DataFrame:
    """
    全てのfoldのoof.csvを結合する

    Args:
        exp_dir: 実験ディレクトリのパス (e.g., /kaggle/working/exp_011_003)

    Returns:
        結合されたOOF予測データフレーム
    """
    oof_dfs = []

    # fold_*ディレクトリを検索
    fold_dirs = sorted(exp_dir.glob("fold_*"))

    if not fold_dirs:
        raise ValueError(f"No fold directories found in {exp_dir}")

    print(f"Found {len(fold_dirs)} fold directories")

    for fold_dir in fold_dirs:
        oof_path = fold_dir / "oof.csv"
        if oof_path.exists():
            df = pd.read_csv(oof_path)
            print(f"Loaded {len(df)} samples from {oof_path}")
            oof_dfs.append(df)
        else:
            print(f"Warning: {oof_path} not found")

    # 全foldを結合
    oof_combined = pd.concat(oof_dfs, ignore_index=True)
    print(f"\nTotal OOF samples: {len(oof_combined)}")

    return oof_combined


def load_train_data(train_path: Path) -> pd.DataFrame:
    """
    訓練データを読み込む

    Args:
        train_path: train.csvのパス

    Returns:
        訓練データフレーム
    """
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} training samples")
    return df


def extract_target_name(sample_id: str) -> str:
    """
    sample_idからtarget名を抽出
    例: ID1012260530__Dry_Green_g -> Dry_Green_g
    """
    return sample_id.split("__")[-1]


def weighted_r2_score(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Weighted R2スコアを計算

    Args:
        y_true: 真の値 (shape: N x 5)
        y_pred: 予測値 (shape: N x 5)

    Returns:
        weighted_r2: 重み付きR2スコア
        r2_scores: 各ターゲットのR2スコア
    """
    weights = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    r2_scores = []

    for i in range(5):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = np.sum((y_t - y_p) ** 2)
        ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)

    r2_scores = np.array(r2_scores)
    weighted_r2 = np.sum(r2_scores * weights) / np.sum(weights)

    return weighted_r2, r2_scores


def calculate_oof_scores(
    oof_df: pd.DataFrame, pred_col: str = "pred"
) -> Dict[str, float]:
    """
    OOF予測のスコアを計算

    Args:
        oof_df: OOFデータフレーム（sample_id, pred, targetを含む）
        pred_col: 予測値のカラム名

    Returns:
        スコア情報を含む辞書
    """
    # sample_idからimage_idを抽出
    oof_df["image_id"] = oof_df["sample_id"].str.split("__").str[0]

    # image_idごとにグループ化して5つのターゲットを並べる
    target_order = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]

    # target_nameカラムを追加
    if "target_name" not in oof_df.columns:
        oof_df["target_name"] = oof_df["sample_id"].apply(extract_target_name)

    # ピボットテーブルで5つのターゲットを列に展開
    pred_pivot = oof_df.pivot_table(
        index="image_id", columns="target_name", values=pred_col, aggfunc="first"
    )[target_order]

    target_pivot = oof_df.pivot_table(
        index="image_id", columns="target_name", values="target", aggfunc="first"
    )[target_order]

    # numpy配列に変換
    y_true = target_pivot.values
    y_pred = pred_pivot.values

    # スコア計算
    weighted_r2, r2_scores = weighted_r2_score(y_true, y_pred)

    # 結果を辞書にまとめる
    scores = {
        "weighted_r2": weighted_r2,
        "Dry_Clover_g_r2": r2_scores[0],
        "Dry_Dead_g_r2": r2_scores[1],
        "Dry_Green_g_r2": r2_scores[2],
        "Dry_Total_g_r2": r2_scores[3],
        "GDM_g_r2": r2_scores[4],
    }

    return scores


def apply_quantile_matching(
    oof_df: pd.DataFrame,
    train_df: pd.DataFrame,
    targets: List[str],
    targets_to_process: List[str] = None,
    log_transform: bool = False,
) -> pd.DataFrame:
    """
    Quantile Matchingを使用してOOF予測値の分布を訓練データの分布に一致させる

    Args:
        oof_df: OOF予測データ
        train_df: 訓練データ
        targets: ターゲット名のリスト（全ターゲット）
        targets_to_process: 処理対象のターゲット名のリスト（Noneの場合は全て）
        log_transform: 対数変換してから分布を合わせるかどうか

    Returns:
        後処理されたOOF予測データ
    """
    oof_processed = oof_df.copy()

    # target_nameカラムを追加（存在しない場合）
    if "target_name" not in oof_processed.columns:
        oof_processed["target_name"] = oof_processed["sample_id"].apply(
            extract_target_name
        )

    # 処理対象のターゲットを決定
    if targets_to_process is None:
        targets_to_process = targets

    for target_name in targets:
        print(f"\nProcessing {target_name}...")

        # このターゲットを処理するかチェック
        if target_name not in targets_to_process:
            print("  Skipped (not in targets_to_process)")
            # 元の予測値をコピー
            oof_mask = oof_processed["target_name"] == target_name
            oof_processed.loc[oof_mask, "pred_processed"] = oof_processed.loc[
                oof_mask, "pred"
            ]
            continue

        # 訓練データの該当ターゲットの値を取得
        train_target = train_df[train_df["target_name"] == target_name]["target"].values

        # OOF予測の該当ターゲットのインデックスを取得
        oof_mask = oof_processed["target_name"] == target_name
        oof_target_pred = oof_processed.loc[oof_mask, "pred"].values

        if len(oof_target_pred) == 0:
            print(f"Warning: No OOF predictions found for {target_name}")
            continue

        print(
            f"  Train samples: {len(train_target)}, OOF samples: {len(oof_target_pred)}"
        )
        print(
            f"  Train - Mean: {train_target.mean():.4f}, Std: {train_target.std():.4f}"
        )
        print(
            f"  OOF (before) - Mean: {oof_target_pred.mean():.4f}, Std: {oof_target_pred.std():.4f}"
        )

        # 対数変換オプション
        if log_transform:
            print("  Applying log transformation...")
            epsilon = 1e-6  # 0の値に対する微小値
            train_target_log = np.log1p(train_target + epsilon)
            oof_target_pred_log = np.log1p(oof_target_pred + epsilon)
            print(
                f"  Train (log) - Mean: {train_target_log.mean():.4f}, Std: {train_target_log.std():.4f}"
            )
            print(
                f"  OOF (log before) - Mean: {oof_target_pred_log.mean():.4f}, Std: {oof_target_pred_log.std():.4f}"
            )
        else:
            train_target_log = train_target
            oof_target_pred_log = oof_target_pred

        # Quantile Transformerを使用して分布を一致させる
        qt = QuantileTransformer(
            n_quantiles=min(1000, len(train_target_log)),
            output_distribution="normal",
            random_state=42,
        )

        # 訓練データで学習
        qt.fit(train_target_log.reshape(-1, 1))

        # OOF予測を正規分布に変換し、訓練データの分布で逆変換
        oof_normalized = stats.rankdata(oof_target_pred_log) / (
            len(oof_target_pred_log) + 1
        )
        oof_normalized = stats.norm.ppf(oof_normalized).reshape(-1, 1)

        # 訓練データの分布で逆変換
        oof_processed_values = qt.inverse_transform(oof_normalized).flatten()

        # 対数変換していた場合は指数変換で元に戻す
        if log_transform:
            oof_processed_values = np.exp(oof_processed_values) - epsilon

        # 負の値を0にクリップ（バイオマスは負になり得ない）
        oof_processed_values = np.clip(oof_processed_values, 0, None)

        # 更新
        oof_processed.loc[oof_mask, "pred_processed"] = oof_processed_values

        print(
            f"  OOF (after) - Mean: {oof_processed_values.mean():.4f}, Std: {oof_processed_values.std():.4f}"
        )

    return oof_processed


def apply_histogram_matching(
    oof_df: pd.DataFrame,
    train_df: pd.DataFrame,
    targets: List[str],
    targets_to_process: List[str] = None,
    log_transform: bool = False,
) -> pd.DataFrame:
    """
    ヒストグラムマッチングを使用してOOF予測値の分布を訓練データの分布に一致させる

    Args:
        oof_df: OOF予測データ
        train_df: 訓練データ
        targets: ターゲット名のリスト（全ターゲット）
        targets_to_process: 処理対象のターゲット名のリスト（Noneの場合は全て）
        log_transform: 対数変換してから分布を合わせるかどうか

    Returns:
        後処理されたOOF予測データ
    """
    oof_processed = oof_df.copy()

    # target_nameカラムを追加（存在しない場合）
    if "target_name" not in oof_processed.columns:
        oof_processed["target_name"] = oof_processed["sample_id"].apply(
            extract_target_name
        )

    # 処理対象のターゲットを決定
    if targets_to_process is None:
        targets_to_process = targets

    for target_name in targets:
        print(f"\nProcessing {target_name}...")

        # このターゲットを処理するかチェック
        if target_name not in targets_to_process:
            print(f"  Skipped (not in targets_to_process)")
            # 元の予測値をコピー
            oof_mask = oof_processed["target_name"] == target_name
            oof_processed.loc[oof_mask, "pred_processed"] = oof_processed.loc[
                oof_mask, "pred"
            ]
            continue

        # 訓練データの該当ターゲットの値を取得
        train_target = train_df[train_df["target_name"] == target_name]["target"].values

        # OOF予測の該当ターゲットのインデックスを取得
        oof_mask = oof_processed["target_name"] == target_name
        oof_target_pred = oof_processed.loc[oof_mask, "pred"].values

        if len(oof_target_pred) == 0:
            print(f"Warning: No OOF predictions found for {target_name}")
            continue

        print(
            f"  Train samples: {len(train_target)}, OOF samples: {len(oof_target_pred)}"
        )
        print(
            f"  Train - Mean: {train_target.mean():.4f}, Std: {train_target.std():.4f}"
        )
        print(
            f"  OOF (before) - Mean: {oof_target_pred.mean():.4f}, Std: {oof_target_pred.std():.4f}"
        )

        # 対数変換オプション
        if log_transform:
            print("  Applying log transformation...")
            epsilon = 1e-6  # 0の値に対する微小値
            train_target_log = np.log1p(train_target + epsilon)
            oof_target_pred_log = np.log1p(oof_target_pred + epsilon)
            print(
                f"  Train (log) - Mean: {train_target_log.mean():.4f}, Std: {train_target_log.std():.4f}"
            )
            print(
                f"  OOF (log before) - Mean: {oof_target_pred_log.mean():.4f}, Std: {oof_target_pred_log.std():.4f}"
            )
        else:
            train_target_log = train_target
            oof_target_pred_log = oof_target_pred

        # ヒストグラムマッチング
        # 訓練データとOOF予測をソート
        train_sorted = np.sort(train_target_log)
        oof_sorted_indices = np.argsort(oof_target_pred_log)

        # 訓練データの分位点をOOF予測にマッピング
        oof_processed_values = np.zeros_like(oof_target_pred_log)

        # 線形補間を使用して訓練データの分布をOOF予測にマッピング
        percentiles = np.linspace(0, 100, len(train_sorted))
        target_percentiles = np.linspace(0, 100, len(oof_target_pred_log))
        matched_values = np.interp(target_percentiles, percentiles, train_sorted)

        oof_processed_values[oof_sorted_indices] = matched_values

        # 対数変換していた場合は指数変換で元に戻す
        if log_transform:
            oof_processed_values = np.exp(oof_processed_values) - epsilon

        # 負の値を0にクリップ
        oof_processed_values = np.clip(oof_processed_values, 0, None)

        # 更新
        oof_processed.loc[oof_mask, "pred_processed"] = oof_processed_values

        print(
            f"  OOF (after) - Mean: {oof_processed_values.mean():.4f}, Std: {oof_processed_values.std():.4f}"
        )

    return oof_processed


def apply_mode_max_matching(
    oof_df: pd.DataFrame,
    train_df: pd.DataFrame,
    targets: List[str],
    targets_to_process: List[str] = None,
    log_transform: bool = False,
) -> pd.DataFrame:
    """
    最頻値と最大値を一致させる分布マッチング

    Args:
        oof_df: OOF予測データ
        train_df: 訓練データ
        targets: ターゲット名のリスト（全ターゲット）
        targets_to_process: 処理対象のターゲット名のリスト（Noneの場合は全て）
        log_transform: 対数変換してから分布を合わせるかどうか

    Returns:
        後処理されたOOF予測データ
    """
    oof_processed = oof_df.copy()

    # target_nameカラムを追加（存在しない場合）
    if "target_name" not in oof_processed.columns:
        oof_processed["target_name"] = oof_processed["sample_id"].apply(
            extract_target_name
        )

    # 処理対象のターゲットを決定
    if targets_to_process is None:
        targets_to_process = targets

    for target_name in targets:
        print(f"\nProcessing {target_name}...")

        # このターゲットを処理するかチェック
        if target_name not in targets_to_process:
            print("  Skipped (not in targets_to_process)")
            # 元の予測値をコピー
            oof_mask = oof_processed["target_name"] == target_name
            oof_processed.loc[oof_mask, "pred_processed"] = oof_processed.loc[
                oof_mask, "pred"
            ]
            continue

        # 訓練データの該当ターゲットの値を取得
        train_target = train_df[train_df["target_name"] == target_name]["target"].values

        # OOF予測の該当ターゲットのインデックスを取得
        oof_mask = oof_processed["target_name"] == target_name
        oof_target_pred = oof_processed.loc[oof_mask, "pred"].values

        if len(oof_target_pred) == 0:
            print(f"Warning: No OOF predictions found for {target_name}")
            continue

        print(
            f"  Train samples: {len(train_target)}, OOF samples: {len(oof_target_pred)}"
        )
        print(
            f"  Train - Mean: {train_target.mean():.4f}, Std: {train_target.std():.4f}"
        )
        print(
            f"  OOF (before) - Mean: {oof_target_pred.mean():.4f}, Std: {oof_target_pred.std():.4f}"
        )

        # 対数変換オプション
        if log_transform:
            print("  Applying log transformation...")
            epsilon = 1e-6  # 0の値に対する微小値
            train_target_log = np.log1p(train_target + epsilon)
            oof_target_pred_log = np.log1p(oof_target_pred + epsilon)
            print(
                f"  Train (log) - Mean: {train_target_log.mean():.4f}, Std: {train_target_log.std():.4f}"
            )
            print(
                f"  OOF (log before) - Mean: {oof_target_pred_log.mean():.4f}, Std: {oof_target_pred_log.std():.4f}"
            )
        else:
            train_target_log = train_target
            oof_target_pred_log = oof_target_pred

        # 最頻値と最大値を計算
        # 最頻値はヒストグラムのビンで最も頻度の高い値の中心値を使用
        train_hist, train_bins = np.histogram(train_target_log, bins=50)
        train_mode_idx = np.argmax(train_hist)
        train_mode = (train_bins[train_mode_idx] + train_bins[train_mode_idx + 1]) / 2
        train_max = train_target_log.max()

        oof_hist, oof_bins = np.histogram(oof_target_pred_log, bins=50)
        oof_mode_idx = np.argmax(oof_hist)
        oof_mode = (oof_bins[oof_mode_idx] + oof_bins[oof_mode_idx + 1]) / 2
        oof_max = oof_target_pred_log.max()

        print(f"  Train - Mode: {train_mode:.4f}, Max: {train_max:.4f}")
        print(f"  OOF (before) - Mode: {oof_mode:.4f}, Max: {oof_max:.4f}")

        # 最頻値と最大値を一致させる区分線形変換
        oof_processed_values = np.zeros_like(oof_target_pred_log)

        for i, val in enumerate(oof_target_pred_log):
            if val <= oof_mode:
                # 最頻値以下: 0からmodeまでをスケーリング
                if oof_mode > 0:
                    scale = train_mode / oof_mode
                    oof_processed_values[i] = val * scale
                else:
                    oof_processed_values[i] = val
            else:
                # 最頻値より大: modeからmaxまでをスケーリング
                if oof_max > oof_mode:
                    scale = (train_max - train_mode) / (oof_max - oof_mode)
                    oof_processed_values[i] = train_mode + (val - oof_mode) * scale
                else:
                    oof_processed_values[i] = train_mode

        # 対数変換していた場合は指数変換で元に戻す
        if log_transform:
            oof_processed_values = np.exp(oof_processed_values) - epsilon

        # 負の値を0にクリップ
        oof_processed_values = np.clip(oof_processed_values, 0, None)

        # 更新
        oof_processed.loc[oof_mask, "pred_processed"] = oof_processed_values

        print(
            f"  OOF (after) - Mean: {oof_processed_values.mean():.4f}, Std: {oof_processed_values.std():.4f}"
        )

    return oof_processed


def apply_min_max_matching(
    oof_df: pd.DataFrame,
    train_df: pd.DataFrame,
    targets: List[str],
    targets_to_process: List[str] = None,
    log_transform: bool = False,
) -> pd.DataFrame:
    """
    最小値と最大値を一致させる線形変換による分布マッチング

    Args:
        oof_df: OOF予測データ
        train_df: 訓練データ
        targets: ターゲット名のリスト（全ターゲット）
        targets_to_process: 処理対象のターゲット名のリスト（Noneの場合は全て）
        log_transform: 対数変換してから分布を合わせるかどうか

    Returns:
        後処理されたOOF予測データ
    """
    oof_processed = oof_df.copy()

    # target_nameカラムを追加（存在しない場合）
    if "target_name" not in oof_processed.columns:
        oof_processed["target_name"] = oof_processed["sample_id"].apply(
            extract_target_name
        )

    # 処理対象のターゲットを決定
    if targets_to_process is None:
        targets_to_process = targets

    for target_name in targets:
        print(f"\nProcessing {target_name}...")

        # このターゲットを処理するかチェック
        if target_name not in targets_to_process:
            print("  Skipped (not in targets_to_process)")
            # 元の予測値をコピー
            oof_mask = oof_processed["target_name"] == target_name
            oof_processed.loc[oof_mask, "pred_processed"] = oof_processed.loc[
                oof_mask, "pred"
            ]
            continue

        # 訓練データの該当ターゲットの値を取得
        train_target = train_df[train_df["target_name"] == target_name]["target"].values

        # OOF予測の該当ターゲットのインデックスを取得
        oof_mask = oof_processed["target_name"] == target_name
        oof_target_pred = oof_processed.loc[oof_mask, "pred"].values

        if len(oof_target_pred) == 0:
            print(f"Warning: No OOF predictions found for {target_name}")
            continue

        print(
            f"  Train samples: {len(train_target)}, OOF samples: {len(oof_target_pred)}"
        )
        print(
            f"  Train - Mean: {train_target.mean():.4f}, Std: {train_target.std():.4f}"
        )
        print(
            f"  OOF (before) - Mean: {oof_target_pred.mean():.4f}, Std: {oof_target_pred.std():.4f}"
        )

        # 対数変換オプション
        if log_transform:
            print("  Applying log transformation...")
            epsilon = 1e-6  # 0の値に対する微小値
            train_target_log = np.log1p(train_target + epsilon)
            oof_target_pred_log = np.log1p(oof_target_pred + epsilon)
            print(
                f"  Train (log) - Mean: {train_target_log.mean():.4f}, Std: {train_target_log.std():.4f}"
            )
            print(
                f"  OOF (log before) - Mean: {oof_target_pred_log.mean():.4f}, Std: {oof_target_pred_log.std():.4f}"
            )
        else:
            train_target_log = train_target
            oof_target_pred_log = oof_target_pred

        # 最小値と最大値を計算
        train_min = train_target_log.min()
        train_max = train_target_log.max()
        oof_min = oof_target_pred_log.min()
        oof_max = oof_target_pred_log.max()

        print(f"  Train - Min: {train_min:.4f}, Max: {train_max:.4f}")
        print(f"  OOF (before) - Min: {oof_min:.4f}, Max: {oof_max:.4f}")

        # 線形変換: y = a * x + b
        # train_min = a * oof_min + b
        # train_max = a * oof_max + b
        # → a = (train_max - train_min) / (oof_max - oof_min)
        # → b = train_min - a * oof_min

        if oof_max > oof_min:
            a = (train_max - train_min) / (oof_max - oof_min)
            b = train_min - a * oof_min
            oof_processed_values = a * oof_target_pred_log + b
        else:
            # OOFの値がすべて同じ場合は訓練データの中央値に設定
            oof_processed_values = np.full_like(
                oof_target_pred_log, (train_min + train_max) / 2
            )

        # 対数変換していた場合は指数変換で元に戻す
        if log_transform:
            oof_processed_values = np.exp(oof_processed_values) - epsilon

        # 負の値を0にクリップ
        oof_processed_values = np.clip(oof_processed_values, 0, None)

        # 更新
        oof_processed.loc[oof_mask, "pred_processed"] = oof_processed_values

        print(
            f"  OOF (after) - Mean: {oof_processed_values.mean():.4f}, Std: {oof_processed_values.std():.4f}"
        )

    return oof_processed


def visualize_predictions(
    oof_df: pd.DataFrame,
    train_df: pd.DataFrame,
    targets: List[str],
    output_dir: Path,
) -> None:
    """
    予測値と真の値、訓練データの分布を可視化

    Args:
        oof_df: OOFデータフレーム（pred, pred_processed, targetを含む）
        train_df: 訓練データフレーム
        targets: ターゲット名のリスト
        output_dir: 可視化結果の保存先ディレクトリ
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # target_nameカラムを追加（存在しない場合）
    if "target_name" not in oof_df.columns:
        oof_df["target_name"] = oof_df["sample_id"].apply(extract_target_name)

    # 全ターゲットの可視化
    n_targets = len(targets)
    fig, axes = plt.subplots(n_targets, 6, figsize=(36, 5 * n_targets))

    if n_targets == 1:
        axes = axes.reshape(1, -1)

    for idx, target_name in enumerate(targets):
        # データを抽出
        train_target = train_df[train_df["target_name"] == target_name]["target"].values
        oof_mask = oof_df["target_name"] == target_name
        oof_target = oof_df.loc[oof_mask, "target"].values
        oof_pred_before = oof_df.loc[oof_mask, "pred"].values
        oof_pred_after = oof_df.loc[oof_mask, "pred_processed"].values

        # 1列目: 散布図（処理前）
        ax1 = axes[idx, 0]
        ax1.scatter(oof_target, oof_pred_before, alpha=0.5, s=20)
        min_val = min(oof_target.min(), oof_pred_before.min())
        max_val = max(oof_target.max(), oof_pred_before.max())
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        ax1.set_xlabel("True Value", fontsize=10)
        ax1.set_ylabel("Predicted Value (Before)", fontsize=10)
        ax1.set_title(f"{target_name}\nBefore Post-processing", fontsize=11)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # R2スコアを表示
        ss_res = np.sum((oof_target - oof_pred_before) ** 2)
        ss_tot = np.sum((oof_target - np.mean(oof_target)) ** 2)
        r2_before = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax1.text(
            0.05,
            0.95,
            f"R² = {r2_before:.4f}",
            transform=ax1.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 2列目: 散布図（処理後）
        ax2 = axes[idx, 1]
        ax2.scatter(oof_target, oof_pred_after, alpha=0.5, s=20, color="orange")
        min_val = min(oof_target.min(), oof_pred_after.min())
        max_val = max(oof_target.max(), oof_pred_after.max())
        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        ax2.set_xlabel("True Value", fontsize=10)
        ax2.set_ylabel("Predicted Value (After)", fontsize=10)
        ax2.set_title(f"{target_name}\nAfter Post-processing", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # R2スコアを表示
        ss_res = np.sum((oof_target - oof_pred_after) ** 2)
        ss_tot = np.sum((oof_target - np.mean(oof_target)) ** 2)
        r2_after = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ax2.text(
            0.05,
            0.95,
            f"R² = {r2_after:.4f}",
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 3列目: 分布のヒストグラム（Train vs Before）
        ax3 = axes[idx, 2]
        bins = np.linspace(
            min(train_target.min(), oof_pred_before.min()),
            max(train_target.max(), oof_pred_before.max()),
            30,
        )
        ax3.hist(
            train_target,
            bins=bins,
            alpha=0.6,
            label="Train (True)",
            color="green",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax3.hist(
            oof_pred_before,
            bins=bins,
            alpha=0.6,
            label="OOF (Before)",
            color="blue",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax3.set_xlabel("Value", fontsize=10)
        ax3.set_ylabel("Density", fontsize=10)
        ax3.set_title(f"{target_name}\nTrain vs Before", fontsize=11)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis="y")

        # 統計情報を表示
        stats_text = f"Train: μ={train_target.mean():.2f}, σ={train_target.std():.2f}\n"
        stats_text += (
            f"Before: μ={oof_pred_before.mean():.2f}, σ={oof_pred_before.std():.2f}"
        )
        ax3.text(
            0.98,
            0.98,
            stats_text,
            transform=ax3.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 4列目: 分布のヒストグラム（Train vs After）
        ax4 = axes[idx, 3]
        bins = np.linspace(
            min(train_target.min(), oof_pred_after.min()),
            max(train_target.max(), oof_pred_after.max()),
            30,
        )
        ax4.hist(
            train_target,
            bins=bins,
            alpha=0.6,
            label="Train (True)",
            color="green",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax4.hist(
            oof_pred_after,
            bins=bins,
            alpha=0.6,
            label="OOF (After)",
            color="orange",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax4.set_xlabel("Value", fontsize=10)
        ax4.set_ylabel("Density", fontsize=10)
        ax4.set_title(f"{target_name}\nTrain vs After", fontsize=11)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

        # 統計情報を表示
        stats_text = f"Train: μ={train_target.mean():.2f}, σ={train_target.std():.2f}\n"
        stats_text += (
            f"After: μ={oof_pred_after.mean():.2f}, σ={oof_pred_after.std():.2f}"
        )
        ax4.text(
            0.98,
            0.98,
            stats_text,
            transform=ax4.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 5列目: 対数スケールのヒストグラム（Train vs Before）
        ax5 = axes[idx, 4]
        epsilon = 1e-6
        train_log = np.log1p(train_target + epsilon)
        before_log = np.log1p(oof_pred_before + epsilon)
        bins_log = np.linspace(
            min(train_log.min(), before_log.min()),
            max(train_log.max(), before_log.max()),
            30,
        )
        ax5.hist(
            train_log,
            bins=bins_log,
            alpha=0.6,
            label="Train (True)",
            color="green",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax5.hist(
            before_log,
            bins=bins_log,
            alpha=0.6,
            label="OOF (Before)",
            color="blue",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax5.set_xlabel("Log(Value)", fontsize=10)
        ax5.set_ylabel("Density", fontsize=10)
        ax5.set_title(f"{target_name}\nLog: Train vs Before", fontsize=11)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis="y")

        # 統計情報を表示
        stats_text = f"Train (log): μ={train_log.mean():.2f}, σ={train_log.std():.2f}\n"
        stats_text += (
            f"Before (log): μ={before_log.mean():.2f}, σ={before_log.std():.2f}"
        )
        ax5.text(
            0.98,
            0.98,
            stats_text,
            transform=ax5.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 6列目: 対数スケールのヒストグラム（Train vs After）
        ax6 = axes[idx, 5]
        after_log = np.log1p(oof_pred_after + epsilon)
        bins_log = np.linspace(
            min(train_log.min(), after_log.min()),
            max(train_log.max(), after_log.max()),
            30,
        )
        ax6.hist(
            train_log,
            bins=bins_log,
            alpha=0.6,
            label="Train (True)",
            color="green",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax6.hist(
            after_log,
            bins=bins_log,
            alpha=0.6,
            label="OOF (After)",
            color="orange",
            density=True,
            edgecolor="black",
            linewidth=0.5,
        )
        ax6.set_xlabel("Log(Value)", fontsize=10)
        ax6.set_ylabel("Density", fontsize=10)
        ax6.set_title(f"{target_name}\nLog: Train vs After", fontsize=11)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3, axis="y")

        # 統計情報を表示
        stats_text = f"Train (log): μ={train_log.mean():.2f}, σ={train_log.std():.2f}\n"
        stats_text += f"After (log): μ={after_log.mean():.2f}, σ={after_log.std():.2f}"
        ax6.text(
            0.98,
            0.98,
            stats_text,
            transform=ax6.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    output_path = output_dir / "prediction_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

    # 個別のターゲットごとの詳細プロット
    for target_name in targets:
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))

        # データを抽出
        train_target = train_df[train_df["target_name"] == target_name]["target"].values
        oof_mask = oof_df["target_name"] == target_name
        oof_target = oof_df.loc[oof_mask, "target"].values
        oof_pred_before = oof_df.loc[oof_mask, "pred"].values
        oof_pred_after = oof_df.loc[oof_mask, "pred_processed"].values

        # 左上: 散布図（処理前）
        ax1 = axes[0, 0]
        ax1.scatter(
            oof_target,
            oof_pred_before,
            alpha=0.6,
            s=30,
            edgecolors="black",
            linewidths=0.5,
        )
        min_val = min(oof_target.min(), oof_pred_before.min())
        max_val = max(oof_target.max(), oof_pred_before.max())
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        ax1.set_xlabel("True Value", fontsize=12)
        ax1.set_ylabel("Predicted Value (Before)", fontsize=12)
        ax1.set_title("Before Post-processing", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ss_res = np.sum((oof_target - oof_pred_before) ** 2)
        ss_tot = np.sum((oof_target - np.mean(oof_target)) ** 2)
        r2_before = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae_before = np.mean(np.abs(oof_target - oof_pred_before))
        rmse_before = np.sqrt(np.mean((oof_target - oof_pred_before) ** 2))

        stats_text = (
            f"R² = {r2_before:.4f}\nMAE = {mae_before:.4f}\nRMSE = {rmse_before:.4f}"
        )
        ax1.text(
            0.05,
            0.95,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # 右上: 散布図（処理後）
        ax2 = axes[0, 1]
        ax2.scatter(
            oof_target,
            oof_pred_after,
            alpha=0.6,
            s=30,
            color="orange",
            edgecolors="black",
            linewidths=0.5,
        )
        min_val = min(oof_target.min(), oof_pred_after.min())
        max_val = max(oof_target.max(), oof_pred_after.max())
        ax2.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect prediction",
        )
        ax2.set_xlabel("True Value", fontsize=12)
        ax2.set_ylabel("Predicted Value (After)", fontsize=12)
        ax2.set_title("After Post-processing", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        ss_res = np.sum((oof_target - oof_pred_after) ** 2)
        ss_tot = np.sum((oof_target - np.mean(oof_target)) ** 2)
        r2_after = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae_after = np.mean(np.abs(oof_target - oof_pred_after))
        rmse_after = np.sqrt(np.mean((oof_target - oof_pred_after) ** 2))

        stats_text = (
            f"R² = {r2_after:.4f}\nMAE = {mae_after:.4f}\nRMSE = {rmse_after:.4f}"
        )
        ax2.text(
            0.05,
            0.95,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # 右上: 残差プロット
        ax3 = axes[0, 2]
        residuals_before = oof_target - oof_pred_before
        residuals_after = oof_target - oof_pred_after
        ax3.scatter(
            oof_pred_before,
            residuals_before,
            alpha=0.6,
            s=30,
            label="Before",
            edgecolors="black",
            linewidths=0.5,
        )
        ax3.scatter(
            oof_pred_after,
            residuals_after,
            alpha=0.6,
            s=30,
            color="orange",
            label="After",
            edgecolors="black",
            linewidths=0.5,
        )
        ax3.axhline(y=0, color="r", linestyle="--", lw=2)
        ax3.set_xlabel("Predicted Value", fontsize=12)
        ax3.set_ylabel("Residuals (True - Pred)", fontsize=12)
        ax3.set_title("Residual Plot", fontsize=14, fontweight="bold")
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 左下: 分布のヒストグラム（Train vs Before）
        ax4 = axes[1, 0]
        bins = 40
        ax4.hist(
            train_target,
            bins=bins,
            alpha=0.6,
            label="Train (True)",
            color="green",
            density=True,
            edgecolor="black",
        )
        ax4.hist(
            oof_pred_before,
            bins=bins,
            alpha=0.6,
            label="OOF (Before)",
            color="blue",
            density=True,
            edgecolor="black",
        )
        ax4.set_xlabel("Value", fontsize=12)
        ax4.set_ylabel("Density", fontsize=12)
        ax4.set_title("Distribution: Train vs Before", fontsize=14, fontweight="bold")
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis="y")

        # 統計情報を表示
        stats_text = f"Train: μ={train_target.mean():.2f}, σ={train_target.std():.2f}\n"
        stats_text += (
            f"Before: μ={oof_pred_before.mean():.2f}, σ={oof_pred_before.std():.2f}"
        )
        ax4.text(
            0.98,
            0.98,
            stats_text,
            transform=ax4.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 中下: 分布のヒストグラム（Train vs After）
        ax5 = axes[1, 1]
        ax5.hist(
            train_target,
            bins=bins,
            alpha=0.6,
            label="Train (True)",
            color="green",
            density=True,
            edgecolor="black",
        )
        ax5.hist(
            oof_pred_after,
            bins=bins,
            alpha=0.6,
            label="OOF (After)",
            color="orange",
            density=True,
            edgecolor="black",
        )
        ax5.set_xlabel("Value", fontsize=12)
        ax5.set_ylabel("Density", fontsize=12)
        ax5.set_title("Distribution: Train vs After", fontsize=14, fontweight="bold")
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis="y")

        # 統計情報を表示
        stats_text = f"Train: μ={train_target.mean():.2f}, σ={train_target.std():.2f}\n"
        stats_text += (
            f"After: μ={oof_pred_after.mean():.2f}, σ={oof_pred_after.std():.2f}"
        )
        ax5.text(
            0.98,
            0.98,
            stats_text,
            transform=ax5.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 右下: Q-Qプロット（補足）
        ax6 = axes[1, 2]
        from scipy import stats as scipy_stats

        # Before のQ-Qプロット
        sorted_train = np.sort(train_target)
        sorted_before = np.sort(oof_pred_before)
        quantiles = np.linspace(0, 100, min(len(sorted_train), len(sorted_before)))
        train_quantiles = np.percentile(sorted_train, quantiles)
        before_quantiles = np.percentile(sorted_before, quantiles)
        ax6.scatter(
            train_quantiles,
            before_quantiles,
            alpha=0.6,
            s=20,
            label="Before",
            edgecolors="black",
            linewidths=0.5,
        )

        # After のQ-Qプロット
        sorted_after = np.sort(oof_pred_after)
        quantiles_after = np.linspace(0, 100, min(len(sorted_train), len(sorted_after)))
        train_quantiles_after = np.percentile(sorted_train, quantiles_after)
        after_quantiles = np.percentile(sorted_after, quantiles_after)
        ax6.scatter(
            train_quantiles_after,
            after_quantiles,
            alpha=0.6,
            s=20,
            color="orange",
            label="After",
            edgecolors="black",
            linewidths=0.5,
        )

        # 対角線
        min_val = min(
            train_quantiles.min(), before_quantiles.min(), after_quantiles.min()
        )
        max_val = max(
            train_quantiles.max(), before_quantiles.max(), after_quantiles.max()
        )
        ax6.plot(
            [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect match"
        )
        ax6.set_xlabel("Train Quantiles", fontsize=12)
        ax6.set_ylabel("OOF Quantiles", fontsize=12)
        ax6.set_title("Q-Q Plot", fontsize=14, fontweight="bold")
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)

        # 3行目: 対数スケールのヒストグラム
        # 左下: 対数ヒストグラム（Train vs Before）
        ax7 = axes[2, 0]
        epsilon = 1e-6
        train_log = np.log1p(train_target + epsilon)
        before_log = np.log1p(oof_pred_before + epsilon)
        after_log = np.log1p(oof_pred_after + epsilon)

        bins_log = 40
        ax7.hist(
            train_log,
            bins=bins_log,
            alpha=0.6,
            label="Train (True)",
            color="green",
            density=True,
            edgecolor="black",
        )
        ax7.hist(
            before_log,
            bins=bins_log,
            alpha=0.6,
            label="OOF (Before)",
            color="blue",
            density=True,
            edgecolor="black",
        )
        ax7.set_xlabel("Log(Value)", fontsize=12)
        ax7.set_ylabel("Density", fontsize=12)
        ax7.set_title(
            "Log Distribution: Train vs Before", fontsize=14, fontweight="bold"
        )
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3, axis="y")

        # 統計情報を表示
        stats_text = f"Train (log): μ={train_log.mean():.2f}, σ={train_log.std():.2f}\n"
        stats_text += (
            f"Before (log): μ={before_log.mean():.2f}, σ={before_log.std():.2f}"
        )
        ax7.text(
            0.98,
            0.98,
            stats_text,
            transform=ax7.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 中下: 対数ヒストグラム（Train vs After）
        ax8 = axes[2, 1]
        ax8.hist(
            train_log,
            bins=bins_log,
            alpha=0.6,
            label="Train (True)",
            color="green",
            density=True,
            edgecolor="black",
        )
        ax8.hist(
            after_log,
            bins=bins_log,
            alpha=0.6,
            label="OOF (After)",
            color="orange",
            density=True,
            edgecolor="black",
        )
        ax8.set_xlabel("Log(Value)", fontsize=12)
        ax8.set_ylabel("Density", fontsize=12)
        ax8.set_title(
            "Log Distribution: Train vs After", fontsize=14, fontweight="bold"
        )
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3, axis="y")

        # 統計情報を表示
        stats_text = f"Train (log): μ={train_log.mean():.2f}, σ={train_log.std():.2f}\n"
        stats_text += f"After (log): μ={after_log.mean():.2f}, σ={after_log.std():.2f}"
        ax8.text(
            0.98,
            0.98,
            stats_text,
            transform=ax8.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # 右下: 対数スケールでの散布図（Before vs After）
        ax9 = axes[2, 2]
        ax9.scatter(
            before_log,
            train_log,
            alpha=0.5,
            s=20,
            label="Before",
            edgecolors="black",
            linewidths=0.5,
        )
        ax9.scatter(
            after_log,
            train_log,
            alpha=0.5,
            s=20,
            color="orange",
            label="After",
            edgecolors="black",
            linewidths=0.5,
        )
        min_val = min(train_log.min(), before_log.min(), after_log.min())
        max_val = max(train_log.max(), before_log.max(), after_log.max())
        ax9.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
        ax9.set_xlabel("Log(Predicted)", fontsize=12)
        ax9.set_ylabel("Log(True)", fontsize=12)
        ax9.set_title("Log-Scale Scatter Plot", fontsize=14, fontweight="bold")
        ax9.legend(fontsize=10)
        ax9.grid(True, alpha=0.3)

        fig.suptitle(
            f"{target_name} - Detailed Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        output_path = output_dir / f"{target_name}_detailed.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Detailed visualization for {target_name} saved to: {output_path}")
        plt.close()


def run_post_processing(config: PostProcessConfig) -> None:
    """
    設定に基づいて後処理を実行

    Args:
        config: 後処理の設定
    """
    # 設定を表示
    config.print_config()

    # データ読み込み
    print("\n" + "=" * 80)
    print("Loading data...")
    print("=" * 80)
    oof_df = load_oof_predictions(config.exp_dir)
    train_df = load_train_data(config.train_path)

    # ターゲット名のリストを取得
    if "target_name" in train_df.columns:
        targets = train_df["target_name"].unique().tolist()
    else:
        # target_nameカラムがない場合、推定
        targets = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]

    print(f"\nTargets: {targets}")

    # 処理対象のターゲットを決定
    targets_to_process = config.targets if config.targets else targets
    if config.targets:
        # 指定されたターゲットが有効かチェック
        invalid_targets = [t for t in targets_to_process if t not in targets]
        if invalid_targets:
            print(f"\nWarning: Invalid target names: {invalid_targets}")
            print(f"Valid targets are: {targets}")
            targets_to_process = [t for t in targets_to_process if t in targets]

        if not targets_to_process:
            raise ValueError("No valid targets to process")

    print(f"Targets to process: {targets_to_process}")

    if config.log_transform:
        print("Log transformation: ENABLED")
    else:
        print("Log transformation: DISABLED")

    # 分布マッチングを適用
    print("\n" + "=" * 80)
    print(f"Applying {config.method} matching...")
    print("=" * 80)

    if config.method == "quantile":
        oof_processed = apply_quantile_matching(
            oof_df, train_df, targets, targets_to_process, config.log_transform
        )
    elif config.method == "histogram":
        oof_processed = apply_histogram_matching(
            oof_df, train_df, targets, targets_to_process, config.log_transform
        )
    elif config.method == "mode_max":
        oof_processed = apply_mode_max_matching(
            oof_df, train_df, targets, targets_to_process, config.log_transform
        )
    elif config.method == "min_max":
        oof_processed = apply_min_max_matching(
            oof_df, train_df, targets, targets_to_process, config.log_transform
        )
    else:
        raise ValueError(f"Unknown method: {config.method}")

    # 結果を保存
    print("\n" + "=" * 80)
    print(f"Saving processed predictions to {config.output_path}")
    print("=" * 80)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    oof_processed.to_csv(config.output_path, index=False)
    print(f"Saved {len(oof_processed)} samples")

    # 統計情報を表示
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    for target_name in targets:
        train_target = train_df[train_df["target_name"] == target_name]["target"]
        oof_mask = oof_processed["target_name"] == target_name
        oof_before = oof_processed.loc[oof_mask, "pred"]
        oof_after = oof_processed.loc[oof_mask, "pred_processed"]

        print(f"\n{target_name}:")
        print(
            f"  Train        - Mean: {train_target.mean():.4f}, Std: {train_target.std():.4f}, "
            f"Min: {train_target.min():.4f}, Max: {train_target.max():.4f}"
        )
        print(
            f"  OOF (before) - Mean: {oof_before.mean():.4f}, Std: {oof_before.std():.4f}, "
            f"Min: {oof_before.min():.4f}, Max: {oof_before.max():.4f}"
        )
        print(
            f"  OOF (after)  - Mean: {oof_after.mean():.4f}, Std: {oof_after.std():.4f}, "
            f"Min: {oof_after.min():.4f}, Max: {oof_after.max():.4f}"
        )

    # スコア計算
    print("\n" + "=" * 80)
    print("Score Calculation")
    print("=" * 80)

    # 後処理前のスコア
    print("\n[Before Post-processing]")
    scores_before = calculate_oof_scores(oof_processed, pred_col="pred")
    print(f"Weighted R2 Score: {scores_before['weighted_r2']:.6f}")
    print("\nIndividual R2 Scores:")
    target_names = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]
    weights = [0.1, 0.1, 0.1, 0.2, 0.5]
    for name, weight in zip(target_names, weights):
        r2_key = f"{name}_r2"
        print(f"  {name:15s} (weight={weight:.1f}): R2 = {scores_before[r2_key]:.6f}")

    # 後処理後のスコア
    print("\n[After Post-processing]")
    scores_after = calculate_oof_scores(oof_processed, pred_col="pred_processed")
    print(f"Weighted R2 Score: {scores_after['weighted_r2']:.6f}")
    print("\nIndividual R2 Scores:")
    for name, weight in zip(target_names, weights):
        r2_key = f"{name}_r2"
        print(f"  {name:15s} (weight={weight:.1f}): R2 = {scores_after[r2_key]:.6f}")

    # スコア改善
    score_diff = scores_after["weighted_r2"] - scores_before["weighted_r2"]
    print(f"\n{'=' * 80}")
    print(f"Score Improvement: {score_diff:+.6f}")
    if score_diff > 0:
        print("✓ Post-processing improved the score!")
    elif score_diff < 0:
        print("✗ Post-processing decreased the score.")
    else:
        print("→ No change in score.")
    print("=" * 80)

    # 可視化
    if config.visualize:
        print("\n" + "=" * 80)
        print("Generating visualizations...")
        print("=" * 80)

        visualize_predictions(oof_processed, train_df, targets, config.viz_dir)
        print("\nVisualization completed!")


if __name__ == "__main__":
    config = PostProcessConfig(
        exp_dir="/kaggle/working/exp_012_004",
        method="quantile",
        targets=None,
        log_transform=True,
        visualize=True,
    )
    run_post_processing(config)
