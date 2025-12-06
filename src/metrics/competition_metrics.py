from typing import Dict

import numpy as np
import pandas as pd
import torch


def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: shape (N, 5)
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


def weighted_r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    y_true, y_pred: shape (N, 5)
    """
    weights = torch.tensor([0.1, 0.1, 0.1, 0.2, 0.5], device=y_true.device)
    r2_scores = []
    for i in range(5):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        ss_res = torch.sum((y_t - y_p) ** 2)
        ss_tot = torch.sum((y_t - torch.mean(y_t)) ** 2)
        r2 = (
            1 - ss_res / ss_tot
            if ss_tot > 0
            else torch.tensor(0.0, device=y_true.device)
        )
        r2_scores.append(r2)
    r2_scores = torch.stack(r2_scores)
    weighted_r2 = torch.sum(r2_scores * weights) / torch.sum(weights)
    return weighted_r2, r2_scores


class CompetitionMetrics:
    def __init__(self):
        pass

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> Dict[str, float]:
        y_true = y_true.clone()
        y_pred = y_pred.clone()
        metrics_value, r2_scores = weighted_r2_score_torch(y_true, y_pred)
        metrics_dict = {
            "weighted_r2": float(metrics_value.numpy()),
            "r2_Dry_Green_g": float(r2_scores[0].numpy()),
            "r2_Dry_Dead_g": float(r2_scores[1].numpy()),
            "r2_Dry_Clover_g": float(r2_scores[2].numpy()),
            "r2_GDM_g": float(r2_scores[3].numpy()),
            "r2_Dry_Total_g": float(r2_scores[4].numpy()),
        }
        return metrics_dict

    def calculate_diff(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        y_true = y_true.clone()
        y_pred = y_pred.clone()
        diffs = y_true - y_pred
        return diffs


def calulate_category_metrics(
    df: pd.DataFrame,
    category: str,
    target_name_list: list[str],
) -> Dict[str, float]:
    metrics_dict: Dict[str, float] = {}
    for category_name in df[category].unique():
        category_df = df[df[category] == category_name]
        y_true_category_name = np.zeros((len(category_df) // 5, 5))
        y_pred_category_name = np.zeros((len(category_df) // 5, 5))
        for i, name in enumerate(target_name_list):
            target_name_core = name.replace("Dry_", "").replace("_g", "")
            y_true_category_name[:, i] = category_df["target"][
                category_df["target_name"] == name
            ]
            y_pred_category_name[:, i] = category_df["pred"][
                category_df["target_name"] == name
            ]
            category_name_score, category_name_r2_scores = weighted_r2_score(
                y_true_category_name, y_pred_category_name
            )
        metrics_dict[f"{category}/{category_name} weighted_r2"] = category_name_score
        for idx, name in enumerate(target_name_list):
            target_name_core = name.replace("Dry_", "").replace("_g", "")
            metrics_dict[f"{category}_{category_name}/{target_name_core}"] = (
                category_name_r2_scores[idx]
            )
    return metrics_dict


def calculate_custom_metric(
    oof_df: pd.DataFrame, valid_df: pd.DataFrame
) -> Dict[str, float]:
    target_name_list = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ]
    # train_dfとmerge
    oof_df = oof_df.drop(columns=["target"])
    valid_df = pd.concat(
        [
            valid_df,
            oof_df,
        ],
        axis=1,
    )
    valid_df["Month"] = pd.to_datetime(valid_df["Sampling_Date"]).dt.month
    valid_df["season"] = valid_df["Month"] % 12 // 3 + 1  # 1:冬, 2:春, 3:夏, 4:秋
    valid_df["season"] = valid_df["season"].map(
        {1: "winter", 2: "spring", 3: "summer", 4: "autumn"}
    )
    metrics_dict: Dict[str, float] = {}
    # State別スコア計算
    metrics_dict.update(calulate_category_metrics(valid_df, "State", target_name_list))
    # Month別スコア計算
    metrics_dict.update(calulate_category_metrics(valid_df, "Month", target_name_list))
    # Species別スコア計算
    metrics_dict.update(
        calulate_category_metrics(valid_df, "Species", target_name_list)
    )
    metrics_dict.update(calulate_category_metrics(valid_df, "season", target_name_list))

    return metrics_dict


if __name__ == "__main__":
    # y_true = torch.tensor(
    #     [
    #         [10.0, 20.0, 30.0, 40.0, 50.0],
    #         [15.0, 25.0, 35.0, 45.0, 55.0],
    #         [20.0, 30.0, 40.0, 50.0, 60.0],
    #     ]
    # )
    # y_pred = torch.tensor(
    #     [
    #         [12.0, 18.0, 29.0, 41.0, 52.0],
    #         [14.0, 27.0, 33.0, 44.0, 57.0],
    #         [19.0, 31.0, 39.0, 49.0, 61.0],
    #     ]
    # )

    # metric = CompetitionMetrics()
    # score = metric(y_true, y_pred)
    # diffs = metric.calculate_diff(y_true, y_pred)
    # print(score)
    # print(diffs)
    from pathlib import Path

    import yaml  # type: ignore

    df = pd.read_csv("/kaggle/input/csiro-biomass/train.csv")
    fold = 0
    exp_dir = Path("/kaggle/working/debug")
    splits_dir = Path("/kaggle/working/splits")
    valid_ids_path = splits_dir / f"fold_{fold}" / "valid.yaml"
    with open(valid_ids_path, "r") as f:
        valid_ids = yaml.safe_load(f)
    valid_df = df[df["sample_id"].isin(valid_ids)].reset_index(drop=True)
    oof_df = pd.read_csv(exp_dir / f"fold_{fold}" / "best_oof.csv")
    custom_metrics = calculate_custom_metric(oof_df, valid_df)
    print(custom_metrics)
