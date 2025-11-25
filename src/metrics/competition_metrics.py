from typing import Dict

import numpy as np
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
            "weighted_r2": metrics_value.numpy(),
            "r2_Dry_Clover_g": r2_scores[0].numpy(),
            "r2_Dry_Dead_g": r2_scores[1].numpy(),
            "r2_Dry_Green_g": r2_scores[2].numpy(),
            "r2_Dry_Total_g": r2_scores[3].numpy(),
            "r2_GDM_g": r2_scores[4].numpy(),
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


if __name__ == "__main__":
    y_true = torch.tensor(
        [
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [15.0, 25.0, 35.0, 45.0, 55.0],
            [20.0, 30.0, 40.0, 50.0, 60.0],
        ]
    )
    y_pred = torch.tensor(
        [
            [12.0, 18.0, 29.0, 41.0, 52.0],
            [14.0, 27.0, 33.0, 44.0, 57.0],
            [19.0, 31.0, 39.0, 49.0, 61.0],
        ]
    )

    metric = CompetitionMetrics()
    score = metric(y_true, y_pred)
    diffs = metric.calculate_diff(y_true, y_pred)
    print(score)
    print(diffs)
