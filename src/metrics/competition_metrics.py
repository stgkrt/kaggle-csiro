from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score


def compute_f1_scores(self, sol, sub, target_gestures):
    # Compute binary F1 (Target vs Non-Target)
    y_true_bin = sol["gesture"].isin(target_gestures).values
    y_pred_bin = sub["gesture"].isin(target_gestures).values
    f1_binary = f1_score(
        y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average="binary"
    )

    # Build multi-class labels for gestures
    y_true_mc = sol["gesture"].apply(
        lambda x: x if x in self.target_gestures else "non_target"
    )
    y_pred_mc = sub["gesture"].apply(
        lambda x: x if x in self.target_gestures else "non_target"
    )

    # Compute macro F1 over all gesture classes
    f1_macro = f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)

    return 0.5 * f1_binary + 0.5 * f1_macro


class CompetitionMetrics:
    def __init__(self, inverse_gesture_dict_path: Path) -> None:
        self.target_gestures = [
            "Above ear - pull hair",
            "Cheek - pinch skin",
            "Eyebrow - pull hair",
            "Eyelash - pull hair",
            "Forehead - pull hairline",
            "Forehead - scratch",
            "Neck - pinch skin",
            "Neck - scratch",
        ]
        self.non_target_gestures = [
            "Write name on leg",
            "Wave hello",
            "Glasses on/off",
            "Text on phone",
            "Write name in air",
            "Feel around in tray and pull out an object",
            "Scratch knee/leg skin",
            "Pull air toward your face",
            "Drink from bottle/cup",
            "Pinch knee/leg skin",
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures
        self.target_le_dict = joblib.load(inverse_gesture_dict_path)

    def __call__(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> float:
        y_true = y_true.clone()
        y_pred = y_pred.clone()
        # y_true and y_pred are in target_gesture or not 1 else 0
        y_true = torch.argmax(y_true, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        y_true_gestures = [self.target_le_dict[x] for x in y_true]
        y_pred_gestures = [self.target_le_dict[x] for x in y_pred]
        solution = pd.DataFrame({"gesture": y_true_gestures})
        submission = pd.DataFrame({"gesture": y_pred_gestures})
        y_true_bin = solution["gesture"].isin(self.target_gestures).values
        y_pred_bin = submission["gesture"].isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average="binary"
        )
        # Build multi-class labels for gestures
        y_true_mc = [
            x if x in self.target_gestures else "non_target"
            for x in solution["gesture"]
        ]
        y_pred_mc = [
            x if x in self.target_gestures else "non_target"
            for x in submission["gesture"]
        ]
        # Compute macro F1 over all gesture classes
        f1_macro = f1_score(y_true_mc, y_pred_mc, average="macro", zero_division=0)
        self.binary_f1 = f1_binary
        self.macro_f1 = f1_macro
        metrics_value = 0.5 * f1_binary + 0.5 * f1_macro
        return metrics_value


if __name__ == "__main__":
    inverse_gesture_dict_path = Path(
        "/kaggle/working/encoders/inverse_gesture_dict.pkl"
    )
    metrics = CompetitionMetrics(inverse_gesture_dict_path=inverse_gesture_dict_path)
    batch_size = 64
    class_num = 18
    y_true = torch.randint(0, 19, (batch_size, class_num))
    y_pred = torch.randint(0, 19, (batch_size, class_num))
    metrics_value = metrics(y_true, y_pred)
    print(f"Competition metrics value: {metrics_value:.4f}")
    # print(f"Binary F1: {metrics.binary_f1.compute():.4f}")
    # print(f"Macro F1: {metrics.macro_f1.compute():.4f}")
    print(f"Binary F1: {metrics.binary_f1:.4f}")
    print(f"Macro F1: {metrics.macro_f1:.4f}")
