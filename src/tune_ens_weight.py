from pathlib import Path

import numpy as np
import optuna
import pandas as pd

# stacking_train.py の oof 読み込み関数を流用
from stacking_train import (
    collect_oof_features,
    evaluate_with_competition_metrics,
    prepare_stacking_data,
)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def load_all_oof(exp_names, working_dir, df_path, splits_dir, n_folds=5):
    oof_list = []
    length = None
    for exp_name in exp_names:
        oof_df, feature_cols, target_cols = collect_oof_features(
            Path(working_dir), exp_name, Path(df_path), Path(splits_dir), n_folds
        )
        preds = oof_df[feature_cols].values
        preds = softmax(preds, axis=1)  # softmax適用
        # oof_list.append(preds)

        if length is None:
            length = len(preds)
            oof_list.append(preds)
        else:
            preds = preds[-length:]
            oof_list.append(preds)

    return np.stack(oof_list, axis=0), oof_df[target_cols].values


def objective(trial, oof_preds, targets, inverse_gesture_dict_path):
    n_models = oof_preds.shape[0]
    weights = np.array([trial.suggest_float(f"w{i}", 0, 1) for i in range(n_models)])
    weights = weights / weights.sum()
    blended = np.tensordot(weights, oof_preds, axes=([0], [0]))
    metrics = evaluate_with_competition_metrics(
        y_true=np.argmax(targets, axis=1),
        y_pred=blended,
        inverse_gesture_dict_path=inverse_gesture_dict_path,
        return_detailed=True,
    )
    return metrics["competition_score"]  # maximize


if __name__ == "__main__":
    exp_names = [
        "exp_044_9_010_splitpublic_cnn",  # CNN best
        "exp_044_9_011_splitpublic_tran",  # trans
        # "exp_044_9_012_splitpublic_lstm",
        "exp_044_9_013_splitpublic_cnnbce",  # bce
        # "exp_045_9_001_splitpublic_cnn",
        # "exp_045_9_002_splitpublic_tran",
        # "exp_045_9_003_splitpublic_lstm",
        "exp_046_9_003_splitpublic_lstm",  # lstm best
        "exp_048_9_001_splitpublic_cnn",
        "exp_048_9_002_splitpublic_tran",
        "exp_048_9_003_splitpublic_lstm",
        # "",
    ]
    working_dir = "/kaggle/working"
    df_path = "/kaggle/working/processed_diff01_cumsum/processed_df.csv"
    splits_dir = "/kaggle/working/splits_public"
    inverse_gesture_dict_path = "/kaggle/working/encoders/inverse_gesture_dict.pkl"
    n_folds = 5

    oof_preds, targets = load_all_oof(
        exp_names, working_dir, df_path, splits_dir, n_folds
    )

    # meanアンサンブルでのスコア
    mean_preds = np.mean(oof_preds, axis=0)
    mean_score = evaluate_with_competition_metrics(
        y_true=np.argmax(targets, axis=1),
        y_pred=mean_preds,
        inverse_gesture_dict_path=inverse_gesture_dict_path,
        return_detailed=True,
    )["competition_score"]

    # study = optuna.create_study(direction="maximize")
    # study.optimize(
    #     lambda trial: objective(trial, oof_preds, targets, inverse_gesture_dict_path),
    #     n_trials=1000,
    # )

    print(f"[mean ensemble] score: {mean_score}")
    print("Best weights:", [study.best_params[f"w{i}"] for i in range(len(exp_names))])
    print("Best score:", study.best_value)
