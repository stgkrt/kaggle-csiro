import argparse
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml  # type: ignore

from src.data.dataset_process import (
    get_each_features_list,
    get_input_features,
    get_max_min_by_group,
)
from src.data.preprocess import (
    preprocess_signals,
)
from src.model.architectures.each_branch_cnn_model import EachBranchCNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_len = 127
feature_scaler_path = Path("/kaggle/working/features/feature_scaler.yaml")
with open(feature_scaler_path, "r") as f:
    feature_scaler = yaml.safe_load(f)
gesture_classes_path = Path(
    "/kaggle/input/cmi2025-models/models/encoders/inverse_gesture_dict.pkl"
)
gesture_classes = joblib.load(gesture_classes_path)

pred1_exp_name = "exp_009_5_002_eachbranch_cnn_model"
pred1_config_path = Path(f"/kaggle/working/{pred1_exp_name}/fold_0/config.yaml")
pred1_feature_config_path = Path(
    f"/kaggle/working/{pred1_exp_name}/fold_0/feature_columns.yaml"
)
with open(pred1_feature_config_path, "r") as f:
    pred1_feature_config = yaml.safe_load(f)
with open(pred1_config_path, "r") as f:
    pred1_config = yaml.safe_load(f)
pred1_imu_cols = pred1_feature_config["imu_cols"]
pred1_thm_cols = pred1_feature_config["thm_cols"]
pred1_tof_cols = pred1_feature_config["tof_cols"]

pred1_models = []
for fold in range(5):
    pred1_model_path = Path(
        f"/kaggle/working/{pred1_exp_name}/fold_{fold}/final_weights.pth"
    )
    pred1_model = EachBranchCNNModel(
        imu_dim=pred1_feature_config["imu_dim"],
        tof_dim=pred1_feature_config["tof_dim"],
        thm_dim=pred1_feature_config["thm_dim"],
        n_classes=pred1_config["model"]["n_classes"],
        default_emb_dim=pred1_config["model"]["default_emb_dim"],
        layer_num=pred1_config["model"]["layer_num"],
    )
    pred1_model.eval()
    pred1_model.load_state_dict(torch.load(pred1_model_path))
    pred1_model.to(device)
    pred1_models.append(pred1_model)


def get_post_cut_features(features, pad_len=127):
    feature_length = len(features)
    padded_features = np.zeros((pad_len, features.shape[1]))
    if feature_length < pad_len:
        padded_features[:feature_length, :] = features
    else:
        padded_features = features[-pad_len:]
    return padded_features


def scale_features(features, features_col_list):
    for i, col in enumerate(features_col_list):
        max_val = feature_scaler[col]["max"]
        min_val = feature_scaler[col]["min"]
        if max_val == min_val:
            features[:, i] = 0.0
        else:
            features[:, i] = (features[:, i] - min_val) / (max_val - min_val + 1e-8)
    return features


def predict_1(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()
    preprocessed_df = preprocess_signals(df_seq)
    (
        imu_features_list,
        thm_features_list,
        tof_features_list,
        data_len_list,
        group_list,
    ) = get_each_features_list(
        preprocessed_df,
        imu_cols=pred1_imu_cols,
        thm_cols=pred1_thm_cols,
        tof_cols=pred1_tof_cols,
    )
    batch_imu_features = np.empty(
        (0, imu_features_list[0].shape[0], imu_features_list[0].shape[1])
    )
    batch_thm_features = np.empty(
        (0, thm_features_list[0].shape[0], thm_features_list[0].shape[1])
    )
    batch_tof_features = np.empty(
        (0, tof_features_list[0].shape[0], tof_features_list[0].shape[1])
    )

    for imu_features, thm_features, tof_features in zip(
        imu_features_list, thm_features_list, tof_features_list
    ):
        imu_features = get_post_cut_features(imu_features, pad_len)
        thm_features = get_post_cut_features(thm_features, pad_len)
        tof_features = get_post_cut_features(tof_features, pad_len)
        # imu_features = scale_features(imu_features, pred1_feature_config["imu_cols"])
        thm_features = scale_features(thm_features, pred1_feature_config["thm_cols"])
        tof_features = scale_features(tof_features, pred1_feature_config["tof_cols"])
        batch_imu_features = np.vstack(
            (batch_imu_features, np.expand_dims(imu_features, axis=0))
        )
        batch_thm_features = np.vstack(
            (batch_thm_features, np.expand_dims(thm_features, axis=0))
        )
        batch_tof_features = np.vstack(
            (batch_tof_features, np.expand_dims(tof_features, axis=0))
        )

    model_input_dict = {
        "imu_features": torch.Tensor(batch_imu_features).float().to(device),
        "thm_features": torch.Tensor(batch_thm_features).float().to(device),
        "tof_features": torch.Tensor(batch_tof_features).float().to(device),
    }

    with torch.no_grad():
        models_output = None
        for model in pred1_models:
            outputs = model(model_input_dict)
            # softmax to get probabilities (batch, n_classes)
            probs = torch.softmax(outputs["logits"], dim=1)
            if models_output is None:
                models_output = probs
            else:
                models_output += probs
        models_output /= len(pred1_models)

    return models_output.cpu().numpy()


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Predict the gesture class for a given sequence and demographics data.

    Args:
        sequence (pl.DataFrame): The sequence data as a Polars DataFrame.
        demographics (pl.DataFrame): The demographics data as a Polars DataFrame.

    Returns:
        str: The predicted gesture class.
    """
    pred1_output = predict_1(sequence, demographics)
    pred1_gesture_class = np.argmax(pred1_output, axis=1)[0]
    return gesture_classes[pred1_gesture_class]
