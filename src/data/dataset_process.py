from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml  # type: ignore
from sklearn.preprocessing import StandardScaler


def get_max_min_by_group(
    df: pd.DataFrame, feature_cols: list[str], group_col: str = "subject"
) -> pd.DataFrame:
    """Get max and min values for each feature column grouped by a specific column."""
    max_min_df = df.groupby(group_col)[feature_cols].agg(["max", "min"]).reset_index()
    max_min_df.columns = [
        f"{col}_{stat}" if stat else col  # type: ignore
        for col, stat in max_min_df.columns
    ]
    max_min_df.rename(columns={group_col: "group"}, inplace=True)
    max_min_df = max_min_df.set_index("group")
    max_min_df = max_min_df.fillna(0.0)
    max_min_df = max_min_df.astype({col: float for col in max_min_df.columns})

    return max_min_df


def make_scaler(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Create a scaler dictionary with max and min values for each feature column."""
    scaler_dict = {}
    for col in feature_cols:
        feature_max = df[col].max()
        feature_min = df[col].min()
        scaler_dict[str(col)] = {
            "max": float(feature_max),
            "min": float(feature_min),
        }
    return scaler_dict


def scalarize_feature_by_column(
    data, output_dir=Path("/kaggle/working/encoders"), feature_cols=[]
):
    scaler_path = output_dir / "feature_scaler.yaml"
    scaled_data = np.zeros_like(data, dtype=np.float32)
    if scaler_path.exists():
        with open(scaler_path, "r") as f:
            scaled_dict = yaml.safe_load(f)
        print(f"Feature scaler loaded from {scaler_path}")
        for i, col in enumerate(feature_cols):
            feature_max = scaled_dict[str(col)]["max"]
            feature_min = scaled_dict[str(col)]["min"]
            if feature_max - feature_min > 0:
                scaled_data[:, :, i] = (data[:, :, i] - feature_min) / (
                    feature_max - feature_min + 1e-8
                )
            else:
                scaled_data[:, :, i] = data[:, :, i]
    else:
        scaled_dict = {}
        for i, col in enumerate(feature_cols):
            feature_max, feature_min = data[:, :, i].max(), data[:, :, i].min()
            scaled_dict[str(col)] = {
                "max": float(feature_max),
                "min": float(feature_min),
            }
            if feature_max - feature_min > 0:
                scaled_data[:, :, i] = (data[:, :, i] - feature_min) / (
                    feature_max - feature_min + 1e-8
                )
            else:
                scaled_data[:, :, i] = data[:, :, i]
        with open(scaler_path, "w") as f:
            yaml.dump(scaled_dict, f)
        print(f"Feature scaler saved to {scaler_path}")

    return scaled_data


def get_features_list(df: pd.DataFrame, features_cols=None, group_col="subject"):
    features_list, data_len_list, group_list = [], [], []
    for _, group in df.groupby("sequence_id"):
        features = (
            group[features_cols].ffill().bfill().fillna(0).values.astype("float32")
        )
        features_list.append(features)
        data_len_list.append(len(features))
        group_list.append(group[group_col].iloc[0])

    return features_list, data_len_list, group_list


def get_each_features_list(
    df: pd.DataFrame,
    imu_cols: list = [],
    thm_cols: list = [],
    tof_cols: list = [],
    group_col="subject",
):
    imu_features_list, thm_features_list, tof_features_list = [], [], []
    data_len_list, group_list = [], []
    for _, group in df.groupby("sequence_id"):
        imu_features = (
            group[imu_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        thm_features = (
            group[thm_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        tof_features = (
            group[tof_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        imu_features_list.append(imu_features)
        thm_features_list.append(thm_features)
        tof_features_list.append(tof_features)
        data_len_list.append(len(imu_features))

        group_list.append(group[group_col].iloc[0])

    return (
        imu_features_list,
        thm_features_list,
        tof_features_list,
        data_len_list,
        group_list,
    )


def get_features_and_labels(
    df: pd.DataFrame,
    imu_cols: list = [],
    thm_cols: list = [],
    tof_cols: list = [],
    features_cols=None,
    label_col="gesture_le",
    orient_label_col="orientation_le",
    behavior_label_col="behavior_le",
):
    imu_features_list, thm_features_list, tof_features_list = [], [], []
    labels_list, orient_labels_list, behavior_labels_list = [], [], []
    data_len_list = []
    for _, group in df.groupby("sequence_id"):
        imu_features = (
            group[imu_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        thm_features = (
            group[thm_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        tof_features = (
            group[tof_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )

        imu_features_list.append(imu_features)
        thm_features_list.append(thm_features)
        tof_features_list.append(tof_features)
        data_len_list.append(len(imu_features))
        labels_list.append(int(group[label_col].iloc[0]))
        orient_labels_list.append(int(group[orient_label_col].iloc[0]))
        behavior_labels_list.append(int(group[behavior_label_col].iloc[0]))
    labels = np.array(labels_list, dtype=np.int64)
    labels_ohe = F.one_hot(
        torch.tensor(labels), num_classes=len(df[label_col].unique())
    ).numpy()
    orient_labels = np.array(orient_labels_list, dtype=np.int64)
    orient_labels_ohe = F.one_hot(
        torch.tensor(orient_labels), num_classes=len(df[orient_label_col].unique())
    ).numpy()
    behavior_labels = np.array(behavior_labels_list, dtype=np.int64)
    behavior_labels_ohe = F.one_hot(
        torch.tensor(behavior_labels), num_classes=len(df[behavior_label_col].unique())
    ).numpy()
    return (
        imu_features_list,
        thm_features_list,
        tof_features_list,
        labels_ohe,
        orient_labels_ohe,
        behavior_labels_ohe,
        data_len_list,
    )


def get_features_and_labels_resize(
    df: pd.DataFrame,
    imu_cols: list = [],
    thm_cols: list = [],
    tof_cols: list = [],
    features_cols=None,
    label_col="gesture_le",
    orient_label_col="orientation_le",
    behavior_label_col="behavior_le",
    feature_size=256,
):
    imu_features_list, thm_features_list, tof_features_list = [], [], []
    labels_list, orient_labels_list, behavior_labels_list = [], [], []
    data_len_list = []
    for _, group in df.groupby("sequence_id"):
        imu_features = (
            group[imu_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        thm_features = (
            group[thm_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        tof_features = (
            group[tof_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        # 線形補間してサイズをそろえる
        imu_features = np.interp(
            np.linspace(0, 1, feature_size),
            np.linspace(0, 1, len(imu_features)),
            imu_features,
        )
        thm_features = np.interp(
            np.linspace(0, 1, feature_size),
            np.linspace(0, 1, len(thm_features)),
            thm_features,
        )
        tof_features = np.interp(
            np.linspace(0, 1, feature_size),
            np.linspace(0, 1, len(tof_features)),
            tof_features,
        )

        imu_features_list.append(imu_features)
        thm_features_list.append(thm_features)
        tof_features_list.append(tof_features)
        data_len_list.append(len(imu_features))
        labels_list.append(int(group[label_col].iloc[0]))
        orient_labels_list.append(int(group[orient_label_col].iloc[0]))
        behavior_labels_list.append(int(group[behavior_label_col].iloc[0]))
    labels = np.array(labels_list, dtype=np.int64)
    labels_ohe = F.one_hot(
        torch.tensor(labels), num_classes=len(df[label_col].unique())
    ).numpy()
    orient_labels = np.array(orient_labels_list, dtype=np.int64)
    orient_labels_ohe = F.one_hot(
        torch.tensor(orient_labels), num_classes=len(df[orient_label_col].unique())
    ).numpy()
    behavior_labels = np.array(behavior_labels_list, dtype=np.int64)
    behavior_labels_ohe = F.one_hot(
        torch.tensor(behavior_labels), num_classes=len(df[behavior_label_col].unique())
    ).numpy()
    return (
        imu_features_list,
        thm_features_list,
        tof_features_list,
        labels_ohe,
        orient_labels_ohe,
        behavior_labels_ohe,
        data_len_list,
    )


def get_features_and_labels_valid_filter(
    df: pd.DataFrame,
    imu_cols: list = [],
    thm_cols: list = [],
    tof_cols: list = [],
    features_cols=None,
    label_col="gesture_le",
    orient_label_col="orientation_le",
    behavior_label_col="behavior_le",
    feature_size=256,
):
    imu_features_list, thm_features_list, tof_features_list = [], [], []
    labels_list, orient_labels_list, behavior_labels_list = [], [], []
    data_len_list = []
    fillna_value = 0.0  # Fill value for missing data
    for _, group in df.groupby("sequence_id"):
        imu_features = (
            group[imu_cols]
            .ffill()
            .bfill()
            .fillna(fillna_value)
            .values.astype("float32")
        )
        thm_features = (
            group[thm_cols]
            .ffill()
            .bfill()
            .fillna(fillna_value)
            .values.astype("float32")
        )
        tof_features = (
            group[tof_cols]
            .ffill()
            .bfill()
            .fillna(fillna_value)
            .values.astype("float32")
        )
        # imu_featuresの3:6chがすべて0のものは除外
        valid_filter = imu_features[:, 3:6].sum(axis=1) != 0
        imu_features = imu_features[valid_filter]
        thm_features = thm_features[valid_filter]
        tof_features = tof_features[valid_filter]
        # tof_featuresの-1を0に置き換える(nanmeanとかしているので対策済みで不要)
        # tof_features[tof_features == -1] = 0.0
        imu_features_list.append(imu_features)
        thm_features_list.append(thm_features)
        tof_features_list.append(tof_features)
        data_len_list.append(len(imu_features))
        labels_list.append(int(group[label_col].iloc[0]))
        orient_labels_list.append(int(group[orient_label_col].iloc[0]))
        behavior_labels_list.append(int(group[behavior_label_col].iloc[0]))
    labels = np.array(labels_list, dtype=np.int64)
    labels_ohe = F.one_hot(
        torch.tensor(labels), num_classes=len(df[label_col].unique())
    ).numpy()
    orient_labels = np.array(orient_labels_list, dtype=np.int64)
    orient_labels_ohe = F.one_hot(
        torch.tensor(orient_labels), num_classes=len(df[orient_label_col].unique())
    ).numpy()
    behavior_labels = np.array(behavior_labels_list, dtype=np.int64)
    behavior_labels_ohe = F.one_hot(
        torch.tensor(behavior_labels), num_classes=len(df[behavior_label_col].unique())
    ).numpy()
    return (
        imu_features_list,
        thm_features_list,
        tof_features_list,
        labels_ohe,
        orient_labels_ohe,
        behavior_labels_ohe,
        data_len_list,
    )


def get_input_features_with_valid_filter(
    df: pd.DataFrame,
    imu_cols: list = [],
    thm_cols: list = [],
    tof_cols: list = [],
):
    imu_features_list, thm_features_list, tof_features_list = [], [], []
    data_len_list = []
    fillna_value = 0.0  # Fill value for missing data
    for _, group in df.groupby("sequence_id"):
        imu_features = (
            group[imu_cols]
            .ffill()
            .bfill()
            .fillna(fillna_value)
            .values.astype("float32")
        )
        thm_features = (
            group[thm_cols]
            .ffill()
            .bfill()
            .fillna(fillna_value)
            .values.astype("float32")
        )
        tof_features = (
            group[tof_cols]
            .ffill()
            .bfill()
            .fillna(fillna_value)
            .values.astype("float32")
        )
        # imu_featuresの3:6chがすべて0のものは除外
        valid_filter = imu_features[:, 3:6].sum(axis=1) != 0
        imu_features = imu_features[valid_filter]
        thm_features = thm_features[valid_filter]
        tof_features = tof_features[valid_filter]
        imu_features_list.append(imu_features)
        thm_features_list.append(thm_features)
        tof_features_list.append(tof_features)
        data_len_list.append(len(imu_features))
    return (
        imu_features_list,
        thm_features_list,
        tof_features_list,
        data_len_list,
    )


def get_features_and_labels_and_meta(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    imu_cols: list = [],
    thm_cols: list = [],
    tof_cols: list = [],
    features_cols=None,
    meta_features_cols=None,
    label_col="gesture_le",
    orient_label_col="orientation_le",
    behavior_label_col="behavior_le",
):
    imu_features_list, thm_features_list, tof_features_list = [], [], []
    meta_features_list = []
    labels_list, orient_labels_list, behavior_labels_list = [], [], []
    data_len_list = []
    for _, group in df.groupby("sequence_id"):
        imu_features = (
            group[imu_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        thm_features = (
            group[thm_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )
        tof_features = (
            group[tof_cols].ffill().bfill().fillna(-1).values.astype("float32")
        )

        imu_features_list.append(imu_features)
        thm_features_list.append(thm_features)
        tof_features_list.append(tof_features)
        data_len_list.append(len(imu_features))
        labels_list.append(int(group[label_col].iloc[0]))
        orient_labels_list.append(int(group[orient_label_col].iloc[0]))
        behavior_labels_list.append(int(group[behavior_label_col].iloc[0]))
        subject = group["subject"].iloc[0]
        meta_features = meta_df[meta_df["subject"] == subject][
            meta_features_cols
        ].values.astype("float32")
        meta_features_list.append(meta_features[0])

    labels = np.array(labels_list, dtype=np.int64)
    labels_ohe = F.one_hot(
        torch.tensor(labels), num_classes=len(df[label_col].unique())
    ).numpy()
    orient_labels = np.array(orient_labels_list, dtype=np.int64)
    orient_labels_ohe = F.one_hot(
        torch.tensor(orient_labels), num_classes=len(df[orient_label_col].unique())
    ).numpy()
    behavior_labels = np.array(behavior_labels_list, dtype=np.int64)
    behavior_labels_ohe = F.one_hot(
        torch.tensor(behavior_labels), num_classes=len(df[behavior_label_col].unique())
    ).numpy()
    return (
        imu_features_list,
        thm_features_list,
        tof_features_list,
        meta_features_list,
        labels_ohe,
        orient_labels_ohe,
        behavior_labels_ohe,
        data_len_list,
    )


def get_input_features(
    features_list, pad_len: int = 127, features_cols=None, scaling=True
):
    features_padded = np.zeros(
        (len(features_list), pad_len, features_list[0].shape[1]),
        dtype=np.float32,
    )
    for i, feat in enumerate(features_list):
        sequence_len = min(len(feat), pad_len)
        # features_padded[i, :sequence_len] = feat[:sequence_len]
        features_padded[i, :sequence_len] = feat[-sequence_len:]

    if scaling:
        features_padded = scalarize_feature_by_column(
            features_padded, feature_cols=features_cols
        )

    return features_padded, pad_len


def get_labels(df: pd.DataFrame):
    labels_list = []
    for _, group in df.groupby("sequence_id"):
        label = group["gesture_le"].iloc[0]
        labels_list.append(label)
    labels = np.array(labels_list, dtype=np.int64)
    labels_ohe = F.one_hot(
        torch.tensor(labels), num_classes=len(df["gesture_le"].unique())
    ).numpy()
    return labels_ohe


if __name__ == "__main__":
    # df_path = "/kaggle/working/processed/processed_with_homotrans_df.csv"
    # df_path = "/kaggle/working/processed_rotations_2/processed_with_rots_df.csv"
    # df_path = "/kaggle/working/processed_tof_region/processed_df.csv"

    # df_path = "/kaggle/working/processed_diff_cumsum/processed_df.csv"
    # df_path = (
    #     "/kaggle/working/processed_diff01_cumsum_swaphandness_height/processed_df.csv"
    # )
    # df_path = (
    #     "/kaggle/working/processed_diff01_cumsum_swaphandness_elbow/processed_df.csv"
    # )
    # df_path = "/kaggle/working/processed_diff01_cumsum_swaphandness4/processed_df.csv"
    df_path = "/kaggle/working/processed_diff01_swaphandness_means/processed_df.csv"
    # df = pd.read_csv(df_path)
    # max_min_df = get_max_min_by_group(
    #     df=df,
    #     feature_cols=[
    #         col for col in df.columns if col.startswith(("acc_", "rot_", "tof_"))
    #     ],
    #     group_col="subject",
    # )
    # subject_ = "SUBJ_004117"
    # print(max_min_df.loc[subject_])

    df = pd.read_csv(df_path)
    imu_cols_path = Path("/kaggle/working/features/imu_cols.yaml")
    thm_cols_path = Path("/kaggle/working/features/thm_cols.yaml")
    tof_cols_path = Path("/kaggle/working/features/tof_agg_cols.yaml")
    if imu_cols_path.exists():
        with open(imu_cols_path, "r") as f:
            imu_cols = yaml.safe_load(f)
    if thm_cols_path.exists():
        with open(thm_cols_path, "r") as f:
            thm_cols = yaml.safe_load(f)
    if tof_cols_path.exists():
        with open(tof_cols_path, "r") as f:
            tof_cols = yaml.safe_load(f)
    print(f"IMU columns: {imu_cols}")
    print(f"Thermal columns: {thm_cols}")
    print(f"ToF columns: {tof_cols}")
    features_cols = (
        imu_cols["imu_cols"] + thm_cols["thm_cols"] + tof_cols["tof_agg_cols"]
    )
    scaler_dict = make_scaler(df, features_cols)
    df_dir = Path(df_path).parent
    scaler_path = df_dir / "feature_scaler.yaml"
    with open(scaler_path, "w") as f:
        yaml.dump(scaler_dict, f)
    print(f"Feature scaler saved to {scaler_path}")
