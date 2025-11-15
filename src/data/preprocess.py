import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml  # type: ignore
from scipy.fft import fft
from sklearn.preprocessing import LabelEncoder

from src.data.rotation_utils import (
    calculate_angular_distance,
    calculate_angular_velocity_from_quat,
    quaternion_to_rotation_matrix,
    quaternions_to_rotation_matrices,
    remove_gravity_from_acc,
)


def label_encoding(df, columns, encoders_dir):
    """
    Apply label encoding to specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to apply label encoding to.

    Returns:
    pd.DataFrame: DataFrame with label encoded columns.
    """
    target_gesture = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Neck - pinch skin",
        "Neck - scratch",
    ]
    non_target_gestures = [
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
    gesture_dict_path = encoders_dir / "gesture_dict.pkl"
    inverse_gesture_dict_path = encoders_dir / "inverse_gesture_dict.pkl"
    if gesture_dict_path.exists() and inverse_gesture_dict_path.exists():
        with open(gesture_dict_path, "rb") as f:
            gesture_dict = pickle.load(f)
        with open(inverse_gesture_dict_path, "rb") as f:
            inverse_gesture_dict = pickle.load(f)
    else:
        # gestureの18個のlabelを0から17に変換するdictionaryを作成
        gesture_dict = {
            label: i for i, label in enumerate(sorted(df["gesture"].unique()))
        }
        inverse_gesture_dict = {v: k for k, v in gesture_dict.items()}
        # dictを保存
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with open(gesture_dict_path, "wb") as f:
            pickle.dump(gesture_dict, f)
        with open(inverse_gesture_dict_path, "wb") as f:
            pickle.dump(inverse_gesture_dict, f)
    # Apply label encoding to the 'gesture' column
    target_non_target_le_path = encoders_dir / "target_non_target_gesture_le.pkl"
    if target_non_target_le_path.exists():
        target_gesture_le = joblib.load(target_non_target_le_path)["target"]
        non_target_gesture_le = joblib.load(target_non_target_le_path)["non_target"]
    else:
        # target_gestureとnon_target_gestureをlabel encodingする
        target_gesture_le = [gesture_dict[g] for g in target_gesture]
        non_target_gesture_le = [gesture_dict[g] for g in non_target_gestures]
        target_non_target_gesture_le = {
            "target": target_gesture_le,
            "non_target": non_target_gesture_le,
        }
        joblib.dump(target_non_target_gesture_le, target_non_target_le_path)

    df["gesture_le"] = df["gesture"].apply(
        lambda x: gesture_dict[x] if x in gesture_dict else -1
    )
    df["gesture_le"] = df["gesture_le"].astype(int)

    return df


def label_encoding_only_target_gesture(df, encoders_dir):
    """
    Apply label encoding to only target gestures in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: DataFrame with label encoded target gestures.
    """
    target_gesture = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Neck - pinch skin",
        "Neck - scratch",
    ]
    gesture_dict_path = encoders_dir / "gesture_dict_only_target.pkl"
    inverse_gesture_dict_path = encoders_dir / "inverse_gesture_dict_only_target.pkl"
    if gesture_dict_path.exists() and inverse_gesture_dict_path.exists():
        with open(gesture_dict_path, "rb") as f:
            gesture_dict = pickle.load(f)
        with open(inverse_gesture_dict_path, "rb") as f:
            inverse_gesture_dict = pickle.load(f)
    else:
        # gestureの18個のlabelを0から17に変換するdictionaryを作成
        gesture_dict = {label: i for i, label in enumerate(sorted(target_gesture))}
        inverse_gesture_dict = {v: k for k, v in gesture_dict.items()}
        # dictを保存
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with open(gesture_dict_path, "wb") as f:
            pickle.dump(gesture_dict, f)
        with open(inverse_gesture_dict_path, "wb") as f:
            pickle.dump(inverse_gesture_dict, f)
    # Apply label encoding to the 'gesture' column
    target_non_target_le_path = encoders_dir / "only_target_le.pkl"
    if target_non_target_le_path.exists():
        target_gesture_le = joblib.load(target_non_target_le_path)["target"]
    else:
        target_gesture_le = [gesture_dict[g] for g in target_gesture]
        target_non_target_gesture_le = {
            "target": target_gesture_le,
        }
        joblib.dump(target_non_target_gesture_le, target_non_target_le_path)

    df["gesture_le"] = df["gesture"].apply(
        lambda x: gesture_dict[x] if x in gesture_dict else -1
    )
    df["gesture_le"] = df["gesture_le"].astype(int)

    return df


def label_split_encoding(df, columns, encoders_dir):
    """
    Apply label encoding to specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to apply label encoding to.

    Returns:
    pd.DataFrame: DataFrame with label encoded columns.
    """
    target_gesture = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Neck - pinch skin",
        "Neck - scratch",
    ]
    non_target_gestures = [
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
    gesture_dict_path = encoders_dir / "split_gesture_dict.pkl"
    inverse_gesture_dict_path = encoders_dir / "inverse_split_gesture_dict.pkl"
    if gesture_dict_path.exists() and inverse_gesture_dict_path.exists():
        with open(gesture_dict_path, "rb") as f:
            gesture_dict = pickle.load(f)
        with open(inverse_gesture_dict_path, "rb") as f:
            inverse_gesture_dict = pickle.load(f)
    else:
        # gestureの18個のlabelを0から17に変換するdictionaryを作成
        gesture_dict = {}
        for i, gesture in enumerate(target_gesture):
            gesture_dict[gesture] = i
        for j, gesture in enumerate(non_target_gestures):
            gesture_dict[gesture] = j + len(target_gesture)
        inverse_gesture_dict = {v: k for k, v in gesture_dict.items()}
        # dictを保存
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with open(gesture_dict_path, "wb") as f:
            pickle.dump(gesture_dict, f)
        with open(inverse_gesture_dict_path, "wb") as f:
            pickle.dump(inverse_gesture_dict, f)
    # Apply label encoding to the 'gesture' column
    target_non_target_le_path = encoders_dir / "target_split_gesture_le.pkl"
    if target_non_target_le_path.exists():
        target_gesture_le = joblib.load(target_non_target_le_path)["target"]
        non_target_gesture_le = joblib.load(target_non_target_le_path)["non_target"]
    else:
        # target_gestureとnon_target_gestureをlabel encodingする
        target_gesture_le = [gesture_dict[g] for g in target_gesture]
        non_target_gesture_le = [gesture_dict[g] for g in non_target_gestures]
        target_non_target_gesture_le = {
            "target": target_gesture_le,
            "non_target": non_target_gesture_le,
        }
        joblib.dump(target_non_target_gesture_le, target_non_target_le_path)

    df["gesture_le"] = df["gesture"].apply(
        lambda x: gesture_dict[x] if x in gesture_dict else -1
    )
    df["gesture_le"] = df["gesture_le"].astype(int)

    return df


def orientation_encoding(df, columns, encoders_dir):
    """
    Apply label encoding to specified orientation columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to apply label encoding to.

    Returns:
    pd.DataFrame: DataFrame with label encoded orientation columns.
    """
    orientation_dict_path = encoders_dir / "orientation_dict.pkl"
    if orientation_dict_path.exists():
        with open(orientation_dict_path, "rb") as f:
            orientation_dict = pickle.load(f)
    else:
        # Create a dictionary for orientation labels
        orientation_dict = {
            label: i for i, label in enumerate(sorted(df["orientation"].unique()))
        }
        # Save the dictionary
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with open(orientation_dict_path, "wb") as f:
            pickle.dump(orientation_dict, f)

    # Apply label encoding to the specified columns
    for col in columns:
        df[col + "_le"] = df[col].apply(
            lambda x: orientation_dict[x] if x in orientation_dict else -1
        )
        df[col + "_le"] = df[col + "_le"].astype(int)

    return df


def behavior_encoding(df, columns, encoders_dir):
    """
    Apply label encoding to specified behavior columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to apply label encoding to.

    Returns:
    pd.DataFrame: DataFrame with label encoded behavior columns.
    """
    behavior_dict_path = encoders_dir / "behavior_dict.pkl"
    if behavior_dict_path.exists():
        with open(behavior_dict_path, "rb") as f:
            behavior_dict = pickle.load(f)
    else:
        # Create a dictionary for behavior labels
        behavior_dict = {
            label: i for i, label in enumerate(sorted(df["behavior"].unique()))
        }
        # Save the dictionary
        encoders_dir.mkdir(parents=True, exist_ok=True)
        with open(behavior_dict_path, "wb") as f:
            pickle.dump(behavior_dict, f)

    # Apply label encoding to the specified columns
    for col in columns:
        df[col + "_le"] = df[col].apply(
            lambda x: behavior_dict[x] if x in behavior_dict else -1
        )
        df[col + "_le"] = df[col + "_le"].astype(int)

    return df


def calculate_rolling_mean(
    df, columns, window_size=16, groupby_col="sequence_id", ignore_value=0.0
):
    """
    Calculate rolling mean for specified columns within each group.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to calculate rolling mean for.
    window_size (int): Size of the rolling window (default: 16).
    groupby_col (str): Column name to group by (default: 'sequence_id').
    ignore_value (float or None): Value to ignore in rolling mean calculation
        (default: None).

    Returns:
    pd.DataFrame: DataFrame with rolling mean columns added.
    """
    # 元のデータフレームをコピー
    df = df.copy()

    # 新しい列を格納する辞書
    new_columns = {}

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue

        if ignore_value is not None:
            # Calculate rolling mean within each group, ignoring specified values
            rolling_mean_values = (
                df.groupby(groupby_col)[col]
                .apply(
                    lambda x: x.replace(ignore_value, np.nan)
                    .rolling(window=window_size, min_periods=1)
                    .mean()
                )
                .reset_index(level=0, drop=True)
            )
            rolling_std_values = (
                df.groupby(groupby_col)[col]
                .apply(
                    lambda x: x.replace(ignore_value, np.nan)
                    .rolling(window=window_size, min_periods=1)
                    .std()
                )
                .reset_index(level=0, drop=True)
            )
        else:
            # Calculate rolling mean within each group
            rolling_mean_values = (
                df.groupby(groupby_col)[col]
                .rolling(window=window_size, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            rolling_std_values = (
                df.groupby(groupby_col)[col]
                .rolling(window=window_size, min_periods=1)
                .std()
            )

        # nanは元の値に置き換える
        rolling_mean_values = rolling_mean_values.fillna(df[col])
        new_columns[f"{col}_rolling_mean_{window_size}"] = rolling_mean_values
        rolling_std_values = rolling_std_values.fillna(df[col])
        new_columns[f"{col}_rolling_std_{window_size}"] = rolling_std_values

    # すべての新しい列を一度に結合
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


def preprocess_signals(df):
    """
    Preprocess the signals in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    pd.DataFrame: DataFrame with preprocessed signals.
    """
    # 新しい列を格納する辞書
    new_columns = {}

    # まず基本的な特徴量を計算
    acc_mag = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    rot_angle = 2 * np.arccos(df["rot_w"].clip(-1, 1))

    new_columns["acc_mag"] = acc_mag
    new_columns["rot_angle"] = rot_angle

    # 一時的にDataFrameに追加して差分計算に使用
    df_temp = df.copy()
    df_temp["acc_mag"] = acc_mag
    df_temp["rot_angle"] = rot_angle
    fillna_value = 0.0

    new_columns["acc_mag_jerk"] = (
        df_temp.groupby("sequence_id")["acc_mag"].diff().fillna(fillna_value)
    )
    new_columns["rot_angle_vel"] = (
        df_temp.groupby("sequence_id")["rot_angle"].diff().fillna(fillna_value)
    )

    print("  Calculating ToF features with vectorized NumPy...")
    tof_pixel_cols = [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
    # tof_pixel_cols = [
    #     f"tof_{i}_v{p}_rolling_mean_5" for i in range(1, 6) for p in range(64)
    # ]
    tof_data_np = df[tof_pixel_cols].replace(-1, np.nan).to_numpy()
    reshaped_tof = tof_data_np.reshape(len(df), 5, 64)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"Mean of empty slice")
        warnings.filterwarnings("ignore", r"Degrees of freedom <= 2 for slice")
        mean_vals, std_vals = (
            np.nanmean(reshaped_tof, axis=2),
            np.nanstd(reshaped_tof, axis=2),
        )
        min_vals, max_vals = (
            np.nanmin(reshaped_tof, axis=2),
            np.nanmax(reshaped_tof, axis=2),
        )

    # ToFの統計値を一度に追加
    for i in range(1, 6):
        new_columns[f"tof_{i}_mean"] = mean_vals[:, i - 1]
        new_columns[f"tof_{i}_std"] = std_vals[:, i - 1]
        new_columns[f"tof_{i}_min"] = min_vals[:, i - 1]
        new_columns[f"tof_{i}_max"] = max_vals[:, i - 1]

    # tof_data = df[tof_pixel_cols].replace(-1, np.nan)
    # tof_region_new_columns = {}
    # for i in range(1, 6):
    #     for mode in [2, 4, 8, 16, 32]:
    #         region_size = 64 // mode
    #         for r in range(mode):
    #             region_data = tof_data.iloc[:,
    #                                           r * region_size : (r + 1) * region_size
    #             ]
    #             tof_region_new_columns.update(
    #                 {
    #                     f"tof{mode}_{i}_region_{r}_mean": region_data.mean(axis=1),
    #                     f"tof{mode}_{i}_region_{r}_std": region_data.std(axis=1),
    #                     f"tof{mode}_{i}_region_{r}_min": region_data.min(axis=1),
    #                     f"tof{mode}_{i}_region_{r}_max": region_data.max(axis=1),
    #                 }
    #             )

    # 重力除去した加速度を計算
    print("  Calculating gravity-free acceleration...")
    linear_accel_list = []
    angular_vel_list = []
    angular_dist_list = []

    # sequence_idごとにグループ化して処理
    print("  Calculating cross-axis energy features...")
    cross_axis_energy_list = []
    for _, group in df.groupby("sequence_id"):
        # 重力除去した加速度を計算
        group_linear_accel = remove_gravity_from_acc(group, group)
        linear_accel_list.append(group_linear_accel)

        # 角速度をクォータニオンから計算
        group_angular_vel = calculate_angular_velocity_from_quat(group)
        angular_vel_list.append(group_angular_vel)

        # 角距離をクォータニオンから計算
        group_angular_dist = calculate_angular_distance(group)
        angular_dist_list.append(group_angular_dist)

        # クロス軸エネルギー特徴量を計算
        group_cross_axis_energy = compute_cross_axis_energy(group)
        cross_axis_energy_list.append(group_cross_axis_energy)

    # 結果を結合
    linear_accel = np.vstack(linear_accel_list)
    angular_vel = np.vstack(angular_vel_list)
    angular_dist = np.concatenate(angular_dist_list)

    # クロス軸エネルギー特徴量をDataFrameに追加
    # 各sequenceの特徴量を該当する全ての行に適用
    cross_axis_features = {}
    for seq_features in cross_axis_energy_list:
        for feature_name, feature_value in seq_features.items():
            if feature_name not in cross_axis_features:
                cross_axis_features[feature_name] = []
            cross_axis_features[feature_name].append(feature_value)

    # sequence_idごとの特徴量を全ての行に展開
    for feature_name, feature_values in cross_axis_features.items():
        feature_array = np.zeros(len(df))
        for i, (_, group) in enumerate(df.groupby("sequence_id")):
            group_indices = group.index
            feature_array[group_indices] = feature_values[i]
        new_columns[feature_name] = feature_array

    new_columns["acc_x_gravity_free"] = linear_accel[:, 0]
    new_columns["acc_y_gravity_free"] = linear_accel[:, 1]
    new_columns["acc_z_gravity_free"] = linear_accel[:, 2]
    new_columns["acc_mag_gravity_free"] = np.sqrt(
        linear_accel[:, 0] ** 2 + linear_accel[:, 1] ** 2 + linear_accel[:, 2] ** 2
    )

    new_columns["angular_vel_x"] = angular_vel[:, 0]
    new_columns["angular_vel_y"] = angular_vel[:, 1]
    new_columns["angular_vel_z"] = angular_vel[:, 2]
    new_columns["angular_vel_mag"] = np.sqrt(
        angular_vel[:, 0] ** 2 + angular_vel[:, 1] ** 2 + angular_vel[:, 2] ** 2
    )

    new_columns["angular_distance"] = angular_dist

    # すべての新しい列を一度に結合
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=df.index)
        # tof_new_df = pd.DataFrame(tof_region_new_columns, index=df.index)
        # df = pd.concat([df, new_df, tof_new_df], axis=1)
        df = pd.concat([df, new_df], axis=1)

    return df


def calculate_position_from_acceleration(df, dt=0.02):
    """
    加速度データから位置を計算する関数

    Parameters:
    df (pd.DataFrame): 加速度データを含むDataFrame
    dt (float): サンプリング間隔（デフォルト: 0.02秒 = 50Hz）

    Returns:
    pd.DataFrame: 位置データが追加されたDataFrame
    """
    # 初期化
    df = df.copy()

    # 新しい列を格納する辞書
    new_columns = {
        "vel_x": np.zeros(len(df)),
        "vel_y": np.zeros(len(df)),
        "vel_z": np.zeros(len(df)),
        "pos_x": np.zeros(len(df)),
        "pos_y": np.zeros(len(df)),
        "pos_z": np.zeros(len(df)),
        "pos_rel_x": np.zeros(len(df)),
        "pos_rel_y": np.zeros(len(df)),
        "pos_rel_z": np.zeros(len(df)),
    }

    # sequence_idごとに処理
    for seq_id in df["sequence_id"].unique():
        mask = df["sequence_id"] == seq_id
        seq_data = df[mask].copy()

        if len(seq_data) <= 1:
            continue

        # 加速度データを取得
        acc_x = seq_data["acc_x"].values
        acc_y = seq_data["acc_y"].values
        acc_z = seq_data["acc_z"].values

        # 重力加速度を除去（簡易的にDCオフセットを除去）
        acc_x_filtered = acc_x - np.mean(acc_x)
        acc_y_filtered = acc_y - np.mean(acc_y)
        acc_z_filtered = acc_z - np.mean(acc_z)

        # 速度を計算（台形積分）
        vel_x = np.zeros(len(acc_x_filtered))
        vel_y = np.zeros(len(acc_y_filtered))
        vel_z = np.zeros(len(acc_z_filtered))

        for i in range(1, len(acc_x_filtered)):
            vel_x[i] = (
                vel_x[i - 1] + (acc_x_filtered[i] + acc_x_filtered[i - 1]) * dt / 2
            )
            vel_y[i] = (
                vel_y[i - 1] + (acc_y_filtered[i] + acc_y_filtered[i - 1]) * dt / 2
            )
            vel_z[i] = (
                vel_z[i - 1] + (acc_z_filtered[i] + acc_z_filtered[i - 1]) * dt / 2
            )

        # 速度のドリフトを除去（線形トレンド除去）
        time_indices = np.arange(len(vel_x))
        if len(time_indices) > 1:
            # 線形フィットしてトレンドを除去
            vel_x_detrend = vel_x - np.polyval(
                np.polyfit(time_indices, vel_x, 1), time_indices
            )
            vel_y_detrend = vel_y - np.polyval(
                np.polyfit(time_indices, vel_y, 1), time_indices
            )
            vel_z_detrend = vel_z - np.polyval(
                np.polyfit(time_indices, vel_z, 1), time_indices
            )
        else:
            vel_x_detrend = vel_x
            vel_y_detrend = vel_y
            vel_z_detrend = vel_z

        # 位置を計算（台形積分）
        pos_x = np.zeros(len(vel_x_detrend))
        pos_y = np.zeros(len(vel_y_detrend))
        pos_z = np.zeros(len(vel_z_detrend))

        for i in range(1, len(vel_x_detrend)):
            pos_x[i] = pos_x[i - 1] + (vel_x_detrend[i] + vel_x_detrend[i - 1]) * dt / 2
            pos_y[i] = pos_y[i - 1] + (vel_y_detrend[i] + vel_y_detrend[i - 1]) * dt / 2
            pos_z[i] = pos_z[i - 1] + (vel_z_detrend[i] + vel_z_detrend[i - 1]) * dt / 2

        # 結果を配列に設定
        indices = seq_data.index
        new_columns["vel_x"][indices] = vel_x_detrend
        new_columns["vel_y"][indices] = vel_y_detrend
        new_columns["vel_z"][indices] = vel_z_detrend
        new_columns["pos_x"][indices] = pos_x
        new_columns["pos_y"][indices] = pos_y
        new_columns["pos_z"][indices] = pos_z

        # 相対位置（sequence内での最初の位置からの変位）
        new_columns["pos_rel_x"][indices] = pos_x - pos_x[0]
        new_columns["pos_rel_y"][indices] = pos_y - pos_y[0]
        new_columns["pos_rel_z"][indices] = pos_z - pos_z[0]

    # 位置の大きさを計算
    new_columns["pos_magnitude"] = np.sqrt(
        new_columns["pos_x"] ** 2
        + new_columns["pos_y"] ** 2
        + new_columns["pos_z"] ** 2
    )
    new_columns["pos_rel_magnitude"] = np.sqrt(
        new_columns["pos_rel_x"] ** 2
        + new_columns["pos_rel_y"] ** 2
        + new_columns["pos_rel_z"] ** 2
    )

    # すべての新しい列を一度に結合
    new_df = pd.DataFrame(new_columns, index=df.index)
    df = pd.concat([df, new_df], axis=1)

    return df


def calculate_acc_diff_cumsum(
    df, columns=["acc_x", "acc_y", "acc_z"], groupby_col="sequence_id", threshold=0.1
):
    """
    加速度データについて、sequence_idごとのdiffのcumsumを特徴量として追加する関数
    threshold (scale elbow) = 0.005

    Parameters:
    df (pd.DataFrame): 元のDataFrame
    columns (list): diffとcumsumを計算する加速度カラム名のリスト
        (default: ["acc_x", "acc_y", "acc_z"])
    groupby_col (str): グループ化に使用するカラム名 (default: "sequence_id")
    threshold (float): diff値がこの閾値以下の場合は0にする (default: 0.1)

    Returns:
    pd.DataFrame: diffのcumsum特徴量が追加されたDataFrame
    """
    df = df.copy()

    print(
        f"Calculating diff cumsum features for {columns} with threshold={threshold}..."
    )

    # 新しい列を格納する辞書
    new_columns = {}

    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
            continue

        # sequence_idごとにdiffを計算
        diff_values = df.groupby(groupby_col)[col].diff().fillna(0)

        # 閾値以下の値を0にする（絶対値で判定）
        diff_values = diff_values.where(np.abs(diff_values) > threshold, 0)

        # cumsumを計算
        diff_cumsum_values = diff_values.groupby(df[groupby_col]).cumsum()

        new_columns[f"{col}_diff"] = diff_values
        new_columns[f"{col}_diff_cumsum"] = diff_cumsum_values
        print(f"Added {col}_diff_cumsum feature (threshold={threshold})")

    # すべての新しい列を一度に結合
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


def normalize_sequence_counter_by_subject_max(df):
    """
    sequence_counterをsubjectごとのmaxで正規化した特徴量を追加する関数

    Parameters:
    df (pd.DataFrame): 元のDataFrame

    Returns:
    pd.DataFrame: sequence_counterが正規化された特徴量が追加されたDataFrame
    """
    df = df.copy()

    print("Normalizing sequence_counter by subject max...")
    subject_max_values = df.groupby("subject")["sequence_counter"].max()
    normalized_values = np.zeros(len(df))
    for subject in df["subject"].unique():
        subject_mask = df["subject"] == subject
        subject_data = df.loc[subject_mask, "sequence_counter"]
        subject_max = subject_max_values.loc[subject]
        if subject_max != 0:
            normalized_values[subject_mask] = subject_data / subject_max
        else:
            normalized_values[subject_mask] = subject_data

    df["sequence_counter_norm"] = normalized_values
    print("Added sequence_counter_norm_by_subject_max feature")

    return df


def compute_cross_axis_energy(df):
    """
    加速度の各軸についてクロス軸エネルギー特徴量を計算する関数

    Parameters:
    df (pd.DataFrame): sequence_idグループのDataFrame

    Returns:
    dict: エネルギー特徴量の辞書
    """
    axes = ["x", "y", "z"]
    features = {}

    # 各軸のエネルギーを計算
    for axis in axes:
        fft_result = fft(df[f"acc_{axis}"].values)
        energy = np.sum(np.abs(fft_result) ** 2)
        features[f"energy_{axis}"] = energy

    # 軸間のエネルギー比を計算
    for i, axis1 in enumerate(axes):
        for axis2 in axes[i + 1 :]:
            features[f"energy_ratio_{axis1}{axis2}"] = features[f"energy_{axis1}"] / (
                features[f"energy_{axis2}"] + 1e-6
            )

    # 軸間のエネルギー相関を計算
    for i, axis1 in enumerate(axes):
        for axis2 in axes[i + 1 :]:
            fft1 = np.abs(fft(df[f"acc_{axis1}"].values))
            fft2 = np.abs(fft(df[f"acc_{axis2}"].values))
            correlation = np.corrcoef(fft1, fft2)[0, 1]
            # NaNの場合は0に置き換え
            if np.isnan(correlation):
                correlation = 0.0
            features[f"energy_corr_{axis1}{axis2}"] = correlation

    return features


def swap_axis_by_handness(df, demographics):
    """
    Swap axes of accelerometer and gyroscope data based on handedness.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing sensor data.
    demographics (pd.DataFrame): The demographics DataFrame containing
                                 subject handedness information.

    Returns:
    pd.DataFrame: The DataFrame with swapped axes for left-handed subjects.
    """
    df = df.copy()
    tmp_df = df.copy()
    left_handed_subjects = demographics[demographics["handedness"] == 0][
        "subject"
    ].values

    for subject in left_handed_subjects:
        subject_mask = df["subject"] == subject
        # Swap axes for accelerometer data
        df.loc[subject_mask, "acc_x"] = -df.loc[subject_mask, "acc_x"]
        # df.loc[subject_mask, "acc_y"] = -df.loc[subject_mask, "acc_y"]
        # df.loc[subject_mask, "acc_z"] = -df.loc[subject_mask, "acc_z"]
        # Swap axes for gyroscope data
        # df.loc[subject_mask, "rot_x"] = -df.loc[subject_mask, "rot_x"]
        df.loc[subject_mask, "rot_y"] = -df.loc[subject_mask, "rot_y"]
        df.loc[subject_mask, "rot_z"] = -df.loc[subject_mask, "rot_z"]
        # tof_3_v* と tof_5_v*を入れ替え
        tof_3_cols = [col for col in df.columns if col.startswith("tof_3_v")]
        tof_5_cols = [col for col in df.columns if col.startswith("tof_5_v")]
        tof_3_values = tmp_df.loc[subject_mask, tof_3_cols].values
        tof_5_values = tmp_df.loc[subject_mask, tof_5_cols].values
        df.loc[subject_mask, tof_3_cols] = tof_5_values
        df.loc[subject_mask, tof_5_cols] = tof_3_values
        # thm_3とthm_5を入れ替え
        thm_3_cols = [col for col in df.columns if col.startswith("thm_3")]
        thm_5_cols = [col for col in df.columns if col.startswith("thm_5")]
        thm_3_values = tmp_df.loc[subject_mask, thm_3_cols].values
        thm_5_values = tmp_df.loc[subject_mask, thm_5_cols].values
        df.loc[subject_mask, thm_3_cols] = thm_5_values
        df.loc[subject_mask, thm_5_cols] = thm_3_values

    return df


def get_tof_thm_mean(df):
    """ToFとTHMの平均値を計算して特徴量として追加する関数"""
    tof_cols = []
    for i in range(1, 6):
        tof_cols.append(f"tof_{i}_mean")
    tof_value = df[tof_cols].replace(-1, np.nan)
    print(tof_value.shape)
    print(tof_value)
    tof_mean = np.nanmean(tof_value, axis=1)
    print(tof_mean)
    df["tof_mean"] = tof_mean
    df["tof_mean"] = df["tof_mean"].fillna(0)
    df["tof_diff"] = df["tof_mean"].diff().fillna(0)
    thm_cols = [col for col in df.columns if col.startswith("thm_")]
    thm_value = df[thm_cols].replace(0, np.nan)
    thm_mean = np.nanmean(thm_value, axis=1)
    df["thm_mean"] = thm_mean
    df["thm_mean"] = df["thm_mean"].fillna(0)
    df["thm_diff"] = df["thm_mean"].diff().fillna(0)
    return df


def scale_by_meta_info(df, demographics):
    """dfのacc_x, acc_y, acc_zについて、
    dfのsubjectからdemographicsのmetaデータを参照してスケーリングする関数
    demographicsのheight_cmを使って、dfのacc_x, acc_y, acc_zをスケーリングする
    """
    for subject in df["subject"].unique():
        subject_mask = df["subject"] == subject
        # height_cm = (
        #     demographics.loc[demographics["subject"] == subject, "height_cm"].values
        #     / 100.0
        # )
        # height_cm = height_cm * np.ones_like(df.loc[subject_mask, "acc_x"].values)
        # for col in ["acc_x", "acc_y", "acc_z"]:
        #     df.loc[subject_mask, col] /= height_cm + 1e-6
        elbow_to_wrist_cm = demographics.loc[
            demographics["subject"] == subject, "elbow_to_wrist_cm"
        ].values
        elbow_to_wrist_cm = elbow_to_wrist_cm * np.ones_like(
            df.loc[subject_mask, "acc_x"].values
        )
        for col in ["acc_x", "acc_y", "acc_z"]:
            df.loc[subject_mask, col] /= elbow_to_wrist_cm + 1e-6
    return df


if __name__ == "__main__":
    DEBUG = False
    ONLY_TARGET_GESTURE = False
    df_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
    demographics_path = (
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
    )
    output_base_dir = "/kaggle/working/processed_diff01_swaphandness_means"
    target_gesture = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Neck - pinch skin",
        "Neck - scratch",
    ]

    df = pd.read_csv(df_path)
    print("data samples = ", len(df))
    if ONLY_TARGET_GESTURE:
        output_base_dir += "_only_target_gesture"
        OUTPUT_DIR = Path(output_base_dir)
        df = df[df["gesture"].isin(target_gesture)].reset_index(drop=True)
        df = label_encoding_only_target_gesture(df, encoders_dir=OUTPUT_DIR)
    else:
        OUTPUT_DIR = Path(output_base_dir)
        df = label_encoding(df, ["gesture"], encoders_dir=OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_gesture_path = OUTPUT_DIR / "gesture_classes.npy"
    save_feature_df_path = OUTPUT_DIR / "processed_df.csv"
    save_feature_cols_path = OUTPUT_DIR / "feature_cols.npy"

    # Load the dataset
    demographics = pd.read_csv(demographics_path)
    if DEBUG:
        # subject_idを100個に制限
        df = df[df["subject"].isin(df["subject"].unique()[:10])].reset_index(drop=True)

    # df = label_split_encoding(df, ["gesture"], encoders_dir=OUTPUT_DIR)
    df = swap_axis_by_handness(df, demographics)
    # df = scale_by_meta_info(df, demographics)
    print("data samples = ", len(df))
    df = orientation_encoding(df, ["orientation"], encoders_dir=OUTPUT_DIR)
    df = behavior_encoding(df, ["behavior"], encoders_dir=OUTPUT_DIR)
    gesture_classes = df["gesture_le"].unique()
    np.save(save_gesture_path, gesture_classes)

    # Preprocess the signals
    print("Preprocessing signals...")
    # processed_df = compute_cross_axis_energy(df)
    processed_df = preprocess_signals(df)
    processed_df = get_tof_thm_mean(processed_df)
    # processed_df = preprocess_signals(processed_df)
    processed_df = normalize_sequence_counter_by_subject_max(processed_df)

    # 加速度のdiff cumsumを計算
    processed_df = calculate_acc_diff_cumsum(processed_df)

    # roll_mean_std_col = ["acc_x", "acc_y", "acc_z"]
    # window_size = 12
    # processed_df = calculate_rolling_mean(
    #     processed_df, columns=roll_mean_std_col, window_size=window_size
    # )
    processed_df.to_csv(save_feature_df_path, index=False)

    imu_cols = [col for col in processed_df.columns if col.startswith(("acc_", "rot_"))]
    imu_cols += ["sequence_counter_norm"]
    imu_cols += [col for col in processed_df.columns if col.startswith("energy_")]

    tof_cols = [col for col in processed_df.columns if col.startswith("tof_")]
    # 位置・速度の特徴量を追加
    position_cols = [
        col for col in processed_df.columns if col.startswith(("pos_", "vel_"))
    ]
    # 重力除去加速度と角速度の特徴量を追加
    gravity_free_cols = [
        col
        for col in processed_df.columns
        if col.startswith("acc_") and "gravity_free" in col
    ]
    angular_vel_cols = [
        col for col in processed_df.columns if col.startswith("angular_vel_")
    ]
    # 角距離の特徴量を追加
    angular_dist_cols = [
        col for col in processed_df.columns if col.startswith("angular_distance")
    ]
    # 加速度のdiff cumsum特徴量を追加
    acc_diff_cumsum_cols = [
        col for col in processed_df.columns if col.endswith("_diff_cumsum")
    ]
    thm_cols = [col for col in processed_df.columns if col.startswith("thm_")]
    # 回転行列の特徴量を追加
    rotation_matrix_cols = [
        col for col in processed_df.columns if col.startswith("rot_matrix_")
    ]
    # agg_suffixes = ["_mean", "_std", "_max", "_min"]
    # tof_agg_cols = []
    # for i in range(1, 6):
    #     tof_agg_cols.extend(
    #         [
    #             f"tof_{i}_mean",
    #             f"tof_{i}_std",
    #             f"tof_{i}_max",
    #             f"tof_{i}_min",
    #         ]
    #     )
    tof_agg_cols = [
        col
        for col in processed_df.columns
        if col.startswith("tof") and col.endswith(("_mean", "_std", "_max", "_min"))
    ]

    # for i in range(1, 6):
    #     for mode in [2, 4, 8, 16, 32]:
    #         for r in range(mode):
    #             tof_agg_cols.extend(
    #                 [
    #                     f"tof{mode}_{i}_region_{r}_mean",
    #                     f"tof{mode}_{i}_region_{r}_std",
    #                     f"tof{mode}_{i}_region_{r}_min",
    #                     f"tof{mode}_{i}_region_{r}_max",
    #                 ]
    #             )

    features_cols = (
        imu_cols
        + position_cols
        + gravity_free_cols
        + angular_vel_cols
        + angular_dist_cols
        + acc_diff_cumsum_cols
        + thm_cols
        + tof_agg_cols
        + rotation_matrix_cols
    )
    np.save(save_feature_cols_path, np.array(features_cols))
    print(f"Number of ToF features: {len(tof_cols)}")
    print(f"Number of IMU features: {len(imu_cols)}")
    print(f"Number of position/velocity features: {len(position_cols)}")
    print(f"Number of angular distance features: {len(angular_dist_cols)}")
    print(f"Number of acc diff cumsum features: {len(acc_diff_cumsum_cols)}")
    print(f"Number of thermal features: {len(thm_cols)}")
    print(f"Number of aggregated ToF features: {len(tof_agg_cols)}")
    print(f"Total number of features: {len(features_cols)}")
    # save feature columns dictionary
    feature_cols_dict = {
        "imu": imu_cols,
        "position": position_cols,
        "gravity_free": gravity_free_cols,
        "angular_velocity": angular_vel_cols,
        "angular_distance": angular_dist_cols,
        "acc_diff_cumsum": acc_diff_cumsum_cols,
        "thermal": thm_cols,
        "tof": tof_cols,
        "tof_agg": tof_agg_cols,
        "rotation_matrix": rotation_matrix_cols,
    }
    # yaml形式で保存する場合
    with open(OUTPUT_DIR / "feature_cols_dict.yaml", "w") as f:
        yaml.dump(feature_cols_dict, f, allow_unicode=True)
    feature_cols_dict_path = OUTPUT_DIR / "feature_cols_dict.pkl"
    with open(feature_cols_dict_path, "wb") as f:
        pickle.dump(feature_cols_dict, f)
