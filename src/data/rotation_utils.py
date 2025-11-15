from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """
    クォータニオン (w, x, y, z) を 3x3 回転行列に変換する

    Args:
        w: クォータニオンの実部 (scalar part)
        x: クォータニオンのi成分
        y: クォータニオンのj成分
        z: クォータニオンのk成分

    Returns:
        3x3の回転行列 (numpy.ndarray)

    Note:
        クォータニオンは正規化されていることを前提とする
        正規化されていない場合は事前に normalize_quaternion() を使用すること
    """
    # クォータニオンの正規化（念のため）
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        # ゼロクォータニオンの場合は単位行列を返す
        return np.eye(3)

    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # 回転行列の計算
    # R = I + 2*K + 2*K^2 (ロドリゲスの公式によるクォータニオン表現)
    # より直接的な計算方式を使用

    rotation_matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )

    return rotation_matrix


def normalize_quaternion(
    w: float, x: float, y: float, z: float
) -> Tuple[float, float, float, float]:
    """
    クォータニオンを正規化する

    Args:
        w, x, y, z: クォータニオンの成分

    Returns:
        正規化されたクォータニオンの成分 (w, x, y, z)
    """
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return 1.0, 0.0, 0.0, 0.0  # 単位クォータニオン
    return w / norm, x / norm, y / norm, z / norm


def quaternions_to_rotation_matrices(
    df: pd.DataFrame,
    w_col: str = "rot_w",
    x_col: str = "rot_x",
    y_col: str = "rot_y",
    z_col: str = "rot_z",
) -> np.ndarray:
    """
    DataFrame内の複数のクォータニオンを回転行列に一括変換する

    Args:
        df: クォータニオン列を含むDataFrame
        w_col, x_col, y_col, z_col: クォータニオン成分の列名

    Returns:
        形状 (n_samples, 3, 3) の回転行列配列
    """
    n_samples = len(df)
    rotation_matrices = np.zeros((n_samples, 3, 3))

    for i in range(n_samples):
        w = df.iloc[i][w_col]
        x = df.iloc[i][x_col]
        y = df.iloc[i][y_col]
        z = df.iloc[i][z_col]

        rotation_matrices[i] = quaternion_to_rotation_matrix(w, x, y, z)

    return rotation_matrices


def rotation_matrix_to_euler_angles(
    R: np.ndarray, order: str = "xyz"
) -> Tuple[float, float, float]:
    """
    回転行列をオイラー角に変換する

    Args:
        R: 3x3回転行列
        order: 回転順序 ('xyz', 'zyx', 'zxy' など)

    Returns:
        オイラー角 (alpha, beta, gamma) [ラジアン]
    """
    if order.lower() == "xyz":
        # X-Y-Z順のオイラー角
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return x, y, z

    elif order.lower() == "zyx":
        # Z-Y-X順のオイラー角（ヨー・ピッチ・ロール）
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # roll
            y = np.arctan2(-R[2, 0], sy)  # pitch
            z = np.arctan2(R[1, 0], R[0, 0])  # yaw
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return z, y, x  # yaw, pitch, roll

    else:
        raise ValueError(f"Unsupported rotation order: {order}")


def get_rotation_features(
    df: pd.DataFrame,
    w_col: str = "rot_w",
    x_col: str = "rot_x",
    y_col: str = "rot_y",
    z_col: str = "rot_z",
) -> pd.DataFrame:
    """
    クォータニオンから様々な回転特徴量を抽出する

    Args:
        df: 入力DataFrame
        w_col, x_col, y_col, z_col: クォータニオン成分の列名

    Returns:
        回転特徴量が追加されたDataFrame
    """
    result_df = df.copy()

    # 回転行列を計算
    rotation_matrices = quaternions_to_rotation_matrices(df, w_col, x_col, y_col, z_col)

    # 回転行列の各要素を特徴量として追加
    for i in range(3):
        for j in range(3):
            result_df[f"rot_matrix_{i}{j}"] = rotation_matrices[:, i, j]

    # オイラー角を計算
    euler_xyz = np.array(
        [rotation_matrix_to_euler_angles(R, "xyz") for R in rotation_matrices]
    )
    euler_zyx = np.array(
        [rotation_matrix_to_euler_angles(R, "zyx") for R in rotation_matrices]
    )

    result_df["euler_x"] = euler_xyz[:, 0]
    result_df["euler_y"] = euler_xyz[:, 1]
    result_df["euler_z"] = euler_xyz[:, 2]

    result_df["yaw"] = euler_zyx[:, 0]
    result_df["pitch"] = euler_zyx[:, 1]
    result_df["roll"] = euler_zyx[:, 2]

    # 回転の大きさ（回転角度）
    # クォータニオンから回転角度を計算: theta = 2 * arccos(|w|)
    w_vals = df[w_col].values
    rotation_angles = 2 * np.arccos(np.clip(np.abs(w_vals), 0, 1))
    result_df["rotation_angle"] = rotation_angles

    return result_df


def remove_gravity_from_acc(acc_data, rot_data):
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[["acc_x", "acc_y", "acc_z"]].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)

    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
            linear_accel[i, :] = acc_values[i, :]

    return linear_accel


def calculate_angular_velocity_from_quat(
    rot_data, time_delta=1 / 200
):  # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i + 1]

        if (
            np.all(np.isnan(q_t))
            or np.all(np.isclose(q_t, 0))
            or np.all(np.isnan(q_t_plus_dt))
            or np.all(np.isclose(q_t_plus_dt, 0))
        ):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt

            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass

    return angular_vel


def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i + 1]

        if (
            np.all(np.isnan(q1))
            or np.all(np.isclose(q1, 0))
            or np.all(np.isnan(q2))
            or np.all(np.isclose(q2, 0))
        ):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            relative_rotation = r1.inv() * r2

            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
            pass

    return angular_dist
