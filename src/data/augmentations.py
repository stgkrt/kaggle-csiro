import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import signal
from scipy.interpolate import interp1d


class TimeSeriesAugmentation:
    def __init__(
        self,
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        time_shift_range: float = 0.1,
        noise_std: float = 0.02,
        magnitude_scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_angle_range: float = 0.1,
        mask_ratio: float = 0.1,
        freq_filter_range: Tuple[float, float] = (0.1, 0.9),
        dropout_ratio: float = 0.2,
        aug_prob: float = 0.5,
        aug_dropout_prob: float = 0.1,
    ):
        self.time_stretch_range = time_stretch_range
        self.time_shift_range = time_shift_range
        self.noise_std = noise_std
        self.magnitude_scale_range = magnitude_scale_range
        self.rotation_angle_range = rotation_angle_range
        self.mask_ratio = mask_ratio
        self.freq_filter_range = freq_filter_range
        self.dropout_ratio = dropout_ratio
        self.aug_prob = aug_prob
        self.aug_dropout_prob = aug_dropout_prob
        self.nan_value = 0.0

    def __call__(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        augmented = {key: data.copy() for key, data in sequence.items()}

        if random.random() < self.aug_prob:
            augmented = self.time_stretch(augmented)

        if random.random() < self.aug_prob:
            augmented = self.time_shift(augmented)

        if random.random() < self.aug_prob:
            augmented = self.add_noise(augmented)

        if random.random() < self.aug_prob:
            augmented = self.magnitude_scale(augmented)

        if random.random() < self.aug_prob:
            augmented = self.rotate_imu(augmented)

        # if random.random() < self.aug_prob:
        #     augmented = self.time_mask(augmented)

        if random.random() < self.aug_dropout_prob:
            augmented = self.coarse_dropout(augmented)

        if random.random() < self.aug_dropout_prob:
            augmented = self.nan_dropout(augmented)

        # exp048では無効化
        # if random.random() < self.aug_prob:
        #     augmented = self.replace_nan(augmented)

        # if random.random() < self.aug_prob:
        #     augmented = self.frequency_filter(augmented)

        return augmented

    # def time_stretch(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    #     stretch_factor = random.uniform(*self.time_stretch_range)
    #     data = sequence["imu_features"]
    #     seq_len = data.shape[0]
    #     original_time = np.linspace(0, 1, seq_len)
    #     new_length = max(1, int(seq_len / stretch_factor))
    #     new_time = np.linspace(0, 1, new_length)

    #     result = {}
    #     for key, data in sequence.items():
    #         num_features = data.shape[1]
    #         stretched_sequence = np.zeros((new_length, num_features))
    #         for i in range(num_features):
    #             interpolator = interp1d(
    #                 original_time,
    #                 data[:, i],
    #                 kind="linear",
    #                 bounds_error=False,
    #                 fill_value="extrapolate",
    #             )
    #             stretched_sequence[:, i] = interpolator(new_time)
    #         result[key] = stretched_sequence

    #     return result

    def time_stretch(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        stretch_factor = random.uniform(0.75, 1.0)
        data = sequence["imu_features"]
        seq_len = data.shape[0]
        cut_length = int(seq_len * stretch_factor)
        original_time = np.linspace(0, 1, seq_len)
        cut_time = np.linspace(0, 1, cut_length)

        result = {}
        for key, data in sequence.items():
            num_features = data.shape[1]
            stretched_sequence = np.zeros((seq_len, num_features))
            for i in range(num_features):
                interpolator = interp1d(
                    cut_time,
                    data[-cut_length:, i],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                stretched_sequence[:, i] = interpolator(original_time)
            result[key] = stretched_sequence

        return result

    def time_shift(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        data = sequence["imu_features"]
        seq_len, num_features = data.shape
        max_shift = int(seq_len * self.time_shift_range)
        shift = random.randint(-max_shift, max_shift)

        result = {}
        for key, data in sequence.items():
            if shift == 0:
                result[key] = data
                continue

            shifted = np.zeros_like(data)
            if shift > 0:
                shifted[shift:] = data[:-shift]
            else:
                shifted[:shift] = data[-shift:]

            result[key] = shifted

        return result

    def add_noise(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = {}
        for key, data in sequence.items():
            noise = np.random.normal(0, self.noise_std, data.shape)
            result[key] = data + noise

        return result

    def magnitude_scale(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        scale_factor = random.uniform(*self.magnitude_scale_range)
        data = sequence["imu_features"]

        result = {}
        for key, data in sequence.items():
            scaled = data.copy()
            # IMUデータの場合のみスケーリングを適用
            if key == "imu":
                scaled[:, -4:] *= scale_factor
            result[key] = scaled

        return result

    def rotate_imu(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        data = sequence["imu_features"]
        rotated = data.copy()
        # IMUデータの場合のみ回転を適用
        angle = random.uniform(-self.rotation_angle_range, self.rotation_angle_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        acc_x = rotated[:, 0] * cos_a - rotated[:, 1] * sin_a
        acc_y = rotated[:, 0] * sin_a + rotated[:, 1] * cos_a
        rotated[:, 0] = acc_x
        rotated[:, 1] = acc_y

        result = {}
        result["imu_features"] = rotated
        for key, data in sequence.items():
            if key != "imu_features":
                result[key] = data

        return result

    def time_mask(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        data = sequence["imu_features"]
        seq_len, num_features = data.shape
        mask_length = int(seq_len * self.mask_ratio)

        result = {}
        for key, data in sequence.items():
            if mask_length == 0:
                result[key] = data
                continue

            start_idx = random.randint(0, seq_len - mask_length)
            masked = data.copy()

            if start_idx > 0 and start_idx + mask_length < seq_len:
                before_mean = np.mean(masked[max(0, start_idx - 5) : start_idx], axis=0)
                after_mean = np.mean(
                    masked[
                        start_idx + mask_length : min(
                            seq_len, start_idx + mask_length + 5
                        )
                    ],
                    axis=0,
                )
                fill_value = (before_mean + after_mean) / 2
            else:
                fill_value = np.mean(masked, axis=0)

            masked[start_idx : start_idx + mask_length] = fill_value
            result[key] = masked

        return result

    def coarse_dropout(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        data = sequence["imu_features"]
        seq_len, num_features = data.shape

        # if np.random.rand() < 0.5:
        #     nan_value = -1.0
        # else:
        #     nan_value = 0.0
        nan_value = 0.0
        result = {}
        for key, data in sequence.items():
            drop_size = int(seq_len * self.dropout_ratio)
            dropped = data.copy()
            start_idx = random.randint(0, seq_len - drop_size)
            if key == "imu_features":
                pass
            elif key == "thm_features" or key == "tof_features":
                dropped[start_idx : start_idx + drop_size] = nan_value
            else:
                dropped[start_idx : start_idx + drop_size] = nan_value
            result[key] = dropped

        return result

    def nan_dropout(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        data = sequence["imu_features"]
        seq_len, num_features = data.shape

        # if np.random.rand() < 0.5:
        #     nan_value = -1.0
        # else:
        #     nan_value = 0.0
        nan_value = 0.0
        result = {}
        for key, data in sequence.items():
            dropout_signals = np.random.rand(*data.shape) < self.dropout_ratio

            dropped = data.copy()
            if key == "imu_features":
                pass
            elif key == "thm_features" or key == "tof_features":
                dropped = np.where(dropout_signals, nan_value, dropped)
            else:
                dropped = np.where(dropout_signals, nan_value, dropped)
            result[key] = dropped

        return result

    def replace_nan(self, sequence: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = {}

        if np.random.rand() < 0.5:
            nan_value = -1.0
        else:
            nan_value = 0.0
        for key, data in sequence.items():
            if key == "imu_features":
                result[key] = data.copy()
            elif key == "thm_features" or key == "tof_features":
                result[key] = nan_value * np.ones_like(data)
        return result

    def frequency_filter(
        self, sequence: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        data = sequence["imu_features"]
        seq_len, num_features = data.shape

        filtered = data.copy()
        # IMUデータの場合のみフィルタリングを適用
        if random.random() < 0.5:
            cutoff = random.uniform(*self.freq_filter_range)
            sos = signal.butter(2, cutoff, btype="low", output="sos")
        else:
            cutoff = random.uniform(0.05, self.freq_filter_range[0])
            sos = signal.butter(2, cutoff, btype="high", output="sos")

        try:
            filtered = signal.sosfilt(sos, data)
        except Exception:
            pass

        result = {}
        result["imu_features"] = filtered
        # 他のデータはそのまま保持
        for key, data in sequence.items():
            if key != "imu_features":
                result[key] = data

        return result
