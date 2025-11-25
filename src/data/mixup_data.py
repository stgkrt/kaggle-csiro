import random

import numpy as np


class Mixup:
    def __init__(
        self,
        X: dict[str, np.ndarray],
        y: dict[str, np.ndarray],
        alpha: float = 0.2,
    ):
        self.alpha = alpha
        self.x_list = X
        self.y_list = y

    def same_idx_mix_input(
        self, x_input: np.ndarray, data_name: str, input_or_label: str
    ) -> np.ndarray:
        if input_or_label == "input":
            x_for_mix = self.x_list[data_name][self.idx]
            x_for_mix = self._adjust_length(x_for_mix, x_input)
        else:
            x_for_mix = self.y_list[data_name][self.idx]
        mixed = self.lam * x_for_mix + (1 - self.lam) * x_input

        return mixed

    def _adjust_length(self, x_input: np.ndarray, x_for_mix: np.ndarray) -> np.ndarray:
        if len(x_input) < len(x_for_mix):
            return np.pad(
                x_input, ((0, len(x_for_mix) - len(x_input)), (0, 0)), mode="constant"
            )
        else:
            return x_input[-len(x_for_mix) :]

    def __call__(
        self, x_input: np.ndarray, y_input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.idx = random.randint(0, len(self.x_list["imu_features"]) - 1)

        x_for_mix, y_for_mix = (
            self.x_list["imu_features"][self.idx],
            self.y_list["label"][self.idx],
        )
        x_for_mix = self._adjust_length(x_for_mix, x_input)

        self.lam = np.random.beta(self.alpha, self.alpha)

        mixed_x = self.lam * x_for_mix + (1 - self.lam) * x_input
        mixed_y = self.lam * y_for_mix + (1 - self.lam) * y_input

        return mixed_x, mixed_y


class CutMixup:
    def __init__(
        self, X: dict[str, np.ndarray], y: dict[str, np.ndarray], alpha: float = 0.2
    ):
        self.alpha = alpha
        self.x_list = X
        self.y_list = y

    def same_idx_mix_input(
        self, x_input: np.ndarray, data_name: str, input_or_label: str
    ) -> np.ndarray:
        if input_or_label == "input":
            mixed = x_input.copy()
            x_for_mix = self.x_list[data_name][self.idx]

            x_for_mix = self._adjust_length(x_for_mix, x_input)
            mixed[self.cutmixup_range_min : self.cutmixup_range_max] = x_for_mix[
                self.cutmixup_range_min : self.cutmixup_range_max
            ]
        else:
            x_for_mix = self.y_list[data_name][self.idx]
            mixed = self.label_rate * x_for_mix + (1 - self.label_rate) * x_input

        return mixed

    def _adjust_length(self, x_input: np.ndarray, x_for_mix: np.ndarray) -> np.ndarray:
        if len(x_input) < len(x_for_mix):
            return np.pad(
                x_input, ((0, len(x_for_mix) - len(x_input)), (0, 0)), mode="constant"
            )
        else:
            return x_input[-len(x_for_mix) :]

    def __call__(
        self, x_input: np.ndarray, y_input: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.cutmixup_range_min = np.random.randint(
            len(x_input) // 2, 2 * len(x_input) // 3
        )
        range_max = min(self.cutmixup_range_min + (len(x_input) // 5), len(x_input) - 1)
        self.cutmixup_range_max = np.random.randint(
            self.cutmixup_range_min + 1, range_max
        )

        self.idx = random.randint(0, len(self.x_list["imu_features"]) - 1)

        x_for_mix = self.x_list["imu_features"][self.idx]
        y_for_mix = self.y_list["label"][self.idx]
        x_for_mix = self._adjust_length(x_for_mix, x_input)

        mixed_x = x_input.copy()
        cut_size = self.cutmixup_range_max - self.cutmixup_range_min + 1
        self.label_rate = cut_size / len(x_input)
        mixed_x[self.cutmixup_range_min : self.cutmixup_range_max] = x_for_mix[
            self.cutmixup_range_min : self.cutmixup_range_max
        ]
        mixed_y = self.label_rate * y_for_mix + (1 - self.label_rate) * y_input

        return mixed_x, mixed_y
