import random
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


class Mixup:
    """Mixup augmentation for images and regression labels with memory buffer.

    Mixup performs linear interpolation between two samples:
    mixed_x = lambda * x1 + (1 - lambda) * x2
    mixed_y = lambda * y1 + (1 - lambda) * y2

    Uses a memory buffer to store samples from previous batches,
    enabling mixup across the entire dataset rather than just within batches.

    Supports both dict-based and tensor-based inputs/labels.

    Args:
        alpha (float): Beta distribution parameter for sampling lambda.
                      Higher values lead to stronger mixing.
        prob (float): Probability of applying mixup (0.0 to 1.0).
        buffer_size (int): Maximum number of samples to store in memory buffer.
                          0 means batch-only mixing (original behavior).
        image_key (str): Key name for image in dict-based inputs (default: "image").
        label_keys (list): List of keys to mix in dict-based labels. If None,
                           mix all keys.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
        buffer_size: int = 0,
        image_key: str = "image",
        label_keys: Optional[list] = None,
    ):
        self.alpha = alpha
        self.prob = prob
        self.buffer_size = buffer_size
        self.image_key = image_key
        self.label_keys = label_keys
        self.image_buffer = []  # type: ignore
        self.label_buffer = []  # type: ignore

    def _update_buffer(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> None:
        """Update memory buffer with new samples.

        Args:
            inputs: Batch of inputs (tensor or dict with image key)
            labels: Batch of labels (tensor or dict with label keys)
        """
        # Extract images
        if isinstance(inputs, dict):
            images = inputs[self.image_key]
        else:
            images = inputs

        # Add samples to buffer
        batch_size = images.size(0)
        for i in range(batch_size):
            if isinstance(inputs, dict):
                input_sample = {k: v[i].clone().cpu() for k, v in inputs.items()}
            else:
                input_sample = images[i].clone().cpu()
            self.image_buffer.append(input_sample)

            if isinstance(labels, dict):
                label_sample = {k: v[i].clone().cpu() for k, v in labels.items()}
            else:
                label_sample = labels[i].clone().cpu()
            self.label_buffer.append(label_sample)

        # Maintain buffer size limit
        if len(self.image_buffer) > self.buffer_size:
            excess = len(self.image_buffer) - self.buffer_size
            self.image_buffer = self.image_buffer[excess:]
            self.label_buffer = self.label_buffer[excess:]

    def _get_mix_samples(
        self, batch_size: int, device: torch.device, is_dict: bool
    ) -> Tuple[
        Union[torch.Tensor, Dict[str, torch.Tensor], None],
        Union[torch.Tensor, Dict[str, torch.Tensor], None],
    ]:
        """Get samples to mix with from buffer or batch.

        Args:
            batch_size: Number of samples needed
            device: Device to place samples on
            is_dict: Whether samples are dict-based

        Returns:
            Tuple of inputs and labels for mixing (or None, None for batch mixing)
        """
        if self.buffer_size == 0 or len(self.image_buffer) == 0:
            # Batch-only mode: return None to indicate batch mixing
            return None, None

        # Sample from buffer
        indices = np.random.choice(
            len(self.image_buffer), size=batch_size, replace=True
        )

        if is_dict:
            # Reconstruct dict batches
            mix_inputs = {}
            for key in self.image_buffer[0].keys():
                mix_inputs[key] = torch.stack(
                    [self.image_buffer[i][key] for i in indices]
                ).to(device)

            mix_labels = {}
            for key in self.label_buffer[0].keys():
                mix_labels[key] = torch.stack(
                    [self.label_buffer[i][key] for i in indices]
                ).to(device)
        else:
            # Tensor batches
            mix_inputs = torch.stack([self.image_buffer[i] for i in indices]).to(device)
            mix_labels = torch.stack([self.label_buffer[i] for i in indices]).to(device)

        return mix_inputs, mix_labels

    def __call__(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[
        Union[torch.Tensor, Dict[str, torch.Tensor]],
        Union[torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Apply mixup augmentation.

        Args:
            inputs: Batch of inputs - either tensor (B, C, H, W) or dict with image key
            labels: Batch of labels - either tensor (B, ...) or dict with label keys

        Returns:
            Tuple of mixed inputs and labels
        """
        # Update buffer with current batch samples
        if self.buffer_size > 0:
            self._update_buffer(inputs, labels)

        if random.random() > self.prob:
            return inputs, labels

        # Determine if dict-based
        is_dict = isinstance(inputs, dict)

        # Extract images and get batch info
        if is_dict:
            images = inputs[self.image_key]
            batch_size = images.size(0)
            device = images.device
        else:
            images = inputs
            batch_size = images.size(0)
            device = images.device

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Get samples to mix with
        mix_inputs, mix_labels = self._get_mix_samples(batch_size, device, is_dict)

        if mix_inputs is None:
            # Batch-only mixing
            index = torch.randperm(batch_size, device=device)
            if is_dict:
                mix_inputs = {k: v[index] for k, v in inputs.items()}
                mix_labels = {k: v[index] for k, v in labels.items()}
            else:
                mix_inputs = inputs[index]
                mix_labels = labels[index]

        # Mix inputs and labels
        if is_dict:
            mixed_inputs = {}
            for key in inputs.keys():  # type: ignore
                mixed_inputs[key] = lam * inputs[key] + (1 - lam) * mix_inputs[key]  # type: ignore

            mixed_labels = {}
            keys_to_mix = (
                self.label_keys if self.label_keys is not None else labels.keys()  # type: ignore
            )
            for key in labels.keys():  # type: ignore
                if key in keys_to_mix:
                    mixed_labels[key] = lam * labels[key] + (1 - lam) * mix_labels[key]  # type: ignore
                else:
                    # Keep original for keys not in label_keys
                    mixed_labels[key] = labels[key]
        else:
            mixed_inputs = lam * inputs + (1 - lam) * mix_inputs  # type: ignore
            mixed_labels = lam * labels + (1 - lam) * mix_labels  # type: ignore

        return mixed_inputs, mixed_labels


class CutMix:
    """CutMix augmentation for images and regression
    labels with grid-based regions and memory buffer.

    CutMix divides the image into a grid and
    replaces selected grid cells with another sample:
    - The image is divided into n_splits x n_splits grid
    - Grid cells are randomly selected based on the lambda value
    - Labels are mixed according to the actual replaced area ratio

    Uses a memory buffer to store samples from previous batches,
    enabling cutmix across the entire dataset rather than just within batches.

    Supports both dict-based and tensor-based inputs/labels.

    Args:
        alpha (float): Beta distribution parameter for sampling area ratio.
                      Higher values lead to more cells being replaced.
        prob (float): Probability of applying cutmix (0.0 to 1.0).
        n_splits (int): Number of grid divisions per dimension
                       (e.g., 2 means 2x2=4 cells).
        buffer_size (int): Maximum number of samples to store in memory buffer.
                          0 means batch-only mixing (original behavior).
        image_key (str): Key name for image in dict-based inputs (default: "image").
        label_keys (list): List of keys to mix in dict-based labels.
                           If None, mix all keys.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        prob: float = 0.5,
        n_splits: int = 2,
        buffer_size: int = 0,
        image_key: str = "image",
        label_keys: Optional[list] = None,
    ):
        self.alpha = alpha
        self.prob = prob
        self.n_splits = n_splits
        self.buffer_size = buffer_size
        self.image_key = image_key
        self.label_keys = label_keys
        self.image_buffer = []
        self.label_buffer = []

    def _get_grid_regions(
        self, size: Tuple[int, ...], lam: float
    ) -> Tuple[np.ndarray, float]:
        """Generate grid-based regions to replace.

        Args:
            size: Image size (B, C, H, W)
            lam: Lambda value for area ratio (proportion to keep from original)

        Returns:
            Tuple of binary mask (n_splits, n_splits) and actual lambda value
        """
        # Total number of grid cells
        total_cells = self.n_splits * self.n_splits

        # Number of cells to replace (based on 1-lam since lam is what we keep)
        n_replace = int(np.round(total_cells * (1 - lam)))
        n_replace = max(
            1, min(n_replace, total_cells - 1)
        )  # At least 1, at most total-1

        # Create mask for which cells to replace
        cell_indices = np.arange(total_cells)
        np.random.shuffle(cell_indices)
        replace_indices = cell_indices[:n_replace]

        # Create 2D mask
        mask = np.zeros(total_cells, dtype=bool)
        mask[replace_indices] = True
        mask = mask.reshape(self.n_splits, self.n_splits)

        # Calculate actual lambda based on selected cells
        actual_lam = 1.0 - (n_replace / total_cells)

        return mask, actual_lam

    def _update_buffer(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> None:
        """Update memory buffer with new samples.

        Args:
            inputs: Batch of inputs (tensor or dict with image key)
            labels: Batch of labels (tensor or dict with label keys)
        """
        # Extract images
        if isinstance(inputs, dict):
            images = inputs[self.image_key]
        else:
            images = inputs

        # Add samples to buffer
        batch_size = images.size(0)
        for i in range(batch_size):
            if isinstance(inputs, dict):
                input_sample = {k: v[i].clone().cpu() for k, v in inputs.items()}
            else:
                input_sample = images[i].clone().cpu()
            self.image_buffer.append(input_sample)

            if isinstance(labels, dict):
                label_sample = {k: v[i].clone().cpu() for k, v in labels.items()}
            else:
                label_sample = labels[i].clone().cpu()
            self.label_buffer.append(label_sample)

        # Maintain buffer size limit
        if len(self.image_buffer) > self.buffer_size:
            excess = len(self.image_buffer) - self.buffer_size
            self.image_buffer = self.image_buffer[excess:]
            self.label_buffer = self.label_buffer[excess:]

    def _get_mix_samples(
        self, batch_size: int, device: torch.device, is_dict: bool
    ) -> Tuple[
        Union[torch.Tensor, Dict[str, torch.Tensor], None],
        Union[torch.Tensor, Dict[str, torch.Tensor], None],
    ]:
        """Get samples to mix with from buffer or batch.

        Args:
            batch_size: Number of samples needed
            device: Device to place samples on
            is_dict: Whether samples are dict-based

        Returns:
            Tuple of inputs and labels for mixing (or None, None for batch mixing)
        """
        if self.buffer_size == 0 or len(self.image_buffer) == 0:
            # Batch-only mode: return None to indicate batch mixing
            return None, None

        # Sample from buffer
        indices = np.random.choice(
            len(self.image_buffer), size=batch_size, replace=True
        )

        if is_dict:
            # Reconstruct dict batches
            mix_inputs = {}
            for key in self.image_buffer[0].keys():
                mix_inputs[key] = torch.stack(
                    [self.image_buffer[i][key] for i in indices]
                ).to(device)

            mix_labels = {}
            for key in self.label_buffer[0].keys():
                mix_labels[key] = torch.stack(
                    [self.label_buffer[i][key] for i in indices]
                ).to(device)
        else:
            # Tensor batches
            mix_inputs = torch.stack([self.image_buffer[i] for i in indices]).to(device)
            mix_labels = torch.stack([self.label_buffer[i] for i in indices]).to(device)

        return mix_inputs, mix_labels

    def __call__(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[
        Union[torch.Tensor, Dict[str, torch.Tensor]],
        Union[torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Apply cutmix augmentation.

        Args:
            inputs: Batch of inputs - either tensor (B, C, H, W) or dict with image key
            labels: Batch of labels - either tensor (B, ...) or dict with label keys

        Returns:
            Tuple of mixed inputs and labels
        """
        # Update buffer with current batch samples
        if self.buffer_size > 0:
            self._update_buffer(inputs, labels)

        if random.random() > self.prob:
            return inputs, labels

        # Determine if dict-based
        is_dict = isinstance(inputs, dict)

        # Extract images and get batch info
        if is_dict:
            images = inputs[self.image_key]
            batch_size = images.size(0)
            device = images.device
        else:
            images = inputs
            batch_size = images.size(0)
            device = images.device

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Get samples to mix with
        mix_inputs, mix_labels = self._get_mix_samples(batch_size, device, is_dict)

        if mix_inputs is None:
            # Batch-only mixing
            index = torch.randperm(batch_size, device=device)
            if is_dict:
                mix_inputs = {k: v[index] for k, v in inputs.items()}
                mix_labels = {k: v[index] for k, v in labels.items()}
            else:
                mix_inputs = inputs[index]
                mix_labels = labels[index]

        # Extract mix_images for grid operation
        if is_dict:
            mix_images = mix_inputs[self.image_key]
        else:
            mix_images = mix_inputs

        # Get grid mask and actual lambda
        mask, actual_lam = self._get_grid_regions(images.size(), lam)

        # Apply cutmix to images based on grid
        mixed_images = images.clone()
        H, W = images.size(2), images.size(3)
        cell_h = H // self.n_splits
        cell_w = W // self.n_splits

        for i in range(self.n_splits):
            for j in range(self.n_splits):
                if mask[i, j]:  # Replace this cell
                    h_start = i * cell_h
                    h_end = (i + 1) * cell_h if i < self.n_splits - 1 else H
                    w_start = j * cell_w
                    w_end = (j + 1) * cell_w if j < self.n_splits - 1 else W

                    mixed_images[:, :, h_start:h_end, w_start:w_end] = mix_images[
                        :, :, h_start:h_end, w_start:w_end
                    ]

        # Mix labels and create output
        if is_dict:
            mixed_inputs = inputs.copy()
            mixed_inputs[self.image_key] = mixed_images

            mixed_labels = {}
            keys_to_mix = (
                self.label_keys if self.label_keys is not None else labels.keys()
            )
            for key in labels.keys():
                if key in keys_to_mix:
                    mixed_labels[key] = (
                        actual_lam * labels[key] + (1 - actual_lam) * mix_labels[key]
                    )
                else:
                    # Keep original for keys not in label_keys
                    mixed_labels[key] = labels[key]
        else:
            mixed_inputs = mixed_images
            mixed_labels = actual_lam * labels + (1 - actual_lam) * mix_labels

        return mixed_inputs, mixed_labels


class MixupCutmixWrapper:
    """Wrapper to randomly apply either Mixup or CutMix with memory buffer support.

    Supports both dict-based and tensor-based inputs/labels.

    Args:
        mixup_alpha (float): Alpha parameter for Mixup
        cutmix_alpha (float): Alpha parameter for CutMix
        prob (float): Overall probability of applying augmentation
        switch_prob (float): Probability of choosing Mixup over CutMix (0.0 to 1.0)
                            0.5 means equal probability for both
        buffer_size (int): Maximum number of samples to store in memory buffer.
                          0 means batch-only mixing (original behavior).
        n_splits (int): Number of grid divisions for CutMix
        image_key (str): Key name for image in dict-based inputs (default: "image").
        label_keys (list): List of keys to mix in dict-based labels.
                           If None, mix all keys.
    """

    def __init__(
        self,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
        buffer_size: int = 0,
        n_splits: int = 2,
        image_key: str = "image",
        label_keys: Optional[list] = None,
    ):
        self.mixup = Mixup(
            alpha=mixup_alpha,
            prob=mixup_prob,
            buffer_size=buffer_size,
            image_key=image_key,
            label_keys=label_keys,
        )
        self.cutmix = CutMix(
            alpha=cutmix_alpha,
            prob=cutmix_prob,
            n_splits=n_splits,
            buffer_size=buffer_size,
            image_key=image_key,
            label_keys=label_keys,
        )
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def __call__(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[
        Union[torch.Tensor, Dict[str, torch.Tensor]],
        Union[torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """Apply either Mixup or CutMix randomly.

        Args:
            inputs: Batch of inputs - either tensor (B, C, H, W) or dict with image key
            labels: Batch of labels - either tensor (B, ...) or dict with label keys

        Returns:
            Tuple of mixed inputs and labels
        """
        inputs, labels = self.mixup(inputs, labels)  # Update buffers first
        inputs, labels = self.cutmix(inputs, labels)  # Update buffers first
        return inputs, labels
