"""Streamlit app to visualize Mixup and CutMix augmentation on CloverHeightSegDataset."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.configs import AugmentationConfig, DatasetConfig
from src.data.augmentations import get_train_transforms
from src.data.clover_height_seg_dataset import CloverHeightSegDataset
from src.data.mixup_data import CutMix, Mixup


@st.cache_resource
def load_dataset_and_dataloader(batch_size=4):
    """Load dataset and dataloader with caching."""
    dataset_config = DatasetConfig()
    aug_config = AugmentationConfig()
    train_transforms = get_train_transforms(aug_config)

    df = pd.read_csv(dataset_config.df_path)
    dataset = CloverHeightSegDataset(
        df,
        data_root_dir=dataset_config.data_root_dir,
        target_cols=dataset_config.target_cols,
        phase="fit",
        transforms=train_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataset, dataloader


def denormalize_image(image_tensor):
    """Convert image tensor to numpy array for visualization.

    Args:
        image_tensor: Tensor of shape (C, H, W) or (H, W, C)

    Returns:
        Numpy array of shape (H, W, C) with values in [0, 1]
    """
    image = image_tensor.cpu().numpy()

    # Handle different shapes
    if image.ndim == 3:
        if image.shape[0] == 3:  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))

    # Clip to valid range
    image = np.clip(image, 0, 1)
    return image


def visualize_sample(inputs, labels, sample_idx=0, title_prefix=""):
    """Visualize a single sample from the batch.

    Args:
        inputs: Dict containing 'image' tensor
        labels: Dict containing various labels
        sample_idx: Index of sample in batch to visualize
        title_prefix: Prefix for the title (e.g., "Original" or "Mixed")
    """
    # Extract data
    image = inputs["image"][sample_idx]
    target_values = labels["labels"][sample_idx]
    include_clover = labels["include_clover_label"][sample_idx].item()
    height = labels["height"][sample_idx].item()
    seg_mask = labels["segmentation_mask"][sample_idx, 0]  # Remove channel dim

    # Denormalize image
    image_np = denormalize_image(image)
    seg_mask_np = seg_mask.cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title(f"{title_prefix} Image")
    axes[0].axis("off")

    # Segmentation mask
    axes[1].imshow(seg_mask_np, cmap="gray")
    axes[1].set_title(f"{title_prefix} Segmentation Mask")
    axes[1].axis("off")

    # Image with mask overlay
    axes[2].imshow(image_np)
    axes[2].imshow(seg_mask_np, cmap="Greens", alpha=0.3)
    axes[2].set_title(f"{title_prefix} Overlay")
    axes[2].axis("off")

    # Add label information
    label_text = f"Include Clover: {include_clover:.2f}\n"
    label_text += f"Height: {height:.2f} cm\n"
    label_text += f"Target Values: {target_values.cpu().numpy()}"

    fig.suptitle(label_text, fontsize=10, y=0.98)
    plt.tight_layout()

    return fig


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Mixup/CutMix Visualization", layout="wide")
    st.title("🌿 Mixup & CutMix Augmentation Visualization")
    st.markdown(
        "Visualize the effect of Mixup and CutMix augmentation on CloverHeightSegDataset"
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Augmentation type selection
    aug_type = st.sidebar.radio(
        "Augmentation Type", options=["Mixup", "CutMix"], index=0
    )

    batch_size = st.sidebar.slider("Batch Size", min_value=2, max_value=8, value=4)

    st.sidebar.subheader(f"{aug_type} Parameters")
    alpha = st.sidebar.slider(
        "Alpha (Beta distribution parameter)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )
    prob = st.sidebar.slider(
        "Probability", min_value=0.0, max_value=1.0, value=1.0, step=0.05
    )
    buffer_size = st.sidebar.number_input(
        "Buffer Size (0 for batch-only mixing)",
        min_value=0,
        max_value=1000,
        value=0,
    )

    # CutMix-specific parameter
    if aug_type == "CutMix":
        n_splits = st.sidebar.slider(
            "Grid Splits (n x n grid)",
            min_value=2,
            max_value=8,
            value=2,
            help="Number of grid divisions per dimension",
        )

    # Load data
    with st.spinner("Loading dataset..."):
        dataset, dataloader = load_dataset_and_dataloader(batch_size)

    st.success(f"Dataset loaded! Total samples: {len(dataset)}")

    # Sample selection
    sample_idx = st.sidebar.slider(
        "Sample Index in Batch", min_value=0, max_value=batch_size - 1, value=0
    )

    # Get a batch
    if st.button("Load New Batch") or "batch" not in st.session_state:
        with st.spinner("Loading batch..."):
            batch_iter = iter(dataloader)
            batch = next(batch_iter)
            st.session_state["batch"] = batch

    batch = st.session_state["batch"]
    inputs, labels = batch

    # Display original samples
    st.header("Original Sample")
    fig_original = visualize_sample(inputs, labels, sample_idx, "Original")
    st.pyplot(fig_original)
    plt.close()

    # Apply augmentation based on selection
    st.header(f"{aug_type} Applied")

    if aug_type == "Mixup":
        augmentation = Mixup(
            alpha=alpha,
            prob=prob,
            buffer_size=buffer_size,
            image_key="image",
            label_keys=[
                "labels",
                "include_clover_label",
                "height",
                "segmentation_mask",
            ],
        )
    else:  # CutMix
        augmentation = CutMix(
            alpha=alpha,
            prob=prob,
            n_splits=n_splits,
            buffer_size=buffer_size,
            image_key="image",
            label_keys=[
                "labels",
                "include_clover_label",
                "height",
                "segmentation_mask",
            ],
        )

    # Apply augmentation
    mixed_inputs, mixed_labels = augmentation(inputs, labels)

    # Display mixed samples
    fig_mixed = visualize_sample(
        mixed_inputs, mixed_labels, sample_idx, f"{aug_type} Mixed"
    )
    st.pyplot(fig_mixed)
    plt.close()

    # Show comparison side by side
    st.header("Side-by-Side Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        image_orig = denormalize_image(inputs["image"][sample_idx])
        st.image(image_orig, caption="Original Image", use_container_width=True)

        mask_orig = labels["segmentation_mask"][sample_idx, 0].cpu().numpy()
        st.image(
            mask_orig, caption="Original Mask", use_container_width=True, clamp=True
        )

    with col2:
        st.subheader(f"{aug_type} Mixed")
        image_mixed = denormalize_image(mixed_inputs["image"][sample_idx])
        st.image(
            image_mixed, caption=f"{aug_type} Mixed Image", use_container_width=True
        )

        mask_mixed = mixed_labels["segmentation_mask"][sample_idx, 0].cpu().numpy()
        st.image(
            mask_mixed,
            caption=f"{aug_type} Mixed Mask",
            use_container_width=True,
            clamp=True,
        )

    # Display label comparison
    st.header("Label Comparison")

    comparison_data = {
        "Metric": ["Include Clover", "Height (cm)"],
        "Original": [
            labels["include_clover_label"][sample_idx].item(),
            labels["height"][sample_idx].item(),
        ],
        f"{aug_type} Mixed": [
            mixed_labels["include_clover_label"][sample_idx].item(),
            mixed_labels["height"][sample_idx].item(),
        ],
    }

    st.table(pd.DataFrame(comparison_data))

    # Display target values
    st.subheader("Target Values")
    col1, col2 = st.columns(2)

    with col1:
        st.text("Original:")
        st.write(labels["labels"][sample_idx].cpu().numpy())

    with col2:
        st.text(f"{aug_type} Mixed:")
        st.write(mixed_labels["labels"][sample_idx].cpu().numpy())


if __name__ == "__main__":
    main()
