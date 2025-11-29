import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import yaml  # type: ignore
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.simple_dataset import SimpleDataset
from src.model.architectures.model_architectures import get_model_architecture
from src.model.model_module import ModelModule


def seed_everything(seed: int = 42) -> None:
    """Set seed for reproducibility"""
    import os
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def load_config(fold_dir: Path) -> dict:
    """Load configuration from checkpoint directory"""
    config_path = fold_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    print(f"Loaded config from {config_path}")
    print(config_dict)

    return config_dict


def load_model_from_checkpoint(
    config: dict, fold_dir: Path, weight_type: str, device: str = "cuda"
):
    """Load trained model from checkpoint"""

    model = get_model_architecture(
        model_name=config["model_name"],
        backbone_name=config["backbone_name"],
        pretrained=False,
        in_channels=config["in_channels"],
        n_classes=config["n_classes"],
    )
    if weight_type == "best":
        model_path = Path(f"{fold_dir}/best_weights.pth")
    elif weight_type == "final":
        model_path = Path(f"{fold_dir}/final_weights.pth")
    print(f"Loading model weights from: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def create_test_dataloader(
    test_df: pd.DataFrame,
    data_root_dir: Path,
    transforms: Compose,
    batch_size: int = 64,
    num_workers: int = 2,
) -> DataLoader:
    """Create test dataloader"""
    test_dataset = SimpleDataset(
        df=test_df,
        data_root_dir=data_root_dir,
        phase="test",
        transforms=transforms,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader


def predict_fold(
    model: ModelModule,
    dataloader: DataLoader,
    device: str = "cuda",
) -> np.ndarray:
    """Run inference on test data"""
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            inputs = batch
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(inputs)
            preds = outputs["logits"]

            # Move predictions to CPU and convert to numpy
            preds = preds.cpu().numpy()
            predictions.append(preds)

    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)

    return predictions


def create_submission(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    target_cols: list[str] = [
        "Dry_Clover_g",
        "Dry_Dead_g",
        "Dry_Green_g",
        "Dry_Total_g",
        "GDM_g",
    ],
) -> pd.DataFrame:
    """Create submission file"""
    # Get unique sample IDs
    sample_ids_targets = test_df["sample_id"].unique()
    sample_ids = [sid.split("_")[0] for sid in sample_ids_targets]
    sample_ids = list(sorted(set(sample_ids)))
    print("Number of samples:", len(sample_ids))
    # Create submission dataframe
    submission_rows = []
    for i, sample_id in enumerate(sample_ids):
        for j, target_name in enumerate(target_cols):
            submission_rows.append(
                {
                    "sample_id": f"{sample_id}__{target_name}",
                    "target": predictions[i, j],
                }
            )

    submission_df = pd.DataFrame(submission_rows)
    assert len(submission_df) == len(test_df)

    return submission_df


def get_infer_transforms(aug_config: dict) -> Compose:
    """Create validation data augmentations based on the given configuration."""
    transforms = []
    transforms.append(
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    )
    transforms.append(
        A.Resize(
            height=aug_config["resize_img_height"],
            width=aug_config["resize_img_width"],
            p=1.0,
        )
    )
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def run_inference(
    exp_dir: Path,
    folds: list[int] = [0, 1, 2, 3, 4],
    weight_type: str = "best",
    test_csv_path: Path = Path("/kaggle/input/csiro-biomass/test.csv"),
    data_root_dir: Path = Path("/kaggle/input/csiro-biomass/"),
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run full inference pipeline"""
    print("Starting inference...")

    # Load test data
    print(f"Loading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    # Create dataloader
    print("Creating dataloader...")
    # Run inference
    preds_sum = np.zeros((len(test_df) // 5, 5))
    for fold in folds:
        print(f"Running inference fold {fold}...")
        fold_dir = exp_dir / f"fold_{fold}"
        # Load configuration
        print("Loading configuration...")
        config = load_config(fold_dir=fold_dir)

        transforms = get_infer_transforms(config["augmentation"])
        test_loader = create_test_dataloader(
            test_df=test_df,
            data_root_dir=data_root_dir,
            transforms=transforms,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model = load_model_from_checkpoint(
            config=config["model"],
            fold_dir=fold_dir,
            weight_type=weight_type,
            device=device,
        )
        fold_predictions = predict_fold(model, test_loader, device=device)
        preds_sum += fold_predictions

    # Average predictions across folds
    predictions = preds_sum / len(folds)

    # Create submission
    print("Creating submission...")
    submission_df = create_submission(
        predictions=predictions,
        test_df=test_df,
    )

    # Save predictions as numpy array
    # np.save(output_dir / "predictions.npy", predictions)
    # print(f"Predictions saved to: {output_dir / 'predictions.npy'}")

    print("Inference completed!")

    return submission_df


if __name__ == "__main__":

    class EXP_CONFIG:
        # exp_dir = Path("/kaggle/input/csiro-biomass-models/models/exp_000_000")
        exp_dir = Path("/kaggle/working/exp_000_000")
        # weight_type = "final"
        weight_type = "best"
        # folds = [0, 1, 2, 3, 4]
        folds = [0]
        test_csv_path = Path("/kaggle/input/csiro-biomass/train.csv")
        data_root_dir = Path("/kaggle/input/csiro-biomass")
        output_dir = Path("/kaggle/working/")
        batch_size = 64
        num_workers = 0
        device = "cuda"

    # Run inference
    submission_df = run_inference(
        exp_dir=EXP_CONFIG.exp_dir,
        folds=EXP_CONFIG.folds,
        test_csv_path=EXP_CONFIG.test_csv_path,
        data_root_dir=EXP_CONFIG.data_root_dir,
        batch_size=EXP_CONFIG.batch_size,
        num_workers=EXP_CONFIG.num_workers,
        device=EXP_CONFIG.device,
    )
    # Save submission
    submission_df.to_csv(EXP_CONFIG.output_dir / "submission.csv", index=False)
    print(f"Submission saved to: {EXP_CONFIG.output_dir / 'submission.csv'}")
    train_df = pd.read_csv(EXP_CONFIG.test_csv_path)
    train_df = train_df[["sample_id", "target"]]
    submission_df = submission_df.rename(columns={"target": "pred"})
    scoring_df = train_df.merge(
        submission_df[["sample_id", "pred"]], on="sample_id", how="left"
    )
    print(scoring_df.head())
