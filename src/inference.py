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
from src.log_utils.pylogger import RankedLogger
from src.model.architectures.model_architectures import get_model_architecture
from src.model.model_module import ModelModule

log = RankedLogger(__name__, rank_zero_only=True)


def load_config(fold_dir: Path) -> dict:
    """Load configuration from checkpoint directory"""
    config_path = fold_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    log.info(f"Loaded config from {config_path}")

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
    model.load_state_dict(torch.load(model_path))
    model.to(device)
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
    )

    return test_loader


def predict_fold(
    model: ModelModule,
    dataloader: DataLoader,
    device: str = "cuda",
) -> np.ndarray:
    """Run inference on test data"""
    model = model.to(device)
    model.eval()

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

    # Create submission dataframe
    submission_rows = []

    for i, sample_id in enumerate(sample_ids):
        for j, target_name in enumerate(target_cols):
            submission_rows.append(
                {
                    "sample_id": f"{sample_id}_{target_name}",
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
    if aug_config["resize"]:
        transforms.append(
            A.Resize(
                height=aug_config["resize_img_height"],
                width=aug_config["resize_img_width"],
                always_apply=True,
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
    output_dir: Path = Path("/kaggle/working/"),
    batch_size: int = 64,
    num_workers: int = 2,
    device: str = "cuda",
) -> pd.DataFrame:
    """Run full inference pipeline"""
    log.info("Starting inference...")

    # Load test data
    log.info(f"Loading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)

    # Create dataloader
    log.info("Creating dataloader...")
    # Run inference
    preds_sum = np.zeros((len(test_df), 5))
    for fold in folds:
        log.info(f"Running inference fold {fold}...")
        fold_dir = exp_dir / f"fold_{fold}"
        # Load configuration
        log.info("Loading configuration...")
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
        print("fold_pred shape", fold_predictions.shape)
        preds_sum += fold_predictions

    # Average predictions across folds
    predictions = preds_sum / len(folds)
    print(predictions.shape)

    # Create submission
    log.info("Creating submission...")
    submission_df = create_submission(
        predictions=predictions,
        test_df=test_df,
    )

    # Save predictions as numpy array
    # np.save(output_dir / "predictions.npy", predictions)
    # log.info(f"Predictions saved to: {output_dir / 'predictions.npy'}")

    log.info("Inference completed!")

    return submission_df


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description="Run inference on test data")
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="List of folds to use for inference",
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="best",
        help="Checkpoint name (best/last or specific filename)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to data root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for dataloader",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    test_csv = Path(args.test_csv) if args.test_csv else None
    data_root = Path(args.data_root) if args.data_root else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Run inference
    submission_df = run_inference(
        exp_dir=exp_dir,
        folds=args.folds,
        test_csv_path=test_csv,
        data_root_dir=data_root,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    # Save submission
    submission_df.to_csv(output_dir / "submission.csv", index=False)
    log.info(f"Submission saved to: {output_dir / 'submission.csv'}")

    print("\nSubmission preview:")
    print(submission_df.head())


if __name__ == "__main__":
    main()
