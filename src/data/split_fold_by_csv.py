from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml  # type: ignore


def load_predictions_and_create_splits(
    predictions_path: Path,
    train_csv_path: Path,
    save_dir: Path,
) -> None:
    """Load predictions CSV with fold information and create splits in the same format.

    Args:
        predictions_path (Path): Path to the predictions CSV file with fold column.
        train_csv_path (Path): Path to the original train.csv file.
        save_dir (Path): Directory to save the split files.
    """
    # Load predictions CSV
    pred_df = pd.read_csv(predictions_path)
    print(f"Loaded predictions from {predictions_path}")
    print(f"Columns: {pred_df.columns.tolist()}")
    print(f"Total rows: {len(pred_df)}")

    # Load train.csv to get all target_name variations
    train_df = pd.read_csv(train_csv_path)
    print(f"\nLoaded train data from {train_csv_path}")

    # Extract sample_id from image_path
    # (e.g., "train/ID1012260530.jpg" -> "ID1012260530")
    pred_df["sample_id_base"] = pred_df["image_path"].apply(
        lambda x: Path(x).stem if isinstance(x, str) else x
    )

    # Get unique folds
    folds = sorted(pred_df["fold"].unique())
    print(f"\nFolds found: {folds}")

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save split type
    split_type = "from_csv_fold"
    split_type_path = save_dir / "split_type.yaml"
    with open(split_type_path, "w") as f:
        yaml.dump(split_type, f)
    print(f"\nSaved split type to {split_type_path}")

    # Save keys information (empty since we're using CSV folds)
    group_keys_path = save_dir / "group_keys.yaml"
    stratify_keys_path = save_dir / "stratify_keys.yaml"
    with open(group_keys_path, "w") as f:
        yaml.dump(["from_csv"], f)
    with open(stratify_keys_path, "w") as f:
        yaml.dump(["fold_from_csv"], f)

    # Create splits for each fold
    for fold in folds:
        fold_dir = save_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Get validation sample_ids for this fold
        val_sample_ids_base = pred_df[pred_df["fold"] == fold][
            "sample_id_base"
        ].tolist()

        # Get training sample_ids (all other folds)
        train_sample_ids_base = pred_df[pred_df["fold"] != fold][
            "sample_id_base"
        ].tolist()

        # Create full sample_ids with target_name suffix
        # Format: ID1012260530__Dry_Clover_g
        target_names = train_df["target_name"].unique()

        train_ids = []
        for base_id in train_sample_ids_base:
            for target_name in target_names:
                train_ids.append(f"{base_id}__{target_name}")

        val_ids = []
        for base_id in val_sample_ids_base:
            for target_name in target_names:
                val_ids.append(f"{base_id}__{target_name}")

        print(f"\nFold {fold}: Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")
        print(
            f"  Train images: {len(train_sample_ids_base)},"
            f"  Val images: {len(val_sample_ids_base)}"
        )

        # Save train and validation IDs to YAML files
        train_ids_path = fold_dir / "train.yaml"
        val_ids_path = fold_dir / "valid.yaml"

        with open(train_ids_path, "w") as f:
            yaml.dump(train_ids, f)

        with open(val_ids_path, "w") as f:
            yaml.dump(val_ids, f)

    print(f"\nAll splits saved to {save_dir}")


if __name__ == "__main__":
    # Paths
    predictions_path = Path("/kaggle/input/rick_folds/all_predictions.csv")
    train_csv_path = Path("/kaggle/input/csiro-biomass/train.csv")
    save_dir = Path("/kaggle/working/splits_rick_folds")

    # Create splits from CSV
    load_predictions_and_create_splits(
        predictions_path=predictions_path,
        train_csv_path=train_csv_path,
        save_dir=save_dir,
    )
