from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yaml  # type: ignore
from sklearn.model_selection import StratifiedGroupKFold


def set_target_bins(
    df: pd.DataFrame,
    target_col: str,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Create target bins for stratification.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The column name for the target variable.
        n_bins (int, optional): Number of bins. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with an additional 'target_bin' column.
    """
    df = df.copy()
    df[f"{target_col}_bin"] = pd.qcut(
        df[target_col], q=n_bins, labels=False, duplicates="drop"
    )
    return df


def split_stratified_group_kfold(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, List[Union[str, int]]]:
    """Split the DataFrame into train and validation sets using Stratified Group K-Fold.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        group_col (str): The column name for grouping.
        target_col (str): The column name for the target variable.
        n_splits (int, optional): Number of splits. Defaults to 5.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.

    Returns:
        Dict containing fold information and train/validation indices.
    """
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits: Dict[str, List[Union[str, int]]] = {
        "fold": [],
        "train_indices": [],
        "val_indices": [],
    }
    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(df, df[stratify_col], df[group_col])
    ):
        splits["fold"].append(fold)
        splits["train_indices"].append(train_idx)
        splits["val_indices"].append(val_idx)

    return splits


def save_splits_to_yaml(
    df: pd.DataFrame,
    df_target: pd.DataFrame,
    splits: Dict[str, List[Union[str, int]]],
    save_dir: Path,
    group_keys: List[str] = ["Sampling_Date", "State"],
    stratify_keys: List[str] = ["target_bin", "Pre_GSHH_NDVI_bin", "Height_Ave_cm_bin"],
) -> None:
    """Save the splits dictionary to a YAML file.

    Args:
        splits (Dict): The splits dictionary.
        save_path (Path): The path to save the YAML file.
    """
    # save train ids
    save_dir.mkdir(parents=True, exist_ok=True)
    # group keysとstratify keysをyamlに保存
    group_keys_path = save_dir / "group_keys.yaml"
    stratify_keys_path = save_dir / "stratify_keys.yaml"
    with open(group_keys_path, "w") as f:
        yaml.dump(group_keys, f)
    with open(stratify_keys_path, "w") as f:
        yaml.dump(stratify_keys, f)
    for fold in splits["fold"]:
        fold_dir = save_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_ids_path = fold_dir / "train.yaml"
        val_ids_path = fold_dir / "valid.yaml"
        train_indices = splits["train_indices"][fold]
        val_indices = splits["val_indices"][fold]
        train_total_ids = df_target.iloc[train_indices]["sample_id"]
        val_total_ids = df_target.iloc[val_indices]["sample_id"]

        train_total_ids = [str(ids.split("_")[0]) for ids in train_total_ids]
        val_total_ids = [str(ids.split("_")[0]) for ids in val_total_ids]
        train_ids = df[df["sample_id"].str.startswith(tuple(train_total_ids))][
            "sample_id"
        ].unique()
        val_ids = df[df["sample_id"].str.startswith(tuple(val_total_ids))][
            "sample_id"
        ].unique()
        print(f"Fold {fold}: Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")
        print(
            f"image data sample num: {len(train_total_ids)}, val data sample num: {len(val_total_ids)}"
        )
        with open(train_ids_path, "w") as f:
            yaml.dump(train_ids.tolist(), f)
        with open(val_ids_path, "w") as f:
            yaml.dump(val_ids.tolist(), f)


if __name__ == "__main__":
    # Example usage
    df_path = Path("/kaggle/input/csiro-biomass/train.csv")
    df = pd.read_csv(df_path)
    print(df.head())
    target_col = "Dry_Total_g"
    df_target = df[df["target_name"] == target_col].reset_index(drop=True)
    stratify_col = "target_bin"
    group_col = "key"
    save_dir = Path(f"/kaggle/working/splits")
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    # else:
    #     print(f"{save_dir} already exists.")
    #     raise FileExistsError(f"{save_dir} already exists.")

    group_keys = ["Sampling_Date", "State"]
    stratify_keys = ["target_bin", "Pre_GSHH_NDVI_bin", "Height_Ave_cm_bin", "Species"]

    # group_keys = ["Sampling_Date", "State"]
    # stratify_keys = ["Species"]

    print("Setting target bins...")
    print("Stratify by:", stratify_keys)
    print("Group by:", group_keys)

    df_target = set_target_bins(df_target, target_col="target", n_bins=5)
    df_target = set_target_bins(df_target, target_col="Pre_GSHH_NDVI", n_bins=5)
    df_target = set_target_bins(df_target, target_col="Height_Ave_cm", n_bins=5)

    print(df_target["Pre_GSHH_NDVI_bin"].value_counts())
    print(df_target["Height_Ave_cm_bin"].value_counts())
    # Sampling_Data + Stateの結合keyを作成
    df_target["group_key"] = df_target[group_keys].astype(str).agg("_".join, axis=1)
    df_target["stratify_key"] = (
        df_target[stratify_keys].astype(str).agg("_".join, axis=1)
    )

    print(df_target["target_bin"].value_counts())
    splits = split_stratified_group_kfold(
        df_target,
        group_col="group_key",
        stratify_col="stratify_key",
        n_splits=5,
        random_state=42,
    )

    save_splits_to_yaml(
        df,
        df_target,
        splits,
        save_dir=save_dir,
        group_keys=group_keys,
        stratify_keys=stratify_keys,
    )
    print(f"Splits saved to {save_dir}")
