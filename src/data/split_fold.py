from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yaml  # type: ignore
from sklearn.model_selection import StratifiedGroupKFold


def split_stratified_group_kfold(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
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
        sgkf.split(df, df[target_col], df[group_col])
    ):
        splits["fold"].append(fold)
        splits["train_indices"].append(train_idx)
        splits["val_indices"].append(val_idx)

    return splits


def split_balanced_nan_group_kfold(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    nan_count_bins: int = 5,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, List[Union[str, int]]]:
    """Split DataFrame balancing both target labels and NaN counts across folds.

    This function creates balanced folds by considering:
    1. The distribution of target labels per group
    2. The distribution of NaN counts per group

    The algorithm works by creating a combined stratification key and using
    a greedy assignment approach to balance both aspects.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        group_col (str): The column name for grouping (e.g., 'subject').
        target_col (str): The column name for the target variable.
        nan_count_bins (int): Number of bins for NaN count discretization.
        n_splits (int, optional): Number of splits. Defaults to 5.
        random_state (int, optional): Random state for reproducibility.
            Defaults to 42.

    Returns:
        Dict containing fold information and train/validation indices.
    """
    np.random.seed(random_state)

    # Calculate NaN counts per group (subject) based on sequence_id
    # Count how many sequences per subject contain NaN values
    sequence_nan_status = (
        df.groupby("sequence_id")
        .apply(lambda x: x.isnull().any().any())
        .reset_index(name="has_nan")
    )

    # Merge back to get subject information for each sequence
    df_with_seq_info = df[["sequence_id", group_col]].drop_duplicates()
    sequence_with_subject = sequence_nan_status.merge(
        df_with_seq_info, on="sequence_id"
    )

    # Count number of sequences with NaN per subject
    nan_counts_per_group = (
        sequence_with_subject.groupby(group_col)["has_nan"]
        .sum()
        .reset_index(name="nan_count")
    )

    # Get target label distribution per group
    target_per_group = (
        df.groupby(group_col)[target_col]
        .first()  # Assuming same target for all rows in a group
        .reset_index()
    )

    # Combine group information
    group_info = target_per_group.merge(nan_counts_per_group, on=group_col)

    # Bin the NaN counts
    group_info["nan_count_binned"] = pd.cut(
        group_info["nan_count"], bins=nan_count_bins, labels=False
    )

    # Handle case where all values fall into one bin
    if group_info["nan_count_binned"].isnull().any():
        group_info["nan_count_binned"] = group_info["nan_count_binned"].fillna(0)

    # Create stratification key combining target and binned NaN count
    group_info["strat_key"] = (
        group_info[target_col].astype(str)
        + "_"
        + group_info["nan_count_binned"].astype(str)
    )

    # Initialize fold assignments
    group_info["fold"] = -1
    fold_sizes = [0] * n_splits

    # Group by stratification key and assign to folds
    for strat_key in group_info["strat_key"].unique():
        groups_in_key = group_info[group_info["strat_key"] == strat_key].copy()
        groups_in_key = groups_in_key.sample(
            frac=1, random_state=random_state
        )  # Shuffle

        # Assign groups to folds in round-robin fashion, prioritizing smaller folds
        for _, group_row in groups_in_key.iterrows():
            # Find the fold with minimum size
            min_fold = np.argmin(fold_sizes)
            group_info.loc[group_info[group_col] == group_row[group_col], "fold"] = (
                min_fold
            )

            # Update fold size (number of samples from this group)
            group_size = len(df[df[group_col] == group_row[group_col]])
            fold_sizes[min_fold] += group_size

    # Create splits dictionary
    splits: Dict[str, List[Union[str, int]]] = {
        "fold": [],
        "train_indices": [],
        "val_indices": [],
    }

    for fold in range(n_splits):
        # Get validation groups for this fold
        val_groups = group_info[group_info["fold"] == fold][group_col].values

        # Get indices
        val_indices = df[df[group_col].isin(val_groups)].index.values
        train_indices = df[~df[group_col].isin(val_groups)].index.values

        splits["fold"].append(fold)
        splits["train_indices"].append(train_indices)
        splits["val_indices"].append(val_indices)

    return splits


def split_sequence_stratified_kfold(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = "subject",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, List[Union[str, int]]]:
    """Split DataFrame using sequence_id as groups with stratification.

    This function ensures that:
    1. All rows from the same sequence_id are kept together
    2. All sequences from the same subject+target combination are in the same fold
    3. No sequence leakage between folds
    4. Balanced distribution of sequences with NaN across folds

    Args:
        df (pd.DataFrame): The DataFrame to split.
        target_col (str): The column name for the target variable.
        group_col (str): The column name for grouping (e.g., 'subject').
        n_splits (int, optional): Number of splits. Defaults to 5.
        random_state (int, optional): Random state for reproducibility.
            Defaults to 42.

    Returns:
        Dict containing fold information and train/validation indices.
    """
    np.random.seed(random_state)

    # Get unique subject+target combinations with their NaN information
    subject_target_combinations = (
        df.groupby([group_col, target_col]).size().reset_index(name="count")
    )

    # Get NaN status for each sequence
    sequence_nan_status = (
        df.groupby("sequence_id")
        .apply(lambda x: x.isnull().any().any())
        .reset_index(name="has_nan")
    )

    # Get subject+target info for each sequence
    sequence_subject_target = (
        df.groupby("sequence_id")[[group_col, target_col]].first().reset_index()
    )

    # Merge sequence info with NaN status
    sequence_info = sequence_subject_target.merge(sequence_nan_status, on="sequence_id")

    # Count NaN sequences for each subject+target combination
    nan_counts_per_combination = (
        sequence_info.groupby([group_col, target_col])["has_nan"]
        .sum()
        .reset_index(name="nan_sequences")
    )

    # Merge back to get complete info
    subject_target_combinations = subject_target_combinations.merge(
        nan_counts_per_combination, on=[group_col, target_col], how="left"
    )
    subject_target_combinations["nan_sequences"] = subject_target_combinations[
        "nan_sequences"
    ].fillna(0)

    # Create stratification key based on target and NaN presence
    subject_target_combinations["has_any_nan"] = (
        subject_target_combinations["nan_sequences"] > 0
    )
    subject_target_combinations["strat_key"] = (
        subject_target_combinations[target_col].astype(str)
        + "_"
        + subject_target_combinations["has_any_nan"].astype(str)
    )

    # Initialize fold assignments
    subject_target_combinations["fold"] = -1
    fold_sizes = [0] * n_splits
    fold_nan_counts = [0] * n_splits

    # Assign subject+target combinations to folds
    for strat_key in subject_target_combinations["strat_key"].unique():
        combinations_in_key = subject_target_combinations[
            subject_target_combinations["strat_key"] == strat_key
        ].copy()
        combinations_in_key = combinations_in_key.sample(
            frac=1, random_state=random_state
        )  # Shuffle

        # Assign combinations to folds, balancing both size and NaN count
        for _, comb_row in combinations_in_key.iterrows():
            # Calculate priority score for each fold
            fold_scores = []
            for fold_idx in range(n_splits):
                size_penalty = fold_sizes[fold_idx]
                # Weight NaN balance heavily
                nan_penalty = fold_nan_counts[fold_idx] * 1000
                fold_scores.append(size_penalty + nan_penalty)

            # Find the fold with minimum score
            min_fold = np.argmin(fold_scores)
            comb_mask = (
                subject_target_combinations[group_col] == comb_row[group_col]
            ) & (subject_target_combinations[target_col] == comb_row[target_col])
            subject_target_combinations.loc[comb_mask, "fold"] = min_fold

            # Update fold statistics
            # Get all sequences for this subject+target combination
            seq_mask = (sequence_info[group_col] == comb_row[group_col]) & (
                sequence_info[target_col] == comb_row[target_col]
            )
            comb_sequences = sequence_info[seq_mask]

            # Count total samples and NaN sequences
            total_samples = 0
            for seq_id in comb_sequences["sequence_id"]:
                total_samples += len(df[df["sequence_id"] == seq_id])

            fold_sizes[min_fold] += total_samples
            fold_nan_counts[min_fold] += int(comb_row["nan_sequences"])

    # Create final splits
    splits: Dict[str, List[Union[str, int]]] = {
        "fold": [],
        "train_indices": [],
        "val_indices": [],
    }

    for fold in range(n_splits):
        # Get validation subject+target combinations for this fold
        val_combinations = subject_target_combinations[
            subject_target_combinations["fold"] == fold
        ]

        # Get indices for validation data
        val_mask = pd.Series(False, index=df.index)
        for _, row in val_combinations.iterrows():
            mask = (df[group_col] == row[group_col]) & (
                df[target_col] == row[target_col]
            )
            val_mask |= mask
        val_indices = df[val_mask].index.values

        # Get indices for training data
        train_indices = df[~val_mask].index.values

        splits["fold"].append(fold)
        splits["train_indices"].append(train_indices)
        splits["val_indices"].append(val_indices)

    return splits


if __name__ == "__main__":
    # save_split_dir = Path("/kaggle/working/splits")
    df_path = Path("/kaggle/working/processed/processed_with_homotrans_df.csv")
    split_num = 5
    seed = 42
    target_col = "gesture_le"
    group_col = "subject"

    # Choose which splitting method to use
    use_balanced_nan = False  # Set to True to use balanced NaN method
    # use_sequence_group = True  # Set to True to use sequence-based grouping
    use_sequence_group = False  # Set to True to use sequence-based grouping

    df = pd.read_csv(df_path)

    if use_sequence_group:
        save_split_dir = Path("/kaggle/working/splits_sequence_group")
        print("Using sequence-based stratified grouping method...")
        splits = split_sequence_stratified_kfold(
            df=df,
            target_col=target_col,
            group_col=group_col,
            n_splits=split_num,
            random_state=seed,
        )
        split_method = "sequence_group"
    elif use_balanced_nan:
        save_split_dir = Path("/kaggle/working/splits_balanced_nan")
        print("Using balanced NaN and group splitting method...")
        splits = split_balanced_nan_group_kfold(
            df=df,
            group_col=group_col,
            target_col=target_col,
            nan_count_bins=5,
            n_splits=split_num,
            random_state=seed,
        )
        split_method = "balanced_nan"
    else:
        print("Using standard StratifiedGroupKFold...")

        save_split_dir = Path("/kaggle/working/splits")
        splits = split_stratified_group_kfold(
            df=df,
            group_col=group_col,
            target_col=target_col,
            n_splits=split_num,
            random_state=seed,
        )
        split_method = "standard"

    for fold, train_indices, valid_indices in zip(
        splits["fold"], splits["train_indices"], splits["val_indices"]
    ):
        fold_dir = save_split_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_sequences = df.iloc[train_indices]["sequence_id"].unique()  # type: ignore
        val_sequences = df.iloc[valid_indices]["sequence_id"].unique()  # type: ignore
        # save ids list to yaml file
        with open(fold_dir / "train.yaml", "w") as f:
            yaml.dump(train_sequences.tolist(), f)
        with open(fold_dir / "valid.yaml", "w") as f:
            yaml.dump(val_sequences.tolist(), f)

        # Print statistics for each fold
        print(f"Fold {fold}:")
        print(f"  Train size: {len(train_indices)}")  # type: ignore
        print(f"  Valid size: {len(valid_indices)}")  # type: ignore

        # Print target distribution
        train_target_dist = df.iloc[train_indices][target_col].value_counts()  # type: ignore
        valid_target_dist = df.iloc[valid_indices][target_col].value_counts()  # type: ignore
        print(f"  Train target distribution: {train_target_dist.to_dict()}")
        print(f"  Valid target distribution: {valid_target_dist.to_dict()}")

        # Print group (subject) distribution
        train_groups = df.iloc[train_indices][group_col].nunique()
        valid_groups = df.iloc[valid_indices][group_col].nunique()
        print(f"  Train groups: {train_groups}")
        print(f"  Valid groups: {valid_groups}")

        # Print sequence distribution
        train_sequences = df.iloc[train_indices]["sequence_id"].nunique()
        valid_sequences = df.iloc[valid_indices]["sequence_id"].nunique()
        print(f"  Train sequences: {train_sequences}")
        print(f"  Valid sequences: {valid_sequences}")

        # Print NaN count statistics for balanced_nan method
        if split_method == "balanced_nan":
            train_nan_counts = []
            valid_nan_counts = []

            # Count sequences with NaN for each subject in train set
            for group in df.iloc[train_indices][group_col].unique():
                group_sequences = df[df[group_col] == group]["sequence_id"].unique()
                sequences_with_nan = 0
                for seq_id in group_sequences:
                    seq_data = df[df["sequence_id"] == seq_id]
                    if seq_data.isnull().any().any():
                        sequences_with_nan += 1
                train_nan_counts.append(sequences_with_nan)

            # Count sequences with NaN for each subject in valid set
            for group in df.iloc[valid_indices][group_col].unique():
                group_sequences = df[df[group_col] == group]["sequence_id"].unique()
                sequences_with_nan = 0
                for seq_id in group_sequences:
                    seq_data = df[df["sequence_id"] == seq_id]
                    if seq_data.isnull().any().any():
                        sequences_with_nan += 1
                valid_nan_counts.append(sequences_with_nan)

            print(
                f"  Train NaN seq counts - Mean: {np.mean(train_nan_counts):.1f}, "
                f"Std: {np.std(train_nan_counts):.1f}"
            )
            print(
                f"  Valid NaN seq counts - Mean: {np.mean(valid_nan_counts):.1f}, "
                f"Std: {np.std(valid_nan_counts):.1f}"
            )

        # Print sequence-level statistics for sequence_group method
        elif split_method == "sequence_group":
            # Check for sequence leakage (should be 0)
            train_seq_set = set(df.iloc[train_indices]["sequence_id"].unique())
            valid_seq_set = set(df.iloc[valid_indices]["sequence_id"].unique())
            overlap = train_seq_set.intersection(valid_seq_set)
            print(f"  Sequence overlap (should be 0): {len(overlap)}")

            # Show NaN sequence distribution
            train_nan_seqs = 0
            valid_nan_seqs = 0
            for seq_id in train_seq_set:
                seq_data = df[df["sequence_id"] == seq_id]
                if seq_data.isnull().any().any():
                    train_nan_seqs += 1
            for seq_id in valid_seq_set:
                seq_data = df[df["sequence_id"] == seq_id]
                if seq_data.isnull().any().any():
                    valid_nan_seqs += 1

            print(f"  Train sequences with NaN: {train_nan_seqs}")
            print(f"  Valid sequences with NaN: {valid_nan_seqs}")

            # Check subject+target combination leakage
            train_subject_targets = set(
                df.iloc[train_indices].apply(
                    lambda x: f"{x[group_col]}_{x[target_col]}", axis=1
                )
            )
            valid_subject_targets = set(
                df.iloc[valid_indices].apply(
                    lambda x: f"{x[group_col]}_{x[target_col]}", axis=1
                )
            )
            subject_target_overlap = train_subject_targets.intersection(
                valid_subject_targets
            )
            print(
                f"  Subject+Target overlap (should be 0): {len(subject_target_overlap)}"
            )
            if len(subject_target_overlap) > 0:
                overlaps_sample = list(subject_target_overlap)[:5]
                print(f"    Overlapping combinations: {overlaps_sample}")

        print()
