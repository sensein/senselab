"""Uses weak supervision to label files as include, exclude, or unsure."""

from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

from senselab.audio.tasks.quality_control.taxonomies import (
    BIOACOUSTIC_ACTIVITY_TAXONOMY,
)
from senselab.audio.tasks.quality_control.taxonomy import TaxonomyNode
from senselab.utils.data_structures.logging import logger


class Label(IntEnum):
    """Label values for weak supervision review results.

    Note: ABSTAIN is used internally by labeling functions when they cannot
    make a decision (e.g., missing data). However, the final predictions
    from the LabelModel will only be INCLUDE or EXCLUDE, never ABSTAIN.
    This is because: (1) the composite labeling function always fires, and
    (2) the LabelModel is configured with cardinality=2 (binary classification).
    """

    INCLUDE = 1
    EXCLUDE = 0
    ABSTAIN = -1  # Used internally only; never appears in final predictions


def get_taxonomy_check_names(taxonomy: TaxonomyNode, activity: str = "bioacoustic") -> List[str]:
    """Extract check function names from taxonomy for a given activity.

    Args:
        taxonomy: The taxonomy tree to extract checks from
        activity: The activity to get checks for

    Returns:
        List of check function names from the taxonomy
    """
    subtree = taxonomy.prune_to_activity(activity)
    if subtree is None:
        raise ValueError(f"Activity '{activity}' not found in taxonomy.")

    evaluations = subtree.get_all_evaluations()

    # Filter to only check functions (those that return bool)
    check_names = []
    for evaluation in evaluations:
        func_name = evaluation.__name__
        # Check functions typically end with "_check"
        if func_name.endswith("_check"):
            check_names.append(func_name)

    return check_names


def check_to_labeling_function(col: str) -> Callable[[pd.Series], int]:
    """Create a labeling function that maps check results to labels.

    Args:
        col: Column name to check for failed quality checks.

    Returns:
        A labeling function that returns:
        - Label.EXCLUDE if the column value is True (check failed)
        - Label.INCLUDE if the column value is False (check passed)
        - Label.ABSTAIN if the column value is None/NaN (missing data)

    Note: ABSTAIN is used internally when data is missing, but the final
    predictions will only be INCLUDE or EXCLUDE due to the composite
    labeling function and binary LabelModel.
    """

    @labeling_function(name=f"{col}")
    def lf(x: pd.Series, _col: str = col) -> int:
        val: Any = getattr(x, _col, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return Label.ABSTAIN
        if isinstance(val, str):
            val = val.strip().lower() in {"1", "true", "t", "yes", "y"}
        return Label.EXCLUDE if bool(val) else Label.INCLUDE

    return lf


def include_no_failed_checks_label_function(
    cols: Sequence[str],
) -> Callable[[pd.Series], int]:
    """Include a file if all given checks are False or None.

    This composite labeling function ensures at least one labeling function
    always fires for every row, preventing ABSTAIN in the final predictions.

    Args:
        cols: Sequence of column names to check for failed quality checks.

    Returns:
        A labeling function that returns:
        - Label.INCLUDE if all checks are False/None (no failures)
        - Label.EXCLUDE if any check is True (at least one failure)

    Note: This function always returns INCLUDE or EXCLUDE, never ABSTAIN.
    This guarantees that every file gets a final label.
    """

    @labeling_function(name="include_no_failed_checks_label_function")
    def lf(x: pd.Series, _cols: Sequence[str] = tuple(cols)) -> int:
        total_true = 0
        for c in _cols:
            v: Any = getattr(x, c, None)
            if isinstance(v, str):
                v = v.strip().lower() in {"1", "true", "t", "yes", "y"}
            total_true += int(bool(v))
        return Label.INCLUDE if total_true == 0 else Label.EXCLUDE

    return lf


def prune_check_columns(
    df_checks: pd.DataFrame, correlation_threshold: float = 0.99
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Prune constant and highly correlated check columns.

    Args:
        df_checks: DataFrame containing only *_check columns.
        correlation_threshold: Drop one of any pair with correlation >=
            this threshold.

    Returns:
        keep_cols: Names of check columns to keep.
        dropped: Dict of reasons -> list of dropped column names.
    """
    X = df_checks.copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    X = X.fillna(False).astype(bool)

    dropped: Dict[str, List[str]] = {"constant": [], "high_corr": []}

    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        dropped["constant"].extend(const_cols)
        X = X.drop(columns=const_cols)

    if X.shape[1] > 1:
        corr = X.astype(int).corr()
        to_drop: set[str] = set()
        cols = list(corr.columns)
        for i in range(len(cols)):
            if cols[i] in to_drop:
                continue
            for j in range(i + 1, len(cols)):
                if cols[j] in to_drop:
                    continue
                if corr.iloc[i, j] >= correlation_threshold:
                    to_drop.add(cols[j])
        if to_drop:
            dropped["high_corr"].extend(sorted(to_drop))
            X = X.drop(columns=list(to_drop))

    keep_cols = list(X.columns)
    return keep_cols, dropped


def calculate_label_function_reliability(L_train: np.ndarray, preds: np.ndarray, lf_names: List[str]) -> pd.DataFrame:
    """Calculate reliability metrics for labeling functions.

    Args:
        L_train: Matrix of labeling function votes
            (n_examples x n_label_functions)
        preds: Label model predictions
        lf_names: Names of the labeling functions

    Returns:
        DataFrame with reliability metrics sorted by agreement and vote counts
    """
    reliability_rows: List[Dict[str, object]] = []

    for j, name in enumerate(lf_names):
        votes = L_train[:, j]
        fired = votes != Label.ABSTAIN
        n_voted = int(fired.sum())

        # Count votes for include/exclude when labeling function fired
        voted_include = int((votes[fired] == Label.INCLUDE).sum()) if n_voted else 0
        voted_exclude = int((votes[fired] == Label.EXCLUDE).sum()) if n_voted else 0

        agree = float((votes[fired] == preds[fired]).mean()) if n_voted else np.nan
        reliability_rows.append(
            {
                "label_function": name,
                "voted_include": voted_include,
                "voted_exclude": voted_exclude,
                "agreement_with_label_model": (round(agree, 4) if agree == agree else None),
            }
        )

    return pd.DataFrame(reliability_rows).sort_values(
        ["agreement_with_label_model", "voted_include"], ascending=[False, False]
    )


def review_files(
    df_path: Union[str, pd.DataFrame],
    correlation_threshold: float = 0.99,
    output_dir: Optional[str] = None,
    save_results: bool = True,
    prune_checks: bool = True,
    taxonomy: TaxonomyNode = BIOACOUSTIC_ACTIVITY_TAXONOMY,
    activity: str = "bioacoustic",
) -> pd.DataFrame:
    """Labels audio files as include or exclude using weak supervision.

    Uses Snorkel's weak supervision framework to combine multiple quality
    check labeling functions into a single prediction per file.

    Args:
        df_path: Path to CSV file containing quality control results, or a
            DataFrame directly. If a DataFrame is provided, output_dir must be
            specified when save_results=True.
        correlation_threshold: Correlation threshold for pruning highly
            correlated columns.
        output_dir: Directory to save results. If None and df_path is a string,
            saves to same directory as input CSV. If None and df_path is a DataFrame,
            defaults to "qc_results" when save_results=True.
        save_results: Whether to save the results to disk.
        prune_checks: Whether to prune constant and highly correlated check columns.
        taxonomy: The taxonomy tree to extract checks from.
        activity: The activity to get checks for.

    Returns:
        DataFrame with snorkel_label column added containing predicted labels.
        Values will be 1 (INCLUDE) or 0 (EXCLUDE), never -1 (ABSTAIN).

    Note:
        The ABSTAIN label is used internally by individual labeling functions
        when data is missing, but the final predictions are always INCLUDE or
        EXCLUDE. This is guaranteed by: (1) a composite labeling function that
        always fires, and (2) a binary LabelModel (cardinality=2).
    """
    # Handle both DataFrame and file path inputs
    if isinstance(df_path, pd.DataFrame):
        df = df_path.copy()
        input_path = None
    else:
        df = pd.read_csv(df_path)
        input_path = Path(df_path)

    logger.info(f"Total files: {len(df)}")

    # Get check names from taxonomy
    taxonomy_check_names = get_taxonomy_check_names(taxonomy, activity)
    logger.info(f"Taxonomy checks for '{activity}': {taxonomy_check_names}")

    # Filter to only columns that exist in both the DataFrame and taxonomy
    available_check_cols = [c for c in df.columns if "check" in c]
    taxonomy_check_cols = [c for c in available_check_cols if c in taxonomy_check_names]

    logger.info(f"Available check columns: {len(available_check_cols)}")
    logger.info(f"Taxonomy-filtered check columns: {len(taxonomy_check_cols)}")

    if not taxonomy_check_cols:
        raise ValueError(f"No check columns found that match taxonomy checks for activity '{activity}'")

    df_checks = df[taxonomy_check_cols]

    if prune_checks:
        keep_cols, dropped = prune_check_columns(df_checks, correlation_threshold=correlation_threshold)
        dropped_total = sum(len(v) for v in dropped.values())
        logger.info(f"Checks total: {df_checks.shape[1]} | kept: {len(keep_cols)} | dropped: {dropped_total}")
        for reason, cols in dropped.items():
            if cols:
                logger.info(f"  - {reason} ({len(cols)}): {cols}")
    else:
        keep_cols = list(df_checks.columns)
        logger.info(f"Checks total: {df_checks.shape[1]} | kept: {len(keep_cols)} | dropped: 0 (pruning disabled)")

    # Check if we have any columns left after pruning
    if not keep_cols:
        logger.warning(
            "All check columns were pruned (likely all constant values). "
            "Cannot perform weak supervision labeling. Assigning all files as INCLUDE."
        )
        df["snorkel_label"] = Label.INCLUDE
        df["review_result_1=include"] = True
        logger.info(f"Assigned all {len(df)} files as INCLUDE (no quality issues detected)")
        return df

    # Build labeling functions + explicit names (avoid mypy complaining
    # about .name)
    lf_list: List[Callable[[pd.Series], int]] = []
    lf_names: List[str] = []
    for c in keep_cols:
        lf_list.append(check_to_labeling_function(c))
        lf_names.append(f"{c}")
    lf_list.append(include_no_failed_checks_label_function(keep_cols))
    lf_names.append("no_checks_true_include_label_function")

    # Apply labeling functions
    applier = PandasLFApplier(lfs=lf_list)
    df_checks_filtered = df_checks[keep_cols]
    L_train = applier.apply(df_checks_filtered)

    # Train LabelModel
    # cardinality=2 means binary classification: only INCLUDE (1) or EXCLUDE (0)
    # ABSTAIN (-1) will never appear in predictions
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=200, log_freq=100, seed=123)

    # Predict
    preds = label_model.predict(L=L_train)
    df["snorkel_label"] = preds
    df["review_result_1=include"] = preds == Label.INCLUDE

    # Counts
    # Note: abstain_mask checks if NO labeling function fired for a row.
    # With the composite labeling function, this should always be 0.
    # Final predictions (preds) will only be INCLUDE or EXCLUDE, never ABSTAIN.
    abstain_mask = (L_train != Label.ABSTAIN).sum(axis=1) == 0
    logger.info(f"Rows with no labeling function votes (should be 0): {int(abstain_mask.sum())}")
    logger.info(f"INCLUDE: {int((preds == Label.INCLUDE).sum())}")
    logger.info(f"EXCLUDE: {int((preds == Label.EXCLUDE).sum())}")

    # Reliability ranking (agreement with LabelModel on rows where labeling
    # function fired)
    reliability_df = calculate_label_function_reliability(L_train, preds, lf_names)
    logger.info("\nlabeling function reliability (agreement with LabelModel):")
    logger.info(reliability_df.to_string(index=False))

    # Save results if requested
    if save_results:
        if output_dir is None:
            if input_path is not None:
                output_path = input_path.parent
            else:
                output_path = Path("qc_results")
                output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Save labeled dataset
        if input_path is not None:
            base_name = input_path.stem
        else:
            base_name = "quality_control_results"
        labeled_csv_path = output_path / f"{base_name}_with_snorkel_labels.csv"
        df.to_csv(labeled_csv_path, index=False)
        logger.info(f"\nSaved labeled dataset to: {labeled_csv_path}")

        # Save reliability ranking
        reliability_csv_path = output_path / f"{base_name}_labeling_function_reliability.csv"
        reliability_df.to_csv(reliability_csv_path, index=False)
        logger.info(f"Saved reliability ranking to: {reliability_csv_path}")

    return df
