"""Uses weak supervision to label files as include, exclude, or unsure."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

INCLUDE: int = 1
EXCLUDE: int = 0
ABSTAIN: int = -1


def check_to_labeling_function(col: str) -> Callable[[pd.Series], int]:
    """Create a labeling function that maps check results to labels.

    Args:
        col: Column name to check for failed quality checks.

    Returns:
        A labeling function that returns EXCLUDE if the column value is True,
        otherwise ABSTAIN.
    """

    @labeling_function(name=f"{col}")
    def lf(x: pd.Series, _col: str = col) -> int:
        val: Any = getattr(x, _col, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ABSTAIN
        if isinstance(val, str):
            val = val.strip().lower() in {"1", "true", "t", "yes", "y"}
        return EXCLUDE if bool(val) else ABSTAIN

    return lf


def include_no_failed_checks_label_function(
    cols: Sequence[str],
) -> Callable[[pd.Series], int]:
    """Include a file if all given checks are False or None.

    Args:
        cols: Sequence of column names to check for failed quality checks.

    Returns:
        A labeling function that returns INCLUDE if all checks are False/None,
        otherwise ABSTAIN.
    """

    @labeling_function(name="include_no_failed_checks_label_function")
    def lf(x: pd.Series, _cols: Sequence[str] = tuple(cols)) -> int:
        total_true = 0
        for c in _cols:
            v: Any = getattr(x, c, None)
            if isinstance(v, str):
                v = v.strip().lower() in {"1", "true", "t", "yes", "y"}
            total_true += int(bool(v))
        return INCLUDE if total_true == 0 else ABSTAIN

    return lf


def prune_check_columns(
    df_checks: pd.DataFrame, correlation_threahold: float = 0.99
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Prune constant and highly correlated check columns.

    Args:
        df_checks: DataFrame containing only *_check columns.
        correlation_threahold: Drop one of any pair with correlation >=
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
                if corr.iloc[i, j] >= correlation_threahold:
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
        DataFrame with reliability metrics sorted by agreement and coverage
    """
    reliability_rows: List[Dict[str, object]] = []
    n_rows = len(preds)

    for j, name in enumerate(lf_names):
        votes = L_train[:, j]
        fired = votes != ABSTAIN
        n_audio_files = int(fired.sum())
        cov = (n_audio_files / n_rows) if n_rows else 0.0
        agree = float((votes[fired] == preds[fired]).mean()) if n_audio_files else np.nan
        reliability_rows.append(
            {
                "label_function": name,
                "coverage": round(cov, 4),
                "n_audio_files": n_audio_files,
                "agreement_with_label_model": (round(agree, 4) if agree == agree else None),
            }
        )

    return pd.DataFrame(reliability_rows).sort_values(
        ["agreement_with_label_model", "coverage"], ascending=[False, False]
    )


def review_files(
    df_path: str,
    correlation_threahold: float = 0.99,
    output_dir: Optional[str] = None,
    save_results: bool = True,
    prune_checks: bool = True,
) -> pd.DataFrame:
    """Labels audio files as include, exclude, or unsure with weak supervision.

    Args:
        df_path: Path to CSV file containing quality control results.
        correlation_threahold: Correlation threshold for pruning highly
            correlated columns.
        output_dir: Directory to save results. If None, saves to same directory
            as input CSV.
        save_results: Whether to save the results to disk.
        prune_checks: Whether to prune constant and highly correlated check columns.

    Returns:
        DataFrame with snorkel_label column added containing predicted labels.
    """
    df = pd.read_csv(df_path)
    print(f"Total files: {len(df)}")

    df_checks = df[[c for c in df.columns if "check" in c]]

    if prune_checks:
        keep_cols, dropped = prune_check_columns(df_checks, correlation_threahold=correlation_threahold)
        dropped_total = sum(len(v) for v in dropped.values())
        print(f"Checks total: {df_checks.shape[1]} | kept: {len(keep_cols)} | " f"dropped: {dropped_total}")
        for reason, cols in dropped.items():
            if cols:
                print(f"  - {reason} ({len(cols)}): {cols}")
    else:
        keep_cols = list(df_checks.columns)
        print(f"Checks total: {df_checks.shape[1]} | kept: {len(keep_cols)} | " f"dropped: 0 (pruning disabled)")

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

    # Verify alignment
    assert len(L_train) == len(df), f"Mismatch: L_train has {len(L_train)} rows, df has {len(df)} rows"

    # Train LabelModel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=200, log_freq=100, seed=123)

    # Predict
    preds = label_model.predict(L=L_train)
    df["snorkel_label"] = preds
    df["review_result_1=include"] = preds == INCLUDE

    # Counts (abstain == no labeling function fired; with composite labeling
    # function this should be 0)
    abstain_mask = (L_train != ABSTAIN).sum(axis=1) == 0
    print(f"ABSTAIN: {int(abstain_mask.sum())}")
    print(f"INCLUDE: {int((preds == INCLUDE).sum())}")
    print(f"EXCLUDE: {int((preds == EXCLUDE).sum())}")

    # Reliability ranking (agreement with LabelModel on rows where labeling
    # function fired)
    reliability_df = calculate_label_function_reliability(L_train, preds, lf_names)
    print("\nlabeling function reliability (agreement with LabelModel):")
    print(reliability_df.to_string(index=False))

    # Save results if requested
    if save_results:
        input_path = Path(df_path)
        if output_dir is None:
            output_path = input_path.parent
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        # Save labeled dataset
        base_name = input_path.stem
        labeled_csv_path = output_path / f"{base_name}_with_snorkel_labels.csv"
        df.to_csv(labeled_csv_path, index=False)
        print(f"\nSaved labeled dataset to: {labeled_csv_path}")

        # Save reliability ranking
        reliability_csv_path = output_path / f"{base_name}_labeling_function_reliability.csv"
        reliability_df.to_csv(reliability_csv_path, index=False)
        print(f"Saved reliability ranking to: {reliability_csv_path}")

    return df
