"""Uses weak supervision to label files as include, exclude, or unsure."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from snorkel.labeling import PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

INCLUDE: int = 1
EXCLUDE: int = 0
ABSTAIN: int = -1


def make_check_lf(col: str) -> Callable[[pd.Series], int]:
    """Create a labeling function: True in column -> EXCLUDE, else ABSTAIN."""

    @labeling_function(name=f"lf_{col}")
    def lf(x: pd.Series, _col: str = col) -> int:
        val: Any = getattr(x, _col, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ABSTAIN
        if isinstance(val, str):
            val = val.strip().lower() in {"1", "true", "t", "yes", "y"}
        return EXCLUDE if bool(val) else ABSTAIN

    return lf


def make_no_checks_true_include_lf(cols: Sequence[str]) -> Callable[[pd.Series], int]:
    """Create an LF: if all given checks are False/NA -> INCLUDE."""

    @labeling_function(name="lf_no_checks_true_include")
    def lf(x: pd.Series, _cols: Sequence[str] = tuple(cols)) -> int:
        total_true = 0
        for c in _cols:
            v: Any = getattr(x, c, None)
            if isinstance(v, str):
                v = v.strip().lower() in {"1", "true", "t", "yes", "y"}
            total_true += int(bool(v))
        return INCLUDE if total_true == 0 else ABSTAIN

    return lf


def _prune_check_columns(df_checks: pd.DataFrame, corr_thresh: float = 0.95) -> Tuple[List[str], Dict[str, List[str]]]:
    """Prune constant and highly correlated check columns.

    Args:
        df_checks: DataFrame containing only *_check columns.
        corr_thresh: Drop one of any pair with correlation >= this threshold.

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
                if corr.iloc[i, j] >= corr_thresh:
                    to_drop.add(cols[j])
        if to_drop:
            dropped["high_corr"].extend(sorted(to_drop))
            X = X.drop(columns=list(to_drop))

    keep_cols = list(X.columns)
    return keep_cols, dropped


def label_files(df_path: str, corr_thresh: float = 0.95) -> pd.DataFrame:
    """Label files using per-check LFs + a composite include LF.

    Steps:
      1) Load CSV and filter out Audio-Check rows.
      2) Prune *_check columns (constants, highly correlated).
      3) Build LFs (per-check EXCLUDE + composite INCLUDE).
      4) Train binary LabelModel, predict labels.
      5) Print counts and LF reliability; return labeled DataFrame.
    """
    df = pd.read_csv(df_path)
    df = df[~df["audio_path_or_id"].astype(str).str.contains("Audio-Check", na=False)]
    print(f"Total files: {len(df)}")

    df_checks = df[[c for c in df.columns if "check" in c]]
    keep_cols, dropped = _prune_check_columns(df_checks, corr_thresh=corr_thresh)

    dropped_total = sum(len(v) for v in dropped.values())
    print(f"Checks total: {df_checks.shape[1]} | kept: {len(keep_cols)} | " f"dropped: {dropped_total}")
    for reason, cols in dropped.items():
        if cols:
            print(f"  - {reason} ({len(cols)}): {cols}")

    # Build LFs + explicit names (avoid mypy complaining about .name)
    lf_list: List[Callable[[pd.Series], int]] = []
    lf_names: List[str] = []
    for c in keep_cols:
        lf_list.append(make_check_lf(c))
        lf_names.append(f"lf_{c}")
    lf_list.append(make_no_checks_true_include_lf(keep_cols))
    lf_names.append("lf_no_checks_true_include")

    # Apply LFs
    applier = PandasLFApplier(lfs=lf_list)
    L_train = applier.apply(df_checks[keep_cols])

    # Train LabelModel
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    # Predict
    preds = label_model.predict(L=L_train)
    df["snorkel_label"] = preds

    # Counts (abstain == no LF fired; with composite LF this should be 0)
    abstain_mask = (L_train != ABSTAIN).sum(axis=1) == 0
    print(f"ABSTAIN: {int(abstain_mask.sum())}")
    print(f"INCLUDE: {int((preds == INCLUDE).sum())}")
    print(f"EXCLUDE: {int((preds == EXCLUDE).sum())}")

    # Reliability ranking (agreement with LabelModel on rows where LF fired)
    reliability_rows: List[Dict[str, object]] = []
    n_rows = len(df)
    for j, name in enumerate(lf_names):
        votes = L_train[:, j]
        fired = votes != ABSTAIN
        n_fired = int(fired.sum())
        cov = (n_fired / n_rows) if n_rows else 0.0
        agree = float((votes[fired] == preds[fired]).mean()) if n_fired else np.nan
        reliability_rows.append(
            {
                "lf": name,
                "coverage": round(cov, 4),
                "n_fired": n_fired,
                "agreement": round(agree, 4) if agree == agree else None,
            }
        )

    reliability_df = pd.DataFrame(reliability_rows).sort_values(["agreement", "coverage"], ascending=[False, False])
    print("\nLF reliability (agreement with LabelModel):")
    print(reliability_df.to_string(index=False))

    return df


# Example call (split long path for style checks)
if __name__ == "__main__":
    path = (
        "/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/wasabi/"
        "eipm-bridge2ai-internal-data-dissemination/"
        "2025-04-04T18.14.48.299Z/"
        "bioacoustic_quality_control_results_with_checks.csv"
    )
    label_files(path)
