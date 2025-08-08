"""Uses weak supervision to label files as include, exclude, or unsure."""

import pandas as pd
import numpy as np
from snorkel.labeling import LFAnalysis, PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel

INCLUDE = 1
EXCLUDE = 0
ABSTAIN = -1

def make_check_lf(col: str):
    @labeling_function(name=f"lf_{col}")
    def lf(x, _col=col):
        val = getattr(x, _col, None)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ABSTAIN
        if isinstance(val, str):
            val = val.strip().lower() in {"1", "true", "t", "yes", "y"}
        return EXCLUDE if bool(val) else ABSTAIN
    return lf

def make_no_checks_true_include_lf(cols):
    @labeling_function(name="lf_no_checks_true_include")
    def lf(x, _cols=tuple(cols)):
        total_true = 0
        for c in _cols:
            v = getattr(x, c, None)
            if isinstance(v, str):
                v = v.strip().lower() in {"1","true","t","yes","y"}
            total_true += int(bool(v))
        return INCLUDE if total_true == 0 else ABSTAIN
    return lf

def _prune_check_columns(df_checks: pd.DataFrame, corr_thresh: float = 0.95):
    X = df_checks.copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype(str).str.strip().str.lower().isin({"1","true","t","yes","y"})
    X = X.fillna(False).astype(bool)

    dropped = {"constant": [], "high_corr": []}

    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        dropped["constant"].extend(const_cols)
        X = X.drop(columns=const_cols)

    if X.shape[1] > 1:
        corr = X.astype(int).corr()
        to_drop = set()
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

def label_files(df_path: str, corr_thresh: float = 0.95):
    df = pd.read_csv(df_path)
    df = df[~df["audio_path_or_id"].astype(str).str.contains("Audio-Check", na=False)]
    print(f"Total files: {len(df)}")

    df_checks = df[[c for c in df.columns if "check" in c]]
    keep_cols, dropped = _prune_check_columns(df_checks, corr_thresh=corr_thresh)

    print(f"Checks total: {df_checks.shape[1]} | kept: {len(keep_cols)} | dropped: {sum(len(v) for v in dropped.values())}")
    for reason, cols in dropped.items():
        if cols:
            print(f"  - {reason} ({len(cols)}): {cols}")

    # LFs: per-check EXCLUDE + composite INCLUDE when no checks fire
    lf_list = [make_check_lf(c) for c in keep_cols]
    lf_list.append(make_no_checks_true_include_lf(keep_cols))

    applier = PandasLFApplier(lfs=lf_list)
    L_train = applier.apply(df_checks[keep_cols])

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    preds = label_model.predict(L=L_train)
    df["snorkel_label"] = preds

    abstain_mask = (L_train != ABSTAIN).sum(axis=1) == 0
    num_abstain = int(abstain_mask.sum())
    num_include = int((preds == INCLUDE).sum())
    num_exclude = int((preds == EXCLUDE).sum())

    print(f"ABSTAIN: {num_abstain}")
    print(f"INCLUDE: {num_include}")
    print(f"EXCLUDE: {num_exclude}")

    # Reliability ranking
    lf_names = [lf.name for lf in lf_list]
    reliability_rows = []
    for j, name in enumerate(lf_names):
        votes = L_train[:, j]
        fired = votes != ABSTAIN
        n_fired = int(fired.sum())
        cov = n_fired / len(df)
        if n_fired == 0:
            agree = np.nan
        else:
            agree = float((votes[fired] == preds[fired]).mean())
        reliability_rows.append({
            "lf": name,
            "coverage": round(cov, 4),
            "n_fired": n_fired,
            "agreement": round(agree, 4) if agree == agree else None
        })

    reliability_df = pd.DataFrame(reliability_rows).sort_values(
        ["agreement", "coverage"], ascending=[False, False]
    )

    print("\nLF reliability (agreement with LabelModel):")
    print(reliability_df.to_string(index=False))

    return df

path = "/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/wasabi/eipm-bridge2ai-internal-data-dissemination/2025-04-04T18.14.48.299Z/bioacoustic_quality_control_results_with_checks.csv"
label_files(path)
