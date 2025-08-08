"""Uses weak supervision to label files as include, exclude, or unsure."""

import pandas as pd
from snorkel.labeling import LFAnalysis, PandasLFApplier, labeling_function
from snorkel.labeling.model import LabelModel


INCLUDE = 1
EXCLUDE = 0
ABSTAIN = -1


# load the evaluation results
# get the check columns
# define label functions that map True to exclude, False to unsure
# train the label model
# apply the label model to the data
# save the results

def make_check_lf(col: str):
    @labeling_function(name=f"lf_{col}")
    def lf(x, _col=col):
        val = getattr(x, _col, None)
        # normalize to boolean
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ABSTAIN
        if isinstance(val, str):
            val = val.strip().lower() in {"1", "true", "t", "yes", "y"}
        return EXCLUDE if bool(val) else ABSTAIN
    return lf

def label_files(df_path: str):
    quality_control_df = pd.read_csv(path)
    quality_control_df = quality_control_df[~quality_control_df["audio_path_or_id"].str.contains("Audio-Check", na=False)]
    print(len(quality_control_df))
    df_check_only = quality_control_df[
        [col for col in quality_control_df.columns if "check" in col]
    ]

    # --- Apply LFs to binary check columns ---
    check_cols = [col for col in df_check_only.columns if "check" in col]
    lf_list = [make_check_lf(c) for c in check_cols]
    applier = PandasLFApplier(lfs=lf_list)
    L_train = applier.apply(df_check_only[check_cols])

    # --- Train Label Model ---
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    # --- Predict labels ---
    preds = label_model.predict(L=L_train)
    df_check_only["snorkel_label"] = preds

    # --- Optional: Analyze ---
    # --- Count ABSTAIN / INCLUDE / EXCLUDE ---
    abstain_mask = (L_train != ABSTAIN).sum(axis=1) == 0
    num_abstain = abstain_mask.sum()
    num_include = (preds == INCLUDE).sum()
    num_exclude = (preds == EXCLUDE).sum()

    print(f"ABSTAIN: {num_abstain}")
    print(f"INCLUDE: {num_include}")
    print(f"EXCLUDE: {num_exclude}")

    # --- Optional: Analyze ---
    LFAnalysis(L=L_train, lfs=lf_list).lf_summary()
    print(df_check_only[["snorkel_label"] + check_cols].head())

path = "/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/wasabi/eipm-bridge2ai-internal-data-dissemination/2025-04-04T18.14.48.299Z/bioacoustic_quality_control_results_with_checks.csv"
label_files(path)
