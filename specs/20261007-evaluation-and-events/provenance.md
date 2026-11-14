# Provenance of ported code

First-party lab code is ported under senselab's Apache-2.0 licence by decision of the principal
investigator. Each row names the senselab symbol, the origin it was ported from, and its authors.

- Origins are cited by full commit hash. Any path segment that names the originating project is
  replaced by `<package>`.
- Authors are credited by confirmed full name. Until a name is confirmed, the entry reads
  "sensein lab member".
- Ported code is re-documented: no docstring or comment is carried over verbatim.

| senselab symbol | Origin | Authors |
|---|---|---|
| `delong_test` | `private sensein repository @ c2b11516b305799544ee6b4c6af52360f80c3b1d:evaluation/delong_and_effect_sizes.py` | sensein lab member |
| `cluster_bootstrap_ci`, `legacy_cluster_bootstrap` | `private sensein repository @ c2b11516b305799544ee6b4c6af52360f80c3b1d:evaluation/bootstrap_ci_groupstrat.py` | sensein lab member |
| `tune_threshold` | `private sensein repository @ c2b11516b305799544ee6b4c6af52360f80c3b1d:mil/mil_utils.py` | sensein lab member |
| `grouped_splits` (stratified group k-fold wrapper) | `private sensein repository @ c95b727c2ee06d226cdd2c168afddf731923e9e4:evaluation/group_stratified_kfold.py` | sensein lab member |
| `event_f1` | `private sensein repository @ c95b727c2ee06d226cdd2c168afddf731923e9e4:evaluation/onset_tolerance_f1.py` | sensein lab member |
| `score_classification` (legacy macro-recall formula for balanced accuracy) | `private sensein repository @ 8c4a035cded9fd5587b86dd149f510267026c2b7:<package>/evaluation/metrics.py` | sensein lab member |
| `segment_map` | clean-room implementation from the ActivityNet temporal-localisation definition; no lab code read | — |
