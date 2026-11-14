"""`senselab-evaluate <predictions.csv> --config <file.yaml> [--out DIR]`.

Two arguments and an output directory: every other setting lives in the config, deep-merged over the
packaged `data/default.yaml`. The command scores each split of the predictions file, writes
`metrics.json` (the evaluation report plus run metadata) and, for one split, `reproduce_metrics.json`
in a reproduction driver's vocabulary. The headline is always balanced accuracy.

Exit status: 0 on success; 2 for a usage or input error (missing config key, bad column, label outside
the schema); 1 when the reports were written but the reproduction export was refused (the reason is in
`metrics.json` and on stderr).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from senselab.utils.data_structures.events import AnnotationSchema
from senselab.utils.evaluation.classification import EvaluationReport, score_splits
from senselab.utils.evaluation.export import reproduce_metrics
from senselab.utils.evaluation.thresholds import tune_threshold

DEFAULT_CONFIG = Path(__file__).parent / "data" / "default.yaml"
REQUIRED_KEYS = ("schema", "protocol")
SINGLE_SPLIT = "all"


def build_parser() -> argparse.ArgumentParser:
    """The command's argument parser: one positional, `--config` and `--out`."""
    parser = argparse.ArgumentParser(
        prog="senselab-evaluate",
        description="Score predictions with protocol-labelled, imbalance-aware reports (headline: balanced accuracy).",
    )
    parser.add_argument("predictions", type=Path, help="predictions CSV")
    parser.add_argument("--config", type=Path, required=True, help="YAML config, deep-merged over the defaults")
    parser.add_argument("--out", type=Path, default=Path("senselab_evaluate_out"), help="output directory")
    return parser


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """`override` merged into a copy of `base`, mappings merged key by key."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path) -> Tuple[Dict[str, Any], str]:
    """The user config deep-merged over the defaults, and the sha256 of the merged mapping.

    The hash is over the parsed values (canonical JSON), so comments do not change it.
    """
    defaults = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(user, dict):
        raise ValueError("the config must be a YAML mapping")
    merged = _deep_merge(defaults, user)
    canonical = json.dumps(merged, sort_keys=True, separators=(",", ":"), default=str)
    return merged, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Rows of a CSV as strings, with no missing-value conversion."""
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _version() -> str:
    """The installed senselab version, or `unknown`."""
    try:
        return metadata.version("senselab")
    except metadata.PackageNotFoundError:
        return "unknown"


def _column(row: Mapping[str, str], name: Optional[str]) -> Optional[str]:
    """A cell, or None when the column is not configured."""
    if name is None:
        return None
    if name not in row:
        raise ValueError(f"column {name!r} is not in the predictions file")
    return row[name]


def _resolve(path_value: str, config_path: Path) -> Path:
    """A config path, taken relative to the config file when it is not absolute and not found as given."""
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return config_path.parent / path


def _score(
    config: Dict[str, Any], config_path: Path, rows: List[Dict[str, str]]
) -> Tuple[EvaluationReport, Dict[str, int]]:
    """Join, split, tune (optionally) and score; returns the evaluation and the row counts."""
    if config["reader"] != {"dtype": "str", "keep_default_na": False}:
        raise ValueError("reader: only dtype str with keep_default_na false is supported")
    schema = AnnotationSchema.from_yaml(_resolve(str(config["schema"]), config_path))
    columns = config["columns"]
    counts = {"n_rows": len(rows), "n_dropped": 0, "n_unmatched_predictions": 0, "n_unscored_ground_truth": 0}
    truth_by_ref: Optional[Dict[str, str]] = None
    if config.get("ground_truth"):
        truth_rows = _read_csv(_resolve(str(config["ground_truth"]), config_path))
        truth_by_ref = {r[columns["item_ref"]]: r[columns["y_true"]] for r in truth_rows}
        seen = {r.get(columns["item_ref"]) for r in rows}
        counts["n_unscored_ground_truth"] = sum(ref not in seen for ref in truth_by_ref)
    tune_split = config["threshold"]["tune_on_split"]
    per_split: Dict[str, Dict[str, List[Any]]] = {}
    for row in rows:
        ref = _column(row, columns["item_ref"])
        if truth_by_ref is not None:
            if ref not in truth_by_ref:
                counts["n_unmatched_predictions"] += 1
                counts["n_dropped"] += 1
                continue
            y_true = truth_by_ref[str(ref)]
        else:
            y_true = _column(row, columns["y_true"]) or ""
        if y_true == "":
            counts["n_dropped"] += 1
            continue
        split = _column(row, columns["split"]) or config["export"]["split"] or SINGLE_SPLIT
        cell = per_split.setdefault(split, {"y_true": [], "y_pred": [], "y_score": [], "groups": []})
        cell["y_true"].append(y_true)
        cell["y_pred"].append(None if tune_split else _column(row, columns["y_pred"]))
        score = _column(row, columns["y_score"])
        cell["y_score"].append(None if score is None else float(score))
        cell["groups"].append(_column(row, columns["group"]))
    threshold = None
    if tune_split:
        if tune_split not in per_split or columns["y_score"] is None or config["positive_label"] is None:
            raise ValueError("threshold.tune_on_split needs that split, columns.y_score and positive_label")
        tuning = per_split.pop(tune_split)
        threshold = tune_threshold(
            tuning["y_true"],
            tuning["y_score"],
            split_name=tune_split,
            positive_label=config["positive_label"],
            objective=config["threshold"]["objective"],
        )
    inputs: Dict[str, Dict[str, Any]] = {}
    for split, cell in per_split.items():
        args: Dict[str, Any] = {
            "y_true": cell["y_true"],
            "y_pred": None if threshold is not None else cell["y_pred"],
            "labels": schema.labels,
            "protocol": config["protocol"],
            "n_dropped": counts["n_dropped"],
            "split_report": None,
            "positive_label": config["positive_label"],
            "threshold": threshold,
            "reference_prevalence": schema.reference_prevalence,
        }
        if columns["y_score"] is not None:
            args["y_score"] = cell["y_score"]
        if columns["group"] is not None:
            args.update(
                groups=cell["groups"],
                n_resamples=config["bootstrap"]["n_resamples"],
                seed=config["bootstrap"]["seed"],
            )
        inputs[split] = args
    if not inputs:
        raise ValueError("no split left to score")
    return score_splits(inputs), counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command.

    Args:
        argv: Arguments (default: the process's).

    Returns:
        The exit status: 0, or 1 when the reproduction export was refused.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config, config_sha256 = load_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(f"cannot read the config: {error}")
    for key in REQUIRED_KEYS:
        if config.get(key) in (None, ""):
            parser.error(f"config key {key!r} is required")
    try:
        evaluation, counts = _score(config, args.config, _read_csv(args.predictions))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export_split = config["export"]["split"] or (
        next(iter(evaluation.reports)) if len(evaluation.reports) == 1 else None
    )
    export: Dict[str, Any] = {"split": export_split, "written": False, "reason": None}
    status = 0
    if export_split is None:
        export["reason"] = "several splits were scored; set export.split to choose one"
    elif export_split not in evaluation.reports:
        export["reason"] = f"export.split {export_split!r} was not scored"
        status = 1
    else:
        try:
            exported = reproduce_metrics(evaluation.reports[export_split])
        except ValueError as error:
            export["reason"] = str(error)
            status = 1
        else:
            (out / "reproduce_metrics.json").write_text(json.dumps(exported, indent=2, allow_nan=False) + "\n")
            export["written"] = True
    if export["reason"] and status:
        print(f"senselab-evaluate: reproduction export refused: {export['reason']}", file=sys.stderr)
    metrics = {
        **evaluation.to_dict(),
        "config_sha256": config_sha256,
        "senselab_version": _version(),
        **counts,
        "reproduce_export": export,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return status


if __name__ == "__main__":
    sys.exit(main())
