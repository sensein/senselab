"""Tests for the senselab-evaluate command line and the reproduction-driver export."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from senselab.utils.evaluation import cli
from senselab.utils.evaluation.classification import ClassificationReport, score_classification
from senselab.utils.evaluation.export import reproduce_metrics


def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _schema(tmp_path: Path, labels: tuple[str, ...] = ("neg", "pos")) -> Path:
    path = tmp_path / f"schema_{len(labels)}.yaml"
    path.write_text(yaml.safe_dump({"schema_id": "s", "task_type": "classification", "labels": list(labels)}))
    return path


def _predictions(tmp_path: Path, n: int = 120, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        y = "pos" if rng.random() < 0.3 else "neg"
        score = float(np.clip((0.7 if y == "pos" else 0.35) + rng.normal(0, 0.15), 0, 1))
        rows.append(
            {
                "item_ref": f"i{i}",
                "y_true": y,
                "y_pred": "pos" if score >= 0.5 else "neg",
                "score": f"{score:.4f}",
                "subject": f"s{i % 24}",
                "split": "val" if i % 24 < 8 else "test",
            }
        )
    return _write_csv(tmp_path / "predictions.csv", rows)


def _config(tmp_path: Path, **overrides: object) -> Path:
    config: dict = {
        "schema": str(_schema(tmp_path)),
        "protocol": "unseen_subject",
        "columns": {"split": "split", "group": "subject", "y_score": "score"},
        "positive_label": "pos",
        "bootstrap": {"n_resamples": 100},
        "export": {"split": "test"},
    }
    config.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _run(tmp_path: Path, *extra: str, config: Path | None = None) -> tuple[int, Path]:
    out = tmp_path / "out"
    code = cli.main(
        [str(_predictions(tmp_path)), "--config", str(config or _config(tmp_path)), "--out", str(out), *extra]
    )
    return code, out


def test_metrics_json_carries_the_report(tmp_path: Path) -> None:
    """metrics.json holds the protocol, prevalence by split, floors, headline values and run metadata."""
    code, out = _run(tmp_path)
    assert code == 0
    data = json.loads((out / "metrics.json").read_text())
    assert data["protocol"] == "unseen_subject"
    assert set(data["prevalence_by_split"]) == {"val", "test"}
    for report in data["reports"].values():
        assert report["headline"] == "balanced_accuracy" and isinstance(report["headline_value"], float)
        assert report["floors"]["balanced_accuracy"] == 0.5
        assert report["ci"]["balanced_accuracy"]["n_resamples"] == 100
    assert len(data["config_sha256"]) == 64 and data["n_rows"] == 120 and data["n_dropped"] == 0
    assert "senselab_version" in data and data["n_unmatched_predictions"] == 0


def test_missing_protocol_exits_2_naming_the_key(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Without a protocol the command exits 2 and says which key is missing."""
    config = _config(tmp_path, protocol=None)
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, config=config)
    assert caught.value.code == 2
    assert "protocol" in capsys.readouterr().err


def test_unknown_flag_exits_2(tmp_path: Path) -> None:
    """Per-knob flags do not exist; an unknown one is an argparse error."""
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, "--seed", "3")
    assert caught.value.code == 2


def test_help_shows_one_positional_and_two_options(capsys: pytest.CaptureFixture) -> None:
    """The command takes one positional argument plus --config and --out, and nothing else."""
    parser = cli.build_parser()
    positionals = [a.dest for a in parser._actions if not a.option_strings]
    options = {s for a in parser._actions for s in a.option_strings}
    assert positionals == ["predictions"] and options == {"-h", "--help", "--config", "--out"}
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    text = capsys.readouterr().out
    assert "predictions" in text and "--config" in text and "--out" in text


def test_row_without_ground_truth_is_dropped_and_counted(tmp_path: Path) -> None:
    """A prediction whose item_ref is not in the ground truth is dropped and counted."""
    predictions = _predictions(tmp_path)
    with predictions.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    truth = _write_csv(tmp_path / "truth.csv", [{"item_ref": r["item_ref"], "y_true": r["y_true"]} for r in rows[1:]])
    config = _config(tmp_path, ground_truth=str(truth))
    out = tmp_path / "out"
    assert cli.main([str(predictions), "--config", str(config), "--out", str(out)]) == 0
    data = json.loads((out / "metrics.json").read_text())
    assert data["n_dropped"] == 1 and data["n_unmatched_predictions"] == 1


def test_config_hash_follows_values_not_comments(tmp_path: Path) -> None:
    """Changing a value changes the hash; changing only a comment does not."""
    base = _config(tmp_path)
    text = base.read_text()
    first = cli.load_config(base)[1]
    base.write_text("# a comment that changes nothing\n" + text)
    assert cli.load_config(base)[1] == first
    base.write_text(text.replace("n_resamples: 100", "n_resamples: 101"))
    assert cli.load_config(base)[1] != first


def test_default_config_has_no_derivation_and_no_headline_key() -> None:
    """The packaged defaults carry no prose key and no way to pick another headline."""
    raw = yaml.safe_load(cli.DEFAULT_CONFIG.read_text())
    assert "derivation" not in raw and "headline" not in raw and raw["protocol"] is None


def test_threshold_tuned_on_validation_is_applied_to_test(tmp_path: Path) -> None:
    """A threshold tuned on the val split scores the test split; the tuning split is not scored."""
    config = _config(tmp_path, threshold={"tune_on_split": "val", "objective": "f1"})
    code, out = _run(tmp_path, config=config)
    assert code == 0
    data = json.loads((out / "metrics.json").read_text())
    assert set(data["reports"]) == {"test"}
    report = data["reports"]["test"]
    assert report["threshold"]["objective"] == "f1" and report["threshold"]["split_name"] == "val"
    assert report["threshold_tuning"] == "validation" and report["headline"] == "balanced_accuracy"


def test_reproduce_export_is_written_in_the_driver_vocabulary(tmp_path: Path) -> None:
    """reproduce_metrics.json uses the driver's keys, names and floors for the exported split."""
    code, out = _run(tmp_path)
    assert code == 0
    data = json.loads((out / "reproduce_metrics.json").read_text())
    assert set(data) <= {"headline", "secondary", "per_class", "confusion", "split_report", "threshold_tuning", "n"}
    (headline,) = data["headline"]
    assert headline["name"] == "balanced_accuracy" and headline["floor"] == 0.5
    assert {"ci_low", "ci_high", "value", "lift", "n"} <= set(headline)
    names = {entry["name"] for entry in data["secondary"]}
    assert names == {"accuracy", "macro_f1", "mcc", "f1", "auroc", "pr_auc"}
    assert set(data["per_class"]) == {"neg", "pos"}
    assert all(isinstance(v, float) for c in data["per_class"].values() for v in c.values())
    assert data["threshold_tuning"] == "none" and set(data["confusion"]) == {"tp", "fp", "fn", "tn"}


def _report(y_true: list, y_pred: list, labels: list) -> ClassificationReport:
    return score_classification(
        y_true, y_pred, labels=labels, protocol="unseen_subject", split_name="test", n_dropped=0, split_report=None
    )


def test_export_refuses_a_declared_class_absent_from_the_split() -> None:
    """Recall of an absent class is undefined, so the export is refused rather than invented."""
    with pytest.raises(ValueError, match="absent"):
        reproduce_metrics(_report(["a", "a", "b"], ["a", "b", "b"], ["a", "b", "c"]))


def test_export_never_predicted_class_has_zero_precision() -> None:
    """A present class that is never predicted exports precision 0.0 (no correct predictions of it)."""
    data = reproduce_metrics(_report(["a", "a", "b", "b"], ["a", "a", "a", "a"], ["a", "b"]))
    assert data["per_class"]["b"]["precision"] == 0.0 and data["per_class"]["b"]["recall"] == 0.0


def test_export_omits_confusion_with_invalid_outputs() -> None:
    """Invalid outputs have no cell in a 2 x 2 table, so the confusion block is left out."""
    report = score_classification(
        ["neg", "pos", "pos"],
        ["neg", "pos", "INVALID"],
        labels=["neg", "pos"],
        protocol="unseen_subject",
        split_name="test",
        n_dropped=0,
        split_report=None,
        positive_label="pos",
    )
    assert "confusion" not in reproduce_metrics(report)


def test_export_refused_by_the_cli_still_writes_metrics(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """When the export is refused, metrics.json is written with the reason and the command exits 1."""
    config = _config(
        tmp_path,
        schema=str(_schema(tmp_path, ("neg", "pos", "maybe"))),
        positive_label=None,
        columns={"split": "split", "group": "subject", "y_score": None},
    )
    code, out = _run(tmp_path, config=config)
    assert code == 1
    data = json.loads((out / "metrics.json").read_text())
    assert "absent" in data["reproduce_export"]["reason"] and not (out / "reproduce_metrics.json").exists()
    assert "absent" in capsys.readouterr().err
