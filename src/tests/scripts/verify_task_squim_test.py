"""The task-extent SQUIM verification report: aggregates only, and the sanity checks it enforces."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "verify_task_squim.py"
_spec = importlib.util.spec_from_file_location("verify_task_squim_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["verify_task_squim_under_test"] = cli
_spec.loader.exec_module(cli)

FIELDS = [
    "participant_id",
    "session_id",
    "task",
    "task_family",
    "branch",
    "verdict",
    "reason",
    "reasons",
    "task_start_s",
    "task_duration_s",
    "snr_db",
    "speech_over_residual_db",
    "squim_pesq",
    "squim_stoi",
    "squim_si_sdr",
]


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _rows(new: bool, *, airway_value: bool = False, voice_value: bool = False) -> list[dict[str, str]]:
    rows = []
    for i in range(40):
        snr = 5.0 + i
        rows.append(
            {
                "participant_id": f"sub-secret{i:03d}",
                "session_id": "ses-x",
                "task": "caterpillar-passage",
                "task_family": "caterpillar-passage",
                "branch": "SPEECH",
                "verdict": "pass" if i % 4 else "review",
                "reason": "" if i % 4 else "low_snr",
                "reasons": "" if i % 4 else "low_snr",
                "task_start_s": "1.0",
                "task_duration_s": "20.0",
                "snr_db": str(snr),
                "speech_over_residual_db": str(snr + 2),
                "squim_pesq": str(1.0 + 0.05 * i if new else 1.5 - 0.01 * i),
                "squim_stoi": str(0.8 + 0.004 * i if new else 0.75 - 0.002 * i),
                "squim_si_sdr": str(5.0 + 0.3 * i if new else -5.0 - 0.1 * i),
            }
        )
    rows.append(
        {
            **rows[0],
            "participant_id": "sub-airway",
            "task": "voluntary-cough",
            "task_family": "voluntary-cough",
            "branch": "AIRWAY",
            "squim_pesq": "2.0" if airway_value else "",
            "squim_stoi": "",
            "squim_si_sdr": "",
        }
    )
    rows.append(
        {
            **rows[0],
            "participant_id": "sub-voice",
            "task": "prolonged-vowel",
            "task_family": "prolonged-vowel",
            "branch": "VOICE",
            "squim_pesq": "1.3" if voice_value else "",
            "squim_stoi": "",
            "squim_si_sdr": "",
        }
    )
    return rows


def test_the_report_is_aggregates_only_and_reads_the_trend(tmp_path: Path) -> None:
    """No participant appears; new rises with SNR and old falls."""
    old = _write(tmp_path / "old.tsv", _rows(False))
    new = _write(tmp_path / "new.tsv", _rows(True))
    logs = tmp_path / "logs" / "slices"
    logs.mkdir(parents=True)
    (logs / "task-squim-slice-0-of-1.jsonl").write_text(
        json.dumps(
            {
                "stem": "sub-secret000_ses-x_task-caterpillar-passage_20261010-144634",
                "status": "ok",
                "windows_n": 2,
                "covered_s": 20.0,
                "extent_s": 20.5,
                "extent_source": "task_audio",
                "elapsed_s": 1.5,
            }
        )
        + "\n"
    )
    out = tmp_path / "verify.md"
    assert cli.main(["--old", str(old), "--new", str(new), "--logs", str(tmp_path / "logs"), "--out", str(out)]) == 0
    text = out.read_text()
    assert "sub-secret" not in text and "ses-x" not in text and "sub-airway" not in text
    assert "| snr_db | PESQ | 40 | -1.000 | 40 | +1.000 |" in text
    assert "| SPEECH | ok | 1 |" in text, "a timestamped manifest stem resolves to its branch"
    assert "### Median by snr_db decile (SPEECH rows only, deciles of SPEECH)" in text
    assert "| voice row carries a new value | 0 | 0 |" in text


def test_an_airway_value_fails_the_sanity_check(tmp_path: Path) -> None:
    """An airway row carrying a new value is a violation and a non-zero exit."""
    old = _write(tmp_path / "old.tsv", _rows(False))
    new = _write(tmp_path / "new.tsv", _rows(True, airway_value=True))
    assert cli.main(["--old", str(old), "--new", str(new), "--out", str(tmp_path / "v.md")]) == 1


def test_a_voice_value_fails_the_sanity_check(tmp_path: Path) -> None:
    """A voice row carrying a new value is a violation and a non-zero exit."""
    old = _write(tmp_path / "old.tsv", _rows(False))
    new = _write(tmp_path / "new.tsv", _rows(True, voice_value=True))
    assert cli.main(["--old", str(old), "--new", str(new), "--out", str(tmp_path / "v.md")]) == 1
