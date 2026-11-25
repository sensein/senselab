"""The PII scan audit: each store under a corpus root read as the release decision reads its ``pii_scan`` records."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any

from senselab.utils.prov_store import ProvStore

audit_driver = importlib.import_module("senselab.audio.workflows.triage.drivers.triage_scan_audit")

DECLINED = {"name": "pii_scan", "scanned": False, "scanned_by": [], "failed": [], "missing": ["gliner"]}
COMPLETE = {"name": "pii_scan", "scanned_by": ["rules", "gliner"], "failed": [], "missing": []}
FAILED = {"name": "pii_scan", "scanned_by": ["rules"], "failed": ["gliner"], "missing": []}
UNATTEMPTED = {"name": "pii_scan", "scanned_by": ["rules"], "failed": [], "missing": ["gliner"]}
NOBODY = {"name": "pii_scan", "scanned_by": [], "failed": [], "missing": []}


def _store(root: Path, name: str, scans: list[dict[str, Any]], *, invalidated: bool = False) -> Path:
    """A run root ``root/<participant>/<name>`` whose store holds these ``pii_scan`` measurements."""
    run_root = root / "sub-1" / "ses-1" / name
    store = ProvStore(run_id=name)
    speech = store.activity(node="SPEECH", step=None, parameters={})
    for attributes in scans:
        scan = store.entity(prov_type="measurement", extent=None, attributes=dict(attributes))
        store.was_generated_by(scan, speech)
        if invalidated:
            store.was_invalidated_by(scan, speech)
    (run_root / "run").mkdir(parents=True)
    store.write_jsonl(run_root / "run" / "store.jsonl")
    return run_root


def test_each_store_is_read_as_the_release_decision_reads_it(tmp_path: Path) -> None:
    """Declined, complete, failed, unattempted, no detector, no record, an invalidated record and a broken store."""
    corpus = tmp_path / "corpus"
    _store(corpus, "declined", [DECLINED])
    _store(corpus, "complete", [COMPLETE])
    _store(corpus, "failed", [FAILED])
    _store(corpus, "unattempted", [UNATTEMPTED])
    _store(corpus, "nobody", [NOBODY])
    _store(corpus, "rescanned", [FAILED, COMPLETE])
    _store(corpus, "none", [])
    _store(corpus, "stale", [COMPLETE], invalidated=True)
    broken = corpus / "sub-2" / "ses-1" / "broken" / "run"
    broken.mkdir(parents=True)
    (broken / "store.jsonl").write_text("not json\n")
    summary = audit_driver.audit(corpus, tmp_path / "out")
    assert summary["stores"] == 9
    assert summary["states"] == {
        "no_pii_scan_record": 2,
        "declined": 1,
        "complete": 2,
        "incomplete": 3,
        "error": 1,
    }
    assert summary["criteria"] == {"stores": 6, "scanned_by_empty": 2, "failed_nonempty": 2, "missing_nonempty": 2}
    assert summary["criteria_declined"] == {
        "stores": 1,
        "scanned_by_empty": 1,
        "failed_nonempty": 0,
        "missing_nonempty": 1,
    }
    assert summary["earlier_rule_scanned"] == 5 and summary["affected"] == 3
    assert summary["release_reading"]["complete_scans_found_nothing_eligible"] == 2
    assert "scan_found_nothing_open" not in summary["release_reading"]
    assert json.loads((tmp_path / "out" / "summary.json").read_text()) == summary
    with (tmp_path / "out" / "not_complete.tsv").open() as handle:
        rows = {Path(row["store"]).parents[1].name: row for row in csv.DictReader(handle, delimiter="\t")}
    assert sorted(rows) == ["broken", "declined", "failed", "nobody", "none", "stale", "unattempted"]
    assert rows["failed"]["state"] == "incomplete" and rows["failed"]["failed"] == "gliner"
    assert rows["failed"]["affected"] == "true" and rows["declined"]["affected"] == "false"
    assert rows["unattempted"]["missing"] == "gliner" and rows["nobody"]["scanned_by_empty"] == "true"
    assert rows["broken"]["state"].startswith("error:")


def test_main_exits_nonzero_where_a_store_is_affected(tmp_path: Path) -> None:
    """0 over complete and declined stores; 1 once a store the earlier rule read as scanned is not; 2 for no root."""
    corpus = tmp_path / "corpus"
    _store(corpus, "complete", [COMPLETE])
    _store(corpus, "declined", [DECLINED])
    assert audit_driver.main([str(corpus), "--out-dir", str(tmp_path / "clean"), "--workers", "1"]) == 0
    _store(corpus, "failed", [FAILED])
    assert audit_driver.main([str(corpus), "--out-dir", str(tmp_path / "affected"), "--workers", "2"]) == 1
    assert json.loads((tmp_path / "affected" / "summary.json").read_text())["affected"] == 1
    assert audit_driver.main([str(tmp_path / "absent"), "--out-dir", str(tmp_path / "x")]) == 2
