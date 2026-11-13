"""`check_provenance`: a run's committed keys each have an execution or a Reuse, and none has two executions."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from pool_run_fixture import ONE_ITEM_CLAIMS, TASK, Run
from pool_stack_fixture import packaged, registry
from senselab_core.cli import main as cli_main
from senselab_core.runtime.execution.provenance_check import ProvenanceIncomplete, check_provenance, committed_keys
from senselab_core.runtime.execution.queue import RUN_QUEUES
from senselab_core.runtime.provenance import PROV_DIR

pytestmark = pytest.mark.contract


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(registry(TASK), tmp_path / "cache"):
        yield


def _served(tmp_path: Path, name: str = "run", texts: tuple[str, ...] = ("a", "b", "c")) -> Run:
    run = Run(tmp_path, name, list(texts), profile=ONE_ITEM_CLAIMS)
    run.serve(run.worker_store())
    return run


def _documents(run: Run) -> list[Path]:
    """Return the documents holding an activity: with one-item claims, one item's each."""
    return [d for d in sorted((run.run_dir / PROV_DIR).iterdir()) if '"senselab:logicalKey"' in d.read_text()]


def test_a_run_whose_items_each_have_an_execution_is_complete(tmp_path: Path) -> None:
    """Three committed keys, three executions, no Reuse."""
    report = check_provenance(_served(tmp_path).run_dir, reuses=0)
    assert (report.committed, report.executions, report.reuses, report.problems) == (3, 3, 0, ())
    report.require()


def test_a_committed_key_whose_activity_was_never_written_is_named(tmp_path: Path) -> None:
    """An outcome on disk and its execution not, as T24-10 left a worker killed after its commit."""
    run = _served(tmp_path)
    _documents(run)[0].unlink()
    report = check_provenance(run.run_dir)
    assert report.executions == 2 and [p.endswith("has no execution and no Reuse") for p in report.problems] == [True]
    with pytest.raises(ProvenanceIncomplete, match="has no execution and no Reuse"):
        report.require()


def test_a_key_executed_twice_is_named(tmp_path: Path) -> None:
    """Two executions of one key (two workers whose stores did not share it) are reported for each key."""
    first, second = _served(tmp_path / "x"), _served(tmp_path / "y")
    for document in _documents(second):
        shutil.copy(document, first.run_dir / PROV_DIR / document.name)
    problems = check_provenance(first.run_dir).problems
    assert len(problems) == 3 and all(p.endswith("has 2 executions") for p in problems)


def test_the_reuse_count_must_be_the_one_the_resume_reported(tmp_path: Path) -> None:
    """A rerun serves every item from the store: three Reuse; a count of two (or none) is a problem."""
    _served(tmp_path, "first")
    again = _served(tmp_path, "again")
    assert check_provenance(again.run_dir, reuses=3).complete
    assert check_provenance(again.run_dir, reuses=2).problems == ("3 Reuse activities, the resume reported 2",)
    assert check_provenance(again.run_dir).complete


def test_a_failed_job_and_a_dead_item_are_not_committed_keys(tmp_path: Path) -> None:
    """A job that failed stored nothing and a dead item has no jobs: neither needs an activity."""
    run = _served(tmp_path)
    [outcome] = sorted((run.run_dir / RUN_QUEUES).glob("*/outcome/*"))[:1]
    record = json.loads(outcome.read_bytes())
    ok = committed_keys(run.run_dir)
    record["jobs"][0]["failure"] = {"fault": "content"}
    outcome.write_bytes(json.dumps(record).encode())
    assert len(committed_keys(run.run_dir)) == len(ok) - 1
    outcome.write_bytes(json.dumps({"outcome": "dead"}).encode())
    assert len(committed_keys(run.run_dir)) <= len(ok) - 1


def test_status_check_provenance_exits_zero_for_a_complete_run_and_one_otherwise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`senselab status --check-provenance` prints the counts and the problems, and says so in its exit status."""
    run = _served(tmp_path)
    assert cli_main(["status", str(run.run_dir), "--check-provenance"]) == 0
    assert "provenance: 3 committed keys, 3 executions, 0 reuses: complete" in capsys.readouterr().out
    _documents(run)[0].unlink()
    assert cli_main(["status", str(run.run_dir), "--check-provenance"]) == 1
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out and "has no execution and no Reuse" in out
