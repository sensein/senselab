"""`check_provenance`: a run's committed keys each have an execution or a Reuse, and none has two executions."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from pool_stack_fixture import packaged, planned, registry, scoring
from senselab_core.cli import main as cli_main
from senselab_core.runtime.execution.provenance_check import ProvenanceIncomplete, check_provenance, committed_keys
from senselab_core.runtime.execution.run import Run, start
from senselab_core.runtime.provenance import PROV_DIR
from senselab_core.testing.pools import POOL_PROFILE, served
from senselab_worker.canonical import jcs_dumps

pytestmark = pytest.mark.contract


TASK = scoring("provenance_check_test.score")


@pytest.fixture(autouse=True)
def stack(tmp_path: Path) -> Iterator[None]:
    """This module's registry, every env this interpreter under an MIT lock; the cache under `tmp_path`."""
    with packaged(registry(TASK), tmp_path / "cache"):
        yield


def _served(tmp_path: Path, name: str = "run", texts: tuple[str, ...] = ("a", "b", "c")) -> Run:
    """Run one job per text on a pool served here; the store is shared by every run under `tmp_path`.

    `name` is only distinguishing: a plan's run directory is named by its hash, so a plan repeated is a new attempt.
    """
    placed, values = planned(TASK, POOL_PROFILE, list(texts))
    with served(tmp_path / f"pool-{name}", store=tmp_path / "store") as (session, _):
        run = start(placed, session, values=values)
        assert run.wait(timeout=120).done
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


def test_several_executions_of_one_key_are_allowed(tmp_path: Path) -> None:
    """ADR-010: one key can carry several executions: two attempts on two stores execute each key twice."""
    first, second = _served(tmp_path / "x"), _served(tmp_path / "y")
    assert first.run_dir == second.run_dir
    report = check_provenance(first.run_dir)
    assert report.executions == 6 and report.complete


def test_a_writer_records_at_most_one_reuse_per_item_per_attempt(tmp_path: Path) -> None:
    """A writer's document holding two Reuse of the one job with that key and node is named; one is not."""
    _served(tmp_path, "first", ("a", "b"))
    one = _served(tmp_path, "one", ("a",))
    assert check_provenance(one.run_dir, reuses=1).complete
    [reused] = [d for d in _documents(one) if "senselab:Reuse" in d.read_text()]
    merged = json.loads(reused.read_text())
    merged["@graph"] += [n for n in merged["@graph"] if "senselab:Reuse" in n["@type"]]
    reused.write_bytes(jcs_dumps(merged))
    report = check_provenance(one.run_dir)
    assert len(report.problems) == 1 and "records 2 Reuse of key" in report.problems[0]


def test_the_reuse_count_must_be_the_one_the_resume_reported(tmp_path: Path) -> None:
    """A rerun serves every item from the store: three Reuse; a count of two (or none) is a problem."""
    _served(tmp_path, "first")
    again = _served(tmp_path, "again")
    assert check_provenance(again.run_dir, reuses=3).complete
    assert check_provenance(again.run_dir, reuses=2).problems == ("3 Reuse activities, the resume reported 2",)
    assert check_provenance(again.run_dir).complete


def test_a_failed_job_is_not_a_committed_key(tmp_path: Path) -> None:
    """A job that failed stored nothing: it needs no activity."""
    run = _served(tmp_path, texts=("a", "b"))
    failing = _served(tmp_path, "failing", ("a", "fail"))
    assert committed_keys(failing.run_dir) <= committed_keys(run.run_dir) and len(committed_keys(failing.run_dir)) == 1


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
