"""The differential driver: manifest slicing, the mirrored tree, and the two phases.

The stores here are real, built by the same sequence the replay driver performs. What is asserted
is the driver's own behaviour: that a task takes its stride and no other, that a recording the
replay has not reached is a status rather than a failure, and that the fold over the rows writes
both products.

``specs/20260922-replay-decisions-over-a-finished-corpus/differential.md`` is the design.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from senselab.audio.workflows.triage.extend import REPLAY_MARKER_STEP, REPLAY_NODE, live_decisions, retire_decisions
from senselab.audio.workflows.triage.nodes.common import software_agent, write_verdict
from senselab.audio.workflows.triage.replay_diff import NO_STORE, OK
from senselab.audio.workflows.triage.run import entity_subdir
from senselab.audio.workflows.triage.vocabulary import FileVerdict, Release, Triage
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "triage_replay_diff.py"
_spec = importlib.util.spec_from_file_location("triage_replay_diff_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["triage_replay_diff_under_test"] = cli
_spec.loader.exec_module(cli)

TIMESTAMP = "20260919-000000"
"""The suffix the corpus run gave every run directory."""


def _decide(store: ProvStore, triage: Triage) -> None:
    """Write one file-level decision into a store.

    Args:
        store: The store to write into.
        triage: What the decision concluded.
    """
    agent = software_agent(store)
    activity = store.activity(node="VERDICT", step=None, parameters={})
    store.was_associated_with(activity, agent)
    write_verdict(
        store,
        activity,
        agent,
        node="VERDICT",
        outcome=triage,
        kind=None,
        why="the fixture's decision",
        detail=FileVerdict(triage=triage, release=Release.NOT_ASSESSED).record(),
    )


def _corpus_run(root: Path, stem: str) -> Path:
    """One finished run in the corpus tree, holding a decision and an enhanced stream.

    Args:
        root: The corpus tree.
        stem: The recording's file stem.

    Returns:
        The run root.
    """
    run_root = root / entity_subdir(stem) / f"{stem}_{TIMESTAMP}"
    (run_root / "run" / "streams").mkdir(parents=True)
    (run_root / "run" / "streams" / "enhanced.flac").write_bytes(b"")
    store = ProvStore(run_id=run_root.name)
    _decide(store, Triage.FLAG)
    store.write_jsonl(run_root / "run" / "store.jsonl")
    return run_root


def _replay_into(corpus_run: Path, out_root: Path, stem: str, triage: Triage) -> Path:
    """Replay one finished run into a mirrored tree, as the replay driver does.

    Args:
        corpus_run: The finished run root.
        out_root: The tree the mirrored root goes under.
        stem: The recording's file stem.
        triage: What the replayed decision concludes.

    Returns:
        The replayed run root.
    """
    target = out_root / entity_subdir(stem) / corpus_run.name
    (target / "run").mkdir(parents=True)
    store = ProvStore.read_jsonl(corpus_run / "run" / "store.jsonl", run_id=f"{corpus_run.name}+replay-cafe")
    agent = software_agent(store)
    retired = retire_decisions(store, live_decisions(store), software=agent)
    _decide(store, triage)
    marker = store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={"config_hash": "cafe", "commit": "b" * 40, "retired": len(retired)},
    )
    store.was_associated_with(marker, agent)
    store.write_jsonl(target / "run" / "store.jsonl")
    return target


def _manifest(path: Path, runs: dict[str, Path]) -> Path:
    """Write the manifest the replay itself took: a stem and its enhanced stream per line.

    Args:
        path: Where the JSONL goes.
        runs: Stem to its finished run root.

    Returns:
        The manifest's path.
    """
    path.write_text(
        "".join(
            json.dumps({"stem": stem, "enhanced": str(run / "run" / "streams" / "enhanced.flac")}) + "\n"
            for stem, run in runs.items()
        ),
        encoding="utf-8",
    )
    return path


class TestRows:
    """One array task reads its stride of the replayed tree and writes its rows."""

    def test_a_task_takes_its_stride_and_names_the_replayed_root(self, tmp_path: Path) -> None:
        """The mirrored root is derived the way the replay derived it, from the stem and the manifest."""
        corpus, out = tmp_path / "corpus", tmp_path / "replay"
        runs = {f"sub-{index}_ses-1_task-vowel": _corpus_run(corpus, f"sub-{index}_ses-1_task-vowel") for index in "ab"}
        for stem, run in runs.items():
            _replay_into(run, out, stem, Triage.PASS)
        manifest = _manifest(tmp_path / "manifest.jsonl", runs)

        assert (
            cli.main(["rows", str(manifest), "--slice-index", "0", "--slice-count", "2", "--out-root", str(out)]) == 0
        )
        rows = [
            json.loads(line)
            for line in (tmp_path / "slices" / "replay-diff-slice-0-of-2.jsonl").read_text().splitlines()
        ]
        assert [row["stem"] for row in rows] == ["sub-a_ses-1_task-vowel"]
        assert rows[0]["status"] == OK
        assert rows[0]["transitions"]["triage"] == "flag->pass"
        assert Path(rows[0]["run_root"]).is_relative_to(out)

    def test_a_recording_the_replay_has_not_reached_is_a_status(self, tmp_path: Path) -> None:
        """Reading the tree while the array is still running does not fail the task."""
        corpus, out = tmp_path / "corpus", tmp_path / "replay"
        stem = "sub-c_ses-1_task-vowel"
        manifest = _manifest(tmp_path / "manifest.jsonl", {stem: _corpus_run(corpus, stem)})

        assert (
            cli.main(["rows", str(manifest), "--slice-index", "0", "--slice-count", "1", "--out-root", str(out)]) == 0
        )
        [row] = [
            json.loads(line)
            for line in (tmp_path / "slices" / "replay-diff-slice-0-of-1.jsonl").read_text().splitlines()
        ]
        assert row["status"] == NO_STORE

    def test_nothing_is_written_under_either_tree(self, tmp_path: Path) -> None:
        """The rows go where the log directory names and nowhere else."""
        corpus, out, logs = tmp_path / "corpus", tmp_path / "replay", tmp_path / "logs"
        stem = "sub-d_ses-1_task-vowel"
        run = _corpus_run(corpus, stem)
        _replay_into(run, out, stem, Triage.PASS)
        manifest = _manifest(tmp_path / "manifest.jsonl", {stem: run})
        before = {path: path.stat().st_mtime_ns for path in list(corpus.rglob("*")) + list(out.rglob("*"))}

        assert (
            cli.main(
                [
                    "rows",
                    str(manifest),
                    "--slice-index",
                    "0",
                    "--slice-count",
                    "1",
                    "--out-root",
                    str(out),
                    "--log-dir",
                    str(logs),
                ]
            )
            == 0
        )
        assert (logs / "slices" / "replay-diff-slice-0-of-1.jsonl").is_file()
        assert {path: path.stat().st_mtime_ns for path in list(corpus.rglob("*")) + list(out.rglob("*"))} == before


class TestReport:
    """The fold over a tree of rows writes both products."""

    def test_both_products_are_written(self, tmp_path: Path) -> None:
        """The JSON is what was counted and the Markdown leads with the recordings that did not move."""
        corpus, out = tmp_path / "corpus", tmp_path / "replay"
        runs = {}
        for index, triage in (("a", Triage.FLAG), ("b", Triage.PASS)):
            stem = f"sub-{index}_ses-1_task-vowel"
            runs[stem] = _corpus_run(corpus, stem)
            _replay_into(runs[stem], out, stem, triage)
        manifest = _manifest(tmp_path / "manifest.jsonl", runs)
        cli.main(["rows", str(manifest), "--slice-index", "0", "--slice-count", "1", "--out-root", str(out)])

        assert cli.main(["report", str(tmp_path / "slices"), "--out", str(tmp_path / "report")]) == 0
        counted = json.loads((tmp_path / "report" / "replay_diff.json").read_text())
        assert counted["compared"] == 2
        assert counted["identical"] == 1
        assert counted["triage"] == {"flag->flag": 1, "flag->pass": 1}
        assert counted["moved_stems"]["triage"] == {"flag->pass": ["sub-b_ses-1_task-vowel"]}
        assert "1 decisions did not move at all" in (tmp_path / "report" / "replay_diff.md").read_text()

    def test_an_empty_tree_is_reported_as_empty(self, tmp_path: Path) -> None:
        """A fold over nothing says so rather than writing a report of zeroes silently."""
        (tmp_path / "slices").mkdir()
        assert cli.main(["report", str(tmp_path / "slices")]) == 1
