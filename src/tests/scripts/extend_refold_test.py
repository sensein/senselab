"""The fold-only driver: a corpus brought to a fold change without reading anything again.

A change to the release vocabulary, a flag ground or a ``verdict.*`` key changes what the store's
existing evidence *means*. Bringing 62,548 recordings to such a change is minutes of CPU through
this driver, or days of GPU through a replay. These pin that it decides again, that it touches
nothing else, and that it refuses to run blind.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.extend import REFOLD_MARKER_STEP, REFOLD_NODE
from senselab.audio.workflows.triage.nodes.common import software_agent, write_verdict
from senselab.audio.workflows.triage.run import LOG_FILE
from senselab.audio.workflows.triage.vocabulary import REDACTION_LLM_ANNOTATION, Outcome
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import word_attributes

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_refold.py"
_spec = importlib.util.spec_from_file_location("extend_refold_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_refold_under_test"] = cli
_spec.loader.exec_module(cli)

SR = 16000


def _config(tmp_path: Path, text: str = "") -> Any:  # noqa: ANN401 — TriageConfig, kept off for brevity
    """The packaged config, with a partial YAML over it when one is given."""
    if not text:
        return load_triage_config()
    path = tmp_path / "override.yaml"
    path.write_text(text, encoding="utf-8")
    return load_triage_config(path)


def _finished_run(root: Path, *, words: Sequence[str] = ("hello", "alicia")) -> Path:
    """A finished run: streams, consensus words, a run log, and one verdict per deciding node."""
    run_root = root / "sub-a_task-free-speech_20260925-000000"
    streams = run_root / "run" / "streams"
    streams.mkdir(parents=True, exist_ok=True)
    wave = (0.05 * np.random.default_rng(0).standard_normal(SR * 3)).astype(np.float32)
    for name in ("enhanced", "plain", "recording"):
        sf.write(str(streams / f"{name}.flac"), wave, SR)

    store = ProvStore(run_id=run_root.name)
    software = store.agent(agent_type="software", version="senselab test-seed")
    activity = store.activity(node="PREPROCESS", step="consensus", parameters={})
    store.was_associated_with(activity, software)
    for name in ("recording", "enhanced", "plain"):
        stream = store.entity(
            prov_type="stream",
            extent=(0.0, 3.0),
            attributes={"name": name, "path": f"streams/{name}.flac", "sampling_rate": SR, "channels": 1},
        )
        store.was_generated_by(stream, activity)
    word_ids = []
    for index, text in enumerate(words):
        extent = (float(index), float(index) + 0.5)
        word_id = store.entity(prov_type="word", extent=extent, attributes=word_attributes(text, extent, index=index))
        store.was_generated_by(word_id, activity)
        word_ids.append(word_id)
    transcript = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "consensus_transcript",
            "signal": "plain",
            "role": "consensus",
            "n_words": len(words),
            "word_ids": word_ids,
            "text": " ".join(words),
        },
    )
    store.was_generated_by(transcript, activity)
    for node, outcome in (("ADMIT", Outcome.PASS), ("SPEECH", Outcome.PASS), ("VERDICT", Outcome.FLAG)):
        act = store.activity(node=node, step="seed", parameters={})
        store.was_associated_with(act, software)
        write_verdict(store, act, software, node=node, outcome=outcome, kind=None, why="seeded", detail={})
    store.write_jsonl(run_root / "run" / "store.jsonl")
    (run_root / "run" / LOG_FILE).write_text(
        json.dumps({"source": str(streams / "recording.flac")}), encoding="utf-8"
    )
    return run_root


def _manifest(tmp_path: Path, run_root: Path) -> Path:
    """The manifest the extend family takes, over one run."""
    path = tmp_path / "manifest.jsonl"
    row = {"stem": run_root.name, "enhanced": str(run_root / "run" / "streams" / "enhanced.flac")}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _hints(tmp_path: Path) -> Path:
    """A ``hints.py`` naming the task this run declares."""
    directory = tmp_path / "scope"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hints.py").write_text(
        "from senselab.audio.data_structures import AudioHints\n\n\ndef build_hint(path):\n"
        '    return (AudioHints(task="free-speech"),)\n',
        encoding="utf-8",
    )
    return directory


def _live_verdicts(run_root: Path) -> list[dict[str, Any]]:
    """Every live verdict entity in the store, by node."""
    store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
    return [
        dict(entity.attributes) for entity in store.entities("verdict") if not store.is_invalidated(entity.id)
    ]


class TestItDecidesAgainWithoutReadingAnything:
    """The point of the driver: the fold moves, the evidence does not."""

    def test_the_standing_verdict_is_retired_and_replaced(self, tmp_path: Path) -> None:
        """One live VERDICT, and it is not the seeded one."""
        run_root = _finished_run(tmp_path / "corpus")
        before = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        held = next(e.id for e in before.entities("verdict") if e.attributes["node"] == "VERDICT")

        summary = cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
            hints=_hints(tmp_path),
        )
        assert summary["counts"] == {"ok": 1}
        after = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        assert after.is_invalidated(held)
        assert [v["node"] for v in _live_verdicts(run_root)].count("VERDICT") == 1

    def test_no_other_node_s_verdict_is_touched(self, tmp_path: Path) -> None:
        """What separates a re-fold from a replay."""
        run_root = _finished_run(tmp_path / "corpus")
        cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
            hints=_hints(tmp_path),
        )
        nodes = [v["node"] for v in _live_verdicts(run_root)]
        assert "ADMIT" in nodes and "SPEECH" in nodes

    def test_it_opens_no_audio(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A fold reads the store. Decoding a stream here would make the pass cost what a replay does."""
        import soundfile

        def _boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise AssertionError("the re-fold opened an audio stream")

        monkeypatch.setattr(soundfile, "read", _boom)
        run_root = _finished_run(tmp_path / "corpus")
        summary = cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
            hints=_hints(tmp_path),
        )
        assert summary["counts"] == {"ok": 1}

    def test_a_reviewer_annotation_survives_the_fold(self, tmp_path: Path) -> None:
        """The expensive half is the reading. A fold change must never cost a re-review."""
        run_root = _finished_run(tmp_path / "corpus")
        store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        software = software_agent(store)
        activity = store.activity(node="REVIEW", step="llm_check", parameters={})
        store.was_associated_with(activity, software)
        annotation = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={"name": REDACTION_LLM_ANNOTATION, "signal": "consensus_transcript", "status": "clean"},
        )
        store.was_generated_by(annotation, activity)
        store.write_jsonl(run_root / "run" / "store.jsonl")

        cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
            hints=_hints(tmp_path),
        )
        after = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        assert not after.is_invalidated(annotation)


class TestItIsIdempotentRatherThanResumable:
    """A fold that concludes what already stands mints the same entity and retires nothing."""

    def test_a_second_pass_changes_nothing(self, tmp_path: Path) -> None:
        """Resubmitting a preempted task is safe: it folds again and reaches the same answer."""
        run_root = _finished_run(tmp_path / "corpus")
        args = {
            "slice_index": 0,
            "slice_count": 1,
            "config": _config(tmp_path),
            "log_dir": tmp_path,
            "hints": _hints(tmp_path),
        }
        manifest = _manifest(tmp_path, run_root)
        cli.run_slice(manifest, **args)
        settled = _live_verdicts(run_root)

        summary = cli.run_slice(manifest, **args)
        assert summary["counts"] == {"unchanged": 1}
        assert _live_verdicts(run_root) == settled

    def test_a_second_pass_leaves_a_live_verdict(self, tmp_path: Path) -> None:
        """The defect this guards: retiring before folding left the recording with no decision.

        The store is content-addressed, so an unchanged fold mints the same entity. Retiring first
        invalidated it and the fold handed it straight back -- measured as 0 live and 2 retired
        after two passes. A preempted slice is re-run by definition, so a corpus re-fold on a
        preemptable queue would have quietly deleted verdicts.
        """
        run_root = _finished_run(tmp_path / "corpus")
        args = {
            "slice_index": 0,
            "slice_count": 1,
            "config": _config(tmp_path),
            "log_dir": tmp_path,
            "hints": _hints(tmp_path),
        }
        manifest = _manifest(tmp_path, run_root)
        for _ in range(3):
            cli.run_slice(manifest, **args)
            live = [v for v in _live_verdicts(run_root) if v["node"] == "VERDICT"]
            assert len(live) == 1, "every pass must leave exactly one live verdict"


class TestItRefusesToRunBlind:
    """Re-folding with no declaration flags every recording it touches."""

    def test_the_hints_argument_is_required(self, tmp_path: Path) -> None:
        """Not optional, because the corpus-wide failure it prevents reads as a finding."""
        run_root = _finished_run(tmp_path / "corpus")
        with pytest.raises(SystemExit):
            cli.main([str(_manifest(tmp_path, run_root)), "--slice-index", "0", "--slice-count", "1"])


class TestTheMarkerSaysWhatFoldedIt:
    """A re-fold is readable back off the store, the way a replay is."""

    def test_the_marker_names_the_configuration_and_the_commit(self, tmp_path: Path) -> None:
        """So a corpus can be asked which fold it is carrying."""
        run_root = _finished_run(tmp_path / "corpus")
        config = _config(tmp_path)
        cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=config,
            log_dir=tmp_path,
            hints=_hints(tmp_path),
            commit="c0ffee",
        )
        store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        markers = [
            a for a in store.activities() if a.node == REFOLD_NODE and a.step == REFOLD_MARKER_STEP
        ]
        assert len(markers) == 1
        assert markers[0].parameters["commit"] == "c0ffee"
        assert markers[0].parameters["config_hash"] == config.config_hash
