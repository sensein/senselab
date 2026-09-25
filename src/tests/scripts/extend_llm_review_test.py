"""The review-only driver: REVIEW over a finished corpus, without replaying the graph.

The replay driver reaches REVIEW by re-running TAXONOMY through REPORT on every recording, which is
the graph's cost paid to get at one node's. This one reads the store, reads the transcript back and
writes the annotation, and its tests pin the three properties that makes it usable over 42,030
recordings on a preemptable queue: it is resumable, it rewrites no decision, and it cannot write
through a ``streams/`` symlink into the corpus it was pointed at.

``specs/20260924-reviewer-over-every-transcript/design.md`` is the design.
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
from senselab.audio.workflows.triage.nodes import review as review_module
from senselab.audio.workflows.triage.nodes.common import software_agent, write_verdict
from senselab.audio.workflows.triage.run import LOG_FILE
from senselab.audio.workflows.triage.vocabulary import REDACTION_LLM_ANNOTATION, Outcome
from senselab.text.tasks.pii_detection.redaction_review import ReviewProposal, ReviewResult
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import word_attributes

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_llm_review.py"
_spec = importlib.util.spec_from_file_location("extend_llm_review_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_llm_review_under_test"] = cli
_spec.loader.exec_module(cli)

SR = 16000
LLM_ON = "redaction:\n  llm_check:\n    enabled: true\n"


def _config(tmp_path: Path, text: str = LLM_ON) -> Any:  # noqa: ANN401 — TriageConfig, kept off for brevity
    """The packaged config with a partial YAML deep-merged over it."""
    path = tmp_path / "override.yaml"
    path.write_text(text, encoding="utf-8")
    return load_triage_config(path)


def _finished_run(root: Path, *, words: Sequence[str] = ("hello", "alicia")) -> Path:
    """One finished run in miniature: a store with consensus words, and the streams beside it."""
    run_root = root / "sub-a_task-free-speech_20260924-000000"
    streams = run_root / "run" / "streams"
    streams.mkdir(parents=True, exist_ok=True)
    wave = (0.05 * np.random.default_rng(0).standard_normal(SR * 5)).astype(np.float32)
    for name in ("enhanced", "plain"):
        sf.write(str(streams / f"{name}.flac"), wave, SR)

    store = ProvStore(run_id=run_root.name)
    software = store.agent(agent_type="software", version="senselab test-seed")
    activity = store.activity(node="PREPROCESS", step="consensus", parameters={})
    store.was_associated_with(activity, software)
    for name in ("recording", "enhanced", "plain"):
        stream = store.entity(
            prov_type="stream",
            extent=(0.0, 5.0),
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
    store.write_jsonl(run_root / "run" / "store.jsonl")
    return run_root


def _seed_verdicts(run_root: Path) -> dict[str, str]:
    """One verdict per deciding node, and the run log a hint is built against.

    Returns:
        The entity id of each seeded verdict, keyed by node.
    """
    store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
    software = software_agent(store)
    seeded: dict[str, str] = {}
    for node, outcome in (("ADMIT", Outcome.PASS), ("SPEECH", Outcome.PASS), ("VERDICT", Outcome.FLAG)):
        activity = store.activity(node=node, step="seed", parameters={})
        store.was_associated_with(activity, software)
        entity_id, _ = write_verdict(
            store, activity, software, node=node, outcome=outcome, kind=None, why="seeded", detail={}
        )
        seeded[node] = entity_id
    store.write_jsonl(run_root / "run" / "store.jsonl")
    (run_root / "run" / LOG_FILE).write_text(
        json.dumps({"source": str(run_root / "run" / "streams" / "recording.flac")}), encoding="utf-8"
    )
    return seeded


def _hints(tmp_path: Path, run_root: Path) -> Path:
    """A ``hints.py`` naming the one task this run declares, as the replay driver's does."""
    directory = tmp_path / "scope"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hints.py").write_text(
        "from senselab.audio.data_structures import AudioHints\n"
        "\n"
        "\n"
        "def build_hint(path):\n"
        '    return (AudioHints(task="free-speech"),)\n',
        encoding="utf-8",
    )
    return directory


def _manifest(tmp_path: Path, run_root: Path) -> Path:
    """The manifest the extend family takes, over one run."""
    path = tmp_path / "manifest.jsonl"
    row = {"stem": run_root.name, "enhanced": str(run_root / "run" / "streams" / "enhanced.flac")}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def _stub(monkeypatch: pytest.MonkeyPatch, *proposal: ReviewProposal) -> list[int]:
    """Replace the reviewer with one that always answers, counting how often it was asked."""
    calls: list[int] = []

    def _fake(original: str, **kw: Any) -> ReviewResult:  # noqa: ANN401
        calls.append(1)
        return ReviewResult(
            available=True,
            reasoning="read it",
            redaction="not_applicable",
            original="carries_pii" if proposal else "clean",
            speakers="one",
            proposal=list(proposal),
            model_id="stub/model",
            revision="a" * 40,
        )

    monkeypatch.setattr(review_module, "review_transcript", _fake)
    return calls


def _annotations(run_root: Path) -> list[dict[str, Any]]:
    """Every live annotation the written store holds."""
    store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
    return [
        dict(entity.attributes)
        for entity in store.entities("measurement")
        if entity.attributes.get("name") == REDACTION_LLM_ANNOTATION and not store.is_invalidated(entity.id)
    ]


class TestItReadsTheStoreAndWritesTheReading:
    """One node over a finished run; no TAXONOMY, no REPORT, no decision re-run."""

    def test_a_finished_run_gains_one_annotation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole product of the pass, and the store is what carries it."""
        run_root = _finished_run(tmp_path / "corpus")
        calls = _stub(monkeypatch)
        summary = cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
        )
        assert summary["counts"] == {"ok": 1}
        assert len(calls) == 1
        annotations = _annotations(run_root)
        assert len(annotations) == 1
        assert annotations[0]["status"] == "clean"

    def test_no_refold_leaves_every_decision_exactly_where_it_was(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The collecting pass. Readings land, and nothing that was decided is decided again."""
        run_root = _finished_run(tmp_path / "corpus")
        _seed_verdicts(run_root)
        store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        before = [dict(entity.attributes) for entity in store.entities("verdict")]

        _stub(monkeypatch)
        cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
        )
        after_store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        after = [dict(entity.attributes) for entity in after_store.entities("verdict")]
        assert after == before


class TestTheDecisionIsTakenAgainOverTheReading:
    """REVIEW writes an input VERDICT reads, so the recorded decision must have seen it."""

    def test_a_landed_reading_retires_the_standing_verdict_and_writes_another(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of the re-fold: the decision on disk is one that read the annotation."""
        run_root = _finished_run(tmp_path / "corpus")
        held = _seed_verdicts(run_root)
        _stub(monkeypatch)
        summary = cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
            hints=_hints(tmp_path, run_root),
        )
        # The counter is the guard against a re-fold that silently did not run: a skip or a raise
        # would leave every assertion below satisfied by the seeded verdict alone.
        assert list(summary["refolds"]) == ["flag/not_assessed"]
        store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        assert store.is_invalidated(held["VERDICT"])
        live = [entity for entity in store.entities("verdict") if not store.is_invalidated(entity.id)]
        assert [entity.attributes["node"] for entity in live].count("VERDICT") == 1

    def test_only_verdicts_own_conclusion_is_retired(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What separates this from the replay driver: every other node's verdict stands."""
        run_root = _finished_run(tmp_path / "corpus")
        held = _seed_verdicts(run_root)
        _stub(monkeypatch)
        cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
            hints=_hints(tmp_path, run_root),
        )
        store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        assert not store.is_invalidated(held["SPEECH"])
        assert not store.is_invalidated(held["ADMIT"])

    def test_the_marker_names_the_configuration_and_the_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A re-fold is readable back off the store, the way a replay is."""
        run_root = _finished_run(tmp_path / "corpus")
        _seed_verdicts(run_root)
        _stub(monkeypatch)
        config = _config(tmp_path)
        cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=config,
            log_dir=tmp_path,
            hints=_hints(tmp_path, run_root),
            commit="c0ffee",
        )
        store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        markers = [
            activity
            for activity in store.activities()
            if activity.node == REFOLD_NODE and activity.step == REFOLD_MARKER_STEP
        ]
        assert len(markers) == 1
        assert markers[0].parameters["commit"] == "c0ffee"
        assert markers[0].parameters["config_hash"] == config.config_hash

    def test_a_pass_that_read_nothing_new_refolds_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second pass is ``present``, and a decision is not retired for having been looked at."""
        run_root = _finished_run(tmp_path / "corpus")
        _seed_verdicts(run_root)
        _stub(monkeypatch)
        args = {"slice_index": 0, "slice_count": 1, "config": _config(tmp_path), "log_dir": tmp_path}
        manifest = _manifest(tmp_path, run_root)
        hints = _hints(tmp_path, run_root)
        cli.run_slice(manifest, hints=hints, **args)
        first = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        settled = [entity.id for entity in first.entities("verdict") if not first.is_invalidated(entity.id)]

        summary = cli.run_slice(manifest, hints=hints, **args)
        assert summary["counts"] == {"present": 1}
        second = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
        assert [entity.id for entity in second.entities("verdict") if not second.is_invalidated(entity.id)] == settled

    def test_the_cli_refuses_to_run_blind_rather_than_flagging_the_corpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-folding with no hint reads every declaration as unresolvable, which flags everything."""
        run_root = _finished_run(tmp_path / "corpus")
        _stub(monkeypatch)
        argv = [str(_manifest(tmp_path, run_root)), "--slice-index", "0", "--slice-count", "1"]
        assert cli.main(argv) == 2
        assert cli.main([*argv, "--no-refold"]) == 0


class TestItIsResumableTheWayTheFamilyIs:
    """A preemptable queue kills tasks; resubmitting the same array must not pay twice."""

    def test_a_second_pass_reads_nothing_again(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The store is the marker: a standing annotation is ``present``, and no model is contacted."""
        run_root = _finished_run(tmp_path / "corpus")
        manifest = _manifest(tmp_path, run_root)
        calls = _stub(monkeypatch)
        cli.run_slice(manifest, slice_index=0, slice_count=1, config=_config(tmp_path), log_dir=tmp_path)
        second = cli.run_slice(manifest, slice_index=0, slice_count=1, config=_config(tmp_path), log_dir=tmp_path)
        assert second["counts"] == {"present": 1}
        assert len(calls) == 1, "the second pass contacted the model again"
        assert len(_annotations(run_root)) == 1

    def test_force_re_reads_and_never_stacks_a_second_live_reading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two live annotations would make the store ambiguous about what the reviewer concluded.

        The store is content-addressed, so a re-read reaching the same conclusion mints the same
        entity and the fingerprint does not move -- which is why the retirement happens after the
        write and only where the id differs.
        """
        run_root = _finished_run(tmp_path / "corpus")
        manifest = _manifest(tmp_path, run_root)
        _stub(monkeypatch)
        cli.run_slice(manifest, slice_index=0, slice_count=1, config=_config(tmp_path), log_dir=tmp_path)
        again = cli.run_slice(
            manifest, slice_index=0, slice_count=1, config=_config(tmp_path), log_dir=tmp_path, force=True
        )
        assert again["counts"] == {"present": 1}, "an identical re-reading is the same entity, not a second one"
        assert len(_annotations(run_root)) == 1


class TestItCannotWriteThroughTheCorpusSStreams:
    """17,924 redacted.flac files were overwritten by a mirror that seeded one as a symlink."""

    def test_the_stream_it_writes_is_never_seeded_as_a_link(self, tmp_path: Path) -> None:
        """The guard, at the mirror rather than at the write."""
        run_root = _finished_run(tmp_path / "corpus")
        streams = run_root / "run" / "streams"
        sf.write(str(streams / "redacted.flac"), np.zeros(SR, dtype=np.float32), SR)
        mirror = cli.mirror_run_root(run_root, tmp_path / "out")
        assert (mirror / "run" / "streams" / "enhanced.flac").is_symlink()
        assert not (mirror / "run" / "streams" / "redacted.flac").exists()

    def test_applying_a_proposal_leaves_the_finished_tree_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The incident itself, as a test: the corpus copy must not move."""
        run_root = _finished_run(tmp_path / "corpus")
        streams = run_root / "run" / "streams"
        sf.write(str(streams / "redacted.flac"), np.zeros(SR, dtype=np.float32), SR)
        untouched = (streams / "redacted.flac").read_bytes()
        _stub(monkeypatch, ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name"))
        summary = cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
            out_root=tmp_path / "out",
            apply=True,
        )
        assert summary["counts"] == {"ok": 1}
        assert (streams / "redacted.flac").read_bytes() == untouched
        assert (tmp_path / "out" / run_root.name / "run" / "streams" / "redacted.flac").is_file()
        assert not _annotations(run_root), "and the corpus store is untouched too"

    def test_mirroring_twice_is_idempotent(self, tmp_path: Path) -> None:
        """A preempted task re-enters its own mirror; seeding must not raise on the second pass."""
        run_root = _finished_run(tmp_path / "corpus")
        first = cli.mirror_run_root(run_root, tmp_path / "out")
        second = cli.mirror_run_root(run_root, tmp_path / "out")
        assert first == second


class TestItOpensNoAudioUnlessAskedTo:
    """The reading is a store read; audio is the CPU-and-IO half and is opt-in."""

    def test_a_review_without_apply_decodes_no_stream(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """What makes the pass cheap enough to put on a scarce GPU."""
        run_root = _finished_run(tmp_path / "corpus")
        for stream in (run_root / "run" / "streams").glob("*.flac"):
            stream.unlink()
        _stub(monkeypatch)
        summary = cli.run_slice(
            _manifest(tmp_path, run_root),
            slice_index=0,
            slice_count=1,
            config=_config(tmp_path),
            log_dir=tmp_path,
        )
        assert summary["counts"] == {"ok": 1}, "it read a run whose audio is not even on disk"
