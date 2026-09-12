"""The reprocessed-outputs extend driver: what it derives, what it retires, and what it refuses.

Nothing here is faked. Neither derivation runs a model — one is Praat over streams the run tree
already holds, the other is arithmetic over scores already in the store — so the runs are real runs
in miniature: FLAC streams on disk, per-span classifier measurements carrying ``raw_scores``, and a
``consensus_taxonomy`` in the string-matched form the finished corpus is in.

``specs/20260912-extend-reprocessed-outputs/design.md`` is the design.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.consensus import (
    SourceHypothesis,
    align_sources,
    render_transcript,
    vocabulary_key,
    word_attributes,
)
from senselab.audio.workflows.triage.extend import CONSENSUS_TRANSCRIPT, ONOMATOPOEIC_TOKENS_KEY, REBRACKET
from senselab.audio.workflows.triage.nodes.common import (
    consensus_words,
    find_measurement,
    lexical_words,
    path_attributes,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.nodes.preprocess import phonation_tracks
from senselab.audio.workflows.triage.nodes.taxonomy import NODE as TAXONOMY_NODE
from senselab.audio.workflows.triage.nodes.taxonomy import PER_SPAN_CLASSIFIERS, _write_consensus_taxonomy
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_reprocessed_outputs.py"
_spec = importlib.util.spec_from_file_location("extend_reprocessed_outputs_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_reprocessed_outputs_under_test"] = cli
_spec.loader.exec_module(cli)

SR = 16000
"""The fixture's sampling rate, the one every triage stream is resampled to."""

DURATION_S = 1.5
"""Long enough for Praat to place several F0 and formant frames, short enough to stay quick."""

PHONATION_TRACKS = "phonation_tracks"
CONSENSUS_TAXONOMY = "consensus_taxonomy"

_LEXICON = {vocabulary_key(str(token)) for token in load_triage_config().require(ONOMATOPOEIC_TOKENS_KEY)}
"""The shipped onomatopoeic vocabulary, which is what the extend pass re-flags against."""

SPELLINGS = {"hear": "Throat Clear", "yamnet": "Throat clearing"}
"""One AudioSet node, two classifier spellings — the pair the identity merge exists for."""

PEAKS = {"hear": 0.8, "yamnet": 0.6}


def _voiced() -> np.ndarray:
    """A buzzy 120 Hz source through two fixed resonances, so F0 and formants both resolve.

    Returns:
        The waveform, mono float32, peak-normalised below full scale.
    """
    grid = np.arange(int(SR * DURATION_S)) / SR
    source = sum(np.sin(2 * np.pi * 120.0 * harmonic * grid) / harmonic for harmonic in range(1, 30))
    shaped = np.asarray(source) * (1.0 + 0.3 * np.sin(2 * np.pi * 3.0 * grid))
    return (0.7 * shaped / np.abs(shaped).max()).astype(np.float32)


def _stream(store: ProvStore, run_dir: Path, name: str, activity: str, agent: str) -> str:
    """Write one conditioned stream to the run tree and register its entity.

    Args:
        store: The store to write into.
        run_dir: The run directory ``streams/`` sits under.
        name: The stream's name, which is also its file stem.
        activity: The conditioning activity.
        agent: The agent answerable for it.

    Returns:
        The stream entity's id.
    """
    relative = f"streams/{name}.flac"
    sf.write(str(run_dir / relative), _voiced(), SR)
    entity_id = store.entity(
        prov_type="stream",
        extent=(0.0, DURATION_S),
        attributes={"name": name, **path_attributes(relative, run_dir), "sampling_rate": SR, "channels": 1},
    )
    store.was_generated_by(entity_id, activity)
    store.was_attributed_to(entity_id, agent)
    return entity_id


def _seed_span_scores(store: ProvStore, agent: str) -> None:
    """One span, and each per-span classifier's own spelling of one AudioSet node over it.

    Args:
        store: The store to write into.
        agent: The agent answerable for the measurements.
    """
    activity = store.activity(node="PREPROCESS", step="spans", parameters={})
    store.was_associated_with(activity, agent)
    span_id = store.entity(prov_type="span", extent=(0.2, 0.7), attributes={"signal": "preemphasised"})
    store.was_generated_by(span_id, activity)
    for classifier, step in PER_SPAN_CLASSIFIERS.items():
        span_activity = store.activity(node="PREPROCESS", step=step, parameters={})
        store.was_associated_with(span_activity, agent)
        entity_id = store.entity(
            prov_type="measurement",
            extent=(0.2, 0.7),
            attributes={
                "name": step,
                "classifier": classifier,
                "signal": "plain",
                "span_id": span_id,
                "raw_scores": {SPELLINGS[classifier]: PEAKS[classifier]},
                "labelled": True,
            },
        )
        store.was_generated_by(entity_id, span_activity)
        store.was_attributed_to(entity_id, agent)


def _seed_string_matched_taxonomy(store: ProvStore, agent: str) -> str:
    """The consolidation the retired writer produced: one row per spelling, neither merged.

    Args:
        store: The store to write into.
        agent: The agent answerable for the measurement.

    Returns:
        The measurement entity's id.
    """
    activity = store.activity(
        node=TAXONOMY_NODE,
        step=CONSENSUS_TAXONOMY,
        parameters={"consolidation_floor": None, "classifiers": ["hear", "yamnet"]},
    )
    store.was_associated_with(activity, agent)
    rows = [
        {
            "label": SPELLINGS[classifier],
            "classifiers": [classifier],
            "peak": PEAKS[classifier],
            "peak_by_classifier": {classifier: PEAKS[classifier]},
            "n_classifiers": 1,
        }
        for classifier in ("hear", "yamnet")
    ]
    return write_measurement(
        store,
        activity,
        agent,
        name=CONSENSUS_TAXONOMY,
        signal="plain",
        attributes={"labels": rows, "n_labels": len(rows), "classifiers": ["hear", "yamnet"]},
        extent=None,
    )


HYPOTHESES: dict[str, tuple[tuple[str, float, float], ...]] = {
    "asr_a": (
        ("I", 0.10, 0.20),
        ("cough", 0.30, 0.45),
        ("khh", 0.60, 0.70),
        ("[BREATH]", 0.85, 0.95),
        ("hello", 1.10, 1.30),
    ),
    "asr_b": (
        ("I", 0.11, 0.21),
        ("Cough,", 0.31, 0.46),
        ("hack", 0.61, 0.72),
        ("[BREATH]", 0.86, 0.96),
        ("hello", 1.12, 1.33),
    ),
}
"""Two recognizers over one recording: an agreement column and a variant column of the lexicon, one
token already bracketed, and two ordinary words that no vocabulary touches."""

LEXICAL_HYPOTHESES: dict[str, tuple[tuple[str, float, float], ...]] = {
    "asr_a": (("I", 0.10, 0.20), ("said", 0.30, 0.45), ("hello", 0.60, 0.80)),
    "asr_b": (("I", 0.11, 0.21), ("said", 0.31, 0.46), ("hello", 0.61, 0.82)),
}
"""A recording no entry of the vocabulary appears in."""


def _seed_consensus(
    store: ProvStore,
    agent: str,
    *,
    lexicon: set[str],
    hypotheses: dict[str, tuple[tuple[str, float, float], ...]],
) -> None:
    """Write the ASR hypotheses and the consensus stream a run under this vocabulary would hold.

    The same writes PREPROCESS's own consensus block makes, in the same order: one measurement per
    recognizer, one ``word`` entity per aligned column, and the transcript listing them.

    Args:
        store: The store to write into.
        agent: The agent answerable for the writes.
        lexicon: The ``words.onomatopoeic_tokens`` vocabulary the run was made under.
        hypotheses: ``{source: ((text, start, end), ...)}``, the recognizers' own words.
    """
    asr = store.activity(node="PREPROCESS", step="asr", parameters={})
    store.was_associated_with(asr, agent)
    measurement_ids: dict[str, str] = {}
    for source, words in hypotheses.items():
        measurement_ids[source] = write_measurement(
            store,
            asr,
            agent,
            name=source,
            signal="plain",
            attributes={
                "role": "asr_hypothesis",
                "source": source,
                "model_id": f"test/{source}",
                "commit_sha": None,
                "words": [{"text": text, "start": start, "end": end} for text, start, end in words],
                "n_words": len(words),
                "timestamp_source": "native",
                "timestamp_model": None,
            },
        )
    consensus = align_sources(
        [
            SourceHypothesis(name=source, words=words, timestamp_source="native", timestamp_model=None)
            for source, words in hypotheses.items()
        ],
        onomatopoeic=lexicon,
    )
    activity = store.activity(node="PREPROCESS", step="consensus", parameters={"sources": sorted(hypotheses)})
    store.was_associated_with(activity, agent)
    word_ids: list[str] = []
    for word in consensus.words:
        word_id = store.entity(prov_type="word", extent=word.extent, attributes=word_attributes(word))
        store.was_generated_by(word_id, activity)
        store.was_attributed_to(word_id, agent)
        word_ids.append(word_id)
    write_measurement(
        store,
        activity,
        agent,
        name=CONSENSUS_TRANSCRIPT,
        signal="plain",
        attributes={
            "role": "consensus",
            **consensus.provenance,
            "sources": [
                {**row, "measurement_id": measurement_ids[str(row["name"])], "agent_id": agent}
                for row in consensus.provenance["sources"]
            ],
            "word_ids": word_ids,
            "text": render_transcript(consensus.words, strong=("", "")),
        },
        derived_from=tuple(measurement_ids[source] for source in sorted(hypotheses)),
    )


def _seed_asr_span(store: ProvStore, agent: str) -> str:
    """One span the ASR proposer contributed, derived from the transcript that proposed it.

    Args:
        store: The store to write into, already carrying a consensus transcript.
        agent: The agent answerable for the writes.

    Returns:
        The span entity's id.
    """
    consensus = find_measurement(store, CONSENSUS_TRANSCRIPT)
    assert consensus is not None, "an ASR-proposed span needs the transcript that proposed it"  # noqa: S101
    activity = store.activity(node="PREPROCESS", step="spans", parameters={"measure": "asr"})
    store.was_associated_with(activity, agent)
    span_id = store.entity(
        prov_type="span",
        extent=(0.10, 0.70),
        attributes={"signal": "consensus", "measure": "asr", "merged_proposals": 3, "contains_clip": False},
    )
    store.was_generated_by(span_id, activity)
    store.was_attributed_to(span_id, agent)
    store.was_derived_from(span_id, consensus.id)
    return span_id


def _seed_run(
    root: Path,
    *,
    taxonomy: str = "old",
    tracks: bool = False,
    streams: bool = True,
    words: str = "null",
    asr_span: bool = False,
) -> None:
    """Write one finished run in the shape the corpus is in, or one of its variations.

    Args:
        root: The run root, created if absent.
        taxonomy: ``"old"`` for the string-matched consolidation, ``"current"`` for the identity
            merge a fresh run writes, ``"none"`` for a store carrying no consolidation at all.
        tracks: Whether the store already carries a ``phonation_tracks`` measurement.
        streams: Whether the conditioned streams exist at all.
        words: ``"null"`` for a consensus written under the null vocabulary the corpus ran with,
            ``"current"`` for one written under the shipped lexicon, ``"lexical"`` for a recording
            no entry of the lexicon appears in, ``"none"`` for a store carrying no transcript.
        asr_span: Whether the store carries a span the ASR proposer contributed.
    """
    run_dir = root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    sf.write(str(run_dir / "streams" / "enhanced.flac"), _voiced(), SR)

    store = ProvStore(run_id=root.name)
    agent = software_agent(store)
    condition = store.activity(node="PREPROCESS", step="condition", parameters={"target_hz": SR})
    store.was_associated_with(condition, agent)
    if streams:
        _stream(store, run_dir, "plain", condition, agent)
        _stream(store, run_dir, "preemphasised", condition, agent)
    _seed_span_scores(store, agent)
    if words != "none":
        _seed_consensus(
            store,
            agent,
            lexicon=_LEXICON if words == "current" else set(),
            hypotheses=LEXICAL_HYPOTHESES if words == "lexical" else HYPOTHESES,
        )
    if asr_span:
        _seed_asr_span(store, agent)
    if taxonomy == "old":
        _seed_string_matched_taxonomy(store, agent)
    elif taxonomy == "current":
        _write_consensus_taxonomy(store, load_triage_config(), agent)
    if tracks:
        phonation_tracks(store, load_triage_config(), run_dir=run_dir)
    store.write_jsonl(run_dir / "store.jsonl")


def _manifest(path: Path, roots: list[Path]) -> Path:
    """Write a manifest naming one enhanced stream per run root.

    Args:
        path: Where the JSONL goes.
        roots: The run roots, in manifest order.

    Returns:
        The manifest's path.
    """
    lines = [
        json.dumps({"stem": root.name, "enhanced": str(root / "run" / "streams" / "enhanced.flac")}) for root in roots
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _store_of(root: Path) -> ProvStore:
    """Read one run's store back under its own run id.

    Args:
        root: The run root.

    Returns:
        The store.
    """
    return ProvStore.read_jsonl(root / "run" / "store.jsonl", run_id=root.name)


def _retired(store: ProvStore) -> set[str]:
    """Every entity the store has marked ``wasInvalidatedBy``.

    Args:
        store: The store.

    Returns:
        The retired entities' ids.
    """
    return {source for relation, source, _ in store.relations() if relation == "wasInvalidatedBy"}


def _log(tmp_path: Path) -> list[dict[str, Any]]:
    """The slice log the driver wrote.

    Args:
        tmp_path: The directory the manifest sits in.

    Returns:
        One record per manifest row, in order.
    """
    path = tmp_path / "slices" / "reprocessed-outputs-slice-0-of-1.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _run(manifest: Path) -> int:
    """Drive the whole manifest as one slice.

    Args:
        manifest: The manifest JSONL.

    Returns:
        The driver's exit code.
    """
    return int(cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]))


@pytest.fixture
def corpus(tmp_path: Path) -> Callable[..., tuple[Path, list[Path]]]:
    """A factory for a manifest over N finished runs.

    Returns:
        A callable taking the run count and :func:`_seed_run`'s keyword arguments.
    """

    def _build(count: int, **kwargs: Any) -> tuple[Path, list[Path]]:  # noqa: ANN401 — _seed_run's own keywords
        roots = []
        for index in range(count):
            root = tmp_path / "corpus" / f"sub-{index:02d}_ses-1_20260912-000000"
            root.mkdir(parents=True, exist_ok=True)
            _seed_run(root, **kwargs)
            roots.append(root)
        return _manifest(tmp_path / "manifest.jsonl", roots), roots

    return _build


class TestThePhonationTracks:
    """An append: the corpus has no tracks at all, because a null search range suppressed them."""

    def test_the_measurement_and_its_sidecar_are_written(self, corpus: Callable[..., tuple[Path, list[Path]]]) -> None:
        """The F0 range is this recording's own, narrowed from the configured wide search."""
        manifest, roots = corpus(1)
        assert find_measurement(_store_of(roots[0]), PHONATION_TRACKS) is None

        assert _run(manifest) == 0

        measurement = find_measurement(_store_of(roots[0]), PHONATION_TRACKS)
        assert measurement is not None
        floor, ceiling = load_triage_config().require("voice.f0_search_range_hz")
        assert floor <= measurement.attributes["f0_min_hz"] < measurement.attributes["f0_max_hz"] <= ceiling
        assert measurement.attributes["f0_signal"] == "preemphasised"
        assert measurement.attributes["formant_signal"] == "plain"

        npz = np.load(roots[0] / "run" / "derivatives" / f"{PHONATION_TRACKS}.npz")
        assert npz["times_s"].size > 0
        assert npz["f1_hz"].size == npz["formant_times_s"].size

    def test_the_tracks_agree_with_what_a_fresh_pass_would_have_written(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """Both paths read the streams out of the store, so both mint the same entity id."""
        manifest, roots = corpus(1)
        _run(manifest)
        extended = find_measurement(_store_of(roots[0]), PHONATION_TRACKS)

        fresh_root = tmp_path / "fresh"
        fresh_root.mkdir()
        _seed_run(fresh_root, tracks=True)
        fresh = find_measurement(_store_of(fresh_root), PHONATION_TRACKS)

        assert extended is not None and fresh is not None
        assert extended.attributes == fresh.attributes

    def test_a_run_with_no_conditioned_stream_is_an_outcome_not_a_crash(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """There is no signal to track; the taxonomy beside it is still made current."""
        manifest, roots = corpus(1, streams=False)

        assert _run(manifest) == 1

        record = _log(tmp_path)[0]
        assert "plain" in str(record[PHONATION_TRACKS])
        assert record[CONSENSUS_TAXONOMY] == "rewritten"
        assert find_measurement(_store_of(roots[0]), CONSENSUS_TAXONOMY) is not None


class TestTheConsensusTaxonomyRewrite:
    """A rewrite: the store already holds a consolidation, and two live ones would contradict."""

    def test_the_new_reading_is_live_and_merges_the_two_spellings(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """One AudioSet node, one row, both classifiers on it, each spelling named."""
        manifest, roots = corpus(1)

        assert _run(manifest) == 0

        measurement = find_measurement(_store_of(roots[0]), CONSENSUS_TAXONOMY)
        assert measurement is not None
        [row] = measurement.attributes["labels"]
        assert row["label"] == SPELLINGS["yamnet"]
        assert row["n_classifiers"] == 2
        assert row["labels_by_classifier"] == {"hear": ["Throat Clear"], "yamnet": ["Throat clearing"]}

    def test_the_superseded_reading_is_retired_rather_than_removed(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The store is append-only: the old rows stay readable and stop being current."""
        manifest, roots = corpus(1)
        before = find_measurement(_store_of(roots[0]), CONSENSUS_TAXONOMY)
        assert before is not None

        _run(manifest)

        store = _store_of(roots[0])
        assert store.is_invalidated(before.id)
        assert store.get_entity(before.id).attributes["n_labels"] == 2
        live = [
            entity
            for entity in store.entities("measurement")
            if entity.attributes.get("name") == CONSENSUS_TAXONOMY and not store.is_invalidated(entity.id)
        ]
        assert len(live) == 1

    def test_the_retirement_names_why_the_old_reading_is_no_longer_current(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """A reader arriving at the retired row follows one edge to the reason."""
        manifest, roots = corpus(1)
        before = find_measurement(_store_of(roots[0]), CONSENSUS_TAXONOMY)
        assert before is not None

        _run(manifest)

        store = _store_of(roots[0])
        [activity_id] = [target for relation, source, target in store.relations() if source == before.id][-1:]
        activity = store.get_activity(activity_id)
        assert activity.node == TAXONOMY_NODE
        assert activity.step == f"{CONSENSUS_TAXONOMY}_superseded"
        assert activity.parameters["superseded"] == before.id
        assert "AudioSet node identity" in activity.parameters["reason"]

    def test_a_store_carrying_no_consolidation_is_given_none(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """Nothing is there to make current, and writing one would assert TAXONOMY had run."""
        manifest, roots = corpus(1, taxonomy="none")

        assert _run(manifest) == 0

        assert find_measurement(_store_of(roots[0]), CONSENSUS_TAXONOMY) is None
        assert _log(tmp_path)[0][CONSENSUS_TAXONOMY] == "absent"


class TestAFreshRun:
    """A run that never had the old form must be left alone, not rewritten into an identical one."""

    def test_a_run_already_carrying_all_three_outputs_is_not_rewritten(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The store is byte-identical and nothing is retired."""
        manifest, roots = corpus(1, taxonomy="current", tracks=True, words="current")
        before = (roots[0] / "run" / "store.jsonl").read_bytes()
        fingerprint = _store_of(roots[0]).fingerprint()

        assert _run(manifest) == 0

        assert (roots[0] / "run" / "store.jsonl").read_bytes() == before
        assert _store_of(roots[0]).fingerprint() == fingerprint
        record = _log(tmp_path)[0]
        assert record["status"] == "skipped"
        assert record[REBRACKET] == "current"
        assert record[PHONATION_TRACKS] == "present"
        assert record[CONSENSUS_TAXONOMY] == "current"

    def test_a_fresh_run_is_never_given_a_retirement_edge(self, corpus: Callable[..., tuple[Path, list[Path]]]) -> None:
        """Recomputing the current consolidation mints the id the store already holds."""
        manifest, roots = corpus(1, taxonomy="current", tracks=True, words="current")
        _run(manifest)
        store = _store_of(roots[0])
        assert [relation for relation, _, _ in store.relations() if relation == "wasInvalidatedBy"] == []


class TestASecondPass:
    """A task that dies and reruns must change nothing, which is what makes the array restartable."""

    def test_a_rerun_over_the_same_corpus_changes_nothing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The fingerprint is what the driver decides on, and it does not move on the second pass."""
        manifest, roots = corpus(2)
        assert _run(manifest) == 0
        after_first = {root: _store_of(root).fingerprint() for root in roots}
        bytes_first = {root: (root / "run" / "store.jsonl").read_bytes() for root in roots}

        assert _run(manifest) == 0

        for root in roots:
            assert _store_of(root).fingerprint() == after_first[root]
            assert (root / "run" / "store.jsonl").read_bytes() == bytes_first[root]
        assert [record["status"] for record in _log(tmp_path)] == ["skipped", "skipped"]

    def test_a_third_pass_retires_nothing_further(self, corpus: Callable[..., tuple[Path, list[Path]]]) -> None:
        """One retirement per superseded reading, however many times the driver is run."""
        manifest, roots = corpus(1)
        _run(manifest)
        after_first = _retired(_store_of(roots[0]))
        _run(manifest)
        _run(manifest)
        assert _retired(_store_of(roots[0])) == after_first
        assert len(after_first) == 1 + 1 + 2


class TestWhatTheRunGains:
    """The store, ``prov/`` and the environment record, and nothing outside the run."""

    def test_nothing_is_written_outside_the_recordings_own_run(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The store gains entities and ``prov/`` is exported; no second tree appears."""
        manifest, roots = corpus(1)
        _run(manifest)
        assert sorted(entry.name for entry in roots[0].iterdir()) == ["prov", "run"]
        assert sorted(entry.name for entry in (roots[0] / "run").iterdir()) == [
            "derivatives",
            "store.jsonl",
            "streams",
        ]

    def test_the_bep028_files_are_re_exported_from_the_merged_store(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """``prov/`` must not disagree with the store it was exported from."""
        manifest, roots = corpus(1)
        _run(manifest)

        io_graph = json.loads((roots[0] / "prov" / "prov-triage_io.json").read_text())
        entity_ids = {node["Id"] for key in ("Files", "prov:Entity") for node in io_graph.get(key, [])}
        for name in (PHONATION_TRACKS, CONSENSUS_TAXONOMY):
            measurement = find_measurement(_store_of(roots[0]), name)
            assert measurement is not None
            assert any(measurement.id in identifier for identifier in entity_ids), name

    def test_the_host_environment_is_recorded(self, corpus: Callable[..., tuple[Path, list[Path]]]) -> None:
        """The pass ran somewhere, and the store says where; no venv is involved in this one."""
        manifest, roots = corpus(1)
        _run(manifest)
        assert [env.kind for env in _store_of(roots[0]).environments()] == ["host"]


class TestOneBadRecording:
    """A run the filesystem refuses is an outcome record, not the end of the slice."""

    def test_a_run_with_no_store_is_recorded_and_the_others_still_extend(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """A missing store is this recording's error; its neighbours are unaffected."""
        manifest, roots = corpus(2)
        (roots[0] / "run" / "store.jsonl").unlink()

        assert _run(manifest) == 1

        assert find_measurement(_store_of(roots[1]), PHONATION_TRACKS) is not None
        assert [record["status"] for record in _log(tmp_path)] == ["error", "ok"]

    def test_a_manifest_row_naming_no_run_root_is_refused_rather_than_guessed(self) -> None:
        """A wrong guess would rewrite the wrong recording's store."""
        with pytest.raises(ValueError, match="no run root"):
            cli.run_root_of(Path("/corpus/enhanced.flac"))


class TestRebracketingTheWords:
    """A rewrite: the corpus spells transcribed coughs as lexical words, and re-flagging is derivable."""

    def _words(self, root: Path) -> list[Any]:
        """The run's live consensus words, in stream order.

        Args:
            root: The run root.

        Returns:
            The live ``word`` entities.
        """
        return consensus_words(_store_of(root))

    def test_the_words_are_re_flagged_and_the_lexical_count_falls(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """Two of the five columns are vocabulary tokens; the other three are untouched."""
        manifest, roots = corpus(1)
        assert [word.attributes["text"] for word in self._words(roots[0])] == [
            "I",
            "cough",
            "khh",
            "[BREATH]",
            "hello",
        ]
        assert len(lexical_words(_store_of(roots[0]))) == 4

        assert _run(manifest) == 0

        assert [word.attributes["text"] for word in self._words(roots[0])] == [
            "I",
            "[COUGH]",
            "[KHH]",
            "[BREATH]",
            "hello",
        ]
        assert len(lexical_words(_store_of(roots[0]))) == 2
        assert _log(tmp_path)[0][REBRACKET] == "rewritten"

    def test_a_variant_column_keeps_both_readings_and_brackets_each_on_its_own_key(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """``[KHH]`` and ``[HACK]`` are still two readings; re-bracketing manufactures no agreement."""
        manifest, roots = corpus(1)
        _run(manifest)
        variant = self._words(roots[0])[2]
        assert variant.attributes["outcome"] == "variant"
        assert [reading["text"] for reading in variant.attributes["variants"]] == ["[KHH]", "[HACK]"]
        assert variant.attributes["readings"] == {"asr_a": "khh", "asr_b": "hack"}
        assert variant.attributes["agreement"] == 0.5

    def test_the_superseded_words_are_retired_and_the_retirement_names_why(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The store is append-only: the old reading stays readable and stops being current."""
        manifest, roots = corpus(1)
        before = {word.attributes["text"]: word.id for word in self._words(roots[0])}

        _run(manifest)

        store = _store_of(roots[0])
        assert store.is_invalidated(before["cough"]) and store.is_invalidated(before["khh"])
        assert store.get_entity(before["cough"]).attributes["bracketed"] is False
        [activity_id] = [target for _, source, target in store.relations() if source == before["cough"]][-1:]
        activity = store.get_activity(activity_id)
        assert (activity.node, activity.step) == ("PREPROCESS", "word_superseded")
        assert activity.parameters["superseded"] == before["cough"]
        assert "words.onomatopoeic_tokens" in activity.parameters["reason"]

    def test_a_word_no_vocabulary_entry_touches_keeps_its_id(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """Only the columns whose reading moved are rewritten; the rest are not re-minted."""
        manifest, roots = corpus(1)
        before = {word.attributes["text"]: word.id for word in self._words(roots[0])}

        _run(manifest)

        after = {word.attributes["text"]: word.id for word in self._words(roots[0])}
        assert after["I"] == before["I"]
        assert after["hello"] == before["hello"]
        assert after["[BREATH]"] == before["[BREATH]"]
        assert after["[COUGH]"] != before["cough"]

    def test_the_transcript_is_retired_and_rewritten_over_the_new_words(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """A transcript listing retired words would contradict the words the store now holds."""
        manifest, roots = corpus(1)
        before = find_measurement(_store_of(roots[0]), CONSENSUS_TRANSCRIPT)
        assert before is not None and before.attributes["text"] == "I cough khh [BREATH] hello"

        _run(manifest)

        store = _store_of(roots[0])
        after = find_measurement(store, CONSENSUS_TRANSCRIPT)
        assert after is not None and after.id != before.id
        assert after.attributes["text"] == "I [COUGH] [KHH] [BREATH] hello"
        assert store.is_invalidated(before.id)
        assert not any(store.is_invalidated(word_id) for word_id in after.attributes["word_ids"])
        assert after.attributes["n_words"] == before.attributes["n_words"]
        assert after.attributes["outcomes"] == before.attributes["outcomes"]

    def test_the_re_flagged_run_is_the_run_the_lexicon_would_have_made(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """Both paths mint the same entity ids, so an extended store and a fresh one are one store."""
        manifest, roots = corpus(1)
        _run(manifest)

        fresh_root = tmp_path / "fresh" / roots[0].name
        fresh_root.mkdir(parents=True)
        _seed_run(fresh_root, words="current")

        extended, fresh = _store_of(roots[0]), _store_of(fresh_root)
        assert [word.id for word in consensus_words(extended)] == [word.id for word in consensus_words(fresh)]
        extended_transcript = find_measurement(extended, CONSENSUS_TRANSCRIPT)
        fresh_transcript = find_measurement(fresh, CONSENSUS_TRANSCRIPT)
        assert extended_transcript is not None and fresh_transcript is not None
        assert extended_transcript.id == fresh_transcript.id

    def test_a_recording_no_vocabulary_entry_appears_in_is_untouched(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """Nothing to re-flag is not the same outcome as nothing to re-flag against."""
        manifest, roots = corpus(1, taxonomy="current", tracks=True, words="lexical")
        before = (roots[0] / "run" / "store.jsonl").read_bytes()

        assert _run(manifest) == 0

        assert (roots[0] / "run" / "store.jsonl").read_bytes() == before
        assert _log(tmp_path)[0][REBRACKET] == "current"

    def test_a_store_carrying_no_transcript_is_given_none(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """PREPROCESS wrote no consensus there; synthesising one would assert that it had."""
        manifest, roots = corpus(1, words="none")

        assert _run(manifest) == 0

        assert find_measurement(_store_of(roots[0]), CONSENSUS_TRANSCRIPT) is None
        assert _log(tmp_path)[0][REBRACKET] == "absent"


class TestTheSpansTheAsrProposed:
    """A re-flagged word changes the proposer's input; the spans it already proposed are kept."""

    def test_the_span_is_kept_and_the_store_records_that_it_was_not_recomputed(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """A recomputed ASR span would carry no per-span measurement, and none is derivable here."""
        manifest, roots = corpus(1, asr_span=True)
        before = [span for span in _store_of(roots[0]).entities("span") if span.attributes.get("measure") == "asr"]
        assert len(before) == 1

        _run(manifest)

        store = _store_of(roots[0])
        assert not store.is_invalidated(before[0].id)
        record = find_measurement(store, REBRACKET)
        assert record is not None
        assert record.attributes["asr_proposed_span_ids"] == [before[0].id]
        assert record.attributes["asr_proposed_spans_recomputed"] is False
        assert (record.attributes["n_lexical_before"], record.attributes["n_lexical_after"]) == (4, 2)
        assert record.attributes["n_rebracketed"] == 2

    def test_the_kept_span_still_names_the_reading_that_proposed_it(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """Its ``wasDerivedFrom`` edge points at the retired transcript, which is the whole record."""
        manifest, roots = corpus(1, asr_span=True)
        proposing = find_measurement(_store_of(roots[0]), CONSENSUS_TRANSCRIPT)
        assert proposing is not None

        _run(manifest)

        store = _store_of(roots[0])
        [span] = [entity for entity in store.entities("span") if entity.attributes.get("measure") == "asr"]
        assert store.derived_from(span.id) == [proposing.id]
        assert store.is_invalidated(proposing.id)
