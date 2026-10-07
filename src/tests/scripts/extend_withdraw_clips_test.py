"""The clip-withdrawal extend driver: a finished run's clip spans read against its own amplitudes.

Nothing here is faked and no audio is opened by the pass under test. A finished run in miniature is
enough: a source WAV on disk, the store ADMIT wrote its ``recording`` stream into, PREPROCESS's clip
spans with the ``clip_amplitude`` measurement beside them — the shape the corpus is in once
``scripts/extend_clip_amplitudes.py`` has passed over it — and the per-span records the general
spans carry.

``specs/20260912-quality-clip-consistency/design.md`` is the design.
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
from senselab.audio.workflows.triage.extend import withdraw_contradicted_clips
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import (
    find_branch_report,
    find_measurement,
    live_entities,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.nodes.preprocess import (
    WITHDRAW_VERB,
    reject_contradicted_clips,
    write_clip_spans,
    write_withdrawn_clips,
)
from senselab.audio.workflows.triage.nodes.quality import (
    CLIP_AMPLITUDE_MEASUREMENT,
    CLIP_FAMILY,
    CLIP_LEVELS,
    CONTEST_VERB,
    CONTRADICTED_CLIP,
    UNCLIPPED_LOUDER_N,
    clip_spans,
    quality,
)
from senselab.audio.workflows.triage.vocabulary import QUALITY
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_withdraw_clips.py"
_spec = importlib.util.spec_from_file_location("extend_withdraw_clips_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_withdraw_clips_under_test"] = cli
_spec.loader.exec_module(cli)

SR = 16000
"""The fixture's sampling rate, so a sample index and a time are one conversion apart."""

CLIP_RANGE = (8000, 8400)
"""The samples the seeded clip span covers."""

CLIP_EXTENT = (CLIP_RANGE[0] / SR, CLIP_RANGE[1] / SR)
"""The clip span's extent, in seconds."""

SECOND_CLIP_RANGE = (12000, 12400)
"""The samples a second seeded clip span covers, where a fixture asks for one."""

SECOND_CLIP_EXTENT = (SECOND_CLIP_RANGE[0] / SR, SECOND_CLIP_RANGE[1] / SR)
"""That span's extent, in seconds."""

SPAN_OVER_CLIP = (0.40, 0.60)
"""A general span overlapping the clip, which the figure flags and the per-span models measure."""

SPAN_ELSEWHERE = (1.20, 1.40)
"""A general span nowhere near it."""

DERIVATION = "withdraw_contradicted_clips"
"""The slice log's per-derivation column."""


def _samples(*, clip_level: float, louder: float | None, second_clip_level: float | None = None) -> np.ndarray:
    """A quiet bed with one or two plateaus, and optionally one unclipped sample louder than one.

    Args:
        clip_level: The level the first plateau holds.
        louder: An amplitude to place at sample 24000, or None to leave the bed there.
        second_clip_level: The level a second plateau holds, or None for a single-plateau recording.

    Returns:
        The recording, mono float32.
    """
    grid = np.arange(2 * SR) / SR
    out = (0.05 * np.sin(2 * np.pi * 220.0 * grid)).astype(np.float32)
    out[CLIP_RANGE[0] : CLIP_RANGE[1]] = np.float32(clip_level)
    if second_clip_level is not None:
        out[SECOND_CLIP_RANGE[0] : SECOND_CLIP_RANGE[1]] = np.float32(second_clip_level)
    if louder is not None:
        out[24000] = np.float32(louder)
    return out


def _seed_general_spans(store: ProvStore, activity: str, agent: str) -> list[str]:
    """Two general spans, each carrying the per-span records the branches key by span id.

    Args:
        store: The store.
        activity: The activity that proposed them.
        agent: The agent answerable for them.

    Returns:
        The span ids: the one overlapping the clip first.
    """
    span_ids: list[str] = []
    for extent, contains_clip in ((SPAN_OVER_CLIP, True), (SPAN_ELSEWHERE, False)):
        span_id = store.entity(
            prov_type="span",
            extent=extent,
            attributes={
                "signal": "preemphasised",
                "measure": "amplitude",
                "merged_proposals": 1,
                "contains_clip": contains_clip,
            },
        )
        store.was_generated_by(span_id, activity)
        store.was_attributed_to(span_id, agent)
        for name in ("span_hear", "span_yamnet"):
            window_id = store.entity(
                prov_type="measurement",
                extent=extent,
                attributes={"name": name, "signal": "plain", "span_id": span_id, "raw_scores": {"speech": 0.9}},
            )
            store.was_generated_by(window_id, activity)
            store.was_attributed_to(window_id, agent)
            store.was_derived_from(window_id, span_id)
        squim_id = store.entity(
            prov_type="assertion",
            extent=extent,
            attributes={"verb": "measure", "name": "squim", "stoi": 0.9, "pesq": 2.0, "si_sdr": 12.0},
        )
        store.was_generated_by(squim_id, activity)
        store.was_attributed_to(squim_id, agent)
        store.was_derived_from(squim_id, span_id)
        span_ids.append(span_id)
    return span_ids


def _seed_run(
    root: Path,
    *,
    clip_level: float = 0.5,
    louder: float | None = 0.9,
    second_clip_level: float | None = None,
    spans: bool = True,
    amplitudes: bool = True,
    contested: bool = False,
) -> Path:
    """Write one finished run: a source WAV, and the store the clip-amplitude pass left behind.

    Args:
        root: The run root, created if absent.
        clip_level: The first plateau's level.
        louder: An unclipped sample's amplitude, or None for a recording nothing contradicts.
        second_clip_level: A second plateau's level, for a run whose clip spans do not all fall.
        spans: Whether the store carries a clip span at all.
        amplitudes: Whether the ``clip_amplitude`` measurement is written beside the spans. False is
            the corpus before ``scripts/extend_clip_amplitudes.py``.
        contested: Whether QUALITY has already run and contested the contradicted spans, which is
            the state ``scripts/extend_quality.py`` left the corpus in.

    Returns:
        The source WAV's path.
    """
    run_dir = root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    source = root.parent / f"{root.name}.wav"
    sf.write(str(source), _samples(clip_level=clip_level, louder=louder, second_clip_level=second_clip_level), SR)
    sf.write(
        str(run_dir / "streams" / "enhanced.flac"),
        _samples(clip_level=clip_level, louder=None, second_clip_level=second_clip_level),
        SR,
    )

    store = ProvStore(run_id=root.name)
    config = load_triage_config()
    admitted = admit(store, source, config, run_dir=run_dir)
    assert admitted.audio is not None
    agent = software_agent(store)
    activity = store.activity(node="PREPROCESS", step="clip_spans", parameters={})
    store.was_associated_with(activity, agent)
    extents = [CLIP_EXTENT] if spans else []
    if spans and second_clip_level is not None:
        extents.append(SECOND_CLIP_EXTENT)
    if amplitudes:
        write_clip_spans(
            store,
            activity,
            agent,
            audio=admitted.audio,
            extents=extents,
            signal="recording",
            guard_samples=int(config.require("quality.clip_edge_guard_samples")),
        )
    else:
        for extent in extents:
            span_id = store.entity(
                prov_type="span", extent=extent, attributes={"family": CLIP_FAMILY, "signal": "recording"}
            )
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, agent)
    _seed_general_spans(store, activity, agent)
    if contested:
        quality(store, "recording", config, run_dir=run_dir)
    store.write_jsonl(run_dir / "store.jsonl")
    return source


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


def _log(tmp_path: Path) -> list[dict[str, Any]]:
    """The slice log the driver wrote.

    Args:
        tmp_path: The directory the manifest sits in.

    Returns:
        One record per manifest row, in order.
    """
    path = tmp_path / "slices" / "withdraw-clips-slice-0-of-1.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _withdrawals(store: ProvStore) -> list[Any]:
    """Every live assertion withdrawing a clip claim.

    Args:
        store: The store.

    Returns:
        The assertion entities.
    """
    return [
        entity
        for entity in live_entities(store, "assertion")
        if entity.attributes.get("reason") == CONTRADICTED_CLIP and entity.attributes.get("verb") == WITHDRAW_VERB
    ]


def _general_spans(store: ProvStore) -> list[Any]:
    """Every live span that is not a clip span.

    Args:
        store: The store.

    Returns:
        The span entities.
    """
    return [entity for entity in live_entities(store, "span") if entity.attributes.get("family") is None]


def _seeded_amplitudes(*, level: float, peak: float) -> ProvStore:
    """A store holding one clip span and a ``clip_amplitude`` measurement carrying chosen numbers.

    The pass reads stored outputs only, so seeding those outputs directly is the comparison's own
    interface — and the only way to place ``peak`` exactly on the margin, which no float32 recording
    can be made to do.

    Args:
        level: The clip span's own level.
        peak: The whole-file unclipped peak.

    Returns:
        The store.
    """
    store = ProvStore(run_id="boundary")
    agent = software_agent(store)
    activity = store.activity(node="PREPROCESS", step="clip_spans", parameters={})
    store.was_associated_with(activity, agent)
    span_id = store.entity(
        prov_type="span", extent=CLIP_EXTENT, attributes={"family": CLIP_FAMILY, "signal": "recording"}
    )
    store.was_generated_by(span_id, activity)
    store.was_attributed_to(span_id, agent)
    write_measurement(
        store,
        activity,
        agent,
        name=CLIP_AMPLITUDE_MEASUREMENT,
        signal="recording",
        attributes={
            "unclipped_peak": peak,
            "unclipped_peak_time_s": 1.5,
            "unclipped_samples_n": 100,
            "edge_guard_samples": 16,
            "clip_spans_n": 1,
            CLIP_LEVELS: {span_id: level},
            UNCLIPPED_LOUDER_N: {span_id: 1},
        },
        derived_from=(span_id,),
    )
    return store


def _margin() -> float:
    """The configured contradiction margin.

    Returns:
        ``quality.clip_contradiction_margin``.
    """
    return float(load_triage_config().require("quality.clip_contradiction_margin"))


def _widened(tmp_path: Path, margin: float) -> Any:  # noqa: ANN401 — TriageConfig, without importing it
    """The packaged configuration with a wider contradiction margin.

    Args:
        tmp_path: Where the partial YAML goes.
        margin: The margin to set.

    Returns:
        The merged configuration.
    """
    override = tmp_path / "margin.yaml"
    override.write_text(f"quality:\n  clip_contradiction_margin: {margin}\n", encoding="utf-8")
    return load_triage_config(override)


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
    """A factory for a manifest over N finished runs the clip-amplitude pass has reached.

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


class TestAContradictedClipIsTakenBack:
    """The span an unclipped sample denies is retired, and the store says who retired it and why."""

    def test_the_span_is_retired_and_the_withdrawal_is_derived_from_it(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The one relation the fresh path cannot record, because it writes no span to point at."""
        manifest, roots = corpus(1)
        before = [span.id for span in clip_spans(_store_of(roots[0]), "recording")]
        assert len(before) == 1

        assert _run(manifest) == 0

        store = _store_of(roots[0])
        assert clip_spans(store, "recording") == []
        assert store.is_invalidated(before[0])
        [withdrawal] = _withdrawals(store)
        assert store.derived_from(withdrawal.id) == before
        assert [record["status"] for record in _log(tmp_path)] == ["ok"]
        assert [record[DERIVATION] for record in _log(tmp_path)] == ["rewritten"]

    def test_the_withdrawal_is_the_entity_the_fresh_path_writes(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """Same vocabulary, same extent, same numbers — so the same id, since an id digests those."""
        manifest, roots = corpus(1)
        config = load_triage_config()
        source = roots[0].parent / f"{roots[0].name}.wav"
        margin = float(config.require("quality.clip_contradiction_margin"))
        audio = admit(ProvStore(run_id="scratch"), source, config, run_dir=roots[0] / "run").audio
        assert audio is not None
        _, withdrawn = reject_contradicted_clips(
            audio,
            [CLIP_EXTENT],
            guard_samples=int(config.require("quality.clip_edge_guard_samples")),
            margin=margin,
        )
        fresh = ProvStore(run_id=roots[0].name)
        fresh_activity = fresh.activity(node="PREPROCESS", step="clip_spans", parameters={})
        fresh_agent = software_agent(fresh)
        [expected] = write_withdrawn_clips(
            fresh, fresh_activity, fresh_agent, withdrawn=withdrawn, signal="recording", margin=margin
        )

        _run(manifest)

        [written] = _withdrawals(_store_of(roots[0]))
        assert written.id == expected
        assert written.attributes == fresh.get_entity(expected).attributes
        assert tuple(written.extent or ()) == CLIP_EXTENT

    def test_the_amplitude_measurement_is_rewritten_over_the_survivors(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The per-span maps lose the retired span; every whole-file value is carried through."""
        manifest, roots = corpus(1)
        before = find_measurement(_store_of(roots[0]), CLIP_AMPLITUDE_MEASUREMENT)
        assert before is not None

        _run(manifest)

        store = _store_of(roots[0])
        after = find_measurement(store, CLIP_AMPLITUDE_MEASUREMENT)
        assert after is not None and after.id != before.id
        assert store.is_invalidated(before.id)
        assert after.attributes[CLIP_LEVELS] == {}
        assert after.attributes[UNCLIPPED_LOUDER_N] == {}
        assert after.attributes["clip_spans_n"] == 0
        for key in ("unclipped_peak", "unclipped_peak_time_s", "unclipped_samples_n", "edge_guard_samples"):
            assert after.attributes[key] == before.attributes[key]

    def test_nothing_is_written_outside_the_recordings_own_run(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The store is replaced and ``prov/`` is exported; no second tree appears."""
        manifest, roots = corpus(1)
        _run(manifest)
        assert sorted(entry.name for entry in roots[0].iterdir()) == ["prov", "run"]
        assert sorted(entry.name for entry in (roots[0] / "run").iterdir()) == ["store.jsonl", "streams"]

    def test_the_bep028_files_are_re_exported_from_the_rewritten_store(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """``prov/`` must not disagree with the store it was exported from."""
        manifest, roots = corpus(1)
        _run(manifest)

        io_graph = json.loads((roots[0] / "prov" / "prov-triage_io.json").read_text())
        entity_ids = {node["Id"] for key in ("Files", "prov:Entity") for node in io_graph.get(key, [])}
        [withdrawal] = _withdrawals(_store_of(roots[0]))
        assert any(withdrawal.id in identifier for identifier in entity_ids), withdrawal.id


class TestThePerSpanRecordsKeepTheirSpans:
    """No non-clip span is re-minted, which is what lets a span's measurements keep pointing at it."""

    def test_every_per_span_record_still_resolves_to_a_live_span(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """``span_hear``, ``span_yamnet`` and ``squim`` key by span id; a new id would orphan them."""
        manifest, roots = corpus(1)
        before = [span.id for span in _general_spans(_store_of(roots[0]))]

        _run(manifest)

        store = _store_of(roots[0])
        live = {span.id for span in _general_spans(store)}
        assert live == set(before)
        records = [
            entity
            for entity in live_entities(store, "measurement")
            if entity.attributes.get("name") in {"span_hear", "span_yamnet"}
        ] + [entity for entity in live_entities(store, "assertion") if entity.attributes.get("name") == "squim"]
        assert len(records) == 6
        for record in records:
            assert set(store.derived_from(record.id)) <= live
            assert store.derived_from(record.id)
        for record in records:
            if "span_id" in record.attributes:
                assert record.attributes["span_id"] in live


class TestAStoreWithNothingToWithdraw:
    """A clip nothing is louder than stands, and its store is not rewritten at all."""

    def test_a_run_whose_clips_stand_is_left_byte_identical(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The loudest sample is inside the span, which is what a clip means."""
        manifest, roots = corpus(1, clip_level=0.98, louder=None)
        before = (roots[0] / "run" / "store.jsonl").read_bytes()

        assert _run(manifest) == 0

        assert (roots[0] / "run" / "store.jsonl").read_bytes() == before
        assert len(clip_spans(_store_of(roots[0]), "recording")) == 1
        assert _withdrawals(_store_of(roots[0])) == []
        assert _log(tmp_path)[0]["status"] == "skipped"
        assert _log(tmp_path)[0][DERIVATION] == "current"

    def test_a_run_with_no_clip_span_is_skipped_rather_than_refused(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """Nothing was asserted, so nothing can be taken back."""
        manifest, roots = corpus(1, spans=False)
        before = (roots[0] / "run" / "store.jsonl").read_bytes()

        assert _run(manifest) == 0

        assert (roots[0] / "run" / "store.jsonl").read_bytes() == before
        assert _log(tmp_path)[0]["status"] == "skipped"
        assert _log(tmp_path)[0][DERIVATION] == "absent"

    def test_a_store_with_no_clip_span_returns_none_without_raising(self, tmp_path: Path) -> None:
        """Read directly, since the driver never calls the pass on such a store."""
        root = tmp_path / "sub-99_ses-1_20260912-000000"
        root.mkdir(parents=True)
        _seed_run(root, spans=False)
        store = _store_of(root)
        assert withdraw_contradicted_clips(store, load_triage_config()) is None


class TestASecondPass:
    """A rerun must move nothing. Where every span fell it short-circuits; the fixpoint path is below."""

    def test_a_rerun_over_the_same_corpus_changes_nothing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The store is byte-identical. With the only span gone, the second pass reads no span at all."""
        manifest, roots = corpus(2)
        _run(manifest)
        after_first = {root: (root / "run" / "store.jsonl").read_bytes() for root in roots}

        assert _run(manifest) == 0

        for root in roots:
            assert (root / "run" / "store.jsonl").read_bytes() == after_first[root]
            assert len(_withdrawals(_store_of(root))) == 1
            assert clip_spans(_store_of(root), "recording") == []
        assert [record["status"] for record in _log(tmp_path)] == ["skipped", "skipped"]
        assert [record[DERIVATION] for record in _log(tmp_path)] == ["absent", "absent"]


class TestAStoreThePassCannotRead:
    """A missing input is an operational fact about the run, and never a quiet skip."""

    def test_clip_spans_with_no_amplitudes_are_refused_and_the_store_is_left_alone(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """This is the corpus before the clip-amplitude pass; there is nothing to read spans against."""
        manifest, roots = corpus(1, amplitudes=False)
        before = (roots[0] / "run" / "store.jsonl").read_bytes()

        assert _run(manifest) == 1

        assert (roots[0] / "run" / "store.jsonl").read_bytes() == before
        assert len(clip_spans(_store_of(roots[0]), "recording")) == 1
        record = _log(tmp_path)[0]
        assert record["status"] == "error"
        assert CLIP_AMPLITUDE_MEASUREMENT in str(record[DERIVATION])

    def test_the_refusal_is_a_lookup_error_naming_the_driver_that_supplies_it(self, tmp_path: Path) -> None:
        """Read directly, so the message is asserted rather than its string in a log."""
        root = tmp_path / "sub-98_ses-1_20260912-000000"
        root.mkdir(parents=True)
        _seed_run(root, amplitudes=False)
        with pytest.raises(LookupError, match="extend_clip_amplitudes"):
            withdraw_contradicted_clips(_store_of(root), load_triage_config())

    def test_a_run_with_no_store_is_recorded_and_the_others_are_still_read(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """A missing store is this recording's error; its neighbours are unaffected."""
        manifest, roots = corpus(2)
        (roots[0] / "run" / "store.jsonl").unlink()

        assert _run(manifest) == 1

        assert clip_spans(_store_of(roots[1]), "recording") == []
        assert [record["status"] for record in _log(tmp_path)] == ["error", "ok"]

    def test_a_manifest_row_naming_no_run_root_is_refused_rather_than_guessed(self) -> None:
        """A wrong guess would rewrite the wrong recording's store."""
        with pytest.raises(ValueError, match="no run root"):
            cli.run_root_of(Path("/corpus/enhanced.flac"))


class TestTheComparisonAtItsBoundary:
    """The third copy of the rule, pinned where a 0.5 % margin actually decides."""

    def test_a_peak_exactly_at_the_margin_is_not_a_contradiction(self) -> None:
        """The comparison is ``>``: equality is inside the tolerance, not outside it."""
        margin = _margin()
        store = _seeded_amplitudes(level=0.5, peak=0.5 * (1.0 + margin))
        assert withdraw_contradicted_clips(store, load_triage_config()) is None
        assert len(clip_spans(store, "recording")) == 1

    def test_one_ulp_above_the_margin_is(self) -> None:
        """The control on the test above, at the tightest separation a float admits."""
        margin = _margin()
        store = _seeded_amplitudes(level=0.5, peak=float(np.nextafter(0.5 * (1.0 + margin), 1.0)))
        assert withdraw_contradicted_clips(store, load_triage_config()) is not None
        assert clip_spans(store, "recording") == []

    def test_a_peak_below_the_spans_own_level_is_never_a_contradiction(self) -> None:
        """0.4995 under a 0.5 clip is the clip standing, whichever side the margin is read on."""
        store = _seeded_amplitudes(level=0.5, peak=0.4995)
        assert withdraw_contradicted_clips(store, load_triage_config()) is None
        assert len(clip_spans(store, "recording")) == 1

    def test_the_margin_is_read_from_the_configuration(self, tmp_path: Path) -> None:
        """0.504 over a 0.5 clip is 0.8 %: withdrawn at the packaged margin, kept at a wider one."""
        store = _seeded_amplitudes(level=0.5, peak=0.504)
        assert withdraw_contradicted_clips(store, _widened(tmp_path, 0.05)) is None

        assert withdraw_contradicted_clips(store, load_triage_config()) is not None
        assert clip_spans(store, "recording") == []


class TestARunWhereOnlySomeClipSpansFall:
    """The survivors are what the rewritten measurement is about, and what the second pass reads."""

    def test_the_rewritten_maps_hold_exactly_the_surviving_span(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """A level keyed to a retired span would be a level no live span can be read against."""
        manifest, roots = corpus(1, clip_level=0.10, second_clip_level=0.98, louder=0.5)
        before = clip_spans(_store_of(roots[0]), "recording")
        assert len(before) == 2

        assert _run(manifest) == 0

        store = _store_of(roots[0])
        survivors = clip_spans(store, "recording")
        assert [span.id for span in survivors] == [before[1].id]
        assert tuple(survivors[0].extent or ()) == SECOND_CLIP_EXTENT
        after = find_measurement(store, CLIP_AMPLITUDE_MEASUREMENT)
        assert after is not None
        assert list(after.attributes[CLIP_LEVELS]) == [before[1].id]
        assert list(after.attributes[UNCLIPPED_LOUDER_N]) == [before[1].id]
        assert after.attributes["clip_spans_n"] == 1
        assert len(_withdrawals(store)) == 1

    def test_the_second_pass_returns_none_with_a_clip_span_still_standing(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The fixpoint branch, not the no-clip-span short circuit: a span is read and kept."""
        manifest, roots = corpus(1, clip_level=0.10, second_clip_level=0.98, louder=0.5)
        _run(manifest)

        store = _store_of(roots[0])
        assert len(clip_spans(store, "recording")) == 1
        assert withdraw_contradicted_clips(store, load_triage_config()) is None

    def test_the_audit_passes_on_what_this_pass_leaves(self, corpus: Callable[..., tuple[Path, list[Path]]]) -> None:
        """QUALITY reading the same numbers must find nothing left to contest."""
        manifest, roots = corpus(1, clip_level=0.10, second_clip_level=0.98, louder=0.5)
        _run(manifest)

        store = _store_of(roots[0])
        result = quality(store, "recording", load_triage_config(), run_dir=roots[0] / "run")
        assert result.report.conformance is True
        assert store.get_entity(result.report_entity_id).attributes["checked_n"] == 1


class TestQualitysOwnFindingsAboutAWithdrawnSpan:
    """The corpus already carries QUALITY's contests; a contest must not outlive its span."""

    def test_the_contest_of_a_withdrawn_span_is_retired_with_it(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """Two live assertions over one extent, one contesting and one withdrawing, double-count."""
        manifest, roots = corpus(1, contested=True)
        seeded = _store_of(roots[0])
        contests = [e for e in live_entities(seeded, "assertion") if e.attributes.get("verb") == CONTEST_VERB]
        assert len(contests) == 1

        assert _run(manifest) == 0

        store = _store_of(roots[0])
        assert [e for e in live_entities(store, "assertion") if e.attributes.get("verb") == CONTEST_VERB] == []
        assert store.is_invalidated(contests[0].id)
        assert len(_withdrawals(store)) == 1

    def test_the_only_live_assertion_pointing_at_a_retired_span_is_its_own_withdrawal(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """Stated over the whole store: a live claim about a span that has gone is the record of its going."""
        manifest, roots = corpus(1, contested=True)
        _run(manifest)

        store = _store_of(roots[0])
        pointing = [
            entity
            for entity in live_entities(store, "assertion")
            if any(store.is_invalidated(source) for source in store.derived_from(entity.id))
        ]
        assert pointing, "the withdrawal itself must point at the span it retired"
        assert {entity.attributes.get("verb") for entity in pointing} == {WITHDRAW_VERB}
        assert {entity.attributes.get("reason") for entity in pointing} == {CONTRADICTED_CLIP}

    def test_no_quality_report_survives_the_withdrawal(self, corpus: Callable[..., tuple[Path, list[Path]]]) -> None:
        """A report counting contests of spans that are gone is a conclusion about nothing."""
        manifest, roots = corpus(1, contested=True)
        seeded = find_branch_report(_store_of(roots[0]), QUALITY)
        assert seeded is not None and seeded.attributes["conformance"] is False

        _run(manifest)

        store = _store_of(roots[0])
        assert find_branch_report(store, QUALITY) is None
        assert store.is_invalidated(seeded.id)

    def test_the_surviving_span_is_untouched_by_the_retirement(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """QUALITY contested one of two spans; the retirement follows that span and not the other."""
        manifest, roots = corpus(1, clip_level=0.10, second_clip_level=0.98, louder=0.5, contested=True)
        seeded = _store_of(roots[0])
        assert len([e for e in live_entities(seeded, "assertion") if e.attributes.get("verb") == CONTEST_VERB]) == 1
        survivor = clip_spans(seeded, "recording")[1].id

        _run(manifest)

        store = _store_of(roots[0])
        assert [span.id for span in clip_spans(store, "recording")] == [survivor]
        assert [e for e in live_entities(store, "assertion") if e.attributes.get("verb") == CONTEST_VERB] == []
        assert find_branch_report(store, QUALITY) is None
