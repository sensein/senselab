"""QUALITY — clip spans read against the amplitudes PREPROCESS measured beside them.

The spans are seeded rather than detected: this module's subject is what QUALITY does with a clip
span it was handed, and ``src/tests/audio/tasks/clipping`` owns where ClipDaT opens an event. The
seeding also keeps the audit testable as an audit — ``_clip_spans`` withdraws a contradicted
candidate before it becomes a span, so a fixture that went through it could never hand QUALITY the
span it exists to contest.

There are two seedings, because there are two ways a store gets its inputs. :func:`_seed` goes
through PREPROCESS's own ``write_clip_spans``, which is a fresh run. :func:`_seed_bare` writes the
spans alone and leaves ``extend_clip_amplitudes`` to append the measurement, which is a finished run
extended in place. Both run over a real recording that went through the real ADMIT, and QUALITY must
not be able to tell them apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.preprocess import extend_clip_amplitudes, write_clip_spans
from senselab.audio.workflows.triage.nodes.quality import (
    CLIP_AMPLITUDE_MEASUREMENT,
    CLIP_FAMILY,
    CLIP_LEVELS,
    CONTRADICTED_CLIP,
    UNCLIPPED_LOUDER_N,
    quality,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

SR = 16000
"""The fixture's sampling rate, so a sample index and a time are one conversion apart."""

QUIET = 0.05
"""The bed every fixture recording sits on. Below every clip level any test places."""

QUANTISATION = 1.0 / 32768
"""One int16 step, which is what the fixture WAV round-trips through, as a real recording does."""


def _bed(duration_s: float = 2.0) -> np.ndarray:
    """A quiet, non-constant bed ADMIT accepts, with nothing in it above :data:`QUIET`.

    Args:
        duration_s: How long the recording is.

    Returns:
        The samples, mono float32.
    """
    grid = np.arange(int(duration_s * SR)) / SR
    return (QUIET * np.sin(2 * np.pi * 220.0 * grid)).astype(np.float32)


def _plateau(samples: np.ndarray, first: int, stop: int, level: float) -> tuple[int, int]:
    """Hold one level across a run of samples, as a saturated waveform does.

    Args:
        samples: The recording, modified in place.
        first: First sample of the run.
        stop: One past its last sample.
        level: The level held.

    Returns:
        ``(first, stop)``, so a caller can seed the span naming the same samples.
    """
    samples[first:stop] = np.float32(level)
    return first, stop


def _seed(
    store: ProvStore,
    tmp_path: Path,
    wav_writer: Callable[..., Path],
    samples: np.ndarray,
    clip_ranges: list[tuple[int, int]],
    config: TriageConfig | None = None,
) -> Path:
    """Write the recording, ADMIT it, and let PREPROCESS place its clip spans over the named samples.

    Args:
        store: The store to seed.
        tmp_path: The run directory.
        wav_writer: The fixture WAV writer.
        samples: The recording.
        clip_ranges: Half-open sample ranges PREPROCESS is to have called clipped.
        config: The configuration PREPROCESS runs under — the edge guard is its decision. The
            packaged one unless a test widens or removes the guard.

    Returns:
        The recording's path on disk.
    """
    path = wav_writer("input.wav", samples, SR)
    settings = config if config is not None else load_triage_config()
    admitted = admit(store, path, settings, run_dir=tmp_path)
    assert admitted.audio is not None
    agent = store.agent(agent_type="software", version="senselab test-seed")
    activity = store.activity(node="PREPROCESS", step="clip_spans", parameters={})
    store.was_associated_with(activity, agent)
    write_clip_spans(
        store,
        activity,
        agent,
        audio=admitted.audio,
        extents=[(first / SR, stop / SR) for first, stop in clip_ranges],
        signal="recording",
        guard_samples=int(settings.require("quality.clip_edge_guard_samples")),
    )
    return path


def _seed_bare(
    store: ProvStore,
    tmp_path: Path,
    wav_writer: Callable[..., Path],
    samples: np.ndarray,
    clip_ranges: list[tuple[int, int]],
) -> Path:
    """Write the recording, ADMIT it, and place clip spans carrying no amplitudes at all.

    This is the completed corpus's shape: 62,550 recordings whose clip spans were written before
    the amplitudes existed, so each span carries its ``family``, its ``signal`` and its extent and
    nothing else, and no ``clip_amplitude`` measurement sits beside them.

    Args:
        store: The store to seed.
        tmp_path: The run directory.
        wav_writer: The fixture WAV writer.
        samples: The recording.
        clip_ranges: Half-open sample ranges PREPROCESS is to have called clipped.

    Returns:
        The recording's path on disk.
    """
    path = wav_writer("input.wav", samples, SR)
    admitted = admit(store, path, load_triage_config(), run_dir=tmp_path)
    assert admitted.audio is not None
    agent = store.agent(agent_type="software", version="senselab test-seed")
    activity = store.activity(node="PREPROCESS", step="clip_spans", parameters={})
    store.was_associated_with(activity, agent)
    for first, stop in clip_ranges:
        span_id = store.entity(
            prov_type="span",
            extent=(first / SR, stop / SR),
            attributes={"family": CLIP_FAMILY, "signal": "recording"},
        )
        store.was_generated_by(span_id, activity)
        store.was_attributed_to(span_id, agent)
    return path


def _override(tmp_path: Path, text: str) -> TriageConfig:
    """The packaged configuration with one partial YAML deep-merged over it.

    Args:
        tmp_path: Where the override file is written.
        text: The partial YAML.

    Returns:
        The merged configuration.
    """
    path = tmp_path / "quality-override.yaml"
    path.write_text(text)
    return load_triage_config(path)


def _contests(store: ProvStore) -> list[Entity]:
    """Every live assertion QUALITY wrote against a clip span.

    Args:
        store: The store QUALITY wrote to.

    Returns:
        The contesting assertions, oldest first.
    """
    return [
        entity for entity in live_entities(store, "assertion") if entity.attributes.get("reason") == CONTRADICTED_CLIP
    ]


class TestAClipNothingContradicts:
    """The check must stay quiet on a recording whose clip really is its ceiling."""

    def test_a_genuine_clip_with_nothing_louder_outside_it_passes(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The loudest sample in the file is inside the clip span, which is what a clip means."""
        samples = _bed()
        clipped = _plateau(samples, 16000, 16400, 0.98)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert _contests(store) == []

    def test_a_recording_with_no_clip_span_is_a_clean_no_op(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Nothing was asserted, so there is nothing to contradict and nothing to write."""
        _seed(store, tmp_path, wav_writer, _bed(), [])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert "no clip span" in result.verdict.why
        assert _contests(store) == []
        verdict = store.get_entity(result.verdict_entity_id)
        assert verdict.attributes["clip_spans_n"] == 0
        assert verdict.attributes["contradicted_n"] == 0

    def test_the_verdict_belongs_to_no_kind(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """QUALITY is a graph edge, not a branch: it is the authority on no kind at all."""
        _seed(store, tmp_path, wav_writer, _bed(), [])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.node == "QUALITY"
        assert result.verdict.kind is None


class TestAClipAnUnclippedSampleDenies:
    """A clip below an unclipped sample is the false positive this check exists to name."""

    def test_a_clip_at_half_scale_under_a_louder_sample_is_contested(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The recording reaches 0.9 without being called clipped, so the 0.5 plateau is not a ceiling."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert CONTRADICTED_CLIP in result.verdict.why
        verdict = store.get_entity(result.verdict_entity_id)
        assert verdict.attributes["contradicted_n"] == 1
        assert verdict.attributes["checked_n"] == 1

    def test_the_offending_sample_is_named_with_its_time_and_amplitude(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A finding a reader cannot go and look at is not evidence."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        result = quality(store, "recording", config, run_dir=tmp_path)
        contest = _contests(store)
        assert len(contest) == 1
        assert contest[0].attributes["verb"] == "contest"
        assert contest[0].attributes["claim"] == "clip"
        assert contest[0].attributes["clip_level"] == pytest.approx(0.5, abs=QUANTISATION)
        assert contest[0].attributes["louder_amplitude"] == pytest.approx(0.9, abs=QUANTISATION)
        assert contest[0].attributes["louder_time_s"] == pytest.approx(24000 / SR)
        assert contest[0].attributes["louder_samples_n"] == 1
        assert contest[0].extent == (8000 / SR, 8400 / SR)
        assert f"{24000 / SR:.3f}s" in result.verdict.why

    def test_the_contested_span_is_kept_and_the_finding_is_derived_from_it(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The span is PREPROCESS's assertion; QUALITY answers it and never withdraws it."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        quality(store, "recording", config, run_dir=tmp_path)
        spans = [entity for entity in live_entities(store, "span") if entity.attributes.get("family") == "clip"]
        assert len(spans) == 1
        assert not store.is_invalidated(spans[0].id)
        contest = _contests(store)[0]
        assert spans[0].id in store.derived_from(contest.id)

    def test_only_the_clip_below_the_louder_sample_is_contested(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The question is asked of each span against every unclipped sample, span by span."""
        samples = _bed()
        loud = _plateau(samples, 4000, 4400, 0.95)
        quiet = _plateau(samples, 12000, 12400, 0.4)
        samples[24000] = np.float32(0.9)
        _seed(store, tmp_path, wav_writer, samples, [loud, quiet])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        contest = _contests(store)
        assert len(contest) == 1
        assert contest[0].extent == (12000 / SR, 12400 / SR)
        assert store.get_entity(result.verdict_entity_id).attributes["checked_n"] == 2

    def test_a_contradiction_is_recorded_rather_than_raised(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A finding about the recording is never an operational failure of the node."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict_entity_id in result.view
        assert _contests(store)[0].id in result.view


class TestTheMargin:
    """Exact comparison would fire on quantisation; the tolerance is the configuration's."""

    def test_a_sample_within_the_margin_is_not_a_contradiction(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """0.502 over a 0.5 clip is 0.4% — inside the detector's own near-threshold band."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.502)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert _contests(store) == []

    def test_a_sample_beyond_the_margin_is_a_contradiction(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The control on the test above: 0.504 over the same clip is 0.8% and does fire."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.504)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert len(_contests(store)) == 1

    def test_the_margin_is_read_from_the_configuration(
        self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A campaign that widens the tolerance gets a quieter check, without a code change."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.504)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        widened = _override(tmp_path, "quality:\n  clip_contradiction_margin: 0.05\n")
        result = quality(store, "recording", widened, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS

    def test_an_unmeasured_margin_raises_before_anything_is_written(
        self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A null threshold is a decision nobody took, and a guessed one would be worse."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        unset = _override(tmp_path, "quality:\n  clip_contradiction_margin: null\n")
        with pytest.raises(ValueError, match="quality.clip_contradiction_margin"):
            quality(store, "recording", unset, run_dir=tmp_path)
        assert store.activities(node="QUALITY") == []


class TestTheEdgeGuard:
    """A sample beside a clip span is that run's own decay, not independent evidence."""

    def test_a_louder_sample_beside_a_span_edge_is_not_unclipped_evidence(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Two samples past the edge of the loud span, so the guard excludes it from the comparison."""
        samples = _bed()
        loud = _plateau(samples, 4000, 4400, 0.95)
        quiet = _plateau(samples, 12000, 12400, 0.4)
        samples[4401] = np.float32(0.9)
        _seed(store, tmp_path, wav_writer, samples, [loud, quiet])
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert _contests(store) == []

    def test_without_the_guard_the_same_sample_contradicts(
        self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The control: the guard is what silences it, and it is the configuration's to set.

        The guard is applied where the samples are, so the override belongs to the PREPROCESS pass
        that measured them; QUALITY reads the guard back off the measurement.
        """
        samples = _bed()
        loud = _plateau(samples, 4000, 4400, 0.95)
        quiet = _plateau(samples, 12000, 12400, 0.4)
        samples[4401] = np.float32(0.9)
        unguarded = _override(tmp_path, "quality:\n  clip_edge_guard_samples: 0\n")
        _seed(store, tmp_path, wav_writer, samples, [loud, quiet], unguarded)
        result = quality(store, "recording", unguarded, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert _contests(store)[0].extent == (12000 / SR, 12400 / SR)
        assert store.get_entity(result.verdict_entity_id).attributes["clip_edge_guard_samples"] == 0


class TestTheStreamItReads:
    """The check is about the original recording, which is the signal the spans were detected on."""

    def test_an_absent_recording_stream_raises(self, store: ProvStore, config: TriageConfig, tmp_path: Path) -> None:
        """Nothing to read is an operational failure, which the runner records as ``errored``."""
        with pytest.raises(LookupError, match="recording"):
            quality(store, "recording", config, run_dir=tmp_path)

    def test_a_clip_span_over_another_signal_is_not_read_against_this_one(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Amplitudes are not comparable across streams, so a span naming another one is not checked."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        elsewhere = [span for span in live_entities(store, "span")][0]
        elsewhere.attributes["signal"] = "plain"
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert store.get_entity(result.verdict_entity_id).attributes["clip_spans_n"] == 0


class TestWhatItMayRead:
    """QUALITY reads stored outputs. Not audio, and not an input it was never handed."""

    def test_the_node_works_with_no_audio_left_on_disk(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Every amplitude was measured by PREPROCESS, so there is nothing left for QUALITY to decode."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        path = _seed(store, tmp_path, wav_writer, samples, [clipped])
        for stream in [path, *(tmp_path / "streams").glob("*")]:
            stream.unlink()
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert _contests(store)[0].attributes["louder_amplitude"] == pytest.approx(0.9, abs=QUANTISATION)

    def test_clip_spans_with_no_amplitude_measurement_refuse(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A weaker finding from a missing input would read as a clean recording; the run errors instead."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        _seed(store, tmp_path, wav_writer, samples, [clipped])
        measurement = [
            entity for entity in live_entities(store, "measurement") if entity.attributes["name"] == "clip_amplitude"
        ][0]
        store.was_invalidated_by(measurement.id, store.activity(node="TEST", step="drop", parameters={}))
        with pytest.raises(LookupError, match=CLIP_AMPLITUDE_MEASUREMENT):
            quality(store, "recording", config, run_dir=tmp_path)
        assert store.activities(node="QUALITY") == []

    def test_no_clip_span_needs_no_measurement(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Nothing was asserted, so there is nothing to read it against, and nothing to refuse."""
        _seed(store, tmp_path, wav_writer, _bed(), [])
        measurement = [
            entity for entity in live_entities(store, "measurement") if entity.attributes["name"] == "clip_amplitude"
        ][0]
        store.was_invalidated_by(measurement.id, store.activity(node="TEST", step="drop", parameters={}))
        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS


class TestAStoreTheMeasurementWasAppendedTo:
    """The corpus's shape after an extend pass: untouched spans, the amplitudes beside them.

    The store is append-only, so an attribute cannot be added to a span that already exists. Every
    level therefore lives in the measurement, keyed by span id, and a span that carries no attribute
    beyond its family and its signal is checked exactly as a freshly written one is.
    """

    def test_spans_carrying_no_amplitudes_are_checked_from_the_measurement(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The finding is the one a fresh run produces, off spans nothing was stamped on."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        _seed_bare(store, tmp_path, wav_writer, samples, [clipped])
        spans = [entity for entity in live_entities(store, "span") if entity.attributes.get("family") == CLIP_FAMILY]
        assert [sorted(span.attributes) for span in spans] == [["family", "signal"]]

        assert extend_clip_amplitudes(store, config, run_dir=tmp_path) is not None

        result = quality(store, "recording", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        contest = _contests(store)
        assert len(contest) == 1
        assert contest[0].attributes["clip_level"] == pytest.approx(0.5, abs=QUANTISATION)
        assert contest[0].attributes["louder_amplitude"] == pytest.approx(0.9, abs=QUANTISATION)
        assert contest[0].attributes["louder_samples_n"] == 1
        assert store.get_entity(result.verdict_entity_id).attributes["checked_n"] == 1

    def test_the_appended_measurement_keys_every_level_by_span_id(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A level a reader cannot attribute to a span is not a level; the key is the span's id."""
        samples = _bed()
        loud = _plateau(samples, 4000, 4400, 0.95)
        quiet = _plateau(samples, 12000, 12400, 0.4)
        _seed_bare(store, tmp_path, wav_writer, samples, [loud, quiet])
        measurement_id = extend_clip_amplitudes(store, config, run_dir=tmp_path)
        assert measurement_id is not None

        amplitudes = store.get_entity(measurement_id).attributes
        spans = [entity for entity in live_entities(store, "span") if entity.attributes.get("family") == CLIP_FAMILY]
        assert set(amplitudes[CLIP_LEVELS]) == {span.id for span in spans}
        assert set(amplitudes[UNCLIPPED_LOUDER_N]) == {span.id for span in spans}
        by_extent = {span.extent: span.id for span in spans}
        assert amplitudes[CLIP_LEVELS][by_extent[(4000 / SR, 4400 / SR)]] == pytest.approx(0.95, abs=QUANTISATION)
        assert amplitudes[CLIP_LEVELS][by_extent[(12000 / SR, 12400 / SR)]] == pytest.approx(0.4, abs=QUANTISATION)

    def test_an_extended_span_is_the_same_entity_a_fresh_run_writes(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """No amplitude is stamped on a span, so the two paths mint the same id for the same extent.

        Entity ids digest the attributes, so a span carrying a level would be identified by it, and
        an extended corpus and a re-run one would disagree about which entity a clip span is.
        """
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        _seed_bare(store, tmp_path, wav_writer, samples, [clipped])
        extended = [e.id for e in live_entities(store, "span") if e.attributes.get("family") == CLIP_FAMILY]

        fresh = ProvStore(run_id=store.run_id)
        _seed(fresh, tmp_path, wav_writer, samples, [clipped])
        written = [e.id for e in live_entities(fresh, "span") if e.attributes.get("family") == CLIP_FAMILY]
        assert extended == written

    def test_the_same_store_refuses_until_the_measurement_is_appended(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The refusal is what makes the extend pass necessary rather than optional."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        samples[24000] = np.float32(0.9)
        _seed_bare(store, tmp_path, wav_writer, samples, [clipped])
        with pytest.raises(LookupError, match=CLIP_AMPLITUDE_MEASUREMENT):
            quality(store, "recording", config, run_dir=tmp_path)
        assert store.activities(node="QUALITY") == []

    def test_appending_twice_writes_nothing_the_second_time(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The measurement carries no path and no timestamp, so a recompute is a set-union no-op."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        _seed_bare(store, tmp_path, wav_writer, samples, [clipped])
        first = extend_clip_amplitudes(store, config, run_dir=tmp_path)
        fingerprint = store.fingerprint()

        assert extend_clip_amplitudes(store, config, run_dir=tmp_path) == first
        assert store.fingerprint() == fingerprint

    def test_a_store_with_no_clip_span_gets_no_measurement(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Nothing was asserted, so there is nothing to measure and QUALITY refuses nothing."""
        _seed_bare(store, tmp_path, wav_writer, _bed(), [])
        assert extend_clip_amplitudes(store, config, run_dir=tmp_path) is None
        assert quality(store, "recording", config, run_dir=tmp_path).verdict.outcome is Outcome.PASS

    def test_a_recording_whose_bytes_changed_is_refused_rather_than_measured(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Amplitudes from other bytes would not belong to the spans they are keyed against."""
        samples = _bed()
        clipped = _plateau(samples, 8000, 8400, 0.5)
        path = _seed_bare(store, tmp_path, wav_writer, samples, [clipped])
        replaced = _bed()
        _plateau(replaced, 8000, 8400, 0.8)
        wav_writer(path.name, replaced, SR)

        with pytest.raises(ValueError, match="digests to"):
            extend_clip_amplitudes(store, config, run_dir=tmp_path)
