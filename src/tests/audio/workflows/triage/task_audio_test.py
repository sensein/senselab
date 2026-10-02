"""Tests for the task-extent cut of the plain, enhanced and released redacted streams."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.redaction.api import RedactionExtent, apply_redactions
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import find_measurement, path_attributes, write_stream
from senselab.audio.workflows.triage.task_audio import (
    DEFINITION,
    MEASUREMENT,
    PRESENT,
    SIDECAR,
    WRITTEN,
    cut_task_audio,
    masks_in_cut,
    sample_bounds,
    task_extent,
)
from senselab.audio.workflows.triage.vocabulary import Release
from senselab.utils.prov_store import ProvStore

RECORDING_HZ = 44_100
PLAIN_HZ = 16_000
DURATION_S = 10.0
MASK = RedactionExtent(start=4.2, end=4.9, category="PERSON")
OUTSIDE = RedactionExtent(start=0.3, end=0.6, category="DATE_TIME")


@pytest.fixture
def config() -> TriageConfig:
    """The packaged triage configuration."""
    return load_triage_config()


def _tone(rate: int, seconds: float, channels: int = 1) -> Audio:
    t = torch.arange(int(round(rate * seconds)), dtype=torch.float32) / rate
    wave = 0.5 * torch.sin(2 * math.pi * 220.0 * t) + 0.1
    return Audio(waveform=wave.repeat(channels, 1), sampling_rate=rate)


def _stream(store: ProvStore, run_dir: Path, name: str, audio: Audio, *, source: str | None = None) -> str:
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    relative, report = write_stream(audio, run_dir, name)
    entity = store.entity(
        prov_type="stream",
        extent=(0.0, audio.waveform.shape[-1] / audio.sampling_rate),
        attributes={
            "name": name,
            **path_attributes(relative, run_dir),
            "sampling_rate": audio.sampling_rate,
            "channels": int(audio.waveform.shape[0]),
            "write_gain": report.gain,
        },
    )
    if source is not None:
        store.was_derived_from(entity, source)
    return entity


def _task(store: ProvStore, start: float, end: float) -> str:
    return store.entity(prov_type="span", extent=(start, end), attributes={"role": "task_extent", "branch": "SPEECH"})


def _seed(
    tmp_path: Path,
    *,
    tasks: tuple[tuple[float, float], ...] = ((2.0, 5.0), (3.0, 7.5)),
    enhanced: bool = True,
    redacted: bool = True,
    release: str = Release.WITH_REDACTION.value,
) -> tuple[ProvStore, Path]:
    run_dir = tmp_path / "run"
    store = ProvStore(run_id="task-audio-test")
    recording = _stream(store, run_dir, "recording", _tone(RECORDING_HZ, DURATION_S, channels=2))
    plain = _stream(store, run_dir, "plain", _tone(PLAIN_HZ, DURATION_S), source=recording)
    if enhanced:
        _stream(store, run_dir, "enhanced", _tone(PLAIN_HZ, DURATION_S), source=plain)
    for start, end in tasks:
        _task(store, start, end)
    if redacted:
        for mask in (OUTSIDE, MASK):
            store.entity(
                prov_type="span",
                extent=(mask.start, mask.end),
                attributes={"name": "redaction", "category": mask.category},
            )
        masked = apply_redactions(_tone(RECORDING_HZ, DURATION_S, channels=2), [OUTSIDE, MASK], fill="silence")
        _stream(store, run_dir, "redacted", masked, source=recording)
        store.entity(
            prov_type="verdict", extent=None, attributes={"node": "REDACT", "outcome": "pass", "fill": "silence"}
        )
    store.entity(prov_type="verdict", extent=None, attributes={"node": "VERDICT", "release": release})
    return store, run_dir


class TestTheExtent:
    """One definition: the hull of the live task-extent spans, padded, clamped to the recording."""

    def test_the_hull_is_padded_on_both_sides(self, tmp_path: Path) -> None:
        """First span's start to last span's end, each moved outward by the pad."""
        store, _ = _seed(tmp_path)
        extent = task_extent(store, padding_s=0.25)
        assert extent is not None
        assert (extent.hull_start_s, extent.hull_end_s) == (2.0, 7.5)
        assert (extent.start_s, extent.end_s) == (1.75, 7.75)
        assert extent.definition == DEFINITION
        assert not extent.clamped_start and not extent.clamped_end

    def test_the_pad_is_clamped_at_both_edges(self, tmp_path: Path) -> None:
        """A task touching the recording's ends gets no pad past them, and says it was clamped."""
        store, _ = _seed(tmp_path, tasks=((0.1, 9.9),))
        extent = task_extent(store, padding_s=0.25)
        assert extent is not None
        assert (extent.start_s, extent.end_s) == (0.0, DURATION_S)
        assert extent.clamped_start and extent.clamped_end

    def test_no_task_span_is_no_extent(self, tmp_path: Path) -> None:
        """Nothing marks the task, so nothing is cut."""
        store, _ = _seed(tmp_path, tasks=())
        assert task_extent(store, padding_s=0.25) is None

    def test_an_invalidated_span_is_not_read(self, tmp_path: Path) -> None:
        """Only live spans make the hull."""
        store, _ = _seed(tmp_path, tasks=((2.0, 5.0),))
        late = _task(store, 6.0, 9.0)
        activity = store.activity(node="TEST", step="retire", parameters={})
        store.was_invalidated_by(late, activity)
        extent = task_extent(store, padding_s=0.0)
        assert extent is not None and extent.end_s == 5.0

    def test_a_negative_pad_is_refused(self, tmp_path: Path) -> None:
        """A pad inward would cut the task itself."""
        store, _ = _seed(tmp_path)
        with pytest.raises(ValueError, match="padding_s"):
            task_extent(store, padding_s=-0.1)


class TestSampleBounds:
    """The start rounds down, the end up, both clamped: the rule masks are applied by."""

    @pytest.mark.parametrize("rate", [8_000, 16_000, 22_050, 44_100, 48_000])
    def test_each_bound_is_within_one_sample_of_the_seconds(self, rate: int) -> None:
        """At every rate, the cut's first and last samples sit within one sample of the extent."""
        lo, hi = sample_bounds(1.2345678, 7.6543219, rate, rate * 10)
        assert 0 <= 1.2345678 * rate - lo < 1
        assert 0 <= hi - 7.6543219 * rate < 1

    def test_bounds_are_clamped(self) -> None:
        """Past either end is the end."""
        assert sample_bounds(-1.0, 99.0, 16_000, 1000) == (0, 1000)


class TestTheCut:
    """Every stream that exists is cut to the same seconds, at its own rate."""

    def test_all_three_are_written_and_aligned(self, tmp_path: Path, config: TriageConfig) -> None:
        """Plain, enhanced and redacted each start and end within one sample of the extent."""
        store, run_dir = _seed(tmp_path)
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.cuts == {"plain": WRITTEN, "enhanced": WRITTEN, "redacted": WRITTEN}
        extent = outcome.extent
        assert extent is not None
        for name, rate, channels in (("plain", PLAIN_HZ, 1), ("enhanced", PLAIN_HZ, 1), ("redacted", RECORDING_HZ, 2)):
            audio = Audio(filepath=str(run_dir / f"streams/task_{name}.flac"))
            assert audio.sampling_rate == rate
            assert audio.waveform.shape[0] == channels
            assert abs(audio.waveform.shape[-1] - extent.duration_s * rate) <= 1
            lo, _ = sample_bounds(extent.start_s, extent.end_s, rate, int(DURATION_S * rate))
            assert abs(lo / rate - extent.start_s) * rate < 1

    def test_the_cut_holds_the_source_samples(self, tmp_path: Path, config: TriageConfig) -> None:
        """The plain cut is the plain stream's own samples over the range, not a resample of them."""
        store, run_dir = _seed(tmp_path)
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.extent is not None
        plain = Audio(filepath=str(run_dir / "streams/plain.flac")).waveform
        got = Audio(filepath=str(run_dir / "streams/task_plain.flac")).waveform
        lo, hi = sample_bounds(outcome.extent.start_s, outcome.extent.end_s, PLAIN_HZ, plain.shape[-1])
        assert torch.allclose(got, plain[..., lo:hi], atol=1e-4)

    def test_the_redacted_cut_carries_exactly_the_masks_inside_it(self, tmp_path: Path, config: TriageConfig) -> None:
        """The mask inside the extent is silent on the same samples; the one outside it is gone."""
        store, run_dir = _seed(tmp_path)
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.extent is not None
        got = Audio(filepath=str(run_dir / "streams/task_redacted.flac")).waveform
        lo, hi = sample_bounds(
            outcome.extent.start_s, outcome.extent.end_s, RECORDING_HZ, int(DURATION_S * RECORDING_HZ)
        )
        inside = masks_in_cut([OUTSIDE, MASK], lo, hi, RECORDING_HZ)
        assert [m["category"] for m in inside] == ["PERSON"]
        a, b = inside[0]["cut_samples"]
        assert a == int(MASK.start * RECORDING_HZ) - lo and b == math.ceil(MASK.end * RECORDING_HZ) - lo
        assert torch.all(got[..., a:b] == 0)
        assert torch.all(got[..., a - 1] != 0) and torch.all(got[..., b] != 0)
        zero = (got == 0).all(dim=0)
        assert int(zero.sum()) == b - a, "silence anywhere but the mask means a mask moved or one leaked in"
        entity = next(e for e in store.entities("stream") if e.attributes.get("name") == "task_redacted")
        assert entity.attributes["masks_verified"] is True
        assert entity.attributes["masks"] == inside

    def test_the_sidecar_and_measurement_record_the_extent(self, tmp_path: Path, config: TriageConfig) -> None:
        """Which definition, the pad, the bounds, and each cut's source and digest."""
        store, run_dir = _seed(tmp_path)
        cut_task_audio(store, config, run_dir=run_dir)
        sidecar = json.loads((run_dir / SIDECAR).read_text())
        assert sidecar["extent"]["definition"] == DEFINITION
        assert sidecar["extent"]["padding_s"] == config.require("task_audio.padding_s")
        for name in ("plain", "enhanced", "redacted"):
            assert len(sidecar["cuts"][name]["checksum_sha256"]) == 64
            assert len(sidecar["cuts"][name]["source_sha256"]) == 64
        measurement = next(e for e in store.entities("measurement") if e.attributes.get("name") == MEASUREMENT)
        assert measurement.attributes["cuts"] == ["enhanced", "plain", "redacted"]

    def test_each_cut_is_derived_from_its_source_and_the_extent(self, tmp_path: Path, config: TriageConfig) -> None:
        """Provenance names the stream cut and the measurement that says where."""
        store, run_dir = _seed(tmp_path)
        cut_task_audio(store, config, run_dir=run_dir)
        by_name = {e.attributes.get("name"): e.id for e in store.entities()}
        for name, source in (("plain", "plain"), ("enhanced", "enhanced"), ("redacted", "redacted")):
            parents = set(store.derived_from(by_name[f"task_{name}"]))
            assert {by_name[source], by_name[MEASUREMENT]} <= parents


class TestWhatIsMissing:
    """A stream that does not exist is not cut, and says why."""

    def test_no_enhanced_stream(self, tmp_path: Path, config: TriageConfig) -> None:
        """Plain and redacted are cut; enhanced is absent."""
        store, run_dir = _seed(tmp_path, enhanced=False)
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.cuts["plain"] == WRITTEN and outcome.cuts["redacted"] == WRITTEN
        assert outcome.cuts["enhanced"].startswith("absent")
        assert not (run_dir / "streams/task_enhanced.flac").exists()

    @pytest.mark.parametrize(
        "release", [Release.WITHOUT_REDACTION.value, Release.WITHHELD.value, Release.NOT_ASSESSED.value]
    )
    def test_no_redacted_cut_unless_the_redacted_copy_is_released(
        self, release: str, tmp_path: Path, config: TriageConfig
    ) -> None:
        """REDACT's stream exists, but the fold releases no masked copy, so none is cut."""
        store, run_dir = _seed(tmp_path, release=release)
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.cuts["redacted"].startswith("absent")
        assert outcome.cuts["plain"] == WRITTEN
        assert not (run_dir / "streams/task_redacted.flac").exists()

    def test_no_task_extent_cuts_nothing(self, tmp_path: Path, config: TriageConfig) -> None:
        """No hull, no cut, no sidecar."""
        store, run_dir = _seed(tmp_path, tasks=())
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.extent is None and not outcome.changed
        assert all(value.startswith("absent") for value in outcome.cuts.values())
        assert not (run_dir / SIDECAR).exists()

    def test_a_release_that_moves_retires_the_redacted_cut(self, tmp_path: Path, config: TriageConfig) -> None:
        """A re-fold to withheld leaves no redacted cut standing, in the store or on disk."""
        store, run_dir = _seed(tmp_path)
        cut_task_audio(store, config, run_dir=run_dir)
        store.entity(
            prov_type="verdict", extent=None, attributes={"node": "VERDICT", "release": Release.WITHHELD.value}
        )
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.cuts["redacted"].startswith("absent") and outcome.cuts["plain"] == PRESENT
        live = [
            e
            for e in store.entities("stream")
            if e.attributes.get("name") == "task_redacted" and not store.is_invalidated(e.id)
        ]
        assert live == []
        assert not (run_dir / "streams/task_redacted.flac").exists()


class TestRerun:
    """A rerun over the same inputs is a cache hit."""

    def test_a_second_pass_writes_nothing(self, tmp_path: Path, config: TriageConfig) -> None:
        """Same store fingerprint, same files, every cut present."""
        store, run_dir = _seed(tmp_path)
        cut_task_audio(store, config, run_dir=run_dir)
        before = store.fingerprint()
        mtimes = {p.name: p.stat().st_mtime_ns for p in (run_dir / "streams").glob("task_*.flac")}
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.cuts == {"plain": PRESENT, "enhanced": PRESENT, "redacted": PRESENT}
        assert not outcome.changed and store.fingerprint() == before
        assert {p.name: p.stat().st_mtime_ns for p in (run_dir / "streams").glob("task_*.flac")} == mtimes

    def test_a_moved_extent_recuts(self, tmp_path: Path, config: TriageConfig) -> None:
        """A new task span changes the extent, so every cut is written again and the old ones retired."""
        store, run_dir = _seed(tmp_path)
        cut_task_audio(store, config, run_dir=run_dir)
        _task(store, 8.0, 9.0)
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.cuts == {"plain": WRITTEN, "enhanced": WRITTEN, "redacted": WRITTEN}
        assert outcome.extent is not None and outcome.extent.end_s == 9.25
        live = [
            e
            for e in store.entities("stream")
            if str(e.attributes.get("name")).startswith("task_") and not store.is_invalidated(e.id)
        ]
        assert len(live) == 3

    def test_a_changed_file_on_disk_is_recut(self, tmp_path: Path, config: TriageConfig) -> None:
        """A cut whose bytes no longer match its recorded digest is not trusted."""
        store, run_dir = _seed(tmp_path)
        cut_task_audio(store, config, run_dir=run_dir)
        (run_dir / "streams/task_plain.flac").write_bytes(b"not audio")
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.cuts["plain"] == WRITTEN and outcome.cuts["enhanced"] == PRESENT

    def test_a_recut_over_the_same_extent_keeps_a_live_measurement(self, tmp_path: Path, config: TriageConfig) -> None:
        """The recut retires the standing measurement and writes an identical one, which must be live."""
        store, run_dir = _seed(tmp_path)
        cut_task_audio(store, config, run_dir=run_dir)
        before = find_measurement(store, MEASUREMENT)
        assert before is not None
        (run_dir / "streams/task_redacted.flac").write_bytes(b"not audio")
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.cuts["redacted"] == WRITTEN
        after = find_measurement(store, MEASUREMENT)
        assert after is not None and after.id != before.id
        assert after.attributes["cuts"] == before.attributes["cuts"]


class TestARemaskedRelease:
    """Where the fold's final masks are not REDACT's own, the released copy is the source masked again."""

    def test_the_cut_carries_the_final_masks_not_the_planned_ones(self, tmp_path: Path, config: TriageConfig) -> None:
        """The ledger widens the mask; the cut is silent over the wider range and derives from the ledger."""
        store, run_dir = _seed(tmp_path)
        wide = RedactionExtent(start=4.0, end=5.5, category="PERSON")
        ledger = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "pii_ledger",
                "signal": "recording",
                "final_masks": [{"start_s": wide.start, "end_s": wide.end, "category": wide.category}],
            },
        )
        outcome = cut_task_audio(store, config, run_dir=run_dir)
        assert outcome.extent is not None
        entity = next(e for e in store.entities("stream") if e.attributes.get("name") == "task_redacted")
        assert entity.attributes["remasked"] is True
        assert ledger in set(store.derived_from(entity.id))
        got = Audio(filepath=str(run_dir / "streams/task_redacted.flac")).waveform
        lo, hi = sample_bounds(
            outcome.extent.start_s, outcome.extent.end_s, RECORDING_HZ, int(DURATION_S * RECORDING_HZ)
        )
        [inside] = masks_in_cut([wide], lo, hi, RECORDING_HZ)
        assert entity.attributes["masks"] == [inside]
        a, b = inside["cut_samples"]
        assert int((got == 0).all(dim=0).sum()) == b - a
        assert torch.all(got[..., a:b] == 0)
