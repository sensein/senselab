"""The shared node helpers: verdict-key shadowing, and the latest-non-invalidated read rule."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.nodes.common import (
    STREAM_SUFFIX,
    clamp_extent,
    consensus_words,
    find_measurement,
    lexical_words,
    resolve_stream,
    software_agent,
    write_stream,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


class TestWriteVerdict:
    """The reserved verdict keys cannot be shadowed by detail."""

    def test_a_reserved_key_in_detail_raises_naming_it(self, store: ProvStore) -> None:
        """A detail dict carrying a reserved key is a ValueError naming the offender, not a silent overwrite."""
        activity_id = store.activity(node="TEST", step=None, parameters={})
        agent_id = software_agent(store)
        with pytest.raises(ValueError, match="kind"):
            write_verdict(
                store,
                activity_id,
                agent_id,
                node="TEST",
                outcome=Outcome.PASS,
                kind=None,
                why="testing",
                detail={"kind": "airway"},
            )

    def test_a_key_that_merely_resembles_a_reserved_one_is_allowed(self, store: ProvStore) -> None:
        """`kinds` (plural) is not reserved and lands in the stored attributes."""
        activity_id = store.activity(node="TEST", step=None, parameters={})
        agent_id = software_agent(store)
        entity_id, _ = write_verdict(
            store,
            activity_id,
            agent_id,
            node="TEST",
            outcome=Outcome.PASS,
            kind=None,
            why="testing",
            detail={"kinds": {"speech": "present"}},
        )
        assert store.get_entity(entity_id).attributes["kinds"] == {"speech": "present"}


class TestFindMeasurement:
    """find_measurement returns the latest non-invalidated match, or None."""

    def test_returns_none_when_nothing_carries_the_name(self, store: ProvStore) -> None:
        """An empty store yields None, not an error."""
        assert find_measurement(store, "hnr") is None

    def test_returns_the_entity_carrying_the_name(self, store: ProvStore) -> None:
        """A single matching measurement is found by its name attribute."""
        entity_id = store.entity(prov_type="measurement", extent=None, attributes={"name": "hnr", "value": 1.0})
        found = find_measurement(store, "hnr")
        assert found is not None
        assert found.id == entity_id

    def test_skips_an_invalidated_measurement(self, store: ProvStore) -> None:
        """When the latest match is invalidated, the earlier surviving one wins."""
        first = store.entity(prov_type="measurement", extent=None, attributes={"name": "hnr", "value": 1.0})
        second = store.entity(prov_type="measurement", extent=None, attributes={"name": "hnr", "value": 2.0})
        activity_id = store.activity(node="TEST", step=None, parameters={})
        store.was_invalidated_by(second, activity_id)
        found = find_measurement(store, "hnr")
        assert found is not None
        assert found.id == first


class TestResolveStream:
    """resolve_stream reads the latest non-invalidated stream and resolves relative sidecars."""

    def test_resolves_a_relative_sidecar_path_against_run_dir(
        self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A relative stored path is joined onto run_dir before loading."""
        wav_writer("sidecar.wav", np.zeros(16000, dtype=np.float32))
        entity_id = store.entity(
            prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "sidecar.wav"}
        )
        found_id, audio = resolve_stream(store, tmp_path, "recording")
        assert found_id == entity_id
        assert audio.waveform.shape[-1] == 16000

    def test_the_latest_stream_wins(self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]) -> None:
        """Two live streams under one name resolve to the later write."""
        wav_writer("first.wav", np.zeros(16000, dtype=np.float32))
        wav_writer("second.wav", np.zeros(8000, dtype=np.float32))
        store.entity(prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "first.wav"})
        second = store.entity(
            prov_type="stream", extent=(0.0, 0.5), attributes={"name": "recording", "path": "second.wav"}
        )
        found_id, audio = resolve_stream(store, tmp_path, "recording")
        assert found_id == second
        assert audio.waveform.shape[-1] == 8000

    def test_skips_an_invalidated_stream(
        self, store: ProvStore, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """An invalidated stream is no longer read as what it was; the surviving one is returned."""
        wav_writer("first.wav", np.zeros(16000, dtype=np.float32))
        wav_writer("second.wav", np.zeros(8000, dtype=np.float32))
        first = store.entity(
            prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "first.wav"}
        )
        second = store.entity(
            prov_type="stream", extent=(0.0, 0.5), attributes={"name": "recording", "path": "second.wav"}
        )
        activity_id = store.activity(node="TEST", step=None, parameters={})
        store.was_invalidated_by(first, activity_id)
        found_id, _ = resolve_stream(store, tmp_path, "recording")
        assert found_id == second


class TestWriteStream:
    """write_stream persists a stream as FLAC and never lets a write clip."""

    def test_writes_flac_and_survives_round_trip(self, tmp_path: Path) -> None:
        """A stream within range round-trips with the classifier-relevant content intact."""
        rng = np.random.default_rng(0)
        samples = (rng.standard_normal((1, 16000)) * 0.2).astype(np.float32)
        audio = Audio(waveform=samples, sampling_rate=16000)
        relative, report = write_stream(audio, tmp_path, "plain")
        assert relative == f"streams/plain{STREAM_SUFFIX}"
        written = tmp_path / relative
        assert written.is_file()
        assert report.gain == 1.0
        reloaded = Audio(filepath=str(written))
        assert reloaded.sampling_rate == 16000
        assert reloaded.waveform.shape == audio.waveform.shape
        # FLAC/PCM_24's quantization floor is far below anything a classifier responds to.
        assert float((reloaded.waveform - audio.waveform).abs().max()) < 1e-4

    def test_resolve_stream_reads_back_what_write_stream_wrote(self, store: ProvStore, tmp_path: Path) -> None:
        """The store round trip: write_stream's path resolves through resolve_stream unchanged."""
        samples = np.full((1, 8000), 0.3, dtype=np.float32)
        audio = Audio(waveform=samples, sampling_rate=16000)
        relative, _ = write_stream(audio, tmp_path, "residual")
        store.entity(prov_type="stream", extent=(0.0, 0.5), attributes={"name": "residual", "path": relative})
        _, reloaded = resolve_stream(store, tmp_path, "residual")
        assert reloaded.waveform.shape[-1] == 8000
        assert float((reloaded.waveform - audio.waveform).abs().max()) < 1e-4

    def test_a_peak_past_unit_range_is_scaled_down_not_clipped(self, tmp_path: Path) -> None:
        """Values a write would clip are scaled to fit instead, and the gain is reported."""
        samples = np.zeros((1, 16000), dtype=np.float32)
        samples[0, 0] = 1.4  # past +-1, as a real residual's peak can be
        audio = Audio(waveform=samples, sampling_rate=16000)
        relative, report = write_stream(audio, tmp_path, "residual")
        assert report.gain == pytest.approx(1.0 / 1.4, rel=1e-6)
        reloaded = Audio(filepath=str(tmp_path / relative))
        assert float(reloaded.waveform.abs().max()) <= 1.0
        # Undoing the recorded gain recovers the original peak, not a truncated one.
        assert float(reloaded.waveform.abs().max()) / report.gain == pytest.approx(1.4, rel=1e-3)

    def test_a_peak_already_at_full_scale_passes_through_unscaled(self, tmp_path: Path) -> None:
        """Clipping already present in the recording is content, not a write-time hazard: no rescale."""
        samples = np.zeros((1, 16000), dtype=np.float32)
        samples[0, 100:200] = 1.0
        samples[0, 300:400] = -1.0
        audio = Audio(waveform=samples, sampling_rate=16000)
        relative, report = write_stream(audio, tmp_path, "residual")
        assert report.gain == 1.0
        reloaded = Audio(filepath=str(tmp_path / relative))
        assert float(reloaded.waveform[0, 100:200].abs().min()) > 0.999
        assert float(reloaded.waveform[0, 300:400].max()) < -0.999


class TestConsensusWords:
    """The one order is ``index``; the lexical subset drops the bracketed words and nothing else."""

    @staticmethod
    def _word(store: ProvStore, text: str, index: int, extent: tuple[float, float]) -> str:
        """One seeded word carrying only what these readers consult."""
        bracketed = text.startswith("[") and text.endswith("]")
        return store.entity(
            prov_type="word", extent=extent, attributes={"text": text, "index": index, "bracketed": bracketed}
        )

    def test_words_come_back_in_index_order_whatever_their_extents_say(self, store: ProvStore) -> None:
        """A time order read back as a sequence is the defect C-7 names; the reader never does it."""
        self._word(store, "late", 2, (0.0, 0.1))
        self._word(store, "early", 0, (5.0, 5.1))
        self._word(store, "middle", 1, (9.0, 9.1))
        assert [w.attributes["text"] for w in consensus_words(store)] == ["early", "middle", "late"]

    def test_an_invalidated_word_is_not_read(self, store: ProvStore) -> None:
        """The store's shared read rule."""
        gone = self._word(store, "gone", 0, (0.0, 0.1))
        self._word(store, "kept", 1, (0.2, 0.3))
        store.was_invalidated_by(gone, store.activity(node="TEST", step=None, parameters={}))
        assert [w.attributes["text"] for w in consensus_words(store)] == ["kept"]

    def test_lexical_words_drop_the_bracketed_ones_and_keep_the_order(self, store: ProvStore) -> None:
        """[UM] is in the stream and not in the lexical subset."""
        self._word(store, "I", 0, (0.0, 0.1))
        self._word(store, "[UM]", 1, (0.2, 0.3))
        self._word(store, "think", 2, (0.4, 0.5))
        assert [w.attributes["text"] for w in consensus_words(store)] == ["I", "[UM]", "think"]
        assert [w.attributes["text"] for w in lexical_words(store)] == ["I", "think"]


class TestClampExtent:
    """A slice end past the decoded audio is float noise or an inconsistency, and the two differ."""

    @staticmethod
    def _audio(sampling_rate: int = 16000, seconds: float = 1.0) -> Audio:
        """Silence of an exact whole number of samples, so its duration is exactly representable."""
        samples = np.zeros((1, int(seconds * sampling_rate)), dtype=np.float32)
        return Audio(waveform=samples, sampling_rate=sampling_rate)

    def test_an_extent_inside_the_audio_is_returned_unchanged(self) -> None:
        """The common case must not be perturbed by the clamp."""
        assert clamp_extent((0.25, 0.75), self._audio()) == (0.25, 0.75)

    def test_an_extent_ending_exactly_at_the_duration_is_returned_unchanged(self) -> None:
        """The boundary itself is inside, so nothing is clamped and nothing is raised."""
        assert clamp_extent((0.0, 1.0), self._audio()) == (0.0, 1.0)

    def test_half_a_sample_past_the_duration_is_clamped(self) -> None:
        """A word extent may exceed the decode by a float hair; that is a rounding artefact.

        On the cluster (torch 2.11.0+cu130) this hair made ``extract_segments`` raise on a file
        that ran clean locally, taking the whole SPEECH branch with it.
        """
        audio = self._audio()
        clamped = clamp_extent((0.5, 1.0 + 0.5 / audio.sampling_rate), audio)
        assert clamped == (0.5, 1.0)

    def test_a_word_timestamps_own_rounding_step_is_inside_the_tolerance(self) -> None:
        """``fuse_word_streams`` rounds word bounds to 1e-4 s, so the end it reports can overshoot.

        Worst case is half that step, 5e-5 s, which is 0.8 of a sample at 16 kHz — a tolerance of
        half a sample would leave exactly this case raising.
        """
        assert clamp_extent((0.5, 1.0 + 5e-5), self._audio()) == (0.5, 1.0)

    def test_the_tolerance_follows_the_sampling_rate(self) -> None:
        """One sample is a different number of seconds at 8 kHz, and the clamp must say so."""
        overshoot = 0.9 / 8000
        assert clamp_extent((0.5, 1.0 + overshoot), self._audio(sampling_rate=8000)) == (0.5, 1.0)
        with pytest.raises(ValueError, match="past the"):
            clamp_extent((0.5, 1.0 + overshoot), self._audio(sampling_rate=16000))

    def test_more_than_one_sample_past_the_duration_raises(self) -> None:
        """The tolerance is a boundary, not a direction: just past it is refused."""
        audio = self._audio()
        with pytest.raises(ValueError, match="past the"):
            clamp_extent((0.5, 1.0 + 1.5 / audio.sampling_rate), audio)

    def test_a_tenth_of_a_second_past_the_duration_still_raises(self) -> None:
        """That far outside the recording is a real inconsistency, not float noise."""
        with pytest.raises(ValueError, match="past the"):
            clamp_extent((0.5, 1.1), self._audio())

    def test_the_message_names_no_transcript_text(self) -> None:
        """The extent's bounds are safe to log; nothing else about it is."""
        with pytest.raises(ValueError) as raised:
            clamp_extent((0.5, 1.1), self._audio())
        assert "1.1" in str(raised.value) and "1.0" in str(raised.value)
