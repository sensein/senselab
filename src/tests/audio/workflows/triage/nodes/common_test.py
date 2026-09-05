"""The shared node helpers: verdict-key shadowing, and the latest-non-invalidated read rule."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.nodes.common import (
    clamp_extent,
    find_measurement,
    resolve_stream,
    software_agent,
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
