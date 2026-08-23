"""ADMIT rejects only decode failure, all-zero and constant. No thresholds, no flag, no models."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.admit import AdmitResult, admit
from senselab.audio.workflows.triage.nodes.common import resolve_stream
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


def _sine(duration_s: float = 1.0, amplitude: float = 0.5, sampling_rate: int = 16000) -> np.ndarray:
    """A mono sine fixture."""
    t = np.arange(int(duration_s * sampling_rate)) / sampling_rate
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


class TestRejections:
    """The three degenerate conditions fail, without exceptions escaping."""

    def test_a_file_that_does_not_decode_fails(self, store: ProvStore, config: TriageConfig, tmp_path: Path) -> None:
        """A text file with a .wav name is a decode failure, not a crash."""
        path = tmp_path / "not_audio.wav"
        path.write_text("this is not a wav file")
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "decode" in result.verdict.why
        assert result.audio is None

    def test_a_missing_file_fails_rather_than_raising(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """A path that does not exist is a decode failure."""
        result = admit(store, tmp_path / "absent.wav", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL

    def test_all_zero_fails(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Every sample exactly zero is unmeasurable."""
        path = wav_writer("zeros.wav", np.zeros(16000, dtype=np.float32))
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "zero" in result.verdict.why

    def test_constant_dc_fails(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A constant nonzero value has no variance and is unmeasurable."""
        path = wav_writer("dc.wav", np.full(16000, 0.25, dtype=np.float32))
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "constant" in result.verdict.why

    def test_zero_frames_fails(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A zero-frame file fails, whether the decoder raises or returns nothing."""
        path = wav_writer("empty.wav", np.zeros(0, dtype=np.float32))
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL


class TestAdmission:
    """Everything non-degenerate passes; there is no level threshold and no flag."""

    def test_a_sine_passes_and_returns_the_decoded_audio(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The pass port carries the decoded audio."""
        path = wav_writer("sine.wav", _sine())
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.audio is not None
        assert result.audio.sampling_rate == 16000

    def test_a_very_quiet_recording_passes_because_there_is_no_level_threshold(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Quiet is not empty: room-tone-level signal is admitted."""
        path = wav_writer("quiet.wav", _sine(amplitude=1e-4))
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS

    def test_the_admitted_audio_is_the_recording_as_supplied(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """No resampling and no channel reduction happen here: 48 kHz stereo stays 48 kHz stereo."""
        stereo = np.stack([_sine(sampling_rate=48000), _sine(sampling_rate=48000)], axis=1)
        path = wav_writer("stereo48k.wav", stereo, sampling_rate=48000)
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.audio is not None
        assert result.audio.sampling_rate == 48000
        assert result.audio.waveform.shape[0] == 2

    def test_admit_never_flags(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The outcome vocabulary is pass or fail; flag does not exist for ADMIT."""
        fixtures = [
            wav_writer("a.wav", _sine()),
            wav_writer("b.wav", np.zeros(16000, dtype=np.float32)),
            wav_writer("c.wav", _sine(amplitude=1e-4)),
            tmp_path / "missing.wav",
        ]
        for path in fixtures:
            result = admit(ProvStore(run_id=f"never-flag-{path.name}"), path, config, run_dir=tmp_path)
            assert result.verdict.outcome in (Outcome.PASS, Outcome.FAIL)


class TestStoreWrites:
    """What ADMIT writes to the store, and what it does not."""

    def test_pass_writes_a_recording_stream_with_provenance(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The recording enters the store as a stream entity, generated and attributed."""
        path = wav_writer("sine.wav", _sine())
        result = admit(store, path, config, run_dir=tmp_path)
        [stream] = store.entities("stream")
        assert stream.attributes["name"] == "recording"
        assert stream.attributes["sampling_rate"] == 16000
        assert stream.attributes["channels"] == 1
        assert Path(stream.attributes["path"]).is_absolute()
        assert store.generated_by(stream.id) is not None
        assert stream.id in result.view
        entity_id, audio = resolve_stream(store, tmp_path, "recording")
        assert entity_id == stream.id
        assert audio.waveform.shape[-1] == 16000

    def test_fail_writes_only_a_verdict(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A rejected file leaves no stream behind — nothing else is claimed about it."""
        path = wav_writer("zeros.wav", np.zeros(16000, dtype=np.float32))
        admit(store, path, config, run_dir=tmp_path)
        assert store.entities("stream") == []
        [verdict] = store.entities("verdict")
        assert verdict.attributes["outcome"] == "fail"

    def test_the_verdict_entity_names_the_node_and_outcome(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """VERDICT reads verdict entities; theirs is the shape that must hold."""
        path = wav_writer("sine.wav", _sine())
        result = admit(store, path, config, run_dir=tmp_path)
        verdict = store.get_entity(result.verdict_entity_id)
        assert verdict.prov_type == "verdict"
        assert verdict.attributes["node"] == "ADMIT"
        assert verdict.attributes["outcome"] == "pass"
        [agent_id] = store.associated_with(store.generated_by(result.verdict_entity_id) or "")
        assert store.get_agent(agent_id).agent_type == "software"

    def test_resolve_stream_raises_on_an_unknown_name(self, store: ProvStore, tmp_path: Path) -> None:
        """A missing stream is a LookupError naming the stream, not a silent None."""
        with pytest.raises(LookupError, match="plain"):
            resolve_stream(store, tmp_path, "plain")
