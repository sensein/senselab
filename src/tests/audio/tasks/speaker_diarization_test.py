"""Tests for speaker diarization."""

import os

if os.getenv("GITHUB_ACTIONS") != "true":
    import pytest

    from senselab.audio.data_structures.audio import Audio
    from senselab.audio.tasks.speaker_diarization.api import diarize_audios
    from senselab.audio.tasks.speaker_diarization.pyannote import PyannoteDiarization, diarize_audios_with_pyannote
    from senselab.utils.data_structures.device import DeviceType
    from senselab.utils.data_structures.model import PyannoteAudioModel
    from senselab.utils.data_structures.script_line import ScriptLine

    @pytest.fixture
    def pyannote_model() -> PyannoteAudioModel:
        """Fixture for Pyannote model."""
        return PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-3.1")

    def test_diarize_audios(resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel) -> None:
        """Test diarizing audios."""
        results = diarize_audios(audios=[resampled_mono_audio_sample], model=pyannote_model)
        assert len(results) == 1
        assert isinstance(results[0][0], ScriptLine)

    def test_diarize_audios_with_pyannote(
        resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel
    ) -> None:
        """Test diarizing audios with Pyannote."""
        results = diarize_audios_with_pyannote(
            audios=[resampled_mono_audio_sample], model=pyannote_model, device=DeviceType.CPU, num_speakers=2
        )
        assert len(results) == 1
        assert isinstance(results[0][0], ScriptLine)

    def test_pyannote_pipeline_factory(pyannote_model: PyannoteAudioModel) -> None:
        """Test Pyannote pipeline factory."""
        pipeline1 = PyannoteDiarization._get_pyannote_diarization_pipeline(
            model=pyannote_model,
            device=DeviceType.CPU,
        )
        pipeline2 = PyannoteDiarization._get_pyannote_diarization_pipeline(
            model=pyannote_model,
            device=DeviceType.CPU,
        )
        assert pipeline1 is pipeline2  # Check if the same instance is returned

        def test_diarize_audios_with_pyannote_invalid_sampling_rate(
            mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel
        ) -> None:
            """Test diarizing audios with unsupported sampling_rate."""
            with pytest.raises(ValueError):
                diarize_audios(audios=[mono_audio_sample], model=pyannote_model)

        def test_diarize_stereo_audios_with_pyannote_invalid(
            resampled_stereo_audio_sample: Audio, pyannote_model: PyannoteAudioModel
        ) -> None:
            """Test diarizing audios with unsupported number of channels."""
            with pytest.raises(ValueError):
                diarize_audios(audios=[resampled_stereo_audio_sample], model=pyannote_model)
