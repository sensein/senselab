"""This script is for testing the voice cloning API."""

import os

if os.getenv("GITHUB_ACTIONS") != "true":

    import pytest

    from senselab.audio.data_structures.audio import Audio
    from senselab.audio.tasks.voice_cloning.api import clone_voices
    from senselab.utils.data_structures.device import DeviceType
    from senselab.utils.data_structures.model import TorchModel

    @pytest.fixture
    def torch_model() -> TorchModel:
        """Fixture for torch model."""
        return TorchModel(path_or_uri="bshall/knn-vc", revision="master")

    def test_clone_voices_length_mismatch(resampled_mono_audio_sample: Audio, torch_model: TorchModel) -> None:
        """Test length mismatch in source and target audios."""
        source_audios = [resampled_mono_audio_sample]
        target_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]

        with pytest.raises(ValueError, match="Source and target audios must have the same length."):
            clone_voices(
                source_audios=source_audios,
                target_audios=target_audios,
                model=torch_model,
                device=DeviceType.CPU
            )

    def test_clone_voices_invalid_topk(resampled_mono_audio_sample: Audio, torch_model: TorchModel) -> None:
        """Test invalid topk value."""
        source_audios = [resampled_mono_audio_sample]
        target_audios = [resampled_mono_audio_sample]

        with pytest.raises(TypeError, match="argument 'k' must be int, not str"):
            clone_voices(
                source_audios=source_audios,
                target_audios=target_audios,
                model=torch_model,
                device=DeviceType.CPU,
                topk="invalid",  # type: ignore[arg-type]
            )

    def test_clone_voices_invalid_prematched_vocoder(resampled_mono_audio_sample: Audio, 
                                                     torch_model: TorchModel) -> None:
        """Test invalid prematched_vocoder value."""
        source_audios = [resampled_mono_audio_sample]
        target_audios = [resampled_mono_audio_sample]

        with pytest.raises(TypeError, match="prematched_vocoder must be a boolean."):
            clone_voices(
                source_audios=source_audios,
                target_audios=target_audios,
                model=torch_model,
                device=DeviceType.CPU,
                prematched_vocoder="invalid",  # type: ignore[arg-type]
            )

    def test_clone_voices_valid_input(resampled_mono_audio_sample: Audio, torch_model: TorchModel) -> None:
        """Test cloning voices with valid input."""
        source_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]
        target_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]

        try:
            cloned_output = clone_voices(
                source_audios=source_audios,
                target_audios=target_audios,
                model=torch_model,
                device=DeviceType.CPU,
                topk=5,
                prematched_vocoder=False,
            )
            assert isinstance(cloned_output, list), "Output must be a list."
            assert len(cloned_output) == 2, "Output list should contain exactly two audio samples."
            assert isinstance(cloned_output[0], Audio), "Each item in the output list should be an instance of Audio."
            source_duration = source_audios[0].waveform.shape[1]
            cloned_duration = cloned_output[0].waveform.shape[1]

            # Set tolerance to 1% of source duration
            tolerance = 0.01 * source_duration

            # Check if the absolute difference is within the tolerance
            assert abs(source_duration - cloned_duration) <= tolerance, (
                f"Cloned audio duration is not within acceptable range. Source: {source_duration}, "
                f"Cloned: {cloned_duration}"
            )

        except Exception as e:
            pytest.fail(f"An unexpected exception occurred: {e}")

    def test_clone_voices_unsupported_model(resampled_mono_audio_sample: Audio) -> None:
        """Test unsupported model."""
        source_audios = [resampled_mono_audio_sample]
        target_audios = [resampled_mono_audio_sample]
        # this uri doesn't exist
        unsupported_model = TorchModel(path_or_uri="sensein/senselab", revision="main")

        with pytest.raises(NotImplementedError, match="Only KNNVC is supported for now."):
            clone_voices(
                source_audios=source_audios,
                target_audios=target_audios,
                model=unsupported_model,
                device=DeviceType.CPU
            )

    def test_clone_voices_stereo_audio(resampled_stereo_audio_sample: Audio, torch_model: TorchModel) -> None:
        """Test unsupported stereo audio."""
        source_audios = [resampled_stereo_audio_sample]
        target_audios = [resampled_stereo_audio_sample]

        with pytest.raises(ValueError, match="Only mono audio files are supported."):
            clone_voices(
                source_audios=source_audios,
                target_audios=target_audios,
                model=torch_model,
                device=DeviceType.CPU
            )

    def test_clone_voices_invalid_sampling_rate(mono_audio_sample: Audio, torch_model: TorchModel) -> None:
        """Test unsupported sampling rate."""
        source_audios = [mono_audio_sample]
        target_audios = [mono_audio_sample]

        with pytest.raises(ValueError, match="Only 16000 sampling rate is supported."):
            clone_voices(
                source_audios=source_audios,
                target_audios=target_audios,
                model=torch_model,
                device=DeviceType.CPU
            )
