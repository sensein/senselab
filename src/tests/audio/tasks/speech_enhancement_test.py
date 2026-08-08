"""Tests for the speech enhancement task."""

from typing import List

import pytest
from speechbrain.inference.separation import SepformerSeparation as separator

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_enhancement import driftse, enhance_audios
from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
from senselab.utils.data_structures import DeviceType, HFModel, SpeechBrainModel


def test_upstream_is_pinned_to_a_full_commit_sha() -> None:
    """Assert the DriftSE upstream pin is a full 40-char commit SHA.

    A branch name or short SHA would let an upstream force-push change what
    this backend runs without any change here. The repository is unlicensed and
    unpackaged, so the pin is the only version contract available.
    """
    assert len(driftse._DRIFTSE_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in driftse._DRIFTSE_COMMIT)


def test_training_and_metric_dependencies_are_not_installed() -> None:
    """Assert the DriftSE venv requirements omit training/metric-only packages.

    Upstream's requirements.txt lists these for training and scoring. The
    inference path imports none of them, and pesq/scoreq in particular are slow
    and fragile to build. util/inference.py imports pesq — the worker must never
    import util.inference.
    """
    excluded = {
        "pesq",
        "pystoi",
        "scoreq",
        "torch-pesq",
        "asteroid-filterbanks",
        "wandb",
        "pytorch-optimizer",
        "torchinfo",
    }
    named = {r.split(">=")[0].split("==")[0].strip().lower() for r in driftse._DRIFTSE_REQUIREMENTS}
    assert not (named & excluded), f"training-only deps in the inference venv: {named & excluded}"


def test_torch_is_named_explicitly_so_ensure_venv_routes_cuda() -> None:
    """Assert torch and torchaudio are named explicitly in the DriftSE requirements.

    ensure_venv's CUDA auto-detection triggers on an explicit torch pin. Left
    transitive, the resolve skips CUDA-aware routing and can land a CPU-only
    wheel on a GPU host.
    """
    named = {r.split(">=")[0].split("==")[0].strip().lower() for r in driftse._DRIFTSE_REQUIREMENTS}
    assert "torch" in named
    assert "torchaudio" in named


def test_worker_script_compiles_standalone() -> None:
    """Assert the worker script string is syntactically valid standalone Python.

    The worker is a string literal executed by another interpreter, so a
    syntax error in it surfaces only at first inference — after the venv build
    and the model download. Compiling it here makes that a unit-test failure.
    """
    compile(driftse._WORKER_SCRIPT, "<driftse worker>", "exec")


def test_worker_never_imports_util_inference() -> None:
    """Assert the worker never imports upstream's util/inference.py.

    util/inference.py imports pesq and pystoi, which are deliberately not in
    the venv. enhancement.py does not import it and neither may the worker.
    """
    assert "util.inference" not in driftse._WORKER_SCRIPT
    assert "from util import inference" not in driftse._WORKER_SCRIPT


def test_worker_loads_the_checkpoint_with_weights_only() -> None:
    """Assert the worker loads the checkpoint with weights_only=True.

    Upstream omits weights_only. The checkpoint is a foreign pickle from an
    unlicensed research repository; loading it with the unrestricted unpickler is
    arbitrary code execution at enhancement time.
    """
    assert "weights_only=True" in driftse._WORKER_SCRIPT


def test_empty_input_returns_empty_without_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert an empty audio list returns [] without touching the Hub or a venv.

    Constructing the HFModel below would otherwise perform a real Hub
    existence check against a private repo; that check is mocked here per
    this project's rule against unmocked HFModel construction in tests.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    assert driftse.enhance_audios_with_driftse([], model=HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)) == []


@pytest.fixture
def speechbrain_model1() -> SpeechBrainModel:
    """Fixture for Hugging Face model."""
    return SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main")


@pytest.fixture
def speechbrain_model2() -> SpeechBrainModel:
    """Fixture for SpeechBrain model."""
    return SpeechBrainModel(path_or_uri="speechbrain/metricgan-plus-voicebank", revision="main")


@pytest.fixture
def speechbrain_model(request: pytest.FixtureRequest) -> SpeechBrainModel:
    """Fixture that dynamically returns test a SpeechBrain model."""
    return request.getfixturevalue(request.param)


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    """Fixture for clearing the cached models between pytest runs."""
    SpeechBrainEnhancer._models = {}


@pytest.mark.parametrize("speechbrain_model", ["speechbrain_model1"], indirect=True)
def test_enhance_audios_stereo_audio(
    resampled_stereo_audio_sample: Audio, speechbrain_model: SpeechBrainModel, cpu_cuda_device: DeviceType
) -> None:
    """Test that enhancing stereo audios raises a ValueError."""
    with pytest.raises(ValueError, match="Audio waveform must be mono"):
        SpeechBrainEnhancer.enhance_audios_with_speechbrain(
            audios=[resampled_stereo_audio_sample], model=speechbrain_model, device=cpu_cuda_device
        )


@pytest.mark.parametrize("speechbrain_model", ["speechbrain_model1", "speechbrain_model2"], indirect=True)
def test_enhance_audios(
    resampled_mono_audio_sample: Audio,
    resampled_mono_audio_sample_x2: Audio,
    speechbrain_model: SpeechBrainModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Test enhancing audios."""
    enhanced_audios = enhance_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2],
        model=speechbrain_model,
        device=cpu_cuda_device,
    )
    assert len(enhanced_audios) == 2
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].waveform.shape == resampled_mono_audio_sample.waveform.shape


@pytest.mark.parametrize("speechbrain_model", ["speechbrain_model1"], indirect=True)
def test_speechbrain_enhancer_get_model(speechbrain_model: SpeechBrainModel, cpu_cuda_device: DeviceType) -> None:
    """Test getting SpeechBrain model."""
    # TODO: add tests like these but with multithreading
    model, _, _ = SpeechBrainEnhancer._get_speechbrain_model(model=speechbrain_model, device=cpu_cuda_device)
    assert model is not None
    assert isinstance(model, separator)
    assert (
        model
        == SpeechBrainEnhancer._models[
            f"{speechbrain_model.path_or_uri}-{speechbrain_model.revision}-{cpu_cuda_device.value}"
        ]
    )


@pytest.mark.parametrize("speechbrain_model", ["speechbrain_model1", "speechbrain_model2"], indirect=True)
def test_enhance_audios_with_speechbrain(
    resampled_mono_audio_sample: Audio,
    resampled_mono_audio_sample_x2: Audio,
    speechbrain_model: SpeechBrainModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Test enhancing audios with SpeechBrain."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2],
        model=speechbrain_model,
        device=cpu_cuda_device,
    )
    assert len(enhanced_audios) == 2
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].waveform.shape == resampled_mono_audio_sample.waveform.shape
    assert enhanced_audios[1].waveform.shape == resampled_mono_audio_sample_x2.waveform.shape


@pytest.mark.parametrize(
    "speechbrain_model",
    ["speechbrain_model1"],
    indirect=True,
)
def test_enhance_audios_incorrect_sampling_rate(
    mono_audio_sample: Audio, speechbrain_model: SpeechBrainModel, cpu_cuda_device: DeviceType
) -> None:
    """Test enhancing audios with incorrect sampling rate."""
    new_audio = Audio(waveform=mono_audio_sample.waveform, sampling_rate=8000)  # Incorrect sample rate for this model
    with pytest.raises(ValueError, match="Audio sampling rate 8000 does not match expected 16000"):
        SpeechBrainEnhancer.enhance_audios_with_speechbrain(
            audios=[new_audio], model=speechbrain_model, device=cpu_cuda_device
        )


def test_enhance_audios_with_different_bit_depths(
    audio_with_different_bit_depths: List[Audio], cpu_cuda_device: DeviceType
) -> None:
    """Test enhancing audios with different bit depths."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(
        audios=audio_with_different_bit_depths, device=cpu_cuda_device
    )
    assert len(enhanced_audios) == 2
    for audio in enhanced_audios:
        assert isinstance(audio, Audio)
        assert audio.waveform.shape == audio_with_different_bit_depths[0].waveform.shape


def test_enhance_audios_with_metadata(audio_with_metadata: Audio, cpu_cuda_device: DeviceType) -> None:
    """Test enhancing audios with metadata."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(
        audios=[audio_with_metadata], device=cpu_cuda_device
    )
    assert len(enhanced_audios) == 1
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].metadata == audio_with_metadata.metadata


def test_enhance_audios_with_extreme_amplitude(
    audio_with_extreme_amplitude: Audio, cpu_cuda_device: DeviceType
) -> None:
    """Test enhancing audios with extreme amplitude values."""
    enhanced_audios = SpeechBrainEnhancer.enhance_audios_with_speechbrain(
        audios=[audio_with_extreme_amplitude], device=cpu_cuda_device
    )
    assert len(enhanced_audios) == 1
    assert isinstance(enhanced_audios[0], Audio)
    assert enhanced_audios[0].waveform.shape == audio_with_extreme_amplitude.waveform.shape


def test_model_caching(resampled_mono_audio_sample: Audio) -> None:
    """Test model caching by enhancing audios with the same model multiple times."""
    SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[resampled_mono_audio_sample], device=DeviceType.CPU)
    assert len(list(SpeechBrainEnhancer._models.keys())) == 1
    SpeechBrainEnhancer.enhance_audios_with_speechbrain(audios=[resampled_mono_audio_sample], device=DeviceType.CPU)
    assert len(list(SpeechBrainEnhancer._models.keys())) == 1
