"""Tests for speaker diarization."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import concatenate_audios
from senselab.audio.tasks.speaker_diarization import api as diarization_api
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_diarization import pyannote as pyannote_module
from senselab.audio.tasks.speaker_diarization.pyannote import PyannoteDiarization, diarize_audios_with_pyannote
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine
from senselab.utils.data_structures.docker import docker_is_running
from senselab.utils.data_structures.model import model_for_task
from senselab.utils.subprocess_venv import _cache_dir_path

if docker_is_running():
    DOCKER_AVAILABLE = True
else:
    DOCKER_AVAILABLE = False

# Honor SENSELAB_VENV_CACHE the same way ensure_venv() does — hardcoding
# Path.home()/".cache"/... here would silently disagree with a cache dir
# override, so the gate below would never match where the venv actually lives.
_CHILD_ADULT_VENV_ROOT = _cache_dir_path() / "child-adult-diarization"
child_adult_venv_present = _CHILD_ADULT_VENV_ROOT.exists()


@pytest.fixture
def pyannote_model() -> PyannoteAudioModel:
    """Fixture for Pyannote model."""
    return PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1")


@pytest.mark.skip(reason="pyannote-audio is a core dependency and always installed in test environment")
def test_pyannote_not_installed(pyannote_model: PyannoteAudioModel) -> None:
    """Test Pyannote not installed."""
    with pytest.raises(ModuleNotFoundError):
        _ = diarize_audios(audios=[Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)], model=pyannote_model)


def test_diarize_audios(
    resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test diarizing audios."""
    results = diarize_audios(audios=[resampled_mono_audio_sample], model=pyannote_model, device=cpu_cuda_device)
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


@pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker is not available",
)
@pytest.mark.skip(reason="This test takes too long, especially on CI")
def test_diarize_audios_with_nvidia_sortformer(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with NVIDIA Sortformer."""
    model: HFModel = HFModel(path_or_uri="nvidia/diar_sortformer_4spk-v1")
    results = diarize_audios(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert results[0], "empty result — the all(...) below would pass vacuously"
    assert all(isinstance(line, ScriptLine) for line in results[0])


def test_diarize_audios_with_pyannote(
    resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test diarizing audios with Pyannote."""
    results = diarize_audios_with_pyannote(
        audios=[resampled_mono_audio_sample], model=pyannote_model, device=cpu_cuda_device, num_speakers=2
    )
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


def test_pyannote_pipeline_factory(pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType) -> None:
    """Test Pyannote pipeline factory."""
    pipeline1 = PyannoteDiarization._get_pyannote_diarization_pipeline(
        model=pyannote_model,
        device=cpu_cuda_device,
    )
    pipeline2 = PyannoteDiarization._get_pyannote_diarization_pipeline(
        model=pyannote_model,
        device=cpu_cuda_device,
    )
    assert pipeline1 is pipeline2  # Check if the same instance is returned


def test_pyannote_pipeline_factory_forwards_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pyannote factory should forward the Hugging Face token to from_pretrained."""
    monkeypatch.setattr(PyannoteDiarization, "_pipelines", {})
    from_pretrained_mock = Mock()
    pipeline_mock = Mock()
    pipeline_mock.to.return_value = pipeline_mock
    from_pretrained_mock.return_value = pipeline_mock

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(pyannote_module.Pipeline, "from_pretrained", from_pretrained_mock)

    PyannoteDiarization._get_pyannote_diarization_pipeline(
        model=PyannoteAudioModel.model_construct(
            path_or_uri="pyannote/speaker-diarization-community-1",
            revision="main",
            info=None,
        ),
        device=DeviceType.CPU,
    )

    assert from_pretrained_mock.call_args.kwargs["token"] == "hf_test_token"


def test_diarize_audios_with_pyannote_invalid_sampling_rate(
    mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test diarizing audios with unsupported sampling_rate."""
    with pytest.raises(ValueError):
        diarize_audios(audios=[mono_audio_sample], model=pyannote_model, device=cpu_cuda_device)


def test_diarize_stereo_audios_with_pyannote_invalid(
    resampled_stereo_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test diarizing audios with unsupported number of channels."""
    with pytest.raises(ValueError):
        diarize_audios(audios=[resampled_stereo_audio_sample], model=pyannote_model, device=cpu_cuda_device)


@pytest.mark.parametrize(
    "model_id",
    [
        "microsoft/VibeVoice-ASR-HF",
        "AlexXu811/whisper-child-adult",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "BUT-FIT/diarizen-wavlm-large-s80-md",
        "nvidia/diar_sortformer_4spk-v1",
    ],
)
def test_model_for_task_resolves_new_diarizers_to_hfmodel(model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The four new backends and Sortformer are HF-hosted, not Pyannote-hosted.

    Resolving them to PyannoteAudioModel would send them through pyannote's
    pipeline loader and fail with an opaque config error rather than dispatching.
    Kept alongside PR #537's dispatch tests below (which start from an already-
    constructed HFModel): this is the only coverage of model.py's prefix table
    itself for these four new backends — model_test.py's routing test only
    covers Sortformer and the Pyannote default.
    """
    # Avoid Hub validation when constructing the HFModel: the real check does a full
    # snapshot_download, not a HEAD request, and would pull multi-GB weights on every
    # cold CI run for a test that only asserts a routing decision.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    assert isinstance(model_for_task(model_id, task="diarization"), HFModel)


def test_vibevoice_prefix_does_not_capture_the_tts_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """`microsoft/VibeVoice-1.5B` is a TTS model, not the ASR diarizer.

    A bare `microsoft/VibeVoice` prefix would route it to
    VibeVoiceAsrForConditionalGeneration.from_pretrained and fail opaquely.
    """
    # Same rationale as above: PyannoteAudioModel is an HFModel subclass, so
    # constructing one for this (deliberately non-matching) id would otherwise
    # trigger a real snapshot_download of the 4.4 GB VibeVoice-1.5B checkpoint.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    assert isinstance(
        model_for_task("microsoft/VibeVoice-1.5B", task="diarization"),
        PyannoteAudioModel,
    )


def test_speaker_hints_warn_when_the_backend_ignores_them(caplog: pytest.LogCaptureFixture) -> None:
    """Only Pyannote honours num_speakers.

    Silently dropping the hint on the other backends makes a misconfigured run
    indistinguishable from a working one.
    """
    import logging

    from senselab.audio.tasks.speaker_diarization.api import _warn_if_speaker_hints_ignored

    with caplog.at_level(logging.WARNING):
        _warn_if_speaker_hints_ignored(
            backend_name="DiariZen",
            num_speakers=2,
            min_speakers=None,
            max_speakers=None,
        )
    assert any("num_speakers" in r.message for r in caplog.records)


_ALL_BACKEND_ATTRS = (
    "diarize_audios_with_nvidia_sortformer",
    "diarize_audios_with_vibevoice",
    "diarize_audios_with_child_adult",
    "diarize_audios_with_moss",
    "diarize_audios_with_diarizen",
)


@pytest.mark.parametrize(
    ("model_id", "backend_attr", "expects_max_new_tokens"),
    [
        ("microsoft/VibeVoice-ASR-HF", "diarize_audios_with_vibevoice", True),
        ("AlexXu811/whisper-child-adult", "diarize_audios_with_child_adult", False),
        ("OpenMOSS-Team/MOSS-Transcribe-Diarize", "diarize_audios_with_moss", True),
        ("BUT-FIT/diarizen-wavlm-large-s80-md", "diarize_audios_with_diarizen", False),
    ],
)
def test_diarize_audios_dispatches_to_the_right_backend(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    backend_attr: str,
    expects_max_new_tokens: bool,
) -> None:
    """Each new prefix must reach its own backend, and no sibling's.

    `model.py`'s prefix list and this module's `elif` chain are two independently
    maintained copies of the same routing table (`model.py` says so itself). Nothing
    short of exercising `diarize_audios` itself would catch e.g. the MOSS prefix
    silently falling into the child-adult branch. Every candidate backend is patched
    so exactly one of them is asserted to have been called; the rest must stay untouched.

    Kept alongside PR #537's per-backend dispatch tests below: those check that the
    right backend fires, this additionally checks that no *sibling* backend fires and
    that the `max_new_tokens` kwarg is forwarded (or omitted) correctly per backend.
    """
    # Avoid Hub validation when constructing the HFModel (see rationale on the
    # model_for_task tests above).
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)

    mocks = {name: Mock(return_value=[[]]) for name in _ALL_BACKEND_ATTRS}
    for name, mock in mocks.items():
        monkeypatch.setattr(diarization_api, name, mock)

    model = HFModel(path_or_uri=model_id)
    audios = [Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)]

    result = diarize_audios(audios=audios, model=model, max_new_tokens=222)

    target_mock = mocks[backend_attr]
    target_mock.assert_called_once()
    assert result == [[]]
    for name, mock in mocks.items():
        if name != backend_attr:
            mock.assert_not_called()

    call_kwargs = target_mock.call_args.kwargs
    assert call_kwargs["model"] is model
    assert call_kwargs["audios"] is audios
    if expects_max_new_tokens:
        assert call_kwargs.get("max_new_tokens") == 222
    else:
        assert "max_new_tokens" not in call_kwargs


def test_diarize_audios_dispatches_to_vibevoice(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios routes a microsoft/VibeVoice* model id to the VibeVoice backend."""
    sentinel = [[ScriptLine(speaker="0", start=0.0, end=1.0, text="hi")]]
    mock_fn = Mock(return_value=sentinel)
    monkeypatch.setattr(diarization_api, "diarize_audios_with_vibevoice", mock_fn)

    # Mock Hub validation independently of any other test's HFModel._hf_cache entry: this
    # test must be safe to run alone (-k, --lf, xdist, a future pytest-randomly), not just
    # in file-definition order behind test_model_for_task_resolves_new_diarizers_to_hfmodel.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    model: HFModel = HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")
    result = diarize_audios(audios=[], model=model)

    assert result is sentinel
    mock_fn.assert_called_once()


def test_diarize_audios_dispatches_to_child_adult(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios routes the AlexXu811/whisper-child-adult model id to the child-adult backend."""
    sentinel = [[ScriptLine(speaker="ADULT", start=0.0, end=1.0)]]
    mock_fn = Mock(return_value=sentinel)
    monkeypatch.setattr(diarization_api, "diarize_audios_with_child_adult", mock_fn)

    # Mock Hub validation independently — see rationale on the VibeVoice dispatch test above.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    model: HFModel = HFModel(path_or_uri="AlexXu811/whisper-child-adult")
    result = diarize_audios(audios=[], model=model)

    assert result is sentinel
    mock_fn.assert_called_once()


def test_diarize_audios_dispatches_to_moss(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios routes an OpenMOSS-Team/MOSS-Transcribe-Diarize model id to the MOSS backend."""
    sentinel = [[ScriptLine(speaker="S01", start=0.0, end=1.0, text="hi")]]
    mock_fn = Mock(return_value=sentinel)
    monkeypatch.setattr(diarization_api, "diarize_audios_with_moss", mock_fn)

    # Mock Hub validation independently — see rationale on the VibeVoice dispatch test above.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    model: HFModel = HFModel(path_or_uri="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    result = diarize_audios(audios=[], model=model)

    assert result is sentinel
    mock_fn.assert_called_once()


def test_diarize_audios_dispatches_to_diarizen(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios routes a BUT-FIT/diarizen model id to the DiariZen backend."""
    sentinel = [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=1.0)]]
    mock_fn = Mock(return_value=sentinel)
    monkeypatch.setattr(diarization_api, "diarize_audios_with_diarizen", mock_fn)

    # Mock Hub validation independently — see rationale on the VibeVoice dispatch test above.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    model: HFModel = HFModel(path_or_uri="BUT-FIT/diarizen-wavlm-large-s80-md")
    result = diarize_audios(audios=[], model=model)

    assert result is sentinel
    mock_fn.assert_called_once()


def test_diarize_audios_with_vibevoice_raises_when_all_segments_unparsable(
    monkeypatch: pytest.MonkeyPatch, resampled_mono_audio_sample: Audio
) -> None:
    """Every audio in the batch failing to parse raises, rather than returning an all-empty result.

    Guards against a future decode()/return_format="parsed" contract break (e.g. an upstream
    transformers revision bump) presenting as silent "no speech detected" that would then get
    cached as a status-ok outcome and persist across runs.
    """
    from senselab.audio.tasks.speaker_diarization import vibevoice as vibevoice_module
    from senselab.audio.tasks.speaker_diarization.vibevoice import diarize_audios_with_vibevoice

    class _FakeModel:
        device = torch.device("cpu")

        def parameters(self) -> Iterator[torch.Tensor]:
            return iter([torch.zeros(1, dtype=torch.float32)])

        def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
            input_ids = kwargs["input_ids"]
            return torch.zeros((1, input_ids.shape[1] + 1), dtype=torch.long)

    class _FakeProcessor:
        def apply_transcription_request(self, audio: str) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}

        def decode(self, generated_ids: torch.Tensor, return_format: str) -> None:
            raise ValueError("simulated decode()/return_format contract break")

    monkeypatch.setattr(
        vibevoice_module.VibeVoiceDiarization,
        "_get_vibevoice_model",
        Mock(return_value=(_FakeProcessor(), _FakeModel())),
    )

    # Mock Hub validation independently — see rationale on the VibeVoice dispatch test above.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    model: HFModel = HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")
    with pytest.raises(RuntimeError, match="failed to parse output for all"):
        diarize_audios_with_vibevoice(audios=[resampled_mono_audio_sample], model=model)


def test_diarize_audios_with_vibevoice_raises_on_all_batch_schema_mismatch(
    monkeypatch: pytest.MonkeyPatch, resampled_mono_audio_sample: Audio
) -> None:
    """A shape/schema mismatch (the realistic upstream break) also counts as a parse failure.

    ``VibeVoiceAsrProcessor.extract_speaker_dict`` doesn't raise on a shape mismatch (renamed
    key, non-numeric Start/End, ...) — it logs and returns the original *string* instead. That
    string fallback, not a raised exception, is how a future transformers revision would
    realistically break this backend, so it must count toward the all-batch-failed check too.
    """
    from senselab.audio.tasks.speaker_diarization import vibevoice as vibevoice_module
    from senselab.audio.tasks.speaker_diarization.vibevoice import diarize_audios_with_vibevoice

    class _FakeModel:
        device = torch.device("cpu")

        def parameters(self) -> Iterator[torch.Tensor]:
            return iter([torch.zeros(1, dtype=torch.float32)])

        def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
            input_ids = kwargs["input_ids"]
            return torch.zeros((1, input_ids.shape[1] + 1), dtype=torch.long)

    class _FakeProcessor:
        def apply_transcription_request(self, audio: str) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.zeros((1, 1), dtype=torch.long)}

        def decode(self, generated_ids: torch.Tensor, return_format: str) -> str:
            # Mirrors extract_speaker_dict()'s own fallback: valid JSON, wrong shape,
            # so it hands back the raw (undecoded) string rather than raising.
            return '[{"start":0.0,"end":1.0,"speaker":0,"text":"hi"}]'

    monkeypatch.setattr(
        vibevoice_module.VibeVoiceDiarization,
        "_get_vibevoice_model",
        Mock(return_value=(_FakeProcessor(), _FakeModel())),
    )

    # Mock Hub validation independently — see rationale on the VibeVoice dispatch test above.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    model: HFModel = HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")
    with pytest.raises(RuntimeError, match="failed to parse output for all"):
        diarize_audios_with_vibevoice(audios=[resampled_mono_audio_sample], model=model)


@pytest.mark.skip(reason="Downloads a 7B model; run manually on a GPU machine")
def test_diarize_audios_with_vibevoice(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with VibeVoice-ASR-HF."""
    from senselab.audio.tasks.speaker_diarization.vibevoice import diarize_audios_with_vibevoice

    model: HFModel = HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")
    results = diarize_audios_with_vibevoice(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert results[0], "empty result — the all(...) below would pass vacuously"
    assert all(isinstance(line, ScriptLine) for line in results[0])


@pytest.mark.skip(reason="Provisions a dedicated venv and downloads a model; run manually")
def test_diarize_audios_with_moss(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with MOSS-Transcribe-Diarize."""
    from senselab.audio.tasks.speaker_diarization.moss import diarize_audios_with_moss

    model: HFModel = HFModel(path_or_uri="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    results = diarize_audios_with_moss(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert results[0], "empty result — the all(...) below would pass vacuously"
    assert all(isinstance(line, ScriptLine) for line in results[0])


@pytest.mark.skip(reason="Provisions a dedicated venv (forked pyannote-audio) and downloads a model; run manually")
def test_diarize_audios_with_diarizen(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with DiariZen."""
    from senselab.audio.tasks.speaker_diarization.diarizen import diarize_audios_with_diarizen

    model: HFModel = HFModel(path_or_uri="BUT-FIT/diarizen-wavlm-large-s80-md")
    results = diarize_audios_with_diarizen(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert results[0], "empty result — the all(...) below would pass vacuously"
    assert all(isinstance(line, ScriptLine) for line in results[0])


@pytest.mark.skipif(
    not child_adult_venv_present or not torch.cuda.is_available(),
    reason=(
        f"child-adult-diarization venv not provisioned at {_CHILD_ADULT_VENV_ROOT}, or no "
        "CUDA available (the backend raises without CUDA rather than falling back to CPU)"
    ),
)
def test_diarize_audios_with_child_adult(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with the USC-SAIL child-adult classifier (requires CUDA)."""
    from senselab.audio.tasks.speaker_diarization.child_adult import diarize_audios_with_child_adult

    # Upstream's own chunking loop only analyzes whole 10s windows (strict
    # `start + 10 < length`), so anything <= 10s produces zero windows and an
    # empty result — indistinguishable from "no adult/child speech detected."
    # Concatenate the ~4.9s fixture 3x (~14.8s) so this test can actually tell
    # a broken backend from a too-short clip.
    long_audio = concatenate_audios([resampled_mono_audio_sample] * 3)
    model: HFModel = HFModel(path_or_uri="AlexXu811/whisper-child-adult")
    results = diarize_audios_with_child_adult(audios=[long_audio], model=model, device=DeviceType.CUDA)
    assert len(results) == 1
    assert results[0], "expected at least one analyzed window for a ~14.8s clip"
    assert all(isinstance(line, ScriptLine) for line in results[0])
    assert all(line.speaker in ("CHILD", "ADULT", "OVERLAP") for line in results[0])


def test_diarize_audios_with_child_adult_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios_with_child_adult raises a clear error when CUDA isn't available/requested."""
    from senselab.audio.tasks.speaker_diarization.child_adult import diarize_audios_with_child_adult

    # Mock Hub validation independently — see rationale on the VibeVoice dispatch test above.
    # This test carries no skip marker (it asserts a CPU-path error, not a real backend run),
    # so it always executes; without this it would only be safe by accident of
    # HFModel._hf_cache already holding this repo from an earlier test in the same session.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    model: HFModel = HFModel(path_or_uri="AlexXu811/whisper-child-adult")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        diarize_audios_with_child_adult(audios=[], model=model, device=DeviceType.CPU)
