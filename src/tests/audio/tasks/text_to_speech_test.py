"""Tests for the text to speech task."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import extract_segments, resample_audios
from senselab.audio.tasks.text_to_speech import qwen_tts, synthesize_texts
from senselab.audio.tasks.text_to_speech.huggingface import HuggingFaceTTS
from senselab.audio.tasks.text_to_speech.qwen_tts import supported_speakers, synthesize_texts_with_qwen
from senselab.utils.data_structures import CoquiTTSModel, DeviceType, HFModel, Language, SenselabModel, TorchModel
from senselab.utils.subprocess_venv import _cache_dir_path

# Honor SENSELAB_VENV_CACHE the same way ensure_venv() does — hardcoding
# Path.home()/".cache"/... here would silently disagree with a cache dir
# override, so the gate below would never match where the venv actually lives.
_QWEN_TTS_VENV_ROOT = _cache_dir_path() / "qwen-tts"
qwen_tts_venv_present = _QWEN_TTS_VENV_ROOT.is_dir()


@pytest.fixture
def _offline_hfmodel_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let an ``HFModel`` be constructed without reaching the Hub.

    Unmocked, the ``revision`` validator calls ``check_hf_repo_exists``, which
    downloads the full snapshot -- this once pulled 20 GB for an unrelated model.
    Both the existence check and the commit-SHA resolution are stubbed,
    independently, per this project's rule against unmocked ``HFModel``
    construction in tests.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)


# Coqui TTS synthesis still uses direct TTS import (not subprocess venv yet).
# Guard this test until it's migrated.
try:
    from TTS.api import TTS  # noqa: F401

    TTS_AVAILABLE = True
except ModuleNotFoundError:
    TTS_AVAILABLE = False


@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for the HF model."""
    return HFModel(path_or_uri="suno/bark-small", revision="main")


@pytest.fixture
def hf_model2() -> HFModel:
    """Fixture for HF model."""
    return HFModel(path_or_uri="facebook/mms-tts-eng", revision="main")


@pytest.fixture
def coqui_tts_model() -> CoquiTTSModel:
    """Fixture for Coqui TTS model."""
    return CoquiTTSModel(path_or_uri="tts_models/multilingual/multi-dataset/xtts_v2", revision="main")


def test_synthesize_texts_with_mms_tts(hf_model2: HFModel, cpu_cuda_device: DeviceType) -> None:
    """Test synthesizing texts with mms-tts-eng (Tier 1)."""
    texts = ["Hello world", "Hello world again."]
    audios = synthesize_texts(texts=texts, model=hf_model2, device=cpu_cuda_device)

    assert len(audios) == 2
    assert isinstance(audios[0], Audio)
    assert audios[0].waveform is not None
    assert audios[0].sampling_rate > 0


def test_synthesize_texts_with_bark(hf_model: HFModel, gpu_device: DeviceType) -> None:
    """Test synthesizing texts with bark-small (Tier 3)."""
    texts = ["Hello world", "Hello world again."]
    audios = synthesize_texts(texts=texts, model=hf_model, device=gpu_device)

    assert len(audios) == 2
    assert isinstance(audios[0], Audio)
    assert audios[0].waveform is not None
    assert audios[0].sampling_rate > 0


@pytest.mark.skipif(not TTS_AVAILABLE, reason="Coqui TTS synthesis not yet migrated to subprocess venv")
def test_synthesize_texts_with_coqui_model(coqui_tts_model: CoquiTTSModel, gpu_device: DeviceType) -> None:
    """Test synthesizing texts."""
    texts = ["Hello world", "Hello world again."]
    audios = synthesize_texts(
        texts=texts, model=coqui_tts_model, device=gpu_device, language=Language(language_code="en")
    )

    assert len(audios) == 2
    assert isinstance(audios[0], Audio)
    assert audios[0].waveform is not None
    assert audios[0].sampling_rate > 0


def test_huggingface_tts_pipeline_factory(hf_model: HFModel, cpu_cuda_device: DeviceType) -> None:
    """Test Hugging Face TTS pipeline factory."""
    pipeline1 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=cpu_cuda_device)
    pipeline2 = HuggingFaceTTS._get_hf_tts_pipeline(model=hf_model, device=cpu_cuda_device)

    assert pipeline1 is pipeline2  # Check if the same instance is returned


def test_invalid_model() -> None:
    """Test synthesize_texts with invalid model."""
    texts = ["Hello world"]
    model: SenselabModel = SenselabModel(path_or_uri="-----", revision="main")

    # TODO Texts like these should be stored in a common utils/constants file such that
    # they only need to be changed in one place
    with pytest.raises(
        NotImplementedError, match="Only Hugging Face models and select Torch models are supported for now."
    ):
        synthesize_texts(texts=texts, model=model)


# ---------------------------------------------------------------------------
# Qwen3-TTS backend
# ---------------------------------------------------------------------------


def test_qwen_tts_prefix_dispatches_to_qwen(_offline_hfmodel_construction: None, mono_audio_sample: Audio) -> None:
    """An ``HFModel`` whose id starts with ``Qwen/Qwen3-TTS`` reaches the qwen_tts backend.

    Without the dedicated prefix check this would instead fall into the generic
    HuggingFaceTTS pipeline path (every HFModel matches ``isinstance(model, HFModel)``),
    which does not understand ``generate_custom_voice``'s speaker/instruct kwargs at all.
    """
    with patch(
        "senselab.audio.tasks.text_to_speech.api.synthesize_texts_with_qwen",
        return_value=[mono_audio_sample],
    ) as qt:
        synthesize_texts(
            texts=["Hello world"],
            model=HFModel(path_or_uri="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
            speaker="Ryan",
            instruct="Very happy.",
        )
    qt.assert_called_once()
    assert qt.call_args.kwargs["speaker"] == "Ryan"
    assert qt.call_args.kwargs["instruct"] == "Very happy."


def test_qwen_tts_requirements_pin_the_package_but_float_torch() -> None:
    """``qwen-tts`` is pinned exactly; ``torch``/``torchaudio`` carry a floor, not a pin.

    ``unasdiff``'s exact ``torch==2.6.0`` pin has no ``cu128`` wheel and failed outright on
    an H100 this session -- a floor lets ``ensure_venv``'s CUDA-aware routing pick whatever
    compatible wheel actually exists on the host's index instead.
    """
    pinned = {r.split("==")[0].strip().lower() for r in qwen_tts._QWEN_TTS_REQUIREMENTS if "==" in r}
    floored = {r.split(">=")[0].strip().lower() for r in qwen_tts._QWEN_TTS_REQUIREMENTS if ">=" in r}
    assert "qwen-tts" in pinned
    assert {"torch", "torchaudio"} <= floored
    assert not ({"torch", "torchaudio"} & pinned), "torch/torchaudio must not be exact-pinned (see H100 failure mode)"


def test_worker_script_compiles_standalone() -> None:
    """Assert the worker script string is syntactically valid standalone Python.

    The worker is a string literal executed by another interpreter, so a syntax error in
    it surfaces only at first inference -- after the venv build and the model download.
    Compiling it here makes that a unit-test failure instead.
    """
    compile(qwen_tts._WORKER_SCRIPT, "<qwen_tts worker>", "exec")


def test_flash_attention_is_never_requested() -> None:
    """Assert neither the worker nor the requirements reach for flash-attn.

    The model card lists ``attn_implementation="flash_attention_2"`` as one *optional*
    example forwarded through ``**kwargs`` (confirmed against the installed ``qwen-tts``
    wheel's own docstring), not a requirement. Installing flash-attn costs a multi-minute
    ``--no-build-isolation`` compile with a C toolchain in every user's cache for no
    measured benefit here.
    """
    assert "flash_attention_2" not in qwen_tts._WORKER_SCRIPT
    assert "attn_implementation" not in qwen_tts._WORKER_SCRIPT
    named = {r.split(">=")[0].split("==")[0].strip().lower() for r in qwen_tts._QWEN_TTS_REQUIREMENTS}
    assert "flash-attn" not in named
    assert "flash_attn" not in named


def test_empty_texts_returns_empty_without_touching_the_hub_or_a_venv() -> None:
    """An empty text list must return ``[]`` before constructing an ``HFModel`` at all.

    No mocking is set up here on purpose: if the empty-list short-circuit were removed or
    reordered after the default-model construction, this test would fail with a real
    (unmocked) Hub existence check rather than silently passing.
    """
    assert synthesize_texts_with_qwen([]) == []


def test_synthesis_stages_via_the_ref_but_pins_the_worker_payload_to_the_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``hf_subprocess_env`` is staged with the caller's ref; the worker payload gets the SHA.

    Passing the resolved SHA to ``hf_subprocess_env`` instead (the pattern
    ``speech_to_text/qwen.py`` and ``speaker_diarization/moss.py`` use) would skip writing
    ``refs/<ref>`` entirely -- ``_point_ref_at`` no-ops once its ``ref`` argument is already
    a SHA -- and strand the wrapper's own two unpinned ``cached_file()`` reads (see the
    module docstring's "partial-pin gap" section) with nothing to resolve offline. This test
    would fail if that ref/SHA split were collapsed back to passing the SHA everywhere.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "c" * 40)

    staging_calls = []

    def fake_hf_subprocess_env(repo_id: str, revision: str = "main", **kwargs: object) -> dict:
        staging_calls.append((repo_id, revision))
        return {}

    monkeypatch.setattr(qwen_tts, "hf_subprocess_env", fake_hf_subprocess_env)

    def fail_ensure_venv(*a: object, **k: object) -> None:
        raise RuntimeError("stop-before-venv")

    monkeypatch.setattr(qwen_tts, "ensure_venv", fail_ensure_venv)

    model: HFModel = HFModel(path_or_uri=qwen_tts._QWEN_TTS_DEFAULT_MODEL)  # revision defaults to "main"
    with pytest.raises(RuntimeError, match="stop-before-venv"):
        synthesize_texts_with_qwen(["hello"], model=model, device=DeviceType.CPU)

    assert staging_calls == [(qwen_tts._QWEN_TTS_DEFAULT_MODEL, "main")], (
        'hf_subprocess_env must be staged with the ref ("main"), not the resolved SHA'
    )


def test_none_speaker_reaches_the_worker_payload_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """``speaker=None`` must reach the worker payload as ``null`` -- the worker picks the default.

    The host would have to load the whole model to know its own speaker list, which would
    double the model load just to pick a default before staging. This test would fail if a
    host-side default speaker were substituted into the payload before it is sent.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "d" * 40)
    monkeypatch.setattr(qwen_tts, "hf_subprocess_env", lambda *a, **k: {})
    monkeypatch.setattr(qwen_tts, "ensure_venv", lambda *a, **k: Path("/tmp/fake-qwen-tts-venv"))
    monkeypatch.setattr(qwen_tts, "venv_python", lambda venv_dir: "/tmp/fake-python")

    captured: dict = {}

    class _FakeCompleted:
        returncode = 0
        stdout = '{"output_paths": [], "sample_rate": 24000}'
        stderr = ""

    def fake_run(cmd: object, input: str = "", **kwargs: object) -> _FakeCompleted:  # noqa: A002
        captured["input"] = input
        return _FakeCompleted()

    monkeypatch.setattr(qwen_tts.subprocess, "run", fake_run)

    model: HFModel = HFModel(path_or_uri=qwen_tts._QWEN_TTS_DEFAULT_MODEL)
    synthesize_texts_with_qwen(["hello"], model=model, device=DeviceType.CPU)

    payload = json.loads(captured["input"])
    assert payload["speaker"] is None


def test_supported_speakers_reads_config_json_directly_without_a_full_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``supported_speakers`` reads a single small file, not a full model snapshot.

    Routing this through ``resolve_model``/``ensure_hf_model`` would download the whole
    multi-GB checkpoint just to read a handful of speaker-name strings. This test would
    fail if that ever happened, since ``resolve_model`` is mocked to raise.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "a" * 40)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"talker_config": {"spk_id": {"ryan": 0, "aiden": 1, "vivian": 2}}}))

    calls = []

    def fake_hf_hub_download(repo_id: str, filename: str, revision: str = "main", **kwargs: object) -> str:
        calls.append((repo_id, filename, revision))
        return str(config_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    def fail_resolve_model(*a: object, **k: object) -> None:
        raise AssertionError("resolve_model must not be called -- it downloads the full snapshot")

    monkeypatch.setattr("senselab.utils.dependencies.resolve_model", fail_resolve_model)

    result = supported_speakers(model=HFModel(path_or_uri=qwen_tts._QWEN_TTS_DEFAULT_MODEL))

    assert result == ["aiden", "ryan", "vivian"]
    assert calls == [(qwen_tts._QWEN_TTS_DEFAULT_MODEL, "config.json", "a" * 40)]


def test_supported_speakers_raises_when_checkpoint_has_no_named_speakers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A checkpoint with no ``talker_config.spk_id`` mapping (e.g. Base/VoiceDesign) raises."""
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "b" * 40)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"talker_config": {}}))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: str(config_path))

    with pytest.raises(ValueError, match="talker_config.spk_id"):
        supported_speakers(model=HFModel(path_or_uri=qwen_tts._QWEN_TTS_DEFAULT_MODEL))


@pytest.mark.skipif(
    not qwen_tts_venv_present,
    reason=(
        f"qwen-tts venv not provisioned at {_QWEN_TTS_VENV_ROOT}; run manually to build it "
        "(downloads the 1.7B checkpoint and takes minutes). NOT YET VERIFIED on a GPU host -- "
        "see this task's report."
    ),
)
def test_qwen_tts_synthesizes_distinct_named_speakers_end_to_end() -> None:
    """Two distinct named speakers produce non-silent, distinct audio.

    This is the property the speaker-ceiling probe actually needs -- N distinct identities,
    not merely "some audio came out". Unverified on a GPU host as of this task.
    """
    import torch

    speakers = supported_speakers()
    assert len(speakers) >= 2

    audios = synthesize_texts_with_qwen(
        ["The quick brown fox jumps over the lazy dog."] * 2,
        speaker=[speakers[0], speakers[1]],
    )
    assert len(audios) == 2
    for audio in audios:
        assert audio.waveform.abs().max() > 0, "silent output — the model produced nothing"

    min_len = min(audios[0].waveform.shape[-1], audios[1].waveform.shape[-1])
    assert not torch.equal(audios[0].waveform[..., :min_len], audios[1].waveform[..., :min_len])
