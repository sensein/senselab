"""Tests for the speech enhancement task."""

import ast
import json
import types
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
import soundfile
import torch
from speechbrain.inference.enhancement import SpectralMaskEnhancement as enhance_model
from speechbrain.inference.separation import SepformerSeparation as separator

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_enhancement import driftse, enhance_audios
from senselab.audio.tasks.speech_enhancement.driftse import enhance_audios_with_driftse
from senselab.audio.tasks.speech_enhancement.speechbrain import SpeechBrainEnhancer
from senselab.utils.data_structures import DeviceType, HFModel, SpeechBrainModel
from senselab.utils.subprocess_venv import _cache_dir_path

# Honor SENSELAB_VENV_CACHE the same way ensure_venv() does — hardcoding
# Path.home()/".cache"/... here would silently disagree with a cache dir
# override, so the gate below would never match where the venv actually lives.
_DRIFTSE_VENV_ROOT = _cache_dir_path() / "driftse"
driftse_venv_present = _DRIFTSE_VENV_ROOT.is_dir()


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


def test_default_model_is_unchanged(mono_audio_sample: Audio) -> None:
    """No existing caller may change behaviour.

    The workflow calls ``enhance_audios`` with the SpeechBrain default and must
    keep reaching the SpeechBrain path.
    """
    with patch(
        "senselab.audio.tasks.speech_enhancement.api.SpeechBrainEnhancer.enhance_audios_with_speechbrain",
        return_value=[mono_audio_sample],
    ) as sb:
        enhance_audios([mono_audio_sample])
    sb.assert_called_once()
    assert sb.call_args.kwargs["model"].path_or_uri == "speechbrain/sepformer-wham16k-enhancement"


def test_hfmodel_with_the_driftse_prefix_dispatches_to_driftse(
    mono_audio_sample: Audio, _offline_hfmodel_construction: None
) -> None:
    """An ``HFModel`` whose id starts with ``LIANGXU123/DriftSE`` reaches DriftSE."""
    with patch(
        "senselab.audio.tasks.speech_enhancement.api.enhance_audios_with_driftse",
        return_value=[mono_audio_sample],
    ) as ds:
        enhance_audios(
            [mono_audio_sample],
            model=HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO),
        )
    ds.assert_called_once()


def test_an_unrecognised_model_still_raises_not_implemented(
    mono_audio_sample: Audio, _offline_hfmodel_construction: None
) -> None:
    """Silently falling through to a default would enhance with a model the caller did not ask for."""
    with pytest.raises(NotImplementedError):
        enhance_audios([mono_audio_sample], model=HFModel(path_or_uri="some/other-model"))


def test_upstream_is_pinned_to_a_full_commit_sha() -> None:
    """Assert both DriftSE pins are full 40-char commit SHAs.

    A branch name or short SHA would let an upstream force-push or re-upload change what this
    backend runs without any change here. The repository is unpackaged, so the code pin is the only
    version contract available.
    """
    for pin in (driftse._DRIFTSE_COMMIT, driftse._DRIFTSE_HF_REVISION):
        assert len(pin) == 40
        assert all(c in "0123456789abcdef" for c in pin)


def test_training_and_metric_dependencies_are_not_installed() -> None:
    """Assert the DriftSE venv requirements omit training-only packages.

    Upstream's requirements.txt lists these for training and scoring and the inference path does
    not reach them, so a build that installs one means the worker started importing something it
    should not. ``pesq`` and ``pystoi`` are deliberately absent from this exclusion set: they *are*
    on the inference import chain, which only a real run revealed (see
    specs/20260818-083214-driftse-upstream-mit/design.md).
    """
    excluded = {
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

    Upstream omits weights_only. The checkpoint is a foreign pickle, and an MIT licence on the
    repository does not make the unrestricted unpickler any less arbitrary code execution at
    enhancement time.
    """
    assert "weights_only=True" in driftse._WORKER_SCRIPT


def test_empty_input_returns_empty_without_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert an empty audio list returns [] without touching the Hub or a venv.

    Constructing the HFModel below would otherwise perform a real Hub existence check; that check
    is mocked here per this project's rule against unmocked HFModel construction in tests.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    assert driftse.enhance_audios_with_driftse([], model=HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)) == []


def test_checkpoint_download_is_one_pinned_file_not_the_whole_snapshot(
    monkeypatch: pytest.MonkeyPatch, mono_audio_sample: Audio
) -> None:
    """Assert the checkpoint arrives as a single ``hf_hub_download`` at a resolved commit SHA.

    Upstream's mirror is 2.4 GB (two 1.14 GB checkpoints plus 1648 demo wavs) and a run reads one
    checkpoint, so ``resolve_model``'s whole-snapshot download is the wrong primitive here. What must
    not regress is the pinning: the file is requested at a 40-hex commit, which is what makes
    ``huggingface_hub`` take its commit-hash shortcut and skip the per-call Hub check that
    rate-limits under parallelism. ``ensure_venv`` is mocked to raise right afterwards, so the
    resolution is observable without spawning a venv.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)

    calls = []

    def fake_hf_hub_download(repo_id: str, filename: str, *, revision: str = "main", **kwargs: object) -> str:
        calls.append((repo_id, filename, revision))
        return "/tmp/fake-driftse/last.ckpt"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

    def fail_resolve_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("resolve_model downloads the entire 2.4 GB mirror; one file is needed")

    monkeypatch.setattr("senselab.utils.dependencies.resolve_model", fail_resolve_model)

    def fail_ensure_venv(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stop-before-venv")

    monkeypatch.setattr(driftse, "ensure_venv", fail_ensure_venv)

    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO, revision=driftse._DRIFTSE_HF_REVISION)
    with pytest.raises(RuntimeError, match="stop-before-venv"):
        driftse.enhance_audios_with_driftse([mono_audio_sample], model=model)

    assert len(calls) == 1, f"expected exactly one file download, got {calls}"
    repo_id, filename, revision = calls[0]
    assert repo_id == driftse._DRIFTSE_HF_REPO
    assert filename == driftse._DRIFTSE_VARIANTS[driftse._DRIFTSE_DEFAULT_VARIANT][0]
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision), (
        f"the checkpoint was requested at a ref, not a commit: {revision!r}"
    )


def test_worker_prefers_the_ema_state_dict() -> None:
    """Assert the worker loads ``ema`` before ``model``, as upstream does at the pinned commit.

    Upstream switched priority in commit 60333a68 and measures ema slightly ahead (PESQ 3.00 against
    2.98 over 824 files). The ``model`` fallback stays so a checkpoint without an ema still loads.
    """
    script = driftse._WORKER_SCRIPT
    assert 'if "ema" in ckpt' in script
    assert 'elif "model" in ckpt' in script
    assert script.index('"ema"') < script.index('elif "model" in ckpt')


def test_sigma_defaults_to_upstreams_own_constant_and_reaches_the_worker() -> None:
    """Assert sigma is a parameter defaulting to 0.05's replacement, not a hardcoded literal.

    Upstream called the old 0.05 misaligned with the paper and changed it to 0.01 in commit
    70bb6ded; an independent reproduction measures 0.05 costing ~0.11 PESQ and ~0.9 dB SI-SDR. A
    hardcoded literal here would silently keep running whichever value was written first.
    """
    import inspect

    assert driftse._DRIFTSE_DEFAULT_SIGMA == 0.01
    signature = inspect.signature(driftse.enhance_audios_with_driftse)
    assert signature.parameters["sigma"].default == driftse._DRIFTSE_DEFAULT_SIGMA
    assert "sigma * torch.randn_like(Y)" in driftse._WORKER_SCRIPT
    assert "0.05" not in driftse._WORKER_SCRIPT


def test_the_worker_rescales_its_output_and_drops_no_tail() -> None:
    """Assert the two level-critical lines of the worker are present, without a checkpoint.

    Upstream ``enhancement.py`` divides the enhanced waveform by its own peak before multiplying by
    the input's; omitting that half is the defect this guard exists for. The second assertion pins
    the chunking: fixed-length windows anchored at the end of the file, never a short remainder that
    the loop skips. See ``specs/20260818-083214-driftse-upstream-mit/design.md``.
    """
    script = driftse._WORKER_SCRIPT
    assert "out_peak = x.abs().max()" in script
    assert "return x / out_peak * norm if out_peak > 1e-8 else x * norm" in script
    assert "starts.append(total - chunk)" in script
    assert "< n_fft" not in script, "a window shorter than the transform means a dropped tail"


def test_every_variant_names_a_checkpoint_and_a_config() -> None:
    """Assert the variant table is complete and its default is one of its own keys.

    The weights mirror holds checkpoints under ``logs/<variant>/`` while the architecture configs
    live in the pinned code clone under ``config/``, so a variant needs both paths to be loadable.
    """
    assert driftse._DRIFTSE_DEFAULT_VARIANT in driftse._DRIFTSE_VARIANTS
    for variant, (checkpoint, config) in driftse._DRIFTSE_VARIANTS.items():
        assert checkpoint.startswith("logs/") and checkpoint.endswith(".ckpt"), variant
        assert config.startswith("config/") and config.endswith(".json"), variant


def test_an_unknown_variant_fails_before_any_download(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd variant must fail loudly, not fall back to the default checkpoint."""
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    with pytest.raises(ValueError, match="unknown DriftSE variant"):
        driftse.enhance_audios_with_driftse([], model=model, variant="distilhubert_typo")


def test_checkpoint_override_skips_the_hub_entirely(
    monkeypatch: pytest.MonkeyPatch, mono_audio_sample: Audio, tmp_path: Path
) -> None:
    """Assert a local ``SENSELAB_DRIFTSE_CHECKPOINT`` override never calls the Hub.

    An operator pointing at local checkpoint files must not need Hub access at all -- neither
    ``hf_hub_download`` nor ``resolve_model`` may run.
    ``ensure_venv`` is mocked to raise right after the override branch, which
    lets this test observe that neither Hub path was taken without spawning a
    real subprocess venv or touching the network.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    monkeypatch.setenv(driftse._DRIFTSE_CHECKPOINT_ENV, str(tmp_path))

    def fail_resolve_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("resolve_model must not be called when a local checkpoint override is set")

    monkeypatch.setattr("senselab.utils.dependencies.resolve_model", fail_resolve_model)

    def fail_hf_hub_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("hf_hub_download must not be called when a local checkpoint override is set")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_hf_hub_download)

    def fail_ensure_venv(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stop-before-venv")

    monkeypatch.setattr(driftse, "ensure_venv", fail_ensure_venv)

    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO, revision=driftse._DRIFTSE_HF_REVISION)
    with pytest.raises(RuntimeError, match="stop-before-venv"):
        driftse.enhance_audios_with_driftse([mono_audio_sample], model=model)


def test_every_driftse_worker_wav_write_names_an_explicit_subtype() -> None:
    """No ``sf.write`` in the DriftSE worker relies on soundfile's PCM_16 default.

    That default clips every sample past +-1, and it has silently corrupted a measurement three
    times in this repository -- once costing three SepFormer streams up to 8.9% of their samples
    and 9.5 dB of agreement with the CPU run.
    """
    tree = ast.parse(driftse._WORKER_SCRIPT)
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write"
    ]
    assert writes, "test premise: the worker writes WAV files"
    for node in writes:
        assert "subtype" in {kw.arg for kw in node.keywords}, (
            f"sf.write at worker line {node.lineno} relies on the PCM_16 default"
        )


def test_driftse_input_wavs_are_written_as_float_not_pcm16(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _offline_hfmodel_construction: None
) -> None:
    """The WAV the host hands the DriftSE worker is FLOAT, and samples past +-1 survive it.

    ``Audio.save_to_file`` writes PCM_16 for a ``.wav``, so an input peaking above full scale
    was clipped before the enhancer ever saw it.
    """
    monkeypatch.setenv(driftse._DRIFTSE_CHECKPOINT_ENV, str(tmp_path))
    monkeypatch.setattr(driftse, "ensure_venv", lambda *a, **k: Path("/tmp/fake-driftse-venv"))
    monkeypatch.setattr(driftse, "venv_python", lambda venv_dir: "python3")

    captured: dict = {}

    def fake_run(
        cmd: list, *, input: str, capture_output: bool, text: bool, timeout: float, env: dict
    ) -> types.SimpleNamespace:
        payload = json.loads(input)
        captured["subtypes"] = [soundfile.info(p).subtype for p in payload["in_paths"]]
        captured["peak"] = max(abs(soundfile.read(p, dtype="float32")[0]).max() for p in payload["in_paths"])
        for in_path, out_path in zip(payload["in_paths"], payload["out_paths"]):
            data, sr = soundfile.read(in_path, dtype="float32")
            soundfile.write(out_path, data, sr, subtype="FLOAT")
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"output_paths": payload["out_paths"]}), stderr="")

    monkeypatch.setattr(driftse.subprocess, "run", fake_run)

    waveform = torch.zeros(1, 16000)
    waveform[0, 100:200] = 1.75
    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO, revision=driftse._DRIFTSE_HF_REVISION)
    driftse.enhance_audios_with_driftse(
        [Audio(waveform=waveform, sampling_rate=16000)],
        model=model,
    )

    assert captured["subtypes"] == ["FLOAT"]
    assert captured["peak"] > 1.5, "an out-of-range sample was clipped on write"


@pytest.mark.parametrize(
    ("model_uri", "expected"),
    [
        ("speechbrain/sepformer-wham16k-enhancement", separator),
        ("speechbrain/sepformer-wsj02mix", separator),
        ("speechbrain/metricgan-plus-voicebank", enhance_model),
        ("speechbrain/mtl-mimic-voicebank", enhance_model),
    ],
)
def test_loader_for_selects_class_by_name(model_uri: str, expected: type) -> None:
    """The interface class is chosen directly from the model name, no load needed."""
    assert SpeechBrainEnhancer._loader_for(model_uri) is expected


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


@pytest.mark.skipif(
    not driftse_venv_present,
    reason=f"driftse venv not provisioned at {_DRIFTSE_VENV_ROOT}; run manually to build it (first run takes minutes)",
)
def test_driftse_enhances_and_preserves_length(mono_audio_sample: Audio) -> None:
    """Length preservation is the cheapest real correctness check available without a reference signal.

    ``istft(length=T_orig)`` must round-trip, and a chunking bug in the
    overlap-add path shows up here immediately as a short or long result.
    """
    from senselab.audio.tasks.preprocessing import resample_audios

    audio = resample_audios([mono_audio_sample], resample_rate=16000)[0]
    out = enhance_audios([audio], model=HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO))

    assert len(out) == 1
    assert out[0].sampling_rate == 16000
    assert out[0].waveform.shape[-1] == audio.waveform.shape[-1]
    assert out[0].waveform.abs().max() > 0, "silent output — the model produced nothing"


@pytest.mark.skipif(
    not driftse_venv_present,
    reason=f"driftse venv not provisioned at {_DRIFTSE_VENV_ROOT}; run manually to build it (first run takes minutes)",
)
def test_driftse_is_reproducible_under_a_fixed_seed(mono_audio_sample: Audio) -> None:
    """Assert a fixed seed makes the stochastic forward pass reproducible.

    ``train_add_gaussian`` (the released checkpoint's setting) makes the
    forward pass consume a Gaussian sample, so without a seed a rerun would
    produce different audio -- which would make any cached artifact keyed on
    this output non-reproducible.
    """
    from senselab.audio.tasks.preprocessing import resample_audios

    audio = resample_audios([mono_audio_sample], resample_rate=16000)[0]
    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    a = enhance_audios_with_driftse([audio], model=model, seed=17)[0]
    b = enhance_audios_with_driftse([audio], model=model, seed=17)[0]

    assert (a.waveform - b.waveform).abs().max() < 1e-5


@pytest.mark.skipif(
    not driftse_venv_present,
    reason=f"driftse venv not provisioned at {_DRIFTSE_VENV_ROOT}; run manually to build it (first run takes minutes)",
)
@pytest.mark.parametrize("variant", sorted(driftse._DRIFTSE_VARIANTS))
def test_driftse_output_keeps_the_input_level_for_both_variants(mono_audio_sample: Audio, variant: str) -> None:
    """Assert the enhanced waveform comes back at the input's level, for either checkpoint.

    The bound is loose on both sides -- enhancement may take a peak away, but it may not move the
    level by an order of magnitude and it may not clip. Why an unrescaled output does exactly that,
    and why only one of the two checkpoints shows it: ``specs/20260818-083214-driftse-upstream-mit``.
    """
    from senselab.audio.tasks.preprocessing import resample_audios

    audio = resample_audios([mono_audio_sample], resample_rate=16000)[0]
    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    out = enhance_audios_with_driftse([audio], model=model, variant=variant)[0]

    in_peak = float(audio.waveform.abs().max())
    out_peak = float(out.waveform.abs().max())
    assert 0.5 * in_peak <= out_peak <= 1.05 * in_peak, (
        f"{variant}: output peak {out_peak} against input peak {in_peak}"
    )
    clipped = float((out.waveform.abs() >= 0.999).double().mean())
    assert clipped < 0.01, f"{variant}: {clipped:.1%} of samples at full scale"


# ── DriftSE worker device selection ───────────────────────────────────


def _stub_driftse_worker(monkeypatch: pytest.MonkeyPatch, captured: dict, tmp_path: Path) -> None:
    """Replace the checkpoint, the venv and the worker subprocess with fakes that record the payload.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        captured: Filled in with ``payload`` and ``timeout``.
        tmp_path: Directory used as the local checkpoint override, so the Hub is never reached.
    """
    import json
    import types

    import soundfile

    monkeypatch.setenv(driftse._DRIFTSE_CHECKPOINT_ENV, str(tmp_path))
    monkeypatch.setattr(driftse, "ensure_venv", lambda *a, **k: Path("/tmp/fake-driftse-venv"))
    monkeypatch.setattr(driftse, "venv_python", lambda venv_dir: "python3")

    def fake_run(
        cmd: list, *, input: str, capture_output: bool, text: bool, timeout: float, env: dict
    ) -> types.SimpleNamespace:
        payload = json.loads(input)
        captured["payload"] = payload
        captured["timeout"] = timeout
        for in_path, out_path in zip(payload["in_paths"], payload["out_paths"]):
            samples, sr = soundfile.read(in_path, dtype="float32", always_2d=True)
            soundfile.write(out_path, samples[:, 0], sr, subtype="FLOAT")
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"output_paths": payload["out_paths"]}), stderr="")

    monkeypatch.setattr(driftse.subprocess, "run", fake_run)


def _synthetic_audio(seconds: float) -> Audio:
    """Return ``seconds`` of 16 kHz mono noise, well inside full scale.

    Args:
        seconds: Duration to generate.

    Returns:
        An ``Audio`` at 16 kHz.
    """
    import torch

    return Audio(waveform=0.1 * torch.randn(1, int(seconds * 16000)), sampling_rate=16000)


def test_the_callers_device_reaches_the_driftse_worker_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _offline_hfmodel_construction: None
) -> None:
    """A caller-selected device is sent to the worker instead of being validated and dropped.

    ``device`` was handed to ``_select_device_and_dtype`` purely for validation and its result
    thrown away, so the worker chose for itself and no caller could select a card.
    """
    captured: dict = {}
    _stub_driftse_worker(monkeypatch, captured, tmp_path)

    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    enhance_audios_with_driftse([_synthetic_audio(1.0)], model=model, device=DeviceType.CPU)

    assert captured["payload"]["device"] == "cpu"


def test_no_device_leaves_the_choice_to_the_driftse_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _offline_hfmodel_construction: None
) -> None:
    """``device=None`` sends ``None``, not a device the host's own torch build happened to see.

    The host interpreter and the venv have separate torch builds; only the venv's answer to
    ``torch.cuda.is_available()`` governs where the worker can run.
    """
    captured: dict = {}
    _stub_driftse_worker(monkeypatch, captured, tmp_path)

    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    enhance_audios_with_driftse([_synthetic_audio(1.0)], model=model)

    assert captured["payload"]["device"] is None


def test_the_driftse_worker_never_requests_a_bare_cuda_device() -> None:
    """A bare ``torch.device("cuda")`` takes whatever index torch defaults to.

    On a multi-GPU host that is card 0 regardless of the caller's choice or of a
    ``CUDA_VISIBLE_DEVICES`` mask, so the worker always names an index.
    """
    script = driftse._WORKER_SCRIPT
    assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in script
    assert "cuda:%d" in script, "the worker must name an explicit CUDA index"
    assert 'args.get("device")' in script, "the worker must read the device the host sent"


def test_an_incompatible_device_is_rejected_before_the_venv(_offline_hfmodel_construction: None) -> None:
    """MPS is not one of this backend's compatible devices and must raise rather than fall back.

    Held before this change too; kept as a guard on the validation the device plumbing reuses.
    """
    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    with pytest.raises(ValueError):
        enhance_audios_with_driftse([_synthetic_audio(1.0)], model=model, device=DeviceType.MPS)


# ── DriftSE worker timeout ────────────────────────────────────────────


def test_the_driftse_default_timeout_scales_with_windows_and_window_length() -> None:
    """The ceiling is derived from the work, not a constant.

    A fixed 1800 s ceiling covers about 25 minutes of audio at the per-window cost measured on
    this host, and the run that exceeded it lost everything the worker had written.
    """
    assert driftse._default_timeout_s(1, 20.0) == driftse._TIMEOUT_FLOOR_S

    big = driftse._default_timeout_s(200, 20.0)
    more_windows = driftse._default_timeout_s(400, 20.0)
    longer_windows = driftse._default_timeout_s(200, 40.0)
    assert big > driftse._TIMEOUT_FLOOR_S
    assert more_windows == pytest.approx(2 * big)
    assert longer_windows == pytest.approx(2 * big)


def test_the_window_count_mirrors_the_workers_own_chunking() -> None:
    """The host counts the windows the worker will actually run, including the flush-to-end one.

    The count only feeds the ceiling, but a count that drifts from the worker's chunking would
    set the ceiling for a different amount of work than the worker performs.
    """
    chunk, hop = 20 * 16000, 18 * 16000
    assert driftse._window_count(16000, chunk, hop) == 1, "shorter than one window"
    assert driftse._window_count(chunk, chunk, hop) == 1, "exactly one window"
    assert driftse._window_count(chunk + 1, chunk, hop) == 2, "one sample over -- a flush-to-end window"
    assert driftse._window_count(38 * 16000, chunk, hop) == 2, "two regular windows, nothing left over"
    assert driftse._window_count(58 * 16000, chunk, hop) == 4, "three regular windows plus a flush-to-end one"

    # If the worker's own windowing moves, this count has to move with it.
    assert "starts = list(range(0, total - chunk + 1, hop_samples))" in driftse._WORKER_SCRIPT
    assert "starts.append(total - chunk)" in driftse._WORKER_SCRIPT


def test_the_derived_driftse_ceiling_reaches_subprocess_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _offline_hfmodel_construction: None
) -> None:
    """The timeout ``subprocess.run`` receives is the derived one, not a hardcoded 1800."""
    captured: dict = {}
    _stub_driftse_worker(monkeypatch, captured, tmp_path)
    # The floor covers the first-use venv build and would otherwise swallow the work term for any
    # input short enough to keep this test cheap.
    monkeypatch.setattr(driftse, "_TIMEOUT_FLOOR_S", 1.0)

    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    enhance_audios_with_driftse([_synthetic_audio(58.0)], model=model)

    assert captured["timeout"] == driftse._default_timeout_s(4, 20.0)
    assert captured["timeout"] != 1800


def test_an_explicit_driftse_timeout_overrides_the_derived_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _offline_hfmodel_construction: None
) -> None:
    """``timeout_s`` is honoured verbatim."""
    captured: dict = {}
    _stub_driftse_worker(monkeypatch, captured, tmp_path)

    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    enhance_audios_with_driftse([_synthetic_audio(1.0)], model=model, timeout_s=42.0)

    assert captured["timeout"] == 42.0


def test_a_non_positive_driftse_timeout_raises(_offline_hfmodel_construction: None) -> None:
    """A zero or negative ceiling would abort the worker instantly; reject it up front."""
    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="timeout_s"):
            enhance_audios_with_driftse([_synthetic_audio(1.0)], model=model, timeout_s=bad)


def test_a_driftse_timeout_names_the_ceiling_the_input_and_the_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _offline_hfmodel_construction: None
) -> None:
    """A ``TimeoutExpired`` becomes an actionable ``RuntimeError``, not a bare stack trace.

    The unhandled exception said only that a subprocess had run too long: not which ceiling it
    hit, how much audio it was given, how far it had got, or which knob raises the ceiling.
    """
    import json
    import subprocess

    import soundfile

    monkeypatch.setenv(driftse._DRIFTSE_CHECKPOINT_ENV, str(tmp_path))
    monkeypatch.setattr(driftse, "ensure_venv", lambda *a, **k: Path("/tmp/fake-driftse-venv"))
    monkeypatch.setattr(driftse, "venv_python", lambda venv_dir: "python3")

    def fake_run(cmd: list, *, input: str, capture_output: bool, text: bool, timeout: float, env: dict) -> None:
        payload = json.loads(input)
        # The first input finishes before the ceiling fires, so the error can report progress.
        samples, sr = soundfile.read(payload["in_paths"][0], dtype="float32", always_2d=True)
        soundfile.write(payload["out_paths"][0], samples[:, 0], sr, subtype="FLOAT")
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(driftse.subprocess, "run", fake_run)

    model: HFModel = HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)
    with pytest.raises(RuntimeError) as exc:
        enhance_audios_with_driftse(
            [_synthetic_audio(58.0), _synthetic_audio(20.0)], model=model, timeout_s=123.0, device=DeviceType.CPU
        )

    message = str(exc.value)
    assert "123s" in message, "the ceiling that fired must be named"
    assert "1/2 output(s) written" in message, "progress at the point of failure must be reported"
    assert "78.0s of audio" in message, "the input being processed must be named"
    assert "5 window(s) of 20s" in message, "the work the ceiling was set for must be named"
    assert driftse._DRIFTSE_DEFAULT_VARIANT in message
    assert "device=cpu" in message
    assert "timeout_s" in message, "the message must name the knob that raises the ceiling"
