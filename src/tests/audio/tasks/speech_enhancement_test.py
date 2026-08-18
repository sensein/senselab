"""Tests for the speech enhancement task."""

from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
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
