"""ClearVoice through the three audio task APIs: dispatch, output contracts, provenance, WAV subtype.

The worker itself is never launched. What is exercised is everything on the host side of it — the
``Audio`` round trip, the files handed over, the device and ceiling in the payload, and the contracts
each entry point applies to what comes back. Anything needing the real weights, the venv or a GPU is
covered by ``src/tests/utils/clearvoice_test.py``'s payload assertions instead, or skipped there.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import soundfile

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.clearvoice import clearvoice_provenance
from senselab.audio.tasks.source_separation import separate_audios
from senselab.audio.tasks.speech_enhancement import enhance_audios
from senselab.audio.tasks.speech_super_resolution import super_resolve_audios
from senselab.utils import clearvoice as cv
from senselab.utils.backend_parameters import PARAMETER_RECORD_KEY
from senselab.utils.data_structures import DeviceType, HFModel
from senselab.utils.portable_audio_io import write_audio

STAGED_SHA = "d" * 40


@pytest.fixture
def offline_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``HFModel(...)`` be constructed without reaching the Hub.

    Both of its validators make a network call otherwise, which would make every test here depend on
    Hub availability to check something that has nothing to do with the Hub.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda **kwargs: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: STAGED_SHA)


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Any]:
    """Stub the venv, the checkpoint staging and the worker subprocess.

    The fake worker writes real WAVs, so the ``Audio`` round trip, the subtype of the files the host
    hands over, and the lazy-load-before-cleanup ordering are all genuinely exercised. Set
    ``captured["n_sources"]`` before calling to make it return more than one source.
    """
    captured: Dict[str, Any] = {"n_sources": 1}
    monkeypatch.setattr(cv, "ensure_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(cv, "venv_python", lambda venv_dir: "python3")
    monkeypatch.setattr(
        cv, "stage_clearvoice_checkpoints", lambda spec, revision="main": (tmp_path / "ckpt", STAGED_SHA)
    )
    monkeypatch.setattr(cv, "stage_s3fd_weights", lambda: tmp_path / "sfd_face.pth")

    def fake_run(cmd: list, **kwargs: Any) -> types.SimpleNamespace:
        payload = json.loads(kwargs["input"])
        captured["payload"] = payload
        captured["timeout"] = kwargs["timeout"]
        captured["in_subtypes"] = [soundfile.info(p).subtype for p in payload["in_paths"]]
        captured["in_rates"] = [soundfile.info(p).samplerate for p in payload["in_paths"]]
        captured["in_peaks"] = [float(np.abs(soundfile.read(p, dtype="float32")[0]).max()) for p in payload["in_paths"]]

        written: List[List[str]] = []
        for index, in_path in enumerate(payload["in_paths"]):
            samples, rate = soundfile.read(in_path, dtype="float32")
            paths = []
            for source in range(captured["n_sources"]):
                out_path = f"{payload['out_dir']}/out_{index}_s{source}.wav"
                write_audio(out_path, samples, rate)
                paths.append(out_path)
            written.append(paths)
        body = {
            "output_paths": written,
            "input_norm_scalars": [2.5 for _ in payload["in_paths"]],
            "device": payload["device"] or "cpu",
        }
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")

    monkeypatch.setattr(cv.subprocess, "run", fake_run)
    return captured


def _model(name: str) -> HFModel:
    return HFModel(path_or_uri=f"alibabasglab/{name}", revision="main")


# ── Enhancement ───────────────────────────────────────────────────────


def test_enhance_audios_dispatches_to_clearvoice_by_model_id(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """Naming a ClearVoice enhancement checkpoint must reach ClearVoice, not SpeechBrain."""
    enhanced = enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"))
    assert len(enhanced) == 1
    assert worker["payload"]["model_name"] == "FRCRN_SE_16K"
    assert worker["payload"]["task"] == "speech_enhancement"


def test_the_enhanced_audio_records_the_commit_that_produced_it(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """The loader cannot be pinned, so the run must record which weights ran by this route."""
    enhanced = enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"))[0]
    assert clearvoice_provenance(enhanced) == ("alibabasglab/FRCRN_SE_16K", STAGED_SHA)
    assert enhanced.metadata["clearvoice"]["capability"] == "speech enhancement"


def test_the_input_reaches_the_worker_as_float_at_the_models_rate(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """PCM_16 would clip the input before the model ever saw it; 48 kHz would be the wrong rate."""
    enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"))
    assert worker["in_subtypes"] == ["FLOAT"]
    assert worker["in_rates"] == [16000]


def test_the_48k_checkpoint_gets_48k_input(offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio) -> None:
    """Each checkpoint's rate comes from the table, not from the caller's file."""
    enhanced = enhance_audios([mono_audio_sample], model=_model("MossFormer2_SE_48K"))
    assert worker["in_rates"] == [48000]
    assert enhanced[0].sampling_rate == 48000


def test_an_out_of_range_input_reaches_the_worker_unclipped(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """Enhancing a clipped copy of the input would be a measurement of the clipping.

    The FLOAT subtype a plain .wav resolves to carries values beyond ±1, so the guarantee here is a
    bit-exact hand-off rather than a refusal: what must not happen is silent clipping at ±1.
    """
    loud = Audio(waveform=mono_audio_sample.waveform * 40.0, sampling_rate=mono_audio_sample.sampling_rate)
    expected_peak = float(loud.waveform.abs().max())
    assert expected_peak > 1.0, "the fixture must actually be out of range for this to test anything"
    enhance_audios([loud], model=_model("FRCRN_SE_16K"))
    assert worker["in_subtypes"] == ["FLOAT"]
    # Resampling to 16 kHz moves the peak slightly; the point is that it is not pinned at 1.0.
    assert worker["in_peaks"][0] > 1.0


def test_a_separation_checkpoint_is_refused_by_the_enhancer(offline_hub: None, worker: Dict[str, Any]) -> None:
    """It would otherwise return two sources to a caller holding one signal."""
    with pytest.raises(ValueError) as exc:
        enhance_audios([], model=_model("MossFormer2_SS_16K"))
    assert "source_separation.separate_audios" in str(exc.value)


def test_more_than_one_source_from_an_enhancer_is_refused_not_flattened(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """The count comes from the output. Taking the first would hide a decomposition."""
    worker["n_sources"] = 2
    with pytest.raises(RuntimeError) as exc:
        enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"))
    assert "2 source(s)" in str(exc.value)
    assert "source_separation" in str(exc.value)


def test_the_device_is_plumbed_to_the_worker(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """A device accepted and dropped is the defect this asserts against."""
    enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"), device=DeviceType.CPU)
    assert worker["payload"]["device"] == "cpu"


def test_mps_is_refused_at_the_task_boundary(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """Upstream would have selected MPS silently; none of the six is verified on it."""
    with pytest.raises(ValueError):
        enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"), device=DeviceType.MPS)


# ── The parameter pathway through the dispatchers ─────────────────────


def test_a_backend_parameter_reaches_the_worker(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """timeout_s is ClearVoice's one tunable, and it must actually arrive."""
    enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"), parameters={"timeout_s": 123.0})
    assert worker["timeout"] == 123.0


def test_the_chosen_parameters_are_recorded_on_the_result(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """A validated parameter that nothing records still leaves the run unable to say what ran."""
    enhanced = enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"), parameters={"timeout_s": 60.0})[0]
    record = enhanced.metadata[PARAMETER_RECORD_KEY]
    assert record["backend"] == "clearvoice"
    assert record["parameters"] == {"timeout_s": 60.0}
    assert record["explicit"] == ["timeout_s"]


def test_a_misspelled_parameter_raises_instead_of_running_the_default(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """This is the whole reason the pathway is not a permissive dict."""
    with pytest.raises(ValueError) as exc:
        enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"), parameters={"timeout": 60.0})
    assert "did you mean 'timeout_s'" in str(exc.value)


def test_a_parameter_belonging_to_another_backend_raises(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """DriftSE's ``variant`` means nothing to ClearVoice and must not be quietly dropped."""
    with pytest.raises(ValueError, match="Unknown parameter"):
        enhance_audios([mono_audio_sample], model=_model("FRCRN_SE_16K"), parameters={"variant": "x"})


def test_driftse_variant_now_reaches_its_backend(
    offline_hub: None, monkeypatch: pytest.MonkeyPatch, mono_audio_sample: Audio
) -> None:
    """The measured defect: enhance_audios never forwarded variant, so one checkpoint was unreachable."""
    seen: Dict[str, Any] = {}

    # The fake mirrors the real signature: validation and the call both go through this object, so a
    # **kwargs stand-in would declare nothing and the dispatcher would refuse the very key under test.
    def fake_driftse(
        audios: list,
        model: object,
        device: object = None,
        seed: int = 0,
        sigma: float = 0.01,
        variant: str = "distillhubert_three_layers_with_z",
        chunk_s: float = 20.0,
        overlap_s: float = 2.0,
        timeout_s: Optional[float] = None,
    ) -> list:
        seen["variant"] = variant
        return list(audios)

    monkeypatch.setattr("senselab.audio.tasks.speech_enhancement.api.enhance_audios_with_driftse", fake_driftse)
    model = HFModel(path_or_uri="LIANGXU123/DriftSE", revision="main")
    variant = "distillhubert_three_layers_pesq_sisdr_ccmse_with_z"
    enhance_audios([mono_audio_sample], model=model, parameters={"variant": variant})
    assert seen == {"variant": variant}


def test_driftse_really_declares_the_variant_the_pathway_forwards() -> None:
    """The test above uses a stand-in, so pin the real backend's declared set separately."""
    from senselab.audio.tasks.speech_enhancement.driftse import enhance_audios_with_driftse
    from senselab.utils.backend_parameters import declared_parameters

    declared = declared_parameters(enhance_audios_with_driftse)
    assert "variant" in declared and "timeout_s" in declared


def test_speechbrain_declares_no_parameters_and_says_so(offline_hub: None, mono_audio_sample: Audio) -> None:
    """A caller aiming a DriftSE knob at SpeechBrain must be told, not ignored."""
    from senselab.utils.data_structures import SpeechBrainModel

    monkeypatched = SpeechBrainModel(path_or_uri="speechbrain/sepformer-wham16k-enhancement", revision="main")
    with pytest.raises(ValueError, match="no tunable parameters"):
        enhance_audios([mono_audio_sample], model=monkeypatched, parameters={"variant": "x"})


def test_an_unknown_model_still_reports_every_supported_backend(offline_hub: None, mono_audio_sample: Audio) -> None:
    """The dispatch table's failure message is how a caller discovers the three backends."""
    from senselab.utils.data_structures import TorchModel

    with pytest.raises(NotImplementedError) as exc:
        enhance_audios([mono_audio_sample], model=TorchModel(path_or_uri="pytorch/vision", revision="main"))
    message = str(exc.value)
    assert "SpeechBrain" in message and "DriftSE" in message and "alibabasglab" in message


# ── Separation ────────────────────────────────────────────────────────


def test_separate_audios_dispatches_to_clearvoice_and_returns_its_sources(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """The separator's two sources come back as two Audio objects per input."""
    worker["n_sources"] = 2
    separated = separate_audios([mono_audio_sample], model=_model("MossFormer2_SS_16K"))
    assert len(separated) == 1 and len(separated[0]) == 2
    assert [source.metadata["clearvoice"]["source_index"] for source in separated[0]] == [0, 1]
    assert worker["payload"]["rms_normalise"] is True


def test_a_separator_returning_one_source_is_refused(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """One signal from a separator is an unseparated mixture presented as a decomposition."""
    worker["n_sources"] = 1
    with pytest.raises(RuntimeError, match="at least two sources"):
        separate_audios([mono_audio_sample], model=_model("MossFormer2_SS_16K"))


def test_the_unapplied_rms_scalar_is_reported_on_each_source(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """Upstream does not restore the level on this branch, so the caller must be able to see that."""
    worker["n_sources"] = 2
    separated = separate_audios([mono_audio_sample], model=_model("MossFormer2_SS_16K"))
    for source in separated[0]:
        assert source.metadata["clearvoice"]["input_norm_scalar"] == 2.5
        assert source.metadata["clearvoice"]["input_norm_applied_to_output"] is False


def test_unasdiff_only_arguments_are_refused_for_the_clearvoice_separator(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """mode and source_classes describe unasdiff's priors; ignoring them would be silent."""
    worker["n_sources"] = 2
    with pytest.raises(ValueError) as exc:
        separate_audios(
            [mono_audio_sample], model=_model("MossFormer2_SS_16K"), mode="sound_sound", source_classes=["Cello"]
        )
    message = str(exc.value)
    assert "mode" in message and "source_classes" in message
    assert "no modes and no class conditioning" in message


def test_a_source_count_the_checkpoint_cannot_honour_is_refused(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """MossFormer2_SS_16K has two decoder heads; asking for three cannot be satisfied."""
    with pytest.raises(ValueError, match="separates exactly 2 sources"):
        separate_audios([mono_audio_sample], model=_model("MossFormer2_SS_16K"), n_sources=3)


def test_unasdiff_still_rejects_the_parameters_mapping(offline_hub: None, mono_audio_sample: Audio) -> None:
    """Every parameter unasdiff declares is already a named argument, so two sources for one value."""
    with pytest.raises(ValueError, match="already"):
        separate_audios([mono_audio_sample], mode="speech_speech", n_sources=2, parameters={"seed": 3})


# ── Super-resolution ──────────────────────────────────────────────────


def test_super_resolve_audios_returns_48k(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """The output rate is the model's, not the input's — which is why this is not enhancement."""
    resolved = super_resolve_audios([mono_audio_sample])
    assert len(resolved) == 1
    assert resolved[0].sampling_rate == 48000
    assert worker["payload"]["model_name"] == "MossFormer2_SR_48K"
    assert worker["payload"]["rms_normalise"] is False


def test_super_resolution_defaults_to_the_only_checkpoint(offline_hub: None, worker: Dict[str, Any]) -> None:
    """One backend, so the default must be it rather than None."""
    from senselab.audio.tasks.speech_super_resolution import DEFAULT_SUPER_RESOLUTION_MODEL

    assert DEFAULT_SUPER_RESOLUTION_MODEL == "alibabasglab/MossFormer2_SR_48K"


def test_an_enhancement_checkpoint_is_refused_by_super_resolution(offline_hub: None) -> None:
    """Each entry point owns one capability; the table is what says which."""
    with pytest.raises(ValueError) as exc:
        super_resolve_audios([], model=_model("FRCRN_SE_16K"))
    assert "speech_enhancement.enhance_audios" in str(exc.value)


def test_super_resolution_records_its_parameters(
    offline_hub: None, worker: Dict[str, Any], mono_audio_sample: Audio
) -> None:
    """Same provenance contract as the other dispatchers."""
    resolved = super_resolve_audios([mono_audio_sample], parameters={"timeout_s": 30.0})
    assert resolved[0].metadata[PARAMETER_RECORD_KEY]["explicit"] == ["timeout_s"]


# ── Empty input ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("entry", "model_name"),
    [(enhance_audios, "FRCRN_SE_16K"), (super_resolve_audios, "MossFormer2_SR_48K")],
)
def test_no_audio_means_no_worker(
    offline_hub: None, worker: Dict[str, Any], entry: Any, model_name: str
) -> None:
    """An empty list must not provision a venv or stage 670 MB of weights."""
    assert entry([], model=_model(model_name)) == []
    assert "payload" not in worker
