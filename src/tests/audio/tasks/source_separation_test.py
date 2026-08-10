"""unasdiff source separation — API contract and class-space handling."""

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.source_separation import separate_audios, unasdiff
from senselab.audio.tasks.source_separation.api import resolve_source_classes
from senselab.utils.data_structures import HFModel


def test_class_map_has_41_classes_in_50_slots() -> None:
    """The prior's 50-wide embedding has only 41 trained rows.

    Passing an index in 41..49 would condition on an untrained embedding row and
    produce plausible-looking noise rather than an error.
    """
    doc = unasdiff.load_fsd_class_map_document()
    assert len(doc["classes"]) == 41
    assert max(doc["classes"].values()) == 40
    assert doc["num_embedding_slots"] == 50


def test_resolve_source_classes_maps_names_to_indices() -> None:
    """Names resolve to the same indices the raw class map carries."""
    assert resolve_source_classes(["Applause", "Cello"]) == [
        unasdiff.load_fsd_class_map_document()["classes"]["Applause"],
        unasdiff.load_fsd_class_map_document()["classes"]["Cello"],
    ]


def test_an_unknown_class_raises_and_names_the_valid_options() -> None:
    """An unmapped name must raise, not fall back to a class index.

    Silently falling back to index 0 would condition the prior on 'Hi-hat' while
    reporting the caller's own label — separation would be wrong and the output
    would claim otherwise.
    """
    with pytest.raises(ValueError) as exc:
        resolve_source_classes(["Helicopter"])
    assert "Helicopter" in str(exc.value)
    assert "Applause" in str(exc.value), "the error must enumerate the valid classes"


def test_upstream_is_pinned_to_a_full_commit_sha() -> None:
    """The upstream clone target is a 40-hex commit, never a mutable ref."""
    assert len(unasdiff._UNASDIFF_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in unasdiff._UNASDIFF_COMMIT)


def test_flash_attn_is_not_required() -> None:
    """flash-attn is absent from the venv's pinned requirements.

    atten_unet.py sets use_flash=False on ImportError and branches to a manual
    softmax attention, so the venv can omit a package that is slow and fragile to
    build. Verified against upstream, not assumed.
    """
    named = {r.split(">=")[0].split("==")[0].strip().lower() for r in unasdiff._UNASDIFF_REQUIREMENTS}
    assert "flash-attn" not in named and "flash_attn" not in named


def test_torch_is_pinned_for_cuda_routing() -> None:
    """Torch and torchaudio are named explicitly so ensure_venv's CUDA routing fires."""
    named = {r.split(">=")[0].split("==")[0].strip().lower() for r in unasdiff._UNASDIFF_REQUIREMENTS}
    assert "torch" in named and "torchaudio" in named


def test_worker_script_compiles_standalone() -> None:
    """The worker is a string literal run by another interpreter.

    A syntax error would otherwise surface only after the venv build and the model download.
    """
    compile(unasdiff._WORKER_SCRIPT, "<unasdiff worker>", "exec")


def test_worker_never_imports_the_benchmark_scripts() -> None:
    """The worker imports the library modules directly, not upstream's benchmark scripts.

    Upstream's ``test_speech_sound.py`` and its siblings call ``torch.cuda.set_device(0)`` at
    module import and abort on any CPU host.
    """
    for forbidden in ("test_speech_sound", "test_soundevent", "test_speech_speech"):
        assert forbidden not in unasdiff._WORKER_SCRIPT


def test_worker_uses_the_ema_weights() -> None:
    """The worker loads the EMA copy of each checkpoint, not the raw training weights.

    ``load_model`` in the benchmark script returns the EMA copy, not ``ckpt['model']``. Loading
    the non-EMA weights runs but separates measurably worse -- a silent quality regression
    rather than a failure.
    """
    assert '"ema"' in unasdiff._WORKER_SCRIPT or "'ema'" in unasdiff._WORKER_SCRIPT


def test_worker_packs_the_mixture_so_degradation_reproduces_it() -> None:
    """The worker packs orig_x so the sampler's internal degradation recomputes the mixture.

    ``p_sample_loop_group`` ignores its ``measurement`` argument and recomputes
    ``degradation(orig_x)``. Packing ``[mixture, zeros...]`` makes that sum equal the mixture
    exactly -- which is what keeps this an inference call and not an oracle.
    """
    assert "zeros" in unasdiff._WORKER_SCRIPT
    assert "orig_x" in unasdiff._WORKER_SCRIPT


def test_sound_modes_require_source_classes(mono_audio_sample: Audio) -> None:
    """The sound prior is class-conditioned.

    Without a class there is no defensible default -- index 0 is 'Hi-hat', and silently
    choosing it would separate against the wrong prior while reporting success.
    """
    with pytest.raises(ValueError, match="source_classes"):
        separate_audios([mono_audio_sample], mode="speech_sound", n_sources=2)


def test_speech_speech_needs_no_source_classes(mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both slots use the speech prior, whose only label is 0."""
    captured = {}

    def fake(audios: list, n_sources: int, source_class_indices: list, **kwargs: object) -> list:
        captured["labels"] = source_class_indices
        return [[audios[0]] * n_sources]

    monkeypatch.setattr("senselab.audio.tasks.source_separation.api.separate_with_unasdiff", fake)
    separate_audios([mono_audio_sample], mode="speech_speech", n_sources=2)
    assert captured["labels"] == [0, 0]


def test_speech_sound_prepends_the_speech_label(mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slot 0 is always the speech prior in speech_sound mode."""
    captured = {}

    def fake(audios: list, n_sources: int, source_class_indices: list, **kwargs: object) -> list:
        captured["labels"] = source_class_indices
        return [[audios[0]] * n_sources]

    monkeypatch.setattr("senselab.audio.tasks.source_separation.api.separate_with_unasdiff", fake)
    separate_audios([mono_audio_sample], mode="speech_sound", n_sources=2, source_classes=["Applause"])
    assert captured["labels"][0] == 0, "slot 0 is the speech prior"
    assert len(captured["labels"]) == 2


def test_source_classes_length_must_match_the_sound_slots(mono_audio_sample: Audio) -> None:
    """A mismatched source_classes length must raise rather than silently truncate/pad."""
    with pytest.raises(ValueError, match="n_sources"):
        separate_audios([mono_audio_sample], mode="speech_sound", n_sources=3, source_classes=["Applause"])


def test_an_unknown_mode_raises(mono_audio_sample: Audio) -> None:
    """An unrecognized mode must raise rather than silently falling through."""
    with pytest.raises(ValueError, match="mode"):
        separate_audios([mono_audio_sample], mode="music_speech", n_sources=2)


def test_a_foreign_model_is_rejected(mono_audio_sample: Audio) -> None:
    """Source separation has one backend; a model naming a different one must not be accepted."""
    foreign: HFModel = HFModel(path_or_uri="openai/whisper-tiny")
    with pytest.raises(ValueError, match="unasdiff"):
        separate_audios([mono_audio_sample], model=foreign, mode="speech_speech", n_sources=2)
