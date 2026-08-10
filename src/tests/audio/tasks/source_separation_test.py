"""unasdiff source separation — API contract and class-space handling."""

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.source_separation import separate_audios, unasdiff
from senselab.audio.tasks.source_separation.api import resolve_source_classes
from senselab.audio.tasks.source_separation.unasdiff import align_permutations
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


def test_a_foreign_model_is_rejected(mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch) -> None:
    """Source separation has one backend; a model naming a different one must not be accepted.

    Both HFModel validators are monkeypatched rather than constructing a real one: an unmocked
    HFModel triggers a real Hub lookup (and, on a cold cache, a full snapshot download) in its
    field validator.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    foreign: HFModel = HFModel(path_or_uri="openai/whisper-tiny")
    with pytest.raises(ValueError, match="unasdiff"):
        separate_audios([mono_audio_sample], model=foreign, mode="speech_speech", n_sources=2)


def test_identity_permutation_when_slots_already_match() -> None:
    """Slots that already agree resolve to the identity permutation."""
    a, b = torch.randn(1000), torch.randn(1000)
    assert align_permutations([a, b], [a, b]) == [0, 1]


def test_swapped_slots_are_detected() -> None:
    """A permutation swap between windows must be detected, not ignored.

    Windows are separated independently, so slot 0 in window k need not be the same source as
    slot 0 in window k+1. Concatenating without this check swaps sources mid-file and the
    result is worse than the mixture.
    """
    a, b = torch.randn(1000), torch.randn(1000)
    assert align_permutations([a, b], [b, a]) == [1, 0]


def test_three_sources_resolve_to_a_full_permutation() -> None:
    """A three-way rotation resolves to a full, correct permutation, not a partial match."""
    a, b, c = torch.randn(1000), torch.randn(1000), torch.randn(1000)
    assert sorted(align_permutations([a, b, c], [c, a, b])) == [0, 1, 2]
    assert align_permutations([a, b, c], [c, a, b]) == [1, 2, 0]


def test_alignment_survives_scaling_and_noise() -> None:
    """Correlation must be scale-invariant and tolerate the disagreement between windows.

    Adjacent windows overlap but are not identical there -- each is the sampler's own estimate.
    """
    a, b = torch.randn(1000), torch.randn(1000)
    noisy_a = 0.7 * a + 0.05 * torch.randn(1000)
    noisy_b = 1.3 * b + 0.05 * torch.randn(1000)
    assert align_permutations([a, b], [noisy_b, noisy_a]) == [1, 0]


def test_long_input_is_chunked_aligned_and_stitched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A long input is windowed, aligned across windows, and overlap-added back to full length.

    Exercises the whole host-side chunking path in ``separate_with_unasdiff`` without a real
    venv: ``subprocess.run`` is replaced with a fake worker that writes synthetic per-window,
    per-source audio to the exact paths the driver asked for, so this proves the windowing
    arithmetic, the alignment call, and the overlap-add stitching are wired together correctly
    -- not that the sampler itself produces good separations, which needs the real venv and GPU
    (see the skip-gated end-to-end test elsewhere in this module).
    """
    import types

    from senselab.audio.tasks.source_separation import unasdiff as u

    monkeypatch.setattr(u, "ensure_venv", lambda *a, **k: __import__("pathlib").Path("/tmp/fake-unasdiff-venv"))
    monkeypatch.setattr(u, "venv_python", lambda venv_dir: "python3")

    def fake_run(
        cmd: list, *, input: str, capture_output: bool, text: bool, timeout: int, env: dict
    ) -> "types.SimpleNamespace":
        payload = __import__("json").loads(input)
        for paths in payload["out_paths"]:
            for p in paths:
                segment = torch.randn(int(u._WINDOW_S * u._TARGET_SR))
                Audio(waveform=segment.unsqueeze(0), sampling_rate=u._TARGET_SR).save_to_file(p)
        return types.SimpleNamespace(
            returncode=0,
            stdout=__import__("json").dumps({"output_paths": payload["out_paths"]}),
            stderr="",
        )

    monkeypatch.setattr(u.subprocess, "run", fake_run)

    n_samples = int(6.0 * u._TARGET_SR)  # longer than one 4 s window -> exactly two windows
    long_audio = Audio(waveform=torch.randn(1, n_samples), sampling_rate=u._TARGET_SR)

    result = u.separate_with_unasdiff(
        [long_audio],
        n_sources=2,
        source_class_indices=[0, 0],
        mode="speech_speech",
        checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
    )

    assert len(result) == 1
    assert len(result[0]) == 2
    for source in result[0]:
        assert source.sampling_rate == u._TARGET_SR
        assert source.waveform.shape[-1] == n_samples
        # Two windows -> exactly one boundary -> one margin, carried for the caller to inspect
        # rather than gated on (see data/permutation_alignment.json's derivation).
        assert len(source.metadata["unasdiff_alignment_margins"]) == 1
