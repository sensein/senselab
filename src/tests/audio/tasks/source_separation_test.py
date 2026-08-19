"""unasdiff source separation — API contract and class-space handling."""

import ast
import json
import subprocess
import types
from pathlib import Path

import pytest
import soundfile
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.source_separation import separate_audios, unasdiff
from senselab.audio.tasks.source_separation.api import resolve_source_classes
from senselab.audio.tasks.source_separation.unasdiff import align_permutations
from senselab.utils.data_structures import DeviceType, HFModel
from senselab.utils.subprocess_venv import _cache_dir_path


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _stub_worker(monkeypatch: pytest.MonkeyPatch, captured: dict) -> types.ModuleType:
    """Replace the venv and the worker subprocess with a fake that records what the host sent.

    Args:
        monkeypatch: The test's monkeypatch fixture.
        captured: Filled in with ``payload``, ``timeout``, ``in_subtypes`` and ``in_peak``.

    Returns:
        The ``unasdiff`` module, with ``ensure_venv``, ``venv_python`` and ``subprocess.run``
        stubbed for the duration of the test.
    """
    from senselab.audio.tasks.source_separation import unasdiff as u

    monkeypatch.setattr(u, "ensure_venv", lambda *a, **k: Path("/tmp/fake-unasdiff-venv"))
    monkeypatch.setattr(u, "venv_python", lambda venv_dir: "python3")

    def fake_run(
        cmd: list, *, input: str, capture_output: bool, text: bool, timeout: float, env: dict
    ) -> types.SimpleNamespace:
        payload = json.loads(input)
        captured["payload"] = payload
        captured["timeout"] = timeout
        captured["in_subtypes"] = [soundfile.info(p).subtype for p in payload["in_paths"]]
        captured["in_peak"] = max(abs(soundfile.read(p, dtype="float32")[0]).max() for p in payload["in_paths"])
        for paths in payload["out_paths"]:
            for p in paths:
                segment = torch.randn(int(u._WINDOW_S * u._TARGET_SR))
                soundfile.write(p, segment.numpy(), u._TARGET_SR, subtype="FLOAT")
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"output_paths": payload["out_paths"]}), stderr="")

    monkeypatch.setattr(u.subprocess, "run", fake_run)
    return u


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


def test_flash_attn_env_var_is_unset_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the env var unset, the effective requirements match the base list exactly."""
    monkeypatch.delenv(unasdiff._UNASDIFF_FLASH_ATTN_ENV, raising=False)
    assert unasdiff._unasdiff_requirements() == unasdiff._UNASDIFF_REQUIREMENTS


def test_flash_attn_env_var_opts_flash_attn_into_the_requirements(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting SENSELAB_UNASDIFF_FLASH_ATTN truthy appends flash-attn to the venv's requirements.

    Opt-in, not unconditional: this branch already watched av==14.4.0 (no wheel) fall back to a
    source build and take an entire venv install down with it, and flash-attn is considerably
    more build-fragile than that (matching CUDA toolkit, --no-build-isolation, 10-30 minutes of
    MAX_JOBS-tuned compilation). Installing it unconditionally would convert upstream's graceful
    ImportError fallback into a hard venv-creation failure on any host without a working nvcc.
    """
    monkeypatch.setenv(unasdiff._UNASDIFF_FLASH_ATTN_ENV, "1")
    assert "flash-attn==2.5.8" in unasdiff._unasdiff_requirements()


def test_flash_attn_env_var_changes_venv_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Toggling the env var changes the requirements list that ensure_venv keys reuse on.

    ensure_venv's marker comparison is `stored["requirements"] == sorted(requirements)`
    (subprocess_venv.py), so a set-vs-unset environment must resolve to two different
    requirements lists -- otherwise flipping the flag would silently reuse whichever venv
    happened to be cached instead of forcing the rebuild the new dependency needs.
    """
    monkeypatch.delenv(unasdiff._UNASDIFF_FLASH_ATTN_ENV, raising=False)
    unset_requirements = unasdiff._unasdiff_requirements()

    monkeypatch.setenv(unasdiff._UNASDIFF_FLASH_ATTN_ENV, "true")
    set_requirements = unasdiff._unasdiff_requirements()

    assert set_requirements != unset_requirements
    assert sorted(set_requirements) != sorted(unset_requirements)


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


def test_irregular_final_window_aligns_on_its_true_shared_region(monkeypatch: pytest.MonkeyPatch) -> None:
    """The flush-to-end final window overlaps by more than one hop, and alignment must follow it.

    ``_window_starts`` appends a last window flush against the end of the signal, so its overlap
    with its predecessor is wider than ``hop_samples``: 4.92 s of audio gives ``starts=[0, 14720]``,
    a 49280-sample shared region rather than the regular 32000. Slicing a fixed ``overlap_samples``
    off each side compares two regions offset by 17280 samples -- different audio -- and because
    the score is a dot product after normalisation, content shared at *different indices*
    contributes nothing. Every permutation then scores ~0 and the margin carries no information
    about which one is right.

    The fake worker returns exact slices of two known sources with the final window's two slots
    deliberately swapped, which is precisely what alignment exists to undo. On the true shared
    region the correct permutation wins by ~2.0 and the stitched output reproduces both sources;
    on a misaligned region the margin collapses into the ambiguous band and the swap survives.
    """
    import types

    from senselab.audio.tasks.source_separation import unasdiff as u

    monkeypatch.setattr(u, "ensure_venv", lambda *a, **k: __import__("pathlib").Path("/tmp/fake-unasdiff-venv"))
    monkeypatch.setattr(u, "venv_python", lambda venv_dir: "python3")

    window_samples = int(u._WINDOW_S * u._TARGET_SR)
    hop_samples = window_samples - int(u._OVERLAP_S * u._TARGET_SR)
    n_samples = 78720  # 4.92 s -- one regular window plus a flush-to-end final one
    starts = u._window_starts(n_samples, window_samples, hop_samples)
    assert starts == [0, n_samples - window_samples], "test premise: two windows, the last one flush"
    assert starts[1] != hop_samples, "test premise: the final window is irregularly spaced"

    torch.manual_seed(0)
    # Scaled well inside full scale: these go through a real WAV round-trip, and unit-variance
    # noise clips on write, which would cap the reconstruction check near 0.95 for a reason
    # having nothing to do with alignment.
    truth = [0.1 * torch.randn(n_samples), 0.1 * torch.randn(n_samples)]

    def fake_run(
        cmd: list, *, input: str, capture_output: bool, text: bool, timeout: int, env: dict
    ) -> "types.SimpleNamespace":
        payload = __import__("json").loads(input)
        for w, paths in enumerate(payload["out_paths"]):
            # The final window comes back with its slots swapped -- the permutation alignment
            # has to detect and undo. Every earlier window keeps the reference order.
            order = [1, 0] if w == len(starts) - 1 else [0, 1]
            for s, p in enumerate(paths):
                segment = truth[order[s]][starts[w] : starts[w] + window_samples]
                Audio(waveform=segment.unsqueeze(0), sampling_rate=u._TARGET_SR).save_to_file(p)
        return types.SimpleNamespace(
            returncode=0,
            stdout=__import__("json").dumps({"output_paths": payload["out_paths"]}),
            stderr="",
        )

    monkeypatch.setattr(u.subprocess, "run", fake_run)

    audio = Audio(waveform=torch.randn(1, n_samples), sampling_rate=u._TARGET_SR)
    result = u.separate_with_unasdiff(
        [audio],
        n_sources=2,
        source_class_indices=[0, 0],
        mode="speech_speech",
        checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
    )

    margin = result[0][0].metadata["unasdiff_alignment_margins"][0]
    assert margin > 1.0, f"margin {margin} -- the compared regions do not correspond to the same audio"

    # The swap was undone, so each stitched source follows one truth signal for its whole
    # length; leaving it in place would cross the two over from sample 14720 onward.
    for s, source in enumerate(result[0]):
        stitched = source.waveform.squeeze(0)
        assert torch.corrcoef(torch.stack([stitched, truth[s]]))[0, 1] > 0.99


def test_diffusion_steps_defaults_to_the_dead_constants_value() -> None:
    """separate_with_unasdiff's default matches the module constant, not a second hardcoded 200.

    Guards against the two falling out of sync the way _DIFFUSION_STEPS and the worker's own
    literal 200 previously did -- this constant is now the single source of truth for the default.
    """
    import inspect

    default = inspect.signature(unasdiff.separate_with_unasdiff).parameters["diffusion_steps"].default
    assert default == unasdiff._DIFFUSION_STEPS == 200


def test_diffusion_steps_must_be_positive() -> None:
    """A non-positive diffusion_steps must raise before reaching the sampler.

    Handing 0 or a negative count to the worker would either fail deep inside the diffusion
    library with an unhelpful traceback or -- worse -- silently produce degenerate output; the
    host validates before any venv/worker machinery runs.
    """
    audio = Audio(waveform=torch.randn(1, 16000), sampling_rate=16000)
    with pytest.raises(ValueError, match="diffusion_steps"):
        unasdiff.separate_with_unasdiff(
            [audio],
            n_sources=2,
            source_class_indices=[0, 0],
            mode="speech_speech",
            diffusion_steps=0,
        )
    with pytest.raises(ValueError, match="diffusion_steps"):
        unasdiff.separate_with_unasdiff(
            [audio],
            n_sources=2,
            source_class_indices=[0, 0],
            mode="speech_speech",
            diffusion_steps=-5,
        )


def test_diffusion_steps_reaches_the_worker_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-supplied diffusion_steps threads through to the worker's JSON payload.

    The worker reads args["diffusion_steps"] instead of a hardcoded 200 (see unasdiff.py); this
    proves the host actually sends a non-default value rather than the worker silently ignoring it.
    """
    import types

    from senselab.audio.tasks.source_separation import unasdiff as u

    monkeypatch.setattr(u, "ensure_venv", lambda *a, **k: __import__("pathlib").Path("/tmp/fake-unasdiff-venv"))
    monkeypatch.setattr(u, "venv_python", lambda venv_dir: "python3")

    captured: dict = {}

    def fake_run(
        cmd: list, *, input: str, capture_output: bool, text: bool, timeout: int, env: dict
    ) -> "types.SimpleNamespace":
        payload = __import__("json").loads(input)
        captured["diffusion_steps"] = payload["diffusion_steps"]
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

    short_audio = Audio(waveform=torch.randn(1, int(u._WINDOW_S * u._TARGET_SR)), sampling_rate=u._TARGET_SR)
    u.separate_with_unasdiff(
        [short_audio],
        n_sources=2,
        source_class_indices=[0, 0],
        mode="speech_speech",
        checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
        diffusion_steps=17,
    )

    assert captured["diffusion_steps"] == 17


def test_separate_audios_forwards_diffusion_steps(mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch) -> None:
    """api.separate_audios threads diffusion_steps through to separate_with_unasdiff unchanged."""
    captured = {}

    def fake(audios: list, n_sources: int, source_class_indices: list, **kwargs: object) -> list:
        captured["diffusion_steps"] = kwargs["diffusion_steps"]
        return [[audios[0]] * n_sources]

    monkeypatch.setattr("senselab.audio.tasks.source_separation.api.separate_with_unasdiff", fake)
    separate_audios([mono_audio_sample], mode="speech_speech", n_sources=2, diffusion_steps=42)
    assert captured["diffusion_steps"] == 42


@pytest.mark.skipif(
    not ((_cache_dir_path() / "unasdiff").is_dir() and _cuda_available()),
    reason="needs the unasdiff venv and CUDA; the sampler backprops through the "
    "priors at every one of 200 steps, so CPU is impractical",
)
def test_unasdiff_separates_a_mixture_into_n_sources(mono_audio_sample: Audio) -> None:
    """Shape and energy, not quality.

    A separation that returns the mixture in every slot passes a shape check, so the
    energy-difference assertion is the one that can actually fail.
    """
    from senselab.audio.tasks.preprocessing import resample_audios

    audio = resample_audios([mono_audio_sample], resample_rate=16000)[0]
    result = separate_audios([audio], mode="speech_sound", n_sources=2, source_classes=["Applause"], seed=17)

    assert len(result) == 1 and len(result[0]) == 2
    for source in result[0]:
        assert source.sampling_rate == 16000
        assert source.waveform.shape[-1] == audio.waveform.shape[-1]

    a, b = result[0][0].waveform, result[0][1].waveform
    assert (a - b).abs().mean() > 1e-4, "both slots returned the same signal"


# ── Worker device selection ───────────────────────────────────────────

_CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


def _reads_os_environ(node: ast.AST) -> bool:
    """True if ``node`` is the expression ``os.environ``."""
    return isinstance(node, ast.Attribute) and node.attr == "environ" and isinstance(node.value, ast.Name)


def _cuda_visible_devices_landmarks(script: str) -> dict:
    """Line numbers of the four events the worker's device handling must order correctly.

    Args:
        script: The worker script source.

    Returns:
        ``{"saves": [...], "restores": [...], "upstream_imports": [...], "cuda_api": [...]}``,
        each a list of line numbers in the order ``ast.walk`` yields them.
    """
    tree = ast.parse(script)
    landmarks: dict = {"saves": [], "restores": [], "upstream_imports": [], "cuda_api": []}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _reads_os_environ(node.func.value):
            named = node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == _CUDA_VISIBLE_DEVICES
            if named and node.func.attr == "get":
                landmarks["saves"].append(node.lineno)
            elif named and node.func.attr == "pop":
                landmarks["restores"].append(node.lineno)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and _reads_os_environ(target.value)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == _CUDA_VISIBLE_DEVICES
                ):
                    landmarks["restores"].append(node.lineno)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("models", "diffusion"):
                    landmarks["upstream_imports"].append(node.lineno)
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "cuda"
            and isinstance(node.value, ast.Name)
            and node.value.id == "torch"
        ):
            landmarks["cuda_api"].append(node.lineno)
    return landmarks


def test_the_worker_restores_cuda_visible_devices_before_it_touches_cuda() -> None:
    """The upstream module-scope GPU pin is saved before the import and put back before any CUDA call.

    ``models/atten_unet.py`` assigns ``CUDA_VISIBLE_DEVICES = "0"`` at module scope, ahead of its
    own ``import torch``. CUDA initialises lazily, so restoring the launcher's value after the
    import but before the first CUDA API call is what makes the pin have no effect. This is a
    static ordering check because the pin's effect is only observable on a multi-GPU host.
    """
    marks = _cuda_visible_devices_landmarks(unasdiff._WORKER_SCRIPT)
    assert marks["saves"], "the worker never reads CUDA_VISIBLE_DEVICES before the upstream import"
    assert marks["restores"], "the worker never restores CUDA_VISIBLE_DEVICES after the upstream import"
    assert marks["upstream_imports"], "test premise: the worker imports the upstream modules"
    assert marks["cuda_api"], "test premise: the worker calls into torch.cuda"

    assert min(marks["saves"]) < min(marks["upstream_imports"]), "the save must precede the pinning import"
    assert max(marks["restores"]) > max(marks["upstream_imports"]), "the restore must follow every upstream import"
    assert max(marks["restores"]) < min(marks["cuda_api"]), "the restore must precede the first CUDA API call"


def test_the_worker_never_requests_a_bare_cuda_device() -> None:
    """Every ``torch.device`` the worker builds for CUDA carries an explicit index.

    A bare ``"cuda"`` takes whatever index torch defaults to, which is the outcome the upstream
    pin produced on a four-GPU node.
    """
    tree = ast.parse(unasdiff._WORKER_SCRIPT)
    bare = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "device"):
            continue
        # Every string constant reachable from the argument, so a ternary picking between
        # "cuda" and "cpu" is caught as readily as a plain literal.
        for inner in ast.walk(node.args[0]) if node.args else []:
            if isinstance(inner, ast.Constant) and inner.value == "cuda":
                bare.append(node.lineno)
    assert not bare, f"bare torch.device('cuda') at worker line(s) {sorted(set(bare))}"


def test_the_callers_device_reaches_the_worker_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-selected device is sent to the worker instead of being validated and dropped.

    ``device`` used to be handed to ``_select_device_and_dtype`` purely for validation and its
    result discarded, so the worker chose for itself and no caller could select a card.
    """
    captured: dict = {}
    u = _stub_worker(monkeypatch, captured)

    audio = Audio(waveform=torch.randn(1, int(u._WINDOW_S * u._TARGET_SR)), sampling_rate=u._TARGET_SR)
    u.separate_with_unasdiff(
        [audio],
        n_sources=2,
        source_class_indices=[0, 0],
        mode="speech_speech",
        checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
        device=DeviceType.CPU,
    )
    assert captured["payload"]["device"] == "cpu"


def test_no_device_leaves_the_choice_to_the_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """``device=None`` sends ``None``, not a device the host's own torch build happened to see.

    The host interpreter and the venv have separate torch builds; only the venv's answer to
    ``torch.cuda.is_available()`` governs where the worker can run.
    """
    captured: dict = {}
    u = _stub_worker(monkeypatch, captured)

    audio = Audio(waveform=torch.randn(1, int(u._WINDOW_S * u._TARGET_SR)), sampling_rate=u._TARGET_SR)
    u.separate_with_unasdiff(
        [audio],
        n_sources=2,
        source_class_indices=[0, 0],
        mode="speech_speech",
        checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
    )
    assert captured["payload"]["device"] is None


def test_an_incompatible_device_is_rejected_before_the_venv() -> None:
    """MPS is not one of this backend's compatible devices and must raise rather than fall back."""
    audio = Audio(waveform=torch.randn(1, 16000), sampling_rate=16000)
    with pytest.raises(ValueError):
        unasdiff.separate_with_unasdiff(
            [audio],
            n_sources=2,
            source_class_indices=[0, 0],
            mode="speech_speech",
            device=DeviceType.MPS,
        )


# ── Worker timeout ────────────────────────────────────────────────────


def test_the_default_timeout_scales_with_windows_and_steps() -> None:
    """The ceiling is derived from the work, not a constant.

    A fixed 3600 s ceiling failed every input past roughly 90 s on an A100 — the run that
    exceeded it lost every window.
    """
    floor = unasdiff._default_timeout_s(1, 1)
    assert floor == unasdiff._TIMEOUT_FLOOR_S

    big = unasdiff._default_timeout_s(200, unasdiff._DIFFUSION_STEPS)
    bigger_input = unasdiff._default_timeout_s(400, unasdiff._DIFFUSION_STEPS)
    more_steps = unasdiff._default_timeout_s(200, 2 * unasdiff._DIFFUSION_STEPS)
    assert big > floor
    assert bigger_input == pytest.approx(2 * big)
    assert more_steps == pytest.approx(2 * big)


def test_the_derived_ceiling_reaches_subprocess_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout ``subprocess.run`` receives is the derived one, not a hardcoded constant."""
    captured: dict = {}
    u = _stub_worker(monkeypatch, captured)

    n_samples = int(6.0 * u._TARGET_SR)  # two windows
    audio = Audio(waveform=torch.randn(1, n_samples), sampling_rate=u._TARGET_SR)
    u.separate_with_unasdiff(
        [audio],
        n_sources=2,
        source_class_indices=[0, 0],
        mode="speech_speech",
        checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
        diffusion_steps=9000,
    )
    expected = u._default_timeout_s(2, 9000)
    assert captured["timeout"] == expected
    assert captured["timeout"] != 3600


def test_an_explicit_timeout_overrides_the_derived_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """``timeout_s`` is honoured verbatim."""
    captured: dict = {}
    u = _stub_worker(monkeypatch, captured)

    audio = Audio(waveform=torch.randn(1, int(u._WINDOW_S * u._TARGET_SR)), sampling_rate=u._TARGET_SR)
    u.separate_with_unasdiff(
        [audio],
        n_sources=2,
        source_class_indices=[0, 0],
        mode="speech_speech",
        checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
        timeout_s=42.0,
    )
    assert captured["timeout"] == 42.0


def test_a_non_positive_timeout_raises() -> None:
    """A zero or negative ceiling would abort the worker instantly; reject it up front."""
    audio = Audio(waveform=torch.randn(1, 16000), sampling_rate=16000)
    for bad in (0, -1.0):
        with pytest.raises(ValueError, match="timeout_s"):
            unasdiff.separate_with_unasdiff(
                [audio],
                n_sources=2,
                source_class_indices=[0, 0],
                mode="speech_speech",
                timeout_s=bad,
            )


def test_a_timeout_names_the_ceiling_the_input_and_the_windows_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``TimeoutExpired`` becomes an actionable ``RuntimeError``, not a bare stack trace.

    The unhandled exception said only that a subprocess ran too long: not which ceiling it hit,
    how much audio it was given, or how far it had got.
    """
    import types

    from senselab.audio.tasks.source_separation import unasdiff as u

    monkeypatch.setattr(u, "ensure_venv", lambda *a, **k: Path("/tmp/fake-unasdiff-venv"))
    monkeypatch.setattr(u, "venv_python", lambda venv_dir: "python3")

    def fake_run(cmd: list, *, input: str, capture_output: bool, text: bool, timeout: float, env: dict) -> None:
        payload = json.loads(input)
        # One window completes before the ceiling fires, so the error can report progress.
        for p in payload["out_paths"][0]:
            segment = torch.randn(int(u._WINDOW_S * u._TARGET_SR))
            soundfile.write(p, segment.numpy(), u._TARGET_SR, subtype="FLOAT")
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(u.subprocess, "run", fake_run)

    n_samples = int(6.0 * u._TARGET_SR)  # two windows
    audio = Audio(waveform=torch.randn(1, n_samples), sampling_rate=u._TARGET_SR)
    with pytest.raises(RuntimeError) as exc:
        u.separate_with_unasdiff(
            [audio],
            n_sources=2,
            source_class_indices=[0, 0],
            mode="speech_speech",
            checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
            timeout_s=123.0,
        )

    message = str(exc.value)
    assert "123s" in message, "the ceiling that fired must be named"
    assert "1/2 window(s) written" in message, "progress at the point of failure must be reported"
    assert "6.0s of audio" in message, "the input being processed must be named"
    assert "speech_speech" in message and "diffusion_steps=200" in message
    assert "timeout_s" in message, "the message must name the knob that raises the ceiling"


def test_separate_audios_forwards_timeout_s(mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch) -> None:
    """api.separate_audios threads timeout_s through to separate_with_unasdiff unchanged."""
    captured = {}

    def fake(audios: list, n_sources: int, source_class_indices: list, **kwargs: object) -> list:
        captured["timeout_s"] = kwargs["timeout_s"]
        return [[audios[0]] * n_sources]

    monkeypatch.setattr("senselab.audio.tasks.source_separation.api.separate_with_unasdiff", fake)
    separate_audios([mono_audio_sample], mode="speech_speech", n_sources=2, timeout_s=99.0)
    assert captured["timeout_s"] == 99.0


def test_separate_audios_forwards_device(mono_audio_sample: Audio, monkeypatch: pytest.MonkeyPatch) -> None:
    """api.separate_audios threads device through to separate_with_unasdiff unchanged."""
    captured = {}

    def fake(audios: list, n_sources: int, source_class_indices: list, **kwargs: object) -> list:
        captured["device"] = kwargs["device"]
        return [[audios[0]] * n_sources]

    monkeypatch.setattr("senselab.audio.tasks.source_separation.api.separate_with_unasdiff", fake)
    separate_audios([mono_audio_sample], mode="speech_speech", n_sources=2, device=DeviceType.CPU)
    assert captured["device"] is DeviceType.CPU


# ── WAV intermediates ─────────────────────────────────────────────────


def test_input_windows_are_written_as_float_not_pcm16(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window files the host hands the worker are FLOAT, and samples past +-1 survive.

    soundfile's WAV default is PCM_16, which clips every such sample before the worker reads it.
    """
    captured: dict = {}
    u = _stub_worker(monkeypatch, captured)

    window_samples = int(u._WINDOW_S * u._TARGET_SR)
    waveform = torch.zeros(1, window_samples)
    waveform[0, 100] = 1.75
    u.separate_with_unasdiff(
        [Audio(waveform=waveform, sampling_rate=u._TARGET_SR)],
        n_sources=2,
        source_class_indices=[0, 0],
        mode="speech_speech",
        checkpoint_dir="/tmp/fake-unasdiff-checkpoints",
    )

    assert captured["in_subtypes"] == ["FLOAT"]
    # PCM_16 caps at 1.0; the exact peak is not asserted because resample_audios filters even at
    # an unchanged rate, which rounds the impulse off.
    assert captured["in_peak"] > 1.5, "an out-of-range sample was clipped on write"


def test_every_worker_wav_write_names_an_explicit_subtype() -> None:
    """No ``sf.write`` in the worker relies on soundfile's PCM_16 default.

    That default has silently corrupted a measurement three times in this repository.
    """
    tree = ast.parse(unasdiff._WORKER_SCRIPT)
    writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write"
    ]
    assert writes, "test premise: the worker writes WAV files"
    for node in writes:
        keywords = {kw.arg for kw in node.keywords}
        assert "subtype" in keywords, f"sf.write at worker line {node.lineno} relies on the PCM_16 default"
