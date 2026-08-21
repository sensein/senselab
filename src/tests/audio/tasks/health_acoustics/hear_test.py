"""HeAR task contract: the 2 s rule, the no-padding rule, the pin, the scan grid, the labels.

Everything here except the last two tests runs without TensorFlow, without the venv and without
Hub access: the worker subprocess is replaced by a stub that records exactly what the parent sent
and fabricates arrays of the right shape. That is deliberate — the properties being guarded are
properties of the *parent*, and the two that need the real model say so and skip.
"""

from __future__ import annotations

import json
import re
import subprocess
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import soundfile
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics import (
    HEAR_EVENT_LABELS,
    HearEmbeddings,
    centred_cosine_similarity,
    detect_health_acoustic_events,
    extract_hear_embeddings_at_times,
    extract_hear_embeddings_from_audios,
    hear,
)
from senselab.utils.subprocess_venv import _cache_dir_path

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SR = hear.HEAR_SAMPLING_RATE
WIN = hear.HEAR_WINDOW_SAMPLES


def _ramp_audio(n_samples: int, sampling_rate: int = SR) -> Audio:
    """An audio whose every sample is distinct, so a padded window is detectable by value.

    A constant or random signal would make "these samples came from the recording" hard to assert;
    a strictly increasing ramp makes each window's content its own fingerprint.
    """
    waveform = torch.arange(1, n_samples + 1, dtype=torch.float32) / (n_samples + 1)
    return Audio(waveform=waveform.unsqueeze(0), sampling_rate=sampling_rate)


def _stub_worker(monkeypatch: pytest.MonkeyPatch, captured: Dict[str, Any], out_dim: int) -> None:
    """Replace venv provisioning, model staging and the worker subprocess with a recorder."""
    monkeypatch.setattr(hear, "ensure_venv", lambda *a, **k: Path("/tmp/fake-hear-venv"))
    monkeypatch.setattr(hear, "venv_python", lambda venv_dir: "python3")
    monkeypatch.setattr(
        hear,
        "stage_hear_snapshot",
        lambda: (hear.HEAR_REVISION, Path("/tmp/fake-hf-cache/snapshots") / hear.HEAR_REVISION),
    )

    def fake_run(
        cmd: List[str], *, input: str, capture_output: bool, text: bool, timeout: float, env: Dict[str, str]
    ) -> types.SimpleNamespace:
        payload = json.loads(input)
        captured["payload"] = payload
        captured["env"] = env
        captured["windows"] = []
        results = []
        for job in payload["jobs"]:
            data, sr = soundfile.read(job["wav"], dtype="float32", always_2d=False)
            captured.setdefault("sampling_rates", []).append(sr)
            captured.setdefault("subtypes", []).append(soundfile.info(job["wav"]).subtype)
            captured["n_samples"] = data.shape[0]
            for start in job["starts"]:
                captured["windows"].append(data[start : start + payload["window_samples"]].copy())
            array = np.arange(len(job["starts"]) * out_dim, dtype=np.float32).reshape(len(job["starts"]), out_dim)
            np.save(job["out"], array)
            results.append({"out": job["out"], "shape": list(array.shape)})
        return types.SimpleNamespace(returncode=0, stdout=json.dumps({"results": results, "batch": 1}), stderr="")

    monkeypatch.setattr(hear.subprocess, "run", fake_run)


# ── The pin ───────────────────────────────────────────────────────────


def test_the_revision_is_a_pinned_commit_not_a_ref() -> None:
    """A ref would let an upstream push change the weights under an unchanged provenance record."""
    assert SHA_RE.match(hear.HEAR_REVISION), f"{hear.HEAR_REVISION!r} is not a 40-hex commit"
    assert hear.HEAR_REVISION == "9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
    assert hear.HEAR_MODEL_ID == "google/hear"


def test_the_worker_is_pinned_by_the_staged_path_and_never_told_a_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker gets a ``snapshots/<sha>`` directory, not a repo id and not a revision string.

    ``tf.saved_model.load`` takes a local directory, so the commit is pinned by the path. What
    must not happen is a revision-shaped field leaking into the payload as a *ref*: the payload
    carries no revision at all, which is why this file sits in
    ``LOADER_CANNOT_PIN_SUBPROCESS_FILES`` in ``revision_pinning_guard_test.py``.
    """
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=hear.HEAR_EMBEDDING_DIM)
    extract_hear_embeddings_from_audios([_ramp_audio(3 * SR)])

    payload = captured["payload"]
    assert hear.HEAR_REVISION in payload["saved_model_dir"]
    assert not [key for key in payload if "revision" in key.lower()]
    assert not [key for key in payload if "model_id" in key.lower()]


def test_staging_goes_through_the_shared_gated_token_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gated repo must use senselab's existing token mechanism, not a HeAR-specific one."""
    seen: Dict[str, Any] = {}

    def fake_resolve_model(repo_id: str, revision: str = "main", *, token: Optional[str] = None) -> tuple[str, Path]:
        seen.update(repo_id=repo_id, revision=revision, token=token)
        return revision, Path("/tmp/fake") / revision

    monkeypatch.setattr(hear, "resolve_model", fake_resolve_model)
    monkeypatch.setattr(hear, "get_huggingface_token", lambda: "hf_dummy")

    sha, path = hear.stage_hear_snapshot()
    assert seen == {"repo_id": "google/hear", "revision": hear.HEAR_REVISION, "token": "hf_dummy"}
    assert sha == hear.HEAR_REVISION and str(path).endswith(hear.HEAR_REVISION)


# ── The 2 s rule ──────────────────────────────────────────────────────


@pytest.mark.parametrize("duration_s", [0.3, 1.0, 1.5, 1.999])
def test_audio_shorter_than_two_seconds_is_refused_not_padded(duration_s: float) -> None:
    """The encoder would accept these silently; that is exactly why the API must not.

    The error has to say *why*, because "too short" invites the caller to pad, which is the
    measured-wrong repair (centred cosine 0.0-0.5 against a ~0.9 class margin).
    """
    audio = _ramp_audio(int(duration_s * SR))
    with pytest.raises(ValueError, match="pad") as excinfo:
        extract_hear_embeddings_from_audios([audio])
    message = str(excinfo.value)
    assert "32000" in message and "2.0s" in message
    assert "extract_hear_embeddings_at_times" in message, "the error must point at the correct repair"


def test_a_short_recording_is_refused_by_the_detector_too() -> None:
    """Same rule on the detection path: its graph rejects non-2 s input outright."""
    with pytest.raises(ValueError, match="pad"):
        detect_health_acoustic_events([_ramp_audio(SR)])


def test_shortness_is_judged_after_resampling() -> None:
    """A 1.5 s file at 48 kHz has 72000 samples but only 24000 at 16 kHz — still too short."""
    audio = _ramp_audio(int(1.5 * 48000), sampling_rate=48000)
    with pytest.raises(ValueError, match="32000"):
        extract_hear_embeddings_from_audios([audio])


def test_exactly_two_seconds_gives_exactly_one_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boundary case must pass, and must not produce a second, padded window."""
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=hear.HEAR_EMBEDDING_DIM)
    [result] = extract_hear_embeddings_from_audios([_ramp_audio(WIN)])
    assert result.embeddings.shape == (1, hear.HEAR_EMBEDDING_DIM)
    assert result.window_starts == [0.0]


# ── No padding, ever ──────────────────────────────────────────────────


def test_every_window_handed_to_the_worker_is_real_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full-length windows whose samples are the recording's own — no zeros appended anywhere.

    Asserted on values, not just lengths: a padded window is full-length too. The ramp fixture
    makes each sample unique, so a window is only verifiable as real by matching the source slice.
    """
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=hear.HEAR_EMBEDDING_DIM)
    n_samples = int(4.7 * SR)  # deliberately not a whole number of hops
    audio = _ramp_audio(n_samples)
    [result] = extract_hear_embeddings_from_audios([audio], hop_length=1.0)

    source = audio.waveform.squeeze(0).numpy()
    assert captured["windows"], "no windows were sent"
    for window, start in zip(captured["windows"], result.window_starts):
        offset = int(round(start * SR))
        assert window.shape == (WIN,)
        assert np.allclose(window, source[offset : offset + WIN], atol=1e-6)
        assert np.count_nonzero(window) == WIN, "a zero sample means padding crept in"


def test_the_tail_is_covered_by_a_full_window_not_a_short_one() -> None:
    """The last window ends at the final sample, so no tail is skipped and none is padded."""
    n_samples = int(4.7 * SR)
    starts = hear.plan_scan_windows(n_samples, SR)
    assert starts[0] == 0
    assert starts[-1] == n_samples - WIN
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    assert all(0 <= s and s + WIN <= n_samples for s in starts)


def test_the_scan_grid_is_the_hop_grid_plus_a_snapped_tail() -> None:
    """Window/hop arithmetic, spelled out: a 10 s recording at a 0.5 s hop."""
    starts = hear.plan_scan_windows(10 * SR, SR // 2)
    # 0, 0.5, ..., 8.0 s -> 17 windows; the last already ends at 10 s, so nothing is appended.
    assert starts == [i * (SR // 2) for i in range(17)]
    assert starts[-1] + WIN == 10 * SR


def test_a_hop_that_divides_the_remainder_exactly_adds_no_duplicate_window() -> None:
    """The snapped tail must not duplicate a window that is already on the grid."""
    n_samples = 5 * SR
    starts = hear.plan_scan_windows(n_samples, SR)
    assert starts == [0, SR, 2 * SR, 3 * SR]
    assert len(set(starts)) == len(starts)


def test_a_hop_wider_than_the_window_warns_about_the_blind_gap() -> None:
    """Legal, but it leaves audio no window ever sees; silence there is unobserved, not absent."""
    with pytest.warns(UserWarning, match="never seen by the model"):
        hear.seconds_to_hop_samples(3.0)


@pytest.mark.parametrize("hop_length", [0, -1.0, 1e-6])
def test_an_unusable_hop_is_rejected(hop_length: float) -> None:
    """Zero, negative, and "rounds to zero samples" are all caller errors, not silent no-ops."""
    with pytest.raises(ValueError):
        hear.seconds_to_hop_samples(hop_length)


def test_centred_windows_slide_inward_at_the_edges_rather_than_padding() -> None:
    """An event 0.2 s in cannot be centred; the window moves, and says where it went."""
    n_samples = 10 * SR
    starts = hear.plan_centred_windows(n_samples, [int(0.2 * SR), 5 * SR, int(9.9 * SR)])
    assert starts[0] == 0, "the leading edge clamps to the start of the recording"
    assert starts[1] == 5 * SR - WIN // 2, "an interior event is centred"
    assert starts[2] == n_samples - WIN, "the trailing edge clamps to the end"
    assert all(0 <= s and s + WIN <= n_samples for s in starts)


def test_a_time_outside_the_recording_is_an_error() -> None:
    """Clamping a time that is not in the recording would silently analyse the wrong audio."""
    with pytest.raises(ValueError, match="outside the recording"):
        hear.plan_centred_windows(10 * SR, [11 * SR])


def test_per_event_extraction_reports_the_window_it_actually_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """The requested times are kept, and the realised starts are reported alongside them."""
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=hear.HEAR_EMBEDDING_DIM)
    audio = _ramp_audio(10 * SR)
    result = extract_hear_embeddings_at_times(audio, times=[0.1, 5.0])

    assert result.metadata["requested_times"] == [0.1, 5.0]
    assert result.window_starts == [0.0, 5.0 - hear.HEAR_WINDOW_SECONDS / 2]
    assert result.hop_seconds is None, "windows placed by centre have no single hop"
    assert result.embeddings.shape == (2, hear.HEAR_EMBEDDING_DIM)
    for window in captured["windows"]:
        assert window.shape == (WIN,) and np.count_nonzero(window) == WIN


def test_empty_times_is_an_error() -> None:
    """Returning an empty result would let a caller's loop silently do nothing."""
    with pytest.raises(ValueError, match="at least one time"):
        extract_hear_embeddings_at_times(_ramp_audio(3 * SR), times=[])


# ── Hand-off to the worker ────────────────────────────────────────────


def test_the_worker_receives_16khz_float_wavs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resampled to HeAR's rate, and written lossless.

    ``FLOAT`` rather than ``PCM_16`` for the reason measured in ``classification/yamnet.py``:
    16-bit quantization noise is louder than a -100 dBFS signal, and HeAR's subject matter (quiet
    breaths, throat clears) lives at the faint end.
    """
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=hear.HEAR_EMBEDDING_DIM)
    extract_hear_embeddings_from_audios([_ramp_audio(3 * 48000, sampling_rate=48000)])

    assert captured["sampling_rates"] == [SR]
    assert captured["subtypes"] == ["FLOAT"]
    assert captured["payload"]["window_samples"] == WIN


def test_a_cpu_request_hides_the_gpus_from_tensorflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """TensorFlow has no per-call device argument, so CPU is enforced through the environment."""
    from senselab.utils.data_structures import DeviceType

    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=hear.HEAR_EMBEDDING_DIM)
    extract_hear_embeddings_from_audios([_ramp_audio(3 * SR)], device=DeviceType.CPU)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "-1"


def test_the_detector_payload_declares_batch_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The detector's signature pins the batch dimension at 1; batching it raises, not degrades."""
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=len(HEAR_EVENT_LABELS))
    detect_health_acoustic_events([_ramp_audio(3 * SR)])
    assert captured["payload"]["batch_size"] == 1


def test_the_worker_script_rechecks_window_bounds() -> None:
    """The no-padding rule is enforced on both sides of the subprocess boundary."""
    assert "never padded" in hear._HEAR_WORKER
    assert "np.pad" not in hear._HEAR_WORKER and "np.zeros" not in hear._HEAR_WORKER


def test_run_hear_rejects_mismatched_plans() -> None:
    """A plan list out of step with the audio list would silently analyse the wrong recording."""
    with pytest.raises(ValueError, match="window plans"):
        hear.run_hear([_ramp_audio(3 * SR)], [], subdir=hear.ENCODER_SUBDIR)


def test_run_hear_rejects_audio_that_was_not_resampled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backend takes 16 kHz only; the API resamples, so reaching here at 48 kHz is a bug."""
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=hear.HEAR_EMBEDDING_DIM)
    audio = _ramp_audio(3 * 48000, sampling_rate=48000)
    with pytest.raises(ValueError, match="16000 Hz"):
        hear.run_hear([audio], [[0]], subdir=hear.ENCODER_SUBDIR)


# ── The detector's labels and semantics ───────────────────────────────


def test_the_label_set_is_upstreams_eight_in_graph_order() -> None:
    """The tuple index is the output column, so order is not ours to change."""
    assert HEAR_EVENT_LABELS == (
        "Cough",
        "Snore",
        "Baby Cough",
        "Breathe",
        "Sneeze",
        "Throat Clear",
        "Laugh",
        "Speech",
    )
    assert len(HEAR_EVENT_LABELS) == 8


def test_detection_windows_carry_all_eight_labels_and_their_own_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-label output: every label is reported, with window bounds that are window bounds."""
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=len(HEAR_EVENT_LABELS))
    [windows] = detect_health_acoustic_events([_ramp_audio(5 * SR)], hop_length=0.25)

    assert len(windows) == len(captured["windows"])
    for window in windows:
        labels = [next(iter(entry)) for entry in window["label_scores"]]
        assert set(labels) == set(HEAR_EVENT_LABELS)
        scores = [next(iter(entry.values())) for entry in window["label_scores"]]
        assert scores == sorted(scores, reverse=True), "label_scores must be descending"
        assert window["win_length"] == hear.HEAR_WINDOW_SECONDS
        assert window["hop_length"] == 0.25
        assert window["end"] - window["start"] == pytest.approx(hear.HEAR_WINDOW_SECONDS)


def test_top_k_trims_the_label_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in only: for a multi-label gate, dropping labels drops negative evidence too."""
    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=len(HEAR_EVENT_LABELS))
    [windows] = detect_health_acoustic_events([_ramp_audio(3 * SR)], top_k=2)
    assert all(len(w["label_scores"]) == 2 for w in windows)


def test_detection_output_feeds_the_plotting_segment_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window dicts are the shape the classification task's segment converter consumes."""
    from senselab.audio.tasks.classification import scene_results_to_segments

    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=len(HEAR_EVENT_LABELS))
    [windows] = detect_health_acoustic_events([_ramp_audio(3 * SR)])
    segments = scene_results_to_segments(windows)
    assert segments and all(segment["label"] in HEAR_EVENT_LABELS for segment in segments)


def test_the_detector_is_reachable_through_classify_audios(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dispatch alias in the classification task must reach this backend, hop included."""
    from senselab.audio.tasks.classification import classify_audios

    captured: Dict[str, Any] = {}
    _stub_worker(monkeypatch, captured, out_dim=len(HEAR_EVENT_LABELS))
    result = classify_audios([_ramp_audio(4 * SR)], model="hear-event-detector", win_length=3.0, hop_length=0.5)
    [windows] = result
    assert windows[0]["win_length"] == hear.HEAR_WINDOW_SECONDS, "win_length must be ignored, not honoured"
    assert windows[0]["hop_length"] == 0.5, "hop_length must be honoured"


@pytest.mark.parametrize(
    "spec,subdir",
    [
        ("hear-event-detector", "event_detector/event_detector_large"),
        ("hear-events", "event_detector/event_detector_large"),
        ("hear-event-detector-large", "event_detector/event_detector_large"),
        ("hear-event-detector-small", "event_detector/event_detector_small"),
    ],
)
def test_detector_specs_map_to_the_right_saved_model(spec: str, subdir: str) -> None:
    """Both bundled detectors are reachable, and named rather than sized by a boolean."""
    assert hear.resolve_event_detector(spec) == subdir


def test_an_unknown_model_spec_is_refused_on_both_paths() -> None:
    """A typo must not silently fall through to the other capability's model."""
    with pytest.raises(ValueError, match="not a HeAR event detector"):
        hear.resolve_event_detector("hear")
    with pytest.raises(ValueError, match="not the HeAR encoder"):
        extract_hear_embeddings_from_audios([_ramp_audio(3 * SR)], model="hear-event-detector")


# ── Similarity ────────────────────────────────────────────────────────


def test_similarity_is_centred_so_the_shared_component_cannot_dominate() -> None:
    """Reproduces in miniature the measurement that makes raw cosine useless.

    Two vectors that differ in their informative part but share a large common component read as
    nearly identical uncentred (the shared part dominates the dot product) and separate once it is
    removed. On real HeAR embeddings this is 0.977 vs 0.918 raw against +0.653 vs -0.256 centred.
    """
    dim = hear.HEAR_EMBEDDING_DIM
    shared = torch.ones(dim) * 10.0
    a = shared.clone()
    a[0] += 1.0
    b = shared.clone()
    b[1] += 1.0
    stacked = torch.stack([a, b])

    raw = torch.nn.functional.cosine_similarity(a, b, dim=0)
    centred = centred_cosine_similarity(stacked)

    assert raw > 0.99, "the raw number is the uninformative one this helper exists to avoid"
    assert centred[0, 1] < 0.0 < centred[0, 0]
    assert centred.shape == (2, 2)


def test_similarity_accepts_a_result_object_and_an_explicit_mean() -> None:
    """A query set may be centred by a pool's mean rather than its own."""
    dim = hear.HEAR_EMBEDDING_DIM
    embeddings = HearEmbeddings(embeddings=torch.randn(4, dim), window_starts=[0.0, 1.0, 2.0, 3.0])
    pool_mean = torch.randn(dim)
    matrix = centred_cosine_similarity(embeddings, mean=pool_mean)
    assert matrix.shape == (4, 4)
    assert torch.allclose(torch.diagonal(matrix), torch.ones(4), atol=1e-5)

    against = centred_cosine_similarity(embeddings, torch.randn(2, dim))
    assert against.shape == (4, 2)


def test_similarity_refuses_to_invent_a_mean_from_one_vector() -> None:
    """Centring one vector by itself zeroes it, making every similarity undefined."""
    with pytest.raises(ValueError, match="at least two vectors"):
        centred_cosine_similarity(torch.randn(1, hear.HEAR_EMBEDDING_DIM))


def test_similarity_checks_the_embedding_width() -> None:
    """A 768-d tensor here means someone passed another model's embeddings."""
    with pytest.raises(ValueError, match="must be"):
        centred_cosine_similarity(torch.randn(3, 768))


def test_pooling_is_the_window_mean() -> None:
    """The file-level summary PR #366 produced, kept as an explicit method rather than a default."""
    result = HearEmbeddings(
        embeddings=torch.stack([torch.zeros(hear.HEAR_EMBEDDING_DIM), torch.ones(hear.HEAR_EMBEDDING_DIM) * 2]),
        window_starts=[0.0, 1.0],
    )
    assert torch.allclose(result.pooled(), torch.ones(hear.HEAR_EMBEDDING_DIM))


# ── Real model, real venv: skipped unless both are already present ────


def _hear_venv_exists() -> bool:
    return (_cache_dir_path() / "hear" / ".senselab-installed").is_file()


def _hear_weights_cached() -> bool:
    """Whether the pinned commit is already staged in this host's HF cache.

    Existence-based, not a Hub call: the point of the gate is that this test never provisions a
    ~600 MB TensorFlow venv, never downloads 1.2 GB of gated weights and never needs a token.
    """
    from huggingface_hub import constants

    root = Path(constants.HF_HUB_CACHE) / "models--google--hear" / "snapshots" / hear.HEAR_REVISION
    return (root / "saved_model.pb").is_file()


requires_real_hear = pytest.mark.skipif(
    not (_hear_venv_exists() and _hear_weights_cached()),
    reason=(
        "needs the provisioned 'hear' TensorFlow venv and the gated google/hear weights already "
        "in the HF cache; both are large and the repo requires accepting Google's Health AI "
        "Developer Foundations terms, so this test never provisions either"
    ),
)


@requires_real_hear
def test_the_real_encoder_returns_512_dimensions_per_window() -> None:
    """End-to-end shape and finiteness against the actual SavedModel."""
    audio = _ramp_audio(int(4.5 * SR))
    [result] = extract_hear_embeddings_from_audios([audio], hop_length=1.0)
    assert result.embeddings.shape == (len(result.window_starts), hear.HEAR_EMBEDDING_DIM)
    assert torch.isfinite(result.embeddings).all()


@requires_real_hear
def test_the_real_detector_returns_eight_probabilities_per_window() -> None:
    """The graph's 8 outputs are probabilities in [0, 1] that do not sum to 1 (multi-label)."""
    [windows] = detect_health_acoustic_events([_ramp_audio(int(3.0 * SR))], hop_length=1.0)
    assert windows
    for window in windows:
        scores = [next(iter(entry.values())) for entry in window["label_scores"]]
        assert len(scores) == 8
        assert all(0.0 <= score <= 1.0 for score in scores)


@requires_real_hear
def test_the_real_detector_rejects_a_non_two_second_window(tmp_path: Path) -> None:
    """The measurement the fixed window rests on: any other length is a graph error.

    Drives the worker directly with ``window_samples`` set to 1 s — a length the parent API cannot
    produce — and requires the SavedModel to fail. Without this, a future "make the window
    configurable" change would look harmless in review, since the encoder *would* accept it.

    The 1 s window is wholly inside the 3 s recording, so the worker's own bounds check passes and
    the failure genuinely comes from the graph; the assertion matches on TensorFlow's own wording
    rather than accepting any exception.
    """
    from senselab.utils.subprocess_venv import parse_subprocess_result, venv_python

    prepared = hear.prepare_audio_for_hear(_ramp_audio(3 * SR))
    _, snapshot = hear.stage_hear_snapshot()
    wav = tmp_path / "a.wav"
    hear.write_hear_wav(wav, prepared)

    payload = hear.build_worker_payload(
        str(snapshot / hear.EVENT_DETECTOR_SUBDIRS["large"]),
        [{"wav": str(wav), "starts": [0], "out": str(tmp_path / "o.npy")}],
        batch_size=1,
    )
    payload["window_samples"] = SR  # 1 s: half of what the graph accepts

    completed = subprocess.run(
        [venv_python(_cache_dir_path() / "hear"), "-c", hear._HEAR_WORKER],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=900,
    )
    with pytest.raises(RuntimeError, match="Graph execution error"):
        parse_subprocess_result(completed, "HeAR")


def test_span_to_hear_buffer_is_exactly_two_seconds() -> None:
    """The buffer is exactly the detector's 2 s window at the input rate."""
    sr = 16000
    audio = Audio(
        waveform=np.random.default_rng(0).standard_normal((1, 5 * sr)).astype("float32") * 0.1,
        sampling_rate=sr,
    )
    buf = hear.span_to_hear_buffer(audio, 1.0, 1.35)
    assert buf.waveform.shape[-1] == 2 * sr
    assert buf.sampling_rate == sr


def test_span_to_hear_buffer_centres_the_span_and_zeroes_the_rest() -> None:
    """Everything outside the span is silence, never neighbouring audio."""
    sr = 16000
    x = np.ones((1, 3 * sr), dtype="float32")
    buf = hear.span_to_hear_buffer(Audio(waveform=x, sampling_rate=sr), 1.0, 1.5)
    w = np.asarray(buf.waveform).squeeze()
    span_len = int(0.5 * sr)
    offset = (2 * sr - span_len) // 2
    assert np.all(w[:offset] == 0.0), "outside the span must be silence, not neighbouring audio"
    assert np.all(w[offset : offset + span_len] == 1.0)


def test_a_span_longer_than_two_seconds_is_refused() -> None:
    """A span the window cannot hold raises rather than being truncated."""
    sr = 16000
    audio = Audio(waveform=np.zeros((1, 5 * sr), dtype="float32"), sampling_rate=sr)
    with pytest.raises(ValueError, match="longer than the 2 s"):
        hear.span_to_hear_buffer(audio, 1.0, 4.0)
