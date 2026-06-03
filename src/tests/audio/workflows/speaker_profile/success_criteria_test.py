"""Success-criteria smoke checks (SC-002/003/004/005) on the composer fixtures.

Unlike the fast vector-level unit tests in ``compare_test.py``, these exercise
the **real** embedding models (ECAPA + ResNet) on the committed synthetic audio
composed by the T010b fixtures (contamination / overlay / noise), and assert the
spec's success criteria end-to-end:

- **SC-002** — contamination tolerance: a profile built from material that
  includes a ~20%-intruder-contaminated file stays closer to held-out clean
  target audio than to the intruder.
- **SC-003** — other-voice detection rate on annotated intruder regions is at
  least 2× the target-only false-positive rate.
- **SC-004** — on a target-only recording, the other-voice false-flag rate is a
  small fraction of duration (< 10%).
- **SC-005** — a clean target-dominant recording scores higher target quality
  than a noisy one.

Slow: builds a real profile and runs per-window embedding extraction on the
long synthetic passages. The shared clean profile and the clean-recording
scoring are memoized so the models run as few times as possible.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("soundfile")

from senselab.audio.workflows.audio_analysis.embeddings import extract_per_window_embeddings  # noqa: E402
from senselab.audio.workflows.speaker_profile import constants as C  # noqa: E402
from senselab.audio.workflows.speaker_profile.build import ProfileInput, build_speaker_profile  # noqa: E402
from senselab.audio.workflows.speaker_profile.compare import (  # noqa: E402
    compare_recording_to_profile,
    compute_target_quality,
)

from .conftest import add_noise_at_snr, compose_contamination, compose_other_voice, load_clip  # noqa: E402

_MODELS = [C.ECAPA_MODEL_ID, C.RESNET_MODEL_ID]

# Long passages give enough windows for stable detection/false-positive rates.
_BUILD_CLEAN = [
    "sub-A-confident/ses-1/rainbow.flac",
    "sub-A-confident/ses-1/north-wind.flac",
]
_TARGET_REC = "sub-A-confident/ses-1/grandfather.flac"  # held out from the profile
_INTRUDER = "speaker-B/clip-00.flac"

_CACHE: dict[str, object] = {}


def _detect(audio) -> dict:  # noqa: ANN001
    """Per-window embeddings on the short detection grid for one recording."""
    return extract_per_window_embeddings(audio=audio, models=_MODELS, window_s=C.DETECT_WINDOW_S, hop_s=C.DETECT_HOP_S)


def _clean_profile():  # noqa: ANN202
    """Build (once) a confident clean profile from the long target passages."""
    if "clean_profile" not in _CACHE:
        inputs = [ProfileInput(audio=load_clip(fid), file_id=fid) for fid in _BUILD_CLEAN]
        _CACHE["clean_profile"] = build_speaker_profile("sub-A", inputs, embedding_models=_MODELS)
    return _CACHE["clean_profile"]


def _mean_unit_per_model(audio) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Mean L2-normalized embedding per model for a recording (a reference vector)."""
    out: dict[str, np.ndarray] = {}
    for model, windows in _detect(audio).items():
        vecs = [np.asarray(w.vector, dtype=np.float64) for w in windows if w.vector.size]
        if not vecs:
            continue
        m = np.mean(np.stack(vecs), axis=0)
        n = np.linalg.norm(m)
        if n > 0:
            out[model] = m / n
    return out


# ──────────────────────────────────────────────────────────────────────────
# SC-002 — contamination tolerance (build level)


def test_sc002_contamination_tolerance() -> None:
    """A ~20%-contaminated enrollment still yields a centroid nearer clean target than intruder."""
    contaminated = compose_contamination("sub-A-confident/ses-1/harvard-09.flac", _INTRUDER, fraction=0.2).audio
    inputs = [ProfileInput(audio=load_clip(fid), file_id=fid) for fid in _BUILD_CLEAN]
    inputs.append(ProfileInput(audio=contaminated, file_id="sub-A-confident/ses-1/harvard-09-contam.flac"))
    profile = build_speaker_profile("sub-A-contam", inputs, embedding_models=_MODELS)

    target_ref = _mean_unit_per_model(load_clip(_TARGET_REC))  # held-out clean target
    intruder_ref = _mean_unit_per_model(load_clip(_INTRUDER))

    assert profile.centroids, "contaminated build produced no centroid"
    for model, centroid in profile.centroids.items():
        c = np.asarray(centroid, dtype=np.float64)
        sim_target = float(c @ target_ref[model])
        sim_intruder = float(c @ intruder_ref[model])
        assert sim_target > sim_intruder, f"{model}: nearer intruder {sim_intruder:.2f} than target {sim_target:.2f}"


# ──────────────────────────────────────────────────────────────────────────
# SC-003 / SC-004 — other-voice detection vs false-positive


def _clean_grandfather_results():  # noqa: ANN202
    """Score (once) the clean target recording against the clean profile."""
    if "clean_results" not in _CACHE:
        profile = _clean_profile()
        results = compare_recording_to_profile(
            _detect(load_clip(_TARGET_REC)), profile.centroids, profile.calibration_band
        )
        _CACHE["clean_results"] = results
    return _CACHE["clean_results"]


def test_sc003_other_voice_detection_beats_false_positive() -> None:
    """Detection rate on the annotated intruder interval ≥ 2× the target-only false-positive rate."""
    profile = _clean_profile()
    interval = (6.0, 10.0)
    overlay = compose_other_voice(_TARGET_REC, _INTRUDER, [interval], intruder_gain=2.0).audio
    results = compare_recording_to_profile(_detect(overlay), profile.centroids, profile.calibration_band)

    in_region = [r for r in results if interval[0] <= 0.5 * (r.start + r.end) <= interval[1]]
    out_region = [r for r in results if not (interval[0] <= 0.5 * (r.start + r.end) <= interval[1])]
    assert in_region and out_region, "overlay recording did not span both regions"

    detection_rate = sum(r.flag == "other_voice" for r in in_region) / len(in_region)
    false_pos_rate = sum(r.flag == "other_voice" for r in out_region) / len(out_region)

    assert detection_rate >= 2 * false_pos_rate, f"detection {detection_rate:.2f} < 2× FP {false_pos_rate:.2f}"
    assert detection_rate > 0.0, "no intruder windows detected"


def test_sc004_target_only_false_flag_under_10pct() -> None:
    """A target-only recording flags other_voice on < 10% of speech-present windows."""
    results = _clean_grandfather_results()
    scored = [r for r in results if r.flag != "unavailable"]
    assert scored, "no scored windows on the clean recording"
    false_flag_rate = sum(r.flag == "other_voice" for r in scored) / len(scored)
    assert false_flag_rate < 0.10, f"target-only false-flag rate {false_flag_rate:.2f} ≥ 0.10"


# ──────────────────────────────────────────────────────────────────────────
# SC-005 — clean target outranks noisy on target quality


def test_sc005_clean_outranks_noisy_quality() -> None:
    """A clean target recording scores higher target quality than a noisy one."""
    profile = _clean_profile()
    clean_results = _clean_grandfather_results()
    noisy = add_noise_at_snr(_TARGET_REC, snr_db=0.0).audio
    noisy_results = compare_recording_to_profile(_detect(noisy), profile.centroids, profile.calibration_band)

    q_clean = compute_target_quality(clean_results, profile.confidence)
    q_noisy = compute_target_quality(noisy_results, profile.confidence)
    assert q_clean.profile_target_quality > q_noisy.profile_target_quality, (
        f"clean {q_clean.profile_target_quality:.2f} ≤ noisy {q_noisy.profile_target_quality:.2f}"
    )
