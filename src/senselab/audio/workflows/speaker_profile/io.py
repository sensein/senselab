"""Load / save the persisted SpeakerProfile artifact (JSON).

Schema is the single source of truth in
``specs/20260527-151905-speaker-profile-embedding/contracts/speaker-profile.schema.md``.
Invariants enforced here:

- Atomic write — write to a sibling ``.tmp`` then ``os.replace`` so readers
  never see a half-written profile.
- Forward-compatible reader — extra unknown keys are ignored.
- Refuses to read a profile whose ``schema_version`` exceeds this reader's
  known ``SCHEMA_VERSION`` (rather than silently misinterpret).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast, get_args

from senselab.audio.workflows.speaker_profile.types import (
    ClusterStats,
    ProfileConfidence,
    ProfileParams,
    ProfileSourceFile,
    SpeakerProfile,
)

SCHEMA_VERSION: int = 1
"""Bumped on any breaking change to the artifact JSON shape."""


# ──────────────────────────────────────────────────────────────────────────
# Save


def save_profile(profile: SpeakerProfile, path: Path) -> None:
    """Atomically write ``profile`` to ``path`` as JSON.

    ``calibration_band`` tuple values are serialized as 2-element lists so the
    on-disk form is plain JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "subject_id": profile.subject_id,
        "confidence": profile.confidence,
        "aggregate_speech_seconds": profile.aggregate_speech_seconds,
        "centroids": {model: list(vec) for model, vec in profile.centroids.items()},
        "calibration_band": {model: [float(lo), float(hi)] for model, (lo, hi) in profile.calibration_band.items()},
        "dominant_cluster": asdict(profile.dominant_cluster),
        "runner_up_cluster": asdict(profile.runner_up_cluster) if profile.runner_up_cluster is not None else None,
        "sources": [asdict(s) for s in profile.sources],
        "params": asdict(profile.params) if profile.params is not None else None,
        "provenance": dict(profile.provenance),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────
# Load


class ProfileSchemaError(ValueError):
    """Raised when the on-disk profile cannot be safely interpreted."""


def load_profile(path: Path) -> SpeakerProfile:
    """Read a profile artifact written by :func:`save_profile`.

    - Unknown keys are ignored (forward-compatible).
    - A ``schema_version`` higher than this reader knows raises
      :class:`ProfileSchemaError` rather than risk silent misinterpretation.
    """
    path = Path(path)
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProfileSchemaError(f"Expected a JSON object at {path}, got {type(raw).__name__}")

    on_disk_version = raw.get("schema_version", 0)
    if not isinstance(on_disk_version, int) or on_disk_version > SCHEMA_VERSION:
        raise ProfileSchemaError(
            f"Profile schema_version {on_disk_version!r} at {path} is newer than reader "
            f"({SCHEMA_VERSION}). Upgrade senselab or downgrade the profile."
        )

    confidence_raw = raw.get("confidence", "insufficient")
    allowed_confidence = get_args(ProfileConfidence)
    if confidence_raw not in allowed_confidence:
        raise ProfileSchemaError(f"Unknown confidence value {confidence_raw!r} (expected one of {allowed_confidence})")
    confidence = cast(ProfileConfidence, confidence_raw)

    centroids_raw = raw.get("centroids", {}) or {}
    centroids: dict[str, list[float]] = {str(model): [float(x) for x in vec] for model, vec in centroids_raw.items()}

    calib_raw = raw.get("calibration_band", {}) or {}
    calibration_band: dict[str, tuple[float, float]] = {
        str(model): (float(pair[0]), float(pair[1])) for model, pair in calib_raw.items()
    }

    dominant_cluster = _cluster_stats_from_dict(raw.get("dominant_cluster"))
    runner_up_raw = raw.get("runner_up_cluster")
    runner_up_cluster = _cluster_stats_from_dict(runner_up_raw) if runner_up_raw is not None else None

    sources_raw = raw.get("sources", []) or []
    sources = [_source_file_from_dict(s) for s in sources_raw if isinstance(s, dict)]

    params_raw = raw.get("params")
    params = _params_from_dict(params_raw) if isinstance(params_raw, dict) else None

    provenance = dict(raw.get("provenance", {}) or {})

    return SpeakerProfile(
        subject_id=str(raw.get("subject_id", "")),
        centroids=centroids,
        confidence=confidence,
        aggregate_speech_seconds=float(raw.get("aggregate_speech_seconds", 0.0)),
        dominant_cluster=dominant_cluster,
        runner_up_cluster=runner_up_cluster,
        calibration_band=calibration_band,
        sources=sources,
        params=params,
        provenance=provenance,
    )


# ──────────────────────────────────────────────────────────────────────────
# Internal: dict → dataclass helpers (forward-compatible — ignore unknown keys)


def _cluster_stats_from_dict(d: Any) -> ClusterStats:  # noqa: ANN401
    if not isinstance(d, dict):
        return ClusterStats(n_windows=0, speech_seconds=0.0, silhouette=None, share=0.0)
    sil = d.get("silhouette")
    return ClusterStats(
        n_windows=int(d.get("n_windows", 0)),
        speech_seconds=float(d.get("speech_seconds", 0.0)),
        silhouette=float(sil) if sil is not None else None,
        share=float(d.get("share", 0.0)),
    )


def _source_file_from_dict(d: dict[str, Any]) -> ProfileSourceFile:
    sess = d.get("session_id")
    reason = d.get("drop_reason")
    return ProfileSourceFile(
        file_id=str(d.get("file_id", "")),
        audio_signature=str(d.get("audio_signature", "")),
        session_id=str(sess) if sess is not None else None,
        speech_seconds_used=float(d.get("speech_seconds_used", 0.0)),
        windows_used=int(d.get("windows_used", 0)),
        kept=bool(d.get("kept", False)),
        drop_reason=str(reason) if reason is not None else None,
    )


def _params_from_dict(d: dict[str, Any]) -> ProfileParams:
    sess = d.get("prefer_session")
    models = d.get("embedding_models", []) or []
    return ProfileParams(
        embedding_models=[str(m) for m in models],
        profile_window_s=float(d.get("profile_window_s", 0.0)),
        profile_hop_s=float(d.get("profile_hop_s", 0.0)),
        detect_window_s=float(d.get("detect_window_s", 0.0)),
        detect_hop_s=float(d.get("detect_hop_s", 0.0)),
        min_confident_speech_s=float(d.get("min_confident_speech_s", 0.0)),
        target_confident_speech_s=float(d.get("target_confident_speech_s", 0.0)),
        ambiguity_share_ratio=float(d.get("ambiguity_share_ratio", 0.0)),
        prefer_session=str(sess) if sess is not None else None,
    )
