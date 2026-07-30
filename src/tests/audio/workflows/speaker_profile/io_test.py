"""Tests for the SpeakerProfile JSON artifact I/O (T010c).

Covers the contract in
``specs/20260527-151905-speaker-profile-embedding/contracts/speaker-profile.schema.md``:

- Round-trip save → load preserves all fields.
- Reader ignores unknown extra keys (forward-compatible).
- Reader refuses a higher ``schema_version`` than it supports.
- ``save_profile`` is atomic — never leaves a partial file at the target path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.io import (
    SCHEMA_VERSION,
    ProfileSchemaError,
    load_profile,
    save_profile,
)
from senselab.audio.workflows.speaker_profile.types import (
    ClusterStats,
    ProfileParams,
    ProfileSourceFile,
    SpeakerProfile,
)


def _example_profile() -> SpeakerProfile:
    """Build a realistic SpeakerProfile for round-trip tests."""
    return SpeakerProfile(
        subject_id="sub-test-001",
        centroids={
            C.ECAPA_MODEL_ID: [0.1, 0.2, 0.3],
            C.WAVLM_DEFAULT_CHECKPOINT: [0.4, 0.5, 0.6, 0.7],
        },
        confidence="ok",
        aggregate_speech_seconds=33.0,
        dominant_cluster=ClusterStats(n_windows=42, speech_seconds=33.0, silhouette=0.27, share=0.86),
        runner_up_cluster=None,
        calibration_band={
            C.ECAPA_MODEL_ID: (0.31, 0.68),
            C.WAVLM_DEFAULT_CHECKPOINT: (0.28, 0.71),
        },
        sources=[
            ProfileSourceFile(
                file_id="sub-test-001/ses-1/free-speech.wav",
                audio_signature="a" * 64,
                session_id="ses-1",
                speech_seconds_used=22.0,
                windows_used=41,
                kept=True,
                drop_reason=None,
            ),
            ProfileSourceFile(
                file_id="sub-test-001/ses-1/cough.wav",
                audio_signature="b" * 64,
                session_id="ses-1",
                speech_seconds_used=0.0,
                windows_used=0,
                kept=False,
                drop_reason="non_speech_task",
            ),
        ],
        params=ProfileParams(
            embedding_models=list(C.DEFAULT_EMBEDDING_MODELS),
            profile_window_s=C.PROFILE_WINDOW_S,
            profile_hop_s=C.PROFILE_HOP_S,
            detect_window_s=C.DETECT_WINDOW_S,
            detect_hop_s=C.DETECT_HOP_S,
            min_confident_speech_s=C.MIN_CONFIDENT_SPEECH_S,
            target_confident_speech_s=C.TARGET_CONFIDENT_SPEECH_S,
            ambiguity_share_ratio=C.AMBIGUITY_SHARE_RATIO,
            prefer_session=None,
        ),
        provenance={"senselab_version": "x.y.z", "built_at": "2026-05-27T15:30:00Z"},
    )


def test_round_trip_save_load(tmp_path: Path) -> None:
    """Saving then loading reproduces the SpeakerProfile field-for-field."""
    original = _example_profile()
    target = tmp_path / "profile.json"
    save_profile(original, target)

    loaded = load_profile(target)

    assert loaded.subject_id == original.subject_id
    assert loaded.confidence == original.confidence
    assert loaded.aggregate_speech_seconds == original.aggregate_speech_seconds
    assert loaded.centroids == original.centroids
    assert loaded.calibration_band == original.calibration_band
    assert loaded.dominant_cluster == original.dominant_cluster
    assert loaded.runner_up_cluster == original.runner_up_cluster
    assert loaded.sources == original.sources
    assert loaded.params == original.params
    # provenance is a dict; loader returns a fresh dict — compare contents
    assert loaded.provenance == original.provenance


def test_save_writes_schema_version(tmp_path: Path) -> None:
    """The persisted JSON stamps the current SCHEMA_VERSION."""
    target = tmp_path / "profile.json"
    save_profile(_example_profile(), target)
    raw = json.loads(target.read_text())
    assert raw["schema_version"] == SCHEMA_VERSION


def test_load_ignores_unknown_keys(tmp_path: Path) -> None:
    """A reader running against a newer-with-extra-keys (same schema_version) file accepts it."""
    target = tmp_path / "profile.json"
    save_profile(_example_profile(), target)
    raw = json.loads(target.read_text())
    raw["future_field_we_dont_know_about"] = {"x": 1}
    raw["sources"][0]["another_unknown"] = True
    target.write_text(json.dumps(raw))

    loaded = load_profile(target)
    assert loaded.subject_id == "sub-test-001"
    assert loaded.sources[0].file_id.endswith("free-speech.wav")


def test_load_refuses_higher_schema_version(tmp_path: Path) -> None:
    """A future-version profile is refused rather than silently misinterpreted."""
    target = tmp_path / "profile.json"
    save_profile(_example_profile(), target)
    raw = json.loads(target.read_text())
    raw["schema_version"] = SCHEMA_VERSION + 1
    target.write_text(json.dumps(raw))

    with pytest.raises(ProfileSchemaError):
        load_profile(target)


def test_save_is_atomic_no_tmp_leftover(tmp_path: Path) -> None:
    """Successful save leaves only the target file (no ``*.tmp`` siblings)."""
    target = tmp_path / "profile.json"
    save_profile(_example_profile(), target)
    siblings = sorted(p.name for p in tmp_path.iterdir())
    assert siblings == ["profile.json"], siblings


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    """Save creates missing parent directories rather than raising."""
    target = tmp_path / "nested" / "deeper" / "profile.json"
    save_profile(_example_profile(), target)
    assert target.exists()


def test_load_rejects_non_object_root(tmp_path: Path) -> None:
    """A JSON file whose root isn't an object is rejected with a ProfileSchemaError."""
    target = tmp_path / "broken.json"
    target.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ProfileSchemaError):
        load_profile(target)


def test_load_rejects_unknown_confidence(tmp_path: Path) -> None:
    """An unrecognized confidence string is rejected (defensive against drift)."""
    target = tmp_path / "profile.json"
    save_profile(_example_profile(), target)
    raw = json.loads(target.read_text())
    raw["confidence"] = "totally_made_up"
    target.write_text(json.dumps(raw))
    with pytest.raises(ProfileSchemaError):
        load_profile(target)
