"""Final per-speaker outputs (T094, contracts/speaker-identity.md).

These artifacts are what an analyst reads. SC-002 requires that the count disagreement be
resolvable *from these files alone* — without opening intermediate per-bucket artifacts —
so anything needed for that has to be present here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from senselab.audio.workflows.audio_analysis.adaptive.fusion import write_speaker_outputs
from senselab.audio.workflows.audio_analysis.speaker_identity import (
    PerSpeakerPresenceTrack,
    SourceCountClaim,
    SourceLabelCorrespondence,
    SpeakerHypothesis,
    speaker_count_posterior,
)


def _posterior():  # noqa: ANN202 — test helper
    return speaker_count_posterior(
        [
            SourceCountClaim("pyannote", 1, kind="independent"),
            SourceCountClaim("sortformer", 1, kind="independent"),
            SourceCountClaim("embedding_silhouette", 5, kind="derived"),
        ],
        gates={"independent": 1.0, "derived": 0.4},
    )


def _hypotheses():  # noqa: ANN202
    return [
        SpeakerHypothesis(
            speaker_id="S0",
            existence_uncertainty=0.18,
            supporting_sources=["pyannote", "embedding_silhouette"],
            source_kinds={"pyannote": "independent", "embedding_silhouette": "derived"},
            first_seen=0.08,
            last_seen=4.84,
            total_active_s=2.31,
            converged=True,
        )
    ]


def _write(tmp_path: Path, **kw: object):  # noqa: ANN202
    return write_speaker_outputs(
        tmp_path,
        posterior=_posterior(),
        hypotheses=_hypotheses(),
        correspondence=[SourceLabelCorrespondence("pyannote", "SPEAKER_00", "S0", "independent", cluster_id="c0")],
        tracks=[PerSpeakerPresenceTrack("S0", 0.0, 0.5, 0.9, 0.1, contributing_sources=["pyannote"])],
        **kw,  # type: ignore[arg-type]
    )


def test_both_artifacts_are_written(tmp_path: Path) -> None:
    """The pair, not one or the other."""
    speakers, presence = _write(tmp_path)
    assert speakers.exists() and presence.exists()


def test_count_disagreement_is_readable_from_the_file_alone(tmp_path: Path) -> None:
    """SC-002: an analyst must not have to open intermediate artifacts.

    Both competing counts and the sources backing each are present in one document.
    """
    speakers, _ = _write(tmp_path)
    doc = json.loads(speakers.read_text())
    cp = doc["count_posterior"]
    assert set(cp["probabilities"]) == {"1", "5"}
    assert cp["support"]["1"] == ["pyannote", "sortformer"]
    assert cp["support"]["5"] == ["embedding_silhouette"]
    assert cp["is_multimodal"] is True


def test_probabilities_sum_to_one_after_serialization(tmp_path: Path) -> None:
    """Rounding for readability must not break the distribution."""
    speakers, _ = _write(tmp_path)
    probs = json.loads(speakers.read_text())["count_posterior"]["probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)


def test_every_supported_count_appears_in_probabilities(tmp_path: Path) -> None:
    """Support and probabilities cannot disagree about which counts exist."""
    speakers, _ = _write(tmp_path)
    cp = json.loads(speakers.read_text())["count_posterior"]
    assert set(cp["support"]) <= set(cp["probabilities"])


def test_source_kinds_are_recorded_per_hypothesis(tmp_path: Path) -> None:
    """FR-007: a consumer must be able to weight a hypothesis by what backs it."""
    speakers, _ = _write(tmp_path)
    spk = json.loads(speakers.read_text())["speakers"][0]
    assert set(spk["source_kinds"].values()) <= {"independent", "derived"}
    assert spk["has_independent_support"] is True


def test_label_correspondence_is_auditable(tmp_path: Path) -> None:
    """FR-005: which source label became which hypothesis, and via which cluster."""
    speakers, _ = _write(tmp_path)
    entry = json.loads(speakers.read_text())["label_correspondence"][0]
    assert entry["source_label"] == "SPEAKER_00"
    assert entry["speaker_id"] == "S0"
    assert entry["cluster_id"] == "c0"


def test_presence_parquet_carries_every_contract_column(tmp_path: Path) -> None:
    """final/per_speaker_presence.parquet columns."""
    _, presence = _write(tmp_path)
    df = pd.read_parquet(presence)
    for col in (
        "speaker_id",
        "start",
        "end",
        "presence_confidence",
        "presence_uncertainty",
        "overlap_with",
        "contributing_sources",
        "round",
        "resolution_kind",
    ):
        assert col in df.columns, f"missing {col}"


def test_empty_tracks_still_write_a_typed_parquet(tmp_path: Path) -> None:
    """An absent file would make "no speakers" indistinguishable from "never ran"."""
    write_speaker_outputs(tmp_path, posterior=_posterior(), hypotheses=[], correspondence=[], tracks=[])
    df = pd.read_parquet(tmp_path / "final" / "per_speaker_presence.parquet")
    assert len(df) == 0
    assert "speaker_id" in df.columns


def test_profile_versions_are_recorded(tmp_path: Path) -> None:
    """A result must be attributable to the policy that produced it."""
    speakers, _ = _write(tmp_path, profile_version="detection-margin/2026-07-29", influence_profile="influence/default")
    doc = json.loads(speakers.read_text())
    assert doc["profile_version"] == "detection-margin/2026-07-29"
    assert doc["influence_profile"] == "influence/default"


def test_outputs_are_byte_identical_across_repeated_writes(tmp_path: Path) -> None:
    """SC-004 / SC-029: determinism, including key ordering inside the posterior."""
    a, _ = _write(tmp_path / "a")
    b, _ = _write(tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()
