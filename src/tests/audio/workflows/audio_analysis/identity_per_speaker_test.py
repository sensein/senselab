"""Per-speaker structure derived from the per-bucket identity evidence (T098).

The per-bucket identity axis stays the evidence-gathering mechanism; these functions read
what it already harvested rather than running any new inference. Everything here is
therefore pure, and a track can never claim a speaker the harvest did not observe.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding
from senselab.audio.workflows.audio_analysis.identity import (
    SILENT_CLUSTER_ID,
    cluster_active_time,
    label_correspondence_rows,
    per_speaker_tracks,
)


def _bucket(start: float, end: float, clusters: dict[str, str], labels: dict[str, str] | None = None) -> dict:
    """One harvested bucket: which cluster each diar model placed here."""
    votes: dict[str, object] = {}
    for model, cluster in clusters.items():
        votes[model] = {
            "speaker_label": (labels or {}).get(model, f"{model}-{cluster}"),
            "cluster_id": cluster,
            "speaker_changed_from_prev": None,
        }
    votes["__cross_diar_label_disagreement__"] = {"cluster_ids": dict(clusters)}
    return {"start": start, "end": end, "votes": votes}


# ── presence per speaker ──────────────────────────────────────────────


def test_a_speaker_all_models_agree_on_is_present_without_doubt() -> None:
    """Unanimity is the anchor: if this carried uncertainty, nothing else could be read."""
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": "S0", "b": "S0"})])
    assert len(rows) == 1
    assert rows[0]["presence_confidence"] == 1.0
    assert rows[0]["presence_uncertainty"] == 0.0
    assert rows[0]["contributing_sources"] == ["a", "b"]


def test_a_speaker_only_one_of_two_models_places_here_is_maximally_uncertain() -> None:
    """A split vote is the case the per-speaker axis exists to expose.

    Under the old single scalar this bucket read as "identity is uncertain" without saying
    *which* speaker was in doubt — so no follow-up could be targeted.
    """
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": "S0", "b": "S1"})])
    by_id = {r["cluster_id"]: r for r in rows}
    assert by_id["S0"]["presence_confidence"] == 0.5
    assert by_id["S0"]["presence_uncertainty"] == pytest.approx(1.0)
    assert by_id["S1"]["presence_uncertainty"] == pytest.approx(1.0)


def test_silence_is_not_a_speaker() -> None:
    """The ``<silent>`` pseudo-cluster is a bookkeeping device, not a person."""
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": SILENT_CLUSTER_ID, "b": SILENT_CLUSTER_ID})])
    assert rows == []


def test_a_model_calling_silence_still_counts_against_a_speaker_it_omits() -> None:
    """Silent models stay in the denominator.

    Confidence is over all models in the bucket, not only the ones that named someone —
    otherwise a lone detection among four silent models reads as certain.
    """
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": "S0", "b": SILENT_CLUSTER_ID, "c": SILENT_CLUSTER_ID})])
    assert rows[0]["presence_confidence"] == pytest.approx(1 / 3)


def test_speakers_active_in_the_same_bucket_are_recorded_as_overlapping() -> None:
    """FR-003: concurrent speech is a per-speaker fact; a single axis cannot express it."""
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": "S0", "b": "S1"})])
    assert {r["cluster_id"]: r["overlap_with"] for r in rows} == {"S0": ["S1"], "S1": ["S0"]}


def test_a_lone_speaker_overlaps_with_nobody() -> None:
    """The overlap list is empty rather than absent when a speaker is alone."""
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": "S0", "b": "S0"})])
    assert rows[0]["overlap_with"] == []


def test_rows_come_out_in_a_fixed_order() -> None:
    """Row order is a property of the evidence.

    The outputs are asserted byte-identical across runs (SC-004), so ordering cannot depend
    on dict insertion order.
    """
    buckets = [_bucket(0.5, 1.0, {"b": "S1", "a": "S0"}), _bucket(0.0, 0.5, {"a": "S1", "b": "S0"})]
    rows = per_speaker_tracks(buckets)
    assert [(r["start"], r["cluster_id"]) for r in rows] == [(0.0, "S0"), (0.0, "S1"), (0.5, "S0"), (0.5, "S1")]


def test_a_speaker_absent_from_a_bucket_gets_no_row_there() -> None:
    """Rows are evidence of presence, not assertions about everyone everywhere.

    A bucket with no claim for a speaker is not the same as claiming that speaker absent at
    confidence zero.
    """
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": "S0"}), _bucket(0.5, 1.0, {"a": "S1"})])
    assert [(r["start"], r["cluster_id"]) for r in rows] == [(0.0, "S0"), (0.5, "S1")]


# ── ranking and correspondence ────────────────────────────────────────


def test_clusters_rank_by_how_long_they_were_active() -> None:
    """Which cluster becomes S0 must be a property of the evidence, not of iteration order."""
    buckets = [
        _bucket(0.0, 0.5, {"a": "Sx"}),
        _bucket(0.5, 1.0, {"a": "Sy"}),
        _bucket(1.0, 1.5, {"a": "Sy"}),
    ]
    assert cluster_active_time(buckets) == {"Sy": pytest.approx(1.0), "Sx": pytest.approx(0.5)}


def test_equally_active_clusters_break_ties_by_name() -> None:
    """A deterministic tiebreak, so equal evidence does not produce run-to-run drift."""
    buckets = [_bucket(0.0, 0.5, {"a": "Sb"}), _bucket(0.5, 1.0, {"a": "Sa"})]
    assert list(cluster_active_time(buckets)) == ["Sa", "Sb"]


def test_each_model_label_is_mapped_to_the_speaker_it_became() -> None:
    """FR-005: a fused speaker id is unusable without its provenance.

    Every diarizer invents its own labels, so a consumer cannot act on a fused id without
    knowing which of its own labels produced it.
    """
    buckets = [
        _bucket(
            0.0,
            0.5,
            {"pyannote": "S0", "sortformer": "S0"},
            labels={"pyannote": "SPEAKER_00", "sortformer": "speaker_2"},
        )
    ]
    rows = label_correspondence_rows(buckets, speaker_ids={"S0": "S0"})
    assert {(r["source"], r["source_label"], r["speaker_id"]) for r in rows} == {
        ("pyannote", "SPEAKER_00", "S0"),
        ("sortformer", "speaker_2", "S0"),
    }


def test_a_model_label_that_moved_between_clusters_reports_both() -> None:
    """Silently keeping one mapping would hide a genuine instability in the clustering."""
    buckets = [
        _bucket(0.0, 0.5, {"a": "S0"}, labels={"a": "SPEAKER_00"}),
        _bucket(0.5, 1.0, {"a": "S1"}, labels={"a": "SPEAKER_00"}),
    ]
    rows = label_correspondence_rows(buckets, speaker_ids={"S0": "S0", "S1": "S1"})
    assert sorted(r["speaker_id"] for r in rows) == ["S0", "S1"]


def test_silence_is_not_given_a_correspondence_row() -> None:
    """There is no speaker for a silence pseudo-label to correspond to."""
    buckets = [_bucket(0.0, 0.5, {"a": SILENT_CLUSTER_ID})]
    assert label_correspondence_rows(buckets, speaker_ids={}) == []


def test_tracks_carry_the_fused_speaker_id_when_one_is_supplied() -> None:
    """Downstream consumers read ``speaker_id``; the cluster id is retained for audit."""
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": "Sx"})], speaker_ids={"Sx": "S0"})
    assert rows[0]["speaker_id"] == "S0" and rows[0]["cluster_id"] == "Sx"


def test_an_unmapped_cluster_keeps_its_own_id_rather_than_being_dropped() -> None:
    """Surplus clusters stay visible.

    A cluster the count posterior does not back is still observed evidence; dropping it
    would make the surplus invisible instead of contested.
    """
    rows = per_speaker_tracks([_bucket(0.0, 0.5, {"a": "Sx"})], speaker_ids={"Sy": "S0"})
    assert rows[0]["speaker_id"] == "Sx"


# ── empirical same-speaker calibration on every path (real-run defect) ──


def test_a_single_speaker_pass_still_reports_its_calibration_band() -> None:
    """The identity axis cannot report low uncertainty without a reachable same-speaker floor.

    Measured on a two-speaker recording: across 446 buckets where every diarizer agreed the
    speaker was unchanged, ECAPA's within-speaker cosine distance had a median of 0.543 and
    *never once* fell below the 0.30 literature default — so "confidently the same speaker"
    was unreachable and the axis averaged 0.66 during unambiguous single-speaker speech.

    The per-pass empirical band exists to fix exactly this, but it was returned from only one
    of the clusterer's exit paths. A pass that settles on one speaker took a path without it
    and silently reverted to the unreachable default — which is the common case, since a
    single dominant talker is what most of these recordings contain.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.embeddings import cluster_pass_speakers

    rng = np.random.default_rng(0)
    centre = rng.normal(size=192)
    centre /= np.linalg.norm(centre)
    entries = []
    for i in range(24):
        v = centre + 0.25 * rng.normal(size=192)
        v /= np.linalg.norm(v)
        entries.append(WindowEmbedding(start_s=i * 0.5, end_s=i * 0.5 + 2.0, vector=v))

    out = cluster_pass_speakers(entries)
    assert out is not None
    assert out["n_speakers"] == 1, "this fixture is deliberately one speaker"
    same = out.get("empirical_same_speaker_floor")
    assert same is not None, "a single-speaker pass reported no same-speaker calibration"
    assert 0.0 < same < 1.0


def test_the_measured_band_actually_admits_the_speaker_it_was_measured_on() -> None:
    """A floor no same-speaker comparison can reach is not a calibration.

    This is the property the fixed default lacked: with the band measured from the pass's own
    within-cluster distances, a typical same-speaker pair must land at or below it.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.embeddings import cluster_pass_speakers

    rng = np.random.default_rng(1)
    centre = rng.normal(size=192)
    centre /= np.linalg.norm(centre)
    vecs = []
    for _ in range(24):
        v = centre + 0.25 * rng.normal(size=192)
        vecs.append(v / np.linalg.norm(v))
    entries = [WindowEmbedding(start_s=i * 0.5, end_s=i * 0.5 + 2.0, vector=v) for i, v in enumerate(vecs)]

    same = cluster_pass_speakers(entries)["empirical_same_speaker_floor"]
    stacked = np.stack(vecs)
    dists = [1.0 - float(stacked[i] @ stacked[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    assert float(np.median(dists)) <= same
