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


def test_overlapping_distance_distributions_report_no_usable_band() -> None:
    """When same- and different-speaker distances overlap, there is no band to report.

    Measured on the higgs conversation with ECAPA over the standard window grid: the
    within-speaker distances have median 0.874 and the between-speaker distances 0.966, and
    the two distributions overlap almost entirely. The embedding simply does not separate
    identity at this scale on this recording.

    The old behaviour substituted the fixed [0.30, 0.70] band whenever the measured one came
    out inverted. That band is not merely imprecise here — it sits below every distance the
    embedding produces, so *every* same-speaker comparison scored as maximally doubtful, and
    the axis reported 0.65 uncertainty on buckets where every diarizer agreed. Returning
    nothing lets the sub-signal drop out instead of fabricating confident dissent.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.embeddings import _empirical_calibration_band

    rng = np.random.default_rng(3)
    # Two "clusters" drawn from the same distribution: labels differ, geometry does not.
    X = rng.normal(size=(24, 64))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    labels = np.array([0] * 12 + [1] * 12)
    assert _empirical_calibration_band(X, labels) is None


def test_a_genuinely_separable_pass_still_gets_its_band() -> None:
    """The guard must not refuse a calibration when the embeddings do discriminate."""
    import numpy as np

    from senselab.audio.workflows.audio_analysis.embeddings import _empirical_calibration_band

    rng = np.random.default_rng(4)
    a, b = rng.normal(size=64), rng.normal(size=64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    rows = [c + 0.15 * rng.normal(size=64) for c in ([a] * 12 + [b] * 12)]
    X = np.stack([r / np.linalg.norm(r) for r in rows])
    band = _empirical_calibration_band(X, np.array([0] * 12 + [1] * 12))
    assert band is not None and band[0] < band[1]


def test_an_uncalibratable_embedding_emits_no_identity_claim() -> None:
    """FR-007: a sub-signal that cannot be calibrated drops out rather than voting.

    Without this the axis cannot fall back on the evidence that *is* available — unanimous
    diarizer agreement — because a saturated derived signal outvotes it.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.identity import harvest_identity_votes

    segs = [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    pass_summary = {
        "duration_s": 3.0,
        "diarization": {"by_model": {"pyannote": {"status": "ok", "result": [segs]}}},
    }
    rng = np.random.default_rng(5)
    windows = []
    for i in range(6):
        v = rng.normal(size=64)
        windows.append(WindowEmbedding(start_s=i * 0.5, end_s=i * 0.5 + 1.0, vector=v / np.linalg.norm(v)))

    votes = harvest_identity_votes(
        pass_summary=pass_summary,
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        per_window_embeddings={"ecapa": windows},
        same_speaker_floor=None,
        diff_speaker_floor=None,
    )
    emitted = [
        v["same_label_uncertainty"]
        for bucket in votes
        for k, v in bucket["votes"].items()
        if "::" in k and isinstance(v, dict)
    ]
    assert emitted and all(u is None for u in emitted)
    # The raw distance is still recorded — dropping the claim must not destroy the evidence.
    cosines = [
        v["embedding_cosine_within_track"]
        for bucket in votes
        for k, v in bucket["votes"].items()
        if "::" in k and isinstance(v, dict)
    ]
    assert any(c is not None for c in cosines)


def test_the_band_is_measured_on_the_comparison_it_calibrates() -> None:
    """The anchors must describe the same statistic the harvester computes.

    Measured on the higgs conversation with pyannote's turns as ground truth: taken over
    *all* window pairs, ECAPA's within- and between-speaker distances overlap (within q75
    0.919 vs between q25 0.916, separation -0.003) and no ordered band exists. But the
    harvester never compares all pairs — it compares each bucket to the most recent prior
    bucket of the same speaker, and on that statistic the same embeddings separate cleanly
    (within q75 0.646, between q25 0.915, separation +0.269). Nearest-centroid classification
    recovers pyannote's labels at 98.5%, so the embeddings discriminate perfectly well.

    Calibrating on all-pairs therefore discarded a usable band and fell back to a fixed one
    that sat below every distance the embedding produces, which is why the axis reported high
    uncertainty on buckets where every diarizer agreed.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.embeddings import _sequential_calibration_band

    rng = np.random.default_rng(7)
    a, b = rng.normal(size=64), rng.normal(size=64)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    # Turn-structured like a conversation, with per-window noise tuned to reproduce the
    # measured statistics: same-speaker distance median ~0.55, different-speaker ~0.82.
    order = ["A"] * 8 + ["B"] * 8 + ["A"] * 8 + ["B"] * 8
    vecs, labels = [], []
    for lab in order:
        base = a if lab == "A" else b
        v = base + 0.15 * rng.normal(size=64)
        vecs.append(v / np.linalg.norm(v))
        labels.append(lab)
    band = _sequential_calibration_band(np.stack(vecs), np.array(labels))
    assert band is not None, "a separable pass produced no band"
    same_floor, diff_floor = band
    assert same_floor < diff_floor
    # A typical same-speaker comparison must reach the "confidently same" anchor.
    consecutive = [1.0 - float(vecs[i] @ vecs[i - 1]) for i in range(1, len(vecs)) if labels[i] == labels[i - 1]]
    assert float(np.median(consecutive)) <= same_floor
