"""Reproduction for F-165 (raised-by D-2, verdict: SURVIVED).

Claim: `speaker.py`'s `harvest_speaker_votes` wordless gate --

    if fused_words and coverage[key] <= 0.0:
        bucket_dict["votes"] = {}
        continue

-- treats "no ASR word landed in this bucket" as "no speech to attribute" and discards the
**entire** votes dict for the bucket, not just the two attribution voters (`speaker_assignment`,
`target_activity`) that logically need word timing. Everything else the function built for that
bucket -- per-diarizer labels, per-embedder cosine doubt, cross-diarizer disagreement, and J1/J2
change-point evidence, all of which the function's own comments say "stays" for other consumers
(e.g. identity_repair reads the change-point entries to place boundaries) -- is wiped along with
it.

This script calls the real `harvest_speaker_votes` on a synthetic 3-second, two-diarizer,
single-embedder pass where:
  - one ASR word lands only in bucket 0 ([0, 1)), so bucket 1 ([1, 2)) has zero word coverage
    while `fused_words` overall is non-empty (the gate's own precondition);
  - a background-mask region marks the whole recording `target_active`, so bucket 1 passes the
    *first* gate (it is not `target_free`) and reaches the wordless gate;
  - the two diarizers disagree about who is speaking in bucket 1 (cross-diarizer disagreement),
    an embedder's change-point series places two boundaries inside bucket 1 (J2), and the
    per-embedder cosine track is populated (J1 overlap_count is also present pass-wide).

It prints bucket 1's votes dict *before* the wordless gate would run (reconstructed from the same
inputs by calling the function up to, but excluding, that gate is not possible without duplicating
internals, so instead this script demonstrates the wipe directly): it calls
`harvest_speaker_votes` twice -- once with `fused_words=None` (gate disabled, per the function's
own documented escape hatch) to show what bucket 1's votes contain when populated, and once with
the real `fused_words` (gate enabled) to show every one of those entries is replaced by `{}`.

Must be run from the repository root. Loads no model, downloads nothing -- pure Python/numpy over
synthetic dicts.
"""

from __future__ import annotations

import sys

import numpy as np

from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.speaker import harvest_speaker_votes
from senselab.audio.tasks.speaker_embeddings.windowing import WindowEmbedding


def _make_pass_summary() -> dict:
    # Two diarizers with declared speaker capacity (occupancy.SPEAKER_CAPACITY), each reporting
    # one continuous segment covering the whole 3s recording but under DIFFERENT labels for the
    # middle third -- so the harmonizer's overlap matcher maps diarizer A's constant "SPEAKER_A"
    # onto the same cluster as diarizer B's "spk0" (2s of overlap) while "spk1" (1s, bucket 1
    # only) stays a distinct cluster -- i.e. real cross-diarizer disagreement in bucket 1.
    diar_a = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_A"},
    ]
    diar_b = [
        {"start": 0.0, "end": 1.0, "speaker": "spk0"},
        {"start": 1.0, "end": 2.0, "speaker": "spk1"},
        {"start": 2.0, "end": 3.0, "speaker": "spk0"},
    ]
    return {
        "duration_s": 3.0,
        "diarization": {
            "by_model": {
                "pyannote/speaker-diarization-community-1": {"status": "ok", "result": [diar_a]},
                "nvidia/diar_sortformer_4spk-v1": {"status": "ok", "result": [diar_b]},
            }
        },
        "background_mask": {
            "result": {
                # Whole recording target_active, so no bucket is target_free -- every bucket
                # reaches the wordless gate rather than being nulled by the first gate.
                "regions": [{"start": 0.0, "end": 3.0, "state": "target_active", "uncertainty": 0.0}]
            }
        },
    }


def _make_embeddings() -> dict[str, list[WindowEmbedding]]:
    # window_s = 1.0, hop_s = 0.5 -> lag = 2 boundary steps of 0.5s each, placing two
    # change-point boundary times (1.0 and 1.5) inside bucket 1 ([1, 2)), and leaving a
    # same-cluster embedding track (diarizer A's constant label) populated across buckets so
    # bucket 1 also carries per-embedder cosine-doubt votes.
    rng = np.random.default_rng(0)
    starts = [0.0, 0.5, 1.0, 1.5, 2.0]
    vectors = [rng.normal(size=8) for _ in starts]
    # Make the window straddling the disagreement (index 2, covering [1.0, 2.0)) noticeably
    # different so the change-point series has real (non-degenerate) distances.
    vectors[2] = vectors[2] + 5.0
    return {
        "ecapa": [
            WindowEmbedding(start_s=s, end_s=s + 1.0, vector=v) for s, v in zip(starts, vectors)
        ]
    }


def main() -> int:
    grid = BucketGrid(win_length=1.0, hop_length=1.0)
    pass_summary = _make_pass_summary()
    embeddings = _make_embeddings()

    # One consensus word inside bucket 0 only -- fused_words is non-empty overall (the gate's own
    # precondition, `if fused_words and ...`), but no word overlaps bucket 1 at all.
    fused_words = [{"text": "hi", "start": 0.2, "end": 0.6}]

    target_bucket = (1.0, 2.0)

    # ---- Run 1: gate disabled (fused_words=None), to show what bucket 1 contains when populated ----
    rows_gate_off = harvest_speaker_votes(
        pass_summary=pass_summary,
        grid=grid,
        per_window_embeddings=embeddings,
        speaker_floors={"ecapa": (0.30, 0.70)},
        fused_words=None,
    )
    row_off = next(r for r in rows_gate_off if (r["start"], r["end"]) == target_bucket)
    votes_off = row_off["votes"]

    # ---- Run 2: gate enabled (real fused_words), the actual production call shape ----
    rows_gate_on = harvest_speaker_votes(
        pass_summary=pass_summary,
        grid=grid,
        per_window_embeddings=embeddings,
        speaker_floors={"ecapa": (0.30, 0.70)},
        fused_words=fused_words,
    )
    row_on = next(r for r in rows_gate_on if (r["start"], r["end"]) == target_bucket)
    votes_on = row_on["votes"]

    print(f"Target bucket: {target_bucket}")
    print()
    print("=== votes with the wordless gate disabled (fused_words=None) ===")
    for k in sorted(votes_off):
        print(f"  {k}: {votes_off[k]}")
    print()
    print("=== votes with the wordless gate enabled (real fused_words, bucket has 0 word coverage) ===")
    print(f"  {votes_on!r}")

    # Confirm this bucket really did have zero word coverage and wasn't target_free (the intended,
    # narrower gate).
    from senselab.audio.workflows.audio_analysis.attribution import target_activity_doubt, word_coverage

    buckets = [(round(float(r["start"]), 6), round(float(r["end"]), 6)) for r in rows_gate_on]
    coverage = word_coverage(fused_words, buckets)
    mask_regions = pass_summary["background_mask"]["result"]["regions"]
    activity = target_activity_doubt(mask_regions, buckets)
    bucket_key = (round(target_bucket[0], 6), round(target_bucket[1], 6))
    print()
    print(f"word_coverage[{bucket_key}] = {coverage[bucket_key]}  (0.0 => 'no speech to attribute')")
    print(f"target_activity_doubt[{bucket_key}] = {activity[bucket_key]}  (state != 'target_free')")

    has_cross_diar = "__cross_diar_label_disagreement__" in votes_off
    has_change_point = any(k.endswith("::change_point") for k in votes_off)
    has_embedding_doubt = any("::" in k and not k.endswith("::change_point") for k in votes_off)
    wiped_to_empty = votes_on == {}

    print()
    print(f"populated votes included cross-diarizer disagreement: {has_cross_diar}")
    print(f"populated votes included change-point (J2) entries:   {has_change_point}")
    print(f"populated votes included per-embedder cosine doubt:   {has_embedding_doubt}")
    print(f"gate-enabled bucket_dict['votes'] wiped to {{}}:      {wiped_to_empty}")

    if (
        wiped_to_empty
        and has_cross_diar
        and has_change_point
        and has_embedding_doubt
        and coverage[bucket_key] <= 0.0
        and activity[bucket_key][1] != "target_free"
    ):
        print()
        print(
            "DEFECT REPRODUCED: bucket 1 had zero word coverage (0.0) and was NOT target_free, "
            "yet the wordless gate replaced its populated votes dict -- containing per-diarizer "
            "labels, cross-diarizer disagreement, per-embedder cosine doubt, AND J1/J2 change-point "
            "evidence -- with an empty dict ({}), instead of nulling only the two word-dependent "
            "attribution voters (speaker_assignment, target_activity)."
        )
        return 0

    print("Could not reproduce the defect as specified.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
