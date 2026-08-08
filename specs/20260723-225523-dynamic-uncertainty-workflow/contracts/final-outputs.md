# Contract: Final outputs (`<run_dir>/final/` + additive attachments)

## final/transcript.json

```jsonc
{
  "calibrated": false,               // true only when a calibration profile was applied (FR-021)
  "policy_hash": "…",
  "generated_from_round": 3,
  "language": "en",
  "words": [
    {
      "text": "hello", "start": 1.24, "end": 1.51,
      "speaker": "cluster_0",         // unified cluster id; null when unattributable
      "confidence": 0.93,
      "sources": ["nyralabs/CrisperWhisper2.0_turbo", "Qwen/Qwen3-ASR-1.7B"],
      "coverage": 0.75,
      "corroboration": 0.91,          // measured independent evidence for the winning text; null
                                      // when no member was measured — NEVER read null as 0
      "member_corroboration": {"Qwen/Qwen3-ASR-1.7B": 0.91, "openai/whisper-large-v3": null},
      "corroboration_evidence": {"p_independent": 0.91, "n_buckets": 2, "n_measured": 2},
      "alternates": [],               // [{text, share, share_uncorroborated, models, corroboration}]
                                      // when the *uncorroborated* winner share < policy threshold
      "flags": []                     // overlap | single_source | …
    }
  ],
  "segments": [ /* utterance-level rollup: start, end, speaker, text, min_word_confidence */ ],
  "corroboration": {
    "evidence_pool": ["frame_brouhaha_vad", "frame_segmentation"],
    "evidence_pool_rejected": {"acoustic_loudness": "never_reports_absence"},
    "pool_derivation": "support.evidence_signal_names + support.informative_evidence",
    "exponent": 1.0,
    "min_corroboration": 0.05,
    "segment_min_corroboration": 0.2,
    "n_words_measured": 412, "n_words_unmeasured": 0,
    "n_words_withheld_from_segments": 3,
    "withheld_word_indices": [88, 89, 210]
  }
}
```

`flags` lost `low_presence` and `hallucination_purged_nearby`. Both thresholded, at a bare
literal, the quantity `corroboration` now carries continuously; the number is the measurement and
the flag was a decision recorded where it could not be re-taken.

A word below `segment_min_corroboration` is **absent from `segments[].text` and present in
`words[]`**, carrying the measurement that excluded it. Keeping it in the readable transcript
would let it win — the deliverable would assert it, and the text consumers downstream (PII,
sentiment, summary) would ingest it. Dropping it from `words[]` would be erasure. `withheld_word_indices`
indexes into `words[]`, so the rollup is reproducible as a pure function of `words[]` plus one
number, with no model re-run.

Invariants: words sorted by (start, end); `0 ≤ confidence ≤ 1`; every `speaker` exists in
final/diarization.json; every word derivable from active votes in the belief store (SC-008).

## final/diarization.json (+ final/diarization.rttm)

```jsonc
{
  "clusters": [{"cluster_id": "cluster_0", "member_labels": {"pyannote/…": ["SPEAKER_00"], "nvidia/…": ["spk0"]},
                 "n_segments": 12, "total_speech_s": 43.1}],
  "segments": [{"start": 0.8, "end": 4.2, "cluster_id": "cluster_0",
                 "boundary_confidence": {"start": 0.9, "end": 0.55},   // from I1 evidence; 0.5 = unrefined
                 "overlap": false}]
}
```

RTTM sidecar for interop. Cluster ids come from the existing unified clustering
(`clustering.py:202`); boundary confidences from I1 change-point evidence where it ran, else 0.5.

## belief/speech_presence.parquet

Fused speech presence at the presence grid: `start, end, within_pass_uncertainty, epistemic,
aleatoric_floor, speech_presence_confidence, status, irreducible_reason?, round, elected_stream,
overlap_posterior?`, plus the attenuation columns `n_attenuated_sources, attenuated_sources,
attenuation` (belief-store.md invariant 8). This is the last round's presence belief — identical
values to `rounds/<K>/belief/speech_presence.parquet`, republished for discoverability.

`speech_presence_confidence` is a *weighted* fold, so the attenuation columns are load-bearing
rather than diagnostic: without them a bucket where every source agreed and one where the only
source that heard a speaker was discounted to the floor are the same row.

## final/convergence.json

Schema in data-model.md (ConvergenceReport). Additional invariants: `budget.spent` reconciles with
iterations.json (US5-3); `run_state=no_speech` runs contain only rounds[0]; every irreducible region's
`residual ≤ floor + epsilon` (D7).

## final/iterations.json

```jsonc
{
  "policy_hash": "…",
  "entries": [ /* InterventionRecord, ordered by (round, plan order) — see data-model.md */ ]
}
```

Byte-identical across identical runs (SC-004). Includes `status: deferred_budget | blocked_guard`
entries — the full decision surface, not only fired actions.

## Label Studio bundle (additive)

- New tracks: `final__consensus_transcript` (TextArea, word-level regions with confidence in meta),
  `final__presence` (Labels, converged/irreducible/open coloring), `final__diarization` (refined
  boundaries). Existing per-model and per-axis tracks unchanged (FR-023, FR-024).
- disagreements.json gains a sibling `final/disagreements_resolved.json`: entries from the round-1
  index annotated with `resolution: converged | irreducible:<reason> | budget_exhausted` and the
  intervention ids that touched them — the before/after story of the loop.

## summary.json (additive keys)

`adaptive: {run_state, termination_reason, converged, rounds_executed, policy_hash, budget: {...},
uncertainty_mass_by_round: {...}}` alongside the existing `global_uncertainty` block (whose semantics
are unchanged; it is now computed from the final belief state).

`run_state` is the loop's own reason for stopping; `termination_reason` is that reason after
non-convergence detection has had its say (FR-011e), and the two differ exactly when a run stopped
with nothing left to fire while its state was still trading places. Consumers deciding whether an
answer settled must read `termination_reason` / `converged`, never `run_state` alone.
