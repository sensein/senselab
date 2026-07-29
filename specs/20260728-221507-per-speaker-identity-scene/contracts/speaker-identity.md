# Contract: per-speaker identity outputs

**Files**: `<run_dir>/final/speakers.json`, `<run_dir>/final/per_speaker_presence.parquet`,
`<run_dir>/final/convergence.json` (extended)

Replaces the single per-bucket identity scalar in the final convergence. Per the project's
pre-alpha position the old shape is **not** retained alongside (FR-001).

## `final/speakers.json`

```json
{
  "profile_version": "detection-margin/2026-07-29",
  "influence_profile": "influence/default",
  "generated_from_round": 3,
  "count_posterior": {
    "probabilities": {"1": 0.22, "4": 0.51, "5": 0.27},
    "support": {
      "1": ["pyannote/speaker-diarization-3.1", "nvidia/sortformer"],
      "4": ["embedding_silhouette"],
      "5": ["embedding_silhouette"]
    },
    "modal_count": 4,
    "is_multimodal": true,
    "converged": false
  },
  "speakers": [
    {
      "speaker_id": "S0",
      "existence_uncertainty": 0.18,
      "supporting_sources": ["pyannote/speaker-diarization-3.1", "embedding_silhouette"],
      "source_kinds": {
        "pyannote/speaker-diarization-3.1": "independent",
        "embedding_silhouette": "derived"
      },
      "first_seen": 0.08, "last_seen": 4.84, "total_active_s": 2.31,
      "converged": true,
      "revisions": [
        {
          "round": 2, "quantity": "speakers.S0.span",
          "before": [0.08, 1.60], "after": [0.08, 4.84],
          "caused_by": "identity_repair",
          "effective_weight": 0.42,
          "resolution_kind": "new_evidence",
          "evidence": {"change_point_prominence": 0.71}
        }
      ]
    }
  ],
  "label_correspondence": [
    {
      "source": "pyannote/speaker-diarization-3.1", "source_label": "SPEAKER_00",
      "speaker_id": "S0", "cluster_id": "c0",
      "source_kind": "independent", "confidence": 0.88
    }
  ]
}
```

### Invariants

- `count_posterior.probabilities` values sum to `1.0 ± 1e-9`; keys are stringified ints.
- Every `support` key appears in `probabilities` (FR-006).
- `is_multimodal` true ⟹ at least two counts exceed the policy threshold (FR-008).
- All sources agree on one speaker ⟹ `probabilities == {"1": 1.0}` and `speakers` has
  length 1 (FR-009, SC-001).
- Zero-speech recording ⟹ `probabilities == {"0": 1.0}`, `speakers == []`.
- Every `speakers[].source_kinds` value is `independent` or `derived` (FR-007).
- Every `revisions[]` entry carries `caused_by`, `effective_weight`, and
  `resolution_kind` (FR-011g, SC-026).
- `resolution_kind == "revision"` ⟹ the accompanying uncertainty drop is **not** reported
  as improved confidence anywhere downstream (FR-011d, SC-027).
- `existence_uncertainty` and per-bucket `presence_uncertainty` are separate fields and
  neither substitutes for the other (FR-004).

## `final/per_speaker_presence.parquet`

One row per `(speaker_id, bucket)`. Same grid as `final/presence.parquet`.

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `speaker_id` | `string` | no | |
| `start` | `double` | no | |
| `end` | `double` | no | |
| `presence_confidence` | `double` | yes | `[0,1]`, this speaker specifically |
| `presence_uncertainty` | `double` | yes | `[0,1]` |
| `overlap_with` | `list<string>` | no | May be empty; non-empty for concurrent speech |
| `contributing_sources` | `list<string>` | no | |
| `round` | `int32` | no | Round that produced the value |
| `resolution_kind` | `string` | no | `new_evidence` \| `revision` \| `unresolved` |

### Invariants

- For every `speaker_id` in `speakers.json`, rows cover the full recording duration
  (SC-003) — gaps are represented as rows with null confidence, never as absent rows.
- Grid matches `final/presence.parquet` exactly (same `start`/`end` pairs).
- Two speakers may simultaneously have high `presence_confidence`; `overlap_with` is then
  non-empty on both rows.

## `final/convergence.json` — added fields

```json
{
  "converged": false,
  "rounds_run": 3,
  "termination_reason": "oscillation",
  "oscillation_states": [{"count": 1}, {"count": 4}],
  "unresolved_quantities": ["count_posterior"],
  "per_quantity_resolution": {"speakers.S0.span": "new_evidence"},
  "influence_weights_applied": [
    {"signal": "embedding_silhouette", "target": "count_posterior",
     "base_weight": 1.0, "uncertainty_gate": 0.55, "derivation_gate": 0.4,
     "effective_weight": 0.22}
  ]
}
```

### Invariants

- `termination_reason == "oscillation"` ⟹ `oscillation_states` non-empty and
  `converged == false` (FR-011e, SC-028).
- Any quantity in `unresolved_quantities` is not presented as settled in any other
  artifact (FR-011h).
- A `derived` signal's `derivation_gate` is strictly `< 1.0` (FR-011c, SC-030).
- Reruns with identical inputs and settings produce byte-identical files (FR-011f,
  SC-004, SC-029) — this requires deterministic iteration order and stable key ordering
  in all serialized dicts.
