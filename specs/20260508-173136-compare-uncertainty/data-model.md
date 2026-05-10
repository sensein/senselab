# Phase 1 Data Model — Comparator Uncertainty Stage

## UncertaintyRow (parquet row)

One row per `(pass, axis, start, end)` bucket. Lives in one of nine parquets per default
two-pass run (3 axes × 2 passes + 3 raw-vs-enhanced deltas).

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `start` | float (seconds) | no | Bucket lower bound (inclusive). |
| `end` | float (seconds) | no | Bucket upper bound (exclusive). |
| `axis` | str | no | One of `presence` / `identity` / `utterance`. Stored once in parquet metadata too; mirrored on every row for joinability. |
| `aggregated_uncertainty` | float in [0, 1] | yes | The headline scalar. NaN only when `comparison_status != "ok"`. |
| `contributing_models` | list[str] | no | Model ids that voted on this bucket. Empty only for `incomparable` / `unavailable` rows. |
| `model_votes` | map[str → struct] | no | Per-model raw signals — see "Per-axis vote shape" below. |
| `comparison_status` | str | no | One of `ok` / `incomparable` / `unavailable` / `one_sided` (last only on raw_vs_enhanced parquets). |

### Per-axis vote shape (the `model_votes` struct)

The struct schema is identical across rows within an axis (Arrow nullable struct fields);
fields not applicable to a given model are stored as null.

**presence**:
| Field | Type | Notes |
|---|---|---|
| `speaks` | bool | The model's binary "is there a speaker?" vote on the bucket. |
| `native_confidence` | float | Optional native scalar in [0, 1] (e.g. AST top-1 score). Null when the model exposes none. |

**identity** (cross-model + raw-vs-enhanced sub-signals):
| Field | Type | Notes |
|---|---|---|
| `speaker_label` | str | Diarization speaker label assigned to this bucket (diar models only). |
| `embedding_cosine_to_prev` | float in [0, 1] | `1 − cos_sim` to prior bucket's embedding on the same speaker track (ECAPA / ResNet entries only). |
| `raw_vs_enh_disagrees` | bool | Did the same diar model assign different labels on raw vs enhanced? (Only on `raw_vs_enhanced/identity.parquet`.) |

**utterance**:
| Field | Type | Notes |
|---|---|---|
| `text` | str | Per-bucket transcript text (ASR contributors). Empty string when no token overlap. |
| `avg_logprob` | float | Whisper native average log-probability across overlapping chunks. Null when unavailable. |
| `phoneme_per_to_ppg` | float in [0, 1] | Phoneme error rate vs PPG reference (ASR rows only when PPG is provisioned). |

## DisagreementsIndex (`<run_dir>/disagreements.json`)

Top-level JSON. Schema documented in `contracts/disagreements.json.md`. Built once per run
by reducing across the 9 uncertainty parquets, ranked by `aggregated_uncertainty desc` with
axis-priority tiebreak (utterance > identity > presence) then `start` ascending.

## PerSegmentSpeakerEmbedding (cache entry)

Cache key tuple: `(audio_signature, "speaker_embeddings_per_segment", seg_signature, model_id, wrapper_hash, senselab_version)`.

Where `seg_signature` is `sha256(f"{seg.start}:{seg.end}:{seg.speaker}")` — stable across
runs as long as the upstream diarization output is unchanged.

Cache value: `{"vector": list[float], "model_id": str, "elapsed_s": float}`.

Used only at bucket-aggregation time for the identity axis's across-time sub-signal.

## BucketGrid (in-memory only)

| Field | Type | Notes |
|---|---|---|
| `win_length` | float (s) | Default `0.5`. |
| `hop_length` | float (s) | Default `0.5` (non-overlapping). |
| `name` | str | Provenance label (`"comparator"`). |

Iterates `(start, end, idx)` tuples covering `[0, duration]` such that
`start = idx × hop_length` and `end = start + win_length`. The last bucket is included only
when `start + win_length ≤ duration`.

## Per-axis aggregation contract

### presence
```python
def presence_uncertainty(votes: list[bool], native_confidences: list[float | None]) -> float:
    n = len(votes)
    if n == 0: raise NoRowError  # caller drops the row
    if n == 1:
        # entropy is 0; fall back to native confidence if available
        nc = next((c for c in native_confidences if c is not None), None)
        return 1.0 - nc if nc is not None else 0.0
    # Shannon entropy of the binary vote, normalized by log(n)
    p_true = sum(votes) / n
    p_false = 1.0 - p_true
    h = -(p_true * log(p_true or 1) + p_false * log(p_false or 1))
    return h / log(n)  # ∈ [0, 1]
```

### identity
```python
def identity_uncertainty(
    speaker_labels: dict[model_id, str],   # cross-model sub-signal
    raw_vs_enh: bool | None,               # raw_vs_enhanced sub-signal (None on per-pass parquets)
    embedding_cosines: dict[emb_model, float],  # across-time sub-signal
    aggregator: str,
) -> float:
    sub_signals: list[float] = []
    if len(speaker_labels) >= 2:
        n_pairs = comb(len(speaker_labels), 2)
        n_agreeing = sum(1 for a, b in combinations(speaker_labels.values(), 2) if a == b)
        sub_signals.append(1 - n_agreeing / n_pairs)
    if raw_vs_enh is not None:
        sub_signals.append(1.0 if raw_vs_enh else 0.0)
    if embedding_cosines:
        sub_signals.append(mean(embedding_cosines.values()))
    if not sub_signals:
        raise NoRowError
    return _apply_aggregator(sub_signals, aggregator)
```

### utterance
```python
def utterance_uncertainty(
    transcripts: dict[model_id, str],         # ASR contributors
    avg_logprobs: dict[model_id, float],      # Whisper-style native confidence
    ppg_per_values: dict[model_id, float],    # ASR-vs-PPG PER (when PPG present)
    aggregator: str,
) -> float:
    sub_signals: list[float] = []
    nonempty = [t for t in transcripts.values() if t.strip()]
    if len(nonempty) >= 2:
        wers = [jiwer.wer(t_i, t_j) for i, t_i in enumerate(nonempty) for t_j in nonempty[i+1:]]
        sub_signals.append(min(mean(wers), 1.0))
    if avg_logprobs:
        sub_signals.append(min(1.0 - exp(mean(avg_logprobs.values())), 1.0))
    if ppg_per_values:
        sub_signals.append(min(mean(ppg_per_values.values()), 1.0))
    if not sub_signals:
        raise NoRowError
    return _apply_aggregator(sub_signals, aggregator)
```

### Aggregator (`--uncertainty-aggregator`)

```python
def _apply_aggregator(sub_signals: list[float], name: str) -> float:
    confidences = [1.0 - u for u in sub_signals]
    if name == "min":
        return 1.0 - min(confidences)              # worst sub-signal wins
    if name == "mean":
        return 1.0 - mean(confidences)
    if name == "harmonic_mean":
        return 1.0 - harmonic_mean(confidences)
    if name == "disagreement_weighted":
        # uncertainty when at least one sub-signal disagrees, scaled by mean uncertainty
        max_u = max(sub_signals)
        return (1.0 - mean(confidences)) * max_u
    raise ValueError(name)
```

## State diagram (per bucket)

```
[upstream tasks complete]
         │
         ├── any contributing model has signal? ─── no ──> drop (no row)
         │
         yes
         │
         ├── all contributing signals fail to combine? ─── yes ──> emit row, status="incomparable"
         │
         │
         ├── required sub-signal unavailable (e.g. PPG missing for utterance)? ─── yes ──> use available sub-signals; status="ok"
         │     (status="unavailable" only when *all* sub-signals are unavailable)
         │
         │
         └── happy path ──> emit row, status="ok", aggregated_uncertainty in [0, 1]
```

Raw-vs-enhanced parquets additionally produce `status="one_sided"` rows when a model contributed
to one pass but not the other.
