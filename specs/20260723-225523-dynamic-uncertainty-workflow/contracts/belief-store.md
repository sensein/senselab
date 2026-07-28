# Contract: Belief store (votes + aggregated belief)

Layout under `<run_dir>/`:

```text
rounds/
├── 0/                      # triage
│   ├── belief/presence.parquet        # presence-only (quality/scene/posterior voters)
│   ├── decisions.json                 # enhancement + no-speech gates (trigger values)
│   └── summary.json
├── 1/                      # baseline
│   ├── belief/votes_{presence,identity,utterance}.parquet
│   ├── belief/{presence,identity,utterance}.parquet
│   ├── elections.json                 # per-region StreamElection
│   └── summary.json
├── <k>/                    # k = 2..K intervention rounds
│   ├── belief/votes_*.parquet         # ONLY rows added/updated this round (append-only design)
│   ├── belief/{presence,identity,utterance}.parquet   # full re-aggregated state
│   ├── regions.json
│   └── summary.json
final/ ...                  # see contracts/final-outputs.md
```

## Rules

1. **Append-only votes**: a round never rewrites earlier rounds' vote files. Shadowing (D5) is
   expressed by setting `shadowed_by`/`status` on the *logical* vote via the current round's file;
   readers reconstruct the live set as: latest status per vote id wins.
2. **Vote id**: `sha1(axis|bucket_start|source_model|stream|scope)` — deterministic; a re-fired
   intervention on the same crop overwrites its own logical vote (idempotent).
3. **Shadowing**: `scope=region:*` from (model, stream) shadows `scope=file` from the same
   (model, stream) in buckets covered by the region core. Never across models, never across streams.
   `status=purged_hallucination` (from P3) excludes a vote from aggregation on **both** presence and
   utterance axes; the row persists.
4. **Aggregation reads only `status=active` votes**, weighted by `policy.family_weights[family] ×
   payload.weight`. Aggregators are the existing pure functions (`aggregate.py`) plus the
   epistemic/aleatoric decomposition (D7).
5. **Incremental re-aggregation**: after an intervention, only buckets intersecting its region core are
   re-aggregated; all other rows carry forward by reference (same values, same `round`).
6. **Final-state mirroring**: the last round's aggregated rows are also written to the pre-existing
   paths `<pass>/uncertainty/<axis>.parquet` and `uncertainty/raw_vs_enhanced/<axis>.parquet` with the
   existing column set intact (FR-024); adaptive columns (`status`, `epistemic`, `aleatoric_floor`,
   `elected_stream`, `irreducible_reason`, `round`) are additive.
7. **Provenance**: every parquet carries `schema.metadata` with `policy_hash`, `wrapper_hash`,
   `senselab_version`, `round`, mirroring `write_axis_parquet` (`io.py:22`).

## Invariants (tested)

- No bucket loses evidence across rounds: `|active ∪ shadowed ∪ purged|` is non-decreasing.
- Re-running `aggregate_all` over the reconstructed live set reproduces every round's belief parquet
  byte-identically (determinism, SC-004).
- For any (model, stream, bucket): at most one active vote.
- `history` in BeliefRow has exactly one entry per round that touched the bucket.
