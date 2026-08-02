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
   `status` is `active | shadowed` and nothing else: no rule may remove a vote from aggregation.
   What a rule withdraws is **weight** — `VoteStore.attenuate_source_in_bucket` multiplies
   `vote.evidence_weight` by the measured corroboration, floored at `floors.MIN_EVIDENCE_WEIGHT`,
   and appends the factor to `provenance.evidence_weight_factors`. P3 acts this way on the presence
   and utterance axes (never identity: evidence that no one spoke here is silent about *which*
   speaker it was).
4. **Aggregation reads only `status=active` votes**, weighted by `policy.family_weights[family] ×
   payload.weight × vote.evidence_weight`. Aggregators are the existing pure functions
   (`aggregate.py`) plus the epistemic/aleatoric decomposition (D7). A source absent from the
   evidence-weight map was never measured and contributes unweighted — absent is not zero. An empty
   map is byte-identical to no map, which is what keeps `parity_check` comparing one quantity.
5. **Incremental re-aggregation**: after an intervention, only buckets intersecting its region core are
   re-aggregated; all other rows carry forward by reference (same values, same `round`).
6. **Final-state mirroring**: the last round's aggregated rows are also written to the pre-existing
   paths `<pass>/uncertainty/<axis>.parquet` and `uncertainty/raw_vs_enhanced/<axis>.parquet` with the
   existing column set intact (FR-024); adaptive columns (`status`, `epistemic`, `aleatoric_floor`,
   `elected_stream`, `irreducible_reason`, `round`) are additive.
7. **Provenance**: every parquet carries `schema.metadata` with `policy_hash`, `wrapper_hash`,
   `senselab_version`, `round`, mirroring `write_axis_parquet` (`io.py:22`).
8. **A withdrawal is published, not merely recorded.** Every belief parquet — per-round and
   `final/` — carries `n_attenuated_sources`, `attenuated_sources` (JSON `{source → weight}`) and
   `attenuation` (JSON list of `{source, evidence_weight, factor, corroboration,
   corroboration_pooling, evidence_sources, weight_map, floor, reason, measured_on, round}`),
   written by `fusion.attenuation_columns`. `n_sources` cannot stand in for this: attenuation
   keeps the source contributing by design, so the count is unchanged across a withdrawal.
   Unattenuated buckets are written as `0` / `"{}"` / `"[]"` — absence is stated, not left as a
   gap. Vote provenance is not a substitute either: `votes_added.parquet` only holds the round a
   vote was *added* in, so a round-3 withdrawal against a round-1 vote appears in no round file.

## Invariants (tested)

- No bucket loses evidence across rounds: `|active ∪ shadowed|` is non-decreasing, and every
  `evidence_weight` stays strictly above 0 — attenuation is bounded below by the shared floor, so a
  vote can be made unable to win but never made to disappear.
- Re-running `aggregate_all` over the reconstructed live set reproduces every round's belief parquet
  byte-identically (determinism, SC-004).
- For any (model, stream, bucket): at most one active vote.
- `history` in BeliefRow has exactly one entry per round that touched the bucket.
