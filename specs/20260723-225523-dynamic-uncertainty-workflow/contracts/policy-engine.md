# Contract: Policy engine

The engine is a pure function:

```python
plan_round(belief: BeliefState, regions: list[Region], budget: BudgetLedger,
           policy: Policy, round_idx: int) -> list[PlannedIntervention]
```

No I/O, no model loads, no randomness. Given equal inputs it returns an identical plan (FR-025).

## Policy file (`adaptive/policy/default.yaml`)

```yaml
version: 1                      # bumped on any semantic change; sha256 of file → policy_hash
thresholds:
  theta_speech: 0.5             # triage speech gate (FR-004)
  theta_enh: 0.4                # quality degradation gate for --enhancement auto (FR-003)
  theta_high: 0.66              # region seed (matches disagreements HIGH)
  theta_low: 0.33               # converged (matches LS low bin)
  epsilon: 0.05                 # min per-intervention improvement (FR-017)
regions:
  gap_merge_s: 0.5
  pad_s: 1.0
  top_n_per_round: 8
  max_region_rounds: 2
budget:
  medium_per_run: 24
  heavy_per_run: 4
  per_round_fraction: 0.5       # ≤ this fraction of remaining budget per round
election:
  weights: {presence_conf: 0.4, quality: 0.3, utterance_agreement: 0.3}
  guard_min_raw_ppg: 0.2        # enhancement-artifact guard (FR-015)
families:                       # FR-008
  whisper: ["openai/whisper-*", "nyralabs/CrisperWhisper*"]
  nemo: ["nvidia/canary-*", "nvidia/stt_en_*"]
  qwen: ["Qwen/Qwen3-ASR*"]
  granite: ["ibm-granite/*"]
reserve_asr_models: ["openai/whisper-large-v3-turbo"]   # U2 escalation pool
rules:                          # enable/disable + per-rule params
  U1_region_reasr: {enabled: true}
  U2_reserve_escalation: {enabled: true}
  U4_overlap_separation: {enabled: false}   # v2
  # ... every rule in contracts/interventions.md
```

## Scheduling

1. Collect candidate (rule, region) pairs where the rule's trigger predicate holds and no guard blocks.
2. Score `priority = expected_gain(rule, region) / cost(rule)`; `expected_gain` is the rule's declared
   heuristic over belief values (e.g. for U1: `epistemic × uncertainty_mass`), never a learned model.
3. Sort by (priority desc, axis priority utterance > identity > presence, region start asc, rule id) —
   total order, no ties.
4. Greedily admit while class budgets and `per_round_fraction` allow; the rest are logged
   `deferred_budget` and become `next_actions` candidates.
5. Execute admitted interventions **sequentially in sorted order** (deterministic vote-store state);
   each runs in the failure envelope (D11).
6. Re-aggregate covered buckets; update region/bucket statuses (FR-017); end round.

Loop termination (FR-019): `round_idx > max_rounds`, or zero fired interventions in a round, or all
regions closed.

## Budget ledger

`BudgetLedger` tracks caps/spent per class; every admit decrements atomically before execution; failed
interventions still count their class cost (they consumed the resources). Ledger state is serialized
into each `rounds/<k>/summary.json` and must reconcile with `final/convergence.json.budget` (US5-3).

## Guards (evaluated before admission)

- `region.status != open` → skip.
- `interventions_remaining == 0` → skip (region closes as irreducible/budget_exhausted per FR-017).
- Axis presence gate (C4): identity/utterance rules require region mean p_voice ≥ theta_speech unless
  the rule is P3/C9 (which exist to adjudicate exactly that disagreement).
- Rule-specific guards declared in contracts/interventions.md (e.g. min crop length, stream
  availability, model availability).

## Determinism requirements

- All floats compared with explicit rounding (1e-9) before ordering.
- `policy_hash = sha256(canonical yaml bytes)` recorded in every round summary, iterations.json,
  convergence.json, and parquet metadata.
- The engine's unit tests replay recorded belief fixtures and assert byte-identical plans.
