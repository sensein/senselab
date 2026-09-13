# Three-lane comparison — senselab v2 fold vs LLM reviewer vs enrolled re-run

The campaign's three verdict lanes over the same 112 recordings (2026-08-25):
senselab triage (`$SCRATCH/triage_b2ai_v2/aggregate/<sub>/per_file.tsv`), the LLM comparator
(`$SCRATCH/agentharness-triage-v2/b2ai-v2-merged-20260825/<sub>/outcomes.jsonl`, ctx-65536 pass
merged with the ctx-131072 re-review of its 8 context overflows), and the enrolled senselab re-run
(`$SCRATCH/triage_b2ai_v2_enr/delta/`).

## Triage axis

63/112 agree (56%). The confusion is one-sided:

| senselab / LLM | n |
|---|---|
| flag / flag | 52 |
| **flag / pass** | **43** |
| pass / pass | 11 |
| pass / flag | 6 |

senselab flags 43 files the LLM passes; the reverse happens 6 times. The senselab flags are
dominated by fold-level contradiction and uncertainty reasons (taxonomy-vs-branch disagreement,
one-line-absent folds) rather than content findings — the LLM, reading the same derivatives,
treats those as resolvable and passes. Neither lane found PII anywhere (0/112 in both).

## Kind axes

| kind | agree | dominant divergence |
|---|---|---|
| speech | 71% | LLM hedges `uncertain` on 28 files senselab decides; hard conflicts 5 |
| airway | 62% | LLM `uncertain` on 32 senselab-`absent` files, `present` on 9 of them |
| voice | 38% | **senselab reads `present` on all 112; LLM reads `absent` on 65** |

Two of these confirm registered items with independent evidence:

- **airway**: the 9 absent/present hard conflicts and 32 hedges are the direction the HeAR-floor
  row predicts — senselab under-reads airway at the 0.5 floor; a reviewer listening through the
  derivatives hears breaths senselab's fold calls absent. Evidence for the phase-C refit.
- **voice**: senselab's VOICE branch claims voiced residual on every file in the corpus — the
  architect review's D-1 (unconditional VOICE; reference model is the outlier) measured against an
  independent reviewer: 65/112 of those claims read `absent` to the LLM. Whichever way D-1 is
  decided, the current behaviour is the extreme end of the disagreement.

## Context-overflow re-reviews (LLM lane hygiene)

8 files overflowed ctx 65536; re-reviewed at 131072. Of the 2 that had carried truncated verdicts,
1 changed (`Prolonged-vowel`: pass→flag, airway absent→uncertain) — truncation was materially
biasing at least some reviews, so the merge (taller wins) is the right policy. The other 6
produced their only verdict in the tall pass.

## Enrolled lane

Orthogonal to both: 0/112 kind changes, 30 PII-attribution flag collapses, release decisions
structurally immune on this corpus (`speaker_count == 1` everywhere). The speech-only source
policy and its measurement live in `README.md` and the register's centroid row.
