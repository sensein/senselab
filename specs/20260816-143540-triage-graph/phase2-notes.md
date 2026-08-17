# Phase 2 working notes

Written mid-session so the analysis survives a restart. Everything here was verified by reading or
running code on `feat/triage-phase2-defects`; nothing is inherited from the register unchecked.

## Status

**Done and in PR #560** (→ `triage`): F-162 (linking policy reaches the fold, plus a call-site
guard), F-150 (`high_uncertainty_rate` is `None` with no denominator), the sdist packaging fix
(81 MB → 18 MB), and the ruff/mypy scoping of `specs/`.

**Next up:** F-165, then the four chain lifts and the extraction boundary.

## F-165 — analysed, not yet implemented

`speaker.py:518-543`, the attribution loop. Two early-exit branches clear the whole votes dict:

```python
if state == "target_free":            # :522
    bucket_dict["votes"] = {}; continue
if fused_words and coverage[key] <= 0.0:   # :527
    bucket_dict["votes"] = {}; continue
```

**Both voters written after those branches are word-independent** — this is stronger than the
register states, and it is the core of the finding:

| voter | source | depends on words? |
| --- | --- | --- |
| `speaker_assignment` (`:545`) | entropy over diarizer cluster assignments | **no** |
| `target_activity` (`:552`) | background-mask region state | **no** |

So the gate uses *word absence* as a proxy for *speech absence*, and on that proxy discards two
measurements that words do not determine. The proxy holds for adult connected speech — the comment
at `:528-541` defends it with "22 of the 29 buckets the axis flagged were wordless … inter-turn
silence" — and fails for any vocalization that is not lexical.

### Mask region states

`target_active`, `nontarget_active`, `indeterminate`, `target_free`, and `None` when no region
covers the bucket (`attribution.target_activity_doubt`, `background_mask.py`). `None` also covers
"the mask stage did not run", so it cannot be treated as evidence of anything.

### Proposed fix

Do not apply the word gate where the mask positively reports vocal activity:

```python
_VOCAL_ACTIVITY = ("target_active", "nontarget_active")
if state not in _VOCAL_ACTIVITY and fused_words and coverage[key] <= 0.0:
```

Behaviour per case, which is the part that must hold:

- **(a) adult inter-turn silence** — the case the current comment protects. True silence is
  `target_free` (handled by the branch above) or `indeterminate`/`None` (gate still applies).
  **Unchanged.**
- **(b) non-lexical vocalization** — `nontarget_active`, so the gate is skipped and the bucket keeps
  its two voters instead of being silently zeroed. **Fixed.**
- **(c) ASR missed real speech** — `target_active` with no words: the gate no longer nulls the
  speaker axis on an ASR failure. Arguably a second fix, and worth calling out in review.

Needs a fixture test per state, failing against current code. Never construct an unmocked `HFModel`.

## F-165 and F-168 are serially dependent — the register has them as independent

This is the finding worth carrying forward.

`data/audioset_source_map.json` maps `"Baby cry, infant cry"` → `"people"` and `"Crying, sobbing"`
→ `"people"`. The `speech` task's target vocabulary
(`data/detection_margin/2026-07-29.json`, `mask.target_event_types_by_task`) is
`["speech", "breath", "mouth_noise"]`, and the only three task types are `speech`, `breath`,
`cough`. `"people"` is a **background source category**, so an infant's cry is classified as
background — like a passerby.

Consequences, in order:

1. The cry lands in a `nontarget_active` bucket, not `target_free`.
2. It therefore reaches the word gate, has no words, and is zeroed. **That is F-165**, and fixing
   F-165 stops the silent drop.
3. But the mask still says the vocalization was *not the target*. **That is F-168**, and until it is
   fixed the retained evidence carries the wrong label — the graph's trim-region output would
   propose trimming an infant's vocalization away as background.

So F-165's fix is necessary and not sufficient. Neither finding's row says this. **Do not claim the
pediatric path works after fixing F-165 alone.**

F-168 must not be hand-fixed here: CLAUDE.md requires a profile be regenerated from measured
verdicts rather than edited, and there are no measured pediatric verdicts in the repo. It stays
verified-latent until that data exists.

## F-144 — deferred against the design, with the measurement

Recorded in full in PR #560's description. Short form: `multimodal_threshold=0.15` is unfitted and
not scale-free — a diarizer that *agrees with the majority* flips `is_multimodal` True→False between
5 and 6 sources, with no change to the audio and no change to the dissenting evidence. But
`speaker_identity.py:585` gates `converged` on it, so deleting it changes convergence. It needs the
evidence-carrying replacement Phase 3 builds when speaker count becomes an `Estimate`.

## Extraction boundary — still open

Deferred from Phase 1 after measuring that the design's justification was wrong (the lazy
`__getattr__` already prevents the import cost; `contracts` and `adaptive` stay unloaded). The real
closure of `stages.py` + `stage_context.py` is 14 modules / 5,523 lines and reaches `axes.py` — the
refiner's own axis vocabulary — transitively through `calibration`, `types` and `grid`. That
`axes.py` edge is the thing to resolve before any move: extraction should not depend on the axis
vocabulary it is supposed to be independent of.
