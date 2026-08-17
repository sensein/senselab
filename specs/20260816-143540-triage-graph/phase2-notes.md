# Phase 2 working notes

Written mid-session so the analysis survives a restart. Everything here was verified by reading or
running code on `feat/triage-phase2-defects`; nothing is inherited from the register unchecked.

## Status

**Done and in PR #560** (→ `triage`): F-162 (linking policy reaches the fold, plus a call-site
guard), F-150 (`high_uncertainty_rate` is `None` with no denominator), the sdist packaging fix
(81 MB → 18 MB), and the ruff/mypy scoping of `specs/`.

**Done on `fix/f165-mask-aware-word-gate`:** F-165 (`d8cb7449`, the mask-aware word gate; `a2af9f75`,
six mutants its tests let through), and the register/prose corrections that came out of it —
including **F-187**, a new finding: the mask's region table never reaches the code that reads it, so
the F-165 fix is inert in production along with two decisions that predate it (below).

**Next up:** the four chain lifts and the extraction boundary.

## F-165 — shipped in `d8cb7449`, and inert until F-187 is decided

The line numbers in this section are the pre-fix ones (`speaker.py:518-543`); the loop now sits at
`speaker.py:553-586`. The analysis below is kept as written, because what shipped is exactly it.

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

### The fix, as shipped in `d8cb7449`

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
  speaker axis on an ASR failure. **Shipped, deliberately, as a second fix in the same commit**, and
  called out as such in `d8cb7449`'s message rather than smuggled in with (b): the two share one
  condition (`state in _VOCAL_ACTIVITY`) and separating them would have meant two spellings of one
  branch. It has its own test.

Five fixture tests shipped, one per region state, all of them failing or unpinned against the
pre-fix code; `a2af9f75` took them to 21 after mutation testing found six mutants they let through
(including a fifth `MASK_STATES` member upstream, which is now a type error here). No unmocked
`HFModel` is constructed.

## F-187 — the mask's regions never reach the code that reads them, so none of this fires yet

Found while testing the F-165 fix, verified twice independently, and filed as a new register row
(`specs/20260815-215106-analyze-audio-audit/register.md`, id allocated after the audit closed).

- `stages.py:569,578-580` puts `BackgroundMask.to_json()` into the pass summary under
  `background_mask.result`.
- `BackgroundMask.to_json()` (`background_mask.py:152-168`) emits **only the aggregate counters** —
  there is no `regions` key. The per-region table exists solely in `to_rows()` →
  `L2/background_mask.parquet`.
- `speaker.py:545` reads `mask_doc.get("regions") or []`, so it is always `[]`,
  `target_activity_doubt` returns `(None, None)` for every bucket, and `state` is always `None`.

So the new `_VOCAL_ACTIVITY` branch never fires — and neither does the pre-existing `target_free`
clear, nor the `target_activity` voter, which has never been emitted at all. Corroborated in the
artifacts: no run under `artifacts/analyze_audio/` has an `L1/signals/target_activity.parquet`
though every other vote name has one, and `L2/background_mask.json` on all three runs carries
exactly 13 counter keys and no `regions`.

**Wiring it through was deliberately not done.** It is a behaviour change, so what it would cost was
measured first — over the three completed runs, each run's final consensus words on the shipped
0.1 s grid against that run's own `L2/background_mask.parquet`:

| run | wordless buckets | states | would stop being cleared |
| --- | --- | --- | --- |
| 21 s two-speaker conversation | 28 | 25 `indeterminate`, 3 `target_active` | 3 |
| each streaming-audio run | 39 | 33 `nontarget_active`, 1 `target_active` | 34 |

**And one semantic question is open, which is the real reason not to wire it blind.** A reviewer
asked whether `nontarget_active` belongs in `_VOCAL_ACTIVITY` at all. It requires only
`nontarget_confidence >= 0.5` (`background_mask.py:261`); that quantity is a `max` over every
non-speech source category (`nontarget_confidence_by_bucket`, `:691-728`, excluding `speech` alone);
and 292 of the 527 labels in `data/audioset_source_map.json` map to `environment`, which is also the
map's `default` for an unmapped label and where `Silence` itself lands. So `nontarget_active` may
mean "something non-speech has mass here", not "a voice was heard" — and 33 of the 34 buckets in the
table above are `nontarget_active`. If that reading is right, the membership is wrong and the fix
spares the wrong buckets. Nobody has measured what those 33 contain. **Open question, not a
verdict**, and it must be answered before the regions are threaded through.

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
