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

**Done on `fix/extraction-axes-edge`:** the extraction boundary (below) — it turned out to be three
one-line edits, not the refactor this file and `design.md` had both recorded — plus four defects
filed as F-188..F-191.

**Next up:** the four chain lifts. Nothing now blocks them.

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
| each streaming-audio run | 39 | 33 `nontarget_active`, 5 `indeterminate`, 1 `target_active` | 34 |

**And one semantic question is open, which is the real reason not to wire it blind.** A reviewer
asked whether `nontarget_active` belongs in `_VOCAL_ACTIVITY` at all. `_classify_bucket`
(`background_mask.py:252-266`) reaches the `nontarget_confidence >= 0.5` test at `:261` only after
the bucket has passed `uncertainty <= max_free_uncertainty` **and** `confidence <= free_at`, so the
state means "the target is confidently *absent*, and some non-speech source scored ≥ 0.5" — the
conjunction the function's own docstring at `:252-254` states. That reading **strengthens** the
concern rather than dissolving it. The second conjunct is a `max` over every non-speech source
category (`nontarget_confidence_by_bucket`, `:691-728`, excluding `speech` alone), and 292 of the
527 labels in `data/audioset_source_map.json` map to `environment`, which is also the map's
`default` for an unmapped label and where `Silence` itself lands. So `nontarget_active` is reachable
with the target confidently absent and nothing but environmental mass present — "something
non-speech has mass here", not "a voice was heard" — and 33 of the 34 buckets in the table above are
`nontarget_active`. If that reading is right, the membership is wrong and the fix spares the wrong
buckets. Nobody has measured what those 33 contain. **Open question, not a verdict**, and it must be
answered before the regions are threaded through.

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

## Extraction boundary — resolved, and it was much smaller than this file recorded

Deferred from Phase 1 after measuring that the design's justification was wrong (the lazy
`__getattr__` already prevents the import cost; `contracts` and `adaptive` stay unloaded). The note
then said the closure of `stages.py` + `stage_context.py` "reaches `axes.py` … transitively through
`calibration`, `types` and `grid`", and treated that as the thing to resolve before any move.

**That sentence conflated two graphs, and the conflation is what made the edge look big.**
Re-measured 2026-08-16, three ways, because the three answer different questions and only one of
them is "what does an import cost". Counts are of package submodules, excluding the package
`__init__` that any import of it loads, and including the two roots. Line totals are as of this
commit and move whenever a docstring does; the module counts are the load-bearing part.

| closure of `stages` + `stage_context` | before | after |
| --- | --- | --- |
| **actually loaded** — `sys.modules` after `import …stages` in a fresh interpreter | 8 / 2,939 | **6 / 2,411** |
| AST, module level, counting `if TYPE_CHECKING:` blocks | 8 / 2,939 | 7 / 2,486 |
| AST, including function-local imports | 14 / 5,530 | 14 / 5,516 |

The first row is the one to quote. The second and third are upper bounds on it: an import under
`TYPE_CHECKING` is not executed, and a function-local import is not executed at import time. The
7th module in row 2 *is* `grid`, reached only under `TYPE_CHECKING` from `sound_sources` — so
counting it in a row labelled "what an import costs" would repeat, one edit later, exactly the
category error this section exists to correct. The first two rows' **before** figures are identical
because nothing in that closure had a `TYPE_CHECKING` import yet; the whole gap between them opened
with edit 2.

At **import time the only path to `axes` was `stages` → `sound_sources` → `grid` → `axes`.**
`calibration` and `types` are reached solely by function-local imports inside `stages.py`
(`:532`, `:718` → `calibration`; `:534`, `:719` → `io` → `types`), so neither was ever on the
import-time path. `import …stage_context` on its own did not load `axes` at all — every path came
through `stages`. And exactly three symbols crossed the edge: `DEFAULT_TIME_GRID`, `AxisName`,
`CALIBRATED_AXES`.

Three edits removed it, none of them a refactor — one moved constant, one deferred import, one
local declaration. Why each is the right shape is under "Rationale relocated out of code" below.

1. **`DEFAULT_TIME_GRID` moved `axes.py` → `grid.py`**, which leaves `grid.py` with **zero
   intra-package imports**.
2. **`sound_sources.py`'s `grid` import moved under `TYPE_CHECKING`.** That single line was the
   whole import-time edge.
3. **`types.UncertaintyAxis` is declared locally** instead of aliasing `axes.AxisName`.

`stages_test.py::test_importing_the_extraction_layer_loads_none_of_the_refiner` is the guard: fresh
interpreter, import `stages`, assert `axes`, `contracts` and `adaptive.*` are all absent from
`sys.modules`. It failed before these edits (`['…audio_analysis.axes']`) and passes after. Without
it the edge grows back silently, which is how it arrived.

**What that guard does not cover, stated so nobody reads it as more:** it is an *import-time*
check. An `axes` import moved inside a function body passes it, and one such edge is live today —
see the next paragraph. The guard's docstring says this; `design.md` says it too.

**The `calibration` → `axes` edge is accepted, not removed.** `calibration.py:46` imports
`CALIBRATED_AXES`, which is genuine axis vocabulary, **at module level** — so importing
`calibration` at all loads `axes`. What keeps it off the import-time path is the other end:
`stages.py` imports `calibration` only from inside function bodies (`:532`, `:718`). The
consequence is that running `run_pass` on the unmodified variant with the default
`PassPlan.background_mask=True` **does** load `axes`; only importing the extraction layer does not.
The reasons to accept it anyway:

- `CALIBRATED_AXES`'s only use is in `validate_profile` (`:89`, the loop at `:107`), reached from
  `load_calibration_profile`, which the extraction closure never calls. The closure uses the
  *detection-margin* half of the module (`load_detection_margin_profile` →
  `validate_detection_margin_profile`, from `stages.py:532,718`) and `noise_floor.py`'s
  `quantile_bias_correction_db`; neither touches `CALIBRATED_AXES`.
- It guards a `temperature` block that `calibration.py:24-30` itself declares reaches no fold.
- `calibration.py`'s two-schema split is deliberate and documented, so splitting the module to
  break the edge costs more than the edge does.

`calibration.py`'s module docstring used to end "Stdlib-only; safe to import anywhere", which that
module-level import falsifies. Corrected in the same pass.

**Consequence for Phase 2: the four chain lifts need not be sequenced behind this.** The design's
"seven-times larger refactor" was an artefact of counting function-local imports against an
import-time claim. The boundary now holds as a checked fact rather than an intention, and the
chains can be lifted in whatever order their own dependencies allow.

### Rationale relocated out of code

CLAUDE.md's Code Style section now puts measurements, the failure behind a choice and rejected
alternatives in `specs/`, not in docstrings and comments. The blocks below were written inline
during this work and are moved here rather than deleted; each cites the file that now carries a
short factual line and a pointer back.

**`grid.DEFAULT_TIME_GRID` — window equals hop.** Not a coincidence. The run that motivated the
constant used a 0.1 s window at a 0.02 s hop: adjacent rows shared 80% of their audio, so 1070 rows
were not 1070 independent measurements and nothing told a consumer so. A fine *resolution* is what
the question justifies; reporting five near-duplicate rows per window is not the same thing, and
the near-duplication was invisible in the output. 100 ms is sufficient for the downstream needs
known today — speech and target-activity onsets resolve at it, and speaker turns and mask regions
are much longer. (F-184 cites this measurement; its row now points here.)

**`grid.DEFAULT_TIME_GRID` — why it lives in `grid.py`.** It was declared in `axes.py`, which
inverted the arrow: an axis *reads* the grid it is estimated on, so the grid does not belong to the
axes. `axes.py` never read it — the name appeared once, as its own assignment target, every other
mention being prose — and the one line importing it back the other way (`grid` → `axes`) was, at
import time, the only path from the extraction layer to the refiner's axis vocabulary. `grid.py`
now has zero intra-package imports, which is what makes the boundary checkable instead of
incidental.

**`sound_sources`'s `BucketGrid` import under `TYPE_CHECKING`.** The runtime uses are
`grid.iter_buckets(...)` on the instance the caller passes, so nothing in the module needs the class
object; the module has `from __future__ import annotations`. Deferring the import keeps
`sound_sources` — and through it `stages` — off that one path.

**`types.UncertaintyAxis` declared locally rather than aliased from `axes`.** The axis set is open:
`task` is declared-but-punted, a fifth may follow, and a type enumerating the members is a promise
the pipeline cannot keep. This alias *was* a three-member `Literal`, justified as "narrower than the
set L2 fuses", and that narrowing is precisely what made `background_mask` unrepresentable in every
consumer that needed to act on it. `str` is now the whole content of the alias, so importing it from
`axes` bought an edge and nothing else — `types` is reachable from the extraction layer. That the
two declarations cannot drift is checked at the source level in `axes_test.py`, because an equality
assertion could not fail: both sides are the builtin `str` whatever either module did.

**`run_config._validate`'s deleted conjunct.** Filed as F-188; the mechanism is in that row.

**`PassPlan.mask_grid` — the measured cost of not sharing presence's grid.** Presence produced 1070
buckets at 100 ms and the mask 43 at 0.5 s, so five presence judgements were projected onto each
mask bucket before the mask could say anything. Every projection is a place to lose localisation;
on a shared grid row *i* of one is row *i* of the other and the coupling is exact. The decision is
D-24 in `specs/20260728-221507-per-speaker-identity-scene/layered-architecture.md`.
