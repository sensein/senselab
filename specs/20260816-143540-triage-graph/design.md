# The triage graph: a single-pass, evidence-honest per-file review decision

**Status:** design, 2026-08-16. Depends on `specs/20260815-215106-analyze-audio-audit/` — the
register, the statistical review, and the two remediation inventories are this design's inputs and
are not restated here.

## Goal

For one recording — or each recording from one target participant — produce, in a single pass:

| output | type |
| --- | --- |
| human-review flag | `bool` + ranked reasons |
| transcript | fused word stream with per-word confidence |
| speaker count | `Estimate` over integers |
| PII present | `Estimate` over {absent, present} + spans |
| recording quality | `Estimate` per degradation axis |
| task match | `Estimate`, only when the expected task is known |
| trim regions | list of spans with a reason each |

The flag is the deliverable; the other six are the evidence for it, and are individually useful.

## What this is not

`analyze_audio` is an **iterative refiner**: it asks "where should I spend more compute to reduce
uncertainty?" and loops until converged, irreducible, or out of budget. The graph asks "should a
human look at this?" and answers **once**.

They are not alternatives and neither replaces the other. They share signal extraction and
measurement; they differ entirely in what they do with it. The natural composition is a cascade —
**the graph is the cheap first pass, and the refiner is what you run on the files the graph could
not adjudicate.** A graph result of "flagged because the evidence was too thin to decide" is
precisely a refiner input, and this is the one place the two are wired together.

Nothing in the graph imports `adaptive/`. Inventory 3 of the decomposition establishes the test:
a module is refiner-only exactly when its data or control flow is keyed by round index or
accumulates state across calls. The graph contains no such module.

## The load-bearing decision: an estimate is a type, not a float

The statistical review's closing recommendation was that if the graph carries exactly one thing
forward, it should be N10 — **an evidence count on every published confidence, and shrinkage
toward a stated prior when that count is small.**

In the current codebase nothing does this, and the review measured the consequences: a bucket
backed by 4 unanimous sources and one backed by 20 both publish `P = 1.000`; a diarizer crashing
produces `speech_presence_confidence` indistinguishable from a diarizer agreeing; adding a
*low-reliability* signal moves published confidence from 0.800 to 0.420. Every one of those is the
same defect — a number published without the count of things that produced it.

So the graph does not publish floats. It publishes:

```python
@dataclass(frozen=True)
class Estimate:
    """A published quantity, with the evidence that produced it."""

    value: float          # what a consumer should use; already shrunk
    raw: float            # the unshrunk sample statistic
    n_evidence: int       # independent contributing sources (0 is legal and meaningful)
    prior: float          # what `value` collapses to as n_evidence → 0
    prior_key: str        # config key naming the prior, so its derivation is findable
    population: str       # the population this estimate was validated on
```

with

```
value = (n_evidence * raw + k * prior) / (n_evidence + k)
```

where `k` is a per-quantity pseudo-count from config. At `n_evidence == 0`, `value == prior` and
`raw` is undefined — the constructor requires `raw=None` in that case rather than accepting a
fabricated `0.0`. This is the register's F-156 fix (a fabricated 0.5 and a measured 0.5 must be
distinguishable) applied as a type rather than as a convention.

Three properties follow, and they are the reason this is the first task rather than a refinement:

- **A crashed backend cannot masquerade as an agreeing one.** F-147's defect becomes
  unrepresentable: a crash contributes no evidence, `n_evidence` drops, and the estimate visibly
  moves toward its prior.
- **Unanimity stops being free.** Four agreeing sources and twenty agreeing sources produce
  different values, which is the whole content of the review's finding on `speaker_identity.py`.
- **The review flag gets a principled second arm.** "We could not tell" is now a readable state
  (`n_evidence` below a configured floor) rather than something inferred from a suspiciously round
  number.

`population` carries the lifespan answer. The graph does not claim child-validated behaviour it
does not have; it names, per estimate, what the underlying model and thresholds were fitted on.
Verified-latent findings F-164, F-167, F-169, F-170, F-168 and F-173 are all "adult-derived value
applied without saying so" — this field is where they surface. Surfacing is not fixing, and the
spec says so out loud: the graph makes the gap legible and leaves the measurement to the lifespan
work the register scopes.

## The review flag

There is no labeled corpus of "recordings a human should have looked at", so the flag is a
**stated rule over the six outputs, not a fitted classifier.** Inventing a classifier here would
add exactly the kind of unmeasured decision the audit spent 176 findings cataloguing.

The flag fires when any of:

1. **A signal is bad enough to matter** — an output's `Estimate.value` crosses a configured
   review band, *and* `n_evidence` is at or above that output's evidence floor.
2. **The evidence is too thin to adjudicate** — any output's `n_evidence` is below its floor. This
   arm is not a fallback; it is the arm that makes the system honest, and it is what routes a file
   to the refiner.
3. **The recording contradicts its declared task** — only evaluated when `AudioHints` supplies one.

Each firing reason is emitted with the output that produced it, the estimate, and which arm fired.
A flag with no reasons is a bug, not a pass.

Arm 2 is the direct fix for F-150, where a total harvest failure yields
`high_uncertainty_rate = 0.0` and reads as a dramatic improvement over a partially-successful run.
Under the rule above, zero harvest means zero evidence, which flags.

## Architecture

Four layers, in dependency order. Layers 1–3 are lifted from `analyze_audio` by the decomposition
inventory; only layer 4 is new code.

**Layer 1 — extraction.** `stages.py`'s six `stage_*` functions plus `run_pass` are already
round-agnostic — plain dict fragments out, no `VoteStore`, no `Region`, no round numbers — and
`run_pass` is genuinely production-exercised, from `scripts/analyze_audio.py:611` rather than from
anywhere inside the package. This is the layer the graph extracts against.

**Correction, measured 2026-08-16, superseding this section's first draft.** The lift was justified
above on the claim that a graph wanting only `run_pass` must import "`contracts.py`, `adaptive/`,
and every other file in the 91-file package". That is false. The package resolves public symbols
through a lazy PEP-562 `__getattr__`, so importing it costs one module, and resolving `run_pass`
loads **9** submodules with `contracts` and `adaptive` both provably unloaded.

**Second correction, same date, superseding the first's second half.** That draft went on to say
the closure of `stages.py` + `stage_context.py` was "14 modules / 5,523 lines" reaching `axes.py`
"transitively through `calibration`, `types` and `grid`", concluded the lift was **a seven-times
larger refactor than costed**, and moved it into Phase 2 on that basis. The 14-module figure counts
every *function-local* import; the import-time closure was **8 modules / 2,939 lines**, and at
import time the only path to `axes` was `stages` → `sound_sources` → `grid` → `axes`, for one
constant. `calibration` and `types` were never on it. Three symbols crossed the edge in total.

The edge is now gone: `DEFAULT_TIME_GRID` moved to `grid.py` (which axes *read*, so it was the
arrow that was inverted), `sound_sources`'s `grid` import went under `TYPE_CHECKING`, and
`types.UncertaintyAxis` — a bare `str` — is declared locally. Import-time closure is **7 modules /
2,503 lines** with `axes` absent, pinned by
`stages_test.py::test_importing_the_extraction_layer_loads_none_of_the_refiner`. One edge is
accepted with its reasoning written down: `calibration` → `axes` for `CALIBRATED_AXES`, used only
in `validate_profile`, which the extraction closure never calls.

The lift therefore buys a conceptual boundary, not an import saving — that part stands — but it is
not a large refactor, and **Phase 2's four chain lifts need not be sequenced behind it**. See
`phase2-notes.md`, "Extraction boundary", for the measurements. Phase 1 shipped the two pieces that
needed no boundary decision.

**Layer 2 — four chain workflows**, each independently importable:

| module | chain | feeds |
| --- | --- | --- |
| `workflows/speaker_clustering/` | window → embed → speech-veto → cluster | speaker count |
| `workflows/transcript_fusion/` | transcribe → align → fuse across backends | transcript |
| `workflows/speaker_identity/` | diarize → harmonize → attribute | speaker count, PII context |
| `workflows/scene_quality/` | Brouhaha → quality → degradation → mask → sources | quality, task match, trim |

Each chain's functions are already pure over plain inputs; they are entangled only by being called
inside `compute.harvest_pass`, whose vote-bucket wrapping stays with the refiner.

**Layer 3 — task promotions.** The 16 signal extractions from Inventory 1 move to
`utils/tasks/` or `audio/tasks/` as the chains that call them are lifted. Promotion happens with
the chain that needs it, not as a separate sweep — a task promoted with no caller is how
`adaptive/provenance.py` became dead code.

**Layer 4 — seven output builders plus the flag.** Small, one per output, none looping, none
importing `adaptive/`. `pii.py`'s `detect_pii_in_pass` is already a thin single-pass adapter over
the standalone `senselab.text.tasks.pii_detection` task and moves structurally unchanged — but it
does **not** move as-is. No register finding touches `pii.py`, which reads as "the one clean
output" and is not what it means: the config sweep found three unfitted thresholds gating the PII
verdict directly (`presidio_score_threshold=0.4`, `gliner_threshold=0.5`, and a `count >= 2`
cross-model corroboration gate) with **no `pii:` config section in existence**, so none can be
changed without editing Python. The PII builder therefore carries the same config work as every
other output, and its `count >= 2` gate is precisely an evidence count — it becomes an `Estimate`
rather than a boolean.

## Input

The graph's input is `Audio` plus the optional `AudioHints` that landed on `alpha` in
`src/senselab/audio/data_structures/audio_hints.py`. `ExpectedSpeech` supplies the task-match
comparison; `TargetSpeakerEmbedding` supplies the reference for deciding whether a detected
additional speaker is an intruder or the expected interviewer. Without hints the graph runs and
omits the task-match output rather than guessing at one.

## What the graph must fix before lifting, and what it must only declare

Every defect the register marked `consumed` was found *inside the refiner*, where multiple rounds
and multiple diarizers partially dilute any single bad vote. **The graph has no such dilution** —
one pass, no second chance. This raises the priority of the wiring defects specifically, and it is
the reason for the split below.

**Fixed as part of lifting** (all are wiring or contract defects, all cheap, all made worse by
single-pass):

| finding | defect | why it cannot be carried |
| --- | --- | --- |
| F-162 | `fuse_consensus_words` called without `policy=` at the only reachable call site | the graph would inherit a config knob that reads as live and is not |
| F-150 | total harvest failure publishes `high_uncertainty_rate = 0.0` | it is the review flag's own input |
| F-147 | `speech_presence_confidence` unearned when a diarizer crashes | subsumed by the `Estimate` contract |
| F-165 | an empty `fused_words` bucket zeroes the entire votes dict | demonstrated, and discards exactly the non-lexical child vocalization the lifespan work cares about |
| F-144 | `multimodal_threshold` is decorative *and* not scale-free | deleted rather than wired: adding a diarizer flips the verdict with no change in the audio |

**Declared, not fixed** — every verified-latent population finding (F-164, F-167, F-168, F-169,
F-170, F-173). Each becomes a `population` value on the affected estimate. Fixing them needs data
this project does not yet have; the register already names the experiment for each.

**Explicitly excluded, and stated as excluded:** `sources.py`'s `screen_candidate` / `plan_excision`
and `foreground.py`'s `suppress_foreground` have no production caller today — designed and tested,
never integrated. The graph does not silently inherit them. Trim regions come from
`background_mask.py`'s `target_free` spans only, and the spec records that suppression depth is
not part of the first version.

## Config discipline

The config sweep found 60 unfitted decision-gating parameters and 14 dead config keys in the
refiner. The graph starts with a rule that prevents the recurrence rather than a promise to be
careful:

- Every decision constant lives in one versioned config with a `derivation` field beside it.
- `derivation: unfitted` is **permitted** — some values genuinely have no fit yet, and forcing a
  fabricated justification is worse. But every unfitted parameter in effect is written into the
  run's own output, so it appears in every artifact rather than only in a file nobody rereads.
- A test asserts every config key is read by some call site and every decision constant in graph
  code appears in config. That is the check that would have caught all 14 dead keys, including the
  four whole `RunConfig` fields built by `_build()` and never read.

## Phasing

Three PRs. Sequential, not independent — each is testable on its own.

**Phase 1 — foundation.** The `Estimate` type and its shrinkage, and the config-discipline guard —
which deletes the four `RunConfig` fields built by `_build()` and never read, rather than
allowlisting them, since pre-alpha policy is to remove outright. No new outputs, no file moves, no
boundary decisions. Everything else depends on this phase.

**Phase 2 — chains, and the extraction boundary.** The four chain workflows lifted, each with its
listed wiring defect fixed and its `population` values filled in — and, drawn in the same pass, the
layer-1 extraction boundary deferred from Phase 1, including what to do about the transitive
`axes.py` dependency. Each chain is one task with its own review gate.

**Phase 3 — outputs.** The seven builders and the flag rule.

## Success criteria

- No published number in the graph's output is a bare float; every one carries its evidence count,
  its prior, and the population it was validated on.
- A run in which every backend crashes produces a flagged file with reason "insufficient evidence",
  not a confident pass.
- Removing one of N agreeing sources changes the published estimate.
- The graph imports nothing from `adaptive/`, enforced by a test.
- Every config key is read; every decision constant is in config; unfitted ones are named in the
  run output.
- Each of the five fixed findings has a regression test that fails against the pre-lift code.

## Out of scope

- Model pluggability, speaker-count refinement beyond what chains 1 and 3 already do, new
  speech-extraction models, and lifespan validation. The register scopes each; the graph surfaces
  the gaps rather than closing them.
- Any change to the refiner's loop, budget, or convergence behaviour. `audio_analysis` keeps
  working through re-exports.
- A fitted review-flag classifier. Revisit when labeled review decisions exist.
