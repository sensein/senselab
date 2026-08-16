# Auditing `analyze_audio`: a verified findings register

**Status:** design approved 2026-08-15. Branch `feat/analyze-audio-audit`, cut from `alpha`
at `f8dffb8b`.

## Goal

Produce a register of verified defects, assumptions and layering problems in
`audio/workflows/audio_analysis`, so that the changes it implies can be decided against the
triage-workflow ("graph") design rather than guessed at.

**This audit changes no code.** Its deliverable is a document.

## Why this comes first

Five concerns were raised together: audit `analyze_audio`; make it take other models; refine
the 1-vs-more speaker decision; add new speech-extraction models; make it work across the
lifespan. The last four are shaped by what the first one finds — which thresholds were never
fitted, which stages assume adult read speech, where a model choice is baked into control flow
rather than config. Specifying them before the audit means specifying against guesses.

A sixth item, the per-file triage workflow, is deliberately *not* blocked on this: it consumes
`analyze_audio`'s outputs rather than its internals. The audit informs it by saying which
outputs to trust.

## What the surface actually is

Measured on `alpha` at `f8dffb8b`, 81 files:

| Layer | Lines | Definition |
| --- | --- | --- |
| Orchestration | 5,423 | Code that imports `senselab.audio.tasks` or `senselab.utils.tasks` |
| Computation | 11,721 | Code that imports no task — statistics, contracts, fusion, plotting |
| Prose | 10,888 | 8,796 docstring lines + 2,092 comment lines |
| Blank | 4,393 | |
| **Total** | **32,425** | |

Two facts follow, and they are the reason the audit is sliced this way rather than by module:

**"32k lines of `analyze_audio`" is misleading.** Only ~5.4k lines are workflow. More than twice
that is general-purpose computation living inside a workflow package — `contracts.py` alone is
1,241 code lines and never touches a task. That is a layering finding before the audit begins,
and it matters to the graph: computation that lives in a workflow cannot be reused by another
workflow without importing it.

**Prose stands at 0.64 lines per line of code**, and reaches 2.76× in `estimates.py`. The
repository's convention is to explain *why* rather than *what*, and that convention has been
applied past the point of usefulness — much of the prose restates readable code, and the part
that is genuinely load-bearing (a measurement behind a threshold) does not need to be wrapped
around every function to survive.

## The four sweeps

Each is one or more subagents holding one question. Two rules bind all of them: every finding
carries `file:line` plus a concrete failure scenario (inputs → wrong output), and every agent
reports what it checked and found **clean**, so silence is distinguishable from absence.

### Sweep A — prose

Runs first, because its output changes what the other sweeps read. Three targets with three
different fates:

- **Descriptive prose that restates readable code** → delete. A docstring reading "Return the
  fraction of sessions where the backend reported exactly `true_k`" above
  `def exact_count_accuracy` earns nothing.
- **Rationale — the *why*, the measurement behind a number** → migrate to a per-module-group
  summary document. Load-bearing, but not per-function.
- **Stale or false prose** → a finding, not a cleanup. This class is not hypothetical: four
  instances were found and fixed in the two days before this audit — a module docstring claiming
  a 2.0 s / 1.0 s default against a 1.0 s / 0.5 s signature; a PII module documenting two
  detectors where three run; a "no boolean anywhere in its output" claim contradicted by a bool
  field; and `p_voice` framing that survived its own consumer's removal.

**The trap:** deleting a rationale that a later reader needs. Mitigated by migrating rather than
deleting anything that states a measurement, a failure, or a rejected alternative.

### Sweep B — the computation layer (11,721 lines)

Audited as mathematics, not as workflow.

- **Unfitted thresholds.** Numeric literals that gate a decision, without a written derivation.
  *Trap:* not every literal is a threshold. A window length chosen for memory, a batch size, a
  Hann overlap satisfying COLA are operational knobs. Report only numbers that change a verdict.
- **Statistics that do not mean what their name claims.** The live precedent is
  `p_voice = 0.5·(silhouette + 1)`: a partition-and-metric-dependent index rescaled into
  something read as a probability. *Trap:* this needs the consumer, not just the producer — a
  defensible computation can be misread downstream, visible only by tracing the value.
- **Unearned confidence.** Agreement computed over sources that are not independent; a
  confidence that does not degrade when an input is missing; `0.0` where `None` is meant.
- **Promotion candidates.** Computation with no workflow dependency that belongs in
  `utils/tasks/` — recorded with its target layer, not moved.

### Sweep C — the orchestration layer (5,423 lines)

The genuinely workflow-shaped concerns.

- **Models baked into control flow.** A backend named in a branch rather than selected through
  config; a hardcoded default that cannot be overridden; a stage that silently requires one
  model's output shape. This sweep scopes the model-pluggability work. *Trap:* prefix dispatch
  to a backend-specific worker is legitimate. The finding is when a *decision* depends on which
  model ran, or when adding a model means editing control flow.
- **Contract violations and ordering dependencies.** A stage's output consumed downstream in a
  way its producer does not guarantee; a stage that must run before another with nothing
  enforcing it.
- **Call-site correctness.** Helpers that are correct while their callers pass the wrong thing.
  This class produced four of the nine findings in the `#550` review and all three of the
  additional defects that review missed, so it gets explicit attention rather than being left to
  emerge.

### Sweep D — assumptions, across both layers

- **Adult / read-speech assumptions.** Behaviour correct only for adult connected speech: VAD
  tuned on adult voices, embeddings whose speaker separation was validated on adults, thresholds
  fitted on read passages applied to spontaneous or sustained phonation. *Trap:* the assumption
  usually lives in a constant or a model choice, not in prose — `PROFILE_WINDOW_S = 2.0` embeds a
  claim about how long a stable voiced segment is. Reason about what a parameter presumes; do not
  grep for "adult".
- **Lifespan gaps.** Per stage: what age range was it validated on, what does it do outside that
  range, does it say so. *Trap:* "unvalidated" is not itself a finding, or this sweep returns 81
  shrugs. The finding is where an unvalidated output is *used as if* validated — a speaker count
  fed into a decision without carrying its own uncertainty, or a training population never
  surfaced to a consumer.

## Verification

Two gates. A candidate is not a finding until it clears both, or is explicitly recorded as
having cleared only the first.

**Gate 1 — refutation.** Each candidate goes to an independent agent instructed to *refute* it,
defaulting to refuted under uncertainty. This is the gate the `#550` review lacked: its findings
were real, but three carried justifications that did not survive checking — a claim about a
senselab version that never existed, a claim that `chmod` modifies directories the user does not
own, and a claim that a ref pointer was never written when it was written by a different path.
Wrong reasons get copied into the next fix, so a real defect with an invented mechanism is not
good enough.

**Gate 2 — reproduction.** Survivors get an execution attempt against the real code.

Outcome sorts the tier:

| Tier | Meaning |
| --- | --- |
| **Demonstrated** | Survived refutation *and* reproduced by executing code. The register proper. |
| **Verified-latent** | Survived refutation; reproduction needs data or hardware not available (a child's voice, a cold cache, a multi-node run). **Must carry the exact experiment that would settle it**, making it a measurement task rather than a permanent maybe. |
| **Unverified concern** | Neither. A separate section, so nothing is mistaken for a finding. |

Requiring reproduction for *every* finding would silently discard the latent class — which is
exactly the class the lifespan work needs. Hence the second tier, stated explicitly rather than
footnoted.

## Register format

One row per finding, following `l1-post-processing-register.md`'s precedent:

| Field | Content |
| --- | --- |
| `id` | Stable identifier |
| `layer` | prose / computation / orchestration / assumption |
| `location` | `file:line` |
| `defect` | One sentence |
| `failure` | Concrete inputs → wrong output |
| `tier` | demonstrated / verified-latent / unverified |
| `severity` | Judged against consequence, not effort |
| `graph_implication` | Does the triage workflow consume this, route around it, or is it irrelevant? |

`graph_implication` is the field that makes the register usable for the decision it exists to
serve. A defect in a signal the graph never reads is a different priority from one in a signal
the graph's review flag depends on.

## Deliverable

`specs/20260815-215106-analyze-audio-audit/`:

- `register.md` — the findings, by tier and severity.
- `summary.md` — the layer measurements, the patterns across findings, and what they imply for
  the graph design and for the four deferred concerns.
- `prose-migration.md` — rationale worth keeping, relocated out of the code.

## Out of scope

- **No code changes.** No fixes, no deletions, no file moves. Every promotion candidate is
  recorded with its target layer and left in place.
- **No prose deletion.** Sweep A identifies and stages; it does not edit source files.
- **The graph / triage workflow itself.** Its own spec, informed by this register.
- **Model pluggability, speaker-count refinement, new extraction models, lifespan validation.**
  Each gets a spec scoped by what this audit finds.

## Success criteria

- Every finding carries `file:line`, a failure scenario, and a tier.
- No finding reaches the register without surviving a refutation attempt.
- Every verified-latent finding names the experiment that would settle it.
- Each sweep reports what it found clean, so absence of findings is distinguishable from absence
  of looking.
- The register answers, for each finding, whether the graph depends on it.
