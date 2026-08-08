# Specification Quality Checklist: Per-Speaker Identity Uncertainty and Background Scene Characterization

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-07-28
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Clarify Session 2026-07-29 — re-validation

Four questions asked and answered; the spec grew from 3 stories / 22 FRs / 10 SCs to
**4 stories / 82 FRs / 33 SCs**. Re-checked against every item above: all still pass.

**One prior checklist claim was invalidated and is now corrected.** Iteration 1 recorded
that ambiguity #2 ("does AST/YAMNet do audio enhancement") had been resolved by covering
both readings. The user's clarification showed **both readings were wrong** — the question
was about *amplification* (automatic gain), not enhancement of any kind. A Terminology
section now separates speech enhancement / amplification / foreground suppression, and
User Story 2 was rewritten. Lesson recorded: covering two interpretations is not the same
as covering the right one, and "informed guess" is not a substitute for asking when the
term itself is in doubt.

**Research findings folded in** (4 agents; load-bearing claims independently spot-checked
against installed code and checkpoint configs):

- Both classifiers are amplitude-**sensitive**; neither self-normalizes. The expected
  asymmetry did not hold.
- Amplification changes **no SNR** — suppression depth is the binding constraint. A 30 dB
  suppression baseline measurably fails.
- Detection reframed from amplification to **per-band floor subtraction**, which is how
  established toolchains make one threshold work at all distances.
- The **3 / 6 / 10 dB** ladder is corroborated from three independent directions.

**Deliberate scope expansions accepted from the user:**

- Background mask with per-region uncertainty, task-metadata-driven (US4, FR-031–045).
- Full mutual influence between all signals with uncertainty-gated weighting
  (FR-011a–h) — flagged in Dependencies as the highest-risk element, with
  self-confirmation and oscillation named as failure modes and guards specified.
- Scope narrowed to lab-like close-microphone collection.

**Outstanding, deferred to planning (not spec defects):**

- Several supporting figures are provisional pending primary-source checks (the
  partial-loudness transition interval; current-generation microphone self-noise
  values). FR-022 requires the derivation to record which figures are verified.
- The floor-conditioned-on-target-activity approach (FR-021h) has **no published
  precedent** and needs empirical validation.
- Whether an alternative classifier with more headroom performs better is untested;
  the spec requires measuring rather than assuming, since headroom arithmetic already
  mispredicted the current pair's ordering.

## Validation Notes (iteration 1, pre-clarify)

**Iteration 1 findings and fixes applied before finalizing:**

1. *Model and library names in requirements* — the initial draft named specific
   models (SepFormer, ECAPA) inside functional requirements. Moved to Assumptions
   and Dependencies as references to "the existing speech enhancement capability"
   so the requirements stay implementation-neutral. AST and YAMNet appear only in
   the verbatim user input and in an Assumptions entry that interprets that input.

2. *"Check if AST/YAMNet does audio enhancement" was ambiguous* — resolvable two
   ways (does the classifier internally normalize/amplify, versus should the
   enhanced variant feed scene analysis). Rather than blocking on a clarification,
   both readings are in scope and each has its own requirement (FR-015 and
   FR-013/FR-014 respectively), documented in Assumptions. Cost of covering both
   is low; a wrong guess would have been costly.

3. *User Story 3 framed as a question, not a deliverable* — the user asked "could
   one carry out...", which does not by itself specify a shipped capability.
   Resolved without a blocking question by giving Story 3 an explicit
   negative-result exit (acceptance scenario 6, SC-007) so a documented "no"
   closes the story legitimately, and by making the capability opt-in (FR-022).

4. *Success criteria initially depended on matching a single human annotation* —
   removed. The standing project decision is not to tune thresholds against one
   ground truth, so SC-005 and SC-007 now require that an effect be *quantified
   and reported with a stated direction*, which is verifiable without a labeled
   benchmark. FR-011 keeps the thresholds configurable for when a benchmark
   exists.

5. *SC-002 originally read "analyst understands the disagreement"* — not
   verifiable. Rewritten as a bounded task ("from the final convergence artifacts
   alone, without opening intermediate per-bucket artifacts, in under one
   minute").

**Deliberate scoping decisions recorded rather than flagged:**

- Per-speaker records may *replace* the current single-scalar identity output;
  no backwards compatibility is required pre-alpha. This follows the project's
  stated position and avoids two names for one quantity.
- Dependency on the open multi-speaker diarization uncertainty pull request
  (#537) is called out in Dependencies. Planning must reconcile with it; this
  spec does not assume either ordering.

**Open risk for the planning phase (not a spec defect):**

- The motivating evidence for User Story 1 is one recording whose true speaker
  count is unverified. The spec is deliberately written so that no requirement
  depends on that recording being multi-speaker — it requires representing
  disagreement, not resolving it in a particular direction. Establishing ground
  truth for that clip would strengthen validation but is not a prerequisite.
