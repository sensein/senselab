# Feature Specification: Per-Speaker Identity Uncertainty and Background Scene Characterization

**Feature Branch**: `20260728-221507-per-speaker-identity-scene`
**Created**: 2026-07-28
**Status**: Draft
**Input**: User description: "we need to change the identity axis as a per speaker uncertainty in the final convergence so that we are both accounting for potential number of speakers and their presence, we need to also check if AST/yamnet does audio enhancement when it's quiet and things are happening in the background of the audio scene, and also if one could carry out near microphone speech suppression and then enhancement, for example could some model/algorithm pull out foreground speech and then enhance the audio for background characteristics?"

## Context

Two observations from validation runs motivate this work.

**Identity collapses to one number.** The identity axis currently reports a single
uncertainty value per time bucket, answering "was it the same speaker?" That
scalar cannot express *how many* people the analysis thinks are present. On a
validation recording containing what sounds like four people introducing
themselves in sequence, two diarization sources each reported one speaker for the
whole clip while the embedding-clustering source reported five distinct regions
aligned to the name boundaries. The axis correctly registered high uncertainty
(0.67), but a consumer reading that number cannot tell whether it means "we
disagree about who spoke" or "we disagree about whether this is one person or
four." The final convergence outputs a fused transcript, a fused diarization, and
a presence track — but no per-speaker uncertainty and no speaker-count statement.

**Neither scene classifier amplifies the signal itself — this has now been
measured.** Background sound-source categorization (speech / people / machine /
environment) is driven by two audio classifiers with very different window
lengths: one analyzes ~10.24 s windows, the other ~0.96 s windows with 50%
overlap. The open research question was whether either performs level
normalization — effectively *amplification* — as part of its own inference. A
code audit of the installed models plus a gain sweep answered it, and the answer
inverts the expected asymmetry:

- **Both classifiers are amplitude-sensitive.** Neither normalizes input level,
  per-example or otherwise. Nothing in the pipeline ahead of them normalizes
  either — decode, downmix, and resample are all level-preserving.
- **The long-window classifier is the *more* gain-brittle of the two**, despite
  being the one with an explicit normalization step. That step divides by *fixed
  dataset-level constants*, so it cannot cancel a per-recording level offset; a
  gain becomes a rigid shift of every input bin. Its reported label set changed at
  every tested gain.
- **The short-window classifier has a hard absolute level floor, but not for the
  reason it appears.** Below roughly −60 dBFS it reports silence regardless of
  content, and this is *source-independent* — tones, beeps, band-limited noise and
  speech all collapse at the same level. Measurement shows this is **not** the
  log-mel stabilizing offset flooring the spectrum: at the collapse point only a few
  percent of bins are floored. It is a **learned, absolute-level-keyed silence
  decision** that fires about 30 dB above where the arithmetic floor begins to bite.
  Because it is monotone and source-independent, it is also the most reliable
  low-level diagnostic either classifier exposes.
- **Label identity itself migrates with level**, in both classifiers, on unchanged
  audio. One classifier's score for a faint source can even *increase* as the source
  gets quieter. Scores are therefore not comparable across segments at different
  levels — which matters because the whole point of this work is comparing across
  segments.

So the mechanism that matters is not adaptive gain — there is none — but an
**absolute floor** below which a classifier reports nothing.

**Amplification is necessary but far from sufficient, and this is the central
design constraint.** Attenuating and re-amplifying is bit-exact in floating point,
so amplification does not *recover* anything; it prevents the floor from destroying
the signal in the first place. Crucially, **amplification changes no
signal-to-noise ratio whatsoever** — it moves signal and residual foreground
together. An oracle experiment makes this concrete: with 30 dB of foreground
suppression and the residual amplified to a healthy level, the result was
*identical* whether a faint background source was present or entirely absent — the
leaked foreground dominated either way. So:

- Amplification fixes the **absolute-floor** failure (silence reported where
  content exists).
- Amplification does nothing for the **buried-under-residual-foreground** failure.
  Suppression depth, not gain, is the binding constraint.
- Worse, amplification **converts a silence false negative into a confident false
  positive** when nothing real is present: an amplified noise floor yields
  plausible-looking environmental labels — waterfall, water, gurgling, static —
  that are statistically indistinguishable from genuine broadband noise and read as
  real findings.

This is why removing foreground speech first matters, and why how *deeply* it is
removed matters more than how much the residual is amplified.

## Clarifications

### Session 2026-07-28

- Q: Is User Story 2 about whether speech enhancement destroys background evidence? → A: No. It is a research question about whether the scene classifiers themselves perform **amplification** (automatic gain / level normalization) of the signal — not speech — as part of inference. "Enhancement" was the wrong word; the intended term is amplification. The long-window (~10 s) classifier is unlikely to do this; the short-window (~1 s) one could. This is also the reason to run classification on a foreground-speech-suppressed waveform: amplification there could surface additional background noise sources.
- Q: Does per-speaker identity uncertainty stay report-only, become an intervention trigger, or fully participate in the adaptive loop including revising fused diarization and transcript speaker labels? → A: Full participation (option C). All signals should be able to **iteratively influence each other toward convergence**, with **uncertainty gating use** — a signal's influence is conditioned on its own uncertainty, so an unreliable signal cannot corrupt reliable ones.
- Q: Which classifier does background work in mask regions — short-window only, both unchanged, or both with the long-window model run on excised mask segments? → A: Both, with the long-window classifier run on **excised** mask segments so its window sees only masked audio (option C). The short-window classifier continues on the regular grid.
- Q: (volunteered) Anything else to add? → A: Add estimation of a **background mask
  uncertainty** covering regions where no speech-relevant events happen, and let that
  mask drive further introspection of what the masked regions contain. Captured as
  User Story 4 (FR-031 to FR-040).
- Q: Does "speech-relevant" for the background mask mean free of foreground/target speech (distant speech stays in the mask and is reportable), free of any speech, or both masks emitted? → A: Free of **target** activity (option A), but the definition of target activity MUST take **task metadata** into account. Some dataset tasks *are* breathing, coughs, or speech, and in those tasks those events are the relevant target signal rather than background. Scope is narrowed to **lab-like collection with the microphone close to the source**; generalization to other recording situations is deferred.
- Q: How should the gain applied to the foreground-suppressed waveform be chosen — fixed dB, single normalization target, per-window adaptive, or a gain sweep? → A: None of those as posed. Different events occur at different distances, so a single gain is not practical. The gain policy must mimic human capability to recognize sound sources at different distances and locations, and must not hallucinate sources from background noise. Establish an overall level target, and treat any background signal below a relative dB threshold as background noise rather than a source event. The threshold is to be derived from human psychophysics of detection, balanced against measured machine capability — not picked arbitrarily.

## Terminology

These three operations are distinct and are never used interchangeably in this
specification:

- **Speech enhancement**: extracting/cleaning the speech content of a recording,
  suppressing everything else. Already exists in the system.
- **Amplification**: scaling signal level, whether by an explicit gain step or by a
  classifier's own internal level normalization. The subject of User Story 2.
- **Foreground suppression**: removing the dominant near-microphone speech so that
  the remaining background can be analyzed. The subject of User Story 3.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Per-speaker identity uncertainty in the final convergence (Priority: P1)

An analyst reviewing a completed run needs to know how many people the analysis
believes are in the recording, how confident it is in that count, and for each
hypothesized speaker, where that person was speaking and how sure the analysis is
about each of those spans. Today they get one identity number per half-second
bucket and must reverse-engineer the rest from intermediate artifacts.

**Why this priority**: This is the requested change and the only one of the three
that alters a published output contract. It is also the one that unblocks
interpretation of every high-identity-uncertainty region found so far — without
it, an analyst cannot distinguish a labeling disagreement from a speaker-count
disagreement, which are different problems with different fixes.

**Independent Test**: Run the analysis on a recording where diarization sources
disagree on speaker count. Confirm the final convergence artifacts state a
speaker-count distribution, list each hypothesized speaker with its own presence
timeline and uncertainty, and identify which source supported each count —
without opening any intermediate per-bucket artifact.

**Acceptance Scenarios**:

1. **Given** a recording where every diarization source reports exactly one
   speaker, **When** the analysis completes, **Then** the final convergence
   reports a speaker-count distribution concentrated on one speaker, emits
   exactly one speaker hypothesis, and no phantom speaker appears.
2. **Given** a recording where sources disagree on speaker count (one source says
   one speaker, another says five), **When** the analysis completes, **Then** the
   count distribution spans the disagreeing counts, and each count is
   attributable to the sources that supported it.
3. **Given** a hypothesized speaker, **When** an analyst reads the final
   convergence, **Then** they see that speaker's own presence timeline on the same
   time grid as the existing presence output, with an uncertainty value per span.
4. **Given** a hypothesized speaker that only one source proposed, **When** an
   analyst reads its record, **Then** the record states which sources proposed it
   and whether any of those sources is derived from another signal rather than
   being an independent observer.
5. **Given** two sources that use unrelated naming conventions for speakers,
   **When** their labels are fused, **Then** the correspondence between each
   source's labels and each fused speaker is recorded and auditable.
6. **Given** a completed run, **When** the same run is repeated with identical
   inputs and settings, **Then** the per-speaker records are identical.

---

### User Story 2 - Determine whether the scene classifiers amplify the signal themselves (Priority: P2)

A researcher needs to know, for each scene classifier, whether its own inference
pipeline normalizes signal level — that is, whether it amplifies a quiet input
before classifying it. The answer decides whether an explicit amplification step is
needed to surface faint background sources, or whether amplification is a no-op
because the classifier already does it internally.

**Why this priority**: The verdict is now established — both classifiers are
amplitude-sensitive — so this story's job shifts from discovery to *pinning*: turn
the finding into a checked-in regression guard, document each classifier's floor,
and record the gain range over which its behavior is trustworthy. That floor is an
input to User Story 3's detection margin, so Story 3 cannot be designed without it.
The audit also surfaced a score-comparability defect (below) that directly distorts
background source categories, which belongs here rather than being deferred.

**Independent Test**: Present each classifier with the same recording at several
known gains and compare its outputs. A classifier whose labels and scores are
unchanged across gains is amplitude-invariant; one whose outputs shift with gain is
amplitude-sensitive. This is verifiable without any change to the analysis
pipeline.

**Acceptance Scenarios**:

1. **Given** one recording rendered at several known gains spanning at least
   30 dB, **When** each scene classifier is run on every version, **Then** the
   degree to which its labels and scores change with gain is reported, yielding an
   amplitude-invariance verdict per classifier.
2. **Given** a classifier reported as amplitude-invariant, **When** its verdict is
   recorded, **Then** the evidence states the gain range over which invariance
   held, since invariance may break down at extreme levels.
3. **Given** the two classifiers differ in window length by roughly an order of
   magnitude, **When** the verdicts are reported, **Then** each verdict is
   attributed to its classifier and window length, rather than being reported as a
   single property of "scene analysis".
4. **Given** a quiet recording with audible background activity, **When** it is
   presented to each classifier both at its original level and amplified, **Then**
   the report states whether amplification changes which background source
   categories are reported.
5. **Given** the classifiers' input preparation, **When** it is audited, **Then**
   any level normalization it applies is documented, so the empirical verdict from
   scenario 1 can be corroborated against the actual mechanism.
6. **Given** the verdicts, **When** User Story 3 is designed, **Then** whether an
   explicit amplification step is required follows from the verdicts rather than
   from assumption.
7. **Given** a completed run, **When** an analyst inspects any scene-analysis
   result, **Then** the record states which audio variant it was computed from and
   at what gain.
8. **Given** each classifier's documented low-level floor, **When** the detection
   margin of User Story 3 is set, **Then** it accounts for the most restrictive
   floor, so a source judged detectable is one the classifiers can actually see.
9. **Given** a recording in which a dominant source and a quieter secondary source
   are both present, **When** background category masses are computed, **Then** the
   secondary source's mass is not suppressed as an artifact of one classifier's
   scores being a mutually-exclusive competition across all classes.
10. **Given** an amplified variant whose peak exceeds full scale, **When** it is
   passed to a classifier, **Then** clipping or requantization is detected and
   reported rather than silently altering the categorization.
11. **Given** the established amplitude-sensitivity verdicts, **When** a model or
   dependency is upgraded, **Then** an automated check fails if level handling
   changed.

---

### User Story 3 - Characterize the background with foreground speech suppressed (Priority: P3)

A researcher analyzing a recording dominated by a close-microphone speaker wants to
know what else is in the room. They want the foreground speech pulled out so the
remaining background can be characterized — and, where the background is faint,
amplified so that its sources rise above the classifier's floor and become
identifiable. Removing the loud foreground first is what makes amplification useful:
it lets the gain act on the background instead of on the speech.

**Why this priority**: The highest-value capability of the three but also the least
certain. It is framed as a feasibility question, it depends on Story 2's baseline
to demonstrate benefit, and it may conclude with a documented negative result. It
must not delay Stories 1 and 2.

**Independent Test**: On a recording with a dominant near-microphone speaker over
audible background activity, produce a foreground-suppressed variant, run
background categorization on it, and compare the recovered categories against
those from the unenhanced and enhanced variants.

**Acceptance Scenarios**:

1. **Given** a recording with a dominant foreground speaker, **When** the
   foreground-suppressed variant is produced, **Then** background source
   categories are reported for it on the same time grid as the existing
   categorization.
1a. **Given** a recording containing only noise floor after foreground suppression,
   **When** the variant is amplified and categorized, **Then** no background source
   category is reported — the noise-floor gate rejects them.
1b. **Given** a recording with two sources of the same type at materially
   different distances, **When** categorization runs, **Then** both are reported,
   each with the margin by which it exceeded the noise floor.
1c. **Given** any reported background source, **When** an analyst reads it,
   **Then** it carries its above-noise-floor margin, so a clearly audible source is
   distinguishable from one at the edge of detectability.
2. **Given** the foreground-suppressed variant, **When** it is evaluated, **Then**
   a measure of residual foreground-speech leakage is reported, so a consumer can
   tell whether a "people" or "speech" category reflects background content or
   leaked foreground.
3. **Given** a recording that audibly contains a background source, **When**
   categories from the foreground-suppressed variant are compared to those from
   the enhanced variant, **Then** the comparison shows which sources each variant
   recovers.
4. **Given** foreground suppression is unavailable or fails on a recording,
   **When** the analysis runs, **Then** background categorization continues on the
   standard variant and the fallback is recorded rather than failing the run.
5. **Given** the foreground-suppressed variant exists, **When** speech tasks run,
   **Then** their input audio is unchanged — transcription and diarization never
   consume the suppressed variant.
6. **Given** the feasibility work concludes the approach does not recover
   background sources better than the existing variants, **When** the work is
   closed, **Then** the negative result is documented with the measurements that
   support it, and no capability is shipped.

---

### User Story 4 - Background mask with its own uncertainty, driving introspection (Priority: P2)

A researcher wants to know **where in the recording nothing speech-relevant is
happening**, with a stated confidence, and then to look inside those regions to find
out what they actually contain. Regions that are confidently free of speech are the
places where background characterization can be trusted without relying on
suppression at all — because there is no foreground there to leak. Regions where the
mask itself is uncertain are precisely where background claims must be discounted.

**Why this priority**: This makes the rest of the background work interpretable and
it is cheap, deriving largely from signals the pipeline already produces. It also
inverts the hard problem: rather than fighting to suppress a dominant talker deeply
enough to hear past them, it identifies the intervals where that fight is
unnecessary. Given that a 30 dB suppression baseline was measured to fail, the
target-free intervals may be where most trustworthy background evidence actually
comes from. It gates the confidence of User Story 3 and should land before it.

**Independent Test**: Run the analysis on a recording with interleaved speech and
target-free intervals. Confirm a background mask is emitted with a per-region
confidence, that confidently-target-free regions are distinguishable from
uncertain ones, and that the contents of the confident regions are characterized and
reportable on their own.

**Acceptance Scenarios**:

1. **Given** a recording with speech and non-speech intervals, **When** the analysis
   completes, **Then** a background mask is emitted marking regions where no
   speech-relevant event occurs, each carrying its own uncertainty rather than being
   a hard binary.
2. **Given** a region where the analysis is confident no speech occurred, **When**
   background characterization runs there, **Then** its findings are reported with
   higher confidence than findings from regions under or adjacent to speech, because
   no foreground leakage is possible.
3. **Given** a region where the mask is uncertain — the analysis cannot tell whether
   speech occurred — **When** background sources are reported there, **Then** they
   are discounted accordingly, and the mask uncertainty is given as the reason.
4. **Given** a confidently target-free region, **When** an analyst requests
   introspection, **Then** the system reports what that region contains — its source
   categories, their above-floor margins, and whether it is merely noise floor.
5. **Given** the interval immediately following loud speech, **When** the mask is
   computed, **Then** that interval is not treated as clean background: reverberant
   tail and the classifier's own temporal context contaminate it even when no speech
   is detected there.
6. **Given** a recording of continuous target activity with no target-free interval,
   **When** the mask is computed, **Then** it is reported as empty and the limitation
   is stated, rather than a degenerate mask being emitted.
7. **Given** a recording that is entirely target-free, **When** the mask is computed,
   **Then** it covers the whole recording and background characterization proceeds
   without any suppression step.
8. **Given** the mask, **When** it is used to select regions for background
   characterization, **Then** the total masked duration is reported, so a consumer
   knows how much of the recording the background findings actually rest on.

---

### Edge Cases

**Per-speaker identity**

- A recording with no speech at all: no speaker hypotheses, count distribution
  concentrated on zero, and no per-speaker presence tracks.
- A speaker who speaks for less than one analysis window: whether they can be
  hypothesized at all, and how the count distribution reflects that limit.
- Overlapping speech, where two speakers occupy the same time span: per-speaker
  presence must be able to report both as present simultaneously.
- Sources that disagree on count by a wide margin (one versus five): the count
  distribution must represent multi-modal disagreement rather than collapsing to a
  majority or a mean.
- A speaker who appears in only one of the two passes: attribution must record
  which pass supported the hypothesis.
- A derived source that produces speaker labels as a by-product of another
  computation rather than as an independent judgment: consumers must be able to
  tell it apart from an independent observer.
- All sources agree on the count but disagree on the boundaries: count uncertainty
  must be low while per-speaker presence uncertainty stays high.

**Mutual influence**

- Two interpretations that each imply the other is wrong, so the loop alternates
  between them across rounds without settling.
- A revision that lowers a signal's uncertainty purely because the signal was
  overwritten, with no new evidence — the self-confirmation case.
- A derived signal and its parent both voting, so their agreement is counted twice and
  a single underlying computation appears to be corroborated.
- A chain of influences long enough that a final speaker label rests on a signal that
  was itself revised twice, with the original evidence no longer visible.
- A revision that improves one axis while degrading another, where a per-axis view
  shows progress and the whole shows none.
- A quantity that never converges but sits adjacent to ones that did, risking its
  presentation as equally settled.

**Scene and background**

- A recording so quiet that the classifier's own input preparation amplifies the
  noise floor into apparent events.
- A recording where enhancement removes all non-speech content, collapsing every
  background category to speech.
- A recording where enhancement introduces artifacts that are then categorized as
  genuine background sources.
- A foreground-suppressed variant that is near-silent because the recording was
  almost entirely foreground speech.
- A foreground-suppressed variant containing leaked foreground speech that would be
  misattributed to background human sounds.
- Background content that is itself speech from a distant talker, which foreground
  suppression should preserve rather than remove.
- Two sources of the same type at very different distances in one recording: the
  near one must not cause the gain policy to bury the far one below the detection
  margin.
- A recording whose noise floor is itself a real source — a steady machine hum or
  ventilation — where noise-floor estimation would absorb the very thing that should
  be reported. The estimator must not silently discard stationary sources.
- A source that sits exactly at the detection margin, where a marginal gain change
  flips it between reported and unreported: the reported margin must make this
  fragility visible rather than presenting the outcome as certain.
- A recording with no discernible noise floor because it was digitally gated or
  heavily compressed upstream, leaving the relative margin undefined.

**Background mask**

- A recording that is continuous speech with no gaps: the mask is empty, and every
  background finding must then depend on suppression quality alone.
- Speech-free gaps shorter than the classifier's analysis window: no window fits
  entirely inside the mask, so no uncontaminated background decision is possible even
  though the mask is non-empty.
- A long reverberant tail after loud speech that voice-activity detection does not
  flag as speech, which would be admitted to the mask and characterized as room
  ambience.
- A distant background talker inside a mask region: target-free but not speech-free.
  It stays in the mask and is reported as a background source (FR-033c).
- Breath and mouth noise from the target participant in an otherwise silent gap —
  target activity, not speech, and acoustically similar to background human sounds.
- A **breathing or cough task**, where the target event is precisely what speech
  detection does not detect. Building the mask from speech activity alone would admit
  the target breaths into the mask and report them as background human sounds — the
  target signal misattributed as an environmental finding.
- A speech task in which the participant coughs: target activity by FR-033, but a
  detector tuned only for speech would admit the cough to the mask.
- Task metadata that is missing, wrong, or describes a task type the system does not
  recognize, where the conservative fallback shrinks the mask rather than risking
  misattribution.
- A mask covering a large fraction of the recording but consisting of many tiny
  fragments, where the total duration looks reassuring but no single region is long
  enough to support a finding.
- Mask regions all shorter than the long-window classifier's analysis window, so every
  excised segment is mostly padding and only the short-window classifier can contribute.
- An excised segment whose boundaries fall mid-event, splitting one background source
  across two segments and halving its apparent duration in each.
- Concatenating adjacent mask segments to reach a usable window length, which would
  create a discontinuity at the join that an onset-sensitive feature reads as an event.

## Requirements *(mandatory)*

### Functional Requirements

**Per-speaker identity (User Story 1)**

- **FR-001**: The final convergence MUST report identity uncertainty per
  hypothesized speaker, rather than only as a single value per time bucket.
- **FR-002**: The system MUST report a speaker-count distribution expressing how
  many speakers the analysis believes are present, capable of representing
  disagreement across more than one plausible count.
- **FR-003**: The system MUST report, for each hypothesized speaker, a presence
  timeline on the same time grid as the existing presence output, with an
  uncertainty value per span.
- **FR-004**: The system MUST separately express uncertainty that a hypothesized
  speaker exists at all and uncertainty about where an existing speaker was
  speaking, so a consumer can tell the two apart.
- **FR-005**: The system MUST record the correspondence between each contributing
  source's speaker labels and each fused speaker hypothesis, so that fusion across
  sources with unrelated naming conventions is auditable.
- **FR-006**: The system MUST attribute each speaker hypothesis and each count in
  the count distribution to the sources that supported it.
- **FR-007**: The system MUST distinguish sources that observe speaker identity
  independently from sources whose speaker labels are derived from another signal,
  so consumers can weight them differently.
- **FR-008**: When contributing sources disagree on speaker count, the system MUST
  surface the disagreement rather than resolving it silently.
- **FR-009**: When all contributing sources agree on a single speaker, the system
  MUST NOT emit additional speaker hypotheses.
- **FR-010**: Per-speaker records MUST be reproducible: identical inputs and
  settings produce identical records.
- **FR-011**: Thresholds and weights governing speaker-count and per-speaker
  uncertainty MUST be configurable rather than fixed in code, so they can be
  re-tuned when a labeled multi-speaker benchmark becomes available.

**Mutual influence and convergence (spans all stories)**

- **FR-011a**: Signals MUST be able to influence one another **iteratively toward
  convergence**, rather than flowing one way. Identity evidence may revise diarization;
  revised diarization may revise per-speaker presence; the background mask may revise
  presence; utterance consensus may revise speaker attribution. Per-speaker identity
  participates fully, including revising fused diarization and transcript speaker
  labels.
- **FR-011b**: A signal's influence on others MUST be **gated by its own uncertainty**.
  A high-uncertainty signal MUST have correspondingly reduced influence, so an
  unreliable signal cannot propagate its error into reliable ones.
- **FR-011c**: Influence gating MUST additionally account for whether a signal is an
  independent observer or **derived** from another signal already in the system
  (FR-007). A derived signal MUST NOT receive the influence weight of an independent
  one, because its agreement with its parent is not corroboration.
- **FR-011d**: The system MUST distinguish uncertainty **resolved by new evidence**
  from uncertainty **resolved by revision**. When a signal is revised and its
  uncertainty subsequently falls, the fall MUST NOT be reported as improved confidence
  unless independent evidence supports it. Without this, the loop confirms its own
  edits — the single largest correctness risk in a mutually-influencing design.
- **FR-011e**: The system MUST detect **non-convergence** — oscillation between states,
  or uncertainty that fails to decrease across rounds — and MUST terminate with the
  condition reported rather than iterating indefinitely or silently emitting whichever
  state the last round happened to produce.
- **FR-011f**: Iteration MUST be **deterministic**: a fixed evaluation order and fixed
  tie-breaking, so that mutual influence does not compromise the byte-reproducibility
  the convergence outputs already provide (SC-004).
- **FR-011g**: Every revision MUST be attributable — which signal caused it, in which
  round, and on what evidence — so a final speaker label can be traced back through the
  chain of influences that produced it.
- **FR-011h**: The system MUST record, for each converged quantity, whether it
  converged, was revised and by what, or remains unresolved. A value that never
  converged MUST NOT be presented as settled.

**Amplitude-invariance research question (User Story 2)**

- **FR-012**: Every scene-analysis result MUST record which audio variant it was
  computed from and the gain applied to that variant.
- **FR-013**: The system MUST be able to present a recording to each scene
  classifier at several known gains and report how much each classifier's labels
  and scores change as a function of gain.
- **FR-014**: The system MUST report an amplitude-invariance verdict per scene
  classifier — self-normalizing or level-sensitive — together with the gain range
  over which the verdict holds.
- **FR-015**: Each verdict MUST be attributed to its own classifier and that
  classifier's analysis window length; verdicts MUST NOT be generalized across
  classifiers.
- **FR-016**: Any level normalization the classifiers apply to their input MUST be
  documented, so the empirical verdict can be corroborated against the mechanism.
- **FR-017**: The system MUST report whether amplifying a quiet recording changes
  which background source categories are reported.
- **FR-017a**: Each classifier's effective low-level floor — the level beneath
  which it reports nothing regardless of content — MUST be documented, since the
  detection margin in FR-021/FR-022 depends on it.
- **FR-017b**: The amplitude-sensitivity verdicts MUST be covered by an automated
  regression guard, so that a model or dependency upgrade that changes level
  handling is detected rather than silently altering background categorization.
- **FR-017c**: Background source category masses MUST be computed from
  per-classifier scores that are comparable across classifiers. Scores that
  represent mutually exclusive competition MUST NOT be summed into category
  masses alongside scores that represent independent per-class presence, because
  doing so structurally suppresses secondary background sources whenever a
  dominant source is present.
- **FR-017d**: The audio path feeding each classifier MUST preserve enough
  headroom and resolution for the amplification in FR-019 to be applied without
  clipping or requantization artifacts, and any such artifact MUST be detected and
  reported rather than silently degrading the input.

**Background characterization (User Story 3)**

- **FR-018**: The system MUST be able to produce an audio variant in which
  near-microphone foreground speech is suppressed.
- **FR-018a**: The system MUST measure and report the achieved **foreground
  suppression depth**, and MUST NOT claim a background source is detectable when the
  residual foreground still exceeds it. Suppression depth — not gain — is the
  binding constraint: amplification moves the background and the residual foreground
  together and changes no signal-to-noise ratio.
- **FR-018b**: Detection MUST be based on **per-band excess over a locally estimated
  noise floor**, not on amplification. The floor MUST be estimated per frequency band
  and subtracted, after which a single threshold applies uniformly regardless of
  source distance. This is what makes the distance problem tractable: the reference
  becomes local rather than global, so a near source and a far source are each judged
  against their own band floor.
- **FR-019**: Amplification MUST be scoped to **conditioning the classifier's input**
  and MUST NOT be the detection mechanism. The applied gain MUST be recorded, MUST
  never clip, and MUST NOT exceed approximately **10 dB**, beyond which measured
  classifier behavior reflects clipping distortion rather than content.
- **FR-019a**: Gain MUST be applied **per analysis segment**, not as one global gain
  for the recording. A single global gain large enough to lift a faint passage
  clips the loud passages, and the long-window classifier cannot isolate itself from
  that contamination.
- **FR-019b**: Amplification MUST be applied before any lossy serialization or
  requantization in the classifier's input path, so that faint content is not
  destroyed before the gain reaches it, and so that a quantization noise floor is
  not amplified in its place.
- **FR-019c**: The overall level target MUST leave enough headroom for the gain of
  FR-019 to be applied without clipping. A broadcast-style target of about
  **−23 LUFS** provides this; a streaming-style target of about −14 LUFS does not.
  The same scalar MUST be applied to the unmodified and suppressed variants, so that
  cross-variant comparisons are not corrupted by independent renormalization.
- **FR-020**: The detection policy MUST surface sound sources occurring at materially
  different distances within a single recording. This follows from FR-018b's local
  reference rather than from any distance-adaptive gain.
- **FR-020a**: The system MUST reject a segment whose **pre-gain** level falls below
  the level at which the classifiers were measured to be trustworthy
  (approximately −45 dBFS RMS in-window), rather than amplifying it and reporting
  whatever labels emerge.
- **FR-020b**: The system MUST apply a noise-character test — such as spectral
  flatness — to each amplified segment and MUST suppress reported source categories
  when the segment is characteristically broadband noise rather than a structured
  source. Measured separation between noise floors and structured sources on this
  statistic is large, making it a cheap and high-value guard.
- **FR-020c**: Labels that an amplified noise floor is known to produce — broadband
  and water-like environmental categories, hum, hiss, static, and silence — MUST NOT
  be reported from an amplified segment unless the noise-character test of FR-020b
  passes. These are the categories most likely to be mistaken for genuine findings.
- **FR-020d**: The system MUST detect each classifier's characteristic
  floor-response signature and discard those windows. A classifier that emits a
  fixed label pattern on digital silence MUST NOT have that pattern reported as
  content, and detection MUST NOT rely on its silence score alone, which can stay
  low while another category in the same signature clears a normal threshold.
- **FR-020e**: Classifier scores MUST NOT be compared or ranked across segments at
  different levels, because score varies with level on unchanged audio and in at
  least one classifier does so non-monotonically.
- **FR-021**: The system MUST estimate the recording's background noise floor
  **per frequency band** and MUST report a candidate background source only when
  its level exceeds that floor by at least a defined relative-dB margin in its
  dominant band. Signal below the margin MUST be reported as noise floor, not as a
  source event. The derived starting margins are:

  | Band-relative SNR | Treatment |
  |---|---|
  | below +3 dB | reject — indistinguishable from noise floor |
  | +3 dB | candidate — tentative only, never a confident source |
  | +6 dB | probable |
  | +10 dB and above | confident background source |

  This 3 / 6 / 10 dB ladder is corroborated from three independent directions, which
  is the strongest evidence available that it is not arbitrary:

  - **Human psychophysics** — masked-threshold and audibility criteria converge on
    the same interval, with roughly +3 dB as the minimum measurability limit and
    +10 dB as confident identification.
  - **Established detection practice** — a dozen independent traditions in
    bioacoustics, ecoacoustics, and noise standards, developed without reference to
    one another, all place their operating points at 3, 6, or 8–10 dB above a
    per-band floor.
  - **Measured classifier capability** — the short-window classifier's reliable
    detection floor against an interferer is roughly 5–10 dB, the long-window
    classifier's roughly 15–20 dB, with noise-family labels contaminating results
    from about 20 dB downward.

  The +10 dB confident margin is therefore defensible from the human and machine
  sides simultaneously, which is the balance this threshold was required to strike.
- **FR-021a**: The noise floor MUST be estimated from the same recording being
  analyzed. A global or cross-recording constant MUST NOT be used as the floor,
  because each recording carries a different unknown level offset.
- **FR-021d**: The noise-floor estimator MUST correct the known downward bias of
  percentile-based estimation. A low-percentile estimate of band power sits a
  calculable amount below the true mean noise power — roughly 10 dB for a tenth
  percentile — and leaving it uncorrected makes every relative-dB gate silently that
  much more permissive.
- **FR-021e**: The gate MUST be evaluated on energy aggregated over an analysis patch,
  never on individual time-frequency bins. Single-bin noise power is far too variable
  for a few-dB threshold to be meaningful, whereas the same threshold is highly
  significant once aggregated over a patch.
- **FR-021f**: The estimator MUST exclude candidate events before re-estimating the
  floor, iterating until stable, so that a sustained source is not absorbed into the
  floor that is supposed to reveal it.
- **FR-021g**: The floor estimate MUST be frozen inside a detected event, so an event
  cannot raise the floor it is being measured against.
- **FR-021h**: The floor MUST be estimated **conditioned on foreground-speaker
  activity**, using the diarization already available. The suppression residual's
  artifact level is correlated with the removed talker's level, so a single
  unconditioned floor over-gates quiet stretches and under-gates busy ones.
- **FR-021i**: The system MUST retain an unsubtracted parallel analysis for stationary
  sources. Per-band floor subtraction deletes continuous narrowband content by
  construction, so a steady hum, drone, or engine — a legitimate machine-category
  source — would otherwise be erased by the very step that reveals transient sources.
- **FR-021j**: Reported detections MUST additionally satisfy a minimum occupancy
  within the analysis patch and a minimum contiguous duration, and MUST use
  hysteresis — triggering at a higher tier and extending boundaries at a lower one —
  since a level threshold alone conflates whether a source is present with where it
  begins and ends.
- **FR-021k**: The presence decision and the temporal-extent decision MUST be
  separable. A single threshold cannot simultaneously get both right, so the margin
  gate MUST determine presence while boundaries are determined independently.
- **FR-021b**: The system MUST estimate the recording chain's own noise floor and
  MUST NOT claim a background source is detectable when its level is within a few dB
  of that floor — in which case the binding limit is the microphone, not human
  hearing, and the report MUST say so rather than implying a perceptual
  justification it cannot support.
- **FR-021c**: The system MUST NOT emit absolute perceptual quantities — loudness in
  sones or phons, or absolute audibility claims — for recordings that carry no
  documented calibration offset. Levels MUST be reported as band-relative or
  difference quantities, or against digital full scale.
- **FR-022**: The relative-dB margin in FR-021 MUST be derived from human
  detection psychophysics balanced against the measured capability of the
  classifiers — including each classifier's documented low-level floor from
  FR-017a — and its derivation MUST be documented alongside the value. It MUST NOT
  be an arbitrary constant. The derivation MUST record which supporting figures are
  verified against primary sources and which are provisional, so the margin's
  evidential basis is auditable.
- **FR-022a**: A candidate source MUST clear **both** the perceptual margin of
  FR-021 and the classifier floor of FR-017a. These are independent limits — one
  perceptual, one machine — and the binding one MUST be identified in the report,
  since which one binds differs by recording.
- **FR-023**: The margin and level target MUST be configurable, so they can be
  re-tuned as evidence accumulates, following the same versioned-profile approach
  the existing calibration uses.
- **FR-024**: Background source categorization MUST be able to consume the
  foreground-suppressed variant and report categories on the existing time grid.
- **FR-025**: The system MUST report which background source categories are
  recovered from the foreground-suppressed variant that are not recovered from the
  unmodified recording, so the benefit of the operation is visible.
- **FR-026**: The system MUST report a measure of residual foreground-speech
  leakage in the foreground-suppressed variant.
- **FR-027**: Every reported background source MUST carry the margin by which it
  exceeded the noise floor, so a consumer can distinguish a clearly audible source
  from one at the edge of detectability.
- **FR-028**: The foreground-suppressed variant MUST NOT be used as input to speech
  tasks such as transcription, alignment, or diarization.
- **FR-029**: When foreground suppression is unavailable or fails, background
  categorization MUST continue on the standard variant, recording the fallback,
  without failing the run.
- **FR-030**: Producing the foreground-suppressed variant MUST be opt-in, so that
  default runs do not pay its cost.

**Background mask (User Story 4)**

- **FR-031**: The system MUST emit a **background mask** identifying regions where no
  **target activity** occurs, on the same time grid as the existing presence output.
  Target activity means activity from the near-microphone participant being recorded,
  not speech specifically.
- **FR-032**: The mask MUST carry a **per-region uncertainty** rather than being a
  hard binary, so that "confidently target-free", "confidently target-active", and
  "cannot tell" are distinguishable states.
- **FR-033**: What counts as target activity MUST be determined by **task metadata**.
  In a breathing task, breaths are target activity; in a cough task, coughs are; in a
  speech task, speech is, and the participant's breaths and mouth noise are also
  target activity rather than background.
- **FR-033a**: The system MUST NOT rely on speech detection alone to build the mask.
  For a task whose target is a non-speech vocal event, speech detection reports no
  activity during the target event, which would admit the target signal into the
  background mask and then report it as a background human-sound source —
  misattributing the very signal being collected. Detection of target activity MUST
  match the task's target event type.
- **FR-033b**: When task metadata is absent or unrecognized, the system MUST fall back
  to the most conservative interpretation — treating any participant-attributable
  vocal activity as target activity — and MUST record that the fallback was used, so a
  mask built without task context is never mistaken for one built with it.
- **FR-033c**: Speech from a source other than the target participant — a distant or
  background talker — MUST remain in the mask and MUST be reportable as a background
  source. Such speech is a finding, not contamination.
- **FR-034**: The mask MUST apply a guard interval around detected speech. The
  interval immediately following speech is contaminated by reverberant tail, and a
  classifier window overlapping speech carries it into its decision, so neither is
  clean background even when no speech is detected in the region itself.
- **FR-035**: Background source findings drawn from confidently-target-free regions
  MUST be reported at higher confidence than findings from regions under or adjacent
  to speech, because foreground leakage is impossible in the former.
- **FR-036**: Where the mask is uncertain, background findings MUST be discounted and
  the mask uncertainty MUST be given as the stated reason, so a weak finding is
  distinguishable from a finding weakened by not knowing whether speech was present.
- **FR-037**: The system MUST support **introspection of a masked region** — reporting
  what that region contains, including its source categories, their above-floor
  margins, and whether it is merely noise floor.
- **FR-038**: The system MUST report the **total masked duration** and its fraction of
  the recording, so a consumer knows how much evidence the background findings rest
  on. A mask covering a negligible fraction MUST be flagged as such.
- **FR-039**: In confidently-target-free regions the system MUST NOT require the
  foreground-suppressed variant, since there is no foreground to remove there. The
  suppression path and its depth constraint apply only where speech and background
  co-occur.
- **FR-040**: An empty mask — a recording of continuous target activity — MUST be reported
  as a stated limitation rather than yielding a degenerate mask or a silent absence of
  background findings.
- **FR-041**: Background categorization over mask regions MUST run the **long-window
  classifier on excised mask segments** rather than on the full timeline, so its
  analysis window contains only masked audio. Running it on the full timeline couples
  masked regions to adjacent target activity and dilutes the result, which measurement
  confirmed.
- **FR-042**: The **short-window classifier MUST continue to run on the regular grid**
  and MUST be retained as the level tripwire regardless of routing, because its silence
  score is the only monotone, source-independent level diagnostic either classifier
  exposes.
- **FR-043**: Excision MUST record each segment's boundaries and the fraction of the
  classifier's window occupied by padding. When a mask region is materially shorter than
  that window, the padding fraction MUST be reported and the result discounted:
  padding-to-signal contrast is itself gain-dependent, and heavily padded short windows
  were measured to be the least stable case.
- **FR-044**: Results computed on excised segments MUST be attributable to the mask
  region they came from and MUST NOT be silently merged with grid-computed results, since
  the two are computed over different audio extents and are not interchangeable.
- **FR-045**: A mask region too short to support an uncontaminated long-window decision
  MUST be identified as such rather than yielding a heavily padded result presented as
  equivalent to a full-window one.

### Key Entities

- **Speaker hypothesis**: A person the analysis believes is present in the
  recording. Carries an identifier stable within the run, the sources that proposed
  it, its existence uncertainty, and a reference to its presence track.
- **Speaker-count distribution**: The analysis's belief over how many speakers are
  present, with the supporting sources for each candidate count.
- **Per-speaker presence track**: For one speaker hypothesis, the time-aligned
  belief that this specific person was speaking, with uncertainty per span.
- **Source-label correspondence**: The mapping from each contributing source's own
  speaker labels to fused speaker hypotheses, plus whether that source is an
  independent observer or derived from another signal.
- **Audio variant**: A named version of the recording that analysis stages consume
  — unmodified, speech-enhanced, or foreground-suppressed — together with the gain
  applied to it, recorded alongside every result computed from it.
- **Amplitude-invariance verdict**: For one scene classifier, whether its inference
  normalizes signal level, the gain range over which that holds, and the
  measurements supporting it. Carries the classifier's analysis window length,
  since window length is the suspected mechanism.
- **Scene variant comparison**: Per time bucket, how background source categories
  differ across audio variants and gains.
- **Background mask**: The set of regions where no speech-relevant event occurs, each
  carrying its own uncertainty so that "confidently target-free" is distinguishable
  from "cannot tell". Carries its total duration and its fraction of the recording,
  and excludes guard intervals adjacent to speech.
- **Masked-region introspection**: For one confidently-target-free region, what it
  contains — source categories with their above-floor margins, or a statement that it
  is only noise floor.
- **Noise floor estimate**: The recording's background noise level, against which
  candidate sources are compared. Distinct from a source: the estimator must not
  absorb a steady real source such as machine hum.
- **Detection margin profile**: The versioned, configurable policy governing the
  overall level target and the relative-dB margin a candidate must clear to be
  reported as a source, together with the written derivation of those values from
  human psychophysics and measured classifier capability.
- **Background characterization result**: Source categories derived from the
  foreground-suppressed variant, each carrying the margin by which it exceeded the
  noise floor, together with the variant's leakage measure and the gain applied.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a recording where all diarization sources agree on one speaker,
  the reported count distribution places at least 90% of its mass on one speaker
  and exactly one speaker hypothesis is emitted.
- **SC-002**: On a recording where sources disagree on speaker count, an analyst
  can determine from the final convergence artifacts alone — without opening
  intermediate per-bucket artifacts — how many speakers each source claimed and
  which sources supported each candidate count, in under one minute. *Verified
  manually — this is a usability criterion, not an automatable CI gate.*
- **SC-003**: A per-speaker presence track is emitted for 100% of hypothesized
  speakers, covering the full recording duration on the same time grid as the
  existing presence output.
- **SC-004**: Repeating a run with identical inputs and settings produces
  byte-identical per-speaker records, matching the reproducibility standard the
  existing convergence outputs already meet.
- **SC-005**: Each scene classifier has a documented amplitude-invariance verdict
  supported by measurements across a gain range of at least 30 dB, so the question
  "does this classifier amplify the signal itself?" has a per-classifier answer
  rather than an assumption.
- **SC-006**: For every scene-analysis result in a completed run, the audio variant
  and the gain it was computed at are identifiable from the artifacts — 100%
  coverage, no unattributed results.
- **SC-007**: On a recording that audibly contains a background source masked by
  foreground speech, the report states which background source categories the
  foreground-suppressed variant recovers that the unmodified recording does not —
  quantifying what the operation buys, or showing it buys nothing.
- **SC-008**: Every reported "people" or "speech" background category derived from
  the foreground-suppressed variant is accompanied by a leakage measure, so no such
  category can be read without knowing whether it may be leaked foreground speech.
- **SC-012**: On a recording containing two instances of the same source type at
  materially different distances (hence different levels), both are reported, each
  with the margin by which it exceeded the noise floor — demonstrating the policy
  handles a range of distances rather than only the loudest source.
- **SC-014**: 100% of reported background sources carry their above-noise-floor
  margin, so no source can be read without knowing how close to undetectable it
  was.
- **SC-015**: On a pair of recordings identical except that one contains a faint
  background source and the other contains none, the reported background categories
  differ. This is the decisive test that the pipeline is detecting content rather
  than reporting amplified residual foreground — a 30 dB-suppression baseline was
  measured to fail it outright.
- **SC-016**: Achieved foreground suppression depth is reported for 100% of runs
  that produce the suppressed variant, so a null result is attributable to
  insufficient suppression rather than to absence of background content.
- **SC-017**: The detection margin's written derivation cites both a human
  psychophysical basis and a measured classifier capability, and the two are shown to
  agree — satisfying the requirement that the threshold balance the two rather than
  privileging either.
- **SC-018**: On amplified pure noise floor, zero background source categories are
  reported. The noise-character guard must eliminate the water-like and broadband
  environmental labels that amplified noise was measured to produce.
- **SC-019**: Every background mask region carries an uncertainty value, and the three
  states — confidently target-free, confidently target-active, and cannot tell — are
  distinguishable from the artifacts alone.
- **SC-020**: Every reported background source states whether it came from a
  confidently-target-free region or from a region under or adjacent to target activity, so its
  exposure to foreground leakage is never unknown.
- **SC-021**: The total masked duration and its fraction of the recording are reported
  for 100% of runs, so background findings can never be read without knowing how much
  of the recording supports them.
- **SC-022**: On a recording of continuous target activity with no target-free interval, the
  mask is reported as empty with the limitation stated — not silently omitted.
- **SC-023**: On a recording containing a background source both during target activity
  and during a target-free interval, the source is reported from the target-free
  interval at higher confidence, demonstrating that the mask improves interpretability
  rather than merely adding a field.
- **SC-024**: On a breathing-task or cough-task recording, zero target events are
  reported as background sources. This is the decisive test that task metadata is
  actually driving the mask rather than speech detection standing in for it.
- **SC-025**: Every emitted mask records whether it was built with recognized task
  metadata or with the conservative fallback, so no mask's provenance is ambiguous.
- **SC-026**: On a recording where one signal is revised by another, the final value is
  traceable to the signal, round, and evidence that changed it — 100% of revisions
  attributable, with no unexplained state changes.
- **SC-027**: A signal whose uncertainty is revised downward as a consequence of being
  edited is reported distinctly from one whose uncertainty fell because independent
  evidence arrived. This is the decisive test against self-confirmation.
- **SC-028**: On a recording constructed to induce oscillation between two speaker
  interpretations, the loop terminates and reports non-convergence rather than
  emitting the last round's state as settled.
- **SC-029**: Repeating a mutually-influencing run with identical inputs and settings
  yields byte-identical outputs, demonstrating that iteration order is deterministic.
- **SC-030**: A derived signal cannot by itself drive a revision that an independent
  signal contradicts — verifying that influence weighting accounts for derivation
  rather than treating every voter as a peer.
- **SC-031**: Every long-window classifier result from a mask region reports the padding
  fraction of its excised segment, so no heavily padded result can be read as though it
  were a full-window decision.
- **SC-032**: On a recording whose mask regions are all shorter than the long-window
  classifier's window, those regions are flagged as unable to support an uncontaminated
  long-window decision, and the short-window classifier's results are used instead
  rather than the run silently producing padded output.
- **SC-033**: Grid-computed and excision-computed results are distinguishable in the
  artifacts, so results computed over different audio extents are never conflated.
- **SC-009**: Default runs — with foreground suppression not requested — complete
  within 10% of their current wall-clock time, so the additions cost nothing when
  unused.
- **SC-010**: The existing three uncertainty axes remain available and
  interpretable after the identity change; no consumer of presence or utterance
  output needs to change.

## Assumptions

- **No backwards compatibility is required.** Per the project's pre-alpha position,
  the per-speaker identity records may replace the current single-scalar identity
  representation in the final convergence rather than sitting alongside it. Two
  names for one quantity is explicitly not wanted.
- **The per-bucket identity axis itself is retained.** The change targets the final
  convergence outputs. The per-bucket axis remains the mechanism by which identity
  evidence is gathered; it gains per-speaker structure downstream rather than being
  deleted.
- **User Story 2 is about amplification, not speech enhancement.** Per the
  clarification above, the question is whether a scene classifier's own inference
  normalizes signal level. Whether the speech-enhanced pass is a good input for
  background categorization is a separate matter and is not what this story
  measures.
- **The amplitude-invariance question is already answered; User Story 2 now pins
  the answer rather than discovering it.** Both classifiers are amplitude-sensitive,
  established by code audit and gain sweep. The expected asymmetry — long-window
  classifier insensitive, short-window one self-normalizing — did not hold: neither
  self-normalizes, and the long-window model is the more gain-brittle. The story's
  remaining value is a checked-in regression guard, a documented gain range, and
  the per-classifier floor that the detection margin depends on.
- **The relevant machine limit is an absolute floor, not adaptive gain.** The
  short-window classifier reports silence below roughly −60 dBFS regardless of
  content. This is a learned, source-independent decision rather than an arithmetic
  consequence of its log-mel offset, so it cannot be tuned away — but it is monotone,
  which makes it usable as a level tripwire. Both classifiers were measured to be
  stable across roughly −35 to −15 dBFS, giving a usable target band.
- **Neither classifier is simply "more brittle" than the other; they fail
  differently.** The long-window classifier's reported label set churns with gain at
  every level tested, while holding speech confidently to very low levels. The
  short-window classifier's top label is stable near unity gain but collapses
  entirely below its silence floor. Any claim that one is more gain-robust must
  specify *for what content and at what level*, since the ordering reverses between
  speech and non-speech sources.
- **Amplification cannot substitute for suppression depth.** Because gain moves
  background and residual foreground together, the achievable benefit is bounded by
  how completely the foreground was removed. This reorders the work: suppression
  quality is the primary variable and gain is secondary.
- **The distance problem is solved by a local reference, not an adaptive gain.** The
  approach of normalizing to a level target and then amplifying is the one that
  established detection toolchains explicitly rejected. Their consensus is to estimate
  a per-band noise floor, subtract it, and then apply one absolute threshold that
  holds at every distance — because after subtraction, quiet bands and loud bands are
  on the same footing. This is why FR-018b makes floor subtraction the detection
  mechanism and demotes gain to input conditioning.
- **Two failure modes of floor subtraction are known and must be designed around.**
  First, subtracting a per-band floor *deletes* continuous narrowband sources — the
  same step that reveals a transient erases a steady hum or engine drone, which are
  legitimate machine-category findings. Hence the parallel unsubtracted pass in
  FR-021i. Second, aggressive spectral subtraction generates "musical noise":
  spurious tonal components appearing and disappearing at random time-frequency
  locations. That is a synthetic event generator feeding the classifier, so a higher
  residual floor is preferable to more aggressive subtraction.
- **An existing in-repo estimator is the natural landing place, and it has the
  documented bias.** The repository's quality-control metrics already compute a
  tenth-percentile per-band spectral floor, but without bias correction, event
  exclusion, or windowing. That is the right instinct and an incomplete
  implementation; FR-021d through FR-021h describe what it needs to become. The
  existing speech-oriented SNR head is not a substitute — it estimates speech SNR,
  not a background-source floor.
- **Throughout-the-clip background splits into two cases, and only one is detectable
  within a single uncalibrated recording.** A source present in nearly every frame is
  absorbed into its own band's noise floor — worse, the bias correction then lifts the floor
  *above* it, so a steady source reads as sub-floor rather than merely as zero-excess.
  *Narrowband* stationary sources (mains hum, a tonal compressor whine) are recoverable by
  comparing a band against its neighbours, which is unaffected by how much of the recording
  the source occupies. *Broadband* stationary sources — ventilation hiss, room rumble, a
  dense music bed — raise every band together, so a neighbour comparison sees nothing, and
  they are not separable from the microphone's own noise floor without a reference the
  recording does not contain: an equipment noise specification, a silent calibration take,
  or a cross-recording baseline. Absent one, the honest output is that the floor's origin is
  undetermined.
- **The SNR gate is a novel component and must be additive.** Established sound-event
  detection practice suppresses false positives entirely in the posterior domain —
  smoothing, class-wise thresholds, minimum durations, hysteresis — and does not gate
  on estimated SNR at all. The gate specified here is therefore not standard practice
  and should be layered alongside that machinery rather than replacing it.
- **An alternative classifier may be better positioned for faint content.** At least
  one comparable AudioSet model applies a large pre-scale ahead of the same
  log floor that the long-window classifier lacks, giving it far more headroom. This
  is worth measuring, but it MUST be measured rather than assumed — headroom
  arithmetic already mispredicted the ordering of the two current classifiers, where
  a learned decision boundary rather than the arithmetic floor turned out to bind.
- **User Story 3 is a feasibility investigation that may ship or may not.** The
  user framed it as "could one carry out..." — a documented negative result with
  supporting measurements is an acceptable outcome, and FR-018 through FR-030
  describe the capability only if the measurement justifies it.
- **The residual of existing speech enhancement is the natural first candidate**
  for a foreground-suppressed variant, since subtracting extracted speech from the
  original approximates the background at no new model cost. Dedicated separation
  models are a fallback if the residual proves inadequate.
- **No new isolated environment is introduced unless a dedicated separation model
  proves necessary**, consistent with the preference expressed during the scene
  analysis work.
- **Validation uses the existing local sample recordings.** A labeled multi-speaker
  benchmark does not exist yet; thresholds are therefore configurable (FR-011) and
  are not tuned against a single annotation. This follows the standing decision not
  to optimize thresholds on one ground truth.
- **Speaker counts are small.** Validation recordings contain roughly one to five
  speakers; the design need not scale to large meetings.
- **The background mask is largely derivable from signals already computed.** The
  existing presence axis distinguishes "confidently no speaker" from "cannot tell",
  which is exactly the distinction the mask needs; the diarization and voice-activity
  outputs supply the boundaries the guard intervals attach to. The mask is therefore
  expected to be a new derived output rather than a new model — **except** for tasks
  whose target is a non-speech vocal event, where target detection matching the task
  type is required and existing speech detection is insufficient (FR-033a).
- **Scope is lab-like collection with the microphone close to the source.**
  Generalization to distant, ambient, or field recording is explicitly deferred. Two
  consequences follow and should shape expectations rather than be discovered later:
  the near-field geometry puts the target far above any background source, so the
  foreground-to-background ratio is large and the +10 dB margin is correspondingly
  hard for a genuine background source to clear; and in a quiet room with a close
  microphone the binding floor is very likely the capture chain's own self-noise
  rather than the room, which is exactly the condition FR-021b requires the system to
  declare.
- **Task metadata is assumed available and trustworthy.** The mask's correctness
  depends on knowing what the target event type is. Where it is missing, FR-033b's
  conservative fallback applies, which will over-exclude and shrink the mask rather
  than risk misattributing target signal as background.
- **The mask may carry more of the evidential weight than suppression does.** Since a
  30 dB suppression baseline was measured to leave the residual foreground dominant,
  the confidently-target-free intervals may be where most trustworthy background
  findings actually originate. If that proves true on validation recordings, the mask
  becomes the primary path and deep suppression becomes an enhancement for the
  co-occurring case — a reordering the plan should be prepared for.
- **Thresholds must be relative, because recordings are not calibrated.** A
  recording's digital level carries no absolute sound-pressure reference unless the
  capture chain was calibrated, which ours is not. Absolute hearing-threshold
  criteria are therefore not directly applicable; the detection margin must be
  expressed relative to the recording's own estimated noise floor. Psychophysics
  informs *how large* that relative margin should be, not an absolute level.
- **Loudness-model criteria are the better-motivated option but are unavailable.**
  Established partial-loudness models define masked threshold precisely and would be
  the theoretically correct criterion. They are ruled out for two independent
  reasons: they require absolute sound-pressure input, and because they are
  nonlinear an unknown level offset does not merely rescale the answer — it changes
  which bands fall below threshold. Separately, such models require the signal and
  background spectra to be supplied *already separated*, which conventional
  pipelines cannot do. Our foreground suppression would actually satisfy that second
  requirement, so if calibrated recordings ever enter scope this becomes a genuinely
  applicable upgrade path rather than a dead end. The band-relative SNR criterion is
  the correct fallback, not a compromise.
- **For consumer-grade recordings the microphone, not human hearing, is often the
  binding floor.** Consumer and phone microphone self-noise sits around or above the
  level of a quiet bedroom, whereas studio-grade capture is roughly 15–20 dB
  quieter. Content genuinely near human threshold was therefore never captured — it
  was buried in electronic noise before quantization. Bit depth is never the limiting
  factor. The spec requires this to be stated rather than papered over (FR-021b),
  because "your microphone could not hear it" is both more defensible and more
  useful than an unsupportable perceptual claim.
- **Automatic gain control in the capture chain, if present, corrupts the very
  ratio being measured.** It continuously re-invalidates any calibration and
  compresses the foreground-to-background ratio. Its presence cannot be assumed
  absent and is a known threat to the measurement.
- **Directional location is not recoverable from mono audio.** The clarification
  asks the gain policy to mimic human recognition of sources at different distances
  *and locations*. Distance has partial monaural cues — level, direct-to-reverberant
  ratio, and high-frequency rolloff — so distance-awareness is achievable. Direction
  requires at least two channels and is not achievable on the mono recordings in
  scope. This specification therefore commits to distance-awareness and treats
  directional localization as out of scope rather than silently promising it.
- **Speaker identifiers are run-local.** Linking speakers across recordings is out
  of scope.

## Dependencies

- **Coordination with in-flight diarization work is required, and the mutual-influence
  decision raises the stakes.** An open pull request on multi-speaker diarization
  uncertainty (#537) overlaps User Story 1 directly. Because identity now participates
  fully in the adaptive loop (FR-011a) rather than only reporting, this work touches the
  intervention catalog and the fusion writers that PR may also be changing. Reconciling
  the two is a prerequisite, not a follow-up — a report-only design could have been
  developed in parallel; this one cannot.
- **The mutual-influence architecture is the highest-risk element of this
  specification.** It is deliberately chosen, and the guards are specified (FR-011b
  through FR-011h), but self-confirmation and oscillation are real failure modes that a
  one-directional design does not have. Planning should sequence the guards *before* the
  influence paths they protect, so the loop is never able to confirm its own edits even
  transiently.
- Relies on the existing three-axis uncertainty workflow, the adaptive convergence
  loop's final output writers, and the background sound-source category mapping
  already in place.
- Relies on the existing speech enhancement capability, both as the enhanced
  variant under audit and as the candidate source of a foreground-suppressed
  variant.

## Out of Scope

- Cross-recording speaker linking or speaker identification against enrolled
  identities.
- Building a labeled multi-speaker benchmark or tuning thresholds against one.
- Improving diarization accuracy itself; this work changes how uncertainty about
  speakers is represented and reported, not how speakers are detected.
- Changing the presence or utterance axes' definitions.
- Improving audio for listening quality; amplification here serves machine
  categorization, not human playback.
- Modifying the scene classifiers themselves. User Story 2 measures their existing
  behavior; it does not change how they normalize their input.
- Recording situations other than lab-like collection with a close microphone.
  Distant, ambient, room-scale, and field recording are deferred; the thresholds and
  the mask's assumptions are calibrated for near-field capture and should not be
  assumed to transfer.
