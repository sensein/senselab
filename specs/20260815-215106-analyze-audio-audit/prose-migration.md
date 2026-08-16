# Prose migration staging — Sweep A `rationale-to-migrate` candidates

This document stages every one of Sweep A's 98 `rationale-to-migrate` findings
(`candidates/deduped.md`, F-9..F-104 plus the two reclassified entries F-105/F-106) for
migration out of in-code docstrings/comments into the design documents named in each entry's
`destination`. **Nothing is migrated here** — this is the record that makes deleting the
in-code copy safe later, per Task 9's brief: quote the rationale verbatim, record where it
came from, and where it is meant to go, so a future deletion pass has something to check
against instead of trusting memory.

These 98 findings were **not** run through the refutation/reproduction gates (`verdicts/`):
they are prose-classification calls, not defect claims, so there is no mechanism to invent or
reproduce. `register.md`'s "Ungated classification" section marks all 130 Sweep A
classification entries (98 of these plus 32 `restates-code`) `graph_implication: irrelevant`
in bulk on that basis — a class label is not a runtime signal the triage graph can consume or
avoid.

Grouped below by destination (39 module-groups, largest first). Within a group, entries are
ordered by finding id. Each entry shows the verbatim source text at the cited `file:line`
range(s) — read directly from the current tree at audit time, not retyped.

## Pattern 4 — four clusters to consolidate once, not migrate per-file

`deduped.md`'s cross-sweep pattern 4 identifies ~7 near-verbatim duplicate rationales spread
across 3+ files with no single canonical home. Migrating each copy into its destination
independently would recreate the duplication one level up (in the design doc instead of the
code). These should be written **once** at migration time, with the per-file copies replaced by
a cross-reference:

- **"silhouette coefficient is not a probability"** — F-24 (`embeddings.py`) and F-52
  (`speech_presence_link.py`), both destined for `l1-post-processing-register.md` item 12 (see
  the `l1-post-processing-register.md` group below).
- **"5 speakers vs 2 diarizers reported" validation anecdote** — F-22 (`speaker_identity.py`),
  echoed in `influence.py`/`support.py`/`reliability.py` (F-35 is the "third independent
  telling"); see `influence/support/reliability weighting design` and `speaker attribution /
  clustering-statistics design` groups below.
- **"three id namespaces once rendered as S0"** — F-23 (`identity_binding.py`), duplicated in
  `harmonize.py`/`clustering.py`, and told a fourth time in F-28's (`joint.py`) "What changes
  from J4" section; both land in the `layered-architecture.md` group below.
- **asr.py's own triple-telling of its `consensus_words` removal history** — F-16, F-18, F-19,
  all in the same file at different line ranges; see the `asr axis design` group below.

---

## background-scene design (14 entries)

### F-27 (raised-by A-27) — `src/senselab/audio/workflows/audio_analysis/invariance.py:1-26`

- destination (as filed): background-scene design (amplification finding)
- rationale: gain-scaling/background-detection cross-reference

Lines 1-26:
```python
"""Invariance probes: perturbations under which a correct model must not change its answer.

The stability factor in :mod:`reliability` compares the raw and enhanced passes. That is a
useful sample, but enhancement is a *genuine transform* — a model is entitled to answer
differently on enhanced audio, so a change there is ambiguous between "this model is unstable"
and "the audio really did change".

These perturbations are different in kind. Each is chosen so that a correct model returns the
same answer, which makes any change in its answer a defect in the model rather than a response
to the signal:

- **Gain scaling.** Changes no signal-to-noise ratio — it moves the source and everything
  around it together. This is the same measurement that reframed background detection away
  from amplification: gain cannot rescue a buried source because it lifts the masker too.
  Speaker count, speaker speaker and transcript are all level-independent facts.
- **Whole-sample time shift.** Padding by an integer number of samples moves the timeline
  without resampling, so no sample value is altered and no interpolation error is introduced.
  A model whose speaker count depends on where its analysis windows happen to land is
  reporting an artifact of framing.
- **Small DC offset.** Speech models operate on mean-removed spectra, so a small constant
  should be invisible. One that is not is leaking a time-domain statistic into its decision.

Measuring these requires re-running inference, so the probe is opt-in rather than part of a
default run. What it buys is an unambiguous reliability signal: unlike the enhanced-pass
comparison, a failure here cannot be explained away as the audio having changed.
"""
```

### F-36 (raised-by A-36) — `src/senselab/audio/workflows/audio_analysis/background_mask.py:1-24`

- destination (as filed): background-scene design (mask semantics)
- rationale: mask-semantics rationale (30 dB suppression baseline leaves residual foreground dominant)

Lines 1-24:
```python
"""Background mask: where no target activity happens (T031-T034, FR-031 to FR-045).

The mask marks regions free of **target activity** — activity from the near-microphone
participant being recorded — not regions free of speech. Two consequences follow, and both
matter more than the naming:

**Background claims are trustworthy in a target-free region without any suppression.**
There is no foreground there to leak, so the suppression-depth constraint that bounds
everything else simply does not apply. Since a 30 dB suppression baseline was measured to
leave residual foreground dominant, these regions may carry most of the trustworthy
background evidence in a recording.

**What counts as target activity depends on the task.** In a breathing or cough task the
target *is* a non-speech vocal event, and speech detection reports no activity while it is
happening. A mask built from speech activity alone would admit the target breaths — and
because AudioSet maps ``Breathing`` and ``Cough`` to the ``people`` category, they would be
reported as a background human-sound source. That is the collected signal misattributed as
an environmental finding, which is why :func:`requires_label_detection` exists and why
FR-033a forbids relying on voice activity alone.

Scope: lab-like collection with the microphone close to the source. A distant talker stays
*in* the mask and is reportable as a background source (FR-033c) — target-free is not
speech-free.
"""
```

### F-37 (raised-by A-37) — `src/senselab/audio/workflows/audio_analysis/background_mask.py:244-255` (``_classify_bucket``)

- destination (as filed): background-scene design (mask classification)
- rationale: measured 0.99/0.99 confidence-and-uncertainty collision producing one whole-file region

Lines 244-255:
```python

    Both committed verdicts demand low uncertainty. ``target_free`` always did — "probably
    nothing there, but I cannot tell" is not a region background claims can rest on — while
    ``target_active`` demanded none, so a bucket at confidence 0.99 *and* uncertainty 0.99
    committed to "target active" on evidence that supported no verdict at all. Measured on a
    21.5 s conversation, that asymmetry produced a single whole-file ``target_active`` region
    at uncertainty 0.9997.

    ``nontarget_active`` requires the target to be absent as well as other content to be
    present: background content underneath active target speech is not a clean region to
    introspect, which is the leakage problem suppression-depth measurement exists for.
    """
```

### F-38 (raised-by A-38) — `src/senselab/audio/workflows/audio_analysis/background_mask.py:462-477` (``_speech_activity_by_bucket``)

- destination (as filed): background-scene design (mask evidence)
- rationale: boolean-collapse-of-1070-buckets rationale

Lines 462-477:
```python

    Each diarizer contributes the **fraction of the bucket it covers**, and the bucket takes the
    mean across diarizers. Not a hit test: a segment clipping a bucket by 10 ms and one filling it
    entirely are not the same evidence, and ``diar_covered_fraction`` was already saying so
    elsewhere in this package while this function asked "does anything overlap".

    That boolean is what made the mask unable to be uncertain. Every bucket of a continuous
    conversation scored exactly 1.0, so every bucket classified identically at margin 1.0 —
    uncertainty 0.0 — and ``build_mask``'s run-length encoding correctly collapsed 1070 identical
    buckets into a single 21 s region reporting no doubt at all. The encoding was right; the
    evidence had already thrown away every distinction it could have encoded.

    ``None`` where no diarizer contributed, so "nobody looked" stays distinguishable from
    "everybody said no" — collapsing the two is what would let an unexamined bucket be
    reported as clean background.
    """
```

### F-39 (raised-by A-39) — `src/senselab/audio/workflows/audio_analysis/noise_floor.py:1-32`

- destination (as filed): background-scene design (noise-floor estimation)
- rationale: quantile-of-exponential-noise derivation (9.8 dB at p10) and explicit "no published precedent" caveat

Lines 1-32:
```python
"""Per-band noise-floor estimation (T056-T059, FR-021a to FR-021i).

Detection works by subtracting a locally estimated per-band floor and applying one margin
threshold to what remains — not by amplification. That is what makes the different-distances
problem tractable: after subtraction a near source and a far source are each judged against
their own band floor, so a single threshold holds at every distance. The approach is taken
from established detection practice (ecoacoustics, marine and terrestrial bioacoustics),
where the design goal is stated directly: after per-bin floor subtraction, power fluctuates
around 0 dB during silence and is considerably higher during an event, so one absolute
threshold suffices.

Four properties, each ruling out a simpler estimator:

**Percentile, not mean or minimum.** A low percentile tolerates high event occupancy — a
tenth percentile survives up to 90% of a band's frames being event — where a mean absorbs
events by construction and a raw minimum carries a large downward bias.

**Bias-corrected.** A ``q``-quantile of exponentially distributed noise power sits a
calculable factor below the mean: about 9.8 dB for a tenth percentile. Uncorrected, every
relative-dB gate is that much more permissive, and the failure looks like generosity rather
than a bug. The correction is validated against synthetic noise in the tests.

**Patch-aggregated, never per-bin.** A single time-frequency bin's log-power has a spread of
about 5.6 dB, so 3 sigma is ~17 dB and a few-dB threshold on one bin is meaningless. Over a
~1 s patch the spread falls below a few tenths of a dB, making the same threshold many sigma.

**Conditioned on target activity.** Every published estimator assumes the floor is
independent of the events. A suppression residual violates that — artifact level correlates
with the removed talker's level — so one unconditioned floor over-gates quiet stretches and
under-gates busy ones. Estimating per activity stratum is the mitigation, and it has **no
published precedent**: validate before relying on it.
"""
```

### F-40 (raised-by A-40) — `src/senselab/audio/workflows/audio_analysis/noise_floor.py:165-197` (``estimate_band_floor_db``)

- destination (as filed): background-scene design (noise-floor estimation)
- rationale: "p10+6dB cut discards two-thirds of exponential noise" derivation

Lines 165-197:
```python
    # The correction is always applied *inside* the loop, because the exclusion reference
    # must be an estimate of the noise mean; using the raw quantile there re-introduces the
    # runaway described below. `apply_bias_correction` only affects the returned value, so a
    # caller comparing the two observes the correction alone rather than the correction plus
    # a different iteration trajectory.
    correction_db = quantile_bias_correction_db(quantile)
    correction_lin = 10.0 ** (correction_db / 10.0)

    surviving = power
    floor_lin = float(np.quantile(surviving, quantile)) * correction_lin
    iterations = 1
    for _ in range(max(0, max_iterations - 1)):
        # Compare against the *corrected* floor -- an estimate of the noise mean -- not the
        # raw quantile. Excluding relative to the raw quantile removes most of the noise
        # distribution itself (a tenth-percentile-plus-6 dB cut discards roughly two thirds
        # of exponentially distributed noise), and re-taking a low quantile of the truncated
        # remainder drives the estimate down every pass. That runaway reads as a very quiet
        # floor, which makes every margin permissive.
        limit = floor_lin * (10.0 ** (event_exclusion_db / 10.0))
        kept = surviving[surviving <= limit]
        if kept.size < max(8, int(0.05 * power.size)):
            # Too few frames left to estimate from; stop rather than chase the floor into a
            # handful of samples.
            break
        new_floor = float(np.quantile(kept, quantile)) * correction_lin
        iterations += 1
        converged = abs(10.0 * math.log10(new_floor / floor_lin)) < 0.1
        floor_lin, surviving = new_floor, kept
        if converged:
            break

    floor_db = 10.0 * math.log10(floor_lin)
    return (floor_db if apply_bias_correction else floor_db - correction_db), iterations
```

### F-41 (raised-by A-41) — `src/senselab/audio/workflows/audio_analysis/noise_floor.py:376-410`

- destination (as filed): background-scene design (stationary source detection)
- rationale: ECMA-74/ISO 7779 prominence-ratio derivation

Lines 376-410:
```python
# ── stationary sources (T068, FR-021i) ─────────────────────────────────
#
# A source present throughout a recording defeats the estimator above, and not by
# accident: it *is* the tenth percentile of its own band, so it is absorbed into the
# floor and its excess over that floor reads ~0 dB. Air conditioning, ventilation,
# mains hum, a music bed -- exactly the "background that exists throughout the clip"
# case, and exactly the sources a background characterizer most wants to name.
#
# The escape is to stop comparing a band against its own history and compare it
# against its NEIGHBOURS instead. A steady narrowband source is prominent relative to
# adjacent third-octave bands even when it is perfectly steady in time, and that
# comparison is unaffected by how much of the recording it occupies.
#
# Standards-grounded rather than invented: ECMA-74 / ISO 7779 define a discrete tone
# as prominent at a Prominence Ratio of about 9 dB (the critical band containing the
# tone against its two neighbours) or a Tone-to-Noise Ratio of about 8 dB. Those are
# the same figures the margin ladder's upper tier converged on, from a different
# tradition.
#
# SCOPE OF THIS DETECTOR, stated precisely because the neighbouring claim is easy to
# overstate. Prominence catches *narrowband* stationary sources -- mains hum, a tonal
# compressor whine, a fan blade rate. A **broadband** stationary source raises every
# band together (ventilation hiss, room rumble, a dense music bed), so a neighbour
# comparison sees nothing.
#
# What that does NOT mean: it does not mean a structured or content-bearing background
# is unmeasurable. Its level, spectral shape, and ratio to the near-field foreground are
# all measurable -- see `foreground_background_ratio_db`. The single narrow thing an
# uncalibrated lone recording cannot do is *attribute* a smooth broadband floor to
# equipment versus room, because both are steady, broadband and spectrally smooth.
#
# And that attribution is recoverable from a cohort: across recordings from one rig the
# equipment contribution is the part common to all of them while the room contribution
# varies, so `cross_recording_baseline` separates the two. `binding_floor` reports which
# limit applies once either that or an equipment specification is supplied.
```

### F-42 (raised-by A-42) — `src/senselab/audio/workflows/audio_analysis/sources.py:1-27`

- destination (as filed): background-scene design (source detection guards)
- rationale: "amplifying a noise floor produces plausible fake environmental labels" rationale

Lines 1-27:
```python
"""Background source detection: margin ladder and fabrication guards (T060-T062).

A candidate becomes a reported background source only by clearing a **margin above its own
band's noise floor**. Never by being amplified: gain moves a source and the residual
foreground together and changes no signal-to-noise ratio, so it cannot promote a tier.

The 3 / 6 / 10 dB ladder is corroborated from three independent directions — human masked
threshold and audibility criteria, a dozen unrelated detection traditions in bioacoustics
and noise standards, and the classifiers' own measured reliable-detection floors. That
convergence is the reason to trust the values; none of them was fitted here.

The guards exist because the failure mode is not a missed source, it is a **fabricated**
one. Amplifying a noise floor produces confident, plausible environmental labels —
waterfall, water, gurgling, static — that are statistically indistinguishable from genuine
broadband noise and read as real findings. Three layers stop that:

- a **pre-gain level reject**, because a segment below the classifiers' measured trust floor
  should not be amplified and interpreted at all;
- a **noise-character test**, because broadband noise separates cleanly from structured
  sources on spectral flatness for the cost of one transform;
- a **quarantine list** for the labels amplified noise actually produces, which may only be
  reported when the noise-character test passes.

Plus a **floor-response signature** check: a classifier below its floor can emit a fixed
label pattern, and one measured signature pairs a silence label at 0.44 with a co-occurring
label at 0.35 — so keying on the silence label alone would let the second one through.
"""
```

### F-43 (raised-by A-43) — `src/senselab/audio/workflows/audio_analysis/sources.py:245-256`

- destination (as filed): background-scene design (excision routing)
- rationale: measured excision-vs-mixed-window comparison (0.705 vs 0.548)

Lines 245-256:
```python
# ── excision routing (T067, FR-041 to FR-045) ──────────────────────────
#
# The long-window classifier runs on *excised* mask segments rather than on the full
# timeline. Measurement drove it: with a loud-then-quiet test signal, excising the
# quiet segment and classifying it alone beat every mixed-window variant (0.705 vs a
# best of 0.548), because one 10.24 s window spanning both halves couples them and the
# loud half dominates the decision. The short-window classifier needs no excision --
# its ~1 s windows already sit entirely inside one half or the other.
#
# The cost is real and is reported rather than absorbed: a mask region shorter than the
# classifier's window is zero-padded, and padding maps to a fixed value while the signal
# region drifts with gain, so the pad-to-signal contrast is itself gain-dependent.
```

### F-44 (raised-by A-44) — `src/senselab/audio/workflows/audio_analysis/foreground.py:1-24`

- destination (as filed): background-scene design (foreground suppression)
- rationale: oracle-experiment rationale (30 dB suppression made present/absent background indistinguishable)

Lines 1-24:
```python
"""Foreground suppression and its depth measurement (T064-T065, FR-018 / FR-018a).

The residual of speech enhancement is the background: subtracting estimated speech from the
original leaves what the enhancer decided was not speech, at no additional model cost.

**Suppression depth, not gain, is the binding constraint.** An oracle experiment settles it:
with 30 dB of suppression and the residual amplified to a healthy level, the reported result
was *identical* whether a faint background source was present or entirely absent — the
leaked foreground dominated either way. Amplification moves the background and the residual
foreground together and changes no signal-to-noise ratio, so no amount of it rescues shallow
suppression. That is why every finding from a suppressed variant carries a depth and a
leakage margin: a null result must be attributable to insufficient suppression rather than
to absence of background content.

Leakage is measured by **projection**, not by level. The component of the residual that is
still correlated with the estimated speech is leaked foreground; the orthogonal component is
what is genuinely not speech. A level-only measure cannot tell a quiet residual that is
mostly leakage from a quiet residual that is mostly background, and those license opposite
conclusions.

Known risk carried from the research: aggressive spectral subtraction generates *musical
noise* — spurious tonal components appearing and disappearing at random time-frequency
locations. That is a synthetic event generator feeding the classifier, so a higher residual
noise floor is preferable to deeper subtraction.
```

### F-45 (raised-by A-45) — `src/senselab/audio/workflows/audio_analysis/foreground.py:121-128` (``is_deep_enough_for``)

- destination (as filed): background-scene design (foreground suppression, consolidate with F-44)
- rationale: same oracle-experiment rationale reused for the depth-below-foreground comparison

Lines 121-128:
```python
    def is_deep_enough_for(self, background_below_foreground_db: float) -> bool:
        """Whether suppression reaches far enough to expose a source at this depth.

        The oracle experiment showed 30 dB of suppression leaving the residual foreground
        dominant over a background 30 dB down, so the comparison is against the source's own
        depth below the foreground — not against a fixed threshold.
        """
        return self.achieved_depth_db > float(background_below_foreground_db)
```

### F-46 (raised-by A-46) — `src/senselab/audio/workflows/audio_analysis/sound_sources.py:31-48` (``AUDIOSET_SCORE_FUNCTION``)

- destination (as filed): background-scene design (sound-source categorization)
- rationale: softmax-vs-sigmoid class-competition rationale

Lines 31-48:
```python
AUDIOSET_SCORE_FUNCTION = "sigmoid"
"""Output transform for AudioSet classification heads (FR-017c).

AudioSet is a **multi-label** task: many classes can be simultaneously present, and a
527-class head trained on it emits per-class evidence, not a choice among alternatives.
Reading it through a softmax makes the classes compete, which suppresses secondary classes
multiplicatively — so a background source at fixed underlying evidence gets a
systematically smaller share of :func:`_window_category_masses` than it should, and the
suppression is worst exactly when a dominant source is present. That is the case this
feature exists to handle, so the competition is not an acceptable approximation.

Per-window normalization below cancels the *scale* difference between a softmax and a
sigmoid, but not the competition structure — hence fixing the transform rather than
post-hoc rescaling.

Ranking is unaffected: both transforms are monotone in the logit, so ``top_k`` selects the
same labels either way. Only the mass proportions change.
"""
```

### F-56 (raised-by A-56) — `src/senselab/audio/workflows/audio_analysis/mask_harvest.py:1-24`

- destination (as filed): background-scene design (D-22)
- rationale: "uncertainty was a property of there being one producer, not of the mask" rationale

Lines 1-24:
```python
"""The ``background_mask`` axis's vote harvest: VAD, ASR words and speaker spans (D-22).

The mask's uncertainty was ``1 - confidence`` of a single derived judgement. That read as a property of
the mask when it was a property of there being **one producer** — and it kept the axis out of
``HARVESTED_AXES``, which is why ``disagreements.json`` never listed it while ``estimates/`` did.

Three sources bear on whether the target was active in a bucket, so the axis's doubt becomes
cross-source disagreement like every other axis's:

===========  ==================================================================
``speech``   a continuous speech probability (Brouhaha's VAD head)
``words``    seconds of ASR word coverage
``speakers`` diarizer occupancy — how much of the bucket a speaker covers
===========  ==================================================================

**What each one *means* depends on the task**, which is why these are votes and not a formula. In a
speech task, all three indicate target *activity*. In a breathing task the target is the breath, speech
detection is silent through it, and a speech vote therefore indicates target **absence** — and since
AudioSet maps ``Breathing`` to ``people``, a mask built from voice activity alone reported the collected
signal as a background source. That is the failure this module's task gate exists to prevent.

Emitted on the **speech-presence grid**, which the mask shares (D-24 correction): the mask is derived
from the presence axis, so on one grid the derivation is exact and needs no projection.
"""
```

### F-57 (raised-by A-57) — `src/senselab/audio/workflows/audio_analysis/mask_harvest.py:37-52` (``TARGET_POLARITY``)

- destination (as filed): background-scene design (task-gated polarity)
- rationale: task-gated polarity rationale (breathing task: speech vote means target absence)

Lines 37-52:
```python
TARGET_POLARITY: Final[Mapping[str, Mapping[str, bool]]] = {
    # task type -> {source -> does a positive reading mean the TARGET was active?}
    "speech": {"speech": True, "words": True, "speakers": True},
    # The breath is the target. Speech detection is silent through it, so speech present means
    # something other than the target was happening. Words and speakers likewise.
    "breathing": {"speech": False, "words": False, "speakers": False},
    # A sustained vowel or /a/ phonation: voiced, so VAD fires on the target, but an ASR transcribing
    # words is hearing something else.
    "phonation": {"speech": True, "words": False, "speakers": True},
}
"""Whether a positive reading from each source indicates **target** activity, per task type.

The mapping is the whole reason these are votes: the same measurement means opposite things about the
target depending on what was asked for. A default would make the breathing case silently wrong, which is
how a collected breath came to be reported as background.
"""
```

---

## layered-architecture.md (9 entries)

### F-23 (raised-by A-23) — `src/senselab/audio/workflows/audio_analysis/identity_binding.py:1-19`

- destination (as filed): layered-architecture.md (D-19)
- rationale: the three-id-namespace ("all once rendered as S0") rationale, near-verbatim repeated in harmonize.py and clustering.py (3 copies)

Lines 1-19:
```python
"""Binding fused speaker ids to each tool's own labels, from spans (D-19's C2, replacing J4).

Three id namespaces stay distinct because all three once rendered as ``S0``: a model's own labels
(``SPEAKER_00``, ``spk0``), the pass-wide cluster that harmonises labels across diarizers (``C0``),
and the fused speaker id in ``final/speakers.json`` (``S0``). This module binds the third to the
first, and the binding is **evidence** rather than a preprocessing step — how well-determined it is
*is* part of the speaker uncertainty, which is what makes its stability a convergence criterion (C2).

**What changes from J4.** It bound ``S_k`` to ``segmentation-3.0``'s activation channels, which are
permutation-arbitrary within each inference: they carried timing but could not name anyone, so the
binding was the only thing supplying a name. With diarizers emitting spans there are no channels —
each tool has its own labels, carrying timing *and* its own identity — and the binding gains something
the channel version could not have: **a speaker bound by one diarizer and unbound by another is a
measurable disagreement.** That is the signature an off-target speaker leaves.

**Two properties carried over unchanged, because both are refusals to decide.** A speaker with no
overlapping label is left *unbound* rather than given the least-bad one; a tool label no speaker
claimed is *reported* rather than dropped, because that is the shape a missed speaker takes.
"""
```

### F-26 (raised-by A-26) — `src/senselab/audio/workflows/audio_analysis/harmonize.py:1-23`

- destination (as filed): layered-architecture.md (D-6)
- rationale: "cross-model statement first guesses same person" framing

Lines 1-23:
```python
"""H2: the common speaker space, and the uncertainty of constructing it.

Every diarizer names its speakers arbitrarily — ``SPEAKER_00``, ``spk0`` — and the names carry no
meaning across models. So any cross-model statement about speaker first *guesses* that two labels
denote the same person. Treated as fact, that guess makes two models which were never correctly
compared read as disagreeing, and speaker uncertainty then stays high in exactly the regions where
per-speaker speech_presence is unambiguous. That is the observation this module exists to address.

**Harmonization is therefore an estimation step and reports its own uncertainty.** Two independent
matchers run over the same labels:

- **temporal overlap** — a one-to-one assignment maximising co-occurrence duration (Hungarian);
- **embedding centroid** — a one-to-one assignment maximising mean-embedding cosine similarity.

Where they agree, the assignment is confident. Where they disagree, that disagreement *is* the
assignment uncertainty, measured with the same estimators as every other axis: normalised Shannon
entropy over the candidate targets, and weighted vote share for the winner. Neither matcher alone
can express doubt about itself, which is why both run (D-6).

Three id namespaces stay distinct because all three once rendered as ``S0``: a model's own labels
(``SPEAKER_00``, ``spk0``), the harmonized cluster produced here (``C0``), and the fused speaker id
in ``final/speakers.json`` (``S0``).
"""
```

### F-28 (raised-by A-28) — `src/senselab/audio/workflows/audio_analysis/joint.py:1-29`

- destination (as filed): layered-architecture.md (D-19/D-7)
- rationale: "J1/J4 have moved" history, duplicating identity_binding.py's "What changes from J4" section

Lines 1-29:
```python
"""L2 joint estimation — signals that exist only by combining others.

Each function here answers a question no single tool was asked. They are L2 by construction: the
inputs are L1 measurements, and the combining rule is a modelling choice that belongs where it can
be seen and changed.

**J1 and J4 have moved.** The count posterior is now cross-diarizer spread
(:mod:`.occupancy`) and the speaker binding is over each tool's own labels
(:mod:`.identity_binding`), both from spans. What lived here was built on
``segmentation-3.0``'s per-speaker channels, whose independence the Poisson-binomial assumed
and which a powerset conversion does not have.

Available now, while J4 (per-speaker presence) still needs rounds, and the reason is worth stating
because it decides what else can be built on the activation channels. `segmentation-3.0` reports
one activation per speaker, but the channel ordering is arbitrary *within a window*: channel 1 in
one window and channel 1 in the next are not the same person. So any quantity that depends on which
channel is whom is ill-defined until the speaker↔channel assignment is resolved, which is the joint
space D-7 hands to L2 rounds. A **count** of active channels is invariant to that permutation, so it
is well-defined immediately — and it is precisely the signal the old noisy-or collapse destroyed,
since `1 − Π(1 − p_k)` answers "is anyone speaking" and discards how many.

**J2 — where the voice changes** (:func:`speaker_change_series`). Compares each embedding window
against the one a whole window-width later, so the two sides are disjoint spans meeting at a
boundary. Adjacent windows at the 50 ms hop share 97.5% of their audio, so the change is present
but low-amplitude and smeared across the window width rather than appearing as a step.

**J7 — which reading the acoustics support** (:func:`phoneme_transcript_agreement`). PPG posteriors
reach the audio without passing through a language model, so they can adjudicate between two ASR
readings of the same span without echoing a third transcriber's opinion.
```

### F-29 (raised-by A-29) — `src/senselab/audio/workflows/audio_analysis/statistics.py:1-36`

- destination (as filed): layered-architecture.md or an estimator-taxonomy note
- rationale: "all called 'uncertainty'" naming-collision history

Lines 1-36:
```python
"""Confidence, variability, and uncertainty — three quantities, three estimators.

The codebase had been calling all of them "uncertainty", which is why a max-doubt fold, a
Shannon entropy, and a max-minus-min spread all ended up in a column of that name. They answer
different questions and only one of them is a probability:

**Confidence** — ``P(proposition)``, in ``[0, 1]``. Estimated as the weighted share of signals
asserting the proposition. Because it is a probability it can be calibrated against ground
truth, and ``0.0`` is the confident claim *"definitely not"* rather than "we did not look".

**Variability** — dispersion of repeated measurements of one quantity: the sample standard
deviation, in the units of the quantity. Deliberately *not* squeezed into ``[0, 1]``; rescaling
it would make it a different statistic and invite reading it as a probability. Needs at least
two measurements — with one, zero would assert perfect agreement that was never observed.

**Uncertainty** — how undetermined the answer is, estimated as Shannon entropy over the
distribution of outcomes, normalised by ``log k`` so an even split reads 1.0 whether there are
two outcomes or five. Raw entropy in nats is unbounded above and so cannot be compared across
axes with different outcome counts.

Uncertainty further decomposes, which is what makes it actionable:

    total = H(mean of the signals' distributions)
    aleatoric = mean of H(each signal's distribution)
    epistemic = total - aleatoric

Epistemic uncertainty is disagreement *between* signals, and it is the reducible part: another
measurement can resolve it. Aleatoric uncertainty is doubt every signal shares, which more
measurements of the same kind cannot remove. Reporting shared internal doubt as reducible would
send the adaptive loop off to gather evidence that cannot help — the decomposition exists so it
can tell the difference.

This is the standard mutual-information decomposition used for ensemble and MC-dropout
uncertainty; the speaker ``0 <= epistemic <= total`` holds by Jensen's inequality, and a
violation means a sign error rather than an interesting finding.
"""
```

### F-30 (raised-by A-30) — `src/senselab/audio/workflows/audio_analysis/measurements.py:1-16`

- destination (as filed): layered-architecture.md (D-18)
- rationale: "frame_mean at a resolution the model never reported" / units:"mixed" history

Lines 1-16:
```python
"""Storing a measurement in its native shape, with its schema attached (D-18).

The bridge between :mod:`.shapes` (what a measurement is), :mod:`.keys` (what it is called) and
:mod:`.stage_io` (whether this stage may name it). Nothing here decides anything about the audio: it
serializes a shape and reads it back unchanged.

**The schema travels with the artifact.** Units, hop, window, vocabulary, top-*k*, speaker capacity and
channel semantics go into the parquet's schema metadata, because a value whose units live somewhere
else is a value a later reader will guess about — and the guesses observed were `frame_mean` at a
resolution the model never reported and six quantities under ``units: "mixed"``.

**Two absences that must not collapse into one.** A file with no rows says *the tool ran and found
nothing*; a missing file says *the tool never ran*. So every write happens even when the shape is
empty. And a frame the tool did not report round-trips as ``None``, never ``0.0`` — parquet nulls
carry that faithfully, and the round-trip is tested for it, because imputing zero manufactures a
confident claim nobody made.
```

### F-32 (raised-by A-32) — `src/senselab/audio/workflows/audio_analysis/influence.py:1-30`

- destination (as filed): layered-architecture.md (D-21 rule 6)
- rationale: "pseudo-diarizer agreeing with itself is not corroboration" rule, duplicated in asr.py/speaker.py

Lines 1-30:
```python
"""Uncertainty-gated influence weighting (T080, FR-011b / FR-011c).

Signals in the adaptive loop influence one another iteratively toward convergence. Two
independent gates bound how far any one signal may move another:

    effective_weight = base_weight × uncertainty_gate × derivation_gate

**The uncertainty gate** shrinks a signal's influence as its own uncertainty rises, so a
signal that does not trust itself cannot propagate its error into signals that do. It is
floored rather than taken to zero: when stability is measured over few perturbation points
the measure is coarse — with two points, normalised entropy can only be 0 or 1 — and a hard
zero would erase a dissenting claim from the posterior entirely rather than down-weighting
it. A maximally-uncertain source is left visible and unable to win.

**The derivation gate** shrinks it further for signals whose labels are a by-product of
another signal already in the system. This is the subtler of the two. A clustering-derived
pseudo-diarizer agreeing with the embeddings it was computed from is *not* corroboration —
it is one computation counted twice, and treating it as two independent votes is how a
single derived signal comes to look like consensus. The gate is required to sit strictly
below the independent gate; a configuration that equalizes them defeats the guard and is
rejected rather than honored.

Deliberately pure and dependency-free so the guards can be tested without the loop, and so
they can be sequenced *before* the influence paths they protect (spec Dependencies).

Lives at the workflow level rather than under ``adaptive/`` because three consumers there need it —
``speaker_identity`` for its source weights, ``fuse`` for the cross-axis evidence-overlap gate, and
the adaptive loop itself. A shared piece imported *upward* out of a subsystem inverts the dependency
and makes the subsystem look like a library for the level above it; the same reasoning moved
non-convergence detection down to ``rounds.py``.
```

### F-33 (raised-by A-33) — `src/senselab/audio/workflows/audio_analysis/calibration.py:1-32`

- destination (as filed): layered-architecture.md or l1-post-processing-register.md
- rationale: declared-and-unread field history (`temperature`, `token_entropy_reference_nats`)

Lines 1-32:
```python
"""Scene-quality calibration profiles (US5, T036 — data-model.md §5).

A ``CalibrationProfile`` is a small versioned JSON mapping raw estimator outputs
(dB) onto the workflow's ``[0, 1]`` degradation scores, plus per-axis
temperatures for the uncertainty aggregators:

```json
{
  "version": "1",
  "snr":        {"type": "linear_db_to_unit", "clean_db": 25.0, "floor_db": 5.0},
  "reverb_c50": {"type": "linear_db_to_unit", "clean_db": 30.0, "floor_db": -5.0},
  "bandwidth":  {"nyquist_ref_hz": 8000.0, "rolloff_pct": 0.95},
  "temperature": {"speech_presence": 1.0, "asr": 1.0}
}
```

Profiles are fitted by ``scripts/calibrate_scene_quality.py`` from synthetic
mixtures (research.md D9). The dB anchors are consumed at runtime by ``quality.py`` /
``degradation.py`` (flat ``*_clean_db``/``*_floor_db`` keys); :func:`profile_to_runtime` bridges the
versioned on-disk shape to that flat runtime convention. Absent profile →
:data:`DEFAULT_PROFILE`, which mirrors the documented uncalibrated defaults in
``quality.py`` (bounded, not fitted).

**``temperature`` and ``token_entropy_reference_nats`` currently reach no fold.** Their only
consumers were ``aggregate.aggregate_asr`` and ``aggregate.aggregate_speech_presence``, which had no
production caller and are deleted; the run's single fold is ``fuse.fuse_axis``, which takes no
temperature. They stay in the schema, validated, because the *question* they answer is real — two
backends' confidences are not on a common scale — and dropping the fields would lose the fitted
values already on disk. But they are declared-and-unread until ``fuse_axis`` takes them, and
``axes.CALIBRATED_AXES`` names the axes that would receive them; see the note there.

Stdlib-only; safe to import anywhere.
```

### F-34 (raised-by A-34) — `src/senselab/audio/workflows/audio_analysis/degradation.py:1-18`

- destination (as filed): layered-architecture.md (L1/L2 calibration boundary)
- rationale: measured "clip((25-snr_db)/20,0,1) returns 0.0 in every bucket of every recording" L1/L2 boundary rationale

Lines 1-18:
```python
"""L2 scene-quality degradation: measurements to ``[0, 1]`` scores against calibrated anchors.

The anchors here are *calibration* — claims about what counts as clean for a task — which is why
they live at L2 and why a fitted profile may replace them. Holding them at L1 destroyed the
underlying measurements: ``clip((25 − snr_db)/20, 0, 1)`` returned ``0.0`` in every bucket of every
recording measured, because clean speech sits at 60–70 dB SNR against a 25 dB anchor. See
``specs/20260728-221507-per-speaker-identity-scene/layered-architecture.md``.

Two rules the functions here obey.

**A missing measurement stays missing.** ``None`` in gives ``None`` out, never ``0.0``. A degraded
score of zero is the confident claim "this audio is clean"; producing it from an estimator that
failed would manufacture confidence from an absence.

**Saturation is visible, not silent.** Every score reports whether it hit an anchor, so a column
pinned at an extreme can be recognised as anchor-limited rather than read as a measurement. That
distinction took a figure and six defects to notice the first time.
"""
```

### F-106 (raised-by A-111 (reclassified per reviewer instruction — was labeled restates-code in sweep-a-prose.md)) — `src/senselab/audio/workflows/audio_analysis/disagreements.py:23-29` (``_row_summary``)

- destination (as filed): layered-architecture.md (L1/L2 boundary register)
- rationale: states an L1/L2 separation principle (fused row = how much/which signals carried doubt; evidence = the L1 per-signal measurement for the same bucket), not a mechanical restatement of the two-dict concatenation below it

Lines 23-29:
```python
def _row_summary(row: Mapping[str, Any], axis: str, evidence: Mapping[str, Any] | None) -> str:
    """One-line human-readable explanation of why a bucket scored high.

    The fused row says *how much* doubt there is and which signals carried it; ``evidence`` is the
    L1 per-signal measurement for the same bucket, which says *what* they measured. Keeping them
    separate is the point — the summary reads a measurement, never a second fold.
    """
```

---

## belief/fusion design (6 entries)

### F-79 (raised-by A-79) — `src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:296-310`

- destination (as filed): belief/fusion design (consensus alignment)
- rationale: "hardcoded to MMS_FA, D-1 moved Canary off MMS" history

Lines 296-310:
```python
    would otherwise be a vote over member timings for a word order none of them emitted, which can
    come out non-monotonic. ``fusion.consensus_alignment: off`` keeps the member-vote timings.

    **Default backend is the Qwen forced aligner, the same one the pre-fusion path uses.** This was
    hard-coded to torchaudio MMS_FA with no way to choose, which left the pipeline running two
    aligners — Qwen3-ForcedAligner before fusion, MMS after — and D-1 moved Canary off MMS precisely
    so that word-boundary differences would "reflect the models, not two different aligners". A third
    aligner appearing after fusion reintroduced what that decision removed.

    The trade is real and worth knowing rather than discovering: Qwen's aligner already times
    Qwen3-ASR (bundled) and Canary (externally), so a Qwen-timed consensus shares its source with
    most members and the published boundary sits closer to theirs by construction. MMS is
    independent of every member but is a third opinion nobody asked for. Consistency won because the
    per-edge confidences measure spread *among members*, which either choice leaves untouched — what
    changes is only whether the published value is drawn from inside or outside that set.
```

### F-81 (raised-by A-81) — `src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:65-74`

- rationale: "flag said background_mask was harvested, method enumerated three axes" mismatch history

Lines 65-74:
```python
_HARVEST_ACCESSORS: Final[frozenset[str]] = frozenset(HARVEST_SOURCES)
"""Axes :meth:`VoteStore.from_harvests` can read straight off a ``PassHarvest``.

Derived from :data:`~.axes.HARVEST_SOURCES` — what a reader can actually *find* — and not from the
``harvested`` flag, which is only what the axis claims. The distinction is the bug: the flag said
``background_mask`` was harvested, this method enumerated three axes in a literal tuple, and
``frozenset(HARVESTED_AXES)`` then reported the mask as covered. So the guard below could not fire,
the caller's ``unharvested`` entry was accepted instead, and the axis was rebuilt from one vote per
mask *region* — 1070 buckets at round 0, one by round 4. Keyed on the declaration a reader
dereferences, an axis nothing can read is *not* in this set and the guard raises."""
```

### F-82 (raised-by A-82) — `src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:206-221`

- destination (as filed): belief/fusion design (SNR gating)
- rationale: measured SNR-gate scope gap (gate reached round 0 only, final/ published 0.2267 vs round 0's 0.0487)

Lines 206-221:
```python
    it is not a replay of the published axis — it is a second, differently-gated fold reported
    under the same name. That is not hypothetical: the gate reached round 0 only, and the loop's
    ungated re-aggregation folded the enhanced pass back in, so ``final/`` published 0.2267 on a
    recording whose round 0 read 0.0487. The axis appeared not to have changed at all.

    Both inputs come from the run's own artifacts rather than from a caller:

    - which perturbations are gated, from ``L1/perturbations.json`` — the register records each
      one's *declared transform*, which is exactly what ``SNR_GATED_TRANSFORMS`` is keyed on. Read
      from there rather than re-derived, so a standalone loop run on someone else's run directory
      cannot disagree with the fold that produced it.
    - identity-pass SNR per bucket, from ``L1/signals/scene_quality.parquet``.

    Returns ``None`` when the run has nothing to gate, which is the correct gate for a run whose
    only perturbation is the identity.
    """
```

### F-83 (raised-by A-83) — `src/senselab/audio/workflows/audio_analysis/adaptive/belief.py:1108-1125`

- destination (as filed): belief/fusion design (aleatoric floor)
- rationale: "every lookup missed, floor assigned 0.0 everywhere — the confident claim of no floor" incident

Lines 1108-1125:
```python
def _attach_floor(row: dict[str, Any], meta: Mapping[str, Any]) -> None:
    """Attach the aleatoric floor, derived from measurements under named anchors.

    The floor is the largest degradation the scene imposes here. It was previously read from
    ``meta["quality_snr"]`` and siblings — *scores*, which neither ingest path has ever carried:
    the harvest holds dB, and the fused presence parquet holds neither. Every lookup missed, the
    floor was assigned ``0.0`` on every bucket of every run, and ``0.0`` is the confident claim
    "this audio imposes no floor" — so the ``snr_floor`` irreducibility verdict could not fire and
    a run could only ever report ``no_reduction_under_available_interventions``.

    So the scores are derived here, from the dB the store does carry, against
    :data:`degradation.DEFAULT_ANCHORS`. An anchored score is an L2 decision, so the anchors and
    the terms that survived travel on the row.

    **Absent is not zero.** With no measurement the floor is ``None``, and a ``None`` floor cannot
    explain a residual — which is the difference between "nothing constrains this bucket" and
    "nobody measured whether anything does".
    """
```

### F-86 (raised-by A-86) — `src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:35-38`

- destination (as filed): belief/fusion design (transcript fusion)
- rationale: "dropped every word overlapping a P3-adjudicated span" history

Lines 35-38:
```python
    This function removes nothing. It used to drop every word of a model overlapping a span P3 had
    adjudicated, which left no record anywhere downstream — and made a word's survival depend on
    whether the intervention had been admitted within budget. Doubt about a word is now carried as
    a measured weight on the word itself (``adaptive.corroboration.apply_corroboration``).
```

### F-96 (raised-by A-96) — `src/senselab/audio/workflows/audio_analysis/adaptive/ls_final.py:217-225`

- rationale: "three answers from one file, only one written down" history

Lines 217-225:
```python
def _final_belief_index(out_dir: Path) -> dict[tuple[str, float, float], dict[str, Any]]:
    """Last round's belief rows indexed by ``(axis, start, end)``.

    Not by stream, and no longer collapsed here: the belief file now holds one row per bucket,
    folded across passes by the writer under a policy it records. This function used to apply its
    own most-doubtful collapse, ``adaptive.plot`` filtered to the fusion stream, and ``evaluate``
    filtered to the transcript's — three answers from one file, only one of which was written
    down. The fold moved to the writer so there is one.
    """
```

---

## policy/triage design (6 entries)

### F-78 (raised-by A-78) — `src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:200-213`

- destination (as filed): policy/triage design (P2 rationale)
- rationale: "what is lost" reduction disclosure for segmentation-3.0

Lines 200-213:
```python
    """Continuous per-frame speech probability over ``span``, from Brouhaha's VAD head.

    P2's purpose is **localisation**: it fires when a region's votes are dominated by coarse voters,
    each casting one identical vote across every bucket it spans, so agreement among them is an
    artifact of window size rather than evidence about the bucket. Re-measuring at frame resolution
    on the crop is the answer.

    Brouhaha rather than ``segmentation-3.0``, whose per-speaker channels nothing uses any more (D-19):
    its VAD head runs at **the same 16.9 ms hop**, so nothing is lost at the one thing P2 exists for.

    **What is lost, stated because it is a real reduction.** This is the same model that already voted
    in round 0, so P2 now buys locality — the same estimator on a crop, which is a genuine
    re-measurement because a model given a short span sees different context — but not a second
    independent opinion. Under ``segmentation-3.0`` it bought both.
```

### F-91 (raised-by A-91) — `src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:266-273`

- destination (as filed): policy/triage design (S1 stream election)
- rationale: "used to concatenate per-bucket text, forcing the axis onto a 1.0s grid" history

Lines 266-273:
```python
def _claims_words(ctx: dict[str, Any], stream: str, region: dict[str, Any]) -> bool:
    """Did any recognizer place a word inside this region, on this stream?

    Asked of the *votes*, not of a transcript: the asr axis emits a recognizer's entry only where one
    of its words actually reaches a bucket, so the vote's presence is the claim. This used to concatenate
    each model's per-bucket ``text`` and test the string for emptiness — the same question, answered
    through a reconstruction of the transcript that existed for no other reader, and that forced the
    axis onto a 1.0 s grid so a whole word could fit inside a bucket.
```

### F-92 (raised-by A-92) — `src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:582-591`

- destination (as filed): policy/triage design (U1/U2 escalation)
- rationale: "environment without g2p_en silently measured a different quantity under the same column name" history

Lines 582-591:
```python
    One path, and no g2p gate. The gate used to route to ``_harvest_word_level``, a second
    harvester emitting pairwise *word*-Levenshtein distances in the same vote schema — so an
    environment without ``g2p_en`` produced an axis measuring a different quantity under the same
    column name, recorded only as ``pair_distance_kind``. The axis now has one voter, the consensus
    word fold, and its word-similarity grading degrades to exact match on its own when g2p is
    absent (``asr.phoneme_similarity``) rather than by switching harvesters.

    The grid comes from the axis rows the loop is already holding, so an escalation cannot land its
    votes on buckets the belief store has no keys for.
    """
```

### F-99 (raised-by A-99) — `src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:20-25`

- rationale: "configuration spread across a file and seventy flags" history

Lines 20-25:
```python
    There is no separate policy file. It lived at ``adaptive/policy/default.yaml`` beside a CLI that
    also carried model ids, grids and stage switches as flags, so a run's configuration was spread
    across a file and seventy arguments and only the file part had an identity. It is now one section
    of ``data/run_config/default.yaml``, and ``path`` is a **run config**, not a bare policy: a file
    whose policy keys sit at the top level would deep-merge into keys nothing reads, which is a
    silent no-op, so its absence is raised rather than tolerated.
```

### F-100 (raised-by A-100) — `src/senselab/audio/workflows/audio_analysis/adaptive/policy.py:74-79`

- rationale: "a floor configurable to zero is not a floor" rationale for raising rather than clamping

Lines 74-79:
```python
    """Reject a policy that sets a weight floor to zero.

    Aggregation drops a voter whose weight reaches zero, and word fusion drops a word whose vote
    weight and coverage contribution both vanish. A zero floor therefore restores erasure through
    configuration — silently, and everywhere at once. A floor that can be configured to zero is
    not a floor, which is why this raises rather than clamping.
```

### F-103 (raised-by A-103) — `src/senselab/audio/workflows/audio_analysis/adaptive/triage.py:1-13`

- rationale: "continuous frame posteriors, never segmentized VAD" design rationale

Lines 1-13:
```python
"""Triage round 0: cheap-signal gating decisions (spec US1, FR-002/003/004).

Design follows ``SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md``: the speech gate is
driven by **continuous frame posteriors** (pyannote ``segmentation-3.0`` raw
scores — never segmentized VAD, whose hysteresis erases brief events),
aggregated on a ~100 ms window (the shortest span where "is this speech?" is
well-posed). Coarse signals (scene taggers, sentence-level ASR) do not vote
here at all. SNR comes from Brouhaha when available, with a percentile DSP
estimator as the ungated fallback.

This module is pure (numpy only): the decision function consumes arrays and a
threshold set, so it is unit-testable and reusable by both
``scripts/analyze_audio.py`` (production round 0) and ad-hoc analyses.
```

---

## adaptive loop design (5 entries)

### F-75 (raised-by A-75) — `src/senselab/audio/workflows/audio_analysis/adaptive/__init__.py:5-9`

- destination (as filed): adaptive loop design (import/dependency strategy)
- rationale: lazy-import-strategy rationale (no torch/model backends at module level)

Lines 5-9:
```python
This subpackage keeps imports light on purpose: no torch / model backends are
imported at module level, so the loop's pure core (belief store, region
proposal, policy engine, fusion, evaluation) runs in minimal environments.
Interventions that need live model backends import them lazily inside their
``execute`` functions and degrade to ``blocked_guard`` when unavailable.
```

### F-76 (raised-by A-76) — `src/senselab/audio/workflows/audio_analysis/adaptive/audio_io.py:110-121`

- destination (as filed): adaptive loop design (audio_io/perturbation dispatch)
- rationale: "used to hardcode two pass names" history

Lines 110-121:
```python
def get_stream_wav(ctx: dict[str, Any], stream: str) -> tuple[Any | None, str | None]:  # noqa: ANN401
    """Waveform for one perturbation: the identity loads the input file, any other is regenerated.

    Dispatch is on the **declared transform** from ``L1/perturbations.json`` (``ctx["perturbations"]``),
    not on the perturbation's name. It used to be a two-armed comparison against the two pass
    names of the day (``perturbations.py`` records which), so a third perturbation was an edit
    here — in a module that has no business knowing any perturbation's name.

    Results are cached in ``ctx["_wav_cache"]``. When the backend for a transform is unavailable the
    caller decides whether the identity is an acceptable fallback (recorded as ``stream_fallback``
    in the intervention log).
    """
```

### F-77 (raised-by A-77) — `src/senselab/audio/workflows/audio_analysis/adaptive/backends.py:191-193`

- destination (as filed): adaptive loop design (model loading/caching)
- rationale: stage-once/load-from-local-snapshot rationale to avoid per-file Hub HEAD (429 source under batch)

Lines 191-193:
```python

# ── I4: overlap posteriors ───────────────────────────────────────────────

```

### F-90 (raised-by A-90) — `src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:54-62`

- destination (as filed): adaptive loop design (artifact access)
- rationale: "returned {} on every run, silently, once outputs moved under L1/" history

Lines 54-62:
```python
    The path comes from :func:`~senselab.audio.workflows.audio_analysis.layout.perturbation_dir` rather
    than being rebuilt here. It was rebuilt as ``run_dir / stream / task_dir`` until the pass
    outputs moved under ``L1/``, at which point this returned ``{}`` on every run — silently, so
    the ASR fusion path received nothing and emitted an empty transcript with no error anywhere.

    A missing directory now warns. An empty result is a legitimate answer only when the stage did
    not run; when the directory itself is absent the caller is asking about a layout that does not
    exist, and returning ``{}`` makes those two indistinguishable — which is precisely how the
    drift above survived to the point of producing a transcript with no words.
```

### F-94 (raised-by A-94) — `src/senselab/audio/workflows/audio_analysis/adaptive/loop.py:583-593`

- rationale: "root inferred from path shape, fragile, works only for the default layout" caveat

Lines 583-593:
```python
def _resolve_input_audio(recorded: str | None, run_dir: Path) -> str | None:
    """Resolve the run's input audio path, re-rooting when the run came from another machine.

    Tries the recorded absolute path first, then the last one/two path components
    relative to the repo root inferred from ``run_dir`` (…/artifacts/analyze_audio/<run>
    → repo). Returns None when nothing exists — audio-dependent rules then guard.

    The root is inferred from the path *shape*, which is fragile: it walks a fixed number of
    parents rather than looking for a marker. It happens to work for the default output layout
    and would not for an arbitrary ``--out-dir``.
    """
```

---

## asr axis design (4 entries)

### F-16 (raised-by A-16) — `src/senselab/audio/workflows/audio_analysis/asr.py:1-27`

- rationale: history of four removed per-bucket ASR quantities

Lines 1-27:
```python
"""Utterance axis vote harvester — "what was said?".

**One voter, one question.** The axis is a resampling of fused word accuracy onto the shared time
grid: the recognizers' words are folded once per pass (``_consensus_word_doubt``), and each bucket
takes the coverage-weighted mean of ``1 - existence_confidence`` over the words reaching it
(``resample_word_doubt``).

Four things used to ride on every bucket beside it, and all four are gone:

- **per-bucket text** (``asr_text_in_window`` with ``fully_contained=True``), a reconstruction of
  what ``final/transcript.json`` already holds at word resolution. It is also what forced the
  1.0 s / 0.5 s grid: with a bucket narrower than a word, a fully-contained read returns nothing,
  so the grid had to be widened and overlapped until words fit. With the derivative as the voter
  that reason is gone, and the axis sits on ``axes.DEFAULT_TIME_GRID`` like the other three.
- **the pairwise phoneme distance** between recognizers, which was already recorded rather than
  scored (D-21 rule 6: its source closure is a subset of the consensus fold's, so counting both
  counts one body of evidence twice). Recorded-and-never-read is not a middle ground; the
  readable form of "which pair diverged" is the transcript's own ``alternates``.
- **``avg_logprob`` / ``token_entropy`` / ``alignment_ctc_score``**, three per-bucket reads that
  no longer reach a fold. The first two are a model's private doubt about a *transcript*, which
  the consensus fold already weighs per word; the third measures an aligner's path posterior given
  a possibly-hallucinated transcript, which was never scored for exactly that reason.

The word-level fields — ``existence_confidence``, ``onset_confidence``, ``offset_confidence`` —
are where localisation and per-edge doubt live. See :func:`resample_word_doubt` for why they are
deliberately not folded into this axis's number.
"""
```

### F-17 (raised-by A-17) — `src/senselab/audio/workflows/audio_analysis/asr.py:100-115` (``phoneme_similarity``)

- destination (as filed): asr axis design (grading/g2p)
- rationale: g2p-fallback rationale ("letters are not sounds")

Lines 100-115:
```python
def phoneme_similarity(a: str, b: str) -> float:
    """How close two words sound, in ``[0, 1]`` — 1.0 identical, 0.0 sharing no phoneme.

    Supplied to the ensemble so word accuracy grades its disagreements instead of counting exact
    matches. The task API stays stdlib-only and receives this as a callable, the same way it
    receives ``calibrator`` and ``speaker_at``: g2p is a workflow dependency and does not belong
    inside a model-independent voting routine.

    ARPAbet with stress markers stripped, so ``AH0`` and ``AH1`` are one phoneme — stress is not a
    lexical difference and counting it would penalise two recognizers that agree on the word.

    Falls back to **exact match** when g2p is unavailable or produces nothing for either side, not
    to grapheme overlap: letters are not sounds, and substituting one measure for the other would
    change the number's meaning invisibly. A homophone pair therefore scores 1.0 where g2p works
    and 0.0 where it does not, which is a real limitation and better than an unrecorded proxy.
    """
```

### F-18 (raised-by A-18) — `src/senselab/audio/workflows/audio_analysis/asr.py:293-320` (``resample_member_doubt``)

- destination (as filed): asr axis design (consolidate with F-16)
- rationale: restates the module docstring's epistemic-uncertainty-was-zero framing

Lines 293-320:
```python
def resample_member_doubt(
    words: Sequence[Mapping[str, Any]],
    buckets: Sequence[tuple[float, float]],
) -> dict[str, dict[tuple[float, float], float | None]]:
    """One series per recognizer: its own doubt about the word sequence, on the axis grid.

    **The recognizers are sources, so the axis aggregates them — not their mean.** This replaces the
    single ``consensus_words`` series. That series was ``1 - existence_confidence``, and
    ``existence_confidence`` contains ``share``, the *weighted mean* of the recognizers' agreement
    with the winning text. Handing an axis a mean has two costs, both measured:

    - ``epistemic_uncertainty`` was structurally ``0.0`` for this axis on every run — not because the
      recognizers agreed, but because the one number reaching the fold had no spread left in it. The
      cross-source disagreement that term exists to measure had been averaged away one layer earlier.
    - ``reliability.signal_stability`` weighted the fused series, so a recognizer whose answer flips
      between perturbations could not be discounted individually.

    Per recognizer and word, doubt is ``1 - agreement × member_confidence``, where ``agreement`` is
    that recognizer's agreement with the winning text (1.0 exact, phoneme similarity otherwise) and
    ``member_confidence`` is its own reported confidence *when it reports one* — absent is absent, so
    the term drops out rather than reading as certainty.

    **The old ``coverage`` term becomes structural.** It was ``min(1, coverage_mass / ensemble_weight)``
    — how much of the ensemble produced anything in this slot — folded in as a multiplier. Per
    recognizer that question needs no term: a recognizer that produced no word in a slot simply has no
    reading there, and ``fuse.per_signal_uncertainty`` drops an absent signal rather than zero-filling
    it. Absence is the measurement.

```

### F-19 (raised-by A-19) — `src/senselab/audio/workflows/audio_analysis/asr.py:426-456` (``harvest_asr_votes``)

- destination (as filed): asr axis design (consolidate with F-16/F-18)
- rationale: third copy of the same consensus_words history in one file

Lines 426-456:
```python
def harvest_asr_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    alignment_by_model: dict[str, Any],
    fused: tuple[list[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes"}`` per bucket for the asr axis.

    ``votes`` holds **one entry per recognizer**, keyed by model id: ``{"value": doubt, **provenance}``,
    where ``doubt`` is that recognizer's disagreement with the fused word sequence, resampled onto this
    bucket (:func:`resample_member_doubt`). A bucket no word reaches carries **no vote at all** rather
    than ``0.0`` — nothing was said there, which is not the same as nothing being in doubt, and
    zero-filling would manufacture confidence (FR-007).

    This used to be a single ``consensus_words`` entry: ``1 - existence_confidence``, whose ``share``
    term is the *weighted mean* of the recognizers' agreement. Handing an axis a mean made
    ``epistemic_uncertainty`` structurally ``0.0`` here on every run — not because the recognizers
    agreed but because the spread had been averaged away before the fold that measures spread saw it —
    and it meant ``signal_stability`` weighted the fused series rather than the recognizers.

    This is **not** the per-model entry that was removed for double-counting (D-21 rule 6). That one
    was each recognizer's *independent* per-bucket reading (``avg_logprob``, ``token_entropy``,
    ``alignment_ctc_score``) sitting beside the fold. These are the fold's own decomposition: their
    weighted mean is the ``share`` the single entry carried, so the evidence is counted once, at the
    resolution where the recognizers were actually compared.

    ``fused`` accepts a consensus fold the caller already performed — ``compute.harvest_pass`` shares
    one with the speaker axis, which reads the same words' ``temporal_confidence``. Omitted, the
    harvest folds them itself, which is what a standalone caller wants.
    """
```

---

## speech-presence design (4 entries)

### F-47 (raised-by A-47) — `src/senselab/audio/workflows/audio_analysis/sound_sources.py:90-106` (``window_label_mass``)

- destination (as filed): speech-presence design (label mass vs top-1), duplicated verbatim in speech_presence.py
- rationale: top-1-discards-evidence example (Music 0.40 / Speech 0.38)

Lines 90-106:
```python
def window_label_mass(window: Any, labels_of_interest: set[str]) -> Optional[float]:  # noqa: ANN401
    """Share of one classification window's score mass falling on a subset of labels.

    Args:
        window: One ``classify_audios`` window dict with ``labels`` and ``scores``.
        labels_of_interest: The label subset to sum, e.g. the speech-related AudioSet classes.

    Returns:
        The subset's share of total mass in ``[0, 1]``, or ``None`` when the window carried no
        scores.

    Replaces ``top-1 label in subset``, which was a threshold disguised as a lookup. AST and
    YAMNet emit several hundred class scores; taking the argmax and asking whether it happens to
    be a speech label throws away everything else. A window whose top label is ``Music`` at 0.40
    with ``Speech`` second at 0.38 voted a confident *no speech* — discarding 0.38 of speech
    evidence. Mass over the subset keeps it, and which labels count as speech stays a task
    parameter rather than being folded into a boolean.
```

### F-48 (raised-by A-48) — `src/senselab/audio/workflows/audio_analysis/speech_presence.py:1-41`

- destination (as filed): speech-presence design (L1 evidence), consolidate with F-47
- rationale: same Music/Speech top-1 example as F-47

Lines 1-41:
```python
"""L1 speech-presence evidence — what each tool measured, in its own units.

There is no presence at L1, only signals. This module runs the readers that project each model's
output onto a reporting bucket and records the numbers; whether a bucket contains speech is decided
in :mod:`speech_presence_link`, under a named policy. Nothing here thresholds, inverts, ranks, or
selects among estimators.

What each signal contributes, and what the measurement is:

- **Diarization models** — ``covered_fraction`` (union of segment overlap with the bucket, as a
  proportion) and ``speaker_label``. Replaces a ``speaks`` bool that could not distinguish a
  segment grazing 5% of a bucket from one covering all of it.
- **ASR models** — ``word_overlap_s`` and ``n_words`` (how much transcript actually lands here),
  the per-chunk ``avg_logprobs`` and ``no_speech_probs`` unpooled, and the unclipped
  ``claim_span_s`` / ``segment_span_s`` so how *wide* the claim is can be measured rather than
  declared.
- **Whisper's silence head** — ``no_speech_prob`` as a sibling signal keyed
  ``<asr_model>::no_speech_prob``, uninverted.
- **AST / YAMNet** — ``speech_label_mass``, the share of class-score mass on the speech label set.
  Not ``top-1 ∈ labels``: the argmax discards several hundred scores, so a window topped by
  ``Music`` at 0.40 with ``Speech`` second at 0.38 used to read as a confident *no speech*.
- **openSMILE HNR** — ``hnr_db``, a ratio in dB and therefore already absolutely calibrated.
- **LUFS** (BS.1770 gated loudness) — ``lufs``, an absolute level, so the same loudness always
  reads the same.
- **Level above floor** — ``excess_db`` above this recording's own measured noise floor, and
  therefore gain-invariant: the question LUFS cannot answer. Together these two replace a single
  per-pass percentile band that answered neither (D-3), since a rank cannot be compared to a fixed
  threshold or across files.
  over the bucket's frames, plus its dispersion and frame count. Not a count of frames whose
  argmax is not ``<silent>``: that collapses each frame's distribution to a hard verdict, the same
  reduction the scene-classifier top-1 made.
- **Frame posteriors** (``segmentation-3.0`` raw scores, Brouhaha's VAD head) — ``frame_mean``,
  ``frame_std``, ``n_frames``, and the per-speaker ``channel_means`` / ``channel_labels`` kept
  intact (D-5), plus the declared ``resolution_s`` and ``native_window_s``.

Windowed speaker embeddings are **not** read here. Clustering them is a derived signal per D-7 —
it needs the whole pass, and its output (per-window silhouette and cluster label) is a conclusion
about speaker structure rather than a measurement of this bucket. The vectors travel on
``PassHarvest.per_window_embeddings`` and ``speech_presence_link.derive_window_clusters`` clusters
them at L2.
"""
```

### F-49 (raised-by A-49) — `src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:1-42`

- destination (as filed): speech-presence design (L1/L2 split)
- rationale: Jensen's-inequality argument for why two ASR confidence statistics differ

Lines 1-42:
```python
"""L2 link layer for the speech-presence axis: measurements → beliefs.

L1 (``speech_presence.py``) reports what each tool measured in that tool's own units — segment
coverage as a fraction, word spans in seconds, Whisper's per-chunk log-probabilities, dB above the
measured noise floor, frame posteriors and their per-speaker channels. Nothing there decides
whether a bucket contains speech.

This module decides. Every threshold, inversion, and pooling that used to sit inside the harvester
lives in :class:`SpeechPresencePolicy`, so each is named, replaceable, and recorded with the run
rather than compiled into the measurement.

Why the split is not cosmetic — three properties it buys that the single layer could not have:

**A verdict can be revisited without re-running a model.** A diarization segment grazing 5% of a
bucket and one covering all of it both set ``speaks=True``. Once the bool was the only survivor,
nothing downstream could tell them apart, and the difference matters most at segment boundaries —
exactly where speaker uncertainty peaks.

**A pooling choice becomes visible.** Whisper's bucket confidence was ``mean(exp(avg_logprob))``.
By Jensen's inequality that strictly exceeds ``exp(mean(avg_logprob))`` whenever the chunks
disagree, so the two are different statistics and one of them had been picked silently. L1 now
emits the per-chunk list and :attr:`SpeechPresencePolicy.asr_confidence_pooling` names the choice.

**"Coarse" stops being a property of a voter.** The old harvester hand-marked AST, YAMNet and the
Whisper segment voters ``coarse: True`` and applied a fixed 0.25 weight below a 0.5 s grid. But a
voter is only coarse *relative to the grid it is reported on*: AST's 10.24 s window is stretched
across 100 buckets at 0.1 s and across none at 10.24 s. That comparison needs both numbers, so it
can only be made here.

Two asymmetries are deliberate and are preserved verbatim from the harvester, because they encode
measured limits of the signals rather than tuning:

- **Level-above-floor abstains at low excess.** The floor is a percentile of this file's own
  frames, so a source that never stops *is* the floor. A low excess is therefore ambiguous between
  "nothing is happening" and "something is happening continuously", and voting absence there made
  the signal contradict correct models on any recording without pauses.
- **HNR abstains at low values.** Whispered and distorted voice both read low, so a low HNR cannot
  distinguish them from silence.

In both cases the signal maps its uninformative end to ``0.5`` rather than to a denial. LUFS keeps
the ability to claim absence, because −90 LUFS is unambiguous on an absolute scale.
"""
```

### F-50 (raised-by A-50) — `src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:144-176` (``_abstaining_ramp``)

- destination (as filed): speech-presence design (signal abstention)
- rationale: measured acoustic_hnr abstention behavior (mean 0.2675 doubt vs 0.0 elsewhere)

Lines 144-176:
```python
def _abstaining_ramp(value: float, low: float, high: float) -> float | None:
    """Linear ramp into ``(0.5, 1]``, or ``None`` at or below ``low`` — where it cannot tell.

    Used where a low reading has two indistinguishable causes (see the module docstring on HNR and
    level-above-floor). Mapping that end to ``0.0`` would let the signal contradict correct models on
    inputs where it simply cannot tell.

    **An abstention is the absence of a claim, so it returns ``None``.** It used to return ``0.5``,
    which ``_directed`` then turned into ``speaks=True`` at confidence ``0.5`` — a half-confident
    *yes* cast in exactly the region where the signal has no opinion, and read by the fold as ``0.5``
    of doubt: the largest contribution a single voter can make. So the buckets where HNR knew least
    were the buckets where it pushed hardest.

    ``link_speech_presence`` has always had the correct rule for this, twenty lines below and stated
    in its own words: *"The signal reported nothing usable in this bucket. Dropping the vote is right:
    a fabricated 0.5 would be indistinguishable from a real abstention."* That is precisely what this
    function was manufacturing.

    Measured on a clean two-speaker conversation, ``acoustic_hnr`` contributed mean 0.2675 doubt
    (median 0.2574, max 0.5000) while all four diarizers, all three recognizers and the brouhaha VAD
    read exactly 0.0000. It is genuine voicing evidence where HNR is high — vowels are periodic —
    which is why the signal stays; what goes is its vote in the range where a low reading means
    "voiceless" and "absent" equally well.

    The graded region is untouched: only ``value <= low`` abstains, so a reading part-way up the ramp
    still votes at the strength the ramp gives it. This is deweighting by *scope* rather than by a
    hand-set discount — the kind ``fuse.cross_axis_inputs`` documents as inadmissible, since a factor
    never measured must not act as a discount.
    """
    ramped = _ramp(value, low, high)
    if ramped <= 0.0:
        return None
    return 0.5 + 0.5 * ramped
```

---

## grid/fuse design (3 entries)

### F-9 (raised-by A-9) — `src/senselab/audio/workflows/audio_analysis/grid.py:15-19`

- destination (as filed): grid/fuse design (doc.md "one grid" section)
- rationale: measured evidence (242/242/19/8 rows, zero shared bucket keys across four resolutions) for why one shared grid replaced four independent ones

Lines 15-19:
```python
    Defaults to :data:`~.axes.DEFAULT_TIME_GRID`, and that is the point rather than a convenience:
    **every axis is on this grid**, so row *i* of one axis is row *i* of another and a cross-axis
    join needs no reconciliation. Measured before it was: the four axes carried 242 / 242 / 19 / 8
    rows on 0.1/0.02, 0.1/0.02, 0.25/0.25 and 1.0/0.5 respectively, shared zero bucket keys, and
    the coupling between them therefore did nothing while reporting that it had run.
```

### F-10 (raised-by A-10) — `src/senselab/audio/workflows/audio_analysis/fuse.py:88-101` (``is_direction_only_claim``)

- destination (as filed): grid/fuse design (vote-folding)
- rationale: measured cost of the vote-folding fix (presence axis fusing 12 vs 8 signals depending on ASR set)

Lines 88-101:
```python
      turbo, Canary-Qwen and Qwen3-ASR all expose ``avg_logprob``/``no_speech_prob`` as ``None``,
      so word coverage is the whole of what they said;
    - the adaptive loop's **missed-speech adjudicator**, whose claim is that two model families
      agree words are here.

    Every other reader of a presence vote already takes such a vote at full strength — see
    ``aggregate.per_source_voice`` and ``support.presence_probability``, both of which map it to
    ``p = 1.0``/``0.0``. :func:`per_signal_uncertainty` did not, and dropped it instead. The cost
    was measured on a real run: with ``--asr-models openai/whisper-*`` the presence axis fused 12
    signals, and on the *shipped* default ASR set only 8 — all three ASR models and both
    diarizers had silently left the axis, because Whisper is the only backend whose per-segment
    ``avg_logprob`` gave the fold a number to read. ``reliability._bucket_beliefs`` had already
    had to reintroduce these voters by hand to measure their stability, so a weight was being
    computed for signals the fold could never use.
```

### F-15 (raised-by A-15) — `src/senselab/audio/workflows/audio_analysis/resolution.py:1-24`

- destination (as filed): grid/fuse design (per-signal resolution)
- rationale: measured VAD saturation from collapsing 17ms frames onto 250ms buckets

Lines 1-24:
```python
"""Per-signal temporal resolution, declared at L1 and converted at L2.

Forcing every signal onto one bucket grid loses information in both directions, and the losses
were both measured on real runs:

- A frame posterior at ~17 ms collapsed onto 250 ms buckets **saturates**. The VAD trace came
  out flat at 1.0 across a conversation with four clear pauses, because a bucket containing one
  speech frame was reported as fully active.
- An AST decision spanning 10.24 s spread across those same buckets **claims precision it does
  not have**, which is why its scene composition row was nearly constant: three real decisions
  stretched over eighty-odd buckets.

So L1 declares its resolution and L2 converts. The declaration travels with the signal because
a resolution inferred at fusion time is a guess about what the harvester did — and the two
failures above are exactly what that guess gets wrong.

Conversion direction matters:

- **Coarser → finer is a hold.** A 10 s decision applies across its whole window; interpolating
  between windows would invent detail the model never produced.
- **Finer → coarser is an integral.** Point-sampling a 17 ms posterior at 250 ms discards
  fourteen of every fifteen measurements and which one survives is arbitrary; averaging keeps
  what they collectively said, and is what stops the saturation above.
"""
```

---

## l1-post-processing-register.md (3 entries)

### F-24 (raised-by A-24) — `src/senselab/audio/workflows/audio_analysis/embeddings.py:396-407`

- destination (as filed): l1-post-processing-register.md item 12, but the comment has no anchor once the dead code it sits on is deleted for real
- rationale: comment on deleted p_voice computation ("a silhouette coefficient is not a probability")

Lines 396-407:
```python
# A per-window map computed as ``0.5 * (silhouette + 1)`` used to live here, rescaling a
# clustering-geometry index into a value that reads as a probability. A silhouette
# coefficient is a property of a chosen partition on a chosen metric, not a probability:
# silhouette computed with cosine and with Euclidean return different numbers for identical
# geometry on unit vectors, so any probability read off it is a probability about a
# parameterisation choice. The L1 post-processing register (item 12) removed the consumer —
# the presence voter that read it as confidence with no ramp and no anchors, contributing a
# near-constant ~0.44 doubt across every bucket while earning the highest fusion weight of
# fifteen signals precisely because it was near-constant (see ``speech_presence_link.py``'s
# comment on ``_silhouette_votes_by_bucket``). Nothing else reads the per-window value, so
# this removes the computation rather than leaving a renamed field with no reader.

```

### F-51 (raised-by A-51) — `src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:327-349`

- destination (as filed): l1-post-processing-register.md item 10
- rationale: removed `_link_hnr` banner, measured 8.12 dB median HNR below the "confidently voiced" anchor

Lines 327-349:
```python
# ``_link_hnr`` lived here. **HNR is voicing evidence, but its ramp was never fitted, so on ordinary
# speech it read as a floor rather than as a measurement.**
#
# The ramp was a code literal — 2 dB to 10 dB — and the register called it "fixed" for that reason.
# Measured on `english_conversation_higgs_audio_v2`, median HNR is **8.12 dB**: *below* the anchor that
# means "confidently voiced", so ordinary conversational speech read as only partly voiced. 102 of its
# 145 votes fell in the graded region, and it ended up the **largest contributor** on the presence axis
# (mean doubt 0.1568, against `ast` 0.1094 and everything else at or below 0.05) while all four
# diarizers, all three recognizers and the brouhaha VAD read exactly 0.0000. Removing it takes presence
# doubt from 0.0250 to 0.0160, and buckets below 0.01 from 66 of 214 to 112.
#
# Making the abstention silent (see :func:`_abstaining_ramp`) was a real fix and not this one: it cut
# HNR's mean from 0.2675 to 0.1568 by removing the fabricated half-confident votes, but the remaining
# doubt is the graded region, which is the anchors' fault and not the abstention's.
#
# **The measurement is not lost.** ``harvest_speech_presence_evidence`` still emits ``hnr_db`` in
# decibels and ``votes._signal_rows_from_buckets`` records it from the *evidence*, not from the votes —
# so ``L1/signals/acoustic_hnr.parquet`` is unchanged and a consumer wanting voicing evidence reads the
# dB directly. What is gone is the unfitted dB→probability step in between.
#
# Reinstating it as a voter means fitting the anchors to measured HNR on known-voiced speech and
# writing them into ``data/`` with their derivation, the way ``detection_margin`` and the scene-quality
# profile are. Recorded as the open item in ``l1-post-processing-register.md`` item 10.
```

### F-52 (raised-by A-52) — `src/senselab/audio/workflows/audio_analysis/speech_presence_link.py:415-445`

- destination (as filed): l1-post-processing-register.md item 12
- rationale: removed `_silhouette_votes_by_bucket` banner, measured weight 1.0 (highest of 15) on the least informative voter

Lines 415-445:
```python
# ``_silhouette_votes_by_bucket`` lived here. **A cluster silhouette is not presence evidence.**
#
# It answered "does a coherent speaker sit here" by reading the silhouette coefficient as a
# confidence — ``_directed(score)`` with no ramp and no anchors, unlike every linker above it
# (``_link_hnr`` ramps between ``policy.hnr_low_db`` and ``hnr_high_db``, ``_link_level_above_floor``
# likewise). Three things were wrong with it, all measured on a clean two-speaker conversation:
#
# - **It answers a different question.** Silhouette measures cluster *geometry* — separation — over
#   every window including silent ones. Silence embeds consistently too, so well-separated silence
#   scores well. It cannot distinguish "a coherent speaker is here" from "this window is coherently
#   not speech".
# - **It carried almost no information.** 214 buckets, doubt spanning 0.4022-0.4996, stdev 0.0227 —
#   a constant to within ±0.05. An ordinary good silhouette of 0.58 became 0.42 of standing doubt.
# - **And the weighting rewarded it for that.** It held weight **1.0**, the highest of all fifteen
#   presence signals, while every informative voter sat at 0.78-0.91: ``reliability.signal_stability``
#   measures cross-pass ``|delta|``, and a near-constant is perfectly stable. The least informative
#   voter earned the most weight.
#
# Together those meant **no bucket could reach zero presence doubt** however unanimous the evidence:
# all four diarizers, all three recognizers and the brouhaha VAD read exactly 0.0000 while the axis
# reported 0.0682. Without this voter it reads 0.0385, and 47 of 214 buckets can reach zero.
#
# Nothing is lost. The clustering this scored already reaches the **speaker** axis as a first-class
# diarization source per D-20 — ``compute.harvest_pass`` injects a synthetic
# ``embedding_silhouette/<model>`` diarizer built from :func:`derive_window_clusters`, whose spans and
# cluster ids feed ``attribution.speaker_assignment_doubt`` directly. Asking the same clustering to
# also vote on presence counted one body of evidence twice, on the axis where it was least apt.
# The vote's ``cluster_id`` had no consumer: label reassignment reads the synthetic diarizer's spans.
#
# ``derive_window_clusters`` below stays — it is what ``compute.harvest_pass`` calls.
# Register: ``l1-post-processing-register.md`` item 12.
```

---

## plotting design (3 entries)

### F-70 (raised-by A-70) — `src/senselab/audio/workflows/audio_analysis/l1_plot.py:1-20`

- destination (as filed): plotting design (L1 evidence figure)
- rationale: "diarizer stopped here / level fell here, neither alone tells the story" rationale

Lines 1-20:
```python
"""``L1/signals.png`` — the evidence plot: every signal, plus level, and no conclusions.

L1 is evidence, so its figure shows what each signal reported and how loud the audio was while
it reported it. That pairing explains most disagreements: "the diarizer stopped here" next to
"the level fell to -60 dBFS here" is usually the whole story, and neither row says it alone.

Level is plotted in **dBFS** rather than raw RMS because a level track is read against full
scale — 0 dBFS is the anchor a reader already has — and amplitude-referenced, so halving the
amplitude reads as -6 dB rather than -3.

Two deliberate omissions:

**No uncertainty rows.** Those are level-2 conclusions drawn *from* this evidence. A figure
that mixes the two invites reading a conclusion as another observation, which is how a derived
signal came to be treated as a peer in the first place.

**No signal is dropped.** A model that ran and reported nothing still gets a row, because
otherwise its silence is indistinguishable from its absence — and "this model reported nothing
here" is frequently the informative part.
"""
```

### F-71 (raised-by A-71) — `src/senselab/audio/workflows/audio_analysis/l1_plot.py:171-196`

- destination (as filed): plotting design (L1 evidence figure)
- rationale: signal-grouping/row-height design rationale (alphabetical order was unreadable)

Lines 171-196:
```python
SIGNAL_GROUPS: tuple[tuple[str, str], ...] = (
    ("frame", "frame posteriors"),
    ("acoustic", "acoustic proxies"),
    ("scene", "scene classifiers"),
    ("diarization", "diarization"),
    ("asr", "ASR"),
    ("other", "other"),
)
"""Display order, grouped by what kind of evidence a signal is.

Alphabetical order interleaved a frame VAD, an acoustic proxy and a diarizer, which made the
figure unreadable: every row looked identical, so a reader could not tell what kind of claim
any of them was making. Grouping is what lets the eye compare like with like."""

_ROW_HEIGHT = {
    "spectrogram": 3.0,
    "scene": 1.4,
    "asr": 1.6,
    "frame": 1.2,
    "acoustic": 1.0,
    "diarization": 0.9,
    "level": 1.2,
    "other": 0.9,
}
"""Relative row heights. A uniform height gave a binary on/off row the same space as a
spectrogram, which wastes the figure on the rows carrying least information."""
```

### F-72 (raised-by A-72) — `src/senselab/audio/workflows/audio_analysis/l2_plot.py:1-13`

- destination (as filed): plotting design (L2 round timeline)
- rationale: "replaces mostly-empty chunked timeline PNGs" rationale

Lines 1-13:
```python
"""``L2/round/<n>/timeline.png`` — one figure per round, drawn after that round's fusion.

A single end-state figure cannot show what the iteration did. Per round, a reader can see
whether a round moved anything and where, which is what says whether the loop is earning its
cost — the same reason the maps themselves are written per round.

This replaces the chunked ``timeline_001.png`` / ``timeline_002.png`` output, whose panels were
mostly empty: a fixed time window rarely lines up with where anything actually happened, so
most chunks showed nothing and the interesting moment was split across two files.

Only fused quantities appear here. The evidence rows live in ``L1/signals.png``; keeping the
two apart is what stops a conclusion being read as another observation.
"""
```

---

## quality/degradation design (3 entries)

### F-53 (raised-by A-53) — `src/senselab/audio/workflows/audio_analysis/quality.py:1-36`

- rationale: "both returned 0.0 in every bucket measured" L1/L2 boundary rationale

Lines 1-36:
```python
"""L1 scene-quality measurements: what the estimators measured, in their own units.

Seven measurements per analysis window, each in native units and none rescaled:

- ``snr_brouhaha_db`` — Brouhaha's SNR head, dB;
- ``c50_brouhaha_db`` — Brouhaha's C50 (clarity) head, dB;
- ``snr_spectral_gating_db`` / ``snr_peak_db`` — senselab's two DSP SNR metrics, dB;
- ``rolloff_95_hz`` — the frequency below which 95% of spectral energy sits, Hz;
- ``proportion_clipped`` — fraction of samples at full scale;
- ``rms`` — root-mean-square energy, uncalibrated.

**Why no degradation scores here.** This module used to emit ``quality_snr`` and
``quality_reverb`` as ``[0, 1]`` scores via ``clip((clean_db - value) / span, 0, 1)`` against 25 dB
and 30 dB anchors. Both returned **0.0 in every bucket of every recording measured**, because
clean speech sits at 60-70 dB SNR and 59.8 dB C50 — far above anchors chosen for conversational
audio. Probing the model directly showed the heads were never the problem: across digital silence,
white noise and clean speech they span −5 to 70 dB SNR and discriminate speech from silence by
+0.98 on the VAD head. A working measurement was destroyed by a clamp sitting on top of it. The
anchors are calibration, so they belong in :mod:`degradation` at L2, where a fitted profile can
replace them and where a saturating choice is visible as a fusion decision rather than baked into
the recorded data.

Two related reductions were removed rather than moved. ``primary_snr_db`` picked Brouhaha and
otherwise averaged the DSP metrics — estimator selection is fusion. ``quality_uncertainty`` took
the standard deviation of all three; because they use different noise-floor definitions, that
spread measured definitional disagreement rather than measurement uncertainty and pinned at 1.0
structurally, even on perfect audio. See
``specs/20260728-221507-per-speaker-identity-scene/l1-post-processing-register.md`` items 17-24.

**Analysis resolution ≠ reporting grid.** The STFT and model estimators are unreliable below
~0.5 s (Brouhaha is trained at 6 s), so measurement happens on a fixed 0.5 s / 0.25 s analysis
window. Reporting buckets are **resampled** from it rather than copied from the nearest window
(``resolution.resample_series``): coarser than the analysis hop integrates, finer holds. The true
resolution stays in provenance so a consumer cannot mistake a repeated value for an independent
one.
"""
```

### F-54 (raised-by A-54) — `src/senselab/audio/workflows/audio_analysis/quality.py:326-352` (``quality_series``)

- destination (as filed): quality/degradation design (D-20/D-25)
- rationale: units:"mixed" honesty and overlapping-window independence caveat

Lines 326-352:
```python
def quality_series(*, audio: Audio, brouhaha: Optional[BrouhahaFrames]) -> dict[str, Series]:
    """One native-resolution :class:`~.shapes.Series` per quality target (D-20, D-25).

    Args:
        audio: The pass audio.
        brouhaha: Per-frame Brouhaha outputs, or ``None`` when the model was unavailable — its two
            targets are then absent from the result rather than present and null, because a model
            that could not load has not measured nothing.

    Returns:
        ``{signal name → Series}`` at the analysis grid this module measures on, **not** at any
        reporting grid. Each series carries its own units, so nothing here is ``units: "mixed"``.

    This replaces :func:`harvest_quality_measurements` for consumers that hold a
    :class:`~.sampler.Sampler`. The difference is not cosmetic:

    - **No resampling.** The old function integrated or held each signal onto a reporting grid
      handed to it, which is a producer making an L2 decision — which grid, and which rule onto it.
      Here the values stay where they were measured and the consumer asks.
    - **Seven targets, seven series.** ``snr``, ``c50``, ``rolloff``, ``clipping`` and the rest answer
      different questions in different units, and one row holding all of them is exactly the bundle
      D-20 dissolved. ``units: "mixed"`` was the honest admission of it.
    - **Window and hop both survive.** The analysis window is 0.5 s at a 0.25 s hop, so adjacent
      values share half their audio. A consumer that treats them as independent samples is wrong, and
      it can only know that if both numbers travel — which they do on ``Series`` and did not on a
      resampled row.
    """
```

### F-104 (raised-by A-104) — `src/senselab/audio/workflows/audio_analysis/acoustic.py:1-18` (`module docstring`)

- destination (as filed): quality/degradation design (loudness measurement)
- rationale: LUFS-vs-percentile loudness rationale, sampled rather than quoted at length

Lines 1-18:
```python
"""Absolutely-calibrated acoustic speech_presence signals.

The acoustic voters were percentile-normalised per recording: a 10th-percentile floor and a
75th-percentile ceiling, described as calibrating to "high vs low for this specific recording".
That makes the value a **rank**, not a level, and it fails in three ways that were all visible
in one figure:

- ~10% of frames pin at 0 and ~25% at 1.0 **by construction**, whatever the audio contains, so
  the voter saturates independently of the signal.
- A uniformly quiet recording still spreads to fill ``[0, 1]``, so quiet frames read as loud —
  the inversion against the dBFS track.
- The dB→``[0, 1]`` mapping differs for every file, so the value cannot be compared to dBFS, to
  another recording, or to a fixed threshold.

Loudness is therefore measured in **LUFS** (BS.1770 gated loudness, via ``pyloudnorm``), which
is absolute: two recordings at the same level report the same number, which is exactly the
property percentile normalisation destroys. The confidence mapping is a fixed dB→``[0, 1]``
ramp anchored on speech levels, so a quiet frame reads quiet.
```

---

## speaker attribution / clustering-statistics design (3 entries)

### F-20 (raised-by A-20) — `src/senselab/audio/workflows/audio_analysis/speaker.py:139-153` (``harvest_speaker_votes``)

- destination (as filed): speaker attribution design (speaker-axis-attribution-design.md, already cross-referenced)
- rationale: measured "same-speaker-as-before" gate replacement (0.666 vs 0.168 doubt)

Lines 139-153:
```python
    """Yield ``{"start", "end", "votes"}`` per bucket for the speaker axis.

    **The axis asks "how sure are we who is speaking here?"** — attribution, not change. Its two
    scored voters come from ``attribution``: ``speaker_assignment`` (do the diarizers agree who is
    here, measured over *all* the answers they gave, since absent a target embedding no speaker is
    privileged) and ``target_activity`` (do we know anyone was active at all). Both are gated by
    ``word_coverage``: a bucket with no words has no speech to attribute and gets no claim.
    Everything else this emits is a *measurement* other consumers read — the cluster assignments, the
    embedding cosines, the change points, the overlap distribution — and is deliberately unscored, so
    the fold sees two voters.

    It asked "was it the same speaker as before?" until 2026-08-05, scored per (diar × embedder) pair
    against embedding cosine. On a 0.1 s grid that asks ten times a second against 0.5 s embedding
    windows, and it read 0.666 on a conversation whose per-speaker presence doubt was 0.168. See
    ``specs/20260728-221507-per-speaker-identity-scene/speaker-axis-attribution-design.md``.
```

### F-21 (raised-by A-21) — `src/senselab/audio/workflows/audio_analysis/speaker_identity.py:1-25`

- destination (as filed): speaker attribution / per-speaker-identity design
- rationale: validation-recording anecdote motivating embedding-clustering as a synthetic diarizer

Lines 1-25:
```python
"""Per-speaker speaker uncertainty (T095-T097, FR-001 to FR-011).

The speaker axis reports one uncertainty value per time bucket, answering "was it the same
speaker?" That scalar cannot express *how many* people the analysis thinks are present, and
the distinction matters: on a validation recording two diarizers each reported one speaker
for the whole clip while embedding clustering reported five distinct regions aligned to name
boundaries. The axis correctly registered high uncertainty, but a consumer reading 0.67
cannot tell "we disagree about who spoke" from "we disagree about whether this is one person
or four" — different problems with different fixes.

So speaker becomes **per speaker**: a distribution over how many speakers are present, one
hypothesis per speaker with its own existence uncertainty, and a speech_presence track per
hypothesis.

Two design commitments:

**Multi-modal disagreement is representable.** A mean or a majority vote would have reported
"one speaker, slightly uncertain" for the case above, which is precisely the wrong summary.
The posterior keeps the competing counts and names which sources backed each.

**Weight comes from the influence gates, not from counting heads.** A clustering-derived
pseudo-diarizer agreeing with the embeddings it was computed from is one computation counted
twice. Reusing ``influence.resolve_influence`` means a derived voter is attenuated by the
same rule everywhere in the loop rather than by a special case here.
"""
```

### F-35 (raised-by A-35) — `src/senselab/audio/workflows/audio_analysis/reliability.py:1-22`

- rationale: "saturated embedding check outvoted unanimous diarizer agreement" incident, third independent telling alongside speaker.py/embeddings.py

Lines 1-22:
```python
"""Per-signal reliability measured by perturbation, for use as aggregation weight.

A sub-signal's own uncertainty is evidence about how far its vote should carry. Without it,
aggregation treats every signal as equally trustworthy, and under max-doubt a single
unreliable signal decides the axis outright — which is exactly how a saturated embedding
check came to outvote unanimous diarizer agreement on a real recording.

The reliability is **derived rather than assigned**, on the same argument the speaker-count
posterior already uses: the raw and enhanced passes are the same recording under a
transform, so each signal's two answers already constitute a stability sample. A signal that
contradicts itself between them has not earned its weight; one that answers identically has.

Two properties are deliberate:

**One pass yields no claim.** A single observation is not a stability sample. Reporting
perfect reliability there would assert something never measured, so a signal with no
perturbation evidence simply keeps its full weight by default.

**Reliability never reaches zero.** With two perturbation points the measure is coarse, so a
hard zero would erase a dissenting claim rather than down-weight it — the same reasoning as
the influence gate's floor in ``influence.py``.
"""
```

---

## L1 shapes / derivative design (2 entries)

### F-60 (raised-by A-60) — `src/senselab/audio/workflows/audio_analysis/shapes.py:1-31`

- destination (as filed): L1 shapes / derivative design (D-18)
- rationale: "forcing shapes through one tabular row" reduction-catalogue rationale

Lines 1-31:
```python
"""The six shapes an L1 measurement can have (D-18).

``SignalRow(measurement: Mapping[str, float])`` fits only the scalar-per-bucket case, and L1's
outputs are six different kinds of object — four of which have no per-bucket scalar form at all.
Forcing them through one tabular row is what produced every reduction the real-run audit found: a
per-speaker probability matrix stored as its mean, 527 label scores stored as a hand-picked sum, a
span set stored as a covered fraction, a transcript stored as a word-overlap duration — each on a
0.1 s grid none of them was measured at, beside provenance describing the measurement that was
discarded.

Each reduction is a **decision**, and every one of them is now an L2 derivative that names its
choice. What L1 stores is the native shape, which is what this module is:

===========  ====================================================================
``Series``   ``(n_frames,)`` at a fixed hop, one named quantity
``Matrix``   ``(n_frames × n_channels)``, channels **named** or **arbitrary**
``Categorical``  ``(n_windows × k)`` over a fixed vocabulary, top-*k* truncated
``Embedding``    ``(n_windows × n_dims)``
``Spans``    variable-length ``[(start, end, label)]`` — on no grid at all
``Tree``     a ``ScriptLine``: text, nested chunks, per-node scores
===========  ====================================================================

**A bucket grid means something different to each of them**, which is the distinction one row type
could not express and :class:`GridRelation` now carries: it is a *resample* for ``Series`` and
``Matrix``, a *projection* for ``Categorical`` and ``Embedding`` (a 0.96 s window is not a 0.1 s
bucket), and a *reduction* for ``Spans`` and ``Tree`` (a transcript has no natural per-bucket
value). Conflating the three is what made one row type look sufficient.

Nothing here folds, thresholds, selects or rescales. A value the tool did not report is ``None``,
never ``0.0`` — zero is a confident claim, and imputing it manufactures confidence nobody expressed.
"""
```

### F-61 (raised-by A-61) — `src/senselab/audio/workflows/audio_analysis/shapes.py:148-159` (``Matrix``)

- rationale: measured "1.0000 in 100% of frames on a half-silent clip" pooled-value example

Lines 148-159:
```python
@dataclass(frozen=True, slots=True)
class Matrix:
    """``n_frames × n_channels`` at a fixed hop — a per-band noise floor, a multi-head output.

    The channels survive L1 because pooling them is a choice among ``mean`` / ``max`` / ``noisy-or``
    that changes the answer. Storing the pooled value made that choice invisibly, and it is what
    returned ``1.0000`` in 100% of frames on a clip that was half digital silence.

    Attributes:
        rows: One tuple per frame, each as wide as ``channels``.
        channels: Column names, in order.
        channel_semantics: Whether those names mean the same thing in every frame.
```

---

## L2 fusion/rounds design (2 entries)

### F-64 (raised-by A-64) — `src/senselab/audio/workflows/audio_analysis/rounds.py:1-24`

- rationale: "regional trust attenuates the wrong claim without silencing the right ones" rationale, with a 5-speaker/4.9s worked example

Lines 1-24:
```python
"""L2 iteration: regional trust and convergence.

Round 0 fuses the L1 signals as harvested, with one weight per signal for the whole recording.
Later rounds do two things round 0 cannot.

**Trust becomes regional.** A signal can be reliable in one stretch and not another, and a
global weight cannot express that. Once the mask says a region is target-free, a diarizer still
placing a speaker there has made a claim *that region* does not support — so its vote is
discounted **there**, and nowhere else. Global down-weighting for a local failure is the exact
mistake that suppressed the source which turned out to be right about the five named speakers
on a 4.9 s recording; regional trust is how the same evidence attenuates the wrong claim
without silencing the right ones.

**The mask's own confidence gates how far it may act.** A mask unsure that a region is
target-free has not earned the right to discount a signal for speaking there. Without that
gate a guess about the mask becomes a verdict about a model — and since the mask is itself
refined across rounds, that error would compound rather than settle.

An ``indeterminate`` region withdraws nothing: "I cannot tell" is not grounds to disbelieve
anyone.

Convergence is deliberately conservative about what counts as *no change*: a bucket that goes
from unmeasured to measured is progress, and treating it as stability would stop the loop
exactly when it had started working.
```

### F-65 (raised-by A-65) — `src/senselab/audio/workflows/audio_analysis/rounds.py:143-157`

- destination (as filed): L2 fusion/rounds design (D-12)
- rationale: cycle-detection window derivation (p+1 rounds to detect a period-p cycle)

Lines 143-157:
```python
DEFAULT_MAX_ROUNDS = 10
"""Round cap (D-12). Named rather than inlined because running out of rounds and agreeing are
different outcomes, and a reader needs to see which budget produced the first."""

EPISTEMIC_TOLERANCE = 1e-3
"""Credited change below which epistemic uncertainty counts as having stopped falling."""

DEFAULT_CYCLE_WINDOW = 4
"""How many recent rounds non-convergence is judged over.

A cycle of period *p* only becomes visible once the window holds a repeat, which takes ``p + 1``
rounds; four therefore catches periods one through three. Bounding it matters in the other
direction too: a state that recurred early and has not since is not *currently* cycling, and
stopping for it would end a run that had begun to make progress.
"""
```

---

## PII detection design (2 entries)

### F-55 (raised-by A-55) — `src/senselab/audio/workflows/audio_analysis/pii.py:19-25`

- rationale: measured near-zero true-positive rate for Presidio's most-severe categories in pediatric/clinical voice data

Lines 19-25:
```python
No category-severity weighting is applied anywhere in this pipeline (no SSN > date scaling):
in pediatric and clinical voice data, the nominally most severe Presidio categories
(``US_SSN``, ``CREDIT_CARD``) have near-zero true-positive rate and are dominated by ASR
digit hallucinations, so weighting them up would inflate exactly the hits a reviewer should
de-prioritise. See :func:`senselab.text.tasks.pii_detection.api._compute_detection_confidence`
for where that scoring actually happens; this module only supplies the cross-ASR pooling it
runs over.
```

### F-105 (raised-by A-122 (reclassified per reviewer instruction — was labeled restates-code in sweep-a-prose.md)) — `src/senselab/audio/workflows/audio_analysis/pii.py:261-268` (``report_to_dict``)

- destination (as filed): PII detection design (audio_analysis adapter), alongside F-55
- rationale: rejected-alternative design note explaining why a redundant per-span `perturbation` field was not carried, not a mechanical restatement of the dict comprehension below it

Lines 261-268:
```python
def report_to_dict(report: PiiPassReport) -> dict[str, Any]:
    """Convert a ``PiiPassReport`` into a JSON-serializable dict.

    Every span in a ``PiiPassReport`` was scanned for the same pass, so rather than
    carrying a redundant per-span ``perturbation`` field on :class:`PiiSpan` itself (which
    would put workflow vocabulary back onto a task-layer type), it's stamped onto each
    serialized span here, uniformly, from ``report.perturbation``.
    """
```

---

## final-outputs design (2 entries)

### F-87 (raised-by A-87) — `src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:378-389`

- rationale: "deliverable presence track used to be rebuilt here, diverging from the round's belief" history

Lines 378-389:
```python
    This is what ``final/`` is: an extraction. The deliverable presence track used to be *rebuilt*
    here from the belief state, into ``L2/speech_presence.parquet``, with columns
    (``speech_presence_confidence``, ``overlap_posterior``) that no round carried — so the number
    a consumer acted on was not the number any round believed, and there was nowhere to look to
    see when it had been decided. The columns now live on the estimate row and this function only
    moves bytes.

    One artifact per active axis, from the axis declaration rather than a list here. The four
    per-axis declarations this replaces named ``speech_presence``, ``asr`` and
    ``background_mask`` — and *not* ``speaker``, so the deliverable set was itself a list of
    three axes with the fourth missing, which is the failure ``axes.AXES`` exists to make
    impossible.
```

### F-88 (raised-by A-88) — `src/senselab/audio/workflows/audio_analysis/adaptive/fusion.py:430-438`

- rationale: "both written to L2 root instead of final/" history

Lines 430-438:
```python

    Where the docstring always said, and where they now go. Both were written to the ``L2`` root
    instead — flattened per-run quantities with no round to belong to — so ``final/`` carried no
    per-speaker output at all while two declarations for it sat unproduced, and every consumer
    reached into the belief tree for a deliverable.

    Replaces the single per-bucket speaker scalar rather than sitting beside it: two names
    for one quantity is how schemas rot, and nothing on the way to alpha needs backwards
    compatibility.
```

---

## identity-repair design (2 entries)

### F-89 (raised-by A-89) — `src/senselab/audio/workflows/audio_analysis/adaptive/identity_repair.py:35-43`

- rationale: "two bare 0.05 literals" naming/derivation history

Lines 35-43:
```python
"""Floor on a window's contribution to its segment's pooled embedding.

Pooling is ``p_voice``-weighted, so an unvoiced or unmeasured window would otherwise contribute
nothing to the vector that decides which speaker the segment belongs to — erasure by weight, in
the one computation where a short, quiet or unmeasured window is most likely to be the boundary
evidence. Two bare ``0.05`` literals used to sit inline here, doing the job of this constant
without naming it or connecting it to the argument that sets it; the number and its derivation
live in :data:`~senselab.audio.workflows.audio_analysis.floors.MIN_EVIDENCE_WEIGHT`.
"""
```

### F-93 (raised-by A-93) — `src/senselab/audio/workflows/audio_analysis/adaptive/interventions.py:866-877`

- rationale: measured "published axis 0.288→0.608 while deliverable stayed 0.1196" gap that the attribution axis exists to remove

Lines 866-877:
```python
        # **The axis's per-speaker term is not recomputed here.** It reads the diarization models'
        # agreement about who is in the bucket, and ``final/per_speaker_presence.parquet`` publishes
        # that same quantity from the *harvest* (``build_speech_presence_tracks(speaker_harvest)``,
        # never from ``refined_identity``). Shadowing it with a value folded over these repaired
        # clusters made the two disagree — the published axis went 0.288 -> 0.608 while the
        # deliverable still read 0.1196 — which is the defect the attribution axis exists to remove.
        #
        # This intervention's product is the refined segmentation above; if a 5-cluster repair against
        # a count posterior of 2 at 0.978 is wrong, that belongs in the identity deliverables where
        # the disagreement is visible, not folded into how sure we are who is speaking.
        touched.setdefault("speaker", set()).add(bk)
        # One vote per bucket, not two: the second was the per-speaker term this no longer shadows.
```

---

## labelstudio/export design (2 entries)

### F-73 (raised-by A-73) — `src/senselab/audio/workflows/audio_analysis/labelstudio.py:1-17`

- rationale: removed TextArea/coarse-grid history

Lines 1-17:
```python
"""Label Studio bundle integration for the three uncertainty axes.

The bundle exposes:
    - one Labels track per fused L2 axis, named ``uncertainty__<axis>``. No pass token: an axis is a
      fold across passes, so there is no per-pass axis to draw.
    - **no transcript text.** There was an ``uncertainty__asr__text`` TextArea rebuilding a
      per-bucket consensus from each model's bucketed transcript; the words are published at word
      resolution in ``final/transcript.json``, and ``adaptive.ls_final`` renders them as
      ``final__consensus_transcript__text`` in the deliverable bundle this one is the input to. Two
      renderings of one transcript at two resolutions is one too many, and the coarse one is what
      forced the asr axis onto a 1.0 s grid of its own.
    - per-pass, per-signal evidence tracks ``<pass>__signal__<signal>`` straight from the L1
      signal rows. That is where "what did each model say on each pass" is legitimately served —
      per pass without being an axis.
    - the scene tracks ``<pass>__presence__{quality,sources}``, which are per-pass
      *measurements* and stay per-pass.
"""
```

### F-74 (raised-by A-74) — `src/senselab/audio/workflows/audio_analysis/labelstudio.py:652-667`

- rationale: "per-speaker presence labelled by speaker, not merged" rationale

Lines 652-667:
```python
def attach_scene_context_tracks_to_ls(
    *,
    ls_tasks: Any,  # noqa: ANN401 — list[dict] or dict, matching attach_uncertainty_tracks_to_ls
    ls_config: str,
    mask_rows: Sequence[Mapping[str, Any]] = (),
    speaker_rows: Sequence[Mapping[str, Any]] = (),
    perturbation: str = "raw",
) -> tuple[Any, str]:
    """Append the background-mask and per-speaker speech_presence tracks to the LS bundle.

    Both answer questions a human reviewer cannot answer from the uncertainty tracks alone.
    The mask decides which background findings are trustworthy, so a reviewer checking those
    findings needs to see the same intervals the machine used (FR-033). Per-speaker speech_presence
    is labelled by speaker rather than merged, because knowing *who* is contested is the
    entire reason the speaker axis moved off a single scalar — a merged track would put the
    same unreadable number back in front of the annotator.
```

---

## speaker/occupancy design (2 entries)

### F-58 (raised-by A-58) — `src/senselab/audio/workflows/audio_analysis/occupancy.py:1-22`

- destination (as filed): speaker/occupancy design (D-19)
- rationale: "honest uncertainty is disagreement across models, not one model's confidence" rationale

Lines 1-22:
```python
"""Speaker occupancy and count, derived from spans across diarizers of differing capacity (D-19).

L2 derivatives over :class:`~.shapes.Spans`. Every diarization tool emits what sortformer and
``community-1`` already emit — ``(start, end, speaker_label)`` at its own boundaries, on no grid — and
occupancy or a count is derived by projecting them here.

**Why this replaces the Poisson-binomial.** ``joint.overlap_count_posterior`` built a count
distribution over ``segmentation-3.0``'s per-speaker channel probabilities, treating them as
independent Bernoullis. They are a **powerset conversion**: the classes are mutually exclusive by
construction and the per-speaker columns are derived from them, so the independence the
Poisson-binomial assumes was never there. What it produced was one model's internal confidence dressed
as a distribution over speaker count.

The honest uncertainty about "how many speakers are active here" is the same as for every other axis:
**disagreement across models.** Each diarizer's spans give a count at time *t*, and the spread across
diarizers is the uncertainty. That is measured rather than assumed, and it composes with D-19's
censoring — a tool at its capacity contributes a *lower bound*, not a point.

**What is kept from the frame-level version**, because it was right: overlap is an *instantaneous*
fact. Two speakers alternating inside a bucket average to 0.5 on each channel, which as a per-bucket
calculation reports an overlap that never occurred. :func:`count_at` evaluates at an instant, so it
cannot produce that artifact at all.
```

### F-59 (raised-by A-59) — `src/senselab/audio/workflows/audio_analysis/occupancy.py:68-79` (``capacity_for``)

- destination (as filed): speaker/occupancy design (D-19)
- rationale: "raising instead of the current design was tried and is wrong at this depth" rationale

Lines 68-79:
```python
def capacity_for(model_id: str) -> Capacity:
    """The declared capacity for ``model_id``, or ``None`` when nothing declared one.

    ``None`` is **not** a permissive default. It means *unknown*, and
    :func:`spans_from_diarization` omits such a tool from the span set rather than including it: a
    tool whose capacity is unknown cannot be censored correctly, and including it uncensored is
    exactly the bias :func:`count_posterior` exists to correct. Omitting loses its evidence, which is
    worse than having it and better than having it wrong.

    Raising instead was tried and is wrong at this depth — one unlisted diarizer would kill the whole
    harvest, so a new model could not be trialled without a table edit first.
    """
```

---

## visualization design (2 entries)

### F-97 (raised-by A-97) — `src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:24-29`

- rationale: measured flat-vs-varying mask-derivative figure discrepancy

Lines 24-29:
```python
Row 8 was missing. This figure hand-listed three axes and drew row 3 where a reader looks for the
fourth, so on a run whose mask derivative is a single ``target_active`` region at uncertainty 0.0
the final figure showed one flat confident band while ``L2/round/<n>/timeline.png`` showed the
same axis varying across 1070 buckets. Two figures disagreeing about one axis, because only one of
them was drawing it. ``axes.AXIS_NAMES`` is the list; a row per name is what keeps it from being
short again.
```

### F-98 (raised-by A-98) — `src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:716-721` (``_fused_axis``)

- destination (as filed): visualization design / belief-store cleanup backlog, worth preserving even after F-4's stale sentence is fixed
- rationale: "this function is scaffolding for a defect, should be deleted rather than maintained"

Lines 716-721:
```python
    **This function is scaffolding for a defect, and should be deleted rather than maintained.**
    The layered design has exactly one axis lineage: L1 emits per-signal measurements, L2 fuses
    them, `final/` holds the result. A second speaker axis exists only because L1 emits a per-pass
    axis fold it is not supposed to compute (item 25) and the belief store was built to ingest it.
    Remove that fold and the belief store has nothing to read but L2's axes — one number, and no
    reason for this comparison to exist.
```

---

## L2 derivative/sampler design (1 entry)

### F-63 (raised-by A-63) — `src/senselab/audio/workflows/audio_analysis/sampler.py:1-27`

- destination (as filed): L2 derivative/sampler design (D-25)
- rationale: measured provenance-describes-a-measurement-the-file-lacks example

Lines 1-27:
```python
"""Query native-resolution signals at the samples a consumer wants, and cache the answers (D-25).

**Producers do not resample.** A producer that reduces onto a target grid has made an L2 decision —
which grid, and which reduction onto it — and destroyed the alternative before anyone could ask for it.
That is the defect D-18 found in the artifacts: ``native_window_s: 0.0619, resolution_s: 0.0169``
recorded on a row spanning ``0.0 → 0.1``, provenance describing a measurement the file did not contain.

So L1 emits at its own resolution (:mod:`.shapes` already does) and the *consumer* asks: this signal,
over this interval, reduced this way. The sampler answers, and remembers.

**The cache key is the derivative key.** D-21 names every projection ``(Target, Operator, Source)``, so
a query is one of those plus an interval — nothing new has to be invented to identify it. Three things
follow:

- D-22's *"materialisation is a caching and inspectability decision, not a semantic one"* becomes
  literal. A derivative is materialised iff something persisted it; the inline and stored forms are the
  same key with the same value.
- :class:`~.shapes.GridRelation` becomes the **dispatch**. ``RESAMPLE`` is arithmetic over finer frames,
  ``PROJECT`` assigns a window's value to the buckets it spans, ``REDUCE`` computes a per-bucket
  quantity the object does not have. Those are exactly the three ways a query can be answered.
- **Over-sampling stops being expressible by accident.** A consumer asking for 100 ms non-overlapping
  buckets gets them whatever the native hop is; a 0.1 s window at a 0.02 s hop cannot arise, because no
  producer chooses the output spacing.

This is not a storage layer. It reads signals and writes nothing; a materialised derivative is still
written by ``derive`` under ``StageIO``.
"""
```

---

## LS-export design (1 entry)

### F-95 (raised-by A-95) — `src/senselab/audio/workflows/audio_analysis/adaptive/ls_final.py:75-80`

- rationale: "read final/ back out of the directory it was about to write" history

Lines 75-80:
```python
    # The run bundle is the belief rendered for an annotator — per-pass uncertainty and scene
    # tracks — and it is *input* here: this stage appends the consensus tracks and writes the
    # deliverable next to them. So it lives under ``L2/``. While it lived in ``final/`` this stage
    # read it back out of the directory it was about to write, and in the integrated path the
    # bundle was not written until after the loop had already run, so the read always missed and
    # the stage silently produced nothing — "not found" being indistinguishable from "no bundle".
```

---

## cache/provenance design (1 entry)

### F-12 (raised-by A-12) — `src/senselab/audio/workflows/audio_analysis/stage_context.py:202-243` (``_commit_sha_for``)

- destination (as filed): cache/provenance design (commit-SHA pinning rules)
- rationale: three-outcome commit-resolution design rationale

Lines 202-243:
```python
    def _commit_sha_for(self, model_id: str | None) -> str | None:
        """Resolve ``model_id`` to this run's commit SHA, or ``None`` when there is no commit to pin.

        Resolution has to happen here, above the load, because the cache key is computed to decide
        *whether* to load at all — a SHA harvested during loading would arrive too late to key on.

        That placement is also why this is *not* the same decision as
        ``signal.resolved_commit_sha``'s, which degrades every failure to ``None``. That function
        fills in a provenance **record**, where "unknown commit" is an honest and cheap answer.
        Here ``None`` is a **key** component (``cached_inference.cache_key``'s ``commit_sha``), and
        every id that degrades to it shares one bucket — so two different upstream commits of the
        same model would collide, and the second run would be served the first one's result. Three
        outcomes, therefore, three treatments:

        - **Not a Hub id at all** — a bare ``None`` (a model-less stage, e.g. ``features``) or a
          name with no ``/`` (a local backend). Short-circuits with no Hub round-trip.
        - **A definitive "there is no commit"** (``RepositoryNotFoundError`` or
          ``HFValidationError``) — ``None`` is then the *correct* value rather than a degradation,
          and it is the same answer every run, so no two commits can collide behind it. Two shapes
          reach it. ``RepositoryNotFoundError`` is the Hub having answered that no such repo
          exists: ``default.yaml`` ships ``yamnet: google/yamnet``, a TensorFlow backend whose id
          happens to contain a ``/`` and so trips the Hub-id heuristic above — the crash being
          fixed. ``HFValidationError`` is the *client* refusing to ask, because the string is not a
          well-formed repo id at all: a local filesystem path (``/scratch/models/foo``) contains
          ``/`` and so trips the same heuristic, but ``model_info`` rejects it before any request.
          That verdict is offline, deterministic and independent of Hub availability, which is what
          makes it definitive rather than a "could not tell" — it cannot be a transient failure
          wearing a not-found's clothes, because no network was involved in reaching it.
        - **Anything else** — a 429, a network error, a ``GatedRepoError`` — propagates. Those all
          mean "we could not tell", which is unsound for a key: the load may well succeed, and its
          result would be stored under a commit-blind key that a later run cannot distinguish from
          any other commit's. ``GatedRepoError`` **subclasses** ``RepositoryNotFoundError``, so it
          has to be excluded by hand; ``dependencies._ensure_hf_model`` makes the identical split
          for the identical reason.

        Alternatives considered: ``_YAMNET_ALIASES`` already accepts a bare ``"yamnet"``
        (``classification/api.py``), so setting ``default.yaml``'s ``yamnet:`` to ``yamnet`` would
        remove this specific crash without touching cache-key semantics at all — strictly smaller.
        It was not taken because it fixes one config value rather than the class: any non-Hub
        backend whose id carries a ``/`` hits the same abort, and the heuristic cannot tell them
        apart without asking the Hub.
        """
```

---

## clustering/statistics design (1 entry)

### F-25 (raised-by A-25) — `src/senselab/audio/workflows/audio_analysis/clustering.py:104-133` (``assign_unified_clusters_with_seed_phase``)

- destination (as filed): clustering/statistics design, beside calibration.py's derivation blocks
- rationale: two-threshold derivation (cross_group 0.75 vs cosine 0.5)

Lines 104-133:
```python
    Why two thresholds:
      - ``cross_group_threshold`` (default 0.75) governs match-across-groups
        (raw vs enhanced, ECAPA-clustering vs ResNet-clustering). Same
        speaker across passes is typically cos_sim 0.85+, different
        speakers within a pass sit around 0.30-0.50, so 0.75 cleanly
        separates them.
      - ``cosine_threshold`` (default 0.5) governs ``other_items`` matching
        — used for pyannote / sortformer labels to snap to an existing seed
        when their mean embedding clears the bar. Lower threshold here is
        intentional: those models segment differently than the synthetic
        source, so their per-label means can be noisier.

    Phase 1 — *seed groups*: walk each group; each ``(key, mean_emb)`` in
    that group is assigned a NEW centroid (within-group items never share
    a cluster id). After the group is consumed, walk a second time across
    the existing centroid pool — if any pair of centroids have cos_sim ≥
    ``cross_group_threshold``, merge them. This is how raw_Peter and
    enh_Peter end up in the same unified cluster.

    Phase 2 — *frozen pool*: each ``(key, mean_emb)`` in ``other_items``
    snaps to the closest seed centroid (no threshold once the pool is
    seeded). If no centroid exists yet, fall back to the legacy
    ``cosine_threshold`` rule so the function degrades gracefully when no
    seeds were provided.
    """
    out: dict[K, str] = {}
    centroids: list[np.ndarray] = []
    # Map cluster_id (C0..) → list of centroid indices that belong to it.
    # We append to ``centroids`` strictly per (key) but the cluster_id may be
    # reused when a cross-group match fires.
```

---

## corroboration/presence design (1 entry)

### F-84 (raised-by A-84) — `src/senselab/audio/workflows/audio_analysis/adaptive/corroboration.py:1-21`

- rationale: measured `acoustic_loudness`/`ast` corroboration pinning near 1.0 under max-pooling

Lines 1-21:
```python
"""Independent presence evidence, derived from the run rather than configured.

One measurement — how far independent evidence supports a speech claim in a span — serving both
consumers that used to erase evidence instead: the belief store's uncorroborated-claim attenuation
and the word-stream ensemble's per-word weight. Sharing the derivation is the point: two
definitions of "corroborated" would drift, and the one that drifted would be the one deciding what
reaches the transcript.

**The pool must exclude claimants, and that is a correctness condition, not a refinement.** The
obvious quantity to reach for — the belief row's ``p_voice`` — is a weighted mean over *all*
presence voters including the ASR models themselves, and ``aggregate._weighted_p_voice`` maps a
voter carrying ``hallucinated: True`` to ``p = 0.1``. Measuring an ASR's claim against that number
is the model indicting itself, the exact failure ``adaptive.provenance.classify_resolution`` exists
to catch. ``support.evidence_signal_names`` excludes ASR and diarizer ids structurally, on the
ground that both infer presence from a decision that already presupposes a speaker.

**A signal that never reports absence makes the measure inert.** Corroboration only ever removes
weight, so it runs entirely on negative evidence; ``support.informative_evidence`` drops voters
that never say "no speech" (measured: ``acoustic_loudness`` median 0.897, ``ast`` 0.728 over 697
buckets — pooled with max they pin corroboration near 1.0). An empty pool is a legitimate outcome
and means the mechanism is inert on this run; it must be *reported*, never silently assumed away.
```

---

## evaluation design (1 entry)

### F-85 (raised-by A-85) — `src/senselab/audio/workflows/audio_analysis/adaptive/evaluate.py:73-77`

- destination (as filed): evaluation design (L1/L2/final boundary)
- rationale: "used to reach into L2/ for intermediates, scoring a scorer" history

Lines 73-77:
```python
    # The evaluator scores the deliverable and nothing else. Every read here is of ``final/``,
    # which is what makes it a consumer of the answer rather than a stage that builds it — it
    # used to reach into ``L2/`` for the presence track, the baseline round's uncertainty mass
    # and the last round's speaker axis, and each of those was a scorer scoring an intermediate.
    final = final_dir(out_dir)
```

---

## influence/support/reliability weighting design (1 entry)

### F-22 (raised-by A-22) — `src/senselab/audio/workflows/audio_analysis/speaker_identity.py:300-308` (``source_kind_for``)

- destination (as filed): influence/support/reliability weighting design (one canonical home removes three repeats)
- rationale: "5 speakers vs 2" anecdote, duplicated near-verbatim in influence.py, support.py, reliability.py (4 copies total)

Lines 300-308:
```python
    The live example, recorded so the decision stays arguable: ``embedding_silhouette`` is
    marked derived because it seeds the cross-model label harmonisation — other diarizers'
    labels snap to its centroids — and the same embeddings drive same-label and change-point
    validation, so that evidence already enters the speaker axis three ways. Against that:
    it runs an embedding model on the audio and clusters the result, which is a direct
    observation; and on one validation recording it reported five speakers where two
    "independent" diarizers reported one, with re-examination suggesting it was the closer
    answer. Down-weighting it may therefore suppress correct results. The gate is
    configurable precisely because that tension is unresolved and needs ground truth.
```

---

## per-speaker-identity-scene design (1 entry)

### F-11 (raised-by A-11) — `src/senselab/audio/workflows/audio_analysis/axes.py:281-334` (``IDENTITY_ONLY_AXES``)

- destination (as filed): per-speaker-identity-scene design (layered-architecture.md)
- rationale: measured 5x enhanced-vs-raw `words` voter reading

Lines 281-334:
```python
IDENTITY_ONLY_AXES: Final[frozenset[str]] = frozenset({"background_mask", "speaker"})
"""Axes whose question is about the recording as read, so only the identity perturbation answers it.

``background_mask`` asks whether a region is free of **target** activity. ``stages.py`` already builds
the mask itself on the unmodified variant alone, and states the measurement behind that: the enhanced
pass masked 50% of a real recording against the unmodified pass's 17.9%, "because speech enhancement
removes the non-speech evidence the mask reads target activity from. A mask built there is
misleadingly generous -- it reports 'safe for background claims' precisely where the background was
destroyed."

That argument was never applied to the mask's **axis**, which harvested ``speakers`` / ``speech`` /
``words`` from every perturbation. On the 48 kHz validation clip its enhanced ``words`` voter read mean
0.0510 against raw's 0.0102 -- 5x higher, in exactly the direction the note predicts, because
enhancement changes what the recognizers find. ``fuse.SnrGate`` happened to suppress it in 40 of 49
buckets by gating on SNR, but not in the 9 below the floor, so the axis was partly built on a pass its
own mask refuses to use.

This is *not* the SNR gate and does not overlap with it. The gate asks "is there anything here for a
repair to repair"; this asks "is this perturbation entitled to answer this question at all", and for
the mask the answer is no at any SNR. A perturbation excluded here still contributes its cross-pass
``|delta|`` to ``reliability.signal_stability``, which is what sets each signal's weight.

``speaker`` was added 2026-08-07, on the same argument and a decisive measurement. **Who is speaking
is a fact about the recording**, and folding a transform's opinion of it produced an axis that
contradicted its own deliverable.

Measured on an 11.26 s recording whose SNR is genuinely low — min −7.4 dB, median −1.5 dB, 105 of 110
buckets below ``triage.snr_floor_db`` — so ``fuse.SnrGate`` admits the enhanced pass almost
everywhere, which is what it is for. At 9.8–10.3 s:

| source | reading |
|---|---|
| all four diarizers, raw pass | ``C0`` unanimously, ``speaker_assignment`` **0.000** |
| the same four, enhanced pass | 2–2 split, ``speaker_assignment`` **1.000** |
| the fused axis | **0.500** |
| ``final/per_speaker_presence.parquet`` | speaker ``S0``, confidence **1.0000**, uncertainty **0.0000**, 4 sources |

The deliverable and the axis describing it disagreed, which
``speaker_attribution_test.test_no_intervention_recomputes_the_per_speaker_term`` already forbids —
it was written when ``I2_recluster`` caused the same contradiction by overwriting the term.

**Reporting the divergence as epistemic instead would not have fixed it.** The deliverable is built by
``build_speech_presence_tracks(speaker_harvest)`` from the *raw* harvest, so it folds one pass by
construction; an axis folding two can never agree with it however well it labels the disagreement. And
making the deliverable fold both is not coherent — there is no answer to "whose ``S0``, raw's or
enhanced's". Either both fold both passes or both fold raw, and only the second is buildable.

The enhanced pass keeps its actual job here: its cross-pass ``|Δ|`` is what
``reliability.signal_stability`` turns into this signal's weight. What it no longer does is vote on
who was speaking.

``speech_presence`` and ``asr`` are unaffected: they ask about content a transform may legitimately
change the reading of, which is what makes the perturbation a sample rather than a contaminant.
"""
```

---

## perturbations/passes design (1 entry)

### F-62 (raised-by A-62) — `src/senselab/audio/workflows/audio_analysis/perturbations.py:1-26, 76-102`

- destination (as filed): perturbations/passes design (D-17)
- rationale: measured raw-vs-enhanced speaker-axis divergence (0.0 vs 0.398, averaging to a false 0.227)

Lines 1-26:
```python
"""The open set of perturbations a run measures under (D-17).

A **perturbation is a transform of the recording**. ``raw`` is the identity; speech enhancement
is one more; the set is open, and a future L2 round may propose another — so L1 is re-enterable
and nothing downstream may assume how many there are or what they are called.

Two assumptions used to be spelled into the code instead of declared here, and both were wrong
in the same way:

- **exactly two.** ``PassLabel`` was ``Literal["raw_16k", "enhanced_16k"]``, the driver ran two
  blocks, and ``get_stream_wav`` branched on those two strings. A third perturbation was a code
  edit in every one of those places.
- **the name carries the transform.** ``variant = "speech_enhanced" if label.startswith("enhanced")``
  inferred what had been done to the audio from how the directory happened to be spelled, so a
  perturbation named ``enhanced_lowpass`` would have claimed to be plain enhancement and a
  perturbation named ``sepformer`` would have claimed to be unmodified.

Here the transform is *declared* beside the name, the parameters travel with it, and
``L1/perturbations.json`` records the whole set — so a reader of a finished run can tell what
each ``L1/perturbation/<k>/`` directory contains without knowing which flag produced it.

Adding a perturbation that reuses a known transform (a second enhancement model, say) is a
register entry and no code edit anywhere. Adding a genuinely *new* transform is one entry in
:data:`TRANSFORMS` plus its implementation in :func:`apply` — one edit, in the one place that
knows how to do it.
"""
```

Lines 76-102:
```python
SNR_GATED_TRANSFORMS: Final[frozenset[str]] = frozenset({"speech_enhanced"})
"""Transforms whose reading only counts where the recording is actually degraded.

**A speech-enhancement model is a repair, and a repair has no standing where nothing is
broken.** Above the SNR floor there is no noise for it to remove, so any change it makes to a
downstream answer is an artifact of the transform rather than evidence about the recording.
Folding it in unconditionally was measured on a clean two-speaker conversation (41–70 dB SNR
throughout): the raw pass placed the speaker axis at exactly 0.0 in 179 of 190 buckets, the
enhanced pass at 0.398 with only 51% zeros, and averaging the two published 0.227 — the
diarizers agreed and the axis said otherwise, in every one of the 178 buckets where nothing was
in dispute.

The gate is on **SNR alone, not on ambiguity.** Admitting the perturbation wherever the raw
sources disagreed was measured too, and it reads better on that clip (0.0202 against 0.0317,
because enhancement resolves five of the seven contested buckets) — but it is the wrong rule:
at genuinely low SNR the raw sources can be unanimously *wrong*, all of them fooled by the same
noise, and that is precisely the case enhancement exists for. An ambiguity requirement locks it
out there. Ambiguity in a high-SNR bucket, meanwhile, is a real disagreement to be resolved on
the recording's own evidence, not arbitrated by a transform.

**Invariance probes are deliberately not listed** (see :mod:`invariance`). Gain scaling, whole-
sample time shift and a small DC offset are chosen so that a *correct* model's answer cannot
change, which makes them meaningful everywhere and at every SNR — gating them by degradation
would remove the only condition under which their disagreement is unambiguously a model defect.
The distinction is the point: enhancement is a transform a model may legitimately answer
differently on, and an invariance probe is one where it may not.
"""
```

---

## plotting design/layout history (1 entry)

### F-69 (raised-by A-69) — `src/senselab/audio/workflows/audio_analysis/plot.py:270-278` (``_load_background_mask_rows``)

- rationale: "written against the flat layout, matched nothing once passes moved under L1/" history

Lines 270-278:
```python
def _load_background_mask_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Read the background mask, if one was written.

    One named path rather than a glob. There is one mask per run — it is only built on the
    unmodified variant, because enhancement removes the non-speech evidence target activity is
    read from — so a glob was never selecting between candidates, only failing quietly when the
    layout moved beneath it. This one was written against the flat layout and matched nothing for
    as long as passes have lived under ``L1/``, which reads exactly like a run with no mask.
    """
```

---

## plotting/layering design (1 entry)

### F-68 (raised-by A-68) — `src/senselab/audio/workflows/audio_analysis/plot.py:1-42`

- rationale: "a default argument decided the layer" naming/layering incident

Lines 1-42:
```python
"""6-row aggregate-uncertainty + per-source-detail timeline plot.

Per FR-006 (revised 2026-05-09): the plot must let a reviewer answer "WHY is this
bucket uncertain?" in addition to "HOW uncertain is it?". Three uncertainty rows show
the headline scalars; three detail rows show the underlying source signals so the
reviewer can drill in directly without opening the parquets.

Rows top-to-bottom:

1. **speech_presence_uncertainty** — raw solid + enhanced dashed in [0, 1]
2. **speaker_uncertainty** — raw solid + enhanced dashed
3. **asr_uncertainty** — raw solid + enhanced dashed
4. **Diarization detail** — per (pass, diar_model), speaker bars at native segment
   times, colored by speaker label. Lets the reviewer see where each diar model
   thinks each speaker is.
5. **Embedding similarity (adjacent windows)** — per (pass, embedding_model), a line
   of ``1 − cos_sim`` between consecutive uniform-window embeddings (default
   2 s window / 1 s hop). Spikes mark speaker-change events independent of any diar
   model's segmentation. Lets the reviewer compare the diar models' label
   transitions against what the audio itself says.
6. **ASR output** — per (pass, asr_model), token-level spans at the actual
   timestamps from the resolved (post-MMS-aligned) ASR result. Lets the reviewer see
   which models returned text where and confirm whether high asr uncertainty
   is real disagreement or punctuation/hesitation noise.

**This figure is an L2 conclusion and writes to ``final/uncertainty_detail.png``.** Rows 1–3 are
the fused axes; a fold across signals *and* perturbations is an axis (D-16), and an axis is L2's.
It previously defaulted to ``L1/timeline.png`` — chosen here, inside the renderer, to escape a
filename collision with the adaptive ``final/timeline.png``. That resolved the collision by
relabelling the figure as "the evidence timeline", which it is not: the first parameter is
``fused_axes``. Two lessons kept in the code rather than only in the spec:

- **A collision between two conclusions is not fixed by moving one into the evidence layer.** It
  is fixed by giving them different names, which is what ``uncertainty_detail`` versus
  ``timeline`` now does.
- **A default argument decided the layer.** The call site passed ``run_dir`` and the callee picked
  the directory, so no reviewer of the call site could see which layer was being written. The
  default is still here for convenience, but it now names the layer this figure belongs to.

The evidence view with no conclusions on it is ``L1/signals.png`` (``l1_plot``), whose docstring
states the rule this module was breaking.
"""
```

---

## provenance/mutual-influence design (1 entry)

### F-101 (raised-by A-101) — `src/senselab/audio/workflows/audio_analysis/adaptive/provenance.py:6-16`

- rationale: "uncertainty can fall for two different reasons, indistinguishable in the number alone" rationale, valuable independent of F-5's wiring gap

Lines 6-16:
```python

The subtler job is :func:`classify_resolution`. In a loop where signals revise one another,
uncertainty can fall for two completely different reasons:

- **New evidence arrived** — the analysis genuinely learned something.
- **The value was overwritten** — and uncertainty was then recomputed *from the overwritten
  value*, so it fell because of the edit, not because of evidence.

Both look identical in the number alone. A loop that cannot distinguish them converges on
its own edits and reports high confidence in them, which is the single largest correctness
risk in the mutual-influence design. So the distinction is structural: every revision
```

---

## region-proposal design (1 entry)

### F-102 (raised-by A-102) — `src/senselab/audio/workflows/audio_analysis/adaptive/regions.py:20-24`

- rationale: "per-(pass,axis) proposal produced two overlapping regions for one ambiguity" history

Lines 20-24:
```python

    Rows must be time-ordered on one axis, and there is one set of them: a region is a span of the
    recording the run is unsure about, not a span of one pass. Proposing per (pass, axis) produced
    two overlapping regions for one ambiguity, each spending budget separately, and made the
    intervention catalogue's target a property of which pass happened to look worse.
```

---

## run summary/global aggregation design (1 entry)

### F-66 (raised-by A-66) — `src/senselab/audio/workflows/audio_analysis/global_summary.py:52-59` (``PASS_FOLD``)

- rationale: "not a minimum: raw/enhanced disagreement is evidence" rationale

Lines 52-59:
```python
PASS_FOLD = "mean over the passes that reported"
"""How per-pass diagnostics are combined into one run-level number.

Named because it is a choice. It is deliberately *not* a minimum: raw and enhanced are the same
recording under a transform, so they are a perturbation sample whose disagreement is evidence —
picking the lower-uncertainty one and reporting it as the run's bottom line discards exactly the
information the second pass was run to obtain.
"""
```

---

## run summary/reporting design (1 entry)

### F-67 (raised-by A-67) — `src/senselab/audio/workflows/audio_analysis/summary.py:1-18`

- rationale: "not-measured treated as zero overstates certainty" rationale

Lines 1-18:
```python
"""``final/summary.md`` — what a person needs to know about a run, without opening a parquet.

``summary.json`` is the machine record and is already large. Someone opening a run wants four
things quickly: how many speakers, how uncertain each axis was and how much of that is
reducible, where the worst regions are, and whether the loop converged or ran out of rounds.
Those live across L2 parquets and JSON, so answering "how did this run go" currently requires
knowing the layout.

Two reporting choices carry weight:

**Unmeasured buckets are counted, never averaged in.** Treating "not measured" as zero would
report a run as more certain than it was, which is the failure mode a summary is most likely to
introduce.

**The worst regions are named with their times.** "Uncertainty was 0.4" is not actionable;
"0.9 at 0.5–1.0 s" is. A mean alone hides a single bad region, which is usually the thing worth
looking at.
"""
```

---

## run-config design (1 entry)

### F-13 (raised-by A-13) — `src/senselab/audio/workflows/audio_analysis/run_config.py:9-14`

- rationale: the "seventy flags, zero shared bucket keys" measurement behind the no-per-knob-flags design

Lines 9-14:
```python
**Why not per-knob flags.** Seventy of them existed, and the run recipes in the repo's own docs
differed from one another only in flags whose right value a reader had no basis to pick. Worse, the
grid flags were live: a caller could set the four axes to four different spacings, which is exactly
what the shipped defaults did, and the result was that every cross-axis coupling in the pipeline ran
against zero shared bucket keys. A knob that no one can choose between settings for is not
configurability; it is an unmeasured decision with a public interface.
```

---

## stage-contracts / D-17 summary (1 entry)

### F-14 (raised-by A-14) — `src/senselab/audio/workflows/audio_analysis/contracts.py:1-58`

- rationale: "enumerating what is forbidden cannot terminate" rationale for the declare-what-is-permitted contracts design

Lines 1-58:
```python
"""D-17 — the pipeline is a DAG of workflows, each declaring its inputs and its outputs.

Three rounds of guards were written against the violation last found, and each missed the next
instance of the same class: a name list that omitted the fourth axis, a regex an alias slipped
past, a glob that saw the workflow package but not ``adaptive/``, three artifact rules that all
pass on a genuine per-pass axis table. **Enumerating what is forbidden cannot terminate.
Declaring what is permitted does.**

This module is that declaration, and it is the only place it exists. Nothing may restate it:
the DAG's edges are *derived* by matching one stage's declared reads against another's declared
writes, and both guards read the same tuples the DAG does.

Four things live here, in this order:

1. :data:`STAGE_CONTRACTS` — for each node (``L1``, an ``L2`` round, ``final``, ``eval``) the
   run-relative path patterns it may read and the artifacts it may write, each artifact carrying
   the **key** its rows are indexed by. The key is what makes the content rules derivable rather
   than enumerated: "an ``L1`` artifact is keyed by one perturbation" yields both *no
   perturbation* and *two perturbations* as violations without either being listed.
2. :data:`MODULE_STAGE` — which stage each pipeline module speaks for. Unlisted modules are
   ``PURE``: they may touch no run-relative path at all. The permission defaults to *none*.
3. :func:`static_violations` — walks the AST of every pipeline module (the whole package,
   ``adaptive/`` included, plus both CLI drivers), resolves local aliases to a fixpoint, and
   flags any read or write of a run-relative path outside the declaring stage's contract.
4. :func:`artifact_violations` — walks a real run's artifact tree and flags any file that is in
   no stage's declared outputs, any file whose *kind* the declaring pattern does not permit, any
   file the guard could not read, and any table whose key contradicts the artifact it was written
   as. Its mirror :func:`unproduced_declarations` flags the opposite: a declared output a complete
   run produces nothing for. Together they catch what static analysis cannot: a writer reached
   through a helper, a file nobody meant to emit, and a declaration nobody satisfies.

**A guard is defeated by the case it does not consider, and every one of those was found by
constructing it rather than by reading the code.** Four were, and closing them is what the shape
of this module is now for:

- a declaration broad enough to permit anything. ``**`` used to be free; it now costs a pinned
  set of ``suffixes``, a ``key`` that prohibits at least one dimension, and conformance to
  :func:`structural_vocabulary` — and :meth:`Artifact.__post_init__` refuses the declaration
  outright rather than leaving the guard to go quiet beneath it.
- a content rule that falls to a file extension. The key rules read every format in
  :data:`TABULAR_SUFFIXES`, the declaration pins which of them may appear where, and a file that
  cannot be read is :class:`UnreadableArtifact` — a finding, never a pass.
- a path bound in a way the resolver did not watch for. Assignment is one of seven binding forms
  (:data:`_BINDING_NODES`); tuple targets, starred targets, ``/=``, walrus and ``for`` were the
  six that were not.
- a real-run fixture that passed on a fragment. Completeness is judged against the declaration,
  by the same :func:`unproduced_declarations` that reports a declaration nothing satisfies.

:data:`KNOWN_DEVIATIONS` records where the tree does not yet conform. Every entry names the
D-17 clause it breaks and what closes it, and a live-ness check fails when an entry stops
matching — so a fixed violation must be deleted from the register rather than left to rot into a
permanent exemption.

**What the static guard cannot see**, stated so its silence is not read as absence: a path
handed to a helper as a parameter (``_write_round_belief(rounds_dir, ...)``) is opaque to it,
because the pattern is decided at the call site. That is precisely the gap
:func:`artifact_violations` exists to close, and it is why both guards are required rather than
either being a cheaper version of the other.
```

---

## support/reliability design (1 entry)

### F-31 (raised-by A-31) — `src/senselab/audio/workflows/audio_analysis/support.py:276-298` (``MIN_LOW_FRACTION``)

- destination (as filed): support/reliability design, migrated with the numbers replaced or dropped, not carried forward as-is
- rationale: the docstring cites specific measured numbers (503/697, 601/697, 0.500, 0.897) that it simultaneously disowns as taken under a since-fixed reading bug ("must be re-measured before they are cited again")

Lines 276-298:
```python
MIN_LOW_FRACTION = 0.02
"""An evidence signal must report "no speech" in at least this fraction of buckets.

This, not the range, is the criterion that matters. Support only ever *removes* weight, so it
runs entirely on negative evidence: a signal that never says "no speech" cannot withhold
support from anything, and including it makes the whole measure inert.

Measured over 697 buckets of a real recording, four of seven candidate evidence signals never
once fell below 0.20 — ``acoustic_hnr`` (median 0.500), ``acoustic_loudness`` (0.897),
``acoustic_spectral_activity`` (0.940) and ``ast`` (0.728). Pooled alongside genuine VAD they
held support at 0.996 for every claimant. The two purpose-built voice detectors behaved as
detectors should: ``frame_segmentation`` reported no speech in 503 of 697 buckets and
``frame_brouhaha_vad`` in 601.

Range alone would not have caught this: ``acoustic_loudness`` swings 0.500 and ``ast`` 0.242
while neither ever reaches a negative verdict. Willingness to say no is the property, and it
is measurable on the run with no per-model judgement.

Caveat on those figures: they were taken while the screen read ``native_confidence`` undirected,
which cannot fall below 0.5 for any voter that took a direction — so part of what they measured was
the reading, not the voter. The screen now uses :func:`presence_probability`. The thresholds are
unchanged because the *property* they test is unchanged, but the per-voter verdicts above must be
re-measured before they are cited again."""
```

---

## types/data-model design (1 entry)

### F-80 (raised-by A-80) — `src/senselab/audio/workflows/audio_analysis/adaptive/types.py:3-23`

- rationale: TypedDict-vs-dataclass design rationale

Lines 3-23:
```python
**Why ``TypedDict`` and not dataclasses**, which is what tasks.md asked for:

1. **These records round-trip through JSON.** ``Region`` is written to
   ``rounds/<n>/regions.json`` by ``loop.py`` and read back by ``plot.py``,
   ``ls_final.py`` and the T039 harness. ``PlannedIntervention`` lands in
   ``final/iterations.json``. A dataclass would need ``to_dict``/``from_dict`` at
   every one of those boundaries, and the dict would remain the real wire format —
   so the dataclass would be a second representation to keep in sync, not a
   replacement.
2. **Candidates are built incrementally.** ``plan_round`` adds ``status``,
   ``error`` and ``intervention_id`` *after* constructing the record. A frozen
   dataclass cannot express that, and a mutable one gives up the guarantee that
   made it attractive.
3. **The actual defect class here is key typos and wrong value types**, not
   mutation — a rule reading ``region["core_strt"]`` or treating
   ``uncertainty_mass`` as a string. ``TypedDict`` catches exactly that, across
   every existing consumer, with zero runtime change and zero migration.

So this replaces ``dict[str, Any]`` annotations with checked shapes rather than
replacing dicts with objects. If a future change removes the JSON round-trip, the
dataclass version becomes worth revisiting.
```

---
