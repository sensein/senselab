# QUALITY

What this node answers: **does the store contradict itself, and is anything true of the recording
that is a property of the room or the device rather than of the content?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md);
shared rules are in [`branch-conventions.md`](branch-conventions.md). This document is grounded in
the code and in [`dag.md`](dag.md); where those disagree the code wins and the disagreement is named.

## QUALITY is not a branch, and the difference is load-bearing

**It reads stored records only.** Entities and their attributes. It decodes no audio, opens no
sidecar and re-derives nothing — every amplitude it compares was measured by the node that held the
signal (`quality.py:8-12`). That is what lets `scripts/extend_quality.py` run it over a finished
store. A capability needing the waveform does not belong here; it belongs in PREPROCESS, with
QUALITY reading what PREPROCESS wrote.

**It runs on every path PREPROCESS completed.** `run.py:310` calls it unconditionally after the
branch loop — whatever routing selected, and whether or not routing raised, since `selected` falls
back to the empty set when `routed is None` (`run.py:296`). So it is the one node positioned to see
every branch's output at once.

**It is not terminal**, though two places say it is. REDACT runs after it (`run.py:311-317`).
`dag.md` has been corrected; `vocabulary.py:29`'s docstring still reads *"The terminal node every
recording reaches"* — a code follow-up, noted here and not edited from this document.

**It is not routed.** No gate selects it and `KIND` is `None` (`quality.py:56`), so the question the
four branch documents must answer — what it does on a recording routed to it whose declared task
belongs elsewhere — does not arise. QUALITY sees everything.

### Where it sits, and whether its `contest` is the contract's

The contract's four stages are PREPROCESS → SCREEN → BRANCHES → VERDICT, and QUALITY is in none of
them. **It is a fifth position: after the branches, before REDACT and VERDICT, reading the whole
store.** The four stages describe the routing-and-measurement path; QUALITY is a cross-cutting audit
over its result. ADMIT, REDACT, REPORT and FIGURE sit outside the four for the same kind of reason.

**Its `contest` is the contract's `contest`.** The contract defines it as *"the span does not carry
what was proposed"*. A clip span proposes the signal reached its ceiling; an unclipped sample louder
than that ceiling says it did not — an instance of the definition under an amplitude test rather
than an absence test. `quality.py:291` writes `verb: CONTEST_VERB`, `CONTEST_VERB = "contest"`
(`:62`), the same wire value AIRWAY writes.

**This is the evidence that the verbs are store-wide.** QUALITY is not a branch and it contests.
What is branch-specific is the *family* a `propose` writes, not the verb vocabulary.

## Capabilities

### Q1 — Clip consistency (**built**)

**Question.** Does any clip span assert a ceiling that a sample outside every clip span exceeds?

**Reads.** PREPROCESS's `clip` spans for their extents and the `clip_amplitude` measurement for
every amplitude including per-span levels (`quality.py:8-12`).

**Computes.** For each clip span with a stored level, whether the whole-file unclipped peak exceeds
that level by more than `quality.clip_contradiction_margin` of it (`quality.py:271`).

**Emits.** One `contest` assertion per contradiction, `wasDerivedFrom` the span it contests
(`quality.py:288-302`). **PREPROCESS's spans are never invalidated here** — the store is append-only
and the span is PREPROCESS's reading.

**Its expected count is zero, and that is the point.** `_clip_spans` applies the same comparison at
detection and never writes a candidate it contradicts, so this is the *audit* of that rule
(`quality.py:31-36`). It deliberately reads nothing about whether the rule ran.

**One trap, recorded in `dag.md`.** The margin is QUALITY's own `config.require`
(`quality.py:239`); only the edge guard comes off the measurement (`:246`). They agree today because
both read the same key, so an override changed between a PREPROCESS run and a later
`extend_quality.py` pass would audit spans against a margin they were never detected under.

**Parameter-free?** No, but `quality.clip_contradiction_margin` is a tolerance on a physical
comparison rather than a classification cut, and it is derived and recorded in `data/`.

### Q2 — Effective bandwidth (**not built; highest-value addition, and it must not measure the vowel**)

**Question.** What frequency band does the *capture chain* carry?

**Effective bandwidth is not the declared sample rate.** A 16 kHz file may carry nothing above 4 kHz
— codec, microphone, or noise suppression. CPP, spectral slope and tilt, spectral moments, HNR and
F3/F4 are **all bandwidth-dependent**, so an undeclared band limit turns device class into a
pseudo-finding.

**Measuring it as an LTAS roll-off measures the content, not the device.** A sustained /a/ has little
energy above 5 kHz **because vowels do not** — and this covariate would be computed on
`prolonged-vowel` and `maximum-phonation-time`, 5,113 recordings, precisely those whose CPP and HNR
it exists to qualify. A wrong covariate is worse than none: it does not merely mislead, it **explains
away real findings** as device artefacts.

**Detect the cliff, not the roll-off.** A codec or anti-alias filter produces a spectral edge —
tens of dB over a fraction of an octave, above which the level sits at the noise floor with near-zero
variance. A vowel's natural spectral decay has neither the slope nor the variance collapse. The
discriminating feature is the **abruptness and the flatness above it**, not the level at any
frequency.

**Prefer broadband content.** Fricatives, coughs and background segments excite the band that matters;
sustained vowels do not. Where a recording has no broadband content, the measurement is
**unavailable** rather than estimated from a vowel.

**Not threshold-free.** An earlier version said it "needs no threshold to report". A roll-off or a
cliff is defined relative to something — a reference level, a slope in dB/octave, a variance
criterion. All are declared as conventions per
[`branch-conventions.md`](branch-conventions.md).

**It is file-level, and that is consistent.** Bandwidth is a property of the capture chain, not of a
span, so it is computed once and referenced by every extent. An earlier version of
`branch-conventions.md` required all covariates per-extent, which contradicted this; that document now
distinguishes per-extent covariates (clipping, SNR, support count) from file-level ones (bandwidth,
AGC and noise-suppression signatures). **An AGC signature is not definable on a 400 ms span at all.**

**Where it runs.** The spectrum needs the waveform, so **PREPROCESS computes it and QUALITY reads
it** — the division that keeps Q1 audio-free.

### Q3 — Background content (**not built; depends on gap spans becoming background**)

**Question.** What is in the regions no proposer claimed, and is any of it a property of the room?

**`dag.md` states the boundary**: the branches answer "is the content the protocol asked for
present"; a mains hum is a property of the room, and questions of that shape are QUALITY's. It also
records that the mechanism cuts against the thesis — gap spans carry no `family` key
(`preprocess.py:1570-1578`), `airway.py:198` selects `family is None`, and **a gap span can carry the
whole AIRWAY verdict alone** because `labelled_n` separates `pass` from `fail`
(`airway.py:376-383`, detail at `:393-398`).

So background is AIRWAY's today, by accident of a missing attribute. If the contract's part (b) types
gap spans as background, they leave AIRWAY's selector and **QUALITY becomes where their content is
concluded on**.

**What it would read.** The gap spans' own per-span classifier measurements — they are in
`state["span_ids"]` and the per-span classifiers run over them (`preprocess.py:1583`, `:1587`;
`_span_hear` at `:1861-1865`), so each already carries HeAR windows and needs no new computation.

**What it would conclude.** That a background region carries a label, and which. Not a verdict on the
recording: a hum, a keyboard, a passing vehicle is an observation about the room.

**Unmeasured.** How often a gap span currently decides an AIRWAY verdict (`dag.md`). That is a count
over the corpus rather than a fit against it, so the no-refits rule permits it.

### Q4 — Exact-duplicate detection (**not built; free**)

**Question.** Has this audio been submitted before?

ADMIT already records `checksum_sha256` on the `recording` stream entity. Two recordings under
different task ids with the same checksum are the same audio.

**Re-submission of a previous recording under a new task id is a known failure mode of app-based
collection at this scale**, and it invalidates any analysis that misses it — a duplicate inflates
whatever it is counted in and, if it crosses task families, corrupts exactly the declared-family
comparisons the corpus is scored against.

**Emits.** A file-level assertion naming the other recording. **Free**: the digest exists, and the
comparison is equality.

**It is a lower bound.** A checksum catches byte-identical duplicates only; a re-encode on upload —
different container, different bitrate, a resample — defeats it entirely. So a zero count means "no
*exact* duplicates found", not "no duplicates". Near-duplicate detection needs an audio fingerprint,
which is not in the inventory.

**Where it runs.** Cross-recording comparison is outside the single-recording store, so this belongs
to a corpus-level pass rather than to a per-recording QUALITY invocation. Recorded here because it is
QUALITY-shaped, with the placement named rather than assumed.

### Q5 — Acquisition consistency (**not built; depends on the declaration**)

**Question.** Does the recording match what the protocol says was recorded?

**Reads.** The declaration's `declared_duration_s`, `sample_rate`, `channels` and `microphone` —
those are the contract's `metadata` key spellings, not the BIDS sidecar's `recording_duration` /
`audio_sample_rate` / `audio_channel_count` / `recording_microphone`, which an earlier version of
this document used and which would send an implementer to keys that do not exist. Compared against
the `stream` entity ADMIT wrote, carrying `sampling_rate`, `channels`, `size_bytes` and
`checksum_sha256`.

**Computes.** Equality for sample rate and channel count; a difference for duration.

**Emits.** A `counts` measurement carrying `declared` and `found` per field, **asserting no
discrepancy**.

**Parameter-free?** Sample rate and channel count are exact. Duration needs a tolerance, and a
tolerance is a number nobody has — so QUALITY states the difference and declines to own a cut. That
refusal is the model the withdrawn `off_task_extent` definitions in SPEECH, VOICE and DDK should
have copied.

**`recording_input_gain` is not used here.** A gain can be set anywhere and the signal can still
clip; it does not predict clipping.

### Q6 — Cross-branch contradiction (**not built; in remit**)

**Question.** Do two branches say incompatible things about the same extent?

QUALITY is the only node positioned to ask: it runs after every branch (`run.py:310`) and reads the
whole store. VERDICT cannot — it reads only verdict entities, `branch_decision`s and
`ruleset_routing` (`verdict.py:218-220`, *"This node reads nothing else"*).

**What makes this hard.** Two branches labelling the same extent differently is usually *not* a
contradiction: a cough during a sentence is genuinely both airway content and an interruption of
speech, and the contract's premise is that content the task did not ask for is still content. A real
contradiction needs a pair of claims that cannot both hold, and enumerating those pairs is a domain
question nobody has answered. **In remit, not yet specifiable** — recorded so the next
QUALITY-shaped check is not built somewhere it does not belong.

### Q7 — SQUIM, described but not concluded on (**not built**)

Per-span SQUIM is written by PREPROCESS over the general spans and again by SPEECH over its own
(`speech.py:1006-1014`). An earlier version of this document cited `speech.py:959-1010` for it; that
range is PII mark code and step 8 begins at `:991`. The same wrong range appeared in
`branch-speech.md` — it propagated between the two documents.

**QUALITY may read it; it must not conclude on it.** `speech.speech_test_stoi_floor` and
`speech.speech_test_si_sdr_floor` are null (`default.yaml:168-169`) and neither can be fitted, so
there is no cut — and inventing one here repeats the error SPEECH avoided by leaving `squim_vote` at
`not_evaluated` (`speech.py:631`).

**What it can do without a cut** is report the distribution. But two qualifiers must travel with it:

- **SQUIM penalises atypical voices.** It was trained to predict perceptual quality of speech, and a
  dysphonic voice scores low for reasons that are the *signal*, not the noise.
- **It is out of domain on coughs, sustained vowels and DDK trains.** Reporting one number across
  span families pools measurements whose validity differs.

So if QUALITY reports SQUIM it **stratifies by span family and marks the out-of-domain ones**.

**Owed.** Any intelligibility floor, and it should come from listening rather than fitting — see
[`branch-listening-sample.md`](branch-listening-sample.md).

## What QUALITY does not emit

**No deviations.** A deviation is a departure from what a *task* asked for, and QUALITY holds no
task declaration — it holds the store. Q5's acquisition comparison is the nearest thing and is a
`counts` measurement, because it compares the recording to its own metadata rather than the
performance to its instruction.

**No spans.** QUALITY proposes nothing; it audits what others proposed.

## What exists today

| capability | status |
| --- | --- |
| Q1 clip consistency | **built**; expected count zero |
| Q2 effective bandwidth | **not built**; highest-value addition; LTAS in PREPROCESS, read here |
| Q3 background content | **not built**; blocked on gap spans being typed background |
| Q4 duplicate detection | **not built**; free, but corpus-level rather than per-recording |
| Q5 acquisition consistency | **not built**; needs the declaration |
| Q6 cross-branch contradiction | **not built**; in remit, not yet specifiable |
| Q7 SQUIM description | **not built**; must stratify and must not conclude |

## What the node emits

```
assertions   contest, one per contradicted clip span
verdict      { signal, preceded_by, clip_spans_n, checked_n, unmeasurable_n,
               contradicted_n, unclipped_samples_n, unclipped_peak,
               unclipped_peak_time_s, clip_contradiction_margin,
               clip_edge_guard_samples, contradictions, flags }
```

**The verdict's basis, exactly** (`quality.py:312-317`):

- `FLAG` when any contradiction was found;
- `PASS` with *"no clip span over the recording; nothing to contradict"* when nothing was measurable;
- `PASS` with *"no clip span sits below an unclipped sample"* otherwise.

`preceded_by` names the nodes whose verdicts were live when QUALITY read the store
(`quality.py:115`, recorded at `:329`).

**What it refuses.** An absent dependency is an operational fact, not a finding: clip spans with no
clip-amplitude measurement raise, and the runner records the node `ERRORED` (`quality.py:14-16`). A
contradiction QUALITY *can* measure is always a finding and never a raise.

## Out of scope

Anything needing the waveform — that belongs in PREPROCESS. Any normative quality judgement.
Withdrawing another node's reading. Any threshold fitted against declared families.

## Unresolved

- Q4's placement: a corpus-level pass rather than per-recording QUALITY.
- Q6's contradiction pairs.
- `vocabulary.py:29`'s docstring still calls QUALITY terminal — a code follow-up.
