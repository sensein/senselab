# SPEECH branch

What the branch answers: **what was said, how was it said, who said it, and did the speaker do what
the task asked?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md);
shared rules are in [`branch-conventions.md`](branch-conventions.md); owed ground truth is in
[`branch-listening-sample.md`](branch-listening-sample.md).

Capabilities below are numbered **S1…S11** and are *not* the same as the nine `# Step` comments in
`speech.py`, which mark the node's present execution order. Where a capability corresponds to a
step, the step is named.

## The state of this branch

SPEECH is the largest node in the graph — 1,104 lines — and most of it runs. Its problem is the
opposite of VOICE's: it **measures a great deal and decides almost nothing**, because five decision
points are gated behind null config.

| `speech.py` step | line | state |
| --- | --- | --- |
| 1 transcript | `:517` | live; the only source of `FAIL` |
| 2 spans | `:571` | live |
| 3 corroborate | `:577` | measures SQUIM and YAMNet; **both votes inert** |
| 4 diarize | `:641` | live, scoped to the lexical hull |
| 5 separation | `:704` | **never selected** (`speech.separation_backend` null) |
| 6 identify | `:774` | word→speaker live to `:798`; `:800-869` enrollment-gated |
| 7 PII | `:900` | live; scan call at `:912` |
| 8 quality | `:991` | SQUIM at `:1006-1014`; reported, never gating |
| 9 non-target | `:1036` | measured; **null** |

## The tasks this branch serves

Declared families, not ground truth. `LEXICAL_SPEECH` (`families.py:31-55`) holds 21 families; those
carrying counts in the corpus profile:

| family | n | shape |
| --- | --- | --- |
| `harvard-sentences-list` | 13,705 | read, known stimulus |
| `free-speech` | 3,074 | spontaneous |
| `productive-vocabulary` | 2,910 | elicited |
| `cape-v-sentences` | 2,370 | read, known stimulus |
| `free-speech-v2` | 2,120 | spontaneous |
| `cape-v-sentences-v2` | 1,224 | read, known stimulus |
| `rainbow-passage` | 897 | read passage |
| `loudness` | 897 | **non-lexical** — three maximal shouts of "hey"; see [`branch-voice.md`](branch-voice.md) V6 |
| `picture-description` | 889 (+373 option1, +329 option2) | spontaneous, prompted |
| `story-recall` | 889 (+660 v2) | spontaneous, prompted |
| `loudness-v2` | 705 | **non-lexical** — "hey" normal then shouted; see [`branch-voice.md`](branch-voice.md) V6 |
| `caterpillar-passage` | 597 | read passage |
| `word-color-stroop` | 472 | read, timed |

These sum to 32,111 against 33,235 declared SPEECH; the table is truncated — the remaining 1,124 are
`LEXICAL_SPEECH` members below the profile's cut (`animal-fluency`, `cinderella-story`,
`open-response-questions`, `random-item-generation` and its v2).

**Read versus spontaneous is the division that matters**, and the declaration carries it as
`speech_type`. Roughly 19,265 declarations are read tasks with a known stimulus — the largest
measurable population in the corpus, and today nothing compares a transcript to it.

**`loudness` and `loudness-v2` are in `LEXICAL_SPEECH` (`families.py:41-42`) but the protocol calls
them `speech_type: "non-lexical"`.** Reading the sidecars settles what they are: a single syllable
— "hey" — shouted three times in v1, or spoken then shouted in v2. Neither is connected speech and
neither carries a stimulus text. An earlier version of this table described both as "read, varying
intensity", which was wrong on both counts.

The measurement they want is [`branch-voice.md`](branch-voice.md) V6's. **The family-set membership
and the protocol's own `speech_type` disagree**, and that discrepancy is worth resolving in
`families.py` rather than in either branch document.

CAPE-V remains the ordinary split: sentence conformance is S3's, voice quality is VOICE's V4.

### A recording routed here whose declared task is not speech

SPEECH routed 41,565 against 33,235 declaring a speech family, and its gate evidence is
`unavailable` on 0 recordings — the only branch with none. On a breath task containing an aside,
SPEECH transcribes it, diarizes it, scans it for PII and records what it found. That content is a
finding about the recording, not a fault of it.

## Capabilities

### S1 — Consensus transcript (**built**, step 1)

Reads `consensus_transcript` and its `word_ids` (`speech.py:517-534`); raises `LookupError` when
absent; splits lexical from bracketed. Emits `FAIL` when no lexical word survives (`:544-565`) — the
branch's only fail path.

### S2 — Speech spans (**built**, step 2)

Lexical word extents grouped where they touch by `group_extents_into_runs`
(`speech.py:573`; the function is `tasks/spans/api.py:223`). **Never the energy envelope** — SPEECH
proposes its own spans from the transcript, which made it the contract's `propose` precedent before
the contract existed. Writes `span` entities, `family: "speech"` (`speech.py:879-883`).

### S3 — Stimulus conformance by forced alignment (**not built; the branch's largest capability**)

**Question.** Did the speaker read the text they were given, and how?

**Use forced alignment, not the consensus aligner.** `align_sources` (`consensus.py:276`) returns a
`Consensus` — it *votes across sources* to produce a merged transcript. Wrapping the stimulus as a
hypothesis therefore yields a fused transcript, not a diff, and substitution, omission and insertion
rows cannot be read off it. It also raises below two hypotheses (`consensus.py:291`). An earlier
version of this document proposed exactly that and left the entry point as an open decision; the
decision is made here.

**`align_transcriptions` is in the inventory** (`tasks/forced_alignment/__init__.py:5`, implemented
at `forced_alignment.py:691`), including an MMS aligner path (`mms_fa.py`).

**Both instruments are needed, and they answer different questions.** A text-level diff of the
transcript against the stimulus says **what** differed — substitutions, omissions and insertions,
enumerated. Forced alignment says **where** and **how** — per-word intervals, acoustic scores,
phoneme durations and pause locations, and therefore rate and phrasing over the ~19,265 read
recordings.

**Alignment alone cannot enumerate the differences.** A forced aligner without skip arcs assigns
*every* stimulus word an interval whether or not it was spoken, so an omission surfaces only as a low
acoustic score — detected by a **score cut**, which is an operating point. An earlier version of this
document claimed "robust omission detection" and simultaneously claimed the capability needed no
threshold; both cannot be true. **The omission score cut is owed.** Substitutions and insertions
cannot be recovered from alignment at all, since the aligner is constrained to the stimulus.

**A forced-alignment per-word score is a typicality score, not a pronunciation score.** It measures
how well the audio matches the model's expectation for that word under a model trained on typical
speech — so it is low for impaired-but-entirely-correct productions. **It must never be presented as
pronunciation accuracy**, and on this corpus it will correlate with impairment rather than with
error.

**Emits.** One `deviate` assertion per departure, carrying the word's extent and its acoustic score.

**Fillers stay in their own channel.** The consensus brackets disfluency and non-speech tokens, so
they never enter the alignment. But **`filler` is a deviation only for read tasks**: on free speech,
picture description and story recall, filled pauses are the phenomenon being studied, and on
`word-color-stroop` hesitation and self-correction are the dependent variable. And **`[breath]` is
never a filler** — a breath during passage reading is how S4 measures breath-group structure.

**Attach recogniser agreement to every mismatch.** ASR error correlates with the impairments this
corpus exists to characterise, so `stimulus_mismatch` will peak on the most impaired speakers whose
reading was in fact perfect. The consensus aligner already produces inter-backend agreement per
column; carrying it on the deviation makes a mismatch where recognisers disagreed visibly a
different object from one where they agreed.

**CAPE-V's six sentences each load a different phonatory condition**, and pooling them discards the
instrument's design. Sentence boundaries fall out of this alignment for free — the cheapest
high-value addition in the five documents, over 3,594 recordings. **Nothing output may be presented
as a CAPE-V score**: CAPE-V is an auditory-perceptual instrument and these are acoustic correlates.

**Parameter-free? No — and an earlier version answered "yes" three paragraphs after establishing
otherwise.** The omission score cut is an operating point and it is owed; that is exactly the
contradiction this section criticises the previous revision for, reintroduced within the section.

What *is* parameter-free is the **enumeration**: the text diff's substitutions and insertions are
differences, not scores against a cut. What is not is **omission detection**, which needs the cut. So
the capability ships in two parts and only one is available now.

The aligner's internal weights are not a decision this branch makes.

### S4 — Connected-speech measures (**not built; the largest gap by population**)

**Question.** How was the connected speech produced?

**Roughly 25,000 recordings currently receive a transcript and nothing else.** The Rainbow and
Caterpillar passages exist to be measured — the Caterpillar was designed to elicit respiratory
phrasing — and both are transcribed and then dropped.

**What to measure**, all components present in the inventory:

- **speech rate and articulation rate**, and **phonation-time ratio** — `extract_speech_rate`
  (`praat_parselmouth.py:91`) returns all three, plus `pause_rate` and `mean_pause_dur`;

  **And it runs on FRCRN-enhanced audio like every other Praat scalar** — see
  [`praat-instrument-audit.md`](praat-instrument-audit.md) finding 0, which governs every measure in
  this capability.

  **The helper also hides three operating points, one data-dependent, and none in any config** —
  `silence_db = -25` (`praat_parselmouth.py:142`), `min_dip = 4` (`:149`) **dropped to 2 when the
  recording's own mean HNR is below 60** (`:154-155`), and `min_pause = 0.3` (`:159`). The HNR switch
  means syllable-detection sensitivity is conditioned on a voice-quality measurement of the recording
  being measured, across ~25,000 recordings of frequently dysphonic speakers. In practice mean HNR is
  far below 60 dB for any real recording, so on three synthetic probes — buzz 50.75 dB, buzz with
  noise 16.90 dB, pure sine 105.60 dB — the `min_dip = 2` branch always took the same side. **That
  has not been measured on the corpus**, and it is not dead code: if Praat returns undefined for the
  mean, `NaN < 60` is `False` and `min_dip` stays at the **stricter 4**, on exactly the recordings
  where pitch could not be measured. See [`branch-ddk.md`](branch-ddk.md) D2 and
  [`praat-instrument-audit.md`](praat-instrument-audit.md) finding 9.

  **Two further deviations from Praat**, both raising the same question: `min_pause` **0.3 s against
  Praat's 0.1 s**, so hesitation pauses at 0.3–0.4 s sit on the edge and `pause_rate` under-reads for
  halting speech; and minimum sounding interval **0.1 s against 0.05 s**, dropping short voiced
  fragments. And the helper's `to_pitch_ac` differs from Praat on six parameters, with the code's own
  comments saying *"can't find a reason for this value being different"* across five consecutive
  lines. **All of it owed** — see [`praat-instrument-audit.md`](praat-instrument-audit.md), and DDK D2
  reads the same helper.
- **pause count, duration and location**, and the **breath-group structure** they imply — but
  **breath groups must not be inferred from ASR bracketing**, which is not a breath detector and
  misses most audible inspirations in read speech. The instrument that would do it is AIRWAY's
  envelope-based breath-event detection (A5); connect them rather than substituting the bracket
  channel;
- **speaking F0 and F0 standard deviation** — `extract_pitch_descriptors` (`:448`);
- **intensity variability** within connected speech — `extract_intensity_descriptors` (`:515`).
  **VOICE V6 owns effort events; this owns intensity within connected speech.** They do not overlap:
  an earlier version said both applied to `loudness` and `loudness-v2` over different extents, but
  **there is no connected speech on either** — both are a single shouted syllable;
- **connected-speech CPP** — `extract_cpp_descriptors` (`:706`). **Withheld on the same grounds V4
  withholds it**, and more strongly: the findings are properties of the function, not of a branch, so
  every caller inherits them. The `> 4` cut is selection on the dependent variable, the peak search
  is capped at 330 Hz, and **finding 3's 70% vuv inflation scales inversely with voiced-run length —
  so it is *worst* on connected speech**, this capability's material, and mildest on the sustained
  vowel where it was first suppressed. S4's population is roughly 25,000 recordings against V4's
  5,113. See [`praat-instrument-audit.md`](praat-instrument-audit.md) findings 2–5, and step 4 for
  the replacement.

For the spontaneous tasks additionally:

**Lexical diversity — but not a bare type-token ratio.** TTR is **length-dependent**: over 30 s and
over 3 minutes it is not the same measure. Use **MATTR or MTLD**, or report TTR only with its token count. **MATTR carries a window length,
which is owed** — naming the measure without it repeats the length-dependence the paragraph exists to
point out. In a document set that mandates support counts, this is the measure whose support count *is*
the confound.

**Disfluency rate — with the caveat that it under-counts where disfluency is greatest.** The
bracketed channel captures **filled pauses**, but part-word and whole-word repetitions in stuttering
are usually emitted as **lexical tokens**, not brackets. So a bracket-derived rate is lowest on the
speakers with the most disfluency — the same impairment-correlated ASR bias S3 handles for
`stimulus_mismatch`, and unhandled here. Attach recogniser agreement, as S3 does.

**Resonance and nasality appear nowhere in this document set**, and that is a decision rather than an
oversight: hypernasality is a first-order dysarthria dimension, and measuring it **requires
nasometry — a second channel, not in the inventory.** Acoustic proxies exist and none is reliable
enough to report unqualified.

**Pause structure is a measurement, not a deviation.** This capability is what replaced the
`off_task_extent` definition an earlier version of this document carried — *"a region carrying no
lexical speech where the task asked for reading"* — which would have made every inter-phrase pause
in a passage reading a deviation and needed an undeclared minimum duration to avoid firing thousands
of times per recording.

**Discourse-content scoring keys are not in the inventory.** Story recall and picture description
have established content-scoring instruments; this document does not invent them.

### Two task families have standard norm-free measures and currently get nothing

**`productive-vocabulary` (2,910) is a verbal fluency task.** What is genuinely key-free is **total
items, unique items, and the inter-response-interval series** with its first-half versus second-half
slope — the last being how retrieval slowing shows itself, and the informative one. All computable
from a time-aligned transcript.

**But *valid* items are not key-free**: scoring an item as belonging to the category needs a category
lexicon, which is not in the inventory. An earlier version listed "total and unique **valid** items"
as needing no key; only total and unique are.

**`word-color-stroop` (472) is a response-latency task, and two of its three measures need data the
audio does not carry.** Latency is measured **from stimulus onset**, which is the app's per-item
presentation timestamp — not in the inventory, and not recoverable from the recording. Error rate
needs the item list, likewise absent. What the audio alone supports is the **inter-response
interval** series, which is not the same measure.

**Both inherit the ASR bias.** A recogniser drops items on impaired speakers, so an item count is
itself impairment-correlated. **Attach recogniser agreement per item**, exactly as S3 does for
`stimulus_mismatch`.

Both are distinguishable from story recall and picture description, where declining to score is the
right call because the instruments require content keys. These two do not.

### S5 — Speaker count (**built, scoped wrong**, step 4)

Reads pyannote community-1 over `[first word start, last word end]` only (`speech.py:641-646`). A
second speaker outside the lexical hull — before the participant starts, after they stop, or in a
pause — is invisible. The contract moves diarization to PREPROCESS as a whole-file shared derivative,
which fixes the scope and makes it available to every branch.

**Emits** a `counts` entry `speaker_count` carrying `found` and `declared` from the declaration's
`targeted_speaker_count`, asserting no discrepancy.

**Diarization over-splits on within-speaker voice-quality change** — which several of these tasks
explicitly instruct. Report pairwise embedding similarity beside the count so an over-split is
visible as one.

The second diarizer never runs: `speech.second_diarizer` is null (`default.yaml:166`), so
`second_record` is always `"not_consulted"` (`speech.py:685-703`).

### S6 — Word→speaker attribution (**built**, step 6)

Always-run through `speech.py:774-798`, `verb: "attribute"`, one assertion per word (`:788-793`) —
the highest-volume assertion verb in the store, and not one of the contract's five. Relevant to the
contract's piece 7, which widens REPORT's assertion read by verb.

`:800-869` is the enrollment path. `speech.enrollment_model` (`default.yaml:171`) and
`speech.target_match_cosine` (`:167`) are both null, so an enrollment supplied without an override
lands in `_flag_before_measuring` (`speech.py:460-470`). A cosine threshold for speaker identity is
owed ground truth and cannot be fitted against declarations.

### S7 — PII (**built**, step 7)

One `scan_for_pii` over the consensus and each recogniser's own transcript (`speech.py:900-989`,
call at `:912`), producing `pii` entities and per-word `verb: "label", label: "pii"` marks at
`:961`. `redact.py:237` selects on exactly that verb/label pair — which is why the contract's
proposal to rename `label` to `mark` was withdrawn. `pii.required_detectors` is populated: the one
decision point in this branch that is neither null nor inert.

### S8 — Speech quality (**measured, inert**, steps 3 and 8)

Per-span SQUIM (`stoi`, `pesq`, `si_sdr`) at `speech.py:1006-1014`, `disruptions`, and a per-span
`proximity` measurement (`:1036-1068`).

Both votes are inert: `yamnet_vote` is `"unavailable"` because `taxonomy.speech_labels` is null
(`default.yaml:181`, read at `speech.py:589`); `squim_vote` is `"not_evaluated"` because
`speech.speech_test_stoi_floor` and `speech.speech_test_si_sdr_floor` are null
(`default.yaml:168-169`, read at `:604-605`).

**Neither floor can be fitted.** [`branch-quality.md`](branch-quality.md) takes up what should read
these numbers instead — including that SQUIM penalises atypical voices and is out of domain on
coughs, sustained vowels and DDK trains.

### S9 — Non-target speech (**gated behind null config**, step 9)

`nontarget_speech_s` is always `None`: `speech.nontarget` (`default.yaml:174`) has all three legs
null (`:175-177`). Three thresholds, all owed.

### S10 — Separation (**never selected**, step 5)

`speech.separation_backend` is null (`default.yaml:172`), so `separation_state` is `"not_selected"`
whenever two or more speakers are found (`speech.py:704-772`). Both backends unreached.

### S11 — Language and truncation checks (**not built**)

Language mismatch against the declaration's `language`, and truncation — the recording beginning or
ending mid-utterance — are both deviations the read tasks can produce and neither is computed.
Repeat readings of the same stimulus likewise.

## Deviations

| type | evidence |
| --- | --- |
| `stimulus_mismatch` | an aligned word that is not the word the stimulus expected, carrying its acoustic score and the recogniser agreement at that column (S3) |
| `filler` | a bracketed disfluency where a **read** task expected lexical content; never `[breath]`; not emitted for spontaneous tasks (S3) |
| `truncation` | the recording begins or ends mid-utterance (S11) |
| `language_mismatch` | the transcript language differs from the declaration (S11) |
| `repeat_reading` | the stimulus was read more than once (S11) |

**A deviation is not evidence of a bad recording.** `filler`, `stimulus_mismatch` and
`repeat_reading` are produced **because of** stuttering, aphasia, apraxia of speech and Parkinson's
disease — they are the finding, not a fault of the recording. A consumer that filters on them is
filtering on impairment. See [`branch-conventions.md`](branch-conventions.md).

**An omitted word has no extent.** The contract requires every deviation to have one, so an omission
is recorded as a **zero-width extent at the alignment point** — the position in the signal where the
expected word should have been. That keeps it a deviation rather than requiring a separate shape.

**`off_task_extent` is withdrawn from this branch** — see S4.

`speaker_count` is a `counts` entry.

## Quality covariates

Every acoustic measurement S4 and S8 emit carries the quality covariates of its own extent, per
[`branch-conventions.md`](branch-conventions.md).

## What exists today

| capability | status |
| --- | --- |
| S1 transcript | built |
| S2 spans | built |
| S3 stimulus conformance | **not built** — `align_transcriptions` exists, nothing calls it with a stimulus |
| S4 connected-speech measures | **not built** — ~25,000 recordings get a transcript and nothing else |
| S5 speaker count | built, scoped to the lexical hull |
| S6 attribution | built; enrollment gated |
| S7 PII | built |
| S8 quality | measured, both votes inert |
| S9 non-target | gated behind null config |
| S10 separation | gated behind null config |
| S11 language and truncation | not built |

## A branch `FAIL` is an absence of detected content

`SPEECH: FAIL` fires only when the consensus produced no lexical word (`speech.py:544-565`). Since
recognisers fail most often on the most impaired speech, that absence is itself impairment-correlated
— the transcript's silence is not the speaker's. See
[`branch-conventions.md`](branch-conventions.md).

**A second population fails for an unrelated reason, and the two must not be pooled.** `loudness` and
`loudness-v2` — **1,602 recordings** — are in `LEXICAL_SPEECH` and therefore routed here by
declaration, while their content is a single non-lexical monosyllable. They will `FAIL` as a
**task-content artefact**: there was never lexical speech to find. That is a fact about the family
set, not about the speaker or the recogniser, and a consumer treating all SPEECH failures alike
mistakes 1,602 correctly-performed recordings for failed transcription.

## What the branch emits

```
spans        family: "speech", one per run of touching lexical words
assertions   attribute (one per word), label/pii, flag, deviate
entities     speaker (one per diarized segment), pii
measurements squim, disruptions, proximity, connected-speech measures (S4)
counts       speaker_count {found, declared}
verdict      { speaker_count, diarization, words_n, speech_s, nontarget_speech_s,
               pii, second_diarizer, separation, flags }
             plus target_speaker and enrollment_id when an enrollment was supplied
```

**The verdict's basis, exactly** (`speech.py:1095-1099`): `FAIL` only from the no-lexical-words row
at `:544-565`; `FLAG` when any flag accumulated; `PASS` otherwise, with
*"words, spans, speakers and quality are in the store"*. The `detail` payload is built at
`:1074-1092`.

An earlier version of this document gave the emit block as
`{lexical_n, bracketed_n, speakers, …, separation_state}`. None of `lexical_n`, `bracketed_n` or
`speakers` exists, and the key is `separation`, not `separation_state`.

## Out of scope

Any normative judgement on a deviation. Voice quality on CAPE-V, which is VOICE's V4. Discourse
content scoring. Any refit of a STOI, SI-SDR, cosine or non-target threshold against declared
families.

## Unresolved

- Whether S3 aligns against the stimulus directly or against a normalised form of it.
- Whether `language` is in the contract's declaration keys.
