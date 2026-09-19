# Corpus speech census — the 2026-09-08 triage run over 62,578 recordings

What the provenance stores under `/orcd/scratch/bcs/002/satra/triage_full_20260908/out/` actually
hold, read on 2026-09-19. The scan scripts are on ORCD scratch at
`/orcd/scratch/bcs/002/satra/checks_20260916/census/`; their outputs, including everything derived
from transcript text, are under `/orcd/scratch/bcs/002/satra/checks_20260916/pii_census/` (mode 700).

**Read the first section before any number below it.** The run this census reads did not run the
branches, so three of the things the census was asked for are not in the stores at all, and one of
them was measured afresh rather than read.

---

## 0. What ran, and what therefore cannot be answered from these stores

Every one of the 62,578 recordings named in
`/orcd/scratch/bcs/002/satra/clipfix_20260913/manifest_all.jsonl` has a `run/store.jsonl`; all
62,578 parse; none errored. Counting the nodes that have at least one activity in a store:

| node | recordings with an activity |
|---|---:|
| ADMIT | 62,578 |
| PREPROCESS | 62,521 |
| TAXONOMY | 62,521 |
| QUALITY | 60,202 |
| routing | **0** |
| AIRWAY | **0** |
| SPEECH | **0** |
| VOICE | **0** |
| REDACT | **0** |
| VERDICT | **0** |

This is not attrition. The run's own driver says so in its first line —
`/orcd/scratch/bcs/002/satra/triage_full_20260908/driver.py`:

> ADMIT + PREPROCESS + TAXONOMY + figure over ONE recording, then verify the store. […] **Branches
> are not run.**

The consequences are exact, and they are the three the brief asked for:

- **PII.** The scan is step 7 of the SPEECH branch. It never ran. There are **0 `pii` entities and
  0 `pii_scan` measurements** across all 62,578 stores. There is nothing to aggregate, nothing to
  recover strings from, and no detector distribution to report.
- **Task conformance.** Conformance is written by a branch as a `branch_report` entity. There are
  **0 `branch_report` entities** in the corpus. No recording carries a conformance of True, False or
  UNDETERMINED — not "UNDETERMINED for all"; the field does not exist.
- **Deviations.** A deviation is an `assertion` with `verb: "deviate"`. There are **0** of them,
  `truncation` included.

The brief framed these as "as of the 2026-09-08 run, and a re-run is owed". That framing is too
generous to the data: there is no 2026-09-08 conformance or deviation reading to be superseded by a
re-run, and the ~132 commits of change since (DDK dissolved into SPEECH, its instrument replaced,
SPEECH's breath detection removed) are beside the point. **The re-run is not owed because the
numbers are stale. It is owed because the numbers do not exist.**

The same check over the other store trees on this scratch — `triage_battery_prov_20260908` and
`triage_battery_20260908`, 388 stores each — finds the same three nodes and no `pii` entity. No
corpus-scale run of the branches exists on this filesystem.

### The one thing that was measured rather than read

Section 1 below reports a **fresh PII scan run today over the transcripts those stores hold**. It is
not a read of the corpus and is labelled as such throughout. It answers the owner's actual question
— is the detector over-reaching, and on what — which the stores cannot.

### Definitions used

"Family" is the project's own `routing_analysis.families.task_family`: the `task-` id with trailing
numeric segments collapsed. The manifest's 796 distinct task ids collapse to **48 families**. The
brief's regex yields the task id; the family is the coarser grouping the branches' expectation tables
are keyed on, so it is what is reported here.

---

## 2. Multiple speakers

### Which measurement carries it, and why this one

Three candidates were checked:

- `enhanced_diarization` and `residual_diarization` — PREPROCESS measurements, one per stream,
  written by `pyannote/speaker-diarization-community-1` with `exclusive: false` (the overlapping
  view, so `overlap_s` and `max_concurrent_speakers` are readable rather than zero by construction).
  Each carries `n_speakers`, `n_segments`, `per_speaker_s`, `speech_s`, `overlap_s` as entity
  attributes, with the segment table in an `.npz` sidecar.
- SPEECH's report detail — **not available.** SPEECH never ran (§0), so there is no per-speaker
  attribution, no enrolment, no target-speaker resolution, and no `nontarget` flag on any span.

**`enhanced_diarization` is used as the headline** because it is the only whole-corpus speaker
measurement that exists, and because the enhanced stream is the one the diarizer was pointed at for
voices. `residual_diarization` is reported beside it as a second, independent reading rather than as
corroboration — see the disagreement below.

Coverage: 61,886 of the 62,521 preprocessed recordings carry both diarizations (98.98 %). The 635
that do not are spread across 44 families (harvard-sentences-list 133, free-speech 38,
respiration-and-cough-fivebreaths 34, …) with a duration profile indistinguishable from the corpus;
the store records no reason, so this is "the instrument did not run", cause unknown. The 57 ADMIT
failures never reached PREPROCESS and are excluded from every denominator here.

### The count

| speakers found (enhanced) | recordings | share of 61,886 |
|---|---:|---:|
| 0 | 12,122 | 0.196 |
| 1 | 47,025 | 0.760 |
| 2 | 2,737 | 0.044 |
| 3 | 2 | 0.00003 |

**2,739 recordings (4.42 %) have a speaker count above one.** Zero is a measurement, not an absence
— it is the ordinary outcome on a breath or cough task, and 9,557 of the 12,122 zeros are in
airway-eliciting families.

The per-family table is **Table B** at the end of this document.

### This number should not be used as it stands

Three things in the data argue that most of the 2,739 are not a second person.

**The second speaker is almost always a fragment.** Over the 2,739 multi-speaker recordings, the
secondary speaker's total speech is:

| p05 | p25 | p50 | p75 | p95 |
|---:|---:|---:|---:|---:|
| 0.07 s | 0.24 s | 0.44 s | 1.00 s | 3.56 s |

2,064 of 2,739 have a secondary speaker under 1 s; 2,668 under 5 s. Only **71 recordings in the
whole corpus (0.11 %)** have a second speaker with at least 5 s of speech.

**The rate tracks duration, not task.** By duration decile of the recording:

| decile | duration range | ≥2 speakers | secondary ≥5 s |
|---|---|---:|---:|
| 1 | 0.09–3.34 s | 0.026 | 0.000 |
| 5 | 5.57–7.32 s | 0.026 | 0.000 |
| 8 | 14.79–23.47 s | 0.043 | 0.001 |
| 9 | 23.47–30.02 s | 0.064 | 0.002 |
| 10 | 30.02–332.56 s | 0.134 | 0.007 |

A flat 2.6 % through the first six deciles, then a 5× climb in the last one. The families with the
highest multi-speaker rates — cinderella-story 0.264, caterpillar-passage 0.197, story-recall-v2
0.183, random-item-generation 0.146 — are the longest families, not the ones a clinician would be
most likely to speak in.

**The two streams disagree almost completely.** Enhanced and residual are two readings of the same
recording:

| | residual ≥2 | residual ≤1 |
|---|---:|---:|
| **enhanced ≥2** | 475 | 2,264 |
| **enhanced ≤1** | 3,049 | 56,098 |

Of the 5,788 recordings where either stream reports more than one voice, the two agree on 475 —
8.2 %. A second voice loud enough to be a real second person should survive enhancement and should
not appear only in the residual.

**The known caveat applies here and is not a hedge.** The project's own measurement record
(`measurements-2026-08-17-span-probe.md`, and the memory item "Speaker embedding batching") is that
heterogeneous-duration batching corrupts short-span speaker embeddings, and that within/between
similarity on short spans is dominated by duration rather than identity. The median secondary
speaker here is 0.44 s. That is squarely inside the regime where the instrument is known not to
separate identity from duration. **The defensible reading of this corpus is: 71 recordings (0.11 %)
carry a second voice with enough speech to be judged, and the remaining 2,668 are a diarizer
decision on spans too short for the instrument to be trusted on.**

### Cross-tab against PII

**Cannot be done.** There are no PII findings in the stores (§0). The hypothesis — that a second
voice in a clinical recording is often a clinician, and that identifying speech comes from there — is
exactly the thing this corpus cannot test, and the fresh scan in §1 cannot test it either, because
the fresh scan has no speaker attribution: it reads a transcript, and the transcript does not say who
said what. Joining a PII finding to a speaker needs SPEECH's own word-to-speaker resolution, which
requires the branch to run.

The nearest thing the stores support is multi-speaker against *lexical content*, which is a much
weaker question:

| | ≥2 speakers | ≤1 speaker | rate |
|---|---:|---:|---:|
| has lexical content | 2,378 | 42,540 | 0.053 |
| no lexical content | 361 | 16,607 | 0.021 |

Recordings with words in them are 2.5× more likely to be called multi-speaker. That is consistent
with a real second voice and equally consistent with the duration confound above (lexical families
are the long ones), and nothing in the store separates the two.

---

## 3. Task verification, and the three recording-quality questions

### 3.1 Conformance and deviations — not answerable from these stores

**Dropped, not estimated.** Conformance is written by a branch as a `branch_report` entity and a
deviation as an `assertion` with `verb: "deviate"`. There are **0 of each across 62,578 stores**, so
there is no True / False / UNDETERMINED to tabulate and no deviation type to count. No proxy is
substituted here and none should be: inferring conformance from a derivative would be inventing a
reading the run never made. **These figures are owed from the corpus re-run now being staged.**

### 3.1b What the stores record instead, which is not a conformance

Each deciding node's own verdict, which is about the node's work rather than about the
participant's:

| node | recordings with a live verdict | outcomes |
|---|---:|---|
| ADMIT | 62,578 | 62,521 pass, 57 fail |
| PREPROCESS | 62,521 | 62,521 pass |
| TAXONOMY | 62,521 | **62,521 flag, 0 pass** |
| QUALITY | 59,936 | 59,936 pass |

TAXONOMY flagged **every single recording**, and the mechanism is in the store. Each `kind` entity's
evidence lines read:

| kind | line | state | evidence present |
|---|---|---|---:|
| airway | acoustic | unavailable | 0 / 62,521 |
| airway | health_acoustic | unavailable | 0 / 62,521 |
| speech | acoustic | unavailable | 0 / 62,521 |
| speech | lexical | unavailable | 45,368 / 62,521 |
| voice | phonation | unavailable | 0 / 62,521 |

Every line is `unavailable` in every recording, including the 45,368 where the lexical line had word
evidence attached. The PREPROCESS verdict says why: `windows.yamnet.default_threshold`,
`windows.ast.default_threshold`, `windows.hear.default_threshold` and `voice.f0_range_hz` are null
in the packaged config — "null because nobody has measured it" — and that absence is recorded in all
62,521 PREPROCESS verdicts. So all three kinds are `uncertain` on all 62,521 recordings and the fold
flags. **This is the "asked for an operating point nobody has measured" path firing on 100 % of the
corpus, recorded correctly.** It means TAXONOMY contributed no discrimination at all, and a re-run
that does not first settle those four values will reproduce this table exactly.

Instrument-did-not-run counts, from the PREPROCESS verdict's `absent` map and worth keeping separate
from findings: `hear_scores` absent in 2,003 recordings, `enhanced_hear`/`residual_hear` in 2,005,
`span_hear`/`span_yamnet`/`squim` in 622, `ast_scores` in 55, `residual` in 2.

**266 QUALITY verdicts are flags, and all 266 are invalidated.** QUALITY ran on 60,202 recordings —
it is in these stores, added by a later `clip_consistency` pass, not absent — and concluded 59,936
pass and 266 flag, every flag a `clip_above_unclipped_sample` contradiction (a clip span sitting
below an unclipped sample the same store records). The live verdict count is 59,936 and they are all
passes, so exactly the 266 flags carry a `wasInvalidatedBy` edge and no replacement verdict was
written: those recordings now have a QUALITY activity, an invalidated flag, and no live QUALITY
conclusion. A further 2,319 preprocessed recordings have no QUALITY activity at all, concentrated in
airway and phonation families (respiration-and-cough-fivebreaths 596, maximum-phonation-time 222,
respiration-and-cough-v2-breath 193, …). 9,688 of the QUALITY verdicts record at least one clip span.

### 3.2 Incomplete, relatively empty, unusually short

The owner's three new questions, each against what the store can actually answer.

#### Incomplete — **the store cannot answer this.**

The instrument for it is the `truncation` deviation ("a production the recording does not contain the
end of"), written by SPEECH, VOICE, AIRWAY and the DDK body. None ran. There is no other record in
the store of a production being cut off: ADMIT's outcome is about decodability, not completeness, and
`level`, `silence` and the taxonomy carry nothing about the end of an intended production.

**What would be needed:** a branch run. `truncation` is the only instrument, and it needs both the
expectation table (what production was asked for) and the alignment (where the recording stops
relative to it).

**The one proxy the stores support, named as a proxy:** the last consensus word's offset against the
recording's end. A transcript whose last word ends at the file boundary is consistent with a cut-off
recording; it is equally consistent with a participant who stopped talking and stopped the recording
in the same motion, and consensus word timings are themselves ASR-derived. Over the 33,086 lexical
family recordings with a timed last word, **5,475 (16.5 %)** have the last word ending within 0.10 s
of the file end. The rate varies from 0.023 (loudness) to 0.343 (open-response-questions). This is a
proxy and nothing more; it is in Table A as "right-edge tight" and should not be quoted as a
truncation rate.

#### Relatively empty — **partly answerable, but not by the instrument intended for it.**

The `acoustically_empty` discard ground comes from the **emptiness bypass**, part of `routing`'s
ruleset evaluation, and it is written onto the file verdict by VERDICT. Neither node ran, so no
recording in this corpus carries `acoustically_empty` — nor `unmeasurable`, the other discard
ground, which is also VERDICT's. ADMIT's own fail verdict is the only live record of a recording
that carried nothing.

What the store does carry, as measurements rather than decisions:

- **ADMIT rejection**, which is the nearest thing to a real emptiness finding and *is* a decision:
  **57 recordings (0.091 % of 62,578) fail ADMIT with `why: "every sample is zero"`** — a genuinely
  empty file, not an empty performance. They span 18 families with no concentration
  (harvard-sentences-list 10, respiration-and-cough-fivebreaths 8, …).
- **No lexical content**: the consensus transcript has zero words, or every position in it is a
  bracketed event token such as a breath or cough marker rather than a word. Over the 41,205
  recordings in speech-eliciting families (lexical + DDK), **1,416 (3.4 %)** have no lexical content.
  Over the whole 62,521, 17,153 (27.4 %) do — but most of that is airway and phonation families
  where no words were ever asked for, which is why the speech-eliciting denominator is the one to
  read. Per family in Table A.
- **Near-silent**: more than 90 % of the `silence` measurement's windows scored as silence at its
  threshold of 0.5. **1,696 of 62,521 (2.7 %)**. This is highest in voluntary-cough (0.53 median
  silent fraction), respiration-and-cough-breath (0.128 above the 90 % line) and
  harvard-sentences-list (0.053).
- **Level**: `rms_dbfs` is recorded for all 62,521. Median −28.2 dBFS; 1,368 below −50 dBFS; 344
  below −60 dBFS. No threshold in this project declares what "too quiet" is, so these are reported
  as distribution points and not as a rate.

The three readings disagree with each other by design — a cough recording with no words is doing the
task, a story-recall recording with no words is not — and none of them is the emptiness bypass. **A
re-run with `routing` is what turns "no lexical content" into "acoustically empty" or into "the
participant did not do the task", and this corpus cannot tell those apart.**

#### Unusually short — **answerable descriptively.**

**Definition used, stated so it can be argued with:** a recording is *short* when its duration is
below half its own family's median duration, and *very short* below a quarter. The family's own
median is the reference because the families span 3.4 s (cape-v-sentences-v2) to 94.3 s
(cinderella-story) and a global constant would call two thirds of the corpus short.

**This is descriptive and is not a fitted threshold.** Nothing was measured to establish that half a
family median marks an unusable recording, no ground truth was consulted, and the ratio was chosen
because it is legible, not because it separates anything. It is here to rank families and to size the
tail, not to gate a recording.

Corpus-wide, over the 62,521 with a measured duration: **4,405 short (7.0 %)** and **2,220 very
short (3.6 %)**. The tail is not uniform:

| family | n | median s | short | very short |
|---|---:|---:|---:|---:|
| cinderella-story | 258 | 94.26 | 0.260 | 0.155 |
| random-item-generation-v2 | 207 | 59.81 | 0.246 | 0.072 |
| random-item-generation | 265 | 53.92 | 0.219 | 0.079 |
| picture-description | 889 | 53.35 | 0.204 | 0.031 |
| free-speech-v2 | 2,119 | 29.72 | 0.141 | 0.034 |
| free-speech | 3,073 | 30.02 | 0.131 | 0.051 |
| maximum-phonation-time-v2 | 813 | 15.44 | 0.130 | 0.021 |
| word-color-stroop | 472 | 75.84 | 0.000 | 0.000 |

The long open-ended families carry the tail; the fixed-script families barely have one. The
absolute tail, for comparison: 1,586 recordings (2.5 %) are under 0.5 s and 2,003 (3.2 %) under 2 s.


---

## Table A — per family: what is in the corpus and the three recording-quality readings

`elicits` is the kind the family's instruction asks for. `n` counts manifest rows; every rate below
is over the recordings of that family that reached PREPROCESS (`n` minus that family's ADMIT
failures), except "right-edge tight", whose denominator is printed because it also excludes
recordings with no timed last word. "No lexical content" is only interpretable against what the
family asked for: 0.957 on breath-sounds is the task being done, 0.038 on story-recall is not. The
totals row is over all 62,521 preprocessed recordings and therefore mixes the two; the
speech-eliciting figure is 1,416 / 41,205 (0.034), in §3.2.

| family | elicits | n | median s | short (<½ median) | very short (<¼ median) | no lexical content | near-silent (>90% silent) | right-edge tight (<0.10 s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| animal-fluency | lexical | 195 | 60.02 | 16 (0.082) | 4 (0.021) | 0 (0.000) | 0 (0.000) | 49/194 (0.253) |
| breath-sounds | airway | 326 | 13.19 | 12 (0.037) | 1 (0.003) | 312 (0.957) | 1 (0.003) | 17/278 (0.061) |
| cape-v-sentences | lexical | 2370 | 3.56 | 17 (0.007) | 6 (0.003) | 18 (0.008) | 12 (0.005) | 339/2369 (0.143) |
| cape-v-sentences-v2 | lexical | 1224 | 3.43 | 3 (0.002) | 0 (0.000) | 2 (0.002) | 1 (0.001) | 244/1224 (0.199) |
| caterpillar-passage | lexical | 598 | 79.89 | 12 (0.020) | 8 (0.013) | 7 (0.012) | 5 (0.008) | 24/594 (0.040) |
| cinderella-story | lexical | 258 | 94.26 | 67 (0.260) | 40 (0.155) | 18 (0.070) | 9 (0.035) | 36/258 (0.140) |
| diadochokinesis-buttercup | ddk | 896 | 7.13 | 5 (0.006) | 1 (0.001) | 7 (0.008) | 3 (0.003) | 139/895 (0.155) |
| diadochokinesis-ka | ddk | 898 | 5.06 | 18 (0.020) | 2 (0.002) | 32 (0.036) | 3 (0.003) | 99/890 (0.111) |
| diadochokinesis-pa | ddk | 896 | 5.50 | 30 (0.033) | 1 (0.001) | 35 (0.039) | 3 (0.003) | 98/883 (0.111) |
| diadochokinesis-pataka | ddk | 897 | 7.34 | 5 (0.006) | 1 (0.001) | 7 (0.008) | 3 (0.003) | 118/893 (0.132) |
| diadochokinesis-ta | ddk | 898 | 5.09 | 17 (0.019) | 1 (0.001) | 18 (0.020) | 3 (0.003) | 62/888 (0.070) |
| diadochokinesis-v2-buttercup | ddk | 702 | 5.09 | 3 (0.004) | 0 (0.000) | 6 (0.009) | 5 (0.007) | 288/701 (0.411) |
| diadochokinesis-v2-kuh | ddk | 702 | 5.10 | 5 (0.007) | 1 (0.001) | 24 (0.034) | 6 (0.009) | 127/696 (0.182) |
| diadochokinesis-v2-puh | ddk | 702 | 5.10 | 5 (0.007) | 2 (0.003) | 27 (0.038) | 3 (0.004) | 130/697 (0.187) |
| diadochokinesis-v2-puhtuhkuh | ddk | 701 | 5.10 | 4 (0.006) | 0 (0.000) | 11 (0.016) | 5 (0.007) | 172/700 (0.246) |
| diadochokinesis-v2-tuh | ddk | 702 | 5.09 | 1 (0.001) | 0 (0.000) | 15 (0.021) | 4 (0.006) | 68/701 (0.097) |
| free-speech | lexical | 3075 | 30.02 | 403 (0.131) | 157 (0.051) | 77 (0.025) | 57 (0.019) | 706/3064 (0.230) |
| free-speech-v2 | lexical | 2121 | 29.72 | 298 (0.141) | 71 (0.034) | 44 (0.021) | 33 (0.016) | 381/2111 (0.180) |
| glides-high-to-low | voice | 1554 | 6.25 | 55 (0.035) | 5 (0.003) | 633 (0.407) | 5 (0.003) | 66/1447 (0.046) |
| glides-low-to-high | voice | 1596 | 6.68 | 45 (0.028) | 0 (0.000) | 722 (0.452) | 4 (0.003) | 45/1519 (0.030) |
| harvard-sentences-list | lexical | 13710 | 4.21 | 931 (0.068) | 880 (0.064) | 911 (0.066) | 725 (0.053) | 3020/13611 (0.222) |
| high-to-low | voice | 43 | 6.38 | 1 (0.023) | 0 (0.000) | 22 (0.512) | 0 (0.000) | 1/33 (0.030) |
| loudness | lexical | 898 | 4.53 | 14 (0.016) | 2 (0.002) | 6 (0.007) | 3 (0.003) | 21/895 (0.023) |
| loudness-v2 | lexical | 705 | 5.57 | 18 (0.026) | 1 (0.001) | 11 (0.016) | 2 (0.003) | 16/705 (0.023) |
| maximum-phonation-time | voice | 2700 | 14.37 | 306 (0.113) | 90 (0.033) | 2190 (0.812) | 34 (0.013) | 118/2478 (0.048) |
| maximum-phonation-time-v2 | voice | 813 | 15.44 | 106 (0.130) | 17 (0.021) | 572 (0.704) | 8 (0.010) | 34/694 (0.049) |
| open-response-questions | lexical | 199 | 30.02 | 11 (0.055) | 6 (0.030) | 5 (0.025) | 3 (0.015) | 68/198 (0.343) |
| picture-description | lexical | 889 | 53.35 | 181 (0.204) | 28 (0.031) | 3 (0.003) | 2 (0.002) | 143/889 (0.161) |
| picture-description-option1 | lexical | 373 | 31.21 | 39 (0.105) | 4 (0.011) | 5 (0.013) | 2 (0.005) | 18/373 (0.048) |
| picture-description-option2 | lexical | 329 | 43.86 | 28 (0.085) | 4 (0.012) | 4 (0.012) | 4 (0.012) | 16/329 (0.049) |
| productive-vocabulary | lexical | 2912 | 8.66 | 325 (0.112) | 44 (0.015) | 62 (0.021) | 27 (0.009) | 157/2895 (0.054) |
| prolonged-vowel | voice | 1605 | 12.11 | 31 (0.019) | 3 (0.002) | 294 (0.183) | 8 (0.005) | 97/1585 (0.061) |
| rainbow-passage | lexical | 898 | 28.68 | 8 (0.009) | 7 (0.008) | 7 (0.008) | 5 (0.006) | 35/895 (0.039) |
| random-item-generation | lexical | 265 | 53.92 | 58 (0.219) | 21 (0.079) | 1 (0.004) | 0 (0.000) | 36/265 (0.136) |
| random-item-generation-v2 | lexical | 207 | 59.81 | 51 (0.246) | 15 (0.072) | 4 (0.019) | 3 (0.014) | 31/206 (0.150) |
| respiration-and-cough-breath | airway | 1788 | 30.02 | 164 (0.092) | 148 (0.083) | 1621 (0.907) | 228 (0.128) | 1105/1595 (0.693) |
| respiration-and-cough-cough | airway | 1789 | 7.02 | 188 (0.105) | 133 (0.074) | 1047 (0.586) | 158 (0.088) | 134/1683 (0.080) |
| respiration-and-cough-fivebreaths | airway | 3580 | 17.72 | 368 (0.103) | 288 (0.081) | 3414 (0.956) | 132 (0.037) | 437/2976 (0.147) |
| respiration-and-cough-threequickbreaths | airway | 1719 | 8.47 | 233 (0.136) | 139 (0.081) | 1606 (0.935) | 55 (0.032) | 187/1583 (0.118) |
| respiration-and-cough-v2-breath | airway | 701 | 20.02 | 13 (0.019) | 8 (0.011) | 663 (0.951) | 51 (0.073) | 79/503 (0.157) |
| respiration-and-cough-v2-hardcough | airway | 698 | 4.53 | 31 (0.044) | 5 (0.007) | 432 (0.619) | 12 (0.017) | 52/686 (0.076) |
| respiration-and-cough-v2-threebreaths | airway | 699 | 10.34 | 32 (0.046) | 5 (0.007) | 649 (0.928) | 6 (0.009) | 33/619 (0.053) |
| respiration-and-cough-v2-threebreathsmouth | airway | 699 | 11.52 | 26 (0.037) | 4 (0.006) | 662 (0.947) | 7 (0.010) | 28/600 (0.047) |
| respiration-and-cough-v2-threebreathsnose | airway | 699 | 12.12 | 28 (0.040) | 2 (0.003) | 672 (0.961) | 14 (0.020) | 33/555 (0.059) |
| story-recall | lexical | 890 | 44.32 | 108 (0.122) | 38 (0.043) | 34 (0.038) | 23 (0.026) | 77/882 (0.087) |
| story-recall-v2 | lexical | 660 | 77.17 | 51 (0.077) | 20 (0.030) | 13 (0.020) | 12 (0.018) | 33/659 (0.050) |
| voluntary-cough | airway | 327 | 13.89 | 34 (0.104) | 7 (0.021) | 226 (0.691) | 4 (0.012) | 4/306 (0.013) |
| word-color-stroop | lexical | 472 | 75.84 | 0 (0.000) | 0 (0.000) | 2 (0.004) | 3 (0.006) | 25/470 (0.053) |
| **all** | | 62,578 (62,521 measured) | | 4405 (0.070) | 2220 (0.036) | 17153 (0.274) | 1696 (0.027) | 9246/60170 (0.154) |

## Table B — per family: speaker counts

`diarized n` is the recordings of that family carrying both diarization measurements. "2nd speaker
≥5 s" is the subset whose secondary speaker has at least five seconds of speech — the only column
here that is not dominated by spans too short for the instrument.

| family | diarized n | 0 speakers | 1 | ≥2 | rate | ≥2 on residual | rate | 2nd speaker ≥5 s | rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| animal-fluency | 191 | 0 | 173 | 18 | 0.094 | 14 | 0.073 | 0 | 0.0000 |
| breath-sounds | 325 | 276 | 47 | 2 | 0.006 | 3 | 0.009 | 0 | 0.0000 |
| cape-v-sentences | 2358 | 16 | 2226 | 116 | 0.049 | 140 | 0.059 | 0 | 0.0000 |
| cape-v-sentences-v2 | 1209 | 6 | 1133 | 70 | 0.058 | 96 | 0.079 | 0 | 0.0000 |
| caterpillar-passage | 590 | 8 | 466 | 116 | 0.197 | 106 | 0.180 | 5 | 0.0085 |
| cinderella-story | 258 | 13 | 177 | 68 | 0.264 | 64 | 0.248 | 4 | 0.0155 |
| diadochokinesis-buttercup | 890 | 7 | 860 | 23 | 0.026 | 36 | 0.040 | 0 | 0.0000 |
| diadochokinesis-ka | 890 | 16 | 834 | 40 | 0.045 | 72 | 0.081 | 1 | 0.0011 |
| diadochokinesis-pa | 884 | 26 | 840 | 18 | 0.020 | 44 | 0.050 | 0 | 0.0000 |
| diadochokinesis-pataka | 882 | 5 | 849 | 28 | 0.032 | 56 | 0.063 | 0 | 0.0000 |
| diadochokinesis-ta | 883 | 15 | 835 | 33 | 0.037 | 87 | 0.099 | 0 | 0.0000 |
| diadochokinesis-v2-buttercup | 694 | 10 | 661 | 23 | 0.033 | 25 | 0.036 | 0 | 0.0000 |
| diadochokinesis-v2-kuh | 694 | 12 | 666 | 16 | 0.023 | 53 | 0.076 | 0 | 0.0000 |
| diadochokinesis-v2-puh | 689 | 18 | 660 | 11 | 0.016 | 43 | 0.062 | 0 | 0.0000 |
| diadochokinesis-v2-puhtuhkuh | 695 | 12 | 678 | 5 | 0.007 | 49 | 0.071 | 0 | 0.0000 |
| diadochokinesis-v2-tuh | 695 | 14 | 665 | 16 | 0.023 | 49 | 0.071 | 0 | 0.0000 |
| free-speech | 3035 | 56 | 2755 | 224 | 0.074 | 309 | 0.102 | 8 | 0.0026 |
| free-speech-v2 | 2093 | 42 | 1847 | 204 | 0.097 | 149 | 0.071 | 5 | 0.0024 |
| glides-high-to-low | 1534 | 186 | 1274 | 74 | 0.048 | 54 | 0.035 | 0 | 0.0000 |
| glides-low-to-high | 1583 | 261 | 1277 | 45 | 0.028 | 42 | 0.027 | 0 | 0.0000 |
| harvard-sentences-list | 13567 | 743 | 12480 | 344 | 0.025 | 919 | 0.068 | 0 | 0.0000 |
| high-to-low | 43 | 5 | 37 | 1 | 0.023 | 2 | 0.047 | 0 | 0.0000 |
| loudness | 890 | 10 | 873 | 7 | 0.008 | 24 | 0.027 | 0 | 0.0000 |
| loudness-v2 | 694 | 11 | 679 | 4 | 0.006 | 17 | 0.024 | 0 | 0.0000 |
| maximum-phonation-time | 2668 | 685 | 1750 | 233 | 0.087 | 111 | 0.042 | 14 | 0.0052 |
| maximum-phonation-time-v2 | 800 | 162 | 563 | 75 | 0.094 | 45 | 0.056 | 6 | 0.0075 |
| open-response-questions | 195 | 3 | 176 | 16 | 0.082 | 17 | 0.087 | 0 | 0.0000 |
| picture-description | 877 | 2 | 783 | 92 | 0.105 | 111 | 0.127 | 2 | 0.0023 |
| picture-description-option1 | 371 | 4 | 339 | 28 | 0.075 | 22 | 0.059 | 1 | 0.0027 |
| picture-description-option2 | 328 | 3 | 293 | 32 | 0.098 | 36 | 0.110 | 1 | 0.0030 |
| productive-vocabulary | 2896 | 56 | 2669 | 171 | 0.059 | 175 | 0.060 | 0 | 0.0000 |
| prolonged-vowel | 1591 | 109 | 1407 | 75 | 0.047 | 55 | 0.035 | 0 | 0.0000 |
| rainbow-passage | 885 | 5 | 799 | 81 | 0.092 | 89 | 0.101 | 4 | 0.0045 |
| random-item-generation | 260 | 1 | 221 | 38 | 0.146 | 54 | 0.208 | 4 | 0.0154 |
| random-item-generation-v2 | 206 | 3 | 176 | 27 | 0.131 | 24 | 0.117 | 3 | 0.0146 |
| respiration-and-cough-breath | 1763 | 1336 | 413 | 14 | 0.008 | 13 | 0.007 | 1 | 0.0006 |
| respiration-and-cough-cough | 1765 | 847 | 903 | 15 | 0.008 | 29 | 0.016 | 0 | 0.0000 |
| respiration-and-cough-fivebreaths | 3538 | 2964 | 555 | 19 | 0.005 | 19 | 0.005 | 0 | 0.0000 |
| respiration-and-cough-threequickbreaths | 1701 | 1309 | 386 | 6 | 0.004 | 10 | 0.006 | 0 | 0.0000 |
| respiration-and-cough-v2-breath | 692 | 594 | 97 | 1 | 0.001 | 2 | 0.003 | 0 | 0.0000 |
| respiration-and-cough-v2-hardcough | 685 | 302 | 379 | 4 | 0.006 | 7 | 0.010 | 0 | 0.0000 |
| respiration-and-cough-v2-threebreaths | 690 | 573 | 115 | 2 | 0.003 | 5 | 0.007 | 0 | 0.0000 |
| respiration-and-cough-v2-threebreathsmouth | 693 | 617 | 75 | 1 | 0.001 | 2 | 0.003 | 0 | 0.0000 |
| respiration-and-cough-v2-threebreathsnose | 692 | 610 | 75 | 7 | 0.010 | 7 | 0.010 | 0 | 0.0000 |
| story-recall | 878 | 26 | 733 | 119 | 0.136 | 102 | 0.116 | 2 | 0.0023 |
| story-recall-v2 | 652 | 12 | 521 | 119 | 0.183 | 67 | 0.103 | 3 | 0.0046 |
| voluntary-cough | 326 | 129 | 195 | 2 | 0.006 | 4 | 0.012 | 0 | 0.0000 |
| word-color-stroop | 468 | 2 | 410 | 56 | 0.120 | 65 | 0.139 | 7 | 0.0150 |
| **all** | 61886 | 12122 | 47025 | 2739 | 0.044 | 3524 | 0.057 | 71 | 0.0011 |

---

## 4. What could not be established

Listed so none of it is mistaken for a null result.

1. **Any PII finding the pipeline itself made.** The scan never ran. The fresh scan in §1 is a
   different instrument reading at a different commit with no speaker attribution and no
   corroboration against the ASR-hypothesis haystacks the graph would also scan.
2. **Whether a PII finding falls in the target speaker's speech or a second voice's.** Needs SPEECH's
   word-to-speaker resolution. Nothing in these stores attributes a word to a speaker: the
   diarization is a time partition with its own labels, and no node joined it to the consensus words.
3. **Task conformance and deviations, at any value.** Not measured, not UNDETERMINED.
4. **Truncation / incompleteness.** No instrument ran. The right-edge proxy is named as a proxy in
   §3.2 and is not a rate for it.
5. **Whether a recording is acoustically empty in the pipeline's sense.** The emptiness bypass is
   `routing`'s and `routing` never ran. "No lexical content" is a different claim.
6. **Why 635 recordings have no diarization and 2,319 have no QUALITY activity.** The stores record
   the absence, not a reason. The run logs under
   `/orcd/scratch/bcs/002/satra/triage_full_20260908/logs/` were not read for this census.
7. **Why 266 QUALITY flag verdicts were invalidated with no replacement.** Observed, unexplained.
8. **Whether the 71 recordings with a substantive second speaker contain a clinician, a family
   member, a television, or the participant twice.** Nothing was listened to.

## 5. Questions for the owner

1. **Is a branch re-run over the corpus in scope, and at what width?** Everything in §1 and §3.1 is
   blocked on it. The full graph over 62,578 recordings is a much larger job than the
   ADMIT/PREPROCESS/TAXONOMY run that produced these stores, and a stratified subset — say 2,000
   recordings sampled across the 48 families — would answer the conformance and deviation questions
   with a usable denominator at a small fraction of the cost.
2. **The four null thresholds.** `windows.yamnet.default_threshold`,
   `windows.ast.default_threshold`, `windows.hear.default_threshold` and `voice.f0_range_hz` are
   null, and the consequence is that TAXONOMY concluded `uncertain` on 100 % of the corpus. A re-run
   that does not fix this produces the same table. Is measuring them the prerequisite to the re-run?
3. **Which diarization stream is the intended authority?** `enhanced` and `residual` agree on 475 of
   the 5,788 recordings where either finds a second voice. The config runs both and nothing in the
   graph reconciles them.
4. **Is `n_speakers ≥ 2` wanted as a reported measurement at all, given the span-length problem?**
   The median second speaker is 0.44 s. Either the diarizer needs a minimum-speech floor per speaker
   before it reports a count, or the count needs to be reported with the secondary duration beside
   it so a consumer can apply its own floor. Proposing a floor here would be a fitted threshold with
   nothing fitted.
5. **Should `truncation` get a non-branch instrument?** It is currently only reachable through a
   branch that also needs an expectation table and an alignment. A PREPROCESS-level right-censoring
   measurement (last word, last voiced frame, or last energy against the file end) would make
   "incomplete" answerable on every recording regardless of routing — but it is a new measurement,
   not a re-read.
6. **The 266 invalidated QUALITY flags.** Is that the extension pass working as intended, or a
   verdict lost in a re-run?
7. **Is the "no lexical content" reading wanted as a task-verification signal**, given that it fires
   on 95.7 % of breath-sounds and 0 % of animal-fluency by design? It is only meaningful against a
   family's expectation, which is the branch's job.

---

## 6. How this was produced

All compute via `sbatch` to `mit_preemptable`; nothing ran on the ORCD login node. Scripts on ORCD
scratch at `/orcd/scratch/bcs/002/satra/checks_20260916/census/`:

| script | what it does | job |
|---|---|---|
| `scan_store.py` | one pass over all 62,578 stores: nodes, verdicts, kinds, PII entities, diarization, level, silence, consensus, deviations | 32 procs, 39 s |
| `extract_text.py` | the transcripts and word boundaries the stores hold | 32 procs, ~60 s |
| `absences.py` | the PREPROCESS `absent` map and the TAXONOMY kind lines, corpus-wide | 16 procs, 2 min |
| `probe_pii.py` | whether a PII scan can run on a compute node at all | 1 proc |
| `pii_scan.py` | the fresh PII scan over every stored transcript | 16-way array × 4 procs |
| `pii_aggregate.py` | the fresh scan's counts and shapes, and the surfaces file | 1 proc |

Outputs under `/orcd/scratch/bcs/002/satra/checks_20260916/pii_census/` (mode 700). The corpus was
read and not written.
