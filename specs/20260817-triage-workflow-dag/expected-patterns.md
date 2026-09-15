# Expected patterns — what each task's instruction asks for, and what would match it

The owner's question: *"is there a list of what algorithm/method/heuristic is used to determine
whether a task was done and what the start end spans are for a task."*

**There is not.** This document is that list.

It is a **method inventory**, not a threshold fit. No number in it is fitted. Where a method needs a
boundary it says what would fit it and marks it owed, per the project rule that a threshold lives in
`data/` with a written derivation and never as a code literal.

## The organising idea

**A recording does not have a fixed temporal layout.** There is no preamble / body / trailing-comment
template to segment it into, and a method that assumes one is measuring the template rather than the
recording.

What there is, is an **instruction**. Each task's instruction implies a set of **expected patterns** —
lexical, acoustic, or an ordered combination — and the recording is matched against them. Structure
is what the match finds, never a shape imposed on it.

This unifies the owner's questions rather than answering them separately:

| the question | the same question, as a pattern match |
| --- | --- |
| **Was the task done?** | were the expected patterns found? |
| **What is the task extent?** | where were they found? |
| **What sub-components does it have?** | the patterns *are* the components; each is found, or not, on its own |

Four outcomes, and all four are findings:

- an expected pattern **found** — the component, with its extent;
- an expected pattern **absent** — the finding, and the only honest answer to "was it done";
- an expected pattern found **where it was not expected** — a different finding, never suppressed to
  fit an assumed order;
- material matching **no** expected pattern — `off_task_extent`.

Whether the counting precedes the vowel, whether it happens at all, whether there is material
matching neither, is an **observation**. `prolonged-vowel`'s instruction says the count comes first;
938 of 1,258 transcripts agree ([`dag.md:184`](dag.md)), and the other 320 are data.

**Declared families are not ground truth.** The declaration says *what to look for*, never *what is
there*. A participant who did not follow the instruction is precisely what the pattern match exists
to find; a method that scores the recording against the declaration is measuring the declaration.
This is the same line
[`../20260913-branch-contract-and-hints/design.md:770-784`](../20260913-branch-contract-and-hints/design.md)
draws for the two counts.

## The family is the wrong grain for an expected pattern

This is the structural finding of the exercise, and it precedes every row of the table.

`task_family` collapses trailing numeric segments (`families.py:134-144`), and the module docstring
says that "folds a repeat index and the Harvard list index and nothing else". **Measured on the
local corpus, that is not what the trailing index is.** Two cases, both verified against the
sidecars:

- **`respiration-and-cough-fivebreaths` carries two different instructions by index.**
  `-1` and `-3`: *"take 5 big breaths in and out through your nose with your mouth closed."*
  `-2` and `-4`: *"take 5 big breaths in and out through your mouth."* The index is a **condition**
  index, not a repeat index, and the family name loses the route — which is exactly what AIRWAY A7
  would be measuring.
- **`harvard-sentences-list-49-1` and `-52-3` are different sentences.** Collapsing to
  `harvard-sentences-list` discards which sentence was expected, which is the whole lexical pattern.

So: **the expected pattern is a property of the recording, carried by its own sidecar. The family is
a routing and scoring unit and nothing more.** A branch that looks the pattern up by family will
measure the wrong thing on at least two families today, and on `productive-vocabulary`,
`cape-v-sentences-(v2)` and `free-speech` — where each index carries a different cue word, sentence
or question — it will have nothing to look up at all. The table below is keyed by family because
that is how the corpus is counted; every row's pattern is resolved per recording.

### And 22 of 112 recordings reach no declared family at all

The BIDS filenames in this corpus spell the version marker with parentheses —
`task-Cape-V-sentences-(v2)-1`, `task-Diadochokinesis-(v2)-KUH`,
`task-Respiration-and-cough-(v2)-ThreeBreathsNose`. `task_id_of` lowercases and strips nothing else
(`families.py:121-131`), so the family is `cape-v-sentences-(v2)`. `families.py` declares
`cape-v-sentences-v2`. **Nothing matches.**

Measured directly over `~/Downloads/b2ai_v31_bids_07_01_v3`: 40 distinct families across 112
recordings, of which **15 families / 22 recordings are undeclared**, every one of them a `(v2)`
spelling; and **all 16 declared `*-v2` families have zero recordings**, 15 of them having an observed
`(v2)`-spelled counterpart and the sixteenth (`random-item-generation-v2`) being absent from the copy
altogether. The existing unit test exercises `cape-v-sentences-v2-4`
(`routing_analysis_test.py:285`), a hyphen spelling that does not occur in this corpus.

The corpus profile the branch documents quote gives non-zero counts for every `*-v2` family
(`respiration-and-cough-v2-breath` 699, `maximum-phonation-time-v2` 813, and so on), so either the
62,547-recording corpus spells them differently or the profile was built with a normaliser
`families.py` does not have. **Which is true is not settled from here**, and it is worth settling
before any per-family method is trusted on v3.1 material.

## Where the instruction comes from

Two JSON sidecars per recording, at two grains
([`../20260913-branch-contract-and-hints/design.md:256-280`](../20260913-branch-contract-and-hints/design.md)).
Both live in the same `audio/` directory. Measured on the local copy: 112 `_recording-metadata.json`
against **36** `_acoustictask-metadata.json`, linked by `recording_acoustic_task_id` →
`acoustic_task_id`, all 112 linking, no orphans. **There are no transcripts in the corpus** — 112
wav, 112 + 36 JSON, 3 `sessions.tsv`, one `PROVENANCE.md`, and nothing else.

**The per-recording grain is the only source of the prompt.** Re-measured on all 112:

| comparison of the two grains | n |
| --- | --- |
| `stimulus_text` identical, both empty | 64 |
| `stimulus_text` identical, both non-empty | 6 |
| `stimulus_text` — acoustictask **empty**, recording has text | 33 |
| `stimulus_text` — recording empty, acoustictask has text | 0 |
| `stimulus_text` — both non-empty and **different** | 9 |
| **`stimulus_text` differs, any way** | **42** |
| `instructions` differ | **72** |
| either field differs | **85** |

`design.md:387` records **44 of 112** for `stimulus_text`; this re-measurement gives **42**. The
direction is the same and the conclusion is unchanged — never read the prompt from the acoustictask
JSON — but the two counts should be reconciled before either is cited as settled.

Two details the earlier measurement did not carry, and both matter:

- **The nine "both non-empty but different" cases are all `free-speech`**, and the acoustictask JSON
  carries the same stale prompt on all nine — *"Can you explain your voice/speech problems and why
  you consulted a physician for them?"* — which matches **none** of the six questions actually asked.
  That is worse than empty: a builder reading the family grain gets a confidently wrong prompt.
- **`instructions` disagree more often than `stimulus_text` does (72 of 112)**, and always in the
  same direction. The acoustictask `instructions` for `diadochokinesis` is the *buttercup* wording
  applied to `/PA/`, `/TA/` and `/KA/`; for `glides` it is the low-to-high wording, so
  `glides-high-to-low` is described **backwards** at the family grain; for `random-item-generation`
  it omits the category entirely. Since the instruction is where the pattern comes from, this is the
  same defect as the `stimulus_text` one and it is larger.

Two fields, two different jobs:

- **`stimulus_text` is machine-readable** and is the lexical half of an expected pattern where it
  exists. It goes into `AudioHints.expected_speech` as ordered `ExpectedSpeech` entries
  ([`design.md:355-362`](../20260913-branch-contract-and-hints/design.md)), whose docstring already
  draws the distinction this document needs: *"which sentence was skipped" is a different question
  from "how close was the whole thing"* (`audio_hints.py:142-144`).
- **`instructions` is prose and must not be parsed**
  ([`design.md:352`](../20260913-branch-contract-and-hints/design.md)). It is read by a human into
  the pattern column below, and that column is checkable against it. The verbatim instruction per
  family is in the appendix.

**`stimulus_text` is empty for every non-lexical family** — all ten DDK families, all ten airway
families, both loudness families, both glide families, `prolonged-vowel` and both
`maximum-phonation-time` families. The target token of a DDK task, the count in a cough task and the
`1, 2, 3 aah` of a prolonged vowel all live **only inside `instructions`**. So for more than half the
corpus the expected pattern is not machine-readable at all and must come from the human-read
expectations table `design.md:336-350` specifies and which does not exist yet.

### `expected_speech` is available at the point of use

`AudioHints` reaches every branch: `run.py:298-300` passes `hint` into `airway(...)`, `speech(...)`
and `voice(...)`, and `quality(...)` at `:310` takes it too. The only definition of `expected_speech`
in `src/senselab` is the field itself (`audio_hints.py:142`, `:152`) — **nothing reads it**, because
no branch implements the comparison yet.

That is an unimplemented consumer, not a plumbing gap. A detection approach that compares consensus
words against `hint.expected_speech` is **implementable today**; the only thing owed is branch code,
which is owed for every row in this document.

Two caveats that belong in prose rather than as a blanket "owed":

- **The hint has to be populated.** `run_triage` takes `hint=None` by default (`run.py:381`), and
  today's builder (`runs/b2ai-v2/make_hints.py`) parses the BIDS `task-` token from the filename and
  never carries `stimulus_text` at all
  ([`design.md:289-294`](../20260913-branch-contract-and-hints/design.md)).
- **A hint is an expectation, not an observation.** The field's own docstring says nothing downstream
  should read it as ground truth. A mismatch against `expected_speech` is a `stimulus_mismatch`
  deviation with an extent, never proof the participant erred — and on this corpus ASR error
  correlates with the impairments the corpus exists to characterise, which is why
  [`branch-speech.md:134-138`](branch-speech.md) requires recogniser agreement to travel with every
  mismatch.

## Which verb carries which answer

The five store-wide verbs
([`../20260913-branch-contract-and-hints/design.md:561-567`](../20260913-branch-contract-and-hints/design.md)),
mapped onto the pattern match:

| the match's result | verb | payload |
| --- | --- | --- |
| this span is where expected pattern P was found | `label` | what the span carries |
| P was proposed here and is not there | `contest` | that it does not carry what was proposed |
| the proposer's boundary for P is wrong | `refine` | `corrected_extent: [start, end]` |
| this is the part of the span that serves the task | `trim` | `task_extent: [start, end]`, plus the `off_task_extent` finding |
| P is here and PREPROCESS proposed nothing | `propose` | mints a `family: "<branch>"` span |
| P's count against the declared count | *not a verb* — a `counts` measurement, `found` beside `declared`, no discrepancy asserted |
| P was expected and is nowhere | *not a verb* — the branch's own verdict |

`refine` was widened on 2026-09-15 beyond extent: **a fired rule may stamp or refine a span's
label** — `label` where the span carried none, `refine` where it carried one the rule sharpens
([`design.md:407-414`](../20260913-branch-contract-and-hints/design.md),
[`family-taxonomy-ruleset.md:383-395`](family-taxonomy-ruleset.md)). The verb table at
`design.md:565` has not been updated to say so and still declares `refine`'s payload as
`corrected_extent` alone. That is a live inconsistency.

**A component span is a `propose` only when nothing proposed the ground.** Where PREPROCESS's
amplitude spans already cover it — which on a held vowel they do, reading 15.89 s on the MPT
recording of the 2026-09-15 run — the component is a `label` plus, where the boundary is wrong, a
`refine`. [`branch-conventions.md:20-23`](branch-conventions.md) scopes both by family, and that
scoping is unresolved for a label written onto an `amplitude` span.

**Material matching nothing already has a carrier.** PREPROCESS writes `measure: "gap"` spans over
every stretch no other span source covered (`preprocess.py:1586-1598`), so `off_task_extent` does not
need a new detector to find *where* the unmatched material is — only a rule for what makes it
off-task rather than ordinary silence.

### Three inconsistencies this document had to navigate

1. **`trim` has no emitter and no method.** `task_extent` appears exactly twice in the whole `specs/`
   tree — `design.md:566` and `:577` — and **zero times in `src/senselab/`**. No branch document
   lists `trim` in its emit block; `dag.md:1816` records that AIRWAY writes none. The spec says what
   `trim` *carries* and nothing about how the extent is *determined*. This document is the first
   statement of that method, per family.
2. **`trim` and `deviate` both claim `off_task_extent`.** `design.md:566` has `trim` carrying it;
   `branch-conventions.md:77` stores every deviation as `assertion, verb: "deviate"` and
   `branch-airway.md:348` emits AIRWAY's that way. One of the two has to go.
3. **The contract's three deviations are not the list.** `branch-conventions.md:93-107` is
   authoritative and carries nine types. `task_extent`, `off_task`, `off_task_extent`,
   `stimulus_mismatch`, `expected_event_count` are **all zero occurrences in `src/senselab/`**;
   `speaker_count` appears only as a null / `no_speaker_count` state. The whole deviation and count
   vocabulary is declared and unbuilt.

## How to read the status marks

Every row needs branch code that does not exist; saying so on each row would make the mark useless.
So the mark is about the **measurement**, not the code:

| mark | meaning |
| --- | --- |
| **implementable today** | every measurement the approach reads is already written into the store by a shipped node. Only branch code is owed, as for every row. |
| **owed a code change** | the computation exists in `src/senselab` but the graph does not put its output where a branch can read it, or a selector excludes it. No new science. |
| **owed a measurement** | nothing in the tree produces the quantity. A new estimator, or a statistic and a stated window, is owed before the approach exists at all. |
| **owed a cut** | the approach is complete; the decision it feeds needs an operating point nobody has fitted. |

---

## The table

One row per (branch, family) pair. Families whose instruction is identical modulo the target syllable
or word share a row and are all named in it. Counts are the corpus profile's, over 62,547 recordings.

Pattern notation: **L** = lexical, **A** = acoustic, **→** = order the instruction states,
**&** = both expected, order not stated.

### VOICE

| branch | task | expected pattern | detection approach |
| --- | --- | --- | --- |
| VOICE | `prolonged-vowel` (1,604) | **L→A.** L: the ordered tokens `one two three`, once. A: one continuous voiced production of a single vowel /a/, held to the timer, F0 holding rather than moving. The only voice family in the corpus that is not purely non-lexical, despite `speech_type: non-lexical` | A's extent: `spans` `measure: "amplitude"` (`peak_over_floor_db`); qualified by `phonation_tracks` `f0_hz` + `strength`. Separating A from L is free on the lexical side — consensus `word` entities carry extents, so the vowel is the voiced production under no lexical word. **Owed a measurement** for the acoustic separator: F0 / formant stationarity and spectral flux, which V1 requires and which do not exist (`branch-voice.md:197-210`) |
| VOICE | `maximum-phonation-time` (2,696) | **A only.** One held /a/ on one breath, to exhaustion. v1's instruction places the deep inhale **before the record tap is mentioned**, so an audible inhale may be inside the file — a second acoustic pattern that is expected but is not the measurement | As above. Duration of the qualified extent is V2's maximum phonation time. **Owed a measurement** (stationarity) and **owed a code change** (VOICE cannot select the amplitude spans at all — see the worked example). The inhale would be an AIRWAY `label`, not an `off_task_extent` |
| VOICE | `maximum-phonation-time-v2` (813) | **A only**, and cleaner: the instruction puts the inhale explicitly *before* the record tap, so no inhale is expected in the file. A behavioural difference from v1, not a wording difference | Same. The v1/v2 contrast is the only within-corpus control for whether the inhale is captured |
| VOICE | `glides-low-to-high` (1,596), `glides-high-to-low` (1,554) | **A only.** One continuous voiced production of /i/ whose F0 sweeps monotonically across the range, in the declared direction. **Not steady** — the opposite of the sustained pattern | `phonation_tracks` `f0_hz` over the amplitude-span extent; the dominant monotone segment and its sign. `sweep_direction_mismatch` is the declared deviation (`branch-conventions.md:105`). **Implementable today** for the track; **owed a cut** for what counts as the dominant monotone segment (V3) |
| VOICE | `high-to-low` (43) | presumed identical to `glides-high-to-low` | **Unverified** — this family has zero recordings in the local copy and its instruction was never read. It may be a dead alias for `glides-high-to-low` |
| VOICE | `loudness` (897) | **L&A, counted.** The token `hey`, **three times**, each at maximal effort | Consensus words give the token and its three extents; `level` (`peak_dbfs`, `rms_dbfs`, `lufs`) and `energy_envelope` give the effort per extent. `expected_event_count: 3` as a `counts` entry. **Implementable today** |
| VOICE | `loudness-v2` (705) | **L→A, contrastive.** `hey` at normal effort **→** `hey` shouted. The measurement is the *within-recording* contrast and needs no norm | Same measurements, differenced across the two extents. **Implementable today**, and the cheapest effort measure in the corpus. V6 is explicit that v1 and v2 are **not one measurement** ([`branch-voice.md:730`](branch-voice.md)) |
| VOICE | `cape-v-sentences` (2,370), `-v2` (1,224) | **A riding on SPEECH's L.** Six sentences each loading a different phonatory condition; the voice-quality measurement is per sentence and pooling discards the instrument's design | Sentence boundaries fall out of SPEECH's S3 alignment for free. **Owed a code change** — they must reach VOICE as selectable spans. Nothing output may be presented as a CAPE-V score |
| VOICE | any AIRWAY- or SPEECH-declared family | **no expected pattern.** VOICE routed 22,277 recordings against 8,306 declaring a voice family | The branch does its best on what it was handed: `label`, `refine` or `contest` on phonation evidence, and it concludes on its own question ([`design.md:487-507`](../20260913-branch-contract-and-hints/design.md)). Connected speech passes voiced-fraction, F0-availability and interruption tests, so **the stationarity qualifiers are what stop V4 computing perturbation over consonants and pauses** — the same owed measurement, here load-bearing over 22,277 recordings |

### SPEECH

| branch | task | expected pattern | detection approach |
| --- | --- | --- | --- |
| SPEECH | `harvard-sentences-list` (13,705), `cape-v-sentences` (2,370), `-v2` (1,224) | **L only**, ordered and fully specified: the words of that recording's `stimulus_text`, in order, once. One sentence per recording — the pattern is per recording, never per family | Forced alignment of the consensus transcript against `hint.expected_speech`, plus a text-level diff. `align_transcriptions` exists (`tasks/forced_alignment/`). **Implementable today** for substitutions and insertions — enumerated differences, not scores. **Owed a cut** for omissions: a skip-arc-free aligner assigns every stimulus word an interval, so an omission surfaces only as a low acoustic score ([`branch-speech.md:115-121`](branch-speech.md)). Carry the consensus `agreement` on every mismatch |
| SPEECH | `rainbow-passage` (897), `caterpillar-passage` (597) | **L only**, ordered, one long passage. `stimulus_text` is identical at both sidecar grains for these | Same as above, at passage length. `[breath]` tokens inside the passage are **not** `filler` — they are how S4 measures breath-group structure |
| SPEECH | `word-color-stroop` (472) | **L only, ordered, and not the displayed words.** The instruction says *name the colour, do not read the word*. `stimulus_text` is the 15-item colour sequence, i.e. the expected **answer** sequence | Ordered alignment against `expected_speech` built from that sequence. **Implementable today.** `filler` must **not** be a deviation here — hesitation and self-correction are the task's dependent variable ([`branch-speech.md:130-132`](branch-speech.md)) |
| SPEECH | `free-speech` (3,074), `free-speech-v2` (2,120) | **L, unordered, with a negative pattern.** No target text. The instruction says *"do not record yourself reading the prompt"*, so `stimulus_text` is an **anti-pattern**: the question appearing verbatim in the transcript is the deviation | Alignment of the transcript against `expected_speech`, read inverted — high verbatim overlap is the finding. Consensus words and the aligner exist; **owed a cut** for how much overlap is reading rather than echoing a question word. **This row cannot use the family grain at all**: the acoustictask JSON carries a stale prompt matching none of the six real questions, on all nine local recordings |
| SPEECH | `story-recall` (889), `story-recall-v2` (660), `cinderella-story` | **L, unordered, partly negative.** Recall *in the participant's own words*. `stimulus_text` is the source story, so semantic coverage is expected and **verbatim** reproduction is the deviation — the participant read rather than recalled | Consensus words against `expected_speech`; n-gram overlap separates recall from reading. **Owed a measurement** — no lexical-overlap or semantic-coverage statistic exists in the graph and `text/tasks/embeddings_extraction` is not wired into triage |
| SPEECH | `productive-vocabulary` (2,910) | **L, weakly specified, per recording.** `stimulus_text` is one cue word per recording (`sunset`, `membership`, `plagiarize`, `entail`, `conga`, `maceration`); expected is definitional speech *about* the cue, not the cue itself | Presence and extent of lexical content is free. Anything beyond presence — whether a definition is *of* its cue — is **owed a measurement**. Six different cues under one family name, so the family grain carries nothing |
| SPEECH | `picture-description` (889), `-option1` (373), `-option2` (329), `open-response-questions`, `animal-fluency` | **L only, unspecified.** Connected speech, no target text, no `stimulus_text` (the stimulus is an image URL in the sidecar) | Presence and extent: consensus `word` entities and `spans` `measure: "asr"` runs. S4's connected-speech measures — rate, pause structure, breath groups — are the real content and are **not built** ([`branch-speech.md:157`](branch-speech.md)) |
| SPEECH | `random-item-generation`, `-v2` | **L, unordered, with a negative constraint.** Items from a named category (*"English words starting with 't'"*, which lives only in the per-recording `instructions`), **and no item repeated** | Consensus words plus `transcript_repeat` — largest repeat count of any normalised token, already computed for `ddk.lexical_repetition` (`default.yaml:287`). A repeat is the deviation the instruction names. **Owed a code change**: `transcript_repeat` lives in `routing_analysis/features.py`, is read only by the ruleset, and is not a store measurement. Category membership is **owed a measurement** |
| SPEECH | `loudness` (897), `loudness-v2` (705) | **L:** the token `hey` | Consensus words. The measurement wanted is VOICE's V6; SPEECH contributes the token's extent. These two are in `LEXICAL_SPEECH` (`families.py:41-42`) while the protocol calls them `speech_type: "non-lexical"` — a `families.py` discrepancy |
| SPEECH | every `SYLLABLE_REPETITION` family (7,989) | **L: none expected.** `/pa/` is not lexical and ASR mostly declines it. These are **positives** for SPEECH's reference set as of 2026-09-15 (`reference_family_set.SPEECH` = `lexical_speech \| syllable_repetition`), so near-zero lexical content is the correct observation, not a miss. `diadochokinesis-buttercup` is the exception — see the DDK table | Consensus words; expected count is zero or near it. `speech.lexical >= 2` firing here is over-routing on function-word artefacts: **71.3%** of that gate's apparent false positives are DDK families ([`dag.md:196-198`](dag.md)) |
| SPEECH | `prolonged-vowel` (1,604) | **L:** the ordered tokens `one two three`, once | Consensus words matched against `expected_speech = ["one","two","three"]`. Measured: **938 of 1,258** transcripts open with `One two three`; the control `maximum-phonation-time` fires `speech.intrusion` at **3.2%** ([`family-taxonomy-ruleset.md:315-322`](family-taxonomy-ruleset.md)). **Implementable today** — the cheapest row in the document |
| SPEECH | any AIRWAY-declared family | **L: none expected.** Any lexical content is off-task by construction | Consensus words with their extents; each contaminating extent is one `off_task_extent`. Measured examples are examiner speech — *"I'll have you do that one more time. [breath]"*, *"So just breathe."* ([`../20260910-taxonomy-routing-evidence/measurements.md:177-182`](../20260910-taxonomy-routing-evidence/measurements.md)). **Implementable today**; AIRWAY owns the deviation ([`branch-airway.md:169`](branch-airway.md)) |

### AIRWAY

| branch | task | expected pattern | detection approach |
| --- | --- | --- | --- |
| AIRWAY | `respiration-and-cough-cough` (1,788) | **A, counted.** Five discrete forced expulsive events — *"After pressing record, cough 5 times"*. `expected_event_count: 5` | Onset/offset per event from `energy_envelope`, corroborated by `span_hear` `Cough` / `span_yamnet` cough-subtree labels joined on `span_id`. **Owed a code change** — AIRWAY counts *label-carrying spans*, not events: `by_label` increments once per (span, label) pair (`airway.py:280`), so a 4 s span holding three coughs counts **1**, and `spans.min_separation_ms` merges adjacent coughs upstream. **Owed a measurement** for a true event boundary (A6) |
| AIRWAY | `respiration-and-cough-v2-hardcough` (698), `voluntary-cough` | **A, one event, maximal effort.** No count stated | As above, minus the count. Unlike `loudness-v2` there is no within-recording contrast to read effort against, so "hard" is **owed a measurement** |
| AIRWAY | `respiration-and-cough-fivebreaths` (3,576) | **A, counted (5) and routed — and the route is per recording, not per family.** Index `-1`/`-3`: nose, mouth closed. Index `-2`/`-4`: mouth. The family name carries neither | Cycle count from envelope peaks/troughs — A5, **not built**, and its three operating points (smoothing window, peak/trough criterion, minimum event duration) are all **owed a cut** ([`branch-airway.md:243-247`](branch-airway.md)). Route is A7 and **may not be measurable**; the `gammatone` 40-channel energy is the only spectral instrument that could carry it |
| AIRWAY | `respiration-and-cough-v2-threebreathsnose` (699), `-threebreathsmouth` (699) | **A, counted (3) and routed**, the route differing *between the two families* and stated in each instruction | Same as above. These two plus the `fivebreaths` index split are the only places the route is a declared contrast, and are therefore the only design that could ever validate A7 |
| AIRWAY | `respiration-and-cough-threequickbreaths` (1,718), `-v2-threebreaths` (699) | **A, counted (3) and timed.** Exhale, then inhale *quickly*. The **interval** is the measurement — a count of three says nothing about whether they were quick | Inter-onset intervals over the counted events; depends entirely on A5's boundaries. **Owed a measurement** |
| AIRWAY | `respiration-and-cough-breath` (1,788), `-v2-breath` (699), `breath-sounds` | **A, uncounted and durational.** Comfortable breathing for a stated duration — 30 s v1, 20 s v2, v2 specifying through the mouth. `expected_event_count` is absent for these | File-level presence is **implementable today**: the `airway.breath` gate reads `residual` `energy_fraction`, and `enhanced_hear_scores` / `residual_hear_scores` carry the `Breathe` label per 2 s window. `declared_duration_s` against actual is free. Per-cycle structure is A5 again |
| AIRWAY | any SPEECH- or VOICE-declared family | **no expected pattern.** AIRWAY routed broadly and its gate evidence is `unavailable` on 56,505 of 62,547 recordings | It labels airway evidence where it finds it and contests what was proposed and is not there. A breath during passage reading is **not** a deviation — it is how S4 measures breath-group structure |

### DDK

DDK is a declared branch with **no node**: `BRANCHES = ("AIRWAY", "SPEECH", "VOICE", "DDK")`
(`vocabulary.py:31`), and `run.py:302-306` marks it `SKIPPED` with
`NO_NODE = "no node implements this branch"`. The graph *can* route to it — two gates do — and then
records that nothing implements it. It is evaluated here on the same terms as the others, per the
owner's instruction that a declared branch is assessed independently of whether it is implemented.

| branch | task | expected pattern | detection approach |
| --- | --- | --- | --- |
| DDK | `diadochokinesis-pa` (896), `-ta` (896), `-ka` (896) | **A, counted, alternating.** One syllable repeated *as fast as possible*, **10 times** — `expected_event_count: 10`. v1 states the count; v2 does not | Envelope modulation spectrum over the located train (D1), cross-checked by syllable-nucleus rate (D2). **Owed a measurement** — D1–D6 are all unbuilt. The gate that fires without a transcript reads `ppg.segment_rate_per_s`, which is **owed a code change**: `extract_ppg_segments` exists (`tasks/features_extraction/ppg.py:349`) but is called only from offline `routing_analysis`; the store holds the whole-file posteriorgram alone, with no per-span query |
| DDK | `diadochokinesis-v2-puh` (702), `-tuh` (702), `-kuh` (702) | **A, uncounted, alternating.** Same repetition *until the timer runs out*, so no `expected_event_count`. The `'puhpuhpuhpuhpuhpuh'` in the instruction is an orthographic illustration, **not** a six-repetition instruction | As above. Train duration as a fraction of the recording is D5, a `counts` entry |
| DDK | `diadochokinesis-pataka` (896), `-v2-puhtuhkuh` (701) | **A, ordered and cyclic.** A three-place sequence repeated in order — sequential motion rate. `/pa-pa-pa/` is a collapse of the sequence and is the clinically meaningful finding | D6 sequence conformance, emitting `syllable_sequence_mismatch` — deliberately **not** `stimulus_mismatch`, since there is no stimulus text and no lexical expectation. **Owed a measurement**: the PPG-phoneme → expected-syllable mapping is unmeasured and the posteriorgram is out of domain on rapid nonsense repetition ([`branch-ddk.md:318-323`](branch-ddk.md)) |
| DDK | `diadochokinesis-buttercup` (896), `-v2-buttercup` (702) | **L&A.** A real English word repeated — so unlike every other DDK family this one **does** have a lexical pattern, and the recognisers will produce it | Consensus words give the repeat count directly via `transcript_repeat`. The only DDK family where the lexical route is the right one, and the only one where `ddk.lexical_repetition >= 3` fires for the correct reason rather than on function-word repetition. **Owed a code change** only (getting `transcript_repeat` into the store) |
| DDK | any lexical-speech family | **no expected pattern.** DDK routed 22,363 against 7,989 declaring a DDK family; repetition occurs in ordinary speech — a stutter, a false start, a repeated word | The branch measures what it finds and says what it is; it does not assert that a Harvard sentence failed to be a DDK task ([`branch-ddk.md:66-70`](branch-ddk.md)) |

QUALITY is not in `BRANCHES` — `vocabulary.py:28-29` calls it "a graph edge, never a branch" — so it
has no rows here. Its Q5 acquisition-consistency capability is the natural home for
`declared_duration_s` against measured duration, which several rows above want.

---

## The worked example: `prolonged-vowel`

The owner named this case, and it is where the whole argument is visible.

**The instruction**, verbatim from the recording-grain sidecar:

> This task helps us analyze features in your voice. Please press the play button to listen to the
> demonstration on how to complete the task. Then, tap the record button and imitate the speaker by
> repeating the sentence "1, 2, 3 aah" in your normal voice. Please hold the sound "aah" until the
> timer runs out.

**Two expected patterns, and the instruction states their order:**

| | pattern | kind | what it is |
| --- | --- | --- | --- |
| P1 | `one`, `two`, `three` | lexical, ordered | three tokens, once |
| P2 | a held /a/ | acoustic | one continuous voiced production, single vowel, F0 holding, to the timer |

`stimulus_text` is **empty** for this family at both grains; the pattern exists only inside
`instructions`. All three local recordings are exactly 12.1 s, the timer.

**Only P2 is the measurement.** Every F0, jitter, shimmer, HNR and CPPS figure for this recording
should be taken over the vowel alone.

### What happens today

**The scalars are whole-file, and the file contains the count-in.** `praat_features`
(`preprocess.py:860`) is documented as *"Praat/Parselmouth's whole-file feature set over the
`enhanced` stream"*. It resolves the whole `enhanced` `Audio` and passes it unsliced. Every Praat
aggregation then takes the whole-object time range:

- `extract_pitch_descriptors` — `call(pitch, "Get mean", 0, 0, unit)` (`praat_parselmouth.py:617`);
- `extract_harmonicity_descriptors` — `call(harmonicity, "Get mean", 0, 0)` (`:740`);
- `extract_jitter` / `extract_shimmer` — `call(..., "Get jitter (…)", 0, 0, …)` (`:1370`, `:1425`);
- `extract_cpp_descriptors` — the mean over every finite frame of the file's cepstrogram (`:1030`).

Praat's `Get mean` on a **Pitch** object intrinsically skips unvoiced frames; that is the only
voicing mask anywhere in the set, and it comes from Praat, not the graph. HNR, CPPS, jitter and
shimmer carry no mask at all.

So on a `prolonged-vowel` recording `mean_f0_hertz` is the mean over the counted `one two three`
**and** the vowel; `mean_hnr_db` and `mean_cpp` additionally average in the silences and the
count-in's consonants. The figures are not the vowel's.

**And VOICE emits no scalars at all.** Its subject is `family == "phonation"` spans (`voice.py:40`,
`:227-231`). Nothing in the repository writes such a span — the detector was retired 2026-09-04 — so
the branch takes the no-span return at `:235-264` and returns `Outcome.FAIL` with `spans_n: 0`,
`phonation_s: 0.0`, on every recording. Measured: 6 of 6 VOICE-routed recordings in the 2026-09-15
run ([`branch-voice.md:11-24`](branch-voice.md)). `period_marks` and `voice_tracks` are never
written.

**The spans that would carry the vowel are in the same store, live, unread.** PREPROCESS's `spans`
block writes amplitude spans (`preprocess.py:1568`) carrying `signal`, `measure`,
`peak_over_floor_db`, `k_db`, `merged_proposals`, `contains_clip` — and **no `family` key at all**,
which is exactly what VOICE's selector excludes. The routing gate `voice.sustained` reads
`[span_longest, amplitude]` and measured **15.89 s** on the MPT recording and **12.27 s** on a glide
in the same run. The recording is routed to VOICE *by* those spans and then fails *for want of* them.

### What the decomposition would change

| | today | with the decomposition |
| --- | --- | --- |
| P1 found | SPEECH transcribes it and it scores an `extra` against the reference set | a SPEECH `label` on the count-in's extent, matched to `expected_speech` |
| P2 found | nothing; VOICE `FAIL` | a VOICE `label` (or `refine`) on the vowel's extent |
| the scalars | whole-file, over count-in + silence + vowel | over P2's extent alone |
| "was the task done" | unanswerable | P1 found / absent and P2 found / absent, independently |
| the task extent | not recorded | `trim` carrying `task_extent` = P2's extent |
| anything else | silent | `off_task_extent` over the `gap` spans it falls in |

### What is owed to build it

1. **A phonation subject VOICE can select.** Either the ruleset stamps a label on the amplitude span
   that fired `voice.sustained` (the owner's 2026-09-15 decision), or VOICE's selector widens. Both
   are **owed a code change**; whether a labelled `amplitude` span is thereby refinable is unresolved
   at [`branch-conventions.md:60-70`](branch-conventions.md).
2. **A separator between the count-in and the vowel.** The lexical half is free — consensus `word`
   entities carry `extent` plus per-source `timings`, so `one two three` locates itself and P2 is the
   voiced production under no lexical word. The acoustic half is V1's stationarity qualifier and is
   **owed a measurement**: nothing named `spectral_flux` or `stationarity` exists anywhere in
   `src/senselab`, and none of F0 stationarity, formant stationarity or spectral flux has a named
   statistic or a window.
3. **Scalars taken over an extent.** `extract_praat_parselmouth_features_from_audios` has no extent
   argument, while every Praat call it makes already accepts a time range and is passed `0, 0`. So
   this is **owed a code change**, not a measurement — but to a function outside the triage module
   with its own consumers.
4. **A rule for where the pattern comes from.** `expected_speech = ["one","two","three"]` cannot be
   read off `stimulus_text`, which is empty. It comes from the human-read expectations table
   `design.md:336-350` specifies, which does not exist.

**Note what is not owed.** The count-in is lexical and needs no new detector. And
`words.onomatopoeic_tokens` does **not** help here — its vocabulary is
`[cough, coughs, coughing, 咳, 呵, ahem, hack, khh, kof, cof]` (`default.yaml:123`), a cough set. The
bracketing rule it drives moves cough-like renderings into the bracketed channel, which is what keeps
`[cough]` out of a lexical word count; it says nothing about numerals. A sustained vowel is not
lexical and no lexical mechanism will find it — which is the general shape of this whole document:
**the lexical half of every pattern is nearly free, and the acoustic half is where everything is
owed.**

---

## The measurements a method here may read

Verified present in the store as of `9fec73c8`. Anything not on this list does not exist.

| measurement / entity | granularity | what a pattern match uses it for |
| --- | --- | --- |
| `word` entities (consensus) — `text`, `bracketed`, `outcome`, `index`, `extent`, per-source `timings`, `readings`, `agreement`, `onset_spread_s`, `temporal_uncertainty_s` | per word | every lexical pattern. Two ASR sources (CrisperWhisper 2.0 turbo, Qwen3-ASR-1.7B), so `agreement` ∈ {0.5, 1.0} |
| `consensus_transcript` | per file | the whole-transcript half of a stimulus comparison |
| `span` with `measure ∈ {amplitude, continuity, asr, gap}`, `peak_over_floor_db`, `merged_proposals` | per span | every acoustic extent. `gap` spans are the carrier for material matching nothing |
| `span` with `family: "clip"`, `family: "speech"` | per span | quality and lexical-run extents |
| `span_hear`, `span_yamnet` (joined on `span_id`) | per span | what a span sounds like. HeAR's eight labels are `Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear, Laugh, Speech`. **There is no `span_ast`** |
| `yamnet_window` / `ast_window` / `hear_window` + the `*_windows` roll-ups | per window | YAMNet 0.96 s / 0.48 s hop; AST 10.24 s non-overlapping; HeAR fixed 2.0 s |
| `enhanced_*` / `residual_*` scores and `*_summary_all` / `*_summary_speech_free` | per window, per file | whether a pattern survives enhancement or lives in the residual |
| `consensus_taxonomy`, `<classifier>_label_summary` | per file | the file-level label picture |
| `phonation_tracks` + `derivatives/phonation_tracks.npz` — `times_s`, `f0_hz`, `strength`, `formant_times_s`, `f1..f4_hz`, `f1..f4_bw_hz`, hop 0.01 s | per frame | every F0 and formant pattern. F0 on `preemphasised`, formants on `plain` — **not the same signal as the scalars**, which are on `enhanced` |
| `praat_features` — ~40 scalars over `enhanced` | per file, whole-file | today's F0 / jitter / shimmer / HNR / CPPS |
| `ppg_posteriorgram` + npz — 40 ARPAbet labels including `<silent>` | per frame, whole file | segment rate and silent fraction. **Not queryable per span**; `extract_ppg_segments` runs at read time in `routing_analysis` and stores nothing |
| `energy_envelope`, `normalized_envelope` (dBFS), `continuity_trace`, `gammatone` (40 ERB channels, hop 0.005 s), the two spectrograms | per frame | extent proposal, and the only spectral instrument that could carry nasal/oral route |
| `residual` — `energy_fraction`, `gain_db`, `bands`, `speech_coverage_fraction` | per file | the `airway.breath` gate |
| `silence` — YAMNet `Silence` projected onto the 0.96/0.48 grid | per window | the only presence/absence-of-sound gate in the graph |
| `level`, `disruptions_file`, `clip_amplitude`, `squim` assertions | per file / per span | effort, and quality covariates on every value |
| `speaker` entities + `diarization_interval` | per turn | who spoke. **Scoped to the lexical word hull only** |

**Absent, and no method here may assume them:** any VAD (no voice-activity model runs anywhere in
the graph — "speech activity" is word-timing- or YAMNet-`Silence`-derived); a whole-file diarization
derivative (settled in docs at `9fec73c8`, which is docs-only; nothing writes one); per-span PPG;
`span_ast`; a phonation-family span producer; per-word ASR confidence on the consensus word; a DDK
node.

## What is missing, consolidated

**Owed a measurement** — nothing in the tree produces it:

| what | who needs it |
| --- | --- |
| F0 stationarity, formant stationarity, spectral flux (each with a named statistic and a window) | VOICE V1; every sustained family; and every non-voice family VOICE routes. The single most load-bearing gap in this document |
| breath / cough event onset and offset | AIRWAY A5, A6; every counted airway family. Today an "event" is a label-carrying amplitude span, and `spans.min_separation_ms` merges adjacent events upstream |
| nasal versus oral route | AIRWAY A7, six families. May not be measurable; `gammatone` is the only candidate instrument |
| lexical / semantic overlap | SPEECH, `story-recall`, the `free-speech` anti-pattern |
| DDK envelope modulation spectrum, syllable-nucleus rate, inter-onset intervals | DDK D1–D3 |
| PPG phoneme → expected syllable mapping | DDK D6 |
| category membership of a produced word | SPEECH, `random-item-generation`, `animal-fluency` |
| vocal tremor | VOICE V4a |
| effort without a within-recording contrast | AIRWAY `hardcough`, VOICE `loudness` v1 |

**Owed a code change** — the computation exists, its output does not reach the reader:

| what | where it exists | why it does not reach a branch |
| --- | --- | --- |
| a phonation subject for VOICE | `preprocess.py:1568` amplitude spans | they carry no `family`; VOICE selects `family == "phonation"` (`voice.py:230`) |
| PPG segments | `tasks/features_extraction/ppg.py:349` | called only from `routing_analysis/features.py:396`; the store holds the whole-file posteriorgram |
| `transcript_repeat` | `routing_analysis/features.py` | a ruleset feature, not a store measurement |
| Praat scalars over an extent | every Praat call already takes a time range | `extract_praat_parselmouth_features_from_audios` has no extent parameter and passes `0, 0` |
| a populated `expected_speech` | `AudioHints.expected_speech` reaches every branch | `runs/b2ai-v2/make_hints.py` parses the filename and never reads `stimulus_text` |
| the expected pattern at the right grain | the per-recording sidecar | nothing reads it; and `task_family` collapses the index that carries the condition |
| whole-file diarization | pyannote runs inside SPEECH over the lexical word hull (`speech.py:642-646`) | moving it to PREPROCESS over `enhanced` and `residual` is settled but docs-only |
| sentence boundaries for CAPE-V | would fall out of S3 alignment | S3 is not built, and they would have to reach VOICE as selectable spans |
| the `(v2)` spelling | `families.py` / `task_id_of` | 22 of 112 local recordings reach no declared family |

**Owed a cut** — the approach is complete, the operating point is not fitted:

omission detection from forced-alignment score (SPEECH S3); the dominant-monotone-segment criterion
(VOICE V3); verbatim-overlap for the `free-speech` anti-pattern and the read-rather-than-recalled
case; A5's envelope smoothing window, peak/trough criterion and minimum event duration; what makes a
`gap` span off-task rather than ordinary silence. Each belongs in `data/` with a written derivation,
fitted against measured verdicts and **not** against the declared families — a cut fitted on
declarations encodes which recordings the protocol labelled, not which carry the pattern.

## What could not be settled

- **Which verb carries `off_task_extent`** — `trim` per `design.md:566`, `deviate` per
  `branch-conventions.md:77` and `branch-airway.md:348`. Both documents are current. This document
  uses `trim` for `task_extent` and leaves the deviation's verb to whoever reconciles the two.
- **Whether `trim` has an emitter at all.** No branch document claims it, and it is in neither the
  nine implementation pieces (`design.md:902-943`) nor the unresolved list (`:947-1004`). Since
  `task_extent` is the direct answer to the owner's question, this is the gap that most needs an
  owner.
- **Whether the corpus spells `v2` with parentheses.** The local v3.1 copy does; the corpus profile's
  non-zero `*-v2` counts say the 62,547-recording corpus does not, or was normalised differently.
  Until that is settled, every per-family figure for a `v2` family is of uncertain provenance.
- **42 against 44** for the `stimulus_text` grain disagreement. Same direction, same conclusion,
  different count; the definitions behind the two measurements should be reconciled.
- **Whether `maximum-phonation-time`'s "we will repeat this task 3 times" means three recordings or
  three attempts within one.** The local corpus has `-1`, `-2`, `-3` as separate recordings, which
  argues for three recordings — but `repeat_attempt` is already a declared VOICE deviation, so the
  two readings produce opposite findings on the same audio. Settling it needs the protocol.
- **Whether a label on an `amplitude` span makes it refinable**, which decides whether the
  prolonged-vowel decomposition is a `refine` or a `propose`. Unresolved at
  `branch-conventions.md:60-70`, `design.md:958-962`, `branch-voice.md:86-93`.
- **`loudness` / `loudness-v2` family-set membership.** `families.py:41-42` puts them in
  `LEXICAL_SPEECH`; the protocol says `speech_type: "non-lexical"` and the only word is `hey`.
  Resolve in `families.py`, not in a branch document.
- **`high-to-low` (43 recordings)** is declared in `VOICE_ELICITING`, has zero recordings in the
  local copy, and may be a dead alias for `glides-high-to-low`. Its instruction was never read and
  its row is assumed.
- **Eight `LEXICAL_SPEECH` families and two `AIRWAY_ELICITING` families** (`animal-fluency`,
  `open-response-questions`, `picture-description-option1`, `random-item-generation-v2`,
  `cape-v-sentences` v1, `breath-sounds`, `voluntary-cough`, and the below-cut remainder) have no
  recordings in the local copy. Their rows are inferred from siblings and are unverified.

## Appendix — the instruction per family, verbatim

Read from `<stem>_recording-metadata.json` in `~/Downloads/b2ai_v31_bids_07_01_v3`, 2026-09-15.
Within a task id, `instructions` and `stimulus_text` are byte-identical across subjects, so each
entry is exact rather than a sample. 40 of the 48 declared families appear; the `(v2)` families are
listed under their declared names with a note.

### VOICE_ELICITING

- **`prolonged-vowel`** — *"This task helps us analyze features in your voice. Please press the play
  button to listen to the demonstration on how to complete the task. Then, tap the record button and
  imitate the speaker by repeating the sentence "1, 2, 3 aah" in your normal voice. Please hold the
  sound "aah" until the timer runs out."* `stimulus_text` empty. Three local recordings, all 12.1 s.
- **`maximum-phonation-time`** (`-1`, `-2`, `-3`) — *"This task helps us analyze the way your
  breathing is connected to your voice. Take a very deep breath and hold out "ah" for as long as
  possible until you completely run out of air. We will repeat this task 3 times."* `stimulus_text`
  empty. The inhale is instructed before the record tap is mentioned.
- **`maximum-phonation-time-v2`** (spelled `(v2)`) — *"…Take a very deep breath, **tap on the record
  button**, and hold out "ah" for as long as possible until you completely run out of air. Try to
  hold it until the progress bar reaches the end of the screen."* The inhale is explicitly placed
  before the tap.
- **`glides-low-to-high`** — *"Please watch the video to help you understand the next task and
  imitate what the speaker does. When you're ready, press the record button and use the sound "ee" to
  gradually move from your lowest note to your highest."*
- **`glides-high-to-low`** — *"Next, please watch this video for the following task and imitate what
  the speaker does. When you're ready, press the record button again and use the sound "ee" to
  gradually move from your highest note to your lowest note."*
- **`high-to-low`** — absent from the local copy; instruction unread.

### SYLLABLE_REPETITION

- **`diadochokinesis-ka` / `-pa` / `-ta`** — *"This task helps us analyze the ease and precision of
  speech sound productions. Record yourself repeating the syllable /KA/ as fast as possible 10
  times."* (`/PA/`, `/TA/` respectively.)
- **`diadochokinesis-pataka`** — *"…Now repeat the word /Pataka/ as fast as possible 10 times."*
- **`diadochokinesis-buttercup`** — *"…Now repeat the word /buttercup/ as fast as possible 10
  times."*
- **`diadochokinesis-v2-puh` / `-tuh` / `-kuh`** (spelled `(v2)`) — *"…Please press the play button
  to listen to the demonstration… Then, tap the record button and imitate the speaker by repeating
  the syllable 'puh' as quickly and consistently as possible until the timer runs out -
  'puhpuhpuhpuhpuhpuh'"*.
- **`diadochokinesis-v2-puhtuhkuh`** — as above, *"…repeating the syllables 'puhtuhkuh'… -
  'puhtuhkuhpuhtuhkuh'"*.
- **`diadochokinesis-v2-buttercup`** — as above, *"…repeating the word 'buttercup'…"*.

All ten have empty `stimulus_text`; the target token is only inside `instructions`.

### AIRWAY_ELICITING

- **`respiration-and-cough-cough`** — *"After pressing record, cough 5 times"*.
- **`respiration-and-cough-breath`** — *"First, let's hear you breathe comfortably for 30 seconds"*.
- **`respiration-and-cough-fivebreaths`** — **two instructions by index.** `-1`, `-3`: *"After
  pressing on record, take 5 big breaths in and out through your nose with your mouth closed."*
  `-2`, `-4`: *"After pressing on record, take 5 big breaths in and out through your mouth."*
- **`respiration-and-cough-threequickbreaths`** — *"After pressing on record, exhale, then inhale
  quickly through your mouth, as if you are trying to catch your breath. Record 3 of these breaths in
  a single recording."*
- **`respiration-and-cough-v2-hardcough`** (spelled `(v2)`) — *"Please do not cover your mouth or
  place your hand between your mouth and the microphone during recording. Breathe normally, then when
  you are ready, tap the record button below and cough HARD as if something were stuck in your
  throat. Tap stop when finished."*
- **`respiration-and-cough-v2-breath`** — *"After pressing on record, breathe comfortably through
  your mouth for 20 seconds, until the timer runs out."*
- **`respiration-and-cough-v2-threebreathsmouth`** — *"After pressing on record, take 3 deep breaths
  in and out through your mouth. Tap stop when finished."*
- **`respiration-and-cough-v2-threebreathsnose`** — *"…take 3 deep breaths in and out through your
  nose with your mouth closed. Tap stop when finished."*
- **`respiration-and-cough-v2-threebreaths`** — *"After pressing on record, exhale normally, then
  inhale quickly through your mouth, as if you are trying to catch your breath. Record 3 of these
  breaths in a single recording. Tap stop when finished."*
- **`breath-sounds`, `voluntary-cough`** — absent from the local copy.

All have empty `stimulus_text`. No airway task names any word.

### LEXICAL_SPEECH

- **`harvard-sentences-list`** and **`cape-v-sentences-v2`** (spelled `(v2)`) — shared instruction
  *"Please read the following sentences out loud in your typical voice."* `stimulus_text` is one
  sentence per recording; 20 distinct Harvard sentences (lists 49 and 52) and the 6 CAPE-V sentences
  appear locally.
- **`rainbow-passage`** — *"This task helps us evaluate how you use breathing to support your voice.
  Please read the following passage out loud in your typical voice."* `stimulus_text`: the passage
  (338 chars).
- **`caterpillar-passage`** — *"This is a passage that contains speech sounds in English by sound
  frequency to test your ability to produce speech sounds. Please read the following passage out loud
  in your typical voice."* `stimulus_text`: the passage (1,035 chars).
- **`word-color-stroop`** — *"This exercise asks you to name out loud the color in which a word is
  displayed. Sometimes the word will be a color word: **do not read the word aloud**, just state the
  color in which it's displayed… Time limit = 5 seconds/item, 15 random words. Total time 75s."*
  `stimulus_text`: *"red brown green red blue green purple green green purple green red blue purple
  blue"* — the 15 expected **spoken answers**, not the displayed words.
- **`free-speech`** — *"This section is meant to hear you speak freely by answering an open-ended
  question. Please answer the following questions and record your answer. Keep talking until the time
  stops. **Do not record yourself reading the prompt** and avoid any information that could identify
  an individual."* `stimulus_text`: one of three questions, per index.
- **`free-speech-v2`** (spelled `(v2)`) — *"…Please answer the following questions as though you were
  having a conversation. We'll record about 30 seconds of your response."* `stimulus_text`: one of
  three different questions, per index.
- **`story-recall`** — *"You are given a text. Read the text so you familiarize yourself with it…
  When you are ready, you will be asked to recall the story. This can be in your own words."*
  `stimulus_text`: the grandfather passage (717 chars) — the **source**, not the expected utterance.
- **`story-recall-v2`** (spelled `(v2)`) — *"…please retell the story in as much detail as
  possible."* `stimulus_text`: the frog story (1,083 chars), with ten image assets.
- **`cinderella-story`** — *"You'll receive a hard copy of the Cinderella storybook from our study
  team to refresh your memory of the story. When ready, click the 'Record' button below to begin
  narrating the story."* `stimulus_text` empty.
- **`productive-vocabulary`** — *"Below you will be provided with a series of words… If you know the
  word, please provide a definition of the word… Once 6 words are defined, the 'Done' button
  appears."* `stimulus_text`: the cue word, one per index — `sunset`, `membership`, `plagiarize`,
  `entail`, `conga`, `maceration`.
- **`picture-description`** — *"Tell me everything you see going on in this picture."*
  `stimulus_text` empty; the stimulus is an image URL in the sidecar.
- **`picture-description-option2`** — *"Describe everything that is happening in the picture (as
  though describing it for the blind), trying to use complete sentences."*
- **`random-item-generation`** — *"Say as many items from the following category as you can. **Do not
  repeat any item.** Your goal is to list as many as possible… Category: English words starting with
  't'."* `stimulus_text` empty — the category is only in the per-recording `instructions`.
- **`loudness`** — *"This helps us to determine the loudness of the voice. When you are ready, press
  on the record button and shout "hey" as loud as possible 3 times in a single recording."*
- **`loudness-v2`** (spelled `(v2)`) — *"…press the record button and say "hey" in your normal voice.
  Then, shout "hey" as loud as you can. Try to reach the target line on the screen."*
- **`animal-fluency`, `cape-v-sentences` (v1), `open-response-questions`,
  `picture-description-option1`, `random-item-generation-v2`** — absent from the local copy.
