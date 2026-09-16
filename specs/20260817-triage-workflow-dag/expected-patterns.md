# Expected patterns — what each task's instruction asks for, and what would match it

The owner's question: *"is there a list of what algorithm/method/heuristic is used to determine
whether a task was done and what the start end spans are for a task."*

**There is not.** This document is that list.

It is a **method inventory**, not a threshold fit. No number in it is fitted. Where a method needs a
boundary it says what would fit it and marks it owed, per the project rule that a threshold lives in
`data/` with a written derivation and never as a code literal.

Each method is written as a **named function over named store derivatives** — the table's detection
column carries the call, [§ The detection functions](#the-detection-functions) carries the body, and
every family's complete derivative requirement is stated once in that family's block rather than
assembled from its rows. Two things follow from the form. An input that does not exist has to be
*named* to be passed, so a row that cannot be implemented says which derivative it is waiting on
rather than reading as prose about a gap. And every function returns whether the task was done, so
**a declared family that produced none of its patterns returns *not done*** — the declaration is
never the answer.

Every corpus figure below was measured on 2026-09-15 over
`/orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04` — 2,005 subjects,
70,520 wav — read-only, by
[`runs/corpus-instructions/scan_instructions.py`](runs/corpus-instructions/scan_instructions.py),
and is reported over the **62,547** triaged recordings that remain once `audio-check` and
`audio-check-v2` are excluded — the corpus profile's count, three short of the 62,550 wav the same
48 families hold, for reasons given below. Instruction and `stimulus_text` counts are over recording
sidecars, of which there are 426 more than there are wav. Durations are `recording_duration` from
the sidecar, and a figure like "12-13 s" means the declared float floored to whole seconds.

An earlier version of this document was written against a 112-recording temporary copy and several
of its findings were artefacts of that copy's filename spelling; where a claim has been withdrawn,
the correction says so rather than deleting the trace.

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
938 of 1,258 transcripts agree ([`dag.md:185`](dag.md)), and the other 320 are data.

**Declared families are not ground truth.** The declaration says *what to look for*, never *what is
there*. A participant who did not follow the instruction is precisely what the pattern match exists
to find; a method that scores the recording against the declaration is measuring the declaration.
This is the same line
[`../20260913-branch-contract-and-hints/design.md:942-948`](../20260913-branch-contract-and-hints/design.md)
draws for the two counts.

## The family is the wrong grain for an expected pattern

This is the structural finding of the exercise, and it precedes every row of the table.

`task_family` collapses trailing numeric segments (`families.py:134-144`), and the module docstring
says that it "folds a repeat index and the Harvard list index and nothing else" (`families.py:3`).
**Measured over the whole dataset, that is not what the trailing index always is.** Two cases where
the index carries a *condition*, both read off the recording-grain sidecars:

- **`respiration-and-cough-fivebreaths` carries two different instructions by index, in an exact
  half-and-half split.** `-1` and `-3`: *"After pressing on record, take 5 big breaths in and out
  through your nose with your mouth closed."* — **1,778** recordings. `-2` and `-4`: *"After
  pressing on record, take 5 big breaths in and out through your mouth."* — **1,778** recordings,
  889 per index in each case. The index is a **route** index, not a repeat index, and the family
  name loses the route — which is exactly what AIRWAY A7 would be measuring. All **894** sessions
  that carry the task carry all four indices, with no exception, so nose and mouth are always the
  same participant in the same session.
- **`maximum-phonation-time-v2` carries an effort escalation by index.** `-1` (676 English
  recordings) is the plain instruction; `-2` (108) appends *"Now try to hold out "ah" for even
  longer."* That is a **within-subject duration contrast**, the only one in the corpus for
  phonation, and collapsing to the family discards it. v1 does not split: `-1`, `-2` and `-3` carry
  identical wording, which is consistent with its *"We will repeat this task 3 times"*.

And four cases where the *stimulus* is per recording, so the family has no pattern to look up:

- **`harvard-sentences-list`** — **1,060** distinct `stimulus_text` values over 13,705 recordings.
  Collapsing to the family discards which sentence was expected, which is the whole lexical pattern.
- **`productive-vocabulary`** — **204** distinct cue words over 2,910 recordings, and 78 recordings
  carry no cue at all.
- **`word-color-stroop`** — **472** distinct colour sequences over 472 recordings: one per
  recording, never shared.
- **`cape-v-sentences` and `-v2` are not the same six sentences.** Three are shared (*"We were away
  a year ago."*, *"The blue spot is on the key again."*, *"My mama makes lemon muffins."*); v1's
  other three are *"How hard did he hit him?"*, *"Peter will keep at the peak."*, *"We eat eggs
  every Easter."*, and v2's are *"I eat eggs every morning."*, *"He helped her hurry home."*,
  *"Papa took a piece of cake."* A sentence list attached to the wrong version is wrong on half of
  it.

So: **the expected pattern is a property of the recording, carried by its own sidecar. The family is
a routing and scoring unit and nothing more.** The table below is keyed by family because that is
how the corpus is counted; every row's pattern is resolved per recording.

## `families.py` matches the corpus

An earlier draft of this document reported that 22 of 112 recordings reached no declared family,
that 15 families were undeclared, and that all 16 declared `*-v2` families had zero recordings.
**Every one of those claims was an artefact** of a temporary 112-recording copy whose BIDS filenames
spelled the version marker with capitals and parentheses — `task-Respiration-and-cough-(v2)-HardCough`.
The authoritative tree does not spell them that way, and the copy has been retired.

Measured over `/orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04`
(2,005 subjects, 70,520 wav, 70,946 `_recording-metadata.json`, 21,725 `_acoustictask-metadata.json`)
on 2026-09-15:

| | |
| --- | --- |
| distinct task families observed | **50** |
| declared in `families.py` | **48** |
| undeclared | **2** — `audio-check`, `audio-check-v2`, excluded from triage (below) |
| declared families with **zero** recordings | **0** |
| task ids containing a parenthesis | **0** |
| wav over the 48 declared families | **62,550** |
| the corpus profile's `n` over the same 48 | **62,547** |

Those last two figures are the argument. They agree to three recordings, and the three are
named: `maximum-phonation-time` (profile 2,696 against 2,697 wav), `rainbow-passage` (897 / 898)
and `random-item-generation-v2` (207 / 208). So `task_id_of` and `task_family`
(`families.py:121-131`, `:134-144`) normalise this tree with nothing left over — matching the
corpus run's own **unassigned families: 0** over 62,547 recordings. The unit test's
`cape-v-sentences-v2-4` (`routing_analysis_test.py:285`) is the corpus spelling, not an invented
one. Nothing here is owed to `families.py`.

### `audio-check` is excluded, and is where the ~8,000 difference goes

`audio-check` (4,202 wav) and `audio-check-v2` (3,768 wav) are the two largest task names in the
dataset — **7,970 together, 11.3% of its 70,520 wav**. Neither appears in `families.py`, in any
`DECLARED_KIND` set, or among the 48 families of the corpus routing table. They are microphone
checks: their recording sidecars carry an **empty `instructions`** string on all 8,136 of them, the
only families in the tree that do, and 1,342 of them are under a second long.

70,520 − 7,970 = 62,550, against **62,578** stores on disk
([`design.md:1222`](../20260913-branch-contract-and-hints/design.md)) and **62,547** scored
([`../20260910-taxonomy-routing-evidence/measurements.md:19`](../20260910-taxonomy-routing-evidence/measurements.md)).
A reader comparing the dataset's
70,520 against the corpus's 62,547 should reach for this exclusion rather than assume a loss.

**They get no row below**, and every per-family figure in this document is over the **62,547**
triaged population, never over 70,520.

## Where the instruction comes from

Two JSON sidecars per recording, at two grains
([`../20260913-branch-contract-and-hints/design.md:256-280`](../20260913-branch-contract-and-hints/design.md)).
Both live in the same `audio/` directory, linked by `recording_acoustic_task_id` →
`acoustic_task_id`. The acoustictask grain is **one sidecar per acoustic task**, and an acoustic
task spans several recording families: 21,725 acoustictask sidecars against 70,946 recording
sidecars. That ratio is the whole mechanism of what follows.

**1,072 recordings do not link at all** — their `recording_acoustic_task_id` resolves to no sidecar
in the session — and **every one of them is `free-speech`**, 1,072 of that family's 3,074. The
figures below are over the 61,738 triaged recordings that do link.

**The per-recording grain is the only source of the prompt.** Measured over the whole dataset,
`audio-check*` excluded:

| comparison of the two grains | n | of 61,738 |
| --- | --- | --- |
| `stimulus_text` identical, both empty | 33,514 | 54.3% |
| `stimulus_text` identical, both non-empty | 3,610 | 5.8% |
| `stimulus_text` — acoustictask **empty**, recording has text | 20,858 | 33.8% |
| `stimulus_text` — recording empty, acoustictask has text | 0 | — |
| `stimulus_text` — both non-empty and **different** | 3,756 | 6.1% |
| **`stimulus_text` differs, any way** | **24,614** | **39.9%** |
| **`instructions` differ** | **40,183** | **65.1%** |
| either field differs | 45,764 | 74.1% |

`design.md:388` records 44 of 112 on the retired copy; the fraction-of-112 figures it and an earlier
draft of this document carried are superseded by these, which are over the corpus. The conclusion is
unchanged and now has a mechanism: **never read the prompt from the acoustictask JSON.**

Three details, all of which matter:

- **The acoustictask `stimulus_text` for `free-speech` is one frozen string** — *"Can you explain
  your voice/speech problems and why you consulted a physician for them?"* — on all 912 v1 and all
  707 v2 sidecars. In v1 that string **is** one of the four questions actually asked, and is right
  on 380 of 3,074 recordings; the other three questions are asked 898 times each. In `free-speech-v2`
  it matches **none** of the six real questions (three English, three Spanish), so the family grain
  is confidently wrong on all 2,120. The earlier claim that it matches none of the real questions
  was right for v2 and wrong for v1, and the distinction is the point: a builder reading the family
  grain gets a plausible prompt that is right on 380 of the 5,194 `free-speech` recordings and wrong
  on the rest.
- **`instructions` disagree more often than `stimulus_text` does (65.1% against 39.9%)**, and the
  mechanism is the grain ratio, not staleness. One acoustictask instruction has to stand for every
  recording family under it, so it can only match one of them. Measured: the `diadochokinesis`
  acoustictask carries the ***buttercup*** wording, so 891 of 896 differ on each of `-ka`, `-pa`,
  `-ta` and `-pataka`; the `glides` acoustictask carries the **low-to-high** wording, so
  `glides-low-to-high` agrees on all 1,597 and `glides-high-to-low` is described **backwards** on
  1,525 of 1,554; the `respiration-and-cough` acoustictask carries only *"Breathing sounds can also
  provide information on your health. Let's record them."*, which names neither a count nor a route,
  so **all four** of its v1 families disagree — 3,556 of 3,576 on `fivebreaths` alone. The v2
  acoustictask is the mirror image: it carries the `hardcough` wording, so `v2-hardcough` agrees on
  all 699 and its four siblings each differ on 675 of 699. Since the
  instruction is where the pattern comes from, this is the same defect as the `stimulus_text` one
  and it is larger.
- **The Spanish sidecars are not translations of the same specificity.** 1,417 of 70,946 recordings
  are `es-419`. Forty-one of the 48 families carry exactly one Spanish instruction, six have no
  Spanish recordings at all, and `random-item-generation-v2` carries **five** — one per category,
  as in English. But for the airway families the Spanish string is the *acoustictask* text —
  *"Los sonidos de la respiración también pueden proporcionar información sobre su salud. Vamos a
  grabarlos."* —
  which carries neither the count nor the route. So for the 20 Spanish `fivebreaths` recordings the
  expected pattern is not recoverable from the sidecar at all, and the same holds across the
  `respiration-and-cough` families. Any per-family pattern table must be read as English-only.

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

**`stimulus_text` is empty on every recording of 36 of the 48 families — 33,430 of 62,547, 53.4% of
the corpus.** All ten DDK families, all eleven airway families, both loudness families, all three
glide families, `prolonged-vowel` and both `maximum-phonation-time` families — 29 in all, as
expected — but also **seven *lexical* families** where the instruction names the target and the
sidecar does not: `animal-fluency`, `cinderella-story`, all three `picture-description` variants,
and both `random-item-generation` families. The target token of a DDK task, the count in a cough
task, the category in a fluency task and the `1, 2, 3 aah` of a prolonged vowel all live **only inside
`instructions`**. `productive-vocabulary` is the partial case: 2,832 of 2,910 carry their cue word
and 78 carry nothing.

So for the majority of the corpus the expected pattern is not machine-readable at all and must come
from the human-read expectations table `design.md:336-350` specifies and which does not exist yet.
The twelve families that do carry `stimulus_text` account for 29,117 recordings.

### `expected_speech` is available at the point of use

`AudioHints` reaches every branch: `run.py:298-300` passes `hint` into `airway(...)`, `speech(...)`
and `voice(...)`, and `quality(...)` at `:310` takes it too. The only occurrence of `expected_speech`
in `src/senselab` is the field itself (`audio_hints.py:142`, `:152`) — **no shipped code reads it**,
because no branch implements the comparison yet. The tests do
(`src/tests/audio/data_structures/audio_hints_test.py`,
`src/tests/audio/workflows/triage/nodes/speech_test.py:746`), which is what keeps the field honest
while the consumer is missing.

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
  [`branch-speech.md:136-140`](branch-speech.md) requires recogniser agreement to travel with every
  mismatch.

## How many instructions each family actually has

The brief for this document assumed version drift *within* a family, of the kind the
`loudness` / `loudness-v2` pair shows between families. **There is almost none.** Counting distinct
`instructions` strings over the English recordings of each declared family, 44 of the 48 carry
**exactly one**. The instruction is a constant per family, which is what makes a per-family
expected-pattern table a legitimate object at all — with four exceptions, and each of the four is a
finding rather than noise:

- **`respiration-and-cough-fivebreaths` (2)** — the nose / mouth route, split by trailing index.
- **`maximum-phonation-time-v2` (2)** — the *"even longer"* escalation on index `-2`.
- **`random-item-generation` (10)** and **`-v2` (10)** — one instruction per category, and two of
  the ten **invert the task's own negative constraint** (below).

Version drift between families is real and is what the `v1`/`v2` pairs carry: `loudness` asks for
*"shout "hey" as loud as possible 3 times"* while `loudness-v2` asks for *"say "hey" in your normal
voice. Then, shout "hey" as loud as you can."* — a counted task and a contrastive one under names
that differ by a suffix. The corpus declares **16** `*-v2` families, of which **8** have an exact v1
counterpart by name (`cape-v-sentences`, `diadochokinesis-buttercup`, `free-speech`, `loudness`,
`maximum-phonation-time`, `random-item-generation`, `respiration-and-cough-breath`, `story-recall`);
the other eight renamed the target as well as the wording, which is why `diadochokinesis-pa` and
`diadochokinesis-v2-puh` are separate families rather than two versions of one. Every pair is a
behavioural difference, not a rewording, and they are set out row by row in the table.

| family | n | distinct English `instructions` | distinct `stimulus_text` |
| --- | ---: | ---: | ---: |
| `harvard-sentences-list` | 13,705 | 1 | 1060 |
| `respiration-and-cough-fivebreaths` | 3,576 | **2** | — |
| `free-speech` | 3,074 | 1 | 4 |
| `productive-vocabulary` | 2,910 | 1 | 204 |
| `maximum-phonation-time` | 2,696 | 1 | — |
| `cape-v-sentences` | 2,370 | 1 | 12 |
| `free-speech-v2` | 2,120 | 1 | 6 |
| `respiration-and-cough-breath` | 1,788 | 1 | — |
| `respiration-and-cough-cough` | 1,788 | 1 | — |
| `respiration-and-cough-threequickbreaths` | 1,718 | 1 | — |
| `prolonged-vowel` | 1,604 | 1 | — |
| `glides-low-to-high` | 1,596 | 1 | — |
| `glides-high-to-low` | 1,554 | 1 | — |
| `cape-v-sentences-v2` | 1,224 | 1 | 6 |
| `loudness` | 897 | 1 | — |
| `rainbow-passage` | 897 | 1 | 1 |
| `diadochokinesis-buttercup` | 896 | 1 | — |
| `diadochokinesis-ka` | 896 | 1 | — |
| `diadochokinesis-pa` | 896 | 1 | — |
| `diadochokinesis-pataka` | 896 | 1 | — |
| `diadochokinesis-ta` | 896 | 1 | — |
| `picture-description` | 889 | 1 | — |
| `story-recall` | 889 | 1 | 2 |
| `maximum-phonation-time-v2` | 813 | **2** | — |
| `loudness-v2` | 705 | 1 | — |
| `diadochokinesis-v2-tuh` | 702 | 1 | — |
| `diadochokinesis-v2-buttercup` | 702 | 1 | — |
| `diadochokinesis-v2-kuh` | 702 | 1 | — |
| `diadochokinesis-v2-puh` | 702 | 1 | — |
| `diadochokinesis-v2-puhtuhkuh` | 701 | 1 | — |
| `respiration-and-cough-v2-breath` | 699 | 1 | — |
| `respiration-and-cough-v2-threebreaths` | 699 | 1 | — |
| `respiration-and-cough-v2-threebreathsnose` | 699 | 1 | — |
| `respiration-and-cough-v2-threebreathsmouth` | 699 | 1 | — |
| `respiration-and-cough-v2-hardcough` | 698 | 1 | — |
| `story-recall-v2` | 660 | 1 | 2 |
| `caterpillar-passage` | 597 | 1 | 2 |
| `word-color-stroop` | 472 | 1 | 472 |
| `picture-description-option1` | 373 | 1 | — |
| `picture-description-option2` | 329 | 1 | — |
| `voluntary-cough` | 327 | 1 | — |
| `breath-sounds` | 326 | 1 | — |
| `random-item-generation` | 265 | **10** | — |
| `cinderella-story` | 258 | 1 | — |
| `random-item-generation-v2` | 207 | **10** | — |
| `open-response-questions` | 199 | 1 | 1 |
| `animal-fluency` | 195 | 1 | — |
| `high-to-low` | 43 | 1 | — |

Read English-only. Forty-one families also carry exactly one Spanish instruction, six have no
Spanish recordings, and `random-item-generation-v2` carries five (1,417 `es-419` recordings across
the tree); for the airway families that string is the generic acoustictask text. The
`stimulus_text` column counts distinct strings including Spanish: `cape-v-sentences`'s 12 is 6
English sentences and 6 Spanish, and `free-speech-v2`'s 6 is 3 and 3.

## Which verb carries which answer

The five store-wide verbs
([`../20260913-branch-contract-and-hints/design.md:564-570`](../20260913-branch-contract-and-hints/design.md)),
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
[`family-taxonomy-ruleset.md:383-395`](family-taxonomy-ruleset.md)). The verb table carries the
widening: `design.md:568` declares `refine`'s payload as `corrected_extent: [start, end]` **and/or**
`corrected_attributes: {key: value}`, at least one. So a span-metadata correction is a `refine`, not
a second `label`, and the rows below use it that way.

**A component span is a `propose` only when nothing proposed the ground.** Where PREPROCESS's
amplitude spans already cover it — which on a held vowel they do, reading 15.89 s on the MPT
recording of the 2026-09-15 run — the component is a `label` plus, where the boundary is wrong, a
`refine`. [`branch-conventions.md:20-23`](branch-conventions.md) scopes both by family, and that
scoping is unresolved for a label written onto an `amplitude` span.

**Material matching nothing already has a carrier.** PREPROCESS writes `measure: "gap"` spans over
every stretch no other span source covered (`preprocess.py:1912`), so `off_task_extent` does not
need a new detector to find *where* the unmatched material is — only a rule for what makes it
off-task rather than ordinary silence.

### Three inconsistencies this document had to navigate

1. **`trim` has no emitter and no method.** `task_extent` appears exactly twice in the whole `specs/`
   tree — `design.md:569` and `:602` — and **zero times in `src/senselab/`**. No branch document
   lists `trim` in its emit block; `dag.md:1823-1824` records that AIRWAY writes none. The spec says
   what `trim` *carries* and nothing about how the extent is *determined*. This document is the first
   statement of that method, per family.
2. **`trim` and `deviate` both claim `off_task_extent`.** `design.md:569` has `trim` carrying it;
   `branch-conventions.md:108` stores every deviation as `assertion, verb: "deviate"` and
   `branch-airway.md:350` emits AIRWAY's that way. One of the two has to go.
3. **The contract's three deviations are not the list.** `branch-conventions.md:123-142` is
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
| **no viable approach** | the determination cannot be made from this design at all — not by a new derivative and not by a fit. Added 2026-09-15 with the five rows [`preprocess-derivatives-for-expected-patterns.md`](preprocess-derivatives-for-expected-patterns.md) § 4 ruled out. |
| **settled** | no method is owed, because the row is not a separate task. |

---

## The table

One row per (branch, family) pair. Families whose instruction is identical modulo the target syllable
or word share a row and are all named in it. Counts are the corpus profile's, over the 62,547
triaged recordings — the dataset's 70,520 wav less the 7,970 `audio-check*` recordings, which get no
row. Every instruction quoted below was read from the recording-grain sidecars of
`bids_adult_2026_09_04` on 2026-09-15 and is verbatim; the appendix carries each in full with its
count.

Pattern notation: **L** = lexical, **A** = acoustic, **→** = order the instruction states,
**&** = both expected, order not stated.

**The detection column is a call, not a description.** Each row names the function that makes the
determination and lists the store derivatives it reads, spelled as the store spells them; `†` marks
an input no node writes today and `‡` one that exists but reaches no branch. The body — the decision
logic, the three things it returns, and the operating points it cannot supply — is in
[§ The detection functions](#the-detection-functions), one block per task family carrying that
family's complete derivative requirement.

### VOICE

| branch | task | expected pattern | detection |
| --- | --- | --- | --- |
| VOICE | `prolonged-vowel` (1,604) | **L→A.** L: the ordered tokens `one two three`, once. A: one continuous voiced production of a single vowel /a/, held to the timer, F0 holding rather than moving. The only voice family in the corpus that is not purely non-lexical, despite `speech_type: non-lexical` | `detect_prolonged_vowel(words, spans, phonation_tracks, continuity_trace, energy_envelope, stream_extent, stimulus_alignment†)` — body: § `prolonged-vowel`. **Owed a code change** for the subject (amplitude spans carry no `family`, and `voice.py:230` selects `family == "phonation"`, so the candidate list is empty before any test runs) and **owed a cut** for the stationarity qualifier — which is a statistic over `continuity_trace` and `f0_hz`, both already in the store, not a missing estimator ([`branch-voice.md:197-210`](branch-voice.md)). The lexical separator is free: the vowel is the voiced production under no lexical word |
| VOICE | `maximum-phonation-time` (2,696) | **A only.** One held /a/ on one breath, to exhaustion. v1's instruction places the deep inhale **before the record tap is mentioned**, so an audible inhale may be inside the file — a second acoustic pattern that is expected but is not the measurement | `detect_sustained_phonation(spans, phonation_tracks, continuity_trace, energy_envelope, stream_extent, words, span_hear, span_yamnet, hints)` — body: § `maximum-phonation-time`. Same owed code change and same owed cut. Duration of the qualified extent is V2's maximum phonation time. The v1 inhale is an AIRWAY `label`, not an `off_task_extent`. **Owed a measurement** (D2) for any voice-quality number over the extent |
| VOICE | `maximum-phonation-time-v2` (813) | **A only**, and cleaner: the instruction puts the inhale explicitly *before* the record tap, so no inhale is expected in the file. A behavioural difference from v1, not a wording difference. **And the index carries a second condition**: `-2` (108 English recordings) appends *"Now try to hold out "ah" for even longer."*, so `-1` → `-2` is a within-subject maximum-duration contrast | `detect_sustained_phonation(..., hints.metadata.task_token‡)` — body: § `maximum-phonation-time`. The two contrasts are free and neither reaches a branch: **owed a code change**, since `task_family` collapses the trailing index (`families.py:134-144`) that carries v2's `-2` effort escalation, the same change `fivebreaths` needs |
| VOICE | `glides-low-to-high` (1,596), `glides-high-to-low` (1,554) | **A only.** One continuous voiced production of /i/ whose F0 sweeps monotonically across the range, in the declared direction. **Not steady** — the opposite of the sustained pattern | `detect_glide(spans, phonation_tracks, continuity_trace, stream_extent, hints)` — body: § `glides-*`. **Implementable today** for the track; **owed a cut** for the dominant monotone segment and its tolerance (V3). `sweep_direction_mismatch` is the declared deviation (`branch-conventions.md:137`). The sustained qualifier does **not** transfer: `continuity_trace` stays high through a glide |
| VOICE | `high-to-low` (43) | **identical to `glides-high-to-low`** — the instruction string is byte-for-byte the same on all 43 recordings, `stimulus_text` empty on all 43 | **Settled** — an alias, not a task: no function of its own, routed into `detect_glide` with `declared = "down"`. Whether `families.py` folds the two is a declaration question, not a branch one |
| VOICE | `loudness` (897) | **L&A, counted.** The token `hey`, **three times**, each at maximal effort | `detect_loudness_token(words, stimulus_alignment†)` and `measure_loudness_effort(words, spans, energy_envelope, level, spectrogram_wideband, phonation_tracks.rms_dbfs†, phonation_tracks.cpps_db†)` — body: § `loudness`. The token, its three extents and the `expected_event_count: 3` count are **implementable today**. Absolute effort has **no viable approach**: `level` is uncalibrated and no SPL reference exists in the graph, so the output is a measurement with its covariates, never a `maximal` / `not maximal` verdict |
| VOICE | `loudness-v2` (705) | **L→A, contrastive.** `hey` at normal effort **→** `hey` shouted. The measurement is the *within-recording* contrast and needs no norm | `detect_loudness_contrast(words, spans, energy_envelope, level, spectrogram_wideband)` — body: § `loudness`. **Implementable today**, and the cheapest effort measure in the corpus: a within-recording difference needs no norm and no calibration. V6 is explicit that v1 and v2 are **not one measurement** ([`branch-voice.md:736`](branch-voice.md)) |
| VOICE | `cape-v-sentences` (2,370), `-v2` (1,224) | **A riding on SPEECH's L.** Six sentences each loading a different phonatory condition; the voice-quality measurement is per sentence and pooling discards the instrument's design | `measure_cape_v_per_sentence(stimulus_alignment†, phonation_tracks, phonation_tracks.{hnr_db,cpps_db,rms_dbfs}†, continuity_trace, words)` — body: § `cape-v-sentences`. **Owed a measurement** twice over: D1 for the six sentence boundaries, D2 for any per-sentence voice-quality number at all — today's are whole-file scalars standing for six deliberately different phonatory conditions. **Owed a code change** so the boundaries reach VOICE as selectable spans. Nothing output may be presented as a CAPE-V score |
| VOICE | any AIRWAY- or SPEECH-declared family | **no expected pattern.** VOICE routed 22,277 recordings against 8,306 declaring a voice family | `measure_voice_without_declared_task(spans, phonation_tracks, continuity_trace, energy_envelope, words)` → `done = UNDETERMINED`; the branch concludes on its own question ([`design.md:486-507`](../20260913-branch-contract-and-hints/design.md)). **Owed a cut**: connected speech passes voiced-fraction, F0-availability and interruption tests, so the stationarity qualifier is what stops V4 computing perturbation over consonants and pauses — the same cut, here load-bearing over 22,277 recordings |

### SPEECH

| branch | task | expected pattern | detection |
| --- | --- | --- | --- |
| SPEECH | `harvard-sentences-list` (13,705), `cape-v-sentences` (2,370), `-v2` (1,224) | **L only**, ordered and fully specified: the words of that recording's `stimulus_text`, in order, once. One sentence per recording — the pattern is per recording, never per family. Measured: **1,060** distinct Harvard sentences, and CAPE-V v1 and v2 share only three of their six sentences. All three families carry the identical instruction, *"Please read the following sentences out loud in your typical voice."* | `detect_read_text(words, spans, stream_extent, stimulus_alignment†)` — body: § `harvard-sentences-list`. **Owed a measurement** (D1): without it the function returns `UNDETERMINED`, because the transcript alone cannot say what was expected. Substitutions and insertions are then enumerated differences carrying the consensus `agreement`; **owed a cut** for omissions, which a skip-arc-free aligner can only surface as a low acoustic score ([`branch-speech.md:115-121`](branch-speech.md)) |
| SPEECH | `rainbow-passage` (897), `caterpillar-passage` (597) | **L only**, ordered, one long passage. Both grains agree on both fields, on all 897 and all 597 — each is its own acoustic task, so the one-to-many collapse that breaks the other families does not arise. `stimulus_text` is one 338-character passage shared by all 898 rainbow recordings, and one of 1,035 characters shared by the 582 English caterpillar recordings (the 15 Spanish ones carry a 1,079-character Spanish passage) | `detect_read_passage(words, spans, stream_extent, hear_scores, stimulus_alignment†)` — body: § `rainbow-passage`. As above at passage length, and here D1's input is legitimately family-scoped. `[breath]` tokens inside the passage are **not** `filler` — the function subtracts them from the deviation list, because they are how S4 measures breath-group structure |
| SPEECH | `word-color-stroop` (472) | **L only, ordered, and not the displayed words.** The instruction says *name the colour, do not read the word*. `stimulus_text` is the 15-item colour sequence, i.e. the expected **answer** sequence — **472 distinct sequences over 472 recordings**, one per recording and never shared, so the pattern is maximally per-recording. Declared 75 s; measured median 75.8 s, with 463 of 472 falling between 74 and 77 s | `detect_stroop_sequence(words, spans, stream_extent, stimulus_alignment†)` — body: § `word-color-stroop`. **Owed a measurement** (D1), built per recording from that recording's own 15-colour answer sequence — 472 distinct sequences over 472 recordings. `filler` is never emitted here: hesitation and self-correction are the task's dependent variable ([`branch-speech.md:130-134`](branch-speech.md)) |
| SPEECH | `free-speech` (3,074) | **L, unordered, with a negative pattern.** No target text. The v1 instruction says *"Do not record yourself reading the prompt"*, so `stimulus_text` is an **anti-pattern**: the question appearing verbatim in the transcript is the deviation. Four questions, per index — three asked 898 times each, one 380 | `detect_free_speech_v1(words, spans, stream_extent, stimulus_alignment†)` — body: § `free-speech`. Presence and extent are **implementable today** from `spans` `measure: "asr"`; the anti-pattern is **owed a measurement** (D1, read inverted) and **owed a cut** (`p_echo_overlap_max`). **This row cannot use the family grain at all**: one frozen prompt covers all 912 sidecars, right on 380 recordings and wrong on 2,694 |
| SPEECH | `free-speech-v2` (2,120) | **L, unordered, and the negative pattern is gone.** v2's instruction drops *"do not record yourself reading the prompt"* entirely and asks instead to answer *"as though you were having a conversation"*. So verbatim echo is **not** declared a deviation here, and treating v1 and v2 alike would invent one. Six questions, three English (683 each) and three Spanish (24 each) | `detect_free_speech_v2(words, spans, stream_extent, declared_duration_s)` — body: § `free-speech-v2`. **Implementable today**, and it emits **no** verbatim-echo deviation: v2 drops the instruction that made echo one, so firing v1's rule here invents it. The acoustictask prompt matches none of the six v2 questions on all 707 sidecars |
| SPEECH | `story-recall` (889), `story-recall-v2` (660) | **L, unordered, partly negative.** Recall *in the participant's own words*. `stimulus_text` is the source story — the 717-character grandfather passage for v1, the 1,083-character frog story for v2 — so semantic coverage is expected and **verbatim** reproduction is the deviation, the participant having read rather than recalled. The five Spanish v1 recordings carry the **frog** story, i.e. v2's source under v1's family name, so even here the source must be read per recording | `detect_story_recall(words, spans, stream_extent, stimulus_alignment†)` — body: § `story-recall`. **Owed a measurement** (D1) and **owed a cut** (`p_verbatim_overlap_max`); the n-gram overlap itself is arithmetic once D1 exists. The source is read per recording — the five Spanish v1 recordings carry v2's story under v1's family name |
| SPEECH | `cinderella-story` (258) | **L, unordered, and nothing machine-readable.** `stimulus_text` is empty on all 258; the source is a physical storybook handed to the participant, so no overlap measure is even definable here | `detect_narrative_presence(words, spans, stream_extent, declared_duration_s)` — body: § `cinderella-story`. Presence and extent **implementable today**, and that is the ceiling rather than a first step: `stimulus_text` is empty on all 258, so **no overlap measure is definable**, D1 included, and the `story-recall` method does not transfer |
| SPEECH | `productive-vocabulary` (2,910) | **L, weakly specified, per recording.** `stimulus_text` is one cue word per recording — **204 distinct cues** across the corpus, and **78 recordings carry no cue at all**. Expected is definitional speech *about* the cue, not the cue itself | `detect_definitional_speech(words, spans, stream_extent, stimulus_alignment†)` — body: § `productive-vocabulary`. Presence and extent **implementable today**; whether the speech defines its cue has **no viable approach** in this graph — a lexicon or a text model, branch-local, no waveform. 204 cues under one family name and 78 recordings with none |
| SPEECH | `picture-description` (889), `-option1` (373), `-option2` (329) | **L only, unspecified.** Connected speech, no target text, `stimulus_text` empty on all 1,591 (the stimulus is an image URL in the sidecar). `picture-description` and `-option1` carry a **byte-identical** instruction — *"Tell me everything you see going on in this picture."* — and differ only by image; `-option2` asks for complete sentences *"as though describing it for the blind"* | `detect_connected_speech(words, spans, hear_scores, stream_extent)` — body: § `picture-description`. **Implementable today** for presence, extent, pause structure and breath groups; S4's measures are unbuilt branch code, not a missing derivative ([`branch-speech.md:157`](branch-speech.md)). The v1/option1 instructions are byte-identical, so **no method here may distinguish them** — any difference found is the image's |
| SPEECH | `open-response-questions` (199) | **L only, unspecified.** *"Please answer the following questions and record your answer."* One `stimulus_text`, a 447-character prompt shared by all 199. Median 30 s | `detect_open_response(words, spans, hear_scores, stream_extent, declared_duration_s)` → `detect_connected_speech` plus the duration count. **Implementable today.** The single shared 447-character prompt makes this the one connected-speech family whose `expected_speech` is legitimately family-scoped |
| SPEECH | `animal-fluency` (195) | **L, unordered, category-bound, with a repetition anti-pattern.** *"say as many animals as you can, while avoiding repeating the same ones. There is a 1 minute timer"*. `stimulus_text` empty on all 195; the category lives only in `instructions`. Median duration 60 s, matching the declared timer | `detect_item_list(words, spans, stream_extent, declared_duration_s)` — body: § `animal-fluency`. Items, their extents, the repetition anti-pattern and the duration check are **implementable today**; category membership has **no viable approach** here — it needs no waveform, has one consumer, and is branch-local rather than a PREPROCESS derivative |
| SPEECH | `random-item-generation` (265), `-v2` (207) | **L, unordered, category-bound — and the negative constraint is category-conditional.** Ten categories per family, one per recording, living only in `instructions`. Eight of the ten (`Animals`, `City names`, `Country names`, `Drinks`, `First names`, `Fruits`, `Jobs`, `English words starting with 't'`) say *"Do not repeat any item"*. The other two, **`Letters` and `Numbers`, say the opposite** — *"random letters or numbers (**repetition allowed**)"* — 48 of 265 v1 recordings and 77 of 203 English v2 recordings | `detect_random_items(words, spans, stream_extent, hints)` → `detect_item_list` with `p_repetition_allowed` read from the recording's **own** category. **Owed a code change**: the category lives only in `instructions` and no grain above the recording carries it, so a family-scoped repeat rule inverts the instruction on the `Letters` and `Numbers` recordings — 48 of 265 v1 and 77 of 203 English v2. Category membership: **no viable approach**, as above |
| SPEECH | `loudness` (897), `loudness-v2` (705) | **L:** the token `hey` | `detect_loudness_token(words, stimulus_alignment†)` — body: § `loudness`. **Implementable today**; SPEECH contributes the token's extents and VOICE's V6 is the measurement wanted. These two are in `LEXICAL_SPEECH` (`families.py:41-42`) while the protocol calls them `speech_type: "non-lexical"` — a `families.py` discrepancy |
| SPEECH | every `SYLLABLE_REPETITION` family (7,989) | **L: none expected.** `/pa/` is not lexical and ASR mostly declines it. These are **positives** for SPEECH's reference set as of 2026-09-15 (`reference_family_set.SPEECH: speech` = `lexical_speech \| syllable_repetition`, `default.yaml:241`), so near-zero lexical content is the correct observation, not a miss. `diadochokinesis-buttercup` is the exception — see the DDK table | `detect_lexical_absence(words, spans)` — body: § `SYLLABLE_REPETITION`. **Implementable today.** These are **positives** for SPEECH's reference set (`default.yaml:241`), so near-zero lexical content is the correct observation, not a miss; `speech.lexical >= 2` firing here is over-routing on function-word artefacts ([`dag.md:195-196`](dag.md)) |
| SPEECH | `prolonged-vowel` (1,604) | **L:** the ordered tokens `one two three`, once. The count-in is **prescribed**, not incidental: the instruction asks the participant to repeat *"1, 2, 3 aah"*, on all 1,575 English recordings | `detect_count_in(words, stimulus_alignment†)` — body: § `prolonged-vowel`. **Implementable today**: D1 sharpens it, but an ordered-run match over the consensus words needs nothing new. Measured: 938 of 1,258 transcripts open with `One two three` ([`dag.md:185`](dag.md)) |
| SPEECH | any AIRWAY-declared family | **L: none expected.** Any lexical content is off-task by construction | `detect_lexical_intrusion(words, spans)` — body: § the AIRWAY families. **Implementable today** and needs no alignment: nothing lexical is expected, so every lexical word is the finding. AIRWAY owns the deviation ([`branch-airway.md:164-165`](branch-airway.md)) |

### AIRWAY

| branch | task | expected pattern | detection |
| --- | --- | --- | --- |
| AIRWAY | `respiration-and-cough-cough` (1,788) | **A, counted.** Five discrete forced expulsive events — *"After pressing record, cough 5 times"*. `expected_event_count: 5` | `detect_cough_series(spans, energy_envelope, span_hear, span_yamnet, words, stream_extent)` — body: § `respiration-and-cough-cough`. **Owed a code change**: the count is of events, not of label-carrying spans — `by_label` increments once per (span, label) pair (`airway.py:280`), so a 4 s span holding three coughs counts 1. **Owed a cut** for A5/A6's four boundary points. The merging case is a series on one exhalation, visible as multiple maxima inside one span — not `spans.min_separation_ms`, which is 30 ms |
| AIRWAY | `respiration-and-cough-v2-hardcough` (698) | **A, one event, maximal effort.** *"cough HARD as if something were stuck in your throat"*. No count stated, and the instruction adds a **recording-hygiene** clause found nowhere else — *"do not cover your mouth or place your hand between your mouth and the microphone"* | `detect_hard_cough(spans, energy_envelope, span_hear, span_yamnet, level, spectrogram_wideband, disruptions_file, band_profile†, stream_extent)` — body: § `respiration-and-cough-v2-hardcough`. Events as above, minus the count. *"Hard"* has **no viable approach**: no within-recording contrast and no SPL reference, so the output is a measurement with its covariates. The hygiene clause is QUALITY's: `detect_occluded_microphone(level, spectrogram_wideband, band_profile†, disruptions_file)`, a level and spectral-tilt finding |
| AIRWAY | `voluntary-cough` (327) | **A, counted (3), maximal effort, with breathing between.** Same *"cough HARD"* wording as `v2-hardcough` but *"Complete this task **3 times** in a single recording"*, with *"then breathe normally again"* between. `expected_event_count: 3`, and the inter-cough breathing is an **expected** pattern, not off-task | `detect_cough_cycles(spans, energy_envelope, span_hear, span_yamnet, hear_scores, words, stream_extent)` — body: § `voluntary-cough`. The one cough family where a cough detector alone is insufficient: the expected pattern is an **alternation**, so material between coughs is matched as breath rather than scored as `off_task_extent`. Same owed cut as the 5-cough row. Median 13.8 s against `v2-hardcough`'s 4.6 s, consistent with three cycles |
| AIRWAY | `respiration-and-cough-fivebreaths` (3,576) | **A, counted (5) and routed — and the route is per recording, not per family.** Index `-1`/`-3`: nose, mouth closed — **1,778** recordings. Index `-2`/`-4`: mouth — **1,778**. The family name carries neither, and the split is exact | `detect_breath_cycles(spans, energy_envelope, span_hear, span_yamnet, hear_scores, words, stream_extent, hints, band_profile†)` — body: § `respiration-and-cough-fivebreaths`. The count is **owed a cut** (A5's operating points, [`branch-airway.md:243-247`](branch-airway.md)). The route has **no viable approach**: the discriminating band is largely above the 8 kHz ceiling and the residual tilt is confounded one-for-one with mouth-to-microphone geometry, which changes *with the route by construction*. The function returns `route = NOT_SEPARABLE_BY_THIS_DESIGN` and carries `band_profile†` so a negative is attributable. The index that carries the route is **owed a code change** — `task_family` collapses it |
| AIRWAY | `respiration-and-cough-v2-threebreathsnose` (699), `-threebreathsmouth` (699) | **A, counted (3) and routed**, the route differing *between the two families* and stated in each instruction | `detect_breath_cycles(..., p_expected_count = 3)` with the route read from the family name — body: § `respiration-and-cough-v2-threebreaths{nose,mouth}`. Same count, same route finding. These two and the `fivebreaths` index split are the only declared route contrasts in the corpus, which is why neither should be treated as validation-grade for A7 |
| AIRWAY | `respiration-and-cough-threequickbreaths` (1,718), `-v2-threebreaths` (699) | **A, counted (3) and timed.** Exhale, then inhale *quickly*. The **interval** is the measurement — a count of three says nothing about whether they were quick | `detect_quick_breaths(..., p_expected_count = 3)` — body: § `respiration-and-cough-threequickbreaths`. The **interval** is the measurement — a count of three says nothing about whether they were quick — and it is arithmetic over the event onsets, so it is **owed the same cut** and nothing more: `p_interval_max_s` cannot be fitted before A5's boundary points are |
| AIRWAY | `respiration-and-cough-breath` (1,788), `-v2-breath` (699) | **A, uncounted and durational.** Comfortable breathing for a stated duration — 30 s v1, 20 s v2, v2 specifying through the mouth. `expected_event_count` is absent for these, and both durations are honoured: 1,440 of 1,788 v1 recordings run 30-31 s and 638 of 699 v2 run 20-21 s | `detect_comfortable_breathing(hear_scores, span_hear, spans, energy_envelope, words, stream_extent, declared_duration_s)` — body: § `respiration-and-cough-breath`. **Implementable today** on HeAR's `Breathe`, which is a positive detection of the thing being asked about. `residual` `energy_fraction` is **not** read: `residual = plain − g·FRCRN(plain)` and FRCRN is a speech enhancer, so a high fraction means *this is not speech* — a cough, a glide, room noise and a near-silent file all satisfy it. `declared_duration_s` against measured is free and is the sharpest signal here: 132 of 1,788 v1 recordings run under a second |
| AIRWAY | `breath-sounds` (326) | **A, counted (3), routed (mouth), and preceded by a declared 60 s of nothing.** *"Please relax for 60 seconds until the task starts. Take three deep breaths in a row in and out of the mouth."* So it is **not** the uncounted durational task its name suggests, and it is closest to `v2-threebreathsmouth` | `detect_breath_sounds(..., p_expected_count = 3, p_relax_s)` — body: § `breath-sounds`. Count and route as the three-breath families. The declared 60 s relax is the only place an instruction **prescribes** material that is not the task, and the function emits it as `off_task_extent` only on a recording long enough to contain it — the measured median is 13.2 s against a declared ~73 s |
| AIRWAY | any SPEECH- or VOICE-declared family | **no expected pattern.** AIRWAY routed broadly and its gate evidence is `unavailable` on 56,505 of 62,547 recordings | `measure_airway_without_declared_task(spans, span_hear, span_yamnet, hear_scores, energy_envelope)` → `done = UNDETERMINED`; it labels airway evidence where it finds it and contests what was proposed and is not there. **Implementable today.** A breath during passage reading is **not** a deviation — it is how S4 measures breath-group structure |

### DDK

DDK is a declared branch with **no node**: `BRANCHES = ("AIRWAY", "SPEECH", "VOICE", "DDK")`
(`vocabulary.py:31`), and `run.py:302-306` marks it `SKIPPED` with
`NO_NODE = "no node implements this branch"`. The graph *can* route to it — two gates do — and then
records that nothing implements it. It is evaluated here on the same terms as the others, per the
owner's instruction that a declared branch is assessed independently of whether it is implemented.

| branch | task | expected pattern | detection |
| --- | --- | --- | --- |
| DDK | `diadochokinesis-pa` (896), `-ta` (896), `-ka` (896) | **A, counted, alternating.** One syllable repeated *as fast as possible*, **10 times** — `expected_event_count: 10`. v1 states the count; v2 does not | `detect_ddk_train(energy_envelope, spans, continuity_trace, stream_extent)` — body: § `diadochokinesis-pa`. **Owed a cut**, not a measurement: D1's rate is the modulation spectrum of `energy_envelope` over the train and D3's intervals are its event onsets, both over an array the store already holds. Praat's `extract_speech_rate` is **not** the instrument — it is already running inside `praat_features` and its `min_dip` and 0.3 s `min_pause` both under-count the fastest trains, biasing the measurement in the direction of the quantity being measured |
| DDK | `diadochokinesis-v2-puh` (702), `-tuh` (702), `-kuh` (702) | **A, uncounted, alternating.** Same repetition *until the timer runs out*, so no `expected_event_count`. The `'puhpuhpuhpuhpuhpuh'` in the instruction is an orthographic illustration, **not** a six-repetition instruction. The timer is **5 s**: 630-646 of each family's 702 recordings run 5-6 s, against a v1 median of 5 s spread over 3-9 s | `detect_ddk_train_timed(..., p_expected_count = None, declared_duration_s)` — body: § `diadochokinesis-v2-puh`. As above with no declared count. D5's train fraction has a fixed 5 s denominator here, which makes the rate directly comparable across participants in a way v1's participant-terminated recordings are not — and none of D1-D6 is built to use it |
| DDK | `diadochokinesis-pataka` (896), `-v2-puhtuhkuh` (701) | **A, ordered and cyclic.** A three-place sequence repeated in order — sequential motion rate. `/pa-pa-pa/` is a collapse of the sequence and is the clinically meaningful finding. v1 asks for the sequence *"10 times"* (`expected_event_count: 10`, i.e. 30 syllables); v2 asks for it *"until the timer runs out"*, which is 5 s, so the two are counted and uncounted respectively | `detect_ddk_sequence(energy_envelope, spans, spectrogram_wideband, stream_extent)` — body: § `diadochokinesis-pataka`. **Owed a cut** for the place decision (`p_burst_window_ms`, `p_place_centroid_bands_hz`, `p_place_margin`) and nothing else: /p/, /t/ and /k/ differ in burst spectrum inside 8 kHz, and `spectrogram_wideband`'s 5 ms window at a 5 ms hop is the classical resolution for it. The **PPG is not the instrument** — it is trained on connected speech and its prior works against the discrimination on a rapid nonsense train ([`branch-ddk.md:318-323`](branch-ddk.md)). `syllable_sequence_mismatch`, deliberately not `stimulus_mismatch` |
| DDK | `diadochokinesis-buttercup` (896), `-v2-buttercup` (702) | **L&A.** A real English word repeated — so unlike every other DDK family this one **does** have a lexical pattern, and the recognisers will produce it. v1 states the count (*"10 times"*, `expected_event_count: 10`); v2 does not (*"until the timer runs out"*), so the two are not one measurement | `detect_repeated_word(words, energy_envelope, spans, stream_extent)` — body: § `diadochokinesis-buttercup`. **Implementable today**: the repeat count is a counter over normalised consensus tokens, so `transcript_repeat` moving into the store is a convenience rather than the capability. The only DDK family where the lexical route is the right one, and the only one where `ddk.lexical_repetition >= 3` fires for the right reason |
| DDK | any lexical-speech family | **no expected pattern.** DDK routed 22,363 against 7,989 declaring a DDK family; repetition occurs in ordinary speech — a stutter, a false start, a repeated word | `measure_ddk_without_declared_task(energy_envelope, spans, words)` → `done = UNDETERMINED`. The branch measures what it finds and says what it is; it does not assert that a Harvard sentence failed to be a DDK task ([`branch-ddk.md:66-70`](branch-ddk.md)) |

QUALITY is not in `BRANCHES` — `vocabulary.py:28-29` calls it "a graph edge, never a branch" — so it
has no rows here. Its Q5 acquisition-consistency capability is the natural home for
`declared_duration_s` against measured duration, which several rows above want.

---

## The detection functions

One function per row of the table above, named in that row. This section is the body of the
detection column: the table says what a row takes and what state it is in, and the block here says
how the determination is made.

**Every function returns the same triple**, which is what the DAG's verbs read:

```
(done, components, deviations)

done        bool | UNDETERMINED — were the expected patterns found. Read off the recording,
            never off the declaration: a declared family that produced none of its patterns
            returns False, and a function whose only instrument is missing returns UNDETERMINED
            rather than guessing.
components  [(label, start, end)] — where each pattern was found. A component over an extent
            some node already proposed is a `label` or a `refine`; the component the task is
            measured over is the `trim` payload, `task_extent`. An empty list is a result.
deviations  [(deviation_type, start, end, evidence)] — the nine types of
            `branch-conventions.md:123-142`, plus `counts` entries ({found, declared}) where
            the instruction declares a number. A count asserts no discrepancy.
```

**Reading the signatures.**

| mark | meaning |
| --- | --- |
| no mark | the input is a derivative PREPROCESS writes today, spelled as the store spells it |
| `†` | the derivative does not exist. Named anyway, so the dependency is visible: `stimulus_alignment` (D1), `phonation_tracks.{hnr_db,rms_dbfs,cpps_db}` (D2), `band_profile` (D3), all three from [`preprocess-derivatives-for-expected-patterns.md`](preprocess-derivatives-for-expected-patterns.md) |
| `‡` | the value exists somewhere in the tree but no branch can read it: `hints.expected_speech` (declared, never populated), the trailing task index (collapsed by `task_family`), `transcript_repeat` (a `routing_analysis` feature, not a store measurement), per-span PPG |
| `p_*` | an operating point. **No number appears in any body.** Every `p_*` is unfitted and belongs in `data/` with a written derivation, per the project rule |

**The store types the signatures are written over.** Verified against
`src/senselab/audio/workflows/triage/nodes/preprocess.py` at this branch's tip; the block list is
`preprocess.py:2945-2985`.

| name in a signature | what it is | fields a body may read |
| --- | --- | --- |
| `words` | the `word` entities `consensus_transcript` names (`preprocess.py:2409`; attributes at `consensus.py:391`) | `text`, `bracketed`, `outcome`, `sources`, `readings`, `timings` (per source), `onset_spread_s`, `offset_spread_s`, `temporal_uncertainty_s`, `variants`, `agreement`, `index`, and the entity's own `extent` |
| `consensus_transcript` | the whole-file stream (`preprocess.py:2456`, name at `:2460`) | `text`, `sources` (model id + commit per source), `word_ids` |
| `spans` | the `span` entities — the amplitude/continuity/ASR loop at `preprocess.py:1875-1891`, the `gap` loop at `:1893-1919` | `extent`, `measure ∈ {amplitude, continuity, asr, gap}`, `signal`, `peak_over_floor_db`, `k_db`, `merged_proposals`, `contains_clip`, `corroborated_by` |
| `energy_envelope` | `derivatives/energy_envelope.npz` (`preprocess.py:1612`) | `envelope_dbfs` per sample, `floor_dbfs` (one **global** value, broadcast), `sampling_rate` |
| `normalized_envelope` | the AGC'd envelope (`:1705`) | the same two arrays |
| `continuity_trace` | `derivatives/continuity_trace.npz` (`:2549`) | `continuity` per sample in `[0, 1]`, `cut_level`, `cut_percentile`. **This is a spectral-stationarity trace already** — cosine similarity between consecutive log-magnitude spectra (`spectral_continuity/api.py:10`) |
| `phonation_tracks` | `derivatives/phonation_tracks.npz` (`:1300`), hop 10 ms | `times_s`, `f0_hz`, `strength`, `formant_times_s`, `f1..f4_hz`, `f1..f4_bw_hz`. F0 on `preemphasised`, formants on `plain` |
| `spectrogram_wideband` / `spectrogram_narrowband` | `derivatives/spectrogram_*.npz` (`:2507`) | `spectrogram` (power), `win_length`, `hop_length`, `n_fft`. 5 ms / 20 ms window, 5 ms hop |
| `gammatone` | `derivatives/gammatone.npz` (`:2584`) | `centre_frequencies_hz` (40 channels, 80–7800 Hz), `energy_db`, hop 5 ms |
| `ppg_posteriorgram` | `derivatives/ppg_posteriorgram.npz` (`:803`), on `enhanced` | `posteriorgram[frame, phoneme]`, `phonemes` (40 ARPAbet incl. `<silent>`), `seconds_per_frame`. **Whole file only** |
| `praat_features` | ~40 whole-file scalars on `enhanced` (`:1179`) | the scalar set. Not re-poolable over an extent |
| `level` | whole-file, on `plain` (`:2087`) | `peak_dbfs`, `rms_dbfs`, `lufs`. **Uncalibrated** — no SPL reference exists anywhere in the graph |
| `silence` | YAMNet `Silence` per window (`:2059`) | `windows[{start, end, score, is_silence}]`, `threshold` |
| `span_hear` / `span_yamnet` | per-span classifier windows (`:2186`, `:2250`) | `span_id`, the window's own `extent`, `raw_scores`, `labels`, `scores`, `labelled`, `isolated_span` |
| `hear_scores` / `yamnet_scores` / `ast_scores` | the classifier's verbatim whole-file windows (`:1926`) | `start`, `end`, `label_scores` (every label, raw), `win_length`, `hop_length`. HeAR's eight labels are `Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear, Laugh, Speech`; HeAR windows are 2.0 s non-overlapping, YAMNet 0.96 s on a 0.48 s hop, AST 10.24 s non-overlapping |
| `hear_windows` / `yamnet_windows` / `ast_windows` | the fold of a membership rule over those scores (`:1952`) | **all three are absent under the shipped config**, measured by calling `load_label_membership` against the packaged default: `windows.{yamnet,hear}.label_thresholds` and `windows.ast.default_threshold` are null, `config.require` raises on a null, and the block records its own absence. YAMNet's and HeAR's floors ship at 0.2 with `label_top_k: 4`, and reach a reader only through `span_hear` / `span_yamnet`, which use `optional_label_membership` and tolerate null overrides. **No body below reads a `*_windows` derivative** |
| `enhanced_hear_scores` / `residual_hear_scores` (and `_yamnet_scores`, `_ast_scores`) | the same per stream — block names `enhanced_hear` / `residual_hear`, measurement names as spelled here (`:2815-2845`, `:2930`) | per-window scores plus `speech_overlap`, and the `_summary_all` / `_summary_speech_free` roll-ups |
| `enhanced_diarization` / `residual_diarization` | one measurement per stream in `diarization.streams` (`:2617`, `:2985`) — **shipped** | `n_speakers`, `n_segments`, `per_speaker_s`, `speech_s`, `overlap_s`, `max_concurrent_speakers`, and `derivatives/<stream>_diarization.npz` carrying `starts`/`ends`/`speakers`/`streams`. The two streams' counts are never summed |
| `residual` | the FRCRN subtraction (`:2667`, measurement at `:2778`) | `energy_fraction`, `enhanced_energy_fraction`, `gain_db`, `bands`, `speech_present`, `speech_coverage_fraction`, `n_consensus_words` |
| `squim` | one assertion per span (`:2133`) | `stoi`, `pesq`, `si_sdr`, over the span's extent |
| `disruptions_file` | on the **un-resampled** `recording` stream (`:2106`) | clipped runs, dropouts, discontinuities, DC, zero-crossing rate, `sampling_rate` |
| `clip_spans` | `span` entities with `family: CLIP_FAMILY` (`:714`, block at `:1519`) | `extent`, `signal`, and one `clip_amplitude` measurement beside them |
| `hints` | `AudioHints` (`audio_hints.py:129`), handed to every branch | `may_contain`, `environment`, `metadata.task_token`‡ (the only carrier of the trailing index), `metadata.speech_type`, `expected_speech`‡ |
| `stream_extent` | ADMIT's `recording` stream extent, `(0.0, duration_s)` (`admit.py:95-98`) | the measured duration every "was it done" test needs |

Three helpers recur and are written once here rather than in thirty bodies:

```
gaps(spans)        = [s for s in spans if s.measure == "gap"]
lexical(words)     = [w for w in words if not w.bracketed]
off_task(components, spans) =
    [("off_task_extent", g.start, g.end, {"measure": "gap"})
     for g in gaps(spans) if g overlaps no component extent]
```

`off_task_extent` is emitted by the branch that owns the task; which verb carries it —
`trim` per `design.md:569` or `deviate` per `branch-conventions.md:108` — is unsettled, and these
bodies emit the finding without choosing.

**`declared_duration_s` is a sidecar-consistency check, not a task-completion measurement.** Where a
body reads it, the finding is that the declaration and the recording disagree; which of the two is
wrong is a separate question, and 2,020 sidecars declare under a second.

### `prolonged-vowel` (1,604) — VOICE and SPEECH

**Requires, across both branches:** `consensus_transcript` + its `word` entities · `spans`
(`amplitude`, `gap`) · `phonation_tracks` (`times_s`, `f0_hz`, `strength`) · `continuity_trace` ·
`energy_envelope` · `stream_extent`.
**Absent:** `stimulus_alignment`† (D1) for the count-in as a declared expectation rather than a
guessed token list; `phonation_tracks.{hnr_db,cpps_db,rms_dbfs}`† (D2) for any voice-quality number
over the vowel — without it the only figures that exist are `praat_features`, which are whole-file
and include the count-in and the silence.
**Unreachable:** `hints.expected_speech`‡.

```
detect_prolonged_vowel(
    words, spans, phonation_tracks, continuity_trace, energy_envelope, stream_extent,
    stimulus_alignment†,
    *, p_count_in_tokens, p_voiced_strength_min, p_voiced_fraction_min,
       p_f0_spread_window_s, p_f0_spread_max_semitones, p_continuity_min, p_vowel_min_s,
) -> (done, components, deviations)

    # P1 — the lexical count-in, prescribed by the instruction ("1, 2, 3 aah")
    count_in = stimulus_alignment.run_for(p_count_in_tokens)          # D1, when it exists
             | longest ordered run in lexical(words) whose normalised texts are
               p_count_in_tokens in order                             # fallback; needs no D1

    # P2 — the held vowel: voiced, under no lexical word, spectrally stationary
    candidates = [s for s in spans
                  if s.measure == "amplitude"
                  and no w in lexical(words) overlaps s.extent]       # the free lexical separator
    for s in candidates:
        f = phonation_tracks sliced to s.extent                       # 10 ms hop
        voiced       = fraction(f.strength >= p_voiced_strength_min)
        f0_spread    = max over sliding p_f0_spread_window_s of
                       robust_spread(semitones(f.f0_hz)) on voiced frames
        stationarity = median(continuity_trace over s.extent)         # already a stationarity trace
        qualifies(s) = voiced >= p_voiced_fraction_min
                   and f0_spread <= p_f0_spread_max_semitones
                   and stationarity >= p_continuity_min
                   and duration(s) >= p_vowel_min_s
    vowel = longest qualifying candidate, else None

    done       = (count_in is not None) and (vowel is not None)
    components = [("count_in", *count_in.extent)] if count_in
               + [("vowel", *vowel.extent)] if vowel                  # `vowel` is the task_extent
    deviations = [("truncation", ...)] if vowel touches stream_extent's edge
               + [("repeat_attempt", ...)] if more than one candidate qualifies
               + off_task(components, spans)
```

```
detect_count_in(words, stimulus_alignment†, *, p_count_in_tokens)
        -> (done, components, deviations)

    match = stimulus_alignment.run_for(p_count_in_tokens)
          | ordered-run match over lexical(words)
    done       = match is not None
    components = [("count_in", *match.extent)] if match
    deviations = [("stimulus_mismatch", *w.extent, {"expected": t, "read": w.text,
                                                    "agreement": w.agreement})
                  for (t, w) in match.substitutions]
```

**Notes.** The lexical half is the cheapest row in the document — 938 of 1,258 transcripts open with
`One two three` ([`dag.md:185`](dag.md)) and the control family fires `speech.intrusion` at 3.2%
([`family-taxonomy-ruleset.md:320-323`](family-taxonomy-ruleset.md)). The acoustic half returns
nothing today for a reason that is not a missing estimator: VOICE selects `family == "phonation"`
spans (`voice.py:230`) and PREPROCESS's amplitude spans carry no `family` at all, so
`candidates` is empty before any test runs. That is the owed code change; the owed *decision* is
`p_f0_spread_*` and `p_continuity_min`, since `continuity_trace` and `f0_hz` are both in the store.
`words.onomatopoeic_tokens` does not help here — its vocabulary is a cough set
(`default.yaml:123`), not numerals.

### `maximum-phonation-time` (2,696), `-v2` (813) — VOICE

**Requires:** `spans` (`amplitude`, `gap`) · `phonation_tracks` · `continuity_trace` ·
`energy_envelope` · `span_hear` · `span_yamnet` (v1's inhale only) · `stream_extent` ·
`hints.metadata.task_token`‡ for the trailing index.
**Absent:** `phonation_tracks.{hnr_db,cpps_db,rms_dbfs}`† (D2).

```
detect_sustained_phonation(
    spans, phonation_tracks, continuity_trace, energy_envelope, stream_extent, words,
    span_hear, span_yamnet, hints,
    *, p_voiced_strength_min, p_voiced_fraction_min, p_f0_spread_window_s,
       p_f0_spread_max_semitones, p_continuity_min, p_vowel_min_s,
       p_inhale_search_s, p_breath_label_set, p_breath_score_min,
) -> (done, components, deviations)

    vowel = the qualifying candidate of detect_prolonged_vowel's P2 test, longest first,
            with no lexical-word exclusion applied — this family expects no lexical material,
            so a lexical word overlapping the production is a finding, not a filter

    version = "v2" if hints.metadata.task_token names a v2 token else "v1"
    if version == "v1":
        # v1 places the deep inhale before the record tap is mentioned, so an audible inhale
        # may be inside the file. It is an AIRWAY `label`, never an off_task_extent.
        inhale = first span in spans within p_inhale_search_s of stream_extent.start
                 with sounds_like(span, span_hear, span_yamnet,        # § the AIRWAY families
                                  p_label_set=p_breath_label_set,
                                  p_score_min=p_breath_score_min)
    done       = vowel is not None
    components = [("phonation", *vowel.extent)]   if vowel     # the task_extent; its duration is V2
               + [("inhale", *inhale.extent)]     if inhale    # v1 only, handed to AIRWAY
    deviations = [("truncation", ...)] if vowel touches stream_extent's edge
               + [("repeat_attempt", ...)] if more than one candidate qualifies
               + [("off_task_extent", *w.extent, {"text": w.text}) for w in lexical(words)]
               + off_task(components, spans)
    counts     = {"phonation_s": duration(vowel), "index": trailing index of task_token}‡
```

**Notes.** Duration of the qualified extent is V2's maximum phonation time, and it is a
norm-bearing scalar: `branch-conventions.md` requires the name to carry its convention. Two
contrasts ride on the trailing index and both are lost before a branch sees them, because
`task_family` collapses it (`families.py:134-144`): the v1/v2 contrast is the corpus's only control
for whether the inhale is captured, and `-1` → `-2` on v2 (700 against 113 recordings) is its only
within-subject phonation-duration contrast. Whether v1's *"we will repeat this task 3 times"* means
three recordings or three attempts inside one is unsettled, and `repeat_attempt` fires opposite ways
under the two readings — so the deviation above is emitted with the reading named in its evidence.

### `glides-low-to-high` (1,596), `glides-high-to-low` (1,554), `high-to-low` (43) — VOICE

**Requires:** `spans` (`amplitude`) · `phonation_tracks` (`f0_hz`, `strength`) ·
`continuity_trace` · `stream_extent` · `hints.metadata.task_token`‡ for the declared direction.
**Absent:** nothing. The measurement is in the store; what is owed is one decision.

```
detect_glide(
    spans, phonation_tracks, continuity_trace, stream_extent, hints,
    *, p_voiced_strength_min, p_voiced_fraction_min, p_monotone_tolerance_semitones,
       p_dominant_segment_min_fraction, p_glide_min_s,
) -> (done, components, deviations)

    declared = "up" if task_token names low-to-high else "down"       # from the family, a hint
    for s in [s for s in spans if s.measure == "amplitude"]:
        f       = phonation_tracks sliced to s.extent
        voiced  = fraction(f.strength >= p_voiced_strength_min)
        # V3: the dominant monotone segment. Longest run of semitones(f.f0_hz) that never
        # reverses by more than p_monotone_tolerance_semitones; its sign is the direction.
        run     = longest tolerant-monotone run over voiced frames
        covers  = duration(run) / duration(s)
        qualifies(s) = voiced >= p_voiced_fraction_min
                   and covers >= p_dominant_segment_min_fraction
                   and duration(s) >= p_glide_min_s
    sweep = longest qualifying span, else None

    done       = sweep is not None
    components = [("glide", *sweep.extent)] if sweep                   # the task_extent
    deviations = [("sweep_direction_mismatch", *sweep.extent,
                   {"declared": declared, "measured": sign(run), "extent_semitones": ...})]
                  if sweep and sign(run) != declared
               + [("truncation", ...)] if sweep touches stream_extent's edge
```

**Notes.** A glide is the opposite of the sustained pattern, so `p_continuity_min` does **not**
transfer: `continuity_trace` stays high through a slowly-moving harmonic structure, which is why it
qualifies a glide and a held vowel alike and cannot separate them — the separation is the monotone
run, not the trace. `high-to-low` is an alias: its instruction is byte-for-byte
`glides-high-to-low`'s on all 43 recordings and its `stimulus_text` is empty on all 43, so it gets
no function of its own and is routed into this one. Whether `families.py` should fold the two is a
declaration question, not a branch one.

### `loudness` (897), `loudness-v2` (705) — VOICE and SPEECH

**Requires, across both branches:** `consensus_transcript` + `word` entities · `spans`
(`amplitude`) · `energy_envelope` · `level` · `spectrogram_wideband` · `stream_extent`.
**Absent:** `phonation_tracks.{rms_dbfs,cpps_db,hnr_db}`† (D2) for the level-invariant half of
effort; `stimulus_alignment`† (D1) for `hey` as a declared token rather than a literal.
**No viable approach** for absolute effort on v1 — see the note.

```
detect_loudness_token(words, stimulus_alignment†, *, p_token, p_expected_count)
        -> (done, components, deviations)

    hits       = ordered matches of p_token over lexical(words)        # SPEECH's half
    done       = len(hits) > 0
    components = [("token", *h.extent) for h in hits]
    counts     = {"expected_event_count": {"found": len(hits), "declared": p_expected_count}}
```

```
measure_loudness_effort(words, spans, energy_envelope, level, spectrogram_wideband,
                        phonation_tracks.rms_dbfs†, phonation_tracks.cpps_db†,
                        *, p_token, p_expected_count)
        -> (done, components, deviations)

    hits = detect_loudness_token(...).components
    for h in hits:
        measure per token, never as a verdict:
            peak_over_floor_db  from energy_envelope over h.extent
            spectral_balance    = high/low band energy ratio of spectrogram_wideband over
                                  h.extent — the level-invariant half of effort
            rms_dbfs, cpps_db   over h.extent                          # D2
        carry the covariates that decide what the number means:
            level.{peak_dbfs, rms_dbfs, lufs} (file), spans.contains_clip, disruptions_file
    done       = len(hits) > 0
    components = [("token", *h.extent) for h in hits]
    deviations = []                                                     # effort asserts nothing
    counts     = {"expected_event_count": {"found": len(hits), "declared": p_expected_count}}
```

```
detect_loudness_contrast(words, spans, energy_envelope, level, spectrogram_wideband,
                         *, p_token, p_min_contrast_db)
        -> (done, components, deviations)

    first, second = the first two matches of p_token over lexical(words), in order
    done          = both exist
    contrast_db   = peak_over_floor_db(second) - peak_over_floor_db(first)
    balance_delta = spectral_balance(second) - spectral_balance(first)
    components    = [("normal", *first.extent), ("loud", *second.extent)]
    deviations    = [("off_task_extent", ...)] for lexical material matching neither
    # A within-recording difference needs no norm and no calibration. Direction is the
    # measurement; p_min_contrast_db decides only whether the contrast is reported as present.
```

**Notes.** `level` is peak dBFS, RMS dBFS and LUFS of an uncalibrated consumer recording with
unknown microphone sensitivity, unknown source-to-microphone distance and, on many handsets, AGC
inside the capture path. **There is no SPL reference in the graph and none can be recovered after
the fact**, so v1's *"maximal effort"* has no viable approach: `measure_loudness_effort` returns a
measurement with its covariates and never a `maximal` / `not maximal` verdict. What survives without
calibration is the level-invariant half — spectral balance, which effort changes largely
independently of gain. v2 is the cheapest effort measure in the corpus because the contrast is
within the recording. V6 is explicit that v1 and v2 are **not one measurement**
([`branch-voice.md:736`](branch-voice.md)). `families.py:41-42` puts both in `LEXICAL_SPEECH` while
the sidecars declare `speech_type: "non-lexical"` on all 897 and all 705; that is a `families.py`
discrepancy, not a branch one.

### `cape-v-sentences` (2,370), `-v2` (1,224) — VOICE and SPEECH

**Requires, across both branches:** `consensus_transcript` + `word` entities · `spans` ·
`phonation_tracks` · `continuity_trace` · `stream_extent`.
**Absent:** `stimulus_alignment`† (D1) — and here it is load-bearing twice, because the per-sentence
boundaries it yields are what VOICE measures over; `phonation_tracks.{hnr_db,cpps_db,rms_dbfs}`†
(D2), without which there is no per-sentence voice-quality number at all.
**Unreachable:** `hints.expected_speech`‡; the six sentences per version, which no table carries.

```
measure_cape_v_per_sentence(
    stimulus_alignment†, phonation_tracks, phonation_tracks.hnr_db†,
    phonation_tracks.cpps_db†, phonation_tracks.rms_dbfs†, continuity_trace, words,
    *, p_voiced_strength_min,
) -> (done, components, deviations)

    sentences = stimulus_alignment.structure_spans()    # D1 yields the six boundaries
    if sentences is empty: return (UNDETERMINED, [], [])
    for s in sentences:
        voiced = frames in s where strength >= p_voiced_strength_min
        report per sentence, on `plain` and over voiced frames only:
            f0, hnr_db, cpps_db, rms_dbfs                # D2 tracks, pooled over s
    done       = every sentence has a realised extent
    components = [("sentence_%d" % i, *s.extent) for i, s in sentences]
    deviations = [("stimulus_mismatch", ...)] from the alignment, carried by SPEECH
```

**Notes.** Six sentences each loading a different phonatory condition; pooling across them discards
the instrument's design, which is exactly what `praat_features` does today — one whole-file number
standing for six deliberately different conditions. v1 and v2 share only three of their six
sentences, so a sentence list attached to the wrong version is wrong on half of it. The boundaries
fall out of SPEECH's alignment for free but must reach **VOICE** as selectable spans, which is the
owed code change. Nothing output here may be presented as a CAPE-V score.

### `harvard-sentences-list` (13,705), `cape-v-sentences` (2,370), `-v2` (1,224) — SPEECH

**Requires:** `consensus_transcript` + `word` entities (`text`, `extent`, `agreement`, `timings`,
`variants`) · `spans` (`asr`, `gap`) · `stream_extent`.
**Absent:** `stimulus_alignment`† (D1).
**Unreachable:** `hints.expected_speech`‡ — 1,060 distinct Harvard sentences live in the
per-recording sidecar and nothing reads them.

```
detect_read_text(
    words, spans, stream_extent, stimulus_alignment†,
    *, p_omission_score_max, p_repeat_overlap_min,
) -> (done, components, deviations)

    a = stimulus_alignment†                        # one column per expected token
    if a is absent: return (UNDETERMINED, [], [])  # the transcript alone cannot say what was expected

    realised = [t for t in a.expected if t.column is not None]
    done     = len(realised) / len(a.expected) >= ... is NOT the test — the test is per token,
               and `done` is "every expected token was realised at least once"
    components = [("read_text", min(start of realised), max(end of realised))]   # task_extent
               + [("sentence_%d" % i, *s.extent) for i, s in a.structure_spans()]
    deviations = [("stimulus_mismatch", *w.extent,
                   {"expected": t.text, "read": w.text, "agreement": w.agreement,
                    "variants": w.variants})
                  for (t, w) in a.substitutions]
               + [("stimulus_mismatch", *w.extent, {"expected": None, "read": w.text})
                  for w in a.insertions]
               + [("omission", t.index, {"expected": t.text})     # a TENTH deviation type,
                  for t in a.expected if t.column is None]        # and extent-free by nature
               + [("repeat_reading", ...)] if a covers the token sequence more than once
                  with overlap >= p_repeat_overlap_min
               + [("filler", *w.extent) for w in words if w.bracketed]   # a read task expects none
               + [("truncation", ...)] if the first or last realised token touches stream_extent
               + off_task(components, spans)
```

**`omission` is not in the deviation vocabulary.** `branch-conventions.md:123-142` is
authoritative and carries nine types; this is a tenth, and the contract's own rule is that a branch
adding a type adds a row there. Naming it `stimulus_mismatch` would be wrong — nothing was read in
the wrong place, something was not read at all — and leaving it unnamed would make the commonest
read-task departure invisible. The same applies to `repeated_item` in § `animal-fluency`.

**Notes.** Substitutions and insertions are enumerated differences, not scores, and every mismatch
carries the consensus `agreement` (∈ {0.5, 1.0} — two ASR sources) so a reader can tell a
disagreement from a departure. **Omissions are the one thing this shape cannot produce**: a
skip-arc-free aligner assigns every stimulus word an interval, so an omission surfaces only as a low
acoustic score ([`branch-speech.md:115-121`](branch-speech.md)) — hence `p_omission_score_max`, and
hence D1 must carry a skip arc or the omission branch above is dead code. `align_transcriptions`
(`tasks/forced_alignment/forced_alignment.py:691`) is the *acoustic* variant and is the heavier
instrument; D1 is a token-list alignment, the same operation `align_sources`
(`consensus.py:276`) already performs for the ASR-against-ASR case with one side carrying no
timings. The pattern is **per recording**: collapsing to the family discards which sentence was
expected, which is the whole lexical pattern.

### `rainbow-passage` (897), `caterpillar-passage` (597) — SPEECH

**Requires:** as above, plus `hear_scores` (raw, `Breathe`) and the `[breath]` bracketed `word`
entities for S4's breath groups.
**Absent:** `stimulus_alignment`† (D1). One shared passage per family, so here D1's input is
family-scoped and available.

```
detect_read_passage(
    words, spans, stream_extent, hear_scores, stimulus_alignment†,
    *, p_omission_score_max, p_repeat_overlap_min, p_breath_group_min_gap_s,
) -> (done, components, deviations)

    base = detect_read_text(words, spans, stream_extent, stimulus_alignment†, ...)
    # S4 rides on the same alignment: pause and breath-group structure over the passage
    gaps_between = inter-word gaps from the per-source `timings` of consecutive lexical words
    breath_marks = [w for w in words if w.bracketed and w.text == "[breath]"]
                 + hear_scores windows whose raw `Breathe` score clears p_breath_score_min
    groups       = runs of lexical words separated by a gap >= p_breath_group_min_gap_s
                   or by a breath mark
    done         = base.done
    components   = base.components + [("breath_group_%d" % i, *g.extent) for i, g in groups]
    deviations   = base.deviations   minus   [("filler", ...) for w in breath_marks]
```

**Notes.** `[breath]` tokens inside a passage are **not** `filler` — they are how S4 measures breath
structure, and scoring them as disfluency inverts the measurement. Both grains agree on both fields
on all 897 and all 597 recordings, so the one-to-many collapse that breaks the other read families
does not arise here.

### `word-color-stroop` (472) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent` ·
`declared_duration_s` (75 s) from the sidecar.
**Absent:** `stimulus_alignment`† (D1), whose expected-token list here is the **answer** sequence.

```
detect_stroop_sequence(words, spans, stream_extent, stimulus_alignment†,
                       *, p_omission_score_max)
        -> (done, components, deviations)

    a = stimulus_alignment† built from the recording's own 15-colour sequence
    base = detect_read_text(...) with two changes:
        - no `filler` deviation is emitted, ever
        - a substitution is scored against the colour, never against the displayed word
    done       = every expected colour was realised at least once
    components = base.components + [("item_%d" % i, *c.extent) for i, c in a.expected]
    deviations = base.deviations without filler
               + [("stimulus_mismatch", *w.extent, {"expected": colour, "read": w.text})
                  for the colour/word confusions the task exists to elicit]
```

**Notes.** The instruction says name the colour, do not read the word, so `stimulus_text` is the
expected *answer* sequence — 472 distinct sequences over 472 recordings, never shared, so the
pattern is maximally per recording and no family-scoped table can serve it. `filler` must **not** be
a deviation here: hesitation and self-correction are the task's dependent variable
([`branch-speech.md:130-134`](branch-speech.md)). Declared 75 s, measured median 75.8 s, 463 of 472
between 74 and 77 s.

### `free-speech` (3,074) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent`.
**Absent:** `stimulus_alignment`† (D1), read **inverted**; a lexical-overlap statistic (the
`p_echo_*` decision below has no fitted value and no estimator in the graph).

```
detect_free_speech_v1(words, spans, stream_extent, stimulus_alignment†,
                      *, p_echo_ngram_n, p_echo_overlap_max)
        -> (done, components, deviations)

    speech_runs = [s for s in spans if s.measure == "asr"]
    done        = any lexical(words) at all — no target text exists, so presence is the pattern
    components  = [("response", *s.extent) for s in speech_runs]       # task_extent = their hull

    # The negative pattern: v1 says "do not record yourself reading the prompt".
    a       = stimulus_alignment† against the recording's own question
    overlap = fraction of p_echo_ngram_n-grams of the prompt found verbatim in lexical(words)
    deviations = [("stimulus_mismatch", *a.echo_extent,
                   {"reading": "verbatim_prompt", "overlap": overlap})]
                 if overlap > p_echo_overlap_max
               + off_task(components, spans)
```

**Notes.** This row cannot use the family grain at all: the acoustictask JSON carries one frozen
prompt for all 912 sidecars, which is right on 380 recordings and wrong on 2,694 — four questions,
per index. `p_echo_overlap_max` is a cut nobody has fitted, and it is the decision that separates
reading from ordinary quotation of a question's words. 1,072 of these recordings link to no
acoustictask sidecar at all, which is the only structural break in the sidecar graph and is
unexplained.

### `free-speech-v2` (2,120) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent` ·
`declared_duration_s` (~30 s).
**Absent:** nothing the row needs. v2 drops the anti-pattern, so D1 is not load-bearing here.

```
detect_free_speech_v2(words, spans, stream_extent, declared_duration_s,
                      *, p_response_min_s)
        -> (done, components, deviations)

    speech_runs = [s for s in spans if s.measure == "asr"]
    response    = hull of speech_runs
    done        = response exists and duration(response) >= p_response_min_s
    components  = [("response", *response)]                            # task_extent
    deviations  = off_task(components, spans)
    counts      = {"declared_duration_s": {"found": duration(stream_extent),
                                           "declared": declared_duration_s}}
    # No verbatim-echo deviation. v2's instruction drops "do not record yourself reading the
    # prompt" and asks for a conversational answer; firing v1's deviation here invents one.
```

**Notes.** The acoustictask prompt matches **none** of the six v2 questions, on all 707 sidecars, so
the family grain is wrong on every one of the 2,120 recordings. Treating v1 and v2 alike is the
error this pair exists to prevent.

### `story-recall` (889), `-v2` (660) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent`.
**Absent:** `stimulus_alignment`† (D1); a lexical-overlap statistic — and note that the overlap
itself is arithmetic once D1 exists, so what is missing is the *cut*, not an estimator.

```
detect_story_recall(words, spans, stream_extent, stimulus_alignment†,
                    *, p_ngram_n, p_verbatim_overlap_max, p_coverage_min)
        -> (done, components, deviations)

    source   = the recording's own source story (v1 grandfather, v2 frog — read per recording,
               because the five Spanish v1 recordings carry v2's story under v1's family name)
    produced = lexical(words)
    coverage = |content tokens of source seen in produced| / |content tokens of source|
    verbatim = longest common p_ngram_n-gram run between source and produced, as a fraction
    done     = coverage >= p_coverage_min                      # recalled, in any words
    components = [("recall", *hull of asr spans)]              # task_extent
    deviations = [("stimulus_mismatch", *run.extent,
                   {"reading": "read_not_recalled", "verbatim": verbatim})]
                 if verbatim > p_verbatim_overlap_max
               + off_task(components, spans)
```

**Notes.** *Recall in your own words*: semantic coverage is expected and verbatim reproduction is
the deviation. An n-gram count against the source answers the question the instruction actually
poses — recalled or read — and verbatim echo is an n-gram phenomenon; a semantic-coverage measure
would need a cut nobody has fitted, to answer a question nobody asked. `text/tasks/embeddings_extraction`
exists and is not wired into triage, and this row is not a reason to wire it.

### `cinderella-story` (258) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent` ·
`declared_duration_s`.
**Absent:** nothing that could help. The source is a physical storybook; `stimulus_text` is empty on
all 258, so **no overlap measure is definable**, D1 included.

```
detect_narrative_presence(words, spans, stream_extent, declared_duration_s,
                          *, p_response_min_s)
        -> (done, components, deviations)

    response   = hull of [s for s in spans if s.measure == "asr"]
    done       = response exists and duration(response) >= p_response_min_s
    components = [("narrative", *response)]                            # task_extent
    deviations = off_task(components, spans)
    counts     = {"declared_duration_s": {"found": duration(stream_extent),
                                          "declared": declared_duration_s}}
```

**Notes.** Presence and extent only, and that is the ceiling rather than a first step: the
`story-recall` n-gram method **does not transfer**, because there is no text to overlap against.
Median 93.9 s; 18 of 258 run under a second.

### `productive-vocabulary` (2,910) — SPEECH

**Requires:** as `cinderella-story`, plus the per-recording cue word.
**Absent:** `stimulus_alignment`† (D1) for the cue; a "is this speech a definition *of* that cue"
measure, which is **not** a PREPROCESS derivative and has no viable approach in this graph.

```
detect_definitional_speech(words, spans, stream_extent, stimulus_alignment†,
                           *, p_response_min_s)
        -> (done, components, deviations)

    response   = hull of [s for s in spans if s.measure == "asr"]
    done       = response exists and duration(response) >= p_response_min_s   # presence only
    components = [("definition", *response)]
    deviations = off_task(components, spans)
    # Whether the speech defines its cue: NO VIABLE APPROACH here. It needs a lexicon or a text
    # model, is branch-local, needs no waveform, and 78 recordings carry no cue at all.
```

**Notes.** 204 distinct cues under one family name, so the family grain carries nothing, and on the
78 cue-less recordings neither grain does.

### `picture-description` (889), `-option1` (373), `-option2` (329) — SPEECH

**Requires:** `consensus_transcript` + `word` entities (with `timings` for the gap structure) ·
`spans` (`asr`, `gap`) · `hear_scores` (raw, `Breathe`) · `stream_extent`.
**Absent:** nothing for presence and extent; S4's connected-speech measures are unbuilt branch code,
not a missing derivative.

```
detect_connected_speech(words, spans, hear_scores, stream_extent,
                        *, p_response_min_s, p_pause_min_s, p_breath_group_min_gap_s)
        -> (done, components, deviations)

    response   = hull of [s for s in spans if s.measure == "asr"]
    done       = response exists and duration(response) >= p_response_min_s
    pauses     = inter-word gaps >= p_pause_min_s, from consecutive lexical words' `timings`
    groups     = runs of lexical words separated by a pause or a `Breathe`-scoring window
    components = [("description", *response)]                          # task_extent
               + [("breath_group_%d" % i, *g.extent) for i, g in groups]
    deviations = off_task(components, spans)
    measures   = {"speech_rate_from_consensus_words_per_s": ..., "pause_fraction": ...,
                  "n_breath_groups": len(groups)}     # named for their convention, not "rate"
```

**Notes.** `picture-description` and `-option1` carry a **byte-identical** instruction and both have
empty `stimulus_text`; the only declared difference is the image, which is not in the sidecar as
anything a method can read. **No method here may distinguish them** — any difference found is the
image's. `-option2` asks for complete sentences as though describing it for the blind, which is a
different instruction and the same detector.

### `open-response-questions` (199) — SPEECH

**Requires:** as `picture-description`, plus the one shared 447-character prompt.
**Absent:** `stimulus_alignment`† (D1) only if the prompt is to be excluded as an echo; presence and
extent need nothing.

```
detect_open_response(words, spans, hear_scores, stream_extent, declared_duration_s,
                     *, p_response_min_s, p_pause_min_s, p_breath_group_min_gap_s)
        -> detect_connected_speech(...) with counts:
           {"declared_duration_s": {"found": duration(stream_extent),
                                    "declared": declared_duration_s}}
```

**Notes.** One `stimulus_text` shared by all 199 recordings makes this the single connected-speech
family whose `expected_speech` is legitimately family-scoped; every other one is per recording or
absent. Median 30 s.

### `animal-fluency` (195) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent` ·
`declared_duration_s` (60 s timer).
**Absent:** category membership — branch-local, needs no waveform, **not** a PREPROCESS derivative;
`transcript_repeat`‡ as a store measurement.

```
detect_item_list(words, spans, stream_extent, declared_duration_s,
                 *, p_repeat_normaliser, p_repetition_allowed)
        -> (done, components, deviations)

    items      = lexical(words) grouped into produced items in order
    normalised = [p_repeat_normaliser(i.text) for i in items]
    repeats    = [i for i in items if normalised[i] seen earlier in the list]
    done       = len(items) > 0
    components = [("item_%d" % k, *i.extent) for k, i in enumerate(items)]
    deviations = [("repeated_item", *i.extent, {"first_at": ...})    # an ELEVENTH type,
                  for i in repeats] if not p_repetition_allowed      # owed a row in
                                                                     # branch-conventions.md
    counts     = {"items": {"found": len(items), "declared": None},
                  "declared_duration_s": {"found": duration(stream_extent),
                                          "declared": declared_duration_s}}
    # Whether an item is in the declared category: NO VIABLE APPROACH in this graph.
    # It is a lexicon or a text embedding, one consumer, and no audio.
```

**Notes.** The category lives only in `instructions`; `stimulus_text` is empty on all 195. Median
duration 60 s, matching the declared timer, so the duration check is the row's sharpest free signal.

### `random-item-generation` (265), `-v2` (207) — SPEECH

**Requires:** as `animal-fluency`, plus the **per-recording category**, which no grain above the
recording carries.
**Absent:** the same two, and one more: `p_repetition_allowed` must come from the recording's own
category, not from the family.

```
detect_random_items(words, spans, stream_extent, hints, *, p_repeat_normaliser)
        -> detect_item_list(..., p_repetition_allowed = category_allows_repetition(hints))

    # Eight of ten categories say "Do not repeat any item". `Letters` and `Numbers` say the
    # opposite — "repetition allowed" — on 48 of 265 v1 and 77 of 203 English v2 recordings.
    # A family-scoped rule inverts the instruction on roughly a fifth of them.
```

**Notes.** `transcript_repeat` — largest repeat count of any normalised token — is already computed
for `ddk.lexical_repetition` (`default.yaml:302`) but lives in `routing_analysis/features.py`, is
read only by the ruleset and is not a store measurement. Moving it is a convenience, not the
capability: a counter over normalised consensus tokens is arithmetic over word entities the store
already holds.

### The AIRWAY families — two functions every one of them uses

Written once here; the family blocks below call them.

```
events_in_span(energy_envelope, span, *, p_smoothing_window_s, p_peak_prominence_db,
               p_event_min_s, p_trough_return_db)
        -> [(start, end)]

    e      = energy_envelope.envelope_dbfs over span.extent, smoothed over p_smoothing_window_s
    floor  = energy_envelope.floor_dbfs                     # one global value, not a local floor
    peaks  = local maxima of e with prominence >= p_peak_prominence_db
    for each peak: onset  = walk back to where e falls p_trough_return_db below the peak
                   offset = walk forward by the same rule
    keep events with duration >= p_event_min_s
    # A5/A6. The merging case is not `spans.min_separation_ms` (30 ms; volitional coughs are
    # seconds apart): it is a series produced on one exhalation, where the envelope never falls
    # back within k_db of the floor between bursts and the whole series is one span. That is
    # multiple maxima inside one span, and separating them is arithmetic over this array.
```

```
sounds_like(span, span_hear, span_yamnet, *, p_label_set, p_score_min) -> bool

    h = [w for w in span_hear   if w.span_id == span.id]
    y = [w for w in span_yamnet if w.span_id == span.id]
    return any window in h or y whose raw_scores[label] >= p_score_min
           for a label in p_label_set
    # p_label_set is `cough_labels` or `breath_labels` as `routing_analysis/labels.py:100-105`
    # defines them: HeAR's own group labels, plus the AudioSet closure for YAMNet. `raw_scores`
    # is always written; `labels` and `scores` are a decision over it — a label is carried only
    # when it is in the window's top `label_top_k` (4) AND clears `default_threshold` (0.2 for
    # YAMNet and HeAR, null for AST), all owner-directed rather than fitted (`default.yaml:91-107`).
    # A cough label ranked fifth is dropped by a size, not by a score, so this body reads
    # raw_scores and applies its own named floor.
```

```
detect_lexical_intrusion(words, spans) -> (done, components, deviations)

    # SPEECH's row over every AIRWAY-declared family: L, none expected, so any lexical
    # content is off-task by construction.
    intrusions = lexical(words)
    done       = len(intrusions) == 0            # "done" here means the negative pattern held
    components = []
    deviations = [("off_task_extent", *w.extent, {"text": w.text, "agreement": w.agreement})
                  for w in intrusions]
    # AIRWAY owns the deviation (`branch-airway.md:164-165`); SPEECH supplies the extents.
    # Measured examples are examiner speech — "I'll have you do that one more time. [breath]",
    # "So just breathe."
```

Measured examples:
[`../20260910-taxonomy-routing-evidence/measurements.md:177-182`](../20260910-taxonomy-routing-evidence/measurements.md).

### `respiration-and-cough-cough` (1,788) — AIRWAY and SPEECH

**Requires:** `spans` (`amplitude`, `gap`) · `energy_envelope` · `span_hear` · `span_yamnet` ·
`consensus_transcript` + `word` entities (for the intrusion row) · `stream_extent`.
**Absent:** nothing. Every instrument is in the store; A5/A6's operating points are decisions.

```
detect_cough_series(spans, energy_envelope, span_hear, span_yamnet, words, stream_extent,
                    *, p_cough_label_set, p_cough_score_min, p_expected_count,
                       p_smoothing_window_s, p_peak_prominence_db, p_event_min_s,
                       p_trough_return_db)
        -> (done, components, deviations)

    carriers = [s for s in spans if s.measure == "amplitude"
                and sounds_like(s, span_hear, span_yamnet,
                                p_label_set=p_cough_label_set, p_score_min=p_cough_score_min)]
    events   = [e for s in carriers for e in events_in_span(energy_envelope, s, ...)]
    done     = len(events) > 0
    components = [("cough_%d" % k, *e) for k, e in enumerate(events)]   # hull is the task_extent
    deviations = detect_lexical_intrusion(words, spans).deviations
               + [("truncation", ...)] if the first or last event touches stream_extent
               + off_task(components, spans)
    counts     = {"expected_event_count": {"found": len(events), "declared": p_expected_count}}
```

**Notes.** The count is of **events**, not of label-carrying spans: `by_label` increments once per
(span, label) pair (`airway.py:280`), so a 4 s span holding three coughs counts 1 today. That is the
defect the `events_in_span` decomposition removes, and it is branch code.

### `respiration-and-cough-v2-hardcough` (698) — AIRWAY, SPEECH and QUALITY

**Requires:** `spans` · `energy_envelope` · `span_hear` · `span_yamnet` · `level` ·
`spectrogram_wideband` · `disruptions_file` · `stream_extent`.
**Absent:** `band_profile`† (D3) for the hygiene clause's spectral-tilt half;
`phonation_tracks.rms_dbfs`† (D2).
**No viable approach** for *"hard"* — see the note.

```
detect_hard_cough(spans, energy_envelope, span_hear, span_yamnet, level,
                  spectrogram_wideband, disruptions_file, band_profile†, stream_extent,
                  *, p_cough_label_set, p_cough_score_min, p_smoothing_window_s,
                     p_peak_prominence_db, p_event_min_s, p_trough_return_db)
        -> (done, components, deviations)

    events = as detect_cough_series, with no declared count
    done   = len(events) > 0
    components = [("cough_%d" % k, *e) for k, e in enumerate(events)]
    measures   = per event, never a verdict:
                   peak_over_floor_db, spectral_balance from spectrogram_wideband
                 with covariates: level.{peak_dbfs, rms_dbfs, lufs}, contains_clip,
                                  disruptions_file, band_profile† (D3)
    # "hard": NO VIABLE APPROACH. There is no within-recording contrast to read effort against
    # (unlike loudness-v2) and no SPL reference anywhere in the graph. The output is a
    # measurement with its covariates; never a `hard` / `not hard` verdict.
```

```
# The hygiene clause — "do not cover your mouth or place your hand between your mouth and the
# microphone" — is a QUALITY expectation, not an AIRWAY one:
detect_occluded_microphone(level, spectrogram_wideband, band_profile†, disruptions_file,
                           *, p_tilt_max_db_per_octave, p_level_min_dbfs)
        -> a level and spectral-tilt finding on the recording, emitted by QUALITY
```

**Notes.** This is the only instruction in the corpus carrying a recording-hygiene clause, and it is
the only place D3 has a job that is not the route question.

### `voluntary-cough` (327) — AIRWAY and SPEECH

**Requires:** as `-cough`, plus `hear_scores` / `span_hear` `Breathe` for the interleaved breaths.
**Absent:** the same as `-hardcough` for the effort half.

```
detect_cough_cycles(spans, energy_envelope, span_hear, span_yamnet, hear_scores, words,
                    stream_extent,
                    *, p_cough_label_set, p_breath_label_set, p_cough_score_min,
                       p_breath_score_min, p_expected_count, p_smoothing_window_s,
                       p_peak_prominence_db, p_event_min_s, p_trough_return_db)
        -> (done, components, deviations)

    coughs  = events as detect_cough_series
    breaths = [s for s in spans if sounds_like(s, span_hear, span_yamnet,
                                               p_label_set=p_breath_label_set,
                                               p_score_min=p_breath_score_min)]
    # The expected pattern is an ALTERNATION: material between coughs is matched as breath,
    # never scored as off_task_extent. A cough detector alone is insufficient here.
    cycles  = [(cough, following breath) pairs in time order]
    done    = len(coughs) > 0
    components = [("cough_%d" % k, *c) for k, c in enumerate(coughs)]
               + [("breath_%d" % k, *b.extent) for k, b in enumerate(breaths)]
    deviations = detect_lexical_intrusion(words, spans).deviations
               + off_task(components, spans)      # gaps matching neither cough nor breath
    counts     = {"expected_event_count": {"found": len(coughs), "declared": p_expected_count}}
```

**Notes.** Median 13.8 s against `-v2-hardcough`'s 4.6 s, consistent with three cough-and-breathe
cycles in one file.

### `respiration-and-cough-fivebreaths` (3,576) — AIRWAY and SPEECH

**Requires:** `spans` (`amplitude`, `gap`) · `energy_envelope` · `span_hear` · `span_yamnet` ·
`hear_scores` · `stream_extent` · `hints.metadata.task_token`‡ for the route index.
**Absent:** `band_profile`† (D3).
**No viable approach** for the route itself.

```
detect_breath_cycles(spans, energy_envelope, span_hear, span_yamnet, hear_scores, words,
                     stream_extent, hints, band_profile†,
                     *, p_breath_label_set, p_breath_score_min, p_expected_count,
                        p_smoothing_window_s, p_peak_prominence_db, p_event_min_s,
                        p_trough_return_db)
        -> (done, components, deviations)

    carriers = [s for s in spans if s.measure == "amplitude"
                and sounds_like(s, span_hear, span_yamnet,
                                p_label_set=p_breath_label_set, p_score_min=p_breath_score_min)]
    cycles   = [e for s in carriers for e in events_in_span(energy_envelope, s, ...)]
    declared_route = "nose" if task_token index ∈ {1, 3} else "mouth"‡   # per recording
    route    = NOT_SEPARABLE_BY_THIS_DESIGN                              # see the note
    done     = len(cycles) > 0
    components = [("breath_%d" % k, *c) for k, c in enumerate(cycles)]
    deviations = detect_lexical_intrusion(words, spans).deviations
               + off_task(components, spans)
    counts     = {"expected_event_count": {"found": len(cycles), "declared": p_expected_count},
                  "declared_route": declared_route, "measured_route": route,
                  "content_band_hz": band_profile†.rolloff_hz}
```

**Notes — why `route` is not a measurement.** The discriminating band sits largely above the 8 kHz
ceiling the 16 kHz working rate imposes, and what remains below it is a spectral-tilt difference
confounded one-for-one with mouth-to-microphone distance and angle. The confound is not incidental:
**a participant told to breathe through the mouth points the mouth at the phone; one told to breathe
through the nose with the mouth closed does not.** Route and geometry change together by
construction, so the 1,778 / 1,778 within-session split is perfectly balanced on route and perfectly
confounded on source-to-microphone transfer — all 894 sessions carrying the task carry all four
indices, so nose and mouth are always the same participant in the same session, and a classifier
fitted on that contrast will separate the two conditions while what it learned stays unidentifiable
from the recordings alone. `gammatone` is not a privileged instrument here: it is an ERB rebinning
of the same short-time spectrum the two spectrograms already carry, and carries no frequency content
they lack. D3's job is to make a
negative *attributable* — a contrast that fails on files whose content stops at 4 kHz failed for a
written-down reason — not to make the route measurable. The route index itself is discarded before
any branch sees it, because `task_family` collapses the trailing segment (`families.py:134-144`).

### `respiration-and-cough-v2-threebreathsnose` (699), `-threebreathsmouth` (699) — AIRWAY and SPEECH

**Requires:** as `fivebreaths`, with the route declared by the family rather than by the index.
**Absent:** `band_profile`† (D3). **No viable approach** for the route.

```
detect_breath_cycles(..., p_expected_count = 3)
    with declared_route read from the family name rather than from the task index
```

**Notes.** These two and the `fivebreaths` index split are the only places the route is a declared
contrast, and therefore the only design that could ever validate A7 — which is why the argument
above matters: neither should be treated as validation-grade for it.

### `respiration-and-cough-threequickbreaths` (1,718), `-v2-threebreaths` (699) — AIRWAY and SPEECH

**Requires:** as `fivebreaths`.
**Absent:** nothing beyond `band_profile`†; the interval measurement is arithmetic over
`events_in_span`'s output.

```
detect_quick_breaths(..., p_expected_count = 3, *, p_interval_max_s)
        -> (done, components, deviations)

    cycles    = detect_breath_cycles(...).components
    intervals = [onset(c[k+1]) - onset(c[k]) for k in range(len(cycles) - 1)]
    done      = len(cycles) > 0
    counts    = {"expected_event_count": {"found": len(cycles), "declared": 3},
                 "inter_onset_interval_s": intervals}
    # "Quick" is the measurement, and it is the interval, not the count: three breaths says
    # nothing about whether they were quick. The whole row depends on A5's boundaries, so
    # p_interval_max_s cannot be fitted before p_peak_prominence_db and p_trough_return_db are.
```

### `respiration-and-cough-breath` (1,788), `-v2-breath` (699) — AIRWAY and SPEECH

**Requires:** `hear_scores` (raw, `Breathe`) · `span_hear` · `spans` · `energy_envelope` ·
`stream_extent` · `declared_duration_s` (30 s v1, 20 s v2).
**Absent:** `band_profile`† (D3) for v2's declared mouth route.

```
detect_comfortable_breathing(hear_scores, span_hear, spans, energy_envelope, words,
                             stream_extent, declared_duration_s,
                             *, p_breath_label_set, p_breath_score_min,
                                p_breath_coverage_min)
        -> (done, components, deviations)

    windows  = [w for w in hear_scores                                  # 2.0 s, non-overlapping
                if any raw score for a label in p_breath_label_set >= p_breath_score_min]
    coverage = covered seconds / duration(stream_extent)
    done     = coverage >= p_breath_coverage_min
    components = [("breathing", *merged extent of windows)]             # task_extent
    deviations = detect_lexical_intrusion(words, spans).deviations
               + off_task(components, spans)
    counts     = {"declared_duration_s": {"found": duration(stream_extent),
                                          "declared": declared_duration_s}}
    # `residual.energy_fraction` is NOT read here. `residual = plain - g·FRCRN(plain)` and FRCRN
    # is a speech enhancer, so a high energy fraction means "FRCRN removed most of this signal",
    # i.e. THIS IS NOT SPEECH — satisfied equally by a cough, a glide, room noise and a
    # near-silent file. As a routing gate (`airway.breath`, `default.yaml:285-288`) "not speech"
    # may be adequate; as this row's presence measurement it is not. HeAR's `Breathe` is a
    # positive detection of the thing being asked about, and it is the half that stands.
```

**Notes.** Both declared durations are honoured — 1,440 of 1,788 v1 recordings run 30-31 s, 638 of
699 v2 run 20-21 s — and on v1 the duration check is the sharpest "was the task done" signal in the
corpus: **132 of 1,788 run under a second.**

### `breath-sounds` (326) — AIRWAY and SPEECH

**Requires:** as the three-breath families, plus `declared_duration_s` (~73 s: 60 s relax + the
task).
**Absent:** `band_profile`† (D3). **No viable approach** for the declared mouth route.

```
detect_breath_sounds(..., p_expected_count = 3, *, p_relax_s)
        -> (done, components, deviations)

    result = detect_breath_cycles(..., p_expected_count = 3)
    if duration(stream_extent) >= p_relax_s + duration(hull of result.components):
        # The one family whose instruction PRESCRIBES material that is not the task.
        deviations += [("off_task_extent", 0.0, p_relax_s, {"reading": "declared_relax_period"})]
    counts += {"declared_duration_s": {"found": duration(stream_extent),
                                       "declared": declared_duration_s}}
```

**Notes.** *"Please relax for 60 seconds until the task starts. Take three deep breaths in a row in
and out of the mouth."* — so it is **not** the uncounted durational task its name suggests; it is
closest to `-v2-threebreathsmouth`. The measured median is 13.2 s, so the relax period is evidently
not inside the file, and the branch above emits the by-instruction `off_task_extent` only on a
recording that actually runs long enough to contain it.

### The `SYLLABLE_REPETITION` families (7,989) — SPEECH's row over all ten

```
detect_lexical_absence(words, spans, *, p_expected_lexical_max)
        -> (done, components, deviations)

    produced = lexical(words)
    done     = len(produced) <= p_expected_lexical_max     # near-zero lexical content is correct
    components = []
    deviations = [("off_task_extent", *w.extent, {"text": w.text}) for w in produced]
    # `/pa/` is not lexical and ASR mostly declines it. These families are POSITIVES for
    # SPEECH's reference set (`reference_family_set.SPEECH: speech` = lexical_speech |
    # syllable_repetition, `default.yaml:241`), so near-zero lexical content is the correct
    # observation, not a miss. `speech.lexical >= 2` firing here is over-routing on
    # function-word artefacts: 71.3% of that gate's apparent false positives are DDK families
    # (`dag.md:195-196`). `diadochokinesis-buttercup` is the exception and has its own block.
```

### The DDK trains — one function they share

```
train_rate(energy_envelope, extent, *, p_modulation_band_hz, p_rate_prominence_min)
        -> (rate_hz, train_extent)

    e     = energy_envelope.envelope_dbfs over extent, mean-removed
    S     = magnitude spectrum of e over p_modulation_band_hz
    peak  = argmax S, kept only if its prominence >= p_rate_prominence_min
    # D1, and it is the instrument for D2 as well. Praat's `extract_speech_rate`
    # (`praat_parselmouth.py:160`, de Jong & Wempe) is ALREADY RUNNING inside `praat_features`
    # on every recording and will not carry DDK, for two reasons internal to the method:
    #   - a candidate intensity peak counts only if the dip to the next peak exceeds `min_dip`
    #     (2 or 4 dB, chosen by a whole-file HNR test), and in a fast /pʌ/ train with weak
    #     bilabial closure the inter-syllable dip is frequently smaller — so the FASTEST trains
    #     are the ones most likely to be under-counted;
    #   - a peak must be voiced and inside a "sounding" interval, whose silence tier is built
    #     with `min_pause = 0.3 s` — longer than an entire DDK syllable cycle.
    # Both failures are correlated with the quantity DDK exists to measure, so the bias is
    # signal-dependent, not noise. The envelope modulation spectrum has neither failure mode
    # and needs no new derivative.
```

### `diadochokinesis-pa` (896), `-ta` (896), `-ka` (896) — DDK and SPEECH

**Requires:** `energy_envelope` · `spans` (`amplitude`, `gap`) · `continuity_trace` ·
`consensus_transcript` + `word` entities (for SPEECH's absence row) · `stream_extent`.
**Absent:** nothing. D1–D6 are unbuilt branch code and unfitted operating points, not missing
derivatives.

```
detect_ddk_train(energy_envelope, spans, continuity_trace, stream_extent,
                 *, p_modulation_band_hz, p_rate_prominence_min, p_train_min_s,
                    p_expected_count, p_smoothing_window_s, p_peak_prominence_db,
                    p_event_min_s, p_trough_return_db)
        -> (done, components, deviations)

    carriers = [s for s in spans if s.measure == "amplitude"
                and duration(s) >= p_train_min_s]
    train    = the carrier with the strongest modulation peak, else None
    if train is None: return (False, [], [])
    rate_hz, _ = train_rate(energy_envelope, train.extent, ...)         # D1 / D2
    onsets     = [e for e in events_in_span(energy_envelope, train, ...)]
    intervals  = [onsets[k+1].start - onsets[k].start for k in ...]     # D3
    done       = rate_hz is not None and len(onsets) > 0
    components = [("train", *train.extent)]                             # task_extent
               + [("syllable_%d" % k, *o) for k, o in enumerate(onsets)]
    deviations = [("truncation", ...)] if train touches stream_extent's edge
               + off_task(components, spans)
    counts     = {"expected_event_count": {"found": len(onsets), "declared": p_expected_count},
                  "ddk_syllable_rate_from_envelope_modulation_hz": rate_hz,
                  "inter_onset_interval_s": intervals,
                  "train_fraction_of_recording": duration(train) / duration(stream_extent)}  # D5
```

**Notes.** v1 states the count (10) and v2 does not. The gate that fires without a transcript reads
`ppg.segment_rate_per_s`, which is owed a code change — `extract_ppg_segments`
(`tasks/features_extraction/ppg.py:349`) is called from `routing_analysis/features.py:421`,
`ppg.py:467`, `ppg.py:543` and `plotting.py:1667`, never from a node, so the store holds the
whole-file posteriorgram alone with no per-span query. The function above needs neither.

### `diadochokinesis-v2-puh` (702), `-tuh` (702), `-kuh` (702) — DDK and SPEECH

**Requires:** as above, plus `declared_duration_s` (a fixed 5 s timer).
**Absent:** nothing.

```
detect_ddk_train_timed(..., p_expected_count = None, declared_duration_s)
        -> detect_ddk_train(...) with
           counts += {"declared_duration_s": {"found": duration(stream_extent),
                                              "declared": declared_duration_s}}
    # No expected_event_count: the instruction says "until the timer runs out". The
    # `'puhpuhpuhpuhpuhpuh'` in the instruction is an orthographic illustration, not a
    # six-repetition instruction.
```

**Notes.** 630-646 of each family's 702 recordings run 5-6 s, against a v1 median of 5 s spread over
3-9 s. The fixed denominator makes D5's train fraction and the rate directly comparable across
participants in a way v1's participant-terminated recordings are not — the cleanest DDK design in
the corpus, and none of D1-D6 is built to use it.

### `diadochokinesis-pataka` (896), `-v2-puhtuhkuh` (701) — DDK and SPEECH

**Requires:** `energy_envelope` · `spans` · `spectrogram_wideband` · `stream_extent`.
**Absent:** nothing — the instrument this row needs is already written. `ppg_posteriorgram` is
**not** the instrument; see the note.

```
detect_ddk_sequence(energy_envelope, spans, spectrogram_wideband, stream_extent,
                    *, p_modulation_band_hz, p_rate_prominence_min, p_train_min_s,
                       p_burst_window_ms, p_place_centroid_bands_hz, p_place_margin,
                       p_expected_sequence, p_expected_count,
                       p_smoothing_window_s, p_peak_prominence_db, p_event_min_s,
                       p_trough_return_db)
        -> (done, components, deviations)

    train  = as detect_ddk_train
    onsets = events_in_span(energy_envelope, train, ...)
    for each onset:
        burst = spectrogram_wideband frames in the first p_burst_window_ms after the onset
        place = argmax over p_place_centroid_bands_hz of the burst's band energies
                → one of {labial, alveolar, velar} when the winning band clears the runner-up
                  by p_place_margin, else `unresolved`
        # /p/, /t/ and /k/ differ in burst spectrum in the textbook way — /t/ high-frequency
        # dominant, /k/ a compact mid-frequency peak, /p/ diffuse and falling — and all three
        # sit comfortably inside the 8 kHz band. `spectrogram_wideband` is a 5 ms window at a
        # 5 ms hop, the classical resolution for exactly this measurement; `gammatone` at the
        # same hop gives it pre-pooled.
    produced = the place sequence, in time order
    done     = produced is non-empty and cycles through p_expected_sequence at least once
    components = [("train", *train.extent)]
               + [("syllable_%d" % k, *o) for k, o in enumerate(onsets)]
    deviations = [("syllable_sequence_mismatch", *o,
                   {"expected": p_expected_sequence[k % 3], "measured": place})
                  for k, (o, place) in enumerate(zip(onsets, produced))
                  if place != p_expected_sequence[k % 3] and place != "unresolved"]
    counts     = {"expected_event_count": {"found": len(onsets), "declared": p_expected_count},
                  "sequence_collapse_fraction": share of positions realised as one place}
```

**Notes.** `/pa-pa-pa/` is a collapse of the sequence and is the clinically meaningful finding, which
is why the deviation is `syllable_sequence_mismatch` and deliberately **not** `stimulus_mismatch`:
there is no stimulus text and no lexical expectation. A per-span PPG query would spend a model pass
to recover less — a phonetic posteriorgram is trained on connected speech, and on a rapid nonsense CV
train with no lexical context its acoustic-model prior works against the discrimination, smearing
adjacent frames toward whatever phoneme sequence the training distribution favours. v1 asks for the
sequence 10 times (30 syllables), v2 until the 5 s timer runs out, so the two are counted and
uncounted respectively and are not one measurement.

### `diadochokinesis-buttercup` (896), `-v2-buttercup` (702) — DDK and SPEECH

**Requires:** `consensus_transcript` + `word` entities · `energy_envelope` · `spans` ·
`stream_extent`.
**Absent:** nothing. `transcript_repeat`‡ is arithmetic over word entities the store already holds.

```
detect_repeated_word(words, energy_envelope, spans, stream_extent,
                     *, p_target_token, p_repeat_normaliser, p_expected_count,
                        p_modulation_band_hz, p_rate_prominence_min)
        -> (done, components, deviations)

    hits    = [w for w in lexical(words)
               if p_repeat_normaliser(w.text) == p_repeat_normaliser(p_target_token)]
    train   = hull of hits
    rate_hz = train_rate(energy_envelope, train, ...)        # the acoustic half, for v2's timer
    done    = len(hits) > 0
    components = [("repeat_%d" % k, *w.extent) for k, w in enumerate(hits)]
               + [("train", *train)]                                    # task_extent
    deviations = off_task(components, spans)
    counts     = {"expected_event_count": {"found": len(hits), "declared": p_expected_count},
                  "ddk_syllable_rate_from_envelope_modulation_hz": rate_hz}
```

**Notes.** The only DDK family with a lexical pattern, so the only one where the recognisers produce
the count directly and the only one where `ddk.lexical_repetition >= 3` fires for the right reason
rather than on function-word repetition. v1 states the count and v2 does not, so the two are not one
measurement.

### Routed against no declared task — VOICE, AIRWAY and DDK

The three rows where the branch was routed to a recording whose family declares none of its
patterns. **Declared families are not ground truth, and neither is a routing**: the branch measures
what it finds, concludes on its own question, and asserts nothing about the other branch's task.

**Requires (VOICE):** `spans` · `phonation_tracks` · `continuity_trace` · `energy_envelope` ·
`consensus_transcript` + `word` entities. **Absent:** `phonation_tracks.{hnr_db,cpps_db,rms_dbfs}`†
(D2) — and here the stationarity qualifier is load-bearing over 22,277 routed recordings, because
connected speech passes voiced-fraction, F0-availability and interruption tests and would otherwise
have perturbation measured over consonants and pauses.

```
measure_voice_without_declared_task(spans, phonation_tracks, continuity_trace,
                                    energy_envelope, words,
                                    *, p_voiced_strength_min, p_voiced_fraction_min,
                                       p_f0_spread_window_s, p_f0_spread_max_semitones,
                                       p_continuity_min)
        -> (done, components, deviations)

    done       = UNDETERMINED                 # no pattern was expected; "done" is not this
                                              # branch's question on this recording
    qualifying = spans passing the stationarity qualifier of detect_prolonged_vowel's P2
    components = [("phonation", *s.extent) for s in qualifying]     # `label`, or `refine` where
                                                                   # the span carried a label
    contests   = [("contest", *s.extent) for s in spans a proposer marked phonation
                  that fail the qualifier]
```

**Requires (AIRWAY):** `spans` · `span_hear` · `span_yamnet` · `hear_scores` · `energy_envelope`.
**Absent:** nothing.

```
measure_airway_without_declared_task(spans, span_hear, span_yamnet, hear_scores,
                                     energy_envelope,
                                     *, p_cough_label_set, p_breath_label_set, p_score_min)
        -> (done, components, deviations)

    done       = UNDETERMINED
    components = [("airway_event", *s.extent) for s in spans
                  if sounds_like(s, span_hear, span_yamnet, ...)]
    contests   = spans a proposer labelled airway that carry no such evidence
    # A breath during passage reading is NOT a deviation — it is how S4 measures breath-group
    # structure. AIRWAY's gate evidence is `unavailable` on 56,505 of 62,547 recordings.
```

**Requires (DDK):** `energy_envelope` · `spans` · `consensus_transcript` + `word` entities.
**Absent:** nothing.

```
measure_ddk_without_declared_task(energy_envelope, spans, words,
                                  *, p_modulation_band_hz, p_rate_prominence_min,
                                     p_repeat_normaliser)
        -> (done, components, deviations)

    done       = UNDETERMINED
    components = [("repetition", *extent) for each modulation peak or repeated token found]
    # Repetition occurs in ordinary speech — a stutter, a false start, a repeated word. The
    # branch measures what it finds and says what it is; it does not assert that a Harvard
    # sentence failed to be a DDK task (`branch-ddk.md:66-70`). DDK is a declared branch with
    # no node: `run.py:302-306` marks it SKIPPED with "no node implements this branch".
```

---

## The worked example: `prolonged-vowel`

The owner named this case, and it is where the whole argument is visible.

**The instruction**, verbatim from the recording-grain sidecar and carried **identically by all
1,575 English recordings** of the family — there is no second English wording:

> This task helps us analyze features in your voice. Please press the play button to listen to the
> demonstration on how to complete the task. Then, tap the record button and imitate the speaker by
> repeating the sentence "1, 2, 3 aah" in your normal voice. Please hold the sound "aah" until the
> timer runs out.

The count-in is therefore **prescribed**, not a participant habit. That matters for how a mismatch
is read: a recording with no `one two three` is a departure from an instruction, not an idiosyncrasy.

**Two expected patterns, and the instruction states their order:**

| | pattern | kind | what it is |
| --- | --- | --- | --- |
| P1 | `one`, `two`, `three` | lexical, ordered | three tokens, once |
| P2 | a held /a/ | acoustic | one continuous voiced production, single vowel, F0 holding, to the timer |

`stimulus_text` is **empty** for this family at both grains, on all 1,604 recordings; the pattern
exists only inside `instructions`. The timer is 12 s and the corpus honours it: **1,187 of 1,604**
recordings run 12-13 s and the median is 12.1 s, so P1 and P2 share a fixed 12 s budget and a long
count-in is a short vowel.

**Only P2 is the measurement.** Every F0, jitter, shimmer, HNR and CPPS figure for this recording
should be taken over the vowel alone.

### What happens today

**The scalars are whole-file, and the file contains the count-in.** `praat_features`
(`preprocess.py:1180`) is documented as *"Praat/Parselmouth's whole-file feature set over the
`enhanced` stream"*. It resolves the whole `enhanced` `Audio` and passes it unsliced. Every Praat
aggregation then takes the whole-object time range:

- `extract_pitch_descriptors` — `call(pitch, "Get mean", 0, 0, unit)` (`praat_parselmouth.py:617`);
- `extract_harmonicity_descriptors` — `call(harmonicity, "Get mean", 0, 0)` (`:740`);
- `extract_jitter` / `extract_shimmer` — `call(..., "Get jitter (…)", 0, 0, …)` (`:1355`, `:1411`);
- `extract_cpp_descriptors` — the mean over every finite frame of the file's cepstrogram (`:1036`).

Praat's `Get mean` on a **Pitch** object intrinsically skips unvoiced frames; that is the only
voicing mask anywhere in the set, and it comes from Praat, not the graph. HNR, CPPS, jitter and
shimmer carry no mask at all.

So on a `prolonged-vowel` recording `mean_f0_hertz` is the mean over the counted `one two three`
**and** the vowel; `mean_hnr_db` and `mean_cpp` additionally average in the silences and the
count-in's consonants. The figures are not the vowel's.

**And VOICE emits no scalars at all.** Its subject is `family == "phonation"` spans (`voice.py:40`,
`:227-231`). The module's only write of one (`voice.py:333-346`, `"family": _PHONATION_FAMILY` at
`:337`) sits **downstream of the no-span return**, and nothing else in the repository writes one, so
no phonation span is ever produced and none is ever bootstrapped — the detector was retired
2026-09-04. The branch takes the no-span return at `:235-264` and returns `Outcome.FAIL` with
`spans_n: 0`, `phonation_s: 0.0`, on every recording. Measured: 6 of 6 VOICE-routed recordings in
the 2026-09-15 run ([`branch-voice.md:11-24`](branch-voice.md)). `period_marks` and `voice_tracks`
are never written.

**The spans that would carry the vowel are in the same store, live, unread.** PREPROCESS's `spans`
block writes amplitude spans (`preprocess.py:1875-1891`) carrying `signal`, `measure`,
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
   at [`branch-conventions.md:61-71`](branch-conventions.md).
2. **A separator between the count-in and the vowel.** The lexical half is free — consensus `word`
   entities carry `extent` plus per-source `timings`, so `one two three` locates itself and P2 is the
   voiced production under no lexical word. The acoustic half is V1's stationarity qualifier and is
   **owed a cut**, not a measurement — a correction to an earlier draft of this document, which
   said nothing named `spectral_flux` or `stationarity` exists anywhere in `src/senselab` and read
   that as a missing estimator. Nothing carries those names, and the measurement is nevertheless in
   the store: `continuity_trace` is the cosine similarity between consecutive log-magnitude spectra
   (`spectral_continuity/api.py:10`), written per sample (`preprocess.py:2549`), and F0 and formant
   stationarity are statistics of `f0_hz` and `f1..f4_hz` at a 10 ms hop. What is owed is a named
   statistic and a window, in `data/` with a derivation.
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

Verified present in the store as of `9fec73c8`. Anything not on this list does not exist. The same
inventory, spelled as a function signature spells it and with the fields a body may read, is in
[§ The detection functions](#the-detection-functions); this table is the consumer's view of it.

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
| `energy_envelope`, `normalized_envelope` (dBFS, per sample), `continuity_trace` (per sample, `[0, 1]`), `gammatone` (40 ERB channels, 80–7800 Hz, hop 0.005 s), `spectrogram_wideband` (5 ms window) and `spectrogram_narrowband` (20 ms window), both at a 5 ms hop | per sample / per frame | extent proposal; event boundaries; burst spectra. **`continuity_trace` is a spectral-stationarity trace by construction** — cosine similarity between consecutive log-magnitude spectra (`spectral_continuity/api.py:10`) — so a stationarity test over it is a statistic, not a missing estimator. `gammatone` is **not** a privileged spectral instrument: it is an ERB rebinning of the same short-time spectrum the two spectrograms carry |
| `residual` — `energy_fraction`, `gain_db`, `bands`, `speech_coverage_fraction` | per file | the `airway.breath` gate |
| `silence` — YAMNet `Silence` projected onto the 0.96/0.48 grid | per window | the only presence/absence-of-sound gate in the graph |
| `level`, `disruptions_file`, `clip_amplitude`, `squim` assertions | per file / per span | effort, and quality covariates on every value |
| `speaker` entities + `diarization_interval` | per turn | who spoke. **Scoped to the lexical word hull only** — the whole-file picture is `{enhanced,residual}_diarization` |

**Absent, and no method here may assume them:** any VAD (no voice-activity model runs anywhere in
the graph — "speech activity" is word-timing- or YAMNet-`Silence`-derived, and
`detect_human_voice_activity_in_audios` relabels diarization segments rather than running a
detector, `tasks/voice_activity_detection/api.py:27`); per-span PPG; `span_ast`; a phonation-family
span producer; per-word ASR confidence on the consensus word; a DDK node.

**Whole-file diarization is no longer on that list.** `{enhanced,residual}_diarization` are shipped
derivatives — one measurement per stream in `diarization.streams`, `n_speakers` and the per-speaker
totals as attributes and the segment table as `derivatives/<stream>_diarization.npz`
(`preprocess.py:2617`, block list at `:2985`, default streams `[enhanced, residual]` at
`default.yaml:189`). The two streams' counts are never summed: `enhanced` says how many voices
survived enhancement, `residual` whether one was removed. SPEECH still runs its own pyannote pass
over the lexical word hull (`speech.py:642-646`) and is to read this derivative instead (owner,
2026-09-15); that read-swap is enumerated in
[`../20260915-preprocess-diarization/design.md`](../20260915-preprocess-diarization/design.md).
`dag.md:526-527`, which says `pyannote` appears nowhere in `nodes/preprocess.py`, is behind the
code.

## What is missing, consolidated

Read off the per-task requirement lists above, not off the rows: a derivative that two rows of one
family both want is one dependency, and a family that four rows touch is still one unit of work.
The unit here is a **task block** — the 31 blocks of the section above, covering the table's 39 rows
— and the ranking is by how many of them cannot be completed without the item.

Two categories, and the distinction decides who does the work:

- a **missing derivative** is a value no branch can read out of the store, whether because nothing
  in the tree computes it or because what computes it writes nowhere a branch looks;
- a **missing decision** is an instrument that is already in the store with no named statistic,
  window or operating point over it. Nothing is owed but a fit, in `data/` with a derivation.

A third table names what is neither: determinations with **no viable approach** in this design, which
no derivative and no fit will settle.

**Two deviation types are owed a row in the vocabulary**, which is neither of the three: `omission`
(a stimulus token nothing realised — read tasks, ~19,300 recordings) and `repeated_item` (a repeat
where the instruction forbids one — `animal-fluency` and the eight `random-item-generation`
categories that forbid it). `branch-conventions.md:123-142` is authoritative, carries nine types,
and its own rule is that a branch adding a type adds a row there.

### Missing derivative — ranked by task blocks it holds up

| what | blocks | recordings | state | which blocks |
| --- | --- | --- | --- | --- |
| `stimulus_alignment` (D1) — the consensus word stream aligned against the declared utterance, with a skip arc | **7 hard, 3 partial** | ~26,800 | absent from the tree; the aligner is not — `align_sources` (`consensus.py:276`) and `harmonize_transcripts` (`harmonize.py:522`) already do this for the ASR-against-ASR case | harvard/CAPE-V read text, passages, stroop, story-recall, free-speech v1's anti-pattern, CAPE-V's per-sentence extents for VOICE, productive-vocabulary's cue; partial on prolonged-vowel, loudness, open-response, all of which have a token-list fallback |
| `phonation_tracks.{hnr_db, rms_dbfs, cpps_db}` (D2) — three more columns on the grid the npz already writes | **7** | 5,113 sustained + 3,193 glides + 3,594 CAPE-V + 2,627 effort, and the 22,277 VOICE routes | in the tree, computed and thrown away: `hnr_track` (`phonation/api.py:108`) and `_rms_track` are called at `voice.py:267` and `:274` on every VOICE run that gets past `:235` — which is none, because the no-span early return fires first; `extract_cpp_descriptors` builds a per-frame `prominence` array at `praat_parselmouth.py:1030` and reduces it to three scalars at `:1035-1039` | prolonged-vowel, maximum-phonation-time ×2, glides, CAPE-V, loudness, hard-cough/voluntary-cough effort, routed-VOICE |
| a phonation subject VOICE can select | **4** | 5,113 + 22,277 | in the store and unreadable: `preprocess.py:1875-1891` writes amplitude spans with no `family` key; `voice.py:230` selects `family == "phonation"`, so the candidate list is empty before any test runs | prolonged-vowel, maximum-phonation-time, glides, routed-VOICE |
| a populated `expected_speech`, at the recording grain | **7, the same as D1** | ~26,800 | `AudioHints.expected_speech` reaches every branch (`audio_hints.py:152`) and nothing populates it: `runs/b2ai-v2/make_hints.py:381` writes `may_contain` and `metadata` from the filename token and never reads `stimulus_text`. On 36 of 48 families `stimulus_text` is empty too, so for those the expectation needs a human-read table that does not exist | D1's blocks. D1 without this is an aligner with nothing to align against |
| the trailing task index, where it carries a condition | **2** | 3,576 + 813 | in the BIDS stem and in `hints.metadata.task_token`; `task_family` collapses it (`families.py:134-144`) and every family-keyed rule loses it | `fivebreaths`' nose/mouth route, `maximum-phonation-time-v2`'s `-2` effort escalation |
| `band_profile` (D3) — content band and long-term spectrum on the un-resampled `recording` stream | **0 blocked, 7 unattributable** | ~13,000 airway | absent; `_rolloff_hz` (`audio_analysis/quality.py:156`) is the core and is a private function in a sibling workflow | every airway block, plus the hard-cough hygiene clause. It unblocks nothing — it makes a negative route result *attributable* to a measured band limit rather than recorded as "not yet fitted" |
| `transcript_repeat` as a store measurement | **0** | 2,265 | in `routing_analysis/features.py`, read only by the ruleset | `random-item-generation`, `animal-fluency`, `buttercup`. Listed for completeness: it is a counter over normalised consensus tokens, so the capability is arithmetic over word entities the store already holds |
| per-span PPG | **0** | — | `extract_ppg_segments` (`ppg.py:349`) is called from `routing_analysis` and `plotting`, never from a node | nothing above calls it. The row that used to want it — DDK sequence conformance — is served by `spectrogram_wideband` instead, at no model cost |

**Two entries that were on this list and are not any more.** Whole-file diarization **ships**:
`{enhanced,residual}_diarization` are real derivatives, one measurement per stream in
`diarization.streams` (`preprocess.py:2617`, `:2985`), carrying `n_speakers` and a segment npz. And
*Praat scalars over an extent* is superseded by D2 rather than owed separately — a track is
re-poolable over whatever extent a branch proposes; a scalar taken over PREPROCESS's own spans is
not, and those spans are amplitude/continuity/ASR/gap, which are the task extent for no family here.

### Missing decision — ranked by task blocks it holds up

Every item is a named statistic, a window, or an operating point over an array that is already in
the store. Each belongs in `data/` with a written derivation, **fitted against measured verdicts and
not against the declared families** — a cut fitted on declarations encodes which recordings the
protocol labelled, not which carry the pattern.

| what | blocks | recordings | the instrument that already exists |
| --- | --- | --- | --- |
| the event boundary: `p_smoothing_window_s`, `p_peak_prominence_db`, `p_trough_return_db`, `p_event_min_s` (A5/A6, and D3's onsets) | **12** | ~13,000 airway + 7,989 DDK | `energy_envelope` per sample with its global floor. The merging case is a series on one exhalation, visible as multiple maxima inside one span — not `spans.min_separation_ms`, which is 30 ms while volitional coughs are seconds apart |
| what makes a `gap` span off-task rather than ordinary silence | **all 31** | 62,547 | `span` `measure: "gap"`, `silence`, `energy_envelope`. Every body above calls `off_task()` and none of them can say where the line falls |
| the stationarity qualifier: `p_f0_spread_window_s`, `p_f0_spread_max_semitones`, `p_continuity_min`, `p_voiced_strength_min`, `p_voiced_fraction_min` (V1) | **5** | 5,113 + 22,277 | **`continuity_trace` is already a spectral-stationarity trace** — cosine similarity between consecutive log-magnitude spectra (`spectral_continuity/api.py:10`), written per sample (`preprocess.py:2549`); F0 and formant stationarity are statistics of `f0_hz` and `f1..f4_hz` at a 10 ms hop. An earlier draft of this document called this "the single most load-bearing gap"; it is load-bearing, and it is a fit, not an estimator |
| the DDK rate: `p_modulation_band_hz`, `p_rate_prominence_min` (D1/D2) | **4** | 7,989 | the modulation spectrum of `energy_envelope` over the train. **Not** Praat's `extract_speech_rate`, which is already running inside `praat_features` and whose two failure modes — a `min_dip` the fastest trains do not clear, and a silence tier built with `min_pause = 0.3 s`, longer than a whole DDK cycle — both bias the measurement in the direction of the quantity being measured |
| the omission cut: `p_omission_score_max` (S3) | **3** | ~19,300 | a skip-arc-free aligner assigns every stimulus word an interval, so an omission surfaces only as a low acoustic score (`branch-speech.md:115-121`). This is the decision that decides whether D1 needs a skip arc |
| the verbatim-overlap cut: `p_echo_ngram_n`, `p_echo_overlap_max`, `p_verbatim_overlap_max` | **2** | 4,623 | n-gram overlap between the source text and the consensus words. Arithmetic once D1 exists; the cut separates reading from quoting |
| the place decision: `p_burst_window_ms`, `p_place_centroid_bands_hz`, `p_place_margin` (D6) | **1** | 1,597 | `spectrogram_wideband`, 5 ms window at 5 ms hop — the classical resolution for a /p/,/t/,/k/ burst-spectrum contrast, all three of which sit inside 8 kHz |
| the dominant monotone segment: `p_monotone_tolerance_semitones`, `p_dominant_segment_min_fraction` (V3) | **1** | 3,193 | `phonation_tracks.f0_hz` over the span |
| the classifier label floors and the `label_top_k` rule (`windows.{yamnet,ast,hear}`) | **9** | ~13,000 airway | the raw scores are written whatever the configuration says; `labels` and `scores` are a decision over them. YAMNet's and HeAR's `default_threshold` ship at 0.2 and `label_top_k` at 4, both **owner-directed rather than fitted**; AST's floor is null (`default.yaml:91-107`). A label must be in the top four *and* clear its floor, so a cough label ranked fifth is dropped by a size, not by a score — which is why every airway body above reads `raw_scores` and applies its own `p_*_score_min`. **And a null `label_thresholds` kills a fold whose floor is set**: `load_label_membership` `require`s all three keys, so `yamnet_windows` and `hear_windows` are absent today even though their floors ship at 0.2 — measured against the packaged config, not inferred. Whether that is the intended reading of a null override is a config question, and it is why no body above reads a `*_windows` derivative |
| vocal tremor's band and statistic (V4a) | **2** | 5,113 | the 2–12 Hz band of the spectrum of `f0_hz` and of `energy_envelope`; the envelope's 40 Hz lowpass passes the whole band |
| the breath-coverage cut: `p_breath_coverage_min` | **2** | 2,487 | HeAR `Breathe` windows, 2.0 s non-overlapping |

### No viable approach — neither a derivative nor a fit will settle these

| what | blocks | recordings | why |
| --- | --- | --- | --- |
| nasal versus oral route (A7) | **3 declared contrasts, 7 families naming a route** | 8,416, of which 4,974 sit in a declared contrast | the discriminating band sits largely above the 8 kHz ceiling, and what remains below it is a spectral tilt confounded one-for-one with mouth-to-microphone geometry — which changes *with the route, by construction*, since a participant breathing through the mouth points it at the phone and one breathing through the nose does not. The 1,778 / 1,778 within-session split is perfectly balanced on route and perfectly confounded on source-to-microphone transfer. Report as *not separable by this design*, not as *not yet fitted*, and do not treat `fivebreaths` as validation-grade for A7 |
| absolute effort — was this maximal | **2** | 1,595 | `level` is peak dBFS, RMS dBFS and LUFS of an uncalibrated consumer recording with unknown microphone sensitivity, unknown distance and, on many handsets, AGC in the capture path. No SPL reference exists in the graph and none is recoverable after the fact. What survives is the level-invariant half — spectral balance — and the honest output is a measurement with its covariates. `loudness-v2` and `voluntary-cough` are unaffected: both carry a within-recording contrast |
| category membership of a produced word | **2** | 667 | a lexicon or a text embedding, one consumer, no waveform. Not a PREPROCESS derivative, and `text/tasks/embeddings_extraction` is not wired into triage |
| whether definitional speech is a definition *of* its cue | **1** | 2,910 | the same, and 78 recordings carry no cue at all |
| any overlap measure for `cinderella-story` | **1** | 258 | the source is a physical storybook; `stimulus_text` is empty on all 258. Presence and extent is the ceiling, not a first step |
| whether a sub-second recording is a verdict or an acquisition failure | **all 31** | 2,020 sidecars | every expected pattern is absent in such a file, so every function above correctly returns *not done* and uninformatively. Whether QUALITY should absorb it is a protocol question |

## What could not be settled

- **Which verb carries `off_task_extent`** — `trim` per `design.md:569`, `deviate` per
  `branch-conventions.md:108` and `branch-airway.md:350`. Both documents are current. This document
  uses `trim` for `task_extent` and leaves the deviation's verb to whoever reconciles the two.
- **Whether `trim` has an emitter at all.** No branch document claims it, and it is in neither the
  nine implementation pieces (`design.md:1114-1165`) nor the unresolved list (`:1166-1229`). Since
  `task_extent` is the direct answer to the owner's question, this is the gap that most needs an
  owner.
- **Whether `maximum-phonation-time`'s "we will repeat this task 3 times" means three recordings or
  three attempts within one.** The corpus has `-1`, `-2`, `-3` as separate recordings carrying the
  identical instruction, 899 apiece, which argues for three recordings — but `repeat_attempt` is
  already a declared VOICE deviation, so the two readings produce opposite findings on the same
  audio. Settling it needs the protocol, not more counting.
- **What a sub-second recording means.** **2,020 of the 62,810 triaged recording sidecars**
  declare a `recording_duration` under one second — none declares 0, and the shortest in the
  corpus is 0.1 s — concentrated in `harvard-sentences-list` (1,135 of 13,960),
  `respiration-and-cough-fivebreaths` (272), `-threequickbreaths` (134), `-breath` (132) and
  `-cough` (124). Every expected pattern is absent in such a file, so the pattern match answers "was
  the task done" with *no* — correctly, but uninformatively. Whether that is a verdict or an
  acquisition failure QUALITY should absorb is not settled here, and it is the single largest
  category of "pattern absent" the corpus will produce.
- **Why 1,072 `free-speech` recordings do not link to an acoustictask sidecar.** Their
  `recording_acoustic_task_id` resolves to nothing in the session, and no other family has a single
  unlinked recording. It costs nothing today, because the recording grain is the one to read — but
  it is the only structural break in the sidecar graph and nobody has explained it.
- **Whether `picture-description` and `picture-description-option1` are one family.** Their
  instructions are byte-identical and both have empty `stimulus_text`; the only declared difference
  is the image, which is not in the sidecar as anything a method can read. No method in the table
  above can tell them apart, so declaring them separately buys nothing and may mislead.
- **Whether a label on an `amplitude` span makes it refinable**, which decides whether the
  prolonged-vowel decomposition is a `refine` or a `propose`. Unresolved at
  `branch-conventions.md:61-71`, `design.md:1174-1181`, `branch-voice.md:92-97`.
- **`loudness` / `loudness-v2` family-set membership.** `families.py:41-42` puts them in
  `LEXICAL_SPEECH`; the sidecars declare `speech_type: "non-lexical"` on all 897 and all 705, and
  the only word is `hey`. Resolve in `families.py`, not in a branch document.
- **Whether `high-to-low` should stay a separate family.** Its instruction is byte-identical to
  `glides-high-to-low`'s on all 43 recordings, so it is an alias by any measurable test. Whether
  `families.py` folds it is a declaration question, and the table treats the two as one task.

## Appendix — the instruction per family, verbatim

Read from `<stem>_recording-metadata.json` under
`/orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04`, 2026-09-15, over
all 2,005 subjects. **All 48 declared families appear**; none is inferred from a sibling. The
bracketed number is how many English recording *sidecars* carry that exact string; the family's own
count is the corpus profile's, which counts *wav*. A few sidecars have no wav — 426 tree-wide, 255
of them `harvard-sentences-list` — so the bracketed number can exceed the family count by one or
two, as it does for `animal-fluency`. Within a family the English `instructions` is byte-identical
across subjects except where a family is listed with more than one, so each entry is exact rather
than a sample.
Spanish (`es-419`, 1,417 recordings tree-wide) is omitted: 41 families carry exactly one Spanish
string, six have no Spanish recordings, `random-item-generation-v2` carries five, and for the
airway families the string is the generic acoustictask text that names neither count nor route.

### VOICE_ELICITING

- **`maximum-phonation-time`** (2,696) — *"This task helps us analyze the way your breathing is
  connected to your voice. Take a very deep breath and hold out “ah” for as long as possible until
  you completely run out of air. We will repeat this task 3 times."* [2,682 English].
  `stimulus_text` empty on every recording.
- **`prolonged-vowel`** (1,604) — *"This task helps us analyze features in your voice. Please
  press the play button to listen to the demonstration on how to complete the task. Then, tap the
  record button and imitate the speaker by repeating the sentence “1, 2, 3 aah” in your normal
  voice. Please hold the sound “aah” until the timer runs out."* [1,575 English]. `stimulus_text`
  empty on every recording.
- **`glides-low-to-high`** (1,596) — *"Please watch the video to help you understand the next task
  and imitate what the speaker does. When you're ready, press the record button and use the sound
  “ee” to gradually move from your lowest note to your highest."* [1,568 English]. `stimulus_text`
  empty on every recording.
- **`glides-high-to-low`** (1,554) — *"Next, please watch this video for the following task and
  imitate what the speaker does. When you're ready, press the record button again and use the
  sound “ee” to gradually move from your highest note to your lowest note."* [1,525 English].
  `stimulus_text` empty on every recording.
- **`maximum-phonation-time-v2`** (813) — **2 distinct English instructions.** `stimulus_text`
  empty on every recording.
  - *"This task helps us analyze the way your breathing is connected to your voice. Take a very
    deep breath, tap on the record button, and hold out “ah” for as long as possible until you
    completely run out of air. Try to hold it until the progress bar reaches the end of the
    screen."* [676]
  - *"This task helps us analyze the way your breathing is connected to your voice. Take a very
    deep breath, tap on the record button, and hold out “ah” for as long as possible until you
    completely run out of air. Try to hold it until the progress bar reaches the end of the
    screen. Now try to hold out “ah” for even longer."* [108]
- **`high-to-low`** (43) — *"Next, please watch this video for the following task and imitate what
  the speaker does. When you're ready, press the record button again and use the sound “ee” to
  gradually move from your highest note to your lowest note."* [43 English]. `stimulus_text` empty
  on every recording.

### SYLLABLE_REPETITION

- **`diadochokinesis-buttercup`** (896) — *"This task helps us analyze the ease and precision of
  speech sound productions. Now repeat the word /buttercup/ as fast as possible 10 times."* [891
  English]. `stimulus_text` empty on every recording.
- **`diadochokinesis-ta`** (896) — *"This task helps us analyze the ease and precision of speech
  sound productions. Record yourself repeating the syllable /TA/ as fast as possible 10 times."*
  [891 English]. `stimulus_text` empty on every recording.
- **`diadochokinesis-pataka`** (896) — *"This task helps us analyze the ease and precision of
  speech sound productions. Now repeat the word /Pataka/ as fast as possible 10 times."* [891
  English]. `stimulus_text` empty on every recording.
- **`diadochokinesis-pa`** (896) — *"This task helps us analyze the ease and precision of speech
  sound productions. Record yourself repeating the syllable /PA/ as fast as possible 10 times."*
  [891 English]. `stimulus_text` empty on every recording.
- **`diadochokinesis-ka`** (896) — *"This task helps us analyze the ease and precision of speech
  sound productions. Record yourself repeating the syllable /KA/ as fast as possible 10 times."*
  [891 English]. `stimulus_text` empty on every recording.
- **`diadochokinesis-v2-puh`** (702) — *"This task helps us analyze the ease and precision of
  speech sound productions. Please press the play button to listen to the demonstration on how to
  complete the task. Then, tap the record button and imitate the speaker by repeating the syllable
  'puh' as quickly and consistently as possible until the timer runs out - 'puhpuhpuhpuhpuhpuh'"*
  [678 English]. `stimulus_text` empty on every recording.
- **`diadochokinesis-v2-tuh`** (702) — *"This task helps us analyze the ease and precision of
  speech sound productions. Please press the play button to listen to the demonstration on how to
  complete the task. Then, tap the record button and imitate the speaker by repeating the syllable
  'tuh' as quickly and consistently as possible until the timer runs out - 'tuhtuhtuhtuhtuhtuh'"*
  [678 English]. `stimulus_text` empty on every recording.
- **`diadochokinesis-v2-buttercup`** (702) — *"This task helps us analyze the ease and precision
  of speech sound productions. Please press the play button to listen to the demonstration on how
  to complete the task. Then, tap the record button and imitate the speaker by repeating the word
  'buttercup' as quickly and consistently as possible until the timer runs out."* [678 English].
  `stimulus_text` empty on every recording.
- **`diadochokinesis-v2-kuh`** (702) — *"This task helps us analyze the ease and precision of
  speech sound productions. Please press the play button to listen to the demonstration on how to
  complete the task. Then, tap the record button and imitate the speaker by repeating the syllable
  'kuh' as quickly and consistently as possible until the timer runs out - 'kuhkuhkuhkuhkuhkuh'"*
  [678 English]. `stimulus_text` empty on every recording.
- **`diadochokinesis-v2-puhtuhkuh`** (701) — *"This task helps us analyze the ease and precision
  of speech sound productions. Please press the play button to listen to the demonstration on how
  to complete the task. Then, tap the record button and imitate the speaker by repeating the
  syllables 'puhtuhkuh' as quickly and consistently as possible until the timer runs out -
  'puhtuhkuhpuhtuhkuh'"* [678 English]. `stimulus_text` empty on every recording.

### AIRWAY_ELICITING

- **`respiration-and-cough-fivebreaths`** (3,576) — **2 distinct English instructions.**
  `stimulus_text` empty on every recording.
  - *"After pressing on record, take 5 big breaths in and out through your nose with your mouth
    closed."* [1,778]
  - *"After pressing on record, take 5 big breaths in and out through your mouth."* [1,778]
- **`respiration-and-cough-cough`** (1,788) — *"After pressing record, cough 5 times"* [1,778
  English]. `stimulus_text` empty on every recording.
- **`respiration-and-cough-breath`** (1,788) — *"First, let's hear you breathe comfortably for 30
  seconds"* [1,778 English]. `stimulus_text` empty on every recording.
- **`respiration-and-cough-threequickbreaths`** (1,718) — *"After pressing on record, exhale, then
  inhale quickly through your mouth, as if you are trying to catch your breath. Record 3 of these
  breaths in a single recording."* [1,708 English]. `stimulus_text` empty on every recording.
- **`respiration-and-cough-v2-threebreaths`** (699) — *"After pressing on record, exhale normally,
  then inhale quickly through your mouth, as if you are trying to catch your breath. Record 3 of
  these breaths in a single recording. Tap stop when finished."* [675 English]. `stimulus_text`
  empty on every recording.
- **`respiration-and-cough-v2-breath`** (699) — *"After pressing on record, breathe comfortably
  through your mouth for 20 seconds, until the timer runs out."* [675 English]. `stimulus_text`
  empty on every recording.
- **`respiration-and-cough-v2-threebreathsnose`** (699) — *"After pressing on record, take 3 deep
  breaths in and out through your nose with your mouth closed. Tap stop when finished."* [675
  English]. `stimulus_text` empty on every recording.
- **`respiration-and-cough-v2-threebreathsmouth`** (699) — *"After pressing on record, take 3 deep
  breaths in and out through your mouth. Tap stop when finished."* [675 English]. `stimulus_text`
  empty on every recording.
- **`respiration-and-cough-v2-hardcough`** (698) — *"Please do not cover your mouth or place your
  hand between your mouth and the microphone during recording. Breathe normally, then when you are
  ready, tap the record button below and cough HARD as if something were stuck in your throat. Tap
  stop when finished."* [675 English]. `stimulus_text` empty on every recording.
- **`voluntary-cough`** (327) — *"You are being asked to record coughing sounds. Breathe normally,
  then when you are ready, push record and cough HARD as if something were stuck in your throat.
  Then breathe normally again. Complete this task 3 times in a single recording."* [324 English].
  `stimulus_text` empty on every recording.
- **`breath-sounds`** (326) — *"You are being asked to record your breathing sounds. Please relax
  for 60 seconds until the task starts. Take three deep breaths in a row in and out of the
  mouth."* [323 English]. `stimulus_text` empty on every recording.

### LEXICAL_SPEECH

- **`harvard-sentences-list`** (13,705) — *"Please read the following sentences out loud in your
  typical voice."* [13,480 English]. `stimulus_text`: **1,060** distinct — one Harvard sentence
  per recording.
- **`free-speech`** (3,074) — *"This section is meant to hear you speak freely by answering an
  open-ended question. Please answer the following questions and record your answer. Keep talking
  until the time stops. Do not record yourself reading the prompt and avoid any information that
  could identify an individual."* [3,056 English]. `stimulus_text`: **4** distinct.
- **`productive-vocabulary`** (2,910) — *"Below you will be provided with a series of words. You
  may or may not know the word. If you know the word, please provide a definition of the word.
  After you have defined the word, press 'Next Word'. If you do NOT know the word, press 'I don't
  know this word, next', and you will be provided a new word. Once 6 words are defined, the 'Done'
  button appears."* [2,880 English]. `stimulus_text`: **204** distinct, 78 carry none.
- **`cape-v-sentences`** (2,370) — *"Please read the following sentences out loud in your typical
  voice."* [2,280 English]. `stimulus_text`: **12** distinct.
- **`free-speech-v2`** (2,120) — *"This section is meant to hear you speak freely by answering an
  open-ended question. We'd like to get a recording of your natural speech. Please answer the
  following questions as though you were having a conversation. We'll record about 30 seconds of
  your response."* [2,049 English]. `stimulus_text`: **6** distinct.
- **`cape-v-sentences-v2`** (1,224) — *"Please read the following sentences out loud in your
  typical voice."* [1,224 English]. `stimulus_text`: **6** distinct.
- **`loudness`** (897) — *"This helps us to determine the loudness of the voice. When you are
  ready, press on the record button and shout “hey” as loud as possible 3 times in a single
  recording."* [892 English]. `stimulus_text` empty on every recording.
- **`rainbow-passage`** (897) — *"This task helps us evaluate how you use breathing to support
  your voice. Please read the following passage out loud in your typical voice."* [893 English].
  `stimulus_text`: one text of 338 characters, shared by all.
- **`story-recall`** (889) — *"You are given a text. Read the text so you familiarize yourself
  with it. You have up to 5 minutes to read it as many times as you want. When you are ready, you
  will be asked to recall the story. This can be in your own words."* [884 English].
  `stimulus_text`: **2** distinct — the 717-character grandfather passage on the 884 English
  recordings, and on the 5 Spanish ones a 1,219-character Spanish text that is
  `story-recall-v2`'s **frog** story, not v1's.
- **`picture-description`** (889) — *"Tell me everything you see going on in this picture."* [884
  English]. `stimulus_text` empty on every recording.
- **`loudness-v2`** (705) — *"This helps us to determine the loudness of the voice. When you're
  ready, press the record button and say “hey” in your normal voice. Then, shout “hey” as loud as
  you can. Try to reach the target line on the screen. Follow the prompts on the screen to
  continue."* [681 English]. `stimulus_text` empty on every recording.
- **`story-recall-v2`** (660) — *"Below you will be presented with a story about a boy and a frog.
  You may click forward to see the next image in the series. Feel free to move back and forth as
  much as you like until you are comfortable with the story. You will have up to 5 minutes to view
  the story, but you are not required to use the full 5 minutes. Once the time is up, please
  retell the story in as much detail as possible. When you are ready, please click the record
  button below. Once you click the record button, the story will disappear."* [637 English].
  `stimulus_text`: **2** distinct — the 1,083-character frog story on the 637 English recordings
  and its 1,219-character Spanish translation on 23.
- **`caterpillar-passage`** (597) — *"This is a passage that contains speech sounds in English by
  sound frequency to test your ability to produce speech sounds. Please read the following passage
  out loud in your typical voice."* [582 English]. `stimulus_text`: **2** distinct.
- **`word-color-stroop`** (472) — *"This exercise asks you to name out loud the color in which a
  word is displayed. Sometimes the word will be a color word: do not read the word aloud, just
  state the color in which it's displayed. For example, if you see the word "brown", and its
  letters are displayed in blue, you will say the word "blue". Please answer as quickly as
  possible. Time limit = 5 seconds/item, 15 random words. Total time 75s. Tap on the red circle
  below to start recording. The recording will continue through all tasks and automatically stop
  at the end."* [467 English]. `stimulus_text`: **472** distinct.
- **`picture-description-option1`** (373) — *"Tell me everything you see going on in this
  picture."* [364 English]. `stimulus_text` empty on every recording.
- **`picture-description-option2`** (329) — *"Describe everything that is happening in the picture
  (as though describing it for the blind), trying to use complete sentences."* [314 English].
  `stimulus_text` empty on every recording.
- **`random-item-generation`** (265) — **10 distinct English instructions.** `stimulus_text` empty
  on every recording.
  - *"Say as many items from the following category as you can. Do not repeat any item. Your goal
    is to list as many as possible. The selection will appear when you start recording. The
    recording will automatically stop at the end. Category: <X>."*
    where `<X>` is one of `Jobs` [34], `City names` [32], `Drinks` [32], `English words starting
    with 't'` [31], `First names` [25], `Animals` [22], `Fruits` [21], `Country names` [20]
  - *"You will have to speak a series of (i) random letters or numbers (repetition allowed), or
    (ii) items from a given category e.g., cities, animals, etc. (repetition not allowed), with
    the goal of maximizing the number of items. The selection will appear when you start
    recording. The recording will automatically stop at the end. Category: <X>."*
    where `<X>` is one of `Letters` [24], `Numbers` [24]
- **`cinderella-story`** (258) — *"You'll receive a hard copy of the Cinderella storybook from our
  study team to refresh your memory of the story. When ready, click the 'Record' button below to
  begin narrating the story. Once you've finished, tap on 'Stop Recording'."* [258 English].
  `stimulus_text` empty on every recording.
- **`random-item-generation-v2`** (207) — **10 distinct English instructions.** `stimulus_text`
  empty on every recording.
  - *"Say as many items from the following category as you can. Do not repeat any item. Your goal
    is to list as many as possible. The selection will appear when you start recording. The
    recording will automatically stop at the end. Category: <X>."*
    where `<X>` is one of `City names` [23], `Drinks` [19], `First names` [18], `Country names`
    [15], `Fruits` [14], `English words starting with 't'` [13], `Animals` [12], `Jobs` [12]
  - *"You will have to speak a series of (i) random letters or numbers (repetition allowed), or
    (ii) items from a given category e.g., cities, animals, etc. (repetition not allowed), with
    the goal of maximizing the number of items. The selection will appear when you start
    recording. The recording will automatically stop at the end. Category: <X>."*
    where `<X>` is one of `Numbers` [39], `Letters` [38]
- **`open-response-questions`** (199) — *"Please answer the following questions and record your
  answer."* [199 English]. `stimulus_text`: one text of 447 characters, shared by all.
- **`animal-fluency`** (195) — *"For this task, you say as many animals as you can, while avoiding
  repeating the same ones. There is a 1 minute timer on this, and we will ask that you continue
  trying until the time is up."* [196 English]. `stimulus_text` empty on every recording.
