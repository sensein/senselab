# Expected patterns — what each task's instruction asks for, and what would match it

The owner's question: *"is there a list of what algorithm/method/heuristic is used to determine
whether a task was done and what the start end spans are for a task."*

**There is not.** This document is that list.

It is a **method inventory**, not a threshold fit. No number in it is fitted. Where a method needs a
boundary it says what would fit it and marks it owed, per the project rule that a threshold lives in
`data/` with a written derivation and never as a code literal.

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

### VOICE

| branch | task | expected pattern | detection approach |
| --- | --- | --- | --- |
| VOICE | `prolonged-vowel` (1,604) | **L→A.** L: the ordered tokens `one two three`, once. A: one continuous voiced production of a single vowel /a/, held to the timer, F0 holding rather than moving. The only voice family in the corpus that is not purely non-lexical, despite `speech_type: non-lexical` | A's extent: `spans` `measure: "amplitude"` (`peak_over_floor_db`); qualified by `phonation_tracks` `f0_hz` + `strength`. Separating A from L is free on the lexical side — consensus `word` entities carry extents, so the vowel is the voiced production under no lexical word. **Owed a measurement** for the acoustic separator: F0 / formant stationarity and spectral flux, which V1 requires and which do not exist (`branch-voice.md:197-210`) |
| VOICE | `maximum-phonation-time` (2,696) | **A only.** One held /a/ on one breath, to exhaustion. v1's instruction places the deep inhale **before the record tap is mentioned**, so an audible inhale may be inside the file — a second acoustic pattern that is expected but is not the measurement | As above. Duration of the qualified extent is V2's maximum phonation time. **Owed a measurement** (stationarity) and **owed a code change** (VOICE cannot select the amplitude spans at all — see the worked example). The inhale would be an AIRWAY `label`, not an `off_task_extent` |
| VOICE | `maximum-phonation-time-v2` (813) | **A only**, and cleaner: the instruction puts the inhale explicitly *before* the record tap, so no inhale is expected in the file. A behavioural difference from v1, not a wording difference. **And the index carries a second condition**: `-2` (108 English recordings) appends *"Now try to hold out "ah" for even longer."*, so `-1` → `-2` is a within-subject maximum-duration contrast | Same. Two contrasts, both free: the v1/v2 one is the only within-corpus control for whether the inhale is captured, and the `-1`/`-2` one is the only within-subject phonation-duration contrast in the corpus. The second is unevenly available — 700 recordings carry `-1` and only 113 carry `-2` — so it is a paired sub-sample, not a corpus-wide design. It also needs the trailing index, which `task_family` discards: **owed a code change**, the same one `fivebreaths` needs |
| VOICE | `glides-low-to-high` (1,596), `glides-high-to-low` (1,554) | **A only.** One continuous voiced production of /i/ whose F0 sweeps monotonically across the range, in the declared direction. **Not steady** — the opposite of the sustained pattern | `phonation_tracks` `f0_hz` over the amplitude-span extent; the dominant monotone segment and its sign. `sweep_direction_mismatch` is the declared deviation (`branch-conventions.md:137`). **Implementable today** for the track; **owed a cut** for what counts as the dominant monotone segment (V3) |
| VOICE | `high-to-low` (43) | **identical to `glides-high-to-low`** — the instruction string is byte-for-byte the same on all 43 recordings, `stimulus_text` empty on all 43 | **Settled**: this is a naming alias, not a separate task, and no separate method is owed. Route it and measure it as `glides-high-to-low`. Whether `families.py` should fold the two is a `families.py` question, not a branch one |
| VOICE | `loudness` (897) | **L&A, counted.** The token `hey`, **three times**, each at maximal effort | Consensus words give the token and its three extents; `level` (`peak_dbfs`, `rms_dbfs`, `lufs`) and `energy_envelope` give the effort per extent. `expected_event_count: 3` as a `counts` entry. **Implementable today** |
| VOICE | `loudness-v2` (705) | **L→A, contrastive.** `hey` at normal effort **→** `hey` shouted. The measurement is the *within-recording* contrast and needs no norm | Same measurements, differenced across the two extents. **Implementable today**, and the cheapest effort measure in the corpus. V6 is explicit that v1 and v2 are **not one measurement** ([`branch-voice.md:736`](branch-voice.md)) |
| VOICE | `cape-v-sentences` (2,370), `-v2` (1,224) | **A riding on SPEECH's L.** Six sentences each loading a different phonatory condition; the voice-quality measurement is per sentence and pooling discards the instrument's design | Sentence boundaries fall out of SPEECH's S3 alignment for free. **Owed a code change** — they must reach VOICE as selectable spans. Nothing output may be presented as a CAPE-V score |
| VOICE | any AIRWAY- or SPEECH-declared family | **no expected pattern.** VOICE routed 22,277 recordings against 8,306 declaring a voice family | The branch does its best on what it was handed: `label`, `refine` or `contest` on phonation evidence, and it concludes on its own question ([`design.md:486-507`](../20260913-branch-contract-and-hints/design.md)). Connected speech passes voiced-fraction, F0-availability and interruption tests, so **the stationarity qualifiers are what stop V4 computing perturbation over consonants and pauses** — the same owed measurement, here load-bearing over 22,277 recordings |

### SPEECH

| branch | task | expected pattern | detection approach |
| --- | --- | --- | --- |
| SPEECH | `harvard-sentences-list` (13,705), `cape-v-sentences` (2,370), `-v2` (1,224) | **L only**, ordered and fully specified: the words of that recording's `stimulus_text`, in order, once. One sentence per recording — the pattern is per recording, never per family. Measured: **1,060** distinct Harvard sentences, and CAPE-V v1 and v2 share only three of their six sentences. All three families carry the identical instruction, *"Please read the following sentences out loud in your typical voice."* | Forced alignment of the consensus transcript against `hint.expected_speech`, plus a text-level diff. `align_transcriptions` exists (`tasks/forced_alignment/`). **Implementable today** for substitutions and insertions — enumerated differences, not scores. **Owed a cut** for omissions: a skip-arc-free aligner assigns every stimulus word an interval, so an omission surfaces only as a low acoustic score ([`branch-speech.md:115-121`](branch-speech.md)). Carry the consensus `agreement` on every mismatch |
| SPEECH | `rainbow-passage` (897), `caterpillar-passage` (597) | **L only**, ordered, one long passage. Both grains agree on both fields, on all 897 and all 597 — each is its own acoustic task, so the one-to-many collapse that breaks the other families does not arise. `stimulus_text` is one 338-character passage shared by all 898 rainbow recordings, and one of 1,035 characters shared by the 582 English caterpillar recordings (the 15 Spanish ones carry a 1,079-character Spanish passage) | Same as above, at passage length. `[breath]` tokens inside the passage are **not** `filler` — they are how S4 measures breath-group structure |
| SPEECH | `word-color-stroop` (472) | **L only, ordered, and not the displayed words.** The instruction says *name the colour, do not read the word*. `stimulus_text` is the 15-item colour sequence, i.e. the expected **answer** sequence — **472 distinct sequences over 472 recordings**, one per recording and never shared, so the pattern is maximally per-recording. Declared 75 s; measured median 75.8 s, with 463 of 472 falling between 74 and 77 s | Ordered alignment against `expected_speech` built from that sequence. **Implementable today.** `filler` must **not** be a deviation here — hesitation and self-correction are the task's dependent variable ([`branch-speech.md:130-134`](branch-speech.md)) |
| SPEECH | `free-speech` (3,074) | **L, unordered, with a negative pattern.** No target text. The v1 instruction says *"Do not record yourself reading the prompt"*, so `stimulus_text` is an **anti-pattern**: the question appearing verbatim in the transcript is the deviation. Four questions, per index — three asked 898 times each, one 380 | Alignment of the transcript against `expected_speech`, read inverted — high verbatim overlap is the finding. Consensus words and the aligner exist; **owed a cut** for how much overlap is reading rather than echoing a question word. **This row cannot use the family grain at all**: the acoustictask JSON carries one frozen prompt for all 912 sidecars, which is right on 380 recordings and wrong on 2,694 |
| SPEECH | `free-speech-v2` (2,120) | **L, unordered, and the negative pattern is gone.** v2's instruction drops *"do not record yourself reading the prompt"* entirely and asks instead to answer *"as though you were having a conversation"*. So verbatim echo is **not** declared a deviation here, and treating v1 and v2 alike would invent one. Six questions, three English (683 each) and three Spanish (24 each) | Same alignment, but the inverted reading applies only to v1. For v2 the measurement is presence and extent of connected speech against a ~30 s declared response. The acoustictask prompt matches **none** of the six, on all 707 sidecars — the family grain is wrong on every one of the 2,120 |
| SPEECH | `story-recall` (889), `story-recall-v2` (660) | **L, unordered, partly negative.** Recall *in the participant's own words*. `stimulus_text` is the source story — the 717-character grandfather passage for v1, the 1,083-character frog story for v2 — so semantic coverage is expected and **verbatim** reproduction is the deviation, the participant having read rather than recalled. The five Spanish v1 recordings carry the **frog** story, i.e. v2's source under v1's family name, so even here the source must be read per recording | Consensus words against `expected_speech`; n-gram overlap separates recall from reading. **Owed a measurement** — no lexical-overlap or semantic-coverage statistic exists in the graph and `text/tasks/embeddings_extraction` is not wired into triage |
| SPEECH | `cinderella-story` (258) | **L, unordered, and nothing machine-readable.** `stimulus_text` is empty on all 258; the source is a physical storybook handed to the participant, so no overlap measure is even definable here | Presence and extent only. The v1/v2 `story-recall` rows' n-gram method **does not transfer**: there is no text to overlap against. Median 93.9 s, 18 of 258 under a second |
| SPEECH | `productive-vocabulary` (2,910) | **L, weakly specified, per recording.** `stimulus_text` is one cue word per recording — **204 distinct cues** across the corpus, and **78 recordings carry no cue at all**. Expected is definitional speech *about* the cue, not the cue itself | Presence and extent of lexical content is free. Anything beyond presence — whether a definition is *of* its cue — is **owed a measurement**. 204 cues under one family name, so the family grain carries nothing, and on the 78 empty ones neither grain does |
| SPEECH | `picture-description` (889), `-option1` (373), `-option2` (329) | **L only, unspecified.** Connected speech, no target text, `stimulus_text` empty on all 1,591 (the stimulus is an image URL in the sidecar). `picture-description` and `-option1` carry a **byte-identical** instruction — *"Tell me everything you see going on in this picture."* — and differ only by image; `-option2` asks for complete sentences *"as though describing it for the blind"* | Presence and extent: consensus `word` entities and `spans` `measure: "asr"` runs. S4's connected-speech measures — rate, pause structure, breath groups — are the real content and are **not built** ([`branch-speech.md:157`](branch-speech.md)). The v1/option1 instruction identity means **no method may distinguish them**; any difference found is the image's |
| SPEECH | `open-response-questions` (199) | **L only, unspecified.** *"Please answer the following questions and record your answer."* One `stimulus_text`, a 447-character prompt shared by all 199. Median 30 s | Presence and extent, as above. The single shared prompt makes this the one connected-speech family where the same `expected_speech` is legitimately family-scoped — every other one is per recording or absent |
| SPEECH | `animal-fluency` (195) | **L, unordered, category-bound, with a repetition anti-pattern.** *"say as many animals as you can, while avoiding repeating the same ones. There is a 1 minute timer"*. `stimulus_text` empty on all 195; the category lives only in `instructions`. Median duration 60 s, matching the declared timer | Consensus words plus `transcript_repeat` for the anti-pattern, as for `random-item-generation`. Category membership (is this an animal?) is **owed a measurement**. `declared_duration_s` against measured is free |
| SPEECH | `random-item-generation` (265), `-v2` (207) | **L, unordered, category-bound — and the negative constraint is category-conditional.** Ten categories per family, one per recording, living only in `instructions`. Eight of the ten (`Animals`, `City names`, `Country names`, `Drinks`, `First names`, `Fruits`, `Jobs`, `English words starting with 't'`) say *"Do not repeat any item"*. The other two, **`Letters` and `Numbers`, say the opposite** — *"random letters or numbers (**repetition allowed**)"* — 48 of 265 v1 recordings and 77 of 203 English v2 recordings | Consensus words plus `transcript_repeat` — largest repeat count of any normalised token, already computed for `ddk.lexical_repetition` (`default.yaml:302`). A repeat is a deviation **only on the eight categories that forbid it**; firing it on `Letters` or `Numbers` inverts the instruction, and a family-scoped rule would do exactly that on roughly a fifth of the recordings. **Owed a code change**: `transcript_repeat` lives in `routing_analysis/features.py`, is read only by the ruleset, and is not a store measurement — and the rule needs the per-recording category, which no grain above the recording carries. Category membership is **owed a measurement** |
| SPEECH | `loudness` (897), `loudness-v2` (705) | **L:** the token `hey` | Consensus words. The measurement wanted is VOICE's V6; SPEECH contributes the token's extent. These two are in `LEXICAL_SPEECH` (`families.py:41-42`) while the protocol calls them `speech_type: "non-lexical"` — a `families.py` discrepancy |
| SPEECH | every `SYLLABLE_REPETITION` family (7,989) | **L: none expected.** `/pa/` is not lexical and ASR mostly declines it. These are **positives** for SPEECH's reference set as of 2026-09-15 (`reference_family_set.SPEECH: speech` = `lexical_speech \| syllable_repetition`, `default.yaml:241`), so near-zero lexical content is the correct observation, not a miss. `diadochokinesis-buttercup` is the exception — see the DDK table | Consensus words; expected count is zero or near it. `speech.lexical >= 2` firing here is over-routing on function-word artefacts: **71.3%** of that gate's apparent false positives are DDK families ([`dag.md:195-196`](dag.md)) |
| SPEECH | `prolonged-vowel` (1,604) | **L:** the ordered tokens `one two three`, once. The count-in is **prescribed**, not incidental: the instruction asks the participant to repeat *"1, 2, 3 aah"*, on all 1,575 English recordings | Consensus words matched against `expected_speech = ["one","two","three"]`. Measured: **938 of 1,258** transcripts open with `One two three` ([`dag.md:185`](dag.md)); the control `maximum-phonation-time` fires `speech.intrusion` at **3.2%** ([`family-taxonomy-ruleset.md:320-323`](family-taxonomy-ruleset.md)). **Implementable today** — the cheapest row in the document |
| SPEECH | any AIRWAY-declared family | **L: none expected.** Any lexical content is off-task by construction | Consensus words with their extents; each contaminating extent is one `off_task_extent`. Measured examples are examiner speech — *"I'll have you do that one more time. [breath]"*, *"So just breathe."* ([`../20260910-taxonomy-routing-evidence/measurements.md:177-182`](../20260910-taxonomy-routing-evidence/measurements.md)). **Implementable today**; AIRWAY owns the deviation ([`branch-airway.md:164-165`](branch-airway.md)) |

### AIRWAY

| branch | task | expected pattern | detection approach |
| --- | --- | --- | --- |
| AIRWAY | `respiration-and-cough-cough` (1,788) | **A, counted.** Five discrete forced expulsive events — *"After pressing record, cough 5 times"*. `expected_event_count: 5` | Onset/offset per event from `energy_envelope`, corroborated by `span_hear` `Cough` / `span_yamnet` cough-subtree labels joined on `span_id`. **Owed a code change** — AIRWAY counts *label-carrying spans*, not events: `by_label` increments once per (span, label) pair (`airway.py:280`), so a 4 s span holding three coughs counts **1**, and `spans.min_separation_ms` merges adjacent coughs upstream. **Owed a measurement** for a true event boundary (A6) |
| AIRWAY | `respiration-and-cough-v2-hardcough` (698) | **A, one event, maximal effort.** *"cough HARD as if something were stuck in your throat"*. No count stated, and the instruction adds a **recording-hygiene** clause found nowhere else — *"do not cover your mouth or place your hand between your mouth and the microphone"* | As above, minus the count. Unlike `loudness-v2` there is no within-recording contrast to read effort against, so "hard" is **owed a measurement**. The hygiene clause is a QUALITY expectation, not an AIRWAY one: an occluded microphone is a level and spectral-tilt finding |
| AIRWAY | `voluntary-cough` (327) | **A, counted (3), maximal effort, with breathing between.** Same *"cough HARD"* wording as `v2-hardcough` but *"Complete this task **3 times** in a single recording"*, with *"then breathe normally again"* between. `expected_event_count: 3`, and the inter-cough breathing is an **expected** pattern, not off-task | Counting is A5/A6 as for the 5-cough row. The interleaved breaths make this the one cough family where a cough detector alone is insufficient: the expected pattern is an alternation, and material between coughs must be matched as breath rather than scored as `off_task_extent`. Median 13.8 s against `v2-hardcough`'s 4.6 s, consistent with three cycles |
| AIRWAY | `respiration-and-cough-fivebreaths` (3,576) | **A, counted (5) and routed — and the route is per recording, not per family.** Index `-1`/`-3`: nose, mouth closed — **1,778** recordings. Index `-2`/`-4`: mouth — **1,778**. The family name carries neither, and the split is exact | Cycle count from envelope peaks/troughs — A5, **not built**, and its three operating points (smoothing window, peak/trough criterion, minimum event duration) are all **owed a cut** ([`branch-airway.md:243-247`](branch-airway.md)). Route is A7 and **may not be measurable**; the `gammatone` 40-channel energy is the only spectral instrument that could carry it. The 1,778/1,778 balance makes this the best-powered route contrast in the corpus, and it is **within session** — all 894 sessions carrying the task carry all four indices |
| AIRWAY | `respiration-and-cough-v2-threebreathsnose` (699), `-threebreathsmouth` (699) | **A, counted (3) and routed**, the route differing *between the two families* and stated in each instruction | Same as above. These two plus the `fivebreaths` index split are the only places the route is a declared contrast, and are therefore the only design that could ever validate A7 |
| AIRWAY | `respiration-and-cough-threequickbreaths` (1,718), `-v2-threebreaths` (699) | **A, counted (3) and timed.** Exhale, then inhale *quickly*. The **interval** is the measurement — a count of three says nothing about whether they were quick | Inter-onset intervals over the counted events; depends entirely on A5's boundaries. **Owed a measurement** |
| AIRWAY | `respiration-and-cough-breath` (1,788), `-v2-breath` (699) | **A, uncounted and durational.** Comfortable breathing for a stated duration — 30 s v1, 20 s v2, v2 specifying through the mouth. `expected_event_count` is absent for these, and both durations are honoured: 1,440 of 1,788 v1 recordings run 30-31 s and 638 of 699 v2 run 20-21 s | File-level presence is **implementable today**: the `airway.breath` gate reads `residual` `energy_fraction`, and `enhanced_hear_scores` / `residual_hear_scores` carry the `Breathe` label per 2 s window. `declared_duration_s` against actual is free, and on v1 it is the sharpest "was the task done" signal in the corpus: **132 of 1,788 run under a second** |
| AIRWAY | `breath-sounds` (326) | **A, counted (3), routed (mouth), and preceded by a declared 60 s of nothing.** *"Please relax for 60 seconds until the task starts. Take three deep breaths in a row in and out of the mouth."* So it is **not** the uncounted durational task its name suggests, and it is closest to `v2-threebreathsmouth` | Count and route as for the three-breath families. The 60 s relax is the one place the instruction **prescribes** material that is not the task — but the measured median is 13.2 s, so the relax period is evidently not inside the file. That is a checkable claim and the check is free: `declared_duration_s` against measured. If a recording does run ~73 s, the first 60 s is `off_task_extent` **by instruction**, which no other family offers |
| AIRWAY | any SPEECH- or VOICE-declared family | **no expected pattern.** AIRWAY routed broadly and its gate evidence is `unavailable` on 56,505 of 62,547 recordings | It labels airway evidence where it finds it and contests what was proposed and is not there. A breath during passage reading is **not** a deviation — it is how S4 measures breath-group structure |

### DDK

DDK is a declared branch with **no node**: `BRANCHES = ("AIRWAY", "SPEECH", "VOICE", "DDK")`
(`vocabulary.py:31`), and `run.py:302-306` marks it `SKIPPED` with
`NO_NODE = "no node implements this branch"`. The graph *can* route to it — two gates do — and then
records that nothing implements it. It is evaluated here on the same terms as the others, per the
owner's instruction that a declared branch is assessed independently of whether it is implemented.

| branch | task | expected pattern | detection approach |
| --- | --- | --- | --- |
| DDK | `diadochokinesis-pa` (896), `-ta` (896), `-ka` (896) | **A, counted, alternating.** One syllable repeated *as fast as possible*, **10 times** — `expected_event_count: 10`. v1 states the count; v2 does not | Envelope modulation spectrum over the located train (D1), cross-checked by syllable-nucleus rate (D2). **Owed a measurement** — D1–D6 are all unbuilt. The gate that fires without a transcript reads `ppg.segment_rate_per_s`, which is **owed a code change**: `extract_ppg_segments` exists (`tasks/features_extraction/ppg.py:349`) and is called from `routing_analysis/features.py:421`, `ppg.py:467`, `ppg.py:543` and `plotting.py:1667` — never from a node, so the store holds the whole-file posteriorgram alone, with no per-span query |
| DDK | `diadochokinesis-v2-puh` (702), `-tuh` (702), `-kuh` (702) | **A, uncounted, alternating.** Same repetition *until the timer runs out*, so no `expected_event_count`. The `'puhpuhpuhpuhpuhpuh'` in the instruction is an orthographic illustration, **not** a six-repetition instruction. The timer is **5 s**: 630-646 of each family's 702 recordings run 5-6 s, against a v1 median of 5 s spread over 3-9 s | As above. Train duration as a fraction of the recording is D5, a `counts` entry — and on v2 the denominator is a fixed 5 s, which makes rate directly comparable across participants in a way v1's participant-terminated recordings are not. That is the cleanest DDK design in the corpus and none of D1-D6 is built to use it |
| DDK | `diadochokinesis-pataka` (896), `-v2-puhtuhkuh` (701) | **A, ordered and cyclic.** A three-place sequence repeated in order — sequential motion rate. `/pa-pa-pa/` is a collapse of the sequence and is the clinically meaningful finding. v1 asks for the sequence *"10 times"* (`expected_event_count: 10`, i.e. 30 syllables); v2 asks for it *"until the timer runs out"*, which is 5 s, so the two are counted and uncounted respectively | D6 sequence conformance, emitting `syllable_sequence_mismatch` — deliberately **not** `stimulus_mismatch`, since there is no stimulus text and no lexical expectation. **Owed a measurement**: the PPG-phoneme → expected-syllable mapping is unmeasured and the posteriorgram is out of domain on rapid nonsense repetition ([`branch-ddk.md:318-323`](branch-ddk.md)) |
| DDK | `diadochokinesis-buttercup` (896), `-v2-buttercup` (702) | **L&A.** A real English word repeated — so unlike every other DDK family this one **does** have a lexical pattern, and the recognisers will produce it. v1 states the count (*"10 times"*, `expected_event_count: 10`); v2 does not (*"until the timer runs out"*), so the two are not one measurement | Consensus words give the repeat count directly via `transcript_repeat`. The only DDK family where the lexical route is the right one, and the only one where `ddk.lexical_repetition >= 3` fires for the correct reason rather than on function-word repetition. **Owed a code change** only (getting `transcript_repeat` into the store) |
| DDK | any lexical-speech family | **no expected pattern.** DDK routed 22,363 against 7,989 declaring a DDK family; repetition occurs in ordinary speech — a stutter, a false start, a repeated word | The branch measures what it finds and says what it is; it does not assert that a Harvard sentence failed to be a DDK task ([`branch-ddk.md:66-70`](branch-ddk.md)) |

QUALITY is not in `BRANCHES` — `vocabulary.py:28-29` calls it "a graph edge, never a branch" — so it
has no rows here. Its Q5 acquisition-consistency capability is the natural home for
`declared_duration_s` against measured duration, which several rows above want.

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
| nasal versus oral route | AIRWAY A7. **Seven** families name a route in their instruction — `respiration-and-cough-fivebreaths` (both, by index), `-threequickbreaths`, `-v2-breath`, `-v2-threebreaths`, `-v2-threebreathsmouth`, `-v2-threebreathsnose`, `breath-sounds` — and only `fivebreaths` and the `v2-threebreaths{nose,mouth}` pair put the two routes in contrast. May not be measurable; `gammatone` is the only candidate instrument |
| lexical / semantic overlap | SPEECH, `story-recall` and `-v2`, and the `free-speech` **v1** anti-pattern — v2 drops the instruction that makes echo a deviation |
| DDK envelope modulation spectrum, syllable-nucleus rate, inter-onset intervals | DDK D1–D3 |
| PPG phoneme → expected syllable mapping | DDK D6 |
| category membership of a produced word | SPEECH, `random-item-generation`, `animal-fluency` |
| vocal tremor | VOICE V4a |
| effort without a within-recording contrast | AIRWAY `respiration-and-cough-v2-hardcough` and `voluntary-cough`, VOICE `loudness` v1 |

**Owed a code change** — the computation exists, its output does not reach the reader:

| what | where it exists | why it does not reach a branch |
| --- | --- | --- |
| a phonation subject for VOICE | `preprocess.py:1875-1891` amplitude spans | they carry no `family`; VOICE selects `family == "phonation"` (`voice.py:230`) |
| PPG segments | `tasks/features_extraction/ppg.py:349` | called from `routing_analysis/features.py:421`, `ppg.py:467`, `ppg.py:543`, `plotting.py:1667` — never from a graph node, so the store holds the whole-file posteriorgram |
| `transcript_repeat` | `routing_analysis/features.py` | a ruleset feature, not a store measurement |
| Praat scalars over an extent | every Praat call already takes a time range | `extract_praat_parselmouth_features_from_audios` has no extent parameter and passes `0, 0` |
| a populated `expected_speech` | `AudioHints.expected_speech` reaches every branch | `runs/b2ai-v2/make_hints.py` parses the filename and never reads `stimulus_text` |
| the expected pattern at the right grain | the per-recording sidecar | nothing reads it; and `task_family` collapses the index that carries the condition |
| whole-file diarization | pyannote runs inside SPEECH over the lexical word hull (`speech.py:642-646`) | moving it to PREPROCESS over `enhanced` and `residual` is settled but docs-only |
| sentence boundaries for CAPE-V | would fall out of S3 alignment | S3 is not built, and they would have to reach VOICE as selectable spans |
| the trailing index, where it carries a condition | the BIDS stem | `task_family` collapses it (`families.py:134-144`), so `fivebreaths`' nose/mouth route and `maximum-phonation-time-v2`'s `-2` escalation are discarded before any branch sees them |

**Owed a cut** — the approach is complete, the operating point is not fitted:

omission detection from forced-alignment score (SPEECH S3); the dominant-monotone-segment criterion
(VOICE V3); verbatim-overlap for the `free-speech` anti-pattern and the read-rather-than-recalled
case; A5's envelope smoothing window, peak/trough criterion and minimum event duration; what makes a
`gap` span off-task rather than ordinary silence. Each belongs in `data/` with a written derivation,
fitted against measured verdicts and **not** against the declared families — a cut fitted on
declarations encodes which recordings the protocol labelled, not which carry the pattern.

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
