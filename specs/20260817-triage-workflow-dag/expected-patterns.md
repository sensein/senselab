# Expected patterns — what each task's instruction asks for, and what would match it

The owner's question: *"is there a list of what algorithm/method/heuristic is used to determine
whether a task was done and what the start end spans are for a task."*

**There is not.** This document is that list.

It is a **method inventory**, not a threshold fit. No number in it is fitted. Where a method needs a
boundary it says what would fit it and marks it owed, per the project rule that a threshold lives in
`data/` with a written derivation and never as a code literal.

**Each branch has exactly two entry points**, per the owner's decision of 2026-09-15: an
`align_<branch>` that evaluates a task of the branch's own kind against what the instruction asked
for, and a `detect_<branch>` that, on a task of any other kind, finds and marks evidence of the
branch's own speciality and evaluates nothing. The per-task variation lives in **data** — one
`Expectation` row per in-family (branch, family) pair — and the code lives in
[§ The code](#the-code-two-entry-points-per-branch), which is Python that compiles and runs rather
than pseudo-code. Every family's complete derivative requirement is stated once in that family's
block rather than assembled from its rows.

Three things follow from the form. An input that does not exist has to be *named* to be passed, so a
row that cannot be implemented says which derivative it is waiting on rather than reading as prose
about a gap. Every in-family entry point returns whether the task was done, so **a declared family
that produced none of its patterns returns *not done*** — the declaration is never the answer. And
per the owner's second decision, of 2026-09-16, **both modes write by `propose` only**: a branch
mints spans in its own family and never edits a span another node proposed, so every extent it has
something to say about is a span it names its own evidence for.

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

## Two modes per branch, and only two

The owner, 2026-09-15:

> *"each branch will have to do an alignment with expectation function eval for tasks that fit the
> branch and for tasks that are given to it but not associated with the branch, it should be
> detecting/refining spans that contain elements the branch specializes for"*

**So a branch has exactly two entry points, and which one runs is one membership test.**

| mode | entry point | when | the question it answers | `done` |
| --- | --- | --- | --- | --- |
| **in-family** | `align_<branch>(task_family, store, hints, params)` | the recording's declared family is one this branch owns | *were the expected patterns found, where, and what deviates from them* | `True` / `False` / `UNDETERMINED` |
| **out-of-family** | `detect_<branch>(store, params)` | the branch was routed here and the family is not its kind | *where is the evidence of this branch's own speciality, and is what was proposed actually there* | **always `UNDETERMINED`** |

The second mode does **not** evaluate the task. It annotates what the branch is expert in, wherever
it occurs, and says nothing about whether the other branch's task was done. This generalises a rule
the owner has stated before for AIRWAY — a non-airway recording routed through AIRWAY should still
find breathing or other airway evidence and mark, refine or refute it — and the generalisation is
what makes it one function per branch rather than one per (branch, family) pair.

**`done = UNDETERMINED` in the out-of-family mode is a rule, not a default.** An earlier draft of
this document gave SPEECH's row over the AIRWAY families `done = (no lexical word was found)` —
reading the absence of lexical content as *"the negative pattern held"*. Under the two-mode rule
that is wrong: an AIRWAY-declared recording is not SPEECH's task, so SPEECH has no *"was it done"*
to answer on it. The lexical intrusion is still found, still given an extent and still handed to AIRWAY,
which owns the deviation ([`branch-airway.md:164-165`](branch-airway.md)); what changes is that
SPEECH no longer renders it as a verdict of its own.

### What is in family for which branch

The membership test is the branch's reference family set, which the packaged config already
declares (`default.yaml:239-243`, verified 2026-09-16):

| branch | in-family families | n | source |
| --- | --- | --- | --- |
| AIRWAY | `AIRWAY_ELICITING` | **11** | `reference_family_set.AIRWAY: airway` (`default.yaml:240`); the set at `families.py:61-75` |
| SPEECH | `LEXICAL_SPEECH` ∪ `SYLLABLE_REPETITION` | **31** | `reference_family_set.SPEECH: speech`, whose own trailing comment spells it `lexical_speech \| syllable_repetition` (`default.yaml:241`); the union is `SPEECH_ELICITING` at `families.py:58` |
| VOICE | `VOICE_ELICITING` | **6** | `reference_family_set.VOICE: voice` (`default.yaml:242`); the set at `families.py:78-87` |
| DDK | `SYLLABLE_REPETITION` | **10** | `reference_family_set.DDK: syllable_repetition` (`default.yaml:243`); the set at `families.py:15-28` |

**58 (branch, family) rows over 48 declared families.** The ten `SYLLABLE_REPETITION` families are
in family for **both** SPEECH and DDK, and that is correct rather than an overlap to resolve:
SPEECH's expectation over them is *no lexical content*, DDK's is *a syllable train*. Both are
alignments against an expectation; neither is the other's.

### VOICE's own instrument families are not VOICE's, and the two-mode rule makes that visible

`VOICE_ELICITING` has **six** members — `glides-high-to-low`, `glides-low-to-high`, `high-to-low`,
`maximum-phonation-time`, `maximum-phonation-time-v2`, `prolonged-vowel` (`families.py:78-87`).
**`cape-v-sentences`, `cape-v-sentences-v2`, `loudness` and `loudness-v2` are in `LEXICAL_SPEECH`**
(`families.py:31-55`), so under the reference family set they are **out of family for VOICE**.

That is the sharpest consequence of the owner's decision in this document. A CAPE-V recording is a
voice-quality instrument — six sentences each loading a different phonatory condition — and
`loudness-v2` is the cheapest effort measurement in the corpus. Under the two-mode rule VOICE meets
both through `detect_voice`, which marks phonation wherever it finds it and **answers no *"was it
done"* at all**. The per-sentence pooling and the within-recording effort contrast are still written
below, as `VOICE_EXPECTATIONS_PENDING_DECLARATION`, and `align_voice` does not read that table.

**The fix is a `families.py` decision, not a branch one**, and one half of it is already on this
document's unsettled list: `families.py:41-42` puts `loudness` and `loudness-v2` in `LEXICAL_SPEECH`
while the sidecars declare `speech_type: "non-lexical"` on all 897 and all 705. CAPE-V is the same
question and was not previously asked, because the old one-function-per-(branch, family) shape let a
`measure_cape_v_per_sentence` sit under a VOICE heading without any membership test ever running.

### What routing hands each mode

Routing hands a branch **one bit** — `will_run` — and nothing else: `run.py:298-300` passes
`(store, "plain", config, hint)` to `airway`, `speech` and `voice`, and no branch reads its own
`branch_decision` (`dag.md:1764-1767`). So the mode test is the branch's, made from the declared
family, and the declared family reaches it two ways:

- **available today, and ugly.** ADMIT writes the resolved source path onto the `recording` stream
  entity (`admit.py:101`, `"path": str(source.resolve())`), and `task_family(task_id_of(stem))` is
  exported from `routing_analysis` (`__init__.py:19`; `families.py:121`, `:134`). A branch can
  therefore compute its own mode from a filename. **Implementable today**, and it puts BIDS-stem
  parsing inside a branch, which is the thing `AudioHints` exists to prevent.
- **the clean route, and `‡`.** `hints.metadata["task_token"]` is the only carrier of the declared
  task, it is written only by the campaign's own builder — `make_hints.py:381`, `:397`, under **this
  spec's own** [`runs/b2ai-v2/`](runs/b2ai-v2/make_hints.py), not under the repo root — and **it is
  read by nothing in `src/senselab`**.
  `AudioHints` has no `task_family` field at all (`audio_hints.py:149-154`).

**Neither route recovers the trailing index**, because `task_family` strips every trailing numeric
segment (`families.py:143`, `_TRAILING_INDEX = re.compile(r"(?:-\d+)+$")` at `:13`). That is the
same loss the `fivebreaths` route and the `maximum-phonation-time-v2` effort escalation both take,
and it is now a loss in the *mode selector's own input* rather than only in a per-family rule.

### What the two modes replace

The previous version of this document carried **37 named detection functions** — 32 per-task, 4
out-of-family (`measure_{voice,airway,ddk}_without_declared_task` and `detect_lexical_intrusion`),
and 1 QUALITY (`detect_occluded_microphone`) — beside three shared instruments and three one-line
helpers. They become:

| | before | after |
| --- | --- | --- |
| in-family determinations | **32** functions, one per (branch, task) | **58 `Expectation` rows** — data, one per in-family (branch, family) pair, built from a 22-field record — dispatched by **4** `align_*` entry points over **13** `Pattern` kinds, i.e. 13 matcher branches in place of 32 functions |
| out-of-family determinations | 4 functions (`measure_{voice,airway,ddk}_without_declared_task`, `detect_lexical_intrusion`) | **4** `detect_*` entry points |
| QUALITY | 1 (`detect_occluded_microphone`) | **1** `detect_quality`, folding it in — QUALITY is not routed and has only the detect mode |
| shared instruments | 3 (`events_in_span`, `sounds_like`, `train_rate`) | the same 3, unchanged in substance, plus the small arithmetic helpers written once |

Nothing was dropped to get there. Every determination the 32 functions made is either a field of an
`Expectation` or a branch of the matcher its `Pattern` selects, and the ten determinations that were
marked *no viable approach* are carried as data too — an `Expectation.unviable` entry that makes the
function emit `NOT_SEPARABLE_BY_THIS_DESIGN` with its reason, rather than silently omitting the
measurement.

### Both modes write by `propose` only — the owner, 2026-09-16

> *"each branch can generate new spans specific to the task of the branch. it doesn't need to edit
> existing spans."*

**A branch mints spans in its own family and never modifies a span another node proposed.** The
mechanics, the four consequences and the code are in
[§ The write path](#the-write-path--propose-only-and-its-four-consequences); two things belong here,
because they change what the two modes *are* rather than how they are written.

**It closes the scoping question the two-mode design would otherwise have opened.**
`branch-conventions.md:20-23` rules that a branch *"`refine`s only a span of the family it is
proposing into"*, and the out-of-family mode is exactly the case that rule was not written for —
AIRWAY finding a cough inside a `harvard-sentences-list` recording wants to sharpen the boundary of
a span AIRWAY does not own. Under propose-only there is nothing to sharpen: AIRWAY mints its own
`family: "airway"` span over the event and names the carrier in `wasDerivedFrom`. The rule is
satisfied by construction.

**And it makes the out-of-family mode's output the same kind of thing as the in-family mode's.**
Both return proposed spans in the branch's own family. What separates them is not what they write
but what they claim: `align_*` answers *"were the expected patterns found"* and `detect_*` returns
`UNDETERMINED`, because no pattern was expected of it.

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

**The owner's 2026-09-16 decision cuts this table down.** *"each branch can generate new spans
specific to the task of the branch. it doesn't need to edit existing spans."* So **every extent a
branch has to say something about is a span it proposes**, and the two verbs that edited someone
else's span are not used by either mode.

| the match's result | verb | payload |
| --- | --- | --- |
| expected pattern P was found here | `propose` | mints a `family: "<branch>"` span, `wasDerivedFrom` the PREPROCESS spans and measurements the extent came from |
| this is the part of the recording that serves the task | `propose` | the same, with `role: "task_extent"`. **Not `trim`** |
| the proposer's boundary for P is wrong | `propose` | the branch mints its own span at the right boundary and names the proposer's in `wasDerivedFrom`. **Not `refine`** — the derivation is now the whole record of the relationship |
| P was proposed here and is not there | `contest` | that it does not carry what was proposed. An assertion *beside* the span, not an edit to it, so it survives the decision unchanged |
| this span carries P | `label` | what the span carries. Also an assertion beside a span |
| P's count against the declared count | *not a verb* — a `counts` measurement, `found` beside `declared`, no discrepancy asserted |
| P was expected and is nowhere | *not a verb* — the branch's own verdict |

`refine` was widened on 2026-09-15 beyond extent — a fired rule may stamp or refine a span's label,
`design.md:568` declaring its payload as `corrected_extent` **and/or** `corrected_attributes`. That
widening is unaffected and unused here: it governs what the *ruleset* may write onto a span, not
what a branch does with one. **No body in this document emits `refine`.**

**A branch mints even where PREPROCESS's spans already cover the ground.** On a held vowel they do —
`voice.sustained` read 15.89 s on the MPT recording of the 2026-09-15 run — and under propose-only
that is not an obstacle: VOICE proposes a `family: "voice"` span over the voiced run and names the
amplitude span in `wasDerivedFrom`. [`branch-conventions.md:20-23`](branch-conventions.md) scopes
minting by family, and a `family: "voice"` span collides with no `family is None` reader, so the
rule is satisfied by construction rather than by an exception.

**Material matching nothing already has a carrier.** PREPROCESS writes `measure: "gap"` spans over
every stretch no other span source covered (`preprocess.py:1912`), so `off_task_extent` does not
need a new detector to find *where* the unmatched material is — only a rule for what makes it
off-task rather than ordinary silence.

### Three inconsistencies this document had to navigate

1. **`trim` has no emitter and no method — and now no job.** `task_extent` appears exactly twice in
   the whole `specs/` tree — `design.md:569` and `:602` — and **zero times in `src/senselab/`**. No
   branch document lists `trim` in its emit block; `dag.md:1823-1824` records that AIRWAY writes
   none. The spec says what `trim` *carries* and nothing about how the extent is *determined*. This
   document is the first statement of that method, per family — and under propose-only the extent is
   a span the branch mints with `role: "task_extent"`, so `trim` is not its carrier either.
2. **`trim` and `deviate` both claim `off_task_extent`.** `design.md:569` has `trim` carrying it;
   `branch-conventions.md:108` stores every deviation as `assertion, verb: "deviate"` and
   `branch-airway.md:350` emits AIRWAY's that way. One of the two has to go, and propose-only
   settles which half of the pair is live: this document emits `off_task_extent` as a `deviate`
   finding and never as a span, because off-task material is the *absence* of the branch's
   speciality and a branch does not mint over ground it is disclaiming.
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

**The detection column is a mode and a row of data, not a call.** Each row says which of the
branch's two entry points meets the recording — `align_<branch>` when the family is in family,
`detect_<branch>` when it is not — and, for the in-family rows, the `Expectation` that entry point
looks up. `†` still marks an input no node writes today and `‡` one that exists but reaches no
branch; the derivative each row needs is stated once in that family's block below rather than
assembled from the row. The bodies — the matcher each `Pattern` selects, the three things it
returns, and the operating points it cannot supply — are in
[§ The code](#the-code-two-entry-points-per-branch).

**A row that reads `Out of family` is not a gap.** It is the second mode doing its job: the branch
annotates its own speciality over the recording and declines to answer *"was the task done"*,
because that question is another branch's. Four such rows were previously written as functions of
their own, and one — `detect_count_in` — turns out to need no successor at all.

### VOICE

| branch | task | expected pattern | detection |
| --- | --- | --- | --- |
| VOICE | `prolonged-vowel` (1,604) | **L→A.** L: the ordered tokens `one two three`, once. A: one continuous voiced production of a single vowel /a/, held to the timer, F0 holding rather than moving. The only voice family in the corpus that is not purely non-lexical, despite `speech_type: non-lexical` | **In family.** `align_voice` → `Expectation(pattern=SUSTAINED, tokens=("one", "two", "three"), token_source="instructions", declared_duration_s=12.0, lexical_separator=True)`. Reads `words` · `spans` · `phonation_tracks` · `continuity_trace` · `energy_envelope` · `stream_extent` · `stimulus_alignment†`. **Owed a code change** for the subject (amplitude spans carry no `family`, and `voice.py:230` selects `family == "phonation"`, so the candidate list is empty before any test runs) and **owed a cut** for the stationarity qualifier — which is a statistic over `continuity_trace` and `f0_hz`, both already in the store, not a missing estimator ([`branch-voice.md:197-210`](branch-voice.md)). The lexical separator is free: the vowel is the voiced production under no lexical word |
| VOICE | `maximum-phonation-time` (2,696) | **A only.** One held /a/ on one breath, to exhaustion. v1's instruction places the deep inhale **before the record tap is mentioned**, so an audible inhale may be inside the file — a second acoustic pattern that is expected but is not the measurement | **In family.** `align_voice` → `Expectation(pattern=SUSTAINED, forbid_lexical=True, expect_inhale=True)`. Reads the same, plus `span_hear` · `span_yamnet` for the inhale. Same owed code change and same owed cut. Duration of the qualified extent is V2's maximum phonation time. The v1 inhale is an AIRWAY `label`, not an `off_task_extent`. **Owed a measurement** (D2) for any voice-quality number over the extent |
| VOICE | `maximum-phonation-time-v2` (813) | **A only**, and cleaner: the instruction puts the inhale explicitly *before* the record tap, so no inhale is expected in the file. A behavioural difference from v1, not a wording difference. **And the index carries a second condition**: `-2` (108 English recordings) appends *"Now try to hold out "ah" for even longer."*, so `-1` → `-2` is a within-subject maximum-duration contrast | **In family.** `align_voice` → `Expectation(pattern=SUSTAINED, forbid_lexical=True, expect_inhale=False)` — the same matcher as v1, one boolean apart, which is what makes the behavioural v1/v2 difference a data difference rather than a second function. The two contrasts are free and neither reaches a branch: **owed a code change**, since `task_family` collapses the trailing index (`families.py:134-144`) that carries v2's `-2` effort escalation, the same change `fivebreaths` needs |
| VOICE | `glides-low-to-high` (1,596), `glides-high-to-low` (1,554) | **A only.** One continuous voiced production of /i/ whose F0 sweeps monotonically across the range, in the declared direction. **Not steady** — the opposite of the sustained pattern | **In family.** `align_voice` → `Expectation(pattern=GLIDE, declared_direction="up")` and `(..., declared_direction="down")` — two rows, one matcher. **Implementable today** for the track; **owed a cut** for the dominant monotone segment and its tolerance (V3). `sweep_direction_mismatch` is the declared deviation (`branch-conventions.md:137`). The sustained qualifier does **not** transfer: `continuity_trace` stays high through a glide |
| VOICE | `high-to-low` (43) | **identical to `glides-high-to-low`** — the instruction string is byte-for-byte the same on all 43 recordings, `stimulus_text` empty on all 43 | **Settled** — an alias, not a task: no matcher of its own, one more row in `align_voice`'s table carrying `declared_direction="down"`. Whether `families.py` folds the two is a declaration question, not a branch one |
| VOICE | `loudness` (897) | **L&A, counted.** The token `hey`, **three times**, each at maximal effort | **Out of family for VOICE today.** `loudness` is `LEXICAL_SPEECH` (`families.py:41`), so `align_voice` never reaches it and `detect_voice` does not evaluate it. The row is written and held in `VOICE_EXPECTATIONS_PENDING_DECLARATION`: `Expectation(pattern=EFFORT, tokens=("hey",), expected_event_count=3, unviable=(("effort_absolute", …),))`. The token, its three extents and the `expected_event_count: 3` count are **implementable today**. Absolute effort has **no viable approach**: `level` is uncalibrated and no SPL reference exists in the graph, so the output is a measurement with its covariates, never a `maximal` / `not maximal` verdict |
| VOICE | `loudness-v2` (705) | **L→A, contrastive.** `hey` at normal effort **→** `hey` shouted. The measurement is the *within-recording* contrast and needs no norm | **Out of family for VOICE today**, as v1 (`families.py:42`). Held in `VOICE_EXPECTATIONS_PENDING_DECLARATION`: `Expectation(pattern=EFFORT, tokens=("hey",), expected_event_count=2, contrast=True)` — the single field `contrast` is the whole v1/v2 difference. **Implementable today**, and the cheapest effort measure in the corpus: a within-recording difference needs no norm and no calibration. V6 is explicit that v1 and v2 are **not one measurement** ([`branch-voice.md:736`](branch-voice.md)) |
| VOICE | `cape-v-sentences` (2,370), `-v2` (1,224) | **A riding on SPEECH's L.** Six sentences each loading a different phonatory condition; the voice-quality measurement is per sentence and pooling discards the instrument's design | **Out of family for VOICE today.** CAPE-V is `LEXICAL_SPEECH` (`families.py:33-34`), so the corpus's one deliberate voice-quality instrument reaches VOICE only through `detect_voice`. Held in `VOICE_EXPECTATIONS_PENDING_DECLARATION`: `Expectation(pattern=PER_SENTENCE, token_source="stimulus_text")`. **Owed a measurement** twice over: D1 for the six sentence boundaries, D2 for any per-sentence voice-quality number at all — today's are whole-file scalars standing for six deliberately different phonatory conditions. **Owed a code change** so the boundaries reach VOICE as selectable spans. Nothing output may be presented as a CAPE-V score |
| VOICE | any AIRWAY- or SPEECH-declared family | **no expected pattern.** VOICE routed 22,277 recordings against 8,306 declaring a voice family | **Out of family.** `detect_voice(store, params)` → `done = UNDETERMINED`; the branch concludes on its own question ([`design.md:486-507`](../20260913-branch-contract-and-hints/design.md)). **Owed a cut**: connected speech passes voiced-fraction, F0-availability and interruption tests, so the stationarity qualifier is what stops V4 computing perturbation over consonants and pauses — the same cut, here load-bearing over 22,277 recordings |

### SPEECH

| branch | task | expected pattern | detection |
| --- | --- | --- | --- |
| SPEECH | `harvard-sentences-list` (13,705), `cape-v-sentences` (2,370), `-v2` (1,224) | **L only**, ordered and fully specified: the words of that recording's `stimulus_text`, in order, once. One sentence per recording — the pattern is per recording, never per family. Measured: **1,060** distinct Harvard sentences, and CAPE-V v1 and v2 share only three of their six sentences. All three families carry the identical instruction, *"Please read the following sentences out loud in your typical voice."* | **In family.** `align_speech` → `Expectation(pattern=ORDERED_TOKENS, token_source="stimulus_text")`. **Owed a measurement** (D1): without it the function returns `UNDETERMINED`, because the transcript alone cannot say what was expected. Substitutions and insertions are then enumerated differences carrying the consensus `agreement`; **owed a cut** for omissions, which a skip-arc-free aligner can only surface as a low acoustic score ([`branch-speech.md:115-121`](branch-speech.md)) |
| SPEECH | `rainbow-passage` (897), `caterpillar-passage` (597) | **L only**, ordered, one long passage. Both grains agree on both fields, on all 897 and all 597 — each is its own acoustic task, so the one-to-many collapse that breaks the other families does not arise. `stimulus_text` is one 338-character passage shared by all 898 rainbow recordings, and one of 1,035 characters shared by the 582 English caterpillar recordings (the 15 Spanish ones carry a 1,079-character Spanish passage) | **In family.** `align_speech` → `Expectation(pattern=ORDERED_TOKENS, token_source="stimulus_text", connected=True)` — `connected` is what adds the breath groups, so a passage is a read text plus one boolean. As above at passage length, and here D1's input is legitimately family-scoped. `[breath]` tokens inside the passage are **not** `filler` — the function subtracts them from the deviation list, because they are how S4 measures breath-group structure |
| SPEECH | `word-color-stroop` (472) | **L only, ordered, and not the displayed words.** The instruction says *name the colour, do not read the word*. `stimulus_text` is the 15-item colour sequence, i.e. the expected **answer** sequence — **472 distinct sequences over 472 recordings**, one per recording and never shared, so the pattern is maximally per-recording. Declared 75 s; measured median 75.8 s, with 463 of 472 falling between 74 and 77 s | **In family.** `align_speech` → `Expectation(pattern=ORDERED_TOKENS, token_source="stimulus_text", declared_duration_s=75.0, emit_filler=False)` — `emit_filler=False` is the one behavioural difference from a read text, and it is a field. **Owed a measurement** (D1), built per recording from that recording's own 15-colour answer sequence — 472 distinct sequences over 472 recordings. `filler` is never emitted here: hesitation and self-correction are the task's dependent variable ([`branch-speech.md:130-134`](branch-speech.md)) |
| SPEECH | `free-speech` (3,074) | **L, unordered, with a negative pattern.** No target text. The v1 instruction says *"Do not record yourself reading the prompt"*, so `stimulus_text` is an **anti-pattern**: the question appearing verbatim in the transcript is the deviation. Four questions, per index — three asked 898 times each, one 380 | **In family.** `align_speech` → `Expectation(pattern=FREE_RESPONSE, token_source="stimulus_text", anti_pattern="verbatim_prompt")`. Presence and extent are **implementable today** from `spans` `measure: "asr"`; the anti-pattern is **owed a measurement** (D1, read inverted) and **owed a cut** (`p_echo_overlap_max`). **This row cannot use the family grain at all**: one frozen prompt covers all 912 sidecars, right on 380 recordings and wrong on 2,694 |
| SPEECH | `free-speech-v2` (2,120) | **L, unordered, and the negative pattern is gone.** v2's instruction drops *"do not record yourself reading the prompt"* entirely and asks instead to answer *"as though you were having a conversation"*. So verbatim echo is **not** declared a deviation here, and treating v1 and v2 alike would invent one. Six questions, three English (683 each) and three Spanish (24 each) | **In family.** `align_speech` → `Expectation(pattern=FREE_RESPONSE, declared_duration_s=30.0)` — `anti_pattern` is **absent**, which is the entire v1/v2 difference and is now one missing field rather than a second function that could drift back toward v1's. **Implementable today**, and it emits **no** verbatim-echo deviation: v2 drops the instruction that made echo one, so firing v1's rule here invents it. The acoustictask prompt matches none of the six v2 questions on all 707 sidecars |
| SPEECH | `story-recall` (889), `story-recall-v2` (660) | **L, unordered, partly negative.** Recall *in the participant's own words*. `stimulus_text` is the source story — the 717-character grandfather passage for v1, the 1,083-character frog story for v2 — so semantic coverage is expected and **verbatim** reproduction is the deviation, the participant having read rather than recalled. The five Spanish v1 recordings carry the **frog** story, i.e. v2's source under v1's family name, so even here the source must be read per recording | **In family.** `align_speech` → `Expectation(pattern=FREE_RESPONSE, token_source="stimulus_text", anti_pattern="verbatim_source")` — the same matcher as `free-speech`, with the other anti-pattern. **Owed a measurement** (D1) and **owed a cut** (`p_verbatim_overlap_max`); the n-gram overlap itself is arithmetic once D1 exists. The source is read per recording — the five Spanish v1 recordings carry v2's story under v1's family name |
| SPEECH | `cinderella-story` (258) | **L, unordered, and nothing machine-readable.** `stimulus_text` is empty on all 258; the source is a physical storybook handed to the participant, so no overlap measure is even definable here | **In family.** `align_speech` → `Expectation(pattern=FREE_RESPONSE, unviable=(("source_overlap", …),))` — no `token_source` and no `anti_pattern`, and the `unviable` row is what makes the ceiling explicit rather than implied by an absence. Presence and extent **implementable today**, and that is the ceiling rather than a first step: `stimulus_text` is empty on all 258, so **no overlap measure is definable**, D1 included, and the `story-recall` method does not transfer |
| SPEECH | `productive-vocabulary` (2,910) | **L, weakly specified, per recording.** `stimulus_text` is one cue word per recording — **204 distinct cues** across the corpus, and **78 recordings carry no cue at all**. Expected is definitional speech *about* the cue, not the cue itself | **In family.** `align_speech` → `Expectation(pattern=FREE_RESPONSE, token_source="stimulus_text", unviable=(("defines_its_cue", …),))`. Presence and extent **implementable today**; whether the speech defines its cue has **no viable approach** in this graph — a lexicon or a text model, branch-local, no waveform. 204 cues under one family name and 78 recordings with none |
| SPEECH | `picture-description` (889), `-option1` (373), `-option2` (329) | **L only, unspecified.** Connected speech, no target text, `stimulus_text` empty on all 1,591 (the stimulus is an image URL in the sidecar). `picture-description` and `-option1` carry a **byte-identical** instruction — *"Tell me everything you see going on in this picture."* — and differ only by image; `-option2` asks for complete sentences *"as though describing it for the blind"* | **In family.** `align_speech` → `Expectation(pattern=FREE_RESPONSE, connected=True)` — **three identical rows**, one per family name, which is the measurable statement that no method here distinguishes them. **Implementable today** for presence, extent, pause structure and breath groups; S4's measures are unbuilt branch code, not a missing derivative ([`branch-speech.md:157`](branch-speech.md)). The v1/option1 instructions are byte-identical, so **no method here may distinguish them** — any difference found is the image's |
| SPEECH | `open-response-questions` (199) | **L only, unspecified.** *"Please answer the following questions and record your answer."* One `stimulus_text`, a 447-character prompt shared by all 199. Median 30 s | **In family.** `align_speech` → `Expectation(pattern=FREE_RESPONSE, token_source="stimulus_text", declared_duration_s=30.0, connected=True)`. **Implementable today.** The single shared 447-character prompt makes this the one connected-speech family whose `expected_speech` is legitimately family-scoped |
| SPEECH | `animal-fluency` (195) | **L, unordered, category-bound, with a repetition anti-pattern.** *"say as many animals as you can, while avoiding repeating the same ones. There is a 1 minute timer"*. `stimulus_text` empty on all 195; the category lives only in `instructions`. Median duration 60 s, matching the declared timer | **In family.** `align_speech` → `Expectation(pattern=ITEM_LIST, declared_duration_s=60.0, repetition_allowed=False, unviable=(("category_membership", …),))`. Items, their extents, the repetition anti-pattern and the duration check are **implementable today**; category membership has **no viable approach** here — it needs no waveform, has one consumer, and is branch-local rather than a PREPROCESS derivative |
| SPEECH | `random-item-generation` (265), `-v2` (207) | **L, unordered, category-bound — and the negative constraint is category-conditional.** Ten categories per family, one per recording, living only in `instructions`. Eight of the ten (`Animals`, `City names`, `Country names`, `Drinks`, `First names`, `Fruits`, `Jobs`, `English words starting with 't'`) say *"Do not repeat any item"*. The other two, **`Letters` and `Numbers`, say the opposite** — *"random letters or numbers (**repetition allowed**)"* — 48 of 265 v1 recordings and 77 of 203 English v2 recordings | **In family.** `align_speech` → `Expectation(pattern=ITEM_LIST, repetition_from_category=True, unviable=(("category_membership", …),))` — `repetition_allowed` is deliberately **not** a field here: it is read per recording, and the matcher returns `UNDETERMINED` when the category cannot be read rather than defaulting to the forbidding rule. **Owed a code change**: the category lives only in `instructions` and no grain above the recording carries it, so a family-scoped repeat rule inverts the instruction on the `Letters` and `Numbers` recordings — 48 of 265 v1 and 77 of 203 English v2. Category membership: **no viable approach**, as above |
| SPEECH | `loudness` (897), `loudness-v2` (705) | **L:** the token `hey` | **In family.** `align_speech` → `Expectation(pattern=ORDERED_TOKENS, tokens=("hey", "hey", "hey"), expected_event_count=3)` and `(..., tokens=("hey", "hey"), expected_event_count=2)`. **Implementable today**; SPEECH contributes the token's extents and VOICE's V6 is the measurement wanted. These two are in `LEXICAL_SPEECH` (`families.py:41-42`) while the protocol calls them `speech_type: "non-lexical"` — a `families.py` discrepancy |
| SPEECH | every `SYLLABLE_REPETITION` family (7,989) | **L: none expected.** `/pa/` is not lexical and ASR mostly declines it. These are **positives** for SPEECH's reference set as of 2026-09-15 (`reference_family_set.SPEECH: speech` = `lexical_speech \| syllable_repetition`, `default.yaml:241`), so near-zero lexical content is the correct observation, not a miss. `diadochokinesis-buttercup` is the exception — see the DDK table | **In family.** `align_speech` → `Expectation(pattern=NO_LEXICAL)` — **eight identical rows**; `buttercup` gets its own, below. **Implementable today.** These are **positives** for SPEECH's reference set (`default.yaml:241`), so near-zero lexical content is the correct observation, not a miss; `speech.lexical >= 2` firing here is over-routing on function-word artefacts ([`dag.md:195-196`](dag.md)) |
| SPEECH | `diadochokinesis-buttercup` (896), `-v2-buttercup` (702) | **L&A**, and the one `SYLLABLE_REPETITION` family with a lexical pattern: a real English word repeated, which the recognisers will produce. So SPEECH's expectation here is the opposite of its other eight syllable-repetition rows | **In family.** `align_speech` → `Expectation(pattern=ORDERED_TOKENS, tokens=("buttercup",) * 10, expected_event_count=10, emit_filler=False)` and `(..., tokens=("buttercup",), declared_duration_s=5.0, emit_filler=False)`. **Implementable today**: the same token matcher every read family uses, over a one-word expectation. SPEECH and DDK both hold this family in family and measure different things over it — the token count and the envelope rate — which is the intended shape, not a conflict |
| SPEECH | `prolonged-vowel` (1,604) | **L:** the ordered tokens `one two three`, once. The count-in is **prescribed**, not incidental: the instruction asks the participant to repeat *"1, 2, 3 aah"*, on all 1,575 English recordings | **Out of family.** `prolonged-vowel` is `VOICE_ELICITING` (`families.py:78-87`), so SPEECH runs `detect_speech(store, params)`, which marks the count-in as a lexical run and answers no *"was it done"*. The evaluation of the count-in is `align_voice`'s, through that row's `tokens` field. `detect_count_in` therefore has no successor function at all: the expectation became data and the finding became the generic lexical marking. **Implementable today**: D1 sharpens it, but an ordered-run match over the consensus words needs nothing new. Measured: 938 of 1,258 transcripts open with `One two three` ([`dag.md:185`](dag.md)) |
| SPEECH | any AIRWAY-declared family | **L: none expected.** Any lexical content is off-task by construction | **Out of family.** `detect_speech(store, params)` → `done = UNDETERMINED`; the lexical extents are still emitted and still handed on. **Implementable today** and needs no alignment: nothing lexical is expected, so every lexical word is the finding. AIRWAY owns the deviation ([`branch-airway.md:164-165`](branch-airway.md)) |

### AIRWAY

| branch | task | expected pattern | detection |
| --- | --- | --- | --- |
| AIRWAY | `respiration-and-cough-cough` (1,788) | **A, counted.** Five discrete forced expulsive events — *"After pressing record, cough 5 times"*. `expected_event_count: 5` | **In family.** `align_airway` → `Expectation(pattern=EVENT_SERIES, label_set="cough", expected_event_count=5)`. **Owed a code change**: the count is of events, not of label-carrying spans — `by_label` increments once per (span, label) pair (`airway.py:280`), so a 4 s span holding three coughs counts 1. **Owed a cut** for A5/A6's four boundary points. The merging case is a series on one exhalation, visible as multiple maxima inside one span — not `spans.min_separation_ms`, which is 30 ms |
| AIRWAY | `respiration-and-cough-v2-hardcough` (698) | **A, one event, maximal effort.** *"cough HARD as if something were stuck in your throat"*. No count stated, and the instruction adds a **recording-hygiene** clause found nowhere else — *"do not cover your mouth or place your hand between your mouth and the microphone"* | **In family.** `align_airway` → `Expectation(pattern=EVENT_SERIES, label_set="cough", expected_event_count=None, unviable=(("effort_absolute", …),))` — the `None` count and the `unviable` row are the whole difference from the 5-cough family. Events as above, minus the count. *"Hard"* has **no viable approach**: no within-recording contrast and no SPL reference, so the output is a measurement with its covariates. The hygiene clause is QUALITY's, and folds into `detect_quality(store, params)` rather than becoming a function of its own: a level and spectral-tilt finding over `level`, `spectrogram_wideband` and `band_profile†` |
| AIRWAY | `voluntary-cough` (327) | **A, counted (3), maximal effort, with breathing between.** Same *"cough HARD"* wording as `v2-hardcough` but *"Complete this task **3 times** in a single recording"*, with *"then breathe normally again"* between. `expected_event_count: 3`, and the inter-cough breathing is an **expected** pattern, not off-task | **In family.** `align_airway` → `Expectation(pattern=EVENT_ALTERNATION, label_set="cough", expected_event_count=3)` — the one AIRWAY family needing its own matcher, because the expected pattern is an alternation and material between coughs must be matched as breath rather than scored off-task. The one cough family where a cough detector alone is insufficient: the expected pattern is an **alternation**, so material between coughs is matched as breath rather than scored as `off_task_extent`. Same owed cut as the 5-cough row. Median 13.8 s against `v2-hardcough`'s 4.6 s, consistent with three cycles |
| AIRWAY | `respiration-and-cough-fivebreaths` (3,576) | **A, counted (5) and routed — and the route is per recording, not per family.** Index `-1`/`-3`: nose, mouth closed — **1,778** recordings. Index `-2`/`-4`: mouth — **1,778**. The family name carries neither, and the split is exact | **In family.** `align_airway` → `Expectation(pattern=EVENT_SERIES, label_set="breath", expected_event_count=5, route_from_index=True, unviable=(("route", …),))`. The count is **owed a cut** (A5's operating points, [`branch-airway.md:243-247`](branch-airway.md)). The route has **no viable approach**: the discriminating band is largely above the 8 kHz ceiling and the residual tilt is confounded one-for-one with mouth-to-microphone geometry, which changes *with the route by construction*. The function returns `route = NOT_SEPARABLE_BY_THIS_DESIGN` and carries `band_profile†` so a negative is attributable. The index that carries the route is **owed a code change** — `task_family` collapses it |
| AIRWAY | `respiration-and-cough-v2-threebreathsnose` (699), `-threebreathsmouth` (699) | **A, counted (3) and routed**, the route differing *between the two families* and stated in each instruction | **In family.** `align_airway` → `Expectation(pattern=EVENT_SERIES, label_set="breath", expected_event_count=3, declared_route="nose")` and `(..., declared_route="mouth")` — the route the family name carries is a field; `fivebreaths` sets `route_from_index` instead, and that one field is where the collapsed index bites. Same count, same route finding. These two and the `fivebreaths` index split are the only declared route contrasts in the corpus, which is why neither should be treated as validation-grade for A7 |
| AIRWAY | `respiration-and-cough-threequickbreaths` (1,718), `-v2-threebreaths` (699) | **A, counted (3) and timed.** Exhale, then inhale *quickly*. The **interval** is the measurement — a count of three says nothing about whether they were quick | **In family.** `align_airway` → `Expectation(pattern=EVENT_SERIES, label_set="breath", expected_event_count=3, timed_intervals=True)` — `timed_intervals` is what turns the count into an interval measurement; it is a field, not a function. The **interval** is the measurement — a count of three says nothing about whether they were quick — and it is arithmetic over the event onsets, so it is **owed the same cut** and nothing more: `p_interval_max_s` cannot be fitted before A5's boundary points are |
| AIRWAY | `respiration-and-cough-breath` (1,788), `-v2-breath` (699) | **A, uncounted and durational.** Comfortable breathing for a stated duration — 30 s v1, 20 s v2, v2 specifying through the mouth. `expected_event_count` is absent for these, and both durations are honoured: 1,440 of 1,788 v1 recordings run 30-31 s and 638 of 699 v2 run 20-21 s | **In family.** `align_airway` → `Expectation(pattern=SOUND_COVERAGE, label_set="breath", declared_duration_s=30.0)` and `(..., declared_duration_s=20.0, declared_route="mouth", unviable=(("route", …),))`. **Implementable today** on HeAR's `Breathe`, which is a positive detection of the thing being asked about. `residual` `energy_fraction` is **not** read: `residual = plain − g·FRCRN(plain)` and FRCRN is a speech enhancer, so a high fraction means *this is not speech* — a cough, a glide, room noise and a near-silent file all satisfy it. `declared_duration_s` against measured is free and is the sharpest signal here: 132 of 1,788 v1 recordings run under a second |
| AIRWAY | `breath-sounds` (326) | **A, counted (3), routed (mouth), and preceded by a declared 60 s of nothing.** *"Please relax for 60 seconds until the task starts. Take three deep breaths in a row in and out of the mouth."* So it is **not** the uncounted durational task its name suggests, and it is closest to `v2-threebreathsmouth` | **In family.** `align_airway` → `Expectation(pattern=EVENT_SERIES, label_set="breath", expected_event_count=3, declared_route="mouth", relax_s=60.0, declared_duration_s=73.0, unviable=(("route", …),))` — `relax_s` is the only field of its kind in the corpus and exists because one instruction prescribes material that is not the task. Count and route as the three-breath families. The declared 60 s relax is the only place an instruction **prescribes** material that is not the task, and the function emits it as `off_task_extent` only on a recording long enough to contain it — the measured median is 13.2 s against a declared ~73 s |
| AIRWAY | any SPEECH- or VOICE-declared family | **no expected pattern.** AIRWAY routed broadly and its gate evidence is `unavailable` on 56,505 of 62,547 recordings | **Out of family.** `detect_airway(store, params)` → `done = UNDETERMINED`; it labels airway evidence where it finds it and contests what was proposed and is not there. **Implementable today.** A breath during passage reading is **not** a deviation — it is how S4 measures breath-group structure |

### DDK

DDK is a declared branch with **no node**: `BRANCHES = ("AIRWAY", "SPEECH", "VOICE", "DDK")`
(`vocabulary.py:31`), and `run.py:302-306` marks it `SKIPPED` with
`NO_NODE = "no node implements this branch"`. The graph *can* route to it — two gates do — and then
records that nothing implements it. It is evaluated here on the same terms as the others, per the
owner's instruction that a declared branch is assessed independently of whether it is implemented.

| branch | task | expected pattern | detection |
| --- | --- | --- | --- |
| DDK | `diadochokinesis-pa` (896), `-ta` (896), `-ka` (896) | **A, counted, alternating.** One syllable repeated *as fast as possible*, **10 times** — `expected_event_count: 10`. v1 states the count; v2 does not | **In family.** `align_ddk` → `Expectation(pattern=SYLLABLE_TRAIN, sequence=("labial"|"alveolar"|"velar",), expected_event_count=10)` — **three rows differing in the place their instruction names**. They were three identical rows until 2026-09-16, when the posteriorgram CV instrument ([`branch-ddk-ppg-instrument.md`](branch-ddk-ppg-instrument.md)) gave the matcher something to read the syllable with; `/pa/` IS labial, so the place is the task definition and not a fit, and `len(sequence)` is now the cycle for a one-syllable train as it already was for a sequential one. **Owed a cut**, not a measurement: D1's rate is the modulation spectrum of `energy_envelope` over the train and D3's intervals are its event onsets, both over an array the store already holds. Praat's `extract_speech_rate` is **not** the instrument — it is already running inside `praat_features` and its `min_dip` and 0.3 s `min_pause` both under-count the fastest trains, biasing the measurement in the direction of the quantity being measured |
| DDK | `diadochokinesis-v2-puh` (702), `-tuh` (702), `-kuh` (702) | **A, uncounted, alternating.** Same repetition *until the timer runs out*, so no `expected_event_count`. The `'puhpuhpuhpuhpuhpuh'` in the instruction is an orthographic illustration, **not** a six-repetition instruction. The timer is **5 s**: 630-646 of each family's 702 recordings run 5-6 s, against a v1 median of 5 s spread over 3-9 s | **In family.** `align_ddk` → `Expectation(pattern=SYLLABLE_TRAIN, sequence=("labial"|"alveolar"|"velar",), declared_duration_s=5.0)` — `expected_event_count` absent and `declared_duration_s` present, which is the whole counted/uncounted difference; the place is named here for the same reason it is on the v1 rows. As above with no declared count. D5's train fraction has a fixed 5 s denominator here, which makes the rate directly comparable across participants in a way v1's participant-terminated recordings are not — and none of D1-D6 is built to use it |
| DDK | `diadochokinesis-pataka` (896), `-v2-puhtuhkuh` (701) | **A, ordered and cyclic.** A three-place sequence repeated in order — sequential motion rate. `/pa-pa-pa/` is a collapse of the sequence and is the clinically meaningful finding. v1 asks for the sequence *"10 times"* (`expected_event_count: 10`, i.e. 30 syllables); v2 asks for it *"until the timer runs out"*, which is 5 s, so the two are counted and uncounted respectively | **In family.** `align_ddk` → `Expectation(pattern=SYLLABLE_SEQUENCE, sequence=("labial", "alveolar", "velar"), expected_event_count=30)` and `(..., declared_duration_s=5.0)`. The expected sequence is data, so a four-place train would be a row and not a rewrite. **Owed a cut** for the place decision (`p_burst_window_ms`, `p_place_centroid_bands_hz`, `p_place_margin`) and nothing else: /p/, /t/ and /k/ differ in burst spectrum inside 8 kHz, and `spectrogram_wideband`'s 5 ms window at a 5 ms hop is the classical resolution for it. The **PPG is not the place authority** — it is trained on connected speech and its prior works against the discrimination on a rapid nonsense train ([`branch-ddk.md:318-323`](branch-ddk.md)), and nobody has measured whether it holds there. Since 2026-09-16 it is read anyway and reported BESIDE the burst spectrum, with an agreement fraction over the onsets both resolved, because that agreement is the only thing that can settle D6's owed question — see [`branch-ddk-ppg-instrument.md`](branch-ddk-ppg-instrument.md). `syllable_sequence_mismatch`, deliberately not `stimulus_mismatch` |
| DDK | `diadochokinesis-buttercup` (896), `-v2-buttercup` (702) | **L&A.** A real English word repeated — so unlike every other DDK family this one **does** have a lexical pattern, and the recognisers will produce it. v1 states the count (*"10 times"*, `expected_event_count: 10`); v2 does not (*"until the timer runs out"*), so the two are not one measurement | **In family.** `align_ddk` → `Expectation(pattern=ORDERED_TOKENS, tokens=("buttercup",), expected_event_count=10)` and `(..., declared_duration_s=5.0)` — the only DDK row whose `Pattern` is a lexical one, and `align_ddk` dispatches it to the same token matcher SPEECH uses. **Implementable today**: the repeat count is a counter over normalised consensus tokens, so `transcript_repeat` moving into the store is a convenience rather than the capability. The only DDK family where the lexical route is the right one — and the only one where the removed `ddk.lexical_repetition >= 3` gate used to fire for the right reason |
| DDK | any lexical-speech family | **no expected pattern.** DDK routed 22,363 against 7,989 declaring a DDK family; repetition occurs in ordinary speech — a stutter, a false start, a repeated word | **Out of family.** `detect_ddk(store, params)` → `done = UNDETERMINED`. The branch measures what it finds and says what it is; it does not assert that a Harvard sentence failed to be a DDK task ([`branch-ddk.md:66-70`](branch-ddk.md)) |

### QUALITY — one mode, and it is the out-of-family one

**QUALITY gets no rows in the table above, and that is a structural statement rather than an
omission.** It is not in `BRANCHES` (`vocabulary.py:31`); `vocabulary.py:29` calls it *"the terminal
node every recording reaches, whatever routed. A graph edge, never a branch."* It has **no
`branch_decision`**, no entry in `branch_gates`, and the runner calls it unconditionally at
`run.py:310` — after the branch loop at `:302`, over the **`recording`** stream rather than `plain`.
Its `KIND` is `None` (`quality.py:57`), so the file-level fold never joins its verdict to a branch
(`vocabulary.py:335` filters on `verdict.kind is not None`). It deletes the hint it is handed
(`quality.py:238`, `del hint, run_dir`).

**So QUALITY is never handed a task, and therefore can never be in family.** It has one entry point,
`detect_quality(store, params)`, and that entry point is the *out-of-family* mode by construction:
task-agnostic, returning `done = UNDETERMINED`, marking what it is expert in wherever it occurs.
Giving it an `align_quality` would require a family whose declared expectation is a quality
property, and no such family exists — the closest thing in the corpus is
`respiration-and-cough-v2-hardcough`'s hygiene clause, which is an expectation **about the
recording** attached to an AIRWAY task, not a task of QUALITY's own.

Two consequences:

- **`detect_occluded_microphone` is not a separate function.** The hygiene clause —
  *"do not cover your mouth or place your hand between your mouth and the microphone"*, the only one
  in the corpus — becomes a branch of `detect_quality`, which runs on that recording as it runs on
  every other. AIRWAY's row for that family carries the cough events and says nothing about the
  microphone; QUALITY carries the tilt finding and says nothing about the cough.
- **`declared_duration_s` against measured duration belongs here**, and several rows above want it.
  It is a sidecar-consistency check, not a task-completion measurement
  ([`preprocess-derivatives-for-expected-patterns.md`](preprocess-derivatives-for-expected-patterns.md)
  § 4.6), which is exactly the shape of a QUALITY finding: 2,020 sidecars declare under a second, and
  which of the declaration and the recording is wrong is a separate question. Q5 is its capability.

What QUALITY does **today** is one internal-consistency check — whether PREPROCESS's own clip spans
contradict PREPROCESS's own amplitudes, `quality.py:266-302`, one `contest` assertion per
contradiction, expected count zero on a fresh store. `detect_quality` below keeps that intact and
adds the two rows above.

---

## The code: two entry points per branch

This section is the whole of the executable part. It is **Python that compiles and runs** — compiled
and exercised end to end over a synthetic store on 2026-09-16, every entry point and every `Pattern`
branch, with assertions that each proposed span has positive duration, carries its branch's own
family and names its evidence — rather than pseudo-code, so that porting it to `nodes/<branch>.py`
is transcription plus type-fixing against the real store objects. What it is *not* is fitted: every
`p_*` is an unfitted operating point and **no number appears in any body**.

### The write path — `propose` only, and its four consequences

> *"each branch can generate new spans specific to the task of the branch. it doesn't need to edit
> existing spans."*

**A branch mints spans in its own family and never modifies a span another node proposed.** Four
things follow, and they are why the code below looks the way it does.

**1. The family-scoping question is closed, not deferred.**
[`branch-conventions.md:20-23`](branch-conventions.md) rules that a branch *"`refine`s only a span of
the family it is proposing into"*. Under propose-only that is satisfied by construction — there is
nothing to refine — so the out-of-family mode has no scoping problem to resolve. An earlier draft of
this section carried that as an open dependency; it is gone.

**2. `refine` is off the critical path, and the two documentation conflicts around it stop
blocking.** `refine` is written by nothing in `src/senselab` — swept 2026-09-16, every occurrence of
the string is prose or an unrelated identifier (`prov_store.py:438`, `praat_parselmouth.py:95`,
`audio_analysis`'s `refined_identity` / `I1_boundary_refinement` / `fuse.py:1089`,
`label_membership.py:82`, and `default.yaml:180`'s comment about *"each branch's refined spans"*),
with **zero call sites**. And `store.md:72` says *"`refine` and `withdraw` are gone as verbs"* while
`design.md:568` declares `refine`'s payload and `branch-conventions.md:97-100` records it being
*widened* on 2026-09-15. Both are **documentation conflicts to reconcile, not blockers on these
branches** — and note `store.md:72` is wrong about `withdraw` independently of `refine`, since
`WITHDRAW_VERB` is declared at `preprocess.py:142` and written at `:605`.

**3. VOICE gets a subject, and that is what makes it implementable at all.** It no longer waits for
a `family == "phonation"` span nothing mints. `voice.py:227-231` selects `family == "phonation"`,
PREPROCESS's amplitude spans carry no `family` key (`preprocess.py:1875-1891`), the detector that
proposed phonation spans was retired on 2026-09-04, and VOICE therefore takes the no-span `FAIL` at
`voice.py:235-264` **on every recording** — while the gate that routed it there, `voice.sustained`,
read the longest **amplitude** span. Under propose-only VOICE reads those amplitude spans plus
`phonation_tracks` as *evidence* and proposes its own `family: "voice"` span over what qualifies.
The selector problem dissolves with the editing problem.

**4. Provenance replaces editing.** Every proposed span names in `wasDerivedFrom` the PREPROCESS
spans and measurements its extent came from. **That link is now the whole record of the
relationship** — where a refinement used to say *"this span's boundary is wrong, here is the right
one"*, a proposal says *"here is my span, derived from that one"*, and the derivation is the only
thing carrying the connection. A body that proposes without it loses information. The minting
helper asserts on it rather than trusting the author.

**What this enables in code that is not changed here.** `report.py:291`'s
`_spans_of_family(store, family, *, voice=...)` exists only because VOICE re-minted a second
phonation span from an existing one and `onset_kind` was the only thing telling the two populations
apart (its own docstring says so). Under propose-only they are different families —
`family: "voice"` against whatever proposed the evidence — so the `voice=` split becomes
unnecessary. `_spans_of_family` has **five** call sites (`report.py:673`, `:688`, `:715`, `:1153`,
`:1154`) and **three** pass `voice=`: `:673` `voice=False`, `:715` `voice=True`, `:1154`
`voice=True`. Those three are what the decision simplifies; the other two already read one family
and are untouched. **Recorded as a simplification the
decision enables; no code is changed by this document.**

**`label` and `contest` are unaffected.** Both are assertions written *beside* a span, not edits to
it, so both remain available to either mode — `contest` is the one verb in this document with live
emitters today (`airway.py:310`, `quality.py:291`).

**One line of the contract's verb table widens.** `design.md:570` gives `propose` as writing a
`span`, `family: "<branch>"`, `wasDerivedFrom` its evidence — and says it carries *"a region
PREPROCESS did not find"*. Under propose-only a branch mints **whether or not PREPROCESS found the
region**, because minting is now the only way it has of saying anything about an extent. The verb,
the family and the derivation are unchanged; what goes is the novelty precondition, which was the
half that made `refine` necessary. `branch-conventions.md:20-23`'s family scoping already permits
this — a `family: "<branch>"` span collides with no `family is None` reader — and its three stated
consequences (a branch measures over its own span; covariates describe the same extent; nothing is
left unreconciled) are exactly what the decision generalises.

### What every entry point returns

**Every entry point returns the same triple**, which is what the DAG's verbs read:

```
done        Were the expected patterns found. Read off the recording, never off the declaration:
            a declared family that produced none of its patterns returns False, and a matcher
            whose only instrument is missing returns UNDETERMINED rather than guessing.
            `detect_*` returns UNDETERMINED always — it evaluates no task.
components  The spans this branch PROPOSES, each in the branch's own family, each naming the
            evidence it came from. An empty list is a result: `_speech_no_lexical` proposes
            nothing, and that is the correct observation on a syllable-repetition recording.
deviations  Every finding that is not a proposed span: the nine deviation types of
            `branch-conventions.md:123-142` plus the two this document owes, the `counts` entries
            ({found, declared}) where the instruction declares a number, the per-extent
            measurements with their covariates, and the `contest`s. A count asserts no
            discrepancy.
```

**The third element carries four kinds, and the store splits them:**

| `Finding.kind` | written as | authority |
| --- | --- | --- |
| `deviation` | `assertion`, `verb: "deviate"`, plus `deviation_type`, the extent and the evidence | `branch-conventions.md:104-110` |
| `count` | `measurement` named `counts`, each entry carrying `found` beside `declared` | `branch-conventions.md:104-110` |
| `measure` | a branch measurement over its own extent, carrying its covariates and its support count | `branch-conventions.md` § *Quality covariates travel with every acoustic measurement* |
| `contest` | `assertion`, `verb: "contest"` — the one with live emitters (`airway.py:310`, `quality.py:291`) | — |

```python
UNDETERMINED = "UNDETERMINED"
NOT_SEPARABLE_BY_THIS_DESIGN = "NOT_SEPARABLE_BY_THIS_DESIGN"
Done = bool | Literal["UNDETERMINED"]


class Proposal(NamedTuple):
    """A span this branch mints in its own family. It never edits a span another node proposed."""

    family: str                      # the branch's own, lowercase (branch-conventions.md:9-18)
    role: str                        # what this span is, inside the task
    start: float
    end: float
    derived_from: tuple[str, ...]    # wasDerivedFrom: the whole record of where the extent came from
    attributes: dict


class Finding(NamedTuple):
    kind: str
    name: str
    start: float | None
    end: float | None
    evidence: dict


class Result(NamedTuple):
    done: Done
    components: list[Proposal]
    deviations: list[Finding]


def proposer(family: str):
    """One minting function per branch. The family is fixed by the branch, never by the caller."""

    def _propose(role: str, extent: tuple[float, float], *derived_from: str, **attributes: object) -> Proposal:
        assert derived_from, "a proposed span names its evidence; the derivation is the whole record"
        assert extent[1] > extent[0], f"{role}: a proposed span has positive duration"
        return Proposal(family, role, float(extent[0]), float(extent[1]), tuple(derived_from), dict(attributes))

    return _propose


voice_span = proposer("voice")
speech_span = proposer("speech")
airway_span = proposer("airway")
ddk_span = proposer("ddk")
quality_span = proposer("quality")


def deviation(
    name: str, start: float | None, end: float | None, /, *derived_from: str, **evidence: object
) -> Finding:
    return Finding("deviation", name, start, end, dict(evidence), tuple(derived_from))


def contest(span_id: str, extent: tuple[float, float], claim: str, reason: str) -> Finding:
    return Finding("contest", claim, extent[0], extent[1], {"reason": reason}, (span_id,))


def count(name: str, found: object, declared: object, *derived_from: str) -> Finding:
    return Finding("count", name, None, None, {"found": found, "declared": declared}, tuple(derived_from))


def measured(
    name: str, start: float | None, end: float | None, value: object, /, *derived_from: str, **covariates: object
) -> Finding:
    return Finding("measure", name, start, end, {"value": value, **dict(covariates)}, tuple(derived_from))


def unviable(name: str, why: str) -> Finding:
    return Finding("measure", name, None, None, {"value": NOT_SEPARABLE_BY_THIS_DESIGN, "why": why})


# -------------------------------------------------------- the store, by name
```

### Reading the signatures

| mark | meaning |
| --- | --- |
| no mark | the input is a derivative PREPROCESS writes today, spelled as the store spells it |
| `†` | the derivative does not exist. Named anyway, so the dependency is visible: `stimulus_alignment` (D1), the extra `phonation_tracks` columns this document spells `hnr_db` / `rms_dbfs` / `cpps_db` (D2), `band_profile` (D3), all three from [`preprocess-derivatives-for-expected-patterns.md`](preprocess-derivatives-for-expected-patterns.md). **D2's own block names no column identifiers** — it specifies *"harmonics-to-noise ratio in dB, short-time RMS, and smoothed cepstral peak prominence, one value per 10 ms frame, over `plain`"* and is explicitly *not a new entity*, three arrays added to `derivatives/phonation_tracks.npz` (`preprocess.py:1300`). The three names below are this document's spelling of them, not that document's |
| `‡` | the value exists somewhere in the tree but no branch can read it: `hints.expected_speech` (declared, never populated), `hints.metadata["task_token"]` and the trailing task index it carries, `transcript_repeat` (a `routing_analysis` feature, not a store measurement), per-span PPG |
| `p_*` | an operating point. **No number appears in any body.** Every `p_*` is unfitted and belongs in `data/` with a written derivation, per the project rule |

D1 is named as a protocol rather than left as a word, so that "what D1 must supply" is a signature
rather than a description. **Its four `...` bodies are the only ellipses in this section, and they
are the Python idiom for a `Protocol` rather than an unwritten body** — D1 does not exist, so there
is nothing to write; what the protocol fixes is the interface every body below calls it through, and
every one of those call sites handles `store.stimulus_alignment is None`:

```python
class Alignment(Protocol):
    """D1, `stimulus_alignment`. Absent from the tree; named so the dependency is visible."""

    expected: Sequence            # one row per declared token: .text, .index, .column
    realised: Sequence            # (expected_token, word) for every token a column realised
    substitutions: Sequence       # (expected_token, word) where the column read something else
    insertions: Sequence          # words no expected token claims

    def structure_spans(self) -> Sequence[tuple[float, float]]: ...
    def run_for(self, tokens: Sequence[str]) -> Sequence: ...
    def omissions_for(self, tokens: Sequence[str]) -> Sequence[str]: ...
    def covers_sequence_twice(self, min_overlap: float) -> bool: ...
```

### The store types the bodies are written over

Verified against `src/senselab/audio/workflows/triage/nodes/preprocess.py` at this branch's tip on
2026-09-16; the block list is `preprocess.py:2945-2986`.

| name in a body | what it is | fields a body may read |
| --- | --- | --- |
| `words` | the `word` entities `consensus_transcript` names (entity at `preprocess.py:2436`; attributes at `consensus.py:391-416`) | `text`, `bracketed`, `outcome`, `sources`, `readings`, `timings` (per source), `onset_spread_s`, `offset_spread_s`, `temporal_uncertainty_s`, `variants`, `agreement`, `index`, and the entity's own `extent` |
| `consensus_transcript` | the whole-file stream (block `_consensus` at `:2409`, measurement at `:2456`, name at `:2460`) | `text`, `sources` (model id + resolved commit per source), `word_ids`, `role` |
| `spans` | the `span` entities — the amplitude/continuity/ASR loop at `preprocess.py:1875-1891`, the `gap` entity loop at `:1905-1920` | always `extent`, `signal`, `measure ∈ {amplitude, continuity, asr, gap}`, `merged_proposals`, `contains_clip`. **`peak_over_floor_db` and `k_db` exist only on `measure == "amplitude"`** and `continuity_cut_percentile` only on `continuity` (`_measure_fields`, `:1782-1787`); `corroborated_by` only when a later proposer overlapped |
| `energy_envelope` | `derivatives/energy_envelope.npz` (`_envelope` at `:1590`, savez `:1612`, name `:1621`) | npz `envelope_dbfs`, `floor_dbfs` — **one global value, `np.full_like`-broadcast, not a local floor**; entity attribute `sampling_rate` |
| `normalized_envelope` | the AGC'd envelope (savez `:1705`, name `:1714`) | the same two arrays |
| `continuity_trace` | `derivatives/continuity_trace.npz` (block `:2522`, savez `:2549`, name `:2554`) | npz **`continuity` only**, per sample in `[0, 1]`; entity attributes `sampling_rate`, `cut_level`, `cut_percentile`. **This is a spectral-stationarity trace already** — cosine similarity between consecutive log-magnitude spectra (`spectral_continuity/api.py:10`), fed the narrowband magnitude |
| `phonation_tracks` | `derivatives/phonation_tracks.npz` (fn `:1255`, savez `:1300`), hop 10 ms | `times_s`, `f0_hz`, `strength`, `formant_times_s`, `f1..f4_hz`, `f1..f4_bw_hz`. F0 on the pre-emphasised stream, formants on `plain` (`:1288-1291`) |
| `spectrogram_wideband` / `spectrogram_narrowband` | `derivatives/spectrogram_*.npz` (savez `:2507`) | npz `spectrogram` (power); attributes `win_length`, `hop_length`, `n_fft`. 5 ms / 20 ms window, 5 ms hop (`default.yaml:69-72`). **No `sampling_rate` attribute** — a body takes the working rate from the resample, which is why `band_power` has one as a parameter |
| `gammatone` | `derivatives/gammatone.npz` (block `:2568`, savez `:2584`) | `centre_frequencies_hz` (40 channels, 80–7800 Hz), `energy_db`; attribute `hop_s` = 0.005. **Read by no body here**: it is an ERB rebinning of the same short-time spectrum the two spectrograms carry |
| `ppg_posteriorgram` | writer `:774`, relative path `:803`, on `enhanced` | `posteriorgram[frame, phoneme]`, `phonemes` (40 ARPAbet incl. `<silent>`), `seconds_per_frame`. **Whole file only**, and read by no body here |
| `praat_features` | ~40 whole-file scalars on `enhanced` (fn `:1179`, measurement `:1223-1231`) | **the scalars are nested under one attribute, `features`**, beside `n_features` and the parameter set. Not re-poolable over an extent |
| `level` | whole-file, on `plain` (block `:2087`, name `:2098`) | `peak_dbfs`, `rms_dbfs`, `lufs`. **Uncalibrated** — no SPL reference exists anywhere in the graph |
| `silence` | YAMNet `Silence` per window (block `:2059`, name `:2079`) | `windows[{start, end, score, is_silence}]`, `threshold` |
| `span_hear` / `span_yamnet` | per-span classifier windows (`:2186`, `:2250`; attributes built at `:314-358`) | `span_id`, the window's own `extent`, `raw_scores`, `default_threshold`, `label_top_k`, `labelled`, `isolated_span`, and `labels` / `scores` when a membership rule exists. A refusal becomes an assertion carrying `unmeasured` (`_mark_unmeasured`, `:2174`) |
| `hear_scores` / `yamnet_scores` / `ast_scores` | the classifier's verbatim whole-file windows (`_scores` at `:1926`) | `start`, `end`, `label_scores` (every label, raw), `win_length`, `hop_length`. HeAR's eight labels are `Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear, Laugh, Speech` (`hear.py:133-142`); HeAR windows 2.0 s non-overlapping, YAMNet 0.96 s on a 0.48 s hop (`yamnet.py:142-143`), AST 10.24 s non-overlapping |
| `hear_windows` / `yamnet_windows` / `ast_windows` | the fold of a membership rule over those scores (`_windows` at `:1952`) | **all three are absent under the shipped config.** `load_label_membership` (`label_membership.py:55`) `config.require`s all three of `label_top_k`, `default_threshold` and `label_thresholds` — at `:70`, `:71` and `:73`, inside the return at `:69-75` — and `label_thresholds` is **null for yamnet, ast and hear alike** (`default.yaml:94`, `:98`, `:105`), so `ast` fails on two nulls. `_windows` calls it at `:1957`; the block runner catches `(ValueError, LookupError)` at `preprocess.py:2991` and records the block absent at `:2994`. `optional_label_membership` (`:78-102`) reads the same key with `config.get` and tolerates the null, which is why `span_hear` / `span_yamnet` *are* labelled and these are not. **No body below reads a `*_windows` derivative** |
| `enhanced_hear_scores` / `residual_hear_scores` (and `_yamnet_`, `_ast_`) | the same per stream (`_stream_classifier_scores` at `:2814`, measurement `:2837-2851`, `_stream_hear` `:2930`) | per-window scores plus `speech_overlap`, and the `_summary_all` / `_summary_speech_free` roll-ups (`_stream_classifier_summaries` at `:2859-2888`) |
| `enhanced_diarization` / `residual_diarization` | one measurement per stream in `diarization.streams` (`:2617`, blocks `:2630`, spliced `:2985`) — **shipped** | `speakers`, `n_speakers`, `n_segments`, `per_speaker_s`, `speech_s`, `overlap_s`, `max_concurrent_speakers`, and `derivatives/<stream>_diarization.npz` carrying `starts`/`ends`/`speakers`/`streams`. The two streams' counts are never summed |
| `residual` | the FRCRN subtraction (block `:2667`, name `:2778`) | `energy_fraction`, `enhanced_energy_fraction`, `gain_db`, `bands`, `speech_present`, `speech_coverage_fraction`, `n_consensus_words` |
| `squim` | one **assertion** per span, `verb: "measure"` (`_squim_for` at `:2133`, the assertion written at `:2162-2166`, invoked by `_squim` at `:2172`) | `stoi`, `pesq`, `si_sdr` over the span's extent — **or `unmeasured` and no scores at all** when SQUIM refuses (`:2160`). A body reading it must handle that |
| `disruptions_file` | on the **un-resampled** `recording` stream (block `:2106`, name `:2125`) | clipped runs, dropouts, discontinuities, DC, zero-crossing rate, `sampling_rate` |
| `clip_spans` | `span` entities with `family: CLIP_FAMILY` (`:712-715`, block `:1519`) | `extent`, `family`, `signal`, and one `clip_amplitude` measurement beside them (`:621`, attributes `:666-674`) carrying `unclipped_peak`, `unclipped_peak_time_s`, `unclipped_samples_n`, `edge_guard_samples`, `clip_spans_n` and the per-span levels |
| `hints` | `AudioHints` (`audio_hints.py:129`), handed to every branch | `may_contain` (`:149`), `targeted_speaker_count` (`:150`), `environment` (`:151`), `expected_speech`‡ (`:152`), `target_speaker` (`:153`), `metadata` (`:154`, a `dict[str, Any]`). **`metadata["task_token"]` is a key, not a field, and appears nowhere in `src/senselab`**‡ |
| `stream_extent` | ADMIT's `recording` stream extent, `(0.0, duration_s)` (`admit.py:95-98`), whose entity also carries `path` (`:101`) | the measured duration every *"was it done"* test needs — and, through `path`, the only in-store carrier of the declared family |

`store.id_of(name)` is the id of the measurement or derivative entity of that name, which is what a
proposal puts in `wasDerivedFrom`.

**`declared_duration_s` is a sidecar-consistency check, not a task-completion measurement.** Where a
body reads it, the finding is that the declaration and the recording disagree; which of the two is
wrong is a separate question, and 2,020 sidecars declare under a second. Its home is QUALITY's Q5.

### The operating points, all of them

```python
@dataclass(frozen=True)
class Params:
    """Every operating point these bodies owe. No number appears here or in any body."""

    p_smoothing_window_s: float
    p_peak_prominence_db: float
    p_trough_return_db: float
    p_event_min_s: float
    p_score_min: float
    p_breath_coverage_min: float
    p_voiced_strength_min: float
    p_voiced_fraction_min: float
    p_f0_spread_window_s: float
    p_f0_spread_max_semitones: float
    p_continuity_min: float
    p_production_min_s: float
    p_monotone_tolerance_semitones: float
    p_dominant_segment_min_fraction: float
    p_response_min_s: float
    p_pause_min_s: float
    p_run_gap_max_s: float
    p_breath_group_min_gap_s: float
    p_omission_score_max: float
    p_repeat_overlap_min: float
    p_echo_ngram_n: int
    p_echo_overlap_max: float
    p_verbatim_overlap_max: float
    p_coverage_min: float
    p_expected_lexical_max: int
    p_interval_max_s: float
    p_modulation_band_hz: tuple[float, float]
    p_rate_prominence_min: float
    p_train_min_s: float
    p_repeat_min_occurrences: int
    p_burst_window_ms: float
    p_place_centroid_bands_hz: dict[str, tuple[float, float]]
    p_place_margin_db: float
    p_effort_split_hz: float
    p_min_contrast_db: float
    p_tilt_max_db_per_octave: float
    p_level_min_dbfs: float
    p_gap_off_task_min_s: float
    p_normalise: Callable[[str], str]
    p_label_sets: dict[str, tuple[str, ...]]
```

### The mode selector, written once

```python
IN_FAMILY: dict[str, dict[str, Expectation]] = {
    "VOICE": VOICE_EXPECTATIONS,
    "SPEECH": SPEECH_EXPECTATIONS,
    "AIRWAY": AIRWAY_EXPECTATIONS,
    "DDK": DDK_EXPECTATIONS,
}

ALIGN: dict[str, Callable] = {
    "VOICE": align_voice,
    "SPEECH": align_speech,
    "AIRWAY": align_airway,
    "DDK": align_ddk,
}

DETECT: dict[str, Callable] = {
    "VOICE": detect_voice,
    "SPEECH": detect_speech,
    "AIRWAY": detect_airway,
    "DDK": detect_ddk,
}


def run_branch(branch: str, task_family: str | None, store, hints, params: Params) -> Result:
    """The whole of the mode decision. The declaration picks the mode; it never supplies the answer."""
    if task_family is not None and task_family in IN_FAMILY[branch]:
        return ALIGN[branch](task_family, store, hints, params)
    return DETECT[branch](store, params)
```

`task_family is None` — an undeclared or unreadable family — takes the out-of-family mode, which is
the safe arm: the branch annotates its speciality and concludes nothing. There is no third arm and
no fallback that guesses a family.

### The expectation, as data

```python
class Pattern(Enum):
    ORDERED_TOKENS = "ordered_tokens"
    FREE_RESPONSE = "free_response"
    ITEM_LIST = "item_list"
    NO_LEXICAL = "no_lexical"
    SUSTAINED = "sustained"
    GLIDE = "glide"
    EFFORT = "effort"
    PER_SENTENCE = "per_sentence"
    EVENT_SERIES = "event_series"
    EVENT_ALTERNATION = "event_alternation"
    SOUND_COVERAGE = "sound_coverage"
    SYLLABLE_TRAIN = "syllable_train"
    SYLLABLE_SEQUENCE = "syllable_sequence"


@dataclass(frozen=True)
class Expectation:
    """What the instruction asked for, as data. One row per in-family task."""

    pattern: Pattern
    tokens: tuple[str, ...] | None = None
    token_source: str | None = None
    expected_event_count: int | None = None
    declared_duration_s: float | None = None
    label_set: str | None = None
    sequence: tuple[str, ...] | None = None
    declared_direction: str | None = None
    declared_route: str | None = None
    route_from_index: bool = False
    relax_s: float | None = None
    repetition_allowed: bool | None = None
    repetition_from_category: bool = False
    lexical_separator: bool = False
    forbid_lexical: bool = False
    emit_filler: bool = True
    expect_inhale: bool = False
    contrast: bool = False
    timed_intervals: bool = False
    anti_pattern: str | None = None
    connected: bool = False
    unviable: tuple[tuple[str, str], ...] = ()
```

**Twenty-two fields, and every one of them was a difference between two functions before.** The
v1/v2 pairs are the clearest case: `maximum-phonation-time` against `-v2` is `expect_inhale`;
`free-speech` against `-v2` is `anti_pattern`; every `diadochokinesis` v1 against its v2 is
`expected_event_count` against `declared_duration_s`. Written as data, the pair cannot drift apart;
written as two functions, it repeatedly did.

### What spans each in-family task proposes

Every row is what the body below actually mints. The rule the table follows is stated once: **a span
is proposed where a measurement is taken over it that no existing entity already carries.** A
realised word already has a `word` entity with its own extent, `agreement` and per-source `timings`,
so a token gets no second span; a held vowel, a sentence, a cough and a train each carry a
measurement of their own, so each does.

| branch | task | spans proposed | what each extent is derived from |
| --- | --- | --- | --- |
| VOICE | `prolonged-vowel` | **2** — `count_in`, `task_extent` | `count_in` from the consensus `word` extents of the matched `one two three`; `task_extent` from the first and last voiced frame of `phonation_tracks` inside the qualifying amplitude span. **Two, because only the second is the voice measurement**: today every Praat scalar is taken over count-in plus silence plus vowel, which is the defect the split removes. `count_in` carries `excluded_from_measurement=True` |
| VOICE | `maximum-phonation-time` | **1** — `task_extent` | the voiced run, as above. The v1 inhale gets **no VOICE span**: under propose-only a branch mints only in its own family and an inhale is airway evidence, so `align_voice` records `inhale_expected_in_file` as a count and `detect_airway` — which runs on the same recording — is what proposes the span over it. Whether routing in fact selects AIRWAY on this family often enough for the hand-off to be routine is a routing-share question this document does not measure |
| VOICE | `maximum-phonation-time-v2` | **1** — `task_extent` | as v1, and no inhale is expected at all |
| VOICE | `glides-*`, `high-to-low` | **1** — `task_extent` | the tolerant-monotone run over `semitones(f0_hz)`, which is the sweep itself rather than the carrier span |
| VOICE | `cape-v-sentences`, `-v2` *(pending declaration)* | **n + 1** — one per sentence, plus `task_extent` | each sentence from `stimulus_alignment.structure_spans()`. **One per sentence because pooling across the six discards the instrument's design** — each loads a different phonatory condition, and today's whole-file `praat_features` is exactly that pooling |
| VOICE | `loudness`, `loudness-v2` *(pending declaration)* | **n**, one per realised token; v2 adds a `task_extent` over the pair | each token from its `word` extent. Per token because the effort measurement is per token; v2's `task_extent` spans the contrast, which is the measurement there |
| SPEECH | read text and passages | **1 + n** — `task_extent`, plus one `structure_*` per alignment unit, plus breath groups where `connected` | `task_extent` from the first and last realised token; `structure_*` from `stimulus_alignment.structure_spans()`. **No span per token** — a `word` entity already carries that ground, and one per token would mint ~8 × 13,705 extents on `harvard-sentences-list` alone with no measurement attached |
| SPEECH | `word-color-stroop` | as above | the answer sequence is the alignment's expected tokens; the structure units are the 15 colour items |
| SPEECH | free response (`free-speech*`, `story-recall*`, `cinderella-story`, `productive-vocabulary`, `picture-description*`, `open-response-questions`) | **1 + n** — `task_extent`, plus one per breath group where `connected` | `task_extent` from the hull of the `measure: "asr"` spans; breath groups from inter-word gaps, HeAR `Breathe` windows and `[breath]` tokens. Each breath group carries its own S4 measurement, which is why it is a span |
| SPEECH | `animal-fluency`, `random-item-generation*` | **1** — `task_extent` | the hull of the produced items. An item is one `word` entity, so no per-item span; `repeated_item` is a deviation over that word's extent |
| SPEECH | the eight non-`buttercup` `SYLLABLE_REPETITION` families | **0** | **Nothing, and that is the finding.** The expectation is that no lexical content occurs; a branch that proposed a speech span here would assert the opposite of what it measured |
| SPEECH | `diadochokinesis-buttercup`, `-v2-buttercup` | **1** — `task_extent` | the hull of the realised `buttercup` tokens |
| SPEECH | `loudness`, `loudness-v2` | **1** — `task_extent` | the hull of the realised `hey` tokens |
| AIRWAY | the counted families (`-cough`, `-v2-hardcough`, `fivebreaths`, both `threebreaths*`, `threequickbreaths`, `v2-threebreaths`, `breath-sounds`) | **n + 1** — **one per event**, plus `task_extent` over their hull | each event from `events_in_span`'s peak-prominence and trough-return walk over `energy_envelope`, derived from the carrier amplitude span it was found in. **One per event is the point**: `by_label` increments once per (span, label) pair (`airway.py:280`), so a 4 s span holding three coughs counts 1 today, and the count compared against the instruction's `expected_event_count` is the number of these spans |
| AIRWAY | `voluntary-cough` | **n + m + 1** — one per cough, one per breath, plus `task_extent` | coughs from `events_in_span`, breaths from the `Breathe`-scoring carrier spans. Both are expected, because the pattern is an alternation and material between coughs must be matched rather than scored off-task |
| AIRWAY | `-breath`, `-v2-breath` | **n + 1** — one per merged run of `Breathe` windows, plus `task_extent` | HeAR's raw 2.0 s windows, merged. Uncounted and durational, so the runs are the structure |
| DDK | every train and sequence family | **1** — `task_extent`, the train | the hull of the syllable onsets inside the carrier span, falling back to the carrier's extent. **An individual syllable is NOT a span**: rate, inter-onset interval variability and sequence collapse are statistics over the onset series, and one span per syllable would add ~30 per recording over 7,989 recordings carrying no measurement of their own. The onsets travel as a `counts` entry; a syllable that is not the one the sequence expected travels as a `syllable_sequence_mismatch` deviation with its own extent, which needs no span |
| DDK | `buttercup`, `-v2-buttercup` | **1** — `task_extent`, the train | the hull of the realised tokens |
| QUALITY | not a task | **0 or 1** — `occluded` only when the tilt finding fires | the stream extent, derived from `level`, `spectrogram_wideband` and `band_profile†`. The clip check proposes nothing: it contests PREPROCESS's own reading, which is an assertion beside a span |

**Where a task's spans cannot be delimited, nothing is minted and the result is `UNDETERMINED`.**
Three sites do this rather than proposing a span whose boundaries are guessed: `_speech_ordered`
when `stimulus_alignment` is absent and the expectation is per recording; `_voice_per_sentence` when
the same is true of the sentence boundaries; and `_speech_item_list` when the recording's own
category cannot be read, since the repetition rule inverts between categories. `align_ddk` returns
`False` with no span when no carrier clears the train minimum — that is a measurement, not an
absent instrument.

### The shared helpers, written once rather than in thirty bodies

```python
def lexical(words: Sequence) -> list:
    return [w for w in words if not w.bracketed]


def gaps(spans: Sequence) -> list:
    return [s for s in spans if s.measure == "gap"]


def amplitude_spans(spans: Sequence) -> list:
    return [s for s in spans if s.measure == "amplitude"]


def asr_spans(spans: Sequence) -> list:
    return [s for s in spans if s.measure == "asr"]


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and a[1] > b[0]


def duration(extent: tuple[float, float] | None) -> float:
    return 0.0 if extent is None else float(extent[1] - extent[0])


def hull(extents: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if not extents:
        return None
    return (min(s for s, _ in extents), max(e for _, e in extents))


def merge(extents: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(extents):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def touches_edge(extent: tuple[float, float], stream_extent: tuple[float, float]) -> bool:
    return extent[0] <= stream_extent[0] or extent[1] >= stream_extent[1]


def off_task(components: Sequence[Proposal], spans: Sequence, p_gap_off_task_min_s: float) -> list[Finding]:
    out: list[Finding] = []
    for g in gaps(spans):
        if duration(g.extent) < p_gap_off_task_min_s:
            continue
        if any(overlaps(g.extent, (c.start, c.end)) for c in components):
            continue
        out.append(deviation("off_task_extent", g.extent[0], g.extent[1], measure="gap"))
    return out


def declared_duration_count(store, declared: float | None) -> list[Finding]:
    if declared is None:
        return []
    return [count("declared_duration_s", round(duration(store.stream_extent), 2), declared)]
```

`off_task_extent` is emitted by the branch that owns the task; which verb carries it — `trim` per
`design.md:569` or `deviate` per `branch-conventions.md:108` — was unsettled. **Propose-only settles
the other half of that question**: `task_extent` is now a proposed span carrying `role:
"task_extent"`, not a `trim` payload, so `trim` is not the carrier of the extent either. This
document's reading is that `trim` has no remaining job; `off_task_extent` stays a `deviate` finding,
because off-task material is the absence of the branch's speciality rather than an instance of it,
and a branch does not mint spans over ground it is disclaiming.

#### Array helpers

```python
def envelope_slice(energy_envelope, extent: tuple[float, float]) -> tuple[np.ndarray, int]:
    sr = float(energy_envelope.sampling_rate)
    envelope = np.asarray(energy_envelope.envelope_dbfs, dtype=float)
    lo = max(0, int(round(extent[0] * sr)))
    hi = min(envelope.size, int(round(extent[1] * sr)))
    if hi <= lo:
        return np.empty(0, dtype=float), lo
    return envelope[lo:hi], lo


def trace_slice(continuity_trace, extent: tuple[float, float]) -> np.ndarray:
    sr = float(continuity_trace.sampling_rate)
    trace = np.asarray(continuity_trace.continuity, dtype=float)
    lo = max(0, int(round(extent[0] * sr)))
    hi = min(trace.size, int(round(extent[1] * sr)))
    return trace[lo:hi] if hi > lo else np.empty(0, dtype=float)


def boxcar(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or x.size == 0:
        return x
    width = min(width, x.size)
    return np.convolve(x, np.ones(width, dtype=float) / float(width), mode="same")


@dataclass
class TrackSlice:
    times_s: np.ndarray
    f0_hz: np.ndarray
    strength: np.ndarray
    voiced: np.ndarray
    hop_s: float


def track_slice(phonation_tracks, extent: tuple[float, float], p_voiced_strength_min: float) -> TrackSlice:
    times = np.asarray(phonation_tracks.times_s, dtype=float)
    inside = (times >= extent[0]) & (times < extent[1])
    hop = float(np.median(np.diff(times))) if times.size > 1 else 0.01
    strength = np.asarray(phonation_tracks.strength, dtype=float)[inside]
    return TrackSlice(
        times_s=times[inside],
        f0_hz=np.asarray(phonation_tracks.f0_hz, dtype=float)[inside],
        strength=strength,
        voiced=strength >= p_voiced_strength_min,
        hop_s=hop,
    )


def semitones(f0_hz: np.ndarray, ref_hz: float | None = None) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=float)
    voiced = np.isfinite(f0) & (f0 > 0.0)
    if not voiced.any():
        return np.full(f0.shape, np.nan)
    if ref_hz is None:
        ref_hz = float(np.median(f0[voiced]))
    out = np.full(f0.shape, np.nan)
    out[voiced] = 12.0 * np.log2(f0[voiced] / ref_hz)
    return out


def robust_spread(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    return float(np.percentile(v, 95.0) - np.percentile(v, 5.0))


def max_windowed_spread(values: np.ndarray, hop_s: float, window_s: float) -> float:
    v = np.asarray(values, dtype=float)
    width = max(2, int(round(window_s / hop_s))) if hop_s > 0.0 else 2
    if v.size <= width:
        return robust_spread(v)
    worst = 0.0
    for i in range(0, v.size - width + 1):
        worst = max(worst, robust_spread(v[i : i + width]))
    return worst


def longest_monotone_run(values: np.ndarray, tolerance: float) -> tuple[int, int, int] | None:
    """Longest run that never reverses by more than `tolerance`. Returns (first, last, sign)."""
    v = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(v))
    if finite.size < 2:
        return None
    best: tuple[int, int, int] | None = None
    for sign in (1, -1):
        run_start = int(finite[0])
        extreme = float(v[run_start])
        previous = run_start
        candidates: list[tuple[int, int, int]] = []
        for raw in finite[1:]:
            index = int(raw)
            value = float(v[index])
            if sign * (value - extreme) >= -tolerance:
                extreme = max(extreme, value) if sign > 0 else min(extreme, value)
            else:
                candidates.append((run_start, previous, sign))
                run_start, extreme = index, value
            previous = index
        candidates.append((run_start, previous, sign))
        for candidate in candidates:
            if best is None or (candidate[1] - candidate[0]) > (best[1] - best[0]):
                best = candidate
    return best


def band_power(spectrogram_block, sampling_rate: float, extent: tuple[float, float],
               lo_hz: float, hi_hz: float) -> float:
    power = np.asarray(spectrogram_block.spectrogram, dtype=float)
    freqs = np.fft.rfftfreq(int(spectrogram_block.n_fft), d=1.0 / sampling_rate)
    bins = (freqs >= lo_hz) & (freqs < hi_hz)
    hop_s = float(spectrogram_block.hop_length) / sampling_rate
    first = max(0, int(extent[0] / hop_s))
    last = min(power.shape[1], int(np.ceil(extent[1] / hop_s)))
    if last <= first or not bins.any():
        return float("nan")
    return float(power[np.ix_(bins, np.arange(first, last))].sum())


def spectral_balance_db(spectrogram_block, sampling_rate: float, extent: tuple[float, float], split_hz: float) -> float:
    low = band_power(spectrogram_block, sampling_rate, extent, 0.0, split_hz)
    high = band_power(spectrogram_block, sampling_rate, extent, split_hz, sampling_rate / 2.0)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0.0:
        return float("nan")
    return float(10.0 * np.log10((high + 1e-20) / low))


def peak_over_floor_db(energy_envelope, extent: tuple[float, float]) -> float:
    e, _ = envelope_slice(energy_envelope, extent)
    if e.size == 0:
        return float("nan")
    return float(e.max() - float(energy_envelope.floor_dbfs))


def acquisition_covariates(store, extent: tuple[float, float]) -> dict:
    return {
        "file_peak_dbfs": store.level.peak_dbfs,
        "file_rms_dbfs": store.level.rms_dbfs,
        "file_lufs": store.level.lufs,
        "contains_clip": any(s.contains_clip for s in store.spans if overlaps(s.extent, extent)),
        "disruptions": store.disruptions_file.summary,
        "uncalibrated": True,
    }
```

#### The three instruments

```python
def events_in_span(energy_envelope, span, params: Params) -> list[tuple[float, float]]:
    """A5/A6 and D3's onsets. Multiple maxima inside one span become separate events."""
    sr = float(energy_envelope.sampling_rate)
    raw, offset = envelope_slice(energy_envelope, span.extent)
    if raw.size < 3:
        return []
    e = boxcar(raw, max(1, int(round(params.p_smoothing_window_s * sr))))
    floor_dbfs = float(energy_envelope.floor_dbfs)

    events: list[tuple[float, float]] = []
    for i in range(1, e.size - 1):
        if not (e[i] >= e[i - 1] and e[i] > e[i + 1]):
            continue
        if e[i] - floor_dbfs < params.p_peak_prominence_db:
            continue
        j, left_min = i - 1, float(e[i])
        while j >= 0 and e[j] < e[i]:
            left_min = min(left_min, float(e[j]))
            j -= 1
        k, right_min = i + 1, float(e[i])
        while k < e.size and e[k] < e[i]:
            right_min = min(right_min, float(e[k]))
            k += 1
        if float(e[i]) - max(left_min, right_min) < params.p_peak_prominence_db:
            continue
        target = float(e[i]) - params.p_trough_return_db
        onset = i
        while onset > 0 and e[onset - 1] > target:
            onset -= 1
        tail = i
        while tail < e.size - 1 and e[tail + 1] > target:
            tail += 1
        start = (offset + onset) / sr
        end = (offset + tail) / sr
        if end - start >= params.p_event_min_s:
            events.append((start, end))
    return merge(events)


def sounds_like(span, span_hear, span_yamnet, label_set: Sequence[str], p_score_min: float) -> bool:
    """`raw_scores` is always written; `labels` is a top-K decision over it (default.yaml:91-107)."""
    for window in list(span_hear) + list(span_yamnet):
        if window.span_id != span.id:
            continue
        for label in label_set:
            if float(window.raw_scores.get(label, 0.0)) >= p_score_min:
                return True
    return False


def train_rate_hz(energy_envelope, extent: tuple[float, float], params: Params) -> float | None:
    """The modulation spectrum of the envelope. Not Praat's `extract_speech_rate` — see the notes."""
    sr = float(energy_envelope.sampling_rate)
    e, _ = envelope_slice(energy_envelope, extent)
    if e.size < 8:
        return None
    windowed = (e - e.mean()) * np.hanning(e.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(e.size, d=1.0 / sr)
    lo, hi = params.p_modulation_band_hz
    band = (freqs >= lo) & (freqs <= hi)
    if not band.any():
        return None
    k = int(np.argmax(np.where(band, spectrum, 0.0)))
    background = float(spectrum[band].mean())
    if background <= 0.0 or float(spectrum[k]) / background < params.p_rate_prominence_min:
        return None
    return float(freqs[k])
```

#### Lexical arithmetic

```python
def ordered_run(expected: Sequence[str], words: Sequence, normalise: Callable[[str], str]):
    """The D1-free fallback: a greedy left-to-right ordered match."""
    matched: list[tuple[str, object]] = []
    omissions: list[str] = []
    cursor = 0
    for token in expected:
        hit = None
        for k in range(cursor, len(words)):
            if normalise(words[k].text) == normalise(token):
                hit = k
                break
        if hit is None:
            omissions.append(token)
        else:
            matched.append((token, words[hit]))
            cursor = hit + 1
    return matched, omissions


def ngram_echo_fraction(source_tokens: Sequence[str], produced_tokens: Sequence[str], n: int) -> float:
    if len(source_tokens) < n or len(produced_tokens) < n:
        return 0.0
    source = {tuple(source_tokens[i : i + n]) for i in range(len(source_tokens) - n + 1)}
    produced = {tuple(produced_tokens[i : i + n]) for i in range(len(produced_tokens) - n + 1)}
    if not source:
        return 0.0
    return len(source & produced) / len(source)


def content_coverage(source_tokens: Sequence[str], produced_tokens: Sequence[str]) -> float:
    source = set(source_tokens)
    if not source:
        return 0.0
    return len(source & set(produced_tokens)) / len(source)


def lexical_runs(words: Sequence, max_gap_s: float) -> list[tuple[float, float]]:
    """Consecutive lexical words separated by no more than `max_gap_s`, as one extent each.

    NOT `merge` over the word extents: `merge` joins only touching or overlapping intervals, and
    ordinary speech has a gap between every pair of words, so `merge` would return one extent per
    word. This is the same grouping SPEECH already does today over the consensus word timings
    (`group_extents_into_runs`, `speech.py:573`).
    """
    runs: list[tuple[float, float]] = []
    for word in words:
        if runs and word.extent[0] - runs[-1][1] <= max_gap_s:
            runs[-1] = (runs[-1][0], max(runs[-1][1], word.extent[1]))
        else:
            runs.append((word.extent[0], word.extent[1]))
    return runs


def inter_word_gaps(words: Sequence, min_gap_s: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for previous, current in zip(words, words[1:]):
        if current.extent[0] - previous.extent[1] >= min_gap_s:
            out.append((previous.extent[1], current.extent[0]))
    return out


def group_by_breaks(words: Sequence, breaks: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not words:
        return []
    groups: list[tuple[float, float]] = []
    start = words[0].extent[0]
    for previous, current in zip(words, words[1:]):
        between = (previous.extent[1], max(current.extent[0], previous.extent[1] + 1e-9))
        if any(overlaps(b, between) for b in breaks):
            groups.append((start, previous.extent[1]))
            start = current.extent[0]
    groups.append((start, words[-1].extent[1]))
    return groups


def breath_group_extents(store, params: Params) -> list[tuple[float, float]]:
    words = lexical(store.words)
    if not words:
        return []
    breaks = inter_word_gaps(words, params.p_breath_group_min_gap_s)
    for window in store.hear_scores:
        if float(window.label_scores.get("Breathe", 0.0)) >= params.p_score_min:
            breaks.append((window.start, window.end))
    for word in store.words:
        if word.bracketed and word.text == "[breath]":
            breaks.append(word.extent)
    return group_by_breaks(words, merge(breaks))
```

---

### VOICE — `align_voice` and `detect_voice`

```python
VOICE_EXPECTATIONS: dict[str, Expectation] = {
    "prolonged-vowel": Expectation(
        pattern=Pattern.SUSTAINED,
        tokens=("one", "two", "three"),
        token_source="instructions",
        declared_duration_s=12.0,
        lexical_separator=True,
    ),
    "maximum-phonation-time": Expectation(pattern=Pattern.SUSTAINED, forbid_lexical=True, expect_inhale=True),
    "maximum-phonation-time-v2": Expectation(pattern=Pattern.SUSTAINED, forbid_lexical=True, expect_inhale=False),
    "glides-low-to-high": Expectation(pattern=Pattern.GLIDE, declared_direction="up"),
    "glides-high-to-low": Expectation(pattern=Pattern.GLIDE, declared_direction="down"),
    "high-to-low": Expectation(pattern=Pattern.GLIDE, declared_direction="down"),
}

VOICE_EXPECTATIONS_PENDING_DECLARATION: dict[str, Expectation] = {
    "loudness": Expectation(
        pattern=Pattern.EFFORT,
        tokens=("hey",),
        expected_event_count=3,
        unviable=(("effort_absolute", "`level` is uncalibrated and no SPL reference exists in the graph"),),
    ),
    "loudness-v2": Expectation(pattern=Pattern.EFFORT, tokens=("hey",), expected_event_count=2, contrast=True),
    "cape-v-sentences": Expectation(pattern=Pattern.PER_SENTENCE, token_source="stimulus_text"),
    "cape-v-sentences-v2": Expectation(pattern=Pattern.PER_SENTENCE, token_source="stimulus_text"),
}
```

**V1's stationarity qualifier, one body and three callers** — `align_voice`'s sustained matcher, its
glide matcher's voicing precondition, and `detect_voice`. It is a **fit, not an estimator**:
`continuity_trace` is already a spectral-stationarity trace, and F0 spread is a statistic of `f0_hz`
at a 10 ms hop. It is also what stops V4 computing perturbation over consonants and pauses on the
22,277 recordings VOICE is routed to without a voice family.

```python
def qualifying_phonation(store, expectation: Expectation, params: Params) -> list[tuple[object, TrackSlice]]:
    """V1's stationarity qualifier, over PREPROCESS's amplitude spans. One body, three callers.

    The subject is `measure == "amplitude"`, not `family == "phonation"`. Under propose-only VOICE
    reads the amplitude spans as evidence and mints its own `family: "voice"` span over what
    qualifies, so it no longer waits for a phonation span that nothing proposes.
    """
    lex = lexical(store.words)
    out: list[tuple[object, TrackSlice]] = []
    for span in amplitude_spans(store.spans):
        if duration(span.extent) < params.p_production_min_s:
            continue
        if expectation.lexical_separator and any(overlaps(w.extent, span.extent) for w in lex):
            continue
        f = track_slice(store.phonation_tracks, span.extent, params.p_voiced_strength_min)
        if f.strength.size == 0:
            continue
        voiced_fraction = float(f.voiced.mean())
        pitch = semitones(np.where(f.voiced, f.f0_hz, np.nan))
        spread = max_windowed_spread(pitch, f.hop_s, params.p_f0_spread_window_s)
        trace = trace_slice(store.continuity_trace, span.extent)
        stationarity = float(np.median(trace)) if trace.size else 0.0
        if (
            voiced_fraction >= params.p_voiced_fraction_min
            and spread <= params.p_f0_spread_max_semitones
            and stationarity >= params.p_continuity_min
        ):
            out.append((span, f))
    return out


def voiced_extent(span, f: TrackSlice) -> tuple[float, float]:
    """The production's own boundaries: the first and last voiced frame inside the carrier span."""
    inside = f.times_s[f.voiced]
    if inside.size == 0:
        return span.extent
    return (float(inside.min()), float(inside.max()) + f.hop_s)
```

```python
def align_voice(task_family: str, store, hints, params: Params) -> Result:
    expectation = VOICE_EXPECTATIONS.get(task_family)
    assert expectation is not None, f"{task_family} is not a VOICE family; the caller owes detect_voice"

    if expectation.pattern is Pattern.SUSTAINED:
        return _voice_sustained(expectation, store, hints, params)
    if expectation.pattern is Pattern.GLIDE:
        return _voice_glide(expectation, store, params)
    if expectation.pattern is Pattern.EFFORT:
        return _voice_effort(expectation, store, params)
    if expectation.pattern is Pattern.PER_SENTENCE:
        return _voice_per_sentence(expectation, store, params)
    raise NotImplementedError(expectation.pattern)
```

```python
def _voice_sustained(expectation: Expectation, store, hints, params: Params) -> Result:
    """Spans proposed: `count_in` where the instruction prescribes one, and `task_extent`.

    Two for `prolonged-vowel`, one for `maximum-phonation-time` and `-v2`. The count-in gets its
    own span precisely because it must be excluded from the vowel's measurement window: every
    Praat scalar today is taken over count-in plus silence plus vowel.
    """
    components: list[Proposal] = []
    findings: list[Finding] = []
    count_in_found = expectation.tokens is None

    if expectation.tokens is not None:
        alignment = store.stimulus_alignment
        if alignment is None:
            matched, omissions = ordered_run(expectation.tokens, lexical(store.words), params.p_normalise)
        else:
            matched = [(t, w) for t, w in alignment.run_for(expectation.tokens)]
            omissions = list(alignment.omissions_for(expectation.tokens))
        if matched:
            count_in_found = True
            components.append(
                voice_span(
                    "count_in",
                    (matched[0][1].extent[0], matched[-1][1].extent[1]),
                    store.id_of("consensus_transcript"),
                    *(w.id for _, w in matched),
                    tokens=[t for t, _ in matched],
                    excluded_from_measurement=True,
                )
            )
        for token in omissions:
            findings.append(deviation("omission", None, None, expected=token))

    candidates = sorted(
        qualifying_phonation(store, expectation, params), key=lambda pair: duration(pair[0].extent), reverse=True
    )
    if not candidates:
        findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
        return Result(False, components, findings)

    carrier, track = candidates[0]
    extent = voiced_extent(carrier, track)
    components.append(
        voice_span(
            "task_extent",
            extent,
            carrier.id,
            store.id_of("phonation_tracks"),
            store.id_of("continuity_trace"),
            production="sustained",
            support_frames=int(track.voiced.sum()),
            carrier_extent=list(carrier.extent),
        )
    )
    findings.append(measured("phonation_s", extent[0], extent[1], round(duration(extent), 3),
                             support_frames=int(track.voiced.sum())))
    if touches_edge(extent, store.stream_extent):
        findings.append(deviation("truncation", extent[0], extent[1]))
    for extra, extra_track in candidates[1:]:
        findings.append(
            deviation(
                "repeat_attempt",
                *voiced_extent(extra, extra_track),
                reading="three_attempts_inside_one_recording",
            )
        )

    if expectation.expect_inhale:
        # v1 places the deep inhale before the record tap is mentioned, so an audible inhale may be
        # inside the file. VOICE proposes NO span for it: under propose-only a branch mints only in
        # its own family, and an inhale is airway evidence. `detect_airway` runs on this same
        # recording, wherever routing selects it — and proposes it there.
        findings.append(count("inhale_expected_in_file", True, None))

    if expectation.forbid_lexical:
        for word in lexical(store.words):
            findings.append(deviation("off_task_extent", word.extent[0], word.extent[1], text=word.text))

    if hints is not None:
        token = hints.metadata.get("task_token")            # ‡ written by no code under src/senselab
        if token is not None:
            findings.append(count("task_index", token.rsplit("-", 1)[-1], None))

    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(count_in_found, components, findings)
```

```python
def _voice_glide(expectation: Expectation, store, params: Params) -> Result:
    """Spans proposed: one, `task_extent`, over the sweep."""
    best: tuple[object, TrackSlice, int, float, tuple[float, float]] | None = None
    for span in amplitude_spans(store.spans):
        if duration(span.extent) < params.p_production_min_s:
            continue
        f = track_slice(store.phonation_tracks, span.extent, params.p_voiced_strength_min)
        if f.strength.size == 0 or float(f.voiced.mean()) < params.p_voiced_fraction_min:
            continue
        pitch = semitones(np.where(f.voiced, f.f0_hz, np.nan))
        run = longest_monotone_run(pitch, params.p_monotone_tolerance_semitones)
        if run is None:
            continue
        first, last, sign = run
        sweep = (float(f.times_s[first]), float(f.times_s[last]) + f.hop_s)
        if duration(sweep) / max(duration(span.extent), 1e-9) < params.p_dominant_segment_min_fraction:
            continue
        if best is None or duration(sweep) > duration(best[4]):
            best = (span, f, sign, abs(float(pitch[last] - pitch[first])), sweep)

    if best is None:
        return Result(False, [], off_task([], store.spans, params.p_gap_off_task_min_s))

    span, f, sign, extent_semitones, sweep = best
    measured_direction = "up" if sign > 0 else "down"
    components = [
        voice_span(
            "task_extent",
            sweep,
            span.id,
            store.id_of("phonation_tracks"),
            production="glide",
            direction=measured_direction,
            support_frames=int(f.voiced.sum()),
        )
    ]
    findings = [measured("glide_extent_semitones", sweep[0], sweep[1], round(extent_semitones, 2))]
    if measured_direction != expectation.declared_direction:
        findings.append(
            deviation(
                "sweep_direction_mismatch",
                sweep[0],
                sweep[1],
                declared=expectation.declared_direction,
                measured=measured_direction,
                extent_semitones=round(extent_semitones, 2),
            )
        )
    if touches_edge(sweep, store.stream_extent):
        findings.append(deviation("truncation", sweep[0], sweep[1]))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(True, components, findings)
```

```python
def _voice_effort(expectation: Expectation, store, params: Params) -> Result:
    """Spans proposed: one per realised token, because the effort measurement is per token."""
    assert expectation.tokens is not None and len(expectation.tokens) == 1
    token = params.p_normalise(expectation.tokens[0])
    hits = [w for w in lexical(store.words) if params.p_normalise(w.text) == token]
    evidence = (store.id_of("energy_envelope"), store.id_of("spectrogram_wideband"))

    levels: list[float] = []
    balances: list[float] = []
    components: list[Proposal] = []
    findings: list[Finding] = [count("expected_event_count", len(hits), expectation.expected_event_count)]
    for index, word in enumerate(hits):
        over_floor = peak_over_floor_db(store.energy_envelope, word.extent)
        balance = spectral_balance_db(
            store.spectrogram_wideband, store.sampling_rate, word.extent, params.p_effort_split_hz
        )
        levels.append(over_floor)
        balances.append(balance)
        role = ("normal", "loud")[index] if expectation.contrast and index < 2 else "token_%d" % index
        components.append(voice_span(role, word.extent, word.id, *evidence, token=word.text, index=index))
        findings.append(
            measured(
                "token_peak_over_floor_db",
                word.extent[0],
                word.extent[1],
                round(over_floor, 2),
                index=index,
                **acquisition_covariates(store, word.extent),
            )
        )
        findings.append(
            measured("token_spectral_balance_db", word.extent[0], word.extent[1], round(balance, 2), index=index)
        )

    if not expectation.contrast:
        # v1's "maximal effort": NOT SEPARABLE BY THIS DESIGN. No within-recording contrast and no
        # SPL reference, so what is emitted is a measurement with its covariates, never a verdict.
        for name, why in expectation.unviable:
            findings.append(unviable(name, why))
        findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
        return Result(len(hits) > 0, components, findings)

    if len(hits) < 2:
        findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
        return Result(False, components, findings)

    # v2: a within-recording difference needs no norm and no calibration.
    contrast_db = levels[1] - levels[0]
    span = (hits[0].extent[0], hits[1].extent[1])
    components.append(voice_span("task_extent", span, hits[0].id, hits[1].id, *evidence, production="effort_contrast"))
    findings.append(measured("effort_contrast_db", span[0], span[1], round(contrast_db, 2)))
    findings.append(
        measured("effort_contrast_balance_db", span[0], span[1], round(balances[1] - balances[0], 2))
    )
    findings.append(count("contrast_clears_p_min_contrast_db", abs(contrast_db) >= params.p_min_contrast_db, None))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(True, components, findings)
```

```python
def _voice_per_sentence(expectation: Expectation, store, params: Params) -> Result:
    """Spans proposed: one per sentence, plus `task_extent` over their hull.

    Pooling across the six sentences discards the instrument's design — each loads a different
    phonatory condition — which is exactly what today's whole-file `praat_features` does.
    """
    alignment = store.stimulus_alignment
    if alignment is None:
        # The boundaries cannot be placed, so no span is proposed and nothing is claimed.
        return Result(UNDETERMINED, [], [unviable("per_sentence_extents", "stimulus_alignment (D1) is absent")])
    sentences = list(alignment.structure_spans())
    if not sentences:
        return Result(UNDETERMINED, [], [])

    tracks = store.phonation_tracks
    times = np.asarray(tracks.times_s, dtype=float)
    components: list[Proposal] = []
    findings: list[Finding] = []
    for index, extent in enumerate(sentences):
        f = track_slice(tracks, extent, params.p_voiced_strength_min)
        components.append(
            voice_span(
                "sentence_%d" % index,
                extent,
                store.id_of("stimulus_alignment"),
                store.id_of("phonation_tracks"),
                sentence_index=index,
                support_frames=int(f.voiced.sum()),
            )
        )
        inside = (times >= extent[0]) & (times < extent[1])
        if not f.voiced.any():
            findings.append(measured("sentence_voice_quality", extent[0], extent[1], None, support_frames=0))
            continue
        for column_name in ("f0_hz", "hnr_db", "cpps_db", "rms_dbfs"):       # the last three are D2 †
            column = getattr(tracks, column_name, None)
            if column is None:
                findings.append(
                    unviable("sentence_%s" % column_name, "phonation_tracks.%s (D2) is absent" % column_name)
                )
                continue
            values = np.asarray(column, dtype=float)[inside][f.voiced]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            findings.append(
                measured(
                    "sentence_%s_over_voiced_frames" % column_name,
                    extent[0],
                    extent[1],
                    round(float(np.median(values)), 3),
                    sentence=index,
                    support_frames=int(values.size),
                )
            )
    whole = hull(sentences)
    assert whole is not None
    components.append(
        voice_span("task_extent", whole, store.id_of("stimulus_alignment"), sentences_n=len(sentences))
    )
    for expected_token, word in alignment.substitutions:
        findings.append(
            deviation("stimulus_mismatch", word.extent[0], word.extent[1],
                      expected=expected_token.text, read=word.text)
        )
    return Result(True, components, findings)
```

```python
def detect_voice(store, params: Params) -> Result:
    """Out of family. Proposes a `family: "voice"` span over every sustained voiced region it finds.

    Task-agnostic by construction, and it shares `qualifying_phonation` with the in-family mode —
    the same V1 qualifier, the same amplitude spans, the same `phonation_tracks`. What differs is
    only that no expectation is consulted and no task is evaluated: `done` is UNDETERMINED, always.
    This is the mode that runs on most of what VOICE is handed — 22,277 routed recordings against
    8,306 declaring a voice family.
    """
    neutral = Expectation(pattern=Pattern.SUSTAINED)       # no lexical separator: mark what is there
    qualifying = qualifying_phonation(store, neutral, params)
    qualifying_ids = {span.id for span, _ in qualifying}

    components: list[Proposal] = []
    findings: list[Finding] = []
    for carrier, track in qualifying:
        extent = voiced_extent(carrier, track)
        components.append(
            voice_span(
                "phonation",
                extent,
                carrier.id,
                store.id_of("phonation_tracks"),
                store.id_of("continuity_trace"),
                production="sustained",
                support_frames=int(track.voiced.sum()),
                carrier_extent=list(carrier.extent),
                evaluates_no_task=True,
            )
        )
        findings.append(
            measured("phonation_s", extent[0], extent[1], round(duration(extent), 3),
                     support_frames=int(track.voiced.sum()))
        )
    for span in store.spans:
        if span.label == "phonation" and span.id not in qualifying_ids:
            findings.append(contest(span.id, span.extent, "phonation", "fails_the_stationarity_qualifier"))
    findings.append(count("phonation_spans", len(components), None))
    return Result(UNDETERMINED, components, findings)
```

`span.label` is the ruleset-written label the owner's 2026-09-15 decision introduced — a fired rule
may stamp or refine a span's label. Contesting one is an assertion beside it, so propose-only does
not touch that path.

---

### SPEECH — `align_speech` and `detect_speech`

```python
SPEECH_EXPECTATIONS: dict[str, Expectation] = {
    "harvard-sentences-list": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
    "cape-v-sentences": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
    "cape-v-sentences-v2": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
    "rainbow-passage": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text", connected=True),
    "caterpillar-passage": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text", connected=True),
    "word-color-stroop": Expectation(
        pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text", declared_duration_s=75.0, emit_filler=False
    ),
    "loudness": Expectation(pattern=Pattern.ORDERED_TOKENS, tokens=("hey", "hey", "hey"), expected_event_count=3),
    "loudness-v2": Expectation(pattern=Pattern.ORDERED_TOKENS, tokens=("hey", "hey"), expected_event_count=2),
    "diadochokinesis-buttercup": Expectation(
        pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",) * 10, expected_event_count=10, emit_filler=False
    ),
    "diadochokinesis-v2-buttercup": Expectation(
        pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",), declared_duration_s=5.0, emit_filler=False
    ),
    "free-speech": Expectation(
        pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text", anti_pattern="verbatim_prompt"
    ),
    "free-speech-v2": Expectation(pattern=Pattern.FREE_RESPONSE, declared_duration_s=30.0),
    "story-recall": Expectation(
        pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text", anti_pattern="verbatim_source"
    ),
    "story-recall-v2": Expectation(
        pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text", anti_pattern="verbatim_source"
    ),
    "cinderella-story": Expectation(
        pattern=Pattern.FREE_RESPONSE,
        unviable=(("source_overlap", "`stimulus_text` is empty on all 258; the source is a physical storybook"),),
    ),
    "productive-vocabulary": Expectation(
        pattern=Pattern.FREE_RESPONSE,
        token_source="stimulus_text",
        unviable=(("defines_its_cue", "a lexicon or a text model, branch-local, and no waveform"),),
    ),
    "picture-description": Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
    "picture-description-option1": Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
    "picture-description-option2": Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
    "open-response-questions": Expectation(
        pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text", declared_duration_s=30.0, connected=True
    ),
    "animal-fluency": Expectation(
        pattern=Pattern.ITEM_LIST,
        declared_duration_s=60.0,
        repetition_allowed=False,
        unviable=(("category_membership", "a lexicon or a text embedding, one consumer, no waveform"),),
    ),
    "random-item-generation": Expectation(
        pattern=Pattern.ITEM_LIST,
        repetition_from_category=True,
        unviable=(("category_membership", "a lexicon or a text embedding, one consumer, no waveform"),),
    ),
    "random-item-generation-v2": Expectation(
        pattern=Pattern.ITEM_LIST,
        repetition_from_category=True,
        unviable=(("category_membership", "a lexicon or a text embedding, one consumer, no waveform"),),
    ),
}
for _family in (
    "diadochokinesis-pa",
    "diadochokinesis-ta",
    "diadochokinesis-ka",
    "diadochokinesis-pataka",
    "diadochokinesis-v2-puh",
    "diadochokinesis-v2-tuh",
    "diadochokinesis-v2-kuh",
    "diadochokinesis-v2-puhtuhkuh",
):
    SPEECH_EXPECTATIONS[_family] = Expectation(pattern=Pattern.NO_LEXICAL)
```

31 rows: `LEXICAL_SPEECH` (21) ∪ `SYLLABLE_REPETITION` (10), which is `reference_family_set.SPEECH`
(`default.yaml:241`) and `SPEECH_ELICITING` (`families.py:58`).

```python
def align_speech(task_family: str, store, hints, params: Params) -> Result:
    expectation = SPEECH_EXPECTATIONS.get(task_family)
    assert expectation is not None, f"{task_family} is not a SPEECH family; the caller owes detect_speech"

    if expectation.pattern is Pattern.ORDERED_TOKENS:
        return _speech_ordered(expectation, store, params)
    if expectation.pattern is Pattern.FREE_RESPONSE:
        return _speech_free_response(expectation, store, params)
    if expectation.pattern is Pattern.ITEM_LIST:
        return _speech_item_list(expectation, store, hints, params)
    if expectation.pattern is Pattern.NO_LEXICAL:
        return _speech_no_lexical(store, params)
    raise NotImplementedError(expectation.pattern)
```

```python
def _speech_ordered(expectation: Expectation, store, params: Params) -> Result:
    """Spans proposed: `task_extent`, plus one per structure unit the alignment yields (a sentence,
    a Stroop item), plus the breath groups where the family is `connected`.

    **No span per token.** A realised token already has a `word` entity carrying its own extent,
    `agreement` and per-source `timings`; minting a second entity over the same ground would
    duplicate ~13,705 x 8 extents on `harvard-sentences-list` alone and add no measurement.
    """
    words = lexical(store.words)
    alignment = store.stimulus_alignment

    if expectation.tokens is not None:
        matched, omissions = ordered_run(list(expectation.tokens), words, params.p_normalise)
        matched_ids = {id(word) for _, word in matched}
        substitutions: list[tuple[object, object]] = []
        insertions = [w for w in words if id(w) not in matched_ids]
        structure: list[tuple[float, float]] = []
        repeated = False
    elif alignment is None:
        # D1 is absent and the expectation is per recording, so no extent can be placed: propose
        # nothing rather than a span whose boundaries are guessed.
        return Result(
            UNDETERMINED,
            [],
            [
                unviable(
                    "expected_token_sequence",
                    "stimulus_alignment (D1) is absent; the transcript alone cannot say what was expected",
                )
            ],
        )
    else:
        matched = [(t, w) for t, w in alignment.realised]
        omissions = [t.text for t in alignment.expected if t.column is None]
        substitutions = list(alignment.substitutions)
        insertions = list(alignment.insertions)
        structure = list(alignment.structure_spans())
        repeated = alignment.covers_sequence_twice(params.p_repeat_overlap_min)

    components: list[Proposal] = []
    findings: list[Finding] = []
    if matched:
        read_extent = (matched[0][1].extent[0], matched[-1][1].extent[1])
        components.append(
            speech_span(
                "task_extent",
                read_extent,
                store.id_of("consensus_transcript"),
                *(w.id for _, w in matched),
                words_n=len(matched),
                expected_n=len(matched) + len(omissions),
            )
        )
        for index, extent in enumerate(structure):
            components.append(
                speech_span(
                    "structure_%d" % index,
                    extent,
                    store.id_of("stimulus_alignment"),
                    store.id_of("consensus_transcript"),
                    structure_index=index,
                )
            )
        if touches_edge(read_extent, store.stream_extent):
            findings.append(deviation("truncation", read_extent[0], read_extent[1]))
        if repeated:
            findings.append(deviation("repeat_reading", read_extent[0], read_extent[1]))

    for expected_token, word in substitutions:
        findings.append(
            deviation(
                "stimulus_mismatch",
                word.extent[0],
                word.extent[1],
                expected=getattr(expected_token, "text", expected_token),
                read=word.text,
                agreement=word.agreement,
                variants=word.variants,
            )
        )
    for word in insertions:
        findings.append(deviation("stimulus_mismatch", word.extent[0], word.extent[1], expected=None, read=word.text))
    for token in omissions:
        # A TENTH deviation type, extent-free by nature: a skip-arc-free aligner assigns every
        # stimulus word an interval, so an omission surfaces only as a low acoustic score.
        findings.append(
            deviation("omission", None, None, expected=token, acoustic_score_max=params.p_omission_score_max)
        )

    if expectation.emit_filler:
        for word in store.words:
            if word.bracketed and word.text != "[breath]":    # `[breath]` is S4's, not a disfluency
                findings.append(deviation("filler", word.extent[0], word.extent[1], text=word.text))

    if expectation.connected:
        for index, extent in enumerate(breath_group_extents(store, params)):
            components.append(
                speech_span(
                    "breath_group_%d" % index,
                    extent,
                    store.id_of("consensus_transcript"),
                    store.id_of("hear_scores"),
                    group_index=index,
                )
            )

    if expectation.expected_event_count is not None:
        findings.append(count("expected_event_count", len(matched), expectation.expected_event_count))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(bool(matched) and not omissions, components, findings)
```

```python
def _speech_free_response(expectation: Expectation, store, params: Params) -> Result:
    """Spans proposed: `task_extent` over the hull of the ASR spans, plus one per breath group
    where the family is `connected` — each carrying its own S4 measurement."""
    runs = asr_spans(store.spans)
    response = hull([s.extent for s in runs])
    components: list[Proposal] = []
    findings: list[Finding] = []
    if response is not None:
        components.append(
            speech_span("task_extent", response, *(s.id for s in runs), words_n=len(lexical(store.words)))
        )
    done: Done = response is not None and duration(response) >= params.p_response_min_s

    if expectation.connected and response is not None and duration(response) > 0.0:
        words = lexical(store.words)
        pauses = inter_word_gaps(words, params.p_pause_min_s)
        groups = breath_group_extents(store, params)
        for index, extent in enumerate(groups):
            components.append(
                speech_span(
                    "breath_group_%d" % index,
                    extent,
                    store.id_of("consensus_transcript"),
                    store.id_of("hear_scores"),
                    group_index=index,
                )
            )
        # Named for their measurement convention, never `rate` — `branch-conventions.md:150-152`.
        findings.append(
            measured(
                "speech_rate_from_consensus_words_per_s",
                response[0],
                response[1],
                round(len(words) / duration(response), 3),
                support_words=len(words),
            )
        )
        findings.append(
            measured(
                "pause_fraction_of_response",
                response[0],
                response[1],
                round(sum(duration(p) for p in pauses) / duration(response), 3),
                support_pauses=len(pauses),
            )
        )
        findings.append(count("breath_groups", len(groups), None))

    if expectation.anti_pattern is not None:
        alignment = store.stimulus_alignment
        if alignment is None:
            findings.append(unviable("anti_pattern_%s" % expectation.anti_pattern, "stimulus_alignment (D1) is absent"))
        else:
            source = [params.p_normalise(t.text) for t in alignment.expected]
            produced = [params.p_normalise(w.text) for w in lexical(store.words)]
            echo = ngram_echo_fraction(source, produced, params.p_echo_ngram_n)
            cut = (
                params.p_echo_overlap_max
                if expectation.anti_pattern == "verbatim_prompt"
                else params.p_verbatim_overlap_max
            )
            findings.append(measured("verbatim_overlap_fraction", None, None, round(echo, 3), n=params.p_echo_ngram_n))
            if echo > cut:
                findings.append(
                    deviation(
                        "stimulus_mismatch",
                        response[0] if response else None,
                        response[1] if response else None,
                        reading=expectation.anti_pattern,
                        overlap=round(echo, 3),
                    )
                )
            if expectation.anti_pattern == "verbatim_source":
                # "Recall in your own words": semantic coverage is expected and verbatim
                # reproduction is the deviation, so coverage is what `done` reads.
                covered = content_coverage(source, produced)
                findings.append(measured("source_content_coverage", None, None, round(covered, 3)))
                done = covered >= params.p_coverage_min

    for name, why in expectation.unviable:
        findings.append(unviable(name, why))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(done, components, findings)
```

```python
def _speech_item_list(expectation: Expectation, store, hints, params: Params) -> Result:
    """Spans proposed: `task_extent` only. An item is one `word` entity with its own extent, so a
    per-item span would duplicate ground and carry no measurement of its own; the repetition
    finding is a deviation over that word's extent."""
    if expectation.repetition_from_category:
        # Eight of ten categories say "Do not repeat any item"; `Letters` and `Numbers` say
        # "repetition allowed" — 48 of 265 v1 and 77 of 203 English v2. A family-scoped rule
        # inverts the instruction on those, so an unreadable category returns UNDETERMINED.
        category = hints.metadata.get("category") if hints is not None else None
        if category is None:
            return Result(
                UNDETERMINED,
                [],
                [
                    unviable(
                        "repetition_rule",
                        "the category lives only in `instructions`; no grain above the recording carries it",
                    )
                ],
            )
        repetition_allowed = category in ("Letters", "Numbers")
    else:
        repetition_allowed = bool(expectation.repetition_allowed)

    items = lexical(store.words)
    components: list[Proposal] = []
    findings: list[Finding] = []
    first_seen: dict[str, float] = {}
    for word in items:
        key = params.p_normalise(word.text)
        if key in first_seen and not repetition_allowed:
            findings.append(
                deviation("repeated_item", word.extent[0], word.extent[1],       # an ELEVENTH type
                          first_at=first_seen[key], text=word.text)
            )
        first_seen.setdefault(key, word.extent[0])

    extent = hull([w.extent for w in items])
    if extent is not None:
        components.append(
            speech_span("task_extent", extent, store.id_of("consensus_transcript"), *(w.id for w in items),
                        items_n=len(items), repetition_allowed=repetition_allowed)
        )
    findings.append(count("items", len(items), None))
    findings.append(count("repetition_allowed", repetition_allowed, None))
    for name, why in expectation.unviable:
        findings.append(unviable(name, why))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(len(items) > 0, components, findings)
```

```python
def _speech_no_lexical(store, params: Params) -> Result:
    """Spans proposed: **none**, and that is the finding.

    `/pa/` is not lexical and ASR mostly declines it. These families are POSITIVES for SPEECH's
    reference set (`default.yaml:241`), so near-zero lexical content is the correct observation
    rather than a miss, and a branch that proposed a speech span here would be asserting the
    opposite of what it measured.
    """
    produced = lexical(store.words)
    findings = [
        deviation("off_task_extent", w.extent[0], w.extent[1], text=w.text, agreement=w.agreement) for w in produced
    ]
    findings.append(count("lexical_words", len(produced), 0))
    return Result(len(produced) <= params.p_expected_lexical_max, [], findings)
```

```python
def detect_speech(store, params: Params) -> Result:
    """Out of family. Proposes a `family: "speech"` span over every run of lexical words it finds.

    Task-agnostic and needing **no alignment**: nothing lexical is expected on a file of another
    branch's kind, so every lexical word is the finding. This is `detect_lexical_intrusion`'s
    successor and also `detect_count_in`'s — on a `prolonged-vowel` recording it proposes a span
    over `one two three`, and `align_voice`, for which that family IS in family, is what decides
    whether the prescribed count-in happened. Two branches, two questions, one recording.
    """
    words = lexical(store.words)
    runs = lexical_runs(words, params.p_run_gap_max_s)
    components: list[Proposal] = []
    findings: list[Finding] = []
    for index, extent in enumerate(runs):
        inside = [w for w in words if overlaps(w.extent, extent)]
        components.append(
            speech_span(
                "lexical_run_%d" % index,
                extent,
                store.id_of("consensus_transcript"),
                *(w.id for w in inside),
                words_n=len(inside),
                text=" ".join(w.text for w in inside),
                agreement=min((w.agreement for w in inside), default=None),
                evaluates_no_task=True,
            )
        )
    for span in store.spans:
        if span.label == "speech" and not any(overlaps(span.extent, extent) for extent in runs):
            findings.append(contest(span.id, span.extent, "speech", "no_consensus_word_inside"))
    findings.append(count("lexical_words", len(words), None))
    return Result(UNDETERMINED, components, findings)
```

---

### AIRWAY — `align_airway` and `detect_airway`

```python
AIRWAY_EXPECTATIONS: dict[str, Expectation] = {
    "respiration-and-cough-cough": Expectation(pattern=Pattern.EVENT_SERIES, label_set="cough", expected_event_count=5),
    "respiration-and-cough-v2-hardcough": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="cough",
        expected_event_count=None,
        unviable=(("effort_absolute", "no within-recording contrast and no SPL reference; `hard` is not measurable"),),
    ),
    "voluntary-cough": Expectation(pattern=Pattern.EVENT_ALTERNATION, label_set="cough", expected_event_count=3),
    "respiration-and-cough-fivebreaths": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="breath",
        expected_event_count=5,
        route_from_index=True,
        unviable=(
            (
                "route",
                "the discriminating band sits above the 8 kHz ceiling and the residual tilt is confounded, "
                "one for one, with mouth-to-microphone geometry",
            ),
        ),
    ),
    "respiration-and-cough-v2-threebreathsnose": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="breath",
        expected_event_count=3,
        declared_route="nose",
        unviable=(("route", "as `fivebreaths`"),),
    ),
    "respiration-and-cough-v2-threebreathsmouth": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="breath",
        expected_event_count=3,
        declared_route="mouth",
        unviable=(("route", "as `fivebreaths`"),),
    ),
    "respiration-and-cough-threequickbreaths": Expectation(
        pattern=Pattern.EVENT_SERIES, label_set="breath", expected_event_count=3, timed_intervals=True
    ),
    "respiration-and-cough-v2-threebreaths": Expectation(
        pattern=Pattern.EVENT_SERIES, label_set="breath", expected_event_count=3, timed_intervals=True
    ),
    "respiration-and-cough-breath": Expectation(
        pattern=Pattern.SOUND_COVERAGE, label_set="breath", declared_duration_s=30.0
    ),
    "respiration-and-cough-v2-breath": Expectation(
        pattern=Pattern.SOUND_COVERAGE,
        label_set="breath",
        declared_duration_s=20.0,
        declared_route="mouth",
        unviable=(("route", "as `fivebreaths`"),),
    ),
    "breath-sounds": Expectation(
        pattern=Pattern.EVENT_SERIES,
        label_set="breath",
        expected_event_count=3,
        declared_route="mouth",
        relax_s=60.0,
        declared_duration_s=73.0,
        unviable=(("route", "as `fivebreaths`"),),
    ),
}
```

```python
def align_airway(task_family: str, store, hints, params: Params) -> Result:
    expectation = AIRWAY_EXPECTATIONS.get(task_family)
    assert expectation is not None, f"{task_family} is not an AIRWAY family; the caller owes detect_airway"

    if expectation.pattern is Pattern.EVENT_SERIES:
        return _airway_event_series(expectation, store, hints, params)
    if expectation.pattern is Pattern.EVENT_ALTERNATION:
        return _airway_alternation(expectation, store, params)
    if expectation.pattern is Pattern.SOUND_COVERAGE:
        return _airway_coverage(expectation, store, params)
    raise NotImplementedError(expectation.pattern)


def airway_events(label_set_name: str, store, params: Params) -> list[tuple[float, float, str]]:
    """Each event carries the id of the carrier span, because the proposal must name its evidence."""
    labels = params.p_label_sets[label_set_name]
    events: list[tuple[float, float, str]] = []
    for span in amplitude_spans(store.spans):
        if not sounds_like(span, store.span_hear, store.span_yamnet, labels, params.p_score_min):
            continue
        for start, end in events_in_span(store.energy_envelope, span, params):
            events.append((start, end, span.id))
    return sorted(events)


def lexical_intrusions(store) -> list[Finding]:
    # AIRWAY owns this deviation (`branch-airway.md:164-165`); SPEECH's detect mode proposes the
    # span over the same ground from the other side. Measured examples are examiner speech —
    # "I'll have you do that one more time. [breath]", "So just breathe."
    return [
        deviation("off_task_extent", w.extent[0], w.extent[1], text=w.text, agreement=w.agreement)
        for w in lexical(store.words)
    ]
```

**No body here reads `airway.cough`, and none may.** That gate was selected under a **scoped**
reference standard — `cough.amplitude_peak_over_floor_db_max` against `declared_cough_vs_breath`,
12,741 airway recordings, **J 0.7946** at 50 dB — and against the full population, `declared_airway`
over **61,721** recordings, the same reading is **J −0.1398** (sensitivity 0.2514, specificity
0.6088; [`family-taxonomy-ruleset.md:265-268`](family-taxonomy-ruleset.md)). Within the airway
families it is the best cough-vs-breath discriminator there is and it stays in the catalogue; **as a
cough detector over an arbitrary recording it is worse than chance**, and a body that treated a
fired `airway.cough` as evidence of a cough would be reading a loudness detector. The shipped gate
reads `[span_label_set_stat, yamnet.cough_labels.peak_over_floor_db_max]` at 50.0 dB
(`default.yaml:289-292`); on a 13-recording field run it was unavailable on 12 and read 43.10 dB on
the one deliberate-cough recording, firing on neither. The bodies below read `raw_scores` and
`events_in_span` instead.

```python
def _airway_event_series(expectation: Expectation, store, hints, params: Params) -> Result:
    """Spans proposed: **one per event**, plus `task_extent` over their hull.

    One per event is the point: `by_label` increments once per (span, label) pair
    (`airway.py:280`), so a 4 s span holding three coughs counts 1 today. The count that is
    compared against the instruction's `expected_event_count` is the number of these spans.
    """
    assert expectation.label_set is not None
    events = airway_events(expectation.label_set, store, params)
    kind = expectation.label_set
    evidence = (store.id_of("energy_envelope"), store.id_of("span_hear"), store.id_of("span_yamnet"))
    components: list[Proposal] = [
        airway_span("%s_%d" % (kind, index), (start, end), span_id, *evidence, label=kind, index=index)
        for index, (start, end, span_id) in enumerate(events)
    ]
    findings: list[Finding] = [count("expected_event_count", len(events), expectation.expected_event_count)]

    onsets = [start for start, _, _ in events]
    intervals = [round(b - a, 3) for a, b in zip(onsets, onsets[1:])]
    if expectation.timed_intervals:
        # "Quick" is the measurement, and it is the interval, not the count: three breaths says
        # nothing about whether they were quick. p_interval_max_s cannot be fitted before
        # p_peak_prominence_db and p_trough_return_db are.
        findings.append(count("inter_onset_interval_s", intervals, None))
        findings.append(
            count("intervals_over_p_interval_max_s", sum(1 for v in intervals if v > params.p_interval_max_s), 0)
        )

    for index, (start, end, _) in enumerate(events):
        extent = (start, end)
        findings.append(
            measured(
                "%s_peak_over_floor_db" % kind,
                start,
                end,
                round(peak_over_floor_db(store.energy_envelope, extent), 2),
                index=index,
                spectral_balance_db=round(
                    spectral_balance_db(store.spectrogram_wideband, store.sampling_rate, extent,
                                        params.p_effort_split_hz),
                    2,
                ),
                **acquisition_covariates(store, extent),
            )
        )

    declared_route = expectation.declared_route
    if expectation.route_from_index:
        token = hints.metadata.get("task_token") if hints is not None else None       # ‡
        index_segment = token.rsplit("-", 1)[-1] if token else ""
        declared_route = {"1": "nose", "3": "nose", "2": "mouth", "4": "mouth"}.get(index_segment)
    if declared_route is not None or expectation.route_from_index:
        findings.append(count("declared_route", declared_route, None))
        findings.append(
            measured(
                "measured_route",
                None,
                None,
                NOT_SEPARABLE_BY_THIS_DESIGN,
                content_band_hz=getattr(store.band_profile, "rolloff_hz", None),
            )
        )

    task = hull([(start, end) for start, end, _ in events])
    if task is not None:
        components.append(
            airway_span(
                "task_extent",
                task,
                *{span_id for _, _, span_id in events},
                events_n=len(events),
                declared_event_count=expectation.expected_event_count,
                declared_route=declared_route,
            )
        )
        if touches_edge(task, store.stream_extent):
            findings.append(deviation("truncation", task[0], task[1]))

    if expectation.relax_s is not None and task is not None:
        # The one family whose instruction PRESCRIBES material that is not the task. Emitted only
        # on a recording long enough to contain it — the measured median is 13.2 s against ~73 s.
        if duration(store.stream_extent) >= expectation.relax_s + duration(task):
            findings.append(deviation("off_task_extent", 0.0, expectation.relax_s, reading="declared_relax_period"))

    findings.extend(lexical_intrusions(store))
    for name, why in expectation.unviable:
        findings.append(unviable(name, why))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(len(events) > 0, components, findings)
```

```python
def _airway_alternation(expectation: Expectation, store, params: Params) -> Result:
    """Spans proposed: one per cough, one per breath, plus `task_extent`.

    The expected pattern is an ALTERNATION, so material between coughs is matched as breath and
    never scored as off_task_extent — a cough detector alone is insufficient. Median 13.8 s
    against `v2-hardcough`'s 4.6 s, consistent with three cough-and-breathe cycles in one file.
    """
    coughs = airway_events("cough", store, params)
    breath_labels = params.p_label_sets["breath"]
    breath_carriers = [
        s for s in store.spans if sounds_like(s, store.span_hear, store.span_yamnet, breath_labels, params.p_score_min)
    ]
    breaths = merge([s.extent for s in breath_carriers])
    evidence = (store.id_of("energy_envelope"), store.id_of("span_hear"), store.id_of("span_yamnet"))

    components: list[Proposal] = [
        airway_span("cough_%d" % k, (start, end), span_id, *evidence, label="cough", index=k)
        for k, (start, end, span_id) in enumerate(coughs)
    ]
    components += [
        airway_span("breath_%d" % k, extent, store.id_of("span_hear"), store.id_of("span_yamnet"),
                    label="breath", index=k)
        for k, extent in enumerate(breaths)
    ]

    cycles = sum(1 for _, cough_end, _ in coughs if any(b[0] >= cough_end for b in breaths))
    findings: list[Finding] = [
        count("expected_event_count", len(coughs), expectation.expected_event_count),
        count("cough_then_breathe_cycles", cycles, expectation.expected_event_count),
    ]
    task = hull([(c.start, c.end) for c in components])
    if task is not None:
        components.append(
            airway_span("task_extent", task, *{p.derived_from[0] for p in components},
                        coughs_n=len(coughs), breaths_n=len(breaths))
        )
    findings.extend(lexical_intrusions(store))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(len(coughs) > 0, components, findings)
```

```python
def _airway_coverage(expectation: Expectation, store, params: Params) -> Result:
    """Spans proposed: one per merged run of `Breathe`-scoring windows, plus `task_extent`.

    `residual.energy_fraction` is NOT read here. residual = plain - g*FRCRN(plain) and FRCRN is a
    speech enhancer, so a high fraction means "FRCRN removed most of this signal", i.e. THIS IS
    NOT SPEECH — satisfied equally by a cough, a glide, room noise and a near-silent file. As the
    `airway.breath` routing gate (`default.yaml:285-288`) "not speech" may be adequate; as this
    row's presence measurement it is not.
    """
    assert expectation.label_set is not None
    labels = params.p_label_sets[expectation.label_set]
    windows = [
        (w.start, w.end)
        for w in store.hear_scores                                     # raw HeAR, 2.0 s, non-overlapping
        if any(float(w.label_scores.get(label, 0.0)) >= params.p_score_min for label in labels)
    ]
    covered = merge(windows)
    total = sum(duration(extent) for extent in covered)
    coverage = total / max(duration(store.stream_extent), 1e-9)
    components: list[Proposal] = [
        airway_span("breathing_%d" % k, extent, store.id_of("hear_scores"), label="Breathe", index=k)
        for k, extent in enumerate(covered)
    ]
    findings: list[Finding] = [
        measured("breath_coverage_fraction", None, None, round(coverage, 3), covered_s=round(total, 2))
    ]
    task = hull(covered)
    if task is not None:
        components.append(
            airway_span("task_extent", task, store.id_of("hear_scores"), runs_n=len(covered),
                        covered_s=round(total, 2))
        )
    findings.extend(lexical_intrusions(store))
    for name, why in expectation.unviable:
        findings.append(unviable(name, why))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(coverage >= params.p_breath_coverage_min, components, findings)
```

```python
def detect_airway(store, params: Params) -> Result:
    """Out of family. Proposes a `family: "airway"` span over every cough and breath event it finds.

    Task-agnostic, and the same `sounds_like` + `events_in_span` machinery the in-family mode
    uses. It **does not read `airway.cough`**: that gate was selected under a scoped reference and
    reads J -0.1398 against `declared_airway` over 61,721 recordings, so at its 50 dB cut it is a
    loudness detector over an arbitrary recording. A breath during passage reading is NOT a
    deviation — it is how S4 measures breath-group structure — so nothing here emits one.
    """
    evidence = (store.id_of("energy_envelope"), store.id_of("span_hear"), store.id_of("span_yamnet"))
    components: list[Proposal] = []
    findings: list[Finding] = []
    for name in ("cough", "breath"):
        labels = params.p_label_sets[name]
        for span in store.spans:
            if not sounds_like(span, store.span_hear, store.span_yamnet, labels, params.p_score_min):
                continue
            events = events_in_span(store.energy_envelope, span, params)
            if not events:
                # The label is there and the envelope resolves no event boundary inside it, so the
                # proposal takes the carrier's own extent and says which it is.
                components.append(
                    airway_span("airway_event", span.extent, span.id, *evidence, label=name,
                                boundaries="carrier_span", evaluates_no_task=True)
                )
                continue
            for index, extent in enumerate(events):
                components.append(
                    airway_span("%s_event" % name, extent, span.id, *evidence, label=name, index=index,
                                boundaries="envelope_event", carrier_extent=list(span.extent),
                                evaluates_no_task=True)
                )
            findings.append(count("%s_events_in_span" % name, len(events), None))

    marked = [(c.start, c.end) for c in components]
    for span in store.spans:
        if span.label in ("cough", "breath") and not any(overlaps(span.extent, e) for e in marked):
            findings.append(contest(span.id, span.extent, span.label, "no_raw_score_over_p_score_min"))
    findings.append(count("airway_events", len(components), None))
    return Result(UNDETERMINED, components, findings)
```

AIRWAY's gate evidence is `unavailable` on 56,505 of 62,547 recordings, so the out-of-family mode is
the one that actually runs on the corpus.

---

### DDK — `align_ddk` and `detect_ddk`

DDK is a declared branch with **no node**: `BRANCHES = ("AIRWAY", "SPEECH", "VOICE", "DDK")`
(`vocabulary.py:31`), `DDK` is absent from the dispatch table (`run.py:297-301`), and the `call is
None` arm at `run.py:304-305` marks it `SKIPPED` with `NO_NODE = "no node implements this branch"`
(`run.py:46`) — **whether or not routing selected it**, because that arm precedes the `in selected`
test. It is written here on the same terms as the others, per the owner's instruction that a
declared branch is assessed independently of whether it is implemented.

```python
DDK_EXPECTATIONS: dict[str, Expectation] = {
    "diadochokinesis-pa": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
    "diadochokinesis-ta": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
    "diadochokinesis-ka": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
    "diadochokinesis-v2-puh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
    "diadochokinesis-v2-tuh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
    "diadochokinesis-v2-kuh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
    "diadochokinesis-pataka": Expectation(
        pattern=Pattern.SYLLABLE_SEQUENCE, sequence=("labial", "alveolar", "velar"), expected_event_count=30
    ),
    "diadochokinesis-v2-puhtuhkuh": Expectation(
        pattern=Pattern.SYLLABLE_SEQUENCE, sequence=("labial", "alveolar", "velar"), declared_duration_s=5.0
    ),
    "diadochokinesis-buttercup": Expectation(
        pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",), expected_event_count=10
    ),
    "diadochokinesis-v2-buttercup": Expectation(
        pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",), declared_duration_s=5.0
    ),
}


def ddk_carrier(store, params: Params):
    best = None
    best_rate: float | None = None
    for span in amplitude_spans(store.spans):
        if duration(span.extent) < params.p_train_min_s:
            continue
        rate = train_rate_hz(store.energy_envelope, span.extent, params)
        if rate is None:
            continue
        if best is None or duration(span.extent) > duration(best.extent):
            best, best_rate = span, rate
    return best, best_rate


def ddk_places(onsets: Sequence[tuple[float, float]], store, params: Params) -> list[str]:
    """D6. /p/, /t/ and /k/ differ in burst spectrum in the textbook way — /t/ high-frequency
    dominant, /k/ a compact mid-frequency peak, /p/ diffuse and falling — and all three sit
    comfortably inside the 8 kHz band. `spectrogram_wideband` is a 5 ms window at a 5 ms hop, the
    classical resolution for exactly this measurement.

    The PPG is NOT the instrument: a phonetic posteriorgram is trained on connected speech, and on
    a rapid nonsense CV train with no lexical context its acoustic-model prior works against the
    discrimination, at the cost of a model pass to recover less.
    """
    places: list[str] = []
    window_s = params.p_burst_window_ms / 1000.0
    for start, _ in onsets:
        burst = (start, start + window_s)
        energies = {
            place: band_power(store.spectrogram_wideband, store.sampling_rate, burst, lo, hi)
            for place, (lo, hi) in params.p_place_centroid_bands_hz.items()
        }
        finite = {k: v for k, v in energies.items() if np.isfinite(v) and v > 0.0}
        if len(finite) < 2:
            places.append("unresolved")
            continue
        ranked = sorted(finite.items(), key=lambda kv: kv[1], reverse=True)
        margin_db = 10.0 * float(np.log10(ranked[0][1] / ranked[1][1]))
        places.append(ranked[0][0] if margin_db >= params.p_place_margin_db else "unresolved")
    return places
```

```python
def align_ddk(task_family: str, store, hints, params: Params) -> Result:
    """Spans proposed: **one**, the train, as `task_extent`.

    An individual syllable is NOT a span. The clinically meaningful quantities — rate, inter-onset
    interval variability, sequence collapse — are statistics over the onset series, and minting
    one span per syllable would add roughly 30 spans per recording over 7,989 recordings carrying
    no measurement of their own. The onsets travel as a `counts` entry, and a syllable that is not
    the one the sequence expected travels as a `syllable_sequence_mismatch` deviation with its own
    extent, which needs no span.
    """
    expectation = DDK_EXPECTATIONS.get(task_family)
    assert expectation is not None, f"{task_family} is not a DDK family; the caller owes detect_ddk"
    if expectation.pattern is Pattern.ORDERED_TOKENS:
        return _ddk_repeated_word(expectation, store, params)

    train, rate_hz = ddk_carrier(store, params)
    if train is None:
        # No carrier clears the train minimum and no modulation peak is prominent, so there is no
        # extent to propose. Nothing is minted rather than a span whose boundaries are guessed.
        return Result(False, [], off_task([], store.spans, params.p_gap_off_task_min_s))

    onsets = events_in_span(store.energy_envelope, train, params)
    starts = [start for start, _ in onsets]
    extent = hull(onsets) or train.extent
    findings: list[Finding] = [
        count("expected_event_count", len(onsets), expectation.expected_event_count),
        measured("ddk_syllable_rate_from_envelope_modulation_hz", extent[0], extent[1], rate_hz),
        count("inter_onset_interval_s", [round(b - a, 3) for a, b in zip(starts, starts[1:])], None),
        count("syllable_onset_s", [round(s, 3) for s in starts], expectation.expected_event_count),
        measured(
            "train_fraction_of_recording",                                          # D5
            extent[0],
            extent[1],
            round(duration(extent) / max(duration(store.stream_extent), 1e-9), 3),
        ),
    ]

    attributes: dict = {"syllables_n": len(onsets), "production": "syllable_train"}
    if expectation.pattern is Pattern.SYLLABLE_SEQUENCE:
        assert expectation.sequence is not None
        cycle = expectation.sequence
        places = ddk_places(onsets, store, params)
        for index, (onset, place) in enumerate(zip(onsets, places)):
            target = cycle[index % len(cycle)]
            if place not in (target, "unresolved"):
                # Deliberately NOT `stimulus_mismatch`: there is no stimulus text and no lexical
                # expectation. /pa-pa-pa/ is a collapse of the sequence and is the finding.
                findings.append(
                    deviation("syllable_sequence_mismatch", onset[0], onset[1], expected=target, measured=place)
                )
        resolved = [p for p in places if p != "unresolved"]
        cycles = sum(
            1 for i in range(len(resolved) - len(cycle) + 1) if tuple(resolved[i : i + len(cycle)]) == cycle
        )
        if resolved:
            dominant = max(set(resolved), key=resolved.count)
            findings.append(
                measured(
                    "sequence_collapse_fraction",
                    extent[0],
                    extent[1],
                    round(resolved.count(dominant) / len(resolved), 3),
                    dominant_place=dominant,
                    support_syllables=len(resolved),
                )
            )
        findings.append(count("realised_cycles", cycles, None))
        attributes.update(production="syllable_sequence", realised_cycles=cycles, resolved_n=len(resolved))
        done: Done = bool(onsets) and cycles >= 1
    else:
        done = rate_hz is not None and len(onsets) > 0

    components = [
        ddk_span("task_extent", extent, train.id, store.id_of("energy_envelope"),
                 store.id_of("spectrogram_wideband"), **attributes)
    ]
    if touches_edge(extent, store.stream_extent):
        findings.append(deviation("truncation", extent[0], extent[1]))
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(done, components, findings)
```

```python
def _ddk_repeated_word(expectation: Expectation, store, params: Params) -> Result:
    """Spans proposed: **one**, the train, over the hull of the realised tokens.

    The only DDK family with a lexical pattern, so the only one where the recognisers produce the
    count directly.
    """
    assert expectation.tokens is not None
    target = params.p_normalise(expectation.tokens[0])
    hits = [w for w in lexical(store.words) if params.p_normalise(w.text) == target]
    findings: list[Finding] = [count("expected_event_count", len(hits), expectation.expected_event_count)]
    train = hull([w.extent for w in hits])
    components: list[Proposal] = []
    if train is not None:
        components.append(
            ddk_span(
                "task_extent",
                train,
                store.id_of("consensus_transcript"),
                store.id_of("energy_envelope"),
                *(w.id for w in hits),
                production="lexical_repetition",
                token=target,
                repeats_n=len(hits),
            )
        )
        findings.append(
            measured(
                "ddk_syllable_rate_from_envelope_modulation_hz",
                train[0],
                train[1],
                train_rate_hz(store.energy_envelope, train, params),
            )
        )
    findings.extend(declared_duration_count(store, expectation.declared_duration_s))
    findings.extend(off_task(components, store.spans, params.p_gap_off_task_min_s))
    return Result(len(hits) > 0, components, findings)
```

```python
def detect_ddk(store, params: Params) -> Result:
    """Out of family. Proposes a `family: "ddk"` span over every rapid repetition train it finds.

    Task-agnostic, from the envelope modulation spectrum — not from Praat's `extract_speech_rate`,
    whose `min_pause = 0.3 s` (`praat_parselmouth.py:228`) exceeds an entire DDK cycle, and not
    from the PPG. Repetition occurs in ordinary speech — a stutter, a false start, a repeated
    word — so the branch measures what it finds and says what it is; it does not assert that a
    Harvard sentence failed to be a DDK task (`branch-ddk.md:66-70`). DDK routes 22,363
    recordings against the 7,989 that declare a DDK family, 14,878 of them declaring none
    (`branch-ddk.md:20`), so this is the mode that dominates its corpus.
    """
    components: list[Proposal] = []
    findings: list[Finding] = []
    for span in amplitude_spans(store.spans):
        if duration(span.extent) < params.p_train_min_s:
            continue
        rate_hz = train_rate_hz(store.energy_envelope, span.extent, params)
        if rate_hz is None:
            continue
        components.append(
            ddk_span(
                "repetition",
                span.extent,
                span.id,
                store.id_of("energy_envelope"),
                production="acoustic_repetition",
                rate_hz=rate_hz,
                evaluates_no_task=True,
            )
        )
        findings.append(
            measured(
                "ddk_syllable_rate_from_envelope_modulation_hz",
                span.extent[0],
                span.extent[1],
                rate_hz,
                reading="acoustic_repetition_not_a_declared_ddk_task",
            )
        )

    occurrences: dict[str, list] = {}
    for word in lexical(store.words):
        occurrences.setdefault(params.p_normalise(word.text), []).append(word)
    for token, words in occurrences.items():
        if len(words) < params.p_repeat_min_occurrences:
            continue
        extent = hull([w.extent for w in words])
        assert extent is not None
        components.append(
            ddk_span(
                "lexical_repetition",
                extent,
                store.id_of("consensus_transcript"),
                *(w.id for w in words),
                production="lexical_repetition",
                token=token,
                repeats_n=len(words),
                evaluates_no_task=True,
            )
        )
        # `transcript_repeat` is already computed in `routing_analysis/features.py`, where no gate
        # has read it since `ddk.lexical_repetition` was removed. Moving it is a convenience: this
        # loop is the capability, over word entities the store already holds.
        findings.append(measured("transcript_repeat", extent[0], extent[1], len(words), token=token))
    return Result(UNDETERMINED, components, findings)
```

---

### QUALITY — `detect_quality`, the only mode it can have

```python
def detect_quality(store, params: Params) -> Result:
    """QUALITY is never routed and is never in family. It has the detect mode and no other.

    Spans proposed: at most **one**, `occluded`, and only when the tilt finding fires. The clip
    check proposes nothing — it contests PREPROCESS's own reading, which is an assertion beside a
    span and never an edit to it.
    """
    components: list[Proposal] = []
    findings: list[Finding] = []

    # Q1, as built today (`quality.py:266-302`): does PREPROCESS's own clip span contradict
    # PREPROCESS's own amplitude? A store-consistency audit whose expected count is zero.
    for span in store.spans:
        if span.family != "clip" or span.signal != store.recording_signal:
            continue
        level_dbfs = store.clip_amplitude.level_dbfs.get(span.id)
        if level_dbfs is None:
            findings.append(count("clip_span_unmeasurable", span.id, None))
            continue
        if store.clip_amplitude.unclipped_peak > level_dbfs * (1.0 + store.clip_contradiction_margin):
            findings.append(contest(span.id, span.extent, "clip", "clip_above_unclipped_sample"))

    # The hygiene clause of `respiration-and-cough-v2-hardcough`, the corpus's only one, read as a
    # QUALITY property of the recording rather than as an AIRWAY expectation of the task.
    extent = store.stream_extent
    rolloff_hz = getattr(store.band_profile, "rolloff_hz", None)                          # D3 †
    if rolloff_hz is None:
        findings.append(unviable("occluded_microphone",
                                 "`band_profile` (D3) is absent; the tilt half has no reference"))
    else:
        balance = spectral_balance_db(
            store.spectrogram_wideband, store.sampling_rate, extent, params.p_effort_split_hz
        )
        octaves = float(np.log2(max(rolloff_hz, 1.0) / max(params.p_effort_split_hz, 1.0)))
        tilt = balance / octaves if octaves != 0.0 else float("nan")
        findings.append(
            measured("spectral_tilt_db_per_octave", extent[0], extent[1], round(tilt, 2), content_band_hz=rolloff_hz)
        )
        if (
            np.isfinite(tilt)
            and tilt < -params.p_tilt_max_db_per_octave
            and store.level.rms_dbfs < params.p_level_min_dbfs
        ):
            findings.append(
                deviation(
                    "occluded_microphone",
                    extent[0],
                    extent[1],
                    tilt_db_per_octave=round(tilt, 2),
                    rms_dbfs=store.level.rms_dbfs,
                    content_band_hz=rolloff_hz,
                )
            )
            components.append(
                quality_span(
                    "occluded",
                    extent,
                    store.id_of("level"),
                    store.id_of("spectrogram_wideband"),
                    store.id_of("band_profile"),
                    tilt_db_per_octave=round(tilt, 2),
                )
            )

    # Q5. A sidecar-consistency check, not a task-completion measurement.
    findings.append(count("declared_duration_s", round(duration(store.stream_extent), 2), store.declared_duration_s))
    return Result(UNDETERMINED, components, findings)
```

---

### The ten `NOT_SEPARABLE_BY_THIS_DESIGN` sites

Each is an `Expectation.unviable` entry or a literal in a body, and **each emits a measurement
saying the determination cannot be made, with its reason** — so a reader can tell *not separable by
this design* from *nobody has written it yet*. None of the ten is reopened here.

**Ten sites, twelve `unviable` entries**, and the difference is only that one site can span several
families: `route` is one determination carried by **five** AIRWAY rows (`fivebreaths`, both
`v2-threebreaths{nose,mouth}`, `v2-breath`, `breath-sounds`, grouped as rows 1–4 below);
`category_membership` is one carried by **three** SPEECH rows; `effort_absolute` is one
determination reached twice, by `v2-hardcough` and by `loudness` v1, and is listed twice because the
two reach it from different directions. Nine of the ten are `unviable` entries; the tenth, the
sub-second recording, is not — it is every matcher returning `False` correctly and
uninformatively, which no entry can express.

| # | site | where it fires | why it stays unviable |
| --- | --- | --- | --- |
| 1 | `route` | `align_airway`, `fivebreaths` (`route_from_index`) | discriminating band above the 8 kHz ceiling; the residual tilt confounded one-for-one with mouth-to-microphone geometry, which changes *with the route by construction* |
| 2 | `route` | `align_airway`, `v2-threebreathsnose` / `-mouth` | as 1, and these two plus the `fivebreaths` split are the only declared route contrasts, so none is validation-grade for A7 |
| 3 | `route` | `align_airway`, `v2-breath` | as 1 |
| 4 | `route` | `align_airway`, `breath-sounds` | as 1 |
| 5 | `effort_absolute` | `align_airway`, `v2-hardcough` | *"hard"* has no within-recording contrast and no SPL reference; the output is a measurement with its covariates |
| 6 | `effort_absolute` | `_voice_effort`'s non-contrast arm, `loudness` | as 5. `loudness-v2` and `voluntary-cough` are unaffected: both carry a within-recording contrast |
| 7 | `category_membership` | `align_speech`, `animal-fluency` and both `random-item-generation` | a lexicon or a text embedding: branch-local, one consumer, no waveform |
| 8 | `defines_its_cue` | `align_speech`, `productive-vocabulary` | the same, and 78 recordings carry no cue at all |
| 9 | `source_overlap` | `align_speech`, `cinderella-story` | the source is a physical storybook; `stimulus_text` empty on all 258, so no overlap measure is definable, D1 included |
| 10 | a sub-second recording | every body, through `done = False` | 2,020 sidecars declare under a second; every expected pattern is absent, so every matcher correctly returns *not done* and uninformatively. Whether QUALITY should absorb it is a protocol question |

**Five detection approaches stay ruled out**, and they are the rulings of
[`preprocess-derivatives-for-expected-patterns.md`](preprocess-derivatives-for-expected-patterns.md)
§ 4.1–4.5 — nasal-versus-oral route from `gammatone` (§ 4.1), D6 sequence conformance from the PPG
posteriorgram (§ 4.2), syllable-nucleus rate via Praat's speech rate (§ 4.3), absolute effort from
`level` (§ 4.4), and `residual` `energy_fraction` as the breath-presence reading (§ 4.5). Nothing
above reads `gammatone` for a route, `ppg_posteriorgram` for a place, `praat_features`'
`extract_speech_rate` for a rate, `level` for a verdict, or `residual.energy_fraction` for breath
presence. **§ 4 has a sixth subsection, § 4.6, and it is a scoping correction rather than a
rule-out**: `declared_duration_s` against measured is *free* and stays, read as a sidecar
consistency check — which is why `declared_duration_count(...)` emits a `count` and never a
`deviation`, and why the capability's home is QUALITY's Q5.

---

## What each family requires, and the row that carries it

The research below is unchanged: what each family's instruction asks for, what it needs out of the
store, what is absent, and the corpus figures behind it. What has changed is the last line of each
block — the pseudo-code function is now the `Expectation` row its branch's table holds, and the body
that reads it is the matcher for that row's `Pattern`, above.

### `prolonged-vowel` (1,604) — VOICE and SPEECH

**Requires, across both branches:** `consensus_transcript` + its `word` entities · `spans`
(`amplitude`, `gap`) · `phonation_tracks` (`times_s`, `f0_hz`, `strength`) · `continuity_trace` ·
`energy_envelope` · `stream_extent`.
**Absent:** `stimulus_alignment`† (D1) for the count-in as a declared expectation rather than a
guessed token list; `phonation_tracks.{hnr_db,cpps_db,rms_dbfs}`† (D2) for any voice-quality number
over the vowel — without it the only figures that exist are `praat_features`, which are whole-file
and include the count-in and the silence.
**Unreachable:** `hints.expected_speech`‡.

**Its row**, in `VOICE_EXPECTATIONS` — `align_voice` dispatches it to the `SUSTAINED` matcher:

```python
"prolonged-vowel": Expectation(
    pattern=Pattern.SUSTAINED,
    tokens=("one", "two", "three"),
    token_source="instructions",
    declared_duration_s=12.0,
    lexical_separator=True,
),
```

**Spans proposed: two.** `count_in`, from the consensus `word` extents of the matched
`one two three`, carrying `excluded_from_measurement=True`; and `task_extent`, from the first and
last voiced frame of `phonation_tracks` inside the qualifying amplitude span. Both are
`family: "voice"`, both name their evidence in `wasDerivedFrom`, and neither edits anything
PREPROCESS wrote. **Two rather than one because only the second is the voice measurement**, which
is the defect the split removes: today every Praat scalar is taken over count-in plus silence plus
vowel.

**`detect_count_in` has no successor function.** The expectation became the `tokens` field of the row
above, and the finding became `detect_speech`'s generic lexical marking — SPEECH is **out of family**
here, `prolonged-vowel` being `VOICE_ELICITING` (`families.py:78-87`), so it proposes a
`family: "speech"` span over the `one two three` run and answers no *"was it done"*. `align_voice`
is what evaluates whether the prescribed count-in happened. Two branches, two questions, one
recording, and one fewer function.

**Notes.** The lexical half is the cheapest row in the document — 938 of 1,258 transcripts open with
`One two three` ([`dag.md:185`](dag.md)) and the control family fires `speech.intrusion` at 3.2%
([`family-taxonomy-ruleset.md:320-323`](family-taxonomy-ruleset.md)). The acoustic half returns
nothing today for a reason that is not a missing estimator: VOICE selects `family == "phonation"`
spans (`voice.py:230`) and PREPROCESS's amplitude spans carry no `family` at all, so its candidate
list is empty before any test runs. **Propose-only removes that blocker rather than fixing the
selector**: `qualifying_phonation` reads the `measure == "amplitude"` spans as evidence and VOICE
mints its own `family: "voice"` span over what qualifies, so nothing has to stamp a family onto a
span PREPROCESS wrote. What remains owed is the *decision* —
`p_f0_spread_*` and `p_continuity_min`, since `continuity_trace` and `f0_hz` are both in the store.
`words.onomatopoeic_tokens` does not help here — its vocabulary is a cough set
(`default.yaml:123`), not numerals.

### `maximum-phonation-time` (2,696), `-v2` (813) — VOICE

**Requires:** `spans` (`amplitude`, `gap`) · `phonation_tracks` · `continuity_trace` ·
`energy_envelope` · `span_hear` · `span_yamnet` (v1's inhale only) · `stream_extent` ·
`hints.metadata.task_token`‡ for the trailing index.
**Absent:** `phonation_tracks.{hnr_db,cpps_db,rms_dbfs}`† (D2).

**Its rows**, both `SUSTAINED`, one boolean apart:

```python
"maximum-phonation-time":    Expectation(pattern=Pattern.SUSTAINED, forbid_lexical=True,
                                         expect_inhale=True),
"maximum-phonation-time-v2": Expectation(pattern=Pattern.SUSTAINED, forbid_lexical=True,
                                         expect_inhale=False),
```

**Spans proposed: one**, `task_extent`, over the voiced run. **The v1 inhale gets no VOICE span** —
under propose-only a branch mints only in its own family, and an inhale is airway evidence, so
`align_voice` records `inhale_expected_in_file` as a count and `detect_airway`, running on the same
recording, is what proposes the span over it — a hand-off that is real wherever routing selects
AIRWAY on this family, which this document does not measure.

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

**Its rows**, three families and one matcher:

```python
"glides-low-to-high": Expectation(pattern=Pattern.GLIDE, declared_direction="up"),
"glides-high-to-low": Expectation(pattern=Pattern.GLIDE, declared_direction="down"),
"high-to-low":        Expectation(pattern=Pattern.GLIDE, declared_direction="down"),
```

**Spans proposed: one**, `task_extent`, over the **tolerant-monotone run itself** rather than over
the carrier amplitude span — the sweep is the production, and the carrier's extent is recorded in
`wasDerivedFrom` and in the span's `carrier_extent` attribute.

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

**SPEECH's rows** — `loudness` is `LEXICAL_SPEECH` (`families.py:41-42`), so SPEECH is **in
family** here and runs the same `ORDERED_TOKENS` matcher every read family uses:

```python
"loudness":    Expectation(pattern=Pattern.ORDERED_TOKENS, tokens=("hey", "hey", "hey"),
                           expected_event_count=3),
"loudness-v2": Expectation(pattern=Pattern.ORDERED_TOKENS, tokens=("hey", "hey"),
                           expected_event_count=2),
```

The tokens are literals rather than a `token_source`, so **D1 is not load-bearing for SPEECH here**:
the instruction fixes the word. SPEECH proposes one span, `task_extent` over the hull of the
realised tokens.

**VOICE's row, and it is not read today.** `loudness` is `LEXICAL_SPEECH`, so under
`reference_family_set.VOICE` it is **out of family for VOICE** and `align_voice` never reaches it.
The row is written and held:

```python
# VOICE_EXPECTATIONS_PENDING_DECLARATION
"loudness": Expectation(
    pattern=Pattern.EFFORT,
    tokens=("hey",),
    expected_event_count=3,
    unviable=(("effort_absolute",
               "`level` is uncalibrated and no SPL reference exists in the graph"),),
),
```

**Spans proposed, when the declaration moves: one per realised token**, because the effort
measurement is per token. The `unviable` entry is what makes `measure_loudness_effort` emit a
measurement with its covariates and never a `maximal` / `not maximal` verdict.

```python
# VOICE_EXPECTATIONS_PENDING_DECLARATION
"loudness-v2": Expectation(pattern=Pattern.EFFORT, tokens=("hey",), expected_event_count=2,
                           contrast=True),
```

**One field, `contrast`, is the whole v1/v2 difference** — and it is the field that makes v2 the
cheapest effort measure in the corpus, because a within-recording difference needs no norm and no
calibration. **Spans proposed, when the declaration moves: one per token plus a `task_extent` over
the pair**, since the contrast is the measurement and it spans both.

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

**VOICE's row, and it is not read today.** `cape-v-sentences` and `-v2` are `LEXICAL_SPEECH`
(`families.py:33-34`), so the corpus's one deliberate voice-quality instrument is **out of family
for VOICE** under the reference family set and reaches it only through `detect_voice`, which
evaluates no task. The rows are written and held:

```python
# VOICE_EXPECTATIONS_PENDING_DECLARATION
"cape-v-sentences":    Expectation(pattern=Pattern.PER_SENTENCE, token_source="stimulus_text"),
"cape-v-sentences-v2": Expectation(pattern=Pattern.PER_SENTENCE, token_source="stimulus_text"),
```

**Spans proposed, when the declaration moves: one per sentence, plus a `task_extent` over their
hull** — because pooling across the six discards the instrument's design, each sentence loading a
different phonatory condition, and today's whole-file `praat_features` is exactly that pooling.
**When `stimulus_alignment` is absent the matcher proposes nothing and returns `UNDETERMINED`**
rather than minting six spans whose boundaries are guessed.

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

**Its rows**, `ORDERED_TOKENS` with the tokens read per recording:

```python
"harvard-sentences-list": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
"cape-v-sentences":       Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
"cape-v-sentences-v2":    Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text"),
```

**Spans proposed: `task_extent` over the realised tokens, plus one `structure_*` per unit the
alignment yields.** **No span per token** — a realised token already has a `word` entity carrying
its own extent, `agreement` and per-source `timings`, and one span per token would mint roughly
8 × 13,705 extents on `harvard-sentences-list` alone with no measurement attached. **Without D1 the
matcher proposes nothing and returns `UNDETERMINED`**: the transcript alone cannot say what was
expected, and a guessed boundary is worse than an admitted absence.

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

```python
"rainbow-passage":     Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text",
                                   connected=True),
"caterpillar-passage": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text",
                                   connected=True),
```

**A passage is a read text plus one boolean.** `connected=True` is what adds the breath groups, so
`detect_read_passage` is not a second function — it is the same `ORDERED_TOKENS` matcher taking one
more branch. **Spans proposed: `task_extent`, the `structure_*` units, and one per breath group**,
each group carrying its own S4 measurement, which is why a group is a span and a token is not.

**Notes.** `[breath]` tokens inside a passage are **not** `filler` — they are how S4 measures breath
structure, and scoring them as disfluency inverts the measurement. Both grains agree on both fields
on all 897 and all 597 recordings, so the one-to-many collapse that breaks the other read families
does not arise here.

### `word-color-stroop` (472) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent` ·
`declared_duration_s` (75 s) from the sidecar.
**Absent:** `stimulus_alignment`† (D1), whose expected-token list here is the **answer** sequence.

```python
"word-color-stroop": Expectation(pattern=Pattern.ORDERED_TOKENS, token_source="stimulus_text",
                                 declared_duration_s=75.0, emit_filler=False),
```

**`emit_filler=False` is the one behavioural difference from a read text, and it is a field.**
Hesitation and self-correction are this task's dependent variable, so scoring them as `filler` would
score the measurement as a defect. The expected tokens are the **answer** sequence — 472 distinct
sequences over 472 recordings — so the `structure_*` spans are the 15 colour items. **Spans
proposed: `task_extent` plus the 15 items**, and nothing at all without D1.

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

```python
"free-speech": Expectation(pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text",
                           anti_pattern="verbatim_prompt"),
```

**Spans proposed: one**, `task_extent` over the hull of the `measure: "asr"` spans, derived from
every one of them. The anti-pattern produces no span — a verbatim echo is a `stimulus_mismatch`
deviation over the response's extent, not a region of its own. Without D1 the echo test emits
`NOT_SEPARABLE_BY_THIS_DESIGN` and the presence half still stands.

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

```python
"free-speech-v2": Expectation(pattern=Pattern.FREE_RESPONSE, declared_duration_s=30.0),
```

**`anti_pattern` is absent, and that is the entire v1/v2 difference.** v2's instruction drops *"do
not record yourself reading the prompt"*, so no verbatim-echo deviation can fire here — and written
as a missing field rather than a second function, it cannot drift back toward v1's. **Spans
proposed: one**, `task_extent`.

**Notes.** The acoustictask prompt matches **none** of the six v2 questions, on all 707 sidecars, so
the family grain is wrong on every one of the 2,120 recordings. Treating v1 and v2 alike is the
error this pair exists to prevent.

### `story-recall` (889), `-v2` (660) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent`.
**Absent:** `stimulus_alignment`† (D1); a lexical-overlap statistic — and note that the overlap
itself is arithmetic once D1 exists, so what is missing is the *cut*, not an estimator.

```python
"story-recall":    Expectation(pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text",
                               anti_pattern="verbatim_source"),
"story-recall-v2": Expectation(pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text",
                               anti_pattern="verbatim_source"),
```

**The same matcher as `free-speech`, with the other anti-pattern** — and `verbatim_source` is also
what makes `done` read content coverage rather than duration, since *"recall in your own words"*
asks for coverage and treats verbatim reproduction as the deviation. The source is read per
recording, because the five Spanish v1 recordings carry v2's story under v1's family name. **Spans
proposed: one**, `task_extent`.

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

```python
"cinderella-story": Expectation(
    pattern=Pattern.FREE_RESPONSE,
    unviable=(("source_overlap",
               "`stimulus_text` is empty on all 258; the source is a physical storybook"),),
),
```

**No `token_source` and no `anti_pattern`**, and the `unviable` row is what makes the ceiling
explicit rather than implied by two absent fields. **Spans proposed: one**, `task_extent`.

**Notes.** Presence and extent only, and that is the ceiling rather than a first step: the
`story-recall` n-gram method **does not transfer**, because there is no text to overlap against.
Median 93.9 s; 18 of 258 run under a second.

### `productive-vocabulary` (2,910) — SPEECH

**Requires:** as `cinderella-story`, plus the per-recording cue word.
**Absent:** `stimulus_alignment`† (D1) for the cue; a "is this speech a definition *of* that cue"
measure, which is **not** a PREPROCESS derivative and has no viable approach in this graph.

```python
"productive-vocabulary": Expectation(
    pattern=Pattern.FREE_RESPONSE,
    token_source="stimulus_text",
    unviable=(("defines_its_cue", "a lexicon or a text model, branch-local, and no waveform"),),
),
```

**Spans proposed: one**, `task_extent`. The cue is per recording — 204 distinct cues, 78 recordings
with none — so the family grain carries nothing here and neither grain does on the cue-less 78.

**Notes.** 204 distinct cues under one family name, so the family grain carries nothing, and on the
78 cue-less recordings neither grain does.

### `picture-description` (889), `-option1` (373), `-option2` (329) — SPEECH

**Requires:** `consensus_transcript` + `word` entities (with `timings` for the gap structure) ·
`spans` (`asr`, `gap`) · `hear_scores` (raw, `Breathe`) · `stream_extent`.
**Absent:** nothing for presence and extent; S4's connected-speech measures are unbuilt branch code,
not a missing derivative.

```python
"picture-description":         Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
"picture-description-option1": Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
"picture-description-option2": Expectation(pattern=Pattern.FREE_RESPONSE, connected=True),
```

**Three identical rows, which is the measurable statement that no method here distinguishes them.**
v1 and `-option1` carry a byte-identical instruction and both have empty `stimulus_text`; any
difference found is the image's. **Spans proposed: `task_extent` plus one per breath group.**

**Notes.** `picture-description` and `-option1` carry a **byte-identical** instruction and both have
empty `stimulus_text`; the only declared difference is the image, which is not in the sidecar as
anything a method can read. **No method here may distinguish them** — any difference found is the
image's. `-option2` asks for complete sentences as though describing it for the blind, which is a
different instruction and the same detector.

### `open-response-questions` (199) — SPEECH

**Requires:** as `picture-description`, plus the one shared 447-character prompt.
**Absent:** `stimulus_alignment`† (D1) only if the prompt is to be excluded as an echo; presence and
extent need nothing.

```python
"open-response-questions": Expectation(pattern=Pattern.FREE_RESPONSE, token_source="stimulus_text",
                                       declared_duration_s=30.0, connected=True),
```

The one connected-speech family whose expectation is legitimately family-scoped: a single
447-character prompt shared by all 199. **Spans proposed: `task_extent` plus one per breath
group** — the same matcher as `picture-description`, with a duration count attached.

**Notes.** One `stimulus_text` shared by all 199 recordings makes this the single connected-speech
family whose `expected_speech` is legitimately family-scoped; every other one is per recording or
absent. Median 30 s.

### `animal-fluency` (195) — SPEECH

**Requires:** `consensus_transcript` + `word` entities · `spans` (`asr`, `gap`) · `stream_extent` ·
`declared_duration_s` (60 s timer).
**Absent:** category membership — branch-local, needs no waveform, **not** a PREPROCESS derivative;
`transcript_repeat`‡ as a store measurement.

```python
"animal-fluency": Expectation(
    pattern=Pattern.ITEM_LIST,
    declared_duration_s=60.0,
    repetition_allowed=False,
    unviable=(("category_membership", "a lexicon or a text embedding, one consumer, no waveform"),),
),
```

**Spans proposed: one**, `task_extent` over the hull of the produced items. **An item gets no span
of its own** — it is one `word` entity with its own extent — and `repeated_item`, the eleventh
deviation type this document owes the vocabulary, is a deviation over that word's extent.

**Notes.** The category lives only in `instructions`; `stimulus_text` is empty on all 195. Median
duration 60 s, matching the declared timer, so the duration check is the row's sharpest free signal.

### `random-item-generation` (265), `-v2` (207) — SPEECH

**Requires:** as `animal-fluency`, plus the **per-recording category**, which no grain above the
recording carries.
**Absent:** the same two, and one more: `p_repetition_allowed` must come from the recording's own
category, not from the family.

```python
"random-item-generation":    Expectation(pattern=Pattern.ITEM_LIST, repetition_from_category=True,
                                         unviable=(("category_membership", "…"),)),
"random-item-generation-v2": Expectation(pattern=Pattern.ITEM_LIST, repetition_from_category=True,
                                         unviable=(("category_membership", "…"),)),
```

**`repetition_allowed` is deliberately not a field here.** It is read from the recording's own
category, and when the category cannot be read the matcher **proposes nothing and returns
`UNDETERMINED`** rather than defaulting to the forbidding rule — because `Letters` and `Numbers` say
the opposite, on 48 of 265 v1 and 77 of 203 English v2 recordings, and a family-scoped default
inverts the instruction on roughly a fifth of them.

**Notes.** `transcript_repeat` — largest repeat count of any normalised token — is already computed
in `routing_analysis/features.py`, where it was read by `ddk.lexical_repetition` until that gate was
removed; it is not a store measurement. Moving it is a convenience, not the capability: a counter
over normalised consensus tokens is arithmetic over word entities the store already holds.

### The AIRWAY families — two instruments every one of them uses

Written once in [§ The three instruments](#the-three-instruments); the family blocks below name them.

`events_in_span` is written once in
[§ The three instruments](#the-three-instruments) and is the same body every AIRWAY and DDK row
calls. A5/A6's four operating points — `p_smoothing_window_s`, `p_peak_prominence_db`,
`p_trough_return_db`, `p_event_min_s` — are what it is owed, and they are the most load-bearing
unfitted cut in this document: twelve task blocks and roughly 21,000 recordings wait on them.

`sounds_like` is written once in
[§ The three instruments](#the-three-instruments). Every AIRWAY body reads `raw_scores` through it
and applies its own named floor, because `labels` is a top-K ∩ threshold decision over those scores
(`default.yaml:91-107`) and a cough label ranked fifth is dropped by a size rather than by a score.

**`detect_lexical_intrusion` is now `detect_speech`'s out-of-family mode**, and the change is not
only structural. The old body returned `done = (no lexical word was found)`, reading the absence of
lexical content as *"the negative pattern held"*. Under the two-mode rule that is wrong: an
AIRWAY-declared recording is not SPEECH's task, so SPEECH has no *"was it done"* to answer on it and
returns `UNDETERMINED`.

The extents survive twice over, from both sides. SPEECH proposes a `family: "speech"` span over each
lexical run; AIRWAY's `lexical_intrusions(store)` emits the `off_task_extent` deviation, because
AIRWAY owns that deviation ([`branch-airway.md:164-165`](branch-airway.md)) and is the branch whose
task the material is off. Measured examples are examiner speech — *"I'll have you do that one more
time. [breath]"*, *"So just breathe."*

Measured examples:
[`../20260910-taxonomy-routing-evidence/measurements.md:177-182`](../20260910-taxonomy-routing-evidence/measurements.md).

### `respiration-and-cough-cough` (1,788) — AIRWAY and SPEECH

**Requires:** `spans` (`amplitude`, `gap`) · `energy_envelope` · `span_hear` · `span_yamnet` ·
`consensus_transcript` + `word` entities (for the intrusion row) · `stream_extent`.
**Absent:** nothing. Every instrument is in the store; A5/A6's operating points are decisions.

```python
"respiration-and-cough-cough": Expectation(pattern=Pattern.EVENT_SERIES, label_set="cough",
                                           expected_event_count=5),
```

**Spans proposed: one per event, plus `task_extent` over their hull.** One per event is the point:
`by_label` increments once per (span, label) pair (`airway.py:280`), so a 4 s span holding three
coughs counts 1 today, and the count compared against `expected_event_count` is the number of these
spans. Each event span names the carrier amplitude span it was found in, plus `energy_envelope`,
`span_hear` and `span_yamnet`, in `wasDerivedFrom`.

**Notes.** The count is of **events**, not of label-carrying spans: `by_label` increments once per
(span, label) pair (`airway.py:280`), so a 4 s span holding three coughs counts 1 today. That is the
defect the `events_in_span` decomposition removes, and it is branch code.

### `respiration-and-cough-v2-hardcough` (698) — AIRWAY, SPEECH and QUALITY

**Requires:** `spans` · `energy_envelope` · `span_hear` · `span_yamnet` · `level` ·
`spectrogram_wideband` · `disruptions_file` · `stream_extent`.
**Absent:** `band_profile`† (D3) for the hygiene clause's spectral-tilt half;
`phonation_tracks.rms_dbfs`† (D2).
**No viable approach** for *"hard"* — see the note.

```python
"respiration-and-cough-v2-hardcough": Expectation(
    pattern=Pattern.EVENT_SERIES,
    label_set="cough",
    expected_event_count=None,
    unviable=(("effort_absolute",
               "no within-recording contrast and no SPL reference; `hard` is not measurable"),),
),
```

**The `None` count and the `unviable` row are the whole difference from the 5-cough family** — same
matcher, same spans, one fewer declaration and one more admitted impossibility. **Spans proposed:
one per event plus `task_extent`.**

**The hygiene clause is QUALITY's, and it is not a separate function.** *"Do not cover your mouth
or place your hand between your mouth and the microphone"* is the only recording-hygiene clause in
the corpus, and it is an expectation **about the recording** attached to an AIRWAY task rather than
a task of QUALITY's own — which is why it folds into `detect_quality`, the mode that runs on every
recording, rather than becoming an `align_quality` that could never have a family. AIRWAY's row for
this family carries the cough events and says nothing about the microphone; QUALITY carries the
tilt finding and says nothing about the cough. It is also the only place D3 has a job that is not
the route question.

**Notes.** This is the only instruction in the corpus carrying a recording-hygiene clause, and it is
the only place D3 has a job that is not the route question.

### `voluntary-cough` (327) — AIRWAY and SPEECH

**Requires:** as `-cough`, plus `hear_scores` / `span_hear` `Breathe` for the interleaved breaths.
**Absent:** the same as `-hardcough` for the effort half.

```python
"voluntary-cough": Expectation(pattern=Pattern.EVENT_ALTERNATION, label_set="cough",
                               expected_event_count=3),
```

**The one AIRWAY family needing its own matcher**, because the expected pattern is an alternation:
material between coughs is matched as breath rather than scored as `off_task_extent`, and a cough
detector alone is insufficient. **Spans proposed: one per cough, one per breath, plus
`task_extent`** — both are expected, so both are minted. Median 13.8 s against `v2-hardcough`'s
4.6 s, consistent with three cough-and-breathe cycles in one file.

**Notes.** Median 13.8 s against `-v2-hardcough`'s 4.6 s, consistent with three cough-and-breathe
cycles in one file.

### `respiration-and-cough-fivebreaths` (3,576) — AIRWAY and SPEECH

**Requires:** `spans` (`amplitude`, `gap`) · `energy_envelope` · `span_hear` · `span_yamnet` ·
`hear_scores` · `stream_extent` · `hints.metadata.task_token`‡ for the route index.
**Absent:** `band_profile`† (D3).
**No viable approach** for the route itself.

```python
"respiration-and-cough-fivebreaths": Expectation(
    pattern=Pattern.EVENT_SERIES,
    label_set="breath",
    expected_event_count=5,
    route_from_index=True,
    unviable=(("route",
               "the discriminating band sits above the 8 kHz ceiling and the residual tilt is "
               "confounded, one for one, with mouth-to-microphone geometry"),),
),
```

**Spans proposed: one per breath cycle, plus `task_extent`.** `route_from_index=True` is the one
field that reads the trailing task index, and it is the field where the collapse bites: the route is
per recording, the index carries it, and `task_family` strips it (`families.py:143`). The route
itself is reported `NOT_SEPARABLE_BY_THIS_DESIGN` with `band_profile†`'s content band beside it, so
a negative is attributable to a measured band limit rather than recorded as *"not yet fitted"*.

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
any branch sees it, because `task_family` collapses the trailing segment (`families.py:134-143`).

### `respiration-and-cough-v2-threebreathsnose` (699), `-threebreathsmouth` (699) — AIRWAY and SPEECH

**Requires:** as `fivebreaths`, with the route declared by the family rather than by the index.
**Absent:** `band_profile`† (D3). **No viable approach** for the route.

```python
"respiration-and-cough-v2-threebreathsnose":  Expectation(pattern=Pattern.EVENT_SERIES,
    label_set="breath", expected_event_count=3, declared_route="nose",
    unviable=(("route", "as `fivebreaths`"),)),
"respiration-and-cough-v2-threebreathsmouth": Expectation(pattern=Pattern.EVENT_SERIES,
    label_set="breath", expected_event_count=3, declared_route="mouth",
    unviable=(("route", "as `fivebreaths`"),)),
```

**`declared_route` is a field here and `route_from_index` is not** — the family name carries the
route, so nothing is lost to the index collapse. Same matcher, same spans: one per cycle plus
`task_extent`.

**Notes.** These two and the `fivebreaths` index split are the only places the route is a declared
contrast, and therefore the only design that could ever validate A7 — which is why the argument
above matters: neither should be treated as validation-grade for it.

### `respiration-and-cough-threequickbreaths` (1,718), `-v2-threebreaths` (699) — AIRWAY and SPEECH

**Requires:** as `fivebreaths`.
**Absent:** nothing beyond `band_profile`†; the interval measurement is arithmetic over
`events_in_span`'s output.

```python
"respiration-and-cough-threequickbreaths": Expectation(pattern=Pattern.EVENT_SERIES,
    label_set="breath", expected_event_count=3, timed_intervals=True),
"respiration-and-cough-v2-threebreaths":   Expectation(pattern=Pattern.EVENT_SERIES,
    label_set="breath", expected_event_count=3, timed_intervals=True),
```

**`timed_intervals` is what turns the count into an interval measurement**, and it is a field rather
than a function. *"Quick"* is the measurement and it is the interval, not the count: three breaths
says nothing about whether they were quick. **Spans proposed: one per cycle plus `task_extent`**;
the intervals are a `counts` entry over the onsets, which need no spans of their own.

### `respiration-and-cough-breath` (1,788), `-v2-breath` (699) — AIRWAY and SPEECH

**Requires:** `hear_scores` (raw, `Breathe`) · `span_hear` · `spans` · `energy_envelope` ·
`stream_extent` · `declared_duration_s` (30 s v1, 20 s v2).
**Absent:** `band_profile`† (D3) for v2's declared mouth route.

```python
"respiration-and-cough-breath":    Expectation(pattern=Pattern.SOUND_COVERAGE, label_set="breath",
                                               declared_duration_s=30.0),
"respiration-and-cough-v2-breath": Expectation(pattern=Pattern.SOUND_COVERAGE, label_set="breath",
                                               declared_duration_s=20.0, declared_route="mouth",
                                               unviable=(("route", "as `fivebreaths`"),)),
```

**Uncounted and durational, so the structure is runs rather than events.** **Spans proposed: one per
merged run of `Breathe`-scoring HeAR windows, plus `task_extent`.** `residual.energy_fraction` is
**not** read: `residual = plain − g·FRCRN(plain)` and FRCRN is a speech enhancer, so a high fraction
means *this is not speech* — satisfied equally by a cough, a glide, room noise and a near-silent
file. As the `airway.breath` routing gate (`default.yaml:285-288`) *"not speech"* may be adequate;
as this row's presence measurement it is not.

**Notes.** Both declared durations are honoured — 1,440 of 1,788 v1 recordings run 30-31 s, 638 of
699 v2 run 20-21 s — and on v1 the duration check is the sharpest "was the task done" signal in the
corpus: **132 of 1,788 run under a second.**

### `breath-sounds` (326) — AIRWAY and SPEECH

**Requires:** as the three-breath families, plus `declared_duration_s` (~73 s: 60 s relax + the
task).
**Absent:** `band_profile`† (D3). **No viable approach** for the declared mouth route.

```python
"breath-sounds": Expectation(
    pattern=Pattern.EVENT_SERIES, label_set="breath", expected_event_count=3,
    declared_route="mouth", relax_s=60.0, declared_duration_s=73.0,
    unviable=(("route", "as `fivebreaths`"),),
),
```

**`relax_s` is the only field of its kind in the corpus**, and it exists because one instruction
prescribes material that is not the task. The declared 60 s is emitted as an `off_task_extent`
deviation — never as a span, since it is ground the branch is disclaiming — and only on a recording
long enough to contain it, the measured median being 13.2 s against a declared ~73 s. **Spans
proposed: one per breath plus `task_extent`**, as the three-breath families.

**Notes.** *"Please relax for 60 seconds until the task starts. Take three deep breaths in a row in
and out of the mouth."* — so it is **not** the uncounted durational task its name suggests; it is
closest to `-v2-threebreathsmouth`. The measured median is 13.2 s, so the relax period is evidently
not inside the file, and the branch above emits the by-instruction `off_task_extent` only on a
recording that actually runs long enough to contain it.

### The `SYLLABLE_REPETITION` families (7,989) — SPEECH's row over eight of the ten

```python
for family in ("diadochokinesis-pa", "diadochokinesis-ta", "diadochokinesis-ka",
               "diadochokinesis-pataka", "diadochokinesis-v2-puh", "diadochokinesis-v2-tuh",
               "diadochokinesis-v2-kuh", "diadochokinesis-v2-puhtuhkuh"):
    SPEECH_EXPECTATIONS[family] = Expectation(pattern=Pattern.NO_LEXICAL)
```

**Eight identical rows, and `buttercup` is not among them** — it has a lexical pattern and takes the
`ORDERED_TOKENS` row instead. **Spans proposed: none, and that is the finding.** The expectation is
that no lexical content occurs, so a branch that proposed a `family: "speech"` span here would
assert the opposite of what it measured. These families are **positives** for SPEECH's reference set
(`default.yaml:241`), so near-zero lexical content is the correct observation rather than a miss;
`speech.lexical >= 2` firing here is over-routing on function-word artefacts.

### The DDK trains — one instrument they share

`train_rate_hz` is written once in
[§ The three instruments](#the-three-instruments), and it is the instrument for the rate and for the
onsets alike. Praat's `extract_speech_rate` is **not** the instrument and is ruled out at
[`preprocess-derivatives-for-expected-patterns.md`](preprocess-derivatives-for-expected-patterns.md)
§ 4.3 — it is already running inside `praat_features` on every recording
(`praat_parselmouth.py:1561`), and its two failure modes both bias the measurement in the direction
of the quantity being measured: `min_dip` (`:218` sets 4, `:223-224` drops it to 2 when a whole-file
HNR test reads under 60) under-counts exactly the fastest trains, and its silence tier is built with
`min_pause = 0.3 s` (`:228`), longer than an entire DDK syllable cycle.

### `diadochokinesis-pa` (896), `-ta` (896), `-ka` (896) — DDK and SPEECH

**Requires:** `energy_envelope` · `spans` (`amplitude`, `gap`) · `continuity_trace` ·
`consensus_transcript` + `word` entities (for SPEECH's absence row) · `stream_extent`.
**Absent:** nothing. D1–D6 are unbuilt branch code and unfitted operating points, not missing
derivatives.

```python
"diadochokinesis-pa": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
"diadochokinesis-ta": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
"diadochokinesis-ka": Expectation(pattern=Pattern.SYLLABLE_TRAIN, expected_event_count=10),
```

**Three identical rows, one per target syllable, because nothing in the matcher reads which syllable
it is.** **Spans proposed: one**, `task_extent` over the train. **An individual syllable is not a
span**: rate, inter-onset interval variability and train fraction are statistics over the onset
series, and one span per syllable would add roughly 30 per recording over 7,989 recordings carrying
no measurement of their own. The onsets travel as a `counts` entry. When no carrier clears
`p_train_min_s` with a prominent modulation peak, the matcher proposes nothing and returns `False` —
a measurement, not an absent instrument.

**Notes.** v1 states the count (10) and v2 does not. The gate that fires without a transcript reads
`ppg.segment_rate_per_s`, which is owed a code change — `extract_ppg_segments`
(`tasks/features_extraction/ppg.py:349`) is called from `routing_analysis/features.py:421`,
`ppg.py:467`, `ppg.py:543` and `plotting.py:1667`, never from a node, so the store holds the
whole-file posteriorgram alone with no per-span query. The function above needs neither.

### `diadochokinesis-v2-puh` (702), `-tuh` (702), `-kuh` (702) — DDK and SPEECH

**Requires:** as above, plus `declared_duration_s` (a fixed 5 s timer).
**Absent:** nothing.

```python
"diadochokinesis-v2-puh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
"diadochokinesis-v2-tuh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
"diadochokinesis-v2-kuh": Expectation(pattern=Pattern.SYLLABLE_TRAIN, declared_duration_s=5.0),
```

**`expected_event_count` absent and `declared_duration_s` present is the whole counted/uncounted
difference**, and it needs no second matcher. The fixed 5 s denominator makes the train fraction and
the rate directly comparable across participants in a way v1's participant-terminated recordings are
not. **Spans proposed: one**, `task_extent`.

**Notes.** 630-646 of each family's 702 recordings run 5-6 s, against a v1 median of 5 s spread over
3-9 s. The fixed denominator makes D5's train fraction and the rate directly comparable across
participants in a way v1's participant-terminated recordings are not — the cleanest DDK design in
the corpus, and none of D1-D6 is built to use it.

### `diadochokinesis-pataka` (896), `-v2-puhtuhkuh` (701) — DDK and SPEECH

**Requires:** `energy_envelope` · `spans` · `spectrogram_wideband` · `stream_extent`.
**Absent:** nothing — the instrument this row needs is already written. `ppg_posteriorgram` is
**not** the instrument; see the note.

```python
"diadochokinesis-pataka":       Expectation(pattern=Pattern.SYLLABLE_SEQUENCE,
                                            sequence=("labial", "alveolar", "velar"),
                                            expected_event_count=30),
"diadochokinesis-v2-puhtuhkuh": Expectation(pattern=Pattern.SYLLABLE_SEQUENCE,
                                            sequence=("labial", "alveolar", "velar"),
                                            declared_duration_s=5.0),
```

**The expected sequence is data**, so a four-place train would be a row and not a rewrite. **Spans
proposed: one**, `task_extent` over the train, carrying `production="syllable_sequence"`,
`realised_cycles` and `resolved_n`. A syllable that is not the one the sequence expected travels as
a `syllable_sequence_mismatch` deviation **with its own extent, which needs no span** — deliberately
not `stimulus_mismatch`, since there is no stimulus text and no lexical expectation.

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

```python
# DDK_EXPECTATIONS
"diadochokinesis-buttercup":    Expectation(pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",),
                                            expected_event_count=10),
"diadochokinesis-v2-buttercup": Expectation(pattern=Pattern.ORDERED_TOKENS, tokens=("buttercup",),
                                            declared_duration_s=5.0),
```

**The only DDK row whose `Pattern` is a lexical one**, and `align_ddk` dispatches it to its own
token matcher rather than to the train matcher. **Spans proposed: one**, `task_extent` over the hull
of the realised tokens, derived from `consensus_transcript`, `energy_envelope` and every matched
`word`. SPEECH holds this family in family too, with its own `ORDERED_TOKENS` row and its own
`task_extent`; the two branches measure different things over the same ground — the token count and
the envelope rate — which is the intended shape, not a conflict.

**Notes.** The only DDK family with a lexical pattern, so the only one where the recognisers produce
the count directly — and the only one where the removed `ddk.lexical_repetition >= 3` gate used to
fire for the right reason rather than on function-word repetition. v1 states the count and v2 does
not, so the two are not one measurement.

### Out of family — `detect_voice`, `detect_airway` and `detect_ddk`

The three rows where the branch was routed to a recording whose family declares none of its
patterns — **which is where most of the corpus lands**, so none of the three is a placeholder.
**Declared families are not ground truth, and neither is a routing**: the branch proposes spans over
what it finds, in its own family, and asserts nothing about the other branch's task. All three
return `done = UNDETERMINED`, because *"was it done"* is not their question on this recording.

**Requires (VOICE):** `spans` · `phonation_tracks` · `continuity_trace` · `energy_envelope` ·
`consensus_transcript` + `word` entities. **Absent:** `phonation_tracks.{hnr_db,cpps_db,rms_dbfs}`†
(D2) — and here the stationarity qualifier is load-bearing over 22,277 routed recordings, because
connected speech passes voiced-fraction, F0-availability and interruption tests and would otherwise
have perturbation measured over consonants and pauses.

**`measure_voice_without_declared_task` is now `detect_voice`**, and it is not a stub: it is the
same `qualifying_phonation` body `align_voice` uses, over the same amplitude spans and the same
`phonation_tracks`, with no expectation consulted. It **proposes a `family: "voice"` span over every
sustained voiced region it finds**, each derived from its carrier span plus `phonation_tracks` and
`continuity_trace`, each carrying `evaluates_no_task=True`, and it contests any span a rule labelled
`phonation` that fails the qualifier.

`done` is `UNDETERMINED` always. This is the mode that runs on most of what VOICE is handed —
**22,277 routed recordings, 14,332 of which declare no voice family**
(`runs/ruleset-score-20260912/ruleset_score.json`, `extra.VOICE`) — so a placeholder here would mean
VOICE doing nothing on **64%** of its own corpus.

**Requires (AIRWAY):** `spans` · `span_hear` · `span_yamnet` · `hear_scores` · `energy_envelope`.
**Absent:** nothing.

**`measure_airway_without_declared_task` is now `detect_airway`**, and it is the owner's standing
AIRWAY rule generalised: a non-airway task routed here still gets its breathing and coughing found
and marked. It **proposes a `family: "airway"` span over every cough and breath event**, using the
same `sounds_like` + `events_in_span` machinery as the in-family mode — the event boundaries where
the envelope resolves them, the carrier's own extent where it does not, with `boundaries` recording
which. It contests any span a rule labelled `cough` or `breath` that carries no raw score over the
floor.

**It does not read `airway.cough`.** That gate reads **J −0.1398** against `declared_airway` over
61,721 recordings, so at its 50 dB cut it is a loudness detector over an arbitrary recording, which
is precisely what this mode is handed. A breath during passage reading is **not** a deviation — it
is how S4 measures breath-group structure — so nothing here emits one. AIRWAY's gate evidence is
`unavailable` on 56,505 of 62,547 recordings, so this is the mode that runs on the corpus.

**Requires (DDK):** `energy_envelope` · `spans` · `consensus_transcript` + `word` entities.
**Absent:** nothing.

**`measure_ddk_without_declared_task` is now `detect_ddk`**, and it **proposes a `family: "ddk"`
span over every rapid repetition train it finds** — acoustically from the envelope modulation
spectrum over each amplitude span, and lexically over any normalised token repeated at least
`p_repeat_min_occurrences` times. Not from Praat's `extract_speech_rate`, whose `min_pause = 0.3 s`
exceeds an entire DDK cycle, and not from the PPG.

Repetition occurs in ordinary speech — a stutter, a false start, a repeated word — so the branch
measures what it finds and says what it is; it does not assert that a Harvard sentence failed to be
a DDK task ([`branch-ddk.md:66-70`](branch-ddk.md)). DDK routes **22,363 recordings against the
7,989 that declare a DDK family, 14,878 of them declaring none**
([`branch-ddk.md:20`](branch-ddk.md)), so this mode is almost the whole of what a built DDK
branch would do. DDK is a declared branch with no node: `run.py:304-305` marks it
`SKIPPED` with `NO_NODE`, selected or not.

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
| P1 found | SPEECH transcribes it and it scores an `extra` against the reference set | `align_voice` **proposes a `family: "voice"` span**, `role: "count_in"`, over the matched tokens' extents, carrying `excluded_from_measurement=True`. SPEECH is out of family here and proposes its own `family: "speech"` span over the same run, evaluating nothing |
| P2 found | nothing; VOICE `FAIL` | `align_voice` **proposes a second span**, `role: "task_extent"`, over the voiced run inside the qualifying amplitude span, `wasDerivedFrom` that span plus `phonation_tracks` and `continuity_trace` |
| the scalars | whole-file, over count-in + silence + vowel | over P2's span alone — which is what the two spans exist to make possible |
| "was the task done" | unanswerable | P1 found / absent and P2 found / absent, independently, and `done` is their conjunction |
| the task extent | not recorded | **a proposed span**, `role: "task_extent"` — not a `trim` payload. Under propose-only the extent is a span the branch mints, so `trim` is not its carrier |
| anything else | silent | `off_task_extent` over the `gap` spans it falls in — a `deviate` finding, never a span, because it is ground the branch is disclaiming |

### What is owed to build it

1. ~~**A phonation subject VOICE can select.**~~ **Closed by propose-only, 2026-09-16.** The
   earlier reading was that either the ruleset stamps a label on the amplitude span that fired
   `voice.sustained`, or VOICE's selector widens — both owed a code change, with the refinability of
   a labelled `amplitude` span unresolved at `branch-conventions.md:61-71`. Neither is needed: VOICE
   reads the amplitude spans as **evidence** and proposes its own `family: "voice"` span over what
   qualifies, naming the carrier in `wasDerivedFrom`. The selector, the label-stamping question and
   the refinability question all dissolve together. What is still owed is branch code, as for every
   row in this document.
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
   with its own consumers. Note what propose-only supplies to it: the extent to pass is now a span
   the branch itself minted, so the caller and the extent arrive together.
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
[§ The code](#the-code-two-entry-points-per-branch); this table is the consumer's view of it.

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
| ~~a phonation subject VOICE can select~~ — **withdrawn 2026-09-16** | **0** | — | The blocker was real and the fix is not a derivative: `preprocess.py:1875-1891` writes amplitude spans with no `family` key and `voice.py:230` selects `family == "phonation"`, so VOICE's candidate list is empty before any test runs. Under the owner's propose-only decision VOICE reads those amplitude spans as *evidence* and mints its own `family: "voice"` span, so nothing has to write a family onto them | none. It is branch code, which every row owes anyway |
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
| what makes a `gap` span off-task rather than ordinary silence — `p_gap_off_task_min_s` | **all 31** | 62,547 | `span` `measure: "gap"`, `silence`, `energy_envelope`. Every body above calls `off_task()` and none of them can say where the line falls. It is now one named parameter with one call site rather than a rule restated thirty times |
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

### Closed since the last version

- **~~Which verb carries `off_task_extent`.~~** Half-closed by propose-only: `task_extent` is a
  **proposed span** carrying `role: "task_extent"`, not a `trim` payload, so `trim` is not the
  carrier of the extent either. This document's reading is that `trim` has no remaining job, and
  that `off_task_extent` stays a `deviate` finding — off-task material is the *absence* of the
  branch's speciality rather than an instance of it, and a branch does not mint spans over ground it
  is disclaiming. `design.md:569` and `branch-conventions.md:108` still disagree on paper and whoever
  reconciles them should know the extent question is no longer part of it.
- **~~Whether a label on an `amplitude` span makes it refinable.~~** Closed by propose-only: there is
  no refinement, so there is nothing to scope. `branch-conventions.md:61-71`, `design.md:1174-1181`
  and `branch-voice.md:92-97` all describe a question that the 2026-09-16 decision removes.

### Documentation conflicts, no longer blocking

These were blockers while the branches needed `refine`. Under propose-only they are conflicts
between documents to reconcile, and no branch waits on them:

- **`refine` is written by nothing in `src/senselab`.** Swept 2026-09-16: every occurrence of the
  string is prose or an unrelated identifier — `prov_store.py:438`, `praat_parselmouth.py:95`,
  `audio_analysis`'s `refined_identity` / `I1_boundary_refinement` / `fuse.py:1089`,
  `label_membership.py:82`, and `default.yaml:180`'s comment that QUALITY judges *"against each
  branch's refined spans"*. **Zero call sites.** The verbs actually written are `label`, `confirm`,
  `contest`, `abstain`, `flag`, `attribute`, `measure` and `withdraw`; there is no `propose`, no
  `refine`, no `trim` and no `deviate` anywhere, and `vocabulary.py` declares no verb vocabulary at
  all — the verbs are string literals at their write sites plus `quality.CONTEST_VERB:62` and
  `preprocess.WITHDRAW_VERB:142`.
- **`store.md:72` says `refine` and `withdraw` are "gone as verbs"**, while `design.md:568` declares
  `refine`'s payload and `branch-conventions.md:97-100` records it being *widened* on 2026-09-15.
  Note `store.md:72` is wrong about `withdraw` independently of `refine`: `WITHDRAW_VERB` is
  declared at `preprocess.py:142` and written at `:605`.

### A code simplification the propose-only decision enables

`report.py:291`'s `_spans_of_family(store, family, *, voice=...)` exists only because VOICE re-minted
a second phonation span from an existing one, and `onset_kind` — which VOICE writes and the retired
proposer did not — was the only thing telling the two populations apart. Its own docstring says so.
Under propose-only they are different **families**, so the `voice=` parameter becomes unnecessary.
`_spans_of_family` has five call sites — `report.py:673`, `:688`, `:715`, `:1153`, `:1154` — of
which three pass `voice=` (`:673` `voice=False`, `:715` and `:1154` `voice=True`); those three are
the ones the decision simplifies. **Recorded, not done: this document changes no code.**

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
