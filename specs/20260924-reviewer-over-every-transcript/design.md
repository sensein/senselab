# The reviewer over every transcript

2026-09-24. The LLM PII reviewer moves out of REDACT and becomes its own node, reads both the
original and the redacted transcript, answers four questions instead of one, and proposes the
redaction the audio is then cut from.

No transcript text and no detected string appears in this file.

## The population, measured

From `~/Downloads/recording_vectors_20260924/recording_vectors.parquet`, 62,548 rows, counting
`release` against `release_ground`:

| release | ground | recordings | lexical speech | detectors read it |
|---|---|---:|:-:|:-:|
| `nothing_to_redact` | SPEECH did not run | 19,097 | no | no |
| `nothing_to_redact` | SPEECH ran, no lexical word | 1,421 | no | no |
| `nothing_to_redact` | every lexical word is in the task's own stimulus | **13,810** | **yes** | **no** |
| `nothing_to_redact` | the scan ran and found nothing | 11,526 | yes | yes |
| `releasable` | — | 12,046 | yes | yes |
| `withheld` | — | 4,647 | yes | yes |
| `not_assessed` | SPEECH left no lexical count | 1 | unknown | no |

**42,030 recordings carry lexical speech.** Before this change the reviewer saw 16,693 of them —
the ones with at least one finding — because it was a step inside REDACT and REDACT runs only where
the scan found something.

The 13,810 matter most and are the reason the reviewer could not stay inside REDACT. SPEECH
declines the PII scan where every lexical word sits in the task's own stimulus. That gate is right
in principle, and it is what stops a bracketed filler and a carrier word being called names. But it
widened twice on 2026-09-23: near-match admits a token within two edits of a stimulus word, and
`expected_names` admits a declared cast. A genuine name within two edits of a stimulus word is now
exempt by construction, and nothing — no detector, and until now no reviewer — looks at it.

A reviewer reached through REDACT sits downstream of that gate. It can second-guess the detectors'
output and never the gate that decides whether the detectors run at all.

## D-1. REVIEW is a node, not a step inside REDACT

`GRAPH_ORDER` gains `REVIEW` between `REDACT` and `VERDICT`. It runs on every admitted recording
and writes an annotation on every path, so "no review happened" is a record rather than an absence.

**Rejected: opening REDACT's own gate to every scanned recording.** It was implemented and reverted
in this session. It does not reach the 13,810, because they were never scanned; it puts 11,526
recordings through a second detector pass whose disagreement rate with the first is unmeasured; and
a disagreeing re-scan would move them from `nothing_to_redact` to `withheld`. That is a real
release-axis movement bought for a population the change does not even reach.

REVIEW takes no run directory and opens no audio. Both transcripts come out of the store —
consensus words and the live `redaction` spans — so the review-only driver needs neither the
recording nor a stream decode.

## D-2. It reads both texts

The earlier design (`specs/20260817-triage-workflow-dag/llm-check.md`) deliberately refused to show
the model the original, so that a finding could never be read back in the clear. The owner reversed
this on 2026-09-24, and the reversal is right on its own terms:

- Whether a redaction *removed* the identifying content cannot be judged without what it removed.
- PII the detectors *missed* cannot be found in a text from which the detectors' hits are already
  gone — the surfaces that would have shown it are the ones removed.

Where no redaction was applied, the request says so outright rather than sending an empty section:
the 13,810 and every recording REDACT never reached are the "redacted not available" case, not a
gap.

## D-3. Four judgments, separately answerable

The completion contract was `REASONING:` prose plus a `FINDINGS:` array. It is now five parts:

```
REASONING: <prose>
REDACTION: complete | incomplete | not_applicable
ORIGINAL:  clean | carries_pii
SPEAKERS:  one | more_than_one | unclear
PROPOSAL:  [ {"text", "action": "redact"|"release", "category", "why"} ]
```

They are kept apart because a recording can be correctly redacted and still carry something else,
or be unredacted and carry nothing, or carry two speakers and no PII. An unparsable or absent
heading yields the empty string rather than a guessed answer: the model not answering a question is
different from any of the answers it could give.

`status` — the coarse state VERDICT reads — is `flagged` when the reading says something
identifying is still in what would be released: `REDACTION: incomplete`, `ORIGINAL: carries_pii`,
or any proposal entry asking for a removal. A proposal of only `release` entries is **not** a flag:
it asks for *less* to be removed, which is not a leak.

`SPEAKERS` is a judgment about the recording's content from the words in it — turn-taking, an
examiner giving instructions, questions answered — and never about the processing pass. It is here
because the acoustic detectors have been measured and are weak: pyannote, Sortformer and VibeVoice
all reported one speaker on a recording confirmed to hold two, no embedding statistic of 26 reaches
0.70 AUC as a detector, and MossFormer2's two streams are not a speaker split
(`specs/20260923-second-voice-detector/`). The transcript carries evidence the acoustics do not. It
is a third prong, not a speaker detector, and it is wired to nothing: in particular it does not
touch `dominant_speaker_share_min`, whose bound is declared unfitted and whose readings come from
that same separation.

## D-4. The prompt carries the task's own facts

`_PROMPT` was one fixed string. The model did not know what task the recording was, which is a
defect for every one of the four judgments:

- `cinderella-story` carries 2,989 findings whose names belong to the story.
- A perfectly-read `harvard-sentences` recording contains exactly the prompt's words.
- A time expression's reading depends on context, and 96.5% of 6,397 `DATE_TIME` marks carry no
  calendar anchor (`specs/20260923-free-speech-review-page/datetime-has-no-specificity.md`).
- A task asking for several read sentences has structure that could look conversational.

The request now carries `TASK`, `ASKED TO SAY` and `NAMES THE TASK'S OWN MATERIALS CONTAIN`, each
omitted where the recording declares none, followed by one sentence:

> Those lines are facts about what was asked for. They are not a conclusion about what is in the
> transcript, and neither matching them nor departing from them settles any of the four questions
> on its own.

**Why worded that way.** `llm-check.md` records that an earlier prompt asking the model to "explain
how you read it off the image" led it to infer the expected answer rather than reason. Telling it
"this is a Cinderella retelling, so names are expected" is the same failure: it would excuse a
genuine disclosure that happens to sit beside story names. The facts are given; the conclusion is
not.

The context makes the prompt per-recording and therefore longer, so input tokens rise. They are now
recorded separately from output tokens on every round.

## D-5. The proposal becomes the applied redaction

The owner's direction, in two steps: first that the reviewer output "a form that refines the
redaction or removes it if unnecessary", then that "the output of this should then be used to
create redacted audio", then that only one artefact is kept.

`refine_plan` maps the proposal onto extents. The proposal is about *text* and an extent is about
*time*, so the consensus words' own timings are the join: the transcript is rendered as one string
with each word's character range recorded, a quoted substring is located in it, and the words it
covers give the bounds.

**An entry that cannot be located fails closed.** A `redact` it cannot place keeps every detector
span and is recorded in `unplaced`; a `release` it cannot place changes nothing. This is the one
rule that must not be inherited from the existing path: today an unlocatable finding is widened to
cover the whole transcript, which is where a 9.1 s `PERSON` span over a participant's sustained
vowel came from.

`apply_proposal` writes the recording's redacted stream from the refined set, reusing REDACT's own
`apply_redactions` and `write_stream`. There is no second artefact. What says the audio came from
the reviewer's span set is the store:

- every applied span is generated by REVIEW's `apply` activity and derived from the annotation;
- every applied span carries `why`, the reviewer's own one-sentence reason, so "why is this span no
  longer redacted" has an answer in the store;
- the detector spans the proposal released are `wasInvalidatedBy` the same activity — invalidated,
  not deleted.

## D-6. What the reviewer can and cannot change

Unchanged and load-bearing: **the reviewer writes no verdict.** REVIEW leaves no `verdict` entity at
all, so it cannot reach the release axis by construction.

Its annotation reaches VERDICT under two keys:

| key | default | what it does |
|---|---|---|
| `verdict.llm_redaction_flags` | `true` | a `flagged` reading is a triage flag ground |
| `verdict.llm_redaction_withholds` | `false` | **UNFITTED.** a reading of residue withholds a recording REDACT passed |

`llm_redaction_withholds` is the weighting the owner asked for — VERDICT favouring the reviewer over
the detectors — and it moves the release axis in exactly one direction: tightening. It turns a pass
into a withholding and never the reverse. `absent`, `disabled` and `nothing_to_read` are not
readings and ground nothing under it, so a GPU queue cannot withhold a recording.

**The floor: nothing moves a withheld recording to released but a person.** No key here can do it.

It ships `false` and declared unfitted. The reviewer has run on ~132 recordings and disagreed with
the detectors 41 times; that is not a fit, and `dominant_speaker_share_min: 0.9` is the precedent
for declaring a bound unfitted rather than inventing one.

**What applying a proposal requires.** The proposal is applied by `apply_proposal`, which is called
only by the driver under `--apply` and by nothing in the graph's default path. So a corpus pass that
wants refined audio asks for it explicitly; a pass that does not gets the annotation and nothing
else. The refined audio is what would be released, but whether it *is* released is still the release
axis's answer, and the reviewer does not set that.

## D-7. The state vocabulary

`not_run` retires. It meant "the detectors marked nothing, so there was nothing to review", which
under REVIEW names a population that is read rather than one that is skipped. The four distinctions
stay separately legible:

| state | means |
|---|---|
| `disabled` | the configuration leaves the reviewer off |
| `nothing_to_read` | the transcript carries no words |
| `absent` | it tried and the model would not load |
| `clean` / `flagged` | it read the text and concluded |

Beside them the annotation carries `detector_state` — `scanned`, `declined`, `unscanned` — which is
what makes the gate legible: a `flagged` beside `declined` is the scan gate having let something
through, and `detector_findings_n == 0` beside `flagged` is a contradicted clean scan.

`detector_state: unscanned` deliberately does not reuse the name `not_run`, so no reader can confuse
a detector state with the retired review state.

The free-speech page renders all of this and `EXTRACT_VERSION` goes to 4, so an extract written
under the old ladder is refused rather than described. Its test that no wording claims a review
happened when none did still passes.

## Open: the disease-name class

The owner named two categories. One is measured: time expressions are over-redacted, 96.5% of
`DATE_TIME` marks carrying no calendar anchor.

The other — specific disease names under-redacted — is **new and unmeasured**, and it is the more
dangerous direction, since a rare diagnosis can identify a person. Nothing in the detector stack
targets it. The corpus offers no way to measure it: there is no labelled set of recordings in which
a diagnosis was disclosed, and constructing one by searching for diagnosis terms would only recover
what a term list already knows, which is the thing a reviewer is supposed to improve on. Only human
review over a sample of what the reviewer flags as `CONDITION` can establish a rate. The prompt
names the class so the reviewer can raise it; nothing here claims to have measured it.
