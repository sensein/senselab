# The third artefact family: a proper noun the task itself supplies

Measured 2026-09-23 over the 11,701 free-response recordings of
`triage_rerun_20260923/out`. Counts, categories and detector names only; no detected string
appears here.

## Three artefacts, not one

| | what reached the scanner | what handles it |
| --- | --- | --- |
| `[UH]` as `PERSON` | a transcription convention, not speech | the bracket exclusion in `_scan_tokens` |
| a stimulus word as `PERSON` | a word the participant was handed on a card | the `in_stimulus` haystack, and the exemption pass |
| **a task-inherent proper noun** | **a name that belongs to the task, on a task with no stimulus text** | **nothing** |

The third has no mechanism at all, because the mechanism that would catch it — comparing the
finding against the task's declared text — has nothing to compare against. `in_stimulus` is then
`None`, which is a third value and not a quieter `False`: it means *the question was not asked*.

## Size

**4,537 of 22,319 findings (20.3%) sit on recordings whose task declares no stimulus text.**

| family | recordings | with stimulus | findings | unchecked | in stimulus | checked, not in it |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cinderella-story` | 258 | **0** | 2,989 | **2,989** | 0 | 0 |
| `picture-description` | 889 | **0** | 748 | **748** | 0 | 0 |
| `picture-description-option2` | 329 | **0** | 436 | **436** | 0 | 0 |
| `picture-description-option1` | 373 | **0** | 308 | **308** | 0 | 0 |
| `productive-vocabulary` | 2,910 | 1,111 | 2,101 | 56 | 810 | 1,235 |
| `free-speech-v2` | 2,120 | 1,560 | 5,275 | 0 | 555 | 4,720 |
| `story-recall-v2` | 660 | 509 | 3,147 | 0 | 2,318 | 829 |
| `story-recall` | 889 | 794 | 2,561 | 0 | 1,493 | 1,068 |
| `free-speech` | 3,074 | 1,831 | 4,581 | 0 | 298 | 4,283 |
| `open-response-questions` | 199 | 87 | 173 | 0 | 3 | 170 |
| **total** | **11,701** | **5,892** | **22,319** | **4,537** | **5,477** | **12,305** |

Recordings carrying at least one unchecked finding: `picture-description` 319 of 889,
`cinderella-story` **227 of 258**, `picture-description-option1` 119 of 373,
`picture-description-option2` 99 of 329, `productive-vocabulary` 35 of 2,910.

Categories among the unchecked: `PERSON` 2,127, `NAME` 1,497, `DATE_TIME` 592, `MISC` 172,
`LOCATION` 105, `NRP` 28, the rest single digits. Detectors: `presidio` 1,973, `rules/ner` 1,171,
`gliner/name` 876, `rules/gazetteer+ner` 294 — spread across all three engines, so this is not one
detector misbehaving.

**`cinderella-story` is 2.2% of the recordings and 65.9% of the unchecked proper-noun load** — 2,618
of 3,936. A thirty-fold concentration.

## Which families have task-inherent proper nouns that are predictable in advance

This is the question that decides whether an allowlist is even definable, and the families differ
sharply.

**`cinderella-story` — yes, closed and knowable.** One physical storybook, handed to every
participant. The cast is fixed before any recording exists, and small. `stimulus_text` is empty on
all 258 by design, which `expected-patterns.md` already records as the reason no overlap measure is
definable here. This is the one family where a list could be written down today, in advance,
without looking at a single transcript.

**`picture-description` ×3 — no, not from text.** 1,591 recordings, 1,492 unchecked findings, and
the stimulus is an image URL. What a describer names is open: they may name what they see, or a
person the scene reminds them of. A per-image list could be derived from each image, but that is a
vision problem and it is per-image, not per-family — and `picture-description` and `-option1` carry
byte-identical instructions and differ only by image, so a family-level list would be wrong for one
of them by construction.

**`story-recall` and `story-recall-v2` — already handled, and the machinery demonstrably works.**
They declare the source story as `stimulus_text` on 794 of 889 and 509 of 660, so the haystack
applies: 1,493 and 2,318 findings read `in_stimulus: true`, and the exemption pass actually granted
581 and 1,331 exemptions. These are the control that shows the mechanism functions when it has
something to work with. The gap is the 95 and 151 recordings that declare none.

**`free-speech`, `free-speech-v2`, `open-response-questions` — open by nature.** The prompt is a
question; a name in the answer is the participant's own. No task-inherent nouns to list, and 0
unchecked findings, because the question text is always declared.

**`productive-vocabulary` — a small, real gap.** 204 distinct cue words, but 78 recordings carry no
cue. That shows up as 56 unchecked findings on 35 recordings. Narrow, and fixable by supplying the
missing cue rather than by an allowlist.

So the answer is one family, not four: **`cinderella-story` has a known, closed cast of
task-inherent proper nouns and no stimulus text to declare them in.**

## No fix shipped

A per-family allowlist derived from the task is the obvious candidate and it is the owner's call.
Three things worth knowing before that call:

1. It would be a *declaration*, not a threshold — the Cinderella cast is a property of the task, so
   it belongs with the task's expectation row, not in a fitted profile. That makes it cheap and
   auditable, and unlike a tuned cut it cannot drift.
2. It changes what `in_stimulus` means. Today `None` honestly says "not asked". Supplying an
   allowlist would make it `False` for a genuinely disclosed name and `True` for Cinderella, which
   is the right answer — but the three-valued distinction must survive for the families where no
   list is possible.
3. `picture-description` would still be uncovered, and it is 1,492 unchecked findings across 1,591
   recordings. An allowlist fixes the concentrated case and not the broad one.

The review layer is the instrument that turns this into evidence. A reviewer working through
`cinderella-story` with the category facet set to `PERSON`/`NAME` and marking `not-identifying`
would produce a measured rate and, incidentally, the allowlist itself.

## How the page shows it

Every card's *what determined this status* panel carries a stimulus-check section. Where the task
declares no stimulus text it says so outright — that no finding could be checked against it, none
was exempted, and a proper noun belonging to the task is indistinguishable here from one the
participant disclosed. Each finding's row reads `no stimulus to check` rather than a blank or a
`false`.
