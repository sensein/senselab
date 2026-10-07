# REDACT's re-scan read its own placeholders, and the reviewer decides a re-scan fail

2026-09-26. Owner: *"isn't it the reviewer's call"*, then *"diagnose, then reviewer rule, but the
reviewer is already assessing these date time ones"*.

## What was withheld

The settled r5 corpus (62,550 recordings, re-folded at `ef2bb815`) withholds 3,700. Of those, 416
carry the reviewer's own ground and **3,284 were withheld by REDACT itself**. All 3,284 are a REDACT
`fail` with `unremediable` non-empty: *"verification found pii on the redacted transcript"*, after the
one re-plan. None is an incomplete scan. Categories surviving: DATE_TIME 1,961 (1,786 DATE_TIME
alone), PERSON 1,302, LOCATION 92, then a tail (VEHICLE_IDENTIFIER 31, NAME 28, US_SSN 17).

What the reviewer read on those 3,284 (parquet `recording_vectors_r5_v7`, exact):

| original | proposes a redaction | n |
| --- | --- | --- |
| clean | no | **2,585** |
| carries_pii | yes | 418 |
| carries_pii | no | 250 |
| clean | yes | 30 |
| (no reading) | no | 1 |

An earlier count said 2,557, with 27 REDACT passes and 29 recordings with no REDACT verdict among
the withheld. Both were a probe defect: it located a store by `glob(stem + "*")`, and a stem such as
`…_task-free-speech` also matches `…_task-free-speech-1_…`, so it read another recording's store.
Every figure here keys on the parquet's `run_dir`.

## The mechanism

The verifier re-scans `_verification_text(_render(residue, planned))`: the residue with each planned
extent folded into a placeholder, `[DATE_TIME]`, `[PERSON+NAME]`. GLiNER labels the placeholder as
the entity it names (`gliner/date` on `[DATE_TIME]`, `gliner/name` on `[PERSON+NAME]`, and sometimes
the bare `DATE_TIME`). The re-plan widens only to a *word* the store marks with the surviving
category and no extent covers, and a placeholder is not a word, so nothing widens, the second
re-scan reads the same placeholder, and the recording is `unremediable`.

Replayed offline over a random 400 of the 3,284 (`rescan_s400.json`; the same detectors, the stored
plan):

| every re-scan span is | recordings |
| --- | --- |
| a placeholder | **348 (87%)** |
| a new span on unmasked words only | 27 (7%) |
| a placeholder and a new span | 25 (6%) |

The new spans are detector readings the first scan did not report on words it left alone —
`couple of weeks`, `summer`, `Cinderella`, `dad`, `Dr.`, `fried` — found once the masked neighbour
changes the context. They are real survivors by the verifier's definition and are left as such.

## The fix

A re-scan span with no word character left once every placeholder the rendered text carries is
removed (bracketed, bare, and each category a merged placeholder joins) is the redaction, not a
survivor (`_mask_tokens`, `_is_mask`). The rendering is unchanged, because `transcript_texts` —
what REVIEW reads — renders through the same `_verification_text`; changing the placeholder there
would change every reading's input.

The reviewer's input is therefore unchanged by the fix, and so is the plan wherever the placeholder
was the only survivor: the one re-plan widened nothing there.

## The reviewer rule

`verdict.llm_rescan_clears` (on). Where REDACT failed with re-scan survivors, and the reviewer read
the original as `clean` and proposed no `redact`, the fold releases **the redacted copy** under
`REVIEWER_CLEARED_RESCAN`. REDACT writes the released triple only on a pass, so the fold writes it
from the store (`settle_release`: the masked `redacted` stream and the planned spans over the
consensus words), and removes it if a later fold withholds.

Not cleared: an incomplete scan or re-scan (it names no survivor), a reading proposing a redaction,
an original read as carrying PII, and every recording with no reading.

## What each moves, on the 3,284

From the 400-sample for the fix, exact for the rule:

| applied | released with redaction | still withheld |
| --- | --- | --- |
| the rule only (a re-fold) | **2,585** (exact) | 699 |
| the fix only (REDACT re-run) | ~2,540 (309/400) | ~750 |
| both | ~2,890 (352/400) | ~395 |

The fix adds, over the rule, about 270 recordings (33/400): placeholder-only survivors whose reading
found the original carrying PII but proposed nothing to hide. A REDACT pass releases those; the rule
does not, by design. Under both, a placeholder-only recording that the reviewer also cleared carries
REDACT's own pass (ground None) rather than `REVIEWER_CLEARED_RESCAN`.

## Applying it to r5

The rule needs only a re-fold: CPU, no REDACT, no reading redone, since REVIEW's input
(`transcript_texts`, built from the stored plan) does not change. The fix needs REDACT to run again,
and the replay driver re-runs everything from TAXONOMY, REVIEW included, retiring every reading
(`REPLAYED_NODES`), so applying it to r5 through that driver costs the whole GPU review. It protects
every later run as it stands.
