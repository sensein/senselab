# Measurement

Same harness and same sample as
[`../20260919-diarization-turns-past-the-decode/measurement.md`](../20260919-diarization-turns-past-the-decode/measurement.md):
a uniform random 800 run directories (seed 20260919) from the 62,521 complete stores of the Sept-8
corpus, replayed through `speech(store, "plain", config, run_dir=..., enrollment=None)` with no
hint — the family comes off the BIDS stem, which is the carrier the corpus run itself has.

**Before** is the clamp-fix-only source tree, so SPEECH actually runs and reports; comparing against
the unfixed base would confound this defect with the one that killed the branch outright.
**After** is both fixes.

## n

`23148627`, an 8-way array on `mit_preemptable`. **Six slices completed — 1, 2, 3, 5, 6, 7 — giving
600 of the 800 recordings**, of which **116 sit on `FREE_RESPONSE` rows**. Slice 0 was mid-run and
slice 4 had produced no output when this was written; both are excluded, and the 600 are taken
whole rather than sampled from.

Families in the 116: `free-speech` 29, `free-speech-v2` 25, `productive-vocabulary` 25,
`story-recall-v2` 10, `picture-description-option1` 8, `story-recall` 8, `picture-description` 4,
`picture-description-option2` 3, `open-response-questions` 2, `cinderella-story` 2.

## Before and after

```
BEFORE (clamp fix only)          AFTER (clamp + free response)
  False         66/116  56.90%     False          3/116   2.59%
  True           3/116   2.59%     True          95/116  81.90%
  UNDETERMINED  47/116  40.52%     UNDETERMINED  18/116  15.52%
```

Per-row, with no recording moving from a reading to a worse one:

```
False        -> True           63
UNDETERMINED -> True           29
UNDETERMINED -> UNDETERMINED   18
True         -> True            3
False        -> False           3
```

## The three groups move exactly as the rows predict

```
anti_pattern=None             n=69   before {False: 66, True: 3}   after {True: 66, False: 3}
anti_pattern=verbatim_prompt  n=29   before {UNDETERMINED: 29}     after {True: 29}
anti_pattern=verbatim_source  n=18   before {UNDETERMINED: 18}     after {UNDETERMINED: 18}
```

- **No anti-pattern row** — nothing ever masked these, so they read the broken term directly.
  66 of 69 said the participant failed; all 66 now read a response.
- **`verbatim_prompt`** (`free-speech`) — masked as `UNDETERMINED` by the absent stimulus. All 29
  now report the response they had in fact measured, and none reads `False`.
- **`verbatim_source`** (`story-recall`, `story-recall-v2`) — the term is reassigned from
  `source_content_coverage`, so an unreadable source still leaves it unmeasured. All 18 unchanged,
  which is the control on the stimulus sub-change.

The three that still read `False` are `productive-vocabulary` recordings carrying **1, 1 and 0**
lexical words. Those are readings, not artefacts.

## Reading the flag counts beside this

A correct SPEECH pass is not by itself enough for a file to pass. A `diadochokinesis-buttercup`
recording in the ORCD smoke reported SPEECH `conformance: True` and still came out `flag`, on
`mismatch: routing routed VOICE, it found no subject` — VOICE was routed to a DDK task, correctly
found no sustained phonation, and the graph folded that correct "not my kind" into a file-level
flag. That is a separate defect being measured elsewhere; it is noted here so that a fall in
flagged counts is not read as the whole of this fix's effect, nor its absence as the fix failing.

## What slices 0 and 4 would add

200 more recordings, of which roughly 39 would sit on `FREE_RESPONSE` rows at the 19.3% rate the
600 show. At the observed split they would be expected to add ~22 `False -> True`, ~10
`UNDETERMINED -> True` and ~6 `UNDETERMINED -> UNDETERMINED`. They cannot change the direction of
any group: the three anti-pattern groups are already unanimous on 69, 29 and 18 rows respectively,
and no mechanism in the fix is sample-dependent. Waiting for them tightens the third significant
figure and nothing else.
