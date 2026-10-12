# Cross-dataset readings, the replaced-task discard and the whole-file QC columns (r37 plan D, E1, F)

Owner-approved r37 plan v2, sections D, E1 and F (2026-10-12). Measured read-only over the 62,550 r36
stores (manifest `r34/manifest_r34.jsonl`, r36 decision tables and recording vectors), COHORT re-run with
`--readings-out` on CPU (never into a store). Scratch: `r18d_20261010/r37x_crossdataset/`.

## D1. Recording condition per participant and session

COHORT reads three readings per recording: PREPROCESS's whole-file plain RMS (`level.rms_dbfs`), the clip
detector's live `clip` spans (seconds), and the plain energy envelope's 95th- over 10th-percentile level
(SNR; the percentiles of `recording_vectors/evaluation.yaml`). Each reading's `condition` block carries the
recording's values and the medians and clipped share of its session, the participant, and the participant's
other sessions. `cohort_fold.hot_or_clipped(reading, scope)` is the flag: a median RMS at or over
`condition.hot_rms_dbfs_min`, or a share of recordings with clip spans at or over
`condition.clipped_fraction_min`.

The clip spans, not `disruptions_file`'s full-scale runs, are the clipping reading: the runs mark 1% of
recordings and 0 of 004d42e9's 39, the participant the trace names as clipped; the spans mark 15.5% of
recordings and 34 of its 39.

Bounds, Tukey upper fences (Q3 + 1.5 IQR) over r36:

| key | distribution | value |
|---|---|---|
| `hot_rms_dbfs_min` | 1,527 participant median RMS: Q1 −32.96, median −28.36, Q3 −23.51, p99 −15.0 dBFS | −9.343 dBFS (flags no participant alone; 004d42e9 is −12.6) |
| `clipped_fraction_min` | 1,527 participant clipped shares: Q1 0.025, median 0.067, Q3 0.205, p95 0.585 | 0.475 |
| `shift_level_db_min` | 351 sessions' |median level − other sessions'|: Q1 1.97, median 4.97, Q3 8.30 dB | 17.783 dB |
| `shift_snr_db_min` | the same for SNR: Q1 1.98, median 3.85, Q3 7.24 dB | 15.117 dB |

A `session_speaker_consistency` review whose session differs from the other sessions by a shift bound, or
whose hot-or-clipped flag the other sessions do not share, is the annotation `session_condition_shift`, not
`session_speaker_inconsistent`.

## D2. Session non-target voices

`compare_speaker` keeps each embedded pool's vector in memory (never in the reading). Per participant, every
non-task pool that fails the enrollment match is clustered by average linkage at `non_target_voice.cosine_min`
(0.398, the fitted one-speaker cut of `comparison.cosine_min.default`). A pool whose cosine to the mean of the
participant's other matching target pools reaches the same cut is `self_talk` (no longer a non-match: it raises
no `other_speaker_in_session`, the annotation `non_task_self_talk`); otherwise a cluster spanning
`recurring_min` (2) recordings is `recurring_non_target_voice` (the ground stands, with the annotation),
else `single`. Pools under `pooled_min_s` (2 s) are not clustered: 168c5e68's opening voice (1.92 s) stays
below what speaker identification decides.

## D3. Shared, uncorroborated non-matches

`participant_nonmatch` counts the participant's compared recordings and those with a non-match. Where at least
`shared_nonmatch.recordings_min` (3) recordings and `fraction_min` (0.169, the Tukey fence over the 214
participants with any non-match; the fence over all 1,508 is 0) of them do not match, and nothing corroborates
another speaker (reviewer second speaker or other speaker, second-opinion `other_voice` at its confident bound,
the dominant-speaker gate), and the voice is not a recurring cluster, the `other_speaker_in_session` ground is
the annotation `speaker_nonmatch_shared`.

## D4. Review grouped by participant

The review page's index carries `participants`: recordings, sessions, verdicts, issue count and the commonest
reasons and annotations per participant. A list toggle groups the selection by participant, most issues first.

## E1. Discard when talk replaces the task (`discard_replaced_task.py`, off)

The trace's rule (b), as general code over readings the fold holds: non-lexical (airway, voice, loudness) —
talk outside the task at ≥ 0.25 of the free-speech rate (2.03 words/s, the r36 median) over a weak task
(`task_mismatch`, or extent < 0.5 of the recording); lexical — no extent or 0 expected tokens matched, with
the reviewer reading off-task speech or more than one speaker; free response — response < 0.1 of the family
median `response_min_s` (r36 medians in the data file) with off-task speech or instructions by another voice.
Discard ground `task_replaced_by_talk` (reason `task_not_found`). `enabled: false` until the owner has listened
to a sample.

## F. Whole-file QC columns (`triage_table.yaml` v3)

`issues_n`, `task_fraction`, whole-file SNR / floor / SQUIM (recording vectors; SQUIM empty for airway), session
background, speech over residual, clip-span and dropout seconds, lexical and off-task words, ASR agreement /
single hypothesis / degenerate count, masked seconds and fraction inside the extent (empty when withheld or
discarded), COHORT target / non-task / consistency cosines and enrollment source.

## Measured over r36 (read-only)

See `r37x_crossdataset/results.json` and `fcheck.json`; the summary is in the hand-off of 2026-10-12.
