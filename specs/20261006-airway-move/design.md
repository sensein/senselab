# The AIRWAY move: branches measure, VERDICT decides

Owner decision, 2026-10-06 ("yes"): move breath, cough and background-speech measurement out of
VERDICT and into the AIRWAY branch. Recorded in `specs/20260925-review-pass-resume/resume.md`
("Owner decisions not yet in code", items 2–4).

## What moved

Until this change VERDICT's `_task_evidence` called `breath_pattern_of`, `cough_pattern_of` and
`background_speech_of` itself and minted the breath or cough task extent (`_settle_breath_extent`).
That was a measurement taken inside the node that is supposed only to decide.

Now, for every declared airway family listed in `data/airway_event_requirements.yaml` (all eleven
AIRWAY families: eight breath, three cough), the AIRWAY branch in align mode
(`nodes/airway_task.py`, called from `nodes/airway.py:airway`):

1. reads the breathing pattern (breath families) or the cough onsets (cough families) and writes
   one measurement, `airway_breath_reading` or `airway_cough_reading`. It holds either `absent`
   (the derivatives the measure could not read) or the reading: `pattern`, `events_n`,
   `vetoed_by`, `train_breaths`, `review` for breath; `onsets_n`, `review` for cough; and the full
   `reading` record in both;
2. writes the measure's extent as the standing task-extent span (`supersedes` the branch spans),
   retiring any earlier superseding span, including the ones VERDICT minted before this change;
3. reads background speech over that extent with the measure's own events, and writes
   `airway_background_speech`.

VERDICT reads those three measurements and nothing else of the measures. The decision rules in
`vocabulary.fold_file_verdict` are unchanged by the move itself.

The review-band membership (`review`) is computed in AIRWAY, because it is a property of the
reading (strict and lenient re-reads on opposite sides of the instructed count, or the train's
confidence), not a decision; VERDICT turns it into the `*_review_low_confidence` flag.

## What AIRWAY's HeAR-gated detector still does

The event-series, alternation and coverage matchers still run in align mode. They feed the
`events_min`, `instructed_count_min_fraction` and coverage gates, whose readings are carried into
`recording_vectors` as `gate_*` columns, and nothing else: for a measure family the fold already
skipped AIRWAY's conformance and uncomputed-gate grounds (`measure_decides`), overrode AIRWAY's
found state with the measure's, and replaced the owner's absent inputs with the measure's. So the
detector no longer places the extent and no longer decides; it is a reported reading only. Its
hull span stays as the span the measure's extent supersedes.

## Inputs at AIRWAY time

The measures read PREPROCESS's derivatives (narrowband spectrogram, phonation tracks, streams,
YAMNet windows) and the consensus words, all present before AIRWAY runs. One input changed: the
fallback hull (`task_extent_bounds`) used by the breath veto, the breath extent fallback and the
background-speech extent, where the measure placed none. At VERDICT time it was the hull of every
branch's task-extent span; at AIRWAY time it is AIRWAY's own. `breath_pattern_of`'s docstring
already said "AIRWAY's own hull, as they were fitted", so this is the fitted behaviour. A recording
where SPEECH or VOICE also wrote a task-extent span can read differently; the airway replay will
show how many.

## An unmeasured store

A store AIRWAY reported on but whose reading is missing (every store written before this change)
names `AIRWAY:airway_breath_reading` or `AIRWAY:airway_cough_reading` as an owner absent input, so
a re-fold without the replay reruns those recordings rather than deciding them on nothing. Where
AIRWAY did not report at all, nothing of AIRWAY's is absent.

## A discard is not released

Owner, 2026-10-06: "yes, discard should not be released". A triage discard now releases
`withheld` on the ground `discarded` (`vocabulary.DISCARDED`, key `discarded`), whatever the
redaction evidence would have said, so `settle_release` empties the release directory; and
`task_audio.cut_task_audio` cuts no stream of a discarded recording and retires any earlier cut
(`absent: discarded`). In r16, 3,880 discarded recordings carried a release value and 2,111 had
task-audio cuts.

## task_mismatch on a cough task

Too few coughs for the instructed count was still a flag after `8864ed00` made `task_mismatch` an
annotation for breath. It is now an annotation for cough too, as the owner decided for every
airway family.

## discard_contested

Owner, 2026-10-06: "if discard is in verdict, there is nothing downstream to un discard it, so
seems fine" — a discard VERDICT should not make is caught in VERDICT. The rule: a measure's
no-event discard (`no_breath_captured`, `no_cough_captured`) is flagged `discard_contested`
instead where AIRWAY's own HeAR-gated event detector (`airway_events_found`) found at least the
family's instructed count of its events, or one event for a family whose instruction names no
count (`data/discard_contested.yaml`: `instructed_fraction` 1.0, `uncounted_events_min` 1). It is
the only independent reading the owner's listens supported: the 2026-10-05 contrast listens found
AIRWAY's detector the one reading that matched all ten (one event in the one recording holding a
breath, none in the nine silent ones; `data/airway_event_requirements.yaml`), where HeAR Breathe
scores and `[breath]` tokens each passed three silent recordings. An earlier ground (unmeasurable,
too short, acoustically empty) is not contested, and a contested recording does not fall through
to `declared_task_absent`.

Agreement on the owner labels (`~/Downloads/triage_listening_labels_20261006.csv`, airway rows,
joined to the r16 parquet): twelve labelled recordings carry an event-ground discard in r16, all
breath. Nine were heard without breath (8 `no_breath`, 1 not judged); three were heard with breath
too soft or little (9d16c147 very soft, 167ac3f5 little, 0dc15213, which the owner accepted as
"ok if flagged or discarded"). The rule contests none of the twelve: the counted ones read 0
detector events and the sustained families (`respiration-and-cough-breath`, `-v2-breath`) carry no
count. So it agrees with the outcome the owner accepted on 12/12, and the labels do not test it
where it would fire.

Where it would fire: over the r16 parquet, 351 of the 1,641 event-ground discards have
`gate_events_min` at or above the threshold (all on counted breath families: threequickbreaths
159, fivebreaths 97, v2-threebreathsmouth 32, v2-threebreaths 26, v2-threebreathsnose 25,
breath-sounds 12); at 1.5×, 2× and 3× the instructed count, 286, 237 and 49. Those r16 discards
come from the breath measure as it stood at r16, before the vocalised-exhale and edge-phase work,
so the replay's count will differ. 351 is not a narrow band: listen to a sample of the contested
recordings from the replay before the corpus is settled, and raise `instructed_fraction` if the
detector is wrong on them.

## The replay that lands this

Owner, 2026-10-06: "yes, rerun should be rerun". `scripts/select_replay_manifest.py` writes the
manifest for `scripts/extend_replay_decisions.py` from a recording-vectors table: every recording
whose declared family is an AIRWAY family, plus every recording whose verdict is `rerun`, each row
saying why it was selected. On the r16 table that is 13,292 recordings: 12,832 by family, 177 by
family and rerun, 283 reruns of other branches' families.

The replay re-decides from TAXONOMY on, so AIRWAY writes its readings and VERDICT folds them. It
recomputes TAXONOMY's consolidation from the stored classifier scores; it does not run a model. A
rerun whose missing input is a PREPROCESS derivative that was never computed (a classifier's raw
scores, the phonation tracks) is still a rerun after the replay, and needs that derivative first:
`scripts/extend_reprocessed_outputs.py` for the phonation tracks and consolidation, or a fresh run
for a classifier that never scored the recording. r16's 460 reruns carry `route_unexplained` (318),
`owning_branch_input_absent` (140), `uncomputed_reading` (23), `preprocess_errored` (3) and
`critical_absence` (2) among their keys. The airway ones whose absence was AIRWAY's HeAR-gated
detector should resolve in the replay, since the measures read only the narrowband spectrogram and
the phonation tracks; `route_unexplained` is a reading of the routing ruleset and is unchanged by
it.

## Clef stays out of airway decisions

Owner, 2026-10-06: "clef can be left out of airway decisions". A confident second-opinion
disagreement (`second_opinion_disagreement`) is no flag ground where AIRWAY owns the declared
family; the answers are still recorded on the verdict.

## Before and after on the owner labels

Measured locally, in memory, over the 80 owner-labelled airway stores copied from the r9 tree
(`~/Downloads/airway_move_eval_20261006/`: 56 breath from `tmp_bxt/labels.jsonl`, 24 cough from
`tmp_bxt/eval_labelled_cough.jsonl` with the owner labels of
`triage_listening_labels_20261006.csv`). "Before" reads each measure over the store as VERDICT saw
it, every branch's task-extent span live; "after" retires the SPEECH and VOICE task-extent spans
first, as at AIRWAY time in a replay. The decision compared is the measure's: breath present
(`breath_present`) or a cough found (`cough_present`).

| | labelled | agree before | agree after | decisions moved | extents moved |
|---|---|---|---|---|---|
| breath | 56 | 53 | 53 | 0 | 0 |
| cough | 24 | 24 | 24 | 0 | 0 |

The three breath disagreements are the ones the breathing-pattern design already names:
5cc93330 and d5a327c7 (heard as no breath or masked, read as alternating breaths) and 0dc15213
(heard with breath, discarded; the owner accepted either outcome).
