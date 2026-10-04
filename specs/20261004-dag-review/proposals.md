# Triage DAG review: what to remove, adjust and add

2026-10-04. Owner's brief: review every part of the DAG, update it, and propose changes against the
B2AI tasks and four goals — (a) quality and task-extent detection, (b) good ASR output for spoken
words, (c) multiple-speaker detection, (d) clear reasoning behind the triage state.

The DAG as it runs is in [`../20260817-triage-workflow-dag/dag.md`](../20260817-triage-workflow-dag/dag.md),
corrected against the code at `89abf051` in the same change as this file. Evidence is the r12 parquet
(`recording_vectors_20261004_r12`, schema 19, policy v7, 62,550 recordings), the r11 evaluation
README, and today's field review (`clef_taskcheck`). Counts below are from r12 unless marked.
Nothing here changes code; every item is a proposal.

Cost key: **refold** = CPU re-fold of the stores (hours); **parquet** = extractor change only;
**GPU** = a model re-run; **model** = a new model to integrate and validate; **labels** = needs
hand-labelled recordings first.

## Top 10

**1. Put the flag grounds in the parquet, as a controlled vocabulary (goal d).** The parquet's
`grounds` column holds only the discard ground (29 non-null of 62,550), so all 10,463 flagged rows
carry no reason. A reader gets `flags_n` and `flag_nodes` and must infer the rest from
`gate_failed_names`, `second_opinion_disagrees` or `person_name_masked_n`. Add a `flag_grounds`
list of ground *keys* (one per row in the dag.md table, e.g. `person_name_review`,
`second_opinion_disagreement`, `gate:instructed_count_min_fraction`, `conformance:SPEECH`) and keep
the free text in the store. At the same time move the quoted words now inside three grounds
(instructions spoken, reviewer second speaker, person-name review) out of `reasons` into the
REVIEW annotation, which restores invariant 6 ("no transcript text reaches a verdict"). Effect:
every flag is countable and explainable from the parquet alone. Cost: code, refold, parquet.

**2. Make acoustic quality part of the graph, as its own axis (goal a).** Noise floor, SNR,
clipping and dropout are judged only by the parquet extractor (`recording_vectors.py`
`QUALITY_CHECKS`) and never reach the triage state; QUALITY checks clip *consistency* only, and its
conformance is `undetermined` on 52,829 recordings. The extractor's rules over-trigger: any clip
span counts as `clipping` (9,896 recordings; `clip_s` p95 0.3 ms, p99 2.1 ms) and any dropout counts
as `dropout` (2,872; `raw_dropout_s` p95 0). Move these into QUALITY as task-extent-scoped
measurements with fitted duration bounds, judged on the stream the release uses (raw issue,
resolved by enhancement or not), and fold them into a third axis `quality = usable / degraded /
unusable` beside `triage` and `release`, rather than into flags. SQUIM stays a reported measure on
speech tasks only. Effect: "is this audio usable" becomes a first-class, explainable column, without
burying it among review flags. Cost: code, refold (the measurements already exist), labels to fit
the bounds.

**3. An acoustically empty recording should discard, not flag (goal d).** All 114 recordings whose
route state is `empty` are flagged, not discarded, because a declared branch forced to run reports
non-conformance first and any flag outranks the empty-discard. That makes
`acoustically_empty` dead code (0 in r12) and sends empty recordings (78 Harvard sentences) to
human review. Fold `empty` before the conformance grounds, keeping those as detail. Cost: code,
refold.

**4. Separate operational grounds from grounds about the participant (goal d).** 651 recordings
are flagged because TAXONOMY had no classifier output (399 of them Harvard sentences), 452 for an
`unexplained` route, plus uncomputed readings and node errors. These say the pipeline is missing a
derivative, not that the participant did something; a reviewer can do nothing with them. Give them
their own state (`triage = rerun`, or an `operational` ground class excluded from the review queue),
and re-run the missing derivatives. Cost: code, refold; GPU for the reruns.

**5. Demote the unfitted sustained-phonation gates until labels fit them (goals a, d).** The flag
gates `voiced_fraction_min` (1,113) and `f0_spread_max_semitones` (1,024; both on only 274) drive
maximum phonation time to 33% flagged and prolonged vowel to 52%. Prolonged vowel also fails
`declared_duration_min_fraction` 601 of 1,603 times (bound 0.5, median reading 0.59, in files of
median 12.1 s whose onset-to-offset median is 6.8 s), which points at the declared duration, not at
the participants. Glides flag 44–47%, driven by `dominant_segment_min_fraction` (736) and
`glide_extent_min_semitones` (547). Keep `production_min_s` and glide direction as grounds; report
the rest without flagging until a listening sample (owner's verified labels plus a stratified 200)
fits them, or until the Clef spectrogram check (item 9) corroborates. Cost: config, refold, labels.

**6. Re-derive the breath and cough count gates (goals a, d).** `instructed_count_min_fraction` is
the largest single failure source (1,517) and contains nearly all `events_min` failures (868 of
920). Its reading has median 1.67 — the event detector usually finds *more* events than asked —
and 10th percentile 0.2, so it errs both ways. Keep `events_min` (no event at all) as the ground,
report the count ratio, and fit a per-family tolerance on labelled recordings before flagging on
it. Cost: config, refold, labels.

**7. Validate a multiple-speaker detector before it decides anything (goal c).** Today:
`dominant_speaker_share_min` fails 628 times; signals agree rarely (separation only 1,379,
diarization and separation 367, reviewer only 142, all three 15). Per the owner, the answer is
better models, not gate tweaks. Build a labelled set (~300 recordings stratified by which signals
fire, across free speech, Harvard sentences with a model speaker, story recall with an examiner),
then evaluate candidates against it: pyannote community-1 overlap output, DiariZen and NeMo
Sortformer (both already have venvs), target-speaker verification against the participant's own
enrolment across sessions (`speech.enrollment_model` is null; ECAPA is already used by
`speaker_vectors.py`), and the transcript signals (reviewer, Clef `other_voice`). Until a detector
is validated, report the share and let only agreeing evidence flag. Cost: labels, model, GPU.

**8. Give ASR the language and the task (goal b).** PREPROCESS passes no language to either ASR
backend (CrisperWhisper 2.0 turbo, Qwen3-ASR); Spanish works only because both auto-detect, and the
1,257 `es-419` recordings flag at 20.7% against 16.6% for English. Pass the sidecar language;
choose per task (verbatim CrisperWhisper for read passages and DDK, where disfluencies are the
signal; Qwen for free speech); prompt with the task lexicon or stimulus for productive vocabulary,
Stroop and fluency lists; keep the forced aligner for word timing. Policy v8 relies on proper-noun
casing, so measure casing agreement too. Measure WER and proper-noun recall on a transcribed subset
before switching. Cost: code, GPU re-run of ASR, labels.

**9. Use the Clef spectrogram check as an independent task-conformance reading (goals a, d).**
The pilot (job 24823992, 121 recordings, 11 families) asks Clef whether the task was performed and
whether the audio is usable, from a log-mel image plus the task's fields. If it agrees with the
gates where they are trusted, use it to adjudicate the noisy VOICE and AIRWAY gates — flag only
where gate and Clef agree, send disagreements to the review queue — instead of fitting every gate
by hand. Cost: GPU (~1 s per recording), labels for calibration.

**10. One human-review queue, built from ground keys, that feeds the labels (goal d).** Review work
is spread over the free-speech page, the parquet viewer and the name-approval config: 1,616
recordings wait on person-name approval (1,275 flagged for nothing else), 724 carry a
second-opinion disagreement (643 of them the false named-diagnosis disagreement v8 removes), 164
reviewer second speakers, 26 spoken instructions. Export one queue keyed by ground (item 1), record
each human decision back into the store as a label, and use those labels to fit the UNFITTED cuts:
the second-opinion 0.8/0.2, items 5–7, and item 2's bounds. Cost: code, then human time.

## Full list

| item | class | evidence (r12) | effect | cost |
| --- | --- | --- | --- | --- |
| ADMIT | KEEP | 29 discards, all `unmeasurable` | — | — |
| PREPROCESS, five streams and derivatives | KEEP | — | — | — |
| PREPROCESS ASR without language | ADJUST | no language argument; `es-419` 20.7% flagged vs 16.6% | item 8 | GPU |
| per-task ASR choice and lexicon prompting | ADD | read and DDK need verbatim; lexicon tasks need vocabulary | item 8 | GPU, labels |
| YAMNet from a TF Hub URI | ADJUST | the only model not pinned by a resolved commit | pin a mirrored artifact by digest | code |
| `diarization.streams: [enhanced]` | KEEP | narrowed 2026-09-20 | — | — |
| TAXONOMY FLAG on no classifier | ADJUST | 651 flags, operational | item 4 | refold |
| routing `unexplained` / `unreadable` flags | ADJUST | 452 + 29 | item 4 | refold |
| `routing.default_branch`, `routing.hint_branch_map` (null) | REMOVE | null and unread in the campaign; the declared family always adds its branch | simpler config | config |
| `voice.hint_tags`, `speech.hint_tags` | REMOVE | "unread as of v2" in their own comments | — | config |
| `speech.second_diarizer`, `target_match_cosine`, `speech_test_*_floor`, `enrollment_model`, `nontarget.*` (null) | ADJUST | null, so the code paths they guard never run | fill from item 7, or remove | config |
| `voice.f0_range_by_population`, `f0_range_ratio_max`, `task_duration_ranges` (null) | ADJUST | null; `declared_duration_min_fraction` suggests a wrong duration source | derive from the registry | config, refold |
| AIRWAY task extent | KEEP | 13,017 align reports carry an extent or a reason (r10 fixes) | — | — |
| AIRWAY `breath_coverage_fraction` | KEEP | measured, read by no gate | report only | — |
| `instructed_count_min_fraction` | ADJUST | 1,517 fails, median reading 1.67 | item 6 | labels, refold |
| `events_min` | KEEP | 920, of which 868 also fail the count gate | the ground for "no event" | — |
| SPEECH DDK decode | KEEP | `repetitions_min` 69 fails | — | — |
| DDK envelope rate beside the decoded rate | REMOVE | two rate estimates, nothing folds them | keep one plus dispersion | parquet |
| `expected_tokens_matched_min` | REMOVE | 1,077 of its 1,119 fails also fail `content_omission_fraction_max` | one ground per read task | config, refold |
| `content_omission_fraction_max` | KEEP | 1,410 fails; `es-419` and `en` pass at the same rate | — | — |
| `response_min_s` | KEEP | 290 fails | — | — |
| `items_min` | KEEP | 5 / 666 | — | — |
| `dominant_speaker_share_min` (flag gate) | ADJUST | 628 fails; signals rarely agree | item 7 | labels, model |
| separation (MossFormer2) as a speaker signal | ADJUST | separation-only 1,379 | report only until validated | refold |
| `production_min_s` | KEEP | 193 fails | — | — |
| `voiced_fraction_min`, `f0_spread_max_semitones` (flag gates) | ADJUST | 1,113 / 1,024, overlap 274 | item 5 | config, labels |
| `continuity_min` | REMOVE | never false (4,947 true, 149 n/a, 16 undetermined) | — | config |
| `declared_duration_min_fraction` | ADJUST | 601 / 1,603 prolonged vowel | item 5 | labels |
| glide gates | ADJUST | glides 44–47% flagged | item 5, item 9 | labels |
| QUALITY clip consistency | KEEP | 286 withdrawn, 6 inconsistent | — | — |
| quality issues in the extractor (`q_raw_issues`) | ADJUST | outside the graph; any clip or dropout counts | item 2 | code, refold |
| SQUIM (`q_plain_squim_*`) | ADJUST | PESQ pinned at its floor (median 1.2); meaningless on non-speech | speech tasks only, never a gate | parquet |
| noise floor beside SNR | REMOVE (one) | floor and SNR correlate (ρ −0.66) | keep SNR | parquet |
| quality axis | ADD | no column says "usable" | item 2 | code |
| REDACT and REVIEW | KEEP | policy v7/v8 under way | — | — |
| person-name review flag | ADJUST | 1,616 recordings, 1,275 flagged for nothing else | exclude from triage flags, keep as a queue (item 10) | refold |
| second-opinion named-diagnosis disagreement | REMOVE | 643 false (v8 in progress) | — | refold |
| second-opinion cuts 0.8/0.2 | ADJUST | UNFITTED | fit from item 10 labels | labels |
| reviewer second speaker, instructions spoken | KEEP | 164 and 26 | as review grounds | — |
| quoted transcript inside `reasons` | REMOVE | violates invariant 6 | item 1 | refold |
| flag-ground keys in the parquet | ADD | 0 of 10,463 flags explained in the parquet | item 1 | code, parquet |
| empty route before conformance | ADJUST | 114 empty recordings flagged, 0 discarded | item 3 | refold |
| operational state | ADD | 651 + 452 + errors are pipeline facts | item 4 | code |
| `deviation_flags`, `undetermined_flags` (off) | KEEP | recorded, not grounds | — | — |
| `hint_mismatch_exempt_families` | KEEP | breath tasks only | — | — |
| `model_speaker_families` | KEEP | Harvard and CAPE-V allow a model speaker | — | — |
| per-family extent rules | ADD | extent is per branch; read tasks have no lead/tail rule; `trimmable` 32,074 is decided in the extractor | per-family extent policy in data/, decided in the graph | code, refold |
| Clef spectrogram task check | ADD | pilot running | item 9 | GPU |
| label-based validation sets | ADD | every cut above is UNFITTED or fitted on few files | items 5–7, 10 | labels |
| unified review queue | ADD | queues spread over three tools | item 10 | code |
| REPORT | KEEP | — | — | — |
| SECOND_OPINION as a driver only | ADJUST | the fold reads it but the runner never runs it | document it as a stage (done in dag.md), or add it to the runner behind config | code |

Counts: REMOVE 8, ADJUST 18, ADD 8 (the rest KEEP).

## What triage state means

The reader's definition is in dag.md § "What the triage state means, for a reader of the
parquet". In short: `discard` = unusable (`grounds` says why); `flag` = a person should look
(`flags_n` and `flag_nodes` say how many grounds and where; the reasons themselves need item 1);
`pass` = no ground applied, which is not a statement about audio quality (item 2) or task
confirmation (`undetermined` conformance never flags). `release` is a separate axis.
