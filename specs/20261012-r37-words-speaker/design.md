# r37 words and speaker (sections B, C and G)

2026-10-12. Owner-approved r37 plan v2 (`~/Downloads/b2ai-reviews/r37_plan.md`, sections B, C, G, H), its
review (`r37_plan_review.md`), the trace of the 24 notes (`trace_notes.md`, scripts in
`/orcd/scratch/bcs/002/satra/r18d_20261010/trace_notes_20261012/`), the earlier 6 notes
(`earlier_notes_r36.md`), and the owner labels in `~/Downloads/b2ai-reviews/owner_labels/`
(`triage_review_be5d2593d524.json`, `triage_review_7423bffbc3c9.json`). Every rule is general; nothing is
fitted to a labelled file. Measured read-only over the r36 stores before any run (below).

## B1. Word timing: text and time are separate choices

The consensus time of a word is the isotonic median of the recognizers' spans, so two recognizers 12 s apart
place it between them, often in silence, and a mask over the word covers the hull of both placements (061cec48:
"release masks out too much"). The text was never the problem: "qwen's raw transcript is more accurate".

- `consensus.arbitrate_time`: a word at least two recognizers located, whose located starts differ by more than
  `data/word_timing.yaml` `arbitration.disagreement_s` (1.0 s, the trace's bound), takes the span of the one
  recognizer whose midpoint lies on sound: inside PREPROCESS's `amplitude` spans (any signal) and, in a voice
  family, outside the held phonation. Two on sound: the fit stands. None on sound: the word is `unlocated`, a
  zero-length extent at its fitted onset; it is not masked by guess and not replaced. Text and readings never
  change.
- The held phonation is a voiced run of the phonation tracks at least `held_voicing_min_s` (1.5 s, VOICE's
  `speech_word_max_s`) long, bridging gaps up to `held_voicing_bridge_s` (0.25 s, VOICE's `break_min_s`), at
  `voiced_strength_min` (0.45). It is the phonation span, not the task extent, so a count-in keeps its times.
- `word_arbitration.arbitrate_words` supersedes each moved word and the transcript (as `rebracket_words` does)
  and writes a `word_arbitration` measurement. PREPROCESS runs it after `spans` and before the stimulus
  alignment; `extend.replay_decisions` runs it before TAXONOMY, so a replay's SPEECH places its findings on the
  arbitrated words. `word_hull` of an arbitrated or unlocated word is its own extent.

## B2. Speaker pools only where there is sound

`cohort_stage.target_spans` and `non_task_spans` merge words across gaps up to 0.5 s, so mis-timed words pooled
silence (017b7678: 37.4 s pooled against 9.6 s of activity). Both pools are now intersected with
`RecordingFacts.activity` (PREPROCESS's amplitude spans; unrestricted only where PREPROCESS recorded no spans
at all; empty where it recorded no contrast). `cohort_stage.yaml` v9.

## B3. Words after the last task event

In an airway or voice family (not loudness), a word starting at or after the end of the task's last event is
speech outside the task whatever it spells: declared task text and misreadings of events are task content only
up to the last event (`task_speech.after_last_event`; the reading records `post_task_ids`).

## C1. The speaker gate on exclusive time

`speaker_share.speakers_within` records, per task extent, the time a speaker other than the largest holds with
the largest silent (`secondary_exclusive_s`, `secondary_exclusive_spans`), and `extent_dominant_exclusive_share`
= dominant / (dominant + exclusive), counting exclusive time only at or over `speech.secondary_exclusive_min_s`
(1.0 s, COHORT's `min_run_s`; the trace found the fire count flat for 0.05-2 s). `dominant_speaker_share_min`
(0.9) reads it; the overlap-inclusive share stays recorded.

A fire nothing else corroborates (reviewer, second opinion not explained by COHORT, a non-matching COHORT pool,
two or more separated speakers: `vocabulary.second_voice_signals`) stays a review ground. Only where
`cohort_fold.hot_or_clipped` reads the participant hot or clipped (the cross-dataset covariate, D1) is it the
annotation `speaker_gate_recording_condition` (SUBJ08).

## C2. The second opinion on the participant's own speech

Where COHORT reads the participant (`cohort_fold.participant_matched`: a pool matched, none did not), the second
opinion's confident `other_voice` is the ground `second_opinion:other_voice_participant` → `off_task_speech`, and
it no longer corroborates another speaker (residual voice, the gate).

## C3. Who read an other_speaker item, and where

`EvidenceItem` carries `source`, `model` and `spans`. The fold fills them for the speaker gate (diarizer, its
exclusive spans), the second opinion (its p against `second_opinion_confident_yes`), the reviewer (its quotes
placed on the words), COHORT (each pool's spans, the non-matching spans) and the residual's other voice.
`triage_decisions` / `triage_evidence` (schema 3) carry them; the review page lists "read by" and draws the
spans as `speaker:<source>` issues.

## G. The earlier notes

- **G1** (07929b7b cinderella). In a narrative family (`task_speech.narrative_family`: the SPEECH families
  whose expectation is `FREE_RESPONSE`) the reviewer's `off_task_speech: some` is the annotation `off_task_aside`
  and its quotes are not masked; `extensive`, and every item-set or ordered family, still flags and masks. The
  second opinion's `policy_identifier_present` is compared with "an identifier mask stands"
  (`RedactionEvidence.pii_masks_final_n`: every final mask but speech outside the task), not "any mask stands".
- **G2** (fac74f45). The second opinion's confident "not free" withholds only when the reviewer proposes more,
  finds residue, or an identifier mask stands; alone it flags (review) and the release stands. Its 0.2 / 0.8
  bounds remain UNFITTED (`config/default.yaml`), to be fitted on the owner's redaction-hold labels.
- **G3** (168c5e68). Lexical speech before a lexical family's task extent is the annotation
  `speech_before_task` with the evidence item `speech_before_task_s` (first word to last word before it).

## Measurement (r36 stores, read-only, commit 1c8821e8)

`measure/condition.py` (each run's level and clipping, then `cohort_stage.condition_summary` per participant),
`measure/fold.py` (per run: the r36 decision; **A** = VERDICT at this commit with the gate fed the r36 share;
**B** = A plus B1, B3, C1 and the participant condition written into the in-memory store, PII findings pointed
at the arbitrated words as a replay would place them), `measure/pools.py` (B2), `measure/analyse.py`. 62,550
runs, 62,521 folded both ways (29 with no consensus transcript skipped in B). Outputs in
`/orcd/scratch/bcs/002/satra/r18d_20261010/r37_words_speaker/`. Hold-out = participant hex digit parity.

| | r36 | A | B |
|---|---|---|---|
| pass / review / discard | 52,087 / 7,597 / 2,837 | 52,762 / 6,922 / 2,837 | 53,304 / 6,380 / 2,837 |
| as_is / redacted / withheld | 55,817 / 3,263 / 604 | 56,729 / 2,537 / 418 | 56,719 / 2,507 / 458 |
| `other_speaker` reason | 1,502 | 1,480 | 902 |

`other_speaker` by group (r36 → B): ordered 622 → 256, narrative 319 → 193, ddk 182 → 80, item set 37 → 32,
airway 232 → 232, voice 110 → 109; by half 589/913 → 261/641.

- **B1**: 9,144 runs (31,087 words) disagree by > 1 s; 7,826 words in 3,919 runs re-timed (CrisperWhisper
  6,662, Qwen 1,164; halves 1,959/1,960); 704 words in 580 runs unlocated. Masks narrowed in 193 runs (median
  2.2 s, 815 s in all), widened in 52. 40 releases move to withheld (an unlocated word under a non-task mask:
  `mask_coverage_failed` 39, `non_task_speech_untimed` 11 with overlap), all already in review.
- **B3**: 2,628 words after the last event in 529 runs; 120 runs gain speech words.
- **C1**: fires 628 → 42 (ordered 13, narrative 16, item set 13); none new; 581 stop (4 of them still flagged by
  another signal); 24 corroborated, 18 uncorroborated kept in review, 5 downgraded on a hot/clipped participant.
  Halves 342/286 → 21/21. The recomputed share matched r36 on 39,456/39,456 readings.
- **C2**: 32 `other_voice` → `off_task_speech`; 21 lose `other_speaker`. In r36, 22 `other_speaker` reasons
  rested on the second opinion alone, 18 of them with COHORT matching the participant.
- **C3**: 1,146 other_speaker items in B: all name a source, 1,118 a model (the residual's 28 have none), 713
  spans (the session-consistency check, 415, is a whole-session reading).
- **G1**: 864 narrative asides (review → pass 675; redacted → as_is 736); identifier disagreements 1,527 → 305.
- **G2**: of 270 r36 `second_opinion_not_free` holds, 84 stay withheld (corroborated), 186 keep their release
  in review (halves 93/93).
- **G3**: 1,420 `speech_before_task` annotations (ordered 1,354, item set 66).
- **Notes**: 26 of 30 meet their expectation (halves 19/21, 7/9). Not met: SUBJ01 and 017b7678 stroop (the
  `other_speaker` is COHORT's target non-match, which B2 and B1 change only on a COHORT re-run), 07929b7b
  pataka (a DDK family has no COHORT pool, so C2 cannot apply; the second opinion alone keeps it in review),
  fac74f45 (review with the release kept, as G2 specifies; the owner's "cleared" also needs the 0.2 bound fit).
