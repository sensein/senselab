# Task checks aligned with what each task asks

Owner, 2026-09-29, after the audit in `/orcd/scratch/bcs/002/satra/tmp_task_audit/`
(`family_gate_audit.csv`, `undetermined_paths.csv`, `harvard_sample.jsonl`). Four fix groups were
approved: what an undecided gate means (A), the read-aloud alignment (B), the voice tasks (C), and
items, counts and speakers (D). The reviewer's result cache (E) and the viewer's theme and keyboard
navigation (F, G) and the reviewer's speaker judgment (H) ride in the same batch. Measurements are
from the r8 corpus (`triage_r8_20260929`, 62,550 recordings) unless a section says otherwise.

## A. What an undecided gate means

### What the audit found

`apply_gates` answered `UNDETERMINED` whenever a reading was absent, whatever the absence was, and
any undetermined gate made the whole conformance undetermined. Neither undetermined conformance nor
an undetermined flag gate flags (`verdict.undetermined_flags: false`; a flag gate flags only on
`passed is False`), so every absence was silent. On r8:

| gate | undetermined | cause |
|---|---|---|
| `dominant_speaker_share_min` | 2,217 | 1,423 nothing spoken (no lexical word, no diarization); 646 words present and no share written; 125 share written null (no diarized segment meets the task extent) |
| `production_min_s` / `voiced_fraction_min` / `continuity_min` / `f0_spread_max_semitones` | 666 / 666 / 617 / 622 | no qualifying carrier: rejected `lexical_separator` (467), `production_min_s` (597), `no_voicing` (183) |
| `events_min` | 144 | AIRWAY's classifier windows never reached the store (`event_instrument` written, no count) |
| `items_min` | 472 | random-item-generation: the category was never read, so no item reading was written |

And one undetermined gate hid a failed sibling in 5 recordings (`gates.py`, the any-undetermined rule
was checked before any-false).

### The rule

The absence is explained by the fold (`verdict.reading_absences`), which alone can see the store, and
the gate answers by the explanation:

| reason | when | answer |
|---|---|---|
| `no_carrier` | a VOICE carrier reading is absent, no carrier reading at all was written, and the tracks arrived | **False** for a conformance gate: the branch looked for a production and found none, so nothing was produced. **Not applicable** for a flag gate: a production that does not exist has no quality to ask about |
| `instrument_absent` | VOICE wrote its tracks-absent sentinel, or AIRWAY its `event_instrument` measurement | undetermined |
| `no_speech` | the speaker share is absent and the store carries no lexical word | **not applicable** |
| `null_no_overlap` | the speaker share was written null | undetermined |
| `null_value` | another reading was written null | undetermined |
| `absent_not_computed` | anything else: the branch had its inputs and wrote no reading | undetermined |
| `bound_unmeasured` | the reading is present and the winning layer's bound is null | undetermined |

A carrier reading absent while another carrier reading was written (`carrier_duration_s` present) is
`absent_not_computed`, not `no_carrier`: a carrier did qualify.

Conformance is then: not-applicable gates take no part; **any False decides False** (the fix to the
hidden-failure case); otherwise any undetermined leaves it undetermined, as does a group with no
applicable gate. The gate record carries `reason`; the gate outcome record carries `reason:
no_owner_report` where the owning branch left no in-family report.

**An undecided conformance gate whose reason is `absent_not_computed` or `instrument_absent` is a flag
ground of its own** (`verdict.uncomputed_reading_flags: true`, "a reading this task is judged on was
not computed", with each gate and reason appended). A reading the task is judged on that should exist
and does not is a defect in the pipeline or its inputs; leaving it silent is how 472
random-item-generation recordings sat unjudged. It is not raised for flag gates: a speaker share that
was not computed is recorded (`gate_dominant_speaker_share_min_reason`) but is not itself a statement
about the recording.

### Where it is visible

Parquet schema 12: `gate_<name>_reason` for each of the 13 applied gates, `gate_reason`,
`gate_exempt`, `gate_not_applicable_n`, and `not_applicable` as a `gate_<name>_passed` value. The
data dictionary describes each; the viewer offers the reasons as categorical axes; the review page
shows "not applicable" and the reason beside each gate.

## B. The read-aloud alignment (ORDERED_TOKENS)

### What the audit found

1,800 of 13,700 Harvard recordings flagged; Caterpillar 49%, Stroop 31%, Rainbow 23%. The bound was
`omissions_max: 0`, whose only derivation was "the value the code used". A 40-recording stratified
sample of flagged Harvard (`harvard_sample.jsonl`) split into: nothing read at all (1,012 corpus-wide:
empty, `[breath]`, one or two words); exactly one omission (540), of which about 230 were alignment
defects (compounds `desk top`/`desktop`, `half way`, `thumb tacks`, `pure bred`; a swap `not to
spend`/`to not spend`), about 230 a single dropped function word, about 80 real misreadings; several
omissions (62), mostly real.

### What changed

1. **Compounds and swaps** (`stimulus.align_stimulus`). An adjacent pair on either side whose
   concatenation is one token of the other side, and which are not both tokens of it on their own, is
   aligned as that one token; an expected pair so joined is realised by the single word. After the
   alignment, an absent expected token is paired with an unpaired word of its own key at most one
   position away (`TRANSPOSITION_WINDOW = 1`): a swap of neighbours. One position, not two, because
   Stroop's list is ordered and a rotation of three answers is a real departure. A plural is not a
   compound (`canoes` stays a substitution of `canoe`). PREPROCESS's stored alignment is not rebuilt
   by a replay, so `stimulus_alignment_rebuild_agrees` reads False wherever the new pairing moved a
   count; it is a record, not a ground.
2. **Nothing read** is its own conformance ground ("read none of the words the stimulus asked for")
   where `expected_tokens_matched` is 0, instead of the generic non-conformance.
3. **The omission gate reads content, as a fraction.** `omissions_max` is replaced outright by
   `content_omission_fraction_max`, reading `expected_content_omitted_fraction`: of the stimulus's
   content tokens (`residue.is_content_word`), the fraction nothing realised; over every token where
   the stimulus has no content token (`hey hey hey`). Function-word drops are left out of both counts:
   the ~230 single dropped "the"/"a"/"to" were the sentence read, not unread. The count
   `expected_tokens_omitted` is still written.
   - Sentences (group `ORDERED_TOKENS`): **0.0** — any content word unread fails. This is not a fit;
     it is the instruction ("read the following sentences"), now applied to the words that carry the
     sentence.
   - **UNFITTED**: `rainbow-passage` 0.1, `caterpillar-passage` 0.1, `word-color-stroop` 0.2. No
     labelled verdicts exist to fit against; the values are proposals from the r8 distribution below,
     for the owner to approve.
4. **Stroop's stimulus is the answers.** `stimulus_text` lists the displayed ink colours, which is
   what the instruction asks to be said: over 400 r8 Stroop recordings the realised fraction has
   median 0.73 and 49% at or above 0.8, which could not happen if the text were the printed words
   (about half of which differ from the ink). The expectation already matched; only the bound changed.
