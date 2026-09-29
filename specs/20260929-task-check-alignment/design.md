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
