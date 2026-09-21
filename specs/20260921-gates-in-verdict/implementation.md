# Every gate in VERDICT: what moved, and why each landed where it did

Implements [`design.md`](design.md). This file carries the reasoning, the equivalence argument and
the measurements; the code carries neither.

## What moved

`branch:` held 31 keys. 16 of them are gates and are now in `verdict.gates`, keyed by the
`Pattern` each expectation row declares. 15 are instrument settings and stayed. The split is the
design's table, applied unchanged.

| moved to `verdict.gates` | stayed in `branch:` |
| --- | --- |
| `continuity_min`, `coverage_min`, `dominant_segment_min_fraction`, `echo_overlap_max`, `f0_spread_max_semitones`, `gap_off_task_min_s`, `interval_max_s`, `monotone_tolerance_semitones`, `production_min_s`, `rate_prominence_min`, `repeat_overlap_min`, `response_min_s`, `score_min`, `train_min_s`, `verbatim_overlap_max`, `voiced_fraction_min` | `burst_window_ms`, `echo_ngram_n`, `effort_split_hz`, `event_min_s`, `f0_spread_window_s`, `label_sets`, `modulation_band_hz`, `pause_min_s`, `peak_prominence_db`, `phoneme_place_classes`, `phoneme_vowel_classes`, `run_gap_max_s`, `smoothing_window_s`, `trough_return_db`, `voiced_strength_min` |

Five gate names are **new**, and every one of them is a rule that was a code literal before, now
written down at exactly the value the code used:

| new gate | what it replaces | packaged value |
| --- | --- | --- |
| `expected_tokens_matched_min` | `bool(matched)` in `_speech_ordered` | `1` |
| `omissions_max` | `and not omissions` in the same expression | `0` |
| `items_min` | `len(items) > 0` in `_speech_item_list` | `1` |
| `events_min` | `if events: return True` in `_events_reading` | `1` |
| `repetitions_min` | `decode.count >= 1 or done is True` in `_with_decode` | `1` |

**No value was re-derived.** `ordered_match_min: 0.75` and `response_is_a_sentence: true` appear in
the design's illustrative YAML; both are refits, both are out of scope by the design's own last
section, and neither is shipped. What ships is `omissions_max: 0`, which is the all-or-nothing rule
the design criticises — now a config key with a denominator, which is what makes refitting it a
one-line change instead of a code change.

## The three things a gate can be, and where each is applied

VERDICT decides. But a gate whose finding carries an **extent** — a rejected carrier, a located
deviation, a per-event count — cannot be applied in VERDICT, because VERDICT mints nothing and
locates nothing. So the gate's *bound* moved to `verdict.gates` in every case, and where it is
*applied* follows from what its finding is:

| kind | applied in | examples |
| --- | --- | --- |
| decides the task's conformance | VERDICT, against a reading measurement | `production_min_s`, `omissions_max`, `coverage_min` |
| selects the carrier the branch then measures over | the branch, against the same bound | `production_min_s`, `voiced_fraction_min`, `train_min_s`, `score_min` |
| locates a deviation or a count | the branch, against the same bound | `gap_off_task_min_s`, `repeat_overlap_min`, `echo_overlap_max`, `interval_max_s` |

The first two overlap on purpose. `production_min_s` rejected 423 glide and MPT carriers at a
0.12 s median, and the design says that gate should keep working; a branch that stopped rejecting
carriers would mint spans over 0.12 s of nothing and change `findings`, `agreement` and the flag
column. So the branch still rejects, reading the bound from its own task group, and VERDICT
re-applies the same bound to the reading the branch reported. One number, one table, two uses:
selection and decision.

`GATE_SPECS` in `nodes/gates.py` is where a gate's reading and its comparison are declared;
`CONFORMANCE_GATES` is where each group's conformance rule is. A gate with `reading=None` is one of
the located kind and is never applied by VERDICT.

## Task groups, and the out-of-family mode

`Pattern` is the key, as the design directs. Two of the twelve groups — `PER_SENTENCE` and
`EFFORT` — are configured empty: they appear only in `VOICE_EXPECTATIONS_PENDING_DECLARATION`,
which `EXPECTATIONS` does not carry, so no reachable row declares them.

`detect_*` declares no task and therefore no `Pattern`. It still needs the instrument-selecting
gates: `detect_voice` qualifies carriers, `detect_airway` walks events. `DETECT_GROUP` binds each
branch's out-of-family mode to the group whose reading its instrument takes — AIRWAY to
`EVENT_SERIES`, SPEECH to `FREE_RESPONSE`, VOICE to `SUSTAINED`. **No conformance gate is ever
applied to an out-of-family report**: VERDICT gates only the branch that owns the declared family
and reported `in_family`. The alternative, a thirteenth `DETECT` group, would duplicate four
values with nothing to key them to.

## `Result` no longer carries a conformance

`Result` was `(done, components, deviations)`. It is now `(components, deviations)`. The design's
rule 1 — *branch conformance becomes UNDETERMINED in every arm* — is then not a convention a future
edit can break; a branch has no field to write a verdict into. `dispatch`'s check that
`detect_*` returned `UNDETERMINED` went with it: the type says it.

Every branch report is written with `conformance=UNDETERMINED`. VERDICT substitutes the gated
conformance onto the one report that belongs to the declared task, and leaves every other alone.
QUALITY is untouched: its conformance is about `store_assertions`, not about a task, and no gate
reads it.

## Absent reading, absent bound: both UNDETERMINED

`apply_gates` answers `UNDETERMINED` when **any** applied gate could not be answered — a reading
nothing wrote, a reading written null, or a bound nobody has measured — and when the group
configures none of the gates its pattern names. It never answers False on an absence. This is the
design's rule 3, enforced in one place; the nine sites that used to make the call individually are
gone.

The corollary is that a branch **must not write a reading it did not take**. Each reading below is
written exactly where the old code would have reached a True or a False, and nowhere else.

## The readings, and the equivalence argument

One row per conformance gate. "Written when" is the condition under which the branch emits the
measurement; the claim in each row is that the new answer equals the old one for every input.

| group | gate | reading | written when | old expression |
| --- | --- | --- | --- | --- |
| `SUSTAINED` | `production_min_s`, `voiced_fraction_min`, `f0_spread_max_semitones`, `continuity_min` | `carrier_duration_s`, `carrier_voiced_fraction`, `carrier_f0_spread_semitones`, `carrier_continuity` | a carrier qualified | `True` with a carrier, else `UNDETERMINED` |
| `GLIDE` | `production_min_s`, `voiced_fraction_min`, `monotone_tolerance_semitones`, `dominant_segment_min_fraction` | `carrier_duration_s`, `carrier_voiced_fraction`, `sweep_monotone_reversal_semitones`, `sweep_dominant_fraction` | a sweep was selected | `True` with a sweep, else `UNDETERMINED` |
| `ORDERED_TOKENS` | `expected_tokens_matched_min`, `omissions_max` | `expected_tokens_matched`, `expected_tokens_omitted` | the consensus, and the stimulus where the row needs one, were read | `bool(matched) and not omissions` |
| `FREE_RESPONSE` | `response_min_s` | `response_duration_s` | a recognizer's hypothesis reached the consensus | `response is not None and duration(response) >= minimum` |
| `FREE_RESPONSE`, `verbatim_source` | `coverage_min` | `source_content_coverage` | the stimulus was read and `echo_ngram_n` is measured | `covered >= coverage_min`, which **replaced** the response term |
| `ITEM_LIST` | `items_min` | `items_produced` | the repetition rule was resolvable | `len(items) > 0` |
| `EVENT_SERIES`, `EVENT_ALTERNATION` | `events_min` | `airway_events_found` | the instruments reached the store and `score_min` is measured | `True` with an event, else `UNDETERMINED` if `score_min` unmeasured else `False` |
| `SOUND_COVERAGE` | — | — | — | `UNDETERMINED` always |
| `SYLLABLE_TRAIN`, `SYLLABLE_SEQUENCE` | `repetitions_min` | `ddk_repetitions_found` | either instrument could look | `_with_decode` over the envelope carrier and the decode |

Three rows need their argument spelled out.

**`SUSTAINED` and `GLIDE` never answer False, before or after.** The branch selects a carrier with
the group's own gates, so the readings it reports clear those gates by construction and the gate
set answers True. With no qualifying carrier there is no selected carrier, so there are no
readings, so the answer is `UNDETERMINED` — which is what the old code returned on the same input,
for the same reason stated differently. The gates are not decorative: a group that named a gate the
instrument does not take a reading for would now answer `UNDETERMINED` rather than silently pass,
which is the behaviour the design asks for and which the old code could not express.

**`FREE_RESPONSE` with `verbatim_source` is gated on coverage alone, not on coverage AND the
response length.** The old code reassigned `done` — `done = covered >= coverage_min` — discarding
whatever the response term had said. An `AND` would be the more principled rule and would change
two families' answers; re-deriving is out of scope, so `conformance_gate_names` returns
`("coverage_min",)` for that row and the response term is reported without deciding.

**`repetitions_min` folds two instruments into one reading, because the old rule was an `OR`.**
`max(decode.count, 1 if carrier else 0)`, written whenever either instrument could look, reproduces
`_with_decode` on all eight of its input combinations, the four asymmetric ones included: an absent
envelope with a readable decode that completed nothing answered `False`, not `UNDETERMINED`, and
still does.

## The one behaviour change this cannot avoid

`qualifying_phonation` admitted a carrier whose F0 spread the window could not resolve — the old
comment said so: *"an unmeasured bound, or an unmeasurable reading, leaves its gate unapplied"*.
Under rule 3 an unmeasurable reading is `UNDETERMINED`, not a pass. So a `SUSTAINED` recording whose
selected carrier has a non-finite spread moves from `True` to `UNDETERMINED`. This is rule 3 doing
exactly what it was written to do, in the direction nobody expected it to matter. The corpus count
is in [`corpus-replay.md`](corpus-replay.md).

## What a verdict now records

`FileVerdict.gates` carries the node, the group, the group's whole bound table, and one row per
gate applied — name, reading, value, bound, comparison, outcome. A conformance can be read
backwards from the verdict alone, without the store and without a re-run.
