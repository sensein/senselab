# Every gate in VERDICT: what moved, and why each landed where it did

Implements [`design.md`](design.md). This file carries the reasoning, the equivalence argument and
the measurements; the code carries neither.

## What moved

`branch:` held 31 keys. 16 of them are gates and are now in `verdict.gates`. 15 are instrument
settings and stayed. The split is the design's table, applied unchanged.

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

All three uses read the same resolved table, so a `by_family` bound reaches the branch's own
carrier selection as well as VERDICT's decision. One number, one resolution, three uses.

## Three layers: family, then group, then default

A bound resolves most-specific-first, and a family overrides its group **key by key**:
`load_gate_bounds(config, group, family)` walks `LAYERS` in order and the first layer that *names*
a gate supplies it, so a family naming one gate inherits every other gate its group names. The
layer that supplied each bound travels with it, on `GateBounds.layers` and on every `AppliedGate`.

Both non-group layers ship empty, each for its own reason.

- **`by_family: {}`** because every value moved at its current setting and no per-family difference
  has been derived. What the layer buys today is that a difference is *expressible*:
  `maximum-phonation-time` declares `expect_inhale` where its v2 does not, inside one `SUSTAINED`
  group, and saying so is now a config line rather than a code change.
- **`default: {}`** because nothing is universal. `gap_off_task_min_s` reaches six of the twelve
  groups and no other gate reaches more than three, so putting one there would be a default with
  nine exclusions rather than a shared rule. A test pins that no gate reaches every group, so the
  day one does, the default stops being empty on purpose rather than by neglect.

`Pattern` keys the group layer. Two of its twelve groups — `PER_SENTENCE` and `EFFORT` — are
configured empty: they appear only in `VOICE_EXPECTATIONS_PENDING_DECLARATION`, which
`EXPECTATIONS` does not carry, so no reachable row declares them.

`detect_*` declares no task and therefore neither a `Pattern` nor a family. It still needs the
instrument-selecting gates: `detect_voice` qualifies carriers, `detect_airway` walks events.
`DETECT_GROUP` binds each branch's out-of-family mode to the group whose reading its instrument
takes — AIRWAY to `EVENT_SERIES`, SPEECH to `FREE_RESPONSE`, VOICE to `SUSTAINED` — and it reads no
family layer, because it has no family. **No conformance gate is ever applied to an out-of-family
report**: VERDICT gates only the branch that owns the declared family and reported `in_family`. The
alternative, a thirteenth `DETECT` group, would duplicate four values with nothing to key them to.

## No gate reads `expected_event_count`

Owner-directed, and it required no change here: none of the thirteen readings a gate reads is that
field, and a test now says so structurally rather than by inspection.

The field holds two unlike things. `respiration-and-cough-fivebreaths` asks for five breaths and
says so in its own name; a participant who gives four departed from the instruction.
`diadochokinesis-pa` carries a ten nobody ever spoke — the instruction is *repeat as fast as you
can* — and a participant who produces eight did the task. Individuals vary, and rate is the
measurement of interest there.

So the two gates that could plausibly have read it read something else instead, and both at a bound
of **one**, which asks whether the asked-for sound happened at all rather than whether a count was
met:

| gate | reads | not |
| --- | --- | --- |
| `events_min` | `airway_events_found` — how many events of the kind the instruction named the walk found | the declared count |
| `repetitions_min` | `ddk_repetitions_found` — repetitions either instrument read | the ten, or the thirty |

The declared count is still **reported**, in both shapes it already had: as a `count` finding
carrying `found` beside `declared`, and — on the DDK families, the ones where nobody gave the
number — as a `declared_event_count` covariate on the repetition measurement itself.
`decode_evidence`'s docstring already said what that arrangement is for: *"the decoded repetition
count and the declared count are written beside each other and nothing folds them: no conformance
term, no score and no gate reads the pair."* That sentence predates this change and is now true of
the gate table as well. A bound waits for each row to declare which kind of count it carries and
for a tolerance to be derived; neither is this change's work.

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

`FileVerdict.gates` carries the node, the group, the declared family, the whole resolved bound
table with the layer of every entry, and one row per gate applied — name, reading, value, bound,
**the layer that supplied it and the key it was keyed under**, comparison, outcome. A conformance
can be read backwards from the verdict alone, without the store and without a re-run, and a
family-specific bound is distinguishable from an inherited one without opening the config. That
last part matters more than it looks: once `by_family` starts filling up, it is the only way to
audit which recordings were judged by a special case.


## Four keys I would put on the other side of the table

The design's table was followed exactly and nothing was silently reclassified. Four of its
left-hand entries fail the design's **own** test for a gate — *"changing one changes the verdict,
not the reading"* — and are recorded here rather than moved.

**`gap_off_task_min_s` is the clearest.** The design's paragraph justifying the right-hand column
says `event_min_s` stays because it is "what the walk will *report*". `gap_off_task_min_s`'s own
config comment is "shortest gap **reported** as off-task extent": the same sentence. Changing it
changes how many `off_task_extent` deviations exist, and a deviation is never a flag ground
(`verdict.deviation_flags: false`), so it cannot change a verdict at all. It is an instrument
setting by the design's criterion and by its consequences.

**`score_min` is the parallel of `voiced_strength_min`, which the design keeps.**
`voiced_strength_min` says what counts as a voiced frame; `score_min` says what counts as a label
being present in a classifier window. Both define what the instrument *sees*; neither says whether
what was seen is good enough. The gate that decides an airway task is `events_min`, over the count
the walk produced — and that one genuinely is a gate. Moving `score_min` is why the branch has to
read `verdict.gates` in order to detect at all.

**`monotone_tolerance_semitones` and `rate_prominence_min` are search parameters.**
`longest_monotone_run(pitch, tolerance)` uses the first to *find* the run, so changing it changes
the sweep's extent, its direction and its dominant fraction — the readings. `rate_prominence_min`
decides whether a modulation peak is *readable as a rate*; below it there is no rate, not a bad
one. Both are applied where the search happens, and both were given a reading so that VERDICT can
at least record them: `sweep_monotone_reversal_semitones` was added for the first, and the second
has no scalar reading and is marked `reading=None`.

None of this changes what shipped. It is the argument to have if the table is revisited.

## Mutation results

`mutations.py` beside this file rewrites one line of the implementation at a time into a plausible
wrong version of itself and runs the tests that should notice. **17 of 17 caught.** Two of them
were added after a run that they would have passed: M10 survived the first pass, and an earlier M16 turned out to be an
**equivalent mutant** — it removed a redundant `family is None` disjunct from a test that `None`
already fails, so the two expressions agree on every input. The redundancy is gone and M16 now
mutates something that can actually be wrong: whether a family-specific bound records its own
family or its group's name.

| # | mutation | caught by |
| --- | --- | --- |
| M1 | an absent reading reads as `False` rather than as no answer | 5 tests in `gates_test`, `verdict_test` |
| M2 | a gate the group does not name is applied anyway | 2 tests, `GLIDE`'s spread among them |
| M3 | a group that applied no gate reads as a pass | 1 test |
| M4 | `f0_spread_max_semitones` compares `at_least` instead of `at_most` | 6 tests across three files |
| M5 | `GLIDE` is configured with the held vowel's spread after all | 2 tests |
| M6 | VERDICT gates an out-of-family report too | 1 test |
| M7 | VERDICT leaves whatever the branch wrote on the report | 4 tests |
| M8 | the sustained carrier's readings are never written | 4 tests |
| M9 | DDK writes a repetition reading even when neither instrument could look | 2 tests |
| M10 | AIRWAY counts events as a reading even where the label cut is unmeasured | **survived** the first run; now 1 test |
| M11 | a recall is gated on how long it ran rather than on coverage | 2 tests |
| M12 | a branch writes its own conformance onto its report again | 13 tests |
| M13 | a family replaces its group's whole mapping instead of overriding key by key | 1 test |
| M14 | the layers resolve least-specific-first, so a group beats its family | 5 tests |
| M15 | a gate is put on the count nobody gave | 10 tests across two files |
| M16 | a family-specific bound records its group, so it reads as an inherited one | 1 test |
| M17 | every applied gate records the default layer, whichever one supplied it | 4 tests |

M10 is the one worth recording. Nothing asserted that AIRWAY writes **no** `airway_events_found`
reading when `score_min` is unmeasured, so the "could not look" and "looked and found none" cases
were indistinguishable to the suite — exactly the defect this change exists to prevent, surviving
inside the change that prevents it. `airway_test` now pins both halves: no reading with the cut
unmeasured and `UNDETERMINED`, a reading of `0` with the cut measured and `False`.

## Found and not fixed

- **`SOUND_COVERAGE` has no conformance term at all**, before or after: 2,455 `-breath` recordings
  answer `UNDETERMINED` on every run. `breath_coverage_fraction` is reported and no gate reads it,
  because `branch.breath_coverage_min` was deleted rather than defaulted (`config-derivations.md`
  § branch). It is now a one-line config question instead of a code change, which is the whole
  point of the table; fitting it is separate work.
- **`interval_max_s` decides a `count` nothing folds.** `intervals_over_p_interval_max_s` reaches
  the report and no verdict.
- **`FREE_RESPONSE` with `verbatim_source` is gated on coverage alone**, reproducing the old code's
  reassignment. An `AND` with `response_min_s` is the more principled rule and would change
  `story-recall` and `story-recall-v2`; that is a refit and out of scope here.
- **The design's motivating example is true of the configuration and not of the code.**
  `f0_spread_max_semitones` never reached a glide: `_voice_glide` does not call
  `qualifying_phonation`, so the sweep families were never bound by the held vowel's spread. One
  flat value governing two unlike tasks was real; its application to a glide was not. The fix is
  the same either way — the value is now per group and `GLIDE` names none — but the corpus was
  never going to move on it, and it did not.
- **`config-derivations.md` carries a `branch.breath_group_min_gap_s` entry** for a key the
  packaged section does not have. Pre-existing drift, untouched.
- **`PER_SENTENCE` and `EFFORT` are unreachable groups.** They appear only in
  `VOICE_EXPECTATIONS_PENDING_DECLARATION`, which `EXPECTATIONS` does not carry. They ship
  configured empty so that a row declaring one does not raise.
- **The base branch already answers AIRWAY differently from the corpus run** on 2,787 recordings,
  from commits landed between them. [`corpus-replay.md`](corpus-replay.md) breaks that down; it is
  present identically on both sides of the A/B and moves nothing here.
- **`by_family` and `default` both ship empty, so the packaged config exercises one layer of
  three.** The resolution order, the key-by-key override and the layer provenance are covered by
  tests and by four mutations, over overrides rather than over shipped values. The first real
  `by_family` entry is the first time the layering runs on a corpus, and it should be replayed
  when it lands.
