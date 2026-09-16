# AIRWAY in the two modes — what the port decided, and what did not survive

[`branch-airway.md`](branch-airway.md) is the branch's design and A1–A7 its capabilities;
[`expected-patterns.md`](expected-patterns.md) `:2397` and `:2622` are the two entry points as
executable pseudocode over a store facade; [`branch-foundation.md`](branch-foundation.md) is the
shared machinery. This document is what porting AIRWAY onto the real objects decided, in the places
the design leaves a choice to code, plus what the restructuring cost.

The implementation is `src/senselab/audio/workflows/triage/nodes/airway.py`; its tests are
`src/tests/audio/workflows/triage/nodes/airway_test.py`.

## D-A1. The flat-topped plateau: the carrier's own extent, marked as a fallback

`events_in_span` reports nothing on a digitally flat rise —
`branch-foundation.md` § *`events_in_span` reports nothing on a flat-topped plateau*, pinned by
`test_a_flat_topped_plateau_yields_no_event`. Coughs are the loudest thing in this corpus and
clipping is common, so a counted cough family would read **zero events on its loudest recordings**.

The three available answers were smooth first, accept a one-sided trough, or record a limitation.

**Rejected: change the walk.** `events_in_extent` lives in `branches.py`, which four branches share
and which this work is not to touch. Whether the walk should skip a plateau when computing
prominence — which is what `scipy.signal.peak_prominences` does, walking while the sample is `<=`
the peak rather than `<` — is a change to a shared instrument and a measurement in its own right.

**Rejected: smooth harder.** The walk already applies `boxcar` at `branch.smoothing_window_s`, and a
boxcar over a flat region longer than its own width leaves the interior flat. Picking a width that
happens to break the ties would be an unmeasured number chosen to make a branch run. `boxcar` has
since been fixed to force an odd kernel (`d83b8bcb`, landed with DDK), which moves event counts by
parity but not the plateau: `test_a_flat_topped_plateau_yields_no_event` still holds over it.

**Chosen: a carrier that scored the label and inside which the walk resolves no boundary
contributes one event at the carrier's own extent.** This is not new: it is the design's own
`detect_airway` fallback — *"the label is there and the envelope resolves no event boundary inside
it, so the proposal takes the carrier's own extent and says which it is"* — applied to both modes
rather than only to the out-of-family one. It takes no number, and it changes a silent zero into
one event whose boundary provenance is written down.

Two things make the residual limitation legible rather than hidden:

* every proposed event span carries `boundaries`, `"envelope_event"` or `"carrier_span"`, so a
  reader can tell a resolved boundary from a fallback without reconstructing the walk;
* `align` emits `count("events_with_carrier_boundaries", n, None)`, so a run says how often the
  fallback fired and a corpus pass can count it.

**What it does not fix.** A carrier holding three clipped coughs still reads **one** event. The
count is then wrong in the same direction as before and by less, and the span says why. Fixing it
needs the walk changed or a second event source, both measurements.

The same property bounds `truncation` from the other side, which the test states: an
envelope-resolved event **cannot** begin at sample 0, because the walk demands prominence over a
trough on its left and a rise whose whole left flank stays above the trough-return target has none.
So a `task_extent` reaching the recording's edge always arrives through the carrier fallback.

## D-A2. `confirm` and `abstain` are gone; `contest` is replaced, not fitted

A2's corroboration emitted `confirm`, and `abstain` when nothing co-located either way. Neither
survives, and neither is a loss:

* the presence test is `sounds_like`, which reads `raw_scores` over `span_hear` **and**
  `span_yamnet` together against one `branch.score_min`. The two classifiers are one evidence set,
  so there is no second opinion to confirm and nothing to abstain from. A YAMNet score alone is
  enough (`test_a_yamnet_score_alone_is_enough`).
* `confirm` is on neither of the contract's verb lists, and `write_findings` has no kind for it.
  A2's own note said the migration was owed.
* `abstain` recorded `colocated_windows_n` — how many windows looked. That number is now implicit
  in the derivation: an event span names every derivative behind it, and a `span_hear` PREPROCESS
  never wrote is simply absent from it.

A3's declared contest set is **replaced rather than fitted**, as the design directs. `detect_airway`
contests a span whose *decided* label falls in a label set and over which no raw score cleared
`branch.score_min` — reason `no_raw_score_over_p_score_min`. `labels` is written only where a
membership rule exists, which under the packaged config is nowhere for the whole-file windows and
everywhere for the per-span ones, so the firing rate is bounded by that and is still **unmeasured on
the corpus**. That count is owed before the branch is read as a contest rate.

A1's `label` assertion is gone too: the label is now an attribute of the span the branch proposes,
and its derivation names the carrier and the classifier measurements. **This cost REPORT, and the
cost is now paid** — see D-A7.

## D-A7. What REPORT was reading, and what it reads now

The substitution predicted in D-A2 was real and was three reads, not one, and a sweep for it turned
up a fourth defect of the same class that has nothing to do with the two-mode restructure.

`report.py:_airway_labels` walked the assertions indexed against a span id for `verb == "label"`
generated by an AIRWAY activity. Nothing in the tree writes that any more: the only live
`verb == "label"` writer is SPEECH's PII mark over a **word**, so the function returned `[]` on every
recording and its three call sites all rendered `unlabelled` — the `airway` timeline lane, the branch
evidence description for AIRWAY's own spans, and the description for the envelope spans AIRWAY read.
It is replaced by `_airway_span_label`, reading `span.attributes["label"]` off the `family: "airway"`
spans the proposer writes, and the lane now selects those spans by family exactly as the `phonation`,
`speech spans` and `voice` lanes select theirs.

The third call site could not be repaired by substitution, because the thing it described never
carried an airway label at all. Those are PREPROCESS's envelope spans, listed as the upstream
evidence the branch read; the label lives on AIRWAY's own span derived from them. They now carry
their own reading, `peak_over_floor_db`, which is what PREPROCESS actually measured over them.

AIRWAY's assertions were being described by a key they do not carry. `_branch_evidence` admits
assertions for AIRWAY alone and described each by `name` or `family`; `write_findings` writes
`deviation_type` or `claim` beside a `verb`, so every one rendered as the bare word `assertion`. They
now render their verb and their claim.

The fourth, unrelated to the restructure: `_airway_hear_raster` and `_airway_hear_windows` — and so
the `airway_hear_span_windows` JSON key — selected measurements named `hear_span_window`. That name
exists nowhere else in the tree. The per-span HeAR re-evaluation is named `span_hear` and is written
by PREPROCESS, not by AIRWAY, which reads it; `figure.py` already selected it by the correct name.
Both readers were therefore empty on every real run and the raster panel was never drawn. The
docstrings claimed AIRWAY evaluated these windows and now say AIRWAY reads them.

**Every one of these was invisible to the test suite**, because `report_test.py`'s fixture and
`conftest.py`'s `seed_voice_store` both build the pre-restructure shape by hand: an
`AIRWAY/classify` activity, a `verb: "label"` assertion carrying `hear_window_ids` and
`merged_proposals`, and a `family: "airway"` span carrying neither `role` nor `label`. Three tests
passed only against that fixture. The fixture now builds what `propose_span` writes, and the
`hear_span_window` entity the raster test fabricated is now named `span_hear`.

## D-A3. `lexical_contamination` becomes a located deviation, and the FLAG path empties

A4's three owed changes all land, and two of them land structurally:

1. **conditioned on the declaration** — by the mode dispatch, not by a guard. `lexical_intrusions`
   is reached only from `align_airway`, so a Harvard reading with a cough in it can no longer have
   every word between two events read as contamination. `detect_airway` emits none
   (`test_out_of_family_no_word_is_off_task`).
2. **one `off_task_extent` deviation per word**, each with its own extent, in place of one
   file-level `flag` over the hull.
3. **the voicing channel is not built.** It is owed, and `branch-airway.md` A4 already names the
   instrument (`voice.f0_search_range_hz`, frame-wise, not `phonation_tracks`) and the exclusion it
   needs. Nothing here invents it.

With A3's flag gone and this one migrated, **AIRWAY raises no flag at all**: the outcome is `PASS`
or `FAIL`, and `flags` stays in the verdict detail as an empty list because `report.py` reads the
key.

**The deviation carries `word_id` and `agreement`, not `text`.** The design's `lexical_intrusions`
writes `text=w.text`. That puts a participant's transcript into a store assertion, which
`write_verdict`'s own contract forbids for `why` and which the branch's previous flag was written
specifically to avoid — the old test pinned `"Marisol" not in json.dumps(...)`. That pin is kept.
REDACT is the only path that releases text.

## D-A4. Two entry points that need the run directory, and a contract with no carrier for it

The design's bodies read `store.energy_envelope`, `store.hear_scores`, `store.spectrogram_wideband`
— a facade. The real derivatives are npz and json sidecars under the run directory, reached through
`read_envelope_track(store, run_dir, …)`, and **the two entry points' signature has no `run_dir`**.

**Chosen: a keyword-only `run_dir: Path | None = None` on both, bound by the node with
`functools.partial`.** The positional signature stays exactly the foundation's `AlignMode` and
`DetectMode`, so `dispatch` is unchanged and so is `run.py`. Passing None raises, naming what is
needed. Putting it on `BranchParams` would have been cleaner and is a `branches.py` change; if the
other three branches need it too, that is where it belongs.

## D-A5. What each mode reads, and what it deliberately does not

**Not `airway.cough`.** Selected under the scoped reference `declared_cough_vs_breath` (12,741
recordings, J 0.7946 at 50 dB) and reading **J −0.1398** against `declared_airway` over 61,721; on a
13-recording field run it was unavailable on 12 and read 43.10 dB on the one deliberate-cough
recording, firing on neither. It is a routing gate, and neither mode reads it.

**Not `residual.energy_fraction`.** `airway.breath` is `[residual, energy_fraction] >= 0.10`, which
is high when FRCRN removed most of the signal — i.e. *this is not speech*. It routes AIRWAY on 46%
of `maximum-phonation-time` and 44–58% of glides. The coverage pattern reads the raw HeAR grid
instead.

**Not `hear_windows` / `yamnet_windows` / `ast_windows`.** All three are written through
`load_label_membership`, which requires `windows.<classifier>.label_thresholds`; all three are null,
so `_windows` raises and none of the three derivatives exists on any run. The coverage pattern reads
the `hear_scores` sidecar, and `sounds_like` reads the per-span windows, which come through
`optional_label_membership` and therefore do exist.

**The selector is `family is None or family == "airway"`**, as `branch-airway.md` requires, and the
widening is load-bearing in `detect` only: `align` reaches its carriers through `amplitude_spans`,
which a branch-minted span does not pass for want of a `measure` attribute.

## D-A6. Departures from the pseudocode, each deliberate

| the design writes | the code writes | why |
| --- | --- | --- |
| `count("declared_route", declared_route, None)` | `count("declared_route", None, route)` | `count(name, found, declared)`. The route is what the instruction *declared*; what was *found* is nothing, and `measured_route` carries `NOT_SEPARABLE_BY_THIS_DESIGN` beside it. The design's order puts the declaration in the `found` slot |
| the route index comes only from `hints.metadata["task_token"]` | the token first, then the `recording` path's own task id | `task_id_of` keeps the trailing index; only `task_family` strips it. The token is written by nothing in `src/senselab`, so reading only it makes the route permanently unrecoverable when a carrier for it exists today |
| `{"1": "nose", "3": "nose", "2": "mouth", "4": "mouth"}` inline | `airway.route_by_task_index` | a protocol's index assignment is data with a derivation, not a literal. It could not go in the `branch` section: `branches_test` pins `set(config[branch]) == set(PARAM_KEYS)` and `PARAM_KEYS` is in `branches.py` |
| `count("%s_events_in_span", …)` once per carrier span | `count("%s_events", …)` once per label set | `write_findings` folds counts into one `counts` measurement keyed by name, so the per-carrier version silently keeps only the last carrier's number. Pinned by `test_the_branch_writes_no_measurement_name_twice_over_one_extent` |
| role `"airway_event"` for a carrier-boundary proposal, `"<kind>_event"` for a resolved one | `"<kind>_event"` always, with `boundaries` saying which | a reader grouping by role would otherwise split one kind of event across two roles, and the distinction it wants is the boundary provenance, which now has its own attribute |
| a merged breath extent derived from `span_hear` and `span_yamnet` | derived from every carrier span it covers, plus those | under propose-only the derivation is the whole record of where the extent came from, and `merge` coalescing three carriers into one extent is exactly when that matters |
| coverage proposes `label="Breathe"` | `label=<label set name>` | every other proposed span's `label` is the set name; the design's own spelling is inconsistent with its two other patterns |
| `round(nan, 2)` reaches the store | `None` | NaN survives `json.dumps` as a token no other reader parses. `peak_over_floor_db` returns NaN for an extent outside the envelope and `spectral_balance_db` for an unreadable band; both are absences, and None is how every other absence in this branch is written |
| the per-event spectral balance always | only where `spectrogram_wideband` reached the store | `branch.effort_split_hz` is null, and reading it for an instrument that is not there would fail the whole branch for a descriptive covariate |
| three activities (`classify`, `confirm`, `lexical`) | one, whose `step` is the mode | the node's body is `dispatch` then `propose_spans`, `write_findings`, `write_verdict`; there are no longer three steps to attribute |

## What the branch cannot do under the packaged config, and must say so

**Every numeric `branch.*` key is null, so both modes raise on every recording.** `sounds_like`
needs `branch.score_min`, the event walk needs four more, `off_task` needs
`branch.gap_off_task_min_s`, and the coverage pattern needs `branch.breath_coverage_min`. The
failure names the key on the recording it was asked about, which is the foundation's design
(`branch-foundation.md` D-F4), and `run.py`'s `_attempt` records it as an error rather than a
verdict. **AIRWAY is therefore ERROR, not FAIL, in production until its operating points are
fitted** — which is the honest state and is different from the FAIL-on-everything it reported
before. `test_an_unmeasured_operating_point_raises_naming_its_key` pins it.

**YAMNet no longer contributes to breath presence.** `sounds_like` compares literal label strings,
and `branch.label_sets.breath` is `[Breathe]` — a HeAR label. YAMNet's `Breathing`, `Wheeze`,
`Gasp`, `Pant` and `Snort` no longer count, where the retired ontology closure
(`corroboration_sets`, still live for `routing_analysis/labels.py`) did reach them. Widening
`label_sets` is a measurement, per `config-derivations.md` § branch, so it is recorded here rather
than done. `classifier_ontology.py` is unchanged and its other consumers are unaffected.

**A7 is not attempted**, and that is the design's instruction rather than an omission: the
discriminating band is above the 16 kHz working rate's ceiling and the residual tilt is confounded
one-for-one with mouth-to-microphone geometry, which changes with the route by construction. The
`route` measurement is `NOT_SEPARABLE_BY_THIS_DESIGN` with `content_band_hz` carried as its
covariate — None on every run, `band_profile` (D3) not existing — so the negative is attributable.

## Still owed

* **A4's voicing channel**, frame-wise over `voice.f0_search_range_hz`, with the exclusion that an
  extent already carrying an airway label is not off task.
* **The contest's firing rate**, counted on the corpus before it is read as one.
* **A6's cough descriptors** beyond the peak and spectral balance, and the epoch-versus-event
  convention, which changes a cough count two- to threefold and which nothing here declares.
* **The event walk's plateau behaviour**, as a measurement over `branches.py`.
* Whether `run_dir` belongs on `BranchParams`.
