# VOICE — what porting the branch decided

[`branch-voice.md`](branch-voice.md) is the branch's design (V1–V8);
[`expected-patterns.md`](expected-patterns.md) § VOICE holds the reference bodies that were
extracted, executed and linted against a synthetic store;
[`branch-foundation.md`](branch-foundation.md) is the shared foundation both sit on. This document
is what porting them to `src/senselab/audio/workflows/triage/nodes/voice.py` decided, in the places
the two designs leave a choice or disagree, plus what did not survive contact with the real objects.

Tests: `src/tests/audio/workflows/triage/nodes/voice_test.py`.

## What the branch now does

Its subject is PREPROCESS's `amplitude` spans qualified by `phonation_tracks` and
`continuity_trace` — the same evidence `voice.sustained` routes on — and it mints its own
`family: "voice"` spans over what qualifies. It waits for no `phonation` span, so the no-span return
that took **6 of 6 VOICE-routed recordings** to `FAIL` is gone, along with the whole
`phonation`-family selector, the period-mark pass and the re-mint.

## D-V1. The old node was replaced outright, and so were its 30 tests

Every capability the retired node had was keyed on the `family == "phonation"` subject: the period
marks, `onset_kind`, `offset_criterion`, `longest_span_criterion`, the `production` tri-state, the
`period_doubling_alias` check, `voice.task_duration_ranges` and the `gate_interval` tri-state. None
of them has an object under propose-only, and `voice_test.py`'s 30 tests tested exactly those. Per
CLAUDE.md § *Pre-alpha: rename and replace outright* the file is replaced rather than extended: 30
tests retired, 72 written. No parallel path and no shim.

The retired `_f0_range`, `_required`, `_rms_track`, `_alias_in_range` and `_task_range` helpers go
with it. `voice.py` no longer imports `senselab.audio.tasks.phonation` at all.

## D-V2. The unreachable whole-file tracks were dropped, not relocated

`voice.py:267` (`hnr_track`) and `:274` (`_rms_track`) sat after the no-span return at `:264`, which
every recording took. Three reasons to drop rather than move them:

1. **Neither design reads them.** The reference `align_voice`/`detect_voice` bodies and V1's own
   qualifier read `phonation_tracks`, `continuity_trace` and the energy envelope — PREPROCESS
   derivatives, already computed. A fresh whole-file Praat harmonicity pass is work no reached body
   consumes.
2. **`hnr_db` is a known-defective array, and reaching it would newly expose the defect.**
   `voice.py:372` wrote Praat's −200 dB undefined-frame markers unmasked — **389 sentinel frames
   measured in a padded 2 s signal** — while `phonation.hnr_floor_interval_db` ships null, so
   nothing masks them downstream. Any mean or percentile over that array is destroyed.
   [`praat-instrument-audit.md`](praat-instrument-audit.md) finding 10;
   [`branch-voice.md`](branch-voice.md) § *What the branch emits* withholds HNR for this reason.
   Relocating the call would move a defect from unreachable to reachable.
3. `voice_tracks.npz` existed only to carry them, and the per-span re-mint it was keyed to is what
   propose-only replaces.

So `voice_tracks.npz` is no longer written. HNR stays **withheld**, which is what the design asks
for; reinstating it is owed the sentinel masking first, not a relocation.

## D-V3. `off_task_extent`'s gap arm was withdrawn; its lexical arm was kept

The two designs disagree here and the disagreement is substantive.

The reference bodies call `off_task(components, store.spans, params.p_gap_off_task_min_s)` at the end
of every VOICE matcher. [`branch-voice.md`](branch-voice.md) § *Deviations* withdraws it explicitly:

> **`off_task_extent` is withdrawn from this branch.** It previously marked *"a region with no
> phonation where the task asked for one"* — which makes the silence around every
> maximum-phonation effort a deviation and needs an undeclared minimum duration to avoid firing
> constantly. **Absence of the target is a measurement**: V2's interruption triple reports it, with
> locations and durations, and reports it better. The one correct instance of `off_task_extent` is
> AIRWAY's, which keys on positively-identified off-task content.

**Chosen: the branch's own design wins, because it gives a reason and the reason is correct.** The
gap-keyed `off_task` helper is not called. Its work is done better by the `interruptions`
measurement, which reports the same absence with locations and a total rather than as a deviation
per gap. VOICE therefore needs no `branch.gap_off_task_min_s`, and its deviation vocabulary is
exactly the three the design tables: `sweep_direction_mismatch`, `truncation`, `repeat_attempt`.

**But the reference's *other* `off_task_extent` was kept, renamed.** Under `forbid_lexical` the
reference emits one per lexical word — which keys on positively-identified lexical content, exactly
the criterion the withdrawal endorses. It is kept and spelled `lexical_content`, so the one name no
longer covers two unrelated findings.

This is the one place the port departs from `expected-patterns.md`. It is recorded here rather than
taken silently.

## D-V4. Only `SUSTAINED` and `GLIDE` are implemented, because only they are reachable

The reference `align_voice` dispatches on four patterns. Two of them — `EFFORT` and `PER_SENTENCE` —
belong solely to the four `VOICE_EXPECTATIONS_PENDING_DECLARATION` rows, which
[`branch-foundation.md`](branch-foundation.md) keeps out of `EXPECTATIONS` because `loudness`,
`loudness-v2`, `cape-v-sentences` and `-v2` are `LEXICAL_SPEECH`. `dispatch` therefore sends all four
to `detect_voice`, and no reachable call can carry those patterns.

`_voice_effort` and `_voice_per_sentence` were **not** ported. Writing them would add:

* two bodies nothing can call, which is the dead code D-V2 removes elsewhere in the same file;
* a dependency on a `store.stimulus_alignment` facade with `structure_spans()`, `substitutions` and
  `omissions_for()` — none of which exists over `ProvStore`;
* for `_voice_per_sentence`, reads of `phonation_tracks.hnr_db`, `cpps_db` and `rms_dbfs`, three
  columns PREPROCESS does not write (they are the design's own "D2 †").

`align_voice` raises `KeyError` on a family with no row — the caller owes `detect_voice` — and
`NotImplementedError` on a row whose pattern no matcher serves, which no row in `VOICE_EXPECTATIONS`
does. A test walks all six rows and asserts every one reaches a matcher, so adding a seventh with a
new pattern fails the suite rather than falling through.

## D-V5. `UNDETERMINED` is what an absent instrument returns

`Done`'s three values had no reachable third case once `PER_SENTENCE` was out: the reference reached
`UNDETERMINED` only from `_voice_per_sentence`'s absent alignment. It is now reached where the
documented meaning actually applies — "when its only instrument is absent". With no
`phonation_tracks`, no boundary can be placed, and:

* `False` would claim the expected pattern was looked for and not found;
* `FAIL` would read, through VERDICT's `_resolved` (`vocabulary.py:220`), as **the kind being
  absent** — i.e. as a finding about the speaker's voice.

Both are wrong for a recording nobody measured. The mode returns `UNDETERMINED` with an `unviable`
finding naming the absence, and the node's outcome is `FLAG`, not `FAIL`. This is the same
discipline [`branch-voice.md`](branch-voice.md) § *A branch `FAIL` is an absence of detected content*
applies to the detector: `FAIL` is reserved for having looked and found no attempt, because a
`FAIL` keyed on voicing concentrates on the most impaired speakers across 62,547 recordings.

## D-V6. The `F0RangeUnavailable` defect was removed, not inherited

The retired node called `derive_f0_range` at entry through `_f0_range` → `_required`, so a frankly
aperiodic recording — the case `F0RangeUnavailable` exists to name — **errored the whole node**
before anything was written. That is the pre-existing defect.

It is removed by construction rather than handled: VOICE reads the F0 contour PREPROCESS already
computed and derives no range of its own, so it makes no such call. `voice.py` does not import
`derive_f0_range`. An aperiodic recording now yields a carrier with zero voiced fraction, which
fails the qualifier and produces `FAIL` with a verdict — "a finding about the voice", which is what
V1 asks for.

Where the absence still lands is PREPROCESS's `phonation_tracks` pass (`preprocess.py:1448`), which
calls `derive_f0_range` and will raise there. That is the right place for it and is not VOICE's to
fix; what VOICE now does is survive the resulting absent measurement (D-V5) instead of adding a
second crash site. Three tests pin it.

## What did not survive contact with the code

**The mode contract carries no `run_dir`, and the derivatives are on disk.** `align_voice` and
`detect_voice` take `(task_family, store, hint, params)` and `(store, params)`; every `read_*` loader
in `branches.py` takes `(store, run_dir)`, and `ProvStore` records no run directory — measurement
`path` attributes are relative. So the modes as specified cannot load a single derivative.

`run_dir` is therefore a **keyword-only parameter** on both, and `voice()` adapts with two local
closures whose parameter names match `AlignMode` and `DetectMode` exactly. The four positional
parameters are unchanged, `branches.py` is untouched, and `dispatch` is called exactly as the
foundation specifies. The name matching is load-bearing: mypy checks Protocol compatibility by
parameter *name*, so closures spelled `inner`/`inner_hint` fail `--strict` against `AlignMode`.

**This is a shared gap, not VOICE's.** AIRWAY needs the energy envelope and DDK needs the envelope
and a spectrogram, so at least three of the four branches meet it. Resolving it properly — a
`run_dir` field on `BranchParams`, or loaders that read an absolute path the store records — is a
foundation change over a file this work was not to touch.

**`phonation_tracks` carries no `path` attribute, so `read_phonation_tracks` returns `None` on every
real store.** Of the four npz derivatives a branch reads, three write
`**path_attributes(...)` — `energy_envelope` (`preprocess.py:1795`), `normalized_envelope`
(`:1889`), `continuity_trace` (`:2741`) — and `phonation_tracks` (`:1490`) does not; its attributes
are `hop_s`, `f0_min_hz`, `f0_max_hz`, `f0_signal`, `formant_signal`. `derivative_arrays` gates on
`measurement.attributes.get("path")` (`branches.py:1489-1491`), so it returns `None`, and
`read_phonation_tracks` with it. The retired node worked around this by hardcoding the location
(`voice.py:255`).

Since `phonation_tracks` is the instrument **both** VOICE modes depend on, using the foundation
loader alone would leave the branch returning `UNDETERMINED` on every recording — the same total
failure in a new spelling. `read_tracks` therefore prefers the loader and falls back to the
conventional sidecar location, gated on the measurement existing so a stray npz with no measurement
beside it is not read as evidence. Two tests pin both carriers.

**The correct fix is one line in PREPROCESS** — `**path_attributes("derivatives/phonation_tracks.npz", run_dir)`
in the `_measurement` call at `preprocess.py:1490` — after which `read_tracks`'s fallback becomes
dead and should be deleted. It is not done here: `preprocess.py` is outside this work's remit and
the change invalidates no cache but does alter the entity, so it wants its own review.

**`max_windowed_spread` does not reject a slow sweep, and must not.** The F0-spread qualifier takes
the worst spread over a `f0_spread_window_s` window precisely so a drift across a long production
does not read as instability — a 100-to-400 Hz glide over sixteen seconds is ~24 semitones total but
~1.5 per second. The first draft of the qualifier test asserted the opposite and failed. Both halves
are now pinned: a fast wobble fails the qualifier, a slow sweep does not.

**`interruptions` counts only *internal* unvoiced runs.** A carrier span is wider than the
production inside it, so the leading and trailing silence would otherwise be reported as
interruptions — on the first draft, the silence before onset *was* the first "interruption". The
walk is bounded by the first and last voiced frame, which is what V2's "internal interruptions"
means. Mutation M11 restores the unbounded walk and one test catches it.

**`_spans_of_family`'s `voice=` parameter is now vacuous for new stores, and still correct for old
ones.** `onset_kind` was written only at the retired `voice.py:341`, and nothing else in `src/` ever
wrote it; `propose_span` does not. So `report.py:715` and `:1154` — the two `voice=True` reads —
return `[]` on any store VOICE writes under propose-only, and `:673`'s `voice=False` becomes a
no-op subset. It is **not dead**, because a store from the completed corpus still carries the
re-minted population and still needs separating. `report.py` is unedited. What the migration owes,
unchanged from [`branch-voice.md`](branch-voice.md) § *Consequence for REPORT*: the VOICE arms need
the span-versus-assertion distinction substituted in, and the substitution must cover the **labels**
at `report.py:714`, `:1134` and `:1173`, which interpolate `onset_kind` into display strings and
would render `.../None` rather than being suppressed.

**`longest_span_s` was kept in the verdict, against the design's request.**
[`branch-voice.md`](branch-voice.md) § *What the branch emits* retires it, and is right that it will
be read as MPT against published norms. But `report.py:104` names it in
`_BRANCH_MEASURES["VOICE"]` and the design itself rules that "the writer's vocabulary may shrink,
the reader's may not", owing the migration what `report.py:104` reads instead. That migration edits
`report.py`, which this work must not. All three keys — `spans_n`, `phonation_s`, `longest_span_s` —
are therefore still written, `longest_span_s` now meaning the longest *proposed* span rather than
the longest voiced run. The qualified duration the design wants read instead is
`phonation_onset_to_offset_s`, written as its own measurement; no scalar named
`maximum_phonation_time` is emitted anywhere, and a test pins that.

**The ruleset-label contest is live code with no object today.** `detect_voice` contests a span
whose `label` is `phonation` that fails the qualifier, as the reference body does. Nothing in
`src/senselab` writes a span `label` yet — the owner's 2026-09-15 decision that a fired rule may
label a span is not implemented, and `evaluate_gate` still drops both the gate value and the span
identity (`routing_analysis/ruleset.py:405-409`, `features.py:1134`). The loop is a no-op until that
lands. It is kept because it is the reference body's own, because a contest is an assertion beside a
span and so touches no propose-only rule, and because its object is a PREPROCESS span rather than
VOICE's own proposal — which is the one form [`branch-voice.md`](branch-voice.md) § *What the branch
emits* permits. Two tests pin both arms against a hand-labelled span.

## Config keys

**None added.** Every operating point the two reached matchers need already exists in the `branch`
section, all null: `production_min_s`, `voiced_strength_min`, `voiced_fraction_min`,
`f0_spread_window_s`, `f0_spread_max_semitones`, `continuity_min`,
`monotone_tolerance_semitones`, `dominant_segment_min_fraction`. `gap_off_task_min_s` is **not**
read, per D-V3.

No number appears in `voice.py`. Against the packaged config the branch raises
`branch.production_min_s has no value …`, naming the key it needed; a test pins that message.
Fitting those eight is what stands between this branch and a measured operating point, and it is
owed listening ([`branch-listening-sample.md`](branch-listening-sample.md)), not a default.

## Still owed, unchanged by this port

V4 voice quality (every Praat scalar is computed on FRCRN-enhanced audio; CPPS, jitter, shimmer and
HNR all withheld), V4a vocal tremor, V6 vocal effort events, V8 vowel identity, and the
`refine` verb — VOICE proposes and contests today but asserts no corrected extent on a PREPROCESS
span, which waits on the family-scoping question
[`branch-conventions.md`](branch-conventions.md) leaves open.

The three stale references [`branch-voice.md`](branch-voice.md) names are now stale in one more way:
the retired-detector string at the old `voice.py:237-238` is gone with the node, so
`config-derivations.md:502` and `taxonomy.md:95` are the only two left naming the superseded
`consensus_taxonomy` rework.
