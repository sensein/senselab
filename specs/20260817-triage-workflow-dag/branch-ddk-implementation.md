# DDK, as built

`src/senselab/audio/workflows/triage/nodes/ddk.py`, with
`src/tests/audio/workflows/triage/nodes/ddk_test.py`. The design is
[`branch-ddk.md`](branch-ddk.md) (D1–D6); the ported bodies are
[`expected-patterns.md`](expected-patterns.md) `align_ddk` and `detect_ddk`; the shared foundation
is [`branch-foundation.md`](branch-foundation.md). This document is what porting decided, where the
built module departs from either, and the measurements taken while building it.

`branch-ddk.md`'s "Everything below is **not built** unless it says otherwise" and its
"What exists today" table are superseded for D1, D3, D5, D6 and partly D4 by what is here.

## What is built

| capability | built | instrument |
| --- | --- | --- |
| D1 locate the train and measure its rate | yes | `ddk_carrier` over `amplitude_spans`, rate from the foundation's `train_rate_hz` |
| D2 nucleus rate as a cross-check | **no** — see D-K3 | — |
| D3 inter-onset interval structure | yes | `intervals_of`, `dispersion`, `trend`, `dispersion_by_position` |
| D4 segment inventory over the train | **no** — see D-K4 | — |
| D5 train duration as a fraction of the recording | yes | `train_fraction_of_recording` |
| D6 sequence conformance | yes | `ddk_places` over `spectrogram_wideband`, not the PPG |

## D-K1. The two mode signatures cannot reach a sidecar, so the node binds the derivatives

`align_<branch>(task_family, store, hint, params)` and `detect_<branch>(store, params)` are the
signatures the foundation declares, and `BranchParams` holds only a `TriageConfig`. Every derivative
path in the store is **relative to a run directory** (`common.path_attributes`), `ProvStore` records
no run directory anywhere, and the foundation's own loaders take one
(`read_envelope_track(store, run_dir, name)`). So no mode can load its own instrument. This is a
hole in the declared contract, not in the port.

Three ways out were available. Adding `run_dir` to `BranchParams` edits `branches.py`, which four
branches share and which this work is not to touch. Recovering a run directory from the store means
inventing a second path convention beside the one PREPROCESS writes. What is done instead:
`DdkReads` is a frozen record of the loaded derivatives, `read_ddk` fills it in `ddk()` — which has
the run directory — and two local closures inside `ddk()` bind it to the two modes. The positional
signatures are exactly what the foundation declares, so `dispatch` and the `AlignMode`/`DetectMode`
protocols are satisfied unchanged, and the derivatives arrive as a keyword-only argument that
defaults to an empty `DdkReads()` — which reads as every instrument absent, the safe state.

The three branches landing in parallel have the same problem, so the shared answer is probably a
`run_dir` on `BranchParams`. That is a `branches.py` decision and is recorded here rather than
taken.

## D-K2. `off_task_extent` is not emitted, following the design against the port

`expected-patterns.md`'s `align_ddk` and `_ddk_repeated_word` both end with
`off_task(components, store.spans, params.p_gap_off_task_min_s)`. `branch-ddk.md` withdraws it from
this branch by name, with the mechanism: it makes every breath pause in a DDK train a deviation and
needs an undeclared minimum duration to avoid firing constantly, and *pause structure is a
measurement* — D3's intervals and trend report it, with locations, and report it better.

The design wins. `ddk.py` calls `off_task` nowhere and reads `branch.gap_off_task_min_s` nowhere.
The consequence is one fewer null key DDK can raise on, and pause structure is carried by
`inter_onset_interval_s` and `interval_dispersion` as the design says.

This is the **only** place the built module contradicts the ported bodies. Everything else is either
the port verbatim over the real objects, or additive.

## D-K3. What is additive to the port, and what is still owed

Added, all parameter-free — no new operating point, no new config key:

* **`interval_dispersion`**, with `trend_s_per_step` and, on sequential families, `by_position`.
  D3 carries regularity (D1's peak sharpness having been demoted for confounding brevity with
  irregularity) and the design's verdict shape names `interval_cv` and `interval_trend`; the port
  emits the raw interval list and neither statistic. `dispersion` and `trend` return `None` below
  their own arithmetic domain rather than a number chosen to fill the field. Neither is compared to
  a threshold anywhere: no normative reading of rate or regularity enters the verdict.
* **`unit` on the rate measurement** — `syllables_per_s` on the four alternating patterns,
  `cycles_or_syllables_per_s` on the three sequential ones, beside an `onset_rate_hz` covariate
  which is syllables ÷ duration and therefore unambiguous. D1 requires the unit to be resolved in
  the output and forbids reporting whichever peak is larger as "the rate".
* **`acquisition_covariates`** on every acoustic measurement, per the design's Quality covariates
  section: both D1's modulation spectrum and D3's onsets are envelope measures and so directly
  sensitive to AGC.

Still owed, and deliberately not built:

* **The f/3f harmonic peak structure.** D1's full unit disambiguation rests on `|X1| ≈ |X2| < |X3|`
  under an (A, B, B) pattern, and the tolerance for `≈` is an owed operating point. A body needing
  it would need a new null key and would then raise on every sequential recording, which buys
  nothing over carrying the unit as ambiguous. The ambiguity is therefore *stated* rather than
  resolved.
* **D2, the Praat nucleus rate.** `extract_speech_rate`'s `min_pause = 0.3 s` exceeds an entire DDK
  cycle, so it under-counts the fastest trains and the bias correlates with the quantity being
  measured; its `min_dip` also flips on a mean-HNR test against the voice quality of the recording
  itself. The onset rate above is the unambiguous cross-check D1 wanted from D2, taken from the
  same envelope, so nothing here consumes Praat.
* **D4, the segment inventory.** It reads the PPG, whose usability on rapid nonsense syllables is
  `branch-ddk.md`'s closing Unresolved item, and it adds a model pass for an inventory no verdict
  field reads.

## D-K4. D6 reads the burst spectrum, not the posteriorgram

`ddk_places` takes the burst window after each onset from `spectrogram_wideband` (a 5 ms window at a
5 ms hop, the classical resolution) and ranks `branch.place_centroid_bands_hz` by band power,
reporting `unresolved` unless the leading band beats the next by `branch.place_margin_db`. The PPG
is not the instrument: /p/, /t/ and /k/ differ in burst spectrum inside the 8 kHz ceiling, and a
posteriorgram's acoustic-model prior works against rapid nonsense CV repetition at the cost of a
model pass to recover less.

Two consequences follow from that and are built:

* An absent `spectrogram_wideband` leaves **every** place `unresolved` without reading
  `burst_window_ms`, `place_centroid_bands_hz` or `place_margin_db` at all, emits a
  `syllable_place` measurement naming the missing derivative, and raises **no**
  `syllable_sequence_mismatch`. An unreadable instrument is an absence, never a substitution.
* A departure is typed `syllable_sequence_mismatch` and never `stimulus_mismatch`: the contract
  defines `stimulus_mismatch` as a lexical word differing from the stimulus text, and DDK carries no
  stimulus text and expects no lexical content. Pinned by test.

`sequence_collapse_fraction` carries its `dominant_place` and `support_syllables`, so `/pa-pa-pa/`
reads 1.0 on `labial` with zero realised cycles. That is the finding the sequential families exist
to produce, and it flags rather than fails: a train was found.

## D-K5. The outcome table, and why an absent instrument is not a fail

`DDK: FAIL` means this branch's detector found no train, never that the speaker produced none.

| state | outcome | why |
| --- | --- | --- |
| no train span and the envelope was absent | `FLAG` | `NO_INSTRUMENT` |
| no train span, envelope present | `FAIL` | `NO_TRAIN` |
| a train span and `done is False` | `FLAG` | the expected pattern was not found over a train that was |
| a train span, `done` true or `UNDETERMINED` | `PASS` | |

`TRAIN_ROLES` is `("task_extent",)`. It once read `("task_extent", "repetition")` and admitted a
`lexical_repetition` span, but no writer ever minted either of those two roles: `"repetition"` was
minted nowhere, and the lexical path minted `role="task_extent"` with `lexical_repetition` in
`production`. Both were removed with the lexical path itself — see
[`ddk-syllable-template.md`](ddk-syllable-template.md).

**The vocabulary makes the absent-instrument row uncomfortable and it is left uncomfortable.**
`vocabulary._resolved` maps `FAIL` to `absent` and everything else to `present`, so a `FLAG` for a
missing derivative reads as the subject being present. `FAIL` would read as the subject being
absent, which is the stronger false claim — an absence is never a negative — so `FLAG` is chosen.
`branch-ddk.md`'s Unresolved already records that `Outcome.FAIL`'s wording is the hazard here and
that `Outcome` is a closed vocabulary with readers; nothing in this work changes it.

## D-K6. The NO_NODE flag, verified

`vocabulary.py`'s `decision.will_run and verdict is None` reason — "DDK was asked to run and never
ran" — stops firing for DDK when the node writes a verdict, because the fold joins a branch verdict
by node name **and only when it carries a `kind`** (`by_branch`, `vocabulary.py:335-337`). `ddk()`
writes `kind="ddk"`, which is `BRANCH_FAMILY["DDK"]`. Pinned both ways:
`test_a_ddk_verdict_stops_the_asked_to_run_and_never_ran_flag` and
`test_without_a_verdict_the_flag_still_fires`.

**The flag is not wrong and `vocabulary.py` is not edited.** Two things it now says instead, both
correct and both worth stating:

1. On the **packaged** config every `branch.*` key DDK reads is null, so a recording with an
   amplitude span raises `branch.train_min_s has no value` and a recording with any transcript
   raises `branch.repeat_min_occurrences has no value`. The branch is then `ERRORED`, and the same
   reason fires with `_silence`'s other phrase — "errored without a verdict" rather than "never
   ran". The flag's *id* is unchanged; what changed is which of the three silences it names. This is
   the declared regime (38 nulls, read lazily, a body that needs one fails on the recording it was
   asked about) and not a regression introduced here.
2. With the keys supplied, a DDK-routed recording carrying no train now reads `FAIL`, which
   `_agreement` scores `MISMATCH` against a `routed` route — "mismatch: routing routed DDK, it found
   no subject". That replaced one flag reason with another on the population
   `ddk.lexical_repetition` over-routed, and it was the honest reading: the gate routed connected
   speech and the branch found no train there. With the gate removed, the population that reaches
   this row is the declared one.

`DDK` is still absent from `GRAPH_ORDER`. Nothing breaks on that — `report.py` and `verdict.py` both
sort unknown nodes last, and `run.py:453` composes `nodes` from `(*GRAPH_ORDER, REPORT_NODE,
*BRANCHES)` — but one consequence is live and pre-existing: `run.py:291`'s
`GRAPH_ORDER[PREPROCESS+1 : VERDICT]` slice does not name DDK, so a failed PREPROCESS leaves DDK
with **no** recorded outcome at all rather than `SKIPPED`. Adding `DDK` to `GRAPH_ORDER` is a
`vocabulary.py` change and is recorded rather than taken.

## Measured: `events_in_extent` loses most of a periodic train on an even smoothing width

This is the sharpest thing porting turned up and it is in `branches.py`, which this work does not
touch. It is pinned by `TestTheEventWalkIsSensitiveToTheSmoothingWindowsParity`.

`boxcar` is `np.convolve(x, ones(w)/w, mode="same")`. For an **even** `w` that centres the window
between two samples, so a smooth maximum comes back as two samples of equal value. The strict
maximum test (`smoothed[i] >= smoothed[i-1] and smoothed[i] > smoothed[i+1]`) then selects the
second of the pair, the left walk (`while smoothed[left] < smoothed[index]`) stops immediately on
its equal neighbour, `left_min` is the peak's own value, and the two-sided prominence computes as
**0.0 dB** — below any positive `peak_prominence_db`. The maximum is discarded.

Measured on one 20-syllable, 5 Hz, 4 s train at a 1 kHz envelope, `peak_prominence_db: 6.0`,
`trough_return_db: 3.0`, `event_min_s: 0.02`, varying only `branch.smoothing_window_s`:

| `smoothing_window_s` | width in samples | syllables found of 20 |
| --- | --- | --- |
| 0.001 | 1 | 20 |
| 0.009 | 9 | 20 |
| **0.010** | **10** | **6** |
| 0.011 | 11 | 20 |
| **0.012** | **12** | **1** |
| 0.015 | 15 | 20 |
| 0.021 | 21 | 20 |

Every odd width finds all twenty; both even widths lose most of them. *How many* an even width loses
is decided in the last bit of the cosine — re-associating the fixture's own phase arithmetic
(`t - 1.0 - 0.1` against `t - 1.1`) moves 6 to 0 — so the test asserts twenty against
fewer-than-twenty rather than the exact figures above.

Three things this is not. It is not a fixture artefact: PREPROCESS writes `energy_envelope` at
`resample.target_hz`, so a 0.01 s window is **160 samples at 16 kHz — even**, and the same
0.012 → 192 is even too. It is not the flat-topped-plateau property the foundation already records
(`test_a_flat_topped_plateau_yields_no_event`), which is about a clipped or limiter-flattened event;
this fires on an ordinary sampled sinusoid, which is what a DDK train is. And it is not DDK-specific
— AIRWAY's cough and breath counts come through the same walk.

The consequence for this branch is direct: the syllable count is the measurand, and
the count beside the row's declaration, `syllable_onset_s`, `inter_onset_interval_s`,
`interval_dispersion` and the
`onset_rate_hz` that labels D1's unit all read off it. A halved onset series doubles the intervals
it does find, which **inflates dispersion at high rates — the fastest, healthiest speakers** — the
same direction as the bias `branch-ddk.md` D3 already records for Praat's 0.1 s minimum sounding
interval, and from a different cause.

The repair is a measurement, not a transcription choice: smooth with an odd width, accept a
one-sided trough, or take the maximum over the plateau rather than its last sample. All three change
what counts as an event for AIRWAY as well, so none is taken here.

## The operating points DDK reads

Every one already exists in the `branch` section and every one is null as shipped, so **no config
key was added and `config-derivations.md` needs no new entry**.

`train_min_s`, `modulation_band_hz`, `rate_prominence_min` (D1's carrier and rate);
`smoothing_window_s`, `peak_prominence_db`, `trough_return_db`, `event_min_s` (the onset walk);
`burst_window_ms`, `place_centroid_bands_hz`, `place_margin_db` (D6); `repeat_min_occurrences`
(detect's lexical loop). `p_normalise` is the function, not a key.

Not read: `gap_off_task_min_s` (D-K2), and every key belonging to the other three branches.

Each is read inside the loop or branch that needs it, so a recording with no amplitude span and no
transcript — which is what a store carrying only ADMIT's stream looks like — completes on the
packaged config rather than raising. That is lazy by construction, not by accident, and it is what
lets the runner's own graph test exercise the node.
