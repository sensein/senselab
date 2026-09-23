# VOICE reports the production it found

Three defects in `nodes/voice.py`, found by the owner reviewing real recordings. All three are the
same shape: the branch made a decision and reported an absence, where the contract is that the
branch reports a reading and VERDICT decides.

## The contract being violated

`branches.Result` carries no conformance. The gate bounds live in `verdict.gates.<layer>.<GROUP>`
and are applied by VERDICT to the readings a branch wrote. `carrier_readings()` says so in as many
words: it emits "one measurement per reading VERDICT's `SUSTAINED` and `GLIDE` gates read".

The branch was *also* reading those same bounds, through `BranchParams.gate(...)`, and using them to
discard carriers before any reading was written. The same bound was therefore applied twice, in two
places, and the first application silently destroyed the input to the second: when the branch
discarded every carrier, VERDICT received no reading, every gate came back `passed: UNDETERMINED,
value: null`, and the branch report read `deviations: []`. A reader could not tell "we found a
production and refused it" from "there was nothing here".

## D-1 — a repeated glide was reported by nothing

`sub-b3bfdba4-…_task-glides-low-to-high`: the task was performed twice, both attempts somewhat
inaccurate; the branch report carried `deviations: []`.

`repeat_attempt` was emitted from exactly one place, inside `_voice_sustained`, which sorts the
qualifying carriers by duration, proposes the longest and emits one `repeat_attempt` per remaining
carrier. `_voice_glide` had no equivalent.

The brief for this work described the losing carriers as being "appended to `discarded` as a gate
`Rejection`". That is not what the code did, and the truth is worse. `discarded` received only spans
that *failed* a gate. A span that passed every gate and yielded a monotone run, and then merely lost
the `duration(sweep) > duration(best[4])` comparison, was written nowhere at all: not a span, not a
deviation, not a rejection measure. It left no trace in the store.

**Fixed** by giving GLIDE the same rule SUSTAINED uses. The rule is identical in kind — every
carrier a production exists inside is an attempt; the winner is proposed; each further one is a
`repeat_attempt` — and differs only in which criteria establish existence, which is inherent to the
pattern (a glide's carrier must additionally hold a monotone run for there to be a sweep). GLIDE now
also emits `attempt_count`, as SUSTAINED already did.

### The declared-direction check was winner-only

`expectation.declared_direction` was read against `direction`, a local computed from `best`. It ran
exactly once, on the proposed sweep. A second attempt running the wrong way produced no
`sweep_direction_mismatch`. Since the owner described both attempts as inaccurate, this is the same
blind spot and not a separate one.

**Fixed**: `_direction_mismatch` is applied to every sweep found, not only the proposed one.

### A flat contour has no direction

Making the check run on every carrier made a latent case reachable. `longest_monotone_run` tries
`sign = 1` first and keeps a later candidate only on a strict `>`, so a perfectly flat contour is
returned as a rising run. Read against `glides-high-to-low` that would have emitted a
`sweep_direction_mismatch` whose `measured: "up"` is the sign of a zero difference — a reading that
was not taken. The check is therefore skipped when `extent_semitones == 0.0`. This is an equality,
not a threshold: `glide_extent_semitones` is still written, and VERDICT gates it.

## D-2 — a quality gate decided that the production never happened

`sub-7d51b647…_task-glides-low-to-high`: six carriers considered, all six rejected. Four by
`production_min_s` (0.051 s, 0.349 s, 0.115 s, 0.169 s), one by `voiced_fraction_min` at 0.0, and
the glide itself — a 5.447 s carrier — by `dominant_segment_min_fraction`, whose longest monotone
sweep covered 44.24% of it against a 50% bound. Short by 5.8 points, and the recording reports no
glide.

`sub-652def69…_task-maximum-phonation-time-2`: five carriers, all rejected. Three by
`production_min_s`, one by `voiced_fraction_min` at 0.0, and an 8.152 s carrier by
`f0_spread_max_semitones` reading 11.856 against 2.0. The task is maximum phonation time, where
duration *is* the measurement. The owner has listened to the recording: the phonation genuinely runs
about 8 s, and the amplitude span and YAMNet agree. An 8.152 s production was found, and the
recording reports no duration.

These are one defect in two patterns, not two defects. A carrier failing a quality gate was
`continue`d, so it could never become the chosen production.

### Which criteria are which

A criterion is an **existence** criterion if failing it means there is nothing to measure. It is a
**quality** criterion if failing it means the thing was measured and was poor. Only the first may
discard.

| criterion | pattern | kind | why |
|---|---|---|---|
| `production_min_s` | both | existence | Below it there is no production; and it is the guard that keeps degenerate windows out of every downstream statistic. |
| `lexical_separator` | SUSTAINED | scoping | Not a quality judgement: it says this carrier is the count-in, not the vowel. Still discards. |
| `no_track_over_carrier` | both | existence | The instrument returned no frames over the span. No reading exists. |
| `no_voicing` (new) | both | existence | The tracker voiced not one frame. Nothing was phonated, so nothing can be measured. |
| `no_monotone_run` | GLIDE | existence | With no run there is no sweep: no direction, no extent, no dominant fraction. There is no reading to hand VERDICT. |
| `voiced_fraction_min` | both | quality | How much of the carrier was voiced. Written as `carrier_voiced_fraction`. |
| `f0_spread_max_semitones` | SUSTAINED | quality | How steady the pitch was. Written as `carrier_f0_spread_semitones`. |
| `continuity_min` | SUSTAINED | quality | How stationary the spectrum was. Written as `carrier_continuity`. |
| `dominant_segment_min_fraction` | GLIDE | quality | How much of the carrier the sweep covered. Written as `sweep_dominant_fraction`. |
| `monotone_tolerance_semitones` | GLIDE | parameter | Not a bound on a reading; it shapes `longest_monotone_run`. |

`voiced_fraction_min` was the one genuine ambiguity. Both owner cases contain a carrier failing it
at exactly 0.0, which is an absence of phonation rather than a poor one. The split taken is that
*any* voicing establishes existence, so 0.0 discards under the new `no_voicing` criterion, while the
configured 0.5 bound becomes a reading. No configured value changed.

### Why the align arm and the detect arm differ

`qualifying_phonation`'s quality gates exist for a measured reason: without them the detect arm
"would propose attempts over runs of connected speech on 14,332 recordings". That reason holds only
where nothing declared what the recording contains.

- **`detect_voice`** runs against a neutral expectation. Nothing says a held vowel was asked for, so
  steadiness is the only thing separating one from connected speech. It keeps applying the quality
  gates, and its behaviour is unchanged.
- **`align_voice`** runs against a declared task family. The instruction already says a held vowel
  or a glide was asked for. The question is not "is there a held vowel somewhere" but "how did this
  speaker do what they were asked to do", and a quality gate must not answer the first by refusing
  to answer the second.

`Qualification` therefore distinguishes the two: `carriers` holds every carrier a production exists
inside, each carrying in `failed` the quality gates it missed, evaluated but not applied; `steady`
is the subset that missed none. The detect arm proposes over `steady` and writes a `carrier_rejected`
measure for the rest. The align arm proposes over `carriers` and writes the qualities as readings.

### Consequences to expect

- **The chosen carrier can change.** `_voice_sustained` picks the longest carrier. With unsteady
  carriers no longer removed, a long unsteady carrier now outranks a short steady one where both
  exist. For MPT, where duration is the measurement, the longest production is the right one and its
  steadiness is a separate reading, so this is the intended direction.
- **`repeat_attempt` will fire more often**, since more carriers survive to be counted.
  `verdict.deviation_flags` is `false` corpus-wide, so this flags no files.
- **`carriers_rejected` in the branch report shrinks for the align arm**, because a quality failure
  is no longer a rejection there. What replaces it is better: a span, with the reading on it.
- **Corpus-wide effect not measured here.** See the census below.

## D-3 — the gate silences recordings in proportion to how disordered the voice is

This is the reason D-2 matters, and it is not a claim about any one instrument being wrong.

`f0_spread_max_semitones <= 2.0` encodes a steady, healthy, sustained vowel. A dysphonic voice
cannot satisfy it. Under the old code the carrier was therefore discarded, the branch reported no
extent and `deviations: []`, and the file passed triage clean. **The graph was dropping recordings in
proportion to how disordered the voice is** — which inverts the purpose of the corpus. The
population the corpus exists to characterise is exactly the population the gate removed.

`maximum-phonation-time` mints no task extent on 68.9% of its recordings, the worst coverage of any
family (`specs/20260922-speaker-vectors/coverage.md`). How much of that is found-and-refused rather
than never-found is the size of this blind spot, and is being counted (below).

### Why the 11.856 reading cannot settle it, and does not need to

Two readings of that number were put forward and neither is established:

- **Octave-halving by the tracker.** A harmonic product spectrum over the loudest 1 s window of the
  `plain` stream puts f0 at 165.0 Hz, and the harmonic amplitudes favour it: against f0 = 165 Hz they
  decay monotonically (0.763, 0.404, 0.122, 0.033, 0.019), while against f0 = 82.5 Hz the supposed
  fundamental is absent and H2 carries everything (0.038, 0.763, 0.368, 0.404, 0.040). On that
  reading Praat's ~87 Hz median is halving, and the spread is the tracker's failure.
- **Genuine subharmonics.** In a dysphonic voice, period doubling and diplophonia are real, and a
  track alternating between ~82 and ~165 Hz may be reporting them correctly. The repo already has
  the concept: `period_doubling_factor: 2.0`, described in config as "the definition of period
  doubling; an identity, not a threshold". The f0 gate does not consult it.

Both may be true at different moments of the same recording, and a single-window HPS cannot
distinguish them. **The defect does not depend on which.** Either way an 8.152 s production existed,
was measured, and was reported as nothing. After this change it is proposed, the 11.856 travels as
`carrier_f0_spread_semitones`, and VERDICT decides against the owner's bound — which is where that
decision belongs.

### Instrument evidence, recorded but not acted on

Pitch strength does not separate the suspect frames — core-band median 0.722 against non-core 0.666
— so `voiced_strength_min = 0.45` passes nearly all of them.

Per-stream medians on that recording (Praat cc, hop 0.01, same search range):

| stream | voiced | median Hz | core % | > 1.25x | spread (all) |
|---|---:|---:|---:|---:|---:|
| plain | 1033 | 86.6 | 29.2 | 33.2 | 30.24 |
| preemphasised (what F0 uses) | 778 | 87.8 | 25.1 | 40.4 | 40.89 |
| enhanced | 557 | 222.7 | 44.7 | 20.5 | 38.11 |
| normalized | 747 | 88.5 | 25.6 | 41.8 | 40.92 |
| residual | 680 | 68.8 | 50.9 | 37.4 | 25.89 |

These figures are the coordinator's, on one recording, reproduced rather than independently
re-measured. n = 1 settles nothing.

**F0 must not move to the `enhanced` stream.** The owner's judgment: FRCRN is trained on normal
speech and regularises the irregularity that *is* the clinical signal in a dysphonic voice. The table
is consistent with it — `enhanced` reports 222.7 Hz where every other stream reports ~87, which reads
as a reconstructed fundamental rather than a clarified one, on 46% fewer voiced frames. The question
is closed; the row stays as evidence.

Still open, and belonging to PREPROCESS and the owner rather than to this change:

1. Did `derive_f0_range` narrow at all here? It appears to have returned the full 50-600 Hz. A 50 Hz
   floor is what makes halving a 165 Hz voice representable, and narrowing is the mechanism already
   meant to prevent it. How often it returns the unnarrowed range corpus-wide is not known.
2. F0 runs on the pre-emphasised stream and formants on `plain`, which is the reverse of the usual
   pairing. A convention question, separate from the above.
3. A **pitch-track validity reading** — core-band fraction, or octave-jump rate — would let the branch
   report "the pitch instrument did not work here" instead of an absence or a bare spread. That is a
   reading, not a judgement, and it fits the contract. Deliberately not added: the statistic needs a
   band parameter, which belongs in `branch.*` with a derivation, and the derivation needs the
   campaign above. A validity reading and a subharmonic reading are not the same thing and should not
   be conflated by one statistic.

### The acceptance criterion, and why `undetermined_flags` is not touched

The owner's criterion on `sub-652def69…`: "if this file got flagged it would be ok." Not that the
branch must mint an extent — that a recording where every carrier was found and refused must not
pass triage silently.

It is met, by the chain the contract already provides rather than by a new mechanism:

1. The branch proposes the 8.152 s carrier and writes `carrier_f0_spread_semitones = 11.856`.
2. `verdict.gate_conformance` (`nodes/verdict.py:506-537`) applies the SUSTAINED conformance gates to
   that reading and replaces the branch report's conformance (`verdict.py:597-604`). The gate fails,
   so conformance is `False`, not `UNDETERMINED`.
3. `conformance_flags: true` makes a reported non-conformance a flag ground
   (`vocabulary.py:741-744`), so the file is flagged.

Before the change, step 1 wrote nothing, so step 2 had `value: null` and answered `UNDETERMINED`,
and step 3 did not fire. **That null was the whole defect.** The proposal to flag on UNDETERMINED was
a way of reaching the file without fixing the null; fixing the null reaches it with a reason
attached, which is strictly better — the flag names the gate, the reading and the bound.

`undetermined_flags` therefore stays `false`, and the "unasked versus unanswerable" split is not
built. It would have been the right shape had the null remained. What it would still cover after
this change is a genuinely smaller and different population: recordings where **every** carrier
failed an *existence* criterion — all below `production_min_s`, or no voicing, or no monotone run.
That is "the branch looked and found nothing measurable", and whether it should flag is the owner's
call, not one this change should pre-empt. The evidence for it is already in the store as
`carrier_rejected` measures with reading, value and bound, and the branch report's
`carriers_rejected_n` counts them; what is missing is only VERDICT keying on it.

**Blast radius.** The number of files that gain a flag under this change is exactly the
found-and-refused population the census below counts: a recording with no extent that held a
rejected carrier of usable length now gets an extent, a failing reading and a flag. The owner has
accepted flagging this one file and has not accepted flagging fifteen thousand. That number is not
yet known, and is the thing to read before this lands anywhere but a branch.

### The census

Job `23526686`, `mit_preemptable`, over the 62,548 replayed runs at
`/orcd/scratch/bcs/002/satra/triage_replay_20260922/out/` (3,510 MPT, 1,596 glides-low-to-high,
1,554 glides-high-to-low). It counts, per family: recordings with no voice `task_extent`; of those,
how many hold a `carrier_rejected` of at least 0.5 s, broken down by gate; and the distribution of
`carrier_s` and `value_read` for `f0_spread_max_semitones` and `dominant_segment_min_fraction`. The
population above the 2.0 bound is the candidate instability/diplophonia cohort and its size is the
size of the blind spot.

Results land at `/orcd/scratch/bcs/002/satra/voice_gate_census_20260923/results/census.json` and
`census.txt`. **Unread at the time of writing.** No figure in this document is taken from it.

## The same shape elsewhere, not fixed here

`VOICE_EXPECTATIONS` serves exactly two patterns, SUSTAINED and GLIDE, and
`test_every_voice_row_is_served_by_a_reachable_matcher` enforces that. So there is no third
`_voice_*` body to check: the asymmetry ran in both directions between these two, and both are
fixed.

**DDK has the same defect, in a worse form.** `nodes/ddk.py:185-211`, `ddk_carrier`, walks the
amplitude spans, keeps the longest one that clears `train_min_s` and yields a readable repetition
rate, and returns only that. Every other candidate — including one that cleared both and merely lost
the `duration(span.extent) > duration(best.extent)` comparison — is dropped with no record at all:
no span, no deviation, and not even the `carrier_rejected` measure VOICE wrote. A second DDK train
is therefore invisible in exactly the way a second glide was, and a carrier that failed
`train_min_s` or produced no rate leaves no trace either.

Not fixed: it is a different branch with its own expectations table, and no owner case has been
raised against it. It is the same defect and should be taken up on the same terms — every carrier
holding a readable train is an attempt; the longest is proposed; each further one is a
`repeat_attempt`; a criterion that reads how *good* the train was must not decide that no train
happened.

## Not done

- **GLIDE's `task_extent` is still the sweep, not the carrier.** The owner's related judgment is that
  the amplitude span is the production and the monotone sweep is a measurement about it, which would
  make the carrier the proposed extent. That is a larger change and is not required by any of the
  cases: the glide span now carries `carrier_extent`, as SUSTAINED's already did, so the production's
  own length is legible without it.
- No gate value changed. No stream changed. No threshold was added. No literal entered the code.
