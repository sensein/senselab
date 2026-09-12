# QUALITY's clip-consistency check

QUALITY is the terminal graph node: it runs after the branches, on every path PREPROCESS completed,
and is the only node that sees the whole recording rather than one branch's view of it. Its first
check reads PREPROCESS's clip spans back against the amplitudes PREPROCESS measured beside them.
That question is a whole-file one — a clip span can only be contradicted by a sample somewhere else
in the same recording, possibly inside a different span — and no branch holds that view.

QUALITY reads stored outputs and nothing else. It decodes no audio and re-derives no statistic: the
node that held the waveform measured every number it compares. This is the node's contract, and the
rest of this document is written against it.

## What a clip span asserts

`preprocess.py` (`_clip_spans`) runs `detect_clip_events`
(`src/senselab/audio/tasks/clipping/api.py`) over the **original** recording at
`clipping.near_threshold` 0.995, merges events within `clipping.merge_gap_ms` 30 ms, and writes each
merged range as a `span` with `family: "clip"`, `signal: "recording"`.

The detector's threshold is the file's own global max and min, not full scale, and each extreme is
guarded on its own magnitude:

```python
watch_max = abs(global_max) >= minimum_extreme
watch_min = abs(global_min) >= minimum_extreme
```

An event *opens* where an extreme is **held** — `_held_extreme` matches `x == value` exactly, over
`MINIMUM_EXTREME_RUN = 2` consecutive samples below digital full scale, or a single sample at it —
and then extends while samples stay within `near_threshold` of that same extreme, tolerating up to
`leniency_samples` consecutive dips below the band before closing at the sample where the tolerance
was exceeded.

The claim a clip span makes is therefore: *over this extent the signal was at the recording's
ceiling*.

## Where the false positive comes from

The two extremes are independent, and neither is required to be near full scale. Two mechanisms
produce a span whose level is not the recording's ceiling:

1. **One-sided saturation.** `global_min` at −1.0 from real negative saturation, `global_max` at
   0.3 because some quantised plateau happens to repeat its value across two consecutive samples.
   `_required_run(0.3)` is 2, the repeat qualifies, and a positive clip event opens at 0.3 — a
   plateau, not a ceiling.
2. **An unheld maximum.** A sub-full-scale `global_max` that occurs once and never repeats opens no
   event at all (`_held_extreme` finds no qualifying run), so the loudest sample in the file sits
   outside every clip span while a lower negative plateau carries one.

Both leave the same internal inconsistency: the recording is said to have reached its ceiling at
amplitude *A*, and some sample nothing called clipped sits above *A*. The lower claim is very
likely a false positive, which is what this check exists to reduce.

## Who measures what

The waveform is in hand exactly once, in `_clip_spans`, which already reads it to detect the events.
Everything the check needs about amplitude is measured there, in that one pass, and stored:

* **On each `clip` span** — `clip_level`, the peak absolute amplitude over the samples the span
  covers (null where the extent names no sample of the signal), and `unclipped_louder_n`, how many
  unclipped samples are louder than that span's own level.
* **Beside them, one `clip_amplitude` measurement** over the same signal — `unclipped_peak`, the
  loudest unclipped sample's absolute amplitude; `unclipped_peak_time_s`, where it sits;
  `unclipped_samples_n`, how many samples were unclipped evidence; and `edge_guard_samples`, the
  guard that produced all of the above. It is `wasDerivedFrom` the recording stream and every span,
  because it is that recording read with those spans excluded.

All of these are scalars, so they sit in entity attributes and there is no sidecar: `path_attributes`
and its digest exist for arrays written to `derivatives/`, and four numbers are not an array.
`preprocess.write_clip_spans` writes the spans and the measurement together — they are one reading
of one waveform, and splitting them would let a store carry spans whose levels were measured under a
different guard from the peak they are compared against.

QUALITY then reads `clip_level` per span against `unclipped_peak`, applies its own tolerance, and
writes assertions. No pass over samples, no `Audio`, no stream load.

## What amplitude represents a clip span

**Its peak |x| over the samples it covers.**

The event's defining evidence is the extreme it *held*: `_held_extreme` opens the run on samples
equal to `global_max`/`global_min` exactly, and everything after that is the run continuing while it
stays inside the band, plus up to `leniency_samples` dips per break that are below the band by
construction. The only level the detector asserts is the extreme, and the extreme is the peak of the
span. A median plateau level would sit below what was asserted — dragged down by the tolerated dips
and the decaying tail — and would manufacture contradictions on genuine, fully saturated clips whose
runs happen to carry a long lenient tail.

The peak is also the right reading under merging. `merge_gap_ms` coalesces events of either polarity
into one span and drops their polarities; the merged span's peak is the larger of the two extremes,
which is the strongest claim the merged span makes and the only one a reader of the store can
recover.

**The limitation that follows.** Merging can hide mechanism 1: a bogus positive event within 30 ms
of a genuine negative one is merged into a single span whose peak is the negative extreme, and that
span is never contested. Separating them would require `_clip_spans` to record per-event polarity
and per-event extents, which it does not; the span is PREPROCESS's unit of assertion, and QUALITY
contests spans rather than events it cannot see.

## What counts as a non-clip sample

Every sample outside every clip span, less a guard band of `quality.clip_edge_guard_samples` on each
side of each span, read from the same channel-averaged signal `detect_clip_events` read
(`x.mean(axis=0)`), on ADMIT's `recording` stream. The original, never `plain`: peak-normalisation
and resampling destroy the flat plateaus clipping consists of — the same reason `disruptions.*` is
measured on the original (`specs/20260817-triage-workflow-dag/config-derivations.md`,
`## disruptions`).

The guard is 3 samples, which is `clipping.leniency_samples`: the same tolerance window, at the same
sample rate, on the same signal. A run closes at the sample where its own leniency was exceeded, so
the samples immediately outside an edge are that run's decay at the detector's own temporal
resolution and are not independent evidence about anything. The cost is bounded — 6 samples per
span, 0.375 ms at 16 kHz — and it is one-directional: a guard can only suppress a finding, never
create one. `clip_edge_guard_samples: 0` restores the naive reading without a code change, which is
what the control test uses.

**The guard is applied in PREPROCESS**, because that is where the samples are. The key stays under
`quality:` — it is a parameter of the consistency check rather than of clip detection, and the
derivation above is the reason it has the value it has. `_clip_spans` reads it, records it in the
step's activity parameters and stamps it on the measurement as `edge_guard_samples`; QUALITY reports
the value it finds there rather than re-reading the key, so the guard a verdict names is always the
guard its numbers were measured under. Changing it is a PREPROCESS-side change that needs the pass
re-run, which is the honest consequence of the statistic being a measurement rather than a view.

Samples *inside* a span, including the interior of a merged one, are claimed clipped and are
therefore not evidence.

## The tolerance

`quality.clip_contradiction_margin: 0.005`. A span is contested only when
`unclipped_peak > clip_level * (1 + margin)`; equality does not fire. This one stays QUALITY's, read
from the configuration at every call, because it is a comparison over stored scalars and needs no
waveform: widening it quietens the check without re-running PREPROCESS.

0.005 is `1 - clipping.near_threshold`, the detector's own band. A sample within 0.5 % of a clip
level is one the detector would have counted as part of that run had it been contiguous with it, so
it cannot be evidence that the level was not the ceiling. The exactly symmetric figure is
`1/0.995 − 1 = 0.005025`; 0.005 is 0.5 ‰ tighter, far below anything measured here, and is the
number `near_threshold` is stated with.

Against quantisation: one int16 step is `1/32768 = 3.05e-5`, absolute, while the margin is relative.
The margin exceeds one step for every clip level above `3.05e-5 / 0.005 = 6.1e-3`.
`clipping.minimum_extreme` is `1e-4`, so a clip span whose level falls between `1e-4` and `6.1e-3`
has a margin narrower than its own file's quantisation and can be contested by a sample one LSB
louder. Such a file peaks below −44 dBFS and is near-silence; the bound is named rather than papered
over with a second, absolute floor nobody has measured. If the corpus turns up contradicted clips at
those levels, the fix is that second floor with its own derivation, not a wider relative margin.

The test fixtures round-trip PCM_16, as a real recording does, so 0.502 (1.004×) and 0.504 (1.008×)
over a 0.5 clip straddle the margin unambiguously after quantisation.

**Where the margin is not applied.** `louder_samples_n` on a contest counts unclipped samples above
the span's own `clip_level`, not above `clip_level × (1 + margin)`. The margin decides *whether* a
span is contested, and that decision is exactly `unclipped_peak > clip_level × (1 + margin)` — one
scalar comparison, unchanged by where the count is taken. The count is descriptive: it tells a
reader whether the contradiction is one stray sample or a sustained louder region. Counting at the
margin-adjusted threshold instead would mean storing the distribution of unclipped magnitudes near
every clip level so that any later margin could be answered exactly; the samples that separate the
two counts lie inside the detector's own near-band, which is 0.5 % of the level, and no reading of
the finding turns on them.

## What the node does about it

It records; it does not rewrite. The store is append-only and the clip spans are PREPROCESS's
reading, so nothing is invalidated here.

* One `assertion` per contested span, `verb: "contest"`, `claim: "clip"`,
  `reason: "clip_above_unclipped_sample"`, `wasDerivedFrom` the span, carrying the span's extent, its
  `clip_level`, the `louder_amplitude`, its `louder_time_s` and how many unclipped samples clear the
  span's level. This is AIRWAY's shape (`nodes/airway.py`, step 2), which is how this graph already
  answers another node's claim without withdrawing it.
* One verdict, `outcome: flag`, `kind: None`. Not a branch, so it is the authority on no kind and
  `fold_file_verdict` gives it no kind-state power; a node flag does raise the file to
  `Triage.FLAG`, which is the right reading of a recording whose own clip evidence contradicts
  itself.
* Never `fail`, and never a raise *for a finding*. `run._attempt` records a raise as `ERRORED`,
  which is an operational fact about the run; a contradicted clip is a finding about the recording,
  and the two must stay distinguishable.
* `pass` with a distinct `why` for each of the two clean cases — no clip span at all, and clip spans
  none of which is contested.

## What it refuses

Three raises, all of them before the store is written to, all of them operational rather than
findings:

* A null `quality.clip_contradiction_margin` — a threshold nobody took a decision on.
* No live `recording` stream. QUALITY does not open it, but it names it as the thing its findings
  are about and records a `used` edge to it, and a store with no such stream is not one this check
  has a subject in.
* Clip spans over that signal with **no live `clip_amplitude` measurement** beside them. This is the
  input the check is built on, and the alternative to refusing is the failure mode that matters
  most here: the node would find no contradiction it could measure and write a `pass`, which reads
  exactly like a recording whose clip evidence is consistent.

The absence of clip spans is not a refusal. Nothing was asserted, so there is nothing to read
against, and a store where `_clip_spans` never ran already records that in PREPROCESS's `absent`.

## Wiring

`run._drive_branches` calls QUALITY after the branch loop and before REDACT, inside the PREPROCESS
gate. The position is the contract, not an accident of line order, and `run_test`'s
`TestTheTerminalNodeContract` pins it: every branch that ran precedes QUALITY, a branch that *raised*
precedes it too (a branch that raised has finished), and QUALITY precedes REDACT and VERDICT. It
reads no branch output and no routing decision, so it runs whatever routing selected and whether or
not routing itself raised — `run_test` pins both of those as well.

The clip spans are PREPROCESS's, and a run that never conditioned has none to read, which is why the
call sits inside the PREPROCESS gate rather than outside it.

## Re-running

`quality.clip_edge_guard_samples` is now read by PREPROCESS, and PREPROCESS writes two things it did
not write before. A store from an earlier run carries clip spans with no `clip_level` and no
`clip_amplitude` measurement, so QUALITY refuses on it: the corpus has to be re-run for the check to
have inputs. The merged configuration's values are unchanged, but the pass that consumes them is
not, so `config_hash` alone does not separate a run made before this change from one made after.

## What this check does and does not tell you

It measures **internal inconsistency of the detector's output**, not clip ground truth. A contested
clip is very likely a false positive; the store says it is contradicted, not that it is wrong. Two
gaps run the other way: an uncontested clip is not thereby correct (the merging limitation above),
and a recording clipped throughout has no unclipped evidence to contradict anything —
`unclipped_samples_n` in the verdict is what names that case.

A third gap follows from reading stored outputs rather than samples. QUALITY can still compare any
span's level against the whole-file unclipped peak, which is the comparison the check is made of,
and against a per-span count of samples louder than that span. What it can no longer reach is
anything about *where* unclipped evidence sits relative to a given span: the loudest unclipped
sample within a second of it, the second-loudest one elsewhere, the shape of the unclipped
magnitude distribution. One peak and one time serve every span. None of that was asked for, and each
of them is a new measurement in `_clip_spans` rather than a change in QUALITY.

Turning a corpus count of contradicted clips into a statement about false clips needs per-span
labels: a sample of contested and uncontested spans, each looked at (waveform plus the spectrum
around it) and called saturated or not, so the contested set's precision can be read against the
uncontested set's. Without that, the count says how often PREPROCESS's clip evidence disagrees with
itself, which is a prevalence, not a rate.
