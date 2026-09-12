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
Everything the check needs about amplitude is measured there, in that one pass, and stored **in one
`clip_amplitude` measurement** over the same signal:

* Whole-file — `unclipped_peak`, the loudest unclipped sample's absolute amplitude;
  `unclipped_peak_time_s`, where it sits; `unclipped_samples_n`, how many samples were unclipped
  evidence; and `edge_guard_samples`, the guard that produced all of the above.
* Per span, **keyed by the span's entity id** — `clip_levels`, the peak absolute amplitude over the
  samples each span covers (null where the extent names no sample of the signal), and
  `unclipped_louder_n`, how many unclipped samples are louder than that span's own level.

It is `wasDerivedFrom` the recording stream and every span, because it is that recording read with
those spans excluded. A `clip` span itself carries `family`, `signal` and its extent, and no
amplitude at all.

All of these are scalars, so they sit in entity attributes and there is no sidecar: `path_attributes`
and its digest exist for arrays written to `derivatives/`, and two small maps of numbers are not an
array. `preprocess.write_clip_spans` writes the spans and the measurement together — they are one
reading of one waveform, and splitting them would let a store carry spans whose levels were measured
under a different guard from the peak they are compared against.

QUALITY then reads each span's level out of the measurement against `unclipped_peak`, applies its
own tolerance, and writes assertions. No pass over samples, no `Audio`, no stream load.

### Why the levels are in the measurement and not on the spans

This was a specification error, corrected here rather than lived with. The per-span levels were
first stamped on each `clip` span as `clip_level` and `unclipped_louder_n` attributes, which is the
natural home for them on a fresh run and forecloses every other one.

**The store is append-only.** An entity is `sha256([run_id, prov_type, extent, attributes])` and
nothing is modified after it is added, so an attribute cannot be added to a span that already
exists — writing the span again with the level in it mints a *different entity*, and the store would
then hold two live clip spans over the same extent. The 62,550 recordings of the completed corpus
carry clip spans written before any of this existed. Under the span-attribute design the only way to
give QUALITY its inputs was a full PREPROCESS re-run — enhancement, YAMNet, AST, HeAR and every ASR
recomputed, roughly a day on a 128-task array — to obtain a handful of amplitude scalars.

A measurement is an entity of its own. It can be appended beside spans that already exist, computed
from the stored spans plus the source audio, and nothing else is recomputed and nothing existing is
touched.

The keying is the span's entity id rather than its index, because an index is only meaningful
against the list that produced it and the reader selects live spans by family and signal rather than
replaying a write order. A level a reader cannot attribute to a particular span is not a level.

There is a second property, and it is the one that makes the corpus's spans and a fresh run's the
*same* entities: with no amplitude in a span's attributes, a clip span is identified by its family,
its signal and its extent alone — exactly what the corpus's spans carry. Had the level stayed an
attribute, an extended corpus and a re-run one would disagree about which entity a given clip span
is, for every clip span in the corpus.

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
guard its numbers were measured under. Changing it is a PREPROCESS-side change that needs the
amplitudes measured again, which is the honest consequence of the statistic being a measurement
rather than a view — but only the amplitudes, not the whole pass, because the extend path below
recomputes them from the stored spans alone.

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
  exactly like a recording whose clip evidence is consistent. The refusal is what makes the extend
  pass below necessary rather than optional.

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

## Extending a finished run

A store from an earlier run carries clip spans and no `clip_amplitude` measurement, so QUALITY
refuses on it. **The corpus needs an append, not a PREPROCESS re-run.** The measurement is a
function of two things the finished run already identifies: its clip spans, which are in the store,
and the source recording, whose absolute path and SHA-256 ADMIT wrote onto the `recording` stream
entity. Neither is a model output and neither needs any other block re-run.

`scripts/extend_clip_amplitudes.py` is that pass, and
`preprocess.extend_clip_amplitudes(store, config, run_dir=...)` is the block it drives. Per
recording: read `store.jsonl` with `ProvStore.read_jsonl(path, run_id=<run root name>)`, select the
live `clip` spans over `recording`, load the source the stream entity names, measure, append one
activity and one measurement, capture environments, write the store back and re-export `prov/`.
Nothing is written outside the recording's own run and no existing record is touched.

**It is its own driver rather than a mode of `extend_ppg_praat.py`.** That driver's whole shape —
the batch size, the device, the venv gate, the whole-batch failure handling — exists for a model
call, and this pass has none: it is a decode and a numpy scan. What the two genuinely share is the
run layout, the store read/write, the BEP028 re-export and the manifest slicing, and those are now
`senselab.audio.workflows.triage.extend`, imported by both, so a layout change cannot fix one driver
and miss the other.

### Where the source audio comes from

Not from the run tree: `run/streams/` holds the conditioned streams, and clip spans are detected on
the *original*. Not from `summary.json` either. The `recording` stream entity ADMIT wrote carries
`path`, an absolute resolved path, together with `checksum_sha256` — so every run names its own
source, the library holds no corpus path, and the manifest is needed only to enumerate which runs an
array task owns.

The digest is checked before anything is measured, and a mismatch refuses that recording. This is
the one failure the pass could otherwise commit silently: amplitudes measured on bytes other than
the ones the spans were detected on would be keyed against those spans and indistinguishable, in the
store, from amplitudes that belong to them.

### Convergence

The PPG extend pass found that `path_attributes` includes `mtime_ns`, so re-running a block that
writes a sidecar produces a byte-identical file at a new mtime, which is a different attribute set,
which is a **differently identified entity**; set-union merging does not save you, and convergence
has to come from skipping.

This measurement writes no sidecar, and neither its activity's parameters nor its own attributes
carry a path, a timestamp or anything else that varies between two readings of the same file. Two
passes therefore mint the *same* activity id and the *same* entity id, and the second is a genuine
set-union no-op. That property is worth having and is not what the driver relies on: a recording
whose store already holds a live `clip_amplitude` measurement is skipped outright and its store is
not rewritten at all, so a completed slice re-run is a pass over `store.jsonl` and no decodes. A
recording with no clip span is skipped too — nothing was asserted, QUALITY passes such a store
without the measurement, and writing an empty one would be noise.

A task killed mid-slice restarts and redoes only what never landed; `store.jsonl` is written to a
`.partial` sibling and `replace`d, so a task killed mid-write leaves no truncated store for the next
pass to read.

### What the extend pass does not converge on

The activity it writes is `PREPROCESS`/`clip_amplitude`, not the fresh path's
`PREPROCESS`/`clip_spans`. The fresh path detects and measures in one activity over one waveform;
the extend pass measures over spans it did not detect, and an activity carrying `near_threshold`,
`leniency_samples`, `minimum_extreme` and `merge_gap_ms` it never ran would be a claim that the
detector executed here. The entities agree between the two paths — the spans are identical and the
measurement is identical — and the activity that generated the measurement differs, which is the
honest reading.

`quality.clip_edge_guard_samples` is read by PREPROCESS on both paths and stamped on the
measurement, so a verdict always names the guard its numbers were measured under. The merged
configuration's values are unchanged by any of this, so `config_hash` alone does not separate a run
made before this change from one made after.

## Running QUALITY over a finished run

The corpus pass ran ADMIT → PREPROCESS → TAXONOMY → FIGURE. ROUTING, the branches and QUALITY never
executed: `branch_decision` and `QUALITY` appear in 0 of 150 sampled stores. So the node itself has
to be run over the finished stores, and `scripts/extend_quality.py` is that pass —
`extend.extend_quality(store, config, run_dir=...)` per recording, over the same run layout, store
read/write, BEP028 re-export and manifest slicing as the two drivers above.

Derivable, and trivially so: QUALITY's contract is that it reads stored outputs only. Every input it
takes — the `recording` stream, the `clip` spans over it, the `clip_amplitude` measurement beside
them — is in the store the finished run left behind, and the last of the three is what
`scripts/extend_clip_amplitudes.py` appended. **That ordering is the pass's one prerequisite**: run
the clip driver first, or every store carrying a clip span is a refusal.

### What a verdict written after the fact may claim

QUALITY writes a verdict, and a verdict is a conclusion. Nothing is retired here — the run carries
no QUALITY verdict to replace, so the question the re-bracketing work answered (never retire a
conclusion into a hole) does not arise. The question that does arise is the opposite one: a verdict
is normally reached at a particular place in the graph, and this one is not.

The position is part of QUALITY's contract — "after every branch, on every path PREPROCESS
completed" — and it is a claim about what *had been written* when the node read. On a corpus store
nothing had: no routing decision, no branch verdict. The reading QUALITY produces is unaffected,
because it reads none of them, and the entity it mints is the entity a fresh run would mint from the
same records. What differs is what stands beside it in the store, and a verdict that did not say so
would be indistinguishable from one reached after a full graph pass.

So the verdict says so, in a field measured off the store rather than asserted by the driver:
**`preceded_by`, the `node` of every live verdict entity other than QUALITY's own**, sorted. A fresh
run's QUALITY verdict carries `ADMIT, PREPROCESS, TAXONOMY, routing` and whichever branches ran; a
corpus store's carries `ADMIT, PREPROCESS, TAXONOMY` and no branch at all. The same field on both
paths, because it is a fact about the store either way, and a reader needs it on the fresh path too:
a branch that raised also wrote no verdict.

QUALITY's own node name is excluded, which is what keeps the field stable under a second pass — see
convergence below.

The activity is the fresh path's `QUALITY`/`clip_consistency`, with the fresh path's parameters,
because that computation really ran on those inputs. There is nothing here for a different name to
protect against, which is the judgement `PREPROCESS`/`clip_amplitude` failed and this one passes.

### The file verdict is not re-folded

VERDICT already ran on these recordings and folded a file verdict from an execution record with no
QUALITY in it. This pass does not re-fold: `fold_file_verdict` folds the runner's per-node `ran`
mapping, which is the runner's record and not a store fact, and a QUALITY flag raising a file to
`Triage.FLAG` on a run whose branches never executed would be a file verdict for a pass that never
happened. The store therefore holds a QUALITY verdict the file verdict predates, and `preceded_by`
naming `VERDICT` is exactly how a reader tells: a fold cannot have included a verdict written after
it. This is consumer note 4 of the re-bracketing design in its own terms — the words were re-flagged
and the conclusions drawn from them were not.

### No clip span is a pass, not a refusal

Roughly 84 % of the corpus carries no clip span (`clip_amplitude` is present in 43 of 150 sampled
stores after the clip pass, which is the ~16 % carrying clip spans). QUALITY passes such a store —
nothing was asserted, so nothing can contradict it — and that verdict is written like any other.
Writing nothing there would leave those recordings indistinguishable from the 62,550 that had not
been read at all, which is the state this pass exists to end. The refusal is the other case, and
only the other case: clip spans present with no `clip_amplitude` beside them, which is a store the
clip driver has not reached, recorded as that recording's error.

### Convergence

Per recording: read the store, take `store.fingerprint()`, run the node, and write the store,
re-export `prov/` and record the environment **only if the fingerprint moved** — the reprocessed
driver's rule, for the same reason. Neither the activity's parameters nor the assertions' nor the
verdict's attributes carry a path, a timestamp or anything else that varies between two readings of
the same records, and `preceded_by` excludes QUALITY itself, so a second pass over an extended store
recomputes exactly the activity, assertions and verdict it already holds: a set-union no-op that
leaves the fingerprint where it was. The driver also skips a store already carrying a live QUALITY
verdict outright, which saves the recomputation rather than buying the convergence.

A refusal leaves the store unwritten. QUALITY adds its software agent before it reads the
measurement, so a refused store's in-memory copy has moved; it is discarded rather than written, and
the recording keeps the store it had.

### Verification

`src/tests/scripts/extend_quality_test.py`, over synthetic finished runs: the verdict and its
contest land, the contested span is kept and the contest is derived from it, `preceded_by` names
what had concluded and never QUALITY itself, a run with no clip span passes rather than refusing, a
clip nothing is louder than passes, clip spans with no amplitudes are refused and that store is left
byte-identical, a second pass is a fingerprint no-op reporting `present`, `prov/` is re-exported
carrying the verdict, the host environment is recorded, and a missing store is one recording's error
with its neighbour still reaching a verdict.

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
