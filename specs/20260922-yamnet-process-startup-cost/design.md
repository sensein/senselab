# YAMNet's cost is starting a process, not classifying audio

A step-level measurement over the finished 62,548-recording corpus attributed **46.6% of PREPROCESS
wall clock to `enhanced_yamnet` at a 100 s mean**
(`specs/20260922-replay-decisions-over-a-finished-corpus/design.md:63`). 100 s is implausible as
inference for a small model over a median recording, so the hypothesis was that the cost is fixed
per invocation: `yamnet.py` ran a fresh interpreter, a fresh TensorFlow import and a fresh TF-Hub
load on every call, and PREPROCESS calls it more than once per recording.

Two things came out of testing it. The fixed cost is real and dominant. The 46.6% is not: it is an
off-by-one in the probe that produced it, and the step it names costs 4.98 s.

## How these figures were taken

Everything below is from `mit_preemptable` with `--exclusive --nodes=1`, so the node ran nothing
else. `loadavg` at job start was 0.00 on node2312 (cost probe) and is echoed by every job script.
This matters: the corpus run itself was on contended nodes at load averages in the hundreds, so its
absolute seconds are not these seconds. Where a corpus-wide projection is made below it is made as
a ratio against the corpus run's own PREPROCESS mean, never by transplanting an idle-node second
into a loaded-node total.

Scripts in this directory, each run through the sbatch beside it:

| script | what it measures | job |
| --- | --- | --- |
| `probe_yamnet_cost.py` | fixed vs marginal cost of one invocation | 23464721 |
| `probe_preprocess_steps.py` | corrected step table; every YAMNet invocation | 23464900 |
| `probe_worker_rss.py` | what a resident worker holds | srun, `mit_quicktest` |
| `probe_equivalence.py` | old path against new, same recordings | 23465390 |

## The split: 4.81 s fixed, 0.0005 s per second of audio

`probe_yamnet_cost.py` runs the shipped worker script unchanged except for per-phase timers, over
band-limited noise at ten durations from 1 s to 240 s, three repeats each, one audio per
invocation. Parent-side wall clock around `subprocess.run`:

```
wall_s = 4.809 + 0.0005 * audio_seconds     (r2 on the duration term 0.010, n=30)
```

The slope is indistinguishable from zero over a 240-fold range in duration. The model's own work is
the only thing that scales, and it scales far below the noise in process start-up:

```
infer + post_s = 0.158 + 0.0009 * audio_seconds    (r2 = 0.978)
```

Per-phase medians, seconds:

| audio s | wall | interpreter + import | TF-Hub load | class map | inference | post |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4.73 | 2.08 | 1.88 | 0.02 | 0.15 | 0.00 |
| 7 (corpus median, interpolated) | 4.81 | — | — | — | 0.16 | 0.00 |
| 21 | 4.65 | 2.06 | 1.84 | 0.02 | 0.18 | 0.00 |
| 120 | 4.74 | 2.05 | 1.87 | 0.02 | 0.25 | 0.00 |
| 240 | 5.00 | 2.05 | 1.92 | 0.02 | 0.38 | 0.01 |

The 4.81 s decomposes as 2.0 s interpreter start plus TensorFlow import, 1.85 s TF-Hub load of the
cached SavedModel, 0.02 s class map, and ~0.95 s of spawn, pipe and parent-side overhead.

**At the corpus median recording of 7.3 s, 99.9% of an invocation is start-up.** The corpus median
is measured here, not assumed: 900 recordings sampled from `corpus_manifest.jsonl` read at the WAV
header give min 0.2 s, p25 4.6 s, median 7.3 s, p75 18.7 s, p90 30.0 s, max 130.5 s. The brief's
working figure of 15 s was already generous.

A second audio inside a process that has paid the fixed cost is nearly free — 21 s of audio for
0.038 s:

```
wall_s = 4.697 + 0.0378 * n_audios_of_21s
```

so 16 recordings in one invocation cost 5.23 s, or 0.33 s each.

## Four invocations per recording

`probe_preprocess_steps.py` wraps `subprocess.run` and records every call with the PREPROCESS step
that was registered when it was made. Job 23464900, exclusive node. Its `loadavg_start` was 5.54 /
12.69 / 29.49, decaying from the previous tenant rather than from a co-tenant — the allocation was
exclusive and `loadavg_end` was 2.90 / 3.86 / 9.32 with our own nine recordings in it. The cost
probe (23464721) and the equivalence run (23465390) both started at 0.00, and it is those two that
carry the absolute seconds; this job's job is the ratio between steps.

| call site | block | batch |
| --- | --- | --- |
| `_yamnet_scores` | `yamnet_scores` | the plain stream, 1 audio |
| `_stream_yamnet("enhanced")` | `enhanced_yamnet` | the enhanced stream, 1 audio |
| `_stream_yamnet("residual")` | `residual_yamnet` | the residual stream, 1 audio |
| `_span_yamnet` | `span_yamnet` | every native-length span, **already batched into one call** |

Measured, per recording:

| recording s | invocations | YAMNet s | share of the whole graph | per call: plain / enhanced / residual / span |
| --- | --- | --- | --- | --- |
| 2.79 | 4 | 21.1 | 10.8% | 6.76 / 4.97 / 4.50 / 4.85 |
| 3.72 | 4 | 19.1 | 23.1% | 4.93 / 4.76 / 4.57 / 4.81 |
| 4.54 | 3 | 14.4 | 12.5% | 4.80 / 5.18 / 4.40 / — |
| 5.58 | 4 | 19.4 | 16.2% | 5.10 / 4.90 / 4.50 / 4.93 |
| 7.29 | 3 | 14.3 | 12.5% | 4.84 / 5.00 / 4.48 / — |
| 10.18 | 4 | 19.4 | 16.6% | 5.08 / 4.89 / 5.09 / 4.35 (3 spans) |
| 18.32 | 4 | 19.0 | 18.4% | 4.88 / 5.01 / 4.54 / 4.58 (10 spans) |
| 30.02 | 4 | 19.3 | 10.6% | 4.52 / 5.17 / 4.60 / 4.98 |
| 73.54 | 4 | 20.5 | 6.8% | 5.24 / 5.21 / 4.93 / 5.16 |

Median 4 invocations and 19.3 s, mean 18.5 s, **12.5% of the whole graph** over these nine — of
which about 0.7 s is inference. The per-call column is flat across a 26-fold range in recording
length, which is the same result as the duration sweep, arrived at through the real pipeline.

Two things the table settles that a code reading would not have.

**Three, not four, when no span reaches YAMNet's native frame.** At 4.54 s and 7.29 s every span was
shorter than 0.96 s, so all of them took the covering-window attribution path, `native_prepared` was
empty, and `_classify_spans_in_batch` returned without calling. The count is 3 or 4, not a constant.

**`_span_yamnet` does not spawn a process per span.** The brief's worry was checked and is not the
case: `_classify_spans_in_batch` (`nodes/preprocess.py:287`) hands every prepared span to one
`classify_audios` call and only falls back to one call per span when the batch as a whole fails. The
probe sees batches of 1, 3 and 10 spans each costing one ~4.5 s invocation. That is the one level of
reuse that already existed, and it is why `span_yamnet` is not the outlier the brief expected.

## The 46.6% was an off-by-one, and this is what the steps actually cost

Every PREPROCESS block calls `_step(...)` — which registers its provenance activity — *before* doing
its work. The replay probe stamped "the interval since the previous registration" onto the activity
being registered, so each interval was labelled with the name of the step that was *starting*, not
the one that had just run. `enhanced_yamnet` is registered immediately after the `residual` block,
whose work is an FRCRN enhancement, a cross-correlation alignment and two stream writes. That is
what the 100 s was.

The same artefact is visible elsewhere in that table without needing this probe:
`enhanced_hear_summary` at 5.97 s and `residual_hear_summary` at 5.67 s are pure Python pooling over
a window list and cannot cost seconds — they are the HeAR inferences that preceded them.

`probe_preprocess_steps.py` attributes `[t_i, t_{i+1}]` to step *i*, which is correct given
registration-before-work. Over nine recordings:

Over nine recordings, grand total 1,330.9 s:

| step | n | mean s | share |
| --- | --- | --- | --- |
| PREPROCESS `asr_qwen` | 9 | 28.64 | 19.4% |
| PREPROCESS `asr_crisperwhisper` | 9 | 19.88 | 13.4% |
| SPEECH `pii` | 7 | 15.89 | 8.4% |
| QUALITY `clip_consistency` | 9 | 10.96 | 7.4% |
| PREPROCESS `residual` | 9 | 10.69 | 7.2% |
| PREPROCESS `gammatone` | 9 | 7.40 | 5.0% |
| PREPROCESS `hear` | 9 | 6.80 | 4.6% |
| PREPROCESS `span_hear` | 9 | 6.59 | 4.5% |
| PREPROCESS `enhanced_hear` | 9 | 6.57 | 4.4% |
| PREPROCESS `residual_hear` | 9 | 6.39 | 4.3% |
| PREPROCESS `yamnet` | 9 | 5.17 | 3.5% |
| PREPROCESS **`enhanced_yamnet`** | 9 | **5.04** | 3.4% |
| PREPROCESS `praat_features` | 9 | 4.79 | 3.2% |
| PREPROCESS `residual_yamnet` | 9 | 4.65 | 3.1% |
| PREPROCESS `span_yamnet` | 9 | 3.87 | 2.6% |

`enhanced_yamnet` is 5.04 s, not 100.20 s — a factor of 20. The replay document's claim that one
step is 46.6% of PREPROCESS does not survive a correctly-attributed probe, and
`specs/20260922-replay-decisions-over-a-finished-corpus/design.md` is corrected alongside this. The
steps that are actually large are the two ASR passes, SPEECH's PII scan, and the FRCRN `residual`
that the 100 s was really measuring. No single step is anywhere near half of a recording; the
largest is 19.4%.

(`span_yamnet`'s 3.87 s mean is below its ~4.8 s per-invocation cost because two of the nine
recordings had no span long enough to reach the model, so the step ran without calling it.)

What survives is the smaller, real finding the wrong number pointed at: YAMNet is ~20 s per
recording and almost none of it is classification.

## What the fix is, and the levels it could have been at

Three levels of reuse were available.

**Within one `_span_yamnet` loop.** Already done, and it is why `span_yamnet` costs one invocation
rather than one per span. Nothing to add.

**Within one recording**, so the four calls share a process. Saves 3 x 4.81 = 14.4 s per recording
and requires the worker to outlive a call.

**Across recordings within one driver task**, so the worker also outlives the recording. The corpus
driver runs ~61 recordings per slice in one process, so this takes the four-per-recording cost from
19.2 s to 4.81 s once per slice — 0.08 s per recording amortised.

The third was chosen, because the mechanism that buys the second buys the third for nothing: once
the worker is a process that answers requests rather than one that exits after answering, how long
it lives is a policy question, not a structural one. It is expressed as exactly that — the worker is
resident by construction, and `yamnet.keep_worker_resident` decides whether PREPROCESS ends it when
the recording does.

### The shape, and why it is not shared with the LLM check

`redaction.llm_check.keep_worker_resident` (`nodes/redact.py:644, 689`,
`text/tasks/pii_detection/redaction_review.py`) is the only existing resident worker in the tree: a
`subprocess.Popen` over the venv interpreter with stdin/stdout held open, a marker-prefixed JSON
line protocol, `sys.stdout` redirected to stderr before any heavy import, two daemon pump threads, a
module-global handle under one lock, `atexit` teardown, and a worker-side exit on EOF.

YAMNet's worker is the same shape, deliberately, and is a second hand-rolled copy of it rather than
a shared helper. Generalising was considered and rejected for now on two grounds, both measured
rather than aesthetic:

1. **The lifetime policies differ for reasons specific to each model.** The LLM check's residency is
   `false` by default because a warm worker holds 70.35 GiB of allocated CUDA memory and took 19 of
   22 recordings out of a corpus run with `CUDA error: out of memory`. YAMNet's is `true` by default
   because a warm worker holds 710–971 MiB of host RAM. A shared helper would have to parameterise
   the one thing each caller decides differently, which is the whole of it.
2. **The error contracts differ.** `redaction_review` flattens every worker failure to
   `ReviewWorkerError`; the one-shot `parse_subprocess_result` reconstructs `ValueError` and
   `TypeError` and collapses the rest to `RuntimeError`. Changing YAMNet's callers' exception types
   would change what PREPROCESS records as a hard failure against an absence, so the new path
   reconstructs exactly what `parse_subprocess_result` would have. A shared helper has to pick one,
   and picking either breaks a caller.

`ensure_venv` is called once per process instead of once per call, so the venv lock is touched
strictly less often than before. The failure mode the brief warned about — a dead worker blocking
others — cannot arise from this change: the lock is `threading.Lock` inside one process, the worker
is checked with `poll()` before reuse and replaced when dead, every request has a wall-clock ceiling
after which the worker is killed, and a killed or crashed parent releases the child through the
worker's own EOF exit as well as `atexit`.

## Memory: what residency costs

`probe_worker_rss.py` reads `/proc/<pid>/status` of a live worker.

| phase | VmRSS MiB | VmHWM MiB |
| --- | --- | --- |
| loaded, idle | 710 | 710 |
| after 5 s of audio | 743 | 747 |
| after 30 s | 821 | 821 |
| after 130 s (corpus maximum) | 971 | 979 |

A corpus task is given 32 GB, so the steady state rises by 2.2–3.0%. The **peak** does not move: the
process already existed at this size during each of the four calls; residency only makes the
interval continuous. That is what makes `true` the defensible default, and it is the opposite of the
LLM check, where the first forward pass is what blows the budget.

## Equivalence

The classifier must return what it returned. `probe_equivalence.py` loads the pre-change module out
of git at `884d481c` as its own module — the shipped code, not a description of it — runs both paths
over the same nine recordings at `top_k=521` (the config's own value, the full label space), and
compares `json.dumps(..., sort_keys=True)` of the complete result. **Exact equality is the accepted
tolerance, not a numeric one**: the two paths run the same TensorFlow graph over the same float32
samples in the same interpreter build, differing only in whether the process exits afterwards, so
any difference at all would be a defect rather than a rounding artefact.

**9 recordings, 2.8 s to 73.5 s, 0 mismatches.** Every window of every recording is identical,
including the `Silence` score the emptiness gate reads and the full 521-label vector. Job 23465390,
node2112, `loadavg_start` 0.00, exclusive. `equiv-23465390.jsonl`:

| recording s | windows | identical | old path s | new path s | worker RSS MiB |
| --- | --- | --- | --- | --- | --- |
| 2.79 | 5 | yes | 6.76 | 4.55 (cold) | 750 |
| 3.72 | 7 | yes | 4.60 | 0.18 | 762 |
| 4.54 | 9 | yes | 5.12 | 0.06 | 781 |
| 5.58 | 11 | yes | 4.87 | 0.06 | 797 |
| 7.29 | 15 | yes | 4.68 | 0.07 | 815 |
| 10.18 | 21 | yes | 5.02 | 0.07 | 852 |
| 18.32 | 38 | yes | 4.83 | 0.08 | 912 |
| 30.02 | 62 | yes | 5.04 | 0.10 | 995 |
| 73.54 | 153 | yes | 5.30 | 0.21 | 1181 |
| **total** | | **0 mismatches** | **47.21** | **5.38** | |

The last two columns are the whole result: 8.8x over the nine, and a call that is not the first
costs 0.06–0.21 s against 4.6–6.8 s. The same job also ran the four-invocations-per-recording shape
from cold: **18.86 s old, 4.50 s new**.

`worker_rss_mib` here is the same worker growing across nine recordings of increasing length, and
it tops out at 1,181 MiB — higher than the 971 MiB of the single-recording RSS probe, because this
process classified all nine without a restart. That is the figure a long-lived driver should be
sized against: **~1.2 GiB**, still 3.7% of a task's 32 GB.

## What was not measured

- **The fixed cost on a loaded node.** Every figure here is from an idle exclusive node. Process
  start-up is import-bound and therefore filesystem- and contention-sensitive, so on the corpus
  run's nodes the 4.81 s is a floor, likely by a large factor, and the saving is correspondingly a
  floor too. No loaded-node measurement was taken.
- **The corpus-wide saving in absolute hours.** Stated below as a ratio only, for the same reason.
- **HeAR, which has the same disease and is bigger.** The corrected step table shows `hear` 6.80 s,
  `span_hear` 6.59 s, `enhanced_hear` 6.57 s and `residual_hear` 6.39 s — four subprocess-venv
  invocations of a second TensorFlow model, 26.4 s per recording against YAMNet's 18.5 s, **17.8% of
  the graph against YAMNet's 12.5%**. Its own fixed/marginal split was not measured and the
  resident-worker change was not applied to it. It is the obvious next target and the larger one.
- **`gammatone` at 7.40 s and `clip_consistency` at 10.96 s**, both of which the probe shows making
  subprocess calls of ~5–11 s. Not investigated.

## Projected saving

Measured directly rather than projected: YAMNet was **12.5% of the whole graph** over the probe's
nine recordings, at a median 19.3 s and 4 invocations. The equivalence job measured the
four-invocation shape end to end at **18.86 s before and 4.50 s after** from cold, and 0.06–0.21 s
per call once the worker is up.

| | per recording | share of the graph, measured |
| --- | --- | --- |
| YAMNet today | 14.3–21.1 s, median 19.3 | 12.5% |
| resident within a recording | 4.5 s | ~3.0% |
| resident across recordings in a task | 0.1–0.3 s | ~0.2% |

Applied to the corpus run's own totals as a ratio — PREPROCESS 90.4% of 10,358,336 recording-seconds
— YAMNet is on the order of **12% of the whole corpus's wall clock** and nearly all of it goes.
Stated as a ratio deliberately: these seconds are idle-node seconds and the corpus ran at load
averages in the hundreds, where process start-up costs more than 4.81 s and the saving is
correspondingly larger, not smaller.

## Is `residual_yamnet` still running, and does anything read it?

Yes to both, and the second is the answer to the question asked. Traced against the code rather than
assumed.

- **`residual_yamnet_scores`** (the measurement and its `derivatives/` sidecar): **no reader**.
  Neither has `residual_ast_scores` nor `residual_hear_scores`. Every `*_scores` read in the tree is
  the plain, unprefixed one — `nodes/taxonomy.py:70` over `SUMMARISED_CLASSIFIERS`,
  `nodes/airway.py:336, 933, 947, 1138` on the literal `hear_scores`. `find_measurement` /
  `find_measurements` (`nodes/common.py:353, 403`) match on exact name equality; there is no prefix,
  suffix or regex store query anywhere in the repo.
- **`residual_yamnet_summary_all`**: **read, and load-bearing in the live pipeline.**
  `routing_analysis/features.py:339-356` matches `f"{stream}_{classifier}_summary_all"` over
  `("enhanced", "residual")` and records `"residual|yamnet"` in `features.classifier_streams`. The
  emptiness bypass reads exactly that: `taxonomy.ruleset.emptiness.peak_streams` is
  `[enhanced|yamnet, residual|yamnet]` (`data/config/default.yaml:435`), evaluated through
  `routing_analysis/ruleset.py:342-364` and `detectors.py:550-555`, reached from `nodes/routing.py:221`.
  Without it the bypass returns `UNAVAILABLE` and the vocabulary reports "the emptiness bypass could
  not be read". It is also read by REPORT's cover (`nodes/figure.py:796`).
- **`residual_yamnet_summary_speech_free`**: read by REPORT's cover only (`nodes/figure.py:837`),
  print and nothing else. The enhanced equivalent is in exactly the same position.

So the residual YAMNet pass is **not** in the position diarization was in before it was narrowed.
Its scores have no reader, but its `summary_all` does, and that summary is computed from those
scores in the same call — the pass cannot be removed without removing a routing input. Nothing was
changed here; this is a finding, and the narrowing decision is the owner's.

The narrowing would in any case buy much less now: with the worker resident the marginal cost of the
residual pass is one request, not one process. Measured at 0.038 s for 21 s of audio.

There is no `streams` config key for classifiers. The six enhanced/residual classifier passes are
literal entries in the `blocks` list (`nodes/preprocess.py:3111-3116`), prefix baked into each
lambda, so narrowing one would be a code change, not a config change.
