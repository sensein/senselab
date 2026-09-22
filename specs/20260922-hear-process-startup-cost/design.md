# HeAR's cost is starting a process too, and it is the larger instance

`specs/20260922-yamnet-process-startup-cost/design.md` measured YAMNet's per-invocation cost,
applied a resident worker, and named HeAR as the larger target it did not touch: four subprocess
invocations of a second TensorFlow model per recording, **17.8% of the whole graph against YAMNet's
12.5%**. This is that target, measured the same way and fixed the same way.

The numbers did not transfer. HeAR's fixed cost is 35% higher, its model load is 82% larger, and its
marginal term is real rather than negligible — 0.0067 s per 2 s window against YAMNet's 0.0009 s of
inference per second of audio. The invocation count differs too: HeAR's is a constant 4 where
YAMNet's was 3 or 4.

## How these figures were taken

`mit_preemptable`, `--exclusive --nodes=1`, so the node ran nothing else. `loadavg_start` was 0.01
on node2027 (cost probe) and is echoed by every job script. The corpus run itself was on contended
nodes at load averages in the hundreds, so its absolute seconds are not these seconds.

| script | what it measures | job |
| --- | --- | --- |
| `probe_hear_cost.py` | fixed vs marginal cost of one invocation, both SavedModels | 23466225 |
| `probe_equivalence.py` | old path against new, same recordings | 23466540 |
| `../20260922-yamnet-process-startup-cost/probe_preprocess_steps.py` | the invocation census | 23464900 |

The invocation census is the YAMNet work's own step probe re-read rather than re-run. Its
`subprocess.run` wrapper sits on the shared `subprocess` module, so it recorded **every** venv
backend's call with the PREPROCESS step that was registered when it was made, HeAR's included. No
second job was needed and none was run; the rows below are job 23464900's, filtered to the four
steps whose name is `hear` or ends in `_hear`.

## The split: 6.48 s fixed, 0.0067 s per 2 s window

`probe_hear_cost.py` runs the shipped worker script unchanged except for per-phase timers, over
band-limited noise at ten durations from 2 s to 240 s, three repeats each, one audio per
invocation, on the event detector — the SavedModel all four PREPROCESS passes use. Parent-side wall
clock around the subprocess:

```
wall_s = 6.482 + 0.0026 * audio_seconds     (r2 on the duration term 0.177, n=30)
```

The duration term is small enough that process start-up noise dominates the fit, which is why the
marginal cost is better read off the model's own work, where it is clean:

```
infer_s = 0.254 + 0.00669 * n_windows       (r2 = 0.998, n=30)
```

**0.0067 s per 2 s window**, plus 0.254 s for the first window of a process, which is the graph's
first trace and is paid once per process rather than once per window. At the config's 2.0 s
non-overlapping hop that is 0.0033 s per second of audio.

Per-phase medians, seconds:

| audio s | windows | wall | interpreter + TF import | SavedModel load | signature | read | inference | first window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 6.27 | 1.87 | 3.39 | 0.00 | 0.000 | 0.263 | 0.263 |
| 5 | 3 | 6.55 | 2.17 | 3.37 | 0.00 | 0.001 | 0.277 | 0.263 |
| 21 | 11 | 6.34 | 1.90 | 3.37 | 0.00 | 0.002 | 0.332 | 0.263 |
| 30 | 15 | 6.53 | 2.01 | 3.35 | 0.00 | 0.003 | 0.356 | 0.261 |
| 120 | 60 | 6.74 | 1.91 | 3.37 | 0.00 | 0.005 | 0.658 | 0.262 |
| 240 | 120 | 7.29 | 2.11 | 3.38 | 0.00 | 0.008 | 1.062 | 0.263 |

The 6.48 s decomposes as 1.9–2.1 s interpreter start plus TensorFlow import, **3.37 s
`tf.saved_model.load`**, 0.00 s signature probe, 0.25 s first-call trace, and ~1.0 s of spawn, pipe
and parent-side overhead.

**The intercept does split differently from YAMNet's, as the brief suspected.** The import half is
the same (2.0 s both), because it is the same TensorFlow. The load half is not: 3.37 s against
YAMNet's 1.85 s TF-Hub load, so the detector's SavedModel is 1.8x the load and the total fixed cost
is 1.35x.

**At the corpus median recording of 7.3 s, 95.7% of an invocation is start-up** — high, but not
YAMNet's 99.9%: inference is 0.28 s of a 6.50 s call rather than 0.006 s of 4.81 s. The marginal
term is small; it is not zero, and at 240 s it is 1.06 s of a 7.29 s call.

Two parent-side costs the one-shot path paid per invocation and the resident path pays once were
measured and are **not** part of the 6.48 s: `ensure_venv` 0.003–0.004 s and `stage_hear_snapshot`
0.001–0.005 s. Hoisting them buys nothing measurable. `ensure_venv` is hoisted anyway, because
touching the venv lock once per process rather than once per call is worth having for its own sake.

A second audio inside a process that has paid the fixed cost is nearly free — 21 s of audio for
0.068 s:

```
wall_s = 6.282 + 0.0680 * n_audios_of_21s
```

so 16 recordings in one invocation cost 7.30 s, or 0.46 s each.

### The encoder is a different shape, and the graph does not use it

`run_hear` also serves the 512-d encoder, which no PREPROCESS pass calls; `hear_scores`,
`enhanced_hear`, `residual_hear` and `span_hear` all go through `detect_health_acoustic_events`,
i.e. the detector. Measured anyway, because the resident worker now holds both:

| model | TF import | SavedModel load | first window | per window |
| --- | --- | --- | --- | --- |
| event detector (MobileNetV3-Large) | 1.9 s | 3.37 s | 0.263 s | 0.0067 s |
| encoder (ViT-L) | 1.9 s | 1.84 s | 1.22 s | 0.106 s |

The encoder loads *faster* and infers 16x slower. For it the fixed cost is not the whole story —
`wall_s = 6.92 + 0.0399 * audio_seconds`, so a 120 s recording is 11.9 s of which 7.4 s is the
model working. Residency still removes the same ~5.7 s of start-up, it is just a smaller share of
what an encoder call costs.

## Four invocations per recording, and it is a constant

Job 23464900, exclusive node, nine recordings from 2.8 s to 73.5 s:

| call site | block | batch |
| --- | --- | --- |
| `_hear_scores` | `hear` | the plain stream, 1 audio |
| `_stream_hear("enhanced")` | `enhanced_hear` | the enhanced stream, 1 audio |
| `_stream_hear("residual")` | `residual_hear` | the residual stream, 1 audio |
| `_span_hear` | `span_hear` | every span, **already batched into one call** |

| recording s | invocations | HeAR s | share of the whole graph | per call: plain / enhanced / residual / span |
| --- | --- | --- | --- | --- |
| 2.79 | 4 | 27.7 | 14.2% | 8.38 / 6.55 / 6.25 / 6.49 |
| 3.72 | 4 | 25.8 | 31.2% | 6.73 / 6.49 / 6.29 / 6.28 |
| 4.54 | 4 | 25.8 | 22.5% | 6.47 / 6.57 / 6.45 / 6.31 |
| 5.58 | 4 | 25.9 | 21.6% | 6.58 / 6.35 / 6.37 / 6.59 |
| 7.29 | 4 | 25.6 | 22.3% | 6.32 / 6.47 / 6.43 / 6.38 |
| 10.18 | 4 | 26.1 | 22.3% | 6.70 / 6.44 / 6.59 / 6.38 |
| 18.32 | 4 | 26.1 | 25.3% | 6.65 / 6.56 / 6.44 / 6.43 |
| 30.02 | 4 | 25.8 | 14.2% | 6.41 / 6.75 / 6.21 / 6.44 |
| 73.54 | 4 | 27.6 | 9.2% | 6.88 / 6.85 / 6.35 / 7.50 |

Median 4 invocations and 25.9 s, mean 26.3 s, **17.8% of the whole graph** over these nine — of
which about 1.3 s is inference. The 2.79 s recording's first call is 8.38 s because it is the first
HeAR invocation of the whole run and pays a cold page cache for the SavedModel.

The two things the brief said to check rather than inherit:

**`_span_hear` does not spawn a process per span.** Same as YAMNet: `_classify_spans_in_batch`
(`nodes/preprocess.py:291`) hands every prepared span to one `detect_health_acoustic_events` call
and only falls back to one call per span when the batch as a whole fails. One invocation per
recording, measured, at every length.

**The count is 4, not "3 or 4".** This is where HeAR and YAMNet diverge. YAMNet's `span_yamnet`
made no call at all on the 4.54 s and 7.29 s recordings, because every span there was shorter than
its 0.96 s native frame and `native_prepared` came out empty. HeAR has no such path:
`span_hear_input` places a short span in a silent 2 s buffer (`hear.py`'s `span_to_hear_buffer`),
so every span always yields something to classify and the batch is never empty. Four invocations on
all nine recordings, min 4, max 4.

## What the fix is

The same three levels were available and the same one was chosen, for the same reason: once the
worker is a process that answers requests rather than one that exits after answering, how long it
lives is a policy question. It is expressed as exactly that — the worker is resident by
construction, and `hear.keep_worker_resident` decides whether PREPROCESS ends it when the recording
does.

| | per recording | share of the graph, measured |
| --- | --- | --- |
| HeAR today | 25.6–27.7 s, median 25.9 | 17.8% |
| resident within a recording | ~6.5 s | ~4.5% |
| resident across recordings in a task | ~0.3 s | ~0.2% |

## Is a shared helper warranted now that there are three?

**No, and the third case sharpens the reason rather than weakening it.** The YAMNet work rejected a
helper on two grounds — lifetime policy and error contract — and said a helper would have to
parameterise exactly the thing each caller decides differently. With three instances in view:

| | `redaction_review` | `yamnet` | `hear` |
| --- | --- | --- | --- |
| default lifetime | `false` | `true` | `true` |
| why | 70.35 GiB CUDA, took 19/22 recordings out of a run | 0.7–1.2 GiB host RAM | host RAM, same argument |
| error contract | everything → `ReviewWorkerError` | reconstruct `ValueError`/`TypeError`, else `RuntimeError`; **worker ends** | reconstruct the same two; **worker survives** |
| models per worker | one | one | **many, lazily, keyed by directory** |
| environment | fixed | fixed | **keyed by device; a device change replaces the worker** |
| payload | JSON in, JSON out | JSON in, JSON out | **JSON in, `.npy` files out** |

The two original grounds still hold, and the third instance adds three more. Every one of them is
in the same place a helper would have to own:

1. **The error contract is not just "which exception type" but "does the worker die".** HeAR
   answers differently from YAMNet, and the difference is measured rather than stylistic:
   `_classify_spans_in_batch` retries a failed batch **one span at a time**, so a fatal-on-error
   contract costs one 3.37 s model load per span of a failing batch. YAMNet's `_span_yamnet` has
   the same retry loop, but its reload is 1.85 s and its batches are smaller; HeAR's is the case
   where it bites. The worker therefore distinguishes a request-level failure (reply the error,
   keep the models) from one outside the request loop (`"fatal": True`, exit). A shared helper
   would have to own that distinction and both callers' answers to it.
2. **HeAR holds a map of models, not a model.** `run_hear` is parameterised by `subdir` — encoder,
   detector-large, detector-small — so the worker loads lazily and keys by directory. YAMNet's
   worker loads one thing at start-up and reports `ready` when it has it. "What does `ready` mean"
   is a different question for the two.
3. **HeAR's worker identity includes the device.** `CUDA_VISIBLE_DEVICES` is fixed when the process
   starts, so a request for another device replaces the worker. YAMNet has no device parameter.

What the three do share is the mechanical shell: `Popen` over the venv interpreter, `sys.stdout`
redirected to stderr before the heavy import, a marker-prefixed JSON line protocol, two daemon pump
threads, a module-global under one lock, `atexit` teardown, a worker-side exit on EOF. That is
about 120 lines and it is the part with no decisions in it. A helper that extracted only that shell
would be worth having; a helper that also owned lifetime, readiness, error policy, model identity
and environment identity would have five parameters, one per caller-specific decision, and each
caller would pass a different value for every one. The recommendation is therefore: **not yet, and
the thing to extract when a fourth arrives is the transport, not the policy.** Recorded here rather
than acted on, because extracting the transport touches `redaction_review` and YAMNet as well, and
that is a change of its own with its own equivalence to prove.

The failure mode the brief warned about — a dead worker blocking others — cannot arise here. The
lock is a `threading.Lock` inside one process, the worker is checked with `poll()` before reuse and
replaced when dead, every request and every start has a wall-clock ceiling after which the worker
is killed, and a killed or crashed parent releases the child through the worker's own EOF exit as
well as `atexit`. `ensure_venv` is called once per process instead of once per call, so the venv
lock is touched strictly less often than before.

## Equivalence

`probe_equivalence.py` loads the pre-change module out of git at the merge-base,
`fb3cb7a2b974721f3e17c35c369b8413e39d655f`, as its own module — the shipped code, not a description
of it — and runs both paths over the same nine recordings at the config's 2.0 s hop, on the
detector, plus three of them on the encoder, plus an alternation that puts the detector back
through a worker that has since loaded the encoder.

**Exact equality is the accepted tolerance, not a numeric one.** The comparison is
`a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()` — bitwise, on the raw
float32. Three things make that the right bar rather than an optimistic one:

- Both paths run the same TensorFlow graph over the same float32 windows in the same interpreter
  build, differing only in whether the process exits afterwards.
- The arrays never pass through JSON. Both paths have the worker `np.save` to a path the parent
  named and the parent `np.load` it back, so there is no float formatting step in which a
  difference could be introduced or hidden.
- The windows are cut from the same `.wav`, written by the same `write_hear_wav` at FLOAT subtype,
  by the same `np.stack` of the same slices.

Any difference at all would therefore be a defect — a model loaded twice answering differently, or
a request crossed with another — rather than a rounding artefact. The probe reports the worst
absolute difference alongside the verdict, so a non-bitwise result would be quantified rather than
merely failing.

Results: job 23466540, pending at the time of writing.

## What was not measured

- **The fixed cost on a loaded node.** Every figure here is from an idle exclusive node. Process
  start-up is import- and filesystem-bound, so on the corpus run's nodes 6.48 s is a floor and the
  saving is correspondingly a floor.
- **Resident-worker RSS as its own probe.** YAMNet had `probe_worker_rss.py`; here the equivalence
  run's `worker_rss_mib` column is the only memory measurement, taken from `/proc` of the live
  worker as it grows across the nine recordings. No separate sweep of a single worker against
  recording length was taken, and no measurement was taken of a worker holding both SavedModels
  after many requests.
- **The GPU path.** Everything here is `CUDA_VISIBLE_DEVICES=-1`. Whether a CUDA-built worker's
  residency is as cheap as the CPU one's is not known, and the LLM check is the standing warning
  that it may not be.
- **`gammatone` at 7.40 s and `clip_consistency` at 10.96 s**, both of which the step probe shows
  making subprocess calls of ~5–11 s. Still not investigated; the two remaining named targets.

## Who reads each of HeAR's four passes

Traced against the code, as the YAMNet work traced `residual_yamnet`. **Nothing was removed — this
is a finding, and the narrowing decision is the owner's.**

| product | reader? | where |
| --- | --- | --- |
| `hear_scores` (+ its `derivatives/` sidecar) | **read, load-bearing live** | AIRWAY's coverage mode `nodes/airway.py:336, 933, 947, 1138`; TAXONOMY `nodes/taxonomy.py:70` over `SUMMARISED_CLASSIFIERS` → `hear_label_summary` → REPORT cover `nodes/figure.py:618` and sweep peaks `routing_analysis/features.py:349-351` |
| `hear_window` (per window) | read, REPORT only | `nodes/report.py:570, 630, 673` — raster, JSON, label counts |
| `hear_windows` (pooled) | read, presentation only | `nodes/report.py:512, 530, 1063, 1651`. Its `windows_by_label`, `labels` and threshold attributes have **no reader** |
| `enhanced_hear_scores` | **no reader** | — |
| `residual_hear_scores` | **no reader** | — |
| `enhanced_hear_summary_all` | read — cover print and the offline sweep; **no live gate** | `nodes/figure.py:796`; `routing_analysis/features.py:352, 769-775` → `{speech,airway}.hear_peak.enhanced`, profiled in `data/detector_profile/2026-09-12.parquet`, swept by `routing_analysis/report.py:905, 978` |
| `residual_hear_summary_all` | read — as above, plus label prevalence; **no live gate** | as above, plus `routing_analysis/report.py:732-744` (`residual` is in `LABEL_PREVALENCE_STREAMS`) |
| `enhanced_hear_summary_speech_free` | read — cover print only | `nodes/figure.py:837` via `:879-880` |
| `residual_hear_summary_speech_free` | read — cover print only | `nodes/figure.py:837` via `:879-880` |
| `span_hear` (+ its `unmeasured` assertion) | **read, load-bearing live, the most widely read of the four** | AIRWAY `nodes/airway.py:222` → `nodes/branches.py:2186-2213` at `:420, 856`; TAXONOMY `nodes/taxonomy.py:98`; `routing_analysis/features.py:55, 746-755, 1138-1145`; FIGURE `nodes/figure.py:1981`; REPORT `nodes/report.py:601, 657`; EXTEND `extend.py:364` |

Read against the question the YAMNet trace was asking:

- The **plain** `hear_scores` pass and the **`span_hear`** pass are both load-bearing in the live
  graph. AIRWAY reads both; TAXONOMY's consensus reads `span_hear`. Neither can be narrowed.
- The **`enhanced_hear`** and **`residual_hear`** passes are in a *weaker* position than
  `residual_yamnet` was. `residual_yamnet` survived its trace because
  `residual_yamnet_summary_all` feeds the emptiness bypass through
  `taxonomy.ruleset.emptiness.peak_streams` (`data/config/default.yaml:436`, which lists
  `enhanced|yamnet` and `residual|yamnet` and nothing else). HeAR's two stream summaries have no
  such reader: `extract_features` does parse them on every recording, but nothing consumes the
  result. What would go dark if those two passes were removed is (a) two lines of REPORT cover
  print, (b) four profiled offline sweep candidates — `{speech,airway}.hear_peak.{enhanced,residual}`
  — and (c) the `residual|hear|*` rows of the label-prevalence table. **No live routing decision
  changes.**
- As with YAMNet there is no config switch. The six stream×classifier passes are literal entries in
  the `blocks` list (`nodes/preprocess.py:3111-3122`) with the prefix baked into each lambda, so
  narrowing one is a code change, not a config change.
- And, as with YAMNet, the narrowing buys much less now than it would have: with the worker
  resident the marginal cost of one stream pass is one request, measured at 0.068 s for 21 s of
  audio, not a 6.48 s process.
