# Persistent subprocess-venv workers

Status: implemented on `feat/task-events`; validated on ORCD (job 25503998); restarted as r30.

## Why

The r18d PREPROCESS array (job 25497192, cancelled) wrote 4,408 per-recording rows. A least-squares
fit over them (`seconds` against `duration_s`, 8 CPUs) gave

    wall ≈ 45.6 s + 5.0 s per audio second

with a median recording of 10 s, so the intercept was most of the cost. Per-call fits of the
driver's own timers (`node_seconds`) located it:

| call | intercept (s) | slope (s per audio s) |
| --- | ---: | ---: |
| Qwen3-ASR-1.7B (qwen-asr-cpu venv) | 21.9 | 1.21 |
| CrisperWhisper 2.0 turbo (crisperwhisper-cpu, CT2 float32) | 17.5 | 0.25 |
| FRCRN_SE_16K (clearvoice-cpu) | 5.0 | 1.72 |
| AST (in process) | 1.8 | 0.20 |
| pyannote diarization (in process) | -2.1 | 0.42 |
| HeAR, YAMNet (already persistent) | 0.4, 0.3 | 0.04, 0.01 |

The three subprocess-venv backends started a fresh interpreter for every call: import torch and the
model library, load the weights, decode one recording, exit. AST and pyannote were already cached
per process by `(model, revision, device)`; YAMNet and HeAR already had long-lived workers
(`specs/20260922-yamnet-process-startup-cost/`, `specs/20260922-hear-process-startup-cost/`).

## What changed

`senselab.utils.venv_worker.serve_in_venv` generalises the YAMNet/HeAR worker. A backend's worker
script defines `load(init)` and `handle(state, request)`; the first call for an identity starts the
interpreter and runs `load` once, every call sends one JSON request line and reads one reply line.

* **Identity.** The caller's tuple `(venv, model, revision, device, compute_type[, ...])`, plus the
  interpreter path and digests of the script, the init payload and the environment. Anything that
  `load` reads is in `init`, so a worker can never serve a request under weights or settings other
  than the ones it loaded. Per-call values (paths, language, decode strategy) are in the request.
* **Revision pinning is unchanged.** The SHA that reached the one-shot payload reaches `init`
  unchanged: Qwen's `model_revision`/`forced_aligner_revision`, CrisperWhisper's staged
  `snapshots/<sha>` directory, ClearVoice's staged checkpoint directory. The SHA is also in the
  identity, so a run that resolves a different commit gets a different worker.
  `revision_pinning_guard_test.py` passes; `hf_load_coverage_test.py` now counts `serve_in_venv(`
  as a subprocess launch so these files stay under review.
* **Reply channel.** The worker's fd 1 is pointed at its stderr before the backend script runs and
  replies go to a private dup of the original stdout, so library prints cannot corrupt a reply.
* **Failure.** A handler exception is returned as `{"error"}` and re-raised in the parent with the
  same type mapping as `parse_subprocess_result` (`ValueError`, `TypeError`, else `RuntimeError`);
  the worker stays loaded. A worker that dies or times out during a request is killed, the call
  raises, a `lost` event is recorded, and the next call starts a new worker and records `restart`.
  A worker found dead between calls records `died_idle` and is restarted before the request.
  `venv_worker_events()` returns the record; the r30 driver writes it into each row.
* **Shutdown.** `shutdown_venv_workers()` is registered with `atexit`.
* **Converted.** CrisperWhisper, Qwen3-ASR transcription, and the five audio-only ClearVoice
  checkpoints. Not converted: Qwen forced alignment (not in PREPROCESS), ClearVoice TSE (its
  upstream path writes tracks per video), ppgs and the rest.

## Thread settings

Every worker reports its settings once loaded (`venv_worker_stats()`). On ORCD with
`--cpus-per-task=8` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=8`, all three report
`torch.get_num_threads() == 8` and an affinity of 8 CPUs: no oversubscription and no
single-threading. See the FRCRN section below for its decode cost.

## Validation

See `validation.md` in this directory.
