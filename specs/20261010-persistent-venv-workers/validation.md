# Validation on ORCD

Job 25503998 (`pi_satra`, node2803, Intel Xeon Platinum 8462Y+, 8 CPUs, 48 GB, shared node at a load
average of 17–24, with probes 25504160, 25504383 and 25505119 on the same node for part of it). The
working tree of this change (branch tip d3f259fd plus the commits that add `venv_worker`) ran the
r18d driver, instrumented (`preprocess_driver_r30.py`), over 20 recordings from
`r18d_20261010/rerun_manifest.jsonl` that already had complete r18d rows, chosen at the 20 duration
quantiles (0.3 s to 119.6 s, 484 s in all). `SENSELAB_RESULT_CACHE=off`, so every model computed;
`SENSELAB_RUN_ID=slurm-25497192`, so every model resolved to the commit r18d used. Output went to
`/orcd/scratch/bcs/002/satra/r30_validate/run/`, never the staging tree.

## Outputs against r18d staging

`compare_r30.py` matched each recording's store against its r18d store:

| output | identical | differing |
| --- | --- | --- |
| Qwen3-ASR words, times, scores, transcript | 20/20 | 0 |
| CrisperWhisper words, times, scores, transcript, decode strategy | 20/20 | 0 |
| pyannote diarization (arrays and attributes) | 20/20 | 0 |
| `enhanced` stream (FLAC sha256) | 17/20 | 3, max abs 2.4e-7 |
| AST / YAMNet / HeAR score files on `plain` | 17/20, 17/20, 16/19 | ≤1.5e-6 |
| the same on `enhanced` and `residual` | 15–16 of 19–20 | ≤8.1e-5 |
| squim, span_yamnet, span_hear (matched by extent) | all spans in 17 recordings | the same 3 recordings, ≤2.4e-5 |

The three recordings that differ are exactly the three whose r18d rows ran on node3306, node3804 and
node4012. Their `plain`-stream YAMNet and HeAR scores differ as well, and neither YAMNet nor HeAR is
touched by this change (both already had persistent workers). On those nodes FRCRN runs at
0.4–0.9 s per audio second against 2.6–3.7 on node2803's class (per-host medians over the 4,408
r18d rows), so they are a different CPU class, and the differences are the kernel dispatch on that
class: on node2803 itself, capping oneDNN at AVX2 or disabling mkldnn changes FRCRN's output by
3.1e-7 and 2.2e-7 (probe 25505119), the same order as the 2.4e-7 seen here. Every recording whose
r18d row ran on node2803's class is identical in every model output listed above. One of them
(r18d on node2803 itself) has `enhanced_ast_scores` and `residual_hear_scores` files whose numbers
differ by 7e-16 and 4e-15, float64 summaries of identical inputs.

Measurements whose attributes differ for other reasons: `consensus_transcript`, `span_*` ids,
`squim`/`proximity`/`disruptions` `stream` references (entity ids are per run), and the products of
code that changed between r18d's commit 3f8997c9 and d3f259fd (`session_floor`, `background_model`,
`pii_ledger`, `quality_join`, `redaction_llm_annotation`, the new `speech_residual`).

## Load against compute

Worker start-up (spawn, imports, weights), from `venv_worker_stats()`: CrisperWhisper 3.3 s,
Qwen3-ASR with its aligner 7.7 s, FRCRN 3.2 s, MossFormer2_SS 3.1 s. Per-recording timings for the
same 20 recordings, r18d (one process per call) against this run (one load per process):

| call | r18d fit | this run |
| --- | --- | --- |
| Qwen3-ASR-1.7B | 16.4 s + 0.19 s/s | 0.8 s + 0.37 s/s |
| CrisperWhisper 2.0 turbo | 14.6 s + 0.06 s/s | 7.9 s + 0.12 s/s |
| FRCRN_SE_16K | 6.8 s + 2.62 s/s | 5.2 s + 2.97 s/s |
| AST | 1.4 s + 0.07 s/s | 0.8 s + 0.11 s/s |
| PREPROCESS | 46.1 s + 3.46 s/s | 19.0 s + 4.16 s/s |

The fixed cost the persistent workers remove is the one Qwen and CrisperWhisper paid per call. What
remains of CrisperWhisper's intercept is decoding, not loading: Whisper encodes a 30 s window for
any input, and a short recording of repeated syllables decodes for 15–20 s (probe 25504160 serves
eight recordings three times from one loaded worker, 4.1–23 s each, identical output every time).
The slopes rose on this run because the node was shared with the probes; they are upper bounds.

## Threads

All three workers report `torch.get_num_threads() == 8`, an 8-CPU affinity and
`OMP/MKL/OPENBLAS_NUM_THREADS=8`. Two findings, neither changed:

* **CrisperWhisper's CT2 engine runs at `intra_threads=4`.** `crisperwhisper.engine.CT2Engine`
  fixes it and `CrisperWhisperModel` does not pass it through. Probe 25504160 patched it to 8: output
  identical on all 8 recordings, time 0–23% lower (16.4 against 19.2 s on the slowest). Left at 4.
* **FRCRN is slow on this CPU class, not misconfigured.** On node2803 a 22.7 s recording decodes in
  75 s at 8 threads and 125 s at 4; flushing denormals changes nothing (probe 25504383). oneDNN at
  AVX512_CORE gives identical output at the same speed; AVX2 is slower; mkldnn off is 20% faster but
  changes the output by 2.2e-7. No setting both speeds it up and keeps its numerics.

## Resident memory

High-water marks after each recording (`/proc/<pid>/status` VmHWM): Qwen3-ASR worker 13.4 GB from its
load on, CrisperWhisper 4.1 GB, the driver 1.8–4.8 GB, FRCRN 0.7 GB growing with the longest input
seen (6.8 GB after 76 s). Their sum reached 29 GB at 76 s, before YAMNet and HeAR (about 0.8 GB
each). The one-shot path never held Qwen and FRCRN at once; this one does, so `--mem=24G` is too
small. r30 keeps the cancelled array's 40 GB.
The r30 driver also ends every persistent worker before and after a recording of 150 s or more
(242 of the 18,199 remaining, the longest 333 s), so FRCRN's activations for those never sit beside
Qwen3-ASR's and CrisperWhisper's weights; the rest stay under 40 GB by the sums above.

## Whole-recording timing

On the same 20 recordings, with SPEECH left out (its PII scan reads the result cache, which this run
switched off and r18d did not: the scan cost 12 s for every lexical recording here and 279 s for the
120 s one, against 1–4 s in r18d), least squares on `seconds` against `duration_s`:

    r18d:  46.2 s + 3.59 s per audio second   (2,663 s in all)
    r30:   18.8 s + 4.31 s per audio second   (2,464 s in all)

The intercept is the persistent workers; the slope is a shared node, and FRCRN's CPU class above all.
The PII scan (`text/tasks/pii_detection/subprocess_backend.py`) is a fourth per-call subprocess, not
converted here; it is reached only on a result-cache miss.
