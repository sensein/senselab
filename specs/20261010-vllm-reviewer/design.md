# The reviewer on a vLLM server, one request at a time (owner-approved, 2026-10-10)

## Why

Measured on one H100 over the 246 reviewer requests of `/orcd/scratch/bcs/002/satra/bench_reviewer_20261009`
(`bench.sbatch`, `bench.py`, `compare.py`, `venv_vllm.sh`; requests templated to token ids with the Gemma
tokenizer at 52f3f65b, greedy, 1,024 new tokens, the checkpoint's `generation_config.json` stop ids):

| arm | requests | calls/s | output tok/s | latency p50 | reproducible |
|---|---|---|---|---|---|
| transformers worker (`_ReviewWorker`), concurrency 1 | 242 of 246 | 0.049 | 17.6 | 15.4 s | -- |
| vLLM 0.31.0, concurrency 1 | 246 of 246 | 0.179 | 66.7 | 4.1 s | 246/246 identical token ids across two runs |
| vLLM 0.31.0, concurrency 32 | 246 | higher | higher | -- | no, with or without `VLLM_BATCH_INVARIANT` |

vLLM at concurrency 1 is 3.7x the transformers worker and exactly reproducible; batching is not, so a server
never takes more than one request at a time. Against the transformers worker vLLM is a different engine (33%
exact completions, all scalar labels agreeing on 92.6%), which is why the engine is part of a reading's identity.

The server: `google/gemma-4-31B-it-qat-w4a16-ct` at 52f3f65bc7a02d555763bc923bd1d9094898219d, vLLM 0.31.0 (torch
2.13, CUDA 13.0 runtime: a driver with CUDA >= 13 and `module load cuda/13.1.0`, ninja on PATH), `--max-model-len
12288 --gpu-memory-utilization 0.90 --enable-prefix-caching --kv-cache-dtype auto --generation-config vllm
--limit-mm-per-prompt '{"image":0,"audio":0}'`. At 0.90 it loaded 19.78 GiB of weights and kept a 56,753-token KV
cache (4.62 requests of 12,288 tokens).

## What was built

- **Engine** (`redaction.llm_check.engine`: `transformers` default, or `vllm`; `redaction.llm_check.vllm.*` the
  server arguments). `redaction_review_vllm.py` holds a worker on the same line protocol as the transformers
  worker: in the `pii-review-vllm` venv it loads the tokenizer from the staged `snapshots/<sha>`, starts one
  `vllm serve` over that directory, and per request applies the chat template the transformers worker applies,
  posts the token ids to `/v1/completions` (temperature 0, top_p 1, top_k -1, the checkpoint's eos ids as
  `stop_token_ids`, `max_tokens` = `max_new_tokens`, `return_token_ids`) and decodes the returned ids with
  `skip_special_tokens`, as the transformers worker decodes its own. The server must read exactly the prompt
  length sent. A prompt that with `max_new_tokens` exceeds `max_model_len` is an absent reading for that row
  and the worker carries on; any other error ends the worker and its server.
- **One request at a time.** One worker per process, one server per worker, and the host serialises calls; the
  identity says `concurrency: 1`.
- **Several servers per card.** Servers sharing a node's visible GPUs start one at a time under a node-local
  lock file (`$TMPDIR/senselab-vllm-start-<host>-<devices>.lock`), so each sees the others' memory when it sizes
  its KV cache. The server dies with its worker (process group, and `PR_SET_PDEATHSIG` on Linux).
- **Identity.** `engine_identity`: `name`, `version` (the pinned 0.31.0), `server_args` (the configured
  arguments, in a fixed order) and `concurrency`. It is in the result-cache key (`review_cache_key` params
  `engine`), on REVIEW's activity parameters, on every round's payload and on the annotation (`engine`, as the
  worker reported it). The worker refuses a server whose `/version` differs or whose snapshot is not the
  resolved commit. A reading that records no engine was the transformers worker's.
- **Venv.** `ensure_venv("pii-review-vllm", ["vllm==0.31.0"], python 3.12, max_cuda_version (13, 0))` from the
  committed lock (`venv_locks/pii-review-vllm.txt`: torch 2.13.0, torchaudio 2.11.0 from `cu130`; transformers
  5.17.0, compressed-tensors 0.17.0, flashinfer-python 0.7.0.post1 as in the benchmark venv). The PyTorch index map
  gained `cu130`; a venv that declares no ceiling now has the default ceiling (12, 8), so every other venv keeps
  routing to `cu128` on a CUDA-13 host. `triton` is installed with torch from its index wherever torch pulls it in.
- **Resume.** `extend_llm_review.py` counts a store as `present` only where its REVIEW annotation is under the
  current prompt version and on the configured engine; anything else is read again and retired. A requeued
  task therefore resumes per row.
- **Server logs.** `SENSELAB_VLLM_LOG_DIR`, where set, receives each server's output (mode 600); vLLM logs no
  request content without `--enable-log-requests`, which is not passed.

## Several servers on one H100

Three servers at `gpu_memory_utilization` 0.30 hold 23.9 GiB each, against 19.78 GiB of weights alone, so the KV
cache cannot hold one 12,288-token request unless activations and graphs fit in about 3 GiB; two at 0.45 leave
about 12 GiB of KV (roughly 15,000 tokens at the per-token size the 0.90 run implies). The smoke test below tries
three and falls back to two; the corpus template takes `SERVERS` and sets 0.90 / `SERVERS` (floored to 0.01).

## Jobs (this directory)

- `review_vllm.sbatch` -- the corpus re-run: one H100 per array task, `SERVERS` client processes each owning one
  server and one shard (`--slice-index task*SERVERS+i --slice-count SLICES*SERVERS`), `--requeue`.
- `second_opinion.sbatch` -- question set 6 over the same rows (clef:27b on the pinned Ollama, `--workers` rows
  at once per server, one server per array task; the array spreads the rows over GPUs).
- `smoke.sbatch` and `smoke_report.py` -- the GPU smoke test.

## Smoke test

Pending: job, results and throughput are recorded here when it finishes.
