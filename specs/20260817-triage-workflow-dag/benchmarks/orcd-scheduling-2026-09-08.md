# ORCD scheduling: fixed vs marginal cost, GPU vs CPU, and what a persistent worker buys

**Status: PARTIAL.** Both jobs hit the 28-minute wall-clock limit before the harness finished its
sweep. Everything below is recomputed from the raw logs by
[`scripts/orcd_scheduling_2026_09_08.py`](scripts/orcd_scheduling_2026_09_08.py); run it yourself
against the two `.out` files to reproduce every number in this document. n=2 per data point
throughout — ranges are given, not point estimates, and no missing measurement is estimated.

Jobs: `gpu-22213820.out` (node3805, A100 80GB PCIe, driver 590.48.01, `cuda_available: true`) and
`cpu-22213836.out` (node2803, 8 `nproc`, `cuda_available: false`). Same commit
(`b9d1b4362f81bcdec0d39ee911ffe7b1f043be7f`), same 8.0 s clip, same harness, same declared backend
list (`yamnet,hear,crisperwhisper,squim`). Both started ~12:52–12:53 and were cancelled by Slurm at
`13:20:51` for exceeding the time limit — a 28-minute budget, confirmed from the log timestamps.

## 1. What this run does not answer

- **GPU memory footprint and SM utilisation.** Nothing in either log records device memory or
  occupancy. The "is 80 GB the right card" question is unanswered by this run; §3(b) below is
  the strongest inference available in its absence, not a substitute for it.
- **`squim` and `clearvoice`.** `squim` is named in both jobs' declared backend list and never
  reached before timeout. `clearvoice` does not appear in the declared backend list at all — it
  was not part of this benchmark's scope, so its absence here says nothing about it.
- **Batch sizes above 16 on CPU.** The CPU job's own meta record declares `batch_sizes: "1,4,16"`
  — 64 was never attempted on CPU, unlike GPU. Any CPU number at batch 64 in this document does
  not exist; none is given.
- **Queue-wait time.** Both logs start from inside the job (`=== host: ... ===`), after Slurm has
  already granted the allocation. Time spent queued is not recorded here.
- **Pre-emptible / restart behaviour.** Neither job was pre-empted; nothing here speaks to how a
  worker would resume after an eviction.
- **`crisperwhisper`, fully.** Neither job produced a single successful `full_call` for it (see
  §4). Its cost profile at any batch size is unmeasured.

## 2. Fixed vs marginal cost, recomputed

Method: ordinary least squares, `seconds = fixed + marginal × batch_size`, over every raw
`full_call` point (not pre-averaged), per backend per host. yamnet's cost curve is nearly linear
and single-source-of-noise-free enough that the fit is close to insensitive to which points feed
it; hear's is not (see the note below the table).

| backend | host | fixed (s) | marginal (s/item) | fixed % of predicted batch-1 call | fixed % of predicted batch-64 call |
|---|---|---|---|---|---|
| yamnet | GPU (A100) | 7.29 | 0.026 | 99.6% | 81.4% |
| hear | GPU (A100) | 8.3–9.9 | 1.39–1.42 | 88–98% | 8–10% |
| yamnet | CPU | not extractable | not extractable | — | — |
| hear | CPU | ~12.6 (order-of-magnitude only) | ~0.47 (order-of-magnitude only) | ~96% | ~63%* |

\* CPU numbers stop at batch 16 (see §1); "batch-64" column for CPU hear is the fitted line's
extrapolation past the last measured point, shown only for symmetry with the GPU row — treat it as
unverified.

**yamnet (GPU): confirms the original estimate, refines it slightly.** Raw points: batch 1 →
7.193/7.350 s, batch 4 → 7.426/7.429 s, batch 16 → 7.686/7.752 s, batch 64 → 8.897/8.997 s. OLS
gives **fixed = 7.29 s, marginal = 0.026 s/item** (vs. the quoted 7.24 s / 0.027 s/item — the same
curve to within the precision n=2 supports). Fixed cost is 99.6% of a batch-1 call and **81.4%** of
a batch-64 call, matching "~81%" and "~99%" as stated. Every batch size has ≤2.1% run-to-run
spread, so this fit is stable regardless of which points anchor it — a two-point fit through just
batch 16 and 64 gives 7.31 s / 0.026 s/item, effectively the same line.

**hear (GPU): the quoted fit is real, but it depends on excluding batch 1, and that exclusion is
defensible.** Raw points: batch 1 → 16.595/10.178 s, batch 4 → 14.334/13.938 s, batch 16 →
31.128/31.030 s, batch 64 → 99.214/99.544 s. Full four-point OLS gives **fixed = 9.89 s, marginal =
1.393 s/item** → fixed cost is 10.0% of a batch-64 call. A two-point fit through only batch 16 and
64 — the two batch sizes where run-to-run spread is ≤0.3% — gives **fixed = 8.31 s, marginal =
1.423 s/item** → 8.0% at batch 64. That second fit is what produced the quoted "~8.31 s fixed,
~8%": it is not wrong, but it is a fit that deliberately drops the noisiest point. **Batch 1's two
calls differ by 63%** (10.178 s to 16.595 s) — an order of magnitude more scatter than any other
hear batch size — so including or excluding it swings the intercept by ~1.6 s and the batch-64
fixed-fraction by 2 points. Report 8–10%, not a single number.

**What batch 1's scatter is telling you.** Within batch 1, `call_index 0` (16.595 s, the very first
`full_call` after `interpreter_plus_import`) is 63% slower than `call_index 1` (10.178 s) at the
*same* batch size — something amortises between the first and second call that batch size alone
does not explain. yamnet shows no such gap at batch 1 (7.193 s → 7.350 s, +2.1%, if anything
slightly slower the second time). This is circumstantial, not measured directly, but it lines up
with `interpreter_plus_import` being a separate, larger cost for hear-adjacent work than for
yamnet (hear GPU: 5.22 s; yamnet GPU: 5.75 s — comparable in isolation, so the batch-1 gap is not
import-driven) and instead looks like a graph/kernel warm-up internal to HeAR's first inference
call. A persistent HeAR worker would pay this once per process lifetime rather than once per call,
which is a stronger argument for a persistent worker than the 8–10% steady-state figure alone
suggests — flagged as an inference, not re-measured here.

**CPU: no reliable fit exists at all for yamnet, and hear's is order-of-magnitude only.** yamnet
CPU raw points *decrease* with batch size — batch 1: 9.679/7.900 s, batch 4: 8.980/6.199 s, batch
16: 7.293/6.747 s — giving OLS **fixed = 8.48 s, marginal = −0.098 s/item**. A negative marginal
cost is not physical; it means call-to-call noise on this CPU node (up to 45% spread within a
single batch size, e.g. batch 4's 8.980 vs. 6.199 s) dominates whatever batch-size signal exists,
and n=2 cannot separate them. hear CPU (batch 1: 20.548/9.131 s, batch 4: 12.356/12.050 s, batch
16: 20.822/20.314 s) gives fixed = 12.57 s, marginal = 0.472 s/item, but batch 1 alone spans a
2.25× range (9.13 s to 20.55 s) — treat this fit as showing hear is cheaper per-item on CPU than on
GPU (see §3c) and nothing more precise.

## 3. Warm `ensure_venv` is confirmed free

yamnet: 0.0108 s (GPU) / 0.0141 s (CPU). hear: 0.0089 s (GPU) / 0.0214 s (CPU). Both backends, both
hosts, both under 22 ms. Confirmed as stated — this cost is not on the table for any optimisation.

## 4. `crisperwhisper`: both "warm" figures are mislabelled cold builds, and one is worse than that

Both jobs' `ensure_venv_warm` records for `crisperwhisper` are immediately preceded in the log by
`Creating isolated venv 'crisperwhisper'` — the defining signature of a cold build, not a warm-path
check. **Confirmed: discard 794.1097 s (GPU) and 579.6854 s (CPU) as "warm" figures.**

They are not even the same kind of number, though:

- **CPU (579.69 s reported) is a clean cold build.** The log's own timestamps —
  `Creating isolated venv` at 12:57:55.282, `Venv 'crisperwhisper' ready` at 13:07:34.926 — span
  579.64 s, matching the reported figure to within 0.05 s. This is 579.69 s of venv construction,
  nothing else.
- **GPU (794.11 s reported) is a cold build *plus* lock-contention wait, and they should not be
  reported as one number.** The log shows the actual build (`Creating isolated venv` at 13:07:35.249
  to `Venv ... ready` at 13:13:37.897) took **362.6 s** — roughly 400–430 s less than the reported
  794.11 s. The gap is a `Stale lock ... heartbeat is infs old (> stale_after=120.0s)` wait that
  precedes the build: the GPU job's `crisperwhisper` request found a lock apparently held by the
  CPU job (`host=node2803`) and had to detect it as stale before proceeding.

**The stale-lock defect recurred in this run, and it recurred on both sides.** After the GPU job's
build finished (13:13:37) and the CPU job later tried to reuse the venv, the CPU log shows the same
message pointed the other way: `Stale lock ... Previous holder was user=satra host=node3805
pid=2763785. Breaking lock and taking over.` at 13:13:37.937 — followed by *rebuilding the venv a
second time from scratch* (`Creating isolated venv` again, ready at 13:18:04.057, another 266 s).
The GPU job then also tried to rebuild a second time at 13:18:04.072 and was killed by the time
limit mid-build. Two independent nodes, sharing `$HOME/.cache/senselab/venvs/`, each spent the run
convinced the other's build was dead and tore it down — this is the same `heartbeat is infs old`
symptom flagged as a suspected defect earlier, now observed a second time, on a fresh run, still
unfixed. `inf` is not what a dead process's last heartbeat age should read; it reads like the
heartbeat write path itself is failing, not like a genuinely stale holder.

**The collision left a broken venv, not just a slow one.** The CPU job's one `full_call` attempt at
`crisperwhisper` (batch 1, after the venv had been fought over and rebuilt twice) ran for 628.7 s
and then failed:

```
RuntimeError: Importing the numpy C-extensions failed ...
libscipy_openblas64_-f48b354e.so: cannot open shared object file: No such file or directory
```

A `.so` referenced by an installed wheel is missing — the signature of an install that two
processes wrote into concurrently, not of a normal cold build. The GPU job never got a `full_call`
data point at all: it was still mid-rebuild when Slurm killed it at 13:20:51. **`crisperwhisper`
therefore has zero valid `full_call` measurements from this run, at any batch size, on either
host** — not "expensive," genuinely unmeasured, and its one attempt demonstrates the shared-venv
race actively corrupts state, not merely delays it.

## 5. CPU vs GPU: confirmed, with the batch-1 caveat carried over

| backend | batch | GPU (s) | CPU (s) | GPU/CPU ratio |
|---|---|---|---|---|
| yamnet | 16 | 7.686 / 7.752 (mean 7.72) | 7.293 / 6.747 (mean 7.02) | GPU ~1.10× slower |
| hear | 16 | 31.128 / 31.030 (mean 31.08) | 20.822 / 20.314 (mean 20.57) | GPU ~1.51× slower |

Confirmed at batch 16 (the only batch size both jobs completed for both backends): neither backend
benefits from the GPU, and HeAR is faster on CPU, by roughly half again. Do not extend this to
batch 1 — hear's batch-1 numbers (GPU 10.18–16.60 s vs. CPU 9.13–20.55 s) overlap each other's
entire range, so no CPU/GPU comparison at batch 1 is defensible from this data.

## 6. The owner's four questions

**(a) Would a pub-sub / batch architecture help?** For yamnet, yes, sharply: at batch 64 on the
GPU, 81% of a 8.95 s average call (7.29 s) is fixed overhead that a shared queue would pay once
across an entire batch of requests rather than once per request. At today's batch-1 traffic
pattern, yamnet spends 99.6% of every call paying for the same thing 64 times. For hear the
efficiency gain from batching is much smaller in steady state (8–10% overhead at batch 64) — the
`marginal` cost genuinely dominates (~1.4 s/item) because HeAR does real per-item work. hear's
larger win from an architecture change is not batching, it's amortising the ~63%-of-a-call
first-call warm-up (§2) across a process lifetime instead of paying it on every cold invocation —
which is a **persistent-worker** argument, not strictly a pub-sub/batching one.

**(b) Are we using the GPU's memory, or would a smaller card do?** Unanswered directly — no memory
or SM utilisation was recorded (§1). The strongest available inference: yamnet and hear are both
*slower* on the A100 than on an 8-core CPU node at the one batch size both were measured on (§5),
which is inconsistent with either backend meaningfully using an 80 GB accelerator — a CPU-only
workload cannot be memory-bound on a GPU it runs slower without. This is circumstantial, not a
memory measurement; the actual question needs `nvidia-smi`/NVML sampling during a run, which this
benchmark did not do.

**(c) Which work is better on CPU?** By the one comparable data point (batch 16): both yamnet and
hear ran faster on the 8-core CPU node than on the A100 — yamnet by ~10%, hear by ~34%. Nothing in
this run shows a backend that is *better* on GPU than CPU; `crisperwhisper` and `squim`, the two
backends most likely to actually need one, produced no usable numbers on either host (§1, §4).

**(d) What should change?** See recommendations below — ordered by measured payoff, all tied to a
number already given in this document.

## 7. Recommendations, ordered by measured payoff

1. **Give yamnet a persistent worker before anything else.** 7.29 s fixed against 0.026 s/item
   (§2) is the largest fixed-to-marginal ratio measured in this run by a wide margin — a worker
   that loads the model once and serves calls over IPC eliminates a cost that is currently 81–99.6%
   of every call. This is the single highest-measured-payoff change available.
2. **Stop yamnet and hear from holding a GPU allocation.** Both ran slower on GPU than on an
   8-core CPU node at the only batch size both completed (§5): yamnet 1.10× slower, hear 1.51×
   slower. Moving them to CPU-only workers frees GPU allocation for backends that might actually
   need it (unmeasured here — §1) without a measured cost to yamnet or hear.
3. **Move venvs and their package cache off personal scratch and off `$HOME`, onto group scratch,
   built once.** The personal-scratch attempt died with a disk quota exceeded partway through a
   fourth venv (28 G `uv-cache` alone); moving to `$HOME/.cache/senselab/venvs/` traded that
   failure for the stale-lock collision this run reproduced a second time (§4), which this time
   corrupted a venv rather than merely delaying it. Neither location is safe for concurrent jobs;
   a single build on shared group scratch, referenced read-only by every job, removes the
   collision surface entirely rather than tuning the lock's stale-detection window.
4. **A persistent worker per backend still pays off for hear, just less dramatically, and mostly
   by removing the first-call warm-up rather than by batching.** At batch 64 steady state, fixed
   cost is only 8–10% of a call (§2) — batching gains are real but modest. The larger opportunity
   is the ~63%-of-a-call gap between hear's first and second invocation at batch 1 (§2), which
   looks like a one-time warm-up that a persistent process would pay once instead of on every cold
   call — flagged as an inference from the data, not itself separately measured.
5. **Fix the heartbeat, not the timeout.** The `stale_after=120.0s` detection did eventually fire
   in both directions in this run, so the mechanism functions — but an `inf`-aged heartbeat (§4) on
   a process that had not in fact died is the earlier-flagged suspected defect, observed again here
   independently. It should be root-caused before any change that adds more concurrent venv
   consumers.

## 8. The constraint that bounds any architecture

Subprocess venvs exist because backends resolve incompatible torch/CUDA toolchains — see
`CLAUDE.md`, "CUDA host configuration": each of `nemo-canary-qwen`, `nemo`, and `qwen-asr` installs
its own `torch` + `torchaudio` into an isolated venv because the host's system CUDA can be newer
than the PyTorch default-wheels CUDA, and a shared environment would force every backend onto one
resolved toolchain regardless of what each needs. **A persistent worker per backend venv is
viable — a single shared worker process across backends is not**, because it would recreate
exactly the toolchain collision the subprocess-venv architecture exists to avoid. Every
recommendation above assumes one worker per venv, not one worker for all of them; nobody should
read "give yamnet a persistent worker" (§7.1) as a step toward collapsing yamnet's, hear's, and
crisperwhisper's workers into a single process.

## 9. What this does not license

- **No claim about GPU memory sizing.** §6(b) is an inference from relative speed, not a memory
  measurement. Do not use it to size a card; get NVML/`nvidia-smi` sampling first.
- **No claim about `squim` or `clearvoice`.** Neither produced a single data point here. Any
  scheduling decision about them needs its own run.
- **No claim about CPU batch sizes above 16, or about `crisperwhisper` at any batch size.** Both
  are entirely absent from this data (§1, §4), not merely uncertain.
- **No claim about queue-wait time or pre-emption recovery.** Neither was measured; nothing here
  should feed a decision about scheduling priority or checkpoint/resume behaviour.
- **No claim that hear's first-call warm-up (§2) is confirmed and quantified.** It is read off a
  63%-vs-2% contrast in run-to-run spread between hear and yamnet at batch 1, with n=2. It is
  plausible enough to motivate a persistent-worker design, not solid enough to size one.
- **No license to collapse per-backend workers into one process.** See §8 — the CUDA toolchain
  constraint is architectural, not a performance tradeoff to be reconsidered later.
- **No precision beyond what n=2 supports.** Every fixed/marginal figure above is a range or is
  explicitly flagged as order-of-magnitude; treat any single-decimal figure quoted elsewhere as a
  convenience, not as more precise than the data underneath it.

## Prior findings carried in, not re-derived here

- Cold `ensure_venv` builds, first attempt (job 22210895, personal scratch): yamnet 706 s, hear
  701 s, crisperwhisper 590 s; main env `uv sync --all-extras --group dev` 333 s. That attempt died
  with `Disk quota exceeded` partway through the fourth venv: `uv-cache` 28 G, crisperwhisper
  6.8 G, hear 2.0 G, yamnet 2.0 G, on a filesystem 12% full with 1% inodes used — a per-user quota,
  not a full filesystem. `UV_CACHE_DIR` must share a filesystem with the venvs or hardlinking is
  defeated, so cache and venvs cannot be separated to dodge the quota.
- Earlier production data: the ten-recording triage array ran 2:54–6:53 per recording on an A100;
  `unasdiff` separation is reported as 43× slower than real time on an A100 (793 s and 1098 s for
  25.5 s of audio) and far worse on CPU. (Not re-derived per scope — see instructions.)

## Reproduce

```bash
uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/orcd_scheduling_2026_09_08.py \
    gpu-22213820.out cpu-22213836.out
```
