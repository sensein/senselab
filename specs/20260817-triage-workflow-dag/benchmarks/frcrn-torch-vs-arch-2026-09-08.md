# FRCRN local/cluster divergence: torch version vs. CPU architecture, crossed

**Verdict, ahead of the numbers: neither.** A controlled 2x2 of torch version (2.13 / 2.14) x CPU
architecture (arm64 / x86_64) produces the same qualitative outcome — FRCRN nulls
`Breath`/`ThreeBreaths` and passes `HardCough` through — in **every one of the four cells**, on both
repeats, on two different physical x86_64 nodes, and in the actual current production venv on both
machines. Torch version has zero measured effect; architecture has zero measured effect. The
"cluster passes non-speech through" side of the originally reported contradiction could not be
reproduced today, on this machine, on the cluster's throwaway experiment venvs, or on the cluster's
own current production `clearvoice-cpu` venv. That is a different, more surprising outcome than
either of the two the brief asked to disambiguate between, and it is reported as such rather than
forced into a torch-or-architecture verdict.

## Deviation from the assigned base commit, and why

The task specified basing this work on `design/triage-workflow-dag` at `568cc1c4`. That commit is
the **parent** of the branch's actual current tip, `233a3ea4` ("refactor(triage): one residual
implementation, no meaning gates, enhanced stream kept") — one commit later. `233a3ea4` is the commit
that *introduces* `senselab.audio.tasks.speech_enhancement.residual.compute_residual`, which this
task's brief explicitly requires ("Use the library ... not a reimplementation"); at `568cc1c4` that
module does not exist yet. `233a3ea4` is also, by coincidence of timing, the commit that first wrote
up this exact local/cluster contradiction and proposed the torch/architecture hypothesis this task
was asked to test — so it is the natural, and only workable, base. This worktree was built on
`233a3ea4` instead of `568cc1c4`; no other commit was cherry-picked or altered.

## Method

Two scripts, both under `scripts/`:

- [`scripts/frcrn_torch_vs_arch_2026_09_08.py`](scripts/frcrn_torch_vs_arch_2026_09_08.py) — one cell
  of the 2x2. Builds a throwaway `ensure_venv` environment (`clearvoice==0.1.2` fixed,
  `numpy<2.0,>=1.24.3` fixed, `torch` pinned to the cell's version, `torchaudio==2.11.0` — see "Why
  torchaudio==2.11.0" below), runs FRCRN_SE_16K's own worker script (imported verbatim from
  `senselab.utils.clearvoice._WORKER_SCRIPT`, not reimplemented) over the three input files, and calls
  `senselab.audio.tasks.speech_enhancement.residual.compute_residual` — the shared library — for every
  reported quantity. Repeats the whole batch twice per invocation for the determinism check.
- [`scripts/frcrn_production_venv_check_2026_09_08.py`](scripts/frcrn_production_venv_check_2026_09_08.py) —
  the same three files and the same four quantities, but through senselab's actual **production**
  `enhance_audios` call (`CLEARVOICE_VENV`/`CLEARVOICE_REQUIREMENTS`, unpinned torch), for comparison
  against the pinned throwaway cells. Added mid-session once the throwaway-venv cells stopped
  reproducing the established cluster figure, to check whether the throwaway venvs themselves were
  the thing that differed from production.
- [`scripts/frcrn_cluster_job_2026_09_08.sh`](scripts/frcrn_cluster_job_2026_09_08.sh) — the sbatch
  entry point that runs both x86_64 cells (`torch==2.13.*`, `torch==2.14.*`) in one job on group flash
  scratch.

Inputs: the three files at `~/Downloads/frcrn_gated_probe_20260907/*__cluster_plain.wav` named in the
brief, copied byte-identical to the cluster (`md5` verified equal on both ends before any run — see
below) and used unmodified in every cell.

```
65031b3b814728af4fc94fd5aa34ac95  Breath__cluster_plain.wav
f1a846833cec5b2db73053886d74f16a  HardCough__cluster_plain.wav
4c278244bf8c20c2708d49ea4698dce9  ThreeBreaths__cluster_plain.wav
```

For each (cell, file), `compute_residual(reference=input, signal=enhanced, sampling_rate=16000,
max_lag_ms=200.0)` (the same `max_lag_ms` PREPROCESS and `residual_without_speech.py` use) reports:
`signal_energy_fraction` (fraction of input energy the enhanced output retained),
`residual_energy_fraction`, `gain_db`, `correlation_signal` (correlation of enhanced with input).

**Checking wheel availability before assuming, per the brief.** Both `torch==2.13.0` and
`torch==2.14.0` publish wheels for `cp311` on `macosx_14_0_arm64` *and* `manylinux_2_28_x86_64` (PyPI
JSON API queried directly — 24 wheel files each, both platforms present for both versions), so no
cell is impossible on the torch axis. `torchaudio`'s most recent release, however, is `2.11.0`
(PyPI's `torchaudio` project stopped tracking torch's version number some time before 2.13/2.14
shipped — consistent with the project's own migration toward `torchcodec`) — no `torchaudio` release
exists that nominally "matches" torch 2.13/2.14 the way `2.13.x`/`2.14.x` would. `torchaudio==2.11.0`
declares no dependency-metadata constraint on a specific torch version, so `ensure_venv`'s two-stage
install (Stage 1 installs the caller's exact torch/torchaudio pins; Stage 2 installs `clearvoice` with
torch/torchaudio specs stripped so it can't re-resolve them) installs cleanly regardless. Whether it
also *works* was checked empirically, not assumed: `torch==2.13.0`/`torch==2.14.0` +
`torchaudio==2.11.0` imports without error and `clearvoice.dataloader.dataloader.audio_norm` (which
does `import torchaudio` at module scope and calls `torchaudio.compliance.kaldi.fbank` /
`torchaudio.functional.compute_deltas` internally) loads and the worker runs end-to-end on every cell
below — confirmed, not a silent substitution. No cell in this 2x2 was unobtainable.

Every venv's `torch`/`torchaudio`/`numpy`/`scipy`/`soundfile` versions were probed and logged per cell
(below) so a version mismatch cannot masquerade as a torch/architecture effect.

## The 2x2

All four cells, two repeats each, all four `signal_energy_fraction` values **bit-identical between
repeat 0 and repeat 1** (confirmed from the raw JSON, not just the printed 4-decimal summaries) —
FRCRN on CPU is deterministic within a fixed environment, for every environment tested here.

| | torch 2.13 | torch 2.14 |
|---|---|---|
| **arm64 (local)** | Breath 0.0903 / HardCough 0.9875 / ThreeBreaths 0.0005 | Breath 0.0903 / HardCough 0.9875 / ThreeBreaths 0.0005 |
| **x86_64 (cluster)** | Breath 0.0903 / HardCough 0.9875 / ThreeBreaths 0.0005 | Breath 0.0903 / HardCough 0.9875 / ThreeBreaths 0.0005 |

(`signal_energy_fraction` — fraction of input energy the enhanced output retained — shown per file;
full four-quantity table follows.)

### Full per-cell, per-file numbers

| cell | env (torch/arch) | file | signal_frac | residual_frac | gain (dB) | corr(enhanced,input) |
|---|---|---|---|---|---|---|
| arm64 / 2.13 | 2.13.0 / arm64 | Breath | 0.09032 | 0.4611 | 7.757 | 0.7341 |
| arm64 / 2.13 | 2.13.0 / arm64 | HardCough | 0.98754 | 0.00055 | 0.052 | 0.9997 |
| arm64 / 2.13 | 2.13.0 / arm64 | ThreeBreaths | 0.00052 | 0.7292 | 27.165 | 0.5204 |
| arm64 / 2.14 | 2.14.0 / arm64 | Breath | 0.09032 | 0.4611 | 7.757 | 0.7341 |
| arm64 / 2.14 | 2.14.0 / arm64 | HardCough | 0.98754 | 0.00055 | 0.052 | 0.9997 |
| arm64 / 2.14 | 2.14.0 / arm64 | ThreeBreaths | 0.00052 | 0.7292 | 27.165 | 0.5204 |
| x86_64 / 2.13 (node3805) | 2.13.0+cpu / x86_64 | Breath | 0.09031 | 0.4612 | 7.757 | 0.7341 |
| x86_64 / 2.13 (node3805) | 2.13.0+cpu / x86_64 | HardCough | 0.98754 | 0.00055 | 0.052 | 0.9997 |
| x86_64 / 2.13 (node3805) | 2.13.0+cpu / x86_64 | ThreeBreaths | 0.00052 | 0.7292 | 27.164 | 0.5204 |
| x86_64 / 2.14 (node3805) | 2.14.0+cpu / x86_64 | Breath | 0.09031 | 0.4612 | 7.757 | 0.7341 |
| x86_64 / 2.14 (node3805) | 2.14.0+cpu / x86_64 | HardCough | 0.98754 | 0.00055 | 0.052 | 0.9997 |
| x86_64 / 2.14 (node3805) | 2.14.0+cpu / x86_64 | ThreeBreaths | 0.00052 | 0.7292 | 27.164 | 0.5204 |
| x86_64 / 2.13+2.14 (**node1623**, different node) | same as above | all three | identical to node3805 row above, to displayed precision | | | |

The arm64 and x86_64 numbers for the same torch version differ only at the 4th–5th significant digit
(e.g. Breath `signal_frac` 0.090317 arm64 vs. 0.090309 x86_64) — ordinary cross-platform
floating-point noise, not a change of regime; both are deep in the "null" bucket, nowhere near the
established cluster figure of "~1.000."

## Determinism

- **Within one environment, across repeats: exact.** Every cell's two repeats are bit-identical
  (`signal_energy_fraction` equal to 16 decimal places in the raw JSON) for all three files.
- **Across two different physical x86_64 nodes (node3805, a `pi_satra` GPU-class node used CPU-only;
  node1623, a plain `mit_quicktest` CPU node): identical**, confirming the x86_64 result is not an
  artifact of one specific machine.
- **Torch 2.13 vs. 2.14, same architecture: identical** on both architectures — torch version changes
  nothing measured here.

## The finding that supersedes the torch-vs-architecture question

Because neither axis of the assigned 2x2 produced the cluster's originally reported "~1.000" figure,
the throwaway-venv methodology itself was checked against the actual, currently-installed **production**
venv on both machines (`senselab.utils.clearvoice.CLEARVOICE_VENV`/`CLEARVOICE_REQUIREMENTS`, unpinned
torch, exactly what `enhance_audios` calls) — via
[`scripts/frcrn_production_venv_check_2026_09_08.py`](scripts/frcrn_production_venv_check_2026_09_08.py):

| production venv | torch | Breath signal_frac | HardCough signal_frac | ThreeBreaths signal_frac |
|---|---|---|---|---|
| local (`~/.cache/senselab/venvs/clearvoice-cpu`) | 2.14.0, arm64 | 0.09691 | 0.98754 | 0.00052 |
| cluster (`/orcd/scratch/bcs/002/satra/senselab-venvs/clearvoice-cpu`) | 2.14.0+cpu, x86_64 | 0.09690 | 0.98754 | 0.00052 |

**The current production cluster venv also nulls `Breath` and `ThreeBreaths` today** — matching the
current production *local* venv almost exactly (0.09691 vs. 0.09690; both cells' `gain_db` 7.54–7.55,
correlation 0.7419 both), and matching (to within the same small cross-build noise already seen above)
every throwaway-venv cell in the 2x2. Seven independent environments were measured in this session —
`arm64`x`{2.13, 2.14, production}`, `x86_64`x`{2.13, 2.14}`x`{node3805, node1623}`, `x86_64`
production — and every single one nulls `Breath`/`ThreeBreaths` and passes `HardCough` through. None
reproduces "cluster ~1.000."

This means the "cluster passes non-speech through" fact that motivated this task is **not
reproducible today**, under any tested combination of torch version, CPU architecture, or physical
node — including the cluster's own current production installation. One concrete, dated fact narrows
where to look next: the cluster's production `clearvoice-cpu` venv carries a completion marker
(`.senselab-installed`) timestamped **2026-09-07 17:27:08 -0400**, i.e. it was (re)built earlier today,
by some other process in this same session's timeframe. If the original "cluster ~1.000" measurement
predates that rebuild, whatever the venv looked like before 17:27 today is gone and unrecoverable from
here — the venv this report measured against is necessarily the *post*-rebuild one. This is offered as
the most concrete lead, not a confirmed mechanism: nothing in this session captured the pre-rebuild
venv's dependency versions to compare directly.

## Verdict

**Neither torch version nor CPU architecture is confirmed as the cause**, per the brief's own
decision rule for this outcome ("Anything else ... means neither explanation is sufficient, and you
should say so rather than forcing a verdict"). Specifically:

- It is not torch version: 2.13 and 2.14 are bit-identical on both architectures.
- It is not CPU architecture: arm64 and x86_64 agree (to ordinary floating-point noise) at both torch
  versions, on two different x86_64 nodes.
- It is not run-to-run instability within an environment: every repeat is bit-identical.
- **It is that the "pass-through" state itself does not reproduce today, anywhere this session could
  test it** — including the cluster's own current production environment, which is the strongest
  version of "the cluster" this report can invoke. That is a different failure of the original
  hypothesis than either arm of the brief's decision rule anticipated, and it is reported as such.

## What this implies for pinning

**No `CLEARVOICE_REQUIREMENTS` ceiling is indicated by this evidence.** Torch version was shown to
have zero effect on the one behavior this investigation could actually vary and measure (the
null/pass-through split on these three files, in the environments reachable today) — pinning a
ceiling would not have prevented, and would not fix, anything demonstrated here. Do not add one on the
strength of this report.

Separately, and worth flagging even though it falls outside this report's disambiguation: `torch
>=2.0.1` with no upper bound already let the production venv's resolved torch drift to 2.14.0 on both
machines with nobody deciding that version — consistent with the concern in the task background,
independent of whether that drift caused the original contradiction. If the *next* investigation
wants to pin something, it should be aimed at reproducing the pre-17:27-today cluster environment
(if that state can be recovered from anywhere — a job log with a package list, a lockfile, a build
artifact) rather than at another architecture/torch sweep, since this session's sweep has now
covered that space and found it flat.

**FRCRN's behavior on duration-filling non-speech input remains untrustworthy regardless of this
result.** The existing register (`residual-without-speech-2026-09-08.md`) already documents that this
class of recording sits near a decision boundary where the model's output can be near-total silence
or near-total pass-through with no stable middle ground, and recommends against using FRCRN's residual
as a background stream for it. Nothing here contradicts that; if anything, a divergence that could not
be reproduced under a controlled, repeated, cross-node, cross-torch-version test is itself evidence
that this boundary is sensitive to conditions this investigation did not manage to isolate — a reason
for more caution about this model class on this input shape, not less.

## Reproduce

```bash
# Local (arm64), either torch version:
uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/frcrn_torch_vs_arch_2026_09_08.py \
  --venv-name clearvoice-exp214-arm64 --torch-spec "torch==2.14.*" \
  --inputs ~/Downloads/frcrn_gated_probe_20260907/*__cluster_plain.wav \
  --repeats 2 --out-dir /tmp/frcrn_arm64_214 --out-json /tmp/frcrn_arm64_214/result.json

# Cluster (x86_64), both torch versions in one job:
sbatch --partition=pi_satra --cpus-per-task=8 --mem=32G --time=2:00:00 \
  --chdir=/orcd/scratch/bcs/002/satra/frcrn-torch-arch-20260908 \
  specs/20260817-triage-workflow-dag/benchmarks/scripts/frcrn_cluster_job_2026_09_08.sh

# Actual production venv, either machine:
uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/frcrn_production_venv_check_2026_09_08.py \
  --inputs ~/Downloads/frcrn_gated_probe_20260907/*__cluster_plain.wav \
  --out-dir /tmp/frcrn_prod_check --out-json /tmp/frcrn_prod_check/result.json
```

Raw JSON for every cell in this report (7 environments) is preserved under
`/orcd/scratch/bcs/002/satra/frcrn-torch-arch-20260908/results/` on the cluster and
`/tmp/frcrn_local_2x2/`, `/tmp/frcrn_prod_check_local/` locally; not copied into the repo (large,
reproducible from the commands above).
