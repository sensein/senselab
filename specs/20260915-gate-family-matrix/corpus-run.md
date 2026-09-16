# Sizing and the sbatch for the corpus run

**Nothing here has been submitted.** The recipe and the cost are stated so the owner can submit.

## Two steps, and only one of them is expensive

The matrix reduces a **features shard**, not the stores. So the corpus run is whichever of these
is not already done:

1. **extract** — `analyze_routing_evidence.py` reads all 62,578 stores once and writes
   `features/part-*.parquet`. This is the whole cost.
2. **reduce** — `gate_family_matrix.py` reads that shard and writes the matrix, the agreement and
   the disagreements. Effectively free.

**Check step 1 first.** The brief names existing runs at
`/orcd/scratch/bcs/002/satra/routing_scratch/{full_out,final_out,deriv_out}/`. Each of those is an
`out_dir` of `analyze_routing_evidence.py`, so each should already hold a `features/` shard
directory beside its `sweeps_by_family.json`:

```bash
ls -la /orcd/scratch/bcs/002/satra/routing_scratch/full_out/features/ | head
uv run python -c "
from pathlib import Path
from senselab.audio.workflows.triage.routing_analysis.report import shard_files, load_feature_column
d = Path('/orcd/scratch/bcs/002/satra/routing_scratch/full_out/features')
print(len(shard_files(d)), 'shards')
print(sum(len(load_feature_column(s, 'stem')) for s in shard_files(d)), 'recordings')
"
```

If that reports ~62.5k recordings, **the extract is already paid for** and the whole job is the
reduce: one core, a couple of minutes, no array. Run it interactively or on one small CPU
allocation. The shard is what the matrix reads; it does not care which sweep wrote it.

If it does not exist, or reports far fewer, run the extract below.

## Measured cost

Measured on this laptop (M-series, macOS, **not idle** — other work was running). Under load a
timing is an upper bound, so the wall-clock figures are pessimistic and the throughput figures are
conservative.

| quantity | measured | over 62,578 recordings |
| --- | --- | --- |
| `extract_features` per store, 1 process | 136 ms (27 stores, 48 MB, 13.1 MB/s) | 2.4 CPU-hours |
| mean store size | 1.78 MB | ~111 GB read once |
| `gate_matrix` reduce | 43,222 records/s | ~1.5 s |
| reduce + routes + disagreements | 11,646 records/s | ~5.4 s |
| resident per held record | 18.7 kB | ~1.1 GB, ~1.6 GB peak with imports |

The store sizes come from a 13-recording run carrying full derivatives; corpus stores under
`triage_full_20260908/out/` may differ, and 111 GB is the figure to sanity-check before trusting
the wall clock.

**The bound on the extract is filesystem, not CPU.** One worker sustains 13 MB/s of JSONL; 16
workers want ~210 MB/s aggregate off shared scratch. That, not the core count, is what sets the
wall clock, which is why the recipe below asks for 16 rather than 48.

## The extract, if it is needed

`analyze_routing_evidence.py` is already idempotent and resumable: it reuses a complete manifest
and skips every stem already in the shard, one part file per 4,000 recordings. So this is **one
job, not an array** — a job that dies resumes from the parts that landed, and sharding it by hand
would have several workers racing to write `part-00000.parquet`.

```bash
#!/bin/bash
#SBATCH --job-name=gate-matrix
#SBATCH --partition=mit_normal          # CPU-only; no GPU is requested or used
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/scratch/bcs/002/satra/gate_matrix/logs/%x-%j.out
#SBATCH --error=/orcd/scratch/bcs/002/satra/gate_matrix/logs/%x-%j.err
set -euo pipefail

REPO=/orcd/scratch/bcs/002/satra/senselab
OUT=/orcd/scratch/bcs/002/satra/gate_matrix
RUNS=/orcd/scratch/bcs/002/satra/triage_full_20260908/out

# A Slurm job does not inherit the login shell, and torchcodec dlopen()s FFmpeg by soname at
# IMPORT time -- verified: `gate_family_matrix.py --help` already has torchcodec loaded. Without
# this the job dies on `import`, not on reading audio, and the message names a .so rather than
# ffmpeg. This is the single most likely reason for the job to fail.
export LD_LIBRARY_PATH="/orcd/scratch/bcs/002/satra/miniforge/lib:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/bin:$PATH"          # uv
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"
export HF_HOME=/orcd/scratch/bcs/002/satra/hf   # nothing is downloaded; keeps any cache off $HOME
export MPLBACKEND=Agg                           # the heatmap is written headless
export OMP_NUM_THREADS=1                         # 16 reader processes, not 16x16 BLAS threads

mkdir -p "$OUT/logs"
cd "$REPO"

# Step 1, the expensive one. Idempotent: reuses the manifest and every completed shard part, so a
# requeue resumes. --expect fails the job rather than reporting success over a tree it mis-resolved.
uv run python scripts/analyze_routing_evidence.py "$RUNS" "$OUT" \
    --workers 16 --expect 62578

# Step 2, seconds. Reads the shard step 1 wrote; runs no model and touches no run directory.
uv run python scripts/gate_family_matrix.py "$OUT/features" "$OUT/matrix"
```

**Expected cost.** 2.4 CPU-hours of extract over 16 workers is ~9 minutes of wall clock if the
filesystem keeps up, and the `--time=04:00:00` is there for the case where it does not: 111 GB at a
degraded 10 MB/s aggregate is ~3 hours. Billed: 16 cores x wall clock, so between ~2.4 and ~48
core-hours. The reduce adds under a minute and ~1.6 GB. No GPU, no model download, no network.

If the shard already exists, drop step 1 and the allocation becomes
`--cpus-per-task=1 --mem=8G --time=00:15:00`.

## What to check in the output before believing it

- `[gates] reducing N recordings` should be ~62,578, and `analyze_routing_evidence.py` will have
  refused the manifest outright if it resolved the wrong tree.
- `features/missing.jsonl` names every store that could not be read. A large file there means the
  matrix's denominators are not the corpus.
- `branch_agreement.json`'s `unassigned_families` should account for the families no reference set
  assigns. The measurements spec counts 48 families over 796 task ids; the four reference sets do
  not name all 48, so this list should be non-empty on the corpus and is worth reading.
- The 13-recording sample had recall 1.000 on all four branches. On 3 subjects that is not a
  measurement. If the corpus disagrees, the corpus is right.
- **`family_routing.parquet` is the answer to "for each task family, how many are routed where and
  what are the decision criteria".** One row per (task family, branch): `routed` and `routed_rate`,
  `firing_gates` naming every gate that sent recordings there with its count, `sole_firing_gates`
  naming the gate each routing hinged on alone — that is the routing which disappears if the gate
  is removed — and `margin.*` over the least-clearing firing gate. `declared`/`agreement` say
  whether the family's reference set names the branch; `beyond_declaration` is additive routing,
  not an error.
- **`family_states.parquet`** carries `branches_routed.0` .. `branches_routed.4` per family. Anything
  above `.1` is additive routing and is intended; `.0` splits into `state.empty` (the bypass fired)
  and `state.unexplained` (content no gate read), and only the second is a charge against the
  ruleset.
- **SPEECH's 2x2 should now show the diadochokinesis families as positives**, not as 7,626 false
  positives: expect roughly tp 39,319 / fp 2,246 / tn 19,077 / fn 1,905, sens ~0.954 and spec
  ~0.895. If `fp` comes back near 9,872 the config override did not take.
- **Read `over-rt` and the budget it sits inside, not `decl/routed`.** The per-branch table's last
  column is a raw count ratio kept so nothing is lost; it is prevalence-dependent and is not a
  precision. `specs/20260915-gate-family-matrix/design.md` § 2026-09-15 says why.
