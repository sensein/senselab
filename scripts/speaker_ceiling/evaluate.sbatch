#!/bin/bash
#
# Phase 2 of the speaker-ceiling sweep: evaluate every backend against the durable corpus
# generate.sbatch already wrote. REQUIRES that corpus to exist and be complete -- this script
# never generates audio itself (probe_speaker_ceilings.py --mode evaluate refuses outright
# without --corpus, and validates every session file before diarizing a single one).
#
# Depends on generate.sbatch's ENTIRE array having finished successfully first. Wire that
# with a Slurm dependency so the two submissions run as one pipeline without you having to
# watch the queue:
#
#   GEN_JOB=$(sbatch --parsable scripts/speaker_ceiling/generate.sbatch)
#   sbatch --dependency=afterok:$GEN_JOB scripts/speaker_ceiling/evaluate.sbatch
#
# `afterok:$GEN_JOB` on the array's parent job id waits for every task in that array to
# complete successfully, not just the first -- exactly what is needed, since evaluation reads
# every k the array was supposed to produce. (Running the two by hand in sequence instead of
# using --dependency also works; --dependency just means you do not have to wait at a
# terminal to submit the second one.)
#
# Sharded per backend (6-way, one array task per backend, each covering every k) rather than
# per (backend, k): a cell is only `sessions_per_count` diarization calls, fast next to
# generation, so the coarser split already parallelizes the expensive part (six backends
# potentially running at once) without the bookkeeping of a 48-way backend x k array. Shard
# finer with --shard-k too (see the commented alternative below) if six backends still queue
# too seriously behind each other on your cluster.
#
# Aggregation is NOT part of this script or any sbatch job: it needs no GPU and reads only
# small JSON files, so run it once by hand (or via a trivial CPU salloc) after this array
# finishes:
#
#   uv run python scripts/probe_speaker_ceilings.py --mode aggregate \
#       --counts 1 2 3 4 5 6 7 8 --sessions 20 \
#       --out artifacts/speaker_ceiling/runs/speaker-ceiling-seed-17 \
#       --corpus artifacts/speaker_ceiling/corpus/seed-17

# Partition: ou_bcs_low rather than mit_preemptable. It is preemptable on the same terms (so the
# resumability below is still load-bearing), but every GPU in it is an a100 or h100 -- all >= 40 GB --
# whereas mit_preemptable's pool includes l4 (24 GB) and a40. An untyped --gres=gpu:1 there can land
# a 7B backend like VibeVoice on a card too small for it, and a systematic OOM would be recorded as
# that backend failing to count rather than as the wrong hardware, which would corrupt the very
# measurement this probe exists to produce. ou_bcs_low's 64-GPU ceiling is also mine alone rather
# than a shared group pool, and it starts immediately.
#SBATCH --job-name=speaker-ceiling-evaluate
#SBATCH --partition=ou_bcs_low
#SBATCH --array=0-5
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/speaker-ceiling-evaluate-%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
# No --qos, same reason as generate.sbatch: each partition supplies its own automatically.

set -euo pipefail

# --- Same environment block as generate.sbatch -- see that script's comments for what each
# line fixes. Kept duplicated rather than sourced from a shared file so each sbatch script
# stays a single self-contained file to hand to sbatch. ------------------------------------
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$(readlink -f ~/orcd/scratch)/miniforge/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$(readlink -f ~/orcd/scratch)/hf-cache-$USER"
mkdir -p "$HF_HOME"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

# Must match generate.sbatch's SEED so both phases resolve the same corpus path and the same
# SENSELAB_RUN_ID (and therefore the same pinned model commits).
SEED="${SPEAKER_CEILING_SEED:-17}"
export SENSELAB_RUN_ID="speaker-ceiling-seed-${SEED}"
SESSIONS_PER_COUNT="${SPEAKER_CEILING_SESSIONS:-20}"
CORPUS_DIR="artifacts/speaker_ceiling/corpus/seed-${SEED}"
OUT_DIR="artifacts/speaker_ceiling/runs/${SENSELAB_RUN_ID}"

# One array task per backend. Order matches evaluate.py's ALL_BACKENDS -- not load-bearing
# (each task passes its own name explicitly via --backends) but kept aligned for readability.
BACKENDS=(pyannote sortformer vibevoice child_adult moss diarizen)
BACKEND="${BACKENDS[$SLURM_ARRAY_TASK_ID]}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs "$OUT_DIR"

_on_preempt() {
    echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID:-N/A} (${BACKEND}): received preemption signal -- exiting nonzero for requeue" >&2
    exit 1
}
trap _on_preempt USR1

uv sync --all-extras
uv run --no-sync python scripts/probe_speaker_ceilings.py --mode evaluate \
    --counts 1 2 3 4 5 6 7 8 \
    --sessions "$SESSIONS_PER_COUNT" \
    --out "$OUT_DIR" \
    --corpus "$CORPUS_DIR" \
    --device cuda \
    --backends "$BACKEND"

# Finer-grained alternative (backend x k, 48-way): add `--array=0-47` above, then derive both
# indices from SLURM_ARRAY_TASK_ID, e.g.:
#   K_VALUES=(1 2 3 4 5 6 7 8)
#   BACKEND="${BACKENDS[$(( SLURM_ARRAY_TASK_ID / 8 ))]}"
#   K="${K_VALUES[$(( SLURM_ARRAY_TASK_ID % 8 ))]}"
# and pass both `--backends "$BACKEND" --shard-k "$K"` to the command above.
