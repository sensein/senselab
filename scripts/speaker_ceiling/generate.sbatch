#!/bin/bash
#
# Phase 1 of the speaker-ceiling sweep: generate a durable corpus, sharded by k across an
# 8-task array on mit_preemptable. Writes ONLY under $CORPUS_DIR -- Phase 2 (evaluate.sbatch)
# never writes here and always runs as a separate, later submission (see that script's
# header for why, and for the exact --dependency command to chain the two).
#
# Submit with:
#   sbatch scripts/speaker_ceiling/generate.sbatch
#
# mit_preemptable is PreemptMode=REQUEUE: a task can be killed and restarted at any moment.
# That is survivable here because generate_corpus() checkpoints per session (a session whose
# .wav and .rttm both already exist and parse is skipped, not regenerated -- see
# scripts/speaker_ceiling/generate.py's "Resumability" section) and because each of the 8
# tasks writes only its own disjoint k=<k>/ subdirectory plus a same-named manifest fragment
# (manifest.k<k>.json), so two tasks can never clobber each other's output even if Slurm
# runs them concurrently on different nodes.

#SBATCH --job-name=speaker-ceiling-generate
#SBATCH --partition=mit_preemptable
#SBATCH --array=1-8
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/speaker-ceiling-generate-%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
# Deliberately no --qos: each partition supplies its own automatically, and naming one here
# fails outright with "Invalid qos specification" (a user's association permits only the
# partition's own QOS regardless of what is named).
#
# --gres=gpu:1 is untyped (any GPU model the partition has) rather than gpu:h100:1: the whole
# point of mit_preemptable is the much larger pool across every GPU model on the cluster, and
# pinning to h100 would throw most of that pool away. The 04:00:00 walltime already has
# headroom over the ~1.6h/task the measured H100 RTF implies (20 sessions x ~4.7 min), to
# cover a slower model landing this task; if that is not enough margin on your cluster's mix,
# tighten to --gres=gpu:h100:1 and shrink --time back down.

set -euo pipefail

# --- Environment this repo needs on this cluster, copied from the known-working recipe in
# project_orcd_gpu_test_jobs.md rather than reinvented -- every line below fixes a distinct
# failure that has already cost a debugging cycle once. -----------------------------------
export PATH="$HOME/.local/bin:$PATH"
# torchcodec dlopens FFmpeg by soname; this miniforge tree is what actually ships
# libavutil.so.56 on this cluster. The ffmpeg/5.1.4 spack module sets no FFMPEG_HOME and does
# not work -- do not substitute it in.
export LD_LIBRARY_PATH="$(readlink -f ~/orcd/scratch)/miniforge/lib:${LD_LIBRARY_PATH:-}"
# Per-user, not the shared HF_HOME tree: its huggingface_hub .locks/ subdirectories are not
# group-writable (created under whichever lab member first downloaded that model), and that
# fails as a scatter of unrelated-looking PermissionErrors rather than one clear message.
export HF_HOME="$(readlink -f ~/orcd/scratch)/hf-cache-$USER"
mkdir -p "$HF_HOME"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"

# One SEED (and therefore one SENSELAB_RUN_ID) for the whole sweep -- both phases -- so every
# task in both this array and evaluate.sbatch's pins the same resolved model commits. Derived
# from SEED rather than a Slurm job id specifically so generate.sbatch and evaluate.sbatch,
# submitted separately, agree on it without any explicit hand-off.
SEED="${SPEAKER_CEILING_SEED:-17}"
export SENSELAB_RUN_ID="speaker-ceiling-seed-${SEED}"
SESSIONS_PER_COUNT="${SPEAKER_CEILING_SESSIONS:-20}"
CORPUS_DIR="artifacts/speaker_ceiling/corpus/seed-${SEED}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
mkdir -p logs "$CORPUS_DIR"

# Checkpoint-and-requeue on Slurm's pre-preemption signal. A clean (0) exit tells Slurm the
# job finished and it will NOT requeue it, so this handler must exit nonzero. It does not
# need to flush anything itself: generate_corpus's per-session write-then-rename already
# means whatever completed before the signal is durable on disk.
_on_preempt() {
    echo "[$(date -Is)] task ${SLURM_ARRAY_TASK_ID:-N/A}: received preemption signal -- exiting nonzero for requeue" >&2
    exit 1
}
trap _on_preempt USR1

# uv sync as its own step, then uv run --no-sync: a bare `uv run` syncs mid-run, and on this
# cluster the venv was built under a different patch interpreter than the project now
# resolves, so a mid-run sync reinstalls packages underneath code that already imported them.
uv sync --all-extras
uv run --no-sync python scripts/probe_speaker_ceilings.py --mode generate \
    --counts 1 2 3 4 5 6 7 8 \
    --sessions "$SESSIONS_PER_COUNT" \
    --seed "$SEED" \
    --corpus "$CORPUS_DIR" \
    --device cuda
# --shard-k is intentionally omitted: probe_speaker_ceilings.py falls back to
# SLURM_ARRAY_TASK_ID when it is unset, and --array=1-8 above makes that value exactly the k
# this task should generate.
