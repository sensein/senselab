#!/bin/bash
#SBATCH --job-name=enh-survival
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

# Each export below has failed silently at least once while the job still reported COMPLETED.
export PATH="$HOME/.local/bin:$PATH"                                      # uv lives here
export LD_LIBRARY_PATH="$(readlink -f ~/orcd/scratch)/miniforge/lib:${LD_LIBRARY_PATH:-}"  # libavutil.so.56
export HF_HOME=/orcd/data/satra/002/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"                       # gated repos 401 without this
export SENSELAB_CACHE="$SCRATCH/senselab-cache"
export SENSELAB_VENV_CACHE="$SCRATCH/senselab-venvs"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export SENSELAB_RUN_ID="slurm-$SLURM_JOB_ID"                              # one commit per repo, whole sweep

REPO="$HOME/orcd/scratch/senselab-bench"
BRANCH="bench/enhancement-survival"
AUDIO="${AUDIO:?set AUDIO to the recording path}"
OUT="$SCRATCH/enh-survival-$SLURM_JOB_ID"
mkdir -p "$OUT"

cd "$REPO"
git fetch -q origin "$BRANCH"
git reset -q --hard "origin/$BRANCH"                                      # else it runs a stale checkout
echo "== commit: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"

uv sync --all-extras --group dev                                          # its own step, never during the run
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run --no-sync python specs/20260819-enhancement-survival/bench.py \
  "$AUDIO" --out "$OUT/results.json"

echo "== rows: $(python3 -c "import json;print(len(json.load(open('$OUT/results.json'))))")"
echo "== output: $OUT/results.json"
