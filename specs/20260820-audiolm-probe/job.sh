#!/bin/bash
#SBATCH --job-name=audiolm
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
set -euo pipefail

SCRATCH="$(readlink -f ~/orcd/scratch)"
POOL="$(readlink -f ~/orcd/pool)"
export SCRATCH POOL
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$SCRATCH/miniforge/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/orcd/data/satra/002/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export UV_CACHE_DIR="$POOL/uv-cache"          # SCRATCH is at 98% of its 1M inode cap
mkdir -p "$UV_CACHE_DIR"

VENV="$POOL/venvs/audiolm"
OUT="$SCRATCH/audiolm-$SLURM_JOB_ID"
mkdir -p "$OUT" "$(dirname "$VENV")"

REPO="$HOME/orcd/scratch/senselab-bench"
BRANCH="bench/enhancement-survival"
AUDIO="${AUDIO:?set AUDIO}"

cd "$REPO"
git fetch -q origin "$BRANCH"
git reset -q --hard "origin/$BRANCH"
echo "== commit: $(git rev-parse --short HEAD)"

# Its own venv: Qwen3-Omni needs a transformers new enough to carry Qwen3OmniMoe, which the
# repo's pinned transformers is not obliged to be.
if [ ! -x "$VENV/bin/python" ]; then
  uv venv --python 3.12 "$VENV"
fi
uv pip install --python "$VENV/bin/python" -q \
  torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV/bin/python" -q \
  "transformers>=4.57" accelerate qwen-omni-utils soundfile librosa numpy av

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$VENV/bin/python" -c "import transformers,torch;print('transformers',transformers.__version__,'torch',torch.__version__)"

"$VENV/bin/python" specs/20260820-audiolm-probe/probe.py \
  "$AUDIO" --out "$OUT/audiolm.json" --work "$OUT"

echo "== output: $OUT/audiolm.json"
