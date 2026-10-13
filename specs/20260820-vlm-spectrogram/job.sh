#!/bin/bash
#SBATCH --job-name=vlm-spec
#SBATCH --partition=pi_satra
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2:00:00
set -euo pipefail

SCRATCH="$(readlink -f ~/orcd/scratch)"
POOL="$(readlink -f ~/orcd/pool)"
export SCRATCH POOL
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$SCRATCH/miniforge/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/orcd/data/satra/002/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export UV_CACHE_DIR="$POOL/uv-cache"
# Qwen3.8 is mostly Gated DeltaNet. Without flash-linear-attention those blocks fall back to a
# torch implementation, which cost ~6 min per answer. fla is Triton-based, so no nvcc is needed;
# causal-conv1d is only an optional extra of it and is not installed. Kept on PYTHONPATH rather
# than installed into the shared senselab venv.
if [[ "${VLM_NO_FLA:-0}" == "1" ]]; then
  echo "== fla:       disabled by VLM_NO_FLA=1 (torch fallback)"
else
  export PYTHONPATH="$POOL/pylibs/fla${PYTHONPATH:+:$PYTHONPATH}"
  echo "== fla:       $POOL/pylibs/fla"
fi

# fla's kernels are Triton, and Triton JIT-compiles a small CPython extension (cuda_utils) on first
# use. The venv's interpreter is /usr/bin/python3.12, whose headers are not installed on this
# cluster: /usr/include/python3.12 holds only pyconfig-64.h, so the gcc call fails on a missing
# Python.h. Borrow headers from a uv-managed CPython of the same version.
uv python install 3.12 >/dev/null 2>&1 || true
PYINC="$HOME/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/include/python3.12"
[[ -f "$PYINC/Python.h" ]] || { echo "no Python.h at $PYINC" >&2; exit 1; }
export C_INCLUDE_PATH="$PYINC${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
echo "== py headers $PYINC"

REPO="$SCRATCH/senselab-audiolm"       # its own checkout; the sweep uses senselab-bench
BRANCH="triage"
AUDIO="${AUDIO:?set AUDIO}"
OUT="$SCRATCH/vlm-spec-$SLURM_JOB_ID"
mkdir -p "$OUT"

cd "$REPO"
git fetch -q origin "$BRANCH"
git reset -q --hard "origin/$BRANCH"
echo "== commit: $(git rev-parse --short HEAD)"

# The senselab venv already carries transformers 5.5.4, which knows qwen3_5. No second venv.
# --no-sync because the venv's interpreter patch differs from what the project resolves, and a
# sync mid-run reinstalls under an interpreter that has already imported.
UV_PROJECT_ENVIRONMENT="$SCRATCH/senselab-bench/.venv" uv run --no-sync python -c \
  "import transformers,torch;print('transformers',transformers.__version__,'torch',torch.__version__)"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

UV_PROJECT_ENVIRONMENT="$SCRATCH/senselab-bench/.venv" uv run --no-sync python \
  specs/20260820-vlm-spectrogram/probe.py "$AUDIO" \
  --out "$OUT/vlm.json" --png "$OUT/spectrogram.png"

echo "== output: $OUT/vlm.json"
echo "== image:  $OUT/spectrogram.png"
