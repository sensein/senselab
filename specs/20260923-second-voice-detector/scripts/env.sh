# Shared environment for every job in this campaign. Sourced, never executed.
#
# The checkout is pinned and never reset while jobs run against it; the venv cache and
# HF cache live off $HOME so a job cannot fill the home quota. LD_LIBRARY_PATH is exported
# explicitly because a Slurm job does not inherit the login shell and torchcodec dlopens
# FFmpeg by soname at import time.
SCRATCH=/orcd/scratch/bcs/002/satra
CHECKOUT="$SCRATCH/senselab-secondvoice"
PINNED_SHA=0f7997bc
W="$SCRATCH/secondvoice_20260923"

MF=$(readlink -f ~/orcd/scratch)/miniforge
export PATH="$HOME/.local/bin:$MF/bin:$PATH"
export LD_LIBRARY_PATH="$MF/lib:${LD_LIBRARY_PATH:-}"
export SENSELAB_CACHE="$SCRATCH/senselab-cache"
export SENSELAB_VENV_CACHE="$SCRATCH/senselab-venvs"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
export HF_HOME=/orcd/data/satra/002/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

cd "$CHECKOUT"
HAVE=$(git rev-parse --short=8 HEAD)
if [ "$HAVE" != "$PINNED_SHA" ]; then
  echo "FATAL: checkout is at $HAVE, expected $PINNED_SHA" >&2
  exit 1
fi
echo "=== host $(hostname) commit $HAVE $(date -Is)"
