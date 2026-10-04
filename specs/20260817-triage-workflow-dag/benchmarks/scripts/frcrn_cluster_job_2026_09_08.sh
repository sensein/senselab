#!/bin/bash
# FRCRN torch x86_64 cell(s): torch==2.13.* (new) and torch==2.14.* (known, reconfirmed) on the
# cluster's own architecture, each run twice for the determinism check. Runs entirely inside the
# job -- no computation on the login node.
set -euo pipefail

REPO=/orcd/scratch/bcs/002/satra/frcrn-torch-arch-20260908
INPUTS_DIR="$REPO/inputs"
OUT_ROOT=/orcd/scratch/bcs/002/satra/frcrn-torch-arch-20260908/results
mkdir -p "$OUT_ROOT"

GS=/orcd/scratch/bcs/002/satra
export PATH="$HOME/ffmpeg/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/ffmpeg/lib:${LD_LIBRARY_PATH:-}"
export UV_CACHE_DIR="$GS/uv-cache"
export SENSELAB_VENV_CACHE="$GS/senselab-venvs"
export HF_HOME=/orcd/data/satra/002/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
[ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
umask 027

cd "$REPO"
echo "=== host: $(hostname) arch: $(uname -m) ==="
echo "=== uv sync ==="
uv sync --all-extras --group dev

INPUTS=("$INPUTS_DIR"/*__cluster_plain.wav)
echo "Inputs: ${INPUTS[*]}"

echo "=== cell: x86_64 torch==2.13.* ==="
uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/frcrn_torch_vs_arch_2026_09_08.py \
  --venv-name clearvoice-exp213-x86 \
  --torch-spec "torch==2.13.*" \
  --inputs "${INPUTS[@]}" \
  --repeats 2 \
  --out-dir "$OUT_ROOT/x86_2p13" \
  --out-json "$OUT_ROOT/x86_2p13/result.json"

echo "=== cell: x86_64 torch==2.14.* (known cell, reconfirmed) ==="
uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/frcrn_torch_vs_arch_2026_09_08.py \
  --venv-name clearvoice-exp214-x86 \
  --torch-spec "torch==2.14.*" \
  --inputs "${INPUTS[@]}" \
  --repeats 2 \
  --out-dir "$OUT_ROOT/x86_2p14" \
  --out-json "$OUT_ROOT/x86_2p14/result.json"

echo "=== DONE ==="
