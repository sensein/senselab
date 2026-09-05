#!/bin/bash
# Run one item of the established three-subject B2AI campaign with corrected PR 567 code.
# Submit this script once per subject as a bounded Slurm array.
#
# Example:
#   TRIAGE_SUB=sub-17cee767-1864-457a-b2ec-446a058a81f8 \
#     sbatch --array=0-27%2 evaluate-three-subjects-2026-08-26.sbatch

#SBATCH --job-name=triage-pr567
#SBATCH --partition=pi_satra
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -uo pipefail

SCRATCH="$(readlink -f ~/orcd/scratch)"
SUB="${TRIAGE_SUB:?TRIAGE_SUB must be exported at submit time}"
CAMPAIGN_ROOT="${TRIAGE_CAMPAIGN_ROOT:-$SCRATCH/triage_b2ai_v2}"
EVALUATION_ROOT="${TRIAGE_EVAL_ROOT:-$SCRATCH/triage-pr567-b2ai-v2-2fed5b7e}"
REPO_DIR="${TRIAGE_REPO_DIR:-$SCRATCH/triage-pr567-prolonged-vowel-c6845ce8/source}"
SUBJECT_ROOT="$CAMPAIGN_ROOT/$SUB"
OUT="$EVALUATION_ROOT/runs/$SUB"
CONFIG="$REPO_DIR/specs/20260817-triage-workflow-dag/runs/b2ai-v2/override.yaml"

export PATH="$HOME/.local/bin:$PATH"
# Keep the installed editable package aligned with the checked-out workflow code.
export UV_PROJECT_ENVIRONMENT="$REPO_DIR/.venv"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
module load ffmpeg/5.1.4 >/dev/null 2>&1 || true
export LD_LIBRARY_PATH="$SCRATCH/miniforge/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/orcd/data/satra/002/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export SENSELAB_CACHE="$SCRATCH/senselab-cache"
export SENSELAB_VENV_CACHE="$SCRATCH/senselab-venvs"
if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
else
  echo "FATAL: no HF token at ~/.cache/huggingface/token" >&2
  exit 78
fi

IDX="${SLURM_ARRAY_TASK_ID:?this script must run as an array task}"
WAV="$(sed -n "$((IDX + 1))p" "$SUBJECT_ROOT/filelist.txt")"
HINT="$(sed -n "$((IDX + 1))p" "$SUBJECT_ROOT/hintlist.txt")"
export SENSELAB_RUN_ID="pr567-airway-hear-${SUB:4:8}-${IDX}-20260826"

[ -n "$WAV" ] || { echo "FATAL: no filelist entry for task $IDX" >&2; exit 78; }
[ -r "$WAV" ] || { echo "FATAL: unreadable wav: $WAV" >&2; exit 78; }
[ -r "$HINT" ] || { echo "FATAL: unreadable hint: $HINT" >&2; exit 78; }
[ -r "$CONFIG" ] || { echo "FATAL: unreadable config: $CONFIG" >&2; exit 78; }
mkdir -p "$OUT"

echo "=== task $IDX on $(hostname) at $(date -Is)"
echo "=== subject: $SUB"
echo "=== wav:     $WAV"
echo "=== hint:    $HINT"
echo "=== out:     $OUT"
cd "$REPO_DIR"
echo "=== commit: $(git log --oneline -1)"

START=$SECONDS
uv run --no-sync python scripts/triage_audio.py "$WAV" --out "$OUT" --config "$CONFIG" --hint "$HINT"
rc=$?
echo "=== task $IDX exit $rc after $((SECONDS - START))s at $(date -Is)"

# A node-raised outcome still has an evidence-bearing report; do not fail the array for it.
if [ "$rc" -eq 1 ]; then
  echo "=== NODE-RAISED (rc=1), evidence retained"
  exit 0
fi
exit "$rc"
