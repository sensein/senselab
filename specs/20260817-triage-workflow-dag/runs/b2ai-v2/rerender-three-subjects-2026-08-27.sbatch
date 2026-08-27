#!/bin/bash
# Re-render one completed B2AI report from its immutable provenance store.
# Submit this script as a bounded array after generating TRIAGE_RERENDER_LIST.

#SBATCH --job-name=triage-report
#SBATCH --partition=pi_satra
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

set -uo pipefail

SCRATCH="$(readlink -f ~/orcd/scratch)"
REPO_DIR="${TRIAGE_REPO_DIR:?TRIAGE_REPO_DIR must name the staged source checkout}"
RUN_LIST="${TRIAGE_RERENDER_LIST:?TRIAGE_RERENDER_LIST must name one run root per line}"
CONFIG="$REPO_DIR/specs/20260817-triage-workflow-dag/runs/b2ai-v2/override.yaml"

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$REPO_DIR/.venv"
export UV_CACHE_DIR="$SCRATCH/uv-cache"
module load ffmpeg/5.1.4 >/dev/null 2>&1 || true
export LD_LIBRARY_PATH="$SCRATCH/miniforge/lib:${LD_LIBRARY_PATH:-}"

IDX="${SLURM_ARRAY_TASK_ID:?this script must run as an array task}"
ROOT="$(sed -n "$((IDX + 1))p" "$RUN_LIST")"
[ -n "$ROOT" ] || { echo "FATAL: no run root for task $IDX" >&2; exit 78; }
[ -r "$ROOT/run/store.jsonl" ] || { echo "FATAL: no provenance store: $ROOT/run/store.jsonl" >&2; exit 78; }
[ -r "$CONFIG" ] || { echo "FATAL: unreadable config: $CONFIG" >&2; exit 78; }

echo "=== report task $IDX on $(hostname) at $(date -Is)"
echo "=== run root: $ROOT"
cd "$REPO_DIR"
echo "=== commit: $(git log --oneline -1)"

uv run --no-sync python - "$ROOT" "$CONFIG" <<'PY'
from pathlib import Path
import sys

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes.report import report
from senselab.utils.prov_store import ProvStore

root = Path(sys.argv[1])
store = ProvStore.read_jsonl(root / "run" / "store.jsonl", run_id=root.name)
artifacts = report(store, root / "summary", load_triage_config(sys.argv[2]), run_dir=root / "run")
print(f"=== summary: {artifacts['summary']}")
PY
