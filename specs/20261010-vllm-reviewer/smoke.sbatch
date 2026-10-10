#!/bin/bash
# GPU smoke test of the vLLM reviewer engine and prompt 14, on copies of the owner-labelled stores.
#
#   W=/orcd/scratch/bcs/002/satra/vllm_smoke_20261010
#   sbatch --output="$W/logs/%x-%j.out" --export=ALL,SENSELAB_SHA=<40-hex pushed commit> smoke.sbatch
#
# Copies the 11 stems' run roots from tefix_20261010/gpu twice (out_a, out_b; every stream materialised so
# nothing writes through a link), reviews out_a with 3 servers on the card (falling back to 2 if any 3-server
# client could not start its server), reviews out_b the same way, asks the second opinion over out_a, re-folds
# both, and reports labels, digests and the fold (smoke_report.py). Prints no transcript text.

#SBATCH --job-name=vllm-smoke
#SBATCH --partition=pi_satra,ou_bcs_low,mit_preemptable
#SBATCH --exclude=node2119,node2621,node3002,node1702
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=1:00:00

set -uo pipefail
umask 077
S=/orcd/scratch/bcs/002/satra
W=$S/vllm_smoke_20261010
T=$S/tefix_20261010
R9=$S/triage_r9_20260929
SHA="${SENSELAB_SHA:?SENSELAB_SHA must be the pinned commit}"
CHECKOUT=$W/senselab-$SHA
PY=$S/senselab-r27/.venv/bin/python
JOB=${SLURM_JOB_ID:-local}
STEMS="022c62b4:open-response 07929b7b:cinderella 0842e76f:productive-vocabulary-3 168c5e68:random-item 2ba9a335:random-item 4bd22301:v2-threebreathsnose 541e6b30:breath-1 8ca6baff:random-item 9488c55d:animal-fluency f5c03456:v2-threebreathsnose fac74f45:random-item"

source $S/triage_design_20260919/env_common.sh
export SENSELAB_CACHE=$W/cache SENSELAB_RUN_ID="slurm-$JOB" PYTHONPATH=$CHECKOUT/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OLLAMA_VULKAN=0
export VLLM_CACHE_ROOT=$W/cache/vllm TRITON_CACHE_DIR=$W/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor FLASHINFER_WORKSPACE_BASE=$W/cache/flashinfer
export SENSELAB_VLLM_LOG_DIR=$W/logs/vllm-$JOB
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/13.1.0 2>/dev/null && export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
mkdir -p $W/logs $W/cache

cd "$CHECKOUT" || exit 2
[ "$(git rev-parse HEAD)" = "$SHA" ] || { echo "FATAL: checkout is not $SHA"; exit 2; }
echo "=== smoke $(date -Is) host=$(hostname) commit=$SHA"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
DRIVER_CUDA=$(nvidia-smi | grep -oE "CUDA Version: [0-9.]+" | awk '{print $3}')
awk -v d="$DRIVER_CUDA" 'BEGIN{exit !(d+0 >= 13.0)}' || { echo "FATAL: driver CUDA $DRIVER_CUDA < 13.0"; exit 3; }

STEMS="$STEMS" python3 - "$T/gpu/rows.jsonl" "$W" <<'PY'
import json, os, shutil, sys
from pathlib import Path

source, work = Path(sys.argv[1]), Path(sys.argv[2])
wanted = [pair.split(":") for pair in os.environ["STEMS"].split()]
rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
chosen = []
for stem, task in wanted:
    hits = [row for row in rows if row["stem"][4:12] == stem and task in row["stem"]]
    assert len(hits) == 1, (stem, task, len(hits))
    chosen.append(hits[0])
for copy in ("a", "b"):
    out_rows = []
    for row in chosen:
        enhanced = Path(row["enhanced"])
        run_root = enhanced.parents[2]
        relative = run_root.relative_to(source.parent / "out")
        target = work / f"out_{copy}" / relative
        if not (target / "run" / "store.jsonl").exists():
            shutil.copytree(run_root, target, symlinks=True)
            for link in (target / "run" / "streams").iterdir():
                if link.is_symlink():
                    real = link.resolve()
                    link.unlink()
                    if real.exists():
                        shutil.copy2(real, link)
        out_rows.append({**row, "enhanced": str(target / enhanced.relative_to(run_root))})
    (work / f"rows_{copy}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in out_rows))
print(f"prepared {len(chosen)} stems x 2 copies")
PY

review_pass() {  # $1 copy (a|b), $2 servers; echoes the error-row count
  local copy=$1 servers=$2 util cfg t0 i pids=()
  util=$(awk -v n="$servers" 'BEGIN{printf "%.2f", int(90 / n) / 100}')
  cfg=$W/review-vllm-$servers.yaml
  printf 'redaction:\n  llm_check:\n    enabled: true\n    keep_worker_resident: true\n    ref: 52f3f65bc7a02d555763bc923bd1d9094898219d\n    engine: vllm\n    vllm:\n      gpu_memory_utilization: %s\n' "$util" > "$cfg"
  t0=$(date +%s)
  for i in $(seq 0 $((servers - 1))); do
    "$PY" -u scripts/extend_llm_review.py $W/rows_$copy.jsonl --slice-index $i --slice-count $servers \
      --config "$cfg" --hints $R9/scope --log-dir $W/review_$copy-$servers --no-refold \
      > $W/logs/client-$JOB-$copy-$servers-$i.log 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "=== review $copy with $servers server(s) at $util: $(( $(date +%s) - t0 ))s wall" >&2
  python3 -c "import glob, json, sys; print(sum(json.load(open(f))['counts'].get('error', 0) for f in glob.glob(sys.argv[1])))" \
    "$W/review_$copy-$servers/slices/llm-review-slice-*-of-$servers.summary.json"
}

SERVERS=3
errors=$(review_pass a 3)
echo "=== 3 servers: $errors error row(s)"
if [ "$errors" != "0" ]; then
  grep -hoE "KV cache[^\"]{0,160}|max seq len[^\"]{0,160}|[Nn]o available memory[^\"]{0,120}|CUDA out of memory[^\"]{0,80}" \
    $W/review_a-3/slices/*.jsonl $SENSELAB_VLLM_LOG_DIR/*.log 2>/dev/null | sort | uniq -c | head -8
  SERVERS=2
  errors=$(review_pass a 2)
  echo "=== 2 servers: $errors error row(s)"
fi
grep -hE "GPU KV cache size|Maximum concurrency|Model loading took|init engine" $SENSELAB_VLLM_LOG_DIR/*.log 2>/dev/null \
  | sed -E 's/^.{0,40}(INFO|WARNING)/\1/' | cut -c1-200 | sort | uniq -c | head -12
errors_b=$(review_pass b $SERVERS)
echo "=== run b with $SERVERS servers: $errors_b error row(s)"
grep -h "median" $W/logs/client-$JOB-*.log | tail -n 20 | sed -E 's/.*\] //' | cut -c1-120

echo "=== second opinion over out_a $(date -Is)"
printf 'redaction:\n  llm_check:\n    enabled: true\n    ref: 52f3f65bc7a02d555763bc923bd1d9094898219d\nsecond_opinion:\n  enabled: true\n' > $W/second_opinion.yaml
"$PY" -u scripts/extend_second_opinion.py $W/rows_a.jsonl --slice-index 0 --slice-count 1 \
  --config $W/second_opinion.yaml --hints $R9/scope --log-dir $W/opinion_a \
  --ollama-binary $S/ollama-models-flash/ollama-0.35.1/bin/ollama --ollama-models $S/ollama-models-flash/models \
  --verified-dir /orcd/data/satra/002/models/ollama-senselab/verified 2>&1 | grep -E "^\[slice|Rows|  [a-z]" | tail -15

{ cat $W/review-vllm-$SERVERS.yaml; printf 'second_opinion:\n  enabled: true\n'; } > $W/fold.yaml
for copy in a b; do
  echo "=== refold $copy $(date -Is)"
  "$PY" -u scripts/extend_refold.py $W/rows_$copy.jsonl --slice-index 0 --slice-count 1 --hints $R9/scope \
    --config $W/fold.yaml --log-dir $W/refold_$copy --commit $SHA 2>&1 | grep -E "Rows|  [a-z]" | tail -8
done

echo "=== report $(date -Is)"
"$PY" -u $CHECKOUT/specs/20261010-vllm-reviewer/smoke_report.py $W/rows_a.jsonl $W/rows_b.jsonl --out $W/report-$JOB.json
echo "=== DONE $(date -Is)"
