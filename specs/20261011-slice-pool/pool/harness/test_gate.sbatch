#!/bin/bash
#SBATCH --job-name=pooltest-gate
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:05:00
#SBATCH --exclude=node2119,node2621,node3002,node1702,node3804
# Test link: gate the pool after its worker array(s); on a resubmission, queue this link again after the new array.
set -uo pipefail
POOL=${1:?pool dir}
P=/orcd/scratch/bcs/002/satra/r18d_20261010/pool
PY=/orcd/scratch/bcs/002/satra/senselab-r28/.venv/bin/python
OUT=$($PY $P/slice_pool.py gate "$POOL"); RC=$?
echo "$OUT"
echo "gate rc=$RC $(date -Is)" >> "$POOL/gate.log"
if [ "$RC" -eq 10 ]; then
  NEW=$(printf '%s\n' "$OUT" | sed -n 's/^RESUBMITTED //p' | tail -1)
  L=$(sbatch --parsable --dependency=afterany:$NEW --partition=${SLURM_JOB_PARTITION} --output="$POOL/logs/gate-%j.out" $P/tests/test_gate.sbatch "$POOL")
  echo "relinked gate $L after $NEW" | tee -a "$POOL/gate.log"
fi
exit 0
