#!/bin/bash
#SBATCH --job-name=r4-report
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --output=${ORCD_GROUP_SCRATCH}/triage_r4_20260924/REPORT.txt
#SBATCH --open-mode=truncate
set -euo pipefail
source ${ORCD_GROUP_SCRATCH}/triage_design_20260919/env_common.sh
R=${ORCD_GROUP_SCRATCH}/triage_r4_20260924
A=${ORCD_GROUP_SCRATCH}/axes_census
echo "=== r4 replay, $(date -Is)"
echo "slice files: $(ls $R/rows/slices 2>/dev/null | wc -l) of 240"
echo "--- rows (must read 62548) ---"
cat $R/rows/slices/* 2>/dev/null | python3 -c '
import sys, json, collections
c = collections.Counter(); roots = set()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    o = json.loads(line)
    c[o.get("replay")] += 1
    roots.add(o.get("run_root"))
total = sum(c.values())
print("rows", total, "distinct run roots", len(roots), dict(c))
print("COMPLETE" if total == 62548 and len(roots) == 62548 else "INCOMPLETE -- resubmit the missing slices")
'
echo
echo "--- r3 against r4 ---"
python3 $A/compare.py $A/r3 $A/r4
