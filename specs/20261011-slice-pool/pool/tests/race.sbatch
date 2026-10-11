#!/bin/bash
#SBATCH --job-name=pooltest-race
#SBATCH --partition=mit_quicktest
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=0:10:00
#SBATCH --exclude=node2119,node2621,node3002,node1702,node3804
#SBATCH --array=0-7%8
# Four processes per array task, every task on its own node where Slurm allows; all start at the same wall minute.
D=${RACE_DIR:?}; N=${RACE_NAMES:-400}
PY=/orcd/scratch/bcs/002/satra/senselab-r28/.venv/bin/python
T=${RACE_START:?}; while [ "$(date +%s)" -lt "$T" ]; do sleep 0.2; done
for k in 1 2 3 4; do $PY /orcd/scratch/bcs/002/satra/r18d_20261010/pool/tests/race.py run "$D" "$N" & done
wait
