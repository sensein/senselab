#!/bin/bash
# r36 publish, NOT chained from SUBMIT_R36.sh: the owner approves before anything is published.
#   bash PUBLISH_R36.sh build   [AFTER_JOB]  # re-run the build only: rvec pool -> merge -> check, table (no publish)
#   bash PUBLISH_R36.sh publish [AFTER_JOB]  # copy -> finalize (verify inside) -> postcheck, review page (INSTALL=1)
# The chain already ran the build (r36/jobs_r36.txt: merge=... check=... table=...). Read f_check's output first.
# `publish` OVERWRITES the published r18d derivatives (rsync --delete-during into .../derivatives/senselab-triage)
# and the review page. finalize asserts one config hash and the r36 commit on every decision row.
# g_verify_r36.sbatch is an optional re-verify of out/ against the published copy: sbatch g_verify_r36.sbatch
set -euo pipefail
W=/orcd/scratch/bcs/002/satra/r18d_20261010
R36=$W/r36
cd $W
MODE=${1:?build or publish}
DEP=${2:+--dependency=afterany:$2}
grep -q R36_COMMIT_PLACEHOLDER common_r36.sh && { echo "common_r36.sh still has the placeholder: r36 was never submitted" >&2; exit 2; }
case "$MODE" in
build)
  PY=/orcd/scratch/bcs/002/satra/senselab-r28/.venv/bin/python
  mkdir -p $R36/recording_vectors && chmod 700 $R36/recording_vectors
  [ -d $R36/pools/rvec ] && mv $R36/pools/rvec $R36/pools/rvec.before.$(date +%Y%m%dT%H%M%S)
  $PY $W/pool/slice_pool.py init $R36/pools/rvec --name r36-w-rvec --slices 64 -- bash $W/r36_slice.sh rvec {i} {n} >&2
  A=$($PY $W/pool/slice_pool.py submit $R36/pools/rvec --workers 64 ${2:+--after $2} --name r36-w-rvec \
    -- --exclude=node2119,node2621,node3002,node1702,node3804 --requeue --partition=pi_satra,ou_bcs_low,mit_preemptable \
       --cpus-per-task=2 --mem=24G --time=12:00:00)
  G=$(sbatch --parsable --dependency=afterany:$A --wrap="$PY $W/pool/slice_pool.py gate --check-only $R36/pools/rvec" \
    --job-name=r36-gate-rvec --partition=pi_satra,ou_bcs_low --time=0:10:00 --mem=1G --output=$R36/logs/%x-%j.out)
  B=$(sbatch --parsable --dependency=afterok:$G f_merge_r36.sbatch)
  C=$(sbatch --parsable --dependency=afterany:$B f_check_r36.sbatch)
  D=$(sbatch --parsable --dependency=afterany:$B f_table_r36.sbatch)
  echo "r36 publish-build rvec=$A gate=$G merge=$B check=$C table=$D" | tee -a r36/jobs_r36.txt >> jobs.txt
  echo "r36 publish-build rvec=$A gate=$G merge=$B check=$C table=$D (merge runs only when every rvec slice is done)"
  ;;
publish)
  for f in recording_vectors.parquet triage_decisions.parquet triage_evidence.parquet triage.tsv; do
    [ -n "$DEP" ] || test -s r36/recording_vectors/$f || { echo "r36/recording_vectors/$f missing: run the chain (or build) first" >&2; exit 2; }
  done
  A=$(sbatch --parsable $DEP f_copy_r36.sbatch)
  B=$(sbatch --parsable --dependency=afterany:$A f_finalize_r36.sbatch)
  C=$(sbatch --parsable --dependency=afterany:$B f_postcheck_r36.sbatch)
  D=$(sbatch --parsable --dependency=afterany:$B --export=ALL,INSTALL=1 g_review_page_r36.sbatch)
  echo "r36 publish copy=$A finalize=$B postcheck=$C review_page=$D" | tee -a r36/jobs_r36.txt >> jobs.txt
  echo "r36 publish copy=$A finalize=$B postcheck=$C review_page=$D"
  ;;
*) echo "usage: PUBLISH_R36.sh build|publish [AFTER_JOB]" >&2; exit 2 ;;
esac
