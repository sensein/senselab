#!/bin/bash
#SBATCH --job-name=r36-link
#SBATCH --partition=pi_satra,ou_bcs_low
#SBATCH --exclude=node2119,node2621,node3002,node1702,node3804
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=/orcd/scratch/bcs/002/satra/r18d_20261010/r36/logs/%x-%j.out
# The r36 chain, one link per job (R36_LINK via --export). Each link runs afterany on the worker array(s) of the pool
# it gates, checks that pool with `slice_pool.py gate` (complete -> go on; incomplete -> resubmit its workers once and
# queue this same link again after them; incomplete after that -> the pool writes `halted` and the chain stops), then
# submits the next lane's workers and the next link. The queue holds one lane's workers plus one link at a time.
#   start   : replay pool (when manifest_replay has rows)                                     -> link replay
#             else cohort pool                                                                   -> link cohort
#   replay  : gate replay -> needs_reread rows? split words/nowords -> bypass pool (CPU) + review pool (GPU) -> link review
#                                               none            -> cohort pool                             -> link cohort
#   review  : gate bypass + review -> words? opinion pool (GPU) -> link opinion; else cohort pool -> link cohort
#   opinion : gate opinion -> cohort pool                                                          -> link cohort
#   cohort  : gate cohort -> consistency (one job) -> refold pool (afterany consistency)          -> link refold
#   refold  : gate refold -> task audio pool                                                       -> link task_audio
#   task_audio: gate task audio -> rvec pool                                                       -> link rvec
#   rvec    : gate rvec -> merge -> check, table; stop. Publishing is PUBLISH_R36.sh, by hand.
# A link that already ran (marker in $R36/chain/) exits without submitting anything.
# Stopping or re-slicing by hand: CANCEL_R36.sh (links first, then workers), never the other way round.
set -euo pipefail
source /orcd/scratch/bcs/002/satra/r18d_20261010/common_r36.sh
unset CUDA_VISIBLE_DEVICES   # common_r36 hides GPUs; --export=ALL must not carry that into the GPU workers
cd $W
R36_LINK=${R36_LINK:?R36_LINK must name a link}
mkdir -p $R36/chain $R36/logs
MARK=$R36/chain/$R36_LINK.submitted
[ -e "$MARK" ] && { echo "link $R36_LINK already submitted: $(cat $MARK)"; exit 0; }
TMPSUF=part.${SLURM_JOB_ID:-$$}
export TMPSUF
POOLPY="$PY $POOL_TOOL"
log() { echo "$*" | tee -a $R36/jobs_r36.txt >> $W/jobs.txt; echo "$*"; }
halt() { log "r36 link $R36_LINK HALTED: $* $(date -Is)"; date -Is > "$R36/chain/$R36_LINK.halted"; exit 1; }
link() {  # $1 dependency job list (a:b), $2 link name
  sbatch --parsable --dependency=afterany:$1 --export=ALL,R36_LINK=$2 $W/r36_link.sbatch
}

# Gate every named pool. All complete: return. Any resubmitted: queue this link again after the new arrays, exit 0.
# Any halted: halt the chain.
gate_all() {
  local again="" pool out rc
  for pool in "$@"; do
    [ -e "$POOLS/$pool/pool.json" ] || continue
    rc=0; out=$($POOLPY gate "$POOLS/$pool") || rc=$?
    printf '%s\n' "$out"
    case $rc in
      0) ;;
      10) again=${again:+$again:}$(printf '%s\n' "$out" | sed -n 's/^RESUBMITTED //p' | tail -1) ;;
      *) halt "pool $pool incomplete after one resubmission ($POOLS/$pool/status.json)" ;;
    esac
  done
  if [ -n "$again" ]; then
    N=$(link "$again" "$R36_LINK")
    log "r36 link $R36_LINK: workers resubmitted ($again); link $R36_LINK again as $N"
    exit 0
  fi
}

# Create a lane's pool (idempotent) and submit its workers. $1 lane, $2 slices, $3 workers, $4 success codes,
# $5 dependency ("" for none), then sbatch args. Prints the job id.
pool_start() {
  local lane=$1 slices=$2 workers=$3 codes=$4 after=$5; shift 5
  $POOLPY init "$POOLS/$lane" --name "r36-w-$lane" --slices "$slices" --success-codes "$codes" \
    -- bash $W/r36_slice.sh "$lane" {i} {n} >&2
  $POOLPY submit "$POOLS/$lane" --workers "$workers" ${after:+--after $after} --name "r36-w-$lane" \
    -- --exclude=$GPU_EXCLUDE --requeue "$@"
}
gpu_pool_start() {  # $1 lane, $2 slices: one worker array per GPU type, all on the same pool
  local lane=$1 slices=$2 jobs="" g j
  $POOLPY init "$POOLS/$lane" --name "r36-w-$lane" --slices "$slices" --success-codes 0 \
    --host-check 'd=$(nvidia-smi | grep -oE "CUDA Version: [0-9.]+" | awk "{print \$3}"); awk -v d="$d" "BEGIN{exit !(d+0 >= 13.0)}"' \
    -- bash $W/r36_slice.sh "$lane" {i} {n} >&2
  for g in h100 a100; do
    j=$($POOLPY submit "$POOLS/$lane" --workers "$W_GPU" --name "r36-w-$lane" \
      -- --exclude=$GPU_EXCLUDE --requeue --partition=$GPU_PARTS --gres=gpu:$g:1 --cpus-per-task=8 --mem=96G --time=8:00:00)
    jobs=${jobs:+$jobs:}$j
  done
  echo "$jobs"
}
start_cohort() {
  A=$(pool_start cohort 128 $W_COHORT 0 "" --partition=$CPU_PARTS --cpus-per-task=4 --mem=24G --time=12:00:00)
  N=$(link $A cohort)
  log "r36 cohort=$A (128 slices, $W_COHORT workers) link_cohort=$N"
}

case "$R36_LINK" in
start)
  NRP=$( [ -s $REPLAY_MANIFEST ] && wc -l < $REPLAY_MANIFEST || echo 0 )
  if [ "$NRP" -gt 0 ]; then
    RS=$(( NRP < 32 ? NRP : 32 ))
    A=$(pool_start replay $RS $W_REPLAY 0,3 "" --partition=$CPU_PARTS --cpus-per-task=4 --mem=24G --time=12:00:00)
    N=$(link $A replay)
    log "r36 commit=$HEAD_SHA replay=$A ($NRP recordings, $RS slices, $W_REPLAY workers; exit 3 = needs_reread = success) link_replay=$N"
  else
    log "r36 commit=$HEAD_SHA no replay manifest rows"
    start_cohort
  fi
  ;;
replay)
  gate_all replay
  $PY - <<'PY'
import collections, glob, json, os
R36 = "/orcd/scratch/bcs/002/satra/r18d_20261010/r36"
review, last = {}, {}
for path in sorted(glob.glob(f"{R36}/rows_replay/slices/*.jsonl")):
    for line in open(path):
        if line.startswith("{"):
            rec = json.loads(line)
            review[rec["stem"]] = str(rec.get("review") or "unrecorded")
rows = [json.loads(l) for l in open(f"{R36}/manifest_replay.jsonl") if l.strip()]
counts = collections.Counter(review.get(r["stem"], "not_in_replay_log") for r in rows)
picked = [{k: r[k] for k in ("stem", "enhanced", "source") if k in r} for r in rows if review.get(r["stem"]) == "needs_reread"]
tmp = f"{R36}/rows_needs_reread.jsonl.{os.environ['TMPSUF']}"
open(tmp, "w").write("".join(json.dumps(r) + "\n" for r in picked))
os.replace(tmp, f"{R36}/rows_needs_reread.jsonl")
json.dump({"replay_rows": len(rows), "needs_reread": len(picked), **dict(counts)}, open(f"{R36}/rows_needs_reread.counts.json", "w"), indent=1)
print("replay review states", dict(counts), "needs_reread", len(picked))
PY
  NR=$(wc -l < $R36/rows_needs_reread.jsonl)
  if [ "$NR" -eq 0 ]; then
    log "r36 replay complete; no needs_reread rows"
    start_cohort
  else
    mkdir -p $R36/review/logs $R36/review_nowords/logs $R36/review_split
    SPLIT_PY=$W/split_words_r34.py   # generic: OUT_DIR ROWS.jsonl...; the reviewer's own words predicate
    SLURM_CPUS_PER_TASK=4 $PY $SPLIT_PY $R36/review_split $R36/rows_needs_reread.jsonl
    [ -s $R36/review_split/errors.txt ] && log "r36 WARNING: $(wc -l < $R36/review_split/errors.txt) needs_reread rows whose store did not open (r36/review_split/errors.txt)"
    cp $R36/review_split/rows_nowords.jsonl $R36/review_nowords/rows.jsonl.$TMPSUF && mv -f $R36/review_nowords/rows.jsonl.$TMPSUF $R36/review_nowords/rows.jsonl
    cp $R36/review_split/rows_words.jsonl $R36/review/rows.jsonl.$TMPSUF && mv -f $R36/review/rows.jsonl.$TMPSUF $R36/review/rows.jsonl
    [ -d $R36/review/cache ] || cp -a $R34/review/cache $R36/review/cache   # warm vLLM/triton caches; r34's tree is only read
    NB=$(wc -l < $R36/review_nowords/rows.jsonl); NWD=$(wc -l < $R36/review/rows.jsonl)
    DEPS=""
    if [ "$NB" -gt 0 ]; then
      BS=$(( (NB + 199) / 200 )); BS=$(( BS > 16 ? 16 : BS ))
      A=$(pool_start bypass $BS $W_BYPASS 0 "" --partition=$CPU_PARTS --cpus-per-task=2 --mem=8G --time=4:00:00)
      DEPS=$A
    fi
    if [ "$NWD" -gt 0 ]; then
      RS=$(( (NWD + 299) / 300 ))
      R=$(gpu_pool_start review $RS)
      DEPS=${DEPS:+$DEPS:}$R
    fi
    [ -n "$DEPS" ] || halt "needs_reread rows, but none landed in words or nowords (r36/review_split)"
    N=$(link $DEPS review)
    log "r36 needs_reread=$NR words=$NWD nowords=$NB bypass=${A:-none} (slices ${BS:-0}) review=${R:-none} (slices ${RS:-0}, 1 vLLM server per GPU, h100+a100 workers) link_review=$N"
  fi
  ;;
review)
  gate_all bypass review
  NWD=$( [ -s $R36/review/rows.jsonl ] && wc -l < $R36/review/rows.jsonl || echo 0 )
  if [ "$NWD" -gt 0 ]; then
    OS=$(( (NWD + 399) / 400 ))
    O=$(gpu_pool_start opinion $OS)
    N=$(link $O opinion)
    log "r36 second_opinion=$O (slices $OS) link_opinion=$N"
  else
    start_cohort
  fi
  ;;
opinion)
  gate_all opinion
  start_cohort
  ;;
cohort)
  gate_all cohort
  B=$(sbatch --parsable $W/c_cohort_consistency_r36.sbatch)
  C=$(pool_start refold 128 $W_REFOLD 0 $B --partition=$CPU_PARTS --cpus-per-task=2 --mem=16G --time=12:00:00)
  N=$(link $C refold)
  log "r36 cohort complete; consistency=$B refold=$C (128 slices, $W_REFOLD workers) link_refold=$N"
  ;;
refold)
  gate_all refold
  D=$(pool_start task_audio 64 $W_TASK_AUDIO 0 "" --partition=$CPU_PARTS --cpus-per-task=2 --mem=8G --time=6:00:00)
  N=$(link $D task_audio)
  log "r36 refold complete; task_audio=$D (64 slices, $W_TASK_AUDIO workers) link_task_audio=$N"
  ;;
task_audio)
  gate_all task_audio
  mkdir -p $RV && chmod 700 $RV
  E=$(pool_start rvec 64 $W_RVEC 0 "" --partition=$CPU_PARTS --cpus-per-task=2 --mem=24G --time=12:00:00)
  N=$(link $E rvec)
  log "r36 task audio complete; rvec=$E (64 slices, $W_RVEC workers) link_rvec=$N"
  ;;
rvec)
  gate_all rvec
  F=$(sbatch --parsable $W/f_merge_r36.sbatch)
  G=$(sbatch --parsable --dependency=afterany:$F $W/f_check_r36.sbatch)
  H=$(sbatch --parsable --dependency=afterany:$F $W/f_table_r36.sbatch)
  log "r36 rvec complete; merge=$F check=$G table=$H (end of the chain; publish with PUBLISH_R36.sh after the owner approves)"
  ;;
*) echo "unknown R36_LINK $R36_LINK" >&2; exit 2 ;;
esac
date -Is > "$MARK"
