#!/bin/bash
# r36: the loudness round, as slice_pool lanes chained by r36_link.sbatch. Run on orcd-login:
#   bash /orcd/scratch/bcs/002/satra/r18d_20261010/SUBMIT_R36.sh <40-hex sha> [REPLAY_STEMS | none]
# REPLAY_STEMS defaults to r36/replay_stems_loudness.txt (the 1,602 _task-loudness / _task-loudness-v2 runs of
# manifest_r34); `none` skips the replay and starts at COHORT.
#   1. clones senselab-r36 from GitHub at <sha> (must be on origin/feat/task-events), r28 venv, PYTHONPATH r36/src
#   2. writes the sha into common_r36.sh (placeholder -> sha), writes the r36 single-job scripts (make_r36.sh)
#   3. r36 inputs: manifest_r36 (= manifest_r34), run_config (= r34's), cohort quantiles (= r34's), manifest_replay
#   4. submits ONE job, link `start`; every later lane is submitted by the link before it (r36_link.sbatch):
#      replay -> [bypass + vLLM review -> second opinion, only for needs_reread rows] -> cohort -> consistency ->
#      refold -> task audio -> rvec -> merge -> check, table. It never publishes: that is PUBLISH_R36.sh publish.
# Refuses while any r35-* or r36-* job is queued: two rounds must never write the stores at once.
# Stopping: CANCEL_R36.sh (links first, then the lanes downstream to upstream).
set -euo pipefail
S=/orcd/scratch/bcs/002/satra
W=$S/r18d_20261010
R36=$W/r36
REPO=$S/senselab-r36
URL=https://github.com/sensein/senselab.git
BRANCH=feat/task-events
SHA=${1:?usage: SUBMIT_R36.sh <40-hex sha> [REPLAY_STEMS|none]}
STEMS=${2:-$R36/replay_stems_loudness.txt}
die() { echo "SUBMIT_R36: $*" >&2; exit 2; }
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || die "the commit must be a full 40-hex SHA (git ls-remote $URL $BRANCH)"
cd $W
mkdir -p $R36/logs $R36/chain
[ -e $R36/chain/submitted ] && { echo "r36 already submitted: $(cat $R36/chain/submitted)"; exit 0; }

# No `squeue | grep -q` under pipefail: capture squeue's output, then test it.
QUEUE=$(squeue -h -u "$USER" -o "%i %j") || die "squeue failed; not submitting"
BUSY=$(printf '%s\n' "$QUEUE" | awk '$2 ~ /^r3[56]-/' )
[ -z "$BUSY" ] || die "r35/r36 jobs are still queued; wait for them (or CANCEL_R36.sh):
$BUSY"

# 1. the pinned checkout
if [ -d $REPO ]; then
  HEAD=$(git -C $REPO rev-parse HEAD)
  [ "$HEAD" = "$SHA" ] || die "$REPO exists at $HEAD, not $SHA (move it aside to re-clone)"
else
  TMPREPO=$REPO.tmp.$$
  git clone --quiet --branch $BRANCH $URL $TMPREPO
  git -C $TMPREPO cat-file -e "$SHA^{commit}" || { rm -rf $TMPREPO; die "$SHA is not on $URL $BRANCH"; }
  git -C $TMPREPO merge-base --is-ancestor "$SHA" origin/$BRANCH || { rm -rf $TMPREPO; die "$SHA is not on origin/$BRANCH"; }
  git -C $TMPREPO checkout --quiet --detach "$SHA"
  mv $TMPREPO $REPO
fi
[ "$(git -C $REPO rev-parse HEAD)" = "$SHA" ] || die "checkout is not $SHA"
[ -z "$(git -C $REPO status --porcelain)" ] || die "$REPO has local changes"
[ -f $REPO/specs/20261010-vllm-reviewer/review_vllm.sbatch ] || echo "note: no specs/20261010-vllm-reviewer in the checkout (r36 does not need it)"

# 2. common_r36.sh carries the sha; the r36 single-job scripts
if grep -q R36_COMMIT_PLACEHOLDER $W/common_r36.sh; then
  sed "s/R36_COMMIT_PLACEHOLDER/$SHA/" $W/common_r36.sh > $W/common_r36.sh.tmp.$$
  mv $W/common_r36.sh.tmp.$$ $W/common_r36.sh
fi
grep -q "^export TRIAGE_EXPECTED_COMMIT=$SHA\$" $W/common_r36.sh || die "common_r36.sh pins another commit: $(grep TRIAGE_EXPECTED_COMMIT= $W/common_r36.sh)"
bash $W/make_r36.sh

# 3. inputs
PY=$S/senselab-r28/.venv/bin/python
mkdir -p $R36/cohort $R36/pools $R36/rows_replay $R36/rows_cohort $R36/rows_refold $R36/rows_task_audio
cp -n $W/r34/manifest_r34.jsonl $R36/manifest_r36.jsonl
cmp -s $W/r34/manifest_r34.jsonl $R36/manifest_r36.jsonl || die "manifest_r36 differs from manifest_r34"
[ "$(wc -l < $R36/manifest_r36.jsonl)" -eq 62550 ] || die "manifest_r36 is not 62,550 rows"
cp -n $W/r34/run_config.yaml $R36/run_config.yaml
cmp -s $W/r34/run_config.yaml $R36/run_config.yaml || die "run_config differs from r34's"
cp -n $W/r34/cohort/cohort_quantiles.json $R36/cohort/cohort_quantiles.json
if [ "$STEMS" = none ]; then
  : > $R36/manifest_replay.jsonl.tmp.$$ && mv $R36/manifest_replay.jsonl.tmp.$$ $R36/manifest_replay.jsonl
else
  [ -s "$STEMS" ] || die "replay stem list $STEMS is missing or empty"
  $PY - "$R36/manifest_r36.jsonl" "$STEMS" "$R36/manifest_replay.jsonl" <<'PY'
import json, os, re, sys
manifest, listing, out = sys.argv[1:4]
stamp = re.compile(r"_\d{8}-\d{6}$")
wanted = {stamp.sub("", l.split("\t")[0].strip()) for l in open(listing) if l.strip()}
rows = [json.loads(l) for l in open(manifest) if l.strip()]
picked = [r for r in rows if stamp.sub("", r["stem"]) in wanted]
found = {stamp.sub("", r["stem"]) for r in picked}
if found != wanted:
    sys.exit(f"{len(wanted - found)} listed stems are not in manifest_r36, e.g. {sorted(wanted - found)[:3]}")
tmp = f"{out}.tmp.{os.getpid()}"
open(tmp, "w").write("".join(json.dumps(r) + "\n" for r in picked))
os.replace(tmp, out)
print(f"manifest_replay: {len(picked)} rows from {len(wanted)} listed stems")
PY
fi

# 4. one job: the start link
A=$(sbatch --parsable --export=ALL,R36_LINK=start $W/r36_link.sbatch)
LINE="r36 commit=$SHA start_link=$A replay_rows=$(wc -l < $R36/manifest_replay.jsonl) (stems: $STEMS) submitted $(date -Is); no publish"
echo "$LINE" | tee -a $R36/jobs_r36.txt >> $W/jobs.txt
echo "$LINE" | tee $R36/chain/submitted
