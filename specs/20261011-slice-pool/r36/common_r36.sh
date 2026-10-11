# Sourced by every r36 job: environment, the pinned senselab-r36 checkout on the r28 venv, and the commit check.
# r36 = r35's corpus (manifest_r36.jsonl, a byte copy of r34/manifest_r34.jsonl) at the loudness commit
# (task_speech v6, cohort_stage v4): the loudness recordings replayed from TAXONOMY (and their needs_reread readings
# re-read), COHORT re-applied over every participant, its consistency read, the whole corpus re-folded, task audio,
# and the publish BUILD only (rvec -> merge -> check, table) into $R36/recording_vectors. Every sliced stage runs
# as a slice_pool (pool/slice_pool.py) under $R36/pools/<lane>. Publishing is PUBLISH_R36.sh, run by hand.
# The commit below is a placeholder until SUBMIT_R36.sh writes the pinned SHA into it.
S=/orcd/scratch/bcs/002/satra
W=$S/r18d_20261010
R34=$W/r34
R35=$W/r35
R36=$W/r36
R18=$S/r18_20261008
R9=$S/triage_r9_20260929
OUT=$R18/out
MANIFEST=$R36/manifest_r36.jsonl          # a byte copy of r34/manifest_r34.jsonl (62,550 runs)
REPLAY_MANIFEST=$R36/manifest_replay.jsonl
HINTS=$R9/scope
CONFIG=$R36/run_config.yaml               # a byte copy of r34/run_config.yaml
RV=$R36/recording_vectors
POOLS=$R36/pools
POOL_TOOL=$W/pool/slice_pool.py
GPU_EXCLUDE=node2119,node2621,node3002,node1702,node3804
CPU_PARTS=pi_satra,ou_bcs_low,mit_preemptable
GPU_PARTS=pi_satra,ou_bcs_low,ou_bcs_normal
SPECS_REVIEW=$S/senselab-r36/specs/20261010-vllm-reviewer
# Workers per pool (slices in brackets). The chain holds at most one pool's workers plus one link in the queue.
W_REPLAY=${W_REPLAY:-16}        # [32]
W_BYPASS=${W_BYPASS:-4}         # [ceil(nowords/200), at most 16]
W_GPU=${W_GPU:-2}               # per GPU type (h100, a100) [review ceil(words/300), opinion ceil(words/400)]
W_COHORT=${W_COHORT:-64}        # [128]
W_REFOLD=${W_REFOLD:-64}        # [128]
W_TASK_AUDIO=${W_TASK_AUDIO:-64} # [64]
W_RVEC=${W_RVEC:-64}            # [64]
source $S/triage_design_20260919/env_common.sh
export PATH="$PATH:$HOME/ffmpeg/bin"
export TRIAGE_EXPECTED_COMMIT=R36_COMMIT_PLACEHOLDER
export SENSELAB_REPO=$S/senselab-r36
export PYTHONPATH=$SENSELAB_REPO/src
[ -n "${R36_GPU:-}" ] || export CUDA_VISIBLE_DEVICES=""
PY="$S/senselab-r28/.venv/bin/python"
case "$TRIAGE_EXPECTED_COMMIT" in
  *PLACEHOLDER*) echo "FATAL: common_r36.sh still carries the commit placeholder; run SUBMIT_R36.sh <sha>" >&2; exit 74 ;;
esac
cd "$SENSELAB_REPO"
HEAD_SHA=$(git rev-parse HEAD)
if [ "$HEAD_SHA" != "$TRIAGE_EXPECTED_COMMIT" ]; then echo "FATAL: checkout is $HEAD_SHA, expected $TRIAGE_EXPECTED_COMMIT" >&2; exit 74; fi
command -v ffmpeg >/dev/null || { echo "FATAL: ffmpeg not on PATH" >&2; exit 73; }
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: HF_TOKEN empty" >&2; exit 73; }
