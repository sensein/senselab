#!/bin/bash
# Test per-slice command: sleeps, writes an attempt record, and fails on purpose.
#   DUMMY_KILL_WORKER_SLICE=i: on its first attempt, slice i SIGKILLs its own worker (the job ends; no release).
#   slices i%5==0 fail their first attempt (exit 1); i%13==0 fail their first two (exit 2);
#   i%7==0 succeed with exit 3 (a declared success code); everything else exits 0.
set -u
I=$1; N=$2; A=${POOL_ATTEMPT:-1}
REC=${DUMMY_RECORDS:?}
mkdir -p "$REC"
LO=${DUMMY_MIN:-15}; HI=${DUMMY_MAX:-40}
T=$(( LO + RANDOM % (HI - LO + 1) ))
START=$(date +%s.%N)
if [ "$I" = "${DUMMY_KILL_WORKER_SLICE:-}" ] && [ "$A" = 1 ]; then
  echo "SIGKILL to the worker (pid $PPID) mid-slice: the claim must go stale and be taken over"
  sleep 5; kill -9 "$PPID"; sleep 60
fi
echo "slice $I/$N attempt $A on $(hostname) sleeping $T s"
sleep "$T"
END=$(date +%s.%N)
RC=0
if [ $((I % 13)) -eq 0 ] && [ "$A" -le 2 ]; then RC=2
elif [ $((I % 5)) -eq 0 ] && [ "$A" -le 1 ]; then RC=1
elif [ $((I % 7)) -eq 0 ]; then RC=3; fi
printf '{"slice": %s, "attempt": %s, "host": "%s", "job": "%s", "start": %s, "end": %s, "rc": %s}\n' \
  "$I" "$A" "$(hostname -s)" "${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-x}" "$START" "$END" "$RC" \
  > "$REC/$I.$A.$$.$RANDOM.json"
exit $RC
