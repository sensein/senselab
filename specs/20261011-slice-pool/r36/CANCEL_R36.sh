#!/bin/bash
# Stop the r36 chain, downstream before upstream: the links first (so no link fires on a cancelled lane and submits
# the next one), then the single jobs and lanes from the end of the chain back to its start. Pools keep their state:
# resubmitting a lane's workers (slice_pool.py submit) resumes it, and done slices are never rerun.
set -uo pipefail
for name in r36-link r36-f-table r36-f-check r36-f-merge r36-w-rvec r36-w-task_audio r36-w-refold \
            r36-c-cohort-consistency r36-w-cohort r36-w-opinion r36-w-review r36-w-bypass r36-w-replay; do
  scancel -u "$USER" -n "$name" && echo "cancelled $name"
  [ "$name" = r36-link ] && sleep 5
done
