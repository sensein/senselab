#!/usr/bin/env bash
# Re-run every measurement in this spec, one benchmark process at a time, three threads.
#   bash specs/20261017-110000-dyadic-turn-taking/bench/run_all.sh [WORK_DIR]
# Needs macOS `say` for the synthetic dialogues. Writes WORK_DIR/{data,results}.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="${1:-/tmp/dyadic-turn-taking}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/opt/ffmpeg@7/lib}"
for seed in 0 1 2; do
  [ -f "$work/data/dialogue_seed$seed/truth.json" ] || uv run python "$here/make_dialogue.py" "$work/data" --seed "$seed" --turns 40
done
# Separation-vs-masking fidelity. MossFormer2 runs on seed 0 only (about 0.14x real time on CPU).
uv run python "$here/evaluate_inputs.py" "$work/data" "$work/results" --seeds 0 --methods masking mossformer2
uv run python "$here/evaluate_inputs.py" "$work/data" "$work/results" --seeds 1 2 --methods masking
(cd "$here" && uv run python rescore.py "$work/data" "$work/results")
# Throughput on a 10-minute recording: masking, then the three MaAI runs serially and in parallel.
uv run python "$here/bench_speed.py" "$work/data/dialogue_seed0" "$work/results/speed_10min.json" --minutes 10 --methods masking
