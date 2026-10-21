#!/bin/bash
# Re-run the ASR Apple-silicon benchmark serially, one model resident at a time.
#   bash specs/20261017-104000-asr-apple-silicon/bench/run_bench.sh [WORK_DIR]
# Needs: the crisperwhisper-cpu and qwen-asr-cpu venvs under ~/.cache/senselab/venvs, the three models in
# ~/.cache/huggingface, and ~250 MB free for one AMI shard (deleted after the mixes are built).
# THREADS (default 3) caps torch/OMP threads. Results: WORK_DIR/res/*.json, summary on stdout.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
WORK=${1:-${TMPDIR:-/tmp}/asr-apple-silicon-bench}
mkdir -p "$WORK/data" "$WORK/res"
PY="$REPO/.venv/bin/python"
HUB=${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}
snap() { echo "$HUB/models--$1/snapshots/$(cat "$HUB/models--$1/refs/main")"; }
CW=$(snap nyralabs--CrisperWhisper2.0_turbo)
QM=$(snap Qwen--Qwen3-ASR-1.7B)
FA=$(snap Qwen--Qwen3-ForcedAligner-0.6B)
CWPY=$HOME/.cache/senselab/venvs/crisperwhisper-cpu/bin/python
QPY=$HOME/.cache/senselab/venvs/qwen-asr-cpu/bin/python
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=${THREADS:-3}

if [ ! -f "$WORK/data/bench1200.wav" ]; then
  mkdir -p "$WORK/dl"
  "$PY" -c "from huggingface_hub import hf_hub_download as d; print(d('edinburghcstr/ami','ihm/test-00000-of-00004.parquet',repo_type='dataset',local_dir='$WORK/dl'))"
  "$PY" -I "$HERE/build_mix.py" "$WORK/dl/ihm/test-00000-of-00004.parquet" "$WORK/data" EN2002a,EN2002b
  rm -rf "$WORK/dl"
  "$PY" -I -c "
import soundfile as sf
w, sr = sf.read('$WORK/data/EN2002a_ihmmix.wav')
sf.write('$WORK/data/bench600.wav', w[300 * sr:900 * sr], sr, subtype='PCM_16')
sf.write('$WORK/data/bench180.wav', w[300 * sr:480 * sr], sr, subtype='PCM_16')
sf.write('$WORK/data/bench1200.wav', w[300 * sr:1500 * sr], sr, subtype='PCM_16')"
fi

run() {  # name python script audio args...
  local name=$1 py=$2 script=$3 audio=$4; shift 4
  /usr/bin/time -l "$py" "$HERE/$script" "$WORK/data/$audio" "$@" --out "$WORK/res/$name.json" \
    > "$WORK/res/$name.log" 2> "$WORK/res/$name.err" || echo "$name FAILED"
  echo "$name $(head -1 "$WORK/res/$name.log") time_peak_fp_gb=$(awk '/peak memory footprint/ {print $1/1e9}' "$WORK/res/$name.err")"
}
run cw600_mps_f16 "$CWPY" crisper_bench.py bench600.wav --model "$CW" --device mps --compute-type float16
run cw600_mps_f32 "$CWPY" crisper_bench.py bench600.wav --model "$CW" --device mps --compute-type float32
run cw180_cpu_f32 "$CWPY" crisper_bench.py bench180.wav --model "$CW" --device cpu --compute-type float32
run q600_mps_f16_c30 "$QPY" qwen_bench.py bench600.wav --model "$QM" --aligner "$FA" --device mps --dtype float16 \
  --batch 1 --chunk-sec 30 --empty-cache --device-map
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
  run q600_mps_f16_c30_wm "$QPY" qwen_bench.py bench600.wav --model "$QM" --aligner "$FA" --device mps \
  --dtype float16 --batch 1 --chunk-sec 30 --empty-cache --device-map
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
  run q1200_mps_f16_c30_wm "$QPY" qwen_bench.py bench1200.wav --model "$QM" --aligner "$FA" --device mps \
  --dtype float16 --batch 1 --chunk-sec 30 --empty-cache --device-map
run q180_mps_f32_c180 "$QPY" qwen_bench.py bench180.wav --model "$QM" --aligner "$FA" --device mps --dtype float32 --batch 1
run q180_cpu_f32_c180 "$QPY" qwen_bench.py bench180.wav --model "$QM" --aligner "$FA" --device cpu --dtype float32 --batch 1
run q180_cpu_f32_c30 "$QPY" qwen_bench.py bench180.wav --model "$QM" --aligner "$FA" --device cpu --dtype float32 \
  --batch 1 --chunk-sec 30
"$PY" "$HERE/wer_check.py" "$WORK/data/EN2002a_ihmmix_refs.json" 300 600 "$WORK"/res/cw600_*.json "$WORK"/res/q600_*.json
"$PY" "$HERE/wer_check.py" "$WORK/data/EN2002a_ihmmix_refs.json" 300 1200 "$WORK"/res/q1200_*.json
"$PY" "$HERE/wer_check.py" "$WORK/data/EN2002a_ihmmix_refs.json" 300 180 "$WORK"/res/cw180_*.json "$WORK"/res/q180_*.json
