#!/bin/bash
# Rebuild the calibration set, transcribe it with both recognizers through senselab, refit, recount fillers.
#   bash specs/20261017-104000-asr-apple-silicon/calibration/reproduce.sh [WORK_DIR]
# ~250 MB download (one AMI test shard, deleted after use), ~3.5 h of audio per recognizer.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SPEC=$(dirname "$HERE")
REPO=$(cd "$SPEC/../.." && pwd)
WORK=${1:-${TMPDIR:-/tmp}/asr-apple-silicon-calibration}
PY="$REPO/.venv/bin/python"
mkdir -p "$WORK/data" "$WORK/dl"
"$PY" -c "from huggingface_hub import hf_hub_download as d; d('edinburghcstr/ami','ihm/test-00000-of-00004.parquet',repo_type='dataset',local_dir='$WORK/dl')"
"$PY" -I "$SPEC/bench/build_mix.py" "$WORK/dl/ihm/test-00000-of-00004.parquet" "$WORK/data" EN2002a,EN2002b
"$PY" -I "$SPEC/bench/build_mix.py" "$WORK/dl/ihm/test-00000-of-00004.parquet" "$WORK/data" EN2002a,EN2002b 2
rm -rf "$WORK/dl"
"$PY" -I -c "
import numpy as np, soundfile as sf
for src, dst, snr, seed in [('EN2002a_ihmmix', 'EN2002a_ihmmix_pink5db', 5, 0), ('EN2002b_top2mix', 'EN2002b_top2mix_pink10db', 10, 1)]:
    w, sr = sf.read('$WORK/data/%s.wav' % src)
    n = np.random.default_rng(seed).standard_normal(len(w)); f = np.fft.rfft(n); k = np.arange(len(f)); k[0] = 1
    n = np.fft.irfft(f / np.sqrt(k), len(w)); speech = w[np.abs(w) > 0.01]
    n *= np.sqrt(np.mean(speech ** 2) / (np.mean(n ** 2) * 10 ** (snr / 10))); x = w + n; x /= np.abs(x).max() / 0.9
    sf.write('$WORK/data/%s.wav' % dst, x.astype(np.float32), sr, subtype='PCM_16')"
cd "$REPO"
export OMP_NUM_THREADS=${THREADS:-3}
uv run python "$SPEC/run_pair.py" "$WORK/calib" "$WORK"/data/*.wav
for w in 60 30; do uv run python "$SPEC/fit_calibration.py" "$WORK/calib" "$WORK/data" "$WORK/fit_window$w.json" $w; done
uv run python "$SPEC/filler_recall.py" "$WORK/calib" "$WORK/data"
