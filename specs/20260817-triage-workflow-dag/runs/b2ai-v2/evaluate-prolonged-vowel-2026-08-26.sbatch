#!/bin/bash
# One post-change evaluation of the B2AI prolonged-vowel task on Engaging.
# The caller stages the checked-out source tree at TRIAGE_EVAL_ROOT/source before submission.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$(readlink -f ~/orcd/scratch)/miniforge/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$(readlink -f ~/orcd/scratch)/hf-cache-$USER"
mkdir -p "$HF_HOME"

ROOT="${TRIAGE_EVAL_ROOT:-$(readlink -f ~/orcd/scratch)/triage-pr567-prolonged-vowel-20260826}"
REPO_DIR="$ROOT/source"
OUT_DIR="$ROOT/output"
AUDIO="/orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/adult/bids_07_01_lang/sub-SUBJ04/ses-SES04/audio/sub-SUBJ04_ses-SES04_task-Prolonged-vowel.wav"
HINT="$REPO_DIR/specs/20260817-triage-workflow-dag/runs/b2ai-v2/hints/sub-SUBJ04_ses-SES04_task-Prolonged-vowel.json"
CONFIG="$REPO_DIR/specs/20260817-triage-workflow-dag/runs/b2ai-v2/override.yaml"

cd "$REPO_DIR"
test -f "$AUDIO"
test -f "$HINT"
test -f "$CONFIG"

# ASR consensus uses g2p-en, while the full production dependencies supply the model backends.
uv sync --extra nlp --group dev
export SENSELAB_RUN_ID="pr567-consensus-timing-prolonged-vowel-20260826"
uv run --no-sync python scripts/triage_audio.py "$AUDIO" --config "$CONFIG" --hint "$HINT" --out "$OUT_DIR"
