# Quickstart: Pedagogical Audio Tutorials

## Prerequisites

```bash
# Full development setup
uv sync --extra articulatory --extra text --extra video --extra senselab-ai --group dev --group docs

# Pre-commit hooks
uv run pre-commit install
```

## Test a single tutorial locally

```bash
export HF_TOKEN=<your-token>
uv run papermill tutorials/audio/audio_recording_and_acoustic_analysis.ipynb /dev/null --cwd . -k python3 --execution-timeout 600
```

## Test all tutorials

```bash
export HF_TOKEN=<your-token>
for nb in tutorials/audio/*.ipynb tutorials/video/*.ipynb tutorials/utils/*.ipynb; do
  echo -n "$(basename $nb): "
  uv run papermill "$nb" /dev/null --cwd . -k python3 --execution-timeout 600 2>&1 | tail -1
done
```

## Verify PR #431 PPG functions

```bash
uv run pytest src/tests/audio/tasks/features_extraction_test.py -v -k "ppg or phoneme"
```

## Key files to modify

| File | Purpose |
|------|---------|
| `tutorials/audio/audio_recording_and_acoustic_analysis.ipynb` | New Tutorial 1 |
| `tutorials/audio/transcription_and_phonemic_analysis.ipynb` | New Tutorial 2 |
| `tutorials/audio/00_getting_started.ipynb` | Update install/conventions |
| `tutorials/audio/shbt205_lab.ipynb` | New — adapted from course materials |
| `tutorials/manifest.json` | Add new tutorial entries |
| `src/senselab/audio/tasks/features_extraction/ppg.py` | PR #431 merge |
