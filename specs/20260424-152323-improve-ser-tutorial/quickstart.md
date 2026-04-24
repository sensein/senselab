# Quickstart: Improve SER Tutorial

## Test the updated tutorial

```bash
export HF_TOKEN=<your-token>
uv run papermill tutorials/audio/speech_emotion_recognition.ipynb /dev/null --cwd . -k python3 --execution-timeout 1200
```

## Key files

| File | Purpose |
|------|---------|
| `tutorials/audio/speech_emotion_recognition.ipynb` | Updated SER tutorial |
