# Quickstart: Auditory Scene Analysis

## Run windowed classification
```bash
uv run python -c "
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.api import classify_audios_in_windows
from senselab.utils.data_structures import HFModel

audio = Audio(filepath='tutorial_audio_files/audio_48khz_mono_16bits.wav')
model = HFModel(path_or_uri='MIT/ast-finetuned-audioset-10-10-0.4593')
results = classify_audios_in_windows([audio], model=model, window_size=1.0, hop_size=0.5)
for r in results[0][:5]:
    print(f'{r[\"start\"]:.1f}-{r[\"end\"]:.1f}s: {r[\"labels\"][0]} ({r[\"scores\"][0]:.3f})')
"
```
