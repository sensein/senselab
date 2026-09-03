# Audio Understanding

Generate text about an audio clip from a free-text prompt: captioning, sound-event
description, and open-ended audio question answering. The response is whatever the
prompt asks for, so transcription is one instruction among many rather than a
separate task.

## Licence — read before use

The only backend is **Audio Flamingo 3** (`nvidia/audio-flamingo-3-hf`), whose weights
are released under the **NVIDIA OneWay Noncommercial License: non-commercial research
use only**. Portions of its training-data generation are additionally subject to the
Qwen Research License and OpenAI's Terms of Use. This is stricter than most weights
senselab loads — it cannot be used in a commercial product. See the
[model card](https://huggingface.co/nvidia/audio-flamingo-3-hf).

## Usage

```python
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.audio_understanding import describe_audios

audio = Audio(filepath="recording.wav")
[caption] = describe_audios(
    [audio],
    prompt="Generate a detailed caption for the input audio.",
)
```

## Constraints

| | |
|---|---|
| Sampling rate | 16 kHz mono; other input is resampled and downmixed automatically |
| Maximum duration | 10 minutes per clip — longer input raises, split it first |
| Internal windowing | the model processes audio in 30 s windows |
| dtype | bfloat16 on CUDA, float32 on CPU |
| Attention | `flash_attention_2` when `flash-attn` is installed on CUDA, otherwise `sdpa` |

Runs in-process — `transformers` ships `AudioFlamingo3ForConditionalGeneration`
natively and senselab already pins `transformers>=5.3`, so no subprocess venv is
needed. An ~8B-parameter model: CUDA is strongly recommended, and weights are cached
per `(model, device, attention)` so repeated calls do not reload them.

vLLM offers a documented 5–7× speedup for this model but requires a prerelease
`transformers>=5.0.0rc1` override and a git build of vLLM, which conflicts with the
pinned environment. Not implemented here.
