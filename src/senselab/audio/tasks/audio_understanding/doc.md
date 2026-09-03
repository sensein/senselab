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

vLLM offers a documented 5–7× speedup for this model. It is not implemented here.

## Reasoning mode (AF-Think)

The base checkpoint is not a reasoning model. Prompts that ask it to think before
answering — the card's `"Please think and reason about the input <media> before you
respond."` — belong to **AF-Think**, a PEFT adapter shipped in the same repository's
`think` subfolder rather than as a separate model. `describe_audios(..., think=True)`
loads it: the extra trainables from `think/non_lora_trainables.bin` go in via
`load_state_dict(strict=False)`, then the LoRA weights via `PeftModel.from_pretrained`.

Sending a reasoning prompt to the base weights is a variant mismatch, so the two are
cached separately.

The transcription checkpoints wrap answers in a fixed `The spoken content of the audio
is "..."` phrasing; `strip_prefix=True` removes it. It is exposed on the processor's
`decode` and not on `batch_decode`, which is why generations are decoded one at a time.

## Running on a cluster

Compute nodes frequently carry no system ffmpeg, and `torchcodec` fails to load
without it, so audio decoding raises before this backend is reached. Install it with
the repo's own script, documented under *System Requirements* in the README — it needs
no root and drops the shared libraries where `torchcodec`'s soname lookup finds them:

```bash
CONDA_PREFIX=~/ffmpeg bash scripts/install-ffmpeg.sh
export LD_LIBRARY_PATH="$HOME/ffmpeg/lib:${LD_LIBRARY_PATH:-}"
```

Prefer that to a cluster module. A module that sets `PATH` only still leaves
`torchcodec` unable to `dlopen` the libraries, and the resulting `libavutil.so.56:
cannot open shared object file` names the symbol rather than the cause.

This is a property of the decoding stack rather than of this backend, so it applies to
any senselab job that reads audio on such a node.
