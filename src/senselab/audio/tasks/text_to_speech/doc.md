# Text to Speech

<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/text_to_speech.ipynb'">Tutorial</button>

## Task Overview

Text-to-speech (TTS) is the task of creating natural-sounding speech from text. This process can be performed in multiple languages and for multiple speakers.

## Models

A variety of models are supported by `senselab` for text-to-speech.
Each model varies in performance, size, license, language support, and more. Performance may also vary depending, among other reasons, on the length of the text or the target speaker (differences in terms of age, dialects, disfluencies). It is recommended to review the model card for each model before use and refer to the most recent literature for an informed decision.

Several text-to-speech models are currently available through `🤗 Transformers`. These models can be explored on the [Hugging Face Hub](https://huggingface.co/models?library=transformers&pipeline_tag=text-to-speech&sort=downloads).
**Note**: Some `Hugging Face` models, despite having the `text-to-speech` label on their model cards, may not work with the text-to-speech pipeline. These models are not supported in `senselab`, and identifying them often requires trial and error.

In addition to the models from 🤗 Transformers, senselab also supports `coqui-tts`, which enable text-to-speech generation (sometimes using a specific target voice accompanied by its corresponding transcript). Voice cloning using a target voice refers to the process of creating a synthetic voice that mimics the characteristics of a specific person's voice, known as the target voice. This involves generating speech that sounds like it was spoken by that person, even though it was produced by a machine.

Popular/recommended models include:
- **[Bark](https://huggingface.co/docs/transformers/model_doc/bark)**
  - [small](https://huggingface.co/suno/bark-small)
  - [standard](https://huggingface.co/suno/bark)
- **[MMS](https://huggingface.co/docs/transformers/model_doc/mms)**
  - [small](https://huggingface.co/facebook/mms-300m)
  - [large](https://huggingface.co/facebook/mms-1b-all)
- **[SpeechT5](https://huggingface.co/docs/transformers/model_doc/speecht5)**
  - [standard](https://huggingface.co/microsoft/speecht5_tts)
- **[Coqui-tts](https://github.com/idiap/coqui-ai-TTS)**
  - [models](https://github.com/idiap/coqui-ai-TTS/blob/dev/TTS/.models.json)

### Qwen3-TTS

[`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
(licence: apache-2.0, a real permissive licence — unlike DriftSE and unasdiff, this model
needs no `sensein` weights mirror and loads straight from the Hub) runs through a
dedicated subprocess venv (`senselab.audio.tasks.text_to_speech.qwen_tts`), reached by
`synthesize_texts` whenever the model id starts with `Qwen/Qwen3-TTS`:

```python
from senselab.audio.tasks.text_to_speech import synthesize_texts
from senselab.audio.tasks.text_to_speech.qwen_tts import supported_speakers
from senselab.utils.data_structures import HFModel

model = HFModel(path_or_uri="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
supported_speakers(model)  # -> ['aiden', 'dylan', 'eric', 'ono_anna', 'ryan', 'serena', 'sohee', 'uncle_fu', 'vivian']

audios = synthesize_texts(
    texts=["She said she would be here by noon."],
    model=model,
    speaker="Ryan",
    language="English",       # or "Auto"
    instruct="Very happy.",   # optional style control
)
```

**Named speakers, not voice cloning.** The checkpoint bakes in 9 speaker identities
directly in its config (`talker_config.spk_id`), selectable by name via
`generate_custom_voice`, with no reference audio required. `supported_speakers()`
reads that mapping straight from `config.json` — a single small file — rather than
downloading the full 1.7B-parameter checkpoint or spinning up the subprocess venv just
to enumerate names. This is what makes the model useful as a speech source for
multi-speaker synthetic sessions with exact ground-truth speaker identity (e.g. the
speaker-ceiling probe): distinct named voices are generated directly instead of cloned
from reference clips.

Not part of any default model list and not wired into any pipeline — reachable only by
naming a `Qwen/Qwen3-TTS...` model id explicitly. See the module's own docstring for the
CUDA-wheel pinning rationale, the upstream package's import chain, and a documented
partial-pin gap in the third-party wrapper's own revision handling.

## Evaluation
### Metrics

For assessing speech quality and intelligibility, we can use quantitative metrics such as:
- **Wideband Perceptual Estimation of Speech Quality (PESQ)**
- **Short-Time Objective Intelligibility (STOI)**
- **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)**

and qualitative metrics such as:
- **Mean Opinion Score (MOS)**

Another way to automatically assess the **intelligibility** of the synthesis is by transcribing the output audio (trusting the ASR system) and computing the **Word Error Rate (WER)** with the reference text.

Also, if targeting a specific speaker's voice, we can perform **speaker verification** to assess how closely the generated audio matches the target voice.
If there are specific **features** in the target voice that we aim to maintain, we can extract these features from the generated audio and verify their presence.

`senselab` can help with all of these evaluations.

### Datasets

To train and evaluate TTS models, a variety of datasets can be used. Some popular datasets include:

- **[LJSpeech](https://keithito.com/LJ-Speech-Dataset/)**: A dataset of single-speaker English speech.
- **[LibriTTS](https://openslr.org/60/)**: A multi-speaker English dataset derived from the LibriVox project.
- **[VCTK](https://datashare.ed.ac.uk/handle/10283/2651)**: A multi-speaker English dataset with various accents.
- **[Common Voice](https://commonvoice.mozilla.org/)**: A multi-language dataset collected by Mozilla.

### Benchmark
The [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) ranks and evaluates text-to-speech models available based on human perception.
For automated benchmarking, we recommend using standard datasets and metrics mentioned above.
