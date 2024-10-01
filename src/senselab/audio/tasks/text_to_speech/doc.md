# Text to Speech

<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/text_to_speech.ipynb'">Tutorial</button>

## Task Overview

Text-to-speech (TTS) is the task of creating natural-sounding speech from text. This process can be performed in multiple languages and for multiple speakers.

## Models

A variety of models are supported by `senselab` for text-to-speech.
Each model varies in performance, size, license, language support, and more. Performance may also vary depending, among other reasons, on the length of the text or the target speaker (differences in terms of age, dialects, disfluencies). It is recommended to review the model card for each model before use and refer to the most recent literature for an informed decision.

Several text-to-speech models are currently available through `🤗 Transformers`. These models can be explored on the [Hugging Face Hub](https://huggingface.co/models?library=transformers&pipeline_tag=text-to-speech&sort=downloads).
**Note**: Some `Hugging Face` models, despite having the `text-to-speech` label on their model cards, may not work with the text-to-speech pipeline. These models are not supported in `senselab`, and identifying them often requires trial and error.

In addition to the models from 🤗 Transformers, senselab also supports Mars5-TTS, which enables text-to-speech generation using a specific target voice, accompanied by its corresponding transcript. Voice cloning using a target voice refers to the process of creating a synthetic voice that mimics the characteristics of a specific person's voice, known as the target voice. This involves generating speech that sounds like it was spoken by that person, even though it was produced by a machine.

Popular/recommended models include:
- **[Bark](https://huggingface.co/docs/transformers/model_doc/bark)**
  - [small](https://huggingface.co/suno/bark-small)
  - [standard](https://huggingface.co/suno/bark)
- **[MMS](https://huggingface.co/docs/transformers/model_doc/mms)**
  - [small](https://huggingface.co/facebook/mms-300m)
  - [large](https://huggingface.co/facebook/mms-1b-all)
- **[SpeechT5](https://huggingface.co/docs/transformers/model_doc/speecht5)**
  - [standard](https://huggingface.co/microsoft/speecht5_tts)
- **[Mars5-TTS](https://github.com/Camb-ai/MARS5-TTS)**
  - [mars5_english](https://huggingface.co/CAMB-AI/MARS5-TTS)

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
