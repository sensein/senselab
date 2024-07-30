# Speech to text


<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/speech_to_text.ipynb'">Tutorial</button>


## Task Overview
Speech-to-Text (STT), also known as Automatic Speech Recognition (ASR), is the process of converting spoken language into written text. This technology has a wide range of applications, including transcription services and voice user interfaces.

Notably, certain models can provide word- or sentence-level timestamps along with the transcribed text, making them ideal for generating subtitles. Additionally, some models are multilingual, and some of them leverage language identification blocks to enhance performance.


## Models
A variety of models are supported by ```senselab``` for ASR tasks. They can be explored on the [Hugging Face Hub](https://huggingface.co/models?library=transformers&pipeline_tag=automatic-speech-recognition&sort=downloads). Each model varies in performance, size, license, language support, and more. Performance may also vary depending on who is the speaker in the processed audio clips (there may be differences in terms of age, dialects, disfluencies). It is recommended to review the model card for each model before use. Also, always refer to the most recent literature for an informed decision.

Popular models include:
- **Whisper**
  - [tiny](https://huggingface.co/openai/whisper-tiny)
  - [small](https://huggingface.co/openai/whisper-small)
  - [medium](https://huggingface.co/openai/whisper-medium)
  - [large](https://huggingface.co/openai/whisper-large-v3)

- **Massively Multilingual Speech**
  - [MMS 1b](https://huggingface.co/facebook/mms-1b-all)

- **Massively Multilingual and Multimodal Machine Translation**
  - [Seamless small](https://huggingface.co/facebook/seamless-m4t-unity-small-s2t)
  - [Seamless medium](https://huggingface.co/facebook/hf-seamless-m4t-medium)
  - [Seamless large](https://huggingface.co/facebook/seamless-m4t-v2-large)

## Evaluation
### Metrics
The primary evaluation metric for ASR systems is the Word Error Rate (WER). WER is calculated as:

    WER = (Substitutions + Insertions + Deletions) / Number of words in the reference

where:
- **Substitutions**: Incorrect words.
- **Insertions**: Extra words added.
- **Deletions**: Words omitted.

Other important metrics include:
- **Character Error Rate (CER)**
- **Match Error Rate (MER)**
- **Word Information Lost (WIL)**
- **Word Information Preserved (WIP)**

For detailed information on these metrics, refer to the [speech to text evaluation module](speech_to_text_evaluation).

### Datasets (English Speech Benchmark - ESB)
The following table lists the datasets included in the [English Speech Benchmark (ESB)](https://arxiv.org/abs/2210.13352), which are generally used for evaluating ASR models in English:

| Dataset        | Domain                  | Speaking Style      | Train (h) | Dev (h) | Test (h) | Transcriptions         | License            |
|----------------|-------------------------|---------------------|-----------|---------|----------|------------------------|--------------------|
| LibriSpeech    | Audiobook               | Narrated            | 960       | 11      | 11       | Normalized             | CC-BY-4.0          |
| Common Voice 9 | Wikipedia               | Narrated            | 1409      | 27      | 27       | Punctuated & Cased     | CC0-1.0            |
| VoxPopuli      | European Parliament     | Oratory             | 523       | 5       | 5        | Punctuated             | CC0                |
| TED-LIUM       | TED talks               | Oratory             | 454       | 2       | 3        | Normalized             | CC-BY-NC-ND 3.0    |
| GigaSpeech     | Audiobook, podcast, YouTube | Narrated, spontaneous | 2500 | 12      | 40       | Punctuated             | Apache-2.0         |
| SPGISpeech     | Financial meetings      | Oratory, spontaneous| 4900      | 100     | 100      | Punctuated & Cased     | User Agreement     |
| Earnings-22    | Financial meetings      | Oratory, spontaneous| 105       | 5       | 5        | Punctuated & Cased     | CC-BY-SA-4.0       |
| AMI            | Meetings                | Spontaneous         | 78        | 9       | 9        | Punctuated & Cased     | CC-BY-4.0          |

For more details on these datasets and how models are evaluated to obtain the ESB score, refer to the ESB paper.
Note that this list of datasets is not exhaustive. If you are interested in benchmarking models in different languages or under specific conditions, consult the relevant literature.

### Benchmark
The [🤗 Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) ranks and evaluates speech recognition models available on the Hugging Face Hub. The leaderboard uses the datasets included in the ESB paper to obtain robust evaluation scores for each model. The ESB score is a macro-average of the WER scores across the ESB datasets, providing a comprehensive indication of a model's performance across various domains and conditions.

## Notes
- It is possible to fine-tune foundational speech models on a specific language without requiring large amounts of data. A detailed blog post on how to fine-tune a pre-trained Whisper checkpoint on labeled data for ASR can be found [here](https://huggingface.co/blog/fine-tune-whisper).

Learn more:
- [Whisper](https://arxiv.org/abs/2212.04356)
- [MMS](https://arxiv.org/abs/2305.13516)
- [Seamless MMS](https://arxiv.org/abs/2308.11596)
- [Wav2vec 2.0](https://arxiv.org/abs/2006.11477)
