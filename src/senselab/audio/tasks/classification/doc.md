# Audio Classification


## Task Overview
Audio Classification refers to the broad category of tasks of processing audio inputs and assigning a label over a set of possible classes. The technology has a wide range applications, including identifying speaker intent, language classification, and even animal species by their sounds.


## Models
A variety of models are supported by ```senselab``` for audio classification tasks. They can be explored on the [Hugging Face Hub](https://huggingface.co/models?library=transformers&pipeline_tag=audio-classification&sort=downloads). Each model varies in performance, size, license, language support, and more. Like with many models and tasks, performance may also vary depending on who the speaker is in the processed audio clips (there may be differences in terms of age, dialects, disfluencies). It is recommended to review the model card for each model before use. Also, always refer to the most recent literature for an informed decision. Unlike other tasks, the exact classification task will be based off of the dataset and class labels used to train the model, such that even two models using the same dataset might classify the audios into different categories (e.g. the LibriSpeech dataset could be used to classify age and/or gender based on the audio clips).

Some popular models for different classifications include:

**Auditory Scene / Sound Event Classification:**
  - [Audio Spectrogram Transformer (AST)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) — 521 AudioSet classes (speech, music, environmental sounds, animals, etc.). Recommended for general-purpose scene analysis.
  - [YAMNet](https://huggingface.co/google/yamnet) — Google's AudioSet classifier (521 classes). TensorFlow-based; not directly supported via `classify_audios` (requires TF runtime).

**Speaker / Demographics:**
  - [Age](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
  - [Gender](https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech)
  - [Adult vs. Child Speech](https://huggingface.co/bookbot/wav2vec2-adult-child-cls)

**Content:**
  - [Music Genre](https://huggingface.co/agercas/distilhubert-finetuned-gtzan)
  - [Emotion Recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)

## Evaluation
### Metrics
The primary evaluation metric for audio classification is accuracy. This can be unweighted (the number of correct labels over the total number of labels) or weighted (where each class is weighted accordingly to how much of the whole dataset the class label represents).

Other important metrics include:
- **Area Under the Curve (AUC)**
- **Precision**
- **Recall**
- **F1-Score**

More information about the different metrics related to classification can be found [here](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) as well as a common function for understanding the results of a classification can be found in [scikit-learn](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html).

### Datasets
The following table lists the datasets included in the [English Speech Benchmark (ESB)](https://arxiv.org/abs/2210.13352), which are generally used for evaluating ASR models in English:

| Dataset                             | Domain                  | Amount of Data (h)  |
|-------------------------------------|-------------------------|---------------------|
| AudioSet                            | Audio Events/Ontology   | 5.8 thousand |
| ICBHI Respiratory Sound Database    | Medical/Respiratory     | 5.5            |
| ESC-50                              | Environmental Audio     | 2.77 |
| MSP Podcast      | Speech Emotion Recognition     | 238             |

Note that this list of datasets is not exhaustive. If you are interested in benchmarking models in different languages or under specific conditions, consult the relevant literature.

## Windowed Scene Classification

### Overview
`classify_audios(..., win_length=1.0)` applies a sliding-window approach to audio classification, enabling temporal analysis of how the acoustic scene evolves over time. Instead of producing a single label for an entire recording, it slices the audio into overlapping windows using `Audio.window_generator()` and classifies each one independently.

### Default Parameters
- **win_length**: Window duration in seconds (e.g., 1.0).
- **hop_length**: Hop duration in seconds. Defaults to `win_length / 2` (50% overlap).
- **top_k**: 5 — the number of top-scoring labels retained per window.

Windows are processed in batches of 32 for memory efficiency. If the audio is shorter than a single window, the full audio is used as one window.

### Recommended Models
- **[MIT/ast-finetuned-audioset-10-10-0.4593](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)** — Audio Spectrogram Transformer fine-tuned on AudioSet with 521 event classes (speech, music, environmental sounds, etc.). This is the recommended default for general-purpose auditory scene analysis.

### Visualization
Use `scene_results_to_segments` to convert the per-window results into segment dicts compatible with `plot_aligned_panels`:

```python
from senselab.audio.tasks.classification import classify_audios, scene_results_to_segments
from senselab.audio.tasks.plotting.plotting import plot_aligned_panels
from senselab.utils.data_structures import HFModel

model = HFModel(path_or_uri="MIT/ast-finetuned-audioset-10-10-0.4593")
results = classify_audios([audio], model=model, win_length=1.0)
segments = scene_results_to_segments(results[0])

plot_aligned_panels(audio, [
    {"type": "waveform"},
    {"type": "spectrogram", "mel": False},
    {"type": "segments", "segments": segments},
], title="Auditory Scene Analysis")
```
