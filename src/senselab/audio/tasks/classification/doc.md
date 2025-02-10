# Audio Classification


## Task Overview
Audio Classification refers to the broad category of tasks of processing audio inputs and assigning a label over a set of possible classes. The technology has a wide range applications, including identifying speaker intent, language classification, and even animal species by their sounds.


## Models
A variety of models are supported by ```senselab``` for audio classification tasks. They can be explored on the [Hugging Face Hub](https://huggingface.co/models?library=transformers&pipeline_tag=audio-classification&sort=downloads). Each model varies in performance, size, license, language support, and more. Like with many models and tasks, performance may also vary depending on who the speaker is in the processed audio clips (there may be differences in terms of age, dialects, disfluencies). It is recommended to review the model card for each model before use. Also, always refer to the most recent literature for an informed decision. Unlike other tasks, the exact classification task will be based off of the dataset and class labels used to train the model, such that even two models using the same dataset might classify the audios into different categories (e.g. the LibriSpeech dataset could be used to classify age and/or gender based on the audio clips).

Some popular models for different classifications include:
  - [Age](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
  - [Gender](https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech)
  - [Music Genre](https://huggingface.co/agercas/distilhubert-finetuned-gtzan/blob/main/config.json)
  - [Adult vs. Child Speech](https://huggingface.co/bookbot/wav2vec2-adult-child-cls)
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
