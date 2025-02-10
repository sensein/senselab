# Speech Emotion Recognition


<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/speech_emotion_recognition.ipynb'">Tutorial</button>


## Task Overview
Speech Emotion Recognition (SER) is the process of recognizing emotion from audio clips, typically of speech. This technology has a wide range of applications, from call center monitoring to possible therapeutic interventions.

SER is typically split between two subcategories: discrete emotion classification and dimensional emotion attribute recognition/approximation. The former is the more common activity and aligns well with the general task of audio classification: given a set of labels, can we accurately attribute one or more of the labels to an audio. For these, most datasets will use [Ekman's Big 6 emotions](https://www.paulekman.com/universal-emotions/) (and neutrality) or some subset of them: happiness, anger, fear, sadness, disgust, and surprise. The latter subtask in SER tries to break emotionality into different continuous attributes (typically dominance, arousal, and valence) which can be estimated from audio clips and for which then different ranges of values combine to create what we consider to be a single emotional label.


## Models
A variety of models are supported by ```senselab``` for SER tasks. They can be explored on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=audio-classification&library=transformers&sort=downloadss) and using search terms like "SER" or "emotion". Each model varies in performance, size, license, language support, and more. Performance may also vary depending on who is the speaker in the processed audio clips (there may be differences in terms of age, dialects, disfluencies). It is recommended to review the model card for each model before use. Also, always refer to the most recent literature for an informed decision.

Most major models for SER currently focus on taking large, pre-trained models and then fine-tuning those towards specific tasks. Below we list 3 of these models with some examples that others have fine-tuned for the SER task.

- **[WavLM](https://arxiv.org/abs/2110.13900)**
Developed by Microsoft researchers in 2022 and still remains one of the top self-supervised audio models on the [SUPERB benchmark](https://superbbenchmark.github.io/#/).
  - [3loi/SER-Odyssey-Baseline-WavLM-Categorical](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Categorical): Categorical WavLM model trained on MSP-Podcast
  - [3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes): Dimensional WavLM model trained on MSP-Podcast
- **[wav2vec2](https://arxiv.org/abs/2006.11477)**
A predecessor to WavLM, wav2vec2 was developed as a framework for creating self-supervised speech representations. While older than WavLM, it still performs quite well on multiple benchmarks.
  - [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition): the XLSR refers to Cross-lingual Representation Learning, a more modern Wav2Vec2 model that performs especially well across languages
  - [audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim): Wav2Vec2 dimensional recognition model trained on MSP-Podcast
- **(HuBERT)[https://arxiv.org/abs/2106.07447]**
Hidden-Unit BERT is a self-supervised speech representation learning approach that takes a BERT-like preidction loss approach.
  - [superb/hubert-large-superb-er](https://huggingface.co/superb/hubert-large-superb-er): The official SUPERB model submission for emotion recognition.


## Evaluation
### Metrics
The primary evaluation metric for audio classification, including SER, is accuracy. This can be unweighted (the number of correct labels over the total number of labels) or weighted (where each class is weighted accordingly to how much of the whole dataset the class label represents).

Other important metrics include:
- **Area Under the Curve (AUC)**
- **Precision**
- **Recall**
- **F1-Score**

More information about the different metrics related to classification can be found [here](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall) as well as a common function for understanding the results of a classification can be found in [scikit-learn](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.classification_report.html). For dimensional SER, mean-squared error is a typical metric since models are trying to minimize the distance between their predicted value and a ground-truth human evaluation for each dimension.


### Datasets
The following table lists the datasets used for training and testing the performance of SER models. Datasets come from a variety of languages, some even are meant to test multi-lingual model capabilities, as well as they vary widely in terms of the actors that were included, the actors' proficientry, the total number of emotional labels used, and more. Of notable importance is the "truthfulness" of the emotions and the setup procedure. Many papers continuously find that performance on SER tasks shows a sharp dropoff when switching between corpora and domains. One possible explanation for this is that most datasets are not "true" expressions of emotions, but rather come from acted scenarios where the actors (possibly professional and/or amateur) will simulate or induce the requested emotion. A drawback of this procedure is that it is believed that this causes over-emphasis on certain characteristics for different emotions that deep learning models will pickup on and use that don't occur in more naturalistic settings, causing a performance drop in real-world scenarios. This remains an open and debated problem within the literature.

| Dataset        | Language                  | Actors' demographics  | Actors' proficiency | # of Emotions | Truthfulness of emotions         | Setup |
|----------------|-------------------------|---------------------|-----------|---------|----------|------------------------|--------------------|
| ASED    | Amharic               | 40M, 25F            | Professional, semi-professional, amateur       | 4 + neutrality   | Simulated              | Controlled Setting         |
| Belfast Database | English               | 100 speakers            | Amateur      | 24      | Induced      | Controlled setting            |
| CASIA      | Mandarin     | 4M, 4F             | Professional       | 5 + neutrality       | Simulated            | Controlled setting           |
| CHEAVD       | Mandarin               | 125M, 113F             | Professional & amateur       | 26      | Simulated and naturalistic          | Controlled and uncontrolled setting   |
| DEMoS     | Italian | 45M, 23F | Amateur | 7 + neutrality      | Induced        | Controlled setting       |
| DES     | Danish     | 2M, 2F| Semi--professional      | 4 + neutrality     | Simulated     | Controlled setting     |
| EMODB    | German     | 5M, 5F| Professional       | 6 + neutrality       | Simulated      | Controlled setting      |
| Emovo            | Italian                | 3M, 3F         | Professional        | 6 + neutrality       | Simulated      | Controlled setting          |
| Emozionalmente            | Italian                | 131M, 299F         | Professional & amateur        | 6 + neutrality       | Simulated    | Uncontrolled setting         |
| eNTERFACE            | English                | 34M, 8F         | Amateur        | 6       | Induced      | Controlled setting         |
| EmoMatchSpanishDB            | Spanish                | 30M, 20F         | Amateur        | 6 + neutrality       | Induced        | Controlled setting       |
| €motion            | English, French, German, Italian                | 39M         | Professional and amateur        | 6 + neutrality       | Simulated & induced   | Controlled setting        |
| IEMOCAP            | English                | 5M, 5F         | Professional        | 3 + neutrality       | Simulated    | Controlled setting          |
| INTERFACE            | English, French, Slovenian, Spanish                | 2M, 1F (English), 1M, 1F (others)         | Professional        | 6 + neutrality       | Simulated    | Controlled setting         |
| MESD            | Spanish                | 4M, 4D (adults), 3M, 5F (children)         | Amateur        | 5 + neutrality       | Simulated      | Controlled setting         |
| MSP-Podcast            | English                | > 1400 speakers         | Professional and amateur        | 7 + neutrality + other       | Naturalistic     | Controlled and uncontrolled setting         |
| RAVDESS            | English                | 12M, 12F         | Professional        | 7 + neutrality       | Simulated     | Controlled setting        |
| RUSLANA            | Russian                | 12M, 49F         | Amateur        | 5 + neutrality       | Simulated     | Controlled setting        |
| SAVEE            | English                | 4M         | Amateur        | 6 + neutrality       | Simulated       | Controlled setting         |
| SUBESCO            | Bangla                | 10M, 10F         | Professional        | 6 + neutrality       | Simulated      | Controlled setting          |


Note that this list of datasets is not exhaustive. If you are interested in benchmarking models in different languages or under specific conditions, consult the relevant literature.

### Benchmark
One of the leading benchmarks for SER comes from the creators of the [SUPERB challenge](https://superbbenchmark.github.io/#/), which benchmarks different self-supervised models and how well they perform on downstream tasks (one of which being emotion recognition). [EMO-SUPERB](https://emosuperb.github.io/index.html) tests models on different available datasets and benchmarks them as an average of their overall performance. Rather than just using one or two similar datasets, this benchmark includes 9 datasets that span some of the different combinations of attributes that were described in the previous table. These datasets help test whether models might one day generalize to novel contexts and out-of-distribution scenarios.
