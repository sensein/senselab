# Forced alignment

<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/forced_alignment.ipynb'">Tutorial</button>

## Task Overview
Task Overview
Forced alignment is the process of aligning units of a transcription (e.g. words, phonemes, characters) with its corresponding audio by determining precise timestamps for each unit​.

Common use cases include:
- **Subtitling and Captioning**: Automatically generating timestamps for subtitles or closed captions in videos​.
- **Linguistic Annotation**: Providing time-aligned transcripts for phonetic research or language documentation​.
- **Dataset Preparation**: Segmenting long audio recordings by sentence or word, which is useful for creating automatic speech recognition (ASR) training datasets or extracting exact utterances​.
- **Other Downstream Tasks**: Enabling applications like word-level audio editing, precise audio search, or improving text-to-speech (TTS) modeling by providing exact durations for each phonetic unit​.

Forced alignment typically uses automatic speech recognition (ASR) models or acoustic models in a constrained decoding mode. Many modern aligners use connectionist temporal classification (CTC) decoding or similar sequence alignment algorithms to obtain token timing. In CTC alignment, the ASR model’s output probabilities (including “blank” tokens for silence) are used to find the most likely alignment for the reference text​. Other systems use sequence alignment techniques (e.g. Viterbi algorithm in an hidden Markov model) to force-match the audio frames to the given word sequence​.

## Models
Several types of models and tools are used for forced alignment, each with different strengths:

- **CTC-Based Models** (e.g., Wav2Vec2, Massively Multilingual Speech (MMS), NeMo Forced Aligner (NFA)): These models align transcript tokens to audio using CTC decoding. They are efficient and support long audio, with many hosted on Hugging Face. They output character or token timestamps, requiring aggregation for word-level alignment.

- **Dedicated Aligners** (e.g., Montreal Forced Aligner (MFA)):
Use traditional acoustic models (e.g., GMM-HMM triphone models) and pronunciation lexicons to align transcripts at the word or phoneme level.

## Evaluation
### Metrics
Forced alignment quality is typically assessed using:

- **Alignment Error Rate (AER)**
  Measures alignment accuracy by comparing predicted and reference word-to-time mappings. Defined as `1 - F1 score`, it penalizes both false and missed alignments. Lower is better.

- **Mean Absolute Error (MAE)**
  Average absolute timing difference (in seconds or ms) between predicted and reference boundaries. Sensitive to outliers. Lower values indicate tighter alignment.

- **Boundary Deviation**
  Often synonymous with MAE. Sometimes reported as mean/median boundary error or success rate within a tolerance (e.g., % within ±20ms).

- **Other Metrics**
  - **F1@tolerance**: Combines precision/recall for boundaries within a given window.
  - **Phone Error Rate (PER)**: At phoneme level—counts insertions, deletions, substitutions.

### Datasets

Forced alignment models are evaluated on datasets with trusted word or phoneme timestamps:

- **TIMIT**
  Classic corpus with precise phoneme and word boundaries. Ideal for phoneme-level evaluation under clean, read-speech conditions. High-quality aligners often reach ~20ms MAE.

- **LibriSpeech Alignments**
  Audiobook dataset with full alignments from MFA released for all 980 hours of LibriSpeech. Tests scalability and robustness on real speech.

- **Common Voice**
  Multilingual, crowd-sourced speech with transcripts. Often used to test multilingual alignment. Lacks gold standard timestamps but helps evaluate performance on diverse accents and languages.

- **Buckeye Corpus**
  Conversational English with detailed alignments. Harder than TIMIT due to casual speech and disfluencies. Used to test aligners on natural dialogue.

### Benchmarks
## Notes
