# Forced alignment

<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/forced_alignment.ipynb'">Tutorial</button>

## Task Overview
Task Overview
[Forced alignment](https://montreal-forced-aligner.readthedocs.io/en/stable/user_guide/index.html) is the process of aligning units of a transcription (e.g. words, phonemes, characters) with its corresponding audio by determining precise timestamps for each unit​.

Common use cases include:
- **Subtitling and Captioning**: Automatically generating timestamps for subtitles or closed captions in videos​.
- **Linguistic Annotation**: Providing time-aligned transcripts for phonetic research or language documentation​.
- **Dataset Preparation**: Segmenting long audio recordings by sentence or word, which is useful for creating automatic speech recognition (ASR) training datasets or extracting exact utterances​.
- **Other Downstream Tasks**: Enabling applications like word-level audio editing, precise audio search, or improving text-to-speech (TTS) modeling by providing exact durations for each phonetic unit​.

[Forced alignment typically uses automatic speech recognition (ASR)](https://research.nvidia.com/labs/conv-ai/blogs/2023/2023-08-forced-alignment/#the-naive-way-but-listing-all-the-possible-paths-using-a-graph) models or acoustic models in a constrained decoding mode. Many modern aligners use connectionist temporal classification (CTC) decoding or similar sequence alignment algorithms to obtain token timing. In CTC alignment, the ASR model’s output probabilities (including “blank” tokens for silence) are used to find the most likely alignment for the reference text​. Other systems use sequence alignment techniques (e.g. Viterbi algorithm in an hidden Markov model) to force-match the audio frames to the given word sequence​.

## Models
Several types of models and tools are used for forced alignment, each with different strengths:

- **CTC-Based Models** (e.g., [Wav2Vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2), [Massively Multilingual Speech (MMS)](https://huggingface.co/docs/transformers/en/model_doc/mms), [NeMo Forced Aligner (NFA)](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tools/nemo_forced_aligner.html)): These models align transcript tokens to audio using CTC decoding. They are efficient and support long audio, with many hosted on Hugging Face. They output character or token timestamps, requiring aggregation for word-level alignment.

- **Dedicated Aligners** (e.g., [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/en/latest/)):
Use traditional acoustic models (e.g., GMM-HMM triphone models) and pronunciation lexicons to align transcripts at the word or phoneme level.

## Evaluation
### Metrics
Forced alignment quality is typically assessed using:

- **[Alignment Error Rate (AER)](https://www.nltk.org/howto/align.html)**
  Measures alignment accuracy by comparing predicted and reference word-to-time mappings. Defined as `1 - F1 score`, it penalizes both false and missed alignments. Lower is better.

- **Mean Absolute Error (MAE)**
  Average absolute timing difference (in seconds or ms) between predicted and reference boundaries. Sensitive to outliers. Lower values indicate tighter alignment.

- **Boundary Deviation**
  Often synonymous with MAE. Sometimes reported as mean/median boundary error or success rate within a tolerance (e.g., % within ±20ms).

- **Other Metrics**
  - **F1@tolerance**: Combines precision/recall for boundaries within a given window.
  - **[Phone Error Rate (PER)](https://montreal-forced-aligner.readthedocs.io/en/v2.2.17/user_guide/implementations/alignment_evaluation.html)**: At phoneme level—counts insertions, deletions, substitutions.

### Datasets

Forced alignment models are evaluated on datasets with trusted word or phoneme timestamps:

- **[TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)**
  Classic corpus with precise phoneme and word boundaries. Ideal for phoneme-level evaluation under clean, read-speech conditions. High-quality aligners often reach ~20ms MAE.

- **[LibriSpeech Alignments](https://zenodo.org/records/2619474)**
  Audiobook dataset with full alignments from MFA released for all 980 hours of LibriSpeech. Tests scalability and robustness on real speech.

- **[Common Voice](https://commonvoice.mozilla.org/en/datasets)**
  Multilingual, crowd-sourced speech with transcripts. Often used to test multilingual alignment. Lacks gold standard timestamps but helps evaluate performance on diverse accents and languages.

- **[Buckeye Corpus](https://buckeyecorpus.osu.edu/)**
  Conversational English with detailed alignments. Harder than TIMIT due to casual speech and disfluencies. Used to test aligners on natural dialogue.

### [Benchmarks](https://arxiv.org/html/2406.19363v1)


#### Table 1: TIMIT Word-Level Alignment Accuracy (%)
**Correctly detected word boundaries within time thresholds (ms)**

| System     | ≤10ms | ≤25ms | ≤50ms | ≤100ms |
|------------|-------|--------|--------|---------|
| MFA        | 41.6  | 72.8   | 89.4   | 97.4    |
| MMS        | 18.6  | 43.5   | 75.7   | 94.7    |
| WhisperX   | 22.4  | 52.7   | 82.4   | 94.2    |

---

#### Table 2: Buckeye Word-Level Alignment Accuracy (%)
**Correctly detected word boundaries within time thresholds (ms); `Thresh500` indicates alignments within ±500ms**

| System     | Thresh500 | ≤10ms | ≤25ms | ≤50ms | ≤100ms |
|------------|-----------|--------|--------|--------|---------|
| MFA        | -         | 39.8   | 69.9   | 84.9   | 91.8    |
| MMS        | -         | 25.0   | 52.7   | 75.0   | 87.9    |
| WhisperX   | -         | 18.8   | 43.1   | 67.4   | 77.4    |
| MFA        | +         | 41.1   | 72.2   | 87.6   | 94.8    |
| MMS        | +         | 25.8   | 54.2   | 77.2   | 90.5    |
| WhisperX   | +         | 22.8   | 52.3   | 81.8   | 93.9    |

---

#### Table 3: Alignment Shift and F1@20ms
**Mean/median boundary shift (ms) and F1-score within 20ms window**

| Method     | Dataset  | Thresh500 | Level | Mean   | Median | F1@20 |
|------------|----------|-----------|--------|--------|--------|--------|
| MFA        | TIMIT    | -         | phon   | 133.4  | 12.5   | 66.0   |
| MFA        | Buckeye  | -         | phon   | 1085.9 | 15.9   | 56.2   |
| MFA        | TIMIT    | -         | word   | 21.9   | 12.5   | 65.7   |
| MMS        | TIMIT    | -         | word   | 68.5   | 29.3   | 35.4   |
| WhisperX   | TIMIT    | -         | word   | 34.3   | 23.5   | 43.5   |
| MFA        | Buckeye  | -         | word   | 976.5  | 13.6   | 63.4   |
| MMS        | Buckeye  | -         | word   | 208.3  | 23.1   | 45.0   |
| WhisperX   | Buckeye  | -         | word   | 11685.3| 30.1   | 35.6   |
| MFA        | Buckeye  | +         | word   | 27.8   | 12.9   | 65.4   |
| MMS        | Buckeye  | +         | word   | 41.0   | 22.2   | 46.3   |
| WhisperX   | Buckeye  | +         | word   | 36.4   | 23.7   | 43.3   |

---

#### Table 4: Phone-Level Alignment Accuracy (%) — MFA Only
**Percentage of correctly detected phone boundaries within time thresholds**

| Dataset  | ≤10ms | ≤25ms | ≤50ms | ≤100ms |
|----------|--------|--------|--------|---------|
| TIMIT    | 38.6   | 72.3   | 81.1   | 84.6    |
| Buckeye  | 35.3   | 60.6   | 68.9   | 72.7    |

## Notes

- **Fine-tuning**: Many aligners (e.g., MFA, Wav2Vec2) support retraining on domain-specific data, improving accuracy for accents, jargon, or speaker-specific speech.

- **Multilingual support**: Tools like WhisperX and MMS support many languages out-of-the-box. MFA supports multiple languages but requires language-specific models and lexicons.

- **Transcript and audio quality**: Accurate, verbatim transcripts and clean, single-speaker audio are essential. Major transcript mismatches or overlapping speech can cause alignment failures.

- **Audio preprocessing**: Most aligners expect 16kHz mono audio. Segment long recordings using voice activity detection (VAD) to improve alignment stability and speed.

- **Phoneme alignment**: Lexicon-based aligners like MFA require pronunciation dictionaries. End-to-end models don’t, but won’t produce phoneme-level alignments unless adapted.

- **Performance**: CTC models are fast and GPU-friendly. MFA can be slower on large datasets but offers high precision. Always verify output visually or by spot-checking timestamps.
