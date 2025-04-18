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
## Evaluation
### Metrics
### Datasets
### Benchmarks
## Notes
