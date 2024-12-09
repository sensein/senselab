# Speech to text evaluation


<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/extract_speaker_embeddings.ipynb'">Tutorial</button>


## Overview

Speaker embeddings are fixed-dimensional vector representations that capture the unique characteristics of a speaker's
voice, allowing for tasks such as speaker identification, verification, and diarization.

Speaker embedding extraction is a crucial task in speaker recognition systems. It involves transforming variable-length
audio signals into fixed-size vector representations that encapsulate speaker-specific information while being robust
to variations in speech content, background noise, and recording conditions.

## Model Architecture:
The default model used in this module (speechbrain/spkrec-ecapa-voxceleb) is based on the ECAPA-TDNN architecture,
which has shown strong performance across various speaker recognition tasks.
Other supported models include ResNet TDNN (speechbrain/spkrec-resnet-voxceleb) and
xvector (speechbrain/spkrec-xvect-voxceleb).

**Note**: Performance can vary significantly depending on the specific dataset, task, and evaluation protocol used.
Always refer to the most recent literature for up-to-date benchmarks.

## Learn more:
- [SpeechBrain](https://speechbrain.github.io/)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
- [ResNet TDNN](https://doi.org/10.1016/j.csl.2019.101026)
- [xvector](https://doi.org/10.21437/Odyssey.2018-15)
