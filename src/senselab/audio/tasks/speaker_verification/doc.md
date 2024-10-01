# Speaker Verification
Last updated: 08/13/2024

<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/speaker_verification.ipynb'">Tutorial</button>

## Task Overview
Speaker verification is identity authentication based on voice features.

This technology is widely used in various applications, including security systems, authentication processes, and personalized user experiences. The core concept revolves around comparing voice characteristics extracted from speech samples to verify the identity of the speaker.

SenseLab speaker verification extracts audio embeddings, finds their cosine similarity, and uses a similarity threshold to determine if two audio files came from the same speaker.

## Models
SenseLab speaker verification extracts audio embeddings using ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`). This model is part of [SpeechBrain](https://huggingface.co/speechbrain), "an open-source and all-in-one conversational AI toolkit based on PyTorch." This model was trained on  [Voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [Voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html), celebrity voice datasets. It is important to ensure that the audio samples used for verification have a sampling rate of 16kHz, as this is the rate that `speechbrain/spkrec-ecapa-voxceleb` was trained on.

- **ECAPA-TDNN**
    - [spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)


## Evaluation
### Metrics
The primary evaluation metric for speaker verification is the Equal Error Rate [EER](https://www.sciencedirect.com/topics/computer-science/equal-error-rate), which is the percent error when the false acceptance rate (FAR) and false rejection rate (FRR) are equal.
- **False Acceptance Rate (FAR)** The probability of a verification system accepting invalid inputs. Similar names may refer to this same metric.
- **False Rejection Rate (FRR)** The probability of a verification system rejecting valid inputs. Similar names may refer to this same metric.

Lower values on these metrics indicate better performance.

### Datasets
Common datasets used for evaluating speaker verification models include:
- [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html): A large-scale speaker verification dataset containing over 100,000 utterances from 1,251 celebrities.

Verification Split
|                 | Dev        | Test     |
|-----------------|------------|----------|
| # of speakers   | 1,211      | 40       |
| # of videos     | 21,819     | 677      |
| # of utterances | 148,642    | 4,874    |

Identification Split
|                 | Dev        | Test     |
|-----------------|------------|----------|
| # of speakers   | 1,251      | 1,251    |
| # of videos     | 21,245     | 1,251    |
| # of utterances | 145,265    | 8,251    |

- [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html): An extension of VoxCeleb1, with more speakers and more utterances, used for training and evaluation.

|                 | Dev        | Test     |
|-----------------|------------|----------|
| # of speakers   | 5,994      | 118      |
| # of videos     | 145,569    | 4,911    |
| # of utterances | 1,092,009  | 36,237   |

For more details on these datasets and the evaluation process, refer to the [VoxCeleb paper](https://arxiv.org/abs/1706.08612).

### Benchmark
See the [Papers With Code leaderboard](https://paperswithcode.com/sota/speaker-verification-on-voxceleb) for rankings of speaker verification by EER on VoxCeleb.

## Notes
- Fine-tuning the model on a specific dataset can further improve accuracy.
