# Speech enhancement


<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/speech_enhancement.ipynb'">Tutorial</button>


## Task Overview
Speech Enhancement is a signal processing task that involves improving the quality of speech signals captured under noisy or degraded conditions. The goal of speech enhancement is to make speech signals clearer, more intelligible, and more pleasant to listen to, which can be used for various applications such as voice recognition, teleconferencing, and hearing aids.


## Models
By now, ```senselab``` supports all ```speechbrain``` models for speech enhancement. These include:
- [SepFormer](https://huggingface.co/speechbrain/sepformer-wham16k-enhancement) for audio clips sampled at 16KHz
- [SepFormer](https://huggingface.co/speechbrain/.sepformer-wham-enhancement) for audio clips sampled at 8KHz.
In the future, more models will be integrated.


## Evaluation
### Metrics

Objective evaluation involves comparing speech enhanced outputs across different downstream tasks:

- Using an automatic speaker verification tool to determine if the original speaker and the enhanced speaker can be distinguished from each other.
- Ensuring the intelligibility of speech content using an automatic speech recognition system to verify that the content remains unchanged.
- Assessing the preservation of the original speech's emotion after speech enhancement.
- ...more...
