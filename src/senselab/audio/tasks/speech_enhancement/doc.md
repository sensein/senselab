# Speech enhancement


<button class="tutorial-button" onclick="window.location.href='https://github.com/sensein/senselab/blob/main/tutorials/audio/speech_enhancement.ipynb'">Tutorial</button>


## Task Overview
Speech Enhancement is a signal processing task that involves improving the quality of speech signals captured under noisy or degraded conditions. The goal of speech enhancement is to make speech signals clearer, more intelligible, and more pleasant to listen to, which can be used for various applications such as voice recognition, teleconferencing, and hearing aids.


## Models
By now, ```senselab``` supports all ```speechbrain``` models for speech enhancement. These include:
- [SepFormer](https://huggingface.co/speechbrain/sepformer-wham16k-enhancement) for audio clips sampled at 16KHz
- [SepFormer](https://huggingface.co/speechbrain/.sepformer-wham-enhancement) for audio clips sampled at 8KHz.
In the future, more models will be integrated.

## DriftSE (one-step diffusion enhancement)

[DriftSE](https://github.com/LiangXu123/DriftSE) (Xu, Caviedes-Nozal, Kleijn, Yan & Olsson,
*Speech Enhancement Based on Drifting Models*, Interspeech 2026 oral, arXiv 2604.24199) reaches the
clean-speech distribution in a **single** network evaluation (1 NFE), against 30 for SGMSE+ and 8 for
UNIVERSE++, and is the one generative enhancer here that runs on CPU. Upstream code and weights are
MIT-licensed.

Two released checkpoints, both mirrored by upstream at
[`LIANGXU123/DriftSE`](https://huggingface.co/LIANGXU123/DriftSE):

| Variant | Reported PESQ / SI-SDR | Notes |
|---|---|---|
| `distillhubert_three_layers_with_z` (default) | 3.00 / 15.6 | latent-drift loss only |
| `distillhubert_three_layers_pesq_sisdr_ccmse_with_z` | 3.45 / 20.6 | trained with PESQ/SI-SDR/CCMSE in the loss |

Both are `with_z` — `train_add_gaussian` true — and both are *ablation* rows in upstream's own
VB-DMD table, which marks them as such; neither is the paper's headline configuration, and the
`no_z` variant the README describes has no released checkpoint. `variant` selects between the two;
there is nothing else to select.

Those numbers are upstream's README at the pinned commit, measured against a copy of the
VoiceBank-DEMAND test set that an independent reproducer could not reproduce; the same enhanced
audio scores about 0.4 PESQ lower against a standard copy
([issue #1](https://github.com/LiangXu123/DriftSE/issues/1), open). Read them as advertised rather
than confirmed.

### Calling it

```python
from senselab.audio.tasks.speech_enhancement import enhance_audios
from senselab.utils.data_structures import HFModel

enhanced = enhance_audios(audios, model=HFModel(path_or_uri="LIANGXU123/DriftSE"))
```

`enhance_audios_with_driftse` takes the parameters the dispatcher cannot: `sigma` (the scale of the
Gaussian added to the model input, default 0.01, upstream's own value — 0.05 measurably degrades
output), `seed` (the released checkpoints make that Gaussian part of the forward pass, so output is
stochastic without one), `variant`, and the chunking. DriftSE runs in an isolated subprocess venv
that clones upstream at a pinned commit on first use; `SENSELAB_DRIFTSE_CHECKPOINT` points it at a
local directory holding `last.ckpt` + `config.json` instead of the Hub.

It is a normal selectable backend, nameable from a workflow config like any other model id, but it is
**not** the default enhancer: whether a one-step generative enhancer should displace SepFormer, and
how a second enhancer's output participates in the perturbation sample, are measurements that have
not been made.

Levels: upstream peak-normalises its input, runs the network, and rescales the output by its own
peak back to that input peak. senselab does the same per window, because the arbitrary gain being
removed is a property of one network evaluation. The output therefore carries the input's peak.

The worker deviates from upstream's `enhancement.py` in three ways — `torch.load(weights_only=True)`,
Hann-tapered overlap-add for long inputs, and a recorded RNG seed. Those deviations, the pinned
commits, the licence history, the venv's dependency set and every measurement behind the choices are
in `specs/20260818-083214-driftse-upstream-mit/design.md`.


## ClearVoice (FRCRN, MossFormerGAN, MossFormer2)

[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (Alibaba Speech Lab,
Apache-2.0) contributes three enhancers, in an isolated subprocess venv:

| Model id | Rate | Upstream PESQ (VB-DMD / DNS-2020) |
|---|---|---|
| `alibabasglab/FRCRN_SE_16K` | 16 kHz | 3.23 / 3.24 |
| `alibabasglab/MossFormerGAN_SE_16K` | 16 kHz | 3.47 / 3.57 |
| `alibabasglab/MossFormer2_SE_48K` | 48 kHz | 3.16 / 2.94 |

```python
from senselab.audio.tasks.speech_enhancement import enhance_audios
from senselab.utils.data_structures import HFModel

enhanced = enhance_audios(audios, model=HFModel(path_or_uri="alibabasglab/FRCRN_SE_16K"))
```

Inputs are resampled to the checkpoint's rate and downmixed to mono; the output has the input's
sample count at that rate. `parameters={"timeout_s": ...}` is the one knob.

### Choosing between them decides which non-speech element survives

Measured against six human-verified events on a real recording — capability facts that are **not** in
upstream's documentation, and the reason an enhancer is not a neutral preprocessing step for any
analysis that treats breaths or coughs as signal:

| | breaths | coughs | input energy kept |
|---|---|---|---|
| `FRCRN_SE_16K` | kept (−2.0, −5.5 dB) | kept (−1.0, +0.1 dB) | most |
| `MossFormer2_SE_48K` | **destroyed** (−37, −40 dB) | kept (−0.6, +0.2 dB) | most |
| `MossFormerGAN_SE_16K` | **destroyed** (−51 dB) | 2 of 4 removed | ~8% |

So: `FRCRN_SE_16K` when non-speech events matter, `MossFormerGAN_SE_16K` when only speech does, and
`MossFormer2_SE_48K` when the recording is fullband and breaths are not of interest.

All three **conserve energy**: output never exceeds input (−12.2 to −0.0 dB), in phase at zero lag,
and clean speech is left essentially untouched — every SepFormer checkpoint tested failed that check.
Peaks stayed ≤ 0.95, so no clipping exposure was observed. Independently reproduced here on clean
16 kHz conversational speech: `FRCRN_SE_16K` returned −0.01 dB RMS at r = 1.0000.

`FRCRN_SE_16K` is one of two ClearVoice checkpoints whose input upstream RMS-normalises to −25 dBFS;
the inverse scalar is applied to the output, and reported in `metadata["clearvoice"]`.

Each returned `Audio` carries `metadata["clearvoice"]` with the model id and the **resolved commit**
its weights came from — necessary because upstream's loader accepts no revision. How that is pinned,
and every deviation from upstream's own inference path, is in
`specs/20260819-clearvoice-integration/design.md`.


## Evaluation
### Metrics

Objective evaluation involves comparing speech enhanced outputs across different downstream tasks:

- Using an automatic speaker verification tool to determine if the original speaker and the enhanced speaker can be distinguished from each other.
- Ensuring the intelligibility of speech content using an automatic speech recognition system to verify that the content remains unchanged.
- Assessing the preservation of the original speech's emotion after speech enhancement.
- ...more...
