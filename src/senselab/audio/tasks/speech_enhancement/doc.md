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
*Speech Enhancement Based on Drifting Models*, Interspeech 2026 oral, arXiv 2604.24199) formulates
enhancement as a distributional equilibrium problem and reaches the clean-speech distribution in a
**single** network evaluation (1 NFE), against 30 for SGMSE+ and 8 for UNIVERSE++. On the DNS 2020
blind test set it reports WV-MOS 2.65 and SCOREQ 2.97.

The drifting field is computed in a frozen self-supervised latent space (HuBERT / WavLM /
DistilHuBERT) during **training** only. Inference is the backbone alone: one forward pass under
`no_grad`, so no SSL encoder is loaded at enhancement time. This is the first generative enhancer in
senselab that is genuinely CPU-viable.

### Why a subprocess venv

Not for dependency conflict — the inference dependency set would satisfy senselab core. The upstream
repository has no installable package and its top-level module names are `backbones`, `util`,
`config` and `data`; injecting a generic `util` onto the host interpreter's `sys.path` is the kind of
hazard that surfaces months later as an unrelated import resolving to the wrong module. The venv
installs only the inference dependency set. Upstream's `requirements.txt` is a *training* dependency
set — the inference path (`enhancement.py` -> `backbones.ncsnpp_v2{,_drift}` + `util.other`) imports
none of `pesq`, `pystoi`, `scoreq`, `torch-pesq`, `asteroid-filterbanks`, `wandb`,
`pytorch-optimizer`, or `torchinfo`, and `pesq`/`scoreq` in particular are slow and fragile to build
for no benefit here. The `latent_ckpt/` archive upstream's README requires for training is therefore
not needed at all.

### Deviations from upstream's script

The worker script reuses upstream's own backbone construction and spectral transforms, but departs
from `enhancement.py` in three ways:

1. **`torch.load(..., weights_only=True)`.** Upstream omits it. The checkpoint is a foreign pickle
   from an unlicensed research repository, so loading it with the unrestricted unpickler is arbitrary
   code execution at enhancement time.
2. **Overlap-add chunking for long inputs.** Upstream runs one STFT over an entire file. The NCSN++
   backbone carries attention layers, so memory grows superlinearly in duration. Enhancement is
   per-segment consistent — there is no cross-segment identity to preserve, unlike separation, where
   which-speaker-is-which must stay fixed across a chunk boundary — so overlap-add (Hann-tapered) is
   safe here in a way it would not be for a separation backend.
3. **A recorded RNG seed.** The released checkpoint sets `train_add_gaussian`, so the forward pass
   consumes a Gaussian sample and is stochastic. An unseeded rerun would produce different audio,
   which would make any cached artifact keyed on this output non-reproducible; `enhance_audios_with_driftse`'s
   `seed` argument makes a run reproducible and is recorded in the log line.

### Licensing

The upstream repository reports no license (no `LICENSE` file, no statement in the README), and is
itself built on SGMSE+ (MIT) without carrying that statement forward. senselab therefore vendors none
of it: the worker clones the repository at a pinned commit into the user's own cache at first use. A
license request was opened upstream on 2026-08-08 and remains unanswered:
<https://github.com/LiangXu123/DriftSE/issues/2>.

The checkpoint mirror under `sensein` is **public**, so the backend is usable during the alpha, and its
licence is **unknown** — those are two separate facts and both matter. Publishing the mirror makes the
weights reachable; it grants no rights over them. No terms have been offered upstream, so treat the
weights as all-rights-reserved by default and consult
<https://github.com/LiangXu123/DriftSE> before any use that turns on licence terms. See the model
registry entry for the pinned revision and file digests.

### Not wired into `audio_analysis`

DriftSE is reachable only by passing an `HFModel` whose id starts with `sensein/driftse` explicitly to
`enhance_audios`. It is not in any default model list and the `audio_analysis` workflow's default
enhancer is unchanged. Deciding how a second enhancer's output participates in the perturbation sample
is a measurement, and it comes after this backend exists.


## Evaluation
### Metrics

Objective evaluation involves comparing speech enhanced outputs across different downstream tasks:

- Using an automatic speaker verification tool to determine if the original speaker and the enhanced speaker can be distinguished from each other.
- Ensuring the intelligibility of speech content using an automatic speech recognition system to verify that the content remains unchanged.
- Assessing the preservation of the original speech's emotion after speech enhancement.
- ...more...
