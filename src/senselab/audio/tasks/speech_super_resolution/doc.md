# Speech super-resolution

## Task overview

Speech super-resolution — bandwidth extension — reconstructs the high-frequency band a recording never
captured, turning low-resolution speech into 48 kHz audio. It is not resampling: interpolating a 16 kHz
signal to 48 kHz adds samples but no content, whereas this predicts the missing band.

It is a **separate task from enhancement**, not a mode of it, because its output contract differs: the
result is 48 kHz whatever the input rate was. Folding it into `enhance_audios` would give one function
two output contracts, decided by which model id was passed.

## Model

One backend, in an isolated subprocess venv:
[`alibabasglab/MossFormer2_SR_48K`](https://huggingface.co/alibabasglab/MossFormer2_SR_48K) from
[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (Apache-2.0). Two stages: a
MossFormer2 predicting a mel representation, then a HiFi-GAN vocoder, followed by upstream's own
bandwidth substitution, which keeps the input's own band and splices in only what was missing.

```python
from senselab.audio.tasks.speech_super_resolution import super_resolve_audios

upsampled = super_resolve_audios(audios)          # 48 kHz out
upsampled = super_resolve_audios(audios, parameters={"timeout_s": 3600})
```

The input is resampled to 48 kHz on the host first, because the model fills in bandwidth rather than
resampling. Upstream reports an effective input rate of at least 16 kHz is needed.

Upstream's own numbers, on VoiceBank-DEMAND downsampled and then restored (log-spectral distance,
lower is better):

| Input rate | 16 kHz | 24 kHz | 32 kHz | 48 kHz |
|---|---|---|---|---|
| Original | 2.80 | 2.60 | 2.29 | 1.46 |
| Restored | **1.93** | **1.52** | **1.50** | 1.42 |

At 48 kHz no super-resolution is applied, so that column measures enhancement rather than bandwidth
extension. When the input is both narrowband and noisy, upstream pairs this with
`MossFormer2_SE_48K` — run `enhance_audios` first, then this.

## Provenance

Upstream's loader accepts no revision, so senselab pre-stages the checkpoint at a resolved commit and
points the loader at that local path. Every returned `Audio` carries `metadata["clearvoice"]` naming
the model and that commit, and `metadata["backend_parameters"]` naming the parameters that ran. The
mechanism, and the 1.74 GB optimizer state deliberately not downloaded, are in
`specs/20260819-clearvoice-integration/design.md`.

## Not measured here

This checkpoint has not been run end to end in this repository: its two-stage weights total 1.6 GB and
no bandwidth-extension ground truth was available. Dispatch, staging, the device, the ceiling and the
output rate are covered by tests; the quality numbers above are upstream's own.
