Health acoustics: Google's HeAR encoder and its bundled health sound event detector.

[HeAR](https://developers.google.com/health-ai-developer-foundations/hear) (Baur et al., 2024,
[arXiv:2403.02522](https://arxiv.org/abs/2403.02522)) is a ViT-L masked autoencoder trained on
313 M two-second clips mined for non-semantic human sounds — coughs, breaths, throat clears,
sneezes. `google/hear` ships two artifacts and this module exposes both.

This is **health acoustics, not speaker identity**: the embedding describes what the sound is, not
who made it, which is why it lives here rather than under `speaker_embeddings/`.

| Capability | Model | Output | Input signature |
|---|---|---|---|
| Encoder | `google/hear` (ViT-L MAE, ~300 M params) | 512-d embedding per window | `x: (None, 32000) float32` |
| Event detector | `event_detector/event_detector_large` (MobileNetV3-L, ~3 M) | 8 presence probabilities | `audio_wav: (1, 32000) float32` |
| Event detector (small) | `event_detector/event_detector_small` (MobileNetV3-S, ~1 M) | 8 presence probabilities | `audio_wav: (1, 32000) float32` |

Both detectors share one spectrogram frontend and one label set:
`Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear, Laugh, Speech`.

## Usage

```python
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics import (
    centred_cosine_similarity,
    detect_health_acoustic_events,
    extract_hear_embeddings_at_times,
    extract_hear_embeddings_from_audios,
)

audio = Audio(filepath="recording.wav")

# Sliding 2 s windows, 50% overlap: one 512-d row per window.
[emb] = extract_hear_embeddings_from_audios([audio], hop_length=1.0)
emb.embeddings.shape          # [n_windows, 512]
emb.window_starts             # seconds, on the recording's timeline
emb.pooled().shape            # [512], the file-level mean

# One embedding per event, taken from 2 s of the real recording around each time.
per_event = extract_hear_embeddings_at_times(audio, times=[2.4, 9.7, 12.1])

# Similarity, always mean-centred (raw cosine is uninformative; see below).
centred_cosine_similarity(per_event)          # [3, 3]

# Event detection: 8 independent probabilities per 2 s window.
[windows] = detect_health_acoustic_events([audio], hop_length=0.25)
windows[0]["label_scores"]    # [{"Breathe": 0.83}, {"Speech": 0.12}, ...] descending
```

## The 2 s window is the model, not a default

Every constraint below was measured on real recordings this week, and each one is either enforced
by the API or stated in the docstring of the function a caller would otherwise get it wrong in.

| Measured | Consequence in this module |
|---|---|
| The **detector rejects** every length but 32000 samples (`InvalidArgumentError` at 0.5, 1.0, 1.5, 3.0, 4.0 s); its batch dimension is pinned at 1 | Window length is not a parameter; the worker reads the pinned batch off the graph and feeds one window at a time |
| The **encoder silently accepts** 160…64000 samples, returning a finite, plausible vector | Sub-2 s input raises `ValueError` naming the measurement, rather than being forwarded |
| **Padding destroys the representation**: centred cosine 0.0–0.5 between framings of the same event, against a ~0.9 class margin | No code path pads. Windows are planned inside the recording, and the worker re-checks each one's bounds |
| **Usable length falls off below 2 s**: centred class margin +0.91 at 2 s, +0.46 at 1 s, +0.29 at 0.3 s; 3 s is *worse* than 2 s | 2 s everywhere; short events are handled by placing a 2 s window of real context around them (`extract_hear_embeddings_at_times`) |
| **Shift is benign**: ±50–200 ms gives cosine 0.93–0.98. **Amplitude is irrelevant**: gains ×0.1…×10 give 1.0000 | The tail window is snapped to end at the last sample, and an edge-adjacent centre slides inward, instead of either being padded. No normalisation is applied |
| **Raw cosine is uninformative**: 0.977 within-class vs 0.918 between. Centred: +0.653 vs −0.256, LOO-NN 0.846 with no training | `centred_cosine_similarity` centres by construction; no raw-cosine helper is offered |
| The **detector is a presence gate**: 40 ms of cough crosses p > 0.5, so its response is a box-car of width (event + 2 s) and events closer than 2 s merge | Documented in `detect_health_acoustic_events`; the returned window bounds are window bounds, never event bounds |

## Backend

TensorFlow, in an isolated subprocess venv (`~/.cache/senselab/venvs/hear`), provisioned on first
use — the same pattern as `classification/yamnet.py`. `google/hear` ships TF SavedModels and
senselab is torch-based; TensorFlow is not a core dependency.

A torch conversion (`google/hear-pytorch`) exists but is not used: it is separately gated (this
project's account is authorized for `google/hear` and not for it, so equivalence cannot be
verified), it takes a spectrogram rather than a waveform and its frontend is not in that
repository, and it carries no event detector — so TensorFlow would be needed for the second
capability regardless. See the module docstring of
`senselab.audio.tasks.health_acoustics.hear` for the full argument.

## Access

`google/hear` is gated under the
[Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms);
acknowledging them while logged in to Hugging Face grants access immediately. Set `HF_TOKEN` (or
`HUGGING_FACE_HUB_TOKEN` / `HUGGINGFACE_HUB_TOKEN`, or put it in a `.env`) — the same mechanism as
senselab's other gated models. Without access the staging step raises `huggingface_hub`'s
`GatedRepoError`, which names the repository and the page to accept the terms on.

The weights are pinned at commit `9b2eb2853c426676255cc6ac5804b7f1fe8e563f` — a SHA, never a ref.
