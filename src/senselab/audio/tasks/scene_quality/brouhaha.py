"""``pyannote/brouhaha`` — joint per-frame VAD + SNR + C50 estimation.

Brouhaha (Lavechin et al., 2022, arXiv:2210.13248) is a ``pyannote-audio``
multitask model that predicts, per frame and in a single forward pass:

- **VAD** — speech-presence probability in ``[0, 1]``;
- **SNR** — estimated signal-to-noise ratio in dB;
- **C50** — room-acoustics clarity in dB (higher = less reverberant).

It is loaded through the same low-level ``Model`` + ``Inference`` path used for
raw segmentation posteriors (not the high-level ``Pipeline``, which would
re-segment and discard the regression heads). The model is gated on HuggingFace;
loading reuses ``ensure_hf_model`` + ``get_huggingface_token`` so cached runs
skip the Hub (constitution VI). No new pip dependency and no subprocess venv
(FR-025).

The scene-quality workflow uses the SNR/C50 heads for the ``quality_snr`` /
``quality_reverb`` degradation scores and the VAD head as a second frame-level
speech-presence voter.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.data_structures.model import get_huggingface_token
from senselab.utils.dependencies import ensure_hf_model, hf_local_files_only, retry_on_transient_error

if TYPE_CHECKING:
    from pyannote.audio import Inference

try:
    from pyannote.audio import Inference, Model

    PYANNOTEAUDIO_AVAILABLE = True
except (ImportError, RuntimeError):  # RuntimeError: native-lib load failures
    PYANNOTEAUDIO_AVAILABLE = False

BROUHAHA_MODEL_ID = "pyannote/brouhaha"
BROUHAHA_REVISION = "main"

# Channel order of the Brouhaha multitask output (frames, 3). Documented by the
# model; validated end-to-end on a real clip in T044.
_VAD_CHANNEL = 0
_SNR_CHANNEL = 1
_C50_CHANNEL = 2


@dataclass
class BrouhahaFrames:
    """Per-frame Brouhaha outputs for one audio.

    Attributes:
        vad: ``(num_frames,)`` speech-presence probability in ``[0, 1]``.
        snr_db: ``(num_frames,)`` estimated SNR in dB.
        c50_db: ``(num_frames,)`` estimated C50 (clarity) in dB.
        frame_hop_s: seconds between consecutive frame starts.
    """

    vad: np.ndarray
    snr_db: np.ndarray
    c50_db: np.ndarray
    frame_hop_s: float

    def mean_in_window(self, start_s: float, end_s: float) -> tuple[float, float, float]:
        """Return ``(mean vad, mean snr_db, mean c50_db)`` over frames overlapping ``[start, end)``.

        Any component with no overlapping frames returns ``nan`` for that value.
        """
        if self.frame_hop_s <= 0 or self.vad.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        lo = max(0, int(np.floor(start_s / self.frame_hop_s)))
        hi = min(self.vad.size, int(np.ceil(end_s / self.frame_hop_s)))
        if hi <= lo:
            return (float("nan"), float("nan"), float("nan"))
        return (
            float(np.nanmean(self.vad[lo:hi])),
            float(np.nanmean(self.snr_db[lo:hi])),
            float(np.nanmean(self.c50_db[lo:hi])),
        )


# Cache Inference objects per (model_id, revision, device) so repeated pass calls
# reuse the initialized model.
_inference_cache: dict[str, "Inference"] = {}


def _get_brouhaha_inference(model: PyannoteAudioModel, device: Optional[DeviceType]) -> "Inference":
    """Load (and cache) a Brouhaha ``Inference`` for the requested model/device."""
    import torch

    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])
    key = f"{model.path_or_uri}-{model.revision}-{device}"
    if key not in _inference_cache:
        # Coordinate the (possibly gated) download once, then load offline.
        ensure_hf_model(str(model.path_or_uri), revision=model.revision, token=get_huggingface_token())
        loaded = retry_on_transient_error(
            Model.from_pretrained,
            model.path_or_uri,
            revision=model.revision,
            token=get_huggingface_token(),
        )
        if loaded is None:
            raise ValueError(f"Brouhaha model {model.path_or_uri} could not be loaded.")
        inference = Inference(loaded, device=torch.device(device.value))
        _inference_cache[key] = inference
    return _inference_cache[key]


def extract_brouhaha_frames(
    audios: list[Audio],
    model: Optional[PyannoteAudioModel] = None,
    device: Optional[DeviceType] = None,
) -> list[Optional[BrouhahaFrames]]:
    """Run Brouhaha once per audio, returning per-frame VAD/SNR/C50.

    Args:
        audios: mono 16 kHz clips.
        model: Brouhaha model spec. Defaults to ``pyannote/brouhaha@main``.
        device: inference device (CPU or CUDA).

    Returns:
        One ``BrouhahaFrames`` per input, or ``None`` for an audio whose inference
        failed. If the model itself cannot be loaded (not installed, gated without
        a token, native-lib failure), every entry is ``None`` — the workflow then
        emits null quality columns rather than aborting (FR-023).
    """
    if not PYANNOTEAUDIO_AVAILABLE:
        logger.warning("pyannote-audio unavailable; Brouhaha scene-quality signals will be null.")
        return [None] * len(audios)

    if model is None:
        model = PyannoteAudioModel(path_or_uri=BROUHAHA_MODEL_ID, revision=BROUHAHA_REVISION)

    # Prefer offline load when the model is already cached (constitution VI).
    hf_local_files_only(str(model.path_or_uri), revision=model.revision)

    try:
        inference = _get_brouhaha_inference(model=model, device=device)
    except Exception as exc:  # noqa: BLE001 — any load failure degrades to null quality (FR-023)
        logger.warning(f"Failed to load Brouhaha model {model.path_or_uri}: {exc}. Scene-quality signals will be null.")
        return [None] * len(audios)

    results: list[Optional[BrouhahaFrames]] = []
    for audio in audios:
        try:
            t0 = time.time()
            # Inference on a whole file returns a SlidingWindowFeature; the pyannote
            # type stubs are ambiguous (union), so treat it as Any here.
            output: Any = inference({"waveform": audio.waveform, "sample_rate": audio.sampling_rate})
            data = np.asarray(output.data, dtype=np.float64)  # (num_frames, 3)
            if data.ndim != 2 or data.shape[1] <= _C50_CHANNEL:
                raise ValueError(f"unexpected Brouhaha output shape {data.shape}")
            frame_hop_s = float(output.sliding_window.step)
            results.append(
                BrouhahaFrames(
                    vad=data[:, _VAD_CHANNEL],
                    snr_db=data[:, _SNR_CHANNEL],
                    c50_db=data[:, _C50_CHANNEL],
                    frame_hop_s=frame_hop_s,
                )
            )
            logger.info(f"Brouhaha inference took {time.time() - t0:.2f}s ({data.shape[0]} frames)")
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(f"Brouhaha inference failed for one audio: {exc}")
            results.append(None)
    return results
