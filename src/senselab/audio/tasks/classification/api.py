"""This module represents the API for the speech classification task within the senselab package.

Currently, it supports only Hugging Face models, with plans to include more in the future.
Users can specify the audio clips to classify, the classification model, the preferred device,
and the model-specific parameters, and senselab handles the rest.
"""

from typing import Any, Dict, List, Optional

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.huggingface import HuggingFaceAudioClassifier
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel


@requires_compatibility("audio.tasks.classification.classify_audios")
def classify_audios(
    audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[AudioClassificationResult]:
    """Classify all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be classified.
        model (SenselabModel): The model used for classification.
        device (Optional[DeviceType]): The device to run the model on (default is None).
        **kwargs: Additional keyword arguments to pass to the classification function.

    Returns:
        List[AudioClassificationResult]: The list of classification results.

    Todo:
        - Include more models (e.g., speechbrain, nvidia)
    """
    if isinstance(model, HFModel):
        return HuggingFaceAudioClassifier.classify_audios_with_transformers(
            audios=audios, model=model, device=device, **kwargs
        )
    else:
        raise NotImplementedError(
            "Only Hugging Face models are supported for now. We aim to support more models in the future."
        )


def classify_audios_in_windows(
    audios: List[Audio],
    model: SenselabModel,
    window_size: float = 1.0,
    hop_size: float = 0.5,
    top_k: int = 5,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[List[Dict[str, Any]]]:
    """Classify audio in overlapping sliding windows for temporal scene analysis.

    Slices each audio into overlapping windows and runs classification on every
    window, producing a time-resolved list of classification results per audio.

    Args:
        audios: The list of audio objects to classify.
        model: The model used for classification.
        window_size: Window duration in seconds (default 1.0).
        hop_size: Hop (step) duration in seconds (default 0.5).
        top_k: Number of top labels/scores to keep per window (default 5).
        device: The device to run the model on (default is None).
        **kwargs: Additional keyword arguments forwarded to ``classify_audios``.

    Returns:
        A list (one entry per input audio) of lists (one entry per window).
        Each window dict contains:

        - ``start`` (float): window start time in seconds.
        - ``end`` (float): window end time in seconds.
        - ``labels`` (List[str]): top-k predicted labels.
        - ``scores`` (List[float]): corresponding scores.
    """
    batch_size = 32

    # Build all window Audio objects and record their provenance.
    all_windows: List[Audio] = []
    window_meta: List[List[Dict[str, Any]]] = []  # per-audio list of {start, end}

    for audio in audios:
        sr = audio.sampling_rate
        waveform = audio.waveform
        n_samples = waveform.shape[1]

        win_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        meta_for_audio: List[Dict[str, Any]] = []

        if n_samples <= win_samples:
            # Audio shorter than one window -> use full audio as a single window.
            all_windows.append(Audio(waveform=waveform, sampling_rate=sr))
            meta_for_audio.append({"start": 0.0, "end": n_samples / sr})
        else:
            start = 0
            while start + win_samples <= n_samples:
                chunk = waveform[:, start : start + win_samples]
                all_windows.append(Audio(waveform=chunk, sampling_rate=sr))
                meta_for_audio.append({"start": start / sr, "end": (start + win_samples) / sr})
                start += hop_samples

        window_meta.append(meta_for_audio)

    # Classify in batches for memory efficiency.
    all_results: List[AudioClassificationResult] = []
    for batch_start in range(0, len(all_windows), batch_size):
        batch = all_windows[batch_start : batch_start + batch_size]
        all_results.extend(classify_audios(batch, model=model, device=device, **kwargs))

    # Map flat results back to per-audio, per-window dicts.
    output: List[List[Dict[str, Any]]] = []
    idx = 0
    for meta_list in window_meta:
        audio_results: List[Dict[str, Any]] = []
        for meta in meta_list:
            result = all_results[idx]
            k = min(top_k, len(result.labels))
            audio_results.append(
                {
                    "start": meta["start"],
                    "end": meta["end"],
                    "labels": result.labels[:k],
                    "scores": result.scores[:k],
                }
            )
            idx += 1
        output.append(audio_results)

    return output


def scene_results_to_segments(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert windowed classification results to ``plot_aligned_panels`` segment format.

    Takes the per-window results produced by :func:`classify_audios_in_windows`
    and returns segment dicts suitable for a ``{"type": "segments", ...}`` panel,
    using the top-1 label from each window.

    Args:
        results: List of window dicts as returned by ``classify_audios_in_windows``
            (each dict has ``start``, ``end``, ``labels``, ``scores``).

    Returns:
        List of dicts with keys ``label``, ``start``, ``end``.
    """
    segments: List[Dict[str, Any]] = []
    for r in results:
        segments.append(
            {
                "label": r["labels"][0],
                "start": r["start"],
                "end": r["end"],
            }
        )
    return segments
