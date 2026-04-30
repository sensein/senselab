"""This module represents the API for the speech classification task within the senselab package.

Currently, it supports only Hugging Face models, with plans to include more in the future.
Users can specify the audio clips to classify, the classification model, the preferred device,
and the model-specific parameters, and senselab handles the rest.

When ``win_length`` is provided, classification runs over sliding windows,
producing per-window results with timestamps. If omitted, the entire audio
is classified as a single unit.
"""

from typing import Any, Dict, List, Optional, Union

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.huggingface import HuggingFaceAudioClassifier
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel


@requires_compatibility("audio.tasks.classification.classify_audios")
def classify_audios(
    audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType] = None,
    win_length: Optional[float] = None,
    hop_length: Optional[float] = None,
    top_k: Optional[int] = None,
    **kwargs: Any,  # noqa: ANN401
) -> Union[List[AudioClassificationResult], List[List[Dict[str, Any]]]]:
    """Classify audios using the given model.

    When ``win_length`` is ``None`` (default), each audio is classified as a
    whole and the return type is ``List[AudioClassificationResult]``.

    When ``win_length`` is provided (in seconds), each audio is sliced into
    overlapping windows and classified per-window.  The return type changes
    to ``List[List[Dict]]`` where each inner dict contains:

    - ``start`` / ``end`` (float): window boundaries in seconds.
    - ``labels`` (List[str]): top-k predicted class names.
    - ``scores`` (List[float]): corresponding confidence values.
    - ``win_length`` / ``hop_length`` (float): the parameters used
      (captured for provenance).

    Args:
        audios: Audio objects to classify.
        model: The classification model.
        device: Device for inference (default: auto-select).
        win_length: Window duration in seconds.  If ``None``, classify
            the full audio.  If set, ``hop_length`` defaults to
            ``win_length / 2`` when not provided.
        hop_length: Hop (step) duration in seconds for windowed mode.
            Ignored when ``win_length`` is ``None``.
        top_k: Keep only the top-k labels per result.  Applies in both
            whole-audio and windowed modes.  ``None`` keeps all labels.
        **kwargs: Forwarded to the backend classifier.

    Returns:
        ``List[AudioClassificationResult]`` in whole-audio mode, or
        ``List[List[Dict]]`` in windowed mode.
    """
    if win_length is not None:
        return _classify_windowed(
            audios=audios,
            model=model,
            device=device,
            win_length=win_length,
            hop_length=hop_length if hop_length is not None else win_length / 2,
            top_k=top_k or 5,
            **kwargs,
        )

    results = _classify_whole(audios=audios, model=model, device=device, **kwargs)

    if top_k is not None:
        results = [AudioClassificationResult(labels=r.labels[:top_k], scores=r.scores[:top_k]) for r in results]
    return results


def _classify_whole(
    audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[AudioClassificationResult]:
    """Classify each audio as a single unit (no windowing)."""
    if isinstance(model, HFModel):
        return HuggingFaceAudioClassifier.classify_audios_with_transformers(
            audios=audios, model=model, device=device, **kwargs
        )
    raise NotImplementedError(
        "Only Hugging Face models are supported for now. We aim to support more models in the future."
    )


def _classify_windowed(
    audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType],
    win_length: float,
    hop_length: float,
    top_k: int,
    **kwargs: Any,  # noqa: ANN401
) -> List[List[Dict[str, Any]]]:
    """Slice audios into overlapping windows and classify each window.

    Uses :meth:`Audio.window_generator` for consistent windowed iteration
    across senselab.  Processes one audio at a time to keep memory bounded.
    """
    batch_size = 32
    output: List[List[Dict[str, Any]]] = []

    for audio in audios:
        sr = audio.sampling_rate
        win_samples = max(1, int(win_length * sr))
        hop_samples = max(1, int(hop_length * sr))
        n_samples = audio.waveform.shape[1]

        # Iterate over windows lazily in batches for memory efficiency.
        audio_results: List[Dict[str, Any]] = []
        batch: List[Audio] = []
        batch_positions: List[int] = []
        pos = 0

        for window in audio.window_generator(win_samples, hop_samples):
            batch.append(window)
            batch_positions.append(pos)
            pos += hop_samples

            if len(batch) >= batch_size:
                results = _classify_whole(batch, model=model, device=device, **kwargs)
                for bp, result in zip(batch_positions, results):
                    audio_results.append(
                        {
                            "start": bp / sr,
                            "end": min(bp + win_samples, n_samples) / sr,
                            "labels": result.labels[:top_k],
                            "scores": result.scores[:top_k],
                            "win_length": win_length,
                            "hop_length": hop_length,
                        }
                    )
                batch.clear()
                batch_positions.clear()

        # Process remaining windows in final batch.
        if batch:
            results = _classify_whole(batch, model=model, device=device, **kwargs)
            for bp, result in zip(batch_positions, results):
                k = min(top_k, len(result.labels)) if result.labels else 0
                audio_results.append(
                    {
                        "start": bp / sr,
                        "end": min(bp + win_samples, n_samples) / sr,
                        "labels": result.labels[:k],
                        "scores": result.scores[:k],
                        "win_length": win_length,
                        "hop_length": hop_length,
                    }
                )

        output.append(audio_results)

    return output


def scene_results_to_segments(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert windowed classification results to ``plot_aligned_panels`` segment format.

    Uses the top-1 label from each window.

    Args:
        results: Per-window dicts from ``classify_audios(..., win_length=...)``.

    Returns:
        Segment dicts with ``label``, ``start``, ``end``.
    """
    return [
        {"label": r["labels"][0] if r["labels"] else "unknown", "start": r["start"], "end": r["end"]} for r in results
    ]
