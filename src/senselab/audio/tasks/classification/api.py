"""This module represents the API for the speech classification task within the senselab package.

Supports HuggingFace models, YAMNet, and HeAR's bundled health sound event
detector (the latter two via isolated subprocess venvs).
Users can specify the audio clips to classify, the classification model,
the preferred device, and the model-specific parameters.

When ``win_length`` is provided, classification runs over sliding windows,
producing per-window results with timestamps. If omitted, the entire audio
is classified as a single unit (HF models) or using the model's own
internal windowing (YAMNet).
"""

from typing import Any, Dict, List, Optional, Union

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.huggingface import HuggingFaceAudioClassifier
from senselab.audio.tasks.classification.label_scores import label_scores, top_label
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel, logger

_YAMNET_ALIASES = {"yamnet", "google/yamnet"}


def _is_yamnet(model: Union[SenselabModel, str]) -> bool:
    """Check if the model refers to YAMNet."""
    if isinstance(model, str):
        return model.lower() in _YAMNET_ALIASES
    return False


def _is_hear_event_detector(model: Union[SenselabModel, str]) -> bool:
    """Check if the model refers to HeAR's bundled health sound event detector."""
    from senselab.audio.tasks.health_acoustics.hear import is_hear_event_detector_spec

    return is_hear_event_detector_spec(model)


@requires_compatibility("audio.tasks.classification.classify_audios")
def classify_audios(
    audios: List[Audio],
    model: Union[SenselabModel, str],
    device: Optional[DeviceType] = None,
    win_length: Optional[float] = None,
    hop_length: Optional[float] = None,
    top_k: Optional[int] = None,
    **kwargs: Any,  # noqa: ANN401
) -> Union[List[AudioClassificationResult], List[List[Dict[str, Any]]]]:
    """Classify audios using the given model.

    When ``win_length`` is ``None`` (default) and the model is an HF model,
    each audio is classified as a whole and the return type is
    ``List[AudioClassificationResult]``.

    When ``win_length`` is provided (in seconds), each audio is sliced into
    overlapping windows and classified per-window.  The return type changes
    to ``List[List[Dict]]`` where each inner dict contains:

    - ``start`` / ``end`` (float): window boundaries in seconds.
    - ``labels`` (List[str]): top-k predicted class names.
    - ``scores`` (List[float]): corresponding confidence values.
    - ``win_length`` / ``hop_length`` (float): the parameters used
      (captured for provenance).

    **YAMNet** (``model="yamnet"``): Runs in an isolated TensorFlow
    subprocess venv.  YAMNet uses fixed 0.96 s windows with 0.48 s hop
    internally, so it always returns windowed results.  The ``win_length``
    and ``hop_length`` parameters are ignored for YAMNet.

    **HeAR event detector** (``model="hear-event-detector"``, or
    ``"hear-event-detector-small"``): Google's health sound event detector
    (``Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear, Laugh,
    Speech``), in an isolated TensorFlow subprocess venv.  Its window is
    fixed at 2 s — the graph rejects every other length — so ``win_length``
    is ignored while ``hop_length`` is honoured.  Scores are independent
    presence probabilities, not a distribution, and the detector is a
    presence gate rather than a locator; see
    :mod:`senselab.audio.tasks.health_acoustics` for both caveats in full.

    Args:
        audios: Audio objects to classify.
        model: The classification model.  Can be an ``HFModel`` for
            HuggingFace pipelines, or ``"yamnet"`` for the YAMNet
            subprocess backend.
        device: Device for inference (default: auto-select).  Ignored
            for YAMNet (TensorFlow manages devices internally).
        win_length: Window duration in seconds.  If ``None``, classify
            the full audio.  If set, ``hop_length`` defaults to
            ``win_length / 2`` when not provided.  Ignored for YAMNet.
        hop_length: Hop (step) duration in seconds for windowed mode.
            Ignored when ``win_length`` is ``None``.  Ignored for YAMNet.
        top_k: Keep only the top-k labels per result.  Applies in both
            whole-audio and windowed modes.  ``None`` keeps all labels
            (defaults to 5 for windowed mode and YAMNet).
        **kwargs: Forwarded to the backend classifier.

    Returns:
        ``List[AudioClassificationResult]`` in whole-audio mode (HF only), or
        ``List[List[Dict]]`` in windowed mode or when using YAMNet.
    """
    if _is_yamnet(model):
        from senselab.audio.tasks.classification.yamnet import YAMNetClassifier

        if win_length is not None:
            logger.info("YAMNet uses fixed 0.96s windows; win_length/hop_length parameters are ignored.")
        return YAMNetClassifier.classify_with_yamnet(audios=audios, top_k=top_k or 5)

    # HeAR's bundled event detector is an audio event classifier, so it is reachable from here
    # (the discoverable home for classifiers) as well as from its own task package. This is a
    # dispatch alias only: the implementation, the isolated TensorFlow venv and the pinned
    # revision all live in audio/tasks/health_acoustics/, because the detector shares that
    # repository, that venv and that 2 s window rule with the HeAR encoder, which is not a
    # classifier and does not belong here.
    if _is_hear_event_detector(model):
        from senselab.audio.tasks.health_acoustics import detect_health_acoustic_events
        from senselab.audio.tasks.health_acoustics.api import DEFAULT_DETECTION_HOP_SECONDS

        if win_length is not None:
            logger.info(
                "HeAR's event detector rejects every input length but 2s (its graph raises), so "
                "win_length is ignored; hop_length is honoured."
            )
        return detect_health_acoustic_events(
            audios=audios,
            model=str(model),
            device=device,
            hop_length=hop_length if hop_length is not None else DEFAULT_DETECTION_HOP_SECONDS,
            top_k=top_k,
        )

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
    model: Union[SenselabModel, str],
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[AudioClassificationResult]:
    """Classify each audio as a single unit (no windowing)."""
    if isinstance(model, HFModel):
        return HuggingFaceAudioClassifier.classify_audios_with_transformers(
            audios=audios, model=model, device=device, **kwargs
        )
    raise NotImplementedError(
        f"Model type {type(model).__name__} is not supported for whole-audio classification. "
        "Use HFModel for HuggingFace pipelines or 'yamnet' for YAMNet."
    )


def _classify_windowed(
    audios: List[Audio],
    model: Union[SenselabModel, str],
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
                            # One dict per label, so a score cannot drift away from the label it
                            # belongs to. Two aligned lists were a slice away from disagreeing.
                            "label_scores": [
                                {lbl: float(sc)} for lbl, sc in zip(result.labels[:top_k], result.scores[:top_k])
                            ],
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
                audio_results.append(
                    {
                        "start": bp / sr,
                        "end": min(bp + win_samples, n_samples) / sr,
                        "label_scores": [
                            {lbl: float(sc)} for lbl, sc in zip(result.labels[:top_k], result.scores[:top_k])
                        ],
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
    return [{"label": (top_label(r) or ("unknown", 0.0))[0], "start": r["start"], "end": r["end"]} for r in results]
