"""PII detection over audio: transcribe, then scan the transcript.

This module exists under ``audio/`` rather than beside the detection logic in
``text/tasks/pii_detection`` for one reason: it needs ``transcribe_audios``. A
module under ``text/`` importing from ``audio/`` would invert the layering that
put detection under ``text/`` in the first place. Importing an audio task
(``speech_to_text``) from another audio task (this one) is an ordinary
task-to-task dependency, not a workflow coupling -- neither this module nor
``text/tasks/pii_detection`` imports anything from
``senselab.audio.workflows.audio_analysis``.
"""

from typing import Any, List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.text.tasks.pii_detection import PiiReport, detect_pii, flatten_script_line
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel

# whisper-tiny: the smallest Whisper checkpoint, chosen so this entry point is
# usable out of the box without the caller having to pick a model first, and
# so it stays cheap enough to exercise in CI. Matches the same default choice
# senselab.audio.workflows.explore_conversation makes for the same "just get
# me a transcript" case.
_DEFAULT_ASR_MODEL_ID = "openai/whisper-tiny"


def detect_pii_in_audios(
    audios: List[Audio],
    asr_model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
    **detect_kwargs: Any,  # noqa: ANN401
) -> List[PiiReport]:
    """Transcribe each audio and scan its transcript for PII.

    One report per input audio, in order. This is a two-step composition, not a
    new detection engine: :func:`~senselab.audio.tasks.speech_to_text.transcribe_audios`
    produces the text, :func:`~senselab.text.tasks.pii_detection.detect_pii` scans it.
    Keeping the transcript passed through unchanged (via
    :func:`~senselab.text.tasks.pii_detection.flatten_script_line`) matters because it
    keeps a transcription bug and a detection bug distinguishable -- a caller debugging
    a bad report can rule one of the two steps out independently.

    Args:
        audios: Audio objects to scan for PII. Typical ASR models expect mono, fixed
            sampling rate; see :func:`transcribe_audios` for the same constraint.
        asr_model: ASR model used to produce the transcript. Defaults to
            ``HFModel(path_or_uri="openai/whisper-tiny")`` when omitted --
            deliberately the smallest available Whisper checkpoint rather than a
            larger, more accurate one, so this entry point stays cheap to call
            without configuration. Callers who need a different model (a
            domain-tuned checkpoint, a non-English model, higher accuracy) pass
            it explicitly; there is no environment-variable or config-file
            override, matching :func:`transcribe_audios` itself.
        device: Preferred device for ASR inference. ``None`` lets the backend
            choose (CUDA if available, else CPU).
        **detect_kwargs: Forwarded to :func:`~senselab.text.tasks.pii_detection.detect_pii`
            (e.g. ``detectors``, ``presidio_score_threshold``, ``gliner_model``,
            ``gliner_labels``, ``gliner_threshold``, ``require_cross_source_corroboration``).

    Returns:
        One ``PiiReport`` per input audio, same order as ``audios``. An audio whose
        transcript comes back empty (silence, an ASR that produced nothing, a
        diarization-only line with no text) is **not** reported as PII-free with high
        confidence: it flattens to the empty string, which ``detect_pii`` treats as
        "never reached a detector" -- ``detector_used=None`` and
        ``detection_confidence=None`` on that report, the same honest "did not run"
        signal a caller gets from any other empty input, rather than the deceptive
        ``detection_confidence=0.0`` that would read as "checked, found nothing."
    """
    model = asr_model if asr_model is not None else HFModel(path_or_uri=_DEFAULT_ASR_MODEL_ID)
    transcripts = transcribe_audios(audios=audios, model=model, device=device)
    texts = [flatten_script_line(line) for line in transcripts]
    reports = detect_pii(texts, **detect_kwargs)
    return reports if isinstance(reports, list) else [reports]
