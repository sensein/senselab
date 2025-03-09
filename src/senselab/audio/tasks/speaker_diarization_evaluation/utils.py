"""Defines utility functions for evaluating speaker diarization results."""

from typing import Dict, List

from senselab.utils.data_structures import ScriptLine

try:
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate, GreedyDiarizationErrorRate

    PYANNOTEAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    PYANNOTEAUDIO_AVAILABLE = False


def calculate_diarization_error_rate(
    hypothesis: List[ScriptLine],
    reference: List[ScriptLine],
    greedy: bool = False,
    return_speaker_mapping: bool = False,
    detailed: bool = False,
) -> Dict:
    """Computes the diarization error rate (DER).

    Diarizztion error rate is the ratio of the sum of the false alarms (when speech is detected but none is there),
    missed detections (when speech is there but not detected), and speaker confusions (when speech is
    attributed to the wrong speaker) to the total ground truth time spoken. For more details see:
    https://docs.kolena.com/metrics/diarization-error-rate/

    Args:
        hypothesis (List[ScriptLine]): the diarization generated as the result from a model
        reference (List[ScriptLine]): annotations that serve as the ground truth diarization
        greedy (bool): whether to use a greedy speaker mapping vs. one that optimizes for minimizing the confusion
        return_speaker_mapping (bool): return the mapping between speakers in the reference and the hypothesis
        detailed (bool): whether to include each component that contributed to the overall diarization error rate

    Returns:
        A dictionary with at least the diarization error rate, its components if detailed was true, and the
          speaker mapping if return_speaker_mapping was given.
    """
    if not PYANNOTEAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`pyannote-audio` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`."
        )

    hypothesis_annotation = Annotation()
    reference_annotation = Annotation()

    for line in hypothesis:
        assert line.speaker
        hypothesis_annotation[Segment(line.start, line.end)] = line.speaker

    for line in reference:
        assert line.speaker
        reference_annotation[Segment(line.start, line.end)] = line.speaker

    metric = GreedyDiarizationErrorRate() if greedy else DiarizationErrorRate()
    der = metric(reference_annotation, hypothesis_annotation, detailed=detailed)
    output = {"diarization error rate": der} if not detailed else der
    if return_speaker_mapping:
        mapping_fn = metric.greedy_mapping if greedy else metric.optimal_mapping
        speaker_mapping = mapping_fn(reference_annotation, hypothesis_annotation)
        output["speaker_mapping"] = speaker_mapping

    return output
