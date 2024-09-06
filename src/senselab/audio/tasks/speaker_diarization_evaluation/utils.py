"""Defines utility functions for evaluating speaker diarization results."""

from typing import Dict, List

from interval import interval as pyinterval

from senselab.utils.data_structures.script_line import ScriptLine


def calculate_diarization_error_rate(
    inference: List[ScriptLine], ground_truth: List[ScriptLine], speaker_mapping: Dict[str, str], detailed: bool = False
) -> Dict[str, float] | float:
    """Computes the diarization error rate (DER).

    Diarizztion error rate is the sum of the false alarms (when speech is detected but none is there),
    missed detections (when speech is there but not detected), and speaker confusions (when speech is
    attributed to the wrong speaker). For more details see:
    https://docs.kolena.com/metrics/diarization-error-rate/

    Args:
        inference (List[ScriptLine]): the diarization generated as the result from a model
        ground_truth (List[ScriptLine]): annotations that serve as the ground truth diarization
        speaker_mapping (Dict[str, str]): mapping between speakers in inference and in ground_truth
        detailed (bool): whether to include each component that contributed to the overall diarization error rate

    Returns:
        Either the diarization error rate (float) or a dictionary containing the diarization error rate and its
          individual components. The individual components typically have units of seconds.
    """
    inference_interval = pyinterval()
    ground_truth_interval = pyinterval()
    speaker_inference_intervals = {}
    speaker_ground_truth_intervals = {}

    for line in inference:
        assert line.speaker
        inference_interval = inference_interval | pyinterval[line.start, line.end]
        if line.speaker not in speaker_inference_intervals:
            speaker_inference_intervals[line.speaker] = pyinterval[line.start, line.end]
        else:
            tmp = speaker_inference_intervals[line.speaker] | pyinterval[line.start, line.end]
            speaker_inference_intervals[line.speaker] = tmp

    for line in ground_truth:
        assert line.speaker
        ground_truth_interval = ground_truth_interval | pyinterval[line.start, line.end]
        if line.speaker not in speaker_ground_truth_intervals:
            speaker_ground_truth_intervals[line.speaker] = pyinterval[line.start, line.end]
        else:
            tmp = speaker_ground_truth_intervals[line.speaker] | pyinterval[line.start, line.end]
            speaker_ground_truth_intervals[line.speaker] = tmp

    inference_interval_length = _interval_length(inference_interval)
    ground_truth_length = _interval_length(ground_truth_interval)
    confusion_rate = _speaker_confusion(
        speaker_inference_intervals,
        speaker_ground_truth_intervals,
        ground_truth_interval,
        speaker_mapping=speaker_mapping,
    )
    false_alarms = _false_alarms(inference_interval, ground_truth_interval, ground_truth_length)
    missed_detections = _missed_detection(inference_interval, ground_truth_interval, inference_interval_length)

    der = (confusion_rate + false_alarms + missed_detections) / ground_truth_length

    if detailed:
        return {
            "false_alarms": false_alarms,
            "missed_detections": missed_detections,
            "speaker_confusions": confusion_rate,
            "der": der,
        }
    else:
        return der


def _false_alarms(inference: pyinterval, ground_truth: pyinterval, ground_truth_length: float) -> float:
    """Calculate the amount of false alarms.

    Calculates the amount of false alarms comparing the total amount of time the union of each
      inference and the ground truth adds.
    """
    false_alarms = 0.0
    # print(ground_truth)
    for interval in inference.components:
        # print(interval)
        false_alarms += _interval_length(interval | ground_truth) - ground_truth_length
    return false_alarms


def _missed_detection(inference: pyinterval, ground_truth: pyinterval, inference_length: float) -> float:
    """Calculate amount of missed detections.

    Calculates the amount of missed detections by comparing the total amount of time the union of each
      ground truth segment and inferred diariazion adds.
    """
    missed_detections = 0.0
    for interval in ground_truth.components:
        missed_detections += _interval_length(interval | inference) - inference_length
    return missed_detections


def _speaker_confusion(
    inferred_speaker_intervals: Dict[str, pyinterval],
    true_speaker_intervals: Dict[str, pyinterval],
    ground_truth: pyinterval,
    speaker_mapping: Dict[str, str],
) -> float:
    """Calculate amount of speaker confusion.

    Calculates the amount of speaker confusion by testing for each inferred speaker the amount of time
      that is added when their inferred speech segments are intersected with their ground truth segments vs.
      when they are intersected with the entire ground truth.
    """
    confusion = 0.0
    for inferred_speaker, inferred_speaker_interval in inferred_speaker_intervals.items():
        total_overlap = _interval_length(inferred_speaker_interval & ground_truth)
        equivalent_true_speaker = speaker_mapping[inferred_speaker]
        non_confused_overlap = 0.0
        if equivalent_true_speaker:
            ground_truth_speaker_interval = true_speaker_intervals[equivalent_true_speaker]
            non_confused_overlap = _interval_length(inferred_speaker_interval & ground_truth_speaker_interval)
        confusion += total_overlap - non_confused_overlap
    return confusion


def _interval_length(interval: pyinterval) -> float:
    """Calculates the length in time that the interval represents."""
    return sum([x[1] - x[0] for x in interval])
