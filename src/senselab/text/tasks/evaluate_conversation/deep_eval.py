"""deep_eval.py."""

from typing import Dict, List

from senselab.text.tasks.evaluate_conversation.metrics import (
    BaseMetric,
    BiasEvaluation,
    ReadabilityScore,
    RougeScore,
    TextStatistics,
    ToxicityEvaluation,
)
from senselab.utils.data_structures.transcript_output import TranscriptOutput


def evaluate_transcript_output(transcript_output: TranscriptOutput, selected_metrics: List[str]) -> TranscriptOutput:
    """Evaluate a conversation based on the provided transcript output and metrics.

    Args:
        transcript_output (TranscriptOutput): The transcript output to evaluate.
        selected_metrics (List[str]): A list of metrics to use for evaluation. Available metrics are:
            - "ReadabilityScore"
            - "RougeScore"
            - "TextStatistics"
            - "ToxicityEvaluation"
            - "BiasEvaluation"

    Returns:
        TranscriptOutput: The transcript output with evaluated metrics for each response.
    """
    if not transcript_output:
        raise ValueError("transcript output is empty!")

    available_metrics = {
        "ReadabilityScore": ReadabilityScore,
        "RougeScore": RougeScore,
        "TextStatistics": TextStatistics,
        "ToxicityEvaluation": ToxicityEvaluation,
        "BiasEvaluation": BiasEvaluation,
    }
    for name in selected_metrics:
        if name not in available_metrics:
            raise ValueError(f"Metric '{name}' is not available. Choose from {list(available_metrics.keys())}")

    selected_metrics_classes = [available_metrics[name] for name in selected_metrics]

    for i, response in enumerate(transcript_output.data):
        if response["speaker"] == "AI":
            assert transcript_output.data[i - 1]["speaker"] == "Tutor"
            response_reference_pair = (response["text"], transcript_output.data[i - 1]["text"])
            response["metrics"] = pipeline(response_reference_pair, selected_metrics_classes)

        if response["speaker"] == "Tutor":
            if i + 1 < len(transcript_output.data) and transcript_output.data[i + 1]["speaker"] == "AI":
                response_reference_pair = (response["text"], transcript_output.data[i + 1]["text"])
                response["metrics"] = pipeline(response_reference_pair, selected_metrics_classes)

    return transcript_output


def pipeline(response_reference_pair: tuple, selected_metrics_classes: list[type[BaseMetric]]) -> Dict:
    """Run the metric pipeline on a single text-reference pair.

    Args:
        response_reference_pair (tuple): A tuple containing the response and reference text.
        selected_metrics_classes (list[BaseMetric]): A list of metric classes to be used for evaluation.

    Returns:
        Dict: A dictionary containing the results of the computed metrics.
    """
    metrics = {}
    for metric_class in selected_metrics_classes:
        result = metric_class().compute_reference_pair(response_reference_pair)
        metrics.update(result)
    return metrics
