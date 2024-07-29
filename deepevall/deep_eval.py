"""deep_eval.py."""

from statistics import harmonic_mean, mean
from typing import Dict, List

from senselab.utils.data_structures.script_line import ScriptLine

from .metrics import RougeMetric

"""
Module for evaluating conversations using various metrics.
"""


def evaluate_conversation(script_lines: List[ScriptLine], metrics: List[str], method: str = "mean") -> Dict:
    """Evaluate a conversation based on the provided script lines and metrics.

    Args:
        script_lines (List[ScriptLine]): A list of script lines to evaluate.
        metrics (List[str]): A list of metrics to use for evaluation.
        method (str): The method to calculate the overall score, either "mean" or "harmonic_mean".

    Returns:
        dict: The evaluation result containing overall score and detailed metrics.
    """
    if not script_lines:
        return {"overall_score": 0, "metrics": []}

    references = [line.text for line in script_lines if line.speaker == "agent"]
    hypotheses = [line.text for line in script_lines if line.speaker == "user"]

    if not references or not hypotheses:
        return {"overall_score": 0, "metrics": []}

    metric_instance = RougeMetric()
    scores = metric_instance.measure(references, hypotheses)

    if method == "mean":
        overall_score = mean([score[metric].fmeasure for score in scores for metric in metrics])
    elif method == "harmonic_mean":
        overall_score = harmonic_mean([score[metric].fmeasure for score in scores for metric in metrics])
    else:
        overall_score = mean([score[metric].fmeasure for score in scores for metric in metrics])

    metrics_results = [{metric: score[metric].fmeasure for metric in metrics} for score in scores]

    return {"overall_score": overall_score, "metrics": metrics_results}
