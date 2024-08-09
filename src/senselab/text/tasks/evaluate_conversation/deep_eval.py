"""deep_eval.py."""

from typing import Dict, List

from senselab.utils.data_structures.script_line import ScriptLine

from .metrics import RougeMetric


def evaluate_conversation(script_lines: List[ScriptLine], metrics: List[str]) -> Dict:
    """Evaluate a conversation based on the provided script lines and metrics.

    Args:
        script_lines (List[ScriptLine]): A list of script lines to evaluate.
        metrics (List[str]): A list of metrics to use for evaluation.

    Returns:
        dict: The evaluation result containing detailed metrics.
    """
    if not script_lines:
        return {"metrics": []}
    references: List[str] = [line.text for line in script_lines if line.speaker == "agent" and line.text is not None]
    hypotheses: List[str] = [line.text for line in script_lines if line.speaker == "user" and line.text is not None]

    if not references or not hypotheses:
        return {"metrics": []}

    metric_instance = RougeMetric()
    scores = metric_instance.measure(references, hypotheses)

    metrics_results = [{metric: score.get(metric, 0.0) for metric in metrics} for score in scores]

    return {"metrics": metrics_results}
