"""api.py."""

from typing import Dict, List

from senselab.utils.data_structures.script_line import ScriptLine

from .deep_eval import evaluate_conversation

"""
API module for evaluating chats.
"""


def evaluate_chat(script_lines: List[ScriptLine]) -> Dict:
    """Evaluate chat using the provided script lines and metrics.

    Args:
        script_lines (List[ScriptLine]): A list of script lines to evaluate.

    Returns:
        dict: The standardized result with overall score and metrics.
    """
    metrics = ["rouge1", "rouge2", "rougeL"]  # Define the metrics you want to use
    result = evaluate_conversation(script_lines, metrics)
    standardized_result = {"overall_score": result["overall_score"], "metrics": result["metrics"]}
    return standardized_result
