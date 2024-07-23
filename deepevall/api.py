from deepevall.deep_eval import evaluate_conversation
from typing import List, Dict, Union

def evaluate_chat(script_lines: List[ScriptLine]) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """Evaluate a chat conversation using defined metrics."""
    metrics = ["rouge1", "rouge2", "rougeL"]
    result = evaluate_conversation(script_lines, metrics)
    return {
        "overall_score": result["overall_score"],
        "metrics": result["metrics"]
    }
