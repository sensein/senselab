from rouge_score import rouge_scorer
from typing import List, Dict
from deepevall.script_line import ScriptLine

class RougeMetric:
    """A class to calculate ROUGE metrics."""

    def __init__(self, name="rouge", description="ROUGE metric calculation") -> None:
        self.name = name
        self.description = description

    def measure(self, references: List[str], hypotheses: List[str]) -> List[Dict[str, float]]:
        """Measure ROUGE scores for the provided references and hypotheses."""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        return [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]

def evaluate_conversation(script_lines: List[ScriptLine], metrics: List[str]) -> Dict[str, Union[float, List[Dict[str, float]]]]:
    """Evaluate a conversation using ROUGE metrics."""
    if not script_lines:
        return {"overall_score": 0, "metrics": []}

    references = [line.text for line in script_lines if line.speaker == "agent"]
    hypotheses = [line.text for line in script_lines if line.speaker == "user"]

    if not references or not hypotheses:
        return {"overall_score": 0, "metrics": []}

    rouge_metric = RougeMetric()
    scores = rouge_metric.measure(references, hypotheses)
    
    overall_score = sum(score[metric].fmeasure for score in scores for metric in metrics) / (len(scores) * len(metrics))
    metrics_results = [{metric: score[metric].fmeasure for metric in metrics} for score in scores]

    return {"overall_score": overall_score, "metrics": metrics_results}
