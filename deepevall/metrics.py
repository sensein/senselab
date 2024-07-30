"""metrics.py."""

from abc import ABC, abstractmethod
from typing import Dict, List

from rouge_score import rouge_scorer

"""
Module for defining metrics used in evaluation.
"""


class Metric(ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def measure(self, references: List[str], hypotheses: List[str]) -> List[Dict[str, Dict[str, float]]]:
        """Measure the metric.

        Args:
            references (List[str]): A list of reference strings.
            hypotheses (List[str]): A list of hypothesis strings.

        Returns:
            List[Dict[str, Dict[str, float]]]: A list of dictionaries containing the result of the measurement.
        """
        pass


class RougeMetric(Metric):
    """ROUGE metric calculation class."""

    def __init__(self, name: str = "rouge", description: str = "ROUGE metric calculation") -> None:
        """Initialize the ROUGE metric with a name and description.

        Args:
            name (str): The name of the metric.
            description (str): The description of the metric.
        """
        self.name = name
        self.description = description

    def measure(self, references: List[str], hypotheses: List[str]) -> List[Dict[str, Dict[str, float]]]:
        """Measure the ROUGE metric for the given references and hypotheses.

        Args:
            references (List[str]): A list of reference strings.
            hypotheses (List[str]): A list of hypothesis strings.

        Returns:
            List[Dict[str, Dict[str, float]]]: A list of dictionaries containing ROUGE scores.
        """
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        return [{key: value.fmeasure for key, value in score.items()} for score in scores]
