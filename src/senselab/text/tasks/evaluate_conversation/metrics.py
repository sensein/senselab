"""Metrics to assess performance on tutor response.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.

All other functions are value-independent.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import textstat
from deepeval.metrics import BiasMetric, PromptAlignmentMetric, ToxicityMetric
from deepeval.test_case import LLMTestCaseParams
from rouge_score import rouge_scorer


class BaseMetric(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def compute_reference_pair(self: "BaseMetric", pair: tuple) -> Dict[str, Any]:
        """Compute the reference pair metrics."""
        raise NotImplementedError


class SingleTextMetric(BaseMetric):
    """Base class for metrics that only need one text."""

    @abstractmethod
    def compute(self: "SingleTextMetric", text: str) -> Dict[str, Any]:
        """Compute metrics for the given text.

        Args:
            text (str): The input text to evaluate.
            reference (str): The reference text to compare against. Will always be None.

        Returns:
            Dict[str, Any]: A dictionary containing the computed metrics.
        """
        raise NotImplementedError

    def compute_reference_pair(self: "SingleTextMetric", pair: tuple) -> Dict[str, Any]:
        """Compute the reference pair.

        This method takes a tuple containing two elements and computes the reference
        for the first element in the tuple. It asserts that the tuple has exactly two elements.

        Args:
            pair (tuple): A tuple containing two elements.

        Returns:
            Dict[str, Any]: The computed reference for the first element in the tuple.
        """
        assert len(pair) == 2
        return self.compute(pair[0])


class ComparativeMetric(BaseMetric):
    """Base class for metrics that compare two texts."""

    @abstractmethod
    def compute(self: "ComparativeMetric", text: str, reference_text: str) -> Dict[str, Any]:
        """Computes the evaluation metrics for a given text against a reference text.

        Args:
            text (str): The text to be evaluated.
            reference_text (str): The reference text to compare against.

        Returns:
            Dict[str, Any]: A dictionary containing the computed metrics.
        """
        raise NotImplementedError

    def compute_reference_pair(self: "ComparativeMetric", pair: tuple) -> Dict[str, Any]:
        """Compute the reference pair metrics.

        This method takes a tuple containing two elements and computes the metrics
        by calling the `compute` method with the two elements of the tuple.

        Args:
            pair (tuple): A tuple containing exactly two elements.

        Returns:
            Dict[str, Any]: A dictionary containing the computed metrics.

        Raises:
            AssertionError: If the length of the tuple is not equal to 2.
        """
        assert len(pair) == 2
        return self.compute(pair[0], pair[1])


class ReadabilityScore(SingleTextMetric):
    """Class to compute readability scores for a text."""

    def compute(self: "ReadabilityScore", text: str) -> Dict[str, float]:
        """Compute the readability and syntactic complexity scores for the text using Flesch metrics.

        The Flesch Reading Ease score indicates how easy a text is to read.
        Higher scores indicate easier readability, with scores ranging from 0 to 100.
        - 90-100: Very easy to read, easily understood by an average 11-year-old student.
        - 60-70: Plain English, easily understood by 13- to 15-year-old students.
        - 0-30: Very difficult to read, best understood by university graduates.

        The Flesch-Kincaid Grade Level score indicates the grade level required to understand the text.
        Lower scores indicate easier readability, with scores corresponding to U.S. school grade levels.

        Args:
            text: The model response text

        Returns:
            Dictionary containing the computed readability and grade level scores:
            - "readability_score": Flesch Reading Ease score
            - "grade_level_score": Flesch-Kincaid Grade Level score
        """
        return {
            "readability_score": textstat.flesch_reading_ease(text),
            "grade_level_score": textstat.flesch_kincaid_grade(text),
        }


class RougeScore(ComparativeMetric):
    """Class to compute ROUGE scores for a text against a reference text."""

    def compute(self: "RougeScore", text: str, reference_text: str) -> Dict[str, float]:
        """Compute the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores for the text.

        ROUGE is a set of metrics for evaluating automatic summarization and machine translation.
        It compares an automatically produced summary or translation against a reference or a set of references.

        The following ROUGE metrics are computed:
        - ROUGE-1: Overlap of unigrams (single words) between the system and reference summaries.
        - ROUGE-2: Overlap of bigrams (two consecutive words) between the system and reference summaries.
        - ROUGE-L: Longest common subsequence (LCS) between the system and reference summaries.

        Args:
            text: The model response text
            reference_text: The human tutor response text

        Returns:
            Dictionary containing the computed ROUGE scores:
            - "rouge-1": ROUGE-1 score
            - "rouge-2": ROUGE-2 score
            - "rouge-l": ROUGE-L score
        """
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

        scores = scorer.score(text, reference_text)
        return {
            "rouge-1": scores["rouge-1"]["f"],
            "rouge-2": scores["rouge-2"]["f"],
            "rouge-l": scores["rouge-l"]["f"],
        }


class TextStatistics(SingleTextMetric):
    """Class to compute basic text statistics such as word count and sentence count."""

    def compute(self: "TextStatistics", text: str) -> Dict[str, int]:
        """Compute basic text statistics such as word count and sentence count.

        Args:
            text: The model response text

        Returns:
            Dictionary containing the computed text statistics:
            - "word_count": Number of words in the text
            - "sentence_count": Number of sentences in the text
            - "char_count": Number of characters in the text
        """
        return {
            "word_count": textstat.lexicon_count(text),
            "sentence_count": textstat.sentence_count(text),
            "char_count": textstat.char_count(text),
        }


class ToxicityEvaluation(SingleTextMetric):
    """Evaluates text toxicity using deepeval's ToxicityMetric."""

    def __init__(self: "ToxicityEvaluation") -> None:
        """Initializes the evaluator with a ToxicityMetric instance."""
        self.evaluator = ToxicityMetric()

    def compute(self: "ToxicityEvaluation", text: str) -> Dict[str, Any]:
        """Compute the toxicity score for the given text.

        The toxicity metric measures the presence of harmful, offensive, or
        inappropriate content in the text. Lower scores indicate less toxic content.

        Args:
            text: The model response text

        Returns:
            Dictionary containing the computed toxicity metrics:
            - "toxicity_score": Overall toxicity score between 0 and 1
            - "toxicity_labels": Specific toxicity categories detected
            - "is_toxic": Boolean indicating if the text exceeds toxicity thresholds
        """
        test_case = LLMTestCaseParams(actual_output=text)

        evaluation = self.evaluator.measure(test_case)
        return {
            "toxicity_score": evaluation.score,
            "toxicity_labels": evaluation.toxicity_labels if hasattr(evaluation, "toxicity_labels") else [],
            "is_toxic": not evaluation.passed,
        }


class BiasEvaluation(SingleTextMetric):
    """Evaluates text bias using deepeval's BiasMetric."""

    def __init__(self: "BiasEvaluation") -> None:
        """Initializes the BiasMetric evaluator."""
        self.evaluator = BiasMetric()

    def compute(self: "BiasEvaluation", text: str) -> Dict[str, Any]:
        """Compute the bias score for the given text.

        The bias metric detects various forms of bias including gender, racial,
        and other demographic biases in the text. Lower scores indicate less bias.

        Args:
            text: The model response text

        Returns:
            Dictionary containing the computed bias metrics:
            - "bias_score": Overall bias score between 0 and 1
            - "bias_types": Types of bias detected
            - "has_bias": Boolean indicating if the text contains significant bias
        """
        test_case = LLMTestCaseParams(actual_output=text)

        evaluation = self.evaluator.measure(test_case)
        return {
            "bias_score": evaluation.score,
            "bias_types": evaluation.bias_types if hasattr(evaluation, "bias_types") else [],
            "has_bias": not evaluation.passed,
        }


class PromptAlignmentEvaluation(ComparativeMetric):
    """Evaluates how well the response aligns with the original prompt using deepeval's PromptAlignmentMetric."""

    # TODO

    # This should compare the LLM response to the system instruction, not the tutor response

    def __init__(self: "PromptAlignmentEvaluation") -> None:
        """Initializes the evaluator with a PromptAlignmentMetric instance."""
        self.evaluator = PromptAlignmentMetric()

    def compute(self: "PromptAlignmentEvaluation", text: str, reference_text: str) -> Dict[str, float]:
        """Compute how well the response aligns with the given prompt/reference.

        The prompt alignment metric measures how well the model's response addresses
        and follows the requirements and context of the original prompt.

        Args:
            text: The model response text
            reference_text: The system instruction

        Returns:
            Dictionary containing the computed alignment metrics:
            - "alignment_score": Overall alignment score between 0 and 1
            - "is_aligned": Boolean indicating if the response adequately aligns with the prompt
        """
        test_case = LLMTestCaseParams(actual_output=text, expected_output=reference_text)

        evaluation = self.evaluator.measure(test_case)
        return {"alignment_score": evaluation.score, "is_aligned": evaluation.passed}
