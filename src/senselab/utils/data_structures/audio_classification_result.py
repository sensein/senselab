"""This module contains the definition of the AudioClassificationResult class."""

from typing import Dict, List, Tuple

from pydantic import BaseModel, model_validator


class AudioClassificationResult(BaseModel):
    """A class to represent a classification result.

    Attributes:
        labels (List[str]): Labels that the classification model is predicting over.
        scores (List[float]): The score for each label the classification model outputs.
    """

    labels: List[str] = []
    scores: List[float] = []

    @model_validator(mode="after")
    def validate_labels_and_scores(self) -> "AudioClassificationResult":
        """Validates that the labels and scores are the same length and sort them."""
        labels = self.labels
        scores = self.scores

        if len(labels) != len(scores):
            raise ValueError("'labels' and 'scores' should be the same length.")

        combined = list(zip(labels, scores))
        combined.sort(key=lambda x: x[1], reverse=True)  # sort scores (and their labels) in reverse order

        sorted_labels, sorted_scores = zip(*combined)
        self.labels = list(sorted_labels)
        self.scores = list(sorted_scores)

        return self

    def get_labels(self) -> List[str]:
        """Get the labels that the model was classifying over.

        Returns:
            List[str]: The full list of labels the model was classifying over,
              are in order of highest to lowest classification score.
        """
        return self.labels

    def get_scores(self) -> List[float]:
        """Get the labels that the model was classifying over.

        Returns:
            List[float]: The full list of scores the model output for each label,
              in order of highest to lowest classification score.
        """
        return self.scores

    def get_label_score_pairs(self) -> List[Tuple[str, float]]:
        """Get the labels and their scores.

        Returns:
            List[Tuple[str,float]]: Pairs of labels and their associated scores from
              the classification model, from highest to lowest score.
        """
        return list(zip(self.labels, self.scores))

    @classmethod
    def from_hf_classification_pipeline(cls, results: List[Dict]) -> "AudioClassificationResult":
        """Creates an AudioClassificationResult instance from HuggingFace's pipeline output.

        Args:
            results (List[Dict]): List of dictionaries of the form {"score": score, "label": label}.

        Returns:
            AudioClassificationResult: An instance of AudioClassificationResult.

        Raises:
            ValueError if results dictionaries are not of the form described.
        """
        labels = []
        scores = []

        for classification_result in results:
            if "label" not in classification_result or "score" not in classification_result:
                raise ValueError("Result dictionary is not of the form {'label': label, 'score': score}")
            labels.append(classification_result["label"])
            scores.append(classification_result["score"])

        return cls(labels=labels, scores=scores)
