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

    def top_label(self) -> str:
        """Gets the top label based on the highest score."""
        return self.labels[0]

    def top_score(self) -> float:
        """Gets the top score, based on highest value."""
        return self.scores[0]

    def top(self) -> Tuple[str, float]:
        """Gets the top label/score pairing, based on highest score value."""
        return self.labels[0], self.scores[0]

    def top_k_labels(self, k: int = 1) -> List[str]:
        """Gets the top k label(s) based on the highest score."""
        if k < 1 or k > len(self.labels):
            raise ValueError("k needs to be between 1 and the number of labels the model classifies over.")
        return self.labels[0:k]

    def top_k_score(self, k: int = 1) -> List[float]:
        """Gets the top k score(s), based on highest value."""
        if k < 1 or k > len(self.scores):
            raise ValueError("k needs to be between 1 and the number of labels the model classifies over.")
        return self.scores[0:k]

    def top_k(self, k: int = 1) -> List[Tuple[str, float]]:
        """Gets the top k label/score pairing(s), based on highest score value."""
        if k < 1 or k > len(self.labels):
            raise ValueError("k needs to be between 1 and the number of labels the model classifies over.")
        return list(zip(self.labels, self.scores))[0:k]

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
