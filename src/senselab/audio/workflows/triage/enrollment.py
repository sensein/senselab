"""The enrollment input: one subject's target-speaker vector, estimated across their recordings.

``specs/20260817-triage-workflow-dag/branch-speech.md`` section 6 is the contract.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from senselab.audio.data_structures import SpeakerEmbeddingProvenance
from senselab.utils.tasks.embedding_distribution import EmbeddingDistribution


class Enrollment(BaseModel):
    """A speaker embedding enrolled across all of one subject's provided recordings.

    Attributes:
        subject_id: Whose voice this is.
        vector: The embedding. Non-empty and finite, which is what this model enforces; the
            estimator is expected to return it unit-norm, and ``refusal_against`` does not depend on
            that because cosine similarity normalises either way.
        provenance: Required. Carries the embedding model and its **resolved** commit. An enrollment
            missing either, or naming a model or a commit the probe does not share, is refused
            rather than compared. Names every recording that contributed in ``source_files``.
        task: The vocal task the enrollment was estimated over, when one was declared.
        distribution: Spread over the contributing windows, when the estimator produced one.
    """

    subject_id: str
    vector: list[float] = Field(min_length=1)
    provenance: SpeakerEmbeddingProvenance
    task: Optional[str] = None
    distribution: Optional[EmbeddingDistribution] = None

    @property
    def sources(self) -> list[str]:
        """Every recording behind the vector.

        Returns:
            The contributing file ids, from the provenance.
        """
        return list(self.provenance.source_files)

    def refusal_against(self, probe_model_id: str, probe_commit: str | None) -> str | None:
        """Why this enrollment cannot be compared with the probe named by model and commit.

        Args:
            probe_model_id: The embedding model the branch will run over the diarized speakers.
            probe_commit: The resolved commit that model will be loaded at. A ref, or None, is
                not a commit and is refused like a mismatched one.

        Returns:
            The refusal, in controlled vocabulary, or None when the enrollment is comparable.
        """
        if self.provenance.model_commit_sha is None:
            return "the enrollment carries no resolved model commit; refused rather than compared"
        if self.provenance.model_id != probe_model_id:
            return (
                f"the enrollment's model {self.provenance.model_id} is not the probe {probe_model_id}; "
                "embeddings from different models are not comparable"
            )
        if probe_commit is None:
            return (
                f"the probe {probe_model_id} carries no resolved model commit; refused rather than "
                "compared against an enrollment that names one"
            )
        if self.provenance.model_commit_sha != probe_commit:
            return (
                f"the enrollment was estimated at commit {self.provenance.model_commit_sha} and the "
                f"probe resolves to {probe_commit}; two commits of one model are not comparable"
            )
        return None

    @field_validator("vector")
    @classmethod
    def _must_be_finite(cls, value: list[float]) -> list[float]:
        """Reject a vector carrying a non-finite component.

        Args:
            value: The candidate vector.

        Returns:
            The vector unchanged.

        Raises:
            ValueError: When any component is NaN or infinite, which no similarity is defined over.
        """
        if any(component != component or component in (float("inf"), float("-inf")) for component in value):
            raise ValueError("every component of an enrollment vector must be finite")
        return value
