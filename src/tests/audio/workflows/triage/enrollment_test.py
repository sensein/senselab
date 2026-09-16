"""The enrollment input: a subject's target vector, with the model and revision behind it."""

import pytest
from pydantic import ValidationError

from senselab.audio.data_structures import SpeakerEmbeddingProvenance
from senselab.audio.workflows.triage.enrollment import Enrollment


def _provenance(**overrides: object) -> SpeakerEmbeddingProvenance:
    """A provenance record naming a model and a resolved commit.

    Args:
        overrides: Fields replacing the defaults.

    Returns:
        The provenance record.
    """
    fields: dict[str, object] = {
        "model_id": "speechbrain/spkrec-ecapa-voxceleb",
        "model_commit_sha": "a" * 40,
        "source_files": ["a.wav", "b.wav"],
    }
    fields.update(overrides)
    return SpeakerEmbeddingProvenance(**fields)  # type: ignore[arg-type]


class TestTheShape:
    """subject_id, vector, provenance, sources — every recording behind the vector is named."""

    def test_an_enrollment_names_every_recording_behind_it(self) -> None:
        """``sources`` is what makes an enrollment reproducible and a file's own contribution visible."""
        enrollment = Enrollment(subject_id="sub-01", vector=[0.6, 0.8], provenance=_provenance())
        assert enrollment.sources == ["a.wav", "b.wav"]

    def test_the_vector_must_be_non_empty(self) -> None:
        """A zero-length embedding is compared against nothing."""
        with pytest.raises(ValidationError):
            Enrollment(subject_id="sub-01", vector=[], provenance=_provenance())

    def test_a_non_finite_component_is_refused(self) -> None:
        """No similarity is defined over a NaN or an infinity."""
        with pytest.raises(ValidationError):
            Enrollment(subject_id="sub-01", vector=[1.0, float("nan")], provenance=_provenance())


class TestRefusal:
    """An enrollment that cannot be compared is refused, and the refusal names why."""

    def test_a_missing_commit_is_refused(self) -> None:
        """Two commits of one model are not comparable, so a bare model id is not provenance."""
        enrollment = Enrollment(
            subject_id="sub-01",
            vector=[1.0],
            provenance=_provenance(model_commit_sha=None, unresolved_reason="hub outage"),
        )
        assert "resolved model commit" in (
            enrollment.refusal_against("speechbrain/spkrec-ecapa-voxceleb", "a" * 40) or ""
        )

    def test_a_different_model_is_refused(self) -> None:
        """Embeddings from different models are not comparable at any threshold."""
        enrollment = Enrollment(subject_id="sub-01", vector=[1.0], provenance=_provenance())
        assert "not the probe" in (enrollment.refusal_against("pyannote/embedding", "a" * 40) or "")

    def test_a_different_commit_of_the_same_model_is_refused(self) -> None:
        """branch-speech.md section 6: 'from two commits of one model, are not comparable'."""
        enrollment = Enrollment(subject_id="sub-01", vector=[1.0], provenance=_provenance())
        refusal = enrollment.refusal_against("speechbrain/spkrec-ecapa-voxceleb", "b" * 40) or ""
        assert "two commits of one model are not comparable" in refusal

    def test_an_unpinned_probe_is_refused_like_a_mismatched_one(self) -> None:
        """A ref resolves to whatever it points at today, so it names no commit to agree with."""
        enrollment = Enrollment(subject_id="sub-01", vector=[1.0], provenance=_provenance())
        assert "carries no resolved model commit" in (
            enrollment.refusal_against("speechbrain/spkrec-ecapa-voxceleb", None) or ""
        )
        assert "not comparable" in (enrollment.refusal_against("speechbrain/spkrec-ecapa-voxceleb", "main") or "")

    def test_a_matching_model_and_commit_is_comparable(self) -> None:
        """The one case that is not a refusal: same model, same resolved commit."""
        enrollment = Enrollment(subject_id="sub-01", vector=[1.0], provenance=_provenance())
        assert enrollment.refusal_against("speechbrain/spkrec-ecapa-voxceleb", "a" * 40) is None
