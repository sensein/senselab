"""Tests for H2: the common speaker space, and the uncertainty of building it (D-6).

Each diarizer names speakers arbitrarily -- ``SPEAKER_00``, ``spk0`` -- so any cross-model
comparison first *guesses* that two labels denote the same person. Propagated as fact, that guess
makes models which were never correctly compared read as disagreeing, which is how speaker
uncertainty stayed high in regions where per-speaker presence was unambiguous.

So harmonization is an estimation step with its own uncertainty. Two independent matchers run --
temporal overlap and embedding centroid -- and where they disagree, that disagreement *is* the
assignment uncertainty.
"""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.harmonize import (
    harmonize_speaker_labels,
    overlap_assignment,
)


def _segs(*spans: tuple[float, float, str]) -> list[dict]:
    return [{"start": s, "end": e, "speaker": lab} for s, e, lab in spans]


def test_overlap_assignment_matches_permuted_labels() -> None:
    """Identical timelines under different label names map onto each other."""
    a = _segs((0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01"))
    b = _segs((0.0, 2.0, "spk1"), (2.0, 4.0, "spk0"))  # same timeline, names swapped
    assert overlap_assignment(a, b) == {"SPEAKER_00": "spk1", "SPEAKER_01": "spk0"}


def test_overlap_assignment_is_one_to_one() -> None:
    """Two labels of one model never collapse onto a single label of another.

    A greedy nearest-match would assign both of A's speakers to whichever of B's overlaps most,
    silently merging two people. The assignment must be a matching, not a lookup.
    """
    a = _segs((0.0, 3.0, "A0"), (3.0, 4.0, "A1"))
    b = _segs((0.0, 2.5, "B0"), (2.5, 4.0, "B1"))
    mapping = overlap_assignment(a, b)
    assert len(set(mapping.values())) == len(mapping)


def test_agreeing_matchers_give_confident_assignment() -> None:
    """When overlap and centroids agree, the assignment carries no uncertainty."""
    per_model = {
        "pyannote": _segs((0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")),
        "sortformer": _segs((0.0, 2.0, "spk1"), (2.0, 4.0, "spk0")),
    }
    # Centroids consistent with the same pairing: SPEAKER_00~spk1, SPEAKER_01~spk0.
    voice_x = np.array([1.0, 0.0, 0.0])
    voice_y = np.array([0.0, 1.0, 0.0])
    centroids = {
        ("pyannote", "SPEAKER_00"): voice_x,
        ("pyannote", "SPEAKER_01"): voice_y,
        ("sortformer", "spk1"): voice_x,
        ("sortformer", "spk0"): voice_y,
    }
    h = harmonize_speaker_labels(per_model, centroids=centroids)
    # Both models' labels for the same voice land on one common id.
    assert h.mapping[("pyannote", "SPEAKER_00")] == h.mapping[("sortformer", "spk1")]
    assert h.mapping[("pyannote", "SPEAKER_01")] == h.mapping[("sortformer", "spk0")]
    assert h.mapping[("pyannote", "SPEAKER_00")] != h.mapping[("pyannote", "SPEAKER_01")]
    # Agreement is only meaningful for the models mapped *into* the space. The reference model's
    # own labels define it, so there is nothing there for a second matcher to corroborate.
    for key in (k for k in centroids if k[0] == "sortformer"):
        assert h.methods_agreed[key] is True
        assert h.uncertainty[key] == pytest.approx(0.0)
        assert h.confidence[key] == pytest.approx(1.0)
    for key in (k for k in centroids if k[0] == "pyannote"):
        assert h.methods_agreed[key] is None, "the reference defines the space"
        assert h.uncertainty[key] == pytest.approx(0.0), "correct by construction"


def test_disagreeing_matchers_produce_maximal_assignment_uncertainty() -> None:
    """Overlap says one pairing, centroids say the other → the assignment is undetermined.

    Reporting a point assignment here, as a single matcher must, would hand downstream code a
    guess dressed as a fact.
    """
    per_model = {
        "pyannote": _segs((0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")),
        "sortformer": _segs((0.0, 2.0, "spk0"), (2.0, 4.0, "spk1")),
    }
    # Centroids assert the *opposite* pairing to the timelines.
    voice_x = np.array([1.0, 0.0, 0.0])
    voice_y = np.array([0.0, 1.0, 0.0])
    centroids = {
        ("pyannote", "SPEAKER_00"): voice_x,
        ("pyannote", "SPEAKER_01"): voice_y,
        ("sortformer", "spk0"): voice_y,  # overlaps SPEAKER_00 but sounds like SPEAKER_01
        ("sortformer", "spk1"): voice_x,
    }
    h = harmonize_speaker_labels(per_model, centroids=centroids)
    contested = [k for k in centroids if k[0] == "sortformer"]
    assert any(h.methods_agreed[k] is False for k in contested)
    assert max(h.uncertainty[k] for k in contested) == pytest.approx(1.0)
    assert min(h.confidence[k] for k in contested) <= 0.5


def test_single_matcher_is_not_reported_as_certain() -> None:
    """Without embeddings only overlap runs, and one matcher cannot corroborate itself."""
    per_model = {
        "pyannote": _segs((0.0, 2.0, "SPEAKER_00")),
        "sortformer": _segs((0.0, 2.0, "spk0")),
    }
    h = harmonize_speaker_labels(per_model, centroids=None)
    assert h.mapping[("pyannote", "SPEAKER_00")] == h.mapping[("sortformer", "spk0")]
    assert h.reference_model == "pyannote"
    for key in (k for k in h.mapping if k[0] != h.reference_model):
        assert h.methods_agreed[key] is None, "no second method ran, so agreement is unknown"
        assert h.uncertainty[key] is None, "unmeasured, not measured-and-zero"


def test_extra_speaker_gets_its_own_common_id() -> None:
    """A speaker one model found and another did not is not forced onto an existing id."""
    per_model = {
        "pyannote": _segs((0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")),
        "sortformer": _segs((0.0, 4.0, "spk0")),  # merged the two into one
    }
    h = harmonize_speaker_labels(per_model, centroids=None)
    common_ids = {h.mapping[k] for k in h.mapping}
    assert len(common_ids) == 2, "pyannote's two speakers remain two"
    assert h.mapping[("sortformer", "spk0")] in common_ids


def test_common_ids_are_stable_under_model_ordering() -> None:
    """The same inputs give the same ids regardless of dict iteration order.

    Ids appear in outputs and in the decision log, so an ordering-dependent id would make two
    identical runs look like they disagreed.
    """
    a = _segs((0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01"))
    b = _segs((0.0, 2.0, "spk1"), (2.0, 4.0, "spk0"))
    first = harmonize_speaker_labels({"pyannote": a, "sortformer": b}, centroids=None)
    second = harmonize_speaker_labels({"sortformer": b, "pyannote": a}, centroids=None)
    assert first.mapping == second.mapping


def test_no_segments_yields_an_empty_harmonization() -> None:
    """Nothing to harmonize is not an error."""
    h = harmonize_speaker_labels({}, centroids=None)
    assert h.mapping == {}
    assert h.uncertainty == {}
