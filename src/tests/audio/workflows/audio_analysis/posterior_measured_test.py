"""The speaker-count posterior weights by measurement, not by declared source kind.

The axis aggregator was converted first; the posterior still resolved a source's authority
from a ``kind`` written into policy. That is the same judgement-from-one-recording the axis
change removed, and it produced the same result: on a 4.9 s group introduction the posterior
held 1 speaker at 0.83 against 5 at 0.17, the minority reading being the one that matched the
five names actually spoken.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.speaker_identity import (
    SourceCountClaim,
    speaker_count_posterior,
)

GATES = {"independent": 1.0, "derived": 0.4}


def test_a_claim_with_unsupported_speech_carries_less_weight() -> None:
    """Physical support replaces the declared gate as the second factor.

    A source whose speaker claims sit where no voice detector reports speech has made claims
    the recording does not back, and that is measured rather than assigned.
    """
    p = speaker_count_posterior(
        [
            SourceCountClaim("a", 1, support=1.0),
            SourceCountClaim("b", 4, support=0.1),
        ],
        gates=GATES,
    )
    assert p.probabilities[1] > p.probabilities[4]
    assert p.weights["b"] < p.weights["a"]


def test_equally_supported_sources_carry_equal_weight() -> None:
    """No source is privileged by name — the property the declared gate violated."""
    p = speaker_count_posterior(
        [SourceCountClaim("embedding_silhouette", 5), SourceCountClaim("pyannote", 1)],
        gates=GATES,
    )
    assert p.weights["embedding_silhouette"] == pytest.approx(p.weights["pyannote"])


def test_a_minority_reading_can_win_on_support() -> None:
    """The audio_48k case, stated as a property.

    Two sources agreeing is not authority: if their claims are unsupported by the voice
    detectors while a third source's are supported, the third should prevail.
    """
    p = speaker_count_posterior(
        [
            SourceCountClaim("merger_a", 1, support=0.2),
            SourceCountClaim("merger_b", 1, support=0.2),
            SourceCountClaim("splitter", 5, support=1.0),
        ],
        gates=GATES,
    )
    assert p.modal_count == 5


def test_support_and_self_uncertainty_compound() -> None:
    """Both factors apply; neither subsumes the other."""
    both = speaker_count_posterior([SourceCountClaim("a", 1, uncertainty=0.8, support=0.2)], gates=GATES)
    one = speaker_count_posterior([SourceCountClaim("a", 1, uncertainty=0.8, support=1.0)], gates=GATES)
    assert both.weights["a"] < one.weights["a"]


def test_an_unmeasured_source_keeps_full_support() -> None:
    """A factor never gathered must not act as a discount."""
    p = speaker_count_posterior([SourceCountClaim("a", 1)], gates=GATES)
    assert p.weights["a"] == pytest.approx(1.0)


def test_support_is_reported_per_source() -> None:
    """FR-006: an analyst must be able to see why a count carried the weight it did."""
    doc = speaker_count_posterior(
        [SourceCountClaim("a", 1, support=0.3), SourceCountClaim("b", 2)], gates=GATES
    ).to_json()
    assert doc["source_support"] == {"a": 0.3, "b": 1.0}
    # The per-count attribution must survive alongside it: both were briefly named "support",
    # and the second silently replaced the first.
    assert doc["support"] == {"1": ["a"], "2": ["b"]}


def test_out_of_range_support_is_refused() -> None:
    """A support outside [0, 1] would silently invert or inflate a weight."""
    with pytest.raises(ValueError, match="support"):
        speaker_count_posterior([SourceCountClaim("a", 1, support=1.5)], gates=GATES)


def test_no_source_name_appears_in_the_posterior_weighting() -> None:
    """The same regression guard the axis has, applied to the posterior."""
    import inspect

    from senselab.audio.workflows.audio_analysis import speaker_identity

    source = inspect.getsource(speaker_identity.speaker_count_posterior)
    for name in ("embedding_silhouette", "pyannote", "sortformer", "speechbrain"):
        assert name not in source
