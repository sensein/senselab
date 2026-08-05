"""The loop's gates compare doubt against `theta_*`, not entropy — they are different scales.

Measured on a clean two-speaker conversation (`english_conversation_higgs_audio_v2.wav`), where the
speaker evidence is confident by every other reading:

    final/speakers.json   2 speakers at 0.978, is_multimodal false
                          speakers[0].existence_uncertainty = 0.0
    per-signal doubt      median 0.0000, mean 0.2072, 77.7% of readings <= 0.25

and yet the speaker axis reported `uncertainty` 0.666, seeding **114 of 214** buckets as
high-uncertainty regions and letting only 23 converge.

The cause is a scale mismatch, not a bad measurement. `uncertainty` is normalised **binary entropy**
of the mean per-signal doubt, and entropy climbs steeply away from zero: H(0.10) = 0.469,
H(0.20) = 0.722. `theta_high = 0.66` and `theta_low = 0.33` are written on a *doubt* scale (they are
the Label Studio high/low bins), so comparing them against entropy silently means "flag anything
above 17% doubt, converge only below 6%" — thresholds nobody chose. Solve H(p) = 0.66 → p = 0.171;
H(p) = 0.33 → p = 0.061.

Compounding it, the loop was spending most of that budget on doubt it cannot remove: of the speaker
axis's 0.666 total, aleatoric was 0.391 and epistemic 0.275, so **41.3%** of what drove region
proposal was reducible. `statistics.py` states the decomposition exists precisely so the loop can
tell those apart.

So the gates read `1 - confidence`: `confidence` is documented as a probability, which is the scale
`theta_*` are on. Each rule keeps its own reducibility test (U1/U2 gate on `epistemic_uncertainty`
themselves) — that is the right place for it, because "do my signals disagree" is a per-rule question
and `epistemic` is structurally 0 for a single-voter axis like `asr`, which must stay investigatable.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.adaptive.convergence import apply_convergence_marks
from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy
from senselab.audio.workflows.audio_analysis.adaptive.regions import propose_regions
from senselab.audio.workflows.audio_analysis.estimates import control_doubt


def _entropy(p: float) -> float:
    """Normalised binary entropy — what the `uncertainty` column holds."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2)


def _row(start: float, doubt: float, **extra: object) -> dict[str, Any]:
    """One estimates row for a bucket whose signals agree at ``doubt``.

    `uncertainty` is the entropy of that doubt and `confidence` its complement, exactly as
    `fuse.fuse_axis` computes them, so the fixture cannot disagree with the producer about which
    column holds which quantity.
    """
    return {
        "start": start,
        "end": round(start + 0.1, 6),
        "uncertainty": _entropy(doubt),
        "epistemic_uncertainty": 0.0,  # signals agree, so nothing here is inter-signal disagreement
        "confidence": 1.0 - doubt,
        "variability": 0.0,
        "triage_score": doubt,
        "status": "open",
        "contributing_signals": ["a", "b"],
        "history": [{"round": 0, "uncertainty": _entropy(doubt), "doubt": doubt}],
        **extra,
    }


# A confident bucket: every signal puts doubt at 10%, so H = 0.469 — above theta_low (0.33) and
# below theta_high (0.66). Under the entropy reading it can neither converge nor be left alone.
CONFIDENT_DOUBT = 0.10
# The measured mean on the clean conversation's speaker axis. H(0.2072) = 0.735, over theta_high.
MEASURED_SPEAKER_DOUBT = 0.2072


def test_the_control_quantity_is_doubt_not_entropy() -> None:
    """`control_doubt` returns the complement of confidence, on `theta_*`'s own scale."""
    row = _row(0.0, CONFIDENT_DOUBT)
    assert control_doubt(row) == pytest.approx(CONFIDENT_DOUBT)
    # And it is emphatically not the column the gates used to read.
    assert row["uncertainty"] == pytest.approx(_entropy(CONFIDENT_DOUBT))
    assert control_doubt(row) < row["uncertainty"]


def test_an_unmeasured_bucket_has_no_control_value() -> None:
    """No confidence means nothing was measured — distinct from confident agreement at zero doubt."""
    assert control_doubt({"confidence": None}) is None
    assert control_doubt({}) is None
    assert control_doubt({"confidence": float("nan")}) is None
    assert control_doubt({"confidence": 1.0}) == pytest.approx(0.0)


def test_confident_agreement_converges_instead_of_staying_open() -> None:
    """A bucket every signal calls 90% settled must converge, not sit open forever.

    Under the entropy reading H(0.10) = 0.469 > theta_low, so it stayed `open` on every round of
    every run — which is how 191 of 214 speaker buckets never converged on clean audio.
    """
    policy = load_policy()
    rows = [_row(i * 0.1, CONFIDENT_DOUBT) for i in range(6)]

    class _State:
        def axis_rows(self, axis: str) -> list[dict[str, Any]]:
            return rows if axis == "speaker" else []

    apply_convergence_marks(_State(), policy=policy, touch_counts={}, budget_left=True)
    assert [r["status"] for r in rows] == ["converged"] * 6, (
        f"theta_low={policy['thresholds']['theta_low']} is a doubt threshold; {CONFIDENT_DOUBT} doubt "
        f"is under it, but H({CONFIDENT_DOUBT}) = {_entropy(CONFIDENT_DOUBT):.3f} is not"
    )


def test_the_measured_speaker_doubt_does_not_seed_a_region() -> None:
    """The clean conversation's own mean speaker doubt must not read as high-uncertainty.

    0.2072 doubt is well under theta_high (0.66). Its entropy, 0.735, is over it — which seeded 114
    of 214 buckets on a recording whose speaker count posterior is 0.978 unimodal.
    """
    policy = load_policy()
    rows = [_row(i * 0.1, MEASURED_SPEAKER_DOUBT) for i in range(8)]
    assert _entropy(MEASURED_SPEAKER_DOUBT) > float(policy["thresholds"]["theta_high"]), (
        "fixture no longer reproduces the reported condition"
    )
    assert propose_regions(rows, axis="speaker", policy=policy, round_idx=1, duration_s=2.0) == []


def test_genuine_doubt_still_seeds_a_region() -> None:
    """The gate must still fire where doubt is real, or the fix would just disable the loop."""
    policy = load_policy()
    rows = [_row(i * 0.1, 0.05) for i in range(3)]
    rows += [_row(0.3 + i * 0.1, 0.9) for i in range(3)]
    rows += [_row(0.6 + i * 0.1, 0.05) for i in range(3)]
    regions = propose_regions(rows, axis="speaker", policy=policy, round_idx=1, duration_s=2.0)
    assert regions, "0.9 doubt is above theta_high and must still propose a region"
    assert regions[0]["core_start"] == pytest.approx(0.3)


def test_a_single_voter_axis_stays_investigatable() -> None:
    """`asr` has one voter, so its `epistemic_uncertainty` is structurally 0.

    Gating on epistemic would have made the axis permanently un-investigatable while its doubt was
    real (measured mean 0.215, max 0.918 on the 48 kHz clip). Doubt does not have that blind spot.
    """
    policy = load_policy()
    rows = [_row(i * 0.1, 0.9, contributing_signals=["consensus_words"], epistemic_uncertainty=0.0) for i in range(3)]
    assert all(r["epistemic_uncertainty"] == 0.0 for r in rows)
    assert propose_regions(rows, axis="asr", policy=policy, round_idx=1, duration_s=2.0), (
        "a lone confident-but-doubtful voter is a reason to add a second, not a reason to stop looking"
    )
