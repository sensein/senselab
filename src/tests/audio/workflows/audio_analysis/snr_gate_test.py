"""A repair perturbation is evidence only where there is something to repair.

The regression, stated as the first test: on a clean two-speaker conversation (41-70 dB SNR
throughout) the raw pass placed the speaker axis at exactly 0.0 in 179 of 190 buckets and the
enhanced pass at 0.398 with only 51% zeros. ``fuse_axis`` averaged the two and published 0.227 — so
in all 178 buckets where every diarizer agreed, the axis reported doubt sourced entirely from a
transform that had nothing to remove.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.fuse import SnrGate, fuse_axis
from senselab.audio.workflows.audio_analysis.perturbations import (
    identity,
    speech_enhancement,
)

BUCKET = (0.0, 0.1)
OTHER = (0.1, 0.2)


def _bucket(start: float, end: float, value: float) -> dict[str, Any]:
    return {"start": start, "end": end, "votes": {"speaker_assignment": {"value": value}}}


def _gate(snr: dict[tuple[float, float], float | None], *, floor_db: float = 10.0) -> SnrGate:
    return SnrGate(floor_db=floor_db, snr_db_by_bucket=snr, gated_passes=frozenset({"enhanced"}))


# ── which perturbations the gate applies to ─────────────────────────────────


def test_enhancement_is_gated_and_the_identity_is_not() -> None:
    """Read off the declared transform, never the name: a pass called anything is still a repair."""
    assert speech_enhancement("speechbrain/sepformer-wham16k-enhancement").admission_requires_low_snr
    assert speech_enhancement("m", name="sepformer_v2").admission_requires_low_snr
    assert not identity().admission_requires_low_snr


def test_an_ungated_pass_is_admitted_at_every_snr() -> None:
    """Invariance probes must not be gated — see ``invariance``.

    Gain scaling, a whole-sample shift and a small DC offset are chosen so a *correct* model cannot
    change its answer, which is what makes their disagreement a model defect at any SNR. Gating them
    by degradation would remove the only condition under which they mean anything.
    """
    gate = _gate({BUCKET: 70.0})
    assert gate.admits("raw", BUCKET)
    assert gate.admits("gain_scaled", BUCKET), "an invariance probe is not a repair"


# ── the admission rule ──────────────────────────────────────────────────────


def test_a_clean_bucket_does_not_admit_the_repair() -> None:
    """70 dB: nothing to remove, so a changed answer reports the transform, not the recording."""
    assert not _gate({BUCKET: 70.0}).admits("enhanced", BUCKET)


def test_a_degraded_bucket_admits_the_repair() -> None:
    """Below the floor is exactly what enhancement exists for."""
    assert _gate({BUCKET: 4.0}).admits("enhanced", BUCKET)


def test_the_floor_is_exclusive() -> None:
    """At the floor is not below it; one comparison, stated once."""
    assert not _gate({BUCKET: 10.0}).admits("enhanced", BUCKET)
    assert _gate({BUCKET: 9.999}).admits("enhanced", BUCKET)


def test_an_unmeasured_snr_does_not_admit() -> None:
    """Unmeasured is not measured-low.

    SNR is the *primary* condition, so "we could not tell how degraded this is" cannot stand in for
    "it is degraded". The bucket records the exclusion so the silence is visible.
    """
    assert not _gate({BUCKET: None}).admits("enhanced", BUCKET)
    assert not _gate({}).admits("enhanced", BUCKET), "a bucket the gate has no reading for"


def test_ambiguity_does_not_admit_the_repair() -> None:
    """Deliberately *not* an OR with raw disagreement, though it measures better on one clip.

    Admitting wherever the raw sources disagreed read 0.0202 against 0.0317 on the conversation
    clip, because enhancement resolves five of its seven contested buckets. It is still the wrong
    rule: at genuinely low SNR the raw sources can be unanimously *wrong*, all fooled by the same
    noise, and an ambiguity requirement locks enhancement out of precisely that case. The gate takes
    no argument about the raw values at all, which is what makes this true by construction.
    """
    gate = _gate({BUCKET: 70.0})
    assert not gate.admits("enhanced", BUCKET), "a contested high-SNR bucket is still high-SNR"


# ── what the fold does with it ──────────────────────────────────────────────


def test_the_repair_cannot_add_doubt_to_a_clean_unanimous_bucket() -> None:
    """The regression. Raw says 0.0, enhanced says 0.8, SNR is 70 dB: the axis reads 0.0."""
    rows = fuse_axis(
        {"raw": [_bucket(*BUCKET, 0.0)], "enhanced": [_bucket(*BUCKET, 0.8)]},
        weights={"speaker_assignment": 1.0},
        snr_gate=_gate({BUCKET: 70.0}),
    )
    assert len(rows) == 1
    assert rows[0]["confidence"] == pytest.approx(1.0), "the transform's dissent must not land here"
    assert rows[0]["contributing_passes"] == ["raw"]
    assert rows[0]["snr_gated_passes"] == ["enhanced"]


def test_the_repair_is_folded_in_where_the_audio_is_degraded() -> None:
    """Same inputs, 4 dB: now both readings count and the mean is published."""
    rows = fuse_axis(
        {"raw": [_bucket(*BUCKET, 0.0)], "enhanced": [_bucket(*BUCKET, 0.8)]},
        weights={"speaker_assignment": 1.0},
        snr_gate=_gate({BUCKET: 4.0}),
    )
    assert rows[0]["confidence"] == pytest.approx(0.6), "mean of 0.0 and 0.8, as certainty"
    assert rows[0]["contributing_passes"] == ["enhanced", "raw"]
    assert rows[0]["snr_gated_passes"] == []


def test_the_gate_decides_per_bucket_not_per_run() -> None:
    """One recording can be clean in one bucket and degraded in the next."""
    rows = fuse_axis(
        {
            "raw": [_bucket(*BUCKET, 0.0), _bucket(*OTHER, 0.0)],
            "enhanced": [_bucket(*BUCKET, 0.8), _bucket(*OTHER, 0.8)],
        },
        weights={"speaker_assignment": 1.0},
        snr_gate=_gate({BUCKET: 70.0, OTHER: 4.0}),
    )
    by_start = {r["start"]: r for r in rows}
    assert by_start[BUCKET[0]]["contributing_passes"] == ["raw"]
    assert by_start[OTHER[0]]["contributing_passes"] == ["enhanced", "raw"]


def test_a_bucket_only_the_repair_reached_still_owes_a_row() -> None:
    """Gated out of everything is a measurement — "nothing was admitted" — not an absent bucket.

    Skipping the row would delete buckets from an axis depending on SNR, so two axes on one grid
    would stop being row-aligned and a cross-axis join would need reconciliation again (D-24).
    """
    rows = fuse_axis(
        {"enhanced": [_bucket(*BUCKET, 0.8)]},
        weights={"speaker_assignment": 1.0},
        snr_gate=_gate({BUCKET: 70.0}),
    )
    assert len(rows) == 1, "the bucket must survive as a row"
    assert rows[0]["confidence"] is None, "no signal spoke — None, never 0.0"
    assert rows[0]["contributing_passes"] == []
    assert rows[0]["snr_gated_passes"] == ["enhanced"]


def test_no_gate_folds_every_pass() -> None:
    """``None`` means a run with nothing to gate, and must not change the old arithmetic."""
    rows = fuse_axis(
        {"raw": [_bucket(*BUCKET, 0.0)], "enhanced": [_bucket(*BUCKET, 0.8)]},
        weights={"speaker_assignment": 1.0},
        snr_gate=None,
    )
    assert rows[0]["confidence"] == pytest.approx(0.6)
    assert rows[0]["snr_gated_passes"] == []


# ── the shared constructor ──────────────────────────────────────────────────


class _Harvest:
    """Just the one attribute ``SnrGate.build`` reads."""

    def __init__(self, quality: dict[tuple[float, float], dict[str, Any]]) -> None:
        self.quality_by_bucket = quality


def test_build_reads_the_identity_pass_snr() -> None:
    """How degraded a recording is is a fact about the recording.

    Reading SNR off the enhanced audio would ask the repair to certify its own necessity — and it
    would pass every time, because raising SNR is exactly what it did.
    """
    gate = SnrGate.build(
        {
            "raw": _Harvest({BUCKET: {"snr_brouhaha_db": 4.0}}),
            "enhanced": _Harvest({BUCKET: {"snr_brouhaha_db": 40.0}}),
        },
        floor_db=10.0,
        gated_passes=frozenset({"enhanced"}),
    )
    assert gate is not None
    assert gate.admits("enhanced", BUCKET), "the raw 4 dB governs, not the enhanced 40 dB"


def test_build_returns_none_when_there_is_nothing_to_gate() -> None:
    """A run whose only perturbation is the identity needs no gate object at all."""
    assert SnrGate.build({"raw": _Harvest({})}, floor_db=10.0, gated_passes=frozenset()) is None


# ── the record has to survive the writers ───────────────────────────────────


def test_every_fold_column_the_schema_declares_is_carried_by_the_writers() -> None:
    """The three estimate writers rebuild rows from an explicit whitelist, and omission is silent.

    ``estimate_frame`` fills any declared column a row omits with null — deliberately, so "the
    producer had nothing to say" is recordable. The cost is that a *new* fold column is dropped
    without complaint: the SNR gate shipped with all 214 rows reading ``snr_gated_passes = None``
    while the gating itself had worked correctly, and no unit test saw it because they all assert on
    ``fuse_axis``'s dict rather than on what reaches the parquet.

    Checked by source inspection because the failure is a *missing* line — there is nothing to call.
    """
    import inspect

    from senselab.audio.workflows.audio_analysis import estimates, fuse
    from senselab.audio.workflows.audio_analysis.adaptive import belief, loop

    rows = fuse_axis(
        {"raw": [_bucket(*BUCKET, 0.3)]},
        weights={"speaker_assignment": 1.0},
        snr_gate=None,
    )
    shared = set(rows[0]) & set(estimates.ESTIMATE_COLUMNS)
    # ``round`` and ``axis`` are stamped by the declaration from the path, never carried.
    shared -= {"round", "axis"}
    assert shared, "the fold and the schema share no columns, so this test is checking nothing"

    for module in (fuse, loop, belief):
        source = inspect.getsource(module)
        missing = sorted(column for column in shared if f'"{column}"' not in source)
        assert not missing, (
            f"{module.__name__} writes estimate rows but never names {missing}; "
            "estimate_frame will write them null rather than refuse"
        )
