"""``final/timeline.png`` — every active axis gets a row, and the mask's value is not its axis.

The defect these pin: the figure hand-listed three axes (``speech_presence``, ``speaker``,
``asr``) and drew the background mask's *region state strip* in the place a reader looks for the
fourth axis. So ``final/timeline.png`` showed the mask as one flat, fully-confident band while
``L2/round/<n>/timeline.png`` showed the same axis varying across 1070 buckets — two figures
disagreeing about one axis because they were drawing two different objects under one name.

``axes.py`` exists to make exactly this impossible ("Any list of three axes is wrong"), and this
module had no tests at all, which is how a hand-written list survived beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from senselab.audio.workflows.audio_analysis.adaptive.plot import build_adaptive_timeline
from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES
from senselab.audio.workflows.audio_analysis.layout import belief_dir, estimates_dir, final_dir

_N = 12
_ROUNDS = (0, 1)


def _axis_frame(axis: str, *, uncertainty: float) -> pd.DataFrame:
    """One axis's estimate: the columns every consumer of ``estimates/`` may read."""
    starts = [i * 0.5 for i in range(_N)]
    frame = pd.DataFrame(
        {
            "start": starts,
            "end": [s + 0.5 for s in starts],
            "axis": [axis] * _N,
            "round": [0] * _N,
            # Varying, so a row that draws it is visibly different from a row that draws a flat
            # region value — which is the confusion under test.
            "uncertainty": [uncertainty + 0.03 * i for i in range(_N)],
            "epistemic_uncertainty": [0.5 * (uncertainty + 0.03 * i) for i in range(_N)],
            "confidence": [1.0 - (uncertainty + 0.03 * i) for i in range(_N)],
            "variability": [0.0] * _N,
            "status": [None] * _N,
        }
    )
    if axis == "speech_presence":
        frame["p_voice"] = [0.8] * _N
    return frame


def _run_tree(root: Path) -> None:
    """The smallest tree ``build_adaptive_timeline`` reads: decisions plus every axis per round."""
    final = final_dir(root)
    final.mkdir(parents=True, exist_ok=True)
    (final / "decisions.json").write_text(
        json.dumps({"interventions": [], "convergence": {"run_state": "converged", "policy_hash": "abc123"}})
    )
    for round_index in _ROUNDS:
        dest = estimates_dir(root, round_index)
        dest.mkdir(parents=True, exist_ok=True)
        for axis in AXIS_NAMES:
            _axis_frame(axis, uncertainty=0.2).to_parquet(dest / f"{axis}.parquet")


def _mask_value(root: Path) -> None:
    """The mask *derivative* as a real run writes it: one region, whole recording, no doubt.

    Recorded from ``english_conversation_higgs_audio_v2_20260804-145231``, whose
    ``L2/background_mask.parquet`` is a single ``target_active`` region spanning 0-21 s at
    ``uncertainty`` 0.0 while its ``background_mask`` axis has 1070 buckets averaging 0.0949.
    Both are correct; only one of them is the axis.
    """
    belief_dir(root).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "region_id": ["m0"],
            "start": [0.0],
            "end": [6.0],
            "state": ["target_active"],
            "uncertainty": [0.0],
            "guard_trimmed_s": [0.0],
        }
    ).to_parquet(belief_dir(root) / "background_mask.parquet")


@pytest.fixture()
def drawn_axes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Build the real figure and hand back its axes, so assertions read rendered structure.

    Asserting on the figure rather than on a helper is deliberate: the defect was that the
    figure's row list disagreed with ``AXIS_NAMES``, and a test of anything smaller than the
    figure could not see that.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ``list[Any]`` rather than a narrower type: each entry is a numpy array of axes whose length is
    # what the pick below compares, and mypy cannot see that through ``plt.subplots``' signature.
    captured: list[list[Any]] = []
    real_subplots = plt.subplots

    def spy(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 — a passthrough spy on plt.subplots
        figure, axes = real_subplots(*args, **kwargs)
        captured.append(list(axes) if hasattr(axes, "__len__") else [axes])
        return figure, axes

    monkeypatch.setattr(plt, "subplots", spy)

    _run_tree(tmp_path)
    _mask_value(tmp_path)
    out = build_adaptive_timeline(tmp_path, transcript={"stream": "raw", "words": []})
    assert out is not None and out.exists(), "the figure did not render, so nothing below is testable"

    assert captured, "plt.subplots was never called"
    return max(captured, key=len)


def test_every_active_axis_has_an_uncertainty_row(drawn_axes: list[Any]) -> None:
    """One row per active axis, named for it — the list cannot be short (``axes.AXES``)."""
    labels = [ax.get_ylabel() for ax in drawn_axes]
    missing = [axis for axis in AXIS_NAMES if not any(axis in label for label in labels)]
    assert not missing, f"axes with no row in final/timeline.png: {missing}; rows drawn: {labels}"


def test_the_mask_state_strip_is_not_labelled_as_the_axis(drawn_axes: list[Any]) -> None:
    """The value and the doubt are different objects and must not answer to one name.

    ``derivatives/`` holds values, ``estimates/`` holds doubt about them (D-22). The mask is in
    both, which is legitimate — and precisely why the row drawing the region states may not be the
    row a reader reads as the axis.
    """
    labels = [ax.get_ylabel() for ax in drawn_axes]
    state_rows = [label for label in labels if "state" in label]
    assert len(state_rows) == 1, f"expected exactly one mask state row, got {state_rows} in {labels}"
    assert "background_mask" not in state_rows[0], (
        f"the mask's region-state strip is labelled {state_rows[0]!r}, which reads as the "
        "background_mask axis; the axis is the fused estimate, not the region value"
    )


def test_the_mask_axis_row_draws_the_estimate_not_the_region_value(drawn_axes: list[Any]) -> None:
    """The mask axis row carries the estimate's varying series, not the region's flat 0.0.

    The fixture's region value is one span at ``uncertainty`` 0.0 and its axis rises from 0.2;
    a row drawing the region cannot produce a spread, so a spread is the discriminating evidence.
    """
    row = next((ax for ax in drawn_axes if "background_mask" in ax.get_ylabel()), None)
    assert row is not None, "no background_mask axis row"

    plotted = [line.get_ydata() for line in row.get_lines() if len(line.get_ydata())]
    assert plotted, "the background_mask row drew no series"
    assert any(max(series) - min(series) > 0.05 for series in plotted), (
        "every series on the background_mask row is flat; the row is drawing the region value "
        f"(one span, uncertainty 0.0) rather than the fused axis: {plotted}"
    )


# ── U3: which aligner re-times the consensus, and whether it is recorded ──────


def test_the_consensus_aligner_defaults_to_the_pipeline_s_own_aligner() -> None:
    """One aligner per pipeline (D-1), and the choice is a policy value rather than a literal.

    ``consensus_align`` hard-coded torchaudio MMS_FA, so the pipeline ran Qwen3-ForcedAligner before
    fusion and MMS after — the two-aligner situation D-1 removed when it moved Canary off MMS "so
    word-boundary differences reflect the models, not two different aligners".
    """
    import inspect

    from senselab.audio.workflows.audio_analysis.adaptive.backends import consensus_align
    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    assert inspect.signature(consensus_align).parameters["backend"].default == "qwen"
    assert str((load_policy()["fusion"]).get("consensus_alignment_backend")) == "qwen"


def test_an_empty_word_list_is_refused_rather_than_aligned() -> None:
    """Nothing to place is a reason, not a silent success that publishes an empty span list."""
    from senselab.audio.workflows.audio_analysis.adaptive.backends import consensus_align

    spans, reason = consensus_align(None, [], backend="qwen")
    assert spans is None and reason == "no_words_to_align"


def test_a_span_count_mismatch_is_refused() -> None:
    """Every span after a divergence would attach to the wrong word, so refuse the whole result.

    Publishing a plausible-looking misalignment is worse than keeping the member timings: the
    timestamps would look authoritative while naming the wrong audio.
    """
    from senselab.audio.workflows.audio_analysis.adaptive import backends

    class _Stub:
        chunks = None
        start, end = 0.0, 0.4

    monkey = getattr(backends, "_word_leaves")
    assert monkey(_Stub()) == [(0.0, 0.4)], "the leaf reader must find a timed node"
    nested = type("L", (), {"chunks": [_Stub(), _Stub()]})()
    assert len(monkey(nested)) == 2, "and must descend to the words"
