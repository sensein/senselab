"""One grid, every axis — the property the four-grid arrangement silently broke.

Measured on a 4.92 s clip before this: ``speech_presence`` 242 rows at 0.1/0.02, ``background_mask``
242 at 0.1/0.02, ``speaker`` 19 at 0.25/0.25, ``asr`` 8 at 1.0/0.5. Four axes, four grids, **zero**
shared bucket keys — so ``fuse.project_axis_onto`` found nothing to project, every cross-axis coupling
ran and did nothing, and each round came out byte-identical to the last. Nothing in the output said so.

These tests state the directive as a check: same row count, same ``(window, hop)``, and the same keys,
for every axis a run produces. They also pin the two things that made the old arrangement possible —
``grid.DEFAULT_TIME_GRID`` being declared and unread, and ``compute_uncertainty_axes`` accepting a
per-axis override — because a re-introduced override would restore the defect with these assertions
still passing on the default.
"""

from __future__ import annotations

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis import BucketGrid, compute_uncertainty_axes
from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES
from senselab.audio.workflows.audio_analysis.grid import DEFAULT_TIME_GRID


@pytest.fixture(autouse=True)
def _offline_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub gated model loaders so the compute call stays offline."""
    import senselab.audio.tasks.scene_quality as sq

    monkeypatch.setattr(sq, "extract_brouhaha_frames", lambda audios, *a, **k: [None] * len(audios))


def _silent_audio(duration_s: float, sr: int = 16000) -> Audio:
    """Trivial Audio object for the compute call."""
    return Audio(waveform=torch.zeros(1, int(duration_s * sr), dtype=torch.float32), sampling_rate=sr)


def _diar_block(segments: list[tuple[float, float, str]]) -> dict:
    """Minimal diar by-model block."""
    from types import SimpleNamespace

    segs = [SimpleNamespace(start=s, end=e, speaker=spk, text="") for s, e, spk in segments]
    return {"status": "ok", "result": [segs], "cache_key": "diar_k"}


def _fuse(grid: BucketGrid) -> dict:
    raw_pass = {
        "duration_s": 2.0,
        "diarization": {"by_model": {"pyannote": _diar_block([(0.0, 2.0, "SPEAKER_00")])}},
    }
    _signals, fused_axes, _, _ = compute_uncertainty_axes(
        passes={"raw": raw_pass},
        grid=grid,
        params={},
        audio={"raw": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
        snr_floor_db=10.0,
        snr_gated_passes=frozenset(),
    )
    return fused_axes


def test_bucketgrid_iter_counts() -> None:
    """A 0.1 s / 0.02 s grid yields far more buckets than 0.5 s over the same span."""
    fine = list(BucketGrid(win_length=0.1, hop_length=0.02).iter_buckets(1.0))
    coarse = list(BucketGrid(win_length=0.5, hop_length=0.5).iter_buckets(1.0))
    assert len(fine) > len(coarse)
    assert len(coarse) == 2  # [0,0.5], [0.5,1.0]


def test_bucketgrid_defaults_to_the_declared_grid() -> None:
    """The default *is* ``DEFAULT_TIME_GRID``, not a second opinion about it.

    It was ``(0.5, 0.5)`` while the constant said ``(0.1, 0.1)``, and nothing read the constant. A
    declared value no consumer uses is not a default; it is documentation that cannot be wrong
    because nothing depends on it.
    """
    grid = BucketGrid()
    assert (grid.win_length, grid.hop_length) == DEFAULT_TIME_GRID


def test_the_default_grid_does_not_overlap() -> None:
    """Window equals hop, so N rows are N independent measurements (D-24)."""
    win, hop = DEFAULT_TIME_GRID
    assert hop == win, "adjacent rows would share audio, and nothing in the output would say so"


def test_every_axis_lands_on_one_grid() -> None:
    """Same row count, same span keys, for every axis a run produces. The directive, as a check."""
    fused_axes = _fuse(BucketGrid())

    assert set(fused_axes) == set(AXIS_NAMES), (
        f"a run must produce every declared axis; got {sorted(fused_axes)} for {sorted(AXIS_NAMES)}"
    )
    counts = {axis: len(result.rows) for axis, result in fused_axes.items()}
    assert len(set(counts.values())) == 1, f"axes are on different grids: {counts}"

    keys = {
        axis: [(round(float(r["start"]), 6), round(float(r["end"]), 6)) for r in result.rows]
        for axis, result in fused_axes.items()
    }
    reference = keys[next(iter(sorted(keys)))]
    for axis, spans in sorted(keys.items()):
        assert spans == reference, f"{axis} does not share bucket keys with the others"


def test_every_axis_records_the_same_window_and_hop() -> None:
    """Recorded per axis and per pass, because the claim being made is per axis.

    Equal values are the point; the *per-axis* record is what lets a reader check it instead of
    inferring it from one entry. ``background_mask`` was absent from this mapping entirely while
    being cut on the presence grid, so nothing recorded what the mask's rows were spaced at.
    """
    fused_axes = _fuse(BucketGrid())
    win, hop = DEFAULT_TIME_GRID
    for axis, result in sorted(fused_axes.items()):
        recorded = result.provenance["grid"]["raw"]
        assert recorded["win_length"] == pytest.approx(win), axis
        assert recorded["hop_length"] == pytest.approx(hop), axis


def test_a_caller_supplied_grid_applies_to_every_axis_at_once() -> None:
    """There is no per-axis override to pass, so a coarser run stays internally consistent."""
    fused_axes = _fuse(BucketGrid(win_length=0.5, hop_length=0.5))
    for axis, result in sorted(fused_axes.items()):
        assert result.provenance["grid"]["raw"]["win_length"] == pytest.approx(0.5), axis
    counts = {axis: len(result.rows) for axis, result in fused_axes.items()}
    assert len(set(counts.values())) == 1, counts


def test_no_per_axis_grid_parameter_survives() -> None:
    """A re-introduced override would restore the defect with every assertion above still passing.

    So the *absence* of the parameter is asserted directly, on the signature. ``asr_grid`` and
    ``speech_presence_grid`` were the two, and the shipped CLI passed both.
    """
    import inspect

    parameters = inspect.signature(compute_uncertainty_axes).parameters
    for forbidden in ("asr_grid", "speech_presence_grid"):
        assert forbidden not in parameters, (
            f"{forbidden} is back; an axis on its own grid shares no bucket keys with the others, "
            "which disables cross-axis coupling silently"
        )
