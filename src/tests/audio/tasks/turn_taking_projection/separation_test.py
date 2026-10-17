"""Mono-to-dyad splitting: masks, source-to-speaker assignment, windowed stitching, input handling."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.turn_taking_projection import separate_dyad
from senselab.audio.tasks.turn_taking_projection import separation as sep
from senselab.audio.tasks.turn_taking_projection.api import _split_input
from senselab.audio.tasks.turn_taking_projection.maai import plan_jobs
from senselab.utils.data_structures import ScriptLine

SR = 16000


def line(speaker: str, start: float, end: float) -> ScriptLine:
    """A labelled segment."""
    return ScriptLine(speaker=speaker, start=start, end=end)


def tone(freq: float, seconds: float) -> np.ndarray:
    """A unit sine."""
    return np.sin(2 * np.pi * freq * np.arange(int(seconds * SR)) / SR).astype(np.float32)


def test_smooth_mask_is_one_inside_and_ramps_outside() -> None:
    """Gain 1 on every active sample, rising before onset and falling after offset, 0 elsewhere."""
    active = np.zeros(1000, dtype=bool)
    active[400:600] = True
    gain = sep.smooth_mask(active, 50)
    assert np.all(gain[400:600] == 1.0)
    assert np.all(gain[:350] == 0.0) and np.all(gain[650:] == 0.0)
    assert np.all(np.diff(gain[350:400]) > 0) and np.all(np.diff(gain[600:650]) < 0)
    assert 0 < gain[350] < 0.01 and 0.99 < gain[399] < 1.0


def test_smooth_mask_clips_ramps_at_the_edges() -> None:
    """A segment touching either end of the signal does not index outside it."""
    active = np.zeros(100, dtype=bool)
    active[:10] = True
    active[95:] = True
    gain = sep.smooth_mask(active, 20)
    assert gain.shape == (100,) and gain[0] == 1.0 and gain[-1] == 1.0


def test_masking_duplicates_overlap_and_silences_neither_speaker_inside_a_segment() -> None:
    """Overlapped speech goes to both channels unattenuated; the gap between turns is zeroed."""
    mix = np.ones(4 * SR, dtype=np.float32)
    diar = [line("A", 0.0, 2.0), line("B", 1.5, 3.0)]
    a, b = sep.mask_dyad(mix, diar, ("A", "B"), ramp_seconds=0.05)
    assert np.all(a[int(1.5 * SR) : 2 * SR] == 1.0) and np.all(b[int(1.5 * SR) : 2 * SR] == 1.0)
    assert np.all(a[int(2.1 * SR) :] == 0.0) and np.all(b[: int(1.4 * SR)] == 0.0)
    assert np.all(a[int(3.1 * SR) :] == 0.0) and np.all(b[int(3.1 * SR) :] == 0.0)


def test_choose_speakers_takes_the_two_most_talkative_by_first_onset() -> None:
    """A third, brief speaker is dropped; order follows who spoke first."""
    diar = [line("X", 0.0, 0.5), line("B", 1.0, 5.0), line("A", 5.0, 9.0), line("X", 9.0, 9.2)]
    assert sep.choose_speakers(diar) == ("B", "A")
    assert sep.choose_speakers(diar, ["A", "X"]) == ("A", "X")
    with pytest.raises(ValueError):
        sep.choose_speakers(diar, ["A", "Z"])
    with pytest.raises(ValueError):
        sep.choose_speakers([line("A", 0, 1)])


def test_assignment_follows_energy_not_source_order() -> None:
    """Source 1 carries A's turns and source 0 carries B's: the assignment is (1, 0)."""
    n = 6 * SR
    s0, s1 = np.zeros(n, np.float32), np.zeros(n, np.float32)
    s1[: 2 * SR] = tone(200, 2)
    s0[3 * SR : 5 * SR] = tone(300, 2)
    s0 += 0.01 * tone(50, 6)
    (ia, ib), shares = sep.assign_sources(np.stack([s0, s1]), [line("A", 0, 2), line("B", 3, 5)], ("A", "B"))
    assert (ia, ib) == (1, 0)
    assert shares[1, 0] > 0.99 and shares[0, 1] > 0.99


def test_assignment_with_one_speaker_present_still_gives_two_distinct_sources() -> None:
    """Only A speaks in the window; A gets the source with A's energy, B the other one."""
    n = 4 * SR
    sources = np.stack([0.01 * tone(80, 4), np.concatenate([tone(200, 2), np.zeros(2 * SR, np.float32)])])
    (ia, ib), _ = sep.assign_sources(sources[:, :n], [line("A", 0, 2)], ("A", "B"))
    assert (ia, ib) == (1, 0)


def test_assignment_uses_recording_time_offsets() -> None:
    """A window starting at 10 s reads the diarization shifted by 10 s."""
    sources = np.stack(
        [
            np.concatenate([np.zeros(SR, np.float32), tone(200, 1)]),
            np.concatenate([tone(300, 1), np.zeros(SR, np.float32)]),
        ]
    )
    diar = [line("A", 10.0, 11.0), line("B", 11.0, 12.0)]
    (ia, ib), _ = sep.assign_sources(sources, diar, ("A", "B"), offset_seconds=10.0)
    assert (ia, ib) == (1, 0)


def test_windowed_separation_undoes_a_per_window_permutation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fake separator that swaps its output order on every other window is stitched back in order."""
    params = {**sep.parameters(), "separation_chunk_seconds": 4.0, "separation_overlap_seconds": 1.0}
    monkeypatch.setattr(sep, "parameters", lambda: params)
    n = 20 * SR
    truth_a, truth_b = np.zeros(n, np.float32), np.zeros(n, np.float32)
    diar: List[ScriptLine] = []
    for k in range(10):
        lo, hi = 2 * k, 2 * k + 1.5
        target, freq, spk = (truth_a, 200, "A") if k % 2 == 0 else (truth_b, 330, "B")
        target[int(lo * SR) : int(hi * SR)] = tone(freq, hi - lo)
        diar.append(line(spk, lo, hi))
    mix = truth_a + truth_b
    calls = {"n": 0}
    windows = sep._windows(n, SR)

    def fake(chunks: List[np.ndarray]) -> List[np.ndarray]:
        out = []
        for i, ((lo, hi), _) in enumerate(zip(windows, chunks)):
            pair = np.stack([truth_a[lo:hi], truth_b[lo:hi]])
            out.append(pair[::-1] * 3.0 if i % 2 else pair * 0.5)
        calls["n"] += 1
        return out

    (a, b), report = sep.separate_windows(mix, diar, ("A", "B"), fake)
    assert calls["n"] == 1 and len(report) == len(windows) > 3
    assert [r["sources"] for r in report[:2]] == [[0, 1], [1, 0]]
    assert np.corrcoef(a, truth_a)[0, 1] > 0.99 and np.corrcoef(b, truth_b)[0, 1] > 0.99
    assert np.allclose(a + b, mix, atol=0.05)


def test_stitch_cross_fades_to_unit_weight() -> None:
    """Two overlapping constant pieces stitch to a constant."""
    pieces = [(0, np.ones((2, 100))), (60, np.ones((2, 100)))]
    out = sep.stitch_windows(pieces, 160)
    assert np.allclose(out, 1.0)


def test_separate_dyad_masks_a_mono_audio_and_labels_its_channels() -> None:
    """The public call resamples, masks, and tags each channel with its speaker and method."""
    mix = Audio(waveform=torch.ones(1, 3 * 8000), sampling_rate=8000)
    a, b = separate_dyad(mix, [line("S1", 0.0, 1.0), line("S2", 1.5, 3.0)])
    assert a.sampling_rate == SR and a.waveform.shape == (1, 3 * SR)
    assert a.metadata["speaker"] == "S1" and b.metadata["speaker"] == "S2"
    assert a.metadata["turn_taking_projection"]["method"] == "masking"
    assert float(a.waveform[0, int(2.0 * SR)]) == 0.0 and float(b.waveform[0, int(2.0 * SR)]) == pytest.approx(
        1.0, abs=1e-3
    )


def test_separate_dyad_rejects_an_unknown_method() -> None:
    """Only the three documented methods exist."""
    mix = Audio(waveform=torch.ones(1, SR), sampling_rate=SR)
    with pytest.raises(ValueError, match="unknown separation method"):
        separate_dyad(mix, [line("A", 0, 0.4), line("B", 0.5, 1.0)], method="magic")


def test_split_input_accepts_stereo_or_a_labelled_pair() -> None:
    """Stereo splits into channels labelled a/b; a pair from separate_dyad keeps its labels."""
    stereo = Audio(waveform=torch.zeros(2, SR), sampling_rate=SR)
    a, b, labels, source = _split_input(stereo, None)
    assert labels == ("a", "b") and source == {"kind": "stereo"} and a.waveform.shape == (1, SR)
    pair = separate_dyad(Audio(waveform=torch.ones(1, SR), sampling_rate=SR), [line("X", 0, 0.4), line("Y", 0.5, 1)])
    _, _, labels, source = _split_input(pair, None)
    assert labels == ("X", "Y") and source["method"] == "masking"
    with pytest.raises(ValueError, match="stereo"):
        _split_input(Audio(waveform=torch.zeros(1, SR), sampling_rate=SR), None)


def test_jobs_run_backchannel_once_per_listener() -> None:
    """bc_b feeds (a, b) and bc_a feeds (b, a): MaAI's bc predicts channel 2's backchannels."""
    jobs = plan_jobs("vap_en_kyoto_mc", "vap_bc_en")
    assert [(j.name, j.swap) for j in jobs] == [("vap", False), ("bc_b", False), ("bc_a", True)]
    assert [j.name for j in plan_jobs("vap_en_kyoto", None)] == ["vap"]
    with pytest.raises(ValueError):
        plan_jobs("vap_en", None)
