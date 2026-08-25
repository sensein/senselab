"""Redaction: pad outward, merge overlaps, and write the declared fill into the audio."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.redaction import RedactionExtent, apply_redactions, plan_redactions

SR = 16000


def _tone(duration_s: float, amplitude: float = 1.0, hz: float = 220.0) -> Audio:
    """A mono tone of constant amplitude, so an extent's own peak is known before it is masked."""
    t = np.arange(int(duration_s * SR), dtype="float32") / SR
    return Audio(waveform=(amplitude * np.sin(2 * np.pi * hz * t)).astype("float32")[None, :], sampling_rate=SR)


class TestPlanning:
    """Padding and merging happen before any audio is touched."""

    def test_padding_is_required(self) -> None:
        """padding_ms has no default; the margin is unmeasured and must be supplied."""
        with pytest.raises(TypeError):
            plan_redactions([RedactionExtent(1.0, 1.2, "PERSON")])  # type: ignore[call-arg]

    def test_extents_are_padded_outward_on_both_sides(self) -> None:
        """100 ms of padding widens (1.0, 1.2) to (0.9, 1.3)."""
        (out,) = plan_redactions([RedactionExtent(1.0, 1.2, "PERSON")], padding_ms=100)
        assert out.start == pytest.approx(0.9)
        assert out.end == pytest.approx(1.3)

    def test_padding_never_produces_a_negative_start(self) -> None:
        """Padding clamps at the start of the recording."""
        (out,) = plan_redactions([RedactionExtent(0.02, 0.1, "PERSON")], padding_ms=100)
        assert out.start == 0.0

    def test_extents_that_overlap_after_padding_are_merged(self) -> None:
        """Two paddings that touch become one redaction carrying both categories."""
        out = plan_redactions(
            [RedactionExtent(1.0, 1.1, "PERSON"), RedactionExtent(1.25, 1.35, "DATE")], padding_ms=100
        )
        assert len(out) == 1, "an audible sliver between two redactions is a leak"
        assert out[0].category == "PERSON+DATE"

    def test_an_inverted_extent_raises(self) -> None:
        """An extent with end < start raises instead of passing through to a silent no-op."""
        with pytest.raises(ValueError, match="PERSON"):
            plan_redactions([RedactionExtent(1.5, 1.0, "PERSON")], padding_ms=100)

    def test_a_negative_end_raises(self) -> None:
        """An extent whose end is negative raises instead of selecting a wrong region."""
        with pytest.raises(ValueError, match="DATE"):
            plan_redactions([RedactionExtent(1.0, -0.5, "DATE")], padding_ms=100)

    def test_a_non_finite_bound_raises(self) -> None:
        """An extent with a NaN or infinite bound raises."""
        with pytest.raises(ValueError, match="PERSON"):
            plan_redactions([RedactionExtent(float("nan"), 1.0, "PERSON")], padding_ms=100)
        with pytest.raises(ValueError, match="DATE"):
            plan_redactions([RedactionExtent(1.0, float("inf"), "DATE")], padding_ms=100)

    def test_replanning_merged_output_does_not_duplicate_categories(self) -> None:
        """A compound category arriving at a merge contributes each label once, in first-seen order."""
        first = plan_redactions(
            [RedactionExtent(1.0, 1.1, "PERSON"), RedactionExtent(1.15, 1.25, "DATE")], padding_ms=50
        )
        second = plan_redactions(
            [RedactionExtent(1.4, 1.5, "DATE"), RedactionExtent(1.55, 1.65, "PERSON")], padding_ms=50
        )
        (out,) = plan_redactions([*first, *second], padding_ms=100)
        assert out.category == "PERSON+DATE"


class TestApplying:
    """Applying masks exactly the planned extents and nothing else."""

    def test_the_redacted_region_is_silent_and_the_rest_is_untouched(self) -> None:
        """Samples inside the extent go to zero; samples outside keep their values."""
        x = np.ones((1, 3 * SR), dtype="float32")
        audio = Audio(waveform=x, sampling_rate=SR)
        out = apply_redactions(audio, [RedactionExtent(1.0, 1.5, "PERSON")], fill="silence")
        w = np.asarray(out.waveform).squeeze()
        assert np.all(w[int(1.0 * SR) : int(1.5 * SR)] == 0.0)
        assert np.all(w[: int(1.0 * SR)] == 1.0)
        assert np.all(w[int(1.5 * SR) :] == 1.0)

    def test_duration_is_preserved(self) -> None:
        """Silencing replaces samples; it never cuts them out."""
        audio = Audio(waveform=np.ones((1, 3 * SR), dtype="float32"), sampling_rate=SR)
        out = apply_redactions(audio, [RedactionExtent(1.0, 1.5, "PERSON")], fill="silence")
        assert np.asarray(out.waveform).shape[-1] == 3 * SR

    def test_the_end_boundary_rounds_up(self) -> None:
        """A fractional end silences the sample it falls inside; truncation would leave it audible."""
        audio = Audio(waveform=np.ones((1, 3 * SR), dtype="float32"), sampling_rate=SR)
        out = apply_redactions(audio, [RedactionExtent(1.0, 1.50003, "PERSON")], fill="silence")
        w = np.asarray(out.waveform).squeeze()
        assert w[24000] == 0.0
        assert w[24001] == 1.0

    def test_an_extent_that_cannot_land_leaves_the_audio_unchanged(self) -> None:
        """A degenerate extent selects no samples; it never zeroes a wrong region."""
        audio = Audio(waveform=np.ones((1, 3 * SR), dtype="float32"), sampling_rate=SR)
        out = apply_redactions(
            audio, [RedactionExtent(1.5, 1.0, "PERSON"), RedactionExtent(1.0, -0.5, "DATE")], fill="silence"
        )
        assert np.all(np.asarray(out.waveform) == 1.0)


class TestTheFill:
    """What is written into a redacted extent, and what is refused."""

    def test_silence_writes_zeros(self) -> None:
        """The historical behaviour, now named rather than implied."""
        audio = _tone(1.0)
        out = apply_redactions(audio, [RedactionExtent(0.2, 0.4, "PERSON")], fill="silence")
        assert float(out.waveform[:, 3200:6400].abs().max()) == 0.0

    def test_bleep_writes_a_tone_at_the_extents_own_level(self) -> None:
        """The extent is masked, not removed, and the level is the extent's own."""
        audio = _tone(1.0, amplitude=0.5)
        out = apply_redactions(audio, [RedactionExtent(0.2, 0.4, "PERSON")], fill="bleep", bleep_hz=1000.0)
        inside = out.waveform[:, 3200:6400]
        assert float(inside.abs().max()) == pytest.approx(0.5, rel=0.05)
        assert float(inside.abs().min()) < 0.05

    def test_noise_is_refused_with_the_measurement_it_is_owed(self) -> None:
        """Shipping an unmeasured spectral shape would be a value nobody fitted (V22)."""
        with pytest.raises(NotImplementedError, match="least damaging"):
            apply_redactions(_tone(1.0), [RedactionExtent(0.2, 0.4, "PERSON")], fill="noise")

    def test_an_unknown_fill_is_refused(self) -> None:
        """A typo must not silently fall back to silence."""
        with pytest.raises(ValueError, match="fill"):
            apply_redactions(_tone(1.0), [RedactionExtent(0.2, 0.4, "PERSON")], fill="beep")

    def test_bleep_without_a_frequency_is_refused(self) -> None:
        """The tone's frequency is a config value, not something this function may invent."""
        with pytest.raises(ValueError, match="bleep_hz"):
            apply_redactions(_tone(1.0), [RedactionExtent(0.2, 0.4, "PERSON")], fill="bleep")

    def test_the_fill_has_no_default(self) -> None:
        """A caller that does not say which fill it used gets no answer rather than silence."""
        with pytest.raises(TypeError):
            apply_redactions(_tone(1.0), [RedactionExtent(0.2, 0.4, "PERSON")])  # type: ignore[call-arg]

    def test_the_duration_is_preserved_under_every_implemented_fill(self) -> None:
        """A redaction masks; it does not shorten."""
        audio = _tone(1.0)
        for fill in ("silence", "bleep"):
            out = apply_redactions(audio, [RedactionExtent(0.2, 0.4, "PERSON")], fill=fill, bleep_hz=1000.0)
            assert out.waveform.shape == audio.waveform.shape

    def test_only_the_extent_is_masked_under_a_bleep(self) -> None:
        """The bleep replaces the extent's samples and leaves every other sample alone."""
        audio = _tone(1.0, amplitude=0.5)
        before = np.asarray(audio.waveform).copy()
        out = apply_redactions(audio, [RedactionExtent(0.2, 0.4, "PERSON")], fill="bleep", bleep_hz=1000.0)
        after = np.asarray(out.waveform)
        assert np.array_equal(after[:, :3200], before[:, :3200])
        assert np.array_equal(after[:, 6400:], before[:, 6400:])
        assert not np.array_equal(after[:, 3200:6400], before[:, 3200:6400])
