"""VOICE under propose-only: it mints its own spans from the evidence it routes on.

Nothing here loads a model or calls Praat. The store surface is built directly — amplitude spans,
``phonation_tracks`` and ``continuity_trace`` — because that surface *is* the branch's subject, and
building it by hand is what lets a test say which of the three qualifiers a recording failed.

What is pinned: that a held vowel yields a ``task_extent`` span of family ``voice``; that the mode
is selected by the declared family and both arms are reachable; that an absent instrument is
``UNDETERMINED`` rather than a verdict about the speaker; that every proposal names its evidence;
and that a genuinely aperiodic recording no longer errors the node.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import voice as voice_module
from senselab.audio.workflows.triage.nodes.branches import (
    PARAM_SECTION,
    UNDETERMINED,
    Expectation,
    Pattern,
    branch_params,
    deviation_names,
    mode_of,
    semitones,
    track_slice,
    windowed_spreads,
)
from senselab.audio.workflows.triage.nodes.common import find_branch_report, find_measurement, live_entities
from senselab.audio.workflows.triage.nodes.gates import GROUP_LAYER
from senselab.audio.workflows.triage.nodes.voice import (
    COUNT_IN,
    PHONATION_ROLE,
    TASK_EXTENT,
    align_voice,
    detect_voice,
    qualifying_phonation,
    read_evidence,
    voice,
)
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import gated_conformance, gated_flags, readings_of

HOP_S = 0.01
"""PREPROCESS's own phonation-track hop, which the fixtures build their frame grid on."""

CONTINUITY_RATE = 100.0
"""The continuity trace's sampling rate in the fixtures, in Hz."""

RATE = 16000
"""The ``plain`` stream's sampling rate in the fixtures, in Hz."""

MPT_STEM = "sub-abc_ses-1_task-maximum-phonation-time"
PROLONGED_STEM = "sub-abc_ses-1_task-prolonged-vowel"
GLIDE_UP_STEM = "sub-abc_ses-1_task-glides-low-to-high"
GLIDE_DOWN_STEM = "sub-abc_ses-1_task-glides-high-to-low"
COUGH_STEM = "sub-abc_ses-1_task-voluntary-cough"
LOUDNESS_STEM = "sub-abc_ses-1_task-loudness"
CAPEV_STEM = "sub-abc_ses-1_task-cape-v-sentences"
HARVARD_STEM = "sub-abc_ses-1_task-harvard-sentences-list"

SETTINGS = {
    "voiced_strength_min": 0.5,
    "f0_spread_window_s": 1.0,
    "sweep_smoothing_frames": 1,
    "sweep_reversal_tolerance_semitones": 1.0,
}
"""Instrument settings supplied by the fixture, overriding the packaged ones for the test signals."""

GATES = {
    "production_min_s": 0.5,
    "voiced_fraction_min": 0.6,
    "f0_spread_max_semitones": 4.0,
    "continuity_min": 0.8,
    "glide_extent_min_semitones": 1.0,
}
"""Gate bounds supplied by the fixture. Each group takes the ones its own body reads."""

GROUP_GATES = {
    Pattern.SUSTAINED: ("production_min_s", "voiced_fraction_min", "f0_spread_max_semitones", "continuity_min"),
    Pattern.GLIDE: (
        "production_min_s",
        "voiced_fraction_min",
        "glide_extent_min_semitones",
    ),
}
"""Which of :data:`GATES` each group names, mirroring the packaged table's shape."""

MEASURED = {**SETTINGS, **GATES}
"""Both halves together, for the assertions that read a fixture value back."""


def params(group: Pattern = Pattern.SUSTAINED, **overrides: Any) -> Any:  # noqa: ANN401
    """The operating points, over a configuration carrying fixture values, bound to one group.

    Args:
        group: The task group whose gates the body will read. Rebound by ``align_voice``.
        **overrides: ``branch.*`` or gate keys to set or clear.

    Returns:
        The record.
    """
    return branch_params(config(**overrides)).bind(group)


def config(**overrides: Any) -> TriageConfig:  # noqa: ANN401
    """A configuration carrying the fixture's instrument settings and its per-group gates.

    Args:
        **overrides: ``branch.*`` or gate keys to set or clear; each lands in its own section.

    Returns:
        The configuration.
    """
    packaged = load_triage_config()
    merged = dict(packaged.values)
    settings = {**SETTINGS, **{k: v for k, v in overrides.items() if k not in GATES}}
    gates = {**GATES, **{k: v for k, v in overrides.items() if k in GATES}}
    merged[PARAM_SECTION] = {**packaged.values[PARAM_SECTION], **settings}
    merged["verdict"] = {
        **packaged.values["verdict"],
        "gates": {
            **packaged.values["verdict"]["gates"],
            GROUP_LAYER: {
                **packaged.values["verdict"]["gates"][GROUP_LAYER],
                **{group.name: {name: gates[name] for name in names} for group, names in GROUP_GATES.items()},
            },
        },
    }
    return TriageConfig(packaged.name, packaged.version, packaged.config_hash, merged)


def _tracks(
    duration_s: float,
    voiced: list[tuple[float, float]],
    *,
    f0_hz: float | list[tuple[float, float, float]] = 120.0,
    strength: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A phonation-track grid: zero strength everywhere but the voiced intervals.

    Args:
        duration_s: How long the grid runs.
        voiced: The intervals the tracker found F0 inside.
        f0_hz: A constant F0, or ``(start, end, hz)`` triples placing a ramp's endpoints.
        strength: The pitch strength inside the voiced intervals.

    Returns:
        ``(times_s, f0_hz, strength)``.
    """
    times = np.arange(0.0, duration_s, HOP_S)
    f0 = np.zeros(times.size)
    power = np.zeros(times.size)
    for start, end in voiced:
        inside = (times >= start) & (times < end)
        power[inside] = strength
        f0[inside] = f0_hz if isinstance(f0_hz, float) else 0.0
    if not isinstance(f0_hz, float):
        for start, end, hz in f0_hz:
            inside = (times >= start) & (times < end)
            f0[inside] = np.linspace(hz, hz, int(inside.sum())) if inside.sum() else 0.0
    return times, f0, power


def _glide_tracks(
    duration_s: float, extent: tuple[float, float], low_hz: float, high_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A monotone F0 sweep from ``low_hz`` to ``high_hz`` across ``extent``.

    Args:
        duration_s: How long the grid runs.
        extent: The interval the sweep occupies.
        low_hz: F0 at the sweep's start.
        high_hz: F0 at its end.

    Returns:
        ``(times_s, f0_hz, strength)``.
    """
    times = np.arange(0.0, duration_s, HOP_S)
    f0 = np.zeros(times.size)
    power = np.zeros(times.size)
    inside = (times >= extent[0]) & (times < extent[1])
    power[inside] = 0.9
    f0[inside] = np.linspace(low_hz, high_hz, int(inside.sum()))
    return times, f0, power


def _octave_error_tracks(
    duration_s: float,
    extent: tuple[float, float],
    centre_hz: float,
    bursts: int,
    burst_frames: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A steady F0 with a few frames the tracker doubled -- the pitch-tracker octave error.

    The production is steady throughout; what is not steady is the tracker. Spread over the whole
    carrier this is a fraction of a semitone, and in the handful of windows a burst falls in it is
    twelve.

    Args:
        duration_s: How long the grid runs.
        extent: The interval the phonation occupies.
        centre_hz: The F0 actually held.
        bursts: How many separate doubling errors to place, spread evenly through the extent.
        burst_frames: How many consecutive frames each error lasts.

    Returns:
        ``(times_s, f0_hz, strength)``.
    """
    times = np.arange(0.0, duration_s, HOP_S)
    f0 = np.zeros(times.size)
    power = np.zeros(times.size)
    inside = np.flatnonzero((times >= extent[0]) & (times < extent[1]))
    power[inside] = 0.9
    f0[inside] = centre_hz
    for step in range(bursts):
        first = inside[int(len(inside) * (step + 1) / (bursts + 1))]
        f0[first : first + burst_frames] = centre_hz * 2.0
    return times, f0, power


def _wobble_tracks(
    duration_s: float, extent: tuple[float, float], centre_hz: float, semitones_peak: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An F0 oscillating by ``semitones_peak`` within every window, which is real instability.

    A slow sweep is deliberately *not* this: the spread is taken over a window, so a drift across
    a long production does not read as unsteady. Only fast variation does.

    Args:
        duration_s: How long the grid runs.
        extent: The interval the phonation occupies.
        centre_hz: The F0 the oscillation is centred on.
        semitones_peak: The oscillation's amplitude, in semitones.

    Returns:
        ``(times_s, f0_hz, strength)``.
    """
    times = np.arange(0.0, duration_s, HOP_S)
    f0 = np.zeros(times.size)
    power = np.zeros(times.size)
    inside = (times >= extent[0]) & (times < extent[1])
    power[inside] = 0.9
    wobble = semitones_peak * np.sin(2.0 * np.pi * 4.0 * times[inside])
    f0[inside] = centre_hz * np.power(2.0, wobble / 12.0)
    return times, f0, power


def seed(
    tmp_path: Path,
    *,
    stem: str | None = MPT_STEM,
    duration_s: float = 20.0,
    amplitude: tuple[tuple[float, float], ...] = ((2.0, 18.0),),
    tracks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    continuity: float | None = 0.95,
    write_tracks: bool = True,
    tracks_path: bool = False,
    words: tuple[tuple[str, float, float], ...] = (),
    span_labels: tuple[tuple[float, float, str], ...] = (),
    plain: np.ndarray | None = None,
) -> tuple[ProvStore, dict[str, Any]]:
    """A store carrying exactly the surface the two VOICE modes read.

    Args:
        tmp_path: The run directory.
        stem: The ``recording`` stream's BIDS stem, or None to omit the stream.
        duration_s: The recording's duration.
        amplitude: The extents PREPROCESS's ``amplitude`` spans cover.
        tracks: ``(times_s, f0_hz, strength)``; None makes the whole of each amplitude span voiced.
        continuity: The constant the continuity trace holds, or None to omit the derivative.
        write_tracks: Whether to write the ``phonation_tracks`` measurement and its sidecar.
        tracks_path: Whether that measurement carries a ``path`` attribute of its own.
        words: The consensus words, as ``(text, start, end)``.
        span_labels: Extra spans carrying a ruleset ``label``, as ``(start, end, label)``.
        plain: The ``plain`` stream's samples at 16 kHz, or None to omit the stream.

    Returns:
        The store and the ids it wrote.
    """
    store = ProvStore(run_id="voice-test")
    (tmp_path / "derivatives").mkdir(parents=True, exist_ok=True)
    ids: dict[str, Any] = {"amplitude": [], "labelled": []}
    if plain is not None:
        (tmp_path / "streams").mkdir(parents=True, exist_ok=True)
        soundfile.write(tmp_path / "streams" / "plain.flac", plain, RATE)
        ids["plain"] = store.entity(
            prov_type="stream",
            extent=(0.0, len(plain) / RATE),
            attributes={"name": "plain", "path": "streams/plain.flac", "sampling_rate": RATE},
        )
    if stem is not None:
        ids["recording"] = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={"name": "recording", "path": f"{stem}.wav", "sampling_rate": 16000},
        )
    for start, end in amplitude:
        ids["amplitude"].append(
            store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"measure": "amplitude", "signal": "preemphasised"},
            )
        )
    for start, end, label in span_labels:
        ids["labelled"].append(
            store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"measure": "amplitude", "signal": "preemphasised", "label": label},
            )
        )
    for index, (text, start, end) in enumerate(words):
        store.entity(
            prov_type="word",
            extent=(start, end),
            attributes={"index": index, "text": text, "bracketed": False},
        )
    if write_tracks:
        times, f0, strength = tracks if tracks is not None else _tracks(duration_s, list(amplitude))
        np.savez(
            tmp_path / "derivatives" / "phonation_tracks.npz",
            times_s=times,
            f0_hz=f0,
            strength=strength,
        )
        attributes: dict[str, Any] = {"name": "phonation_tracks", "signal": "preemphasised", "hop_s": HOP_S}
        if tracks_path:
            attributes["path"] = "derivatives/phonation_tracks.npz"
        ids["tracks"] = store.entity(prov_type="measurement", extent=None, attributes=attributes)
    if continuity is not None:
        trace = np.full(int(duration_s * CONTINUITY_RATE), continuity)
        np.savez(tmp_path / "derivatives" / "continuity_trace.npz", continuity=trace)
        ids["continuity"] = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "continuity_trace",
                "signal": "preemphasised",
                "path": "derivatives/continuity_trace.npz",
                "sampling_rate": CONTINUITY_RATE,
            },
        )
    return store, ids


def voice_spans(store: ProvStore) -> list[Any]:
    """Every span VOICE proposed, earliest first.

    Args:
        store: The provenance store.

    Returns:
        The ``family: "voice"`` spans.
    """
    found = [span for span in live_entities(store, "span") if span.attributes.get("family") == "voice"]
    return sorted(found, key=lambda span: span.extent or (0.0, 0.0))


class TestTheQualifierSeparatesAHeldVowelFromConnectedSpeech:
    """Without it the detect arm would propose attempts over connected speech on 14,332 recordings.

    ``Qualification.steady`` is that separation. It is what the detect arm proposes over, because
    nothing declared those recordings to hold a held vowel. The align arm reads the same qualities
    and reports them instead, so these tests read ``steady`` rather than ``carriers``.
    """

    def test_a_wandering_f0_fails_the_spread_qualifier(self, tmp_path: Path) -> None:
        """Connected speech has a high voiced fraction and a usable contour; it is not steady."""
        store, _ = seed(tmp_path, tracks=_wobble_tracks(20.0, (2.0, 18.0), 120.0, 6.0))
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).steady
            == []
        )

    def test_a_slow_sweep_is_not_read_as_instability(self, tmp_path: Path) -> None:
        """The discriminating half: a *local* spread is the statistic, by design.

        A 100-to-400 Hz drift across sixteen seconds is a sustained production, not a wobble.
        """
        store, _ = seed(tmp_path, tracks=_glide_tracks(20.0, (2.0, 18.0), 100.0, 400.0))
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).steady
            != []
        )

    def test_a_low_continuity_recording_fails_the_stationarity_qualifier(self, tmp_path: Path) -> None:
        """The spectral trace is the qualifier the F0 statistics cannot supply."""
        store, _ = seed(tmp_path, continuity=0.1)
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).steady
            == []
        )

    def test_a_mostly_unvoiced_carrier_fails_the_voiced_fraction(self, tmp_path: Path) -> None:
        """A carrier the tracker found almost no F0 in is not a phonation carrier."""
        store, _ = seed(tmp_path, amplitude=((2.0, 18.0),), tracks=_tracks(20.0, [(2.0, 4.0)]))
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).steady
            == []
        )

    def test_an_absent_continuity_trace_fails_the_qualifier_rather_than_passing_it(self, tmp_path: Path) -> None:
        """An absent qualifier must not read as a satisfied one."""
        store, _ = seed(tmp_path, continuity=None)
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).steady
            == []
        )


class TestTheModeIsSelectedByTheDeclaredFamily:
    """Both arms, both ways, and no third arm."""

    def test_a_voice_family_reaches_the_align_arm(self, tmp_path: Path) -> None:
        """``maximum-phonation-time`` is one of VOICE's six in-family rows."""
        store, _ = seed(tmp_path, stem=MPT_STEM)
        assert mode_of("VOICE", store) == ("align", "maximum-phonation-time")

    def test_another_branchs_family_reaches_the_detect_arm(self, tmp_path: Path) -> None:
        """A cough task is AIRWAY's; VOICE marks its speciality and evaluates nothing."""
        store, _ = seed(tmp_path, stem=COUGH_STEM)
        assert mode_of("VOICE", store) == ("detect", "voluntary-cough")
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.node == "VOICE"

    def test_the_detect_arm_evaluates_no_task(self, tmp_path: Path) -> None:
        """It takes no reading a conformance gate reads, so VERDICT can answer nothing about it."""
        store, _ = seed(tmp_path, stem=COUGH_STEM)
        result = detect_voice(store, params(), run_dir=tmp_path)
        assert gated_conformance(result, Pattern.SUSTAINED, settings=config()) == UNDETERMINED

    def test_the_detect_arm_still_proposes_the_phonation_it_finds(self, tmp_path: Path) -> None:
        """Sustained phonation occurs inside sentence reading and free speech; it is marked there."""
        store, _ = seed(tmp_path, stem=HARVARD_STEM)
        result = detect_voice(store, params(), run_dir=tmp_path)
        assert [proposal.role for proposal in result.components] == [PHONATION_ROLE]
        assert result.components[0].attributes["evaluates_no_task"] is True

    def test_no_derivable_task_family_takes_the_detect_arm(self, tmp_path: Path) -> None:
        """A path that is not a BIDS stem names no family, and None is the safe arm."""
        store, _ = seed(tmp_path, stem="not-a-bids-stem")
        assert mode_of("VOICE", store) == ("detect", None)
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED
        assert voice_spans(store)[0].attributes["role"] == PHONATION_ROLE

    def test_an_absent_recording_entity_takes_the_detect_arm(self, tmp_path: Path) -> None:
        """No carrier at all is the same safe arm, not an error."""
        store, _ = seed(tmp_path, stem=None)
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED

    def test_the_hint_carrier_selects_the_mode_before_the_path_does(self, tmp_path: Path) -> None:
        """``task_token`` is the clean route and is read first."""
        store, _ = seed(tmp_path, stem=COUGH_STEM)
        hint = AudioHints(metadata={"task_token": "glides-low-to-high"})
        assert mode_of("VOICE", store, hint) == ("align", "glides-low-to-high")


class TestThePendingDeclarationRowsTakeTheDetectArm:
    """``loudness`` and ``cape-v-sentences`` are LEXICAL_SPEECH, so out of family for VOICE."""

    @pytest.mark.parametrize("stem", [LOUDNESS_STEM, CAPEV_STEM])
    def test_a_pending_declaration_family_never_reaches_align(self, tmp_path: Path, stem: str) -> None:
        """The four rows are deliberately out of EXPECTATIONS; changing that is a families decision."""
        store, _ = seed(tmp_path, stem=stem)
        mode, family = mode_of("VOICE", store)
        assert mode == "detect"
        assert family in {"loudness", "cape-v-sentences"}

    def test_align_voice_refuses_a_family_it_has_no_row_for(self, tmp_path: Path) -> None:
        """The caller owes detect_voice; reaching align is the caller's error, not a silent pass."""
        store, _ = seed(tmp_path, stem=LOUDNESS_STEM)
        with pytest.raises(KeyError):
            align_voice("loudness", store, None, params(), run_dir=tmp_path)

    def test_every_voice_row_is_served_by_a_reachable_matcher(self, tmp_path: Path) -> None:
        """No row in the dispatch table may fall through to NotImplementedError."""
        store, _ = seed(tmp_path)
        for family in ("maximum-phonation-time", "maximum-phonation-time-v2", "prolonged-vowel"):
            assert align_voice(family, store, None, params(), run_dir=tmp_path) is not None
        for family in ("glides-low-to-high", "glides-high-to-low", "high-to-low"):
            assert align_voice(family, store, None, params(), run_dir=tmp_path) is not None


class TestAnAbsentInstrumentIsUndetermined:
    """Boundaries that cannot be placed are not an absence of voice."""

    def test_absent_tracks_make_the_sustained_arm_undetermined(self, tmp_path: Path) -> None:
        """``UNDETERMINED`` is what a mode returns when its only instrument is absent."""
        store, _ = seed(tmp_path, write_tracks=False)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert gated_conformance(result, Pattern.SUSTAINED, settings=config()) == UNDETERMINED
        assert result.components == []

    def test_absent_tracks_make_the_glide_arm_undetermined(self, tmp_path: Path) -> None:
        """The same rule on the other in-family matcher."""
        store, _ = seed(tmp_path, stem=GLIDE_UP_STEM, write_tracks=False)
        result = align_voice("glides-low-to-high", store, None, params(), run_dir=tmp_path)
        assert gated_conformance(result, Pattern.GLIDE, settings=config()) == UNDETERMINED

    def test_the_absence_is_recorded_as_unviable_rather_than_omitted(self, tmp_path: Path) -> None:
        """A measurement that could not be taken is written, not silently dropped."""
        store, _ = seed(tmp_path, write_tracks=False)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert [finding.name for finding in result.deviations] == ["phonation_extent"]
        assert result.deviations[0].evidence["value"] == "NOT_SEPARABLE_BY_THIS_DESIGN"

    def test_an_absent_instrument_does_not_read_as_an_absent_voice(self, tmp_path: Path) -> None:
        """``conformance is False`` means this branch looked and found no attempt. It did not look.

        The distinction is what keeps a non-conformance off the most impaired speakers at scale.
        """
        store, _ = seed(tmp_path, write_tracks=False)
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED

    def test_nothing_qualifying_is_undetermined_because_a_qualifier_is_not_a_verdict(self, tmp_path: Path) -> None:
        """A qualifier says where it is safe to measure, not whether the speaker did the task.

        The instrument was there, it found a carrier, and a qualifier discarded it. That the
        branch could not place a boundary it trusts is not evidence the instruction was ignored.
        """
        store, _ = seed(tmp_path, continuity=0.1)
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED


class TestEveryProposalNamesItsEvidence:
    """Under propose-only the derivation is the whole record of where an extent came from."""

    def test_the_detect_arms_span_is_derived_from_the_same_evidence(self, tmp_path: Path) -> None:
        """The two modes share the qualifier, so they share the derivation."""
        store, ids = seed(tmp_path, stem=COUGH_STEM)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        span = voice_spans(store)[0]
        assert ids["amplitude"][0] in store.derived_from(span.id)

    def test_the_count_in_is_derived_from_the_words_that_realised_it(self, tmp_path: Path) -> None:
        """A count-in's evidence is lexical, so its derivation is the words."""
        store, _ = seed(
            tmp_path,
            stem=PROLONGED_STEM,
            amplitude=((6.0, 18.0),),
            tracks=_tracks(20.0, [(6.0, 18.0)]),
            words=(("one", 1.0, 1.4), ("two", 1.6, 2.0), ("three", 2.2, 2.8)),
        )
        voice(store, "plain", config(), None, run_dir=tmp_path)
        count_in = [span for span in voice_spans(store) if span.attributes["role"] == COUNT_IN]
        assert count_in and len(store.derived_from(count_in[0].id)) == 3


class TestTheCountInIsItsOwnSpan:
    """It gets one precisely because it must be excluded from the vowel's measurement window."""

    def test_the_count_in_is_marked_excluded_from_measurement(self, tmp_path: Path) -> None:
        """Every Praat scalar today is taken over count-in plus silence plus vowel."""
        store, _ = seed(
            tmp_path,
            stem=PROLONGED_STEM,
            amplitude=((6.0, 18.0),),
            tracks=_tracks(20.0, [(6.0, 18.0)]),
            words=(("one", 1.0, 1.4), ("two", 1.6, 2.0), ("three", 2.2, 2.8)),
        )
        result = align_voice("prolonged-vowel", store, None, params(), run_dir=tmp_path)
        assert result.components[0].attributes["excluded_from_measurement"] is True

    def test_a_missing_count_in_token_is_an_omission(self, tmp_path: Path) -> None:
        """The instruction prescribes three; two realised leaves one omitted."""
        store, _ = seed(
            tmp_path,
            stem=PROLONGED_STEM,
            amplitude=((6.0, 18.0),),
            tracks=_tracks(20.0, [(6.0, 18.0)]),
            words=(("one", 1.0, 1.4), ("two", 1.6, 2.0)),
        )
        result = align_voice("prolonged-vowel", store, None, params(), run_dir=tmp_path)
        omissions = [finding for finding in result.deviations if finding.name == "omission"]
        assert [finding.evidence["expected"] for finding in omissions] == ["three"]


class TestTheAperiodicCaseNoLongerErrorsTheNode:
    """A frankly aperiodic voice is a finding about the voice, not a crash."""

    def test_the_node_no_longer_derives_an_f0_range_at_all(self, tmp_path: Path) -> None:
        """``derive_f0_range`` raising ``F0RangeUnavailable`` used to error the whole node.

        VOICE now reads the tracks PREPROCESS already computed, so it makes no such call. The probe
        is behavioural: a ``derive_f0_range`` that raises on every input changes nothing here.
        """
        assert not hasattr(voice_module, "derive_f0_range")

    def test_an_aperiodic_recording_completes_with_a_report(self, tmp_path: Path) -> None:
        """Zero voiced frames throughout: a report rather than a traceback, and no accusation.

        A frankly aperiodic voice is the case the tracker is least able to read, so it is the last
        recording whose silence should be folded as the speaker not having tried.
        """
        store, _ = seed(tmp_path, tracks=_tracks(20.0, []))
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED
        assert result.report_entity_id

    def test_an_aperiodic_recording_out_of_family_also_completes(self, tmp_path: Path) -> None:
        """The detect arm carries the same property on the 14,332 out-of-family recordings."""
        store, _ = seed(tmp_path, stem=COUGH_STEM, tracks=_tracks(20.0, []))
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED


class TestTheTracksAreFoundByEitherCarrier:
    """The ``phonation_tracks`` measurement carries no ``path``, unlike every other derivative."""

    def test_the_tracks_are_read_when_the_measurement_carries_a_path(self, tmp_path: Path) -> None:
        """The clean route, which is what the foundation's loader alone supports."""
        store, _ = seed(tmp_path, tracks_path=True)
        assert read_evidence(store, tmp_path).tracks is not None

    def test_the_tracks_are_read_when_it_carries_none(self, tmp_path: Path) -> None:
        """What PREPROCESS actually writes today; the loader alone would return None."""
        store, _ = seed(tmp_path, tracks_path=False)
        assert read_evidence(store, tmp_path).tracks is not None

    def test_no_measurement_at_all_reads_no_sidecar(self, tmp_path: Path) -> None:
        """A stray sidecar with no measurement beside it is not evidence."""
        store, _ = seed(tmp_path, write_tracks=True)
        stripped, _ = seed(tmp_path, write_tracks=False)
        assert read_evidence(stripped, tmp_path).tracks is None
        assert read_evidence(store, tmp_path).tracks is not None


class TestTheNodeWritesWhatItFound:
    """The store side: spans, findings, the activity's reads and the verdict's own record."""

    def test_the_activity_records_which_mode_ran_and_on_what(self, tmp_path: Path) -> None:
        """A run has to be able to say which arm it took without re-deriving the family."""
        store, _ = seed(tmp_path)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        activity = [each for each in store.activities() if each.node == "VOICE"][-1]
        assert activity.parameters["mode"] == "align"
        assert activity.parameters["declared_task_family"] == "maximum-phonation-time"

    def test_the_activity_used_the_carriers_it_read(self, tmp_path: Path) -> None:
        """The amplitude spans are the subject, so the graph records that they were read."""
        store, ids = seed(tmp_path)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        activity = [each for each in store.activities() if each.node == "VOICE"][-1]
        assert ids["amplitude"][0] in store.uses_of(activity.id)

    def test_the_report_carries_the_keys_the_summary_reads(self, tmp_path: Path) -> None:
        """``common.BRANCH_MEASURES["VOICE"]`` names three, and a writer may not drop a reader's."""
        store, _ = seed(tmp_path)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        report = find_branch_report(store, "VOICE")
        assert report is not None
        assert {"spans_n", "phonation_s", "longest_span_s"} <= set(report.attributes)

    def test_the_report_names_the_mode_and_the_conformance_value(self, tmp_path: Path) -> None:
        """Conformance is the branch's answer and belongs in the record, not only in the return."""
        store, _ = seed(tmp_path, stem=COUGH_STEM)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        report = find_branch_report(store, "VOICE")
        assert report is not None
        assert report.attributes["mode"] == "detect"
        assert report.attributes["conformance"] == UNDETERMINED


class TestARulesetLabelledSpanIsContestedNotRewritten:
    """Contesting is an assertion beside a span, so propose-only does not touch that path."""

    def test_a_labelled_span_failing_the_qualifier_is_contested(self, tmp_path: Path) -> None:
        """The object of a contest is a PREPROCESS span, never VOICE's own proposal."""
        store, _ = seed(
            tmp_path,
            stem=COUGH_STEM,
            amplitude=(),
            tracks=_tracks(20.0, []),
            span_labels=((3.0, 6.0, "phonation"),),
        )
        result = detect_voice(store, params(), run_dir=tmp_path)
        contested = [finding for finding in result.deviations if finding.kind == "contest"]
        assert contested and contested[0].evidence["reason"] == "fails_the_stationarity_qualifier"

    def test_a_labelled_span_that_qualifies_is_not_contested(self, tmp_path: Path) -> None:
        """The discriminating half: it qualified, so there is nothing to contest."""
        store, _ = seed(
            tmp_path,
            stem=COUGH_STEM,
            amplitude=(),
            tracks=_tracks(20.0, [(3.0, 15.0)]),
            span_labels=((3.0, 15.0, "phonation"),),
        )
        result = detect_voice(store, params(), run_dir=tmp_path)
        assert [finding for finding in result.deviations if finding.kind == "contest"] == []


class TestTheSpreadQualifierJudgesATypicalWindowNotTheWorstOne:
    """``f0_spread_max_semitones`` is a per-window bound; the statistic must be a window's spread.

    Read as the maximum over every window, a bound written to tolerate tracker noise was certain to
    find it: the number of windows grows with the duration of the production being qualified, so a
    longer held vowel was more likely to be discarded than a short one. Measured across 300
    recordings in ``specs/20260817-triage-workflow-dag/voice-flag-grounds.md``.
    """

    def test_a_steady_vowel_with_a_few_octave_errors_still_qualifies(self, tmp_path: Path) -> None:
        """The production is steady; the tracker is not. One is the speaker, the other is not."""
        store, _ = seed(tmp_path, tracks=_octave_error_tracks(20.0, (2.0, 18.0), 120.0, bursts=2))
        qualification = qualifying_phonation(
            read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
        )
        assert [carrier.span.id for carrier in qualification.carriers]
        assert qualification.carriers[0].f0_spread_semitones < MEASURED["f0_spread_max_semitones"]

    def test_the_same_recording_read_by_the_worst_window_would_be_discarded(self, tmp_path: Path) -> None:
        """The defect, stated as the contrast: the worst window reads an octave on this signal."""
        store, _ = seed(tmp_path, tracks=_octave_error_tracks(20.0, (2.0, 18.0), 120.0, bursts=2))
        evidence = read_evidence(store, tmp_path)
        assert evidence.tracks is not None
        track = track_slice(evidence.tracks, (2.0, 18.0), MEASURED["voiced_strength_min"])
        pitch = semitones(np.where(track.voiced, track.f0_hz, np.nan))
        spreads = windowed_spreads(pitch, track.hop_s, MEASURED["f0_spread_window_s"])
        assert spreads.max() > 11.0
        assert float(np.median(spreads)) < 1.0

    def test_the_verdict_does_not_change_with_the_length_of_the_production(self, tmp_path: Path) -> None:
        """Duration independence is the property the maximum did not have.

        The same signal and the same error rate, held four seconds and twenty-four: a qualifier
        must read them alike, or it penalises the task for being performed well.
        """
        short, _ = seed(
            tmp_path / "short",
            amplitude=((1.0, 5.0),),
            duration_s=6.0,
            tracks=_octave_error_tracks(6.0, (1.0, 5.0), 120.0, bursts=1),
        )
        long_, _ = seed(
            tmp_path / "long",
            amplitude=((1.0, 25.0),),
            duration_s=26.0,
            tracks=_octave_error_tracks(26.0, (1.0, 25.0), 120.0, bursts=6),
        )
        expectation = Expectation(pattern=Pattern.SUSTAINED)
        assert len(qualifying_phonation(read_evidence(short, tmp_path / "short"), expectation, params()).steady) == 1
        assert len(qualifying_phonation(read_evidence(long_, tmp_path / "long"), expectation, params()).steady) == 1

    def test_genuine_instability_is_still_discarded(self, tmp_path: Path) -> None:
        """The gate must keep rejecting what it was for; a fix that accepts everything is no fix."""
        store, _ = seed(tmp_path, tracks=_wobble_tracks(20.0, (2.0, 18.0), 120.0, 6.0))
        qualification = qualifying_phonation(
            read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
        )
        assert qualification.steady == []
        assert [rejection.gate for carrier in qualification.carriers for rejection in carrier.failed] == [
            "f0_spread_max_semitones"
        ]


class TestADiscardedCarrierIsReportedRatherThanVanishing:
    """A branch reports what it found; a carrier it threw away is part of what it found."""

    def test_the_rejection_names_the_gate_and_the_value_it_read(self, tmp_path: Path) -> None:
        """Without this the store cannot separate an absent production from a discarded one."""
        store, _ = seed(tmp_path, stem=COUGH_STEM, continuity=0.1)
        result = detect_voice(store, params(), run_dir=tmp_path)
        rejected = [finding for finding in result.deviations if finding.name == "carrier_rejected"]
        assert [finding.evidence["value"] for finding in rejected] == ["continuity_min"]
        assert rejected[0].evidence["value_read"] == pytest.approx(0.1, abs=0.01)
        assert rejected[0].evidence["bound"] == MEASURED["continuity_min"]
        assert rejected[0].derived_from

    def test_the_rejection_is_a_measurement_and_not_a_deviation(self, tmp_path: Path) -> None:
        """It is a reading of this branch's instrument, not a departure by the speaker."""
        store, _ = seed(tmp_path, stem=COUGH_STEM, continuity=0.1)
        result = detect_voice(store, params(), run_dir=tmp_path)
        rejected = [finding for finding in result.deviations if finding.name == "carrier_rejected"]
        assert {finding.kind for finding in rejected} == {"measure"}
        assert "carrier_rejected" not in deviation_names(result.deviations)


class TestAQualityGateDoesNotDecideThatTheProductionNeverHappened:
    """In the align arm the instruction says what was asked for, so a poor attempt is still one."""

    def test_the_detect_arm_still_applies_the_quality_qualifier(self, tmp_path: Path) -> None:
        """Nothing declared this recording to hold a held vowel, so steadiness is what separates one."""
        store, _ = seed(
            tmp_path,
            stem=COUGH_STEM,
            amplitude=((2.0, 18.0),),
            tracks=_wobble_tracks(20.0, (2.0, 18.0), 120.0, 6.0),
        )
        result = detect_voice(store, params(), run_dir=tmp_path)
        assert result.components == []
        assert [finding.evidence["value"] for finding in result.deviations if finding.name == "carrier_rejected"] == [
            "f0_spread_max_semitones"
        ]


class TestConformanceIsNeverFalse:
    """These instruments establish that a sustained production happened, not that none did."""

    def test_no_reachable_body_returns_a_false_conformance(self) -> None:
        """A structural guard: a future ``Result(False, ...)`` here is a new accusation.

        The branch has no fitted criterion for the absence of phonation, so every ``False`` it
        could write today would rest on a qualifier's reading rather than on the speaker's.
        """
        source = Path(voice_module.__file__).read_text()
        assert "Result(False" not in source
        assert "conformance=UNDETERMINED" in source

    def test_an_empty_recording_is_undetermined_rather_than_a_non_conformance(self, tmp_path: Path) -> None:
        """No amplitude span at all: the branch had no subject, which is not a failed attempt."""
        store, _ = seed(tmp_path, amplitude=())
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED


def _held(pieces: tuple[tuple[float, float, float], ...], total_s: float) -> np.ndarray:
    """Held harmonic tones over room noise, as the ``plain`` stream.

    Args:
        pieces: ``(start, end, f0_hz)`` per tone.
        total_s: The stream's duration.

    Returns:
        The samples.
    """
    rng = np.random.default_rng(0)
    x = 1e-3 * rng.standard_normal(int(total_s * RATE))
    for start, end, hz in pieces:
        a, b = int(start * RATE), int(end * RATE)
        phase = 2 * np.pi * hz * np.arange(b - a) / RATE
        x[a:b] += 0.2 * np.sum([np.sin(k * phase) / k for k in range(1, 8)], axis=0)
    return x.astype(np.float32)


class TestTheAlignArmReadsThePhonationAttempt:
    """The declared voice task's extent is the phonation attempt on the plain stream."""

    def test_a_held_vowel_is_the_task_extent(self, tmp_path: Path) -> None:
        """The extent runs from the vowel's onset to its offset, whatever the tracks say."""
        store, _ = seed(
            tmp_path, duration_s=8.0, amplitude=(), write_tracks=False, plain=_held(((1.0, 6.0, 140.0),), 8.0)
        )
        voice(store, "plain", config(), run_dir=tmp_path)
        [span] = [span for span in voice_spans(store) if span.attributes.get("role") == TASK_EXTENT]
        assert span.extent is not None
        assert span.extent[0] == pytest.approx(1.0, abs=0.1)
        assert span.extent[1] == pytest.approx(6.0, abs=0.1)

    def test_the_reading_reaches_the_store(self, tmp_path: Path) -> None:
        """VOICE writes the reading VERDICT decides on."""
        store, _ = seed(
            tmp_path, duration_s=8.0, amplitude=(), write_tracks=False, plain=_held(((1.0, 6.0, 140.0),), 8.0)
        )
        voice(store, "plain", config(), run_dir=tmp_path)
        reading = find_measurement(store, voice_module.PHONATION_READING)
        assert reading is not None
        assert reading.attributes["found"] is True
        assert reading.attributes["absent"] == []

    def test_silence_proposes_no_extent(self, tmp_path: Path) -> None:
        """Nothing over the floor is no attempt, and no span."""
        store, _ = seed(tmp_path, duration_s=5.0, amplitude=(), write_tracks=False, plain=_held((), 5.0))
        voice(store, "plain", config(), run_dir=tmp_path)
        assert not [span for span in voice_spans(store) if span.attributes.get("role") == TASK_EXTENT]
        reading = find_measurement(store, voice_module.PHONATION_READING)
        assert reading is not None and reading.attributes["found"] is False

    def test_an_absent_plain_stream_is_an_absent_input(self, tmp_path: Path) -> None:
        """Without the plain stream the reading names it as absent."""
        store, _ = seed(tmp_path, duration_s=8.0)
        voice(store, "plain", config(), run_dir=tmp_path)
        reading = find_measurement(store, voice_module.PHONATION_READING)
        assert reading is not None and reading.attributes["absent"] == ["plain"]

    def test_a_glide_reports_its_travel(self, tmp_path: Path) -> None:
        """A held vowel in a rising-glide task travels nowhere and reads as held."""
        store, _ = seed(
            tmp_path,
            stem=GLIDE_UP_STEM,
            duration_s=6.0,
            amplitude=(),
            write_tracks=False,
            plain=_held(((1.0, 5.0, 190.0),), 6.0),
        )
        voice(store, "plain", config(), run_dir=tmp_path)
        reading = find_measurement(store, voice_module.PHONATION_READING)
        assert reading is not None and reading.attributes["shape"] == "held"
