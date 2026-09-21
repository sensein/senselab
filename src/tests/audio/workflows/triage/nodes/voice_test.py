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
from senselab.audio.workflows.triage.nodes.common import find_branch_report, live_entities
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
from tests.audio.workflows.triage.nodes.conftest import gated_conformance, readings_of

HOP_S = 0.01
"""PREPROCESS's own phonation-track hop, which the fixtures build their frame grid on."""

CONTINUITY_RATE = 100.0
"""The continuity trace's sampling rate in the fixtures, in Hz."""

MPT_STEM = "sub-abc_ses-1_task-maximum-phonation-time"
PROLONGED_STEM = "sub-abc_ses-1_task-prolonged-vowel"
GLIDE_UP_STEM = "sub-abc_ses-1_task-glides-low-to-high"
GLIDE_DOWN_STEM = "sub-abc_ses-1_task-glides-high-to-low"
COUGH_STEM = "sub-abc_ses-1_task-voluntary-cough"
LOUDNESS_STEM = "sub-abc_ses-1_task-loudness"
CAPEV_STEM = "sub-abc_ses-1_task-cape-v-sentences"
HARVARD_STEM = "sub-abc_ses-1_task-harvard-sentences-list"

SETTINGS = {"voiced_strength_min": 0.5, "f0_spread_window_s": 1.0}
"""Instrument settings supplied by the fixture, overriding the packaged ones for the test signals."""

GATES = {
    "production_min_s": 0.5,
    "voiced_fraction_min": 0.6,
    "f0_spread_max_semitones": 4.0,
    "continuity_min": 0.8,
    "monotone_tolerance_semitones": 1.0,
    "dominant_segment_min_fraction": 0.5,
}
"""Gate bounds supplied by the fixture. Each group takes the ones its own body reads."""

GROUP_GATES = {
    Pattern.SUSTAINED: ("production_min_s", "voiced_fraction_min", "f0_spread_max_semitones", "continuity_min"),
    Pattern.GLIDE: (
        "production_min_s",
        "voiced_fraction_min",
        "monotone_tolerance_semitones",
        "dominant_segment_min_fraction",
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


def _node_readings(store: ProvStore) -> dict[str, Any]:
    """Every measurement the node wrote, keyed by name, as VERDICT reads them.

    Args:
        store: The provenance store the node wrote into.

    Returns:
        Measurement name to its value.
    """
    return {
        str(entity.attributes["name"]): entity.attributes.get("value") for entity in live_entities(store, "measurement")
    }


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


def _zigzag_tracks(
    duration_s: float, extent: tuple[float, float], low_hz: float, high_hz: float, legs: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A fully voiced carrier whose pitch reverses, so no one monotone run dominates it.

    Args:
        duration_s: How long the grid runs.
        extent: The interval the phonation occupies.
        low_hz: F0 at the bottom of each leg.
        high_hz: F0 at the top of each leg.
        legs: How many monotone legs to divide the extent into.

    Returns:
        ``(times_s, f0_hz, strength)``.
    """
    times = np.arange(0.0, duration_s, HOP_S)
    f0 = np.zeros(times.size)
    power = np.zeros(times.size)
    inside = np.flatnonzero((times >= extent[0]) & (times < extent[1]))
    power[inside] = 0.9
    edges = np.linspace(0, inside.size, legs + 1).astype(int)
    for leg in range(legs):
        first, last = edges[leg], edges[leg + 1]
        ends = (low_hz, high_hz) if leg % 2 == 0 else (high_hz, low_hz)
        f0[inside[first:last]] = np.linspace(ends[0], ends[1], last - first)
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

    Returns:
        The store and the ids it wrote.
    """
    store = ProvStore(run_id="voice-test")
    (tmp_path / "derivatives").mkdir(parents=True, exist_ok=True)
    ids: dict[str, Any] = {"amplitude": [], "labelled": []}
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


def measurements(store: ProvStore, name: str) -> list[Any]:
    """Every live measurement of one name.

    Args:
        store: The provenance store.
        name: The measurement's name.

    Returns:
        The entities, in write order.
    """
    return [entity for entity in live_entities(store, "measurement") if entity.attributes.get("name") == name]


def assertions(store: ProvStore, verb: str) -> list[Any]:
    """Every live assertion of one verb.

    Args:
        store: The provenance store.
        verb: ``deviate`` or ``contest``.

    Returns:
        The entities, in write order.
    """
    return [entity for entity in live_entities(store, "assertion") if entity.attributes.get("verb") == verb]


class TestASustainedVowelYieldsTheHeldVowelSpan:
    """The branch's foundational capability, and the thing it failed to do on every recording."""

    def test_a_held_vowel_is_proposed_as_a_task_extent_of_family_voice(self, tmp_path: Path) -> None:
        """Sixteen seconds of held phonation becomes one span, in VOICE's own family."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert [proposal.role for proposal in result.components] == [TASK_EXTENT]
        assert {proposal.family for proposal in result.components} == {"voice"}

    def test_the_extent_is_the_productions_own_voiced_boundaries(self, tmp_path: Path) -> None:
        """Not the carrier span's: the first and last voiced frame inside it."""
        store, _ = seed(tmp_path, amplitude=((2.0, 18.0),), tracks=_tracks(20.0, [(4.0, 15.0)]))
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        start, end = result.components[0].start, result.components[0].end
        assert start == pytest.approx(4.0, abs=0.02)
        assert end == pytest.approx(15.0, abs=0.02)

    def test_the_carriers_own_extent_travels_on_the_span(self, tmp_path: Path) -> None:
        """A reader needs the region the span was cut from, not only the cut."""
        store, _ = seed(tmp_path, amplitude=((2.0, 18.0),), tracks=_tracks(20.0, [(4.0, 15.0)]))
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert result.components[0].attributes["carrier_extent"] == [2.0, 18.0]

    def test_the_onset_to_offset_duration_is_measured_under_a_qualified_name(self, tmp_path: Path) -> None:
        """Never ``maximum_phonation_time``: that name is read against published norms."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        names = {finding.name for finding in result.deviations if finding.kind == "measure"}
        assert "phonation_onset_to_offset_s" in names
        assert "maximum_phonation_time" not in names

    def test_the_voiced_duration_is_reported_beside_the_extent(self, tmp_path: Path) -> None:
        """The extent is an upper bound; the voiced total is the other half of V2's triple."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        names = {finding.name for finding in result.deviations if finding.kind == "measure"}
        assert {"phonation_onset_to_offset_s", "voiced_duration_s", "interruptions"} <= names

    def test_the_three_qualifiers_travel_on_the_span(self, tmp_path: Path) -> None:
        """A type-3 voice is a finding about the voice, so the qualifiers are always carried."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        attributes = result.components[0].attributes
        assert {"voiced_fraction", "f0_spread_semitones", "stationarity", "support_frames"} <= set(attributes)

    def test_an_interruption_is_located_rather_than_only_counted(self, tmp_path: Path) -> None:
        """V2's third item: the number, the locations and the total duration."""
        store, _ = seed(tmp_path, amplitude=((2.0, 18.0),), tracks=_tracks(20.0, [(3.0, 8.0), (11.0, 17.0)]))
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        found = [finding for finding in result.deviations if finding.name == "interruptions"]
        assert found and found[0].evidence["value"] == 1
        assert found[0].evidence["locations"][0][0] == pytest.approx(8.0, abs=0.05)

    def test_a_carrier_shorter_than_the_minimum_is_not_a_production(self, tmp_path: Path) -> None:
        """``production_min_s`` is the shortest carrier a production may be found in."""
        store, _ = seed(tmp_path, amplitude=((2.0, 2.2),), duration_s=5.0)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert result.components == []
        assert readings_of(result).get("carrier_duration_s") is None
        assert gated_conformance(result, Pattern.SUSTAINED, settings=config()) == UNDETERMINED


class TestTheQualifierSeparatesAHeldVowelFromConnectedSpeech:
    """Without it V1 would propose attempts over runs of connected speech on 14,332 recordings."""

    def test_a_wandering_f0_fails_the_spread_qualifier(self, tmp_path: Path) -> None:
        """Connected speech has a high voiced fraction and a usable contour; it is not steady."""
        store, _ = seed(tmp_path, tracks=_wobble_tracks(20.0, (2.0, 18.0), 120.0, 6.0))
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).carriers
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
            ).carriers
            != []
        )

    def test_a_low_continuity_recording_fails_the_stationarity_qualifier(self, tmp_path: Path) -> None:
        """The spectral trace is the qualifier the F0 statistics cannot supply."""
        store, _ = seed(tmp_path, continuity=0.1)
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).carriers
            == []
        )

    def test_a_mostly_unvoiced_carrier_fails_the_voiced_fraction(self, tmp_path: Path) -> None:
        """A carrier the tracker found almost no F0 in is not a phonation carrier."""
        store, _ = seed(tmp_path, amplitude=((2.0, 18.0),), tracks=_tracks(20.0, [(2.0, 4.0)]))
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).carriers
            == []
        )

    def test_an_absent_continuity_trace_fails_the_qualifier_rather_than_passing_it(self, tmp_path: Path) -> None:
        """An absent qualifier must not read as a satisfied one."""
        store, _ = seed(tmp_path, continuity=None)
        assert (
            qualifying_phonation(
                read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
            ).carriers
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

    def test_the_task_extent_is_derived_from_its_carrier_and_the_tracks(self, tmp_path: Path) -> None:
        """The carrier is the region; the tracks are what placed the boundary inside it."""
        store, ids = seed(tmp_path)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        span = voice_spans(store)[0]
        assert set(store.derived_from(span.id)) == {ids["amplitude"][0], ids["tracks"], ids["continuity"]}

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

    def test_the_declared_duration_count_names_the_recording_stream(self, tmp_path: Path) -> None:
        """The count is read off the recording's extent, so the stream it came from is its evidence."""
        store, ids = seed(
            tmp_path,
            stem=PROLONGED_STEM,
            duration_s=20.0,
            amplitude=((6.0, 18.0),),
            tracks=_tracks(20.0, [(6.0, 18.0)]),
            words=(("one", 1.0, 1.4), ("two", 1.6, 2.0), ("three", 2.2, 2.8)),
        )
        result = align_voice("prolonged-vowel", store, None, params(), run_dir=tmp_path)
        declared = [finding for finding in result.deviations if finding.name == "declared_duration_s"]
        assert [finding.derived_from for finding in declared] == [(ids["recording"],)]

    def test_no_proposal_is_written_without_a_derivation(self, tmp_path: Path) -> None:
        """``propose_span`` is the only writer and it refuses one; this pins the branch's side."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert all(proposal.derived_from for proposal in result.components)


class TestTheCountInIsItsOwnSpan:
    """It gets one precisely because it must be excluded from the vowel's measurement window."""

    def test_the_count_in_and_the_vowel_are_two_spans(self, tmp_path: Path) -> None:
        """``prolonged-vowel`` prescribes a lexical count-in; MPT forbids lexical content."""
        store, _ = seed(
            tmp_path,
            stem=PROLONGED_STEM,
            amplitude=((6.0, 18.0),),
            tracks=_tracks(20.0, [(6.0, 18.0)]),
            words=(("one", 1.0, 1.4), ("two", 1.6, 2.0), ("three", 2.2, 2.8)),
        )
        result = align_voice("prolonged-vowel", store, None, params(), run_dir=tmp_path)
        assert [proposal.role for proposal in result.components] == [COUNT_IN, TASK_EXTENT]

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

    def test_a_carrier_holding_the_count_in_is_not_the_vowel(self, tmp_path: Path) -> None:
        """``lexical_separator`` excludes a carrier the count-in overlaps."""
        store, _ = seed(
            tmp_path,
            stem=PROLONGED_STEM,
            amplitude=((1.0, 3.0),),
            tracks=_tracks(20.0, [(1.0, 3.0)]),
            words=(("one", 1.0, 1.4), ("two", 1.6, 2.0), ("three", 2.2, 2.8)),
        )
        result = align_voice("prolonged-vowel", store, None, params(), run_dir=tmp_path)
        assert [proposal.role for proposal in result.components] == [COUNT_IN]

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

    def test_no_count_in_at_all_does_not_decide_the_vowel(self, tmp_path: Path) -> None:
        """A held vowel carries no lexical content by design; the count-in cannot be its verdict.

        The transcript of a sustained vowel has no words in it, so keying ``done`` to an ordered
        run of ``one``/``two``/``three`` made the task's own definition guarantee a ``False``.
        """
        store, _ = seed(tmp_path, stem=PROLONGED_STEM, amplitude=((6.0, 18.0),), tracks=_tracks(20.0, [(6.0, 18.0)]))
        result = align_voice("prolonged-vowel", store, None, params(), run_dir=tmp_path)
        assert gated_conformance(result, Pattern.SUSTAINED, settings=config()) is True
        assert [proposal.role for proposal in result.components] == ["task_extent"]
        assert [finding.evidence["expected"] for finding in result.deviations if finding.name == "omission"] == [
            "one",
            "two",
            "three",
        ]

    def test_a_family_prescribing_no_count_in_is_done_on_the_vowel_alone(self, tmp_path: Path) -> None:
        """MPT declares no tokens, so nothing lexical is owed."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert gated_conformance(result, Pattern.SUSTAINED, settings=config()) is True


class TestTheGlideReadsTheSweepAgainstItsDeclaredDirection:
    """V3: direction is measured, and the declaration is what it is read against."""

    def test_an_upward_sweep_is_proposed_with_its_direction(self, tmp_path: Path) -> None:
        """The dominant monotone segment, and which way it ran."""
        store, _ = seed(tmp_path, stem=GLIDE_UP_STEM, tracks=_glide_tracks(20.0, (2.0, 18.0), 100.0, 300.0))
        result = align_voice("glides-low-to-high", store, None, params(), run_dir=tmp_path)
        assert result.components[0].attributes["direction"] == "up"
        assert result.components[0].attributes["production"] == "glide"

    def test_a_sweep_running_the_wrong_way_is_a_deviation(self, tmp_path: Path) -> None:
        """``sweep_direction_mismatch``, with both directions named."""
        store, _ = seed(tmp_path, stem=GLIDE_UP_STEM, tracks=_glide_tracks(20.0, (2.0, 18.0), 300.0, 100.0))
        result = align_voice("glides-low-to-high", store, None, params(), run_dir=tmp_path)
        found = [finding for finding in result.deviations if finding.name == "sweep_direction_mismatch"]
        assert found and found[0].evidence == {
            "declared": "up",
            "measured": "down",
            "extent_semitones": found[0].evidence["extent_semitones"],
        }

    def test_a_downward_declaration_matched_by_a_downward_sweep_deviates_not_at_all(self, tmp_path: Path) -> None:
        """The same recording against the other declaration."""
        store, _ = seed(tmp_path, stem=GLIDE_DOWN_STEM, tracks=_glide_tracks(20.0, (2.0, 18.0), 300.0, 100.0))
        result = align_voice("glides-high-to-low", store, None, params(), run_dir=tmp_path)
        assert [finding.name for finding in result.deviations if finding.kind == "deviation"] == []

    def test_the_semitone_extent_is_measured(self, tmp_path: Path) -> None:
        """An octave and a half in Hz is a semitone count, which is what a reader compares."""
        store, _ = seed(tmp_path, stem=GLIDE_UP_STEM, tracks=_glide_tracks(20.0, (2.0, 18.0), 100.0, 200.0))
        result = align_voice("glides-low-to-high", store, None, params(), run_dir=tmp_path)
        found = [finding for finding in result.deviations if finding.name == "glide_extent_semitones"]
        assert found and found[0].evidence["value"] == pytest.approx(12.0, abs=0.5)

    def test_no_monotone_segment_leaves_the_sweep_undetermined(self, tmp_path: Path) -> None:
        """A steady vowel is not a sweep, and no sweep found is not the task not performed."""
        store, _ = seed(tmp_path, stem=GLIDE_UP_STEM, amplitude=((2.0, 3.0),), tracks=_tracks(20.0, [(2.0, 2.2)]))
        result = align_voice("glides-low-to-high", store, None, params(), run_dir=tmp_path)
        assert gated_conformance(result, Pattern.GLIDE, settings=config()) == UNDETERMINED
        assert result.components == []


class TestTheDeviationsAreTheOnesTheDesignNames:
    """Three types, and a deviation is not evidence of a bad recording."""

    def test_an_attempt_running_to_the_boundary_is_truncated(self, tmp_path: Path) -> None:
        """A truncated trial is right-censored and must never pool with complete ones."""
        store, _ = seed(tmp_path, amplitude=((0.0, 20.0),), tracks=_tracks(20.0, [(0.0, 20.0)]))
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        found = [finding for finding in result.deviations if finding.name == "truncation"]
        assert found and found[0].evidence["reading"] == "right_censored"

    def test_an_attempt_inside_the_recording_is_not_truncated(self, tmp_path: Path) -> None:
        """The discriminating half: the same code on a complete trial."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert [finding.name for finding in result.deviations if finding.name == "truncation"] == []

    def test_a_second_attempt_is_a_repeat_attempt(self, tmp_path: Path) -> None:
        """Taking a maximum silently discards false starts, so each is reported."""
        store, _ = seed(
            tmp_path,
            amplitude=((2.0, 8.0), (11.0, 19.0)),
            tracks=_tracks(20.0, [(2.0, 8.0), (11.0, 19.0)]),
        )
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert len([finding for finding in result.deviations if finding.name == "repeat_attempt"]) == 1

    def test_the_attempt_count_is_reported_rather_than_only_the_longest(self, tmp_path: Path) -> None:
        """``attempt_count`` is a counts entry."""
        store, _ = seed(
            tmp_path,
            amplitude=((2.0, 8.0), (11.0, 19.0)),
            tracks=_tracks(20.0, [(2.0, 8.0), (11.0, 19.0)]),
        )
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        found = [finding for finding in result.deviations if finding.name == "attempt_count"]
        assert found and found[0].evidence["found"] == 2

    def test_the_longest_attempt_is_the_one_proposed(self, tmp_path: Path) -> None:
        """The others are deviations, not competing task extents."""
        store, _ = seed(
            tmp_path,
            amplitude=((2.0, 8.0), (11.0, 19.0)),
            tracks=_tracks(20.0, [(2.0, 8.0), (11.0, 19.0)]),
        )
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert len(result.components) == 1
        assert result.components[0].start == pytest.approx(11.0, abs=0.02)

    def test_lexical_content_in_a_no_lexical_task_is_a_deviation(self, tmp_path: Path) -> None:
        """MPT forbids lexical content; the deviation keys on positively identified words."""
        store, _ = seed(tmp_path, words=(("hello", 1.0, 1.5),))
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        found = [finding for finding in result.deviations if finding.name == "lexical_content"]
        assert found and found[0].evidence["text"] == "hello"

    def test_the_inhale_is_counted_and_never_proposed(self, tmp_path: Path) -> None:
        """An inhale is airway evidence; a branch mints only in its own family."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        assert [finding.name for finding in result.deviations if finding.name == "inhale_expected_in_file"]
        assert {proposal.family for proposal in result.components} == {"voice"}

    def test_the_v2_row_expects_no_inhale(self, tmp_path: Path) -> None:
        """The discriminating half: v1 places the inhale before the record tap, v2 does not."""
        store, _ = seed(tmp_path)
        result = align_voice("maximum-phonation-time-v2", store, None, params(), run_dir=tmp_path)
        assert [finding.name for finding in result.deviations if finding.name == "inhale_expected_in_file"] == []


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

    def test_the_proposed_span_reaches_the_store_with_its_family_and_role(self, tmp_path: Path) -> None:
        """``propose_spans`` stamps both; a reader has only the store to go on."""
        store, _ = seed(tmp_path)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        span = voice_spans(store)[0]
        assert span.attributes["family"] == "voice"
        assert span.attributes["role"] == TASK_EXTENT

    def test_the_measurements_reach_the_store(self, tmp_path: Path) -> None:
        """A measure finding becomes its own measurement entity over its extent."""
        store, _ = seed(tmp_path)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        assert measurements(store, "phonation_onset_to_offset_s")

    def test_the_counts_fold_into_one_measurement(self, tmp_path: Path) -> None:
        """Every count becomes one ``counts`` entry carrying found beside declared."""
        store, _ = seed(tmp_path)
        voice(store, "plain", config(), None, run_dir=tmp_path)
        counts = measurements(store, "counts")
        assert len(counts) == 1
        assert "attempt_count" in counts[0].attributes["entries"]

    def test_a_deviation_becomes_an_assertion_beside_the_span(self, tmp_path: Path) -> None:
        """Never an edit to one: the store is append-only and VOICE proposes only."""
        store, _ = seed(tmp_path, amplitude=((0.0, 20.0),), tracks=_tracks(20.0, [(0.0, 20.0)]))
        voice(store, "plain", config(), None, run_dir=tmp_path)
        assert [entity.attributes["deviation_type"] for entity in assertions(store, "deviate")] == ["truncation"]

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

    def test_a_found_attempt_reads_as_the_kind_being_present(self, tmp_path: Path) -> None:
        """The branch answers nothing; it reports the readings the SUSTAINED gates then clear."""
        store, _ = seed(tmp_path)
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED
        assert result.report.kind == "voice"
        readings = _node_readings(store)
        assert readings["carrier_duration_s"] >= GATES["production_min_s"]
        assert readings["carrier_voiced_fraction"] >= GATES["voiced_fraction_min"]

    def test_a_deviation_is_recorded_rather_than_read_as_a_non_conformance(self, tmp_path: Path) -> None:
        """A truncated attempt is still an attempt; ``conformance is False`` is reserved for none."""
        store, _ = seed(tmp_path, amplitude=((0.0, 20.0),), tracks=_tracks(20.0, [(0.0, 20.0)]))
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED
        assert _node_readings(store)["carrier_duration_s"] >= GATES["production_min_s"]
        assert "truncation" in result.report.deviations


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


class TestAnUnmeasuredOperatingPointIsRecordedRatherThanRaised:
    """No number in code: every ``branch.*`` key is read, never a literal — and none is refused.

    The packaged file now ships a reasoned value for every key VOICE reads, so the packaged config
    alone no longer raises. What is still pinned is that clearing a key by override does not raise
    either: the qualifier it gated is skipped, the key is named in the report's ``unmeasured``, and
    the dependent conformance is ``False`` here because an unqualified carrier search finds none.
    """

    def test_an_unmeasured_operating_point_is_recorded_rather_than_raised(self, tmp_path: Path) -> None:
        """``production_min_s`` cleared by override: no carrier ever qualifies, and the ask is named."""
        store, _ = seed(tmp_path)
        result = voice(store, "plain", config(production_min_s=None), None, run_dir=tmp_path)
        # UNDETERMINED, not False: an unmeasured qualifier could neither admit nor reject, so
        # claiming the instruction was not met would rest on a number nobody chose.
        assert result.report.conformance == UNDETERMINED
        assert "verdict.gates.by_group.SUSTAINED.production_min_s" in result.report.unmeasured

    def test_one_null_key_does_not_fail_a_body_that_needs_another(self, tmp_path: Path) -> None:
        """``BranchParams`` reads lazily; the glide arm never reads the sustained arm's keys."""
        store, _ = seed(tmp_path, stem=GLIDE_UP_STEM, tracks=_glide_tracks(20.0, (2.0, 18.0), 100.0, 300.0))
        result = align_voice("glides-low-to-high", store, None, params(f0_spread_max_semitones=None), run_dir=tmp_path)
        assert result.components[0].attributes["direction"] == "up"


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
        assert len(qualifying_phonation(read_evidence(short, tmp_path / "short"), expectation, params()).carriers) == 1
        assert len(qualifying_phonation(read_evidence(long_, tmp_path / "long"), expectation, params()).carriers) == 1

    def test_genuine_instability_is_still_discarded(self, tmp_path: Path) -> None:
        """The gate must keep rejecting what it was for; a fix that accepts everything is no fix."""
        store, _ = seed(tmp_path, tracks=_wobble_tracks(20.0, (2.0, 18.0), 120.0, 6.0))
        qualification = qualifying_phonation(
            read_evidence(store, tmp_path), Expectation(pattern=Pattern.SUSTAINED), params()
        )
        assert qualification.carriers == []
        assert [rejection.gate for rejection in qualification.rejected] == ["f0_spread_max_semitones"]


class TestADiscardedCarrierIsReportedRatherThanVanishing:
    """A branch reports what it found; a carrier it threw away is part of what it found."""

    def test_the_rejection_names_the_gate_and_the_value_it_read(self, tmp_path: Path) -> None:
        """Without this the store cannot separate an absent production from a discarded one."""
        store, _ = seed(tmp_path, continuity=0.1)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        rejected = [finding for finding in result.deviations if finding.name == "carrier_rejected"]
        assert [finding.evidence["value"] for finding in rejected] == ["continuity_min"]
        assert rejected[0].evidence["value_read"] == pytest.approx(0.1, abs=0.01)
        assert rejected[0].evidence["bound"] == MEASURED["continuity_min"]
        assert rejected[0].derived_from

    def test_the_rejection_is_a_measurement_and_not_a_deviation(self, tmp_path: Path) -> None:
        """It is a reading of this branch's instrument, not a departure by the speaker."""
        store, _ = seed(tmp_path, continuity=0.1)
        result = align_voice("maximum-phonation-time", store, None, params(), run_dir=tmp_path)
        rejected = [finding for finding in result.deviations if finding.name == "carrier_rejected"]
        assert {finding.kind for finding in rejected} == {"measure"}
        assert "carrier_rejected" not in deviation_names(result.deviations)

    def test_the_report_carries_the_rejections_for_a_reader_who_opens_no_findings(self, tmp_path: Path) -> None:
        """The summary a corpus reader sees must not have to be rebuilt from the derivatives."""
        store, _ = seed(tmp_path, continuity=0.1)
        result = voice(store, "plain", config(), None, run_dir=tmp_path)
        report = find_branch_report(store, "VOICE")
        assert report is not None
        assert report.attributes["carriers_rejected_n"] >= 1
        assert report.attributes["carriers_rejected"][0]["gate"] == "continuity_min"
        assert result.report.conformance == UNDETERMINED

    def test_the_glide_records_the_sweep_it_located_and_discarded(self, tmp_path: Path) -> None:
        """``sweep_found: False`` about a sweep the branch found states more than it measured."""
        store, _ = seed(
            tmp_path,
            stem=GLIDE_UP_STEM,
            amplitude=((2.0, 18.0),),
            tracks=_zigzag_tracks(20.0, (2.0, 18.0), 100.0, 300.0),
        )
        result = align_voice("glides-low-to-high", store, None, params(), run_dir=tmp_path)
        discarded = [
            finding
            for finding in result.deviations
            if finding.name == "carrier_rejected" and finding.evidence["value"] == "dominant_segment_min_fraction"
        ]
        assert discarded, "the located sweep must be recorded, not silently dropped"
        assert discarded[0].evidence["direction"] in ("up", "down")
        assert discarded[0].evidence["sweep_s"] > 0.0
        assert gated_conformance(result, Pattern.GLIDE, settings=config()) == UNDETERMINED


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
