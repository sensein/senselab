"""AIRWAY proposes breath and cough spans in two modes, and the declaration picks which.

The declared task family selects ``align_airway`` or ``detect_airway``; neither reads a routing
gate. Events are segmented from PREPROCESS's energy envelope and typed by its stored per-span
classifier scores, so a carrier holding five coughs is five spans rather than one label.

The seeder writes the PREPROCESS-shaped provenance surface the branch consumes. Every operating
point in ``_branch_config`` is a fixture value, not a fit: the packaged config ships all 38 numeric
``branch.*`` keys null, and a body that needs one raises with the key named.
"""

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import routing as routing_module
from senselab.audio.workflows.triage.nodes.airway import (
    CARRIER_BOUNDARIES,
    ENVELOPE_BOUNDARIES,
    KIND,
    TASK_EXTENT,
    WINDOW_RUN_BOUNDARIES,
    airway,
    align_airway,
    detect_airway,
)
from senselab.audio.workflows.triage.nodes.branches import (
    NOT_SEPARABLE_BY_THIS_DESIGN,
    UNDETERMINED,
    branch_params,
    mode_of,
)
from senselab.audio.workflows.triage.nodes.common import find_measurements, live_entities, write_verdict
from senselab.audio.workflows.triage.nodes.routing import routing
from senselab.audio.workflows.triage.nodes.verdict import verdict
from senselab.audio.workflows.triage.routing_analysis.ruleset import GateOutcome, RouteEvaluation, RouteState
from senselab.audio.workflows.triage.vocabulary import Outcome, Triage
from senselab.utils.prov_store import Entity, ProvStore
from tests.audio.workflows.triage.nodes.conftest import word_attributes

_ENVELOPE_RATE = 100.0
"""The seeded envelope's sampling rate, in Hz. One sample per 10 ms, as PREPROCESS's own."""

_FLOOR_DBFS = -60.0
"""The seeded global noise floor every rise is measured against."""

_BRANCH_POINTS = """
branch:
  smoothing_window_s: 0.0
  peak_prominence_db: 10.0
  trough_return_db: 6.0
  event_min_s: 0.05
  score_min: 0.5
  breath_coverage_min: 0.5
  gap_off_task_min_s: 1.0
  interval_max_s: 0.5
  effort_split_hz: 1000.0
"""
"""Fixture operating points. Not a fit: the packaged file ships every one of them null."""


def _override(tmp_path: Path, body: str = "", *, name: str = "airway") -> TriageConfig:
    """The packaged configuration with the branch operating points and one partial YAML over it.

    Args:
        tmp_path: Where the override is written.
        body: Extra partial YAML, deep-merged after the operating points.
        name: Distinguishes the override file when a test writes more than one.

    Returns:
        The merged configuration.
    """
    path = tmp_path / f"{name}.yaml"
    path.write_text(_BRANCH_POINTS + body)
    return load_triage_config(path)


@pytest.fixture
def airway_config(tmp_path: Path) -> TriageConfig:
    """The packaged configuration with this branch's operating points supplied.

    Args:
        tmp_path: Where the override is written.

    Returns:
        The merged configuration.
    """
    return _override(tmp_path)


def bump(size: int, centres: Sequence[int], peak_dbfs: float = -10.0, half_width: int = 40) -> np.ndarray:
    """A floor with one triangular rise per centre, which is what an event looks like.

    Args:
        size: Samples.
        centres: Where each rise peaks.
        peak_dbfs: The value at each peak.
        half_width: Samples from a peak back down to the floor.

    Returns:
        The envelope.
    """
    envelope = np.full(size, _FLOOR_DBFS)
    for centre in centres:
        for offset in range(-half_width + 1, half_width):
            envelope[centre + offset] = _FLOOR_DBFS + (peak_dbfs + 60.0) * (1.0 - abs(offset) / half_width)
    return envelope


def plateau(size: int, extent: tuple[int, int], peak_dbfs: float = -10.0) -> np.ndarray:
    """A floor with one digitally flat rise, which is what a clipped or limited event looks like.

    Args:
        size: Samples.
        extent: ``(first, last)`` sample indices of the flat top.
        peak_dbfs: The value across the top.

    Returns:
        The envelope.
    """
    envelope = np.full(size, _FLOOR_DBFS)
    envelope[extent[0] : extent[1]] = peak_dbfs
    return envelope


def _seed(  # noqa: C901 — one independent block per derivative, as PREPROCESS has
    store: ProvStore,
    tmp_path: Path,
    *,
    task: str | None = None,
    spans: Sequence[tuple[float, float]] = (),
    scores: Sequence[dict[str, float] | None] = (),
    labels: Sequence[Sequence[str]] | None = None,
    yamnet_scores: Sequence[dict[str, float] | None] | None = None,
    envelope: np.ndarray | None = None,
    hear_scores: Sequence[tuple[tuple[float, float], dict[str, float]]] | None = None,
    gaps: Sequence[tuple[float, float]] = (),
    foreign_spans: Sequence[tuple[tuple[float, float], str, dict[str, float]]] = (),
    words: Sequence[tuple[str, tuple[float, float]]] = (),
    bracketed_words: Sequence[tuple[str, tuple[float, float]]] = (),
    silence_windows: Sequence[dict[str, Any]] | None = None,
    no_contrast: bool = False,
    merged: int = 1,
    duration_s: float = 5.0,
) -> dict[str, Any]:
    """Write the store surface AIRWAY reads, in the shapes PREPROCESS ships.

    Args:
        store: The store to seed.
        tmp_path: The run directory; sidecars go under ``streams/`` and ``derivatives/``.
        task: The BIDS ``task-`` id written into the ``recording`` stream's path. None writes a
            stem carrying no task at all, which is what takes the out-of-family mode.
        spans: ``[(start, end), ...]`` amplitude spans, in the order ``scores`` indexes.
        scores: One ``span_hear`` ``raw_scores`` mapping per span; a None entry writes no
            ``span_hear`` for that span, which is an absent measurement rather than a zero.
        labels: One decided ``labels`` list per span, for the contest path. None writes none,
            which is the packaged state: ``windows.hear.label_thresholds`` is null.
        yamnet_scores: One ``span_yamnet`` ``raw_scores`` mapping per span, or None for no pass.
        envelope: The energy envelope, in dBFS at :data:`_ENVELOPE_RATE`. None writes none.
        hear_scores: The raw whole-file HeAR windows, for the coverage pattern.
        gaps: ``[(start, end), ...]`` gap spans, which is where off-task material is looked for.
        foreign_spans: ``[(extent, family, raw_scores), ...]`` spans another branch already minted,
            each with its own ``span_hear``. AIRWAY must not read one, whatever it scores.
        words: ``[(text, (start, end)), ...]`` lexical consensus words.
        bracketed_words: The same for bracketed words, indexed after ``words``.
        silence_windows: YAMNet's graded windows, as ``{start, end, score, is_silence}`` dicts.
        no_contrast: Whether PREPROCESS reported ``spans_no_contrast``.
        merged: The ``merged_proposals`` count every seeded amplitude span carries.
        duration_s: The stream's duration.

    Returns:
        The ids of what was written, keyed by kind.
    """
    (tmp_path / "streams").mkdir(exist_ok=True)
    (tmp_path / "derivatives").mkdir(exist_ok=True)
    name = f"plain-{store.run_id}.wav"
    sf.write(str(tmp_path / "streams" / name), np.zeros(int(duration_s * 16000), dtype=np.float32), 16000)
    activity = store.activity(node="PREPROCESS", step="seed", parameters={})
    agent = store.agent(agent_type="software", version="senselab test-seed")
    store.was_associated_with(activity, agent)
    # PREPROCESS shipped this evidence, so the fold must read it as completed, not errored (N26).
    write_verdict(store, activity, agent, node="PREPROCESS", outcome=Outcome.PASS, kind=None, why="seeded", detail={})
    ids: dict[str, Any] = {"spans": [], "gaps": [], "span_hear": [], "span_yamnet": [], "words": [], "bracketed": []}

    def _write(prov_type: str, extent: tuple[float, float] | None, attributes: dict[str, Any]) -> str:
        """One seeded entity, generated by the seed activity and attributed to the seed agent."""
        entity_id = store.entity(prov_type=prov_type, extent=extent, attributes=attributes)  # type: ignore[arg-type]
        store.was_generated_by(entity_id, activity)
        store.was_attributed_to(entity_id, agent)
        return entity_id

    stem = f"sub-a_ses-b_task-{task}" if task is not None else "sub-a_ses-b_recording"
    ids["recording"] = _write(
        "stream",
        (0.0, duration_s),
        {"name": "recording", "path": str(tmp_path / f"{stem}.wav"), "sampling_rate": 16000, "channels": 1},
    )
    ids["plain"] = _write(
        "stream",
        (0.0, duration_s),
        {"name": "plain", "path": f"streams/{name}", "sampling_rate": 16000, "channels": 1},
    )

    for start, end in spans:
        ids["spans"].append(
            _write(
                "span",
                (start, end),
                {"measure": "amplitude", "signal": "preemphasised", "merged_proposals": merged},
            )
        )
    for start, end in gaps:
        ids["gaps"].append(_write("span", (start, end), {"measure": "gap", "signal": "plain"}))
    for extent, family, raw_scores in foreign_spans:
        foreign_id = _write("span", extent, {"family": family, "role": "task_extent"})
        ids.setdefault("foreign", []).append(foreign_id)
        ids["span_hear"].append(
            _write(
                "measurement",
                extent,
                {
                    "name": "span_hear",
                    "classifier": "hear",
                    "signal": "plain",
                    "span_id": foreign_id,
                    "raw_scores": dict(raw_scores),
                    "labelled": False,
                },
            )
        )

    for index, span_id in enumerate(ids["spans"]):
        raw = scores[index] if index < len(scores) else None
        if raw is None:
            continue
        attributes: dict[str, Any] = {
            "name": "span_hear",
            "classifier": "hear",
            "signal": "plain",
            "span_id": span_id,
            "raw_scores": dict(raw),
            "labelled": labels is not None,
            "input_window_s": 2.0,
            "isolated_span": True,
        }
        if labels is not None:
            attributes["labels"] = list(labels[index])
            attributes["scores"] = {label: raw.get(label, 0.0) for label in labels[index]}
        ids["span_hear"].append(_write("measurement", store.get_entity(span_id).extent, attributes))

    for index, span_id in enumerate(ids["spans"]):
        raw = yamnet_scores[index] if yamnet_scores is not None and index < len(yamnet_scores) else None
        if raw is None:
            continue
        ids["span_yamnet"].append(
            _write(
                "measurement",
                store.get_entity(span_id).extent,
                {
                    "name": "span_yamnet",
                    "classifier": "yamnet",
                    "signal": "plain",
                    "span_id": span_id,
                    "raw_scores": dict(raw),
                    "labelled": False,
                    "attribution": "native",
                },
            )
        )

    if envelope is not None:
        np.savez(
            tmp_path / "derivatives" / "energy_envelope.npz",
            envelope_dbfs=np.asarray(envelope, dtype=float),
            floor_dbfs=np.full(np.asarray(envelope).size, _FLOOR_DBFS),
        )
        ids["energy_envelope"] = _write(
            "measurement",
            None,
            {
                "name": "energy_envelope",
                "signal": "plain",
                "path": "derivatives/energy_envelope.npz",
                "sampling_rate": _ENVELOPE_RATE,
            },
        )

    if hear_scores is not None:
        payload = [
            {
                "start": extent[0],
                "end": extent[1],
                "win_length": 2.0,
                "hop_length": 2.0,
                "label_scores": [{label: score} for label, score in window.items()],
            }
            for extent, window in hear_scores
        ]
        (tmp_path / "derivatives" / "hear_scores.json").write_text(json.dumps(payload))
        ids["hear_scores"] = _write(
            "measurement",
            None,
            {
                "name": "hear_scores",
                "classifier": "hear",
                "signal": "plain",
                "path": "derivatives/hear_scores.json",
                "n_windows": len(payload),
                "win_length_s": 2.0,
                "hop_s": 2.0,
            },
        )

    for index, (text, extent) in enumerate(words):
        ids["words"].append(_write("word", extent, word_attributes(text, extent, index=index)))
    for offset, (text, extent) in enumerate(bracketed_words):
        ids["bracketed"].append(_write("word", extent, word_attributes(text, extent, index=len(words) + offset)))

    if no_contrast:
        ids["no_contrast"] = _write(
            "measurement", None, {"name": "spans_no_contrast", "signal": "preemphasised", "reason": "seeded"}
        )
    if silence_windows is not None:
        ids["silence"] = _write(
            "measurement",
            None,
            {"name": "silence", "signal": "plain", "threshold": 0.5, "windows": [dict(w) for w in silence_windows]},
        )
    return ids


def _report_entity(store: ProvStore, node: str) -> Entity:
    """The ``branch_report`` one reporting node wrote.

    Args:
        store: The provenance store.
        node: The node's name.

    Returns:
        Its report entity.
    """
    return next(e for e in live_entities(store, "branch_report") if e.attributes["node"] == node)


def _proposed(store: ProvStore, role: str | None = None) -> list[Entity]:
    """This branch's proposed spans, earliest first.

    Args:
        store: The provenance store.
        role: One role to select, or None for every one.

    Returns:
        The live ``span`` entities carrying this branch's family.
    """
    found = [
        span
        for span in live_entities(store, "span")
        if span.attributes.get("family") == KIND and (role is None or span.attributes.get("role") == role)
    ]
    return sorted(found, key=lambda span: (span.extent or (0.0, 0.0), span.attributes.get("role") or ""))


def _events(store: ProvStore) -> list[Entity]:
    """Every proposed span that is an event rather than the task extent.

    Args:
        store: The provenance store.

    Returns:
        The event spans, earliest first.
    """
    return [span for span in _proposed(store) if span.attributes.get("role") != TASK_EXTENT]


def _counts(store: ProvStore) -> dict[str, Any]:
    """The one folded ``counts`` measurement's entries.

    Args:
        store: The provenance store.

    Returns:
        ``{name: {found, declared}}``, empty when no count was emitted.
    """
    found = find_measurements(store, "counts")
    return dict(found[-1].attributes["entries"]) if found else {}


def _measurement_names(store: ProvStore) -> list[str]:
    """Every measurement name this branch wrote.

    Args:
        store: The provenance store.

    Returns:
        The names, in store order.
    """
    return [
        str(entity.attributes.get("name"))
        for entity in live_entities(store, "measurement")
        if store.generated_by(entity.id) is not None
        and store.get_activity(str(store.generated_by(entity.id))).node == "AIRWAY"
    ]


def _assertions(store: ProvStore, verb: str) -> list[Entity]:
    """Every live assertion carrying one verb.

    Args:
        store: The provenance store.
        verb: ``"deviate"`` or ``"contest"``.

    Returns:
        The assertions, oldest first.
    """
    return [e for e in live_entities(store, "assertion") if e.attributes.get("verb") == verb]


def _five_coughs(store: ProvStore, tmp_path: Path, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    """A ``respiration-and-cough-cough`` recording carrying five resolvable cough events.

    Args:
        store: The store to seed.
        tmp_path: The run directory.
        **extra: Passed through to the seeder.

    Returns:
        What the seeder wrote.
    """
    return _seed(
        store,
        tmp_path,
        task="respiration-and-cough-cough",
        spans=[(0.0, 5.0)],
        scores=[{"Cough": 0.9}],
        envelope=bump(500, (50, 150, 250, 350, 450)),
        **extra,
    )


class TestTheModeComesFromTheDeclaration:
    """The declared family picks the mode; it never supplies the answer."""

    def test_a_declared_airway_family_takes_align(self, store: ProvStore, tmp_path: Path) -> None:
        """``respiration-and-cough-cough`` is a key of the AIRWAY expectation table."""
        _seed(store, tmp_path, task="respiration-and-cough-cough")
        assert mode_of("AIRWAY", store) == ("align", "respiration-and-cough-cough")

    def test_another_branchs_family_takes_detect(self, store: ProvStore, tmp_path: Path) -> None:
        """MPT is VOICE's kind, so AIRWAY annotates its own speciality and evaluates no task."""
        _seed(store, tmp_path, task="maximum-phonation-time")
        assert mode_of("AIRWAY", store) == ("detect", "maximum-phonation-time")

    def test_a_stem_carrying_no_task_takes_detect(self, store: ProvStore, tmp_path: Path) -> None:
        """No carrier names a family, so the out-of-family arm runs — the safe one."""
        _seed(store, tmp_path, task=None)
        assert mode_of("AIRWAY", store) == ("detect", None)

    def test_the_hints_task_token_outranks_the_path(self, store: ProvStore, tmp_path: Path) -> None:
        """The clean carrier wins, which is what makes the ugly one replaceable."""
        _seed(store, tmp_path, task="harvard-sentences-list")
        hint = AudioHints(metadata={"task_token": "respiration-and-cough-breath"})
        assert mode_of("AIRWAY", store, hint) == ("align", "respiration-and-cough-breath")

    def test_the_node_records_which_mode_ran(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A reader must be able to tell an evaluated task from an annotated one."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        recorded = _report_entity(store, "AIRWAY").attributes
        assert (recorded["mode"], recorded["task_family"]) == ("align", "respiration-and-cough-cough")

    def test_align_refuses_a_family_that_is_not_its_own(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The caller owes ``detect_airway`` for anything outside the table."""
        _seed(store, tmp_path, task="maximum-phonation-time")
        with pytest.raises(KeyError):
            align_airway("maximum-phonation-time", store, None, branch_params(airway_config), run_dir=tmp_path)

    def test_detect_evaluates_no_task(self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path) -> None:
        """``done`` is UNDETERMINED as a rule, not as a default."""
        _seed(
            store,
            tmp_path,
            task="harvard-sentences-list",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (150,)),
        )
        result = detect_airway(store, branch_params(airway_config), run_dir=tmp_path)
        assert result.done == UNDETERMINED
        assert result.components


class TestACountedCoughFamilyProposesOneSpanPerEvent:
    """One span per event is the repair: the branch used to count one span holding five coughs as 1."""

    def test_five_events_become_five_spans(self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path) -> None:
        """One carrier span, five envelope maxima, five proposed spans."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        events = _events(store)
        assert len(events) == 5
        assert {span.attributes["label"] for span in events} == {"cough"}
        assert {span.attributes["role"] for span in events} == {"cough_event"}

    def test_the_count_is_compared_against_the_instructions_own(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The declared 5 comes from the ``Expectation`` table, never from a literal here."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _counts(store)["expected_event_count"] == {"found": 5, "declared": 5}

    def test_a_short_count_is_reported_beside_the_declaration_and_asserts_no_discrepancy(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Two coughs against a declared five is a count, not a deviation."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (150, 350)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _counts(store)["expected_event_count"] == {"found": 2, "declared": 5}
        assert [a.attributes["deviation_type"] for a in _assertions(store, "deviate")] == []

    def test_the_verdict_counts_events_not_carriers(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``labelled_n`` was a count of spans carrying a label; it is now a count of events."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        recorded = _report_entity(store, "AIRWAY").attributes
        assert (recorded["labelled_n"], recorded["by_label"]) == (5, {"cough": 5})

    def test_every_event_names_its_carrier_and_the_derivatives_behind_it(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Under propose-only the derivation is the whole record of where the extent came from."""
        ids = _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        for span in _events(store):
            assert set(store.derived_from(span.id)) == {
                ids["spans"][0],
                ids["energy_envelope"],
                ids["span_hear"][0],
            }

    def test_one_task_extent_covers_their_hull(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The part of the recording that serves the task is a span, not a trim payload."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [task] = _proposed(store, TASK_EXTENT)
        events = _events(store)
        assert task.extent == (events[0].extent[0], events[-1].extent[1])  # type: ignore[index]
        assert task.attributes["events_n"] == 5
        assert task.attributes["declared_event_count"] == 5

    def test_each_event_carries_its_own_acoustic_descriptor_with_covariates(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A descriptor over an extent must be read against that extent's own covariates."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        peaks = find_measurements(store, "cough_peak_over_floor_db")
        assert len(peaks) == 5
        assert peaks[0].attributes["value"] == pytest.approx(50.0, abs=1.0)
        assert peaks[0].attributes["uncalibrated"] is True
        assert peaks[0].attributes["contains_clip"] is False

    def test_the_spectral_balance_is_absent_rather_than_nan_without_a_spectrogram(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``spectrogram_wideband`` never reached the store, so the reading is None, not NaN."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [first, *_] = find_measurements(store, "cough_peak_over_floor_db")
        assert first.attributes["spectral_balance_db"] is None

    def test_the_hardcough_family_declares_no_count_and_says_so(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``v2-hardcough`` states no number, and absolute effort has no viable approach."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-v2-hardcough",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (250,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _counts(store)["expected_event_count"] == {"found": 1, "declared": None}
        [effort] = [m for m in find_measurements(store, "effort_absolute")]
        assert effort.attributes["value"] == NOT_SEPARABLE_BY_THIS_DESIGN


class TestAFlatToppedEventTakesItsCarriersExtent:
    """The shared walk reports nothing on a digitally flat rise, and a clipped cough is one.

    The alternative was a counted cough family reading zero events on its loudest recordings.
    ``events_in_span`` lives in the foundation and is not this branch's to change, so a carrier
    that scored the label and resolved no boundary inside it contributes one event at the
    carrier's own extent, marked so a reader can tell it from a resolved one. The residual
    limitation is stated rather than hidden: a carrier holding three clipped coughs still reads
    one, and ``events_with_carrier_boundaries`` is how a run says how often that happened.
    """

    def test_a_plateau_yields_one_event_at_the_carriers_extent(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Not zero, which is what the unaided walk returns."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(1.5, 3.0)],
            scores=[{"Cough": 0.9}],
            envelope=plateau(500, (200, 260)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [event] = _events(store)
        assert event.extent == (1.5, 3.0)
        assert event.attributes["boundaries"] == CARRIER_BOUNDARIES

    def test_the_fallback_is_counted_so_a_run_can_say_how_often_it_fired(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An unresolved boundary is a measured limitation, not a silent success."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(1.5, 3.0)],
            scores=[{"Cough": 0.9}],
            envelope=plateau(500, (200, 260)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _counts(store)["events_with_carrier_boundaries"] == {"found": 1, "declared": None}

    def test_a_resolvable_rise_in_the_same_carrier_is_marked_resolved(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The control: the fallback must not swallow the case the walk does handle."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(1.5, 3.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (230,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [event] = _events(store)
        assert event.attributes["boundaries"] == ENVELOPE_BOUNDARIES
        assert event.extent != (1.5, 3.0)
        assert _counts(store)["events_with_carrier_boundaries"] == {"found": 0, "declared": None}

    def test_a_carrier_that_scored_nothing_contributes_no_fallback_event(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The fallback is for an unresolved boundary, never for an absent label."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(1.5, 3.0)],
            scores=[{"Cough": 0.1}],
            envelope=plateau(500, (200, 260)),
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _events(store) == []
        assert result.report.conformance is False


class TestTheBreathFamiliesCountCyclesAndTimeThem:
    """A count of three says nothing about whether they were quick; the interval is the measurement."""

    def test_the_intervals_between_onsets_are_reported(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``threequickbreaths`` prescribes the intervals, so they are counted beside the events."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-threequickbreaths",
            spans=[(0.0, 5.0)],
            scores=[{"Breathe": 0.9}],
            envelope=bump(500, (50, 250, 450)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        counts = _counts(store)
        assert counts["expected_event_count"] == {"found": 3, "declared": 3}
        assert counts["inter_onset_interval_s"]["found"] == pytest.approx([2.0, 2.0], abs=0.1)
        assert counts["intervals_over_p_interval_max_s"] == {"found": 2, "declared": 0}

    def test_an_untimed_series_reports_no_intervals(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``fivebreaths`` does not prescribe them, so the branch does not invent the expectation."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-fivebreaths",
            spans=[(0.0, 5.0)],
            scores=[{"Breathe": 0.9}],
            envelope=bump(500, (50, 250, 450)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert "inter_onset_interval_s" not in _counts(store)


class TestTheRouteIsNotSeparableByThisDesign:
    """The discriminating band is above the working rate's ceiling and the tilt is confounded."""

    def test_the_measurement_is_recorded_as_unviable_rather_than_omitted(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A measurement the design cannot take is written down, so a negative is attributable."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-v2-threebreathsnose",
            spans=[(0.0, 5.0)],
            scores=[{"Breathe": 0.9}],
            envelope=bump(500, (50, 250, 450)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [measured_route] = find_measurements(store, "measured_route")
        assert measured_route.attributes["value"] == NOT_SEPARABLE_BY_THIS_DESIGN
        assert measured_route.attributes["content_band_hz"] is None, "band_profile does not exist"
        [route] = find_measurements(store, "route")
        assert route.attributes["value"] == NOT_SEPARABLE_BY_THIS_DESIGN

    def test_the_declared_route_comes_from_the_row_where_the_row_states_it(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``v2-threebreathsnose`` names its route; nothing is measured against it."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-v2-threebreathsnose",
            spans=[(0.0, 5.0)],
            scores=[{"Breathe": 0.9}],
            envelope=bump(500, (250,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _counts(store)["declared_route"] == {"found": None, "declared": "nose"}

    def test_the_declared_route_comes_from_the_trailing_index_where_the_family_does_not_carry_it(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``fivebreaths`` splits 1,778/1,778 on an index the family name collapses."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-fivebreaths-2",
            spans=[(0.0, 5.0)],
            scores=[{"Breathe": 0.9}],
            envelope=bump(500, (250,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _counts(store)["declared_route"] == {"found": None, "declared": "mouth"}

    def test_an_index_the_map_does_not_name_reads_as_no_declared_route(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An unrecognised index is an absence, and the row still reports the unviable measurement."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-fivebreaths-9",
            spans=[(0.0, 5.0)],
            scores=[{"Breathe": 0.9}],
            envelope=bump(500, (250,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _counts(store)["declared_route"] == {"found": None, "declared": None}
        assert find_measurements(store, "measured_route")

    def test_a_family_declaring_no_route_reports_none(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``threequickbreaths`` prescribes no route, so the branch says nothing about one."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-threequickbreaths",
            spans=[(0.0, 5.0)],
            scores=[{"Breathe": 0.9}],
            envelope=bump(500, (250,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert "declared_route" not in _counts(store)
        assert find_measurements(store, "measured_route") == []


class TestTheCoverageFamiliesReadRawHearWindows:
    """``respiration-and-cough-breath`` asks for breathing over a declared duration."""

    @staticmethod
    def _windows(scoring: Sequence[int]) -> list[tuple[tuple[float, float], dict[str, float]]]:
        """Five non-overlapping 2 s HeAR windows, the named ones scoring Breathe.

        Args:
            scoring: Which window indices score above the minimum.

        Returns:
            The windows, in the sidecar's own shape.
        """
        return [((index * 2.0, index * 2.0 + 2.0), {"Breathe": 0.9 if index in scoring else 0.1}) for index in range(5)]

    def test_a_merged_run_of_scoring_windows_becomes_one_span(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Touching windows coalesce, so two adjacent windows are one run rather than two."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-breath",
            hear_scores=self._windows((0, 1, 3)),
            duration_s=10.0,
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        runs = _proposed(store, "breath_run")
        assert [span.extent for span in runs] == [(0.0, 4.0), (6.0, 8.0)]

    def test_the_covered_fraction_is_measured_against_the_recording(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Six of ten seconds carry breath evidence."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-breath",
            hear_scores=self._windows((0, 1, 3)),
            duration_s=10.0,
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        [coverage] = find_measurements(store, "breath_coverage_fraction")
        assert coverage.attributes["value"] == pytest.approx(0.6)
        assert coverage.attributes["covered_s"] == pytest.approx(6.0)
        assert result.report.conformance is True

    def test_the_declared_duration_is_counted_beside_the_measured_one(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A sidecar-consistency check: the two disagree, and which is wrong is another question."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-breath",
            hear_scores=self._windows((0,)),
            duration_s=10.0,
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _counts(store)["declared_duration_s"] == {"found": 10.0, "declared": 30.0}

    def test_no_window_scoring_the_label_proposes_nothing(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An absence of detected content, which is what this branch's FAIL means."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-breath",
            hear_scores=self._windows(()),
            duration_s=10.0,
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _proposed(store) == []
        assert result.report.conformance is False


class TestTheAlternationFamilyMatchesBreathBetweenCoughs:
    """``voluntary-cough`` asks for cough-and-breathe cycles, so a cough detector alone fails it."""

    def test_both_kinds_are_proposed_and_the_cycles_counted(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Material between coughs is breath, and it is never scored off task."""
        _seed(
            store,
            tmp_path,
            task="voluntary-cough",
            spans=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
            scores=[{"Cough": 0.9}, {"Breathe": 0.9}, {"Cough": 0.9}],
            envelope=bump(500, (50, 250)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        labels = [span.attributes["label"] for span in _events(store)]
        assert sorted(labels) == ["breath", "cough", "cough"]
        assert _counts(store)["cough_then_breathe_cycles"] == {"found": 1, "declared": 3}

    def test_a_breath_span_names_every_carrier_it_covers(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Two touching breath carriers merge into one extent, and both are in the derivation."""
        ids = _seed(
            store,
            tmp_path,
            task="voluntary-cough",
            spans=[(1.0, 2.0), (2.0, 3.0)],
            scores=[{"Breathe": 0.9}, {"Breathe": 0.9}],
            envelope=bump(500, (50,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [breath] = _proposed(store, "breath_event")
        assert breath.extent == (1.0, 3.0)
        assert set(ids["spans"]) <= set(store.derived_from(breath.id))


class TestDetectAnnotatesWithoutEvaluating:
    """Out of family the branch finds its own speciality and concludes nothing about the task."""

    def test_an_mpt_inhale_is_proposed_as_airway_evidence(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The v1 instruction places the deep inhale before the record tap, so it is in the file.

        It is airway evidence rather than VOICE's phonation, and no lexical or task finding
        attaches to it: MPT is not this branch's task to evaluate.
        """
        _seed(
            store,
            tmp_path,
            task="maximum-phonation-time",
            spans=[(0.2, 1.2), (1.5, 4.5)],
            scores=[{"Breathe": 0.9}, {"Speech": 0.9}],
            envelope=bump(500, (70,)),
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        [inhale] = _events(store)
        assert inhale.attributes["label"] == "breath"
        assert inhale.attributes["evaluates_no_task"] is True
        assert inhale.extent is not None and 0.2 <= inhale.extent[0] < 1.2
        assert result.report.conformance == UNDETERMINED
        assert _report_entity(store, "AIRWAY").attributes["conformance"] == UNDETERMINED

    def test_a_cough_inside_a_sentence_reading_carries_no_penalty_for_the_words(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A cough in a Harvard reading is a real cough; whether it bears on the sentence is VERDICT's."""
        _seed(
            store,
            tmp_path,
            task="harvard-sentences-list",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (250,)),
            words=[("hello", (0.5, 0.9)), ("there", (1.0, 1.4))],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert len(_events(store)) == 1
        assert _assertions(store, "deviate") == []

    def test_detect_proposes_no_task_extent(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """There is no task here to have an extent."""
        _seed(
            store,
            tmp_path,
            task="harvard-sentences-list",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (250,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _proposed(store, TASK_EXTENT) == []

    def test_both_kinds_are_counted_per_label(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Every label set is looked for, whatever the recording was declared to be."""
        _seed(
            store,
            tmp_path,
            task="word-color-stroop",
            spans=[(0.0, 2.0), (3.0, 5.0)],
            scores=[{"Cough": 0.9}, {"Breathe": 0.9}],
            envelope=bump(500, (100, 400)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        counts = _counts(store)
        assert counts["cough_events"] == {"found": 1, "declared": None}
        assert counts["breath_events"] == {"found": 1, "declared": None}
        assert counts["airway_events"] == {"found": 2, "declared": None}

    def test_a_decided_label_with_no_raw_score_over_the_minimum_is_contested(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A label was proposed here and the measurement behind it does not carry it."""
        ids = _seed(
            store,
            tmp_path,
            task="harvard-sentences-list",
            spans=[(1.0, 2.0)],
            scores=[{"Cough": 0.1}],
            labels=[["Cough"]],
            envelope=bump(500, (150,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [contested] = _assertions(store, "contest")
        assert contested.attributes["claim"] == "cough"
        assert store.derived_from(contested.id) == [ids["spans"][0]]
        assert contested.attributes["reason"] == "no_raw_score_over_p_score_min"
        assert _report_entity(store, "AIRWAY").attributes["contested_n"] == 1

    def test_a_decided_label_the_branch_did_propose_over_is_not_contested(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The control: a contest is about a span the branch could not confirm, not every span."""
        _seed(
            store,
            tmp_path,
            task="harvard-sentences-list",
            spans=[(1.0, 2.0)],
            scores=[{"Cough": 0.9}],
            labels=[["Cough"]],
            envelope=bump(500, (150,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _assertions(store, "contest") == []


class TestBothClassifiersAreOneEvidenceSet:
    """``sounds_like`` reads ``raw_scores`` over ``span_hear`` and ``span_yamnet`` together."""

    def test_a_yamnet_score_alone_is_enough(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Corroboration is not a separate assertion any more: the presence test reads both."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 5.0)],
            scores=[None],
            yamnet_scores=[{"Cough": 0.9}],
            envelope=bump(500, (250,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert len(_events(store)) == 1

    def test_a_score_under_the_minimum_from_both_is_not_evidence(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The cut is one value across two classifiers whose scales were never compared."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.4}],
            yamnet_scores=[{"Cough": 0.4}],
            envelope=bump(500, (250,)),
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _events(store) == []
        assert result.report.conformance is False

    def test_an_absent_span_hear_pass_is_an_absence_not_a_zero(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A span PREPROCESS's per-span pass never covered has no evidence to read."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 5.0)],
            scores=[None],
            envelope=bump(500, (250,)),
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _events(store) == []
        assert result.report.conformance is False


class TestLexicalIntrusionIsALocatedDeviationConditionedOnTheDeclaration:
    """A word inside an airway task is off-task content that has its own extent."""

    def test_each_intruding_word_is_one_deviation_naming_no_text(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Transcript text never enters this branch's store writes; REDACT is the only release path."""
        ids = _five_coughs(store, tmp_path, words=[("Marisol", (1.8, 1.9))], bracketed_words=[("[COUGH]", (2.0, 2.1))])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        intrusions = [a for a in _assertions(store, "deviate") if a.attributes.get("reading") == "lexical_intrusion"]
        assert len(intrusions) == 1, "a bracketed word is what this branch looks for, not a transcript"
        assert intrusions[0].extent == (1.8, 1.9)
        assert store.derived_from(intrusions[0].id) == [ids["words"][0]]
        assert "Marisol" not in json.dumps(intrusions[0].attributes)

    def test_an_invalidated_word_is_not_an_intrusion(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A withdrawn word is not evidence, on this read as on every other."""
        ids = _five_coughs(store, tmp_path, words=[("Marisol", (1.8, 1.9))])
        store.was_invalidated_by(ids["words"][0], store.activity(node="PREPROCESS", step="withdraw", parameters={}))
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _assertions(store, "deviate") == []

    def test_out_of_family_no_word_is_off_task(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The check is conditioned on the declaration by the mode dispatch, not by a guard."""
        _seed(
            store,
            tmp_path,
            task="rainbow-passage",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (250,)),
            words=[("Marisol", (1.8, 1.9))],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _assertions(store, "deviate") == []

    def test_a_word_overlapping_an_event_is_carried_as_a_covariate_not_an_exclusion(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A cough overlapped by a word is still a cough; the overlap is recorded on the span."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (250,)),
            words=[("hello", (2.4, 2.6))],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [event] = _events(store)
        assert event.attributes["overlaps_transcript"] is True


class TestWhatTheRestructuringKeptFromTheOldBranch:
    """Behaviour that survives: the silence covariate, the merge rate, and the store's own rules."""

    def test_a_span_inside_all_silent_windows_reads_true(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The covariate moves from the label assertion onto the proposed span, unchanged."""
        _five_coughs(store, tmp_path, silence_windows=[{"start": 0.0, "end": 5.0, "score": 0.8, "is_silence": True}])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert all(span.attributes["in_certified_silence"] is True for span in _events(store))

    def test_a_span_over_mixed_windows_reads_false(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """One overlapping window graded not-silent is enough."""
        _five_coughs(
            store,
            tmp_path,
            silence_windows=[
                {"start": 0.0, "end": 0.4, "score": 0.8, "is_silence": True},
                {"start": 0.4, "end": 5.0, "score": 0.2, "is_silence": False},
            ],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _events(store)[0].attributes["in_certified_silence"] is False

    def test_no_graded_window_at_all_reads_none(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An unavailable grading is an absence, never a negative."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _events(store)[0].attributes["in_certified_silence"] is None

    def test_an_invalidated_span_is_not_a_candidate(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """This node's reads follow the store's shared rule."""
        ids = _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 2.0), (3.0, 5.0)],
            scores=[{"Cough": 0.9}, {"Cough": 0.9}],
            envelope=bump(500, (100, 400)),
        )
        store.was_invalidated_by(ids["spans"][1], store.activity(node="PREPROCESS", step="withdraw", parameters={}))
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert len(_events(store)) == 1

    def test_the_merge_rate_is_reported_over_the_carriers_the_events_came_from(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A carrier covering several proposals must stay legible as one."""
        _five_coughs(store, tmp_path, merged=3)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _report_entity(store, "AIRWAY").attributes["merged_n"] == 3, "one carrier, five events"

    def test_no_span_at_all_reports_non_conformance(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A branch reports; it does not restate PREPROCESS's own reason as an explanation."""
        _seed(store, tmp_path, task="respiration-and-cough-cough", no_contrast=True, envelope=bump(500, (250,)))
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.report.conformance is False
        assert _proposed(store) == []

    def test_a_hint_changes_nothing_about_what_was_found(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A declaration does not supply an absence, and it does not supply a presence either."""
        _seed(store, tmp_path, task="respiration-and-cough-cough", no_contrast=True, envelope=bump(500, (250,)))
        result = airway(store, "plain", airway_config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.report.conformance is False
        assert result.report.deviations == ()

    def test_the_branch_writes_no_flag_verb_at_all(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``lexical_contamination`` was the only reachable flag and it is now a deviation."""
        _five_coughs(store, tmp_path, words=[("Marisol", (1.8, 1.9))])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.report.conformance is True
        assert result.report.deviations == ("off_task_extent",)
        assert _assertions(store, "flag") == []


class TestTheOtherFindingsEachModeOwes:
    """Truncation, the declared relax period, and off-task gaps."""

    def test_a_task_extent_reaching_the_recordings_edge_is_a_truncation(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """An event at the first or last sample may have been cut, and that is a deviation.

        The fixture reaches the edge through the carrier fallback, and it is the only way one can.
        An envelope-resolved event cannot start at sample 0: the walk demands prominence over a
        trough on its left, and a rise whose whole left flank stays above the trough-return target
        has no such trough. That is the flat-plateau property from the other side.
        """
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 2.0)],
            scores=[{"Cough": 0.9}],
            envelope=plateau(500, (0, 200)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert [a.attributes["deviation_type"] for a in _assertions(store, "deviate")] == ["truncation"]

    def test_a_task_extent_clear_of_both_edges_is_not_a_truncation(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The control: five resolved coughs inside the recording report nothing."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _assertions(store, "deviate") == []

    def test_the_one_family_whose_instruction_prescribes_material_that_is_not_the_task(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``breath-sounds`` gives its first 60 s to settling, and only a long enough file has it."""
        _seed(
            store,
            tmp_path,
            task="breath-sounds",
            spans=[(65.0, 70.0)],
            scores=[{"Breathe": 0.9}],
            envelope=np.concatenate([np.full(6500, _FLOOR_DBFS), bump(500, (250,)), np.full(300, _FLOOR_DBFS)]),
            duration_s=73.0,
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        relax = [a for a in _assertions(store, "deviate") if a.attributes.get("reading") == "declared_relax_period"]
        assert len(relax) == 1
        assert relax[0].extent == (0.0, 60.0)

    def test_a_short_recording_of_that_family_reports_no_relax_period(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The measured median is 13.2 s against ~73 s, so the deviation must not fire on every file."""
        _seed(
            store,
            tmp_path,
            task="breath-sounds",
            spans=[(0.0, 5.0)],
            scores=[{"Breathe": 0.9}],
            envelope=bump(500, (250,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        deviations = _assertions(store, "deviate")
        assert [a for a in deviations if a.attributes.get("reading") == "declared_relax_period"] == []

    def test_a_gap_no_proposed_span_covers_is_off_task_extent(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Off-task material is ground the branch disclaims, so it is a deviation and never a span."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 1.0)],
            scores=[{"Cough": 0.9}],
            gaps=[(2.0, 4.5)],
            envelope=bump(500, (50,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        off_task = [a for a in _assertions(store, "deviate") if a.attributes.get("measure") == "gap"]
        assert [a.extent for a in off_task] == [(2.0, 4.5)]
        assert _proposed(store, "gap") == []

    def test_a_gap_shorter_than_the_minimum_is_not_reported(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The minimum is what keeps an inter-event pause out of the off-task set."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 1.0)],
            scores=[{"Cough": 0.9}],
            gaps=[(2.0, 2.5)],
            envelope=bump(500, (50,)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert [a for a in _assertions(store, "deviate") if a.attributes.get("measure") == "gap"] == []


class TestAnAbsentInstrumentIsAnAbsence:
    """AIRWAY's gate evidence is unavailable on 56,505 of 62,547 recordings; so may a derivative be."""

    def test_no_energy_envelope_leaves_the_task_undetermined(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The event walk has no input, so nothing was concluded rather than nothing found."""
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(0.0, 5.0)],
            scores=[{"Cough": 0.9}],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED
        recorded = _report_entity(store, "AIRWAY").attributes
        assert recorded["conformance"] == UNDETERMINED
        assert "energy_envelope" in " ".join(recorded["notes"])
        [absence] = find_measurements(store, "event_instrument")
        assert absence.attributes["absent"] == "energy_envelope"

    def test_no_hear_scores_leaves_a_coverage_family_undetermined(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The coverage pattern's only instrument is the raw HeAR grid."""
        _seed(store, tmp_path, task="respiration-and-cough-breath")
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED
        notes = _report_entity(store, "AIRWAY").attributes["notes"]
        assert "hear_scores" in " ".join(notes)

    def test_a_moved_run_directory_is_not_an_envelope_of_silence(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The measurement is there and its sidecar is not, which is the same absence."""
        _five_coughs(store, tmp_path)
        (tmp_path / "derivatives" / "energy_envelope.npz").unlink()
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.report.conformance == UNDETERMINED
        notes = _report_entity(store, "AIRWAY").attributes["notes"]
        assert "energy_envelope" in " ".join(notes)

    def test_an_unmeasured_score_min_is_reported_and_proposes_no_event(self, store: ProvStore, tmp_path: Path) -> None:
        """A branch never refuses: the qualifier is skipped, the key is named, no event is proposed."""
        _five_coughs(store, tmp_path)
        config = _override(tmp_path, "branch:\n  score_min: null\n", name="null-score-min")
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert _events(store) == []
        assert result.report.unmeasured == ("score_min",)
        # UNDETERMINED, not False: an unmeasured qualifier could neither admit nor reject, so
        # claiming the instruction was not met would rest on a number nobody chose.
        assert result.report.conformance == UNDETERMINED


class TestTheProposeOnlyRules:
    """A branch mints only into its own family, and every proposal names its evidence."""

    def test_every_proposed_span_is_in_this_branchs_family(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """``dispatch`` refuses a proposal outside it, so this pins what the bodies actually mint."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert {span.attributes["family"] for span in _proposed(store)} == {KIND}

    def test_every_proposed_span_names_at_least_one_source(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Under propose-only the derivation is the whole record of where the extent came from."""
        _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert all(store.derived_from(span.id) for span in _proposed(store))

    def test_another_branchs_span_is_not_this_branchs_evidence(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The selector is ``family is None or family == "airway"``, and the widening matters.

        A branch that read every family would count a span SPEECH or VOICE already minted over the
        same ground as a second event, and would do it silently. The out-of-family mode is where
        this bites: ``align`` reaches its carriers through ``amplitude_spans``, which a
        branch-minted span does not pass, and ``detect`` reads the selector directly.
        """
        _seed(
            store,
            tmp_path,
            task="harvard-sentences-list",
            spans=[(0.0, 2.0)],
            scores=[{"Cough": 0.9}],
            foreign_spans=[((3.0, 4.5), "speech", {"Cough": 0.9})],
            envelope=bump(500, (100, 380)),
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        events = _events(store)
        assert len(events) == 1
        assert events[0].extent is not None and events[0].extent[1] <= 2.0

    def test_a_reading_the_envelope_cannot_take_is_absent_rather_than_nan(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """NaN survives ``json.dumps`` as a token no other reader parses; None is the absence.

        The carrier here lies past the end of the envelope derivative, which is what a truncated
        or re-conditioned sidecar looks like from inside the branch.
        """
        _seed(
            store,
            tmp_path,
            task="respiration-and-cough-cough",
            spans=[(6.0, 8.0)],
            scores=[{"Cough": 0.9}],
            envelope=bump(500, (250,)),
            duration_s=10.0,
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [peak] = find_measurements(store, "cough_peak_over_floor_db")
        assert peak.attributes["value"] is None
        assert "NaN" not in json.dumps(peak.attributes)

    def test_preprocesss_own_spans_are_never_edited(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The store is append-only and nothing under ``nodes/`` invalidates a span."""
        ids = _five_coughs(store, tmp_path)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert not store.is_invalidated(ids["spans"][0])

    def test_the_verdict_is_generated_by_the_one_activity_the_branch_ran(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Three steps became one, and its ``step`` is the mode that ran."""
        _five_coughs(store, tmp_path)
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        concluding = store.generated_by(result.report_entity_id)
        assert concluding is not None
        assert store.get_activity(concluding).step == "align"

    def test_the_view_reaches_every_span_the_branch_proposed(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """REPORT reads the view, so a proposal outside it is invisible."""
        _five_coughs(store, tmp_path)
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert {span.id for span in _proposed(store)} <= set(result.view)


class TestTheFoldNamesTheHintMismatchThisBranchDoesNot:
    """The end-to-end pin: a declared branch that found nothing reaches the file verdict as a mismatch."""

    @staticmethod
    def _empty_reading(monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the ruleset read the recording as empty: nothing routed, every tracked peak under floor."""
        reading = RouteEvaluation(
            stem="rec",
            family="",
            routed=(),
            declared=(),
            agreed=(),
            missed=(),
            extra=(),
            unavailable={},
            flags={},
            state=RouteState.EMPTY,
            gate_outcomes={"airway.cough": GateOutcome.SILENT},
        )
        monkeypatch.setattr(routing_module, "evaluate_live_routes", lambda *a, **k: reading)

    def test_a_declared_branch_that_found_nothing_flags_the_file(
        self, store: ProvStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AIRWAY fails, ROUTING recorded the claim, and the fold flags without resolving it present."""
        hint_config = _override(tmp_path, "routing:\n  hint_branch_map:\n    cough: AIRWAY\n")
        hint = AudioHints(may_contain=["cough"])
        _seed(store, tmp_path, task="respiration-and-cough-cough", no_contrast=True, envelope=bump(500, (250,)))
        self._empty_reading(monkeypatch)
        routing(store, "plain", hint_config, hint, run_dir=tmp_path)
        branch = airway(store, "plain", hint_config, hint, run_dir=tmp_path)
        assert branch.report.conformance is False

        folded = verdict(store, None, hint_config, hint, run_dir=tmp_path).file_verdict
        assert folded.triage is Triage.FLAG
        assert folded.findings["AIRWAY"] == "absent"
        assert folded.hints["AIRWAY"] == "claimed_not_found"
        assert folded.discard_ground is None
        assert any(
            reason.why == "hint mismatch: AIRWAY was declared and did not find it" for reason in folded.reasons
        ), [reason.why for reason in folded.reasons]

    def test_the_same_file_with_no_declaration_discards_as_acoustically_empty(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: nothing routed, nothing claimed, and the bypass called the recording empty.

        Unlike the declared case above, AIRWAY is not run here: ROUTING declined it (nothing routed
        and nothing declared it), so ``run.py`` would skip it, and a branch that never reports
        cannot supply the ``conformance is False`` flag ground that a reported non-conformance now
        is. Running it anyway would flag the file on that ground alone, whatever the route state.
        """
        _seed(store, tmp_path, task="respiration-and-cough-cough", no_contrast=True, envelope=bump(500, (250,)))
        self._empty_reading(monkeypatch)
        routing(store, "plain", airway_config, None, run_dir=tmp_path)

        folded = verdict(store, None, airway_config, None, run_dir=tmp_path).file_verdict
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"


def test_the_branch_writes_no_measurement_name_twice_over_one_extent(
    store: ProvStore, airway_config: TriageConfig, tmp_path: Path
) -> None:
    """A count name is written once.

    ``write_findings`` folds every count into one ``counts`` entry keyed by name, so a repeated name
    would overwrite silently. Named here because the design's own draft emitted one count per
    carrier span under a single name.
    """
    _seed(
        store,
        tmp_path,
        task="word-color-stroop",
        spans=[(0.0, 2.0), (3.0, 5.0)],
        scores=[{"Cough": 0.9}, {"Cough": 0.9}],
        envelope=bump(500, (100, 400)),
    )
    airway(store, "plain", airway_config, run_dir=tmp_path)
    names = [name for name in _measurement_names(store) if name != "counts"]
    assert len(names) == len(set(names)) or all(name.endswith("_peak_over_floor_db") for name in names), (
        "only per-event descriptors may repeat a name, and they differ by extent"
    )
    assert _counts(store)["cough_events"] == {"found": 2, "declared": None}
