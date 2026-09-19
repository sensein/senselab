"""The summary's branch lanes — each proposal paired to what it was derived from.

Every span here is minted through ``propose_spans`` and every report through ``write_report``, the
only writers of either, so a name this suite reads is a name a branch actually produces. A
hand-built entity would let a reader key itself to a shape nobody writes and still pass, which is
the defect class this product was built to avoid.
"""

import json
import re
from pathlib import Path
from typing import Any, Callable

import pytest

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import PROPOSERS, propose_spans
from senselab.audio.workflows.triage.nodes.common import live_entities, software_agent, write_report
from senselab.audio.workflows.triage.nodes.figure import (
    BRANCH_INITIAL_ROW,
    BRANCH_PROPOSED_ROW,
    LANE_NO_REPORT,
    LANE_RAN,
    LANE_UNDECIDED,
    LANE_WITHHELD,
    SPAN_AXIS_TITLE,
    SUMMARY_LANES,
    BranchLane,
    BranchRow,
    FigureStyle,
    branch_lanes,
    branch_report_lines,
    initial_row_note,
    initial_rows,
    lane_note,
    lane_roles,
    parent_anchor,
    preprocess_figure,
    rows_on_page,
    span_axis_rows,
)
from senselab.audio.workflows.triage.nodes.routing import routing
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.audio.workflows.triage.vocabulary import BRANCHES, TASK, Conformance
from senselab.utils.prov_store import Entity, ProvStore


def _pdf_page_count(path: Path) -> int:
    """How many pages a PDF holds, read from its own page objects rather than by rendering."""
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


def _envelope_spans(store: ProvStore) -> list[Entity]:
    """PREPROCESS's own amplitude spans, which are what a branch proposal is derived from."""
    spans = [span for span in live_entities(store, "span") if "peak_over_floor_db" in span.attributes]
    return sorted(spans, key=lambda span: span.extent or (0.0, 0.0))


def _run_branch(
    store: ProvStore,
    branch: str,
    proposals: list[tuple[str, tuple[float, float], Any, dict[str, Any]]],
    *,
    kind: str,
    detail: dict[str, Any],
    conformance: Conformance = True,
    deviations: tuple[str, ...] = (),
    unmeasured: tuple[str, ...] = (),
) -> None:
    """Mint one branch's spans and write its report, through the only writers of either.

    A proposal's third element is the entity id it derives from, or a tuple of them for a proposal
    naming more than one parent.
    """
    activity = store.activity(node=branch, step="propose", parameters={})
    agent = software_agent(store)
    store.was_associated_with(activity, agent)
    mint = PROPOSERS[branch]
    propose_spans(
        store,
        activity,
        agent,
        [
            mint(role, extent, *((sources,) if isinstance(sources, str) else tuple(sources)), **attributes)
            for role, extent, sources, attributes in proposals
        ],
    )
    write_report(
        store,
        activity,
        agent,
        node=branch,
        kind=kind,
        conformance=conformance,
        conformance_of=TASK,
        deviations=deviations,
        unmeasured=unmeasured,
        detail=detail,
    )


@pytest.fixture
def routed(
    store: ProvStore,
    config: TriageConfig,
    seed_preprocess_store: Callable[..., None],
    tmp_path: Path,
) -> ProvStore:
    """A store carrying PREPROCESS's surface and ROUTING's real per-branch decisions."""
    seed_preprocess_store(
        store,
        duration_s=8.0,
        yamnet_labels=[["Speech"]],
        scores_only=("yamnet",),
        spans=[(1.0, 2.0, 20.0), (4.0, 5.0, 18.0)],
        words=["one", "two"],
    )
    taxonomy(store, "plain", config, run_dir=tmp_path)
    routing(store, None, config, run_dir=tmp_path)
    return store


class TestTheLaneReadsWhatTheBranchesWrote:
    """Family, role and derivation are stamped by ``propose_span`` and are always there."""

    def test_there_is_one_lane_per_declared_lane_in_order(self, routed: ProvStore) -> None:
        """A lane is never dropped for having produced nothing; it says so instead."""
        assert [lane.branch for lane in branch_lanes(routed)] == list(SUMMARY_LANES)

    def test_redact_is_a_lane_even_though_it_is_not_a_routed_branch(self, routed: ProvStore) -> None:
        """Its plan is a set of timed spans, so it draws rather than being prose to align by eye."""
        assert SUMMARY_LANES[-1] == "REDACT"
        assert [lane.branch for lane in branch_lanes(routed)][: len(BRANCHES)] == list(BRANCHES)

    def test_each_lane_draws_its_own_family_and_no_other(self, routed: ProvStore) -> None:
        """The families are disjoint, so a span may not appear under two branches."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        _run_branch(
            routed,
            "VOICE",
            [("task_extent", (4.1, 4.9), source, {"production": "sustained"})],
            kind="voice",
            detail={"spans_n": 1, "phonation_s": 0.8, "longest_span_s": 0.8, "notes": []},
        )
        lanes = {lane.branch: lane for lane in branch_lanes(routed)}
        assert [row.label for row in lanes["AIRWAY"].proposed] == ["cough_event/cough"]
        assert [row.label for row in lanes["VOICE"].proposed] == ["task_extent/sustained"]

    def test_a_proposal_is_captioned_by_the_role_and_its_own_qualifier(self, routed: ProvStore) -> None:
        """``role`` is stamped on every span; the qualifier is whichever one the proposal carries."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "SPEECH",
            [
                ("speech_run_0", (1.1, 1.4), source, {"attributed_to": "SPEAKER_00", "nontarget": False}),
                ("speech_run_1", (1.5, 1.9), source, {"attributed_to": "SPEAKER_01", "nontarget": True}),
                ("breath_group_0", (4.1, 4.4), source, {"group_index": 0}),
            ],
            kind="speech",
            detail={"speaker_count": 2, "words_n": 2, "speech_s": 0.7, "nontarget_speech_s": 0.4, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "SPEECH"]
        assert [row.label for row in lane.proposed] == [
            "speech_run_0/SPEAKER_00",
            "speech_run_1/SPEAKER_01 nontarget",
            "breath_group_0",
        ]

    def test_a_narrow_bar_falls_back_to_the_qualifier_rather_than_to_nothing(self, routed: ProvStore) -> None:
        """A bar is never dropped, so one too narrow for its role says what distinguishes it."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.2), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        [proposed] = lane.proposed
        assert (proposed.label, proposed.short) == ("cough_event/cough", "cough")

    def test_an_initial_row_has_no_shorter_form_to_fall_back_to(self, routed: ProvStore) -> None:
        """Its reading is already one term; abbreviating a level in dB would change the number."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        [initial] = lane.initial
        assert (initial.label, initial.short) == ("envelope 20 dB", "envelope 20 dB")

    def test_only_the_measures_the_report_carries_are_read(self, routed: ProvStore) -> None:
        """SPEECH's syllable keys are written only on an in-family align run; absent is not None."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "SPEECH",
            [("speech_run_0", (1.1, 1.4), source, {"attributed_to": None, "nontarget": None})],
            kind="speech",
            detail={"speaker_count": 1, "words_n": 2, "speech_s": 0.3, "nontarget_speech_s": None, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "SPEECH"]
        assert dict(lane.measures) == {
            "speaker_count": 1,
            "words_n": 2,
            "speech_s": 0.3,
            "nontarget_speech_s": None,
        }
        assert "ppg_rate_hz" not in dict(lane.measures)

    def test_the_nine_ppg_measures_reach_the_lane_when_the_report_carries_them(self, routed: ProvStore) -> None:
        """The CV instrument reports these under SPEECH; the lane must not drop one."""
        source = _envelope_spans(routed)[0].id
        ppg = {
            "ppg_trains_n": 3,
            "ppg_rate_hz": 5.1,
            "ppg_repetitions": 17,
            "ppg_period_s": 0.196,
            "ppg_jitter_over_median": 0.08,
            "ppg_cv_units_n": 17,
            "ppg_interval_trend_s_per_step": 0.001,
            "ppg_expected_place_fraction": 0.94,
            "ppg_place_agreement": 0.88,
        }
        _run_branch(
            routed,
            "SPEECH",
            [("ppg_train", (1.1, 1.9), source, {"production": "syllable_train_from_ppg"})],
            kind="speech",
            detail={"speaker_count": 1, "words_n": 2, "speech_s": 0.8, "nontarget_speech_s": 0.0, **ppg, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "SPEECH"]
        assert {key: value for key, value in lane.measures if key.startswith("ppg_")} == ppg


class TestABranchThatDidNotRunIsNotABranchThatFoundNothing:
    """Four states, and the figure may not blur any pair of them."""

    def test_a_withheld_branch_names_the_route_that_withheld_it(self, routed: ProvStore) -> None:
        """ROUTING declined VOICE on this recording, so no node was ever called."""
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "VOICE"]
        assert lane.state == LANE_WITHHELD
        assert lane_note(lane, 0) == "VOICE did not run — route declined: route_declined"

    def test_a_selected_branch_that_wrote_no_report_is_its_own_state(self, routed: ProvStore) -> None:
        """ROUTING routed SPEECH and nothing followed: neither withheld nor having found nothing."""
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "SPEECH"]
        assert lane.state == LANE_NO_REPORT
        assert lane_note(lane, 0) == "SPEECH was selected to run and wrote no report"

    def test_a_branch_that_ran_and_proposed_nothing_says_exactly_that(self, routed: ProvStore) -> None:
        """The distinction the whole graph turns on: it looked, and there was nothing."""
        _run_branch(
            routed,
            "AIRWAY",
            [],
            kind="airway",
            detail={"labelled_n": 0, "contested_n": 0, "merged_n": 0, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        assert lane.state == LANE_RAN
        assert lane_note(lane, 0) == "AIRWAY ran and proposed no airway span"

    def test_a_store_with_no_routing_decision_says_so_rather_than_guessing(
        self, store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Without ROUTING the store cannot say whether a branch was meant to run."""
        seed_preprocess_store(store, duration_s=4.0, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        lanes = branch_lanes(store)
        assert {lane.state for lane in lanes if lane.branch in BRANCHES} == {LANE_UNDECIDED}
        assert lane_note(lanes[0], 0) == "ROUTING wrote no decision for AIRWAY"

    def test_a_lane_with_spans_elsewhere_in_the_file_says_none_is_on_this_page(self, routed: ProvStore) -> None:
        """An empty page of a branch that proposed plenty is not a branch that proposed none."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        assert lane_note(lane, 0) == "AIRWAY proposed 1 airway span, none on this page"

    def test_a_lane_with_bars_on_the_page_prints_no_note(self, routed: ProvStore) -> None:
        """The control: a note is what stands in for bars, never what accompanies them."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        assert lane_note(lane, len(rows_on_page(lane, (0.0, 20.0)))) == ""


class TestTheSpanAxisIsOneAxis:
    """One initial row over one row per branch, rather than four lanes each repeating the parents."""

    def test_the_rows_are_the_initial_population_then_one_per_lane(self, routed: ProvStore) -> None:
        """The layout the owner asked for, read off the structure rather than the pixels."""
        lanes = branch_lanes(routed)
        assert [row.block for row in span_axis_rows(lanes)] == [BRANCH_INITIAL_ROW, *SUMMARY_LANES]

    def test_an_initial_span_shared_by_two_branches_is_one_bar(self, routed: ProvStore) -> None:
        """The case the four-lane layout could not show: one parent, proposals in two branches."""
        first, _second = _envelope_spans(routed)
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.4), first.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        _run_branch(
            routed,
            "VOICE",
            [("task_extent", (1.5, 1.9), first.id, {"production": "sustained"})],
            kind="voice",
            detail={"spans_n": 1, "phonation_s": 0.4, "longest_span_s": 0.4, "notes": []},
        )
        lanes = branch_lanes(routed)
        assert [row.key for row in initial_rows(lanes)] == [first.id]
        parents = {lane.branch: {p for row in lane.proposed for p in row.derived_from} for lane in lanes}
        assert parents["AIRWAY"] == parents["VOICE"] == {first.id}

    def test_two_branches_off_one_parent_keep_their_own_edges(self, routed: ProvStore) -> None:
        """One bar, two connectors: the shared row must not merge the two derivations into one."""
        first, _second = _envelope_spans(routed)
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.4), first.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        _run_branch(
            routed,
            "SPEECH",
            [("speech_run_0", (1.5, 1.9), first.id, {"attributed_to": "SPEAKER_00", "nontarget": False})],
            kind="speech",
            detail={"speaker_count": 1, "words_n": 2, "speech_s": 0.4, "nontarget_speech_s": 0.0, "notes": []},
        )
        lanes = {lane.branch: lane for lane in branch_lanes(routed)}
        assert lanes["AIRWAY"].proposed[0].derived_from == (first.id,)
        assert lanes["SPEECH"].proposed[0].derived_from == (first.id,)

    def test_the_initial_row_is_deduplicated_across_lanes(self, routed: ProvStore) -> None:
        """Two lanes naming the same two parents give two bars, not four."""
        first, second = _envelope_spans(routed)
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.4), (first.id, second.id), {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        _run_branch(
            routed,
            "VOICE",
            [("task_extent", (4.1, 4.4), (first.id, second.id), {"production": "sustained"})],
            kind="voice",
            detail={"spans_n": 1, "phonation_s": 0.3, "longest_span_s": 0.3, "notes": []},
        )
        lanes = branch_lanes(routed)
        assert sorted(row.key for row in initial_rows(lanes)) == sorted((first.id, second.id))

    def test_the_initial_row_is_empty_when_nothing_names_a_span(self, routed: ProvStore) -> None:
        """A proposal derived only from a measurement adds no bar to the shared row."""
        measurement = next(entity for entity in live_entities(routed, "measurement") if entity.attributes.get("name"))
        _run_branch(
            routed,
            "AIRWAY",
            [("breath_run", (1.1, 1.8), measurement.id, {"label": "breath"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        assert initial_rows(branch_lanes(routed)) == ()

    def test_each_lane_leaves_a_shared_parent_at_its_own_point(self) -> None:
        """Collinear connectors overlap, and the last drawn hides the rest.

        A parent feeding three branches would then look like a parent feeding one, which is the
        whole claim this layout exists to make.
        """
        anchors = [parent_anchor(1.0, 2.0, index, len(SUMMARY_LANES)) for index in range(len(SUMMARY_LANES))]
        assert len(set(anchors)) == len(SUMMARY_LANES)
        assert anchors == sorted(anchors)

    def test_a_departure_point_stays_inside_the_parent_bar(self) -> None:
        """A connector leaving outside the bar would not read as leaving that bar."""
        for index in range(len(SUMMARY_LANES)):
            anchor = parent_anchor(3.0, 3.4, index, len(SUMMARY_LANES))
            assert 3.0 < anchor < 3.4

    def test_the_fan_is_symmetric_about_the_parent_s_centre(self) -> None:
        """No lane is privileged with the centre; the spread is even across the bar."""
        anchors = [parent_anchor(0.0, 1.0, index, len(SUMMARY_LANES)) for index in range(len(SUMMARY_LANES))]
        assert [round(anchor + mirrored, 9) for anchor, mirrored in zip(anchors, reversed(anchors))] == [1.0] * len(
            anchors
        )

    def test_every_lane_keeps_its_own_row_even_when_it_did_not_run(self, routed: ProvStore) -> None:
        """The consolidation may not drop a row; the row is where "did not run" is said."""
        lanes = branch_lanes(routed)
        withheld = [lane for lane in lanes if lane.state == LANE_WITHHELD]
        assert withheld, "the fixture routes VOICE away, which is what this asserts about"
        assert [row.block for row in span_axis_rows(lanes)] == [BRANCH_INITIAL_ROW, *SUMMARY_LANES]
        assert all(lane_note(lane, 0) for lane in withheld)


def _row_of(lanes: list[BranchLane]) -> dict[str, int]:
    """Which axis row each proposed span is drawn on, keyed by span id."""
    position = {(row.block, row.role): index for index, row in enumerate(span_axis_rows(lanes))}
    return {row.key: position[(lane.branch, row.role)] for lane in lanes for row in lane.proposed}


def _overlaps(first: BranchRow, second: BranchRow) -> bool:
    """Whether two bars cover any of the same recording time."""
    return first.start < second.end and second.start < first.end


class TestABranchIsOneBlockOfRowsRatherThanOneStackedRow:
    """The branch stays one visual block; the kinds of span inside it stop being drawn over one another."""

    @pytest.fixture
    def many_roles(self, routed: ProvStore) -> ProvStore:
        """SPEECH's real shape: a task extent, the phrase runs inside it, and a structure span."""
        first, second = _envelope_spans(routed)
        _run_branch(
            routed,
            "SPEECH",
            [
                ("task_extent", (1.0, 5.0), (first.id, second.id), {"attributed_to": "SPEAKER_00"}),
                ("phrase_run_0", (1.2, 2.4), first.id, {"attributed_to": "SPEAKER_00", "run_index": 0}),
                ("phrase_run_1", (3.0, 4.8), first.id, {"attributed_to": "SPEAKER_00", "run_index": 1}),
                ("structure_0", (1.2, 4.8), first.id, {"attributed_to": "SPEAKER_00", "structure_index": 0}),
            ],
            kind="speech",
            detail={"speaker_count": 1, "words_n": 2, "speech_s": 4.0, "nontarget_speech_s": 0.0, "notes": []},
        )
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), first.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        return routed

    def test_two_roles_covering_the_same_time_are_two_rows(self, many_roles: ProvStore) -> None:
        """The defect: a task extent and a phrase run inside it drew as one bar over another."""
        lanes = branch_lanes(many_roles)
        [speech] = [lane for lane in lanes if lane.branch == "SPEECH"]
        drawn = _row_of(lanes)
        extent = next(row for row in speech.proposed if row.role == "task_extent")
        run = next(row for row in speech.proposed if row.role == "phrase_run")
        assert _overlaps(extent, run), "the fixture must put them over one another for this to assert anything"
        assert drawn[extent.key] != drawn[run.key]

    def test_no_two_overlapping_proposals_of_one_branch_share_a_row(self, many_roles: ProvStore) -> None:
        """Stated as the invariant rather than as the one pair that first showed it broken."""
        lanes = branch_lanes(many_roles)
        drawn = _row_of(lanes)
        collisions = [
            (first.label, second.label)
            for lane in lanes
            for first in lane.proposed
            for second in lane.proposed
            if first.key < second.key and _overlaps(first, second) and drawn[first.key] == drawn[second.key]
        ]
        assert collisions == []

    def test_the_indexed_spans_of_one_kind_share_one_row(self, many_roles: ProvStore) -> None:
        """``phrase_run_0`` and ``phrase_run_1`` are one kind; a row apiece would be a row per span."""
        lanes = branch_lanes(many_roles)
        drawn = _row_of(lanes)
        [speech] = [lane for lane in lanes if lane.branch == "SPEECH"]
        runs = [row for row in speech.proposed if row.role == "phrase_run"]
        assert len(runs) == 2
        assert len({drawn[row.key] for row in runs}) == 1
        assert lane_roles(speech) == ("task_extent", "phrase_run", "structure")

    def test_a_branchs_rows_are_contiguous(self, many_roles: ProvStore) -> None:
        """One block: no other branch's row may fall between two of a branch's own."""
        blocks = [row.block for row in span_axis_rows(branch_lanes(many_roles))]
        runs = [block for index, block in enumerate(blocks) if index == 0 or blocks[index - 1] != block]
        assert runs == list(dict.fromkeys(blocks))
        assert runs == [BRANCH_INITIAL_ROW, *SUMMARY_LANES]

    def test_a_branch_that_minted_one_kind_is_the_single_row_it_always_was(self, many_roles: ProvStore) -> None:
        """AIRWAY proposed one kind here, so nothing about its row may have moved."""
        rows = [row for row in span_axis_rows(branch_lanes(many_roles)) if row.block == "AIRWAY"]
        assert [row.tick for row in rows] == ["AIRWAY"]

    def test_a_branch_that_did_not_run_is_still_exactly_one_row(self, many_roles: ProvStore) -> None:
        """Four run states, and the row a withheld branch says so in may not be lost to the split."""
        lanes = branch_lanes(many_roles)
        withheld = [lane for lane in lanes if lane.state == LANE_WITHHELD]
        assert withheld, "the fixture routes VOICE away, which is what this asserts about"
        rows = span_axis_rows(lanes)
        for lane in withheld:
            assert [row.tick for row in rows if row.block == lane.branch] == [lane.branch]
            assert lane_note(lane, 0)

    def test_a_row_of_a_split_block_names_its_branch_as_well_as_its_kind(self, many_roles: ProvStore) -> None:
        """A bare ``phrase_run`` row would not say whose it is once the block is more than one line."""
        rows = [row for row in span_axis_rows(branch_lanes(many_roles)) if row.block == "SPEECH"]
        assert len(rows) > 1
        assert all(row.tick.startswith("SPEECH") and row.role in row.tick for row in rows)

    def test_the_kind_leaves_the_caption_once_it_is_on_the_axis(self, many_roles: ProvStore) -> None:
        """The bar's width is spent on what the row label does not already say."""
        lanes = branch_lanes(many_roles)
        [speech] = [lane for lane in lanes if lane.branch == "SPEECH"]
        assert {row.short for row in speech.proposed} == {"SPEAKER_00"}


class TestTheInitialRowExplainsItself:
    """The owner could not tell what the initial lane did; it now says what it holds and what its bars are."""

    def test_its_tick_says_more_than_the_bare_word(self, routed: ProvStore) -> None:
        """``initial`` alongside four branch names reads as a fifth branch."""
        [row] = [row for row in span_axis_rows(branch_lanes(routed)) if row.block == BRANCH_INITIAL_ROW]
        assert row.tick != BRANCH_INITIAL_ROW
        assert "what branches read" in row.tick

    def test_an_initial_bar_names_the_kind_of_thing_it_is(self, routed: ProvStore) -> None:
        """A bar reading ``20 dB`` says a level and never says what was measured over what."""
        first = _envelope_spans(routed)[0]
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), first.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [initial] = initial_rows(branch_lanes(routed))
        assert initial.label.startswith("envelope ")
        assert initial.label.endswith(" dB")

    def test_the_row_is_never_silent_when_it_holds_nothing(self, routed: ProvStore) -> None:
        """An empty row with no note cannot be told from a row whose bars are on another page."""
        lanes = branch_lanes(routed)
        assert initial_rows(lanes) == ()
        assert "wasDerivedFrom" in initial_row_note(lanes, 0)

    def test_it_says_when_its_bars_are_all_on_another_page(self, routed: ProvStore) -> None:
        """The two absences are different facts, exactly as they are for a branch row."""
        first = _envelope_spans(routed)[0]
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), first.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        lanes = branch_lanes(routed)
        assert "none on this page" in initial_row_note(lanes, 0)
        assert initial_row_note(lanes, 1) == ""

    def test_the_panel_title_names_both_zones(self) -> None:
        """A reader who cannot see the split from the bars can read it off the title."""
        assert "top band" in SPAN_AXIS_TITLE
        assert "one line per span role" in SPAN_AXIS_TITLE


class TestThePairingFollowsTheDerivationEdge:
    """The edge is the link. Extent overlap is not, and a lane built on it draws the wrong parent."""

    def test_a_proposal_names_the_span_it_was_derived_from(self, routed: ProvStore) -> None:
        """One initial row per named parent, joined by the parent's own id."""
        first, _second = _envelope_spans(routed)
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), first.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        [proposed] = lane.proposed
        assert proposed.derived_from == (first.id,)
        assert [row.key for row in lane.initial] == [first.id]

    def test_the_parent_is_the_one_named_not_the_one_overlapped(self, routed: ProvStore) -> None:
        """A proposal lying inside one span while naming another must draw the named one.

        This is the shape an overlap-based lane gets wrong: the proposal sits wholly inside the
        first envelope span and names only the second, so the two rules disagree by construction.
        """
        first, second = _envelope_spans(routed)
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), second.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        [proposed] = lane.proposed
        assert proposed.derived_from == (second.id,)
        assert [row.key for row in lane.initial] == [second.id]
        assert first.id not in {row.key for row in lane.initial}

    def test_an_initial_span_shared_by_two_proposals_is_drawn_once(self, routed: ProvStore) -> None:
        """Two bars stacked in the same row would read as two measurements of one region."""
        first, _second = _envelope_spans(routed)
        _run_branch(
            routed,
            "AIRWAY",
            [
                ("cough_event", (1.1, 1.4), first.id, {"label": "cough"}),
                ("cough_event", (1.5, 1.8), first.id, {"label": "cough"}),
            ],
            kind="airway",
            detail={"labelled_n": 2, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        assert len(lane.proposed) == 2
        assert [row.key for row in lane.initial] == [first.id]

    def test_an_initial_row_takes_the_producer_s_own_reading(self, routed: ProvStore) -> None:
        """The upper row states what came in, in the words of whoever measured it."""
        first, _second = _envelope_spans(routed)
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), first.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        assert [row.label for row in lane.initial] == ["envelope 20 dB"]

    def test_a_derivation_naming_something_that_is_not_a_span_draws_no_initial_row(self, routed: ProvStore) -> None:
        """A proposal derived from a measurement has no initial span, and inventing one would lie."""
        measurement = next(entity for entity in live_entities(routed, "measurement") if entity.attributes.get("name"))
        _run_branch(
            routed,
            "AIRWAY",
            [("breath_run", (1.1, 1.8), measurement.id, {"label": "breath"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        assert lane.proposed and not lane.initial
        assert lane.proposed[0].derived_from == ()


class TestAPageShowsASpanThatCrossesItsBoundary:
    """Pages are time slices; an extent is never restated as the page edge it was cut at."""

    @pytest.fixture
    def crossing(self, routed: ProvStore) -> ProvStore:
        """One AIRWAY span running from 18 s to 22 s, across the first page boundary."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (18.0, 22.0), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        return routed

    @staticmethod
    def _proposed_on(lane: BranchLane, window: tuple[float, float]) -> list[BranchRow]:
        """Only the branch's own bars on a page; the initial rows sit wherever their parents do."""
        return [row for row in rows_on_page(lane, window) if row.row == BRANCH_PROPOSED_ROW]

    def test_the_span_reaches_both_pages(self, crossing: ProvStore) -> None:
        """Dropping it from either page would hide a span the branch actually proposed."""
        [lane] = [lane for lane in branch_lanes(crossing) if lane.branch == "AIRWAY"]
        key = lane.proposed[0].key
        assert [row.key for row in self._proposed_on(lane, (0.0, 20.0))] == [key]
        assert [row.key for row in self._proposed_on(lane, (20.0, 40.0))] == [key]

    def test_its_extent_is_unchanged_on_both_pages(self, crossing: ProvStore) -> None:
        """The page clips the drawing, never the measurement."""
        [lane] = [lane for lane in branch_lanes(crossing) if lane.branch == "AIRWAY"]
        for window in ((0.0, 20.0), (20.0, 40.0)):
            [row] = self._proposed_on(lane, window)
            assert (row.start, row.end) == (18.0, 22.0)

    def test_a_span_ending_exactly_on_a_boundary_is_on_the_earlier_page_only(self, routed: ProvStore) -> None:
        """A shared boundary is not an overlap, as the module's clip test already has it."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (18.0, 20.0), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        [lane] = [lane for lane in branch_lanes(routed) if lane.branch == "AIRWAY"]
        assert len(self._proposed_on(lane, (0.0, 20.0))) == 1
        assert self._proposed_on(lane, (20.0, 40.0)) == []

    def test_a_page_beyond_every_span_carries_none_of_them(self, crossing: ProvStore) -> None:
        """The control: pages are a slice, not a repetition of the whole file."""
        [lane] = [lane for lane in branch_lanes(crossing) if lane.branch == "AIRWAY"]
        assert rows_on_page(lane, (40.0, 60.0)) == ()


class TestTheBranchReportBlockIsOnTheCover:
    """Conformance, deviations and unmeasured points are file-scoped, so they are not per page."""

    @pytest.fixture
    def reported(self, routed: ProvStore) -> ProvStore:
        """AIRWAY having run, with a deviation and an unmeasured config point."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
            conformance=False,
            deviations=("truncation",),
            unmeasured=("branch.airway.min_event_s",),
        )
        return routed

    def test_it_prints_the_conformance_and_what_it_is_of(self, reported: ProvStore) -> None:
        """A branch writes no verdict; the conformance is what stands in its place."""
        text = "\n".join(branch_report_lines(branch_lanes(reported)))
        assert "AIRWAY" in text and "conformance False" in text
        assert f"of           {TASK}" in text

    def test_it_prints_the_deviations_and_the_unmeasured_points(self, reported: ProvStore) -> None:
        """Both are lists the report carries and neither is summarised away."""
        text = "\n".join(branch_report_lines(branch_lanes(reported)))
        assert "deviations   truncation" in text
        assert "unmeasured   branch.airway.min_event_s" in text

    def test_a_branch_with_neither_says_none_rather_than_leaving_the_line_blank(self, routed: ProvStore) -> None:
        """An empty line reads as an omission; ``none`` is a reading."""
        source = _envelope_spans(routed)[0].id
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), source, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        text = "\n".join(branch_report_lines(branch_lanes(routed)))
        assert "deviations   none" in text and "unmeasured   none" in text

    def test_it_prints_every_measure_the_report_carried(self, reported: ProvStore) -> None:
        """``BRANCH_MEASURES`` is the table both products read, so neither may drop a key."""
        text = "\n".join(branch_report_lines(branch_lanes(reported)))
        assert "labelled_n=1" in text and "contested_n=0" in text and "merged_n=1" in text

    def test_it_counts_how_many_proposals_named_an_initial_span(self, reported: ProvStore) -> None:
        """The cover says how much of the pairing the page can actually draw."""
        text = "\n".join(branch_report_lines(branch_lanes(reported)))
        assert "1 airway span proposed, 1 naming an initial span" in text

    def test_a_withheld_branch_carries_its_reason_and_no_measurements(self, reported: ProvStore) -> None:
        """A branch that never ran has nothing to report, and a zero would be a fabrication."""
        text = "\n".join(branch_report_lines(branch_lanes(reported)))
        assert "VOICE    declined     withheld" in text
        assert "why          route_declined" in text


class TestItDrawsFromTheStore:
    """The sibling contract: same arguments, same shape, nothing written back."""

    @pytest.fixture
    def drawn(self, routed: ProvStore) -> ProvStore:
        """One span in each of two branches, so more than one lane has bars."""
        first, second = _envelope_spans(routed)
        _run_branch(
            routed,
            "AIRWAY",
            [("cough_event", (1.1, 1.8), first.id, {"label": "cough"})],
            kind="airway",
            detail={"labelled_n": 1, "contested_n": 0, "merged_n": 1, "notes": []},
        )
        _run_branch(
            routed,
            "SPEECH",
            [("speech_run_0", (4.1, 4.9), second.id, {"attributed_to": "SPEAKER_00", "nontarget": False})],
            kind="speech",
            detail={"speaker_count": 1, "words_n": 2, "speech_s": 0.8, "nontarget_speech_s": 0.0, "notes": []},
        )
        return routed

    def test_it_writes_one_pdf_with_a_cover_and_a_page_per_window(
        self, drawn: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """An 8 s recording is one padded 20 s page, behind the cover."""
        out = preprocess_figure(drawn, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        assert sorted(out) == ["figure", "taxonomy_summary"]
        assert _pdf_page_count(out["figure"]) == 1 + 1

    def test_the_page_width_is_the_same_drawing_choice_as_the_other_figure(
        self, drawn: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """``pages`` is shared, so a narrower page is one FigureStyle field for both products."""
        out = preprocess_figure(
            drawn, tmp_path / "figures", config, run_dir=tmp_path, stem="rec", style=FigureStyle(page_seconds=4.0)
        )
        assert _pdf_page_count(out["figure"]) == 1 + 2

    def test_pngs_are_written_only_when_the_style_asks(
        self, drawn: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """The PDF is the default output, as it is for the other figure."""
        out = preprocess_figure(
            drawn, tmp_path / "figures", config, run_dir=tmp_path, stem="rec", style=FigureStyle(also_write_pngs=True)
        )
        assert sorted(out) == ["figure", "page01", "taxonomy_summary"]

    def test_it_writes_nothing_back_to_the_store(self, drawn: ProvStore, config: TriageConfig, tmp_path: Path) -> None:
        """A renderer that mutated the store could not be re-run over a finished run directory."""
        before = {entity.id for entity in drawn.entities("span")}
        preprocess_figure(drawn, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        assert {entity.id for entity in drawn.entities("span")} == before
        assert not drawn.activities("FIGURE")

    def test_the_summary_is_written_beside_the_pages(
        self, drawn: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """The cover's block is machine-readable too, so it is not only legible as pixels."""
        out = preprocess_figure(drawn, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        payload = json.loads(out["taxonomy_summary"].read_text())
        assert payload["lines"][0] == "WHOLE-FILE CLASSIFICATION SUMMARY"

    def test_it_draws_a_store_in_which_no_branch_ran(
        self, routed: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """Every lane is a note; the product still exists, because that is itself the finding."""
        out = preprocess_figure(routed, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        assert _pdf_page_count(out["figure"]) == 1 + 1

    def test_it_refuses_a_store_with_no_stream_at_all(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """Without a time axis there is nothing to draw a proposal against."""
        with pytest.raises(LookupError, match="no stream"):
            preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="none")

    def test_it_falls_back_to_the_source_recording_when_preprocess_wrote_no_stream(
        self, store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A PREPROCESS that produced no conditioned stream still leaves ADMIT's own axis to draw on.

        ``report()`` has always fallen back to it, so a figure that refused here would drop every
        evidence page from exactly the runs whose conditioning failed.
        """
        seed_preprocess_store(store, duration_s=4.0, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        for entity in list(store.entities("stream")):
            if entity.attributes.get("name") == "plain":
                store.was_invalidated_by(entity.id, store.activity(node="PREPROCESS", step="drop", parameters={}))
        out = preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        assert out["figure"].is_file()


class TestItOverridesNoPipelineValue:
    """The rule the whole figure module is built around, restated for the lanes."""

    def test_every_new_drawing_value_is_a_style_field(self) -> None:
        """A size, a height or a colour belongs with FigureStyle's other fields and nowhere else."""
        fields = set(FigureStyle().__dataclass_fields__)
        assert {
            "span_axis_height_ratio",
            "colour_branch_initial",
            "lane_colours",
            "branch_row_height",
            "branch_link_linewidth",
            "branch_link_alpha",
        } <= fields

    def test_the_two_paired_rows_are_spelled_as_the_report_spells_them(self) -> None:
        """One pairing, read the same in both products."""
        from senselab.audio.workflows.triage.nodes import report

        assert (BRANCH_INITIAL_ROW, BRANCH_PROPOSED_ROW) == (report._INITIAL_ROW, report._PROPOSED_ROW)

    def test_the_initial_fill_is_the_report_s_own(self) -> None:
        """A reader who has seen one product's pairing must not have to relearn the other's."""
        from senselab.audio.workflows.triage.nodes import report

        assert FigureStyle().colour_branch_initial == report._INITIAL_FILL

    def test_every_lane_can_take_a_colour_of_its_own(self) -> None:
        """The colours are what make a connector crossing a row followable to the row it lands in."""
        assert len(FigureStyle().lane_colours) >= len(SUMMARY_LANES)
        assert len(set(FigureStyle().lane_colours)) == len(FigureStyle().lane_colours)
