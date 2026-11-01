"""The summary's cover: its margins, and the blocks the decision page owns instead of it."""

from pathlib import Path
from typing import Any, Iterator

import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.figure import (
    FigureStyle,
    branch_lanes,
    branch_report_lines,
    cover_lines,
    cover_margins,
    summary_pages,
    summary_panel_lines,
)
from senselab.audio.workflows.triage.nodes.report import report
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.report_test import _seed_report_store, _write

#: Every classifier column the cover lays side by side, with enough labels to fill the block.
_LABELS = {
    "yamnet": ["Speech", "Narration, monologue", "Male speech, man speaking", "Inside, small room", "Breathing"],
    "ast": ["Cough", "Speech", "Sneeze", "Sniff", "Gasp"],
    "hear": ["Breathe", "Cough", "Speech", "Snore", "Throat"],
}

#: A section header the cover prints, as it prints it.
_COVER_SECTIONS = (
    "SOURCE",
    "CONSENSUS ALIGNMENT",
    "WHOLE-FILE CLASSIFICATION SUMMARY",
    "ROUTE STATES AND GATE OUTCOMES",
    "GATES THAT COULD NOT BE READ",
    "BRANCH REPORTS",
)

#: A section header the decision record prints, as ``_decision_blocks`` prints it.
_DECISION_SECTIONS = (
    "DECISION SUMMARY",
    "PRIMARY EVIDENCE",
    "SCREENING AND ROUTING",
    "ROUTING GATES",
    "MEASURED BRANCH FINDINGS",
    "SUPPORTING EVIDENCE",
    "ANALYTIC RECORD",
)


def _seed_label_summaries(store: ProvStore) -> None:
    """One whole-file label summary per classifier, in the shape TAXONOMY writes."""
    from senselab.audio.workflows.triage.nodes.common import software_agent

    software = software_agent(store)
    for classifier, labels in _LABELS.items():
        activity = store.activity(
            node="TAXONOMY", step=f"{classifier}_label_summary", parameters={"classifier": classifier}
        )
        store.was_associated_with(activity, software)
        entity = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": f"{classifier}_label_summary",
                "signal": "plain",
                "classifier": classifier,
                "n_windows": 24,
                "win_length_s": 0.96,
                "hop_s": 0.48,
                "labels": {
                    label: {"peak": 0.9 - 0.1 * index, "median": 0.4 - 0.05 * index, "n_windows": 24 - index}
                    for index, label in enumerate(labels)
                },
            },
        )
        store.was_generated_by(entity, activity)
        store.was_attributed_to(entity, software)


@pytest.fixture
def cover_store(store: ProvStore, tmp_path: Path) -> ProvStore:
    """A completed run whose cover carries every block it can carry."""
    _seed_report_store(store, tmp_path, full=True, duration_s=25.0)
    _seed_label_summaries(store)
    return store


@pytest.fixture
def pdf_config(tmp_path: Path) -> TriageConfig:
    """The packaged config in the paginated form."""
    return load_triage_config(_write(tmp_path, "report:\n  format: pdf\n"))


def _rendered_pages(
    store: ProvStore, config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[list[str]], list[str]]:
    """Render the real summary, returning ``(decision_pages, cover_lines)`` as the renderer built them."""
    from senselab.audio.workflows.triage.nodes import figure as figure_module
    from senselab.audio.workflows.triage.nodes import report as report_module

    decision: list[list[str]] = []
    cover: list[list[str]] = []
    real_text = report_module._text_figure
    real_panel = figure_module._taxonomy_panel

    def _spy_text(lines: list[str], title: str, **kwargs: Any) -> Any:  # noqa: ANN401
        decision.append(list(lines))
        return real_text(lines, title, **kwargs)

    def _spy_panel(axis: Any, lines: list[str], style: FigureStyle) -> Any:  # noqa: ANN401
        cover.append(list(lines))
        return real_panel(axis, lines, style)

    monkeypatch.setattr(report_module, "_text_figure", _spy_text)
    monkeypatch.setattr(figure_module, "_taxonomy_panel", _spy_panel)
    report(store, tmp_path / "summary", config, run_dir=tmp_path)
    assert cover, "the run produced no cover to read"
    return decision, cover[0]


def _only_cover(store: ProvStore, config: TriageConfig, tmp_path: Path, **kwargs: Any) -> Any:  # noqa: ANN401
    """The cover figure alone, the evidence pages built and closed behind it."""
    from matplotlib import pyplot

    pages: Iterator[tuple[str, Any]] = summary_pages(store, config, run_dir=tmp_path, **kwargs)
    cover = None
    for name, figure in pages:
        if name == "cover":
            cover = figure
        else:
            pyplot.close(figure)
    assert cover is not None, "summary_pages yielded no cover"
    return cover


class TestTheCoverAndTheDecisionPageShareNoBlock:
    """Page 1 owns the decision; page 2 owns the figure. Neither prints the other's blocks."""

    def test_no_line_is_printed_on_both_pages(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reader turning from page 1 to page 2 must not meet a line they have already read."""
        decision, cover = _rendered_pages(cover_store, pdf_config, tmp_path, monkeypatch)
        record = {line.rstrip() for page in decision for line in page if line.strip()}
        block = {line.rstrip() for line in cover if line.strip()}

        assert record and block, "one of the two pages printed nothing"
        assert not record & block, f"printed on both pages: {sorted(record & block)}"

    def test_neither_page_prints_the_other_s_section_headers(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each page keeps its own headings: the decision record's, and the figure's."""
        decision, cover = _rendered_pages(cover_store, pdf_config, tmp_path, monkeypatch)
        record = {line for page in decision for line in page}
        block = set(cover)

        assert record & set(_DECISION_SECTIONS), "the decision record printed no section header at all"
        assert block & set(_COVER_SECTIONS), "the cover printed no section header at all"
        assert not record & set(_COVER_SECTIONS), f"the decision page prints {sorted(record & set(_COVER_SECTIONS))}"
        assert not block & set(_DECISION_SECTIONS), f"the cover prints {sorted(block & set(_DECISION_SECTIONS))}"

    def test_the_route_states_the_decision_page_gives_are_not_repeated_on_the_cover(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Routing decided; the decision record says so. The cover restating it is the redundancy."""
        decision, cover = _rendered_pages(cover_store, pdf_config, tmp_path, monkeypatch)
        record = "\n".join(line for page in decision for line in page)
        block = "\n".join(cover)

        assert "recording: routed" in record and "routes: AIRWAY=routed" in record
        assert "ROUTE STATES AND GATE OUTCOMES" not in block
        assert "recording: routed" not in block
        assert "gates fired:" not in block

    def test_the_branch_measures_the_decision_page_gives_are_not_repeated_on_the_cover(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``MEASURED BRANCH FINDINGS`` is the numbers behind the verdict, and it is on page 1."""
        decision, cover = _rendered_pages(cover_store, pdf_config, tmp_path, monkeypatch)
        record = "\n".join(line for page in decision for line in page)
        block = "\n".join(cover)

        assert "AIRWAY: labelled_n=1" in record, "the decision page must still carry the measures"
        assert "measures" not in block, "the cover restates the decision page's measured findings"
        assert "conformance" not in block, "the cover restates the conformance the decision page gives"
        assert "spans        1 airway span proposed" in block, "the cover lost what the span axis draws"

    def test_the_classifier_scores_are_not_the_decision_page_s_label_counts(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two different measurements of one classifier; neither is a restatement of the other."""
        decision, cover = _rendered_pages(cover_store, pdf_config, tmp_path, monkeypatch)
        record = "\n".join(line for page in decision for line in page)
        block = "\n".join(cover)

        assert "SUPPORTING EVIDENCE" in record and "yamnet: Speech (7)" in record
        assert "WHOLE-FILE CLASSIFICATION SUMMARY" in block and "peak 0.90 median 0.40" in block

    def test_the_reason_a_gate_could_not_be_read_survives_on_the_cover(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Page 1 names the unreadable gates; only the cover ever said why, so the why stays."""
        from senselab.audio.workflows.triage.nodes.figure import _route_decisions

        decisions = _route_decisions(cover_store)
        assert decisions, "the fixture wrote no route decision to attach an unavailable gate to"
        decisions[0].attributes["unavailable_gates"] = {"voice.glide": "no f0 track in the store"}

        _, cover = _rendered_pages(cover_store, pdf_config, tmp_path, monkeypatch)
        block = "\n".join(cover)

        assert "GATES THAT COULD NOT BE READ" in block
        assert "no f0 track in the store" in block


class TestTheCoverStaysInsideThePageMargins:
    """Measured against the margin the style declares, not against a hand-tuned fraction."""

    def test_every_drawn_artist_is_inside_the_margin_box_at_the_default_style(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Title and body both, on the Letter landscape page the report renders."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot

        style = FigureStyle(figure_inches=(11.0, 8.5))
        cover = _only_cover(cover_store, pdf_config, tmp_path, style=style, decision_record=True)
        cover.canvas.draw()
        renderer = cover.canvas.get_renderer()
        width_px, height_px = cover.canvas.get_width_height()
        horizontal, vertical = cover_margins(style)

        drawn = [*cover.texts, *cover.axes[0].texts]
        assert len(drawn) >= 2, "the cover must draw a title and a body block"
        for artist in drawn:
            box = artist.get_window_extent(renderer=renderer)
            left, right = box.x0 / width_px, box.x1 / width_px
            bottom, top = box.y0 / height_px, box.y1 / height_px
            assert left >= horizontal - 1e-3, f"{artist!r} starts left of the margin at {left:.4f}"
            assert right <= 1.0 - horizontal + 1e-3, f"{artist!r} runs past the right margin to {right:.4f}"
            assert bottom >= vertical - 1e-3, f"{artist!r} drops below the bottom margin to {bottom:.4f}"
            assert top <= 1.0 - vertical + 1e-3, f"{artist!r} rises above the top margin to {top:.4f}"

        # The box itself, not only what this run's content happened to fill: a short cover can sit
        # inside a margin its own axis never respected, and the next long one then runs off the page.
        box = cover.axes[0].get_position()
        assert box.x0 >= horizontal - 1e-3, f"the cover's axis starts at {box.x0:.4f}"
        assert box.x1 <= 1.0 - horizontal + 1e-3, f"the cover's axis ends at {box.x1:.4f}"
        assert box.y0 >= vertical - 1e-3, f"the cover's axis drops to {box.y0:.4f}"
        assert box.y1 <= 1.0 - vertical + 1e-3, f"the cover's axis rises to {box.y1:.4f}"
        pyplot.close(cover)

    def test_the_title_does_not_sit_on_the_body(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Reserving the title's band is what lets the margin hold without the two colliding."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot

        style = FigureStyle(figure_inches=(11.0, 8.5))
        cover = _only_cover(cover_store, pdf_config, tmp_path, style=style, decision_record=True)
        cover.canvas.draw()
        renderer = cover.canvas.get_renderer()
        title = cover.texts[0].get_window_extent(renderer=renderer)
        body = cover.axes[0].texts[0].get_window_extent(renderer=renderer)
        pyplot.close(cover)

        assert body.y1 <= title.y0, "the cover's first line is drawn under its own title"

    def test_the_margin_is_the_style_s_and_nothing_else_s(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Widen the declared margin and the drawn content moves in with it."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot

        edges = []
        for margin_in in (0.5, 1.25):
            style = FigureStyle(figure_inches=(11.0, 8.5), cover_margin_in=margin_in)
            cover = _only_cover(cover_store, pdf_config, tmp_path, style=style, decision_record=True)
            cover.canvas.draw()
            box = cover.axes[0].texts[0].get_window_extent(renderer=cover.canvas.get_renderer())
            edges.append(box.x0 / cover.canvas.get_width_height()[0])
            pyplot.close(cover)

        assert edges[1] > edges[0] + 0.05, f"the declared margin did not move the content: {edges}"


class TestTheStandaloneCoverKeepsWhatItNeeds:
    """``preprocess_figure`` has no decision page, so nothing may vanish from its cover."""

    def test_it_keeps_the_route_states_and_gate_outcomes(self, cover_store: ProvStore) -> None:
        """Without them the standalone figure never says whether a branch was routed at all."""
        text = "\n".join(summary_panel_lines(cover_store, FigureStyle()))

        assert "ROUTE STATES AND GATE OUTCOMES" in text
        assert "recording: routed" in text
        assert "gates fired:" in text

    def test_it_keeps_each_branch_s_route_state_conformance_and_measures(self, cover_store: ProvStore) -> None:
        """The whole branch report, since no earlier page carried any of it."""
        text = "\n".join(branch_report_lines(branch_lanes(cover_store)))

        assert "ran · conformance True" in text
        assert "measures     labelled_n=1" in text

    def test_the_default_is_the_whole_cover(self, cover_store: ProvStore) -> None:
        """A caller that says nothing gets everything; only a decision record subtracts."""
        style = FigureStyle()
        whole = cover_lines(cover_store, summary_panel_lines(cover_store, style))
        trimmed = cover_lines(cover_store, summary_panel_lines(cover_store, style, decision_record=True))

        assert len(whole) > len(trimmed), "declaring a decision record removed nothing from the cover"
        assert set(trimmed) <= set(whole), "the trimmed cover invented a line the whole one lacks"

    def test_the_figure_it_writes_draws_the_whole_cover(
        self, cover_store: ProvStore, pdf_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Measured on what ``preprocess_figure`` actually hands the panel, not on the helper."""
        from matplotlib import pyplot

        from senselab.audio.workflows.triage.nodes import figure as figure_module
        from senselab.audio.workflows.triage.nodes.figure import preprocess_figure

        drawn: list[list[str]] = []
        real_panel = figure_module._taxonomy_panel

        def _spy_panel(axis: Any, lines: list[str], style: FigureStyle) -> Any:  # noqa: ANN401
            drawn.append(list(lines))
            return real_panel(axis, lines, style)

        monkeypatch.setattr(figure_module, "_taxonomy_panel", _spy_panel)
        preprocess_figure(cover_store, tmp_path / "figures", pdf_config, run_dir=tmp_path, stem="rec")
        pyplot.close("all")

        assert drawn, "preprocess_figure drew no cover"
        text = "\n".join(drawn[0])
        assert "ROUTE STATES AND GATE OUTCOMES" in text
        assert "measures     labelled_n=1" in text
