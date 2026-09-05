"""The per-span rasters draw the model's own scores, whatever the labelling threshold says."""

from pathlib import Path
from typing import Any, Callable

import pytest

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.figure import (
    FigureStyle,
    _raster_rows,
    _span_scores,
    summary_panel_lines,
)
from senselab.audio.workflows.triage.nodes.preprocess import _span_window_attributes
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.utils.prov_store import ProvStore


class TestTheWindowAttributes:
    """``labelled`` separates "no threshold was set" from "nothing cleared the bar"."""

    def test_a_null_threshold_keeps_the_scores_and_writes_no_labels(self) -> None:
        """The model ran, so its output is a measurement; only the decision over it is missing."""
        attributes = _span_window_attributes(
            name="span_hear",
            classifier="hear",
            span_id="span-1",
            raw_window={"label_scores": [{"Cough": 0.8}, {"Breathe": 0.1}]},
            default_threshold=None,
            label_thresholds={},
            extra={},
        )
        assert attributes["raw_scores"], "a null threshold must not destroy the model's output"
        assert attributes["labelled"] is False
        assert attributes["default_threshold"] is None
        assert "labels" not in attributes
        assert "scores" not in attributes

    def test_a_set_threshold_labels_as_well(self) -> None:
        """With a threshold there is a decision to record, and it sits beside the scores."""
        attributes = _span_window_attributes(
            name="span_hear",
            classifier="hear",
            span_id="span-1",
            raw_window={"label_scores": [{"Cough": 0.8}, {"Breathe": 0.1}]},
            default_threshold=0.5,
            label_thresholds={},
            extra={},
        )
        assert attributes["labelled"] is True
        assert attributes["default_threshold"] == 0.5
        assert attributes["raw_scores"]
        assert "Cough" in attributes["labels"]
        assert "Breathe" not in attributes["labels"], "0.1 is below the 0.5 bar"

    def test_nothing_clearing_the_bar_is_not_the_same_state(self) -> None:
        """An empty label list with a threshold set is "ran, found nothing" — a real finding."""
        attributes = _span_window_attributes(
            name="span_hear",
            classifier="hear",
            span_id="span-1",
            raw_window={"label_scores": [{"Cough": 0.1}]},
            default_threshold=0.5,
            label_thresholds={},
            extra={},
        )
        assert attributes["labelled"] is True
        assert attributes["labels"] == []


class TestTheRasterRows:
    """Rows are the union of each span's own strongest labels, taken over the whole file."""

    def test_rows_are_the_union_of_each_spans_top_k(self) -> None:
        """A label strong on one span earns a row even if every other span ignores it."""
        per_span = {
            "a": {"Speech": 0.9, "Cough": 0.8, "Snore": 0.7, "Laugh": 0.6, "Sneeze": 0.5},
            "b": {"Breathe": 0.95, "Speech": 0.4, "Cough": 0.3, "Snore": 0.2, "Sneeze": 0.1},
        }
        rows = _raster_rows(per_span, 2, "file")
        assert set(rows) == {"Speech", "Cough", "Breathe"}, "two per span, unioned"
        assert "Sneeze" not in rows, "never in either span's top two"

    def test_rows_are_ranked_by_the_file_wide_peak(self) -> None:
        """A stable, file-wide order is what lets a reader scan one label down every page."""
        per_span = {
            "a": {"Speech": 0.5, "Cough": 0.9},
            "b": {"Speech": 0.95, "Cough": 0.1},
        }
        assert _raster_rows(per_span, 2, "file") == ["Speech", "Cough"]

    def test_a_label_that_never_scored_earns_no_row(self) -> None:
        """A row of zeros is noise; the raster is for what the model actually saw."""
        per_span = {"a": {"Speech": 0.9, "Cough": 0.0}}
        assert _raster_rows(per_span, 4, "file") == ["Speech"]

    def test_only_the_file_scope_is_implemented(self) -> None:
        """Per-page rows would move a label between pages, which the owner ruled out."""
        with pytest.raises(ValueError, match="must be 'file'"):
            _raster_rows({"a": {"Speech": 0.9}}, 4, "page")


class TestTheRasterReadsRawScores:
    """The panel draws the model's output, not the subset that cleared a threshold."""

    def test_scores_come_from_the_raw_output(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """Unlabelled windows still fill the raster, which is the whole point of the change."""
        seed_preprocess_store(
            store,
            spans=[(0.0, 1.0, 20.0)],
            span_hear_labels=[["Cough"]],
            span_unlabelled=("hear",),
        )
        per_span = _span_scores(store, "span_hear")
        assert per_span, "raw scores must reach the figure without a labelling threshold"
        assert next(iter(per_span.values()))["Cough"] == pytest.approx(0.9)


class TestTheAirwayLineUnderUnlabelledWindows:
    """A line that counts labels cannot be judged when nothing was labelled."""

    def test_an_unlabelled_pass_reads_unavailable_not_absent(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """Reading absent here would be a false negative no downstream branch could recover."""
        seed_preprocess_store(
            store,
            spans=[(0.0, 1.0, 20.0)],
            span_hear_labels=[["Cough"]],
            span_yamnet_labels=[["Cough"]],
            span_unlabelled=("hear", "yamnet"),
        )
        taxonomy(store, "plain", config, run_dir=tmp_path)
        kinds = [e for e in store.entities() if e.prov_type == "kind" and e.attributes.get("kind") == "airway"]
        assert kinds, "TAXONOMY wrote no airway kind"
        lines: dict[str, Any] = kinds[-1].attributes.get("lines") or {}
        for name in ("health_acoustic", "acoustic"):
            assert lines[name]["state"] == "unavailable", f"{name} judged an unlabelled pass"


class TestTheSummaryFits:
    """The whole-file readout is laid out across the page and cannot overrun it."""

    def test_no_panel_line_exceeds_the_declared_width(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """A string test is what let the clipping ship; this one bounds the rendered width."""
        from senselab.audio.workflows.triage.nodes.figure import _SUMMARY_COLUMN_WIDTH

        seed_preprocess_store(store, yamnet_labels=[["Speech"], ["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        lines = summary_panel_lines(store, FigureStyle())
        widest = max(len(line) for line in lines)
        assert widest <= 2 + 3 * _SUMMARY_COLUMN_WIDTH, f"a line is {widest} characters wide"

    def test_the_rendered_text_stays_inside_its_axis(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """Measured against the drawn axis, not against a character count."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from senselab.audio.workflows.triage.nodes.figure import _taxonomy_panel

        seed_preprocess_store(store, yamnet_labels=[["Speech"], ["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        style = FigureStyle()
        figure, axis = plt.subplots(figsize=(style.figure_inches[0], 2.0))
        text = _taxonomy_panel(axis, summary_panel_lines(store, style), style)
        figure.canvas.draw()
        assert text is not None, "_taxonomy_panel must return its artist so its extent can be measured"
        extent = text.get_window_extent(renderer=figure.canvas.get_renderer())
        axis_extent = axis.get_window_extent()
        plt.close(figure)
        assert extent.x1 <= axis_extent.x1 + 1.0, "the summary overruns the right edge of its axis"
