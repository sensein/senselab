"""The per-span rasters draw the model's own scores, whatever the labelling threshold says."""

from pathlib import Path
from typing import Any, Callable

import pytest

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import resolve_stream
from senselab.audio.workflows.triage.nodes.figure import (
    FigureStyle,
    _raster_rows,
    _span_scores,
    summary_panel_lines,
)
from senselab.audio.workflows.triage.nodes.preprocess import _span_window_attributes
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.utils.prov_store import ProvStore


def _seed_residual_header(
    store: ProvStore,
    *,
    gain_db: float = 0.03,
    enhanced_energy_fraction: float = 0.98,
    energy_fraction: float = 0.014,
    speech_present: bool = True,
) -> None:
    """The ``residual`` measurement alone, in the shape PREPROCESS writes it."""
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "residual",
            "gain_db": gain_db,
            "enhanced_energy_fraction": enhanced_energy_fraction,
            "energy_fraction": energy_fraction,
            "speech_present": speech_present,
        },
    )


def _seed_residual_summary(
    store: ProvStore, classifier: str, labels: dict[str, dict[str, float]], n_windows: int
) -> None:
    """One classifier's ``residual_<classifier>_summary_all`` measurement."""
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": f"residual_{classifier}_summary_all",
            "classifier": classifier,
            "n_windows": n_windows,
            "n_windows_total": n_windows,
            "labels": labels,
        },
    )


def _seed_preprocess_verdict(store: ProvStore, absent: dict[str, str]) -> None:
    """PREPROCESS's own verdict entity, carrying only the ``absent`` map ``_absent_reasons`` reads."""
    store.entity(prov_type="verdict", extent=None, attributes={"node": "PREPROCESS", "detail": {"absent": absent}})


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
        _seed_residual_header(store)
        _seed_residual_summary(
            store,
            "yamnet",
            {
                "a-label-name-long-enough-to-probe-the-declared-width-limit": {
                    "max_score": 0.91,
                    "mean_score": 0.62,
                    "n_windows": 12,
                },
            },
            n_windows=12,
        )
        _seed_residual_summary(
            store,
            "ast",
            {"Buzz": {"max_score": 0.5, "mean_score": 0.3, "n_windows": 4}},
            n_windows=4,
        )
        lines = summary_panel_lines(store, FigureStyle())
        widest = max(len(line) for line in lines)
        assert widest <= 2 + 3 * _SUMMARY_COLUMN_WIDTH, f"a line is {widest} characters wide"

    def test_the_cover_names_the_source_file_within_the_panel_width(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The full path belongs on the cover, wrapped no wider than the summary beneath it."""
        from senselab.audio.workflows.triage.nodes.figure import cover_lines

        seed_preprocess_store(store, yamnet_labels=[["Speech"], ["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        panel = summary_panel_lines(store, FigureStyle())
        lines = cover_lines(store, panel)

        assert lines[0] == "SOURCE", "the cover must lead with the recording it describes"
        source = "".join(line.strip() for line in lines[1 : lines.index("")])
        recorded = str(store.get_entity(resolve_stream(store, tmp_path, "recording")[0]).attributes["path"])
        assert source == recorded, "the wrapped path must reassemble to the one ADMIT recorded"
        assert max(len(line) for line in lines) <= max(len(line) for line in panel), (
            "a cover line reaches further right than the summary panel does"
        )

    def test_the_rendered_text_stays_inside_its_axis(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """Measured against the drawn axis, not against a character count.

        Renders the whole cover — source, consensus alignment block and summary — so the bound
        holds with the alignment block present, not just the summary panel alone.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from senselab.audio.workflows.triage.nodes.figure import _taxonomy_panel, cover_lines

        seed_preprocess_store(
            store,
            yamnet_labels=[["Speech"], ["Speech"]],
            scores_only=("yamnet",),
            words=[
                {"text": "hello"},
                {"text": "world", "timings": {"asr_crisperwhisper": (3.0, 3.5), "asr_qwen": (3.0, 3.5)}},
            ],
        )
        taxonomy(store, "plain", config, run_dir=tmp_path)
        _seed_residual_header(store)
        _seed_residual_summary(
            store,
            "yamnet",
            {
                "a-label-name-long-enough-to-probe-the-declared-width-limit": {
                    "max_score": 0.91,
                    "mean_score": 0.62,
                    "n_windows": 12,
                }
            },
            n_windows=12,
        )
        _seed_residual_summary(
            store, "ast", {"Buzz": {"max_score": 0.5, "mean_score": 0.3, "n_windows": 4}}, n_windows=4
        )
        style = FigureStyle()
        figure, axis = plt.subplots(figsize=(style.figure_inches[0], 2.0))
        lines = cover_lines(store, summary_panel_lines(store, style))
        text = _taxonomy_panel(axis, lines, style)
        figure.canvas.draw()
        assert text is not None, "_taxonomy_panel must return its artist so its extent can be measured"
        extent = text.get_window_extent(renderer=figure.canvas.get_renderer())
        axis_extent = axis.get_window_extent()
        plt.close(figure)
        assert extent.x1 <= axis_extent.x1 + 1.0, "the cover overruns the right edge of its axis"


class TestTheResidualSummary:
    """The residual stream's own whole-file classification summary, on the cover page."""

    def test_it_sits_between_the_main_summary_and_the_kind_states(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The slot a previous pass identified as having ample room."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        _seed_residual_header(store)
        _seed_residual_summary(store, "yamnet", {"Buzz": {"max_score": 0.6, "mean_score": 0.4, "n_windows": 2}}, 2)
        _seed_residual_summary(store, "ast", {"Hum": {"max_score": 0.5, "mean_score": 0.3, "n_windows": 3}}, 3)

        lines = summary_panel_lines(store, FigureStyle())
        top = lines.index("WHOLE-FILE CLASSIFICATION SUMMARY")
        bottom = lines.index("KIND STATES AND EVIDENCE LINES")
        residual = next(i for i, line in enumerate(lines) if line.strip().startswith("RESIDUAL"))
        assert top < residual < bottom

    def test_the_header_states_the_residuals_own_provenance(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """Gain, both energy fractions, and whether speech was present, read straight off the store."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        _seed_residual_header(
            store, gain_db=0.03, enhanced_energy_fraction=0.98, energy_fraction=0.014, speech_present=True
        )
        _seed_residual_summary(store, "yamnet", {"Buzz": {"max_score": 0.6, "mean_score": 0.4, "n_windows": 2}}, 2)
        _seed_residual_summary(store, "ast", {"Hum": {"max_score": 0.5, "mean_score": 0.3, "n_windows": 3}}, 3)

        text = "\n".join(summary_panel_lines(store, FigureStyle()))
        assert "+0.03 dB" in text
        assert "98.0%" in text
        assert "1.4%" in text
        assert "speech present" in text

    def test_it_reports_yamnet_and_ast_but_never_hear_or_the_speech_free_variant(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The owner's ruling: just the all-windows YAMNet/AST summary, kept simple."""
        seed_preprocess_store(
            store, yamnet_labels=[["Speech"]], ast_labels=[["Speech"]], hear_labels=[["Cough"]], scores_only=()
        )
        taxonomy(store, "plain", config, run_dir=tmp_path)
        _seed_residual_header(store)
        _seed_residual_summary(store, "yamnet", {"Buzz": {"max_score": 0.6, "mean_score": 0.4, "n_windows": 2}}, 2)
        _seed_residual_summary(store, "ast", {"Hum": {"max_score": 0.5, "mean_score": 0.3, "n_windows": 3}}, 3)

        lines = summary_panel_lines(store, FigureStyle())
        residual_start = next(i for i, line in enumerate(lines) if line.strip().startswith("RESIDUAL"))
        residual_end = next(i for i in range(residual_start, len(lines)) if lines[i] == "")
        residual_text = "\n".join(lines[residual_start:residual_end])
        assert "yamnet" in residual_text and "ast" in residual_text
        assert "hear" not in residual_text
        assert "speech_free" not in residual_text and "speech-free" not in residual_text

    def test_a_disabled_block_states_that_reason_not_zeros(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """``residual.enabled: false`` is the default, and this is the state most stores are in."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        _seed_preprocess_verdict(
            store,
            {
                "residual": "ValueError: residual.enabled is false",
                "residual_yamnet": "LookupError: residual is absent",
                "residual_ast": "LookupError: residual is absent",
            },
        )

        text = "\n".join(summary_panel_lines(store, FigureStyle()))
        assert "residual.enabled is false" in text
        assert "0.00 dB" not in text
        assert "0.0%" not in text

    def test_a_gated_block_states_the_gates_own_reason(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """A different reason text from the disabled case, so a reader does not conflate the two."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        _seed_preprocess_verdict(
            store,
            {
                "residual": (
                    "ValueError: FRCRN's enhanced output retained only 0.0500 of the input's energy, below the "
                    "configured minimum 0.5000 -- it nulled its input rather than passing it through"
                ),
                "residual_yamnet": "LookupError: residual is absent",
                "residual_ast": "LookupError: residual is absent",
            },
        )

        text = "\n".join(summary_panel_lines(store, FigureStyle()))
        assert "nulled its input" in text
        assert "residual.enabled is false" not in text

    def test_a_block_that_ran_with_no_windows_says_so_rather_than_an_empty_list(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """A residual too short to yield one AST window is not the same fact as the block never running."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        _seed_residual_header(store)
        _seed_residual_summary(store, "yamnet", {"Buzz": {"max_score": 0.6, "mean_score": 0.4, "n_windows": 2}}, 2)
        _seed_residual_summary(store, "ast", {}, 0)

        text = "\n".join(summary_panel_lines(store, FigureStyle()))
        assert "produced no windows" in text
        assert "peak 0.00" not in text

    def test_a_classifier_absent_from_the_residual_alone_states_its_own_reason(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The residual itself ran; only AST's own pass over it failed."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        _seed_residual_header(store)
        _seed_residual_summary(store, "yamnet", {"Buzz": {"max_score": 0.6, "mean_score": 0.4, "n_windows": 2}}, 2)
        _seed_preprocess_verdict(store, {"residual_ast": "RuntimeError: worker timed out"})

        text = "\n".join(summary_panel_lines(store, FigureStyle()))
        assert "ast: absent" in text
        assert "worker timed out" in text


class TestTheConsensusAlignmentBlock:
    """How the consensus transcript was aligned, and how much to trust its timings."""

    def test_it_reports_the_stored_provenance_and_word_uncertainty(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """Every figure the block prints is read straight off the seeded store, not recomputed."""
        from senselab.audio.workflows.triage.nodes.figure import consensus_alignment_lines

        seed_preprocess_store(
            store,
            words=[
                {"text": "hello"},
                {"text": "world", "timings": {"asr_crisperwhisper": (3.0, 3.5), "asr_qwen": (3.0, 3.5)}},
            ],
        )
        measurement = next(
            e for e in store.entities("measurement") if e.attributes.get("name") == "consensus_transcript"
        )
        attrs = measurement.attributes

        lines = consensus_alignment_lines(store)

        assert lines[0] == "CONSENSUS ALIGNMENT"
        assert (
            lines[1] == f"  {attrs['algorithm']} · {attrs['n_sources']} sources · reference {attrs['reference_source']}"
        )
        for row, line in zip(attrs["sources"], lines[2 : 2 + len(attrs["sources"])]):
            assert line == f"    {row['name']}: {row['n_words']} words ({row['timestamp_source']})"
        outcomes = attrs["outcomes"]
        offset = 2 + len(attrs["sources"])
        assert lines[offset] == (
            f"  outcomes: agreement {outcomes['agreement']} (100%)"
            f"  variant {outcomes['variant']} (0%)"
            f"  insertion {outcomes['insertion']} (0%)"
            f"  of {attrs['n_words']}"
        )
        assert lines[offset + 1] == (
            f"  time fit: {attrs['n_words_time_shifted']} words shifted, "
            f"max shift {float(attrs['max_time_shift_s']):.2f}s"
        )
        # "hello" reads identically at its own extent on both sources (overlaps); "world" is timed
        # at 3.0-3.5s by both sources while its derived extent lands near 0.9-1.2s (off-source).
        assert lines[offset + 2] == "  uncertainty: sum 2.30s · median 1.15s · >1s 1 words · off-source extent 1/2"

    def test_it_states_absence_when_no_consensus_reached_the_store(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """A single-recognizer run raises before any consensus_transcript reaches the store."""
        from senselab.audio.workflows.triage.nodes.figure import consensus_alignment_lines

        seed_preprocess_store(store, yamnet_labels=[["Speech"]])

        lines = consensus_alignment_lines(store)

        assert lines[0] == "CONSENSUS ALIGNMENT"
        assert lines[1].startswith("  absent:"), "an absent consensus must be stated, not silenced"
        assert "0" not in lines[1], "an absence is not the same fact as a zero count"

    def test_a_word_off_its_own_sources_is_counted_and_one_on_them_is_not(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """The fit can place a word where none of its own recognizers put it; count only that one."""
        from senselab.audio.workflows.triage.nodes.figure import _consensus_word_stats

        seed_preprocess_store(
            store,
            words=[
                {"text": "inside"},
                {"text": "outside", "timings": {"asr_crisperwhisper": (3.0, 3.5), "asr_qwen": (3.0, 3.5)}},
            ],
        )

        stats = _consensus_word_stats(store)

        assert stats is not None
        assert stats["n_words"] == 2
        assert stats["n_off_source"] == 1, "exactly the word timed far from its own sources is counted"
