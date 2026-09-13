"""FIGURE — the paging rule, the padded tail, and what a panel says when its element is absent."""

import json
import re
from pathlib import Path
from typing import Callable

import pytest

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.figure import (
    FigureStyle,
    _asr_lane_panel,
    _words,
    pages,
    preprocess_figure,
    taxonomy_summary_lines,
)
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.utils.prov_store import ProvStore


def _pdf_page_count(path: Path) -> int:
    """How many pages a PDF holds.

    Args:
        path: The PDF.

    Returns:
        The page count, read from the file's own ``/Type /Page`` objects rather than by rendering.
    """
    return len(re.findall(rb"/Type\s*/Page[^s]", path.read_bytes()))


class TestThePagingRule:
    """Every page is the configured width, so a span's drawn width means one thing throughout."""

    def test_a_recording_shorter_than_a_page_still_gets_one_full_page(self) -> None:
        """A 7 s recording is one 20 s page, not one 7 s page."""
        assert pages(7.0, FigureStyle()) == [(0.0, 20.0)]

    def test_an_exact_multiple_needs_no_padding(self) -> None:
        """40 s is two pages that end exactly where the recording does."""
        assert pages(40.0, FigureStyle()) == [(0.0, 20.0), (20.0, 40.0)]

    def test_a_partial_tail_is_padded_out_to_a_full_page(self) -> None:
        """25 s is two pages, the second running to 40 s with 15 s of padding."""
        assert pages(25.0, FigureStyle()) == [(0.0, 20.0), (20.0, 40.0)]

    def test_every_page_has_the_same_width(self) -> None:
        """The property the padding exists for."""
        widths = {round(end - start, 6) for start, end in pages(53.7, FigureStyle())}
        assert widths == {20.0}

    def test_the_page_width_is_configurable_and_is_a_drawing_choice(self) -> None:
        """page_seconds lives in FigureStyle, so changing it cannot reach the pipeline."""
        assert pages(25.0, FigureStyle(page_seconds=10.0)) == [
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 30.0),
        ]

    def test_padding_can_be_turned_off_leaving_a_short_final_page(self) -> None:
        """Without the pad the final page is ragged, which is what the default avoids."""
        assert pages(25.0, FigureStyle(pad_short_pages=False)) == [(0.0, 20.0), (20.0, 25.0)]

    def test_a_zero_length_recording_still_gets_a_page(self) -> None:
        """One page, so a caller never receives an empty page list to special-case."""
        assert pages(0.0, FigureStyle()) == [(0.0, 20.0)]

    def test_a_non_positive_page_width_is_refused(self) -> None:
        """No number of pages would cover the recording, so this cannot be defaulted."""
        with pytest.raises(ValueError, match="page_seconds must be positive"):
            pages(10.0, FigureStyle(page_seconds=0.0))

    def test_padding_never_changes_which_audio_a_page_covers(self) -> None:
        """The padded and ragged forms agree on every page's start and on the real audio covered."""
        padded = pages(25.0, FigureStyle())
        ragged = pages(25.0, FigureStyle(pad_short_pages=False))
        assert [start for start, _ in padded] == [start for start, _ in ragged]
        assert min(start for start, _ in padded) == 0.0
        assert max(min(end, 25.0) for _, end in padded) == 25.0


class TestItDrawsFromTheStore:
    """It reads what PREPROCESS and TAXONOMY left behind and writes nothing back."""

    def test_it_writes_one_image_per_page(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """A 25 s recording renders one PDF: a summary cover plus two timeline pages."""
        seed_preprocess_store(store, duration_s=25.0, yamnet_labels=[["Speech"]], words=["one"])
        out = preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        assert sorted(out) == ["figure", "taxonomy_summary"]
        assert out["figure"].is_file() and out["figure"].suffix == ".pdf"
        assert _pdf_page_count(out["figure"]) == 1 + 2  # cover, then two 20 s windows

    def test_the_pdf_holds_one_page_per_window(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """A 70 s recording is a cover plus four 20 s pages, the last padded — all one file."""
        seed_preprocess_store(store, duration_s=70.0, yamnet_labels=[["Speech"]], words=["one"])
        out = preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        assert _pdf_page_count(out["figure"]) == 1 + 4  # cover, then four windows

    def test_pngs_are_written_only_when_the_style_asks(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The PDF is the default output; loose pages are opt-in, never both by accident."""
        seed_preprocess_store(store, duration_s=25.0, yamnet_labels=[["Speech"]], words=["one"])
        out = preprocess_figure(
            store,
            tmp_path / "figures",
            config,
            run_dir=tmp_path,
            stem="rec",
            style=FigureStyle(also_write_pngs=True),
        )
        assert sorted(out) == ["figure", "page01", "page02", "taxonomy_summary"]
        assert out["page01"].is_file()

    def test_it_writes_nothing_back_to_the_store(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """A renderer that mutated the store could not be re-run over a finished run."""
        seed_preprocess_store(store, duration_s=5.0, yamnet_labels=[["Speech"]])
        before = len(store.to_jsonl().splitlines()) if hasattr(store, "to_jsonl") else None
        entities_before = {entity.id for entity in store.entities("span")}
        preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        assert {entity.id for entity in store.entities("span")} == entities_before
        assert not store.activities("FIGURE")
        if before is not None:
            assert len(store.to_jsonl().splitlines()) == before

    def test_padding_changes_no_span_extent(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The pad is a display device: every extent is byte-identical after rendering."""
        seed_preprocess_store(store, duration_s=25.0, yamnet_labels=[["Speech"]], words=["one"])
        extents = {entity.id: entity.extent for entity in live_entities(store, "span")}
        preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        assert {entity.id: entity.extent for entity in live_entities(store, "span")} == extents

    def test_a_recording_shorter_than_a_page_renders_one_padded_page(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The common case for a short task recording: a cover and one timeline page."""
        seed_preprocess_store(store, duration_s=3.0, yamnet_labels=[["Speech"]])
        out = preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="short")
        assert sorted(out) == ["figure", "taxonomy_summary"]
        assert _pdf_page_count(out["figure"]) == 1 + 1  # cover, then one window

    def test_it_refuses_a_store_with_no_conditioned_stream(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """A blank page would misreport a missing stream as something measured."""
        with pytest.raises(LookupError, match="no conditioned stream"):
            preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="none")


class TestTheWholeFileTaxonomyPanel:
    """It aggregates over the recording and names what is missing rather than filling it in."""

    def test_it_lists_each_label_with_its_peak_and_median(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The aggregation TAXONOMY writes is what the panel prints."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"], ["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        text = "\n".join(taxonomy_summary_lines(store, FigureStyle()))
        assert "WHOLE-FILE CLASSIFICATION SUMMARY" in text
        assert "yamnet:" in text
        assert "peak" in text and "median" in text

    def test_it_names_the_classifier_that_produced_no_summary(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """Absent is stated, never drawn as an empty row that reads as a measurement."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        text = "\n".join(taxonomy_summary_lines(store, FigureStyle()))
        assert "ast: absent" in text
        assert "hear: absent" in text

    def test_it_reports_each_kind_state_and_its_lines(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The kind entities are file-scoped, which is what makes them a whole-file summary."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        text = "\n".join(taxonomy_summary_lines(store, FigureStyle()))
        assert "KIND STATES AND EVIDENCE LINES" in text
        for kind in ("airway", "speech", "voice"):
            assert f"  {kind}:" in text

    def test_an_unavailable_line_prints_a_null_floor_rather_than_a_number(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """Under the packaged config the floors are null, and the panel says so."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        text = "\n".join(taxonomy_summary_lines(store, FigureStyle()))
        assert "floor —" in text, "a null floor must read as absent, not as a value"
        assert "unavailable" in text

    def test_it_says_taxonomy_never_ran_when_no_kind_reached_the_store(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A store with no kind element is a different fact from every kind being uncertain."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]])
        text = "\n".join(taxonomy_summary_lines(store, FigureStyle()))
        assert "TAXONOMY wrote no kind element" in text

    def test_the_summary_is_written_beside_the_pages(
        self,
        store: ProvStore,
        config: TriageConfig,
        seed_preprocess_store: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """The readout is machine-readable too, so it is not only legible as pixels."""
        seed_preprocess_store(store, duration_s=5.0, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
        taxonomy(store, "plain", config, run_dir=tmp_path)
        out = preprocess_figure(store, tmp_path / "figures", config, run_dir=tmp_path, stem="rec")
        payload = json.loads(out["taxonomy_summary"].read_text())
        assert payload["lines"][0] == "WHOLE-FILE CLASSIFICATION SUMMARY"


class TestItOverridesNoPipelineValue:
    """The rule the scratch tool broke: a figure may not set a pipeline parameter."""

    def test_the_style_shares_no_field_name_with_a_pipeline_key(self) -> None:
        """FigureStyle governs drawing alone, so no field of it can shadow a config path."""
        forbidden = {
            "default_threshold",
            "label_thresholds",
            "f0_range_hz",
            "k_db",
            "continuity_cut_percentile",
            "presence_floor",
        }
        assert not forbidden & set(FigureStyle().__dataclass_fields__)

    def test_it_reads_the_configuration_and_never_writes_one(self, tmp_path: Path) -> None:
        """No override file is produced anywhere under the run directory."""
        import senselab.audio.workflows.triage.nodes.figure as figure_module

        source = Path(figure_module.__file__).read_text()
        assert "write_text" not in source.split("taxonomy_summary.json")[0], (
            "the module must not write a config override; the only file it writes is its own output"
        )


class TestTheWordLaneDrawsEachSourceInItsOwnBand:
    """Test 30: agreement reads as stacked bars, disagreement as separated ones, nothing compounds."""

    @staticmethod
    def _word(index: int, text: str, outcome: str, timings: dict[str, tuple[float, float]], extent: tuple) -> dict:
        """One of ``_words``' dicts, built directly."""
        return {
            "index": index,
            "extent": extent,
            "text": text,
            "outcome": outcome,
            "variants": [],
            "readings": {name: text for name in timings},
            "timings": timings,
            "sources": list(timings),
            "bracketed": text.startswith("["),
        }

    @staticmethod
    def _draw(words: list[dict], window: tuple[float, float] = (0.0, 20.0), **style: object) -> tuple:
        """Draw the lane on a bare axis and hand back the axis with its patches.

        The lane draws filled bands only — the word's derived extent carries no outline — so a
        transparent patch means something has started drawing edges again.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, axis = plt.subplots()
        _asr_lane_panel(axis, words, ["a", "b"], window, FigureStyle(**style), "absent")  # type: ignore[arg-type]
        from matplotlib.colors import to_rgba

        fills = [p for p in axis.patches if to_rgba(p.get_facecolor())[3] > 0.0]
        outlines = [p for p in axis.patches if to_rgba(p.get_facecolor())[3] == 0.0]
        return axis, fills, outlines

    def test_a_two_source_word_draws_two_fills_in_two_bands_and_no_edges(self) -> None:
        """Agreement is two bars stacked at the same time, never one darker bar."""
        word = self._word(0, "hello", "agreement", {"a": (1.0, 1.4), "b": (1.05, 1.45)}, (1.025, 1.425))
        _, fills, outlines = self._draw([word])
        assert len(fills) == 2 and outlines == []
        assert all(patch.get_linewidth() == 0.0 for patch in fills), "a band drew an edge"
        assert len({round(float(p.get_y()), 6) for p in fills}) == 2
        assert {p.get_facecolor()[:3] for p in fills} == {
            matplotlib_colour("#fdae6b"),
            matplotlib_colour("#6baed6"),
        }

    def test_a_one_source_word_draws_one_fill_in_its_sources_band(self) -> None:
        """An insertion leaves the other band empty."""
        word = self._word(0, "uh", "insertion", {"b": (2.0, 2.2)}, (2.0, 2.2))
        _, fills, outlines = self._draw([word])
        assert len(fills) == 1 and outlines == []
        assert fills[0].get_facecolor()[:3] == matplotlib_colour("#6baed6")
        # Source "b" is second in source order, so its band is the upper half of the row: with the
        # default row height 0.52 the row spans -0.26..0.26 and the upper band starts at 0.0.
        assert fills[0].get_y() == pytest.approx(0.0)

    def test_disjoint_readings_draw_at_disjoint_times_in_different_bands(self) -> None:
        """The 4.1 s case: two bars, two times, two bands, and nothing drawn between them."""
        word = self._word(3, "the", "agreement", {"a": (35.30, 35.36), "b": (31.20, 31.84)}, (33.25, 33.60))
        _, fills, outlines = self._draw([word], window=(20.0, 40.0))
        (a_fill,) = [p for p in fills if p.get_x() > 34.0]
        (b_fill,) = [p for p in fills if p.get_x() < 32.0]
        assert a_fill.get_y() != b_fill.get_y()
        assert a_fill.get_x() >= b_fill.get_x() + b_fill.get_width()
        assert outlines == [], "the derived extent must not be drawn where no source put the word"

    def test_a_word_with_an_off_page_extent_is_drawn_where_a_source_put_it(self) -> None:
        """The page filter is the hull of timings and extent, not the derived extent alone."""
        word = self._word(0, "late", "agreement", {"a": (19.5, 19.8), "b": (21.0, 21.3)}, (20.25, 20.55))
        _, fills, outlines = self._draw([word], window=(0.0, 20.0))
        assert len(fills) == 1 and fills[0].get_x() == 19.5
        assert outlines == []
        assert fills[0].get_width() == pytest.approx(0.3), "the bar is the source's own reading"

    def test_the_row_is_the_stream_position_not_the_page_position(self) -> None:
        """A word at stream position 5 draws on row 1 of 4 even as the first word on its page."""
        word = self._word(5, "sixth", "agreement", {"a": (25.0, 25.3), "b": (25.0, 25.3)}, (25.0, 25.3))
        _, fills, _ = self._draw([word], window=(20.0, 40.0), asr_rows=4, asr_row_height=0.5)
        assert min(patch.get_y() for patch in fills) == pytest.approx(1 - 0.25)

    def test_the_title_names_each_source_with_its_colour_and_agreement_is_bold(self) -> None:
        """The page carries its own legend, and the weight is the outcome."""
        words = [
            self._word(0, "I", "agreement", {"a": (0.5, 0.7), "b": (0.5, 0.7)}, (0.5, 0.7)),
            self._word(1, "uh", "insertion", {"a": (0.8, 0.9)}, (0.8, 0.9)),
        ]
        axis, _, _ = self._draw(words)
        from matplotlib.colors import to_rgba

        assert axis.get_title() == "consensus ASR", "the legend belongs beside the lane, not in the title"
        # The legend is placed in axes coordinates; the words are placed in data coordinates.
        in_axes = [text for text in axis.texts if text.get_transform() == axis.transAxes]
        in_data = [text for text in axis.texts if text.get_transform() != axis.transAxes]
        assert {text.get_text(): to_rgba(text.get_color()) for text in in_axes} == {
            "a": to_rgba("#fdae6b"),
            "b": to_rgba("#6baed6"),
        }, "each source must be named in its own colour, never as a hex string"
        assert not any("#" in text.get_text() for text in axis.texts), "a hex code reached the page"
        assert {text.get_text(): text.get_fontweight() for text in in_data} == {"I": "bold", "uh": "normal"}

    def test_words_reads_the_stream_by_index_and_the_source_order(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """``_words`` hands the lane the stream in index order and the measurement's sources."""
        seed_preprocess_store(store, words=["one", {"text": "[UM]", "sources": ["asr_crisperwhisper"]}, "two"])
        words, sources = _words(store)
        assert [w["text"] for w in words] == ["one", "[UM]", "two"]
        assert [w["index"] for w in words] == [0, 1, 2]
        assert sources == ["asr_crisperwhisper", "asr_qwen"]
        assert words[1]["sources"] == ["asr_crisperwhisper"] and words[1]["bracketed"]


def matplotlib_colour(hex_colour: str) -> tuple[float, float, float]:
    """A hex colour as the RGB triple matplotlib reports on a patch."""
    from matplotlib.colors import to_rgb

    return to_rgb(hex_colour)
