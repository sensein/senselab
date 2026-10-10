"""Untimed recognizer spans: the adapter, the word hull, and an unconfirmed word in the speech-outside-task count."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.consensus import word_timing_parameters
from senselab.audio.workflows.triage.nodes.common import (
    consensus_words,
    find_measurement,
    lexical_words,
    word_hull,
)
from senselab.audio.workflows.triage.nodes.preprocess import preprocess
from senselab.audio.workflows.triage.task_speech import task_speech_of
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import _audio, _line, _seed_admit, _stub_models, word_attributes

BREATH_STEM = "sub-x_ses-y_task-respiration-and-cough-v2-threebreathsnose"
BREATH = "respiration-and-cough-v2-threebreathsnose"


def _word(store: ProvStore, text: str, extent: tuple[float, float], index: int, **attributes: object) -> str:
    base = word_attributes(text, extent, index=index, **attributes)  # type: ignore[arg-type]
    return store.entity(prov_type="word", extent=extent, attributes=base)


def _stream(store: ProvStore) -> None:
    store.entity(prov_type="stream", extent=(0.0, 40.0), attributes={"name": "recording", "path": f"{BREATH_STEM}.wav"})


class TestWordHull:
    """The hull is every located span; a missing one is left out and a point one is widened."""

    def test_a_zero_span_does_not_stretch_the_hull_to_zero(self, store: ProvStore) -> None:
        """One recogniser at ``(0, 0)`` beside one at 30 s: the hull is the timed reading's, not 0-30 s."""
        _word(
            store,
            "out",
            (30.0, 30.4),
            0,
            timings={"asr_crisperwhisper": (30.0, 30.4), "asr_qwen": (0.0, 0.0)},
        )
        [word] = consensus_words(store)
        assert word_hull(word) == pytest.approx((30.0, 30.4))

    def test_a_point_span_is_widened_into_a_location(self, store: ProvStore) -> None:
        """A word whose only reading is a point has a positive hull around it."""
        width = word_timing_parameters()["point_width_s"]
        _word(store, "in", (12.0, 12.0), 0, sources=["asr_qwen"], timings={"asr_qwen": (12.0, 12.0)})
        [word] = consensus_words(store)
        assert word_hull(word) == pytest.approx((12.0 - width / 2, 12.0 + width / 2))

    def test_a_word_no_source_located_keeps_its_zero_length_extent(self, store: ProvStore) -> None:
        """No located span: the hull is the extent itself, which has no length."""
        attributes = word_attributes("x", (3.0, 3.0), index=0, sources=["asr_qwen"], timings={"asr_qwen": (3.0, 3.0)})
        attributes["timings"] = {}
        store.entity(prov_type="word", extent=(3.0, 3.0), attributes=attributes)
        [word] = consensus_words(store)
        assert word_hull(word) == (3.0, 3.0)


class TestAnUnconfirmedWord:
    """A word only one recogniser read and none located is not counted as lexical speech, and does not withhold."""

    def test_it_is_left_out_of_the_lexical_words_and_the_speech_outside_the_task(self, store: ProvStore) -> None:
        """Two timed agreed words and one unconfirmed one: two speech words, none of them untimed."""
        _stream(store)
        _word(store, "so", (5.0, 5.3), 0)
        lone = word_attributes("um", (5.3, 5.3), index=1, sources=["asr_qwen"], timings={"asr_qwen": (5.3, 5.3)})
        lone.update(timings={}, unconfirmed=True, untimed_sources=["asr_qwen"], temporal_uncertainty_s=None)
        store.entity(prov_type="word", extent=(5.3, 5.3), attributes=lone)
        _word(store, "where", (5.4, 5.8), 2)
        assert [str(word.attributes["text"]) for word in lexical_words(store)] == ["so", "where"]
        read = task_speech_of(store, BREATH)
        assert read.lexical_n == 2 and read.words_n == 2 and read.untimed_ids == ()

    def test_a_confirmed_word_that_cannot_be_placed_is_still_untimed(self, store: ProvStore) -> None:
        """Both recognisers read it and neither located it: it stays in the count, untimed."""
        _stream(store)
        both = word_attributes("where", (0.0, 0.0), index=0)
        both.update(timings={}, untimed_sources=["asr_crisperwhisper", "asr_qwen"], temporal_uncertainty_s=None)
        store.entity(prov_type="word", extent=(0.0, 0.0), attributes=both)
        read = task_speech_of(store, BREATH)
        assert read.words_n == 1 and len(read.untimed_ids) == 1


class TestTheAdapter:
    """PREPROCESS's recogniser adapter: a ``(0, 0)`` chunk is a missing span, kept and counted."""

    def test_a_whole_file_of_zero_spans_is_kept_counted_and_named(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every Qwen chunk at ``(0, 0)``: kept without times, counted, and the aligner named as failed."""
        _seed_admit(store, tmp_path, wav_writer)
        zeros = [ScriptLine(text=token, start=0.0, end=0.0, score=0.9) for token in ("hello", "there", "world")]
        qwen = ScriptLine(text="hello there world", start=0.0, end=0.0, chunks=zeros, score=0.9)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=qwen)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        hypothesis = find_measurement(store, "asr_qwen")
        assert hypothesis is not None
        assert hypothesis.attributes["untimed_chunks_n"] == 3
        assert [(w["start"], w["end"]) for w in hypothesis.attributes["words"]] == [(None, None)] * 3
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        assert consensus.attributes["aligner_failed"] == ["asr_qwen"]
        assert consensus.attributes["unconfirmed_n"] == 1
        words = consensus_words(store)
        assert all(word_hull(word)[0] >= 0.5 for word in words), "no hull is stretched to zero"
        assert [bool(word.attributes.get("unconfirmed")) for word in words] == [False, True, False]
