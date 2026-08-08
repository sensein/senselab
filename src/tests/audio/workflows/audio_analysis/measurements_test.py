"""Round-tripping a measurement without losing what made it a measurement.

Two absences this design turns on, and both are tested here rather than assumed:

- **Empty is not absent.** A file with no rows says the tool ran and found nothing; a missing file
  says it never ran. Writing only non-empty shapes would collapse them, and a glob matching nothing
  then looks like a stage that produced nothing.
- **``None`` is not ``0.0``.** A frame the tool did not report has to survive as a null, because zero
  is a confident claim and imputing it manufactures confidence nobody expressed.

And the schema has to survive with the bytes: a value whose units live elsewhere is a value the next
reader guesses about, which is how ``units: "mixed"`` and a resolution the model never reported both
came to be stored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis.keys import Route, SignalKey
from senselab.audio.workflows.audio_analysis.measurements import (
    METADATA_KEY,
    read_measurement,
    suffix_for,
    write_measurement,
)
from senselab.audio.workflows.audio_analysis.shapes import (
    Categorical,
    Embedding,
    LabelScore,
    Matrix,
    Series,
    Span,
    Spans,
    Tree,
    Window,
)
from senselab.audio.workflows.audio_analysis.stage_io import Stage, StageIO, UnauthorizedArtifact


@pytest.fixture()
def writer(tmp_path: Path) -> StageIO:
    """An L1 capability rooted in a temp run directory."""
    return StageIO.for_stage(Stage.L1, run_dir=tmp_path)


@pytest.fixture()
def reader(tmp_path: Path) -> StageIO:
    """A round-0 derive capability, which is what reads L1."""
    return StageIO.for_stage(Stage.DERIVE, round=0, run_dir=tmp_path)


def _key(target: str = "snr", producer: str = "pyannote/brouhaha") -> SignalKey:
    return SignalKey(target=target, producer=producer, route=Route())


# ── the round trip, per shape ──────────────────────────────────────────


def test_a_series_round_trips_with_its_units_and_hop(writer: StageIO, reader: StageIO) -> None:
    """The provenance describes the rows present, rather than a measurement they are not at."""
    original = Series(values=(0.1, 0.2, 0.3), hop_s=0.0169, window_s=0.0619, units="dB")
    write_measurement(writer, _key(), original)
    assert read_measurement(reader, _key()) == original


def test_an_unmeasured_frame_survives_as_none(writer: StageIO, reader: StageIO) -> None:
    """The load-bearing round trip. A null that read back as 0.0 would assert a measurement."""
    write_measurement(writer, _key(), Series(values=(0.4, None, 0.6), hop_s=0.01, window_s=0.06, units="dB"))
    restored = read_measurement(reader, _key())
    assert isinstance(restored, Series)
    assert restored.values == (0.4, None, 0.6)
    assert restored.measured_count == 2


def test_a_matrix_round_trips_its_channels_in_order(writer: StageIO, reader: StageIO) -> None:
    """Channel order is read from the metadata, so a reordered table cannot permute them."""
    original = Matrix(
        rows=((0.1, 0.9), (0.2, None)),
        channels=("band_0_100", "band_100_200"),
        hop_s=0.1,
        window_s=0.1,
        units="dB",
    )
    write_measurement(writer, _key(target="noise_floor", producer="band_percentile"), original)
    restored = read_measurement(reader, _key(target="noise_floor", producer="band_percentile"))
    assert restored == original
    assert isinstance(restored, Matrix)
    assert restored.channel("band_100_200") == (0.9, None)


def test_a_matrix_keeps_its_channel_semantics(writer: StageIO, reader: StageIO) -> None:
    """Whether a mean across channels is meaningful at all is not recoverable from the numbers."""
    original = Matrix(
        rows=((0.1, 0.2),),
        channels=("spk_0", "spk_1"),
        channel_semantics="arbitrary",
        hop_s=0.1,
        window_s=0.1,
        units="probability",
    )
    write_measurement(writer, _key(target="occupancy", producer="some/diarizer"), original)
    restored = read_measurement(reader, _key(target="occupancy", producer="some/diarizer"))
    assert isinstance(restored, Matrix)
    assert restored.channels_are_comparable_across_frames is False


def test_a_categorical_round_trips_its_vocabulary_and_top_k(writer: StageIO, reader: StageIO) -> None:
    """Without ``top_k``, a label below the cutoff and a label that scored nothing read alike."""
    original = Categorical(
        windows=(
            Window(start=0.0, end=0.96, scores=(LabelScore("Speech", 0.8), LabelScore("Music", 0.1))),
            Window(start=0.48, end=1.44, scores=(LabelScore("Speech", 0.7),)),
        ),
        vocabulary_id="audioset",
        vocabulary_size=527,
        top_k=7,
    )
    key = _key(target="scene_labels", producer="MIT/ast-finetuned-audioset")
    write_measurement(writer, key, original)
    restored = read_measurement(reader, key)
    assert restored == original
    assert isinstance(restored, Categorical)
    assert restored.mass_is_truncated is True
    assert restored.windows_overlap is True


def test_a_window_that_scored_nothing_survives_as_an_empty_window(writer: StageIO, reader: StageIO) -> None:
    """A long ``(window, label)`` layout would drop the row and shorten the timeline silently."""
    original = Categorical(
        windows=(
            Window(start=0.0, end=0.96, scores=()),
            Window(start=0.96, end=1.92, scores=(LabelScore("Speech", 0.5),)),
        ),
        vocabulary_id="audioset",
        vocabulary_size=527,
        top_k=7,
    )
    key = _key(target="scene_labels", producer="yamnet")
    write_measurement(writer, key, original)
    restored = read_measurement(reader, key)
    assert isinstance(restored, Categorical)
    assert len(restored.windows) == 2
    assert restored.windows[0].scores == ()


def test_an_embedding_round_trips_at_fixed_width(writer: StageIO, reader: StageIO) -> None:
    """Vectors are a cache the diarizer is built from, so they must come back bit-comparable."""
    original = Embedding(vectors=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)), window_s=2.0, hop_s=0.05)
    key = _key(target="speaker_vectors", producer="speechbrain/spkrec-ecapa-voxceleb")
    write_measurement(writer, key, original)
    restored = read_measurement(reader, key)
    assert restored == original
    assert isinstance(restored, Embedding)
    assert restored.dims == 3


def test_spans_round_trip_with_their_capacity(writer: StageIO, reader: StageIO) -> None:
    """Capacity is what makes a count a lower bound rather than a point (D-19)."""
    original = Spans(
        spans=(Span(0.0, 1.5, "spk0", confidence=0.9), Span(1.5, 2.0, "spk1")),
        capacity=4,
    )
    key = _key(target="speaker_spans", producer="nvidia/diar_sortformer_4spk-v1")
    write_measurement(writer, key, original)
    restored = read_measurement(reader, key)
    assert restored == original
    assert isinstance(restored, Spans)
    assert restored.is_censored_at(4) is True
    assert restored.spans[1].confidence is None, "a span with no confidence is not one with zero"


def test_an_unbounded_capacity_round_trips_as_the_string_not_as_a_number(writer: StageIO, reader: StageIO) -> None:
    """``"unbounded"`` and a large int are different claims, and only one is true of a clusterer."""
    key = _key(target="speaker_spans", producer="pyannote/speaker-diarization-community-1")
    write_measurement(writer, key, Spans(spans=(), capacity="unbounded"))
    restored = read_measurement(reader, key)
    assert isinstance(restored, Spans)
    assert restored.capacity == "unbounded"
    assert restored.is_censored_at(99) is False


def test_a_tree_round_trips_as_json_with_its_timestamp_source(writer: StageIO, reader: StageIO) -> None:
    """Flattening a ScriptLine into rows is the reduction L1 is not allowed to make."""
    original = Tree(
        script_line={"text": "hello world", "chunks": [{"text": "hello", "start": 0.0, "end": 0.4}]},
        timestamp_source="bundled_aligner",
    )
    key = _key(target="transcript", producer="Qwen/Qwen3-ASR-1.7B")
    assert suffix_for(original) == ".json"
    write_measurement(writer, key, original)
    assert read_measurement(reader, key) == original


# ── empty is not absent ────────────────────────────────────────────────


def test_an_empty_shape_is_still_written(writer: StageIO, reader: StageIO) -> None:
    """A tool that ran and found nothing stays distinguishable from a tool that never ran."""
    key = _key(target="speaker_spans", producer="some/diarizer")
    path = write_measurement(writer, key, Spans(spans=(), capacity=4))
    assert path.exists()
    restored = read_measurement(reader, key)
    assert isinstance(restored, Spans)
    assert restored.spans == ()


def test_a_missing_measurement_raises_rather_than_reading_as_empty(reader: StageIO) -> None:
    """The failure mode this whole design keeps hitting: a name resolving to nothing, silently."""
    with pytest.raises(FileNotFoundError):
        read_measurement(reader, _key(target="never", producer="ran"))


# ── the schema travels with the artifact ───────────────────────────────


def test_provenance_is_stored_beside_the_shape(writer: StageIO, reader: StageIO) -> None:
    """Model revision and parameters, so the number is reproducible from the audio alone."""
    import pyarrow.parquet as pq

    key = _key()
    path = write_measurement(
        writer,
        key,
        Series(values=(1.0,), hop_s=0.1, window_s=0.1, units="dB"),
        provenance={"revision": "abc123", "device": "cuda"},
    )
    meta = pq.read_table(path).schema.metadata[METADATA_KEY]
    import json

    payload = json.loads(meta)
    assert payload["provenance"]["revision"] == "abc123"
    assert payload["units"] == "dB"
    assert payload["kind"] == "Series"


def test_a_table_without_the_metadata_is_refused_rather_than_guessed(writer: StageIO, reader: StageIO) -> None:
    """A parquet whose units are unknowable is worse than absent, because it looks usable."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    key = _key()
    path = writer.path_for(key, ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"value": pa.array([1.0])}), path)
    with pytest.raises(ValueError, match="metadata"):
        read_measurement(reader, key)


# ── the capability is enforced on both sides ───────────────────────────


def test_writing_a_key_this_stage_does_not_own_is_refused(tmp_path: Path) -> None:
    """The refusal happens before any bytes exist, unlike the artifact guard it replaces."""
    io = StageIO.for_stage(Stage.DERIVE, round=0, run_dir=tmp_path)
    with pytest.raises(UnauthorizedArtifact):
        write_measurement(io, _key(), Series(values=(1.0,), hop_s=0.1, window_s=0.1, units="dB"))


def test_reading_a_key_this_stage_may_not_see_is_refused(tmp_path: Path) -> None:
    """L1 measures the audio; a signal derived from a signal is a derivative."""
    writer = StageIO.for_stage(Stage.L1, run_dir=tmp_path)
    write_measurement(writer, _key(), Series(values=(1.0,), hop_s=0.1, window_s=0.1, units="dB"))
    with pytest.raises(UnauthorizedArtifact):
        read_measurement(writer, _key())
