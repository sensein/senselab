"""Storing a measurement in its native shape, with its schema attached (D-18).

The bridge between :mod:`.shapes` (what a measurement is), :mod:`.keys` (what it is called) and
:mod:`.stage_io` (whether this stage may name it). Nothing here decides anything about the audio: it
serializes a shape and reads it back unchanged.

**The schema travels with the artifact.** Units, hop, window, vocabulary, top-*k*, speaker capacity and
channel semantics go into the parquet's schema metadata, because a value whose units live somewhere
else is a value a later reader will guess about — and the guesses observed were `frame_mean` at a
resolution the model never reported and six quantities under ``units: "mixed"``.

**Two absences that must not collapse into one.** A file with no rows says *the tool ran and found
nothing*; a missing file says *the tool never ran*. So every write happens even when the shape is
empty. And a frame the tool did not report round-trips as ``None``, never ``0.0`` — parquet nulls
carry that faithfully, and the round-trip is tested for it, because imputing zero manufactures a
confident claim nobody made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, cast

import pyarrow as pa
import pyarrow.parquet as pq

from senselab.audio.workflows.audio_analysis.keys import Key
from senselab.audio.workflows.audio_analysis.shapes import (
    Categorical,
    Embedding,
    LabelScore,
    Matrix,
    Measurement,
    Series,
    Span,
    Spans,
    Tree,
    Window,
)
from senselab.audio.workflows.audio_analysis.stage_io import StageIO

__all__ = ["METADATA_KEY", "read_measurement", "suffix_for", "write_measurement"]

METADATA_KEY = b"senselab_measurement"
"""Schema-metadata key holding the shape's own description.

One key rather than one per field, so a reader either has the whole description or none of it. A
partially-populated metadata dict is how ``native_window_s`` came to sit beside a value that was not
at it.
"""

_TREE_SUFFIX = ".json"
_TABLE_SUFFIX = ".parquet"


def suffix_for(shape: Measurement) -> str:
    """The file suffix a shape belongs in.

    A :class:`~.shapes.Tree` is JSON because it is a tree — flattening a ``ScriptLine`` into rows is
    the reduction L1 is not allowed to make. Everything else is tabular.
    """
    return _TREE_SUFFIX if isinstance(shape, Tree) else _TABLE_SUFFIX


def write_measurement(
    io: StageIO,
    key: Key,
    shape: Measurement,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Serialize ``shape`` at the location ``key`` names, if ``io`` permits it.

    Args:
        io: The writing stage's capability. Refuses a key this stage does not own, before any bytes
            exist.
        key: What the measurement is. The path is derived from it; none is accepted.
        shape: The measurement, in its native shape and at its own resolution.
        provenance: Model revision, parameters, timings — merged into the metadata beside the shape's
            own description.

    Returns:
        The path written.

    Raises:
        UnauthorizedArtifact: When this stage may not write this key.
    """
    dest = io.path_for(key, suffix_for(shape))
    dest.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {"kind": type(shape).__name__, **_describe(shape)}
    if provenance:
        metadata["provenance"] = dict(provenance)
    required = io.required_columns(key)
    if required:
        metadata["required_columns"] = list(required)

    if isinstance(shape, Tree):
        dest.write_text(
            json.dumps(
                {"metadata": metadata, "script_line": shape.script_line, "timestamp_source": shape.timestamp_source},
                indent=2,
                default=str,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return dest

    table = _to_table(shape).replace_schema_metadata(
        {METADATA_KEY: json.dumps(metadata, default=str, sort_keys=True).encode("utf-8")}
    )
    pq.write_table(table, dest)
    return dest


def read_measurement(io: StageIO, key: Key, *, suffix: Optional[str] = None) -> Measurement:
    """Read back exactly what :func:`write_measurement` stored.

    Args:
        io: The reading stage's capability.
        key: What to read.
        suffix: Which file, when the shape is not known in advance. Defaults to trying parquet and
            then JSON, since a ``Tree`` is the only JSON shape.

    Returns:
        The shape, with every ``None`` still ``None``.

    Raises:
        UnauthorizedArtifact: When this stage may not read this key.
        FileNotFoundError: When nothing was written — which is a different state from an empty shape
            and is reported as the different thing it is.
    """
    if not io.may_read(key):
        raise _unauthorized(io, key)
    for candidate in [suffix] if suffix else [_TABLE_SUFFIX, _TREE_SUFFIX]:
        path = io.run_dir / key.relative_path(cast(str, candidate))
        if path.exists():
            return _read_path(path)
    raise FileNotFoundError(f"no measurement stored for {key} under {io.run_dir}")


def _unauthorized(io: StageIO, key: Key) -> Exception:
    """The read-side refusal, mirroring ``path_for``'s write-side one."""
    from senselab.audio.workflows.audio_analysis.stage_io import UnauthorizedArtifact

    return UnauthorizedArtifact(f"stage {io.stage.value} may not read {key}")


def _read_path(path: Path) -> Measurement:
    """Reconstruct a shape from its file, using the metadata rather than guessing from columns."""
    if path.suffix == _TREE_SUFFIX:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Tree(script_line=payload["script_line"], timestamp_source=payload["timestamp_source"])
    table = pq.read_table(path)
    raw = (table.schema.metadata or {}).get(METADATA_KEY)
    if raw is None:
        raise ValueError(f"{path} has no {METADATA_KEY.decode()} metadata; its units are unknowable")
    meta = json.loads(raw)
    return _from_table(table, meta)


# ── per-shape serialization ────────────────────────────────────────────


def _describe(shape: Measurement) -> dict[str, Any]:
    """The shape's own schema: everything a reader needs that the rows do not carry."""
    if isinstance(shape, Series):
        return {"hop_s": shape.hop_s, "window_s": shape.window_s, "units": shape.units, "start_s": shape.start_s}
    if isinstance(shape, Matrix):
        return {
            "hop_s": shape.hop_s,
            "window_s": shape.window_s,
            "units": shape.units,
            "start_s": shape.start_s,
            "channels": list(shape.channels),
            "channel_semantics": shape.channel_semantics,
        }
    if isinstance(shape, Categorical):
        return {
            "vocabulary_id": shape.vocabulary_id,
            "vocabulary_size": shape.vocabulary_size,
            "top_k": shape.top_k,
            "units": shape.units,
        }
    if isinstance(shape, Embedding):
        return {"window_s": shape.window_s, "hop_s": shape.hop_s, "dims": shape.dims}
    if isinstance(shape, Spans):
        return {"capacity": shape.capacity}
    return {"timestamp_source": shape.timestamp_source}


def _to_table(shape: Measurement) -> pa.Table:
    """One shape to one table, in the layout that keeps its structure intact."""
    if isinstance(shape, Series):
        # No frame-index column: the row order *is* the frame order, and start_s + hop_s give the
        # time. A stored index would be a second source of truth about the same thing.
        return pa.table({"value": pa.array(list(shape.values), type=pa.float64())})
    if isinstance(shape, Matrix):
        return pa.table(
            {
                name: pa.array([row[index] for row in shape.rows], type=pa.float64())
                for index, name in enumerate(shape.channels)
            }
        )
    if isinstance(shape, Categorical):
        # One row per *window*, labels and scores as lists — so a window that scored nothing is an
        # empty list rather than an absent row. A long (window, label) layout would lose it.
        return pa.table(
            {
                "start": pa.array([w.start for w in shape.windows], type=pa.float64()),
                "end": pa.array([w.end for w in shape.windows], type=pa.float64()),
                "labels": pa.array([[s.label for s in w.scores] for w in shape.windows], type=pa.list_(pa.string())),
                "scores": pa.array([[s.score for s in w.scores] for w in shape.windows], type=pa.list_(pa.float64())),
            }
        )
    if isinstance(shape, Embedding):
        return pa.table({"vector": pa.array([list(v) for v in shape.vectors], type=pa.list_(pa.float64()))})
    if isinstance(shape, Spans):
        return pa.table(
            {
                "start": pa.array([s.start for s in shape.spans], type=pa.float64()),
                "end": pa.array([s.end for s in shape.spans], type=pa.float64()),
                "label": pa.array([s.label for s in shape.spans], type=pa.string()),
                "confidence": pa.array([s.confidence for s in shape.spans], type=pa.float64()),
            }
        )
    raise TypeError(f"{type(shape).__name__} is not a tabular shape")


def _from_table(table: pa.Table, meta: Mapping[str, Any]) -> Measurement:
    """The inverse of :func:`_to_table`, dispatched on the recorded ``kind``."""
    kind = meta["kind"]
    if kind == "Series":
        return Series(
            values=tuple(table.column("value").to_pylist()),
            hop_s=meta["hop_s"],
            window_s=meta["window_s"],
            units=meta["units"],
            start_s=meta["start_s"],
        )
    if kind == "Matrix":
        channels = tuple(meta["channels"])
        # Column order is read from the metadata, not from the table, so a writer that reordered
        # columns cannot silently permute the channels.
        columns = [table.column(name).to_pylist() for name in channels]
        return Matrix(
            rows=tuple(zip(*columns)) if columns else (),
            channels=channels,
            hop_s=meta["hop_s"],
            window_s=meta["window_s"],
            units=meta["units"],
            channel_semantics=meta["channel_semantics"],
            start_s=meta["start_s"],
        )
    if kind == "Categorical":
        starts = table.column("start").to_pylist()
        ends = table.column("end").to_pylist()
        labels = table.column("labels").to_pylist()
        scores = table.column("scores").to_pylist()
        return Categorical(
            windows=tuple(
                Window(
                    start=start,
                    end=end,
                    scores=tuple(LabelScore(label=n, score=v) for n, v in zip(names, values)),
                )
                for start, end, names, values in zip(starts, ends, labels, scores)
            ),
            vocabulary_id=meta["vocabulary_id"],
            vocabulary_size=meta["vocabulary_size"],
            top_k=meta["top_k"],
            units=meta["units"],
        )
    if kind == "Embedding":
        return Embedding(
            vectors=tuple(tuple(v) for v in table.column("vector").to_pylist()),
            window_s=meta["window_s"],
            hop_s=meta["hop_s"],
        )
    if kind == "Spans":
        return Spans(
            spans=tuple(
                Span(start=start, end=end, label=label, confidence=confidence)
                for start, end, label, confidence in zip(
                    table.column("start").to_pylist(),
                    table.column("end").to_pylist(),
                    table.column("label").to_pylist(),
                    table.column("confidence").to_pylist(),
                )
            ),
            capacity=meta["capacity"],
        )
    raise ValueError(f"unknown measurement kind {kind!r}; it was written by a newer writer")
