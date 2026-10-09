"""The review page's side files: each block of records as one zstd-compressed Parquet file in nested columns.

A shard holds, per recording, what the page draws when it opens that recording. Paths are given once per
record as the run directory and each stream's path under it; evidence items name an entry of the shard's
evidence dictionary (item, group, unit, comparison and threshold), stored as the file's key-value metadata,
and carry only their value, effect and decisive flag. A speech task's transcript is carried as its word
entries and marks, which the page renders. Values whose shape is open (an evidence value or threshold, a
mark, the LLM reviewer's record, the release ground) are JSON text.

The page fetches ``shard-NNNN.parquet`` when it is served. Opened from ``file://`` it cannot fetch, so each
shard also has a ``shard-NNNN.js`` wrapper carrying the same bytes base64-encoded.
"""

from __future__ import annotations

import base64
import io
import json
import posixpath
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

EVIDENCE_METADATA = "senselab.review.evidence"
EVIDENCE_KEYS = ("name", "group", "unit", "comparison", "threshold")
MARK_KEYS = ("k", "c", "s", "nm", "pr", "hr", "tr", "pk", "kd", "dx", "d", "brk", "tx", "nt", "stim", "o")
OVERLAY_MARK_KEYS = ("k", "c", "s", "hr", "kd", "dx", "d")
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 9

_SPAN = pa.list_(pa.float64())
_TIMED_WORD = pa.struct([("s", pa.float64()), ("e", pa.float64()), ("t", pa.string())])
SPEECH = pa.struct(
    [
        ("shown_kind", pa.string()),
        ("shown_source", pa.string()),
        ("models", pa.list_(pa.struct([("source", pa.string()), ("model_id", pa.string())]))),
        (
            "own",
            pa.list_(
                pa.struct(
                    [
                        ("source", pa.string()),
                        ("model_id", pa.string()),
                        ("text", pa.string()),
                        ("words", pa.list_(_TIMED_WORD)),
                    ]
                )
            ),
        ),
        (
            "words",
            pa.list_(
                pa.struct(
                    [
                        ("a", pa.float64()),
                        ("o", pa.string()),
                        ("r", pa.list_(pa.string())),
                        ("s", pa.float64()),
                        ("e", pa.float64()),
                        ("t", pa.string()),
                    ]
                )
            ),
        ),
        ("entries", pa.list_(pa.struct([("t", pa.string()), ("b", pa.bool_()), ("m", pa.int32())]))),
        ("marks", pa.string()),
        (
            "pii",
            pa.list_(pa.struct([("c", pa.string()), ("s", pa.string()), ("h", pa.string()), ("stim", pa.int32())])),
        ),
        ("release_ground", pa.string()),
        ("why", pa.string()),
        ("redact_why", pa.string()),
        ("condition_kind", pa.string()),
        ("language", pa.string()),
        ("names_proposed", pa.list_(pa.string())),
        ("llm", pa.string()),
    ]
)
SCHEMA = pa.schema(
    [
        ("stem", pa.string()),
        ("run_dir", pa.string()),
        ("spec", pa.string()),
        ("spec_stream", pa.string()),
        ("figure", pa.string()),
        ("commit", pa.string()),
        ("config_hash", pa.string()),
        ("missing", pa.list_(pa.string())),
        ("extent", _SPAN),
        ("streams", pa.list_(pa.struct([("n", pa.string()), ("p", pa.string())]))),
        (
            "overlay",
            pa.struct(
                [
                    ("events", pa.list_(_SPAN)),
                    ("activity", pa.list_(_SPAN)),
                    ("issues", pa.list_(pa.struct([("s", pa.float64()), ("e", pa.float64()), ("k", pa.string())]))),
                ]
            ),
        ),
        (
            "evidence",
            pa.list_(pa.struct([("i", pa.int32()), ("v", pa.string()), ("e", pa.string()), ("d", pa.bool_())])),
        ),
        ("speech", SPEECH),
    ]
)


def _json(value: Any) -> str:  # noqa: ANN401 -- any JSON value
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str)


def _text(value: Any) -> str | None:  # noqa: ANN401 -- a stored scalar
    return None if value is None else str(value)


def run_dir_of(record: Mapping[str, Any]) -> str | None:
    """The directory every relative path of a record lies under: its streams' and figure's common parent.

    Args:
        record: The review record.

    Returns:
        The common directory, or None where the record holds no relative path.
    """
    relative = [str(p) for p in (record.get("streams") or {}).values() if p and not str(p).startswith("/")]
    figure = record.get("figure")
    if figure and not str(figure).startswith("/"):
        relative.append(str(figure))
    if not relative:
        return None
    common = posixpath.commonpath([posixpath.dirname(p) for p in relative])
    return common or None


def _under(path: Any, run_dir: str | None) -> str | None:  # noqa: ANN401 -- a stored path
    """A path as the shard stores it: under ``run_dir`` where it lies there, else as given."""
    if path is None:
        return None
    text = str(path)
    if run_dir and text.startswith(run_dir + "/"):
        return text[len(run_dir) + 1 :]
    return text


def _mark(mark: Mapping[str, Any]) -> dict[str, Any]:
    """A mark with only the fields the page renders."""
    kept = {key: mark[key] for key in MARK_KEYS if key in mark and key != "o"}
    overlays = [{key: o[key] for key in OVERLAY_MARK_KEYS if key in o} for o in mark.get("o") or ()]
    if overlays:
        kept["o"] = overlays
    return kept


def speech_row(speech: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A speech view in the shard's columns.

    Args:
        speech: The extract's speech view, or None.

    Returns:
        The row's ``speech`` struct, or None.
    """
    if not speech:
        return None
    words = speech.get("words") or []
    entries = []
    for position, entry in enumerate(speech.get("entries") or ()):
        text = str(entry[0])
        consensus = position < len(words) and str(words[position][5]) == text
        owner = int(entry[2]) if len(entry) > 2 else -1
        entries.append({"t": None if consensus else text, "b": bool(entry[1]), "m": owner})
    shown = speech.get("shown") or {}
    return {
        "shown_kind": _text(shown.get("kind")),
        "shown_source": _text(shown.get("source")),
        "models": [
            {"source": _text(m.get("source")), "model_id": _text(m.get("model_id"))} for m in speech.get("models") or ()
        ],
        "own": [
            {
                "source": _text(o.get("source")),
                "model_id": _text(o.get("model_id")),
                "text": _text(o.get("text")),
                "words": [{"s": w[0], "e": w[1], "t": _text(w[2])} for w in o.get("words") or ()],
            }
            for o in speech.get("own") or ()
        ],
        "words": [
            {"a": w[0], "o": _text(w[1]), "r": [_text(r) for r in w[2] or ()], "s": w[3], "e": w[4], "t": _text(w[5])}
            for w in words
        ],
        "entries": entries,
        "marks": _json([_mark(m) for m in speech.get("marks") or ()]),
        "pii": [
            {"c": _text(p.get("c")), "s": _text(p.get("s")), "h": _text(p.get("h")), "stim": int(p.get("stim") or 0)}
            for p in speech.get("pii") or ()
        ],
        "release_ground": None if speech.get("release_ground") is None else _json(speech["release_ground"]),
        "why": _text(speech.get("why")),
        "redact_why": _text(speech.get("redact_why")),
        "condition_kind": _text(speech.get("condition_kind")),
        "language": _text(speech.get("language")),
        "names_proposed": [str(n) for n in speech.get("names_proposed") or ()],
        "llm": _json(speech.get("llm") or {}),
    }


def shard_rows(block: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A block of records as the shard's rows, and the evidence dictionary they index.

    Args:
        block: The records, in page order.

    Returns:
        The rows, and the dictionary of evidence items as ``{name, group, unit, comparison, threshold}``.
    """
    dictionary: dict[str, int] = {}
    entries: list[dict[str, Any]] = []
    rows = []
    for record in block:
        evidence = []
        for item in record.get("evidence") or ():
            entry = {key: item.get(key) for key in EVIDENCE_KEYS}
            key = _json(entry)
            if key not in dictionary:
                dictionary[key] = len(entries)
                entries.append(entry)
            evidence.append(
                {
                    "i": dictionary[key],
                    "v": _json(item.get("value")),
                    "e": _text(item.get("effect")),
                    "d": bool(item.get("decisive")),
                }
            )
        run_dir = run_dir_of(record)
        overlay = record.get("overlay") or {}
        extent = record.get("extent")
        rows.append(
            {
                "stem": record["stem"],
                "run_dir": run_dir,
                "spec": _text(record.get("spec")),
                "spec_stream": _text(record.get("spec_stream")),
                "figure": _under(record.get("figure"), run_dir),
                "commit": _text(record.get("commit")),
                "config_hash": _text(record.get("config_hash")),
                "missing": [str(m) for m in record.get("missing") or ()],
                "extent": None if extent is None else [float(v) for v in extent],
                "streams": [{"n": str(n), "p": _under(p, run_dir)} for n, p in (record.get("streams") or {}).items()],
                "overlay": {
                    "events": [[float(v) for v in s[:2]] for s in overlay.get("events") or ()],
                    "activity": [[float(v) for v in s[:2]] for s in overlay.get("activity") or ()],
                    "issues": [{"s": s[0], "e": s[1], "k": _text(s[2])} for s in overlay.get("issues") or ()],
                },
                "evidence": evidence,
                "speech": speech_row(record.get("speech")),
            }
        )
    return rows, entries


def parquet_bytes(block: Sequence[Mapping[str, Any]]) -> bytes:
    """One shard's Parquet file.

    Args:
        block: The records, in page order.

    Returns:
        The file's bytes: one row group, zstd-compressed, the evidence dictionary in its metadata.
    """
    rows, dictionary = shard_rows(block)
    schema = SCHEMA.with_metadata({EVIDENCE_METADATA: _json(dictionary)})
    table = pa.Table.from_pylist(rows, schema=schema)
    sink = io.BytesIO()
    pq.write_table(
        table,
        sink,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
        row_group_size=max(1, len(rows)),
    )
    return sink.getvalue()


def wrapper_script(number: int, data: bytes) -> str:
    """The ``file://`` wrapper of one shard: its Parquet bytes, base64-encoded, handed to the page.

    Args:
        number: The shard's number.
        data: Its Parquet bytes.

    Returns:
        The script text.
    """
    return f'ReviewPage.shardParquet({number},"{base64.b64encode(data).decode("ascii")}");\n'
