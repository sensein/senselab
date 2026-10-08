"""Assemble the review page: a compact index inlined into one HTML file, and per-recording side files.

The index carries what the Explore and Review tabs filter on (identity, decision, every evidence
value, a speech task's plain transcript for search). Each side file carries a block of recordings'
spectrograms, overlays, stream paths, full evidence and transcript view, and is loaded by a script
tag when a recording in its block is drawn, so the page works opened from ``file://``.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.review_page.spectrogram import settings

SOURCE_DIR = Path(__file__).parent
VIEWER_DIR = SOURCE_DIR.parent / "viewer"
SHELL = SOURCE_DIR / "shell.html"
INLINE_PATTERN = re.compile(r"/\*@INLINE:(?P<path>[^@]+)@\*/")
DATA_MARKER = "/*@INDEX@*/"
PARTS = {
    "viewer/styles.css": VIEWER_DIR / "styles.css",
    "viewer/theme.js": VIEWER_DIR / "theme.js",
    "viewer/axes.js": VIEWER_DIR / "axes.js",
    "viewer/facets.js": VIEWER_DIR / "facets.js",
    "viewer/corpus.js": VIEWER_DIR / "corpus.js",
    "review.css": SOURCE_DIR / "review.css",
    "review.js": SOURCE_DIR / "review.js",
}
SCALARS = ("participant", "session", "task", "family", "branch", "verdict", "release", "reason", "run_status")
LISTS = ("reasons", "annotations")
SHARD_FIELDS = (
    "stem",
    "spec",
    "spec_stream",
    "overlay",
    "streams",
    "figure",
    "evidence",
    "speech",
    "missing",
    "extent",
    "commit",
    "config_hash",
)
SHARD_DIR = "data"
SPEECH_TEXT = re.compile(r"<[^>]+>")


def ordered(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Records in page order: participant, session, stem.

    Args:
        records: The review records.

    Returns:
        The records, sorted.
    """
    return sorted(records, key=lambda r: (str(r.get("participant") or ""), str(r.get("session") or ""), r["stem"]))


def _dictionary(values: Sequence[Any]) -> dict[str, Any]:
    """One scalar column as distinct values and per-row codes, -1 for absent."""
    distinct: dict[Any, int] = {}
    codes = []
    for value in values:
        if value is None:
            codes.append(-1)
            continue
        codes.append(distinct.setdefault(value, len(distinct)))
    return {"values": list(distinct), "codes": codes}


def _list_dictionary(values: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """One list column as distinct terms and per-row code lists."""
    distinct: dict[Any, int] = {}
    codes = [[distinct.setdefault(term, len(distinct)) for term in (row or ())] for row in values]
    return {"values": list(distinct), "codes": codes}


def _plain_text(speech: Mapping[str, Any] | None) -> str | None:
    """A speech task's transcript as plain lower-case text, for search."""
    if not speech or not speech.get("html"):
        return None
    return " ".join(html.unescape(SPEECH_TEXT.sub(" ", str(speech["html"]))).lower().split())


def _kind(values: Iterable[Any]) -> str:
    present = [v for v in values if v is not None]
    if present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return "numeric"
    return "categorical"


SOURCE_BASE = "/_source"
"""The URL route a served page fetches a stream outside the scan root through: the stream's absolute path
appended to it (``scripts/triage_review_serve.py`` serves it)."""


def index_of(
    records: Sequence[Mapping[str, Any]], *, title: str, audio_base: str, source_base: str = SOURCE_BASE
) -> dict[str, Any]:
    """The page's index over records already in page order.

    Args:
        records: The review records, in page order.
        title: The page title.
        audio_base: The URL prefix from the page to the scan root, used when the page is served.
        source_base: The URL route an absolute stream path (one outside the scan root) is appended to.

    Returns:
        The index the page reads.
    """
    columns: dict[str, Any] = {"stem": [r["stem"] for r in records]}
    for name in SCALARS:
        columns[name] = _dictionary([r.get(name) for r in records])
    for name in LISTS:
        columns[name] = _list_dictionary([r.get(name) or () for r in records])
    columns["items"] = _list_dictionary([sorted({item["name"] for item in r.get("evidence") or ()}) for r in records])
    columns["decisive"] = _list_dictionary(
        [sorted({item["name"] for item in r.get("evidence") or () if item.get("decisive")}) for r in records]
    )
    columns["extent_duration_s"] = [
        None if not r.get("extent") else round(float(r["extent"][1]) - float(r["extent"][0]), 3) for r in records
    ]
    columns["duration_s"] = [None if r.get("duration_s") is None else round(float(r["duration_s"]), 3) for r in records]
    columns["text"] = [_plain_text(r.get("speech")) for r in records]
    evidence: dict[str, dict[str, Any]] = {}
    for row, record in enumerate(records):
        for item in record.get("evidence") or ():
            entry = evidence.setdefault(
                item["name"], {"group": item.get("group"), "unit": item.get("unit"), "rows": [], "values": []}
            )
            value = item.get("value")
            entry["rows"].append(row)
            entry["values"].append(str(value).lower() if isinstance(value, bool) else value)
    for entry in evidence.values():
        entry["kind"] = _kind(entry["values"])
        if entry["kind"] == "categorical":
            entry["values"] = [None if v is None else str(v) for v in entry["values"]]
    spec = settings()
    payload = {
        "title": title,
        "n": len(records),
        "shard_size": int(spec["shards"]["records"]),
        "shard_dir": SHARD_DIR,
        "audio_base": audio_base,
        "source_base": source_base,
        "spec": {k: int(spec["spectrogram"][k]) for k in ("frames", "bands", "levels")},
        "counts": dict(Counter(str(r.get("verdict")) for r in records)),
    }
    index = {"build": payload, "cols": columns, "evidence": evidence}
    payload["id"] = hashlib.sha256(json.dumps(index, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return index


def shard_script(number: int, block: Sequence[Mapping[str, Any]]) -> str:
    """One side file: a script handing its block of records to the page.

    Args:
        number: The block's number.
        block: Its records, in page order.

    Returns:
        The script text.
    """
    body = json.dumps([{k: r.get(k) for k in SHARD_FIELDS} for r in block], separators=(",", ":"), default=str)
    return f"ReviewPage.shard({number},{_script_safe(body)});\n"


def _script_safe(text: str) -> str:
    """JSON made safe to sit inside a script element."""
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _inline(match: re.Match[str]) -> str:
    part = PARTS.get(match.group("path"))
    if part is None:
        raise ValueError(f"the shell inlines {match.group('path')}, which PARTS does not list")
    return part.read_text(encoding="utf-8").rstrip("\n")


def page_html(index: Mapping[str, Any], mark_style: str = "") -> str:
    """The single page, with the index inlined.

    Args:
        index: The page index.
        mark_style: The marked-transcript CSS rules, from the free-speech review page.

    Returns:
        The HTML.
    """
    text = SHELL.read_text(encoding="utf-8")
    named = set(INLINE_PATTERN.findall(text))
    missing = set(PARTS) - named
    if missing:
        raise ValueError(f"PARTS lists {sorted(missing)}, which the shell does not inline")
    text = INLINE_PATTERN.sub(_inline, text)
    text = text.replace("/*@MARKSTYLE@*/", mark_style)
    text = text.replace("@TITLE@", html.escape(str(index["build"]["title"])))
    return text.replace(DATA_MARKER, _script_safe(json.dumps(index, separators=(",", ":"), default=str)))


def write_page(
    records: Iterable[Mapping[str, Any]],
    out_dir: Path,
    *,
    title: str,
    audio_base: str,
    mark_style: str = "",
    source_base: str = SOURCE_BASE,
) -> dict[str, Any]:
    """Write ``index.html`` and its side files.

    Args:
        records: The review records.
        out_dir: The directory to write into.
        title: The page title.
        audio_base: The URL prefix from the page to the scan root.
        mark_style: The marked-transcript CSS rules.
        source_base: The URL route an absolute stream path is appended to.

    Returns:
        The page's id, record count and sizes in bytes.
    """
    rows = ordered(records)
    index = index_of(rows, title=title, audio_base=audio_base, source_base=source_base)
    size = int(index["build"]["shard_size"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SHARD_DIR).mkdir(exist_ok=True)
    shard_bytes = 0
    for number, start in enumerate(range(0, len(rows), size)):
        path = out_dir / SHARD_DIR / f"shard-{number:04d}.js"
        path.write_text(shard_script(number, rows[start : start + size]), encoding="utf-8")
        shard_bytes += path.stat().st_size
    page = out_dir / "index.html"
    page.write_text(page_html(index, mark_style), encoding="utf-8")
    return {
        "id": index["build"]["id"],
        "recordings": len(rows),
        "shards": (len(rows) + size - 1) // size,
        "index_bytes": page.stat().st_size,
        "shard_bytes": shard_bytes,
    }
