"""Build a local HTML review page for the free-response transcripts in a triage corpus.

Two phases, because the corpus lives on a cluster and the page is read on a laptop.

``extract`` walks a finished run tree, keeps the recordings whose declared family carries
``Pattern.FREE_RESPONSE``, and writes one compact JSON line per recording.

``render`` turns those lines into a self-contained HTML page: no network reference, no external
asset, grouped by participant, each recording one paragraph of consensus text with its redaction
findings marked inline.

The extract output and the rendered page both carry transcribed speech. They are local artefacts.
See ``specs/20260923-free-speech-review-page/design.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.nodes.gates import Pattern
from senselab.audio.workflows.triage.recording_vectors import (
    RUN_SUBDIR,
    STORE_NAME,
    Entity,
    StoreView,
    identity,
    recording_dirs,
    stem_of,
)
from senselab.audio.workflows.triage.routing_analysis.families import task_family, task_id_of

PII_LABEL = "pii"
LABEL_VERB = "label"
VERDICT_NODE = "VERDICT"
REDACT_NODE = "REDACT"
SPEECH_NODE = "SPEECH"
PII_SCAN = "pii_scan"

TASK_EXTENT_ROLE = "task_extent"
"""The ``span`` role the in-family branch mints for the interval it judged the task to occupy."""

SPEECH_FAMILY = "speech"
"""The span family SPEECH stamps; every free-response recording routes to SPEECH."""

MEASUREMENT_MARKER = '"prov_type": "measurement"'
KEPT_MEASUREMENT_MARKER = f'"name": "{PII_SCAN}"'

RELEASE_ORDER = ("releasable", "withheld", "nothing_to_redact", "not_assessed", "unrecorded")


def free_response_families() -> frozenset[str]:
    """The families the graph itself declares as free response.

    Returns:
        Every family name in :data:`EXPECTATIONS` whose expectation carries
        :attr:`Pattern.FREE_RESPONSE`.
    """
    return frozenset(
        name
        for group in EXPECTATIONS.values()
        for name, expectation in group.items()
        if expectation.pattern is Pattern.FREE_RESPONSE
    )


def read_store_light(path: Path) -> StoreView:
    """Read a store, skipping the measurement payloads this page never reads.

    Args:
        path: The ``store.jsonl``.

    Returns:
        The view, in the same shape :func:`recording_vectors.read_store` returns.
    """
    view = StoreView()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            if MEASUREMENT_MARKER in line and KEPT_MEASUREMENT_MARKER not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                view.malformed_lines += 1
                continue
            kind = record.get("record")
            if kind == "entity":
                extent = record.get("extent")
                view.entities.append(
                    Entity(
                        id=str(record.get("id")),
                        prov_type=str(record.get("prov_type")),
                        extent=(float(extent[0]), float(extent[1])) if extent else None,
                        attributes=record.get("attributes") or {},
                    )
                )
            elif kind == "relation":
                relation, source, target = record.get("relation"), record.get("source"), record.get("target")
                if relation == "wasInvalidatedBy":
                    view.invalidated.add(str(source))
                elif relation == "wasDerivedFrom":
                    view.derived.setdefault(str(source), []).append(str(target))
    return view


def marked_words(view: StoreView) -> dict[str, list[str]]:
    """Which PII categories the live label assertions place on each word.

    Args:
        view: The store view.

    Returns:
        Word entity id to the categories marked on it, in first-seen order.
    """
    marked: dict[str, list[str]] = {}
    for assertion in view.live("assertion"):
        attributes = assertion.attributes
        if attributes.get("verb") != LABEL_VERB or attributes.get("label") != PII_LABEL:
            continue
        category = str(attributes.get("category") or "")
        if not category:
            continue
        for word_id in view.derived.get(assertion.id, []):
            categories = marked.setdefault(word_id, [])
            if category not in categories:
                categories.append(category)
    return marked


def scan_state(view: StoreView) -> bool | None:
    """Whether SPEECH's PII scan ran over this recording's transcript.

    Args:
        view: The store view.

    Returns:
        False when a live ``pii_scan`` measurement records the scan as declined, True when one
        records it as run, and None when the store carries none.
    """
    state: bool | None = None
    for measurement in view.live("measurement"):
        if measurement.attributes.get("name") != PII_SCAN:
            continue
        state = bool(measurement.attributes.get("scanned", True))
    return state


def overlaps(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Whether two half-open intervals meet.

    Args:
        first: One interval.
        second: The other.

    Returns:
        True when they share any time.
    """
    return first[0] < second[1] and first[1] > second[0]


def word_hull(word: Entity) -> tuple[float, float]:
    """A word's extent, widened to every recognizer's placement of it.

    Args:
        word: A live consensus ``word`` entity.

    Returns:
        ``(earliest start, latest end)``, or ``(0.0, 0.0)`` when nothing timed it.
    """
    spans = [(float(span[0]), float(span[1])) for span in (word.attributes.get("timings") or {}).values()]
    if word.extent is not None:
        spans.append((float(word.extent[0]), float(word.extent[1])))
    if not spans:
        return (0.0, 0.0)
    return min(span[0] for span in spans), max(span[1] for span in spans)


def finding_key(stem: str, categories: Sequence[str], start: int, end: int) -> str:
    """A mark's identity, stable across re-extraction and independent of page order.

    Args:
        stem: The recording's BIDS stem.
        categories: The mark's categories, in store order.
        start: The first consensus word index the mark covers.
        end: One past the last.

    Returns:
        A short content-addressed key.
    """
    payload = "|".join([stem, "+".join(categories), str(start), str(end)])
    return hashlib.sha1(payload.encode(), usedforsecurity=False).hexdigest()[:16]


def marks_of(
    stem: str,
    words: Sequence[Entity],
    categories: Sequence[Sequence[str]],
    hulls: Sequence[tuple[float, float]],
    pii: Sequence[Entity],
    extent: tuple[float, float] | None,
) -> list[dict[str, Any]]:
    """The reviewable marks: maximal runs of adjacent words sharing one category set.

    Args:
        stem: The recording's BIDS stem.
        words: The consensus words, in index order.
        categories: Each word's categories, aligned with ``words``.
        hulls: Each word's timing hull, aligned with ``words``.
        pii: The live ``pii`` entities, read for detector attribution.
        extent: The task extent, or None when the recording carries none.

    Returns:
        One record per mark, in reading order.
    """
    marks: list[dict[str, Any]] = []
    index = 0
    while index < len(words):
        current = list(categories[index])
        if not current:
            index += 1
            continue
        cursor = index + 1
        while cursor < len(words) and list(categories[cursor]) == current:
            cursor += 1
        run = words[index:cursor]
        hull = (
            min(hulls[position][0] for position in range(index, cursor)),
            max(hulls[position][1] for position in range(index, cursor)),
        )
        contributing = _contributing(hull, current, pii)
        surface = " ".join(str(word.attributes.get("text") or "") for word in run)
        marks.append(
            {
                "k": finding_key(stem, current, index, cursor),
                "c": current,
                "d": [str(finding.attributes.get("source") or "") for finding in contributing],
                "dn": len(contributing),
                "i": [index, cursor],
                "nt": cursor - index,
                "nc": len(surface),
                "brk": 1 if any(word.attributes.get("bracketed") for word in run) else 0,
                "stim": 1 if any(finding.attributes.get("in_stimulus") for finding in contributing) else 0,
                "tx": _extent_flag(hull, extent),
            }
        )
        index = cursor
    return marks


def _contributing(hull: tuple[float, float], categories: Sequence[str], pii: Sequence[Entity]) -> list[Entity]:
    """The findings that plausibly produced one mark, tightest first.

    The join is geometric and approximate. A ``pii`` entity's extent is the hull of every
    recognizer's placement of its words, so it is generally wider than the mark it produced and can
    reach neighbouring unmarked words; and the label assertion carries no pointer back to the
    finding. A finding whose extent contains the mark is preferred over one that merely meets it,
    and the narrowest comes first, but a store can carry two findings of one category over nested
    word ranges and then the attribution is genuinely ambiguous — which is what ``dn`` records.

    Args:
        hull: The mark's timing hull.
        categories: The mark's categories.
        pii: The live ``pii`` entities.

    Returns:
        The candidate findings, containing ones first and narrowest first within each group.
    """
    candidates: list[tuple[int, float, Entity]] = []
    for finding in pii:
        if str(finding.attributes.get("category") or "") not in categories or finding.extent is None:
            continue
        span = (float(finding.extent[0]), float(finding.extent[1]))
        if not overlaps(hull, span):
            continue
        contains = span[0] <= hull[0] and span[1] >= hull[1]
        candidates.append((0 if contains else 1, span[1] - span[0], finding))
    return [finding for _, _, finding in sorted(candidates, key=lambda item: (item[0], item[1]))]


def _extent_flag(hull: tuple[float, float], extent: tuple[float, float] | None) -> int:
    """Whether a mark falls inside the task extent.

    Args:
        hull: The mark's timing hull.
        extent: The task extent, or None when the recording carries none.

    Returns:
        1 inside, 0 outside, -1 when the recording declares no extent.
    """
    if extent is None:
        return -1
    return 1 if overlaps(hull, extent) else 0


def task_extent(view: StoreView) -> tuple[float, float] | None:
    """The interval the in-family branch judged the task to occupy.

    Args:
        view: The store view.

    Returns:
        ``(start, end)``, or None when the branch minted no such span. Absence is the branch's
        record that it found no task, not a read failure.
    """
    span = view.last("span", role=TASK_EXTENT_ROLE, family=SPEECH_FAMILY)
    if span is None or span.extent is None:
        return None
    return (float(span.extent[0]), float(span.extent[1]))


def recording_record(run_root: Path, families: frozenset[str]) -> dict[str, Any] | None:
    """One recording's row, or None when it is not a free-response recording.

    Args:
        run_root: The directory holding ``run/store.jsonl``.
        families: The family names to keep.

    Returns:
        The row, or None when the declared family is out of scope.
    """
    stem = stem_of(run_root)
    task = task_id_of(stem)
    if task_family(task) not in families:
        return None
    view = read_store_light(run_root / RUN_SUBDIR / STORE_NAME)
    verdict = view.last("verdict", node=VERDICT_NODE)
    attributes = verdict.attributes if verdict is not None else {}
    declared = str(attributes.get("declared_family") or task_family(task))
    if declared not in families:
        return None
    participant, session, _ = identity(stem)
    marked = marked_words(view)
    entities = sorted(view.live("word"), key=lambda entity: int(entity.attributes.get("index", 0)))
    categories = [marked.get(entity.id, []) for entity in entities]
    hulls = [word_hull(entity) for entity in entities]
    pii = view.live("pii")
    marks = marks_of(stem, entities, categories, hulls, pii, task_extent(view))
    owner = {position: number for number, mark in enumerate(marks) for position in range(mark["i"][0], mark["i"][1])}
    words: list[list[Any]] = []
    characters = 0
    lexical = 0
    for position, entity in enumerate(entities):
        text = str(entity.attributes.get("text") or "")
        bracketed = 1 if entity.attributes.get("bracketed") else 0
        words.append([text, bracketed, owner.get(position, -1)])
        characters += len(text)
        if not bracketed:
            lexical += 1
    redact = view.last("verdict", node=REDACT_NODE)
    return {
        "p": participant or "",
        "ses": session or "",
        "task": task,
        "stem": stem,
        "fam": declared,
        "rel": str(attributes.get("release") or ""),
        "rg": attributes.get("release_ground"),
        "tri": str(attributes.get("triage") or attributes.get("outcome") or ""),
        "why": str(attributes.get("why") or ""),
        "rwhy": str(redact.attributes.get("why") or "") if redact is not None else "",
        "scan": scan_state(view),
        "w": words,
        "f": marks,
        "pii": [
            {
                "c": str(finding.attributes.get("category") or ""),
                "s": str(finding.attributes.get("source") or ""),
                "h": str(finding.attributes.get("haystack") or ""),
                "stim": bool(finding.attributes.get("in_stimulus")),
            }
            for finding in pii
        ],
        "nw": len(words),
        "nl": lexical,
        "ch": characters,
    }


def _worker(payload: tuple[str, list[str]]) -> dict[str, Any] | None:
    """Pool entry point.

    Args:
        payload: The run directory and the family names to keep.

    Returns:
        The row, an error record, or None.
    """
    run_root, families = payload
    try:
        return recording_record(Path(run_root), frozenset(families))
    except Exception as error:  # noqa: BLE001 -- one unreadable store must not stop the sweep
        return {"error": f"{type(error).__name__}: {error}", "run_dir": run_root}


def candidates_under(corpus: Path, families: frozenset[str]) -> list[Path]:
    """The run directories whose stem names a free-response family.

    Args:
        corpus: The run tree.
        families: The family names to keep.

    Returns:
        The directories, in walk order.
    """
    return [run_root for run_root in recording_dirs(corpus) if task_family(task_id_of(stem_of(run_root))) in families]


def _rows(payloads: Sequence[tuple[str, list[str]]], workers: int) -> Iterator[dict[str, Any] | None]:
    """Read every candidate store, in a pool or in this process.

    Args:
        payloads: One ``(run directory, family names)`` per candidate.
        workers: How many processes to read with; one reads serially.

    Yields:
        Each row, error record or None.
    """
    if workers <= 1:
        yield from (_worker(payload) for payload in payloads)
        return
    with Pool(workers) as pool:
        yield from pool.imap_unordered(_worker, list(payloads), chunksize=8)


def extract(corpus: Path, out: Path, workers: int) -> dict[str, Any]:
    """Walk a corpus and write one JSON line per free-response recording.

    Args:
        corpus: The run tree, ``<root>/sub-*/ses-*/<stem>_<timestamp>/``.
        out: The JSONL to write.
        workers: How many processes to read stores with.

    Returns:
        The sweep's counts.
    """
    families = free_response_families()
    candidates = candidates_under(corpus, families)
    counts: Counter[str] = Counter()
    characters = 0
    participants: set[str] = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    payloads = [(str(run_root), sorted(families)) for run_root in candidates]
    with out.open("w") as handle:
        for row in _rows(payloads, workers):
            if row is None:
                counts["skipped"] += 1
                continue
            if "error" in row:
                counts["errors"] += 1
                handle.write(json.dumps(row) + "\n")
                continue
            counts["recordings"] += 1
            counts[f"family:{row['fam']}"] += 1
            counts[f"release:{row['rel'] or 'unrecorded'}"] += 1
            characters += int(row["ch"])
            participants.add(str(row["p"]))
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    return {
        "candidates": len(candidates),
        "participants": len(participants),
        "characters": characters,
        "counts": dict(sorted(counts.items())),
    }


@dataclass
class Corpus:
    """The extract output, folded into what the page needs.

    Attributes:
        participants: Participant id to its recordings, each already in page shape.
        families: Family name to how many recordings carry it.
        releases: Release state to how many recordings carry it.
        categories: PII category to how many marks carry it.
        detectors: Detector name to how many marks it contributed to.
        recordings: How many recordings in total.
        characters: How many transcript characters in total.
        marks: How many reviewable marks in total.
        errors: The rows the sweep could not read.
    """

    participants: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    families: Counter[str] = field(default_factory=Counter)
    releases: Counter[str] = field(default_factory=Counter)
    categories: Counter[str] = field(default_factory=Counter)
    detectors: Counter[str] = field(default_factory=Counter)
    recordings: int = 0
    characters: int = 0
    marks: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        """Fold one extract row in.

        Args:
            row: The row.
        """
        self.participants.setdefault(str(row["p"]), []).append(row)
        self.families[str(row["fam"])] += 1
        self.releases[str(row["rel"]) or "unrecorded"] += 1
        for mark in row.get("f") or []:
            self.marks += 1
            for category in mark["c"]:
                self.categories[str(category)] += 1
            for detector in mark["d"] or ["unattributed"]:
                self.detectors[str(detector)] += 1
        self.recordings += 1
        self.characters += int(row["ch"])


def load(path: Path) -> Corpus:
    """Read an extract file.

    Args:
        path: The JSONL :func:`extract` wrote.

    Returns:
        The folded corpus.
    """
    corpus = Corpus()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "error" in row:
                corpus.errors.append(row)
                continue
            corpus.add(row)
    for rows in corpus.participants.values():
        rows.sort(key=lambda row: (str(row["ses"]), str(row["task"])))
    return corpus


def paragraph(words: Sequence[Sequence[Any]], marks: Sequence[dict[str, Any]]) -> str:
    """One recording's consensus text as marked-up prose.

    Each mark the extract identified becomes one ``<mark>`` carrying its review key and its facets,
    so a two-word name reads and reviews as one finding. Bracketed tokens keep their own class.

    Args:
        words: The word entries, ``[text, bracketed, mark index or -1]``.
        marks: The recording's marks, indexed by the third field of a word entry.

    Returns:
        The HTML for the paragraph's contents.
    """
    pieces: list[str] = []
    index = 0
    while index < len(words):
        owner = int(words[index][2]) if len(words[index]) > 2 else -1
        if owner < 0 or owner >= len(marks):
            pieces.append(_plain(words[index]))
            index += 1
            continue
        cursor = index + 1
        while cursor < len(words) and len(words[cursor]) > 2 and int(words[cursor][2]) == owner:
            cursor += 1
        pieces.append(_mark(marks[owner], words[index:cursor]))
        index = cursor
    return " ".join(pieces)


def _mark(mark: dict[str, Any], run: Sequence[Sequence[Any]]) -> str:
    """One reviewable mark.

    Args:
        mark: The mark record.
        run: The word entries it covers.

    Returns:
        The mark's HTML.
    """
    label = html.escape("+".join(str(name) for name in mark["c"]))
    detectors = html.escape(" ".join(str(name) for name in mark["d"]) or "unattributed")
    inner = " ".join(_plain(item) for item in run)
    return (
        f'<mark class="pii" data-k="{html.escape(str(mark["k"]))}" data-c="{label}" '
        f'data-d="{detectors}" data-brk="{mark["brk"]}" data-tx="{mark["tx"]}" '
        f'data-nt="{mark["nt"]}" data-stim="{mark["stim"]}" tabindex="0">'
        f'<span class="cat">{label}</span>{inner}</mark>'
    )


def _plain(entry: Sequence[Any]) -> str:
    """One word, escaped, with the bracketed convention marked.

    Args:
        entry: The word entry.

    Returns:
        The word's HTML.
    """
    text = html.escape(str(entry[0]))
    return f'<span class="bracket">{text}</span>' if entry[1] else text


def _chip(release: str) -> str:
    """The release state's chip.

    Args:
        release: The release value.

    Returns:
        The chip's HTML.
    """
    value = release or "unrecorded"
    return f'<span class="chip r-{html.escape(value)}">{html.escape(value.replace("_", " "))}</span>'


def recording_html(row: dict[str, Any]) -> str:
    """One recording's card.

    Args:
        row: The extract row.

    Returns:
        The card's HTML.
    """
    marks = row.get("f") or []
    categories = Counter(str(name) for mark in marks for name in mark["c"])
    fired = "yes" if marks else "no"
    summary = ", ".join(f"{html.escape(name)}&times;{count}" for name, count in sorted(categories.items())) or "none"
    ground = row.get("rg")
    ground_html = f'<div class="ground">{html.escape(str(ground))}</div>' if ground else ""
    why = row.get("rwhy") or row.get("why") or ""
    why_html = f'<div class="why">{html.escape(str(why))}</div>' if why else ""
    scan_text = {True: "scanned", False: "scan declined", None: "no scan recorded"}[row.get("scan")]
    body = paragraph(row.get("w") or [], marks) or '<span class="empty">no consensus words</span>'
    stem = str(row.get("stem") or f"{row['p']}_{row['ses']}_task-{row['task']}")
    return (
        f'<article class="rec" data-rel="{html.escape(str(row["rel"]) or "unrecorded")}" '
        f'data-fam="{html.escape(str(row["fam"]))}" data-fired="{fired}" '
        f'data-stem="{html.escape(stem)}" data-nf="{len(marks)}">'
        f'<header><span class="task">{html.escape(str(row["task"]))}</span>'
        f'<span class="fam">{html.escape(str(row["fam"]))}</span>{_chip(str(row["rel"]))}'
        f'<span class="meta">{row["nl"]} lexical / {row["nw"]} tokens &middot; {scan_text} '
        f"&middot; findings: {summary}</span></header>"
        f"{ground_html}{why_html}"
        f'<p class="text">{body}</p>'
        f'<div class="recnote"><label>note on this recording '
        f'<textarea class="rnote" rows="1" data-stem="{html.escape(stem)}"></textarea></label></div>'
        f"</article>"
    )


def participant_html(participant: str, rows: Sequence[dict[str, Any]]) -> str:
    """One participant's section.

    Args:
        participant: The participant id.
        rows: Their recordings, already ordered.

    Returns:
        The section's HTML.
    """
    fired = sum(1 for row in rows if row.get("f"))
    releases = Counter(str(row["rel"]) or "unrecorded" for row in rows)
    tally = " ".join(_chip(name) for name in RELEASE_ORDER if releases.get(name))
    cards = "".join(recording_html(row) for row in rows)
    return (
        f'<section class="participant" id="{html.escape(participant)}" data-p="{html.escape(participant)}">'
        f'<h2><span class="pid">{html.escape(participant)}</span>'
        f'<span class="count">{len(rows)} recordings &middot; {fired} with findings</span>{tally}</h2>'
        f"{cards}</section>"
    )


def render(corpus: Corpus, title: str) -> str:
    """The whole page.

    Args:
        corpus: The folded extract.
        title: The page title.

    Returns:
        The self-contained HTML document.
    """
    order = sorted(corpus.participants)
    sections = "".join(participant_html(participant, corpus.participants[participant]) for participant in order)
    jump = "".join(
        f'<li><a href="#{html.escape(participant)}" data-p="{html.escape(participant)}">'
        f"{html.escape(participant[4:16])}<em>{len(corpus.participants[participant])}</em></a></li>"
        for participant in order
    )
    families = "".join(
        f'<label><input type="checkbox" class="fam-f" value="{html.escape(name)}" checked> '
        f"{html.escape(name)} <em>{count}</em></label>"
        for name, count in sorted(corpus.families.items())
    )
    releases = "".join(
        f'<label><input type="checkbox" class="rel-f" value="{html.escape(name)}" checked> '
        f"{html.escape(name.replace('_', ' '))} <em>{corpus.releases[name]}</em></label>"
        for name in RELEASE_ORDER
        if corpus.releases.get(name)
    )
    categories = "".join(
        f'<label><input type="checkbox" class="cat-f" value="{html.escape(name)}" checked> '
        f"{html.escape(name)} <em>{count}</em></label>"
        for name, count in corpus.categories.most_common()
    )
    detectors = "".join(
        f'<label><input type="checkbox" class="det-f" value="{html.escape(name)}" checked> '
        f"{html.escape(name)} <em>{count}</em></label>"
        for name, count in corpus.detectors.most_common()
    )
    errors = f'<p class="errors">{len(corpus.errors)} stores unreadable</p>' if corpus.errors else ""
    return _DOCUMENT.format(
        title=html.escape(title),
        style=_STYLE,
        script=_SCRIPT,
        participants=len(order),
        recordings=corpus.recordings,
        characters=f"{corpus.characters:,}",
        marks=corpus.marks,
        families=families,
        releases=releases,
        categories=categories,
        detectors=detectors,
        verdicts=_VERDICT_CONTROLS,
        jump=jump,
        sections=sections,
        errors=errors,
    )


VERDICTS = (
    ("identifying", "1", "a real disclosure; the redaction is right"),
    ("not-identifying", "2", "the category fits but nobody is identified"),
    ("not-the-category", "3", "not an instance of the category at all"),
    ("unsure", "4", "needs a second look"),
)
"""The review vocabulary, with its keyboard shortcut and what each verdict claims."""

_VERDICT_CONTROLS = "".join(
    f'<button class="verdict v-{name}" data-v="{name}" title="{html.escape(why)} (key {key})">'
    f"{name}<kbd>{key}</kbd></button>"
    for name, key, why in VERDICTS
)


_STYLE = """
:root{--bg:#fbfaf8;--fg:#1d1c1a;--mut:#6b6860;--line:#e2ded6;--card:#fff;--acc:#7a4b12;
--pii:#fde8c8;--piib:#c98a2b;--brk:#8d8a83;--brkbg:#f0eeea;--catbg:#f7dcb0;--catfg:#7a4b12;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
#wrap{display:grid;grid-template-columns:250px minmax(0,1fr);align-items:start}
#rail{position:sticky;top:0;height:100vh;overflow:auto;border-right:1px solid var(--line);
padding:14px 10px;background:var(--bg)}
#rail h1{font-size:15px;margin:0 0 8px}
#rail .sum{font-size:12px;color:var(--mut);margin-bottom:10px}
#rail input[type=search],#rail select{width:100%;padding:6px 8px;border:1px solid var(--line);
border-radius:6px;font-size:13px;background:var(--card);color:var(--fg)}
#rail fieldset{border:0;padding:0;margin:12px 0 0}
#rail legend{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);
padding:0;margin-bottom:4px}
#rail label{display:block;font-size:12.5px;cursor:pointer;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
#rail label em,#jump em,.cat-chip em{color:var(--mut);font-style:normal;font-size:11px}
#jump{list-style:none;padding:0;margin:6px 0 0;font-size:12px;max-height:40vh;overflow:auto}
#jump a{display:flex;justify-content:space-between;gap:6px;padding:2px 4px;border-radius:4px;
color:var(--fg);text-decoration:none;font-variant-numeric:tabular-nums;
font-family:ui-monospace,Menlo,monospace}
#jump a:hover{background:var(--line)}
main{padding:18px 26px 120px;min-width:0}
.participant{margin:0 0 26px;border-top:1px solid var(--line);padding-top:14px}
.participant h2{font-size:14px;margin:0 0 10px;display:flex;gap:10px;align-items:baseline;
flex-wrap:wrap}
.pid{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.count{color:var(--mut);font-weight:400;font-size:12px}
.rec{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;
margin:0 0 10px;max-width:76ch}
.rec header{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;font-size:12px;
margin-bottom:6px}
.task{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--acc)}
.fam,.meta{color:var(--mut)}
.meta{font-size:11.5px}
.ground,.why{font-size:11.5px;color:var(--mut);font-style:italic;margin:0 0 6px}
.text{margin:0;font-size:16px}
.empty{color:var(--mut);font-style:italic}
.bracket{font-family:ui-monospace,Menlo,monospace;font-size:.82em;color:var(--brk);
background:var(--brkbg);border-radius:3px;padding:0 3px;letter-spacing:.02em}
mark.pii{background:var(--pii);color:var(--fg);border-bottom:2px solid var(--piib);
border-radius:3px;padding:0 2px}
mark.pii .cat{font-size:9.5px;letter-spacing:.06em;color:var(--catfg);background:var(--catbg);
border-radius:3px;padding:0 3px;margin-right:4px;vertical-align:.18em;
font-family:ui-monospace,Menlo,monospace}
.chip{font-size:10.5px;letter-spacing:.04em;padding:1px 7px;border-radius:9px;border:1px solid}
.r-releasable{background:#e7f3e7;border-color:#8fbf8f;color:#2c5c2c}
.r-withheld{background:#fbe6e4;border-color:#d08e86;color:#8a2f24}
.r-nothing_to_redact{background:#eaeef6;border-color:#8fa0c0;color:#2f4670}
.r-not_assessed,.r-unrecorded{background:#f1efe9;border-color:#bdb7a8;color:#6b6350}
.cat-chip{display:inline-block;font-size:11px;background:var(--card);border:1px solid var(--line);
border-radius:9px;padding:1px 7px;margin:0 3px 3px 0}
.errors{color:#8a2f24;font-size:12px}
#status{position:fixed;right:14px;bottom:12px;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:5px 10px;font-size:12px;color:var(--mut)}
.hidden{display:none !important}
mark.pii{cursor:pointer}
mark.pii:focus{outline:2px solid var(--acc);outline-offset:1px}
mark.pii.dim{background:transparent;border-bottom:1px dotted var(--line);opacity:.45}
mark.pii.dim .cat{background:transparent;color:var(--mut)}
mark.pii.sel{box-shadow:0 0 0 2px var(--acc)}
mark.pii[data-v]{border-bottom-width:3px}
mark.pii[data-v="identifying"]{border-bottom-color:#8a2f24}
mark.pii[data-v="not-identifying"]{border-bottom-color:#2f4670}
mark.pii[data-v="not-the-category"]{border-bottom-color:#2c5c2c}
mark.pii[data-v="unsure"]{border-bottom-color:#6b6350;border-bottom-style:dashed}
.recnote{margin-top:8px}
.recnote label{font-size:11px;color:var(--mut);display:block}
.recnote textarea{width:100%;font:inherit;font-size:12.5px;background:var(--bg);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:4px 6px;resize:vertical}
.recnote textarea:placeholder-shown{opacity:.7}
#rail .num{width:58px;padding:3px 5px;border:1px solid var(--line);border-radius:5px;
background:var(--card);color:var(--fg);font-size:12px}
#rail .row{display:flex;gap:6px;align-items:center;font-size:12px;color:var(--mut);margin-top:3px}
#rail .facets{max-height:22vh;overflow:auto}
#rail button{font:inherit;font-size:12px;padding:3px 8px;border:1px solid var(--line);
border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
#panel{position:fixed;right:14px;bottom:44px;width:330px;background:var(--card);
border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:12.5px;
box-shadow:0 6px 24px rgba(0,0,0,.18);z-index:9}
#panel h3{margin:0 0 6px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;
color:var(--mut)}
#panel .facts{color:var(--mut);font-size:11.5px;margin-bottom:6px;word-break:break-word}
#panel .surface{background:var(--pii);border-radius:4px;padding:3px 6px;margin-bottom:8px;
max-height:80px;overflow:auto}
#panel .verdicts{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px}
.verdict{font:inherit;font-size:11.5px;padding:3px 7px;border:1px solid var(--line);
border-radius:6px;background:var(--bg);color:var(--fg);cursor:pointer;display:flex;gap:5px}
.verdict kbd{font-size:9.5px;color:var(--mut);border:1px solid var(--line);border-radius:3px;
padding:0 3px}
.verdict.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.verdict.on kbd{color:#fff;border-color:rgba(255,255,255,.5)}
#panel textarea{width:100%;font:inherit;font-size:12px;background:var(--bg);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:4px 6px;resize:vertical}
#panel .close{float:right;border:0;background:none;color:var(--mut);cursor:pointer;font-size:14px}
#progress{font-size:11.5px;color:var(--mut);margin-top:6px}
#io textarea{width:100%;height:70px;font-family:ui-monospace,Menlo,monospace;font-size:10.5px;
background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
@media (prefers-color-scheme:dark){
:root{--bg:#171614;--fg:#eceae5;--mut:#9a958c;--line:#33312d;--card:#1f1e1b;--acc:#d9a45f;
--pii:#4a3413;--piib:#c08a38;--brk:#a09b91;--brkbg:#2a2825;--catbg:#5f4418;--catfg:#f0d7a8;}
.r-releasable{background:#1d2e1d;border-color:#4f7a4f;color:#a8d3a8}
.r-withheld{background:#331e1b;border-color:#8a4b42;color:#e8a89e}
.r-nothing_to_redact{background:#1c2334;border-color:#4a5c86;color:#a7bce4}
.r-not_assessed,.r-unrecorded{background:#282622;border-color:#5a5449;color:#bdb5a5}
.errors{color:#e8a89e}
mark.pii[data-v="identifying"]{border-bottom-color:#e8a89e}
mark.pii[data-v="not-identifying"]{border-bottom-color:#a7bce4}
mark.pii[data-v="not-the-category"]{border-bottom-color:#a8d3a8}
.verdict.on{color:#171614}
.verdict.on kbd{color:#171614;border-color:rgba(0,0,0,.4)}
}
"""

_SCRIPT = """
const KEY='senselab.fsreview.v1';
const sections=[...document.querySelectorAll('.participant')];
const cards=[...document.querySelectorAll('.rec')];
const marks=[...document.querySelectorAll('mark.pii')];
const jump=[...document.querySelectorAll('#jump a')];
const bar=document.getElementById('status');
const q=document.getElementById('q');
const firedSel=document.getElementById('fired');
const brkSel=document.getElementById('brk');
const txSel=document.getElementById('tx');
const revSel=document.getElementById('rev');
const minNf=document.getElementById('minnf');
const minNt=document.getElementById('minnt');
const maxNt=document.getElementById('maxnt');
const panel=document.getElementById('panel');
const progress=document.getElementById('progress');

/* ---- store: localStorage is a convenience, the export is the record ---- */
let store={findings:{},recordings:{}};
function load(){
  try{
    const raw=localStorage.getItem(KEY);
    if(raw){const parsed=JSON.parse(raw);
      store={findings:parsed.findings||{},recordings:parsed.recordings||{}};}
  }catch(e){store={findings:{},recordings:{}};}
}
function save(){
  try{localStorage.setItem(KEY,JSON.stringify(store));}
  catch(e){bar.textContent='judgment kept in memory only \\u2014 storage unavailable, use Export';}
}
load();

const byKey=new Map();
for(const m of marks)byKey.set(m.dataset.k,m);
const cardOf=new Map();
for(const m of marks)cardOf.set(m,m.closest('.rec'));
const haystack=new Map();
for(const r of cards)haystack.set(r,(r.textContent+' '+r.closest('.participant').dataset.p).toLowerCase());
const marksIn=new Map();
for(const r of cards)marksIn.set(r,[...r.querySelectorAll('mark.pii')]);

function paint(m){
  const rec=store.findings[m.dataset.k];
  if(rec&&rec.v)m.dataset.v=rec.v; else m.removeAttribute('data-v');
}
for(const m of marks)paint(m);
for(const t of document.querySelectorAll('.rnote')){
  const rec=store.recordings[t.dataset.stem];
  if(rec&&rec.n)t.value=rec.n;
  t.placeholder='';
  t.addEventListener('change',()=>{
    const stem=t.dataset.stem;
    if(t.value.trim())store.recordings[stem]={n:t.value,t:new Date().toISOString()};
    else delete store.recordings[stem];
    save();});
}

/* ---- facets ---- */
function checked(cls){
  return new Set([...document.querySelectorAll('.'+cls)].filter(i=>i.checked).map(i=>i.value));}
function allChecked(cls){
  return [...document.querySelectorAll('.'+cls)].every(i=>i.checked);}
function markMatches(m,cats,dets,brk,tx,rev,lo,hi){
  if(!cats.has(m.dataset.c))return false;
  const own=m.dataset.d.split(' ');
  if(!own.some(d=>dets.has(d)))return false;
  if(brk!=='any'&&m.dataset.brk!==brk)return false;
  if(tx!=='any'&&m.dataset.tx!==tx)return false;
  const n=+m.dataset.nt;
  if(n<lo||n>hi)return false;
  if(rev!=='any'){
    const v=(store.findings[m.dataset.k]||{}).v||'';
    if(rev==='unreviewed'){if(v)return false;}
    else if(rev==='reviewed'){if(!v)return false;}
    else if(v!==rev)return false;
  }
  return true;
}
function apply(){
  const needle=q.value.trim().toLowerCase();
  const fams=checked('fam-f'), rels=checked('rel-f');
  const cats=checked('cat-f'), dets=checked('det-f');
  const fired=firedSel.value, brk=brkSel.value, tx=txSel.value, rev=revSel.value;
  const nf=+minNf.value||0;
  const lo=+minNt.value||1, hi=+maxNt.value||9999;
  const narrowed=!allChecked('cat-f')||!allChecked('det-f')||brk!=='any'||tx!=='any'
    ||rev!=='any'||lo>1||hi<9999;
  let shownR=0, shownP=0, shownM=0;
  for(const s of sections){
    let any=false;
    for(const r of s.querySelectorAll('.rec')){
      let ok=fams.has(r.dataset.fam)&&rels.has(r.dataset.rel);
      if(ok&&fired!=='any') ok=r.dataset.fired===fired;
      if(ok&&nf) ok=+r.dataset.nf>=nf;
      if(ok&&needle) ok=haystack.get(r).includes(needle);
      let hits=0;
      if(ok){
        for(const m of marksIn.get(r)){
          const hit=markMatches(m,cats,dets,brk,tx,rev,lo,hi);
          m.classList.toggle('dim',narrowed&&!hit);
          if(hit)hits++;
        }
        if(narrowed&&hits===0)ok=false;
      }
      r.classList.toggle('hidden',!ok);
      if(ok){any=true;shownR++;shownM+=narrowed?hits:marksIn.get(r).length;}
    }
    s.classList.toggle('hidden',!any);
    if(any)shownP++;
  }
  const live=new Set(sections.filter(s=>!s.classList.contains('hidden')).map(s=>s.dataset.p));
  for(const a of jump)a.parentElement.classList.toggle('hidden',!live.has(a.dataset.p));
  bar.textContent=shownP+' participants \\u00b7 '+shownR+' recordings \\u00b7 '+shownM+' findings';
  tally();
}
function tally(){
  let done=0;
  for(const m of marks)if((store.findings[m.dataset.k]||{}).v)done++;
  const notes=Object.keys(store.recordings).length;
  progress.textContent=done+' of '+marks.length+' findings judged \\u00b7 '+notes+' recording notes';
}

/* ---- review panel ---- */
let current=null;
function open(m){
  if(current)current.classList.remove('sel');
  current=m; m.classList.add('sel');
  const rec=store.findings[m.dataset.k]||{};
  panel.hidden=false;
  panel.querySelector('.surface').textContent=m.textContent.slice(m.dataset.c.length);
  panel.querySelector('.facts').textContent=
    m.dataset.c+' \\u00b7 '+(m.dataset.d||'unattributed')+' \\u00b7 '+m.dataset.nt+' token(s)'
    +(m.dataset.brk==='1'?' \\u00b7 touches a bracketed token':'')
    +(m.dataset.tx==='1'?' \\u00b7 inside the task extent':m.dataset.tx==='0'?' \\u00b7 outside the task extent':'')
    +(m.dataset.stim==='1'?' \\u00b7 in the stimulus':'');
  for(const b of panel.querySelectorAll('.verdict'))b.classList.toggle('on',b.dataset.v===rec.v);
  panel.querySelector('textarea').value=rec.n||'';
  panel.querySelector('textarea').focus({preventScroll:true});
}
function setVerdict(v){
  if(!current)return;
  const k=current.dataset.k;
  const prev=store.findings[k]||{};
  if(prev.v===v){delete prev.v;}else{prev.v=v;}
  prev.t=new Date().toISOString();
  prev.c=current.dataset.c; prev.d=current.dataset.d;
  if(!prev.v&&!prev.n)delete store.findings[k]; else store.findings[k]=prev;
  save(); paint(current);
  for(const b of panel.querySelectorAll('.verdict'))
    b.classList.toggle('on',b.dataset.v===(store.findings[k]||{}).v);
  tally();
}
for(const m of marks){
  m.addEventListener('click',e=>{e.preventDefault();open(m);});
  m.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open(m);}});
}
for(const b of panel.querySelectorAll('.verdict'))
  b.addEventListener('click',()=>setVerdict(b.dataset.v));
panel.querySelector('textarea').addEventListener('change',e=>{
  if(!current)return;
  const k=current.dataset.k;
  const prev=store.findings[k]||{};
  if(e.target.value.trim())prev.n=e.target.value; else delete prev.n;
  prev.t=new Date().toISOString();
  prev.c=current.dataset.c; prev.d=current.dataset.d;
  if(!prev.v&&!prev.n)delete store.findings[k]; else store.findings[k]=prev;
  save(); tally();});
panel.querySelector('.close').addEventListener('click',()=>{
  panel.hidden=true; if(current)current.classList.remove('sel'); current=null;});
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  const hit=[...panel.querySelectorAll('.verdict')].find(b=>b.querySelector('kbd').textContent===e.key);
  if(hit&&current){e.preventDefault();setVerdict(hit.dataset.v);}
  if(e.key==='Escape'&&current){panel.hidden=true;current.classList.remove('sel');current=null;}});

/* ---- export and import: the durable artefact ---- */
function payload(){
  return JSON.stringify({schema:'senselab.fsreview',version:1,
    exported:new Date().toISOString(),findings:store.findings,recordings:store.recordings},null,1);
}
document.getElementById('export').addEventListener('click',()=>{
  const text=payload();
  document.getElementById('iotext').value=text;
  try{
    const url=URL.createObjectURL(new Blob([text],{type:'application/json'}));
    const a=document.createElement('a');
    a.href=url; a.download='free-speech-review.json';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(()=>URL.revokeObjectURL(url),2000);
  }catch(e){/* the textarea above is the fallback */}
});
document.getElementById('import').addEventListener('click',()=>{
  const text=document.getElementById('iotext').value.trim();
  if(!text)return;
  try{
    const parsed=JSON.parse(text);
    store={findings:Object.assign({},store.findings,parsed.findings||{}),
           recordings:Object.assign({},store.recordings,parsed.recordings||{})};
    save();
    for(const m of marks)paint(m);
    for(const t of document.querySelectorAll('.rnote')){
      const rec=store.recordings[t.dataset.stem]; if(rec&&rec.n)t.value=rec.n;}
    apply();
  }catch(e){bar.textContent='import failed: that is not the export JSON';}
});
document.getElementById('file').addEventListener('change',e=>{
  const f=e.target.files&&e.target.files[0]; if(!f)return;
  const reader=new FileReader();
  reader.onload=()=>{document.getElementById('iotext').value=String(reader.result);
    document.getElementById('import').click();};
  reader.readAsText(f);
});

for(const el of document.querySelectorAll('.fam-f,.rel-f,.cat-f,.det-f'))
  el.addEventListener('change',apply);
for(const el of [firedSel,brkSel,txSel,revSel,minNf,minNt,maxNt])el.addEventListener('change',apply);
let timer;q.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(apply,140);});
for(const [id,cls] of [['allcat','cat-f'],['nocat','cat-f'],['alldet','det-f'],['nodet','det-f']])
  document.getElementById(id).addEventListener('click',()=>{
    for(const el of document.querySelectorAll('.'+cls))el.checked=id.startsWith('all');
    apply();});
document.getElementById('all').addEventListener('click',e=>{
  e.preventDefault();
  for(const el of document.querySelectorAll('.fam-f,.rel-f,.cat-f,.det-f'))el.checked=true;
  firedSel.value='any';brkSel.value='any';txSel.value='any';revSel.value='any';
  minNf.value='';minNt.value='';maxNt.value='';q.value='';apply();});
apply();
"""

_DOCUMENT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>{title}</title>
<style>{style}</style></head>
<body><div id="wrap">
<nav id="rail">
<h1>{title}</h1>
<div class="sum">{participants} participants &middot; {recordings} recordings &middot;
{marks} findings &middot; {characters} characters</div>
{errors}
<input type="search" id="q" placeholder="search transcripts, tasks, ids">
<fieldset><legend>recording</legend>
<select id="fired"><option value="any">redaction fired or not</option>
<option value="yes">findings only</option><option value="no">no findings</option></select>
<div class="row">at least <input class="num" type="number" id="minnf" min="0" step="1">
findings</div>
</fieldset>
<fieldset><legend>release</legend>{releases}</fieldset>
<fieldset><legend>family</legend>{families}</fieldset>
<fieldset><legend>finding &mdash; category
<button id="allcat" type="button">all</button><button id="nocat" type="button">none</button></legend>
<div class="facets">{categories}</div></fieldset>
<fieldset><legend>finding &mdash; detector
<button id="alldet" type="button">all</button><button id="nodet" type="button">none</button></legend>
<div class="facets">{detectors}</div></fieldset>
<fieldset><legend>finding &mdash; shape</legend>
<select id="brk"><option value="any">bracketed token or not</option>
<option value="1">touches a bracketed token</option>
<option value="0">no bracketed token</option></select>
<select id="tx"><option value="any">task extent, any</option>
<option value="1">inside the task extent</option>
<option value="0">outside the task extent</option>
<option value="-1">no task extent recorded</option></select>
<div class="row">span <input class="num" type="number" id="minnt" min="1" step="1"
placeholder="min"> to <input class="num" type="number" id="maxnt" min="1" step="1"
placeholder="max"> tokens</div>
</fieldset>
<fieldset><legend>my review</legend>
<select id="rev"><option value="any">judged or not</option>
<option value="unreviewed">unjudged only</option><option value="reviewed">judged only</option>
<option value="identifying">identifying</option>
<option value="not-identifying">not-identifying</option>
<option value="not-the-category">not-the-category</option>
<option value="unsure">unsure</option></select>
<div id="progress"></div>
</fieldset>
<fieldset id="io"><legend>export &middot; import</legend>
<div class="row"><button id="export" type="button">Export JSON</button>
<button id="import" type="button">Import</button></div>
<input type="file" id="file" accept="application/json,.json">
<textarea id="iotext" spellcheck="false"
placeholder="the export lands here too; paste an export here and press Import"></textarea>
</fieldset>
<p><a href="#" id="all">reset filters</a></p>
<fieldset><legend>participants</legend><ul id="jump">{jump}</ul></fieldset>
</nav>
<main>{sections}</main>
</div>
<div id="panel" hidden><button class="close" type="button" title="close">&times;</button>
<h3>this finding</h3><div class="facts"></div><div class="surface"></div>
<div class="verdicts">{verdicts}</div>
<textarea rows="2" placeholder="note (optional)"></textarea></div>
<div id="status"></div>
<script>{script}</script></body></html>
"""


def buckets(order: Sequence[str], size: int) -> Iterator[list[str]]:
    """Split participants into buckets of at most ``size``.

    Args:
        order: The participant ids, in page order.
        size: The bucket size.

    Yields:
        Each bucket.
    """
    for start in range(0, len(order), size):
        yield list(order[start : start + size])


def write_pages(corpus: Corpus, out: Path, title: str, shard_size: int) -> dict[str, Any]:
    """Write the page, or a shard per bucket of participants plus an index.

    Args:
        corpus: The folded extract.
        out: The HTML file when unsharded, the directory when sharded.
        title: The page title.
        shard_size: Participants per shard; 0 for one file.

    Returns:
        What was written.
    """
    if not shard_size:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(corpus, title))
        out.chmod(0o600)
        return {"page": str(out), "bytes": out.stat().st_size, "participants": len(corpus.participants)}
    out.mkdir(parents=True, exist_ok=True)
    order = sorted(corpus.participants)
    written: list[dict[str, Any]] = []
    for number, bucket in enumerate(buckets(order, shard_size), start=1):
        part = Corpus()
        for participant in bucket:
            for row in corpus.participants[participant]:
                part.add(row)
        path = out / f"shard-{number:03d}.html"
        path.write_text(render(part, f"{title} {number}"))
        path.chmod(0o600)
        written.append({"file": path.name, "participants": len(bucket), "bytes": path.stat().st_size})
    index = out / "index.html"
    index.write_text(_index_html(title, corpus, written))
    index.chmod(0o600)
    return {"index": str(index), "shards": written}


def _index_html(title: str, corpus: Corpus, written: Iterable[dict[str, Any]]) -> str:
    """The shard index.

    Args:
        title: The page title.
        corpus: The whole folded extract.
        written: The shard records.

    Returns:
        The index document.
    """
    rows = "".join(
        f'<li><a href="{html.escape(str(item["file"]))}">{html.escape(str(item["file"]))}</a> '
        f"&mdash; {item['participants']} participants, {item['bytes']:,} bytes</li>"
        for item in written
    )
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>'
        f"<style>{_STYLE}</style></head><body><main><h1>{html.escape(title)}</h1>"
        f"<p>{len(corpus.participants)} participants, {corpus.recordings} recordings, "
        f"{corpus.characters:,} characters.</p><ul>{rows}</ul></main></body></html>"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description="Free-speech transcript review page.")
    sub = parser.add_subparsers(dest="command", required=True)

    extractor = sub.add_parser("extract", help="walk a corpus and write one JSON line per recording")
    extractor.add_argument("corpus", type=Path)
    extractor.add_argument("--out", type=Path, required=True)
    extractor.add_argument("--workers", type=int, default=8)

    renderer = sub.add_parser("render", help="turn an extract into a self-contained HTML page")
    renderer.add_argument("data", type=Path)
    renderer.add_argument("--out", type=Path, required=True)
    renderer.add_argument("--title", default="Free-speech review")
    renderer.add_argument("--shard-size", type=int, default=0, help="participants per file; 0 for one file")

    sub.add_parser("families", help="print the free-response family names the graph declares")

    arguments = parser.parse_args(argv)
    if arguments.command == "families":
        for name in sorted(free_response_families()):
            print(name)
        return 0
    if arguments.command == "extract":
        print(json.dumps(extract(arguments.corpus, arguments.out, arguments.workers), indent=2))
        return 0
    corpus = load(arguments.data)
    print(json.dumps(write_pages(corpus, arguments.out, arguments.title, arguments.shard_size), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
