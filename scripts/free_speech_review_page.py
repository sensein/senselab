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
from typing import Any, Iterable, Iterator, Mapping, Sequence

from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS, expected_names
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
REDACTION_EXEMPTIONS = "redaction_exemptions"

TASK_EXTENT_ROLE = "task_extent"
"""The ``span`` role the in-family branch mints for the interval it judged the task to occupy."""

SPEECH_FAMILY = "speech"
"""The span family SPEECH stamps; every free-response recording routes to SPEECH."""

MEASUREMENT_MARKER = '"prov_type": "measurement"'
KEPT_MEASUREMENTS = (PII_SCAN, REDACTION_EXEMPTIONS)
KEPT_MEASUREMENT_MARKERS = tuple(f'"name": "{name}"' for name in KEPT_MEASUREMENTS)

RELEASE_ORDER = (
    "release_without_redaction",
    "release_with_redaction",
    "withheld",
    "not_assessed",
    "unrecorded",
)
"""The graph's own release axis, most permissive first, plus the page's own ``unrecorded``."""

EXTRACT_SCHEMA = "senselab.fsreview.extract"
EXTRACT_VERSION = 5
"""5 is the first version read from a graph that has both the release split and REVIEW.

Two changes landed between 3 and here, and each alone would make an older extract describe the
wrong thing. The release axis now names which artefact may be handed on: a version-3 row's ``rel``
is the old vocabulary, in which ``releasable`` means what ``release_with_redaction`` now means and
``nothing_to_redact`` spans both ``release_without_redaction`` and ``not_assessed``, so the chips,
the facet and the determination panel would all name the wrong artefact. And the reviewer is now
REVIEW rather than a step inside REDACT: under 3 it was reached only through REDACT, so it read the
recordings the detectors marked and no others, and ``not_run`` was its state for a transcript the
detectors marked nothing in. REVIEW reads every transcript, so that state is gone and its sentence
would be false twice over -- about which recordings were read, and about why one was not.

Both sides of the merge that brought them together had independently called themselves 4, so 4
names two different graphs and vouches for neither. The version is what lets the page refuse to
speak over an extract whose graph it cannot vouch for.
"""


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
            if MEASUREMENT_MARKER in line and not any(marker in line for marker in KEPT_MEASUREMENT_MARKERS):
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


def pii_scan_of(view: StoreView) -> dict[str, Any] | None:
    """SPEECH's latest live ``pii_scan`` record.

    Args:
        view: The store view.

    Returns:
        Its attributes, or None when the store carries none.
    """
    found: dict[str, Any] | None = None
    for measurement in view.live("measurement"):
        if measurement.attributes.get("name") == PII_SCAN:
            found = dict(measurement.attributes)
    return found


def scan_state(scan: Mapping[str, Any] | None) -> bool | None:
    """Whether SPEECH's PII scan ran over this recording's residue.

    Args:
        scan: The ``pii_scan`` attributes, or None.

    Returns:
        True when some detector ran (``scanned_by`` is non-empty), False when the record shows none
        did, and None when the store carries no record.
    """
    if scan is None:
        return None
    return bool(scan.get("scanned_by"))


def residue_of(scan: Mapping[str, Any] | None, entities: Sequence[Entity]) -> dict[str, Any] | None:
    """The lexical residue the detectors and the reviewer read, as the page shows it.

    Args:
        scan: The ``pii_scan`` attributes, or None.
        entities: The transcript's word entities in order.

    Returns:
        ``n`` residue words, ``m`` the method that set the rest aside, ``c`` whether any residue
        word is content, and ``i`` the residue words' positions in ``entities``; None when the
        store carries no scan record or one written before the residue existed.
    """
    if scan is None or scan.get("residue_method") is None:
        return None
    ids = set(scan.get("residue_word_ids") or ())
    return {
        "n": int(scan.get("residue_words_n") or 0),
        "m": str(scan.get("residue_method")),
        "c": bool(scan.get("residue_content")),
        "i": [position for position, entity in enumerate(entities) if entity.id in ids],
    }


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
                "stim": _mark_stimulus(contributing),
                "tx": _extent_flag(hull, extent),
            }
        )
        index = cursor
    return marks


def _mark_stimulus(contributing: Sequence[Entity]) -> int:
    """How the stimulus question was answered for one mark, keeping "not asked" its own state.

    Tri-state for the same reason the finding is: a boolean here collapses a mark nothing could be
    checked against into one that was checked and did not match.

    Args:
        contributing: The findings attributed to the mark.

    Returns:
        1 when any contributing finding matched, 0 when one was checked and none matched, -1 when
        none was checkable or none contributed.
    """
    states = [tristate(finding.attributes.get("in_stimulus")) for finding in contributing]
    if 1 in states:
        return 1
    return 0 if 0 in states else -1


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
    scan = pii_scan_of(view)
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
        "scan": scan_state(scan),
        "res": residue_of(scan, entities),
        "w": words,
        "f": marks,
        "pii": [
            {
                "c": str(finding.attributes.get("category") or ""),
                "s": str(finding.attributes.get("source") or ""),
                "h": str(finding.attributes.get("haystack") or ""),
                "stim": tristate(finding.attributes.get("in_stimulus")),
            }
            for finding in pii
        ],
        "d": determination(view, attributes, redact),
        "names": len(expected_names(declared)),
        "nw": len(words),
        "nl": lexical,
        "ch": characters,
    }


def tristate(value: Any) -> int:  # noqa: ANN401 -- a store attribute is any type
    """A three-valued store flag as an integer, keeping None apart from False.

    ``in_stimulus`` is None when the task supplies no stimulus text, so there was no haystack to
    check the finding against. That is not the same claim as "checked, and not in the stimulus",
    and collapsing the two is what hides a task-inherent proper noun.

    Args:
        value: The attribute.

    Returns:
        1 for True, 0 for False, -1 for None.
    """
    if value is None:
        return -1
    return 1 if value else 0


def _redact_detail(redact: Entity | None) -> dict[str, Any]:
    """What REDACT itself concluded.

    ``release_ground`` is None exactly when REDACT decided, so on the two states REDACT decides —
    ``release_with_redaction`` and ``withheld`` — this ``why`` is the only account there is. A
    withholding the reviewer made carries its own ground instead.

    Args:
        redact: REDACT's verdict entity, or None when it wrote none.

    Returns:
        Its outcome, reason and the category lists that explain a pass carrying survivors.
    """
    if redact is None:
        return {"outcome": "", "why": ""}
    attributes = redact.attributes
    return {
        "outcome": str(attributes.get("outcome") or ""),
        "why": str(attributes.get("why") or ""),
        "outstanding": sorted(str(name) for name in attributes.get("outstanding") or []),
        "survived": sorted(str(name) for name in attributes.get("survived") or []),
        "expected_survivors": sorted(str(name) for name in attributes.get("expected_survivors") or []),
        "redactions_n": int(attributes.get("redactions_n") or 0),
        "artifacts_withheld": bool(attributes.get("artifacts_withheld")),
    }


def determination(view: StoreView, verdict: Mapping[str, Any], redact: Entity | None) -> dict[str, Any]:
    """Everything that determined the recording's release state.

    Args:
        view: The store view.
        verdict: The live VERDICT entity's attributes.
        redact: REDACT's own verdict entity, or None when it wrote none.

    Returns:
        The account a reader needs: which node decided, what each node concluded, whether the LLM
        reviewer ran, which gates were evaluated against which reading, and what the exemption pass
        had to work with.
    """
    exemptions = view.last("measurement", name=REDACTION_EXEMPTIONS)
    exempt = exemptions.attributes if exemptions is not None else {}
    gates = dict(verdict.get("gates") or {})
    return {
        "redact": _redact_detail(redact),
        "nodes": [
            {
                "node": str(reason.get("node") or ""),
                "outcome": str(reason.get("outcome") or ""),
                "why": str(reason.get("why") or ""),
            }
            for reason in verdict.get("reasons") or []
        ],
        "llm": dict(verdict.get("llm_redaction") or {}),
        "gates": {
            "applied": list(gates.get("applied") or []),
            "flagging": list(gates.get("flagging") or []),
            "bounds": dict(gates.get("bounds") or {}),
            "layers": dict(gates.get("layers") or {}),
            "group": str(gates.get("group") or ""),
        },
        "ran": dict(verdict.get("ran") or {}),
        "absences": sorted(str(name) for name in (verdict.get("critical_absences") or {})),
        "exempt": {
            "declared": bool(exempt.get("expected_speech_declared")),
            "n": int(exempt.get("n") or 0),
            "n_findings": int(exempt.get("n_findings") or 0),
            "recorded": exemptions is not None,
        },
        "stim": _stimulus_tally(view),
    }


def _stimulus_tally(view: StoreView) -> list[int]:
    """How the recording's own findings answered the stimulus question.

    The declaration and the answer came apart once a family could declare a cast: a task with no
    stimulus text can still have its findings checked, against ``expected_names`` or a declared
    carrier. So the panel reports what the check actually returned rather than inferring it from
    whether prompt text was declared.

    Args:
        view: The store view.

    Returns:
        ``[in the stimulus, checked and not in it, not checkable]``.
    """
    tally = [0, 0, 0]
    for finding in view.live("pii"):
        state = tristate(finding.attributes.get("in_stimulus"))
        tally[0 if state == 1 else 1 if state == 0 else 2] += 1
    return tally


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
        handle.write(json.dumps({"schema": EXTRACT_SCHEMA, "version": EXTRACT_VERSION}) + "\n")
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
        "schema": EXTRACT_SCHEMA,
        "version": EXTRACT_VERSION,
        "candidates": len(candidates),
        "participants": len(participants),
        "characters": characters,
        "counts": dict(sorted(counts.items())),
    }


class ValuePool:
    """A deduplicating table of JSON values, so a repeated string costs one integer.

    The determination of a release state is overwhelmingly repetition: nine node names, a few
    dozen distinct ``why`` sentences, one gate specification per group. Emitting it verbatim on
    11,701 cards would cost more than the transcripts do.

    Attributes:
        values: The distinct values, in first-seen order.
    """

    def __init__(self) -> None:
        """Start empty."""
        self.values: list[Any] = []
        self._index: dict[str, int] = {}

    def add(self, value: Any) -> int:  # noqa: ANN401 -- any JSON value may be pooled
        """The index of a value, adding it when it is new.

        Args:
            value: Any JSON-serialisable value.

        Returns:
            Its index in :attr:`values`.
        """
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        found = self._index.get(key)
        if found is None:
            found = len(self.values)
            self._index[key] = found
            self.values.append(value)
        return found


GATE_SPEC_KEYS = ("gate", "reading", "op", "bound", "group", "keyed_under", "layer", "ground")
"""The fields of an evaluated gate that do not vary between recordings."""

UNDETERMINED = "UNDETERMINED"
"""What ``passed`` reads when the gate was applied and could not be answered."""


def gate_state(passed: Any) -> int:  # noqa: ANN401 -- a store attribute is any type
    """A gate's outcome as an integer, keeping "could not be answered" out of "passed".

    ``passed`` is True, False, or the string ``UNDETERMINED`` — a truthy string, so a boolean
    coercion turns an unanswerable gate into a passing one.

    Args:
        passed: The gate record's ``passed`` field.

    Returns:
        1 passed, 0 failed, -1 applied but unanswerable.
    """
    if passed is True:
        return 1
    if passed is False:
        return 0
    return -1


def pooled_determination(determination: Mapping[str, Any], pool: ValuePool) -> dict[str, Any]:
    """One recording's determination, with every repeated part replaced by a pool index.

    Args:
        determination: The row's ``d`` mapping.
        pool: The shared pool.

    Returns:
        The compact record the page reads.
    """
    redact = determination.get("redact") or {}
    gates = determination.get("gates") or {}
    evaluated: list[list[Any]] = []
    for kind in ("applied", "flagging"):
        for gate in gates.get(kind) or []:
            spec = {key: gate.get(key) for key in GATE_SPEC_KEYS if gate.get(key) is not None}
            spec["kind"] = kind
            evaluated.append([pool.add(spec), gate.get("value"), gate_state(gate.get("passed"))])
    return {
        "r": pool.add(redact),
        "n": [
            [pool.add(node.get("node") or ""), pool.add(node.get("outcome") or ""), pool.add(node.get("why") or "")]
            for node in determination.get("nodes") or []
        ],
        "l": pool.add(determination.get("llm") or {}),
        "g": evaluated,
        "b": pool.add(
            {
                "bounds": gates.get("bounds") or {},
                "layers": gates.get("layers") or {},
                "group": gates.get("group") or "",
            }
        ),
        "a": pool.add(determination.get("ran") or {}),
        "x": pool.add(determination.get("absences") or []),
        "e": pool.add(determination.get("exempt") or {}),
        "s": determination.get("stim") or [0, 0, 0],
        "nn": int(determination.get("names") or 0),
        "f": [
            [
                pool.add(finding.get("c") or ""),
                pool.add(finding.get("s") or ""),
                pool.add(finding.get("h") or ""),
                finding.get("stim", -1),
            ]
            for finding in determination.get("findings") or []
        ],
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
        reviewer: Reviewer state to how many recordings carry it.
        recordings: How many recordings in total.
        characters: How many transcript characters in total.
        marks: How many reviewable marks in total.
        version: The extract's schema version; 2 for one written before the header existed.
        errors: The rows the sweep could not read.
    """

    participants: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    families: Counter[str] = field(default_factory=Counter)
    releases: Counter[str] = field(default_factory=Counter)
    categories: Counter[str] = field(default_factory=Counter)
    detectors: Counter[str] = field(default_factory=Counter)
    reviewer: Counter[str] = field(default_factory=Counter)
    recordings: int = 0
    characters: int = 0
    marks: int = 0
    version: int = 2
    errors: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any]) -> None:
        """Fold one extract row in.

        Args:
            row: The row.
        """
        self.participants.setdefault(str(row["p"]), []).append(row)
        self.reviewer[_llm_status(row)] += 1
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
            if row.get("schema") == EXTRACT_SCHEMA:
                corpus.version = int(row.get("version") or EXTRACT_VERSION)
                continue
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
    residue = row.get("res")
    residue_html = ""
    if residue is not None:
        words = row.get("w") or []
        said = " ".join(str(words[position][0]) for position in residue["i"] if position < len(words))
        content = "content" if residue["c"] else "function words only"
        residue_html = (
            f'<div class="residue">residue: {residue["n"]} words ({html.escape(residue["m"])}, {content})'
            + (f" &middot; <q>{html.escape(said)}</q>" if said else "")
            + "</div>"
        )
    body = paragraph(row.get("w") or [], marks) or '<span class="empty">no consensus words</span>'
    stem = str(row.get("stem") or f"{row['p']}_{row['ses']}_task-{row['task']}")
    return (
        f'<article class="rec" data-rel="{html.escape(str(row["rel"]) or "unrecorded")}" '
        f'data-fam="{html.escape(str(row["fam"]))}" data-fired="{fired}" '
        f'data-stem="{html.escape(stem)}" data-nf="{len(marks)}" '
        f'data-llm="{html.escape(_llm_status(row))}">'
        f'<header><span class="task">{html.escape(str(row["task"]))}</span>'
        f'<span class="fam">{html.escape(str(row["fam"]))}</span>{_chip(str(row["rel"]))}'
        f'<span class="meta">{row["nl"]} lexical / {row["nw"]} tokens &middot; {scan_text} '
        f"&middot; findings: {summary}</span></header>"
        f"{ground_html}{why_html}{residue_html}"
        f'<p class="text">{body}</p>'
        f'<div class="whyrow">{_triage_controls(stem)}{_release_controls(stem)}'
        f'<button type="button" class="whybtn" data-stem="{html.escape(stem)}">'
        f"what determined this status</button></div>"
        f'<div class="recnote"><label>note on this recording '
        f'<textarea class="rnote" rows="1" data-stem="{html.escape(stem)}"></textarea></label></div>'
        f"</article>"
    )


ROW_TRIAGE = (("+1", "+", "plus"), ("-1", "-", "minus"), ("flag", "f", "flag"))
"""The owner's own row vocabulary, each with its key and the class that styles it.

Deliberately non-specific. ``flag`` means come back to this, not a fourth quality judgment, and
these are not the finding-level verdicts under another name.
"""


_TRIAGE_GROUP = (
    '<span class="trigroup">'
    + "".join(
        f'<button class="tri t-{slug}" data-v="{html.escape(value)}">{html.escape(value)}'
        f"<kbd>{html.escape(key)}</kbd></button>"
        for value, key, slug in ROW_TRIAGE
    )
    + "</span>"
)
"""The control group, identical on every card; the card's own ``data-stem`` names the row."""


RELEASE_DECISIONS = (
    ("release_without_redaction", "o", "without"),
    ("release_with_redaction", "d", "with"),
)
"""The two releases a reviewer can say a recording warrants, each with its key and its class slug.

The values are the graph's own ``Release`` member values, so an exported decision joins to a verdict
without a mapping between them. ``specs/20260924-which-artefact-is-releasable/design.md``.
"""

_DECISION_GROUP = (
    '<span class="decgroup">'
    + "".join(
        f'<button class="dec d-{slug}" data-v="{html.escape(value)}">'
        f"{html.escape(slug)} redaction<kbd>{html.escape(key)}</kbd></button>"
        for value, key, slug in RELEASE_DECISIONS
    )
    + "</span>"
)
"""The release-decision group, identical on every card; the card's own ``data-stem`` names the row."""


def _release_controls(stem: str) -> str:
    """The per-recording release-decision buttons.

    Args:
        stem: The recording's BIDS stem, carried by the enclosing card rather than repeated.

    Returns:
        The control group's HTML.
    """
    return _DECISION_GROUP


def _triage_controls(stem: str) -> str:
    """The per-recording triage buttons.

    Args:
        stem: The recording's BIDS stem, carried by the enclosing card rather than repeated.

    Returns:
        The control group's HTML.
    """
    return _TRIAGE_GROUP


LLM_STATES = ("disabled", "nothing_to_read", "absent", "clean", "flagged")
"""Every state the reviewer records, in the order the ladder reaches them.

``disabled`` is the config leaving it off; ``nothing_to_read`` is a transcript with no words in it;
the last three are readings it actually took. ``not_run`` retired when the reviewer stopped being
reached through the detectors: it meant "the detectors marked nothing", which is now a population
that gets read rather than one that gets skipped.
"""

LLM_RAN = ("absent", "clean", "flagged")
"""The states in which the reviewer was actually invoked. Only these are a reading."""


def _llm_status(row: Mapping[str, Any]) -> str:
    """The reviewer's state for one recording.

    Args:
        row: The extract row.

    Returns:
        One of :data:`LLM_STATES`, or ``unrecorded`` when REDACT left no annotation.
    """
    status = str(((row.get("d") or {}).get("llm") or {}).get("status") or "")
    return status if status in LLM_STATES else "unrecorded"


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
        f'<span class="jp">{html.escape(participant[4:16])}</span>'
        f'<span class="meter"><i class="mr"></i><i class="mj"></i></span>'
        f'<em class="jn">{len(corpus.participants[participant])}</em></a></li>'
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
    if corpus.recordings and not any(corpus.reviewer.get(state) for state in LLM_RAN):
        states = ", ".join(f"{name} {count}" for name, count in corpus.reviewer.most_common())
        errors += (
            f'<p class="noreview"><b>The LLM reviewer did not run on any of these '
            f"{corpus.recordings} recordings</b> ({states}). Nothing here was corroborated or "
            f"contradicted by a reviewer, and no reviewer verdict exists to read. A pass with "
            f"<code>redaction.llm_check.enabled</code> set and a GPU is what would produce one.</p>"
        )
    if corpus.version < EXTRACT_VERSION:
        errors += (
            f'<p class="errors">This extract is version {corpus.version}, written before the '
            f"release axis and the reviewer ladder changed. What the page says about which artefact "
            f"may be handed on, and about the LLM reviewer, does not describe the run that produced "
            f"it. Re-extract before reading either.</p>"
        )
    pool = ValuePool()
    rows: dict[str, Any] = {}
    for participant in order:
        for row in corpus.participants[participant]:
            determination = dict(row.get("d") or {})
            determination["findings"] = row.get("pii") or []
            determination["names"] = row.get("names") or 0
            rows[str(row.get("stem") or "")] = pooled_determination(determination, pool)
    why = json.dumps({"pool": pool.values, "rows": rows}, separators=(",", ":"))
    return _DOCUMENT.format(
        why=why.replace("</", "<\\/"),
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
html{overflow-x:hidden}
#wrap{display:grid;grid-template-columns:minmax(190px,230px) minmax(0,1fr);align-items:start;
max-width:100%}
#rail{position:sticky;top:0;height:100dvh;overflow:auto;border-right:1px solid var(--line);
padding:14px 10px;background:var(--bg);min-width:0}
#rail h1{font-size:15px;margin:0}
.railhead{display:flex;gap:8px;align-items:baseline;justify-content:space-between;
margin-bottom:8px}
#railtoggle{display:none}
#keys{margin:0 0 6px;font-size:12px}
#keys summary{cursor:pointer;color:var(--mut);font-size:11px;text-transform:uppercase;
letter-spacing:.06em}
#keys dl{display:grid;grid-template-columns:auto minmax(0,1fr);gap:1px 8px;margin:6px 0 0}
#keys dt{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--acc);
white-space:nowrap}
#keys dd{margin:0;font-size:11.5px;color:var(--mut)}
#keys .note{font-size:11px;margin:6px 0 0}
#outsum{font-weight:400;text-transform:none;letter-spacing:0}
#rail .sum{font-size:12px;color:var(--mut);margin-bottom:10px}
#rail input[type=search],#rail select{width:100%;padding:6px 8px;border:1px solid var(--line);
border-radius:6px;font-size:13px;background:var(--card);color:var(--fg)}
#rail fieldset{border:0;padding:0;margin:12px 0 0}
#rail legend{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);
padding:0;margin-bottom:4px}
#rail label{display:block;font-size:12.5px;cursor:pointer;white-space:nowrap;overflow:hidden;
text-overflow:ellipsis}
#rail label em,#jump em,.cat-chip em{color:var(--mut);font-style:normal;font-size:11px}
#jump{list-style:none;padding:0;margin:6px 0 0;font-size:12px;max-height:38dvh;overflow:auto}
#jump a{display:grid;grid-template-columns:minmax(0,1fr) 34px auto;gap:6px;align-items:center;
padding:2px 4px;border-radius:4px;color:var(--fg);text-decoration:none;
font-variant-numeric:tabular-nums}
#jump a:hover{background:var(--line)}
#jump a.here{background:var(--line);outline:1px solid var(--acc)}
.jp{font-family:ui-monospace,Menlo,monospace;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap}
.meter{position:relative;height:6px;border-radius:3px;background:var(--line);overflow:hidden}
.meter i{position:absolute;top:0;bottom:0;left:0;width:0}
.meter .mr{background:var(--acc);opacity:.85}
.meter .mj{background:#2c5c2c;opacity:.9;top:3px}
.jn{font-size:11px;color:var(--mut);font-style:normal;min-width:2ch;text-align:right}
main{padding:18px 26px 140px;min-width:0;max-width:100%}
main *{overflow-wrap:anywhere}
.participant{margin:0 0 26px;border-top:1px solid var(--line);padding-top:14px}
.participant h2{font-size:14px;margin:0 0 10px;display:flex;gap:10px;align-items:baseline;
flex-wrap:wrap}
.pid{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}
.count{color:var(--mut);font-weight:400;font-size:12px}
.rec{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;
margin:0 0 10px;max-width:76ch;scroll-margin-top:12px;scroll-margin-bottom:96px}
.rec.active{outline:2px solid var(--acc);outline-offset:1px}
.rec header{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;font-size:12px;
margin-bottom:6px}
.task{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--acc)}
.fam,.meta{color:var(--mut)}
.meta{font-size:11.5px}
.ground,.why,.residue{font-size:11.5px;color:var(--mut);font-style:italic;margin:0 0 6px}
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
.r-release_without_redaction{background:#e7f3e7;border-color:#8fbf8f;color:#2c5c2c}
.r-withheld{background:#fbe6e4;border-color:#d08e86;color:#8a2f24}
.r-release_with_redaction{background:#eaeef6;border-color:#8fa0c0;color:#2f4670}
.r-not_assessed,.r-unrecorded{background:#f1efe9;border-color:#bdb7a8;color:#6b6350}
.cat-chip{display:inline-block;font-size:11px;background:var(--card);border:1px solid var(--line);
border-radius:9px;padding:1px 7px;margin:0 3px 3px 0}
.errors{color:#8a2f24;font-size:12px}
.noreview{font-size:11.5px;background:var(--pii);border-left:3px solid var(--piib);
padding:6px 8px;border-radius:4px;margin:6px 0}
.noreview code{font-family:ui-monospace,Menlo,monospace;font-size:10.5px}
#status{position:fixed;right:14px;bottom:12px;max-width:calc(100vw - 28px);background:var(--card);
border:1px solid var(--line);border-radius:8px;padding:5px 10px;font-size:12px;color:var(--mut);
z-index:8}
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
#panel{position:fixed;right:14px;bottom:44px;width:min(330px,calc(100vw - 28px));
max-height:60dvh;overflow:auto;background:var(--card);
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
.whyrow{margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.trigroup{display:inline-flex;gap:3px}
.tri{font:inherit;font-size:11.5px;padding:2px 7px;border:1px solid var(--line);border-radius:6px;
background:var(--bg);color:var(--mut);cursor:pointer;display:inline-flex;gap:4px;align-items:center}
.tri kbd{font-size:9px;color:var(--mut);border:1px solid var(--line);border-radius:3px;padding:0 3px}
.tri:hover{color:var(--fg);border-color:var(--acc)}
.tri.on{color:#fff;border-color:transparent}
.tri.on kbd{color:#fff;border-color:rgba(255,255,255,.5)}
.t-plus.on{background:#2c5c2c}
.t-minus.on{background:#8a2f24}
.t-flag.on{background:#7a4b12}
.rec[data-t="+1"]{border-left:3px solid #2c5c2c}
.rec[data-t="-1"]{border-left:3px solid #8a2f24}
.rec[data-t="flag"]{border-left:3px solid #c98a2b}

.decgroup{display:inline-flex;gap:4px;margin-right:8px}
.dec{font:inherit;font-size:11.5px;padding:2px 9px;border:1px solid var(--line);
border-radius:6px;background:var(--bg);color:var(--mut);cursor:pointer}
.dec kbd{font-size:9.5px;opacity:.6;margin-left:5px}
.dec:hover{color:var(--fg);border-color:var(--acc)}
.dec.on{color:#fff;border-color:transparent}
.d-without.on{background:#2c5c2c}
.d-with.on{background:#2f4670}
.rec[data-dec="release_without_redaction"]{border-right:3px solid #2c5c2c}
.rec[data-dec="release_with_redaction"]{border-right:3px solid #2f4670}
.whybtn{font:inherit;font-size:11.5px;padding:2px 9px;border:1px solid var(--line);
border-radius:6px;background:var(--bg);color:var(--mut);cursor:pointer}
.whybtn:hover{color:var(--fg);border-color:var(--acc)}
#why{position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);width:min(760px,94vw);
max-height:86dvh;overflow:auto;background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:14px 18px;font-size:13px;box-shadow:0 10px 40px rgba(0,0,0,.28);z-index:20}
#why .tw{overflow-x:auto;max-width:100%}
#why h3{margin:0 0 4px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut)}
#why h4{margin:14px 0 4px;font-size:12px;letter-spacing:.03em;color:var(--mut);
text-transform:uppercase}
#why .decisive{font-size:14px;margin:6px 0 2px}
#why .decisive b{font-weight:600}
#why table{border-collapse:collapse;width:100%;min-width:380px;font-size:12px}
#why th{text-align:left;font-weight:500;color:var(--mut);border-bottom:1px solid var(--line);
padding:3px 6px 3px 0}
#why td{padding:3px 6px 3px 0;border-bottom:1px solid var(--line);vertical-align:top}
#why tr.off td{color:var(--mut);font-style:italic}
#why .ok{color:#2c5c2c}
#why .bad{color:#8a2f24}
#why .neutral{color:var(--mut)}
#why .note{font-size:12px;color:var(--mut);margin:4px 0 0}
#why .warn{background:var(--pii);border-left:3px solid var(--piib);padding:6px 9px;
border-radius:4px;margin:8px 0;font-size:12.5px}
#scrim{position:fixed;inset:0;background:rgba(0,0,0,.35);z-index:19}
#io textarea{width:100%;max-width:100%;height:70px;font-family:ui-monospace,Menlo,monospace;
font-size:10.5px;background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:6px}
#io input[type=file]{max-width:100%;font-size:11px}

/* One column once the rail and a readable measure no longer both fit. */
@media (max-width:820px){
#wrap{grid-template-columns:minmax(0,1fr)}
#rail{position:static;height:auto;max-height:none;overflow:visible;border-right:0;
border-bottom:1px solid var(--line)}
#railtoggle{display:inline-block}
#railbody{display:none}
#rail.open #railbody{display:block}
#jump{max-height:46dvh}
main{padding:14px 16px 150px}
.rec{max-width:100%}
#panel{left:8px;right:8px;bottom:52px;width:auto}
#why{width:96vw;padding:12px 14px}
}
@media (max-width:420px){
main{padding:12px 10px 120px}
.rec{padding:10px 11px}
/* Stay a corner pill rather than a full-width band: a fixed bar spanning the
   column covers a whole line of transcript at every scroll position. */
#status{left:auto;right:8px;bottom:8px;font-size:11px;padding:4px 8px;opacity:.95}
}
#status{pointer-events:none}
/* A pointer that cannot hover gets larger hit targets. */
@media (hover:none){
.tri,.dec,.whybtn,.verdict{padding:6px 10px}
mark.pii{padding:1px 3px}
}
@media (prefers-reduced-motion:reduce){*{scroll-behavior:auto !important}}
@media (prefers-color-scheme:dark){
:root{--bg:#171614;--fg:#eceae5;--mut:#9a958c;--line:#33312d;--card:#1f1e1b;--acc:#d9a45f;
--pii:#4a3413;--piib:#c08a38;--brk:#a09b91;--brkbg:#2a2825;--catbg:#5f4418;--catfg:#f0d7a8;}
.r-release_without_redaction{background:#1d2e1d;border-color:#4f7a4f;color:#a8d3a8}
.r-withheld{background:#331e1b;border-color:#8a4b42;color:#e8a89e}
.r-release_with_redaction{background:#1c2334;border-color:#4a5c86;color:#a7bce4}
.r-not_assessed,.r-unrecorded{background:#282622;border-color:#5a5449;color:#bdb5a5}
.rec[data-dec="release_without_redaction"]{border-right-color:#4f7a4f}
.rec[data-dec="release_with_redaction"]{border-right-color:#4a5c86}
.d-without.on{background:#3a6b3a}
.d-with.on{background:#3d5687}
.errors{color:#e8a89e}
#why .ok{color:#a8d3a8}
#why .bad{color:#e8a89e}
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
const llmSel=document.getElementById('llm');
const minNf=document.getElementById('minnf');
const minNt=document.getElementById('minnt');
const maxNt=document.getElementById('maxnt');
const panel=document.getElementById('panel');
const progress=document.getElementById('progress');

/* ---- store: localStorage is a convenience, the export is the record ---- */
let store={findings:{},recordings:{},triage:{},release:{}};
function load(){
  try{
    const raw=localStorage.getItem(KEY);
    if(raw){const parsed=JSON.parse(raw);
      store={findings:parsed.findings||{},recordings:parsed.recordings||{},
             triage:parsed.triage||{},release:parsed.release||{}};}
  }catch(e){store={findings:{},recordings:{},triage:{},release:{}};}
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
/* what a search reads: the recording, never the controls beside it. A button labelled in the
   page's own vocabulary would otherwise match every card. */
function haystackOf(card){
  const parts=[card.closest('.participant').dataset.p];
  for(const selector of ['header','.ground','.why','.text'])
    {const el=card.querySelector(selector); if(el)parts.push(el.textContent);}
  return parts.join(' ').toLowerCase();
}
const haystack=new Map();
for(const r of cards)haystack.set(r,haystackOf(r));
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
    save();tally();});
}

/* ---- row triage: the owner's +1 / -1 / flag, one per recording ---- */
const triSel=document.getElementById('tri');
const TRIKEYS={'+':'+1','=':'+1','-':'-1','f':'flag','F':'flag'};
function paintRow(card){
  const rec=store.triage[card.dataset.stem];
  const value=rec&&rec.v;
  if(value)card.dataset.t=value; else card.removeAttribute('data-t');
  for(const b of card.querySelectorAll('.tri'))b.classList.toggle('on',b.dataset.v===value);
}
function setRow(card,value){
  const stem=card.dataset.stem;
  const held=(store.triage[stem]||{}).v;
  if(held===value)delete store.triage[stem];
  else store.triage[stem]={v:value,t:new Date().toISOString()};
  save();paintRow(card);tally();outline();
  if(triSel.value!=='any')apply();
}
/* ---- the release decision: which artefact this reviewer would hand on ---- */
const decSel=document.getElementById('dec');
const DECKEYS={'o':'release_without_redaction','O':'release_without_redaction',
               'd':'release_with_redaction','D':'release_with_redaction'};
function paintDecision(card){
  const rec=store.release[card.dataset.stem];
  const value=rec&&rec.v;
  if(value)card.dataset.dec=value; else card.removeAttribute('data-dec');
  for(const b of card.querySelectorAll('.dec'))b.classList.toggle('on',b.dataset.v===value);
}
function setDecision(card,value){
  const stem=card.dataset.stem;
  const held=(store.release[stem]||{}).v;
  if(held===value)delete store.release[stem];
  else store.release[stem]={v:value,t:new Date().toISOString()};
  save();paintDecision(card);tally();
  if(decSel.value!=='any')apply();
}
let activeCard=null;
let pointerOwns=true;
function markActive(card,scroll){
  if(activeCard&&activeCard!==card)activeCard.classList.remove('active');
  activeCard=card;
  if(!card)return;
  card.classList.add('active');
  if(scroll)card.scrollIntoView({block:'center',behavior:'smooth'});
  here(card.closest('.participant'));
}
document.addEventListener('mousemove',()=>{pointerOwns=true;},{passive:true});
for(const card of cards){
  paintRow(card);
  paintDecision(card);
  card.addEventListener('mouseenter',()=>{if(pointerOwns)markActive(card,false);});
  card.addEventListener('focusin',()=>markActive(card,false));
  for(const b of card.querySelectorAll('.tri'))
    b.addEventListener('click',()=>{markActive(card,false);setRow(card,b.dataset.v);});
  for(const b of card.querySelectorAll('.dec'))
    b.addEventListener('click',()=>{markActive(card,false);setDecision(card,b.dataset.v);});
}

/* ---- moving between samples, following the filters ---- */
function visibleCards(){
  return cards.filter(c=>!c.classList.contains('hidden')
    &&!c.closest('.participant').classList.contains('hidden'));
}
function step(delta){
  const live=visibleCards();
  if(!live.length)return;
  pointerOwns=false;
  let at=activeCard?live.indexOf(activeCard):-1;
  if(at<0){
    /* not on a visible card: enter the list at whichever end the reader is moving toward */
    markActive(delta>0?live[0]:live[live.length-1],true);
    return;
  }
  const next=Math.min(live.length-1,Math.max(0,at+delta));
  markActive(live[next],true);
}
function stepParticipant(delta){
  const live=visibleCards();
  if(!live.length)return;
  pointerOwns=false;
  const current=activeCard?activeCard.closest('.participant'):null;
  const order=[];
  for(const c of live){const s=c.closest('.participant');if(order[order.length-1]!==s)order.push(s);}
  let at=current?order.indexOf(current):-1;
  if(at<0){markActive(delta>0?live[0]:live[live.length-1],true);return;}
  const target=order[Math.min(order.length-1,Math.max(0,at+delta))];
  markActive(live.find(c=>c.closest('.participant')===target),true);
}
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  if(e.key==='j'){e.preventDefault();step(1);}
  else if(e.key==='k'){e.preventDefault();step(-1);}
  else if(e.key==='J'){e.preventDefault();stepParticipant(1);}
  else if(e.key==='K'){e.preventDefault();stepParticipant(-1);}
});
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const value=TRIKEYS[e.key];
  if(!value)return;
  const card=activeCard||(e.target.closest&&e.target.closest('.rec'));
  if(!card)return;
  e.preventDefault();setRow(card,value);
});
document.addEventListener('keydown',e=>{
  if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.metaKey||e.ctrlKey||e.altKey)return;
  const value=DECKEYS[e.key];
  if(!value)return;
  const card=activeCard||(e.target.closest&&e.target.closest('.rec'));
  if(!card)return;
  e.preventDefault();setDecision(card,value);
});

/* ---- the outline ---- */
const jumpOf=new Map();
for(const a of jump)jumpOf.set(a.dataset.p,a);
const sectionCards=new Map();
for(const s of sections)sectionCards.set(s.dataset.p,[...s.querySelectorAll('.rec')]);
const sectionMarks=new Map();
for(const s of sections)sectionMarks.set(s.dataset.p,[...s.querySelectorAll('mark.pii')]);
let hereP=null;
function here(section){
  const key=section&&section.dataset.p;
  if(hereP===key)return;
  const was=hereP&&jumpOf.get(hereP);
  if(was)was.classList.remove('here');
  hereP=key;
  const now=key&&jumpOf.get(key);
  if(now){now.classList.add('here');
    if(now.scrollIntoView)now.scrollIntoView({block:'nearest'});}
}
function outline(){
  let seen=0,markedRows=0,judged=0,totalMarks=0;
  for(const s of sections){
    const key=s.dataset.p;
    const own=sectionCards.get(key)||[];
    const shown=own.filter(c=>!c.classList.contains('hidden')).length;
    const marked=own.filter(c=>c.dataset.t).length;
    const own2=sectionMarks.get(key)||[];
    const done=own2.filter(m=>m.hasAttribute('data-v')).length;
    seen+=shown;markedRows+=marked;judged+=done;totalMarks+=own2.length;
    const a=jumpOf.get(key);
    if(!a)continue;
    a.querySelector('.jn').textContent=shown;
    a.querySelector('.mr').style.width=own.length?(100*marked/own.length)+'%':'0';
    a.querySelector('.mj').style.width=own2.length?(100*done/own2.length)+'%':'0';
  }
  const out=document.getElementById('outsum');
  if(out)out.textContent='\\u2014 '+markedRows+' rows marked, '+judged+' findings judged, '
    +seen+' shown';
}

/* ---- facets ---- */
function checked(cls){
  return new Set([...document.querySelectorAll('.'+cls)].filter(i=>i.checked).map(i=>i.value));}
function allChecked(cls){
  return [...document.querySelectorAll('.'+cls)].every(i=>i.checked);}
function markMatches(m,cats,dets,brk,tx,rev,lo,hi){
  if(!m.dataset.c.split('+').some(c=>cats.has(c)))return false;
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
  const tri=triSel.value, llmWant=llmSel.value, dec=decSel.value;
  const nf=+minNf.value||0;
  const lo=+minNt.value||1, hi=+maxNt.value||9999;
  const narrowed=!allChecked('cat-f')||!allChecked('det-f')||brk!=='any'||tx!=='any'
    ||rev!=='any'||lo>1||hi<9999;
  let shownR=0, shownP=0, shownM=0;
  for(const s of sections){
    let any=false;
    for(const r of s.querySelectorAll('.rec')){
      let ok=fams.has(r.dataset.fam)&&rels.has(r.dataset.rel);
      if(ok&&tri!=='any'){
        const held=r.dataset.t||'';
        ok=tri==='marked'?!!held:tri==='unmarked'?!held:held===tri;
      }
      if(ok&&dec!=='any'){
        const said=r.dataset.dec||'';
        ok=dec==='decided'?!!said:dec==='undecided'?!said:said===dec;
      }
      if(ok&&llmWant!=='any'){
        const st=r.dataset.llm||'';
        ok=llmWant==='ran'?(st==='absent'||st==='clean'||st==='flagged'):st===llmWant;
      }
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
  if(activeCard&&(activeCard.classList.contains('hidden')
    ||activeCard.closest('.participant').classList.contains('hidden'))){
    activeCard.classList.remove('active');activeCard=null;here(null);
  }
  outline();
  tally();
}
function tally(){
  let done=0;
  for(const m of marks)if((store.findings[m.dataset.k]||{}).v)done++;
  /* every term counts the cards on this page. One localStorage namespace spans every shard, so
     counting the store's own keys can report more decisions than there are rows. */
  let notes=0, rows=0, said=0;
  for(const c of cards){
    const stem=c.dataset.stem;
    if((store.recordings[stem]||{}).n)notes++;
    if((store.triage[stem]||{}).v)rows++;
    if((store.release[stem]||{}).v)said++;
  }
  progress.textContent=done+' of '+marks.length+' findings judged \\u00b7 '+rows+' of '
    +cards.length+' rows marked \\u00b7 '+said+' of '+cards.length
    +' rows given a release decision \\u00b7 '+notes+' recording notes';
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
    +(m.dataset.stim==='1'?' \\u00b7 in the stimulus'
      :m.dataset.stim==='0'?' \\u00b7 checked, not in the stimulus'
      :' \\u00b7 no stimulus to check against');
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
  tally();outline();
  if(revSel.value!=='any')apply();
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
  return JSON.stringify({schema:'senselab.fsreview',version:3,
    exported:new Date().toISOString(),findings:store.findings,recordings:store.recordings,
    triage:store.triage,release:store.release},null,1);
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
           recordings:Object.assign({},store.recordings,parsed.recordings||{}),
           triage:Object.assign({},store.triage,parsed.triage||{}),
           release:Object.assign({},store.release,parsed.release||{})};
    save();
    for(const m of marks)paint(m);
    for(const card of cards){paintRow(card);paintDecision(card);}
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

/* ---- what determined this status ---- */
let WHY=null;
try{WHY=JSON.parse(document.getElementById('whydata').textContent);}catch(e){WHY=null;}
const whyBox=document.getElementById('why');
function esc(s){const d=document.createElement('div');d.textContent=String(s);return d.innerHTML;}
function P(i){return WHY&&WHY.pool[i];}
function scrimOn(){
  if(document.getElementById('scrim'))return;
  const s=document.createElement('div');s.id='scrim';
  s.addEventListener('click',closeWhy);document.body.appendChild(s);
}
function closeWhy(){
  whyBox.hidden=true;
  const s=document.getElementById('scrim');if(s)s.remove();
}
whyBox.querySelector('.close').addEventListener('click',closeWhy);

function stimWord(v){
  if(v===1)return '<span class="neutral">in the stimulus</span>';
  if(v===0)return 'not in the stimulus';
  return '<span class="neutral">no stimulus to check</span>';
}
function buildWhy(stem,card){
  const r=WHY&&WHY.rows[stem];
  if(!r)return '<p class="note">this recording carries no recorded determination.</p>';
  const out=[];
  const rel=card.dataset.rel;
  const rd=P(r.r)||{};
  const ground=card.querySelector('.ground');

  out.push('<p class="decisive">release is <b>'+esc(rel.replace(/_/g,' '))+'</b>');
  if(rd.outcome)out.push(' \\u2014 <b>REDACT decided it</b>, returning <b>'+esc(rd.outcome)+'</b>');
  else if(ground)out.push(' \\u2014 <b>the fold decided it</b>: '+esc(ground.textContent));
  out.push('</p>');
  if(rd.outcome){
    out.push('<p>'+esc(rd.why)+'</p>');
    out.push('<p class="note">A release ground is recorded only when the fold decides. REDACT '
      +'decided here, so the ground is empty by construction and this sentence is the account.</p>');
    const bits=[];
    if(rd.redactions_n!=null)bits.push(esc(rd.redactions_n)+' redaction(s) planned');
    if(rd.outstanding&&rd.outstanding.length)
      bits.push('still found after redacting: <b>'+esc(rd.outstanding.join(', '))+'</b>');
    if(rd.expected_survivors&&rd.expected_survivors.length)
      bits.push('survived but accounted for by the stimulus: '+esc(rd.expected_survivors.join(', ')));
    if(bits.length)out.push('<p class="note">'+bits.join(' \\u00b7 ')+'</p>');
  }else if(!ground){
    out.push('<p class="note">no release ground was recorded and REDACT left no verdict.</p>');
  }

  /* the LLM reviewer: never let "did not run" read as "agreed" */
  const llm=P(r.l)||{};
  const st=llm.status||'';
  const beside=llm.detector_outcome
    ?' It was taken beside a detector <b>'+esc(llm.detector_outcome)+'</b>.':'';
  const DETECTORS={scanned:'the detectors read this transcript',
    declined:'the detectors never read this transcript: the scan was declined because every '
      +'lexical word is in the task\\'s own stimulus',
    unscanned:'no detector ever reached this transcript'};
  const reached=DETECTORS[llm.detector_state]
    ?'<p class="note">Detectors: '+esc(DETECTORS[llm.detector_state])+'.</p>':'';
  out.push('<h4>the LLM reviewer</h4>');
  if(st==='disabled'){
    out.push('<div class="warn"><b>Switched off.</b> The reviewer is disabled in the '
      +'configuration, so no review was attempted and none of what follows was corroborated by '
      +'one. This is not a verdict about the recording.</div>');
  }else if(st==='nothing_to_read'){
    out.push('<div class="warn"><b>There was no text to read.</b> The reviewer is enabled, and '
      +'this recording\\'s transcript carries no words, so there was nothing to read back. '
      +'It reached no conclusion about this recording.</div>');
  }else if(st==='absent'){
    out.push('<div class="warn"><b>It tried and could not load.</b> That is not the same as '
      +'finding nothing, and it is not a clean reading. '+(llm.failure?esc(llm.failure):'')
      +'</div>');
  }else if(st==='clean'){
    out.push('<p><b>It ran and flagged nothing</b>, over '+esc(llm.iterations||0)+' iteration(s)'
      +(llm.model_id?', model '+esc(llm.model_id):'')+'.'+beside+'</p>');
    if(llm.detector_outcome==='fail')
      out.push('<p class="note">A clean reading beside a detector failure is a disagreement, not a '
        +'confirmation: the reviewer read the redacted text and saw nothing the detectors still '
        +'saw.</p>');
    out.push(reached);
  }else if(st==='flagged'){
    /* A reading is flagged on any of three grounds, and only one of them names categories: the
       category list comes from the proposal's removal entries. A reading that judges the words to
       carry pii without proposing a span is flagged with an empty list, and rendering a bare
       count of it says nothing about why. Name the grounds that actually fired. */
    const cats=llm.flagged||[];
    const why=[];
    if(llm.original==='carries_pii')why.push('the words themselves carry something identifying');
    if(llm.redaction==='incomplete')why.push('the applied redaction did not remove what identifies the speaker');
    if(llm.proposal_redact_n)why.push('it proposes '+esc(llm.proposal_redact_n)+' further removal(s)');
    out.push('<p><b>It flagged this recording</b>'
      +(cats.length?' on '+esc(cats.length)+': '+esc(cats.join(', ')):'')
      +' \\u2014 over '+esc(llm.iterations||0)+' iteration(s)'
      +(llm.model_id?', model '+esc(llm.model_id):'')+'.'+beside
      +(llm.failure?' '+esc(llm.failure):'')+'</p>');
    if(why.length)
      out.push('<p class="note">Why: '+why.join('; ')+'.</p>');
    else
      out.push('<p class="warn">It is flagged, and none of the three readings says why. '
        +'That is a gap in the record, not a clean result.</p>');
    if(llm.detector_state==='declined'||llm.detector_state==='unscanned')
      out.push('<p class="note">It flagged a transcript no detector read. That is a reading about '
        +'the scan gate rather than about the detectors, and it is the only check on that gate.</p>');
    out.push(reached);
    if(llm.proposal_redact_n)
      out.push('<p class="note">It would remove '+esc(llm.proposal_redact_n)
        +' the detectors left in place.</p>');
    if(llm.proposal_release_n)
      out.push('<p class="note">It would stop removing '+esc(llm.proposal_release_n)
        +' of what is currently removed.</p>');
  }else{
    out.push('<div class="warn"><b>No annotation was recorded.</b> REVIEW left no reviewer '
      +'measurement, so nothing is known about whether a review happened.</div>');
  }
  /* Speakers is a third question, answered independently of the two about pii. It used to be
     rendered only inside the flagged branch, so a clean reading that had noticed a second voice
     said nothing about it — which is the one case the note exists for. */
  if(llm.speakers==='more_than_one')
    out.push('<p class="note"><b>It reads the words as showing more than one person speaking</b> in '
      +'this recording. A reading of the transcript, not of the audio.</p>');
  else if(llm.speakers==='unclear')
    out.push('<p class="note">It could not tell from the words whether more than one person '
      +'speaks here.</p>');

  /* gates: evaluated, and declared-but-never-evaluated */
  const evaluated=r.g||[], profile=P(r.b)||{bounds:{},layers:{}};
  const seen=new Set(evaluated.map(g=>(P(g[0])||{}).gate));
  out.push('<h4>gates</h4><div class="tw"><table><tr><th>gate</th><th>reading</th>'
    +'<th>value</th><th>bound</th><th>outcome</th></tr>');
  for(const g of evaluated){
    const spec=P(g[0])||{};
    const val=(typeof g[1]==='number')?(Math.round(g[1]*1000)/1000):g[1];
    const state=g[2];
    const cell=state===1?'<span class="ok">passed</span>'
      :state===0?'<span class="bad">FAILED</span>'
      :'<span class="neutral">could not be answered</span>';
    const missing=state===-1
      ?(g[1]==null?' \\u2014 nothing measured the reading':(spec.bound==null
        ?' \\u2014 nobody has measured the bound':'')):'';
    out.push('<tr'+(state===-1?' class="off"':'')+'><td>'+esc(spec.gate)
      +(spec.kind==='flagging'?' <span class="neutral">(flagging)</span>':'')
      +'</td><td>'+esc(spec.reading||'\\u2014')+'</td><td>'+(g[1]==null?'\\u2014':esc(val))
      +'</td><td>'+esc((spec.op||'').replace(/_/g,' '))+' '
      +(spec.bound==null?'\\u2014':esc(spec.bound))
      +'</td><td>'+cell+missing+'</td></tr>');
  }
  let unevaluated=0;
  for(const name of Object.keys(profile.bounds||{})){
    if(seen.has(name))continue;
    unevaluated++;
    out.push('<tr class="off"><td>'+esc(name)+'</td><td colspan="3">declared for this group, '
      +'never evaluated \\u2014 no reading was available</td><td>\\u2014</td></tr>');
  }
  out.push('</table></div>');
  if(!evaluated.length)
    out.push('<p class="note">No gate was evaluated on this recording'
      +(unevaluated?' \\u2014 the branch that owns this family left no in-family report, so its '
        +'conformance is undetermined rather than failed.':'.')+'</p>');
  const failed=evaluated.filter(g=>g[2]===0).length;
  const unanswered=evaluated.filter(g=>g[2]===-1).length;
  out.push('<p class="note">'+evaluated.length+' evaluated ('+failed+' failed, '+unanswered
    +' unanswerable), '+unevaluated+' declared but never evaluated. A gate that was never '
    +'evaluated did not pass and did not fail.</p>');

  /* what each node concluded, and what ran */
  const ran=P(r.a)||{};
  out.push('<h4>nodes</h4><div class="tw"><table><tr><th>node</th><th>state</th>'
    +'<th>outcome</th><th>why</th></tr>');
  const named=new Set();
  for(const n of r.n||[]){
    const name=P(n[0]);named.add(name);
    const outcome=P(n[1]);
    out.push('<tr><td>'+esc(name)+'</td><td>'+esc(ran[name]||'\\u2014')+'</td><td class="'
      +(outcome==='pass'?'ok':outcome==='fail'?'bad':'neutral')+'">'+esc(outcome)
      +'</td><td>'+esc(P(n[2]))+'</td></tr>');
  }
  for(const name of Object.keys(ran)){
    if(named.has(name))continue;
    out.push('<tr class="off"><td>'+esc(name)+'</td><td>'+esc(ran[name])
      +'</td><td>\\u2014</td><td>ran, but folded no verdict of its own</td></tr>');
  }
  out.push('</table></div>');
  const absences=P(r.x)||[];
  if(absences.length)
    out.push('<p class="note">critical absences: '+esc(absences.join(', '))+'</p>');

  /* the findings themselves */
  const findings=r.f||[];
  out.push('<h4>findings that drove it ('+findings.length+')</h4>');
  if(!findings.length){
    out.push('<p class="note">none.</p>');
  }else{
    out.push('<div class="tw"><table><tr><th>category</th><th>detector</th>'
      +'<th>read from</th><th>stimulus check</th></tr>');
    for(const f of findings)
      out.push('<tr><td>'+esc(P(f[0]))+'</td><td>'+esc(P(f[1]))+'</td><td>'+esc(P(f[2]))
        +'</td><td>'+stimWord(f[3])+'</td></tr>');
    out.push('</table></div>');
  }

  /* the stimulus, which is the third artefact family */
  const ex=P(r.e)||{};
  const stim=r.s||[0,0,0];
  const checked=stim[0]+stim[1];
  const sources=[];
  if(ex.declared)sources.push('the declared prompt text');
  if(r.nn)sources.push('the task\\u2019s declared cast of '+esc(r.nn)+' name(s)');
  out.push('<h4>the stimulus check</h4>');
  if(!ex.recorded){
    out.push('<p class="note">No exemption pass was recorded.</p>');
  }else{
    if(checked||stim[2]){
      out.push('<p>Of '+(checked+stim[2])+' finding(s): <b>'+stim[0]+'</b> matched what the task '
        +'itself supplies, <b>'+stim[1]+'</b> were checked and did not match, and <b>'+stim[2]
        +'</b> could not be checked at all.</p>');
    }
    if(sources.length)
      out.push('<p class="note">Checked against '+sources.join(' and ')+'.</p>');
    if(stim[2]&&!sources.length)
      out.push('<div class="warn">This task supplies <b>no stimulus text and no declared cast</b>, '
        +'so nothing could be checked. A proper noun belonging to the task itself is '
        +'indistinguishable here from one the participant disclosed.</div>');
    else if(stim[2])
      out.push('<div class="warn">'+stim[2]+' finding(s) could not be checked: what the task '
        +'declares does not reach them.</div>');
    out.push('<p class="note">'+esc(ex.n)+' of '+esc(ex.n_findings)+' finding(s) were exempted as '
      +'expected speech.</p>');
  }
  return out.join('');
}
for(const b of document.querySelectorAll('.whybtn')){
  b.addEventListener('click',()=>{
    whyBox.querySelector('.whybody').innerHTML=buildWhy(b.dataset.stem,b.closest('.rec'));
    whyBox.hidden=false;scrimOn();
  });
}
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!whyBox.hidden)closeWhy();});

for(const el of document.querySelectorAll('.fam-f,.rel-f,.cat-f,.det-f'))
  el.addEventListener('change',apply);
for(const el of [firedSel,brkSel,txSel,revSel,triSel,decSel,llmSel,minNf,minNt,maxNt])
  el.addEventListener('change',apply);
let timer;q.addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(apply,140);});
for(const [id,cls] of [['allcat','cat-f'],['nocat','cat-f'],['alldet','det-f'],['nodet','det-f']])
  document.getElementById(id).addEventListener('click',()=>{
    for(const el of document.querySelectorAll('.'+cls))el.checked=id.startsWith('all');
    apply();});
document.getElementById('all').addEventListener('click',e=>{
  e.preventDefault();
  for(const el of document.querySelectorAll('.fam-f,.rel-f,.cat-f,.det-f'))el.checked=true;
  firedSel.value='any';brkSel.value='any';txSel.value='any';revSel.value='any';triSel.value='any';
  decSel.value='any';llmSel.value='any';
  minNf.value='';minNt.value='';maxNt.value='';q.value='';apply();});
const rail=document.getElementById('rail');
const railToggle=document.getElementById('railtoggle');
railToggle.addEventListener('click',()=>{
  const open=rail.classList.toggle('open');
  railToggle.setAttribute('aria-expanded',open?'true':'false');
});
if(window.matchMedia&&window.matchMedia('(min-width:821px)').matches)rail.classList.add('open');
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
<div class="railhead"><h1>{title}</h1>
<button id="railtoggle" type="button" aria-expanded="true">filters</button></div>
<div id="railbody">
<div class="sum">{participants} participants &middot; {recordings} recordings &middot;
{marks} findings &middot; {characters} characters</div>
{errors}
<details id="keys"><summary>keyboard</summary><dl>
<dt>j / k</dt><dd>next / previous sample</dd>
<dt>J / K</dt><dd>next / previous participant</dd>
<dt>+ or = / - / f</dt><dd>mark the row +1 / -1 / flag</dd>
<dt>o / d</dt><dd>release this recording without / with redaction</dd>
<dt>1 &ndash; 4</dt><dd>judge the selected finding</dd>
<dt>Esc</dt><dd>close a panel</dd>
</dl><p class="note">Movement follows the filters, and the card it lands on is what the row-mark
and release keys act on.</p></details>
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
<select id="llm"><option value="any">LLM reviewer, any state</option>
<option value="ran">it actually ran</option>
<option value="flagged">it flagged something</option>
<option value="clean">it ran and flagged nothing</option>
<option value="absent">it could not load</option>
<option value="nothing_to_read">there was no text to read</option>
<option value="disabled">switched off</option>
</select>
<select id="tri"><option value="any">row mark, any</option>
<option value="marked">marked</option><option value="unmarked">unmarked</option>
<option value="+1">+1</option><option value="-1">-1</option><option value="flag">flag</option>
</select>
<select id="dec"><option value="any">release decision, any</option>
<option value="decided">decided</option><option value="undecided">undecided</option>
<option value="release_without_redaction">release without redaction</option>
<option value="release_with_redaction">release with redaction</option>
</select>
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
<fieldset id="outline"><legend>outline <span id="outsum"></span></legend>
<ul id="jump">{jump}</ul></fieldset>
</div></nav>
<main>{sections}</main>
</div>
<script type="application/json" id="whydata">{why}</script>
<div id="why" hidden><button class="close" type="button" title="close">&times;</button>
<h3>what determined this status</h3><div class="whybody"></div></div>
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


def census(path: Path) -> dict[str, Any]:
    """What one extract holds, in the terms a rebuild is judged by.

    Args:
        path: The JSONL :func:`extract` wrote.

    Returns:
        Counts only: totals, and the same totals per family.
    """
    totals: Counter[str] = Counter()
    families: dict[str, Counter[str]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema") == EXTRACT_SCHEMA:
                continue
            if "error" in row:
                totals["unreadable"] += 1
                continue
            family = str(row["fam"])
            per = families.setdefault(family, Counter())
            for counter in (totals, per):
                counter["recordings"] += 1
                counter["findings"] += len(row.get("pii") or [])
                counter["marks"] += len(row.get("f") or [])
                counter[f"release:{row['rel'] or 'unrecorded'}"] += 1
                counter[f"llm:{_llm_status(row)}"] += 1
            for finding in row.get("pii") or []:
                key = {1: "in_stimulus", 0: "not_in_stimulus", -1: "unchecked"}[int(finding.get("stim", -1))]
                totals[key] += 1
                per[key] += 1
            for mark in row.get("f") or []:
                if mark.get("brk"):
                    totals["bracketed_marks"] += 1
                    per["bracketed_marks"] += 1
                    for category in mark["c"]:
                        totals[f"bracketed:{category}"] += 1
    return {
        "totals": dict(sorted(totals.items())),
        "families": {k: dict(sorted(v.items())) for k, v in sorted(families.items())},
    }


def compare(before: Path, after: Path) -> dict[str, Any]:
    """What a rebuild changed, against the extract the current page was built from.

    Args:
        before: The earlier extract.
        after: The rebuilt one.

    Returns:
        Each count in both, with its delta; families present in either.
    """
    old_census, new_census = census(before), census(after)

    def rows(left: Mapping[str, int], right: Mapping[str, int]) -> dict[str, list[int]]:
        return {
            key: [left.get(key, 0), right.get(key, 0), right.get(key, 0) - left.get(key, 0)]
            for key in sorted(set(left) | set(right))
        }

    return {
        "totals": rows(old_census["totals"], new_census["totals"]),
        "families": {
            family: rows(old_census["families"].get(family, {}), new_census["families"].get(family, {}))
            for family in sorted(set(old_census["families"]) | set(new_census["families"]))
        },
    }


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

    census_parser = sub.add_parser("census", help="what one extract holds, in counts")
    census_parser.add_argument("data", type=Path)

    compare_parser = sub.add_parser("compare", help="what a rebuild changed against an earlier extract")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)

    arguments = parser.parse_args(argv)
    if arguments.command == "families":
        for name in sorted(free_response_families()):
            print(name)
        return 0
    if arguments.command == "extract":
        print(json.dumps(extract(arguments.corpus, arguments.out, arguments.workers), indent=2))
        return 0
    if arguments.command == "census":
        print(json.dumps(census(arguments.data), indent=2))
        return 0
    if arguments.command == "compare":
        print(json.dumps(compare(arguments.before, arguments.after), indent=2))
        return 0
    corpus = load(arguments.data)
    print(json.dumps(write_pages(corpus, arguments.out, arguments.title, arguments.shard_size), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
