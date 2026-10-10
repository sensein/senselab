"""The REDACT node: every PII finding padded, merged, masked with the declared fill, and verified.

The last step of the SPEECH branch, run only where SPEECH found PII. The scan it plans from reads the
recording's lexical residue only -- the words that are neither non-lexical nor what the task asked
for (``residue.py``, and ``nodes/speech.py`` step 7) -- and runs only where that residue is not
empty. A store carrying findings but no scan measurement is refused rather than concluded over.

Every non-invalidated ``pii`` entity is redacted regardless of speaker, except one the declared
stimulus accounts for (:func:`_expected_exemptions`, recorded as an ``exempt``/``expected_speech``
assertion) and one every word of which is the task's own content read off its events
(:func:`task_content`, an ``exempt``/``task_event`` assertion). Extents are padded by
``redaction.padding_ms`` and merged by ``plan_redactions``, then filled with ``redaction.fill`` at
``redaction.bleep_hz`` when that is a bleep. A word carries its PII marking through a live
``assertion`` whose ``verb`` is ``"label"`` and ``label`` is ``"pii"``.

No recognizer runs here: verification is a re-scan of the redacted residue, :func:`residue_words`,
with the same detectors, judged complete by ``pii.required_detectors``. A
surviving finding is a fail, an incomplete re-scan is a flag, and a finding the verifier still
sees is re-planned exactly once, what survives that being ``unremediable``; ``audio_check`` is
the constant ``"bounded"`` on every path.
The LLM reviewer is not here. It reads the same residue, whether or not REDACT was reached, so it is
its own node: see ``nodes/review.py``. :func:`transcript_texts` is what it reads, rendered by the
same renderer that writes the released pair.

A pass releases three artifacts under ``artifacts_dir``: the masked audio, the flat redacted
transcript, and the redacted consensus stream as ``consensus.json``, whose records carry each
surviving word's timings, readings, variants and agreement and fold each planned extent into one
placeholder with its category, bounds and word count but no surface. One renderer produces both the
records and the flat text. The masked audio is also a store stream, written under ``run_dir`` and
registered as ``redacted`` on every path.

See ``specs/20260817-triage-workflow-dag/redact.md`` and ``llm-check.md``, and
``specs/20260919-pii-against-the-stimulus/design.md``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Collection, Mapping, Sequence

import yaml

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.redaction.api import RedactionExtent, apply_redactions, plan_redactions
from senselab.audio.workflows.triage.cohort import COHORT, CONDITION_KINDS, OTHER, load_cohort_profile
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import (
    branch_params,
    declared_carrier,
    declared_task_family,
    expected_names,
)
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    cache_attributes,
    consensus_words,
    find_measurement,
    find_verdict,
    live_entities,
    path_attributes,
    resolve_stream,
    software_agent,
    word_hull,
    write_stream,
    write_verdict,
)
from senselab.audio.workflows.triage.residue import is_content_word, is_name_homograph, is_proper_form
from senselab.audio.workflows.triage.stimulus import NearMatch, near_match, split_prompts
from senselab.audio.workflows.triage.task_content import task_content_ids, task_events
from senselab.audio.workflows.triage.task_lexicon import TaskLexicon, declared_names_lexicon, task_lexicon
from senselab.audio.workflows.triage.task_speech import (
    TASK_SPEECH_READING,
    item_set_family,
    task_speech_parameters,
    words_min_for,
)
from senselab.audio.workflows.triage.vocabulary import (
    REDACTION_LLM_ANNOTATION,
    UNPLACED_CLEARED,
    UNPLACED_OPEN,
    UNPLACED_PLACED,
    UNPLACED_UNREAD,
    Outcome,
    Release,
    reviewer_named_no_words,
)
from senselab.text.tasks.pii_detection.api import scan_for_pii
from senselab.text.tasks.pii_detection.redaction_policy import (
    age_positions,
    country_runs,
    date_positions,
    is_number,
    is_state_word,
    kinship_positions,
    state_positions,
)
from senselab.text.tasks.pii_detection.redaction_policy import fold as policy_fold
from senselab.text.tasks.pii_detection.redaction_policy import policy as policy_tables
from senselab.text.tasks.pii_detection.redaction_policy import version as policy_version
from senselab.text.tasks.pii_detection.redaction_review import (
    PLACE_HISTORICAL,
    PLACE_REASONS,
    RELABEL_PLACE,
    RELABELS,
    review_inputs_complete,
    safe_harbor_codes,
)
from senselab.utils.prov_store import Entity, ProvStore
from senselab.utils.tasks.cached_inference import annotate_result_origin

NODE = "REDACT"

_FILL_KEY = "redaction.fill"
_BLEEP_HZ_KEY = "redaction.bleep_hz"
_PADDING_KEY = "redaction.padding_ms"
_REQUIRED_DETECTORS_KEY = "pii.required_detectors"
_LABEL_VERB = "label"  # the store's assertion verb for a label
_PII_LABEL = "pii"  # the marking SPEECH places on a word carrying a finding
_RESERVED_CATEGORY_CHAR = "+"  # plan_redactions' merge separator
_UNPLACED_PLACEHOLDER = "[UNPLACED]"  # a word the store places nowhere; a category-less placeholder
_AUDIO_CHECK = "bounded"  # what a text re-scan can claim about the audio, on every path
STREAM_NAME = "redacted"  # the store-held name the masked audio resolves under, beside plain/enhanced/residual
CONSENSUS_ARTIFACT_SCHEMA = "senselab.triage.redacted_consensus"  # what the JSON artifact claims to be
CONSENSUS_ARTIFACT_VERSION = 1  # bumped when a record's field set changes
_TERMINATORS_KEY = "stimulus.sentence_terminators"
_EXEMPT_VERB = "exempt"  # the store's assertion verb for a redaction not made
_EXPECTED_LABEL = "expected_speech"  # what accounted for it: the stimulus the participant was asked to read
TASK_EVENT_LABEL = "task_event"  # what accounted for it: the task's own events, which the words lie on
_EXEMPTION_MEASUREMENT = "redaction_exemptions"


@dataclass(frozen=True)
class RedactResult(NodeResult):
    """What REDACT returns.

    Attributes:
        artifacts: The released paths, ``{"audio": ..., "transcript": ..., "consensus": ...}``;
            empty on anything but a pass.
    """

    artifacts: dict[str, Path]


@dataclass(frozen=True)
class _Verification:
    """What re-scanning the redacted consensus text established.

    Attributes:
        verified: Whether the re-scan ran completely and found nothing.
        survived: The categories found on the redacted text, sorted; never matched text.
        scan_ran: Whether a complete re-scan happened at all — an empty ``failures`` is not evidence
            that one did.
        failed: Detectors the verification re-scan attempted and that raised. Names only.
        missing: Detectors ``pii.required_detectors`` names that the verification re-scan never
            attempted. Reported in the verdict separately from the planning scan's own
            ``scan_failed`` and ``scan_missing``.
        cache: The result-cache use of the re-scan (:func:`cache_attributes`), None when it did not
            reach the detectors.
        spans: Each surviving span as ``(category, text)``, its placeholders removed; held in memory to
            place it on words, never written.
    """

    verified: bool
    survived: list[str]
    scan_ran: bool
    failed: list[str]
    missing: list[str]
    cache: dict[str, Any] | None = None
    spans: tuple[tuple[str, str], ...] = ()


def padding_ms(config: TriageConfig) -> int:
    """The redaction margin, in whole milliseconds.

    Args:
        config: The triage configuration.

    Returns:
        The margin.

    Raises:
        ValueError: If ``redaction.padding_ms`` has no value, is not a number, is not finite, is not
            integral, or is negative.
    """
    raw = config.require(_PADDING_KEY)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{_PADDING_KEY} must be a number of milliseconds, not {type(raw).__name__}")
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise ValueError(f"{_PADDING_KEY} must be finite, got {raw!r}")
        if raw != int(raw):
            raise ValueError(f"{_PADDING_KEY} must be a whole number of milliseconds, got {raw!r}")
    value = int(raw)
    if value < 0:
        raise ValueError(f"{_PADDING_KEY} must be >= 0, got {value}; a negative margin narrows every extent")
    return value


_NAME_APPROVALS_KEY = "redaction.name_approvals"


def name_approvals(config: TriageConfig, stem: str) -> tuple[str, ...]:
    """The person names a human approved for release in one recording.

    Args:
        config: The triage configuration; ``redaction.name_approvals`` maps a recording's file stem to
            the names approved in it, each quoted as the transcript writes it.
        stem: The recording's file stem.

    Returns:
        The approved names, in the order the configuration lists them; empty where none.
    """
    table = config.get(_NAME_APPROVALS_KEY) or {}
    if not isinstance(table, Mapping):
        raise ValueError(f"{_NAME_APPROVALS_KEY} must map a recording's file stem to a list of names")
    return tuple(str(name) for name in table.get(stem) or ())


def _scan_evidence(scan: Entity, required: list[str]) -> tuple[list[str], list[str], list[str]]:
    """What the store's ``pii_scan`` measurement says about its own completeness.

    Args:
        scan: The measurement entity.
        required: The detector set ``pii.required_detectors`` names.

    Returns:
        ``(scanned_by, failed, missing)`` — the detector names that ran, those that were attempted
        and failed, and those required but never attempted, each sorted. Names only.
    """
    scanned_by = sorted(str(name) for name in scan.attributes.get("scanned_by") or [])
    failed = sorted(str(name) for name in scan.attributes.get("failed") or [])
    missing = sorted(set(required) - set(scanned_by) - set(failed))
    return scanned_by, failed, missing


def _findings(store: ProvStore) -> list[Entity]:
    """The ``pii`` entities still standing, by the store's latest-non-invalidated rule.

    Args:
        store: The provenance store.

    Returns:
        The non-invalidated findings.
    """
    return live_entities(store, "pii")


def _extents_from_findings(findings: list[Entity]) -> list[RedactionExtent]:
    """Every finding, regardless of speaker, as a redaction extent.

    Args:
        findings: The live ``pii`` entities.

    Returns:
        One extent per finding.

    Raises:
        ValueError: If a finding's category is empty or carries the reserved merge character, or if
            a finding has no extent. The error names bounds and category, never any matched text.
    """
    extents = []
    for finding in findings:
        category = finding.attributes.get("category", "")
        if not category or _RESERVED_CATEGORY_CHAR in category:
            raise ValueError(
                f"category {category!r} is empty or contains the reserved merge character; "
                "it cannot be planned without silently decomposing on re-planning"
            )
        if finding.extent is None:
            raise ValueError(f"pii finding {finding.id} has no extent; nothing locatable can be redacted")
        extents.append(RedactionExtent(start=finding.extent[0], end=finding.extent[1], category=category))
    return extents


def _verification_text(records: list[dict[str, Any]]) -> str:
    """The redacted transcript as the re-scan reads it, the bracketed tokens dropped.

    See ``specs/20260922-brackets-are-not-speech/design.md``.

    Args:
        records: :func:`_render`'s records.

    Returns:
        The join of every released surface and placeholder except a bracketed consensus word's.
    """
    kept = [record for record in records if not (record["kind"] == "word" and record["bracketed"])]
    return " ".join(token for token in (_token(record) for record in kept) if token)


def _mask_tokens(records: list[dict[str, Any]]) -> tuple[str, ...]:
    """Every placeholder the rendered text carries, as a detector may echo it back.

    Args:
        records: :func:`_render`'s records.

    Returns:
        Each placeholder with and without its brackets, and each category a merged placeholder
        joins, longest first.
    """
    tokens: set[str] = set()
    for record in records:
        if record["kind"] == "word":
            continue
        token = str(record["token"])
        bare = token.strip("[]")
        tokens.update({token, bare, *bare.split(_RESERVED_CATEGORY_CHAR)})
    return tuple(sorted((token for token in tokens if token), key=len, reverse=True))


def _is_mask(text: str, masks: Sequence[str]) -> bool:
    """Whether a re-scan span is a placeholder and nothing the recording said.

    Args:
        text: The span's text.
        masks: :func:`_mask_tokens`' output for the scanned text.

    Returns:
        True when no word character remains once every placeholder is removed.
    """
    rest = text
    for mask in masks:
        rest = rest.replace(mask, " ")
    return not any(ch.isalnum() for ch in rest)


def _verify(records: list[dict[str, Any]], required: list[str]) -> _Verification:
    """Re-scan the redacted residue with the same detectors; no recognizer runs.

    A span that is a placeholder the plan wrote is the redaction itself and does not survive.

    Args:
        records: :func:`_render`'s records over the residue, under the plan being verified.
        required: The detector set ``pii.required_detectors`` names.

    Returns:
        What the re-scan established. A re-scan that skipped a required detector counts as not
        having run.
    """
    scan = scan_for_pii(_verification_text(records))
    scan = scan[0] if isinstance(scan, list) else scan
    cache = cache_attributes(scan.cache)
    missing = sorted(set(required) - set(scan.detectors_used) - set(scan.failures))
    failed = sorted(scan.failures)
    if failed or not scan.detectors_used or missing:
        return _Verification(verified=False, survived=[], scan_ran=False, failed=failed, missing=missing, cache=cache)
    masks = _mask_tokens(records)
    kept = [span for span in scan.spans if not _is_mask(str(span.text or ""), masks)]
    survived = sorted({span.category for span in kept})
    spans = tuple((str(span.category), _without_masks(str(span.text or ""), masks)) for span in kept)
    return _Verification(
        verified=not survived, survived=survived, scan_ran=True, failed=[], missing=[], cache=cache, spans=spans
    )


def _without_masks(text: str, masks: Sequence[str]) -> str:
    """A re-scan span's text with every placeholder the plan wrote removed.

    Args:
        text: The span's text.
        masks: :func:`_mask_tokens`' output for the scanned text.

    Returns:
        The words the recording said, single-spaced.
    """
    for mask in masks:
        text = text.replace(mask, " ")
    return " ".join(text.split())


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two extents share any temporal intersection > 0.

    Args:
        a: One extent.
        b: The other.

    Returns:
        True when they intersect.
    """
    return a[0] < b[1] and a[1] > b[0]


def _pii_marking_assertions(store: ProvStore) -> list[Entity]:
    """Every live ``label``/``pii`` assertion.

    Args:
        store: The provenance store.

    Returns:
        The assertion entities, oldest first.
    """
    return [
        assertion
        for assertion in live_entities(store, "assertion")
        if assertion.attributes.get("verb") == _LABEL_VERB and assertion.attributes.get("label") == _PII_LABEL
    ]


def _pii_marked_words(store: ProvStore) -> dict[str, dict[str, str]]:
    """Which PII categories the store's live label assertions place on each word, and by which one.

    Args:
        store: The provenance store.

    Returns:
        A mapping from word entity id to ``{category: assertion id}``; a word nothing marks is
        absent. Where two assertions place the same category on one word the earlier is retained.
    """
    marked: dict[str, dict[str, str]] = {}
    for assertion in _pii_marking_assertions(store):
        category = str(assertion.attributes.get("category") or "")
        if not category:
            continue
        for source_id in store.derived_from(assertion.id):
            marked.setdefault(source_id, {}).setdefault(category, assertion.id)
    return marked


def _word_record(word: Entity) -> dict[str, Any]:
    """One surviving consensus word, with its times and its source agreement.

    Args:
        word: A live ``word`` entity a planned extent does not reach.

    Returns:
        The record. Only the fields named here are copied; the store's element id and the entity's
        raw attribute mapping are not among them.
    """
    attributes = word.attributes
    hull = word_hull(word)
    extent = word.extent
    return {
        "kind": "word",
        "index": int(attributes["index"]),
        "text": str(attributes.get("text") or ""),
        "bracketed": bool(attributes.get("bracketed")),
        "outcome": attributes.get("outcome"),
        "start_s": None if extent is None else float(extent[0]),
        "end_s": None if extent is None else float(extent[1]),
        "hull_start_s": hull[0],
        "hull_end_s": hull[1],
        "agreement": float(attributes["agreement"]) if attributes.get("agreement") is not None else None,
        "sources": sorted(str(name) for name in attributes.get("sources") or []),
        "readings": {str(name): str(text) for name, text in (attributes.get("readings") or {}).items()},
        "timings": {
            str(name): [float(span[0]), float(span[1])] for name, span in (attributes.get("timings") or {}).items()
        },
        "variants": [
            {
                "text": str(variant.get("text") or ""),
                "sources": sorted(str(name) for name in variant.get("sources") or []),
                "share": float(variant["share"]) if variant.get("share") is not None else None,
            }
            for variant in attributes.get("variants") or []
        ],
        "onset_spread_s": attributes.get("onset_spread_s"),
        "offset_spread_s": attributes.get("offset_spread_s"),
        "temporal_uncertainty_s": attributes.get("temporal_uncertainty_s"),
    }


def _render(
    words: list[Entity], planned: list[RedactionExtent], owners: Mapping[str, int] | None = None
) -> tuple[list[dict[str, Any]], str, int]:
    """The redacted consensus stream, as records and as the flat text derived from them.

    Words are released in stream order. Every word an extent hides folds into one ``redaction``
    record emitted at the first such position; a word the store places nowhere becomes an
    ``unplaced`` record. Neither record carries ``text``, ``readings`` or ``variants``.

    Args:
        words: PREPROCESS's consensus words, in stream order.
        planned: The extents.
        owners: Which extent hides each masked word, by index into ``planned``, as the fold's
            ledger records it. None hides every word an extent overlaps, which is REDACT's own plan.

    Returns:
        ``(records, text, unplaced_n)``, ``text`` being the join of each record's placeholder or
        surface.
    """
    records: list[dict[str, Any]] = []
    position: dict[int, int] = {}
    unplaced = 0
    for word in words:
        if word.extent is None:
            unplaced += 1
            records.append({"kind": "unplaced", "index": int(word.attributes["index"]), "token": _UNPLACED_PLACEHOLDER})
            continue
        if owners is None:
            hull = word_hull(word)
            index = next((i for i, p in enumerate(planned) if _overlaps(hull, (p.start, p.end))), None)
        else:
            index = owners.get(word.id)
        if index is None:
            records.append(_word_record(word))
        elif index not in position:
            position[index] = len(records)
            records.append(
                {
                    "kind": "redaction",
                    "index": int(word.attributes["index"]),
                    "token": f"[{planned[index].category}]",
                    "category": planned[index].category,
                    "start_s": float(planned[index].start),
                    "end_s": float(planned[index].end),
                    "words_n": 1,
                }
            )
        else:
            records[position[index]]["words_n"] += 1
    return records, " ".join(token for token in (_token(record) for record in records) if token), unplaced


def _token(record: dict[str, Any]) -> str:
    """What one record contributes to the flat transcript.

    Args:
        record: A record from :func:`_render`.

    Returns:
        The surface for a word, the placeholder for anything else.
    """
    return str(record["text"]) if record["kind"] == "word" else str(record["token"])


@dataclass(frozen=True)
class _Exemption:
    """One PII candidate the declared stimulus accounts for, and the evidence that it does.

    Attributes:
        finding_id: The ``pii`` entity that was not planned.
        category: Its category.
        extent: Its extent, unpadded.
        word_ids: The consensus words it covers, in stream order.
        expected_keys: The stimulus's own normalised tokens the covered run matched, taken from the
            prompt and never from the transcript.
        prompt: Which ``expected_speech`` entry accounted for it.
        unit: Which structure unit of that entry.
        unit_text: That unit verbatim, as the prompt spelled it.
    """

    finding_id: str
    category: str
    extent: tuple[float, float]
    word_ids: tuple[str, ...]
    expected_keys: tuple[str, ...]
    prompt: int
    unit: int
    unit_text: str


def _expected_units(
    hint: AudioHints | None,
    task_family: str | None,
    *,
    terminators: str,
    lexicon: TaskLexicon | None = None,
) -> list[tuple[int, str, list[str]]]:
    """Everything the task declared, as the structure units a covered run is placed inside.

    Three declaration sources, one unit list: the recording's declared prompts split into their
    structure units, the carrier a syllable family names, and the proper nouns the family's
    expectation row declares. Each of the latter two is its own unit, so a run is placed inside one
    name and never across two.

    Args:
        hint: What the recording was declared to contain, or None.
        task_family: The declared family, a key of ``SPEECH_EXPECTATIONS``, or None.
        terminators: The characters that close a unit inside one prompt.
        lexicon: The family's task lexicon; each phrase not already a declared name is its own unit.

    Returns:
        ``(prompt index, unit text, tokens)`` per unit. Empty when the task declared nothing. A
        unit from a family declaration carries prompt index ``-1``: it came from the expectation
        row, not from an ``expected_speech`` entry.
    """
    units: list[tuple[int, str, list[str]]] = []
    if hint is not None and hint.expected_speech:
        units.extend(split_prompts(list(hint.expected_speech), terminators=terminators))
    carrier = declared_carrier(task_family)
    declared = (*((carrier,) if carrier else ()), *expected_names(task_family))
    units.extend((-1, name, name.split()) for name in declared)
    named = {name.casefold() for name in declared}
    for phrase in lexicon.texts() if lexicon is not None else ():
        if phrase not in named:
            units.append((-1, phrase, phrase.split()))
    return units


def _expected_exemptions(
    findings: Sequence[Entity],
    words: Sequence[Entity],
    units: Sequence[tuple[int, str, list[str]]],
    normalise: Callable[[str], str],
    near: NearMatch,
) -> list[_Exemption]:
    """Which findings the declared stimulus accounts for.

    A candidate is exempt only when every word SPEECH placed it on (its ``word_ids``) carries a
    non-empty normalised key and those keys occur in order and contiguously inside one declared unit,
    each within ``near`` of the unit's own token. A finding without placed words, or one on every
    consensus word, is never exempt.

    Args:
        findings: The live ``pii`` entities.
        words: PREPROCESS's consensus words, in stream order.
        units: The declared structure from :func:`_expected_units`.
        normalise: The branches' own lexical normaliser, ``BranchParams.p_normalise``.
        near: How far a transcript token may be from a stimulus token and still be that token.

    Returns:
        One exemption per accounted-for finding, in the findings' own order. Empty when ``units``
        is empty, which is the no-hint path.
    """
    if not units:
        return []
    keyed = [(prompt, text, [normalise(token) for token in tokens]) for prompt, text, tokens in units]
    exemptions: list[_Exemption] = []
    by_id = {word.id: word for word in words}
    for finding in findings:
        ids = [str(i) for i in (finding.attributes.get("word_ids") or ())]
        covered = [by_id[i] for i in ids if i in by_id]
        if finding.extent is None or not covered or len(covered) != len(ids) or len(covered) == len(words):
            continue
        keys = [normalise(str(word.attributes.get("text") or "")) for word in covered]
        if not all(keys):
            continue
        for unit_index, (prompt_index, unit_text, unit_keys) in enumerate(keyed):
            offset = near.run_offset(unit_keys, keys)
            if offset is None:
                continue
            exemptions.append(
                _Exemption(
                    finding_id=finding.id,
                    category=str(finding.attributes.get("category", "")),
                    extent=(float(finding.extent[0]), float(finding.extent[1])),
                    word_ids=tuple(word.id for word in covered),
                    expected_keys=tuple(unit_keys[offset : offset + len(keys)]),
                    prompt=prompt_index,
                    unit=unit_index,
                    unit_text=unit_text,
                )
            )
            break
    return exemptions


def word_keys(word: Entity) -> set[str]:
    """Every token one consensus word reads as: its own surface and each recogniser's reading of it.

    Args:
        word: A consensus word.

    Returns:
        The surfaces as :func:`_match_token` folds them, the empty ones dropped.
    """
    readings = (word.attributes.get("readings") or {}).values()
    texts = [str(word.attributes.get("text") or ""), *(str(reading) for reading in readings)]
    return {key for key in (_match_token(text) for text in texts) if key}


def finding_keys(findings: Sequence[Entity], by_id: Mapping[str, Entity]) -> dict[str, set[str]]:
    """The token each finding read on each word it covers: that word's reading in the finding's own text.

    Args:
        findings: The ``pii`` entities.
        by_id: The consensus words by id.

    Returns:
        ``{word id: tokens}``, as :func:`_match_token` folds them; the consensus surface where a finding
        names no ``haystack`` or was read off the consensus.
    """
    keys: dict[str, set[str]] = {}
    for finding in findings:
        haystack = str(finding.attributes.get("haystack") or "consensus")
        for word_id in finding.attributes.get("word_ids") or ():
            word = by_id.get(str(word_id))
            if word is None:
                continue
            readings = word.attributes.get("readings") or {}
            surface = word.attributes.get("text") if haystack == "consensus" else readings.get(haystack)
            key = _match_token(str(surface or ""))
            if key:
                keys.setdefault(word.id, set()).add(key)
    return keys


@dataclass(frozen=True)
class TaskContent:
    """The consensus words that are the task's own content because they lie on its events.

    Attributes:
        event_ids: Every timed word whose hull lies on the declared family's task events
            (:func:`~senselab.audio.workflows.triage.task_content.task_content_ids`).
        propagated_ids: Words a finding covered off the events that read as a token a finding read on them.
        keys: The tokens the findings read on ``event_ids``.
        events_n: How many task events were read.
    """

    event_ids: frozenset[str] = frozenset()
    propagated_ids: frozenset[str] = frozenset()
    keys: frozenset[str] = frozenset()
    events_n: int = 0

    @property
    def ids(self) -> frozenset[str]:
        """Every word that is task content, on the events or by propagation."""
        return self.event_ids | self.propagated_ids


def task_content(
    store: ProvStore,
    words: Sequence[Entity],
    findings: Sequence[Entity],
    family: str | None,
    lexicon_ids: Collection[str] = frozenset(),
) -> TaskContent:
    """Which consensus words are the declared task's own content, read off its events.

    Args:
        store: The provenance store, read for the family's task readings.
        words: The consensus words, in stream order.
        findings: The live ``pii`` entities.
        family: The declared family.
        lexicon_ids: The words of the task's own lexicon.

    Returns:
        The words on the events, and every finding word elsewhere that reads as a token a finding read
        on them; empty for a lexical family or where no reading read the task.
    """
    events = task_events(store, family)
    if not events:
        return TaskContent()
    on_events = task_content_ids(words, events, family=family, lexicon_ids=lexicon_ids)
    by_id = {word.id: word for word in words}
    read = finding_keys(findings, by_id)
    keys = frozenset(key for word_id in on_events & set(read) for key in read[word_id])
    elsewhere = frozenset(word_id for word_id, tokens in read.items() if word_id not in on_events and tokens & keys)
    return TaskContent(frozenset(on_events), elsewhere, keys, len(events))


def _task_event_exemptions(findings: Sequence[Entity], content: TaskContent) -> list[Entity]:
    """The findings every word of which is the task's own content.

    Args:
        findings: The live ``pii`` entities not already exempt.
        content: :func:`task_content`'s reading.

    Returns:
        The findings, in their own order.
    """
    return [
        finding
        for finding in findings
        if finding.extent is not None
        and (ids := {str(i) for i in finding.attributes.get("word_ids") or ()})
        and ids <= content.ids
    ]


def _write_artifacts(
    redacted: Audio,
    transcript_text: str,
    records: list[dict[str, Any]],
    artifacts_dir: Path,
) -> dict[str, Path]:
    """Write the releasable set; takes no store and no element id.

    Args:
        redacted: The verified redacted audio.
        transcript_text: The verified redacted transcript.
        records: The redacted consensus stream from :func:`_render`.
        artifacts_dir: The release directory.

    Returns:
        The written paths, keyed ``audio``/``transcript``/``consensus``.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audio_path = artifacts_dir / "audio.wav"
    redacted.save_to_file(str(audio_path))
    transcript_path = artifacts_dir / "transcript.txt"
    transcript_path.write_text(transcript_text + "\n")
    consensus_path = artifacts_dir / "consensus.json"
    consensus_path.write_text(
        json.dumps(
            {
                "schema": CONSENSUS_ARTIFACT_SCHEMA,
                "version": CONSENSUS_ARTIFACT_VERSION,
                "text": transcript_text,
                "n_records": len(records),
                "n_words": sum(1 for record in records if record["kind"] == "word"),
                "n_redactions": sum(1 for record in records if record["kind"] == "redaction"),
                "n_unplaced": sum(1 for record in records if record["kind"] == "unplaced"),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {"audio": audio_path, "transcript": transcript_path, "consensus": consensus_path}


def _register_redacted_stream(
    store: ProvStore,
    redacted: Audio,
    *,
    run_dir: Path,
    activity_id: str,
    agent_id: str,
    source_id: str,
    fill: str,
) -> str:
    """Persist the redacted audio under ``run_dir`` and register it as the stream named ``redacted``.

    Args:
        store: The provenance store.
        redacted: The masked audio.
        run_dir: The run directory streams live under.
        activity_id: REDACT's ``apply`` activity.
        agent_id: The agent answerable for it.
        source_id: The stream the redaction was applied to.
        fill: What was written into each extent.

    Returns:
        The stream entity's id.
    """
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    relative, report = write_stream(redacted, run_dir, STREAM_NAME)
    duration_s = float(redacted.waveform.shape[-1]) / float(redacted.sampling_rate)
    stream_id = store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={
            "name": STREAM_NAME,
            **path_attributes(relative, run_dir),
            "sampling_rate": redacted.sampling_rate,
            "channels": int(redacted.waveform.shape[0]),
            "write_gain": report.gain,
            "fill": fill,
        },
    )
    store.was_generated_by(stream_id, activity_id)
    store.was_attributed_to(stream_id, agent_id)
    store.was_derived_from(stream_id, source_id)
    return stream_id


RESCAN_SOURCE = "redact_rescan"
"""The ``source`` of a finding REDACT's re-scan read on words no finding covered."""


class _Rescan:
    """REDACT's re-scan survivors, placed on words and judged by the fold's own plan.

    A survivor placed on residue words no planned extent covers becomes a ``pii`` finding of its own; the
    fold's mask plan (:func:`fold_mask_plan`) then says which of its words stay masked, under every
    exemption and policy release the fold applies. Those words are re-planned as masks; a survivor whose
    words the plan releases, or the declared stimulus accounts for, is attributed. A survivor that places
    on no uncovered word, or on a word with no usable timing, is outstanding.
    """

    def __init__(
        self,
        store: ProvStore,
        software: str,
        residue: Sequence[Entity],
        exempt_word_ids: Collection[str],
        detectors: Sequence[str],
    ) -> None:
        """Hold what classifying a re-scan reads.

        Args:
            store: The provenance store, where survivor findings are written.
            software: The software agent.
            residue: The residue words, in stream order.
            exempt_word_ids: The words the declared stimulus accounts for.
            detectors: The detectors the scan ran.
        """
        self.store = store
        self.software = software
        self.residue = list(residue)
        self.tokens = _tokens(self.residue)
        self.exempt = set(exempt_word_ids)
        self.detectors = list(detectors)
        self.findings: list[Entity] = []
        self._written: dict[tuple[str, tuple[str, ...]], str] = {}
        self._activity: str | None = None

    def _place(self, text: str, planned: Sequence[RedactionExtent]) -> list[Entity]:
        def open_(word: Entity) -> bool:
            return not any(_overlaps(word_hull(word), (e.start, e.end)) for e in planned)

        hits = [word for word in _place(text, self.tokens) if open_(word)]
        return hits or [word for word in _place_substring(text, self.residue) if open_(word)]

    def _write(self, category: str, words: Sequence[Entity]) -> None:
        key = (category, tuple(word.id for word in words))
        if key in self._written:
            return
        if self._activity is None:
            self._activity = self.store.activity(node=NODE, step="rescan", parameters={})
            self.store.was_associated_with(self._activity, self.software)
        hull = (min(word_hull(w)[0] for w in words), max(word_hull(w)[1] for w in words))
        finding_id = self.store.entity(
            prov_type="pii",
            extent=hull,
            attributes={
                "category": category,
                "source": RESCAN_SOURCE,
                "haystack": "consensus",
                "word_ids": [word.id for word in words],
                "detectors_used": self.detectors,
                "detectors_failed": [],
            },
        )
        self.store.was_generated_by(finding_id, self._activity)
        self.store.was_attributed_to(finding_id, self.software)
        for word in words:
            self.store.was_derived_from(finding_id, word.id)
        self._written[key] = finding_id
        self.findings.append(self.store.get_entity(finding_id))

    def classify(
        self, checked: _Verification, planned: Sequence[RedactionExtent], plan: Callable[[], MaskPlan]
    ) -> tuple[list[RedactionExtent], list[str], list[str]]:
        """Place one re-scan's survivors and judge them by the fold's plan.

        Args:
            checked: The re-scan.
            planned: The extents the scanned text was rendered under.
            plan: The fold's mask plan over the store, called once the survivors are written.

        Returns:
            ``(added, attributed, outstanding)``: an extent per survivor word the fold keeps masked, and the
            surviving categories accounted for and left outstanding, sorted.
        """
        placed: list[tuple[str, list[Entity]]] = []
        outstanding: set[str] = set()
        for category, text in checked.spans:
            hits = self._place(text, planned)
            if not hits or any(word_hull(w)[1] <= word_hull(w)[0] for w in hits):
                outstanding.add(category)
                continue
            placed.append((category, hits))
        for category, hits in placed:
            if not all(word.id in self.exempt for word in hits):
                self._write(category, hits)
        kept = set(plan().owners()) if placed else set()
        added: list[RedactionExtent] = []
        attributed: set[str] = set()
        for category, hits in placed:
            masked = [word for word in hits if word.id in kept and word.id not in self.exempt]
            if not masked:
                attributed.add(category)
                continue
            added.extend(RedactionExtent(start=word_hull(w)[0], end=word_hull(w)[1], category=category) for w in masked)
        return added, sorted(attributed - outstanding), sorted(outstanding)


def fold_mask_plan(
    store: ProvStore, config: TriageConfig, hint: AudioHints | None, *, reviewer_applies: bool = False
) -> MaskPlan:
    """The fold's mask plan over the store, with every argument VERDICT reads from the config and the hint.

    Args:
        store: The provenance store.
        config: The triage configuration.
        hint: What the recording was declared to contain.
        reviewer_applies: Whether the reviewer's ``release`` entries may unmask words.

    Returns:
        :func:`mask_plan`'s plan.
    """
    from senselab.audio.workflows.triage.live_evidence import recording_stem  # noqa: PLC0415
    from senselab.audio.workflows.triage.vocabulary import FoldPolicy  # noqa: PLC0415

    policy = FoldPolicy.from_config(config)
    return mask_plan(
        store,
        reviewer_applies=reviewer_applies,
        padding_ms=padding_ms(config),
        condition_categories=policy.condition_categories,
        protected_categories=policy.trim_protected_categories,
        cohort_conditions=policy.cohort_conditions,
        lexicon=task_lexicon(config, declared_task_family(store), hint),
        language=None if hint is None else str(hint.metadata.get("language") or "") or None,
        name_approvals=name_approvals(config, recording_stem(store)),
        task_text=task_texts(hint),
    )


def shipped_texts(store: ProvStore, plan: MaskPlan) -> tuple[str, str]:
    """The consensus transcript as recorded, and as the release would ship it under a mask plan.

    Args:
        store: The provenance store.
        plan: The final mask plan (:func:`fold_mask_plan`).

    Returns:
        ``(original, masked)``: the consensus words' text with bracketed tokens dropped, and the same text
        with each final mask's words replaced by its placeholder.
    """
    words = consensus_words(store)
    original = _verification_text(_render(words, [])[0])
    masked = _verification_text(_render(words, plan.final, plan.owners())[0])
    return original, masked


def redact(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
    artifacts_dir: Path,
    task_family: str | None = None,
) -> RedactResult:
    """Redact every PII finding from the recording and verify the redacted text before releasing it.

    Args:
        store: The provenance store, holding SPEECH's ``pii`` entities and scan measurement and
            PREPROCESS's consensus words.
        source: The store-held stream name, ``"recording"`` (N17).
        config: The triage configuration.
        hint: What the recording was declared to contain. When it carries ``expected_speech``, a
            PII candidate the declared stimulus accounts for is exempted from redaction and
            recorded as such.
        run_dir: The run directory sidecar paths are relative to.
        artifacts_dir: The release directory; must not contain or be contained by ``run_dir``.
        task_family: The declared family, a key of ``SPEECH_EXPECTATIONS``. Its expectation row's
            carrier and declared names account for a candidate the same way a declared prompt
            does; with no family, only the prompt does.

    Returns:
        The verdict, the view over what this node wrote, and the released artifacts — empty unless
        the outcome is a pass.

    Raises:
        ValueError: If ``redaction.fill`` has no value, if ``redaction.padding_ms`` has no usable
            value (see :func:`padding_ms`), if ``artifacts_dir`` and ``run_dir`` contain one
            another, if any ``redaction.llm_check`` key is unmeasured, if the store carries no PII
            scan measurement (N15), or if a finding's category is unusable (see
            :func:`_extents_from_findings`).
        LookupError: If no live stream carries ``source``.
    """
    fill = str(config.require(_FILL_KEY))
    bleep_hz = config.get(_BLEEP_HZ_KEY)
    margin_ms = padding_ms(config)
    required_detectors = sorted(str(name) for name in config.require(_REQUIRED_DETECTORS_KEY))
    run_resolved, release_resolved = run_dir.resolve(), artifacts_dir.resolve()
    if run_resolved.is_relative_to(release_resolved) or release_resolved.is_relative_to(run_resolved):
        raise ValueError(
            f"artifacts_dir {artifacts_dir} and run_dir {run_dir} must not contain one another; "
            "the store and the release directory must not be sweepable by one publish step"
        )
    scan_measurement = find_measurement(store, "pii_scan")
    if scan_measurement is None:
        raise ValueError("no PII scan measurement in the store (N15); an unscanned recording is unchecked, not clean")
    scanned_by, scan_failed, scan_missing = _scan_evidence(scan_measurement, required_detectors)
    scan_incomplete = bool(scan_failed) or bool(scan_missing) or not scanned_by
    findings = _findings(store)
    words = consensus_words(store)
    residue = residue_words(store)
    lexicon = task_lexicon(config, task_family, hint)
    units = _expected_units(
        hint,
        task_family,
        terminators=str(config.require(_TERMINATORS_KEY)),
        lexicon=lexicon,
    )
    exemptions = _expected_exemptions(findings, words, units, branch_params(config).p_normalise, near_match(config))
    exempt_findings = {exemption.finding_id for exemption in exemptions}
    lexicon_ids = {words[i].id for i in lexicon.positions([str(w.attributes.get("text") or "") for w in words])}
    content = task_content(store, words, findings, task_family, lexicon_ids)
    task_exempt = _task_event_exemptions([f for f in findings if f.id not in exempt_findings], content)
    exempt_findings |= {finding.id for finding in task_exempt}
    exempt_word_ids = frozenset(word_id for exemption in exemptions for word_id in exemption.word_ids) | frozenset(
        str(word_id) for finding in task_exempt for word_id in finding.attributes.get("word_ids") or ()
    )
    extents = _extents_from_findings([finding for finding in findings if finding.id not in exempt_findings])
    consensus = find_measurement(store, "consensus_transcript")
    consulted = _pii_marking_assertions(store)
    planned = plan_redactions(extents, padding_ms=margin_ms)
    records, transcript_text, unplaced_n = _render(words, planned)
    checked = (
        _verify(_render(residue, planned)[0], required_detectors)
        if not scan_incomplete
        else _Verification(verified=False, survived=[], scan_ran=False, failed=[], missing=[])
    )
    software = software_agent(store)
    rescan = _Rescan(store, software, residue, exempt_word_ids, scanned_by)
    replanned_n = 0
    attributed: list[str] = []
    outstanding: list[str] = []
    while checked.scan_ran and checked.survived:
        added, attributed, outstanding = rescan.classify(checked, planned, lambda: fold_mask_plan(store, config, hint))
        if not added and (replanned_n or not outstanding):
            break
        replanned_n += 1
        extents.extend(added)
        planned = plan_redactions(extents, padding_ms=margin_ms)
        records, transcript_text, unplaced_n = _render(words, planned)
        checked = _verify(_render(residue, planned)[0], required_detectors)
        attributed, outstanding = [], []
    if checked.scan_ran and checked.survived and not (attributed or outstanding):
        _, attributed, outstanding = rescan.classify(checked, planned, lambda: fold_mask_plan(store, config, hint))
    unremediable = list(outstanding)
    findings = [*findings, *rescan.findings]

    stream_id, recording = resolve_stream(store, run_dir, source)
    redacted = apply_redactions(recording, planned, fill=fill, bleep_hz=bleep_hz)

    view: list[str] = [entity.id for entity in rescan.findings]

    plan_act = store.activity(node=NODE, step="plan", parameters={"padding_ms": margin_ms, "replanned_n": replanned_n})
    store.was_associated_with(plan_act, software)
    store.used(plan_act, scan_measurement.id)
    for finding in findings:
        store.used(plan_act, finding.id)
    for assertion in consulted:
        store.used(plan_act, assertion.id)
    span_ids: list[str] = []
    for extent in planned:
        span_id = store.entity(
            prov_type="span",
            extent=(extent.start, extent.end),
            attributes={"name": "redaction", "category": extent.category},
        )
        store.was_generated_by(span_id, plan_act)
        store.was_attributed_to(span_id, software)
        for finding in findings:
            if finding.extent is not None and _overlaps(finding.extent, (extent.start, extent.end)):
                store.was_derived_from(span_id, finding.id)
        span_ids.append(span_id)
        view.append(span_id)

    for exemption in exemptions:
        exempt_id = store.entity(
            prov_type="assertion",
            extent=exemption.extent,
            attributes={
                "verb": _EXEMPT_VERB,
                "label": _EXPECTED_LABEL,
                "category": exemption.category,
                "expected_keys": list(exemption.expected_keys),
                "expected_prompt": exemption.prompt,
                "expected_unit": exemption.unit,
                "expected_unit_text": exemption.unit_text,
                "words_n": len(exemption.word_ids),
            },
        )
        store.was_generated_by(exempt_id, plan_act)
        store.was_attributed_to(exempt_id, software)
        store.was_derived_from(exempt_id, exemption.finding_id)
        for word_id in exemption.word_ids:
            store.was_derived_from(exempt_id, word_id)
        view.append(exempt_id)
    for finding in task_exempt:
        covered = [str(word_id) for word_id in finding.attributes.get("word_ids") or ()]
        bounds = finding.extent if finding.extent is not None else (0.0, 0.0)
        exempt_id = store.entity(
            prov_type="assertion",
            extent=(float(bounds[0]), float(bounds[1])),
            attributes={
                "verb": _EXEMPT_VERB,
                "label": TASK_EVENT_LABEL,
                "category": str(finding.attributes.get("category", "")),
                "words_n": len(covered),
                "events_n": content.events_n,
                "propagated": any(word_id in content.propagated_ids for word_id in covered),
            },
        )
        store.was_generated_by(exempt_id, plan_act)
        store.was_attributed_to(exempt_id, software)
        store.was_derived_from(exempt_id, finding.id)
        for word_id in covered:
            store.was_derived_from(exempt_id, word_id)
        view.append(exempt_id)
    exemptions_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": _EXEMPTION_MEASUREMENT,
            "signal": "consensus_transcript",
            "n": len(exemptions),
            "task_event_n": len(task_exempt),
            "by_category": dict(Counter(exemption.category for exemption in exemptions)),
            "expected_speech_declared": bool(units),
            "n_units": len(units),
            "n_findings": len(findings),
        },
    )
    store.was_generated_by(exemptions_id, plan_act)
    store.was_attributed_to(exemptions_id, software)
    view.append(exemptions_id)

    apply_act = store.activity(node=NODE, step="apply", parameters={"redactions_n": len(planned), "fill": fill})
    store.was_associated_with(apply_act, software)
    store.used(apply_act, stream_id)
    for span_id in span_ids:
        store.used(apply_act, span_id)
    for word in words:
        store.used(apply_act, word.id)
    redacted_stream_id = _register_redacted_stream(
        store, redacted, run_dir=run_dir, activity_id=apply_act, agent_id=software, source_id=stream_id, fill=fill
    )
    for span_id in span_ids:
        store.was_derived_from(redacted_stream_id, span_id)
    view.append(redacted_stream_id)

    verify_act = store.activity(node=NODE, step="verify", parameters={"required_detectors": required_detectors})
    store.was_associated_with(verify_act, software)
    if checked.cache is not None and not checked.cache["hit"]:
        annotate_result_origin(
            checked.cache["key"], {"run": store.run_id, "activity": verify_act, "node": NODE, "step": "verify"}
        )
    if consensus is not None:
        store.used(verify_act, consensus.id)
    for word in words:
        store.used(verify_act, word.id)
    for span_id in span_ids:
        store.used(verify_act, span_id)

    artifacts: dict[str, Path] = {}
    if scan_incomplete:
        outcome = Outcome.FAIL
        reasons = []
        if scan_failed:
            reasons.append(f"detectors failed: {', '.join(scan_failed)}")
        if scan_missing:
            reasons.append(f"required detectors were not attempted: {', '.join(scan_missing)}")
        if not scanned_by:
            reasons.append("no detector ran")
        why = (
            f"the store's pii scan is incomplete ({'; '.join(reasons)}); "
            "an unchecked recording is not a clean one (N15)"
        )
    elif not checked.scan_ran:
        outcome = Outcome.FLAG
        parts = []
        if checked.failed:
            parts.append(f"detectors failed: {', '.join(checked.failed)}")
        if checked.missing:
            parts.append(f"required detectors were not attempted: {', '.join(checked.missing)}")
        if not parts:
            parts.append("no detector ran")
        why = (
            f"the re-scan over the redacted text is incomplete ({'; '.join(parts)}); an unverified artifact is withheld"
        )
    elif outstanding:
        outcome = Outcome.FAIL
        why = "verification found pii on the redacted transcript: " + ", ".join(outstanding)
    else:
        outcome = Outcome.PASS
        why = "every finding redacted; the redacted transcript re-scans clean"
        if exemptions or task_exempt:
            accounted = [
                *([f"{len(exemptions)} the declared stimulus accounts for"] if exemptions else []),
                *([f"{len(task_exempt)} on the task's own events"] if task_exempt else []),
            ]
            why = (
                f"every finding redacted except {' and '.join(accounted)}; the redacted transcript carries nothing else"
            )

    if outcome is Outcome.PASS:
        artifacts = _write_artifacts(redacted, transcript_text, records, artifacts_dir)
    verdict_id, verdict = write_verdict(
        store,
        verify_act,
        software,
        node=NODE,
        outcome=outcome,
        kind=None,
        why=why,
        detail={
            "redactions_n": len(planned),
            "by_category": dict(Counter(extent.category for extent in planned)),
            "padding_ms": margin_ms,
            "fill": fill,
            "verified": checked.verified,
            "survived": checked.survived,
            "unremediable": unremediable,
            "outstanding": outstanding,
            "expected_exempt_n": len(exemptions),
            "task_event_exempt_n": len(task_exempt),
            "expected_exempt_by_category": dict(Counter(exemption.category for exemption in exemptions)),
            "expected_survivors": attributed,
            "rescan_findings_n": len(rescan.findings),
            "expected_speech_declared": bool(units),
            "replanned_n": replanned_n,
            "scan_failed": scan_failed,
            "scan_missing": scan_missing,
            "verify_failed": checked.failed,
            "verify_missing": checked.missing,
            "verify_cache": checked.cache,
            "required_detectors": required_detectors,
            "unplaced_words_n": unplaced_n,
            "audio_check": _AUDIO_CHECK,
            "artifacts_withheld": not artifacts,
        },
    )
    view.append(verdict_id)
    return RedactResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, artifacts=artifacts)


RELEASED_FILES = ("audio.wav", "transcript.txt", "consensus.json")
"""What a release of the redacted copy writes into the release directory."""

_QUOTE_EDGE = "\"'`.,;:!?()[]{}<>-\u2018\u2019\u201c\u201d\u2026"  # stripped from a token's two ends before matching


def _match_token(text: str) -> str:
    """One token as a reviewer quote and a transcript word are compared.

    Args:
        text: A word's surface, or one whitespace-separated piece of a quote.

    Returns:
        The token lower-cased, curly apostrophes made straight, and punctuation stripped from both
        ends. Empty where nothing but punctuation was there.
    """
    return text.replace("\u2019", "'").strip(_QUOTE_EDGE).lower()


MASKED = "masked"
"""A word a mask still hides in the released copy."""

UNMASKED_BY_REVIEWER = "unmasked_by_reviewer"
"""A word a mask hid that a reviewer ``release`` entry named, and the fold let the entry unmask."""

UNMASKED_BY_TRIM = "unmasked_by_trim"
"""A word a mask covered and does not keep: not a residue content word (a closed-class word, a filler,
a marker, task content), or a content word no detector marked, which only the padding reached."""

RELEASED_BY_KIND = "released_by_kind"
"""A content word the policy releases by its kind whatever the reviewer said: a time of day, a weekday, a
relative reference or a length of time in a DATE_TIME-family span (:func:`time_by_kind`), a kinship
word, or a country that is the whole of a place name. :attr:`MaskWord.kind` says which."""

RELEASED_CONDITION = "released_condition"
"""A word a mask covered that a health condition the reviewer listed names: never masked."""

RELEASED_NOT_PROPER = "released_not_proper"
"""A word a detector finding covered that is written with no capital letter, is no number, and no reviewer
``redact`` entry and no date, age or state rule keeps: never masked, in any language."""

UNMASKED_BY_APPROVAL = "unmasked_by_approval"
"""A person's name a human approved for release (``redaction.name_approvals``)."""

PROPOSED_BY_REVIEWER = "proposed_by_reviewer"
"""A word a reviewer ``redact`` entry named that no mask hides."""

WORD_STATES = (
    MASKED,
    UNMASKED_BY_REVIEWER,
    UNMASKED_BY_TRIM,
    RELEASED_BY_KIND,
    RELEASED_CONDITION,
    RELEASED_NOT_PROPER,
    UNMASKED_BY_APPROVAL,
    PROPOSED_BY_REVIEWER,
)

KIND_TIME = "time"
KIND_KINSHIP = "kinship"
KIND_COUNTRY = "country"
RELEASE_KINDS = (KIND_TIME, KIND_KINSHIP, KIND_COUNTRY)
"""Why a word was released by kind."""

LOCK_DATE = "date"
LOCK_AGE = "age"
LOCK_PLACE = "place"
LOCK_PERSON = "person"
LOCKS = (LOCK_DATE, LOCK_AGE, LOCK_PLACE, LOCK_PERSON)
"""Why the policy keeps a word masked whatever the reviewer said: a date element, an age, a place below a
country, or a person's name awaiting a human's approval."""

POLICY_VERSION = policy_version()
"""The redaction policy the fold applies (``version`` of ``data/redaction_policy.yaml``); recorded on the ledger."""

MASK_UNCHANGED = "unchanged"
MASK_TRIMMED = "trimmed"
MASK_PARTLY_UNMASKED = "partly_unmasked"
MASK_UNMASKED = "unmasked"
MASK_OUTCOMES = (MASK_UNCHANGED, MASK_TRIMMED, MASK_PARTLY_UNMASKED, MASK_UNMASKED)
"""What became of one planned mask: kept whole, trimmed to its content words only, some of its words
unmasked by the reviewer, or none of its words left masked."""

PII_LEDGER = "pii_ledger"
"""The measurement VERDICT writes: every PII span, the words it covers and their state."""

FAMILIES_PATH = Path(__file__).parents[1] / "data" / "pii_category_families.yaml"
"""Which detector categories name the same kind of content, keyed by the reviewer's category for it."""

DETECTOR = "detector"
"""A mask placed on a detector finding's own words."""

REVIEWER = "reviewer"
"""A mask placed on a reviewer ``redact`` entry's words where it places a detector finding the fold could
not place itself, or names a country."""

POLICY = "policy"
"""A mask the policy places on words it always masks -- a date element, an age, a state -- that no
detector finding covered."""

PROPAGATED = "propagated"
"""A mask over words no finding covered that read as a name kept masked elsewhere in the recording."""

PROPAGATION_MASK = "mask"
"""A word masked because the same token is a name kept masked elsewhere in the recording."""

PROPAGATION_RELEASE = "release"
"""A word released because a reviewer ``release`` entry named the same token elsewhere in the recording."""

NON_TASK_SPEECH = "non_task_speech"
"""The source of a mask over a run of lexical speech outside a non-lexical task (``task_speech``)."""

NON_TASK_SPEECH_CATEGORY = "NON_TASK_SPEECH"
"""The category such a mask carries."""

PERSON_FAMILY = "PERSON"
LOCATION_FAMILY = "LOCATION"

AGREED_MASKED = "masked"
"""A reviewer ``redact`` entry every content word of which a mask already hides: agreement."""

AGREED_PLACED = "placed_unplaced"
"""A reviewer ``redact`` entry in the family of a detector finding the fold could not place on words: the
reviewer's quote places it, and its words are masked."""

NEW = "new"
"""A reviewer ``redact`` entry naming a content word no mask hides, or one its quote cannot be placed on:
the reviewer proposing to hide more."""

BY_KIND = "by_kind"
"""A reviewer ``redact`` entry every content word of which the policy releases by kind (:data:`RELEASED_BY_KIND`):
it proposes hiding nothing."""

COUNTRY_MASKED = "country_masked"
"""A reviewer ``redact`` entry naming a country, which the policy otherwise releases: the country is masked,
and the entry proposes nothing further."""

AGREEMENTS = (AGREED_MASKED, AGREED_PLACED, NEW, BY_KIND, COUNTRY_MASKED)
"""Every agreement a reviewer ``redact`` entry can carry."""


@lru_cache(maxsize=1)
def _families() -> dict[str, str]:
    """Every category the families file names, mapped to its family.

    Returns:
        ``{category: family}``, upper-cased; each family maps to itself.
    """
    raw = yaml.safe_load(FAMILIES_PATH.read_text()) or {}
    table: dict[str, str] = {}
    for family, members in (raw.get("families") or {}).items():
        for member in [family, *(members or ())]:
            table[str(member).upper()] = str(family).upper()
    return table


def same_extents(one: Sequence[RedactionExtent], other: Sequence[RedactionExtent]) -> bool:
    """Whether two sets of masks silence the same audio, whatever each one is labelled.

    Args:
        one: A set of extents.
        other: Another.

    Returns:
        True where both hold the same ``(start, end)`` bounds in the same order.
    """
    return [(float(e.start), float(e.end)) for e in one] == [(float(e.start), float(e.end)) for e in other]


@lru_cache(maxsize=1)
def name_families() -> frozenset[str]:
    """The families whose findings name something by a proper noun.

    Returns:
        ``name_families`` from ``data/pii_category_families.yaml``, upper-cased.
    """
    raw = yaml.safe_load(FAMILIES_PATH.read_text()) or {}
    return frozenset(str(family).upper() for family in raw.get("name_families") or ())


def category_family(category: str) -> str:
    """The family a detector or reviewer category belongs to, in the reviewer's own terms.

    Args:
        category: A category as a detector or the reviewer wrote it.

    Returns:
        Its family from ``data/pii_category_families.yaml``; the category itself, upper-cased, where no
        family lists it.
    """
    key = str(category or "").upper()
    return _families().get(key, key)


TIME_RELEASE_PATH = Path(__file__).parents[1] / "data" / "time_release.yaml"
"""The words of a DATE_TIME-family finding released by kind: times of day and durations."""

DATE_TIME_FAMILY = "DATE_TIME"

_ORDINAL_DIGITS = re.compile(r"^\d+(st|nd|rd|th)$")
_YEAR_DIGITS = re.compile(r"^\d{4}s?$")


@lru_cache(maxsize=1)
def _time_release() -> dict[str, frozenset[str]]:
    """The lists of ``data/time_release.yaml``, each as a set.

    Returns:
        ``{list name: words}``.
    """
    raw = yaml.safe_load(TIME_RELEASE_PATH.read_text()) or {}
    return {str(key): frozenset(str(word).lower() for word in value or ()) for key, value in raw.items()}


def _time_token(text: str) -> str:
    """One word as :func:`time_by_kind` compares it.

    Args:
        text: A word's surface.

    Returns:
        The redaction policy's folded token (:func:`~senselab.text.tasks.pii_detection.redaction_policy.fold`)
        with dots and hyphens dropped.
    """
    return policy_fold(text).replace(".", "").replace("-", "")


def time_by_kind(texts: Sequence[str]) -> set[int]:
    """Which words of one DATE_TIME-family span name a time, a weekday, a season, a relative reference or a duration.

    A time-of-day word, a weekday, a season, a relative day ("yesterday"), a clock marker after a number, a
    plural unit, or a singular unit beside a quantity or after a relative modifier ("last year") is
    released, and with any of them the span's quantities, numbers and modifiers. A span holding a
    ``blockers`` word, a four-digit number or an ordinal releases nothing.

    Args:
        texts: The span's word surfaces, in order.

    Returns:
        The positions released.
    """
    lists = _time_release()
    tokens = [_time_token(text) for text in texts]
    if any(
        token in lists["blockers"]
        or token in lists["ordinal_words"]
        or _ORDINAL_DIGITS.match(token)
        or _YEAR_DIGITS.match(token)
        for token in tokens
    ):
        return set()

    def number(token: str) -> bool:
        return token.isdigit() or token in lists["quantities"]

    quantified = any(number(token) for token in tokens)
    relative = any(token in lists["relative_modifiers"] for token in tokens)
    core: set[int] = set()
    for position, token in enumerate(tokens):
        if (
            token in lists["time_of_day"]
            or token in lists["plural_units"]
            or token in lists["weekdays"]
            or token in lists["relative_days"]
            or token in lists["seasons"]
        ):
            core.add(position)
        elif token in lists["units"] and (quantified or relative):
            core.add(position)
        elif token in lists["clock_markers"] and position and number(tokens[position - 1]):
            core.add(position)
        elif token[:-2].isdigit() and token[-2:] in ("am", "pm"):
            core.add(position)
    if not core:
        return set()
    return core | {
        position
        for position, token in enumerate(tokens)
        if number(token) or token in lists["modifiers"] or token in lists["relative_modifiers"] or not token
    }


PLACED_WORDS = "words"
PLACED_SUBSTRING = "substring"


@dataclass(frozen=True)
class MaskWord:
    """One word a planned mask covers, and what became of it.

    Attributes:
        word_id: The consensus word's entity id.
        text: Its surface.
        state: One of :data:`WORD_STATES` other than :data:`PROPOSED_BY_REVIEWER`.
        named: Whether a reviewer ``release`` entry named it, whether or not the fold applied it.
        content: Whether it counts as content for the trim: a residue content word, or one a
            detector marked in a protected category.
        finding: Whether a detector marked it, rather than the padding reaching it.
        categories: The categories the detectors marked on it, sorted.
        proper: Whether it is written as a proper noun
            (:func:`~senselab.audio.workflows.triage.residue.is_proper_form`).
        propagated: Whether its state was carried to it from another occurrence of the same token: a
            name kept masked elsewhere (:data:`PROPAGATION_MASK`), or a reviewer ``release`` entry naming the
            same token elsewhere (:data:`PROPAGATION_RELEASE`). A token is a word's consensus surface or
            any recogniser's reading of it, case folded and edge punctuation stripped (:func:`word_keys`).
        kind_cut: Whether the name kind cut released it: at an edge of a mask whose findings are all
            of :func:`name_families` and which holds a proper noun, a word written all lower-case that
            the transcript also uses outside every finding; or, in a recording declared in a language
            other than English, any word of such a mask not written as a proper noun.
        with_head: Whether it was released with its name's head: a word written all lower-case inside
            a finding of :func:`name_families` whose capitalised word a reviewer ``release`` entry
            named, and which no ``redact`` entry quotes.
        locked: One of :data:`LOCKS` where the policy keeps it masked whatever the reviewer said;
            empty otherwise.
        kind: One of :data:`RELEASE_KINDS` where it is :data:`RELEASED_BY_KIND`; empty otherwise.
        propagation: :data:`PROPAGATION_MASK` or :data:`PROPAGATION_RELEASE` where ``propagated``; empty otherwise.
        propagated_from: The word ids whose decision it carries, where ``propagated``.
        source_findings: The ``pii`` finding ids covering those words.
    """

    word_id: str
    text: str
    state: str
    named: bool
    content: bool
    finding: bool
    categories: tuple[str, ...] = ()
    proper: bool = False
    propagated: bool = False
    propagation: str = ""
    propagated_from: tuple[str, ...] = ()
    source_findings: tuple[str, ...] = ()
    kind_cut: bool = False
    with_head: bool = False
    locked: str = ""
    kind: str = ""


@dataclass(frozen=True)
class MaskOutcome:
    """One planned mask and what the word-level rule made of it.

    Attributes:
        planned: REDACT's padded, merged extent.
        words: Every consensus word the extent covers, in stream order.
        final: The extents that stay masked, one per run of adjacent kept words; the planned extent
            itself where no word changed state, and empty where no word stays masked.
        final_words: The masked word ids each of ``final`` hides, aligned with it.
        outcome: One of :data:`MASK_OUTCOMES`.
        task_words_n: How many words outside the residue -- the task's own content, which no mask
            ever covers -- the finding's own extent reached.
        categories: The detector categories of the findings the mask stands for, sorted; the
            reviewer's, the policy's or the source name's family for a :data:`REVIEWER`, :data:`POLICY` or
            :data:`PROPAGATED` mask.
        source: :data:`DETECTOR`, :data:`REVIEWER`, :data:`POLICY` or :data:`PROPAGATED`.
    """

    planned: RedactionExtent
    words: tuple[MaskWord, ...]
    final: tuple[RedactionExtent, ...]
    final_words: tuple[tuple[str, ...], ...]
    outcome: str
    task_words_n: int = 0
    categories: tuple[str, ...] = ()
    source: str = DETECTOR


@dataclass(frozen=True)
class ReviewerSpan:
    """One reviewer ``redact`` entry, placed on the words it names.

    Attributes:
        category: The category the reviewer gave it, upper-cased.
        text: The quote, verbatim.
        placed: :data:`PLACED_WORDS` where it matched whole-token runs, :data:`PLACED_SUBSTRING`
            where only a substring of the residue matched, empty where nothing did.
        word_ids: Every word it names, over every place it occurs.
        texts: Their surfaces.
        masked_ids: Those of them a final mask already hides.
        content_ids: Those of them that are content words the policy does not release by kind; only
            these are proposed to be hidden.
        agreement: One of :data:`AGREEMENTS`.
        entry_index: Its position in the reading's ``proposal``.
    """

    category: str
    text: str
    placed: str
    word_ids: tuple[str, ...]
    texts: tuple[str, ...]
    masked_ids: tuple[str, ...]
    content_ids: tuple[str, ...] = ()
    agreement: str = NEW
    entry_index: int = -1


@dataclass(frozen=True)
class ConditionSpan:
    """One health condition the reviewer listed, placed on the words it names. Never masked.

    Attributes:
        text: The quote, verbatim.
        why: The reviewer's one-sentence reason.
        word_ids: Every word it names, over every place it occurs.
        texts: Their surfaces.
        condition_kind: :data:`~senselab.audio.workflows.triage.cohort.COHORT` where it names a condition the
            study recruits for, :data:`~senselab.audio.workflows.triage.cohort.OTHER` otherwise.
        cohort_diagnosis: The cohort diagnosis its text matches; empty where none does.
    """

    text: str
    why: str
    word_ids: tuple[str, ...]
    texts: tuple[str, ...]
    condition_kind: str
    cohort_diagnosis: str = ""


@dataclass(frozen=True)
class UnplacedFinding:
    """A detector finding SPEECH could not place on the transcript's words, and what became of it.

    Attributes:
        category: The detector's category.
        family: Its family (:func:`category_family`).
        state: :data:`UNPLACED_PLACED`, :data:`UNPLACED_CLEARED`, :data:`UNPLACED_OPEN` or
            :data:`UNPLACED_UNREAD`.
        text: The detector's own text for it.
    """

    category: str
    family: str
    state: str
    text: str = ""


@dataclass(frozen=True)
class MaskPlan:
    """Which words stay masked once the policy, the reviewer's unmasks and the content-word trim are applied.

    Attributes:
        masks: One outcome per planned mask, in stream order.
        reviewer_applied: Whether the reviewer's ``release`` entries were applied.
        release_unplaced: ``release`` quotes that match no run of residue words.
        release_off_mask: ``release`` quotes that match words no mask covers.
        proposals: The reviewer's ``redact`` entries, placed.
        padding_ms: The margin kept around each run of masked words.
        unplaced: The detector findings SPEECH could not place on words, and what became of each.
        redact_planned: REDACT's own padded, merged extents, which the released copy replaces
            wherever the final masks differ from them.
        task_lexicon_ids: The words a finding covered that are the task's own vocabulary
            (:mod:`~senselab.audio.workflows.triage.task_lexicon`), which no mask covers.
        named_no_words: Whether the reading reported ``flagged`` with no proposal entry, so it moved
            no mask.
        releases: The reviewer's ``release`` entries as it wrote them -- text, category, the Safe
            Harbor identifier it named and its reason -- and whether each placed on words.
        conditions: The health conditions the reviewer listed, placed; none is masked.
        condition_indices: The ``proposal`` positions of entries in a condition category, which a
            reading written before conditions had their own part carries; they propose nothing.
        language: The language the recording declares, as its sidecar writes it; empty where none.
        name_approvals: The person names a human approved for release in this recording.
        name_release_proposed: The reviewer ``release`` quotes naming a person's name the policy keeps
            masked until a human approves it.
        task_text_ids: The words of the task's own texts (:func:`task_text_positions`) that a finding
            covered or a date, age or state rule would have masked, which no mask covers.
        task_event_ids: The words a finding covered that are the task's own content read off its events
            (:func:`task_content`), which no mask covers.
        task_content_only: Whether every finding located on words touched a non-lexical task's own events
            (:func:`task_content`) and kept no word once the task's own words were dropped, and none was
            left unplaced.
        reviewer_precedence: Whether the reviewer's reading recorded the task and the whole consensus
            transcript with its PII annotations and every recogniser's reading (``review_inputs_complete``),
            so its ``release`` of a token outranks a name kept masked at another occurrence.
        non_task_speech_ids: The timed words of lexical speech outside a non-lexical task
            (``task_speech_reading``), each kept masked whatever the content-word trim says.
        non_task_speech_untimed: Such words with no usable timing, which no mask can place.
        non_task_speech_extensive: Whether an item-set family's speech outside its item runs is extensive.
    """

    masks: tuple[MaskOutcome, ...]
    reviewer_applied: bool
    release_unplaced: tuple[str, ...]
    release_off_mask: tuple[str, ...]
    proposals: tuple[ReviewerSpan, ...]
    padding_ms: int
    unplaced: tuple[UnplacedFinding, ...] = ()
    redact_planned: tuple[RedactionExtent, ...] = ()
    task_lexicon_ids: tuple[str, ...] = ()
    named_no_words: bool = False
    releases: tuple[Mapping[str, Any], ...] = ()
    conditions: tuple[ConditionSpan, ...] = ()
    condition_indices: tuple[int, ...] = ()
    language: str = ""
    name_approvals: tuple[str, ...] = ()
    name_release_proposed: tuple[str, ...] = ()
    task_text_ids: tuple[str, ...] = ()
    task_event_ids: tuple[str, ...] = ()
    task_content_only: bool = False
    reviewer_precedence: bool = False
    non_task_speech_ids: tuple[str, ...] = ()
    non_task_speech_untimed: tuple[str, ...] = ()
    non_task_speech_extensive: bool = False

    @property
    def non_task_speech_masked_n(self) -> int:
        """How many words stay masked as lexical speech outside a non-lexical task."""
        return len(self.non_task_speech_ids)

    @property
    def planned(self) -> list[RedactionExtent]:
        """REDACT's own extents, in stream order."""
        return list(self.redact_planned)

    @property
    def final(self) -> list[RedactionExtent]:
        """The extents that stay masked, in stream order."""
        return [extent for mask in self.masks for extent in mask.final]

    @property
    def changed(self) -> bool:
        """Whether the final masks differ from REDACT's own extents, so REDACT's copy is not the one released."""
        return not same_extents(self.final, self.redact_planned)

    @property
    def agreed(self) -> frozenset[int]:
        """The ``proposal`` positions that propose hiding nothing more: agreeing entries and listed conditions."""
        return frozenset(span.entry_index for span in self.proposals if span.agreement != NEW) | frozenset(
            self.condition_indices
        )

    @property
    def policy_masks_n(self) -> int:
        """How many masks the policy itself placed, over words no detector finding covered."""
        return sum(1 for mask in self.masks if mask.source == POLICY and mask.final)

    @property
    def propagated_masked_n(self) -> int:
        """How many words stay masked because the same token is a name kept masked elsewhere."""
        return len(
            {
                word.word_id
                for mask in self.masks
                for word in mask.words
                if word.state == MASKED and word.propagation == PROPAGATION_MASK
            }
        )

    @property
    def person_names_masked(self) -> int:
        """How many words of a person's name stay masked: the ones a human may be asked to release."""
        return len(
            {
                word.word_id
                for mask in self.masks
                for word in mask.words
                if word.state == MASKED and word.locked == LOCK_PERSON
            }
        )

    def unplaced_in(self, state: str) -> list[str]:
        """The families of the unplaced findings in one state, sorted and without repeats."""
        return sorted({finding.family for finding in self.unplaced if finding.state == state})

    def owners(self) -> dict[str, int]:
        """Which final extent hides each masked word, by index into :attr:`final`."""
        owners: dict[str, int] = {}
        index = 0
        for mask in self.masks:
            for members in mask.final_words:
                for word_id in members:
                    owners.setdefault(word_id, index)
                index += 1
        return owners

    def count(self, state: str) -> int:
        """How many words are in one state; :data:`PROPOSED_BY_REVIEWER` counts unmasked proposed words."""
        if state == PROPOSED_BY_REVIEWER:
            return len(
                {
                    i
                    for span in self.proposals
                    if span.agreement == NEW
                    for i in span.content_ids
                    if i not in span.masked_ids
                }
            )
        return len({word.word_id for mask in self.masks for word in mask.words if word.state == state})

    def categories(self, state: str) -> list[str]:
        """The categories carrying at least one word in one state, sorted."""
        if state == PROPOSED_BY_REVIEWER:
            return sorted(
                {
                    span.category
                    for span in self.proposals
                    if span.agreement == NEW and set(span.content_ids) - set(span.masked_ids)
                }
            )
        return sorted({mask.planned.category for mask in self.masks if any(word.state == state for word in mask.words)})

    def record(self, *, release: str | None, release_ground: str | None) -> dict[str, Any]:
        """The ledger, as the measurement VERDICT writes carries it.

        Args:
            release: The fold's release axis value.
            release_ground: The fold's release ground.

        Returns:
            JSON-ready attributes. Word surfaces are the store's own transcript words; the released
            artifacts carry none of the masked ones.
        """
        owners = self.owners()
        words = [word for mask in self.masks for word in mask.words]
        return {
            "name": PII_LEDGER,
            "policy_version": POLICY_VERSION,
            "language": self.language,
            "release": release,
            "release_ground": release_ground,
            "reviewer_applied": self.reviewer_applied,
            "reviewer_precedence": self.reviewer_precedence,
            "padding_ms": self.padding_ms,
            "masks": [
                {
                    "category": mask.planned.category,
                    "start_s": float(mask.planned.start),
                    "end_s": float(mask.planned.end),
                    "outcome": mask.outcome,
                    "source": mask.source,
                    "categories": list(mask.categories),
                    "safe_harbor": list(safe_harbor_codes(mask.planned.category)),
                    "task_words_n": mask.task_words_n,
                    "words": [
                        {
                            "id": word.word_id,
                            "text": word.text,
                            "state": word.state,
                            "named": word.named,
                            "content": word.content,
                            "finding": word.finding,
                            "categories": list(word.categories),
                            "proper": word.proper,
                            "propagated": word.propagated,
                            "propagation": word.propagation,
                            "propagated_from": list(word.propagated_from),
                            "source_findings": list(word.source_findings),
                            "kind_cut": word.kind_cut,
                            "with_head": word.with_head,
                            "locked": word.locked,
                            "kind": word.kind,
                        }
                        for word in mask.words
                    ],
                }
                for mask in self.masks
            ],
            "final_masks": [
                {
                    "category": extent.category,
                    "start_s": float(extent.start),
                    "end_s": float(extent.end),
                    "word_ids": sorted(word_id for word_id, owner in owners.items() if owner == index),
                }
                for index, extent in enumerate(self.final)
            ],
            "proposals": [
                {
                    "category": span.category,
                    "text": span.text,
                    "placed": span.placed,
                    "word_ids": list(span.word_ids),
                    "texts": list(span.texts),
                    "masked_ids": list(span.masked_ids),
                    "content_ids": list(span.content_ids),
                    "agreement": span.agreement,
                    "entry_index": span.entry_index,
                    "safe_harbor": list(safe_harbor_codes(span.category)),
                }
                for span in self.proposals
            ],
            "conditions": [
                {
                    "text": condition.text,
                    "why": condition.why,
                    "word_ids": list(condition.word_ids),
                    "texts": list(condition.texts),
                    "state": RELEASED_CONDITION,
                    "condition_kind": condition.condition_kind,
                    "cohort_diagnosis": condition.cohort_diagnosis,
                }
                for condition in self.conditions
            ],
            "releases": [dict(entry) for entry in self.releases],
            "name_approvals": list(self.name_approvals),
            "name_release_proposed": list(self.name_release_proposed),
            "unplaced_findings": [
                {"category": finding.category, "family": finding.family, "state": finding.state, "text": finding.text}
                for finding in self.unplaced
            ],
            "release_unplaced": list(self.release_unplaced),
            "release_off_mask": list(self.release_off_mask),
            "task_lexicon_ids": list(self.task_lexicon_ids),
            "task_text_ids": list(self.task_text_ids),
            "task_event_ids": list(self.task_event_ids),
            "non_task_speech_ids": list(self.non_task_speech_ids),
            "non_task_speech_untimed": list(self.non_task_speech_untimed),
            "non_task_speech_extensive": self.non_task_speech_extensive,
            "task_content_only": self.task_content_only,
            "reviewer_named_no_words": self.named_no_words,
            "counts": {
                "masks_n": len(self.masks),
                "final_masks_n": len(self.final),
                "policy_masks_n": self.policy_masks_n,
                "task_words_n": sum(mask.task_words_n for mask in self.masks),
                "task_lexicon_words_n": len(self.task_lexicon_ids),
                "task_text_words_n": len(self.task_text_ids),
                "task_event_words_n": len(self.task_event_ids),
                "non_task_speech_masked_n": self.non_task_speech_masked_n,
                "non_task_speech_untimed_n": len(self.non_task_speech_untimed),
                **{
                    f"relabel_{relabel}_n": sum(1 for entry in self.releases if entry.get("relabel") == relabel)
                    for relabel in RELABELS
                },
                **{
                    f"place_reason_{reason}_n": sum(1 for entry in self.releases if entry.get("place_reason") == reason)
                    for reason in PLACE_REASONS
                },
                **{
                    f"{outcome}_n": sum(1 for mask in self.masks if mask.outcome == outcome)
                    for outcome in MASK_OUTCOMES
                },
                **{f"{state}_n": self.count(state) for state in WORD_STATES},
                **{
                    f"released_{kind}_n": len(
                        {w.word_id for w in words if w.state == RELEASED_BY_KIND and w.kind == kind}
                    )
                    for kind in RELEASE_KINDS
                },
                **{
                    f"locked_{lock}_n": len({w.word_id for w in words if w.state == MASKED and w.locked == lock})
                    for lock in LOCKS
                },
                "person_name_masked_n": self.person_names_masked,
                "name_release_proposed_n": len(self.name_release_proposed),
                "propagated_n": len({w.word_id for w in words if w.propagated}),
                "propagated_masked_n": self.propagated_masked_n,
                "kind_cut_n": len({w.word_id for w in words if w.kind_cut}),
                "unplaced_n": len(self.unplaced),
                **{
                    f"proposals_{agreement}_n": sum(1 for span in self.proposals if span.agreement == agreement)
                    for agreement in AGREEMENTS
                },
                "conditions_n": len(self.conditions),
                **{
                    f"{kind}_condition_n": sum(1 for condition in self.conditions if condition.condition_kind == kind)
                    for kind in CONDITION_KINDS
                },
            },
            "categories": {state: self.categories(state) for state in WORD_STATES},
            "cohort_diagnoses": sorted(
                {condition.cohort_diagnosis for condition in self.conditions if condition.cohort_diagnosis}
            ),
        }


def _tokens(words: Sequence[Entity]) -> list[tuple[str, Entity]]:
    """The residue words a quote is matched against, each as its matching token.

    Args:
        words: The residue words, in stream order.

    Returns:
        ``(token, word)`` for every non-bracketed, timed word whose token is not empty.
    """
    tokens = [
        (_match_token(str(word.attributes.get("text") or "")), word)
        for word in words
        if not word.attributes.get("bracketed") and word.extent is not None
    ]
    return [(token, word) for token, word in tokens if token]


def _place(quote: str, tokens: Sequence[tuple[str, Entity]]) -> list[Entity]:
    """Every word a quote names, as whole-token runs, over every place it occurs.

    Args:
        quote: The reviewer's quote.
        tokens: :func:`_tokens`' output.

    Returns:
        The words, in stream order, without repeats. Empty where the quote matches no run.
    """
    wanted = [token for token in (_match_token(piece) for piece in quote.split()) if token]
    width = len(wanted)
    surfaces = [token for token, _ in tokens]
    named: dict[str, Entity] = {}
    if width:
        for start in range(len(surfaces) - width + 1):
            if surfaces[start : start + width] == wanted:
                for _, word in tokens[start : start + width]:
                    named[word.id] = word
    return list(named.values())


def _place_substring(quote: str, words: Sequence[Entity]) -> list[Entity]:
    """The words a quote overlaps as a substring of the residue text, where no whole-token run matched.

    Args:
        quote: The reviewer's quote.
        words: The residue words, in stream order.

    Returns:
        Every word the first occurrence overlaps. Empty where the text does not contain the quote.
    """
    parts: list[str] = []
    spans: list[tuple[int, int, Entity]] = []
    cursor = 0
    for word in words:
        if word.attributes.get("bracketed"):
            continue
        surface = str(word.attributes.get("text") or "")
        if parts:
            cursor += 1
        parts.append(surface)
        spans.append((cursor, cursor + len(surface), word))
        cursor += len(surface)
    needle = quote.strip().lower()
    at = " ".join(parts).lower().find(needle) if needle else -1
    if at == -1:
        return []
    end = at + len(needle)
    return [word for first, last, word in spans if first < end and last > at]


def _final_extents(
    kept: Sequence[Entity],
    words: Sequence[Entity],
    kept_ids: set[str],
    category: str,
    padding_s: float,
) -> list[tuple[RedactionExtent, tuple[str, ...]]]:
    """The extents that silence one mask's kept words and leave every other word audible.

    Kept words adjacent in the stream form one run. A run's extent is its words' timing hull widened
    by the padding on each side; the widening stops where the nearest word left unmasked on that
    side ends (or begins), and the extent never shrinks below the run's own consensus extents.

    Args:
        kept: The mask's words that stay masked.
        words: Every timed consensus word of the recording, in stream order.
        kept_ids: The ids of every word any mask keeps.
        category: The mask's category.
        padding_s: The margin, in seconds.

    Returns:
        One ``(extent, word_ids)`` per run, in stream order.
    """
    order = {word.id: position for position, word in enumerate(words)}
    runs: list[list[Entity]] = []
    for word in sorted(kept, key=lambda item: order[item.id]):
        if runs and order[word.id] == order[runs[-1][-1].id] + 1:
            runs[-1].append(word)
        else:
            runs.append([word])
    out: list[tuple[RedactionExtent, tuple[str, ...]]] = []
    for run in runs:
        first, last = order[run[0].id], order[run[-1].id]
        previous = next((words[i] for i in range(first - 1, -1, -1) if words[i].id not in kept_ids), None)
        following = next((words[i] for i in range(last + 1, len(words)) if words[i].id not in kept_ids), None)
        start = min(word_hull(word)[0] for word in run) - padding_s
        end = max(word_hull(word)[1] for word in run) + padding_s
        if previous is not None and previous.extent is not None:
            start = max(start, float(previous.extent[1]))
        if following is not None and following.extent is not None:
            end = min(end, float(following.extent[0]))
        start = min(start, min(float(word.extent[0]) for word in run if word.extent is not None))
        end = max(end, max(float(word.extent[1]) for word in run if word.extent is not None))
        out.append((RedactionExtent(start=start, end=end, category=category), tuple(word.id for word in run)))
    return out


@dataclass(frozen=True)
class _Finding:
    """One live, non-exempt detector finding and the residue words SPEECH placed it on."""

    finding_id: str
    category: str
    extent: tuple[float, float]
    word_ids: tuple[str, ...]
    task_words_n: int


def _located_findings(
    store: ProvStore, residue_ids: set[str], task_ids: Collection[str] = frozenset()
) -> tuple[list[_Finding], list[dict[str, str]]]:
    """The findings on the words SPEECH placed them on, and those it could not place.

    Args:
        store: The provenance store.
        residue_ids: The residue's word ids; empty where the scan read none, which admits every word.
        task_ids: Words that are the task's own content though inside the residue (its task
            lexicon); a finding never masks them, and they count as its task words.

    Returns:
        ``(located, unplaced)``: located in stream order, each covering the residue words of its
        ``word_ids``; unplaced as the live ``pii_scan`` records them, ``{category, text, ...}``. A
        finding REDACT exempted as declared stimulus is neither.

    Raises:
        ValueError: If a finding carries no ``word_ids``, which a store written before SPEECH recorded
            them does not; such a store is replayed, not re-folded.
    """
    exempt = {
        source
        for assertion in live_entities(store, "assertion")
        if assertion.attributes.get("verb") == _EXEMPT_VERB
        for source in store.derived_from(assertion.id)
    }
    located: list[_Finding] = []
    for finding in live_entities(store, "pii"):
        if finding.extent is None or finding.id in exempt:
            continue
        ids = finding.attributes.get("word_ids")
        if ids is None:
            raise ValueError("a pii finding names no word_ids; the store predates SPEECH's placement, replay it")
        members = tuple(str(i) for i in ids if (not residue_ids or str(i) in residue_ids) and str(i) not in task_ids)
        task = sum(1 for i in ids if (residue_ids and str(i) not in residue_ids) or str(i) in task_ids)
        located.append(
            _Finding(
                finding.id,
                str(finding.attributes.get("category") or ""),
                (float(finding.extent[0]), float(finding.extent[1])),
                members,
                task,
            )
        )
    located.sort(key=lambda item: (item.extent[0], item.extent[1]))
    scan = find_measurement(store, "pii_scan")
    unplaced = [dict(record) for record in ((scan.attributes.get("unplaced_findings") or ()) if scan else ())]
    return located, unplaced


def _runs(ids: Sequence[str], order: Mapping[str, int]) -> list[list[str]]:
    """Word ids grouped into runs adjacent in the stream, in stream order."""
    runs: list[list[str]] = []
    for word_id in sorted(ids, key=lambda item: order[item]):
        if runs and order[word_id] == order[runs[-1][-1]] + 1:
            runs[-1].append(word_id)
        else:
            runs.append([word_id])
    return runs


def _approved(approvals: Sequence[str], tokens: Sequence[tuple[str, Entity]]) -> set[str]:
    """The word ids every approved name places on, over every place it occurs."""
    return {word.id for approval in approvals for word in _place(str(approval), tokens)}


_SUFFIXES = ("ing", "es", "ed", "s")


def _stem(key: str) -> str:
    """A folded token with one inflectional suffix dropped, where at least four letters stay."""
    for suffix in _SUFFIXES:
        if key.endswith(suffix) and len(key) - len(suffix) >= 4:
            return key[: -len(suffix)]
    return key


def _stems(text: str) -> set[str]:
    """The stems one word of text matches by: the word whole, its hyphen joints dropped, and each piece."""
    key = policy_fold(text)
    if not key:
        return set()
    pieces = [piece for piece in key.split("-") if piece]
    return {_stem(key), _stem(key.replace("-", "")), *(_stem(piece) for piece in pieces)}


def task_text_keys(task_text: Sequence[str]) -> frozenset[str]:
    """The stems of every word of the task's own texts: its stimulus, its target words, its instructions.

    Args:
        task_text: The texts, as the recording declares them.

    Returns:
        :func:`_stems` of every whitespace-separated word, and of each run of hyphen-joined words written
        apart ("ninety-three" also as "ninetythree").
    """
    keys: set[str] = set()
    for text in task_text:
        for word in str(text or "").split():
            keys |= _stems(word)
    return frozenset(keys)


def task_texts(hint: Any) -> tuple[str, ...]:  # noqa: ANN401 — an AudioHints or None
    """The task's own texts a recording declares: what it was asked to say or define, and its instructions.

    Args:
        hint: The recording's declaration (``AudioHints``), or None.

    Returns:
        Each expected prompt's text, then the instructions; empty where the recording declares none.
    """
    if hint is None:
        return ()
    prompts = [
        str(expected.text)
        for expected in (getattr(hint, "expected_speech", None) or [])
        if getattr(expected, "text", None)
    ]
    instructions = getattr(hint, "instructions", None)
    return tuple(prompts + ([str(instructions)] if instructions else []))


def task_text_positions(texts: Sequence[str], task_text: Sequence[str]) -> set[int]:
    """Which transcript words are the task's own text, inflections included.

    Args:
        texts: The transcript's word surfaces, in order.
        task_text: The task's stimulus, target words and instructions.

    Returns:
        The positions of content words whose whole stem, or every hyphen piece's stem, is a word of the
        task's texts.
    """
    keys = task_text_keys(task_text)
    if not keys:
        return set()
    found: set[int] = set()
    for position, text in enumerate(texts):
        if not is_content_word(text):
            continue
        key = policy_fold(text)
        pieces = [piece for piece in key.split("-") if piece]
        if (
            _stem(key) in keys
            or _stem(key.replace("-", "")) in keys
            or (pieces and all(_stem(p) in keys for p in pieces))
        ):
            found.add(position)
    return found


def _identifier_shaped(text: str) -> bool:
    """Whether a word could be part of a number, a date or an age.

    Args:
        text: A word's surface.

    Returns:
        True for a digit, an ``@``, or a number, ordinal or decade word, whole or as a hyphenated piece
        ("twenty-second", "fifties").
    """
    if any(ch.isdigit() for ch in text) or "@" in text:
        return True
    key = policy_fold(text)
    if not key:
        return False
    tables = policy_tables()
    words = {"thousand"} | set(tables["ordinal_words"]) | set(tables["age_decades"])
    return is_number(key) or key in words or any(is_number(piece) or piece in words for piece in key.split("-"))


@dataclass(frozen=True)
class _NonTaskSpeech:
    """What the owning branch's ``task_speech_reading`` gives the mask plan.

    Attributes:
        timed: The speech words outside the task with usable timing, where the reading reaches its bound.
        untimed: Those with none, where it does.
        item_ids: An item-set family's item-run words, which are task content.
        extensive: Whether an item-set family's speech outside the item runs reaches ``extensive_fraction``.
    """

    timed: frozenset[str] = frozenset()
    untimed: tuple[str, ...] = ()
    item_ids: frozenset[str] = frozenset()
    extensive: bool = False


def _non_task_speech(store: ProvStore) -> _NonTaskSpeech:
    """The speech outside the task, as the owning branch's reading names it (``task_speech``).

    Args:
        store: The provenance store, read for the ``task_speech_reading`` measurement.

    Returns:
        The words to mask and the task-content words; empty for a family whose speech outside the task is
        not read, where no reading was written, or (for the words to mask) where it stays under its bound.
    """
    family = declared_task_family(store)
    bound = words_min_for(family)
    reading = find_measurement(store, TASK_SPEECH_READING) if bound is not None else None
    if reading is None or bound is None:
        return _NonTaskSpeech()
    attributes = reading.attributes
    item_ids = (
        frozenset(str(i) for i in attributes.get("task_content_ids") or ()) if item_set_family(family) else frozenset()
    )
    fraction = attributes.get("off_task_fraction")
    extensive = (
        item_set_family(family)
        and fraction is not None
        and float(fraction) >= float(task_speech_parameters()["extensive_fraction"])
    )
    if int(attributes.get("words_n") or 0) < bound:
        return _NonTaskSpeech(item_ids=item_ids)
    untimed = tuple(str(word_id) for word_id in attributes.get("untimed_ids") or ())
    timed = frozenset(str(word_id) for word_id in attributes.get("word_ids") or ()) - set(untimed)
    return _NonTaskSpeech(timed, untimed, item_ids, extensive)


def mask_plan(
    store: ProvStore,
    *,
    reviewer_applies: bool,
    padding_ms: int,
    condition_categories: Sequence[str] = ("CONDITION",),
    protected_categories: Sequence[str] = (),
    cohort_conditions: str | None = None,
    lexicon: TaskLexicon | None = None,
    language: str | None = None,
    name_approvals: Sequence[str] = (),
    task_text: Sequence[str] = (),
) -> MaskPlan:
    """Which words stay masked under the redaction policy, the reviewer's unmasks and the content-word trim.

    A mask stands for one detector finding, or several covering exactly the same words, and covers the
    residue words that finding was placed on -- never a neighbour its padding or timing reaches, never a
    task word, never the whole transcript. The policy then masks, whatever a detector or the reviewer
    said, every residue word that writes a date element (a year, a month, a holiday), an age
    or a state (:mod:`~senselab.text.tasks.pii_detection.redaction_policy`), placing a :data:`POLICY`
    mask where no finding covers it; it keeps a person's name, written as a proper noun, masked until a
    human approves it (``name_approvals``), and a place below a country masked. It releases by kind a
    time of day, a weekday, a season, a relative reference or a length of time (:func:`time_by_kind`), a kinship
    word, and a country that is the whole of a place name unless a reviewer ``redact`` entry names it.
    A health condition the reviewer lists is never masked. A word otherwise leaves its mask when a
    reviewer ``release`` entry names it -- whole-token runs over the whole consensus transcript the
    reviewer read, kept on residue words, at every place the quote occurs -- or
    names the same token elsewhere, unless a ``redact`` entry places on that token;
    or when it is not a residue content word
    (:func:`~senselab.audio.workflows.triage.residue.is_content_word`). In a recording declared in a
    language other than English, a word of a name-family mask not written as a proper noun is released.
    A word lying on the declared task's own events (:func:`task_content`) is task content: no finding,
    reviewer entry or rule masks it, and neither any other word reading as the token a finding read on it.
    A decision about a token holds at every occurrence (:func:`word_keys`, the consensus surface or any
    recogniser's reading): a reviewer ``release`` releases every covered occurrence the policy does not
    lock, and a name kept masked -- adjacent kept name words matched as one run -- masks every other
    occurrence that is not task content, approved, a condition or released by kind, placing a
    :data:`PROPAGATED` mask where no finding covers it. Where the reading recorded the task and the whole
    consensus transcript with its PII annotations and every recogniser's reading (``review_inputs``,
    :func:`review_inputs_complete`), a release that frees an
    occurrence of a token outranks the name kept masked: the token's person or place lock is lifted at
    every covered occurrence and no mask spreads onto a released word. Otherwise the kept name wins.
    The kept words are cut into one extent per adjacent run, padded up to the nearest unmasked word.
    The rules are in ``specs/20261003-redaction-policy-v7/design.md`` and
    ``specs/20261007-task-events-in-background/design.md`` ("Redaction decisions per token").

    Args:
        store: The provenance store, carrying SPEECH's findings and residue, REDACT's plan and
            REVIEW's annotation.
        reviewer_applies: Whether the fold lets the reviewer's ``release`` entries unmask words.
        padding_ms: ``redaction.padding_ms``, the margin kept around each run of masked words.
        condition_categories: Upper-cased categories whose ``proposal`` entries are health conditions,
            as a reading written before conditions had their own part carries them.
        protected_categories: Upper-cased detector categories under which a marked word written as a
            proper noun counts as content: the trim never releases it.
        cohort_conditions: The cohort profile a condition's text is read against, as
            ``verdict.cohort_conditions``; every condition is ``other`` where None.
        lexicon: The declared family's task lexicon; a word inside one of its phrases is task content
            no mask covers. Where None, the family's declared names, compared exactly.
        language: The language the recording declares (its sidecar's ``language``), or None.
        name_approvals: The person names a human approved for release in this recording.
        task_text: The task's own texts -- its stimulus or target words and its instructions; a word of
            them, inflections included (:func:`task_text_positions`), is task content no mask covers.

    Returns:
        The plan.
    """
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    reading = dict(annotation.attributes) if annotation is not None else {}
    entries = list(reading.get("proposal") or ())
    condition_set = {category.upper() for category in condition_categories}
    condition_indices = tuple(
        index for index, entry in enumerate(entries) if str(entry.get("category") or "").upper() in condition_set
    )
    read = reading.get("status") in ("clean", "flagged")
    try:
        residue = residue_words(store)
    except ValueError:
        residue = []
    residue_ids = {word.id for word in residue}
    tokens = _tokens(residue)
    words = [word for word in consensus_words(store) if word.extent is not None]
    transcript_tokens = _tokens(words)
    by_id = {word.id: word for word in words}
    order = {word.id: position for position, word in enumerate(words)}
    vocabulary = lexicon if lexicon is not None else declared_names_lexicon(declared_task_family(store))
    word_texts = [str(w.attributes.get("text") or "") for w in words]
    declared_ids = {words[i].id for i in vocabulary.positions(word_texts)}
    lexicon_ids = frozenset(declared_ids)
    task_text_ids = {words[i].id for i in task_text_positions(word_texts, task_text)}
    declared_ids |= task_text_ids
    live_findings = live_entities(store, "pii")
    task_words = task_content(store, words, live_findings, declared_task_family(store), lexicon_ids)
    off_task = _non_task_speech(store)
    excluded_ids = declared_ids | task_words.ids | off_task.item_ids
    located, unplaced_records = _located_findings(store, residue_ids, excluded_ids)
    finding_word_ids_of = {
        entity.id: {str(i) for i in (entity.attributes.get("word_ids") or ())} for entity in live_findings
    }
    finding_word_ids = {word_id for ids in finding_word_ids_of.values() for word_id in ids}
    read_keys = finding_keys(live_findings, by_id)
    keys_of = {word.id: word_keys(word) for word in words}
    findings_of: dict[str, list[str]] = {}
    for entity in live_findings:
        for word_id in entity.attributes.get("word_ids") or ():
            findings_of.setdefault(str(word_id), []).append(entity.id)
    unplaced_categories = [str(record.get("category") or "") for record in unplaced_records]
    redact_ran = find_verdict(store, NODE) is not None
    non_english = bool(language) and not str(language).strip().lower().startswith("en")

    groups: dict[tuple[str, ...], list[_Finding]] = {}
    for finding in located:
        groups.setdefault(finding.word_ids, []).append(finding)
    # A word's categories are those of the located findings placed on it. An unplaced finding marks
    # every word of the transcript, and those marks say nothing about any one word.
    found: dict[str, set[str]] = {}
    for finding in located:
        for word_id in finding.word_ids:
            found.setdefault(word_id, set()).add(finding.category)

    protected = {category.upper() for category in protected_categories}
    previous = {
        word.id: (str(words[i - 1].attributes.get("text") or "") if i else None) for i, word in enumerate(words)
    }

    def proper(word: Entity) -> bool:
        return is_proper_form(str(word.attributes.get("text") or ""), previous.get(word.id))

    def text_of(word: Entity) -> str:
        return str(word.attributes.get("text") or "")

    # A capitalised function word is part of a name only beside the name's own words: "The" in "The
    # Green Mile" is, a lone "She" a detector tagged PERSON is not. The name homographs stand alone.
    in_name: set[str] = set()
    for finding in located:
        if finding.category.upper() not in protected:
            continue
        spanned = [by_id[word_id] for word_id in finding.word_ids if word_id in by_id]
        if any(proper(word) and is_content_word(text_of(word)) for word in spanned):
            in_name.update(word.id for word in spanned)

    def protected_name(word: Entity) -> bool:
        if not (protected & {category.upper() for category in found.get(word.id, set())} and proper(word)):
            return False
        if is_content_word(text_of(word)) or word.id in in_name:
            return True
        letters = "".join(ch for ch in text_of(word) if ch.isalpha())
        return is_name_homograph(text_of(word)) or (len(letters) >= 2 and letters.isupper())

    def capitalised(word: Entity) -> bool:
        if non_english:
            return proper(word)
        surface = text_of(word).strip(_QUOTE_EDGE)
        return surface[:1].isupper() and _match_token(surface) not in ("i", "i'm", "i'll", "i've", "i'd")

    def content(word: Entity) -> bool:
        if word.id in task_words.ids or (residue_ids and word.id not in residue_ids):
            return False
        if protected_name(word):
            return True
        return is_content_word(text_of(word))

    # The policy reads the residue's words in order, bracketed markers left out, so a cue and its
    # number ("I'm [UH] 73") stay adjacent.
    scan = [
        word
        for word in (residue or words)
        if word.extent is not None and not word.attributes.get("bracketed") and word.id not in excluded_ids
    ]
    scan_texts = [text_of(word) for word in scan]
    with_task = [
        word for word in (residue or words) if word.extent is not None and not word.attributes.get("bracketed")
    ]
    with_task_texts = [text_of(word) for word in with_task]
    task_policy_ids = {
        with_task[i].id for i in date_positions(with_task_texts) | state_positions(with_task_texts)
    } & task_text_ids
    date_ids = {scan[i].id for i in date_positions(scan_texts)}
    age_ids = {scan[i].id for i in age_positions(scan_texts)}
    state_ids = {scan[i].id for i in state_positions(scan_texts)}
    country_ids = {scan[i].id for start, end in country_runs(scan_texts) for i in range(start, end)}
    kinship_ids = {scan[i].id for i in kinship_positions(scan_texts)}
    always_ids = (date_ids | state_ids) - kinship_ids

    family_of_word: dict[str, str] = {}
    for member_ids, members in groups.items():
        for word_id in member_ids:
            family_of_word.setdefault(word_id, category_family(members[0].category))

    def quoted_words(quote: str) -> list[Entity]:
        hits = [word for word in _place(quote, transcript_tokens) if word.id in residue_ids]
        return hits or _place(quote, tokens)

    def placed_words(quote: str) -> tuple[list[Entity], str]:
        hits = quoted_words(quote)
        if hits:
            return hits, PLACED_WORDS
        hits = _place_substring(quote, residue)
        return hits, (PLACED_SUBSTRING if hits else "")

    profile = load_cohort_profile(cohort_conditions) if cohort_conditions else None
    condition_quotes = [
        (str(item.get("text") or ""), str(item.get("why") or ""))
        for item in reading.get("conditions") or ()
        if isinstance(item, Mapping) and str(item.get("text") or "").strip()
    ] + [(str(entries[index].get("text") or ""), str(entries[index].get("why") or "")) for index in condition_indices]
    conditions: list[ConditionSpan] = []
    condition_ids: set[str] = set()
    for quote, why in condition_quotes:
        hits, _ = placed_words(quote)
        diagnosis = profile.diagnosis(quote) if profile is not None else None
        conditions.append(
            ConditionSpan(
                text=quote,
                why=why,
                word_ids=tuple(word.id for word in hits),
                texts=tuple(text_of(word) for word in hits),
                condition_kind=COHORT if diagnosis else OTHER,
                cohort_diagnosis=diagnosis or "",
            )
        )
        condition_ids.update(word.id for word in hits)

    named: set[str] = set()
    unplaced_quotes: list[str] = []
    releases: list[dict[str, Any]] = []
    not_person_ids: set[str] = set()
    relabelled_place_ids: set[str] = set()
    place_reason_ids: set[str] = set()
    historical_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if str(entry.get("action")) != "release" or index in condition_indices:
            continue
        hits = quoted_words(str(entry.get("text") or ""))
        category = str(entry.get("category") or "OTHER").upper()
        relabel = str(entry.get("relabel") or "").strip().lower()
        place_reason = str(entry.get("place_reason") or "").strip().lower()
        relabel = relabel if relabel in RELABELS else ""
        place_reason = place_reason if place_reason in PLACE_REASONS else ""
        releases.append(
            {
                "text": str(entry.get("text") or ""),
                "category": category,
                "safe_harbor": str(entry.get("safe_harbor") or "") or "".join(safe_harbor_codes(category)[:1]),
                "why": str(entry.get("why") or ""),
                "relabel": relabel,
                "place_reason": place_reason,
                "placed": bool(hits),
            }
        )
        if not hits:
            unplaced_quotes.append(str(entry.get("text") or ""))
            continue
        named.update(word.id for word in hits)
        hit_ids = {word.id for word in hits}
        if relabel == RELABEL_PLACE:
            relabelled_place_ids |= hit_ids
        elif relabel:
            not_person_ids |= hit_ids
        if place_reason:
            place_reason_ids |= hit_ids
        if place_reason == PLACE_HISTORICAL:
            historical_ids |= hit_ids
    always_ids -= historical_ids & state_ids
    redact_placed = {
        index: placed_words(str(entry.get("text") or ""))
        for index, entry in enumerate(entries)
        if str(entry.get("action")) != "release" and index not in condition_indices
    }
    country_redacted = {word.id for hits, _ in redact_placed.values() for word in hits if word.id in country_ids}
    blocked_tokens = {
        _match_token(str(word.attributes.get("text") or "")) for hits, _ in redact_placed.values() for word in hits
    }
    approved_ids = _approved(name_approvals, tokens)
    name_kinds = name_families()

    covered_ids = {word_id for member_ids in groups for word_id in member_ids}
    # What the policy decides of each word a detector mask covers, before the reviewer: kept whatever
    # it says (locked), or released by kind.
    locked: dict[str, str] = (
        {word_id: LOCK_DATE for word_id in date_ids}
        | {word_id: LOCK_AGE for word_id in age_ids}
        | {word_id: LOCK_PLACE for word_id in state_ids}
    )
    kind_of: dict[str, str] = {}
    for member_ids, members in groups.items():
        families = {category_family(finding.category) for finding in members}
        group = [by_id[word_id] for word_id in member_ids if word_id in by_id]
        if families == {DATE_TIME_FAMILY}:
            timed = time_by_kind([text_of(word) for word in group])
            for position, word in enumerate(group):
                if position in timed and all(
                    category_family(category) == DATE_TIME_FAMILY for category in found.get(word.id, ())
                ):
                    kind_of.setdefault(word.id, KIND_TIME)
            if any(finding.category.upper() == "AGE" for finding in members):
                for word in group:
                    if word.id not in kind_of and content(word):
                        locked.setdefault(word.id, LOCK_AGE)
        if families == {"LOCATION"}:
            propers = [word for word in group if capitalised(word)]
            whole_country = bool(propers) and all(word.id in country_ids for word in propers)
            for word in group:
                if word.id in country_ids and whole_country and word.id not in country_redacted:
                    kind_of.setdefault(word.id, KIND_COUNTRY)
                elif capitalised(word):
                    locked.setdefault(word.id, LOCK_PLACE)
        if families == {"PERSON"}:
            for word in group:
                if capitalised(word) and word.id not in approved_ids:
                    locked.setdefault(word.id, LOCK_PERSON)
    lists = _time_release()
    standalone_time = lists["time_of_day"] | lists["weekdays"] | lists["relative_days"] | lists["seasons"]
    standalone_ids = {word.id for word in scan if _time_token(text_of(word)) in standalone_time} - always_ids
    for word_id in standalone_ids & covered_ids:
        kind_of.setdefault(word_id, KIND_TIME)
        locked.pop(word_id, None)
    for word_id in kinship_ids:
        kind_of[word_id] = KIND_KINSHIP
        locked.pop(word_id, None)
    for word_id in country_redacted:
        locked.setdefault(word_id, LOCK_PLACE)
    for word_id in condition_ids:
        if word_id not in always_ids:
            locked.pop(word_id, None)
    for word_id in always_ids:
        kind_of.pop(word_id, None)

    def surface_key(word_id: str) -> str:
        return _match_token(text_of(by_id[word_id])) if word_id in by_id else ""

    def same_term(ids: set[str]) -> set[str]:
        tokens = {surface_key(word_id) for word_id in ids} - {""}
        return ids | {
            word_id
            for word_id in family_of_word
            if keys_of.get(word_id, set()) & tokens and surface_key(word_id) not in blocked_tokens
        }

    not_person_ids = same_term(not_person_ids)
    relabelled_place_ids = same_term(relabelled_place_ids)
    place_reason_ids = same_term(place_reason_ids)
    historical_ids = same_term(historical_ids)
    always_ids -= historical_ids & state_ids
    for word_id in not_person_ids:
        if locked.get(word_id) == LOCK_PERSON:
            locked.pop(word_id)
    for word_id in relabelled_place_ids:
        if locked.get(word_id) == LOCK_PERSON:
            locked[word_id] = LOCK_PLACE
    for word_id in place_reason_ids:
        if locked.get(word_id) == LOCK_PLACE and (word_id not in state_ids or word_id in historical_ids):
            locked.pop(word_id)

    named_tokens = {surface_key(word_id) for word_id in named} - {""}
    propagated = {
        word_id
        for word_id in family_of_word
        if word_id not in named
        and content(by_id[word_id])
        and keys_of.get(word_id, set()) & named_tokens
        and surface_key(word_id) not in blocked_tokens
    }
    released_from = {
        word_id: tuple(
            sorted(source for source in named if surface_key(source) and surface_key(source) in keys_of[word_id])
        )
        for word_id in propagated
    }
    applied = reviewer_applies and bool(named)
    released = ((named | propagated) - set(locked)) if applied else set()
    reviewer_wins = applied and review_inputs_complete(reading.get("review_inputs"))
    if reviewer_wins:
        for token in named_tokens:
            occurrences = {
                word_id
                for word_id in named | propagated
                if surface_key(word_id) == token or token in keys_of.get(word_id, set())
            }
            if not occurrences & released:
                continue
            for word_id in occurrences:
                if locked.get(word_id) in (LOCK_PERSON, LOCK_PLACE):
                    locked.pop(word_id)
                    released.add(word_id)
    approved_release = {word_id for word_id in approved_ids if word_id not in locked}
    # A name the reviewer released takes its lower-case words with it: "Gladiator fighter" tagged
    # PERSON, with "Gladiator" released, does not leave "fighter" masked.
    with_head: set[str] = set()
    if applied or approved_release:
        for member_ids, members in groups.items():
            if not all(category_family(finding.category) in name_kinds for finding in members):
                continue
            group = [by_id[word_id] for word_id in member_ids if word_id in by_id]
            heads = [word for word in group if text_of(word)[:1].isupper() and word.id in released | approved_release]
            if not heads:
                continue
            with_head.update(
                word.id
                for word in group
                if word.id not in released
                and word.id not in locked
                and text_of(word) == text_of(word).lower()
                and _match_token(text_of(word)) not in blocked_tokens
            )
        released |= with_head
    ordinary = {
        _match_token(str(word.attributes.get("text") or ""))
        for word in words
        if word.id not in covered_ids and word.id not in finding_word_ids
    }

    def common_word(word: Entity) -> bool:
        text = str(word.attributes.get("text") or "")
        return text == text.lower() and _match_token(text) in ordinary

    name_only: set[str] = set()
    elsewhere: set[str] = set()
    for member_ids, members in groups.items():
        group = [by_id[word_id] for word_id in member_ids]
        families = {category_family(finding.category) for finding in members}
        if non_english and families <= name_kinds | {"OTHER"}:
            name_only.update(word.id for word in group if not proper(word))
        elif families <= name_kinds:
            if any(proper(word) for word in group):
                first, last = 0, len(group) - 1
                while first <= last and not proper(group[first]) and common_word(group[first]):
                    name_only.add(group[first].id)
                    first += 1
                while last > first and not proper(group[last]) and common_word(group[last]):
                    name_only.add(group[last].id)
                    last -= 1
        else:
            elsewhere.update(member_ids)
    kind_cut = name_only - elsewhere - always_ids - set(locked)
    redact_hit_ids = {word.id for hits, _ in redact_placed.values() for word in hits}
    # A name finding that holds a capitalised word keeps its lower-case words for the name trim above.
    named_groups = {
        word_id
        for member_ids, members in groups.items()
        if {category_family(finding.category) for finding in members} <= name_kinds
        and any(capitalised(by_id[word_id]) for word_id in member_ids if word_id in by_id)
        for word_id in member_ids
    }
    not_proper = {
        word_id
        for word_id in covered_ids - named_groups
        if word_id in by_id
        and not any(ch.isupper() for ch in text_of(by_id[word_id]))
        and word_id not in always_ids
        and word_id not in age_ids
        and locked.get(word_id) != LOCK_AGE
        and word_id not in redact_hit_ids
        and word_id not in kind_of
        and word_id not in kind_cut
        and word_id not in condition_ids
        and not _identifier_shaped(text_of(by_id[word_id]))
        and not is_state_word(text_of(by_id[word_id]))
    }
    keepable = {
        word_id
        for member_ids in groups
        for word_id in member_ids
        if content(by_id[word_id])
        and word_id not in kind_cut
        and word_id not in kind_of
        and word_id not in not_proper
        and (word_id not in condition_ids or word_id in always_ids)
    } | (always_ids & covered_ids)
    kept_ids = (keepable - released - approved_release) | (always_ids & covered_ids)

    # The policy's own masks: words it always masks that no finding covered.
    policy_masks: list[tuple[str, list[Entity]]] = []
    for word_ids, family in (
        (date_ids - covered_ids, DATE_TIME_FAMILY),
        ((state_ids & always_ids) - covered_ids, "LOCATION"),
    ):
        for run in _runs(sorted(word_ids & set(order)), order):
            policy_masks.append((family, [by_id[word_id] for word_id in run]))
            kept_ids |= set(run)

    proposals: list[ReviewerSpan] = []
    reviewer_masks: list[tuple[str, list[Entity]]] = []
    unplaced_families = {category_family(category) for category in unplaced_categories}
    placed_families: set[str] = set()

    for index, (hits, placed) in redact_placed.items():
        entry = entries[index]
        quote = str(entry.get("text") or "")
        category = str(entry.get("category") or "OTHER").upper()
        family = category_family(category)
        timed = time_by_kind([text_of(word) for word in hits]) if family == DATE_TIME_FAMILY else set()
        by_kind_ids = (
            {word.id for position, word in enumerate(hits) if position in timed} | kinship_ids | standalone_ids
        )
        effective = [
            word
            for word in hits
            if content(word) and (word.id not in by_kind_ids or word.id in always_ids) and word.id not in condition_ids
        ]
        countries = [word for word in effective if word.id in country_ids]
        rest = [word for word in effective if word.id not in country_ids]
        uncovered_countries = [word for word in countries if word.id not in kept_ids]
        if uncovered_countries:
            reviewer_masks.append(("LOCATION", uncovered_countries))
            kept_ids |= {word.id for word in uncovered_countries}
        uncovered = [word for word in rest if word.id not in kept_ids]
        if hits and not effective:
            agreement = BY_KIND
        elif countries and not rest:
            agreement = COUNTRY_MASKED
        elif hits and not uncovered:
            agreement = AGREED_MASKED
            if family in unplaced_families:
                placed_families.add(family)
        elif hits and family in unplaced_families and redact_ran:
            agreement = AGREED_PLACED
            placed_families.add(family)
            reviewer_masks.append((family, uncovered))
            kept_ids |= {word.id for word in uncovered}
        else:
            agreement = NEW
        proposals.append(
            ReviewerSpan(
                category=category,
                text=quote,
                placed=placed,
                word_ids=tuple(word.id for word in hits),
                texts=tuple(text_of(word) for word in hits),
                masked_ids=tuple(word.id for word in hits if word.id in kept_ids),
                content_ids=tuple(word.id for word in effective),
                agreement=agreement,
                entry_index=index,
            )
        )

    reviewer_name_ids = {word.id for family, group in reviewer_masks if family in name_kinds for word in group}
    name_ids = {
        word_id
        for word_id in kept_ids
        if {category_family(category) for category in found.get(word_id, ())} & name_kinds
        or locked.get(word_id) in (LOCK_PERSON, LOCK_PLACE)
        or word_id in reviewer_name_ids
    }
    patterns: list[tuple[tuple[frozenset[str], ...], tuple[str, ...]]] = []
    for run in _runs(sorted(name_ids & set(order)), order):
        pattern = tuple(
            frozenset(key for key in (read_keys.get(word_id) or {surface_key(word_id)}) if key) - task_words.keys
            for word_id in run
        )
        if all(pattern) and any(is_content_word(key) for keys in pattern for key in keys):
            patterns.append((pattern, tuple(run)))
    spread_from: dict[str, tuple[str, ...]] = {}
    for at in range(len(words)):
        for pattern, sources in patterns:
            width = len(pattern)
            if at + width > len(words) or not all(
                keys_of[words[at + offset].id] & pattern[offset] for offset in range(width)
            ):
                continue
            for word in words[at : at + width]:
                if (
                    word.id in kept_ids
                    or word.id in excluded_ids
                    or word.id in approved_ids
                    or word.id in condition_ids
                    or word.id in kind_of
                    or (reviewer_wins and word.id in released)
                    or word.attributes.get("bracketed")
                ):
                    continue
                spread_from[word.id] = tuple(sorted({*spread_from.get(word.id, ()), *sources}))
    kept_ids |= set(spread_from)
    propagated_masks: list[tuple[str, list[Entity]]] = []
    for run in _runs(sorted(set(spread_from) - covered_ids), order):
        source = spread_from[run[0]][0]
        family = family_of_word.get(source) or (LOCATION_FAMILY if locked.get(source) == LOCK_PLACE else PERSON_FAMILY)
        propagated_masks.append((family, [by_id[word_id] for word_id in run]))
    speech_ids = (set(off_task.timed) - excluded_ids) & set(order)
    speech_untimed = off_task.untimed
    kept_ids |= speech_ids
    speech_masks = [
        (NON_TASK_SPEECH_CATEGORY, [by_id[word_id] for word_id in run]) for run in _runs(sorted(speech_ids), order)
    ]

    def state_of(word: Entity) -> str:
        if word.id in kept_ids:
            return MASKED
        if word.id in condition_ids:
            return RELEASED_CONDITION
        if word.id in kind_of and content(word):
            return RELEASED_BY_KIND
        if word.id in not_proper and content(word):
            return RELEASED_NOT_PROPER
        if word.id in approved_release and word.id in keepable:
            return UNMASKED_BY_APPROVAL
        if word.id in released and word.id in keepable:
            return UNMASKED_BY_REVIEWER
        return UNMASKED_BY_TRIM

    pending: list[tuple[RedactionExtent, list[Entity], tuple[str, ...], str, int]] = []
    for member_ids, members in groups.items():
        low = min(finding.extent[0] for finding in members)
        high = max(finding.extent[1] for finding in members)
        family = category_family(members[0].category)
        pending.append(
            (
                RedactionExtent(start=low, end=high, category=family),
                [by_id[word_id] for word_id in member_ids],
                tuple(sorted({finding.category for finding in members})),
                DETECTOR,
                max(finding.task_words_n for finding in members),
            )
        )
    for source, masks in (
        (REVIEWER, reviewer_masks),
        (POLICY, policy_masks),
        (PROPAGATED, propagated_masks),
        (NON_TASK_SPEECH, speech_masks),
    ):
        for family, group in masks:
            if not group:
                continue
            hull = (min(word_hull(word)[0] for word in group), max(word_hull(word)[1] for word in group))
            pending.append((RedactionExtent(start=hull[0], end=hull[1], category=family), group, (family,), source, 0))
    redact_planned = planned_extents(store)
    silent = [
        extent
        for extent in redact_planned
        if not any(_overlaps(word_hull(word), (extent.start, extent.end)) for word in words)
    ]
    pending.sort(key=lambda item: (item[0].start, item[0].end))

    owner: dict[str, int] = {}
    for position in sorted(range(len(pending)), key=lambda i: len(pending[i][1])):
        for word in pending[position][1]:
            if word.id in kept_ids:
                owner.setdefault(word.id, position)

    outcomes: list[MaskOutcome] = []
    for position, (extent, group, categories, source, task_words_n) in enumerate(pending):
        states = tuple(
            MaskWord(
                word_id=word.id,
                text=text_of(word),
                state=(state := state_of(word)),
                named=word.id in named,
                content=content(word),
                finding=word.id in found,
                categories=tuple(sorted(found.get(word.id, set()))),
                proper=proper(word),
                propagated=word.id in propagated or word.id in spread_from,
                propagation=PROPAGATION_MASK
                if word.id in spread_from
                else PROPAGATION_RELEASE
                if word.id in propagated
                else "",
                propagated_from=spread_from.get(word.id) or released_from.get(word.id) or (),
                source_findings=tuple(
                    sorted(
                        {
                            finding_id
                            for source_id in (spread_from.get(word.id) or released_from.get(word.id) or ())
                            for finding_id in findings_of.get(source_id, ())
                        }
                    )
                ),
                kind_cut=word.id in kind_cut,
                with_head=word.id in with_head,
                locked=locked.get(word.id, "") if state == MASKED else "",
                kind=kind_of.get(word.id, "") if state == RELEASED_BY_KIND else "",
            )
            for word in group
        )
        own = [word for word in group if owner.get(word.id) == position]
        runs = _final_extents(own, words, kept_ids, extent.category, padding_ms / 1000.0) if own else []
        kept_here = [word for word in group if word.id in kept_ids]
        if group and len(kept_here) == len(group):
            outcome = MASK_UNCHANGED
        elif not kept_here:
            outcome = MASK_UNMASKED
        elif any(word.state == UNMASKED_BY_REVIEWER for word in states):
            outcome = MASK_PARTLY_UNMASKED
        else:
            outcome = MASK_TRIMMED
        outcomes.append(
            MaskOutcome(
                planned=extent,
                words=states,
                final=tuple(run for run, _ in runs),
                final_words=tuple(members for _, members in runs),
                outcome=outcome,
                task_words_n=task_words_n,
                categories=categories,
                source=source,
            )
        )

    # REDACT's own extent stands wherever it hides exactly the kept words of the masks inside it and
    # reaches no other word, so the copy REDACT wrote is the one released.
    for extent in redact_planned:
        reach = [word for word in words if _overlaps(word_hull(word), (extent.start, extent.end))]
        if not reach or any(word.id not in kept_ids for word in reach):
            continue
        reach_ids = {word.id for word in reach}
        inside = [
            position
            for position, outcome in enumerate(outcomes)
            if outcome.final_words and {word_id for members in outcome.final_words for word_id in members} <= reach_ids
        ]
        covered = {word_id for position in inside for members in outcomes[position].final_words for word_id in members}
        if not inside or covered != reach_ids:
            continue
        order_ids = [word.id for word in reach]
        labelled = replace(extent, category=outcomes[inside[0]].planned.category)
        for rank, position in enumerate(inside):
            outcomes[position] = replace(
                outcomes[position],
                final=(labelled,) if rank == 0 else (),
                final_words=(tuple(order_ids),) if rank == 0 else (),
            )
    # A planned extent that reaches no word masks audio no transcript word accounts for; it stands as
    # REDACT planned it.
    for extent in silent:
        parts = tuple(part for part in extent.category.split("+") if part)
        labelled = replace(extent, category=category_family(parts[0]) if parts else extent.category)
        outcomes.append(
            MaskOutcome(
                planned=labelled,
                words=(),
                final=(labelled,),
                final_words=((),),
                outcome=MASK_UNCHANGED,
                categories=parts or (extent.category,),
            )
        )
    outcomes.sort(key=lambda mask: (mask.planned.start, mask.planned.end))

    cleared = read and reading.get("original") == "clean"
    unplaced = tuple(
        UnplacedFinding(
            category=str(record.get("category") or ""),
            family=category_family(str(record.get("category") or "")),
            state=UNPLACED_PLACED
            if category_family(str(record.get("category") or "")) in placed_families
            else UNPLACED_CLEARED
            if cleared
            else UNPLACED_OPEN
            if read
            else UNPLACED_UNREAD,
            text=str(record.get("text") or ""),
        )
        for record in unplaced_records
    )
    off_mask = [
        str(entry.get("text") or "")
        for index, entry in enumerate(entries)
        if str(entry.get("action")) == "release"
        and index not in condition_indices
        and (hits := quoted_words(str(entry.get("text") or "")))
        and not any(word.id in family_of_word for word in hits)
    ]
    name_release_proposed = [
        str(entry.get("text") or "")
        for index, entry in enumerate(entries)
        if str(entry.get("action")) == "release"
        and index not in condition_indices
        and any(locked.get(word.id) == LOCK_PERSON for word in quoted_words(str(entry.get("text") or "")))
    ]
    return MaskPlan(
        masks=tuple(outcomes),
        reviewer_applied=applied,
        release_unplaced=tuple(unplaced_quotes),
        release_off_mask=tuple(off_mask),
        proposals=tuple(proposals),
        padding_ms=int(padding_ms),
        unplaced=unplaced,
        redact_planned=tuple(redact_planned),
        task_lexicon_ids=tuple(sorted(declared_ids & finding_word_ids)),
        task_text_ids=tuple(sorted(task_text_ids & (finding_word_ids | task_policy_ids))),
        task_event_ids=tuple(sorted(task_words.ids & finding_word_ids)),
        task_content_only=bool(located)
        and not unplaced_records
        and all(
            not finding.word_ids and finding_word_ids_of[finding.finding_id] & task_words.ids for finding in located
        ),
        reviewer_precedence=reviewer_wins,
        non_task_speech_ids=tuple(sorted(speech_ids, key=lambda word_id: order[word_id])),
        non_task_speech_untimed=speech_untimed,
        non_task_speech_extensive=off_task.extensive,
        named_no_words=reviewer_named_no_words(reading),
        releases=tuple(releases),
        conditions=tuple(conditions),
        condition_indices=condition_indices,
        language=str(language or ""),
        name_approvals=tuple(str(approval) for approval in name_approvals),
        name_release_proposed=tuple(name_release_proposed),
    )


def released_masks(store: ProvStore) -> tuple[list[RedactionExtent], dict[str, int] | None]:
    """The masks the fold released, as the ledger VERDICT wrote records them.

    Args:
        store: The provenance store, after VERDICT.

    Returns:
        ``(extents, owners)``: the final extents and which of them hides each masked word. REDACT's
        planned extents and None where no ledger stands, which renders by geometry.
    """
    ledger = find_measurement(store, PII_LEDGER)
    if ledger is None:
        return planned_extents(store), None
    extents: list[RedactionExtent] = []
    owners: dict[str, int] = {}
    for index, entry in enumerate(ledger.attributes.get("final_masks") or ()):
        extents.append(
            RedactionExtent(
                start=float(entry["start_s"]), end=float(entry["end_s"]), category=str(entry.get("category") or "")
            )
        )
        for word_id in entry.get("word_ids") or ():
            owners[str(word_id)] = index
    return extents, owners


def _holds_copy(artifacts_dir: Path, records: list[dict[str, Any]]) -> bool:
    """Whether the release directory holds the copy these records render.

    Args:
        artifacts_dir: The release directory.
        records: :func:`_render`'s records for the copy wanted.

    Returns:
        True where all of :data:`RELEASED_FILES` exist and the consensus artifact carries exactly
        these records, each mask's bounds included; or where it is not one this module wrote.
    """
    if not all((artifacts_dir / name).exists() for name in RELEASED_FILES):
        return False
    try:
        written = json.loads((artifacts_dir / "consensus.json").read_text())
    except (OSError, ValueError):
        return True
    if not isinstance(written, dict) or written.get("schema") != CONSENSUS_ARTIFACT_SCHEMA:
        return True
    return written.get("records") == json.loads(json.dumps(records))


def _has_stream(store: ProvStore, name: str) -> bool:
    """Whether a live stream carries this name.

    Args:
        store: The provenance store.
        name: The stream's ``name`` attribute.

    Returns:
        True where one does.
    """
    return any(entity.attributes.get("name") == name for entity in live_entities(store, "stream"))


def _masked_source(store: ProvStore, run_dir: Path) -> tuple[str, Audio]:
    """The stream REDACT's ``redacted`` copy was masked from, loaded from its sidecar.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The source stream's id and its audio.

    Raises:
        LookupError: If the store holds no ``redacted`` stream or does not say what it came from.
    """
    redacted_id, _ = resolve_stream(store, run_dir, STREAM_NAME)
    for source_id in store.derived_from(redacted_id):
        source = store.get_entity(source_id)
        if source.prov_type == "stream":
            path = Path(source.attributes["path"])
            return source_id, Audio(filepath=str(path if path.is_absolute() else run_dir / path))
    raise LookupError("the redacted stream records no source stream")


@dataclass(frozen=True)
class ReleasedAudio:
    """The audio of the redacted copy the fold releases, and what it was made from.

    Attributes:
        audio: The masked audio, at the masked source's own rate and channels.
        derived_from: The store ids it was made from: REDACT's ``redacted`` stream where the final
            masks are REDACT's own, else the masked source stream and the PII ledger.
        masks: The final masks, in seconds on the recording's time base.
        fill: What each mask was filled with.
        remasked: Whether the source was masked again with the final masks.
    """

    audio: Audio
    derived_from: tuple[str, ...]
    masks: tuple[RedactionExtent, ...]
    fill: str
    remasked: bool


MASK_SOURCE = "recording"
"""The store-held stream REDACT masks, and the fold masks where REDACT never ran."""


def released_audio(store: ProvStore, run_dir: Path, *, bleep_hz: float | None, fill: str) -> ReleasedAudio:
    """The audio of the redacted copy, as :func:`settle_release` writes it under ``redacted``.

    Args:
        store: The provenance store, after VERDICT.
        run_dir: The run directory sidecar paths are relative to.
        bleep_hz: ``redaction.bleep_hz``, for a re-masked copy under a bleep fill.
        fill: ``redaction.fill``, used where REDACT never ran and the fold's policy masks are the only
            masks; REDACT's own fill otherwise.

    Returns:
        The masked audio and its provenance.

    Raises:
        LookupError: If no ledger stands, or REDACT ran and wrote no ``redacted`` stream.
    """
    verdict = find_verdict(store, NODE)
    final, _ = released_masks(store)
    ledger = find_measurement(store, PII_LEDGER)
    if verdict is None:
        if ledger is None or not final:
            raise LookupError("REDACT concluded nothing and the fold masked nothing; there is no redacted copy")
        source_id, source = resolve_stream(store, run_dir, MASK_SOURCE)
        masked = apply_redactions(source, final, fill=fill, bleep_hz=bleep_hz)
        return ReleasedAudio(masked, (source_id, ledger.id), tuple(final), fill, remasked=True)
    fill = str(verdict.attributes.get("fill") or "")
    if same_extents(final, planned_extents(store)):
        stream_id, redacted = resolve_stream(store, run_dir, STREAM_NAME)
        return ReleasedAudio(redacted, (stream_id,), tuple(final), fill, remasked=False)
    source_id, source = _masked_source(store, run_dir)
    sources = (source_id,) if ledger is None else (source_id, ledger.id)
    masked = apply_redactions(source, final, fill=fill, bleep_hz=bleep_hz)
    return ReleasedAudio(masked, sources, tuple(final), fill, remasked=True)


def settle_release(
    store: ProvStore,
    release: str | None,
    release_ground: str | None,
    *,
    run_dir: Path,
    artifacts_dir: Path,
    bleep_hz: float | None,
    fill: str,
) -> dict[str, Path]:
    """Make the release directory hold exactly the copy the fold released.

    The directory holds a redacted copy only under ``redacted``; every other release
    empties it, including of a copy REDACT itself wrote on a pass. The copy masks the fold's final
    masks, as its ledger records them (:func:`released_masks`), and its text hides exactly the words
    the ledger keeps masked. Where the final extents are REDACT's own, the audio is REDACT's
    ``redacted`` stream: the copy REDACT wrote stands where it passed and already carries this
    text, and is written again otherwise. Elsewhere, and where REDACT never ran but the policy masked
    words, the source stream is masked with the final masks alone.

    Args:
        store: The provenance store, after VERDICT.
        release: The fold's release axis value.
        release_ground: The fold's release ground. Recorded on the ledger; not read here.
        run_dir: The run directory sidecar paths are relative to.
        artifacts_dir: The release directory.
        bleep_hz: ``redaction.bleep_hz``, for a re-masked copy under a bleep fill.
        fill: ``redaction.fill``, for a copy the fold masks where REDACT never ran.

    Returns:
        The written paths, keyed as :func:`_write_artifacts` keys them; empty where nothing was
        written.
    """
    verdict = find_verdict(store, NODE)
    if release != Release.REDACTED.value:
        for name in RELEASED_FILES:
            (artifacts_dir / name).unlink(missing_ok=True)
        return {}
    final, owners = released_masks(store)
    if verdict is None and not final:
        return {}
    planned = planned_extents(store)
    words = consensus_words(store)
    records, text, _ = _render(words, final, owners)
    if verdict is not None and same_extents(final, planned):
        if verdict.attributes.get("outcome") == Outcome.PASS.value and (
            _holds_copy(artifacts_dir, records) or not _has_stream(store, STREAM_NAME)
        ):
            return {}
    audio = released_audio(store, run_dir, bleep_hz=bleep_hz, fill=fill).audio
    return _write_artifacts(audio, text, records, artifacts_dir)


REDACTION_SPAN = "redaction"
"""The ``span`` name REDACT gives each planned extent. What a later reader recovers the plan from."""


def planned_extents(store: ProvStore) -> list[RedactionExtent]:
    """The extents REDACT planned over this recording, recovered from the store's own spans.

    Args:
        store: The provenance store.

    Returns:
        The live redaction spans as extents, in stream order. Empty where REDACT never ran or
        planned nothing, which are the same thing to a reader of the released text.
    """
    spans = [
        span
        for span in live_entities(store, "span")
        if span.attributes.get("name") == REDACTION_SPAN and span.extent is not None
    ]
    extents = [
        RedactionExtent(start=span.extent[0], end=span.extent[1], category=str(span.attributes.get("category") or ""))
        for span in spans
        if span.extent is not None
    ]
    return sorted(extents, key=lambda extent: (extent.start, extent.end))


def residue_words(store: ProvStore) -> list[Entity]:
    """The consensus words SPEECH's scan read: the recording's lexical residue, in stream order.

    Args:
        store: The provenance store.

    Returns:
        The words the live ``pii_scan`` measurement names. Empty where SPEECH recorded no scan, or
        where the residue is empty.

    Raises:
        ValueError: If the live ``pii_scan`` measurement names no residue, which a store written
            before the residue existed does not.
    """
    scan = find_measurement(store, "pii_scan")
    if scan is None:
        return []
    ids = scan.attributes.get("residue_word_ids")
    if ids is None:
        raise ValueError("the pii_scan measurement names no residue; the store predates the lexical residue")
    wanted = {str(word_id) for word_id in ids}
    return [word for word in consensus_words(store) if word.id in wanted]


def transcript_texts(store: ProvStore) -> tuple[str, str | None]:
    """The recording's lexical residue, and the text an applied redaction made of it.

    One renderer serves both, and the words are the ones SPEECH's scan read, so what a reviewer
    reads, what the detectors read and what the re-scan verifies cannot drift. See
    ``specs/20260925-lexical-only-pii-pathway/design.md``.

    Args:
        store: The provenance store.

    Returns:
        ``(original, redacted)``. ``original`` is empty where the recording has no residue.
        ``redacted`` is None where no redaction was planned.
    """
    words = residue_words(store)
    records, _, _ = _render(words, [])
    original = _verification_text(records)
    planned = planned_extents(store)
    if not planned:
        return original, None
    redacted_records, _, _ = _render(words, planned)
    return original, _verification_text(redacted_records)
