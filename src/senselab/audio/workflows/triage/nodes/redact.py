"""The REDACT node: every PII finding padded, merged, masked with the declared fill, and verified.

The last step of the SPEECH branch, run only where SPEECH found PII. The scan it plans from reads the
recording's lexical residue only -- the words that are neither non-lexical nor what the task asked
for (``residue.py``, and ``nodes/speech.py`` step 7) -- and runs only where that residue is not
empty. A store carrying findings but no scan measurement is refused rather than concluded over.

Every non-invalidated ``pii`` entity is redacted regardless of speaker, except one the declared
stimulus accounts for (:func:`_expected_exemptions`, recorded as an ``exempt``/``expected_speech``
assertion). Extents are padded by ``redaction.padding_ms`` and merged by ``plan_redactions``, then
filled with ``redaction.fill`` at ``redaction.bleep_hz`` when that is a bleep. A word carries its
PII marking through a live ``assertion`` whose ``verb`` is ``"label"`` and ``label`` is ``"pii"``.

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
from senselab.audio.workflows.triage.residue import is_content_word, is_proper_form
from senselab.audio.workflows.triage.stimulus import NearMatch, near_match, split_prompts
from senselab.audio.workflows.triage.task_lexicon import TaskLexicon, declared_names_lexicon, task_lexicon
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
from senselab.text.tasks.pii_detection.redaction_review import safe_harbor_codes
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
    """

    verified: bool
    survived: list[str]
    scan_ran: bool
    failed: list[str]
    missing: list[str]
    cache: dict[str, Any] | None = None


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
    survived = sorted({span.category for span in scan.spans if not _is_mask(str(span.text or ""), masks)})
    return _Verification(verified=not survived, survived=survived, scan_ran=True, failed=[], missing=[], cache=cache)


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


def _matches_surviving(word: Entity, category: str, planned: list[RedactionExtent], marked: Mapping[str, str]) -> bool:
    """Whether this word carries a surviving category that no planned extent already covers.

    Args:
        word: A live consensus ``word`` entity.
        category: A category the verification re-scan still saw.
        planned: The padded, merged extents the failing pass produced.
        marked: The PII categories the store's label assertions place on this word, each mapped to
            the assertion that placed it.

    Returns:
        True when the re-plan should widen to this word; a word a planned extent already covers is
        excluded.
    """
    if category not in marked or word.extent is None:
        return False
    hull = word_hull(word)
    return not any(_overlaps(hull, (extent.start, extent.end)) for extent in planned)


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


def _expected_survivors(
    survived: Sequence[str],
    words: Sequence[Entity],
    marked: Mapping[str, Mapping[str, str]],
    planned: Sequence[RedactionExtent],
    exempt_word_ids: frozenset[str],
) -> list[str]:
    """The surviving categories that nothing but an exempt word can account for.

    A category is attributed to the exemption only when at least one exempt word carries it that no
    planned extent covers, and no non-exempt word carries it in the same position. With no
    exemptions this returns nothing.

    Args:
        survived: What the re-scan still saw.
        words: PREPROCESS's consensus words.
        marked: Which PII categories the store's label assertions place on each word.
        planned: The padded, merged extents the pass produced.
        exempt_word_ids: The words covered by an exemption.

    Returns:
        The attributable categories, sorted.
    """
    attributable: list[str] = []
    for category in sorted(set(survived)):
        uncovered = [
            word
            for word in words
            if category in marked.get(word.id, {})
            and word.extent is not None
            and not any(_overlaps(word_hull(word), (extent.start, extent.end)) for extent in planned)
        ]
        if uncovered and all(word.id in exempt_word_ids for word in uncovered):
            attributable.append(category)
    return attributable


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
    units = _expected_units(
        hint,
        task_family,
        terminators=str(config.require(_TERMINATORS_KEY)),
        lexicon=task_lexicon(config, task_family, hint),
    )
    exemptions = _expected_exemptions(findings, words, units, branch_params(config).p_normalise, near_match(config))
    exempt_findings = {exemption.finding_id for exemption in exemptions}
    exempt_word_ids = frozenset(word_id for exemption in exemptions for word_id in exemption.word_ids)
    extents = _extents_from_findings([finding for finding in findings if finding.id not in exempt_findings])
    consensus = find_measurement(store, "consensus_transcript")
    consulted = _pii_marking_assertions(store)
    marked = _pii_marked_words(store)

    planned = plan_redactions(extents, padding_ms=margin_ms)
    records, transcript_text, unplaced_n = _render(words, planned)
    checked = (
        _verify(_render(residue, planned)[0], required_detectors)
        if not scan_incomplete
        else _Verification(verified=False, survived=[], scan_ran=False, failed=[], missing=[])
    )
    attributed = _expected_survivors(checked.survived, residue, marked, planned, exempt_word_ids)
    outstanding = [category for category in checked.survived if category not in attributed]
    replanned_n = 0
    unremediable: list[str] = []
    widened: list[tuple[tuple[float, float], str]] = []
    if checked.scan_ran and outstanding:
        replanned_n = 1
        for category in outstanding:
            for word in residue:
                marks = marked.get(word.id, {})
                if word.id in exempt_word_ids or word.extent is None:
                    continue
                if not _matches_surviving(word, category, planned, marks):
                    continue
                hull = word_hull(word)
                extents.append(RedactionExtent(start=hull[0], end=hull[1], category=category))
                widened.append((hull, marks[category]))
        planned = plan_redactions(extents, padding_ms=margin_ms)
        records, transcript_text, unplaced_n = _render(words, planned)
        checked = _verify(_render(residue, planned)[0], required_detectors)
        attributed = _expected_survivors(checked.survived, residue, marked, planned, exempt_word_ids)
        outstanding = [category for category in checked.survived if category not in attributed]
        unremediable = list(outstanding)

    stream_id, recording = resolve_stream(store, run_dir, source)
    redacted = apply_redactions(recording, planned, fill=fill, bleep_hz=bleep_hz)

    software = software_agent(store)
    view: list[str] = []

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
        for bounds, assertion_id in widened:
            if _overlaps(bounds, (extent.start, extent.end)):
                store.was_derived_from(span_id, assertion_id)
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
    exemptions_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": _EXEMPTION_MEASUREMENT,
            "signal": "consensus_transcript",
            "n": len(exemptions),
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
        if exemptions:
            why = (
                f"every finding redacted except {len(exemptions)} the declared stimulus accounts for; "
                "the redacted transcript carries nothing else"
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
            "expected_exempt_by_category": dict(Counter(exemption.category for exemption in exemptions)),
            "expected_survivors": attributed,
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

PROPOSED_BY_REVIEWER = "proposed_by_reviewer"
"""A word a reviewer ``redact`` entry named that no mask hides."""

WORD_STATES = (MASKED, UNMASKED_BY_REVIEWER, UNMASKED_BY_TRIM, PROPOSED_BY_REVIEWER)

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
not place itself."""

AGREED_MASKED = "masked"
"""A reviewer ``redact`` entry every content word of which a detector mask already hides: agreement."""

AGREED_PLACED = "placed_unplaced"
"""A reviewer ``redact`` entry in the family of a detector finding the fold could not place on words: the
reviewer's quote places it, and its words are masked."""

NEW = "new"
"""A reviewer ``redact`` entry naming a content word no mask hides, or one its quote cannot be placed on:
the reviewer proposing to hide more."""

TASK_CONTENT = "task_content"
"""A reviewer ``redact`` entry in a human-review category (a health condition) every word of which is the
task's own content -- its stimulus, or its task lexicon: the task said it, so it identifies nobody and is
held for no one."""

AGREEMENTS = (AGREED_MASKED, AGREED_PLACED, NEW, TASK_CONTENT)
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


PLACED_WORDS = "words"
PLACED_SUBSTRING = "substring"


@dataclass(frozen=True)
class MaskWord:
    """One word a planned mask covers, and what became of it.

    Attributes:
        word_id: The consensus word's entity id.
        text: Its surface.
        state: :data:`MASKED`, :data:`UNMASKED_BY_REVIEWER` or :data:`UNMASKED_BY_TRIM`.
        named: Whether a reviewer ``release`` entry named it, whether or not the fold applied it.
        content: Whether it counts as content for the trim: a residue content word, or one a
            detector marked in a protected category.
        finding: Whether a detector marked it, rather than the padding reaching it.
        categories: The categories the detectors marked on it, sorted.
        proper: Whether it is written as a proper noun
            (:func:`~senselab.audio.workflows.triage.residue.is_proper_form`).
        propagated: Whether a reviewer ``release`` entry named the same term elsewhere in the recording
            -- the same surface under the same family -- and unmasked this occurrence with it.
        kind_cut: Whether the name kind cut released it: at an edge of a mask whose findings are all
            of :func:`name_families` and which holds a proper noun, a word written all lower-case that
            the transcript also uses outside every finding.
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
    kind_cut: bool = False


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
            reviewer's category for a :data:`REVIEWER` mask.
        source: :data:`DETECTOR` or :data:`REVIEWER`.
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
        human_review: Whether its category is one the fold routes to human review.
        condition_kind: For a human-review entry, :data:`~senselab.audio.workflows.triage.cohort.COHORT`
            or :data:`~senselab.audio.workflows.triage.cohort.OTHER`; empty otherwise.
        cohort_diagnosis: The cohort diagnosis its text matches; empty where none does.
        agreement: One of :data:`AGREEMENTS`.
        entry_index: Its position in the reading's ``proposal``.
    """

    category: str
    text: str
    placed: str
    word_ids: tuple[str, ...]
    texts: tuple[str, ...]
    masked_ids: tuple[str, ...]
    human_review: bool
    condition_kind: str = ""
    cohort_diagnosis: str = ""
    agreement: str = NEW
    entry_index: int = -1


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
    """Which words stay masked once the reviewer's unmasks and the content-word trim are applied.

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
        """The ``proposal`` positions of the reviewer's ``redact`` entries that agree with the masks."""
        return frozenset(span.entry_index for span in self.proposals if span.agreement != NEW)

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
                    for i in span.word_ids
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
                    if span.agreement == NEW and set(span.word_ids) - set(span.masked_ids)
                }
            )
        return sorted({mask.planned.category for mask in self.masks if any(word.state == state for word in mask.words)})

    @property
    def human_review_kind(self) -> str | None:
        """Which kind of condition the human-review proposals name: ``other`` if any does, else ``cohort``."""
        kinds = {span.condition_kind for span in self.proposals if span.human_review and span.agreement == NEW}
        if not kinds:
            return None
        return OTHER if OTHER in kinds else COHORT

    def record(self, *, release: str, release_ground: str | None) -> dict[str, Any]:
        """The ledger, as the measurement VERDICT writes carries it.

        Args:
            release: The fold's release axis value.
            release_ground: The fold's release ground.

        Returns:
            JSON-ready attributes. Word surfaces are the store's own transcript words; the released
            artifacts carry none of the masked ones.
        """
        owners = self.owners()
        return {
            "name": PII_LEDGER,
            "release": release,
            "release_ground": release_ground,
            "reviewer_applied": self.reviewer_applied,
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
                            "kind_cut": word.kind_cut,
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
                    "human_review": span.human_review,
                    "condition_kind": span.condition_kind,
                    "cohort_diagnosis": span.cohort_diagnosis,
                    "agreement": span.agreement,
                    "entry_index": span.entry_index,
                    "safe_harbor": list(safe_harbor_codes(span.category)),
                }
                for span in self.proposals
            ],
            "releases": [dict(entry) for entry in self.releases],
            "unplaced_findings": [
                {"category": finding.category, "family": finding.family, "state": finding.state, "text": finding.text}
                for finding in self.unplaced
            ],
            "release_unplaced": list(self.release_unplaced),
            "release_off_mask": list(self.release_off_mask),
            "task_lexicon_ids": list(self.task_lexicon_ids),
            "reviewer_named_no_words": self.named_no_words,
            "counts": {
                "masks_n": len(self.masks),
                "final_masks_n": len(self.final),
                "task_words_n": sum(mask.task_words_n for mask in self.masks),
                "task_lexicon_words_n": len(self.task_lexicon_ids),
                **{
                    f"{outcome}_n": sum(1 for mask in self.masks if mask.outcome == outcome)
                    for outcome in MASK_OUTCOMES
                },
                **{f"{state}_n": self.count(state) for state in WORD_STATES},
                "propagated_n": len({w.word_id for mask in self.masks for w in mask.words if w.propagated}),
                "kind_cut_n": len({w.word_id for mask in self.masks for w in mask.words if w.kind_cut}),
                "unplaced_n": len(self.unplaced),
                **{
                    f"proposals_{agreement}_n": sum(1 for span in self.proposals if span.agreement == agreement)
                    for agreement in AGREEMENTS
                },
                **{
                    f"{kind}_condition_n": sum(
                        1
                        for span in self.proposals
                        if span.human_review and span.agreement == NEW and span.condition_kind == kind
                    )
                    for kind in CONDITION_KINDS
                },
            },
            "categories": {state: self.categories(state) for state in WORD_STATES},
            "human_review": any(span.human_review and span.agreement == NEW for span in self.proposals),
            "human_review_kind": self.human_review_kind,
            "cohort_diagnoses": sorted(
                {
                    span.cohort_diagnosis
                    for span in self.proposals
                    if span.human_review and span.agreement == NEW and span.cohort_diagnosis
                }
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


def mask_plan(
    store: ProvStore,
    *,
    reviewer_applies: bool,
    padding_ms: int,
    human_review_categories: Sequence[str] = (),
    protected_categories: Sequence[str] = (),
    cohort_conditions: str | None = None,
    lexicon: TaskLexicon | None = None,
) -> MaskPlan:
    """Which words stay masked: one mask per detector finding, the reviewer's unmasks, and the trim.

    A mask stands for one finding, or several covering exactly the same words, and covers the residue
    words that finding was placed on -- never a neighbour its padding or timing reaches, never a task
    word, never the whole transcript. A finding SPEECH could not place masks nothing; it is recorded as
    unplaced, and a reviewer ``redact`` entry of its family places it on the words the reviewer
    quotes. A word leaves its mask when a reviewer ``release`` entry names it -- whole-token runs, at
    every place the quote occurs -- or names the same term (surface and family) elsewhere, unless a
    ``redact`` entry places on that term; or when it is not a residue content word
    (:func:`~senselab.audio.workflows.triage.residue.is_content_word`). A ``redact`` entry whose content
    words a mask already hides agrees with the masks. The kept words are cut into one extent per
    adjacent run, padded up to the nearest unmasked word. The rule and its derivation are in
    ``specs/20260927-mask-placement-and-second-speaker/design.md``.

    Args:
        store: The provenance store, carrying SPEECH's findings and residue, REDACT's plan and
            REVIEW's annotation.
        reviewer_applies: Whether the fold lets the reviewer's ``release`` entries unmask words.
        padding_ms: ``redaction.padding_ms``, the margin kept around each run of masked words.
        human_review_categories: Upper-cased categories whose ``redact`` entries the fold routes to
            human review; marked on the ledger's proposals.
        protected_categories: Upper-cased detector categories under which a marked word written as a
            proper noun counts as content: the trim never releases it, and only a reviewer
            ``release`` entry naming it does.
        cohort_conditions: The cohort profile a human-review proposal's text is read against, as
            ``verdict.cohort_conditions``; every such proposal is ``other`` where None.
        lexicon: The declared family's task lexicon; a word inside one of its phrases is task content
            no mask covers. Where None, the family's declared names, compared exactly.

    Returns:
        The plan.
    """
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    reading = dict(annotation.attributes) if annotation is not None else {}
    entries = list(reading.get("proposal") or ())
    read = reading.get("status") in ("clean", "flagged")
    try:
        residue = residue_words(store)
    except ValueError:
        residue = []
    residue_ids = {word.id for word in residue}
    tokens = _tokens(residue)
    words = [word for word in consensus_words(store) if word.extent is not None]
    by_id = {word.id: word for word in words}
    vocabulary = lexicon if lexicon is not None else declared_names_lexicon(declared_task_family(store))
    declared_ids = {words[i].id for i in vocabulary.positions([str(w.attributes.get("text") or "") for w in words])}
    located, unplaced_records = _located_findings(store, residue_ids, declared_ids)
    finding_word_ids = {
        str(i) for finding in live_entities(store, "pii") for i in (finding.attributes.get("word_ids") or ())
    }
    unplaced_categories = [str(record.get("category") or "") for record in unplaced_records]
    redact_ran = find_verdict(store, NODE) is not None

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

    def content(word: Entity) -> bool:
        if residue_ids and word.id not in residue_ids:
            return False
        if protected & {category.upper() for category in found.get(word.id, set())} and proper(word):
            return True
        return is_content_word(str(word.attributes.get("text") or ""))

    family_of_word: dict[str, str] = {}
    for member_ids, members in groups.items():
        for word_id in member_ids:
            family_of_word.setdefault(word_id, category_family(members[0].category))

    def term(word_id: str) -> tuple[str, str]:
        return _match_token(str(by_id[word_id].attributes.get("text") or "")), family_of_word.get(word_id, "")

    def placed_words(quote: str) -> tuple[list[Entity], str]:
        hits = _place(quote, tokens)
        if hits:
            return hits, PLACED_WORDS
        hits = _place_substring(quote, residue)
        return hits, (PLACED_SUBSTRING if hits else "")

    named: set[str] = set()
    unplaced_quotes: list[str] = []
    releases: list[dict[str, Any]] = []
    for entry in entries:
        if str(entry.get("action")) != "release":
            continue
        hits = _place(str(entry.get("text") or ""), tokens)
        category = str(entry.get("category") or "OTHER").upper()
        releases.append(
            {
                "text": str(entry.get("text") or ""),
                "category": category,
                "safe_harbor": str(entry.get("safe_harbor") or "") or "".join(safe_harbor_codes(category)[:1]),
                "why": str(entry.get("why") or ""),
                "placed": bool(hits),
            }
        )
        if not hits:
            unplaced_quotes.append(str(entry.get("text") or ""))
            continue
        named.update(word.id for word in hits)
    redact_placed = {
        index: placed_words(str(entry.get("text") or ""))
        for index, entry in enumerate(entries)
        if str(entry.get("action")) != "release"
    }
    blocked_tokens = {
        _match_token(str(word.attributes.get("text") or "")) for hits, _ in redact_placed.values() for word in hits
    }
    named_terms = {term(word_id) for word_id in named if word_id in family_of_word}
    propagated = {
        word_id
        for word_id in family_of_word
        if word_id not in named
        and content(by_id[word_id])
        and term(word_id) in named_terms
        and term(word_id)[0] not in blocked_tokens
    }
    applied = reviewer_applies and bool(named)
    released = (named | propagated) if applied else set()
    name_kinds = name_families()
    covered_ids = {word_id for member_ids in groups for word_id in member_ids}
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
        if all(category_family(finding.category) in name_kinds for finding in members):
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
    kind_cut = name_only - elsewhere
    keepable = {
        word_id
        for member_ids in groups
        for word_id in member_ids
        if content(by_id[word_id]) and word_id not in kind_cut
    }
    kept_ids = keepable - released

    proposals: list[ReviewerSpan] = []
    reviewer_masks: list[tuple[str, list[Entity]]] = []
    unplaced_families = {category_family(category) for category in unplaced_categories}
    placed_families: set[str] = set()
    review_set = {category.upper() for category in human_review_categories}
    profile = load_cohort_profile(cohort_conditions) if cohort_conditions else None
    every_token = _tokens(words)

    def task_content(quote: str, hits: Sequence[Entity]) -> bool:
        if hits:
            return all(word.id in declared_ids for word in hits)
        whole = _place(quote, every_token)
        return bool(whole) and all(
            (bool(residue_ids) and word.id not in residue_ids) or word.id in declared_ids for word in whole
        )

    for index, (hits, placed) in redact_placed.items():
        entry = entries[index]
        quote = str(entry.get("text") or "")
        category = str(entry.get("category") or "OTHER").upper()
        family = category_family(category)
        uncovered = [word for word in hits if word.id not in kept_ids and content(word)]
        if category in review_set and task_content(quote, hits):
            agreement = TASK_CONTENT
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
        held = category in review_set and agreement != TASK_CONTENT
        diagnosis = (profile.diagnosis(quote) if profile is not None else None) if held else None
        proposals.append(
            ReviewerSpan(
                category=category,
                text=quote,
                placed=placed,
                word_ids=tuple(word.id for word in hits),
                texts=tuple(str(word.attributes.get("text") or "") for word in hits),
                masked_ids=tuple(word.id for word in hits if word.id in kept_ids),
                human_review=held,
                condition_kind=(COHORT if diagnosis else OTHER) if held else "",
                cohort_diagnosis=diagnosis or "",
                agreement=agreement,
                entry_index=index,
            )
        )

    def state_of(word: Entity) -> str:
        if word.id in kept_ids:
            return MASKED
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
    for family, group in reviewer_masks:
        if not group:
            continue
        hull = (min(word_hull(word)[0] for word in group), max(word_hull(word)[1] for word in group))
        pending.append((RedactionExtent(start=hull[0], end=hull[1], category=family), group, (family,), REVIEWER, 0))
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
                text=str(word.attributes.get("text") or ""),
                state=state_of(word),
                named=word.id in named,
                content=content(word),
                finding=word.id in found,
                categories=tuple(sorted(found.get(word.id, set()))),
                proper=proper(word),
                propagated=word.id in propagated,
                kind_cut=word.id in kind_cut,
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
        order = [word.id for word in reach]
        labelled = replace(extent, category=outcomes[inside[0]].planned.category)
        for rank, position in enumerate(inside):
            outcomes[position] = replace(
                outcomes[position],
                final=(labelled,) if rank == 0 else (),
                final_words=(tuple(order),) if rank == 0 else (),
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
        for entry in entries
        if str(entry.get("action")) == "release"
        and (hits := _place(str(entry.get("text") or ""), tokens))
        and not any(word.id in family_of_word for word in hits)
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
        named_no_words=reviewer_named_no_words(reading),
        releases=tuple(releases),
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


def _masked_source(store: ProvStore, run_dir: Path) -> Audio:
    """The stream REDACT's ``redacted`` copy was masked from, loaded from its sidecar.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The source audio.

    Raises:
        LookupError: If the store holds no ``redacted`` stream or does not say what it came from.
    """
    redacted_id, _ = resolve_stream(store, run_dir, STREAM_NAME)
    for source_id in store.derived_from(redacted_id):
        source = store.get_entity(source_id)
        if source.prov_type == "stream":
            path = Path(source.attributes["path"])
            return Audio(filepath=str(path if path.is_absolute() else run_dir / path))
    raise LookupError("the redacted stream records no source stream")


def settle_release(
    store: ProvStore,
    release: str,
    release_ground: str | None,
    *,
    run_dir: Path,
    artifacts_dir: Path,
    bleep_hz: float | None,
) -> dict[str, Path]:
    """Make the release directory hold exactly the copy the fold released.

    The directory holds a redacted copy only under ``release_with_redaction``; every other release
    empties it, including of a copy REDACT itself wrote on a pass. The copy masks the fold's final
    masks, as its ledger records them (:func:`released_masks`), and its text hides exactly the words
    the ledger keeps masked. Where the final extents are REDACT's own, the audio is REDACT's
    ``redacted`` stream: the copy REDACT wrote stands where it passed and already carries this
    text, and is written again otherwise. Elsewhere the source stream is re-masked with the final
    masks alone.

    Args:
        store: The provenance store, after VERDICT.
        release: The fold's release axis value.
        release_ground: The fold's release ground. Recorded on the ledger; not read here.
        run_dir: The run directory sidecar paths are relative to.
        artifacts_dir: The release directory.
        bleep_hz: ``redaction.bleep_hz``, for a re-masked copy under a bleep fill.

    Returns:
        The written paths, keyed as :func:`_write_artifacts` keys them; empty where nothing was
        written.
    """
    verdict = find_verdict(store, NODE)
    if verdict is None:
        return {}
    if release != Release.WITH_REDACTION.value:
        for name in RELEASED_FILES:
            (artifacts_dir / name).unlink(missing_ok=True)
        return {}
    planned = planned_extents(store)
    final, owners = released_masks(store)
    words = consensus_words(store)
    records, text, _ = _render(words, final, owners)
    if same_extents(final, planned):
        if verdict.attributes.get("outcome") == Outcome.PASS.value and (
            _holds_copy(artifacts_dir, records) or not _has_stream(store, STREAM_NAME)
        ):
            return {}
        _, redacted = resolve_stream(store, run_dir, STREAM_NAME)
        return _write_artifacts(redacted, text, records, artifacts_dir)
    fill = str(verdict.attributes.get("fill") or "")
    masked = apply_redactions(_masked_source(store, run_dir), final, fill=fill, bleep_hz=bleep_hz)
    return _write_artifacts(masked, text, records, artifacts_dir)


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
