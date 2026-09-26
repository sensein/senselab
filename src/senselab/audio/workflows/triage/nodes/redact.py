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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.redaction.api import RedactionExtent, apply_redactions, plan_redactions
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import branch_params, declared_carrier, expected_names
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
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
from senselab.audio.workflows.triage.stimulus import NearMatch, near_match, split_prompts
from senselab.audio.workflows.triage.vocabulary import (
    REDACTION_LLM_ANNOTATION,
    REVIEWER_RESET_SOME_MASKS,
    Outcome,
    Release,
)
from senselab.text.tasks.pii_detection.api import scan_for_pii
from senselab.utils.prov_store import Entity, ProvStore

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
_COVERAGE_TOLERANCE_S = 1e-9  # float slack on an equality


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
    """

    verified: bool
    survived: list[str]
    scan_ran: bool
    failed: list[str]
    missing: list[str]


def _padding_ms(config: TriageConfig) -> int:
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
    missing = sorted(set(required) - set(scan.detectors_used) - set(scan.failures))
    failed = sorted(scan.failures)
    if failed or not scan.detectors_used or missing:
        return _Verification(verified=False, survived=[], scan_ran=False, failed=failed, missing=missing)
    masks = _mask_tokens(records)
    survived = sorted({span.category for span in scan.spans if not _is_mask(str(span.text or ""), masks)})
    return _Verification(verified=not survived, survived=survived, scan_ran=True, failed=[], missing=[])


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


def _render(words: list[Entity], planned: list[RedactionExtent]) -> tuple[list[dict[str, Any]], str, int]:
    """The redacted consensus stream, as records and as the flat text derived from them.

    Words are released in stream order. Every word a planned extent overlaps folds into one
    ``redaction`` record emitted at the first such position; a word the store places nowhere becomes
    an ``unplaced`` record. Neither record carries ``text``, ``readings`` or ``variants``.

    Args:
        words: PREPROCESS's consensus words, in stream order.
        planned: The padded, merged extents.

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
        hull = word_hull(word)
        index = next((i for i, p in enumerate(planned) if _overlaps(hull, (p.start, p.end))), None)
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
    hint: AudioHints | None, task_family: str | None, *, terminators: str
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
    return units


def _covered_words(finding: Entity, words: Sequence[Entity]) -> list[Entity]:
    """The consensus words a finding's own extent reaches, by the hull of their source timings.

    Args:
        finding: A live ``pii`` entity.
        words: PREPROCESS's consensus words, in stream order.

    Returns:
        The covered words, in stream order. Empty when the finding carries no extent.
    """
    if finding.extent is None:
        return []
    bounds = (float(finding.extent[0]), float(finding.extent[1]))
    return [word for word in words if word.extent is not None and _overlaps(word_hull(word), bounds)]


def _expected_exemptions(
    findings: Sequence[Entity],
    words: Sequence[Entity],
    units: Sequence[tuple[int, str, list[str]]],
    normalise: Callable[[str], str],
    near: NearMatch,
) -> list[_Exemption]:
    """Which findings the declared stimulus accounts for.

    A candidate is exempt only when every word its extent reaches carries a non-empty normalised
    key, those words' own hulls span the whole of the extent, and their keys occur in order and
    contiguously inside one declared unit, each within ``near`` of the unit's own token. A finding
    that reaches every consensus word is never exempt.

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
    for finding in findings:
        covered = _covered_words(finding, words)
        if not covered or len(covered) == len(words):
            continue
        keys = [normalise(str(word.attributes.get("text") or "")) for word in covered]
        if not all(keys):
            continue
        hulls = [word_hull(word) for word in covered]
        assert finding.extent is not None  # _covered_words returns nothing without one
        if (
            min(start for start, _ in hulls) > float(finding.extent[0]) + _COVERAGE_TOLERANCE_S
            or max(end for _, end in hulls) < float(finding.extent[1]) - _COVERAGE_TOLERANCE_S
        ):
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
            value (see :func:`_padding_ms`), if ``artifacts_dir`` and ``run_dir`` contain one
            another, if any ``redaction.llm_check`` key is unmeasured, if the store carries no PII
            scan measurement (N15), or if a finding's category is unusable (see
            :func:`_extents_from_findings`).
        LookupError: If no live stream carries ``source``.
    """
    fill = str(config.require(_FILL_KEY))
    bleep_hz = config.get(_BLEEP_HZ_KEY)
    padding_ms = _padding_ms(config)
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
    units = _expected_units(hint, task_family, terminators=str(config.require(_TERMINATORS_KEY)))
    exemptions = _expected_exemptions(findings, words, units, branch_params(config).p_normalise, near_match(config))
    exempt_findings = {exemption.finding_id for exemption in exemptions}
    exempt_word_ids = frozenset(word_id for exemption in exemptions for word_id in exemption.word_ids)
    extents = _extents_from_findings([finding for finding in findings if finding.id not in exempt_findings])
    consensus = find_measurement(store, "consensus_transcript")
    consulted = _pii_marking_assertions(store)
    marked = _pii_marked_words(store)

    planned = plan_redactions(extents, padding_ms=padding_ms)
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
        planned = plan_redactions(extents, padding_ms=padding_ms)
        records, transcript_text, unplaced_n = _render(words, planned)
        checked = _verify(_render(residue, planned)[0], required_detectors)
        attributed = _expected_survivors(checked.survived, residue, marked, planned, exempt_word_ids)
        outstanding = [category for category in checked.survived if category not in attributed]
        unremediable = list(outstanding)

    stream_id, recording = resolve_stream(store, run_dir, source)
    redacted = apply_redactions(recording, planned, fill=fill, bleep_hz=bleep_hz)

    software = software_agent(store)
    view: list[str] = []

    plan_act = store.activity(node=NODE, step="plan", parameters={"padding_ms": padding_ms, "replanned_n": replanned_n})
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
            "padding_ms": padding_ms,
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


@dataclass(frozen=True)
class MaskReset:
    """Which of REDACT's masks the reviewer's ``release`` entries reset to the original words.

    Attributes:
        planned: REDACT's masks, in stream order.
        kept: The masks no ``release`` entry reset, in stream order.
        reset_n: How many masks were reset.
        unplaced: The ``release`` quotes that match no run of the recording's residue words.
        straddling_n: Masks some but not all of whose residue words a quote names, over an
            original not read as clean; kept.
    """

    planned: list[RedactionExtent]
    kept: list[RedactionExtent]
    reset_n: int
    unplaced: tuple[str, ...]
    straddling_n: int


def reviewer_reset(store: ProvStore) -> MaskReset:
    """Which masks the reviewer's ``release`` entries reset, decided the same way on every read.

    A quote is matched as a run of whole tokens against the residue words the reviewer read, at
    every place it occurs. Where the reviewer read the original as clean, a mask is reset once a
    matched run names any residue word it hides: the other words a padded mask folds in are part of
    an original the reviewer already judged. Otherwise a mask is reset only where every residue word
    it hides lies inside a matched run. A mask hiding no residue word is kept, and a quote matching
    nothing resets nothing.

    Args:
        store: The provenance store, carrying REDACT's planned spans, SPEECH's residue and REVIEW's
            annotation.

    Returns:
        The reset. Nothing is reset where REDACT planned no mask, REVIEW wrote no annotation, or
        the store predates the lexical residue.
    """
    planned = planned_extents(store)
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    quotes = [
        str(entry.get("text") or "")
        for entry in ((annotation.attributes.get("proposal") or ()) if annotation is not None else ())
        if str(entry.get("action")) == "release"
    ]
    unchanged = MaskReset(planned=planned, kept=list(planned), reset_n=0, unplaced=(), straddling_n=0)
    if not planned or not quotes:
        return unchanged
    try:
        residue = residue_words(store)
    except ValueError:
        return unchanged
    tokens = [
        (_match_token(str(word.attributes.get("text") or "")), word)
        for word in residue
        if not word.attributes.get("bracketed") and word.extent is not None
    ]
    tokens = [(token, word) for token, word in tokens if token]
    surfaces = [token for token, _ in tokens]
    released: set[str] = set()
    unplaced: list[str] = []
    for quote in quotes:
        wanted = [token for token in (_match_token(piece) for piece in quote.split()) if token]
        width = len(wanted)
        starts = [i for i in range(len(surfaces) - width + 1) if width and surfaces[i : i + width] == wanted]
        if not starts:
            unplaced.append(quote)
            continue
        for start in starts:
            released.update(word.id for _, word in tokens[start : start + width])
    judged_clean = annotation is not None and annotation.attributes.get("original") == "clean"
    kept: list[RedactionExtent] = []
    straddling = 0
    for extent in planned:
        hidden = [word.id for _, word in tokens if _overlaps(word_hull(word), (extent.start, extent.end))]
        named = [word_id for word_id in hidden if word_id in released]
        if hidden and (len(named) == len(hidden) or (judged_clean and named)):
            continue
        if named:
            straddling += 1
        kept.append(extent)
    return MaskReset(
        planned=planned,
        kept=kept,
        reset_n=len(planned) - len(kept),
        unplaced=tuple(unplaced),
        straddling_n=straddling,
    )


def _holds_full_copy(artifacts_dir: Path, planned_n: int) -> bool:
    """Whether the release directory holds a redacted copy carrying every planned mask.

    Args:
        artifacts_dir: The release directory.
        planned_n: How many masks REDACT planned.

    Returns:
        True where all of :data:`RELEASED_FILES` exist and the consensus artifact records
        ``planned_n`` redactions, or is not one this module wrote.
    """
    if not all((artifacts_dir / name).exists() for name in RELEASED_FILES):
        return False
    try:
        written = json.loads((artifacts_dir / "consensus.json").read_text())
    except (OSError, ValueError):
        return True
    if not isinstance(written, dict) or written.get("schema") != CONSENSUS_ARTIFACT_SCHEMA:
        return True
    return int(written.get("n_redactions") or 0) == planned_n


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

    The directory holds the redacted copy only under ``release_with_redaction``; every other release
    empties it, including of a copy REDACT itself wrote on a pass. A pass released as planned keeps
    the copy REDACT wrote, unless an earlier fold emptied the directory or thinned the copy, in which
    case it is written again. Otherwise the copy is written from the store: REDACT's masked
    ``redacted`` stream where every planned mask is kept, and the source stream re-masked with the
    kept masks alone where the reviewer reset some (:data:`REVIEWER_RESET_SOME_MASKS`).

    Args:
        store: The provenance store, after VERDICT.
        release: The fold's release axis value.
        release_ground: The fold's release ground.
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
    if release_ground != REVIEWER_RESET_SOME_MASKS:
        planned = planned_extents(store)
        if verdict.attributes.get("outcome") == Outcome.PASS.value and (
            _holds_full_copy(artifacts_dir, len(planned)) or not _has_stream(store, STREAM_NAME)
        ):
            return {}
        records, text, _ = _render(consensus_words(store), planned)
        _, redacted = resolve_stream(store, run_dir, STREAM_NAME)
        return _write_artifacts(redacted, text, records, artifacts_dir)
    kept = reviewer_reset(store).kept
    records, text, _ = _render(consensus_words(store), kept)
    fill = str(verdict.attributes.get("fill") or "")
    masked = apply_redactions(_masked_source(store, run_dir), kept, fill=fill, bleep_hz=bleep_hz)
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
