"""The REDACT node: every PII finding padded, merged, masked with the declared fill, and verified.

REDACT is the last step of the SPEECH branch and runs only when SPEECH's PII scan found something;
the runner gates on that, and this node refuses an incoherent store — findings with no scan
measurement — rather than concluding over one. Every non-invalidated ``pii`` entity is redacted
regardless of speaker. Extents are padded and merged by ``plan_redactions``; the margin is the
``redaction.padding_ms`` config key and must be a non-negative whole number of milliseconds. What is
written into an extent is ``redaction.fill``, which ships with no default, at ``redaction.bleep_hz``
when it is a bleep.

**No recognizer runs here.** Verification is a re-scan of the redacted consensus text with the same
detectors, judged complete by ``pii.required_detectors``: a surviving finding is a fail, an
incomplete re-scan is a flag. A finding the planner placed and the verifier still sees is remediable
exactly once — the verifier's words are fed back for a single re-planning pass, and what survives
that is ``unremediable``. The verdict's ``audio_check`` is the constant ``"bounded"`` on every path:
the re-scan establishes that the redacted text no longer carries the finding and nothing about the
audio. See ``specs/20260817-triage-workflow-dag/redact.md``.

An **optional LLM check** re-reads the redacted transcript, off unless ``redaction.llm_check.enabled``
says otherwise. It iterates — a round that flags something masks its concerns and reviews again, to
``redaction.llm_check.max_iterations`` — and every round's chain of thought is stored as its own
``redaction_llm_review`` measurement, which is the point of the step rather than a by-product. It
**annotates and never decides**: its summary is a ``redaction_llm_annotation`` measurement carrying
the status, the iteration count, the flagged categories, the model and the commit it loaded, written
on every path including the ones where it did not run, and this node's outcome is its detector path's
alone. What an annotation of ``flagged`` or ``absent`` means is VERDICT's, under
``verdict.llm_redaction_flags``. It is asked only where a release was in prospect — never over an
outcome the detector path already withheld. The step's own cost is in the store: the ``llm_check``
activity carries ``started`` and ``ended`` on every path, and each round's measurement carries
``elapsed_s``, the ``load_s`` of that round, and the tokens generated.

Three artifacts are released, not two: the masked audio, the flat redacted transcript, and the
**redacted consensus stream** as ``consensus.json`` — the consensus structure PREPROCESS built, with
each surviving word's extent, per-source timings, readings, variants and agreement share, and each
planned extent folded into one placeholder record carrying its category, its padded bounds and the
number of words it swallowed but no surface of any kind. One renderer produces both the records and
the flat text, so the two cannot disagree about what was masked.

The masked audio is a **stream in the store**, written under ``run_dir`` and registered as
``redacted``, so a consumer resolves it by name the way it resolves ``plain``, ``enhanced`` and
``residual``. It is registered on every path, pass or not: ``run_dir`` is the store side of the
disjointness check, never the release side. The released file under ``artifacts_dir`` is written
only on a pass, and is additional to the stream rather than a replacement for it.

A word carries a PII marking through a live ``assertion`` entity whose ``verb`` is ``"label"``,
whose ``label`` is ``"pii"``, and which is ``wasDerivedFrom`` the word — the store's shared shape for
a label. Only a pass produces a released pair; a flag withholds exactly like a fail, and the
verdict's ``artifacts_withheld`` records it. Artifacts are written into a directory disjoint from the
run directory, and carry no store element id.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.redaction.api import RedactionExtent, apply_redactions, plan_redactions
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import branch_params
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    consensus_words,
    find_measurement,
    live_entities,
    path_attributes,
    resolve_stream,
    software_agent,
    word_hull,
    write_stream,
    write_verdict,
)
from senselab.audio.workflows.triage.stimulus import split_prompts
from senselab.audio.workflows.triage.vocabulary import REDACTION_LLM_ANNOTATION, Outcome
from senselab.text.tasks.pii_detection.api import scan_for_pii
from senselab.text.tasks.pii_detection.redaction_review import (
    review_payload,
    review_redacted_text,
    shutdown_review_worker,
)
from senselab.utils.prov_store import Entity, ProvStore

NODE = "REDACT"

_FILL_KEY = "redaction.fill"
_BLEEP_HZ_KEY = "redaction.bleep_hz"
_PADDING_KEY = "redaction.padding_ms"
_REQUIRED_DETECTORS_KEY = "pii.required_detectors"
_LABEL_VERB = "label"  # the store's assertion verb, a vocabulary term not a value
_PII_LABEL = "pii"  # the marking SPEECH places on a word carrying a finding
_RESERVED_CATEGORY_CHAR = "+"  # plan_redactions' merge separator; a string, not a threshold
_UNPLACED_PLACEHOLDER = "[UNPLACED]"  # a word the store places nowhere; a category-less placeholder
_AUDIO_CHECK = "bounded"  # what a text re-scan can claim about the audio, on every path
STREAM_NAME = "redacted"  # the store-held name the masked audio resolves under, beside plain/enhanced/residual
CONSENSUS_ARTIFACT_SCHEMA = "senselab.triage.redacted_consensus"  # what the JSON artifact claims to be
CONSENSUS_ARTIFACT_VERSION = 1  # bumped when a record's field set changes
_TERMINATORS_KEY = "stimulus.sentence_terminators"
_EXEMPT_VERB = "exempt"  # the store's assertion verb for a redaction deliberately not made
_EXPECTED_LABEL = "expected_speech"  # what accounted for it: the stimulus the participant was asked to read
_EXEMPTION_MEASUREMENT = "redaction_exemptions"
_COVERAGE_TOLERANCE_S = 1e-9  # float slack on an equality, not a margin
_LLM_SECTION = "redaction.llm_check"
_LLM_REVIEW_MEASUREMENT = "redaction_llm_review"
_LLM_PLACEHOLDER = "[LLM_{category}]"  # what a concern is masked with inside the loop, never in a release


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
        failed: Detectors the **verification** re-scan attempted and that raised. Names only; a
            failure's message may quote the scanned input.
        missing: Detectors ``pii.required_detectors`` names that the verification re-scan never
            attempted. Kept apart from ``failed`` for the reason the planning scan keeps them apart:
            "it broke" and "nobody ran it" are different findings, and the second is the silent one.
            Both are reported in the verdict separately from the planning scan's own ``scan_failed``
            and ``scan_missing``, because a store whose planning scan was complete and whose
            verification was not is a different state from the reverse, and an operator reading one
            pair of keys for both could not tell which half failed.
    """

    verified: bool
    survived: list[str]
    scan_ran: bool
    failed: list[str]
    missing: list[str]


def _padding_ms(config: TriageConfig) -> int:
    """The redaction margin, in whole milliseconds, as a validity check rather than a tunable.

    Args:
        config: The triage configuration.

    Returns:
        The margin.

    Raises:
        ValueError: If ``redaction.padding_ms`` has no value, is not a number, is not finite, is not
            integral, or is negative. A negative or fractional margin is refused rather than coerced.
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
        and failed, and those required but never attempted, each sorted. Only names: a failure's
        message may quote the scanned input.
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
    """Every finding, regardless of speaker; the membership check that secures the error path.

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


def _verify(transcript_text: str, required: list[str]) -> _Verification:
    """Re-scan the redacted consensus text with the same detectors.

    No recognizer runs. Re-transcribing would draw a second sample from the recognizers, which is a
    different measurement of a different signal rather than a check on this one, and the claim about
    the audio is bounded either way.

    Args:
        transcript_text: The redacted consensus transcript.
        required: The detector set ``pii.required_detectors`` names.

    Returns:
        What the re-scan established. A finding that survives fails; a re-scan that skipped a
        required detector did not run, which is not a clean result.
    """
    scan = scan_for_pii(transcript_text)
    scan = scan[0] if isinstance(scan, list) else scan
    missing = sorted(set(required) - set(scan.detectors_used) - set(scan.failures))
    failed = sorted(scan.failures)
    if failed or not scan.detectors_used or missing:
        return _Verification(verified=False, survived=[], scan_ran=False, failed=failed, missing=missing)
    survived = sorted({span.category for span in scan.spans})
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
    """Every live ``label``/``pii`` assertion, which is the whole of what the marking read consults.

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
        A mapping from word entity id to ``{category: assertion id}``. A word nothing marks is
        absent rather than mapped to an empty mapping. The assertion id is kept so a span the
        re-plan widened can be derived from the marking that caused it; where two assertions place
        the same category on one word the earlier is retained, which is the store's write order.
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
        True when the re-plan should widen to this word. A word a planned extent already covers is
        excluded, so the re-plan widens what the first pass missed rather than re-planning it.
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
        The record. The store's own element id is not in it, and neither is the entity's raw
        attribute mapping: only the fields named here are copied, so a field PREPROCESS adds later
        cannot reach a released artifact without this function being edited.
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

    Words are released in stream order. A word overlapping a planned extent is replaced along with
    its padded-in neighbours, matching what the audio lost; one ``redaction`` record is emitted at
    the first overlapping position and later overlapping positions are folded into it. A word the
    store places nowhere overlaps no extent, so it becomes an ``unplaced`` record rather than being
    released verbatim. A record for a redacted or unplaced position carries **no** ``text``, no
    ``readings`` and no ``variants``: those are per-recognizer surfaces of the very token the
    redaction exists to withhold.

    Args:
        words: PREPROCESS's consensus words, in stream order.
        planned: The padded, merged extents.

    Returns:
        ``(records, text, unplaced_n)``. ``text`` is the join of each record's placeholder or
        surface, so the flat artifact and the structured one cannot disagree about what was masked.
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
        expected_keys: The **stimulus's own** normalised tokens the covered run matched, taken from
            the prompt rather than from the transcript, so a fault in the matcher cannot put
            transcript text into the audit record.
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


def _expected_units(hint: AudioHints | None, *, terminators: str) -> list[tuple[int, str, list[str]]]:
    """The declared stimulus, split into structure units and their verbatim tokens.

    Args:
        hint: What the recording was declared to contain, or None.
        terminators: The characters that close a unit inside one prompt.

    Returns:
        ``(prompt index, unit text, tokens)`` per unit. Empty when there is no hint or no
        ``expected_speech``, which is what makes the no-hint path identical to the old behaviour.
    """
    if hint is None or not hint.expected_speech:
        return []
    return split_prompts(list(hint.expected_speech), terminators=terminators)


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


def _contiguous_run(haystack: Sequence[str], needle: Sequence[str]) -> int | None:
    """Where ``needle`` occurs in ``haystack`` as a contiguous run, or None.

    Args:
        haystack: One unit's normalised tokens.
        needle: The covered words' normalised keys.

    Returns:
        The offset of the first occurrence, or None. A run rather than a subsequence: two words the
        prompt happens to contain in different sentences do not account for them said together.
    """
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return start
    return None


def _expected_exemptions(
    findings: Sequence[Entity],
    words: Sequence[Entity],
    units: Sequence[tuple[int, str, list[str]]],
    normalise: Callable[[str], str],
) -> list[_Exemption]:
    """Which findings the declared stimulus accounts for.

    A candidate is exempt only when **every** word its extent reaches carries a non-empty normalised
    key, those words' own hulls span the whole of the extent, and their keys occur in order and
    contiguously inside one declared unit. A finding that reaches every consensus word is never
    exempt: that is SPEECH's signature for a finding its locator could not place, and a
    whole-transcript extent would be accounted for by a whole-passage prompt without anything having
    been matched.

    The span condition is what makes "every word it reaches" trustworthy. SPEECH builds a located
    finding's extent as the hull of the words it covers, so the covered words' hulls reconstruct it
    exactly — unless one was missed, and a missed word is one this function would be accounting for
    without having looked at it. Any extent the identified words do not reach is therefore refused.

    Args:
        findings: The live ``pii`` entities.
        words: PREPROCESS's consensus words, in stream order.
        units: The declared structure from :func:`_expected_units`.
        normalise: The branches' own lexical normaliser, ``BranchParams.p_normalise``.

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
            offset = _contiguous_run(unit_keys, keys)
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

    The verifier re-scans the released text, in which an exempt word stands verbatim, so it sees the
    candidate again. A category is attributed to the exemption only when there is at least one
    exempt word carrying it that no planned extent covers **and** no non-exempt word carrying it in
    the same position. With no exemptions the first condition can never hold, so this returns
    nothing and every survivor is a failure exactly as before.

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


@dataclass(frozen=True)
class _LlmCheck:
    """What the optional reviewer established, as the ``redaction_llm_annotation`` measurement records it.

    Attributes:
        status: ``disabled`` when the config leaves it off, ``not_run`` when the detector path had
            already withheld, ``absent`` when the model could not be reached, ``clean`` when it
            read the transcript and flagged nothing, ``flagged`` when it flagged something.
        iterations: How many reviews ran.
        flagged: The categories the reviewer named, sorted. Categories only; the substrings and the
            reasoning are in the per-iteration measurements beside it, so nothing VERDICT reads to
            decide on carries transcript text.
        model_id: The repo asked, or the empty string when nothing was.
        revision: The commit the reviewer loaded, or None.
        failure: Why it did not run, when it did not.
    """

    status: str
    iterations: int
    flagged: tuple[str, ...]
    model_id: str
    revision: str | None
    failure: str | None

    def as_detail(self) -> dict[str, Any]:
        """The mapping the annotation measurement carries.

        Returns:
            The record, flat, so a report and VERDICT both read it without knowing this class.
        """
        return {
            "status": self.status,
            "iterations": self.iterations,
            "flagged": list(self.flagged),
            "model_id": self.model_id,
            "revision": self.revision,
            "failure": self.failure,
        }


def _stamp() -> str:
    """Now, as the ISO 8601 UTC string an activity's ``started``/``ended`` carries.

    Returns:
        The timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _llm_settings(config: TriageConfig) -> dict[str, Any]:
    """Every ``redaction.llm_check`` key, read in one place.

    Args:
        config: The triage configuration.

    Returns:
        The settings.

    Raises:
        ValueError: If a key is unmeasured. Reading them together means a run that turns the step on
            with half a configuration is refused before any model is contacted.
    """
    names = (
        "enabled",
        "model_id",
        "ref",
        "max_iterations",
        "max_new_tokens",
        "timeout_s",
        "keep_worker_resident",
    )
    return {name: config.require(f"{_LLM_SECTION}.{name}") for name in names}


def _mask_concerns(text: str, findings: Any) -> str:  # noqa: ANN401 — the backend's own finding type
    """Replace each concern's substring with its placeholder, for the next round only.

    Args:
        text: The text the round reviewed.
        findings: That round's findings.

    Returns:
        The text the next round reviews. A substring the reviewer named but the text does not
        contain is left alone: nothing is guessed at on the model's behalf.
    """
    masked = text
    for finding in findings:
        if finding.text and finding.text in masked:
            masked = masked.replace(finding.text, _LLM_PLACEHOLDER.format(category=finding.category))
    return masked


def _llm_check(transcript_text: str, settings: Mapping[str, Any]) -> tuple[_LlmCheck, list[dict[str, Any]]]:
    """Read the redacted transcript back with the reviewer, bounded, capturing every chain of thought.

    The loop is here rather than in the model: one review per call, and a round that flags something
    masks it and reviews again, so the reviewer sees the effect of its own concerns. The masking is
    local to the loop — nothing it produces is released — and whether **any** round flagged is what
    decides the check, not whether the last one did.

    Every round of one check shares one loaded model. Whether the *next recording* does is
    ``redaction.llm_check.keep_worker_resident``: with it unset the weights are released when the
    check ends, so nothing else on the card has to live beside them.

    Args:
        transcript_text: The redacted transcript, as it would be released.
        settings: :func:`_llm_settings`' mapping.

    Returns:
        ``(check, reviews)`` — the summary and one payload per round, in order. A round that could
        not run is recorded as such; the first round failing is ``absent``, a later one leaves the
        flag that already stands and records the failure beside it.
    """
    try:
        return _llm_rounds(transcript_text, settings)
    finally:
        if not settings["keep_worker_resident"]:
            shutdown_review_worker()


def _llm_rounds(transcript_text: str, settings: Mapping[str, Any]) -> tuple[_LlmCheck, list[dict[str, Any]]]:
    """The bounded review / mask / re-review loop itself, without the worker's lifetime.

    Args:
        transcript_text: The redacted transcript, as it would be released.
        settings: :func:`_llm_settings`' mapping.

    Returns:
        :func:`_llm_check`'s pair.
    """
    reviews: list[dict[str, Any]] = []
    current = transcript_text
    flagged: list[str] = []
    model_id = str(settings["model_id"])
    revision: str | None = None
    for iteration in range(1, int(settings["max_iterations"]) + 1):
        result = review_redacted_text(
            current,
            model_id=model_id,
            ref=str(settings["ref"]),
            max_new_tokens=int(settings["max_new_tokens"]),
            timeout_s=int(settings["timeout_s"]),
        )
        reviews.append({**review_payload(result), "iteration": iteration})
        revision = result.revision or revision
        if not result.available:
            if flagged:
                return (
                    _LlmCheck("flagged", iteration, tuple(sorted(set(flagged))), model_id, revision, result.failure),
                    reviews,
                )
            return _LlmCheck("absent", iteration, (), model_id, revision, result.failure), reviews
        if not result.findings:
            status = "flagged" if flagged else "clean"
            return _LlmCheck(status, iteration, tuple(sorted(set(flagged))), model_id, revision, None), reviews
        flagged.extend(finding.category for finding in result.findings)
        current = _mask_concerns(current, result.findings)
    return (
        _LlmCheck("flagged", int(settings["max_iterations"]), tuple(sorted(set(flagged))), model_id, revision, None),
        reviews,
    )


def _write_artifacts(
    redacted: Audio,
    transcript_text: str,
    records: list[dict[str, Any]],
    artifacts_dir: Path,
) -> dict[str, Path]:
    """Write the releasable set. Takes no store and no element id, so it cannot embed one.

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
) -> RedactResult:
    """Redact every PII finding from the recording and verify the redacted text before releasing it.

    Args:
        store: The provenance store, holding SPEECH's ``pii`` entities and scan measurement and
            PREPROCESS's consensus words.
        source: The store-held stream name, ``"recording"`` (N17).
        config: The triage configuration.
        hint: What the recording was declared to contain. When it carries ``expected_speech``, a
            PII candidate the declared stimulus accounts for is exempted from redaction and
            recorded as such. With no hint or no ``expected_speech`` nothing is exempted and
            the node behaves exactly as it did before.
        run_dir: The run directory sidecar paths are relative to.
        artifacts_dir: The release directory; must not contain or be contained by ``run_dir``.

    Returns:
        The verdict, the view over what this node wrote, and the released artifacts — empty unless
        the outcome is a pass.

    Raises:
        ValueError: If ``redaction.fill`` has no value, if ``redaction.padding_ms`` has no usable
            value (see :func:`_padding_ms`), if ``artifacts_dir`` and ``run_dir`` contain one
            another, if any ``redaction.llm_check`` key is unmeasured, if the store carries no PII
            scan measurement (N15) — an incoherent store, as
            distinct from a complete store with nothing to scan, which concludes — or if a finding's
            category is unusable (see :func:`_extents_from_findings`).
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
    units = _expected_units(hint, terminators=str(config.require(_TERMINATORS_KEY)))
    exemptions = _expected_exemptions(findings, words, units, branch_params(config).p_normalise)
    exempt_findings = {exemption.finding_id for exemption in exemptions}
    exempt_word_ids = frozenset(word_id for exemption in exemptions for word_id in exemption.word_ids)
    extents = _extents_from_findings([finding for finding in findings if finding.id not in exempt_findings])
    consensus = find_measurement(store, "consensus_transcript")
    consulted = _pii_marking_assertions(store)
    marked = _pii_marked_words(store)

    planned = plan_redactions(extents, padding_ms=padding_ms)
    records, transcript_text, unplaced_n = _render(words, planned)
    checked = (
        _verify(transcript_text, required_detectors)
        if not scan_incomplete
        else _Verification(verified=False, survived=[], scan_ran=False, failed=[], missing=[])
    )
    attributed = _expected_survivors(checked.survived, words, marked, planned, exempt_word_ids)
    outstanding = [category for category in checked.survived if category not in attributed]
    replanned_n = 0
    unremediable: list[str] = []
    widened: list[tuple[tuple[float, float], str]] = []
    if checked.scan_ran and outstanding:
        replanned_n = 1
        for category in outstanding:
            for word in words:
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
        checked = _verify(transcript_text, required_detectors)
        attributed = _expected_survivors(checked.survived, words, marked, planned, exempt_word_ids)
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

    llm_settings = _llm_settings(config)
    review_started = _stamp()
    if outcome is not Outcome.PASS:
        llm = _LlmCheck("not_run", 0, (), "", None, "the detector path withheld; there was nothing to release")
        reviews: list[dict[str, Any]] = []
    elif not llm_settings["enabled"]:
        llm = _LlmCheck("disabled", 0, (), "", None, None)
        reviews = []
    else:
        llm, reviews = _llm_check(transcript_text, llm_settings)
    review_act = store.activity(
        node=NODE,
        step="llm_check",
        parameters={
            "model_id": llm.model_id or str(llm_settings["model_id"]),
            "max_iterations": int(llm_settings["max_iterations"]),
            "enabled": bool(llm_settings["enabled"]),
        },
        started=review_started,
        ended=_stamp(),
    )
    store.was_associated_with(review_act, software)
    if llm.status not in ("disabled", "not_run"):
        if llm.revision is not None:
            store.was_associated_with(
                review_act, store.agent(agent_type="model", model_id=llm.model_id, commit_sha=llm.revision)
            )
        else:
            store.was_associated_with(
                review_act,
                store.agent(
                    agent_type="model",
                    model_id=llm.model_id,
                    unresolved_reason=llm.failure or "the reviewer did not load",
                ),
            )
        if consensus is not None:
            store.used(review_act, consensus.id)
        for span_id in span_ids:
            store.used(review_act, span_id)
        for review in reviews:
            review_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={"name": _LLM_REVIEW_MEASUREMENT, "signal": "redacted_transcript", **review},
            )
            store.was_generated_by(review_id, review_act)
            store.was_attributed_to(review_id, software)
            view.append(review_id)
    annotation_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": REDACTION_LLM_ANNOTATION, "signal": "redacted_transcript", **llm.as_detail()},
    )
    store.was_generated_by(annotation_id, review_act)
    store.was_attributed_to(annotation_id, software)
    view.append(annotation_id)

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
