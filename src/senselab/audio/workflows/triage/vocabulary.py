"""The triage graph's shared vocabulary and the file-level fold."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class Outcome(Enum):
    """What a node concluded."""

    PASS = "pass"
    FLAG = "flag"
    FAIL = "fail"


class KindState(Enum):
    """Whether a kind is in the recording."""

    PRESENT = "present"
    ABSENT = "absent"
    UNDECIDED = "undecided"


class RunState(Enum):
    """Whether a node ran at all."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERRORED = "errored"


class Release(Enum):
    """Whether a redacted artifact may be handed on."""

    RELEASABLE = "releasable"
    WITHHELD = "withheld"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class NodeVerdict:
    """One node's conclusion."""

    node: str
    outcome: Outcome
    kind: str | None
    why: str


@dataclass(frozen=True)
class FileVerdict:
    """The graph's conclusion about one recording."""

    triage: Outcome
    release: Release
    kinds: dict[str, KindState]
    reasons: list[NodeVerdict] = field(default_factory=list)
    ran: dict[str, RunState] = field(default_factory=dict)


_BRANCH_FOR_KIND = {"airway": "AIRWAY", "speech": "SPEECH", "voice_no_words": "VOICE"}


def fold_file_verdict(
    node_verdicts: Sequence[NodeVerdict],
    kind_predictions: Mapping[str, KindState],
    ran: Mapping[str, RunState],
    release: Release = Release.NOT_ASSESSED,
) -> FileVerdict:
    """Combine every node's verdict into one for the recording.

    A branch ``fail`` means that branch had no subject, which is normal. It is read against what TAXONOMY
    predicted for its kind, and a disagreement between the two is a flag.

    Args:
        node_verdicts: Every node's conclusion, in graph order.
        kind_predictions: TAXONOMY's prediction per kind.
        ran: Whether each node ran.
        release: REDACT's release state, if it ran.

    Returns:
        The file verdict, carrying every contributing reason rather than only the deciding one.
    """
    reasons: list[NodeVerdict] = list(node_verdicts)
    kinds = dict(kind_predictions)
    by_kind = {v.kind: v for v in node_verdicts if v.kind}

    contradictions: list[NodeVerdict] = []
    for kind, predicted in kind_predictions.items():
        node = _BRANCH_FOR_KIND.get(kind, kind.upper())
        verdict = by_kind.get(kind)
        state = ran.get(node)
        if verdict is None:
            if state in (RunState.SKIPPED, RunState.ERRORED) and predicted in (
                KindState.PRESENT,
                KindState.UNDECIDED,
            ):
                contradictions.append(
                    NodeVerdict(
                        node, Outcome.FLAG, kind, f"contradiction: {kind} was {predicted.value} and {node} never ran"
                    )
                )
            continue
        if predicted is KindState.PRESENT and verdict.outcome is Outcome.FAIL:
            contradictions.append(
                NodeVerdict(
                    node, Outcome.FLAG, kind, f"contradiction: {kind} predicted present, {node} found no subject"
                )
            )
        elif predicted is KindState.ABSENT and verdict.outcome is Outcome.PASS:
            kinds[kind] = KindState.PRESENT
            contradictions.append(
                NodeVerdict(node, Outcome.FLAG, kind, f"contradiction: {kind} predicted absent, {node} passed")
            )
        elif predicted is KindState.UNDECIDED:
            kinds[kind] = KindState.PRESENT if verdict.outcome is Outcome.PASS else KindState.ABSENT
    reasons.extend(contradictions)

    admit = next((v for v in node_verdicts if v.node == "ADMIT"), None)
    if admit and admit.outcome is Outcome.FAIL:
        return FileVerdict(Outcome.FAIL, release, kinds, [admit], dict(ran))
    if any(v.outcome is Outcome.FLAG for v in reasons):
        return FileVerdict(Outcome.FLAG, release, kinds, reasons, dict(ran))
    if kinds and all(s is KindState.ABSENT for s in kinds.values()):
        reasons.append(NodeVerdict("VERDICT", Outcome.FAIL, None, "every kind is absent; no branch had a subject"))
        return FileVerdict(Outcome.FAIL, release, kinds, reasons, dict(ran))
    return FileVerdict(Outcome.PASS, release, kinds, reasons, dict(ran))
