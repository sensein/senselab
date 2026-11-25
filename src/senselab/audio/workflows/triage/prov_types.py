"""The provenance entity types triage writes, registered with the store when the triage package loads (plan step 6c).

``branch_report`` is what a node that reports rather than decides writes: the three branches and QUALITY. It
carries task conformance and deviations and no outcome, which is why it is not a ``verdict``: a reader folding
verdicts must not pick up a decision no branch made.

``kind`` is **historical**: TAXONOMY's evidence fold wrote it until 2026-09-13 and nothing writes it now. It stays
registered because the store is append-only: a reader that refuses a record it previously wrote makes finished runs
unopenable. The writer's vocabulary may shrink; this one may not. See ``specs/20260912-ruleset-in-pipeline/design.md``.
"""

from __future__ import annotations

from typing import Literal, get_args

from senselab.utils.prov_store import register_prov_types

TRIAGE_PROV_TYPE = Literal[
    "span",
    "word",
    "speaker",
    "interval",
    "measurement",
    "kind",
    "stream",
    "pii",
    "verdict",
    "branch_report",
    "assertion",
    "target_match",
    "branch_decision",
    "enrollment",
]
"""Every entity type a triage store may hold, for type checking; :data:`TRIAGE_PROV_TYPES` is what is registered."""

TRIAGE_PROV_TYPES: tuple[str, ...] = get_args(TRIAGE_PROV_TYPE)

register_prov_types(TRIAGE_PROV_TYPES)
