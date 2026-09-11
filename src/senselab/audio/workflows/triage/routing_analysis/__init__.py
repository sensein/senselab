"""Measuring what TAXONOMY would have to read to route a recording to a branch.

Reads a completed triage run's provenance stores, extracts the evidence each candidate routing
detector would use, and scores every detector over a threshold sweep against a stated reference
standard. Runs no model and writes nothing into a run directory.

The reference standards, their weakness and the measurements taken with this module live in
``specs/20260910-taxonomy-routing-evidence/``.
"""

from senselab.audio.workflows.triage.routing_analysis.detectors import (
    DETECTORS,
    Detector,
    detector_value,
    sweep_points,
)
from senselab.audio.workflows.triage.routing_analysis.families import declared_kinds, task_family, task_id_of
from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures, extract_features, read_store
from senselab.audio.workflows.triage.routing_analysis.report import (
    BASELINE_RULES,
    REFERENCE_STANDARDS,
    Confusion,
    ReferenceStandard,
    RoutingRule,
    bucket_augmentation,
    bucket_coverage,
    disagreements,
    dump_features,
    load_features,
    prevalence,
    score_detector,
    taxonomy_as_run,
    write_report,
)
from senselab.audio.workflows.triage.routing_analysis.ruleset import (
    FAMILY_SETS,
    FamilyTally,
    Gate,
    GateOutcome,
    RouteEvaluation,
    Ruleset,
    evaluate_gate,
    evaluate_routes,
    gate_value,
    load_ruleset,
    max_token_repeat,
    score_branches,
    tally_families,
)

__all__ = [
    "BASELINE_RULES",
    "DETECTORS",
    "FAMILY_SETS",
    "REFERENCE_STANDARDS",
    "Confusion",
    "Detector",
    "FamilyTally",
    "Gate",
    "GateOutcome",
    "RecordingFeatures",
    "ReferenceStandard",
    "RouteEvaluation",
    "RoutingRule",
    "Ruleset",
    "bucket_augmentation",
    "bucket_coverage",
    "declared_kinds",
    "detector_value",
    "disagreements",
    "dump_features",
    "evaluate_gate",
    "evaluate_routes",
    "extract_features",
    "gate_value",
    "load_features",
    "load_ruleset",
    "max_token_repeat",
    "prevalence",
    "read_store",
    "score_branches",
    "score_detector",
    "sweep_points",
    "tally_families",
    "task_family",
    "task_id_of",
    "taxonomy_as_run",
    "write_report",
]
