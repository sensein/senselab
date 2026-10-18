"""Iterative metric-driven ranking workflow.

Rank a corpus of audio items (files or segments) by a versioned metric that
combines already-computed signals, refine the metric iteratively from spot-check
annotations, and track how items move between metric versions.

See ``specs/20260604-173646-iterative-metric-ranking/`` for the spec, plan, and
contracts. Public entrypoints are re-exported here.
"""

from __future__ import annotations

from senselab.audio.workflows.ranking.evaluate import evaluate_ranking
from senselab.audio.workflows.ranking.metric import score_items
from senselab.audio.workflows.ranking.rank import rank_corpus, recalibrate_and_propose, update_metric_manual
from senselab.audio.workflows.ranking.triage import apply_triage_threshold
from senselab.audio.workflows.ranking.types import (
    Annotation,
    MetricDefinition,
    MetricVersion,
    MovementReport,
    Ranking,
    RankingItem,
    SeparationResult,
    SignalTerm,
    TriageThreshold,
)

__all__ = [
    "Annotation",
    "MetricDefinition",
    "MetricVersion",
    "MovementReport",
    "Ranking",
    "RankingItem",
    "SeparationResult",
    "SignalTerm",
    "TriageThreshold",
    "apply_triage_threshold",
    "evaluate_ranking",
    "rank_corpus",
    "recalibrate_and_propose",
    "score_items",
    "update_metric_manual",
]
