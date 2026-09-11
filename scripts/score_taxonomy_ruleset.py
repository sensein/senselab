#!/usr/bin/env python
"""Score the family taxonomy ruleset over a completed triage run's extracted features.

    uv run python scripts/score_taxonomy_ruleset.py <features.jsonl> <out_dir> [--config FILE]

``features.jsonl`` is what ``analyze_routing_evidence.py`` wrote under ``features/``. The ruleset
is read from the triage configuration, so a partial override changes the gates without touching
this script. ``out_dir`` receives ``ruleset_score.json``: the corpus totals, the per-branch counts,
the per-branch sensitivity and specificity against the reference family sets, and one tally per
family. Every recording lands in exactly one of ``routed``, ``empty`` and ``unexplained``, and only
the last of those is a charge against the ruleset.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.routing_analysis.report import Confusion, load_features
from senselab.audio.workflows.triage.routing_analysis.ruleset import (
    AXES,
    ROUTE_STATES,
    RouteEvaluation,
    Ruleset,
    branches_on,
    evaluate_routes,
    load_ruleset,
    score_branches,
    tally_families,
)
from senselab.audio.workflows.triage.vocabulary import BRANCHES

SCORE_FILE = "ruleset_score.json"


def corpus_totals(evaluations: list[RouteEvaluation]) -> tuple[int, dict[str, int], dict[str, dict[str, int]]]:
    """Fold every evaluation into one set of corpus-wide counts.

    Args:
        evaluations: One evaluation per recording.

    Returns:
        The recording count, how many landed in each
        :class:`~senselab.audio.workflows.triage.routing_analysis.ruleset.RouteState`, and
        per-branch counts for each of
        :data:`~senselab.audio.workflows.triage.routing_analysis.ruleset.AXES`.
    """
    per_axis = {axis: dict.fromkeys(BRANCHES, 0) for axis in AXES}
    states = dict.fromkeys(ROUTE_STATES, 0)
    for evaluation in evaluations:
        for axis, counts in per_axis.items():
            for branch in branches_on(evaluation, axis):
                counts[branch] += 1
        states[evaluation.state.value] += 1
    return len(evaluations), states, per_axis


def print_table(
    recordings: int,
    states: dict[str, int],
    per_axis: dict[str, dict[str, int]],
    scores: dict[str, Confusion],
    tallies: dict[str, dict[str, Any]],
) -> None:
    """Write the corpus totals, the per-branch scores and the worst unexplained families to stdout.

    Args:
        recordings: How many recordings were scored.
        states: How many landed in each route state.
        per_axis: The per-branch counts from :func:`corpus_totals`.
        scores: The per-branch 2x2 against the reference family sets.
        tallies: Family name to its tally, as dictionaries.
    """
    print(f"\nrecordings {recordings}")
    for name in ROUTE_STATES:
        print(f"  {name:12s} {states[name]:8d} ({states[name] / recordings:.1%})")
    print()
    print(f"{'branch':8s} {'routed':>8s} {'declared':>9s} {'agreed':>8s} {'missed':>8s} {'extra':>8s} {'unavail':>8s}")
    for branch in BRANCHES:
        print(
            f"{branch:8s} {per_axis['routed'][branch]:8d} {per_axis['declared'][branch]:9d} "
            f"{per_axis['agreed'][branch]:8d} {per_axis['missed'][branch]:8d} "
            f"{per_axis['extra'][branch]:8d} {per_axis['unavailable'][branch]:8d}"
        )

    print(f"\n{'branch':8s} {'tp':>8s} {'fp':>8s} {'tn':>8s} {'fn':>8s} {'sens':>7s} {'spec':>7s}")
    for branch, table in scores.items():
        sensitivity = f"{table.sensitivity:.3f}" if table.sensitivity is not None else "-"
        specificity = f"{table.specificity:.3f}" if table.specificity is not None else "-"
        print(
            f"{branch:8s} {table.tp:8d} {table.fp:8d} {table.tn:8d} {table.fn:8d} {sensitivity:>7s} {specificity:>7s}"
        )

    worst = sorted(tallies.values(), key=lambda tally: -int(tally["states"]["unexplained"]))[:12]
    print(f"\n{'family':40s} {'n':>6s} {'empty':>7s} {'unexplained':>13s} {'rate':>7s}")
    for tally in worst:
        count = int(tally["recordings"])
        unexplained, blank = int(tally["states"]["unexplained"]), int(tally["states"]["empty"])
        print(f"{tally['family']:40s} {count:6d} {blank:7d} {unexplained:13d} {unexplained / count:7.1%}")


def main(argv: list[str] | None = None) -> int:
    """Score the ruleset and write the report.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="a features.jsonl written by analyze_routing_evidence.py")
    parser.add_argument("out_dir", type=Path, help="where the score is written")
    parser.add_argument("--config", type=Path, default=None, help="a partial triage config override")
    arguments = parser.parse_args(argv)

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_triage_config(arguments.config)
    ruleset: Ruleset = load_ruleset(config)
    print(f"[ruleset] {len(ruleset.gates)} gates, config hash {config.config_hash}", flush=True)

    records = load_features(arguments.features)
    print(f"[ruleset] scoring {len(records)} recordings", flush=True)
    evaluations = [evaluate_routes(record, ruleset) for record in records]
    tallies = {family: asdict(tally) for family, tally in tally_families(evaluations).items()}
    scores = score_branches(evaluations)
    recordings, states, per_axis = corpus_totals(evaluations)

    totals: dict[str, Any] = {"recordings": recordings, "states": states, **per_axis}
    score = {
        "config_hash": config.config_hash,
        "totals": totals,
        "branches": {branch: table.as_json() for branch, table in scores.items()},
        "families": tallies,
    }
    (arguments.out_dir / SCORE_FILE).write_text(json.dumps(score, indent=1))
    print_table(recordings, states, per_axis, scores, tallies)
    print(f"\n[ruleset] wrote {arguments.out_dir / SCORE_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
