#!/usr/bin/env python
"""Score the family taxonomy ruleset over a completed triage run's extracted features.

    uv run python scripts/score_taxonomy_ruleset.py <features.jsonl> <out_dir> [--config FILE]

``features.jsonl`` is what ``analyze_routing_evidence.py`` wrote under ``features/``. The ruleset
is read from the triage configuration, so a partial override changes the gates without touching
this script. ``out_dir`` receives ``ruleset_score.json``: the corpus totals, the per-branch counts
and one tally per family.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.routing_analysis.report import load_features
from senselab.audio.workflows.triage.routing_analysis.ruleset import (
    RouteEvaluation,
    Ruleset,
    evaluate_routes,
    load_ruleset,
    tally_families,
)
from senselab.audio.workflows.triage.vocabulary import BRANCHES

SCORE_FILE = "ruleset_score.json"


TOTAL_FIELDS = ("declared", "confirmed", "discovered", "unconfirmed", "unavailable")


def corpus_totals(evaluations: list[RouteEvaluation]) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Fold every evaluation into one set of corpus-wide counts.

    Args:
        evaluations: One evaluation per recording.

    Returns:
        The recording count, the fall-through count, and per-branch counts for each of
        :data:`TOTAL_FIELDS`.
    """
    per_field = {field: dict.fromkeys(BRANCHES, 0) for field in TOTAL_FIELDS}
    fell_through = 0
    for evaluation in evaluations:
        for field, counts in per_field.items():
            for branch in getattr(evaluation, field):
                counts[branch] += 1
        fell_through += int(evaluation.fell_through)
    return len(evaluations), fell_through, per_field


def print_table(
    recordings: int, fell_through: int, per_field: dict[str, dict[str, int]], tallies: dict[str, dict[str, Any]]
) -> None:
    """Write the corpus totals and the worst fall-through families to stdout.

    Args:
        recordings: How many recordings were scored.
        fell_through: How many routed to no branch at all.
        per_field: The per-branch counts from :func:`corpus_totals`.
        tallies: Family name to its tally, as dictionaries.
    """
    print(f"\nrecordings {recordings}   fell through every branch {fell_through} ({fell_through / recordings:.1%})\n")
    print(f"{'branch':8s} {'declared':>9s} {'confirmed':>10s} {'rate':>7s} {'discovered':>11s} {'unavailable':>12s}")
    for branch in BRANCHES:
        declared = per_field["declared"][branch]
        confirmed = per_field["confirmed"][branch]
        rate = f"{confirmed / declared:.3f}" if declared else "-"
        print(
            f"{branch:8s} {declared:9d} {confirmed:10d} {rate:>7s} "
            f"{per_field['discovered'][branch]:11d} {per_field['unavailable'][branch]:12d}"
        )

    worst = sorted(tallies.values(), key=lambda tally: -int(tally["fell_through"]))[:12]
    print(f"\n{'family':40s} {'n':>6s} {'fell through':>13s} {'rate':>7s}")
    for tally in worst:
        count, fallen = int(tally["recordings"]), int(tally["fell_through"])
        print(f"{tally['family']:40s} {count:6d} {fallen:13d} {fallen / count:7.1%}")


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
    recordings, fell_through, per_field = corpus_totals(evaluations)

    totals = {"recordings": recordings, "fell_through": fell_through, **per_field}
    score = {"config_hash": config.config_hash, "totals": totals, "families": tallies}
    (arguments.out_dir / SCORE_FILE).write_text(json.dumps(score, indent=1))
    print_table(recordings, fell_through, per_field, tallies)
    print(f"\n[ruleset] wrote {arguments.out_dir / SCORE_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
