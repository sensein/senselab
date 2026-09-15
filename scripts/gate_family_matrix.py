#!/usr/bin/env python
"""Build the family x ruleset-gate matrix over a triage corpus and draw it.

    uv run python scripts/gate_family_matrix.py <features> <out_dir> [--config FILE] [--profile FILE]

``features`` is the shard directory ``analyze_routing_evidence.py`` wrote under ``features/``, or
one shard file out of it. The gates come from ``taxonomy.ruleset`` in the triage configuration, so
a partial override changes what is measured without touching this script. Reads stores only through
that shard, runs no model, and writes nothing into a run directory.

``out_dir`` receives five outputs. ``gate_matrix.parquet`` is one row per (task family, gate):
how many recordings the gate fired on, was silent on and could not be read on, the three rates, and
the distribution of what it read. ``branch_agreement.json`` carries the per-branch 2x2 against the
declared family with recall, precision and specificity, plus the families the reference sets assign
to no branch. ``disagreements.parquet`` is one row per recording and branch the two sides differ
on, naming the gate the difference turns on and what it read; ``disagreement_groups.parquet`` is
the same grouped by task family. ``gate_matrix_fired.png`` and ``gate_matrix_unavailable.png`` are
the heatmap's two panels.

The declared task family is a reference standard and not ground truth. Every rate here is an
agreement rate with the declaration, and a disagreement can be routing reading the recording
correctly against a declaration the participant did not follow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures
from senselab.audio.workflows.triage.routing_analysis.gate_matrix import (
    DECIDING_CLASSES,
    DISAGREEMENT_KINDS,
    MISSED,
    DisagreementGroup,
    GateMatrix,
    gate_matrix,
    group_disagreements,
    load_disagreement_profile,
    qualify_disagreements,
    routing_branch_of,
    unassigned_families,
)
from senselab.audio.workflows.triage.routing_analysis.gate_plot import (
    FIRED_PANEL,
    UNAVAILABLE_PANEL,
    HeatmapStyle,
    write_gate_matrix,
)
from senselab.audio.workflows.triage.routing_analysis.report import load_features
from senselab.audio.workflows.triage.routing_analysis.ruleset import (
    BranchScore,
    Ruleset,
    evaluate_routes,
    load_ruleset,
    score_branches,
)
from senselab.audio.workflows.triage.routing_analysis.tables import write_rows
from senselab.audio.workflows.triage.vocabulary import BRANCHES

MATRIX_FILE = "gate_matrix.parquet"
AGREEMENT_FILE = "branch_agreement.json"
DISAGREEMENT_FILE = "disagreements.parquet"
GROUP_FILE = "disagreement_groups.parquet"
FIRED_FIGURE = "gate_matrix_fired.png"
UNAVAILABLE_FIGURE = "gate_matrix_unavailable.png"

TOP_GROUPS = 25
"""How many disagreement groups the stdout table shows before it stops."""

FRAMING = (
    "Declared task family is a reference standard, NOT ground truth. Every rate below is an "
    "agreement rate with the declaration; either side can be the one that is wrong."
)


def _rate(value: float | None) -> str:
    """One rate formatted for a table, or a dash where it is undefined.

    Args:
        value: A rate or None.

    Returns:
        Three decimal places, or ``-`` for a rate that does not exist. A dash is never a zero.
    """
    return "-" if value is None else f"{value:.3f}"


def print_matrix(matrix: GateMatrix, ruleset: Ruleset) -> None:
    """Write the family x gate fired and unavailable rates to stdout.

    Args:
        matrix: The matrix.
        ruleset: The loaded ruleset, read for which branch each gate routes.
    """
    print(f"\nfamily x gate, {matrix.n_recordings} recordings over {len(matrix.families)} families")
    print(
        f"  {'branch':<8s} " + " ".join(f"{routing_branch_of(ruleset, gate) or 'flag':>10s}" for gate in matrix.gates)
    )
    header = " ".join(f"{gate.split('.')[-1][:10]:>10s}" for gate in matrix.gates)
    print(f"  {'family':<40s} {'n':>5s} {header}")
    for family in matrix.families:
        cells = [matrix.cell(family, gate) for gate in matrix.gates]
        fired = " ".join(f"{_rate(cell.fired_rate_evaluable):>10s}" for cell in cells)
        print(f"  {family:<40s} {matrix.recordings[family]:5d} {fired}")
    print("\n  a dash is a gate no recording of that family could evaluate, not a gate that never fired")
    print(f"  {'family':<40s} {'n':>5s} " + " ".join(f"{gate.split('.')[-1][:10]:>10s}" for gate in matrix.gates))
    for family in matrix.families:
        unavailable = " ".join(f"{_rate(matrix.cell(family, gate).unavailable_rate):>10s}" for gate in matrix.gates)
        print(f"  {family:<40s} {matrix.recordings[family]:5d} {unavailable}")
    print("  (the second block is the unavailable rate)")


def print_agreement(scores: dict[str, BranchScore], unassigned: dict[str, int], n_recordings: int) -> None:
    """Write each branch's agreement with the declared family to stdout.

    Args:
        scores: The per-branch 2x2 tables.
        unassigned: Families the reference sets assign to no branch, and their counts.
        n_recordings: How many recordings were scored.
    """
    print(f"\nbranch agreement with the declared family, {n_recordings} recordings")
    print(f"  {FRAMING}")
    print(
        f"\n  {'branch':<8s} {'reference set':<20s} {'agreed':>7s} {'extra':>7s} {'silent':>7s} "
        f"{'missed':>7s} {'recall':>7s} {'precis':>7s} {'specif':>7s}"
    )
    for branch in BRANCHES:
        score = scores[branch]
        table = score.against_reference
        print(
            f"  {branch:<8s} {score.reference_family_set:<20s} {table.tp:7d} {table.fp:7d} "
            f"{table.tn:7d} {table.fn:7d} {_rate(table.sensitivity):>7s} "
            f"{_rate(table.precision):>7s} {_rate(table.specificity):>7s}"
        )
    print("  agreed=declared and routed  extra=routed not declared  missed=declared not routed  silent=neither")
    print("\n  with each branch's construction exclusions held out of the population:")
    for branch in BRANCHES:
        score = scores[branch]
        if not score.held_out_family_set:
            continue
        kept = score.excluding_construction
        print(
            f"  {branch:<8s} held out {score.held_out_family_set} ({score.n_held_out} recordings): "
            f"recall {_rate(kept.sensitivity)} precision {_rate(kept.precision)} "
            f"specificity {_rate(kept.specificity)}"
        )
    if unassigned:
        total = sum(unassigned.values())
        print(f"\n  {len(unassigned)} families ({total} recordings) are in NO branch's reference set.")
        print("  They are in no denominator above, on either side. Listed here, not folded in:")
        for family, count in unassigned.items():
            print(f"    {family:<40s} {count:6d}")
    else:
        print("\n  every family seen is assigned to at least one branch by a reference set")


def print_groups(groups: list[DisagreementGroup], band: float, source: str) -> None:
    """Write the disagreement groups to stdout, misses first.

    Args:
        groups: What :func:`~senselab.audio.workflows.triage.routing_analysis.gate_matrix.
            group_disagreements` returned.
        band: The relative margin inside which a reading is reported as a threshold question.
        source: Which profile the band came from.
    """
    print("\ndisagreements by task family, deciding gate and finding")
    print(f"  near_threshold = deciding gate within {band:.0%} of its own threshold (relative margin)")
    print(f"  a declared reporting convention from {source}, not a fitted cut")
    print(
        "  near_threshold: a threshold question   far: an evidence question   unavailable: a missing-feature question"
    )
    for kind in DISAGREEMENT_KINDS:
        rows = [group for group in groups if group.kind == kind]
        label = "declared and NOT routed" if kind == MISSED else "routed and NOT declared"
        print(f"\n  {kind.upper()} — {label}: {sum(group.n for group in rows)} over {len(rows)} family/branch groups")
        if not rows:
            print("    none")
            continue
        print(
            f"    {'family':<38s} {'branch':<7s} {'n':>5s} {'near':>5s} {'far':>5s} {'unav':>5s} "
            f"{'m p50':>8s} {'m p90':>8s}  deciding gates"
        )
        for group in rows[:TOP_GROUPS]:
            median = group.margins.get("median")
            ninth = group.margins.get("p90")
            print(
                f"    {group.family:<38s} {group.branch:<7s} {group.n:5d} "
                f"{group.findings['near_threshold']:5d} {group.findings['far']:5d} "
                f"{group.findings['unavailable']:5d} "
                f"{'-' if median is None else f'{median:+8.3f}'} "
                f"{'-' if ninth is None else f'{ninth:+8.3f}'}  {group.gates_summary()}"
            )
        if len(rows) > TOP_GROUPS:
            print(f"    ... {len(rows) - TOP_GROUPS} more groups in {GROUP_FILE}")


def main(argv: list[str] | None = None) -> int:
    """Build the matrix, score the agreement, qualify the disagreements and draw the heatmap.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        0 on success.

    Raises:
        SystemExit: When the shard holds no recordings, which is a run that read nothing and would
            otherwise report success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="a features shard written by analyze_routing_evidence.py")
    parser.add_argument("out_dir", type=Path, help="where the matrix, the tables and the figures are written")
    parser.add_argument("--config", type=Path, default=None, help="a partial triage config override")
    parser.add_argument(
        "--profile", type=Path, default=None, help="a disagreement profile; default the packaged newest"
    )
    parser.add_argument("--no-annotate", action="store_true", help="draw the heatmap without per-cell numbers")
    arguments = parser.parse_args(argv)

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_triage_config(arguments.config)
    ruleset: Ruleset = load_ruleset(config)
    profile = load_disagreement_profile(None if arguments.profile is None else str(arguments.profile))
    band = float(profile["near_threshold_band"])
    print(f"[gates] {len(ruleset.gates)} gates, config hash {config.config_hash}", flush=True)

    records: list[RecordingFeatures] = load_features(arguments.features)
    if not records:
        raise SystemExit(f"no recordings in {arguments.features}; nothing to reduce")
    print(f"[gates] reducing {len(records)} recordings", flush=True)

    matrix = gate_matrix(records, ruleset)
    evaluations = [evaluate_routes(record, ruleset) for record in records]
    scores = score_branches(evaluations, ruleset)
    unassigned = unassigned_families(records, ruleset)
    disagreements = qualify_disagreements(records, ruleset, near_threshold_band=band)
    groups = group_disagreements(disagreements)

    header = {
        "config_hash": config.config_hash,
        "recordings": len(records),
        "near_threshold_band": band,
        "disagreement_profile": profile["source"],
        "reference": "declared task family; a reference standard, not ground truth",
    }
    n_cells = write_rows(matrix.rows(), arguments.out_dir / MATRIX_FILE, header)
    n_rows = write_rows([entry.as_json() for entry in disagreements], arguments.out_dir / DISAGREEMENT_FILE, header)
    n_groups = write_rows([group.as_json() for group in groups], arguments.out_dir / GROUP_FILE, header)
    agreement: dict[str, Any] = {
        **header,
        "framing": FRAMING,
        "finding_classes": list(DECIDING_CLASSES),
        "branches": {branch: score.as_json() for branch, score in scores.items()},
        "unassigned_families": unassigned,
    }
    (arguments.out_dir / AGREEMENT_FILE).write_text(json.dumps(agreement, indent=1))

    style = HeatmapStyle(annotate=not arguments.no_annotate)
    subtitle = f"{len(records)} recordings, config {config.config_hash[:12]}"
    for panel, name in ((FIRED_PANEL, FIRED_FIGURE), (UNAVAILABLE_PANEL, UNAVAILABLE_FIGURE)):
        write_gate_matrix(matrix, ruleset, arguments.out_dir / name, panel=panel, style=style, subtitle=subtitle)

    print_matrix(matrix, ruleset)
    print_agreement(scores, unassigned, len(records))
    print_groups(groups, band, profile["source"])
    print(
        f"\n[gates] wrote {n_cells} cells, {n_rows} disagreements, {n_groups} groups and 2 figures "
        f"to {arguments.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
