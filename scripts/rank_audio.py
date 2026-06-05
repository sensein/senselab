#!/usr/bin/env python
"""``rank_audio`` — rank a corpus by a metric, refine it, and track movement.

Thin CLI wrapper over :mod:`senselab.audio.workflows.ranking`. Subcommands:
``rank``, ``evaluate``, ``sample``, ``annotate``, ``update-metric``,
``recalibrate``, ``threshold``, ``movement``. See
``specs/20260604-173646-iterative-metric-ranking/contracts/rank-cli.md``.

Exit codes:
  0  Success.
  2  Usage / invalid metric (e.g. references an unknown signal).
  3  Requested value not evaluable (insufficient annotated data).
  4  Recalibration refused (insufficient / low-variety annotations).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from senselab.audio.workflows.ranking import annotate, evaluate, io, movement, rank
from senselab.audio.workflows.ranking.metric import MetricError
from senselab.audio.workflows.ranking.store import RankingStore, _defn_from_dict
from senselab.audio.workflows.ranking.triage import apply_triage_threshold
from senselab.audio.workflows.ranking.types import Annotation, MetricDefinition


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_definition(path: Path) -> MetricDefinition:
    return _defn_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _emit(obj: object, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, default=str, indent=2))


def cmd_rank(args: argparse.Namespace) -> int:
    """Score + rank a corpus, creating a new metric version."""
    store = RankingStore(args.store)
    try:
        defn = _load_definition(Path(args.metric))
        ranking = rank.rank_corpus(
            store, args.signals, defn, created_at=_now(),
            band_fraction=args.band_fraction, as_version=args.as_version,
        )
    except (MetricError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bands = {b: sum(1 for it in ranking.items if it.band == b) for b in ("top", "middle", "bottom")}
    print(f"version={ranking.version_id} scored={ranking.n_scored} unscorable={ranking.n_unscorable} bands={bands}")
    _emit({"version_id": ranking.version_id, "n_scored": ranking.n_scored,
           "n_unscorable": ranking.n_unscorable, "bands": bands}, args.json)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Report rank-agreement + band separation for a version."""
    store = RankingStore(args.store)
    ranking = io.read_ranking(store.ranking_path(args.version))
    result = evaluate.evaluate_ranking(
        ranking, annotate.load_active_annotations(store), separation_target=args.separation_target
    )
    print(
        f"version={result.version_id} evaluable={result.evaluable} "
        f"spearman={result.rank_agreement_spearman} kendall={result.rank_agreement_kendall_tau_b} "
        f"band_agreement={result.band_pairwise_agreement} margin={result.band_quality_margin} "
        f"meets_target={result.meets_separation_target}"
    )
    if result.reason:
        print(f"note: {result.reason}")
    _emit(asdict(result), args.json)
    return 0 if result.evaluable else 3


def cmd_sample(args: argparse.Namespace) -> int:
    """Select items to spot-check."""
    store = RankingStore(args.store)
    ranking = io.read_ranking(store.ranking_path(args.version))
    ids = annotate.sample_items(ranking, args.n, strategy=args.strategy, threshold_rank=args.threshold)
    for iid in ids:
        print(iid)
    _emit({"items": ids}, args.json)
    return 0


def cmd_annotate(args: argparse.Namespace) -> int:
    """Record a quality annotation (latest-wins)."""
    store = RankingStore(args.store)
    unit = store.unit() or "file"
    try:
        annotate.add_annotation(
            store,
            Annotation(item_id=args.item, label=args.label, score=args.score, unit=unit,
                       reviewed_under_version=args.version, reviewer=args.reviewer,
                       created_at=_now(), note=args.note or ""),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"annotated {args.item} label={args.label} score={args.score}")
    return 0


def cmd_update_metric(args: argparse.Namespace) -> int:
    """Manual metric revision → new version."""
    store = RankingStore(args.store)
    try:
        defn = _load_definition(Path(args.metric))
        ranking = rank.update_metric_manual(store, args.signals, defn, created_at=_now(),
                                             band_fraction=args.band_fraction)
    except (MetricError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"version={ranking.version_id} (manual) scored={ranking.n_scored}")
    return 0


def cmd_recalibrate(args: argparse.Namespace) -> int:
    """Assisted recalibration; writes a new version only with --accept."""
    store = RankingStore(args.store)
    result = rank.recalibrate_and_propose(store, args.signals, base_version_id=args.base)
    print(
        f"status={result.status} annotations={result.n_annotations_used} pairs={result.n_pairs} "
        f"levels={result.n_distinct_levels} before={result.agreement_before} after={result.agreement_after}"
    )
    if result.message:
        print(f"note: {result.message}")
    _emit({k: v for k, v in asdict(result).items() if k != "proposed_definition"}, args.json)
    if result.status == "refused" or result.proposed_definition is None:
        return 4
    if args.accept:
        versions = store.list_versions()
        ranking = rank.rank_corpus(
            store, args.signals, result.proposed_definition, created_at=_now(),
            band_fraction=args.band_fraction, origin="recalibrated",
            parent_version_id=versions[-1] if versions else None, recal=result,
        )
        print(f"accepted → version={ranking.version_id} (recalibrated)")
    return 0


def cmd_threshold(args: argparse.Namespace) -> int:
    """Triage cut readout (auto-accept vs human-review)."""
    store = RankingStore(args.store)
    ranking = io.read_ranking(store.ranking_path(args.version))
    if args.at_rank is not None:
        result = apply_triage_threshold(ranking, annotate.load_active_annotations(store),
                                        cut=args.at_rank, cut_kind="rank")
    else:
        result = apply_triage_threshold(ranking, annotate.load_active_annotations(store),
                                        cut=args.at_percentile, cut_kind="percentile")
    print(
        f"auto_accept={result.n_auto_accept} human_review={result.n_human_review} "
        f"unscorable_routed={result.n_unscorable_routed} above={result.above_counts} "
        f"below={result.below_counts} auto_accept_poor_rate={result.auto_accept_poor_rate}"
    )
    _emit(asdict(result), args.json)
    return 0


def cmd_movement(args: argparse.Namespace) -> int:
    """Compare two versions and write a movement report."""
    store = RankingStore(args.store)
    from_r = io.read_ranking(store.ranking_path(getattr(args, "from")))
    to_r = io.read_ranking(store.ranking_path(args.to))
    try:
        report = movement.compute_movement(from_r, to_r, annotate.load_active_annotations(store))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = store.movement_dir / f"{report.from_version}__{report.to_version}.json"
    io.write_movement_report(out, report)
    print(
        f"from={report.from_version} to={report.to_version} band_summary={report.band_summary} "
        f"added={len(report.added)} removed={len(report.removed)} "
        f"became_unscorable={len(report.became_unscorable)} → {out}"
    )
    _emit({"band_summary": report.band_summary, "added": report.added, "removed": report.removed,
           "became_unscorable": report.became_unscorable}, args.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="rank_audio", description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    def _store(p: argparse.ArgumentParser) -> None:
        p.add_argument("--store", required=True, help="ranking store directory")

    p_rank = sub.add_parser("rank", help="produce a ranking for a metric")
    _store(p_rank)
    p_rank.add_argument("--signals", required=True)
    p_rank.add_argument("--metric", required=True)
    p_rank.add_argument("--as-version", default=None)
    p_rank.add_argument("--band-fraction", type=float, default=0.20)
    p_rank.set_defaults(func=cmd_rank)

    p_eval = sub.add_parser("evaluate", help="ranking-quality report")
    _store(p_eval)
    p_eval.add_argument("--version", required=True)
    p_eval.add_argument("--separation-target", type=float, default=0.80)
    p_eval.set_defaults(func=cmd_evaluate)

    p_sample = sub.add_parser("sample", help="select items to spot-check")
    _store(p_sample)
    p_sample.add_argument("--version", required=True)
    p_sample.add_argument("--n", type=int, default=20)
    p_sample.add_argument("--strategy", choices=["spread", "near-threshold", "disagreement"], default="spread")
    p_sample.add_argument("--threshold", type=int, default=None, help="rank for near-threshold strategy")
    p_sample.set_defaults(func=cmd_sample)

    p_ann = sub.add_parser("annotate", help="record a quality annotation")
    _store(p_ann)
    p_ann.add_argument("--item", required=True)
    p_ann.add_argument("--label", choices=["good", "acceptable", "poor"], default=None)
    p_ann.add_argument("--score", type=float, default=None)
    p_ann.add_argument("--version", default=None)
    p_ann.add_argument("--reviewer", default=None)
    p_ann.add_argument("--note", default=None)
    p_ann.set_defaults(func=cmd_annotate)

    p_upd = sub.add_parser("update-metric", help="manual metric revision")
    _store(p_upd)
    p_upd.add_argument("--signals", required=True)
    p_upd.add_argument("--metric", required=True)
    p_upd.add_argument("--band-fraction", type=float, default=0.20)
    p_upd.set_defaults(func=cmd_update_metric)

    p_rec = sub.add_parser("recalibrate", help="assisted recalibration from annotations")
    _store(p_rec)
    p_rec.add_argument("--signals", required=True)
    p_rec.add_argument("--base", default=None)
    p_rec.add_argument("--accept", action="store_true")
    p_rec.add_argument("--band-fraction", type=float, default=0.20)
    p_rec.set_defaults(func=cmd_recalibrate)

    p_thr = sub.add_parser("threshold", help="triage cut readout")
    _store(p_thr)
    p_thr.add_argument("--version", required=True)
    group = p_thr.add_mutually_exclusive_group(required=True)
    group.add_argument("--at-rank", type=int, default=None)
    group.add_argument("--at-percentile", type=float, default=None)
    p_thr.set_defaults(func=cmd_threshold)

    p_mov = sub.add_parser("movement", help="compare two versions")
    _store(p_mov)
    p_mov.add_argument("--from", required=True)
    p_mov.add_argument("--to", required=True)
    p_mov.set_defaults(func=cmd_movement)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
