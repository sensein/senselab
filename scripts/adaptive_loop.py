#!/usr/bin/env python3
"""Adaptive uncertainty loop (prototype driver).

Runs the loop from ``specs/20260723-225523-dynamic-uncertainty-workflow`` over a
completed ``analyze_audio.py`` run directory: round 1 ingests the run's
uncertainty parquets into the belief store (with a re-aggregation parity
check), rounds 2..K execute the policy-ranked intervention catalog (stream
election, uncorroborated-speech attenuation, missed-speech correction, cache-replay
reserve-ASR escalation; live-backend rules defer with guard reasons), and the
final round fuses a consensus transcript / diarization / speech_presence track with a
full decision audit trail.

Usage:
    uv run python scripts/adaptive_loop.py artifacts/analyze_audio/<run_dir> \
        --cache-dir artifacts/analyze_audio_cache \
        --ground-truth ~/Downloads/updated-label-XXXX.json

Works in minimal environments (no torch): heavy senselab imports are bypassed
via submodule loading when the full package cannot import.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def _ensure_light_importable() -> None:
    """Make senselab importable when running from a checkout without installation.

    The package ``__init__`` chain is lazy (PEP 562 — see T046 in
    ``specs/20260723-225523-dynamic-uncertainty-workflow/architecture-review.md``),
    so the loop's pure submodules import without torch/transformers; all that is
    needed here is the src/ path when senselab isn't pip-installed.
    """
    try:
        import senselab.audio.workflows.audio_analysis.aggregate  # noqa: F401
    except ImportError:
        if str(_REPO_SRC) not in sys.path:
            sys.path.insert(0, str(_REPO_SRC))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="Completed analyze_audio run directory")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/analyze_audio_cache"))
    parser.add_argument("--policy", type=Path, default=None, help="Policy YAML overriding the default")
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default: run_dir)")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--aggregator", default=None, help="Override (default: from run's disagreements.json)")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Label Studio export JSON")
    args = parser.parse_args(argv)

    from senselab.audio.workflows.audio_analysis.layout import evidence_dir

    if not (evidence_dir(args.run_dir) / "passes.json").exists():
        print(f"ERROR: {args.run_dir} is not an analyze_audio run dir (no L1/passes.json)", file=sys.stderr)
        return 2

    _ensure_light_importable()
    from senselab.audio.workflows.audio_analysis.adaptive.evaluate import evaluate_against_ground_truth
    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop

    result = run_adaptive_loop(
        args.run_dir,
        cache_dir=args.cache_dir,
        policy_path=args.policy,
        out_dir=args.out,
        max_rounds=args.max_rounds,
        aggregator=args.aggregator,
    )
    out_dir = Path(result["out_dir"])
    # Both, when they disagree: the loop's own reason for stopping is not the same claim as
    # whether the answer settled, and an oscillating run stops with "nothing left to fire".
    verdict = result["termination_reason"]
    if verdict != result["run_state"]:
        verdict = f"{verdict} (loop stopped: {result['run_state']})"
    print(f"termination: {verdict}  rounds: {result['rounds']}")
    print("replay:", json.dumps(result["replay_check"]))
    print(
        f"interventions fired: {result['n_interventions_fired']}  "
        f"fused words: {result['n_words_fused']}  stream: {result['fusion_stream']}"
    )
    print(f"final/: {out_dir / 'final'}")

    # Visual timeline. run_adaptive_loop now emits this itself, so re-running it
    # here only adds the ground-truth overlay this script uniquely supports.
    try:
        from senselab.audio.workflows.audio_analysis.adaptive.plot import build_adaptive_timeline

        plot_path = build_adaptive_timeline(
            out_dir,
            transcript=result["transcript"],
            gt_path=args.ground_truth,
            title=args.run_dir.name,
        )
        if plot_path is not None:
            print(f"timeline: {plot_path}")
    except Exception as exc:  # noqa: BLE001 — plotting must never fail the run
        print(f"warn: timeline plot failed: {exc!r}", file=sys.stderr)

    if args.ground_truth is not None:
        eval_doc = evaluate_against_ground_truth(
            out_dir=out_dir, gt_path=args.ground_truth, word_streams=result["word_streams"]
        )
        t = eval_doc["transcript"]
        print("\n── evaluation vs ground truth ──")
        print(f"speech_presence: {json.dumps(eval_doc['speech_presence'])}")
        print(f"fused WER: {t['fused']['wer']}  (normalized: {t['fused']['wer_normalized']})")
        for m, s in (t.get("per_model") or {}).items():
            print(f"  {m}: WER {s['wer']} (normalized {s['wer_normalized']})")
        print(f"diarization: {json.dumps(eval_doc['diarization'])}")
        print(f"localization: {json.dumps(eval_doc['localization'])}")
        print(f"eval.json: {out_dir / 'final' / 'eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
