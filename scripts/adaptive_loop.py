#!/usr/bin/env python3
"""Adaptive uncertainty loop (prototype driver).

Runs the loop from ``specs/20260723-225523-dynamic-uncertainty-workflow`` over a
completed ``analyze_audio.py`` run directory: round 1 ingests the run's
uncertainty parquets into the belief store (with a re-aggregation parity
check), rounds 2..K execute the policy-ranked intervention catalog (stream
election, hallucination adjudication, missed-speech correction, cache-replay
reserve-ASR escalation; live-backend rules defer with guard reasons), and the
final round fuses a consensus transcript / diarization / presence track with a
full decision audit trail.

Usage:
    uv run python scripts/adaptive_loop.py artifacts/e2e_runs/<run_dir> \
        --cache-dir artifacts/analyze_audio_cache \
        --ground-truth ~/Downloads/updated-label-XXXX.json

Works in minimal environments (no torch): heavy senselab imports are bypassed
via submodule loading when the full package cannot import.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def _ensure_light_importable() -> None:
    """Make ``senselab.audio.workflows.audio_analysis.*`` submodules importable.

    In a full environment the normal import works and this is a no-op. In a
    minimal environment (no torch/pydantic), the package ``__init__`` chain
    would fail on heavy imports — so we register namespace stubs for the parent
    packages, letting Python load the *pure* submodules (aggregate, harvesters,
    grid, adaptive.*) directly from their file paths.
    """
    try:
        import senselab.audio.workflows.audio_analysis.aggregate  # noqa: F401

        return
    except Exception:  # noqa: BLE001 — any heavy-dep failure routes to stubs
        pass
    if str(_REPO_SRC) not in sys.path:
        sys.path.insert(0, str(_REPO_SRC))
    path = _REPO_SRC
    name = ""
    for part in ("senselab", "audio", "workflows", "audio_analysis"):
        path = path / part
        name = f"{name}.{part}" if name else part
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]  # type: ignore[attr-defined]
            mod.__package__ = name
            sys.modules[name] = mod


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

    if not (args.run_dir / "summary.json").exists():
        print(f"ERROR: {args.run_dir} is not an analyze_audio run dir (no summary.json)", file=sys.stderr)
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
    print(f"run_state: {result['run_state']}  rounds: {result['rounds']}")
    print("parity:", json.dumps(result["parity_check"]))
    print(
        f"interventions fired: {result['n_interventions_fired']}  "
        f"fused words: {result['n_words_fused']}  stream: {result['fusion_stream']}"
    )
    print(f"final/: {out_dir / 'final'}")

    # Visual timeline — best-effort sidecar (mirrors analyze_audio's timeline.png).
    try:
        from senselab.audio.workflows.audio_analysis.adaptive.plot import build_adaptive_timeline

        plot_path = build_adaptive_timeline(out_dir, gt_path=args.ground_truth, title=args.run_dir.name)
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
        print(f"presence: {json.dumps(eval_doc['presence'])}")
        print(f"fused WER: {t['fused']['wer']}  (normalized: {t['fused']['wer_normalized']})")
        for m, s in (t.get("per_model") or {}).items():
            print(f"  {m}: WER {s['wer']} (normalized {s['wer_normalized']})")
        print(f"diarization: {json.dumps(eval_doc['diarization'])}")
        print(f"localization: {json.dumps(eval_doc['localization'])}")
        print(f"eval.json: {out_dir / 'final' / 'eval.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
