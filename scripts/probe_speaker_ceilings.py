"""Generate (or reuse) a synthetic corpus, run every backend over it, and emit a speaker-ceiling profile.

Ties together the three separable pieces of the speaker-ceiling probe:
``scripts/speaker_ceiling/generate.py`` (corpus), ``scripts/speaker_ceiling/evaluate.py``
(per-backend counts), and ``scripts/speaker_ceiling/derive.py`` (curve -> ceiling). See
``specs/20260809-112417-speaker-ceiling-probe/plan.md`` for why the corpus is TTS-composed
rather than drawn from NeMo's ``MultiSpeakerSimulator`` (that class composes real single-
speaker recordings with published word alignments; it does not synthesize speech, and this
effort never costed sourcing an aligned corpus).

**This script does not update the four unmeasured ``max_speakers`` declarations.**
Emitting a profile from a smoke-test run (or from no data at all) would be worse than
leaving them honestly ``None`` -- that update is a separate, deliberate step taken only
after a real sweep on a GPU (``k`` = 1..8, 20 sessions each, all six backends).

Two refusals, both hard errors (see ``evaluate.py``):

1. any (backend, k) cell has fewer completed sessions than were required;
2. a backend produced zero successful sessions at the smallest ``k`` swept -- that
   backend's row would otherwise be measuring the harness, not the backend.

Usage::

    uv run python scripts/probe_speaker_ceilings.py \\
        --counts 1 2 3 4 5 6 7 8 --sessions 20 --out artifacts/speaker_ceiling/<run> \\
        --device cuda

    # Dry run on CPU, one backend, no GPU/venv needed:
    uv run python scripts/probe_speaker_ceilings.py \\
        --counts 1 2 --sessions 2 --out /tmp/ceiling-dry --device cpu --backends pyannote
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# scripts/ is deliberately not an importable package (pyproject sets
# pythonpath = ["src"]) -- put scripts/speaker_ceiling/ on sys.path directly rather than
# attempting a `from scripts.speaker_ceiling...` import that would raise
# ModuleNotFoundError both here and under pytest.
_SPEAKER_CEILING_DIR = Path(__file__).resolve().parent / "speaker_ceiling"
if str(_SPEAKER_CEILING_DIR) not in sys.path:
    sys.path.insert(0, str(_SPEAKER_CEILING_DIR))

from derive import DEFAULT_ACCURACY_THRESHOLD, derive_ceiling  # noqa: E402
from evaluate import (  # noqa: E402
    ALL_BACKENDS,
    BACKENDS_BY_NAME,
    InsufficientMeasurementError,
    SessionOutcome,
    check_smallest_count_has_successes,
    check_sweep_is_complete,
    confusion_from_outcomes,
    curve_from_outcomes,
    evaluate_backend,
    refusal_reasons_from_outcomes,
)
from generate import generate_corpus  # noqa: E402

from senselab.utils.data_structures import DeviceType  # noqa: E402

_CAVEAT = (
    "Measured on a TTS-composed synthetic corpus (see corpus_manifest.method and "
    "corpus_manifest.tts_model/session_params below): clean, synthetically distinct voices, "
    "no room acoustics, no channel variation, and vocoder characteristics shared across every "
    "speaker in a session. That plausibly makes counting easier than real speech (more "
    "separable identities) and could make it harder (shared synthesis artifacts). Either way "
    "this profile is an upper bound on well-conditioned audio, not a guarantee about a real "
    "recording -- see generate.py's module docstring for the full reasoning."
)


def evaluate_all(
    backend_names: Sequence[str],
    corpus_dir: Path,
    manifest: dict,
    counts: Sequence[int],
    device: Optional[DeviceType],
) -> Dict[str, Dict[int, List[SessionOutcome]]]:
    """Run every named backend over the corpus, printing progress as it goes."""
    outcomes: Dict[str, Dict[int, List[SessionOutcome]]] = {}
    for name in backend_names:
        backend = BACKENDS_BY_NAME[name]
        print(f"evaluating {name} ({backend.model_id}) over k={list(counts)} ...", flush=True)
        outcomes[name] = evaluate_backend(backend, corpus_dir, manifest, counts, device)
    return outcomes


def build_profile(
    outcomes: Mapping[str, Mapping[int, Sequence[SessionOutcome]]],
    manifest: dict,
    counts: Sequence[int],
    sessions_per_count: int,
    threshold: float,
    device_label: Optional[str],
) -> dict:
    """Reduce every backend's outcomes to a profile: confusion, curve, ceiling, and the corpus.

    The threshold and the corpus manifest are recorded once at the top level and again,
    per backend, beside the curve they were applied to -- a reader who pulls one
    backend's block out of the profile still knows exactly what judgement produced its
    ceiling and what the underlying audio was, without cross-referencing the rest of the
    document. See the module docstring's caveat: this profile is a ceiling on clean,
    synthetically distinct voices, not a guarantee about real recordings.
    """
    backends_out: Dict[str, dict] = {}
    for name, by_k in outcomes.items():
        confusion = {str(k): confusion_from_outcomes(sessions) for k, sessions in sorted(by_k.items())}
        refusal_reasons = {
            str(k): reasons
            for k, sessions in sorted(by_k.items())
            if (reasons := refusal_reasons_from_outcomes(sessions))
        }
        curve = curve_from_outcomes(by_k)
        backends_out[name] = {
            "model_id": BACKENDS_BY_NAME[name].model_id,
            "confusion": confusion,
            "refusal_reasons": refusal_reasons,
            "accuracy_curve": {str(k): v for k, v in sorted(curve.items())},
            "ceiling": derive_ceiling(curve, threshold=threshold),
            "threshold": threshold,
        }

    return {
        "generated_on": datetime.now(timezone.utc).isoformat(),
        "counts": list(counts),
        "sessions_per_count": sessions_per_count,
        "threshold": threshold,
        "device": device_label,
        "caveat": _CAVEAT,
        "corpus_manifest": manifest,
        "backends": backends_out,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. See the module docstring for usage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=int, nargs="+", required=True, help="speaker counts (k) to sweep")
    parser.add_argument("--sessions", type=int, required=True, help="sessions required per speaker count")
    parser.add_argument("--out", type=Path, required=True, help="working directory for the corpus and profile")
    parser.add_argument("--seed", type=int, default=17, help="corpus generation seed (ignored with --corpus)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="device (default: auto)")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[b.name for b in ALL_BACKENDS],
        choices=[b.name for b in ALL_BACKENDS],
        help="which backends to evaluate (default: all six)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_ACCURACY_THRESHOLD, help="ceiling-derivation accuracy threshold"
    )
    parser.add_argument(
        "--corpus", type=Path, default=None, help="reuse an existing corpus directory instead of generating one"
    )
    args = parser.parse_args(argv)

    device = DeviceType(args.device) if args.device else None
    args.out.mkdir(parents=True, exist_ok=True)

    if args.corpus is not None:
        corpus_dir = args.corpus
        print(f"reusing corpus at {corpus_dir}")
    else:
        corpus_dir = args.out / "corpus"
        print(f"generating corpus at {corpus_dir} (k={args.counts}, sessions={args.sessions}, seed={args.seed}) ...")
        generate_corpus(
            out_dir=corpus_dir,
            counts=args.counts,
            sessions_per_count=args.sessions,
            seed=args.seed,
            device=device,
        )

    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        print(
            f"ERROR: no manifest.json under {corpus_dir} -- was this corpus written by generate_corpus?",
            file=sys.stderr,
        )
        return 2
    manifest = json.loads(manifest_path.read_text())

    outcomes = evaluate_all(args.backends, corpus_dir, manifest, args.counts, device)

    try:
        check_sweep_is_complete(outcomes, args.sessions)
        check_smallest_count_has_successes(outcomes, min(args.counts))
    except InsufficientMeasurementError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    profile = build_profile(outcomes, manifest, args.counts, args.sessions, args.threshold, args.device)
    profile_path = args.out / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2))

    print(f"\nwrote profile to {profile_path}")
    for name, block in profile["backends"].items():
        print(f"  {name}: ceiling={block['ceiling']}  curve={block['accuracy_curve']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
