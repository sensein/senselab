"""Generate a synthetic corpus once, evaluate every backend against it, and emit a profile.

Ties together the three separable pieces of the speaker-ceiling probe:
``scripts/speaker_ceiling/generate.py`` (corpus), ``scripts/speaker_ceiling/evaluate.py``
(per-backend counts), ``scripts/speaker_ceiling/aggregate.py`` (assembling checkpoints into
a profile), and ``scripts/speaker_ceiling/derive.py`` (curve -> ceiling). See
``specs/20260809-112417-speaker-ceiling-probe/plan.md`` for why the corpus is TTS-composed
rather than drawn from NeMo's ``MultiSpeakerSimulator`` (that class composes real single-
speaker recordings with published word alignments; it does not synthesize speech, and this
effort never costed sourcing an aligned corpus).

**This script does not update the four unmeasured ``max_speakers`` declarations.**
Emitting a profile from a smoke-test run (or from no data at all) would be worse than
leaving them honestly ``None`` -- that update is a separate, deliberate step taken only
after a real sweep on a GPU (``k`` = 1..8, 20 sessions each, all six backends).

Two phases, not one
--------------------
Generation and evaluation are **separate CLI modes that must run as separate invocations**,
not one combined "generate-then-evaluate" call. This is deliberate, for three reasons, in
order of importance:

1. **Cross-backend comparison becomes exact.** Every backend must be scored against
   byte-identical audio, or a difference between two backends' ceilings could just be a
   difference in what corpus each one happened to generate for itself. Phase separation
   guarantees this; letting each evaluation task generate its own corpus only makes it
   likely, not certain.
2. **A requeue cannot change the data.** On a preemptable partition a killed generation task
   restarts. If evaluation shared that task, a requeue could evaluate against subtly
   different audio than the run it replaced -- the corpus must be a finished, durable
   artifact *before* any evaluation touches it.
3. **Re-runs are free.** Adding a backend, changing ``--threshold``, or recomputing the
   curve costs no GPU time once the corpus exists -- which matters because the 0.8 threshold
   is explicitly a judgement someone may want to revisit against the same numbers.

Three modes (``--mode {generate,evaluate,aggregate}``)
--------------------------------------------------------
**generate**: writes a durable corpus under ``--corpus`` (defaults to
``artifacts/speaker_ceiling/corpus/seed-<seed>/`` -- not a temp dir, so it survives the run
and can be re-evaluated for free). Can be sharded by *k* with ``--shard-k`` (or
``SLURM_ARRAY_TASK_ID``) for speed, since each *k* writes its own disjoint ``k=<k>/``
subdirectory; the array's tasks never touch each other's output. This is the only mode that
writes to ``--corpus``, and the only mode that costs GPU-hours proportional to audio length.

**evaluate**: *requires* ``--corpus`` -- refuses outright without it, so no evaluation run
can silently generate its own audio (reason 1 above). Validates the corpus before touching
a single backend (see ``aggregate.validate_corpus_for_counts``): manifest present and
parseable, every requested *k*'s session wav/rttm files present and readable, and every wav
at :data:`generate.CORPUS_SAMPLE_RATE`. Can be sharded by backend (``--backends``) and/or by
*k* (``--shard-k``) -- fan-out here is cheap, since diarizing is fast next to generation.
Every completed (backend, *k*) cell is checkpointed to ``<out>/cells/`` immediately (see
``evaluate.write_cell``), carrying the corpus's identity (seed, resolved TTS commit) so a
later aggregation can catch two cells that were quietly evaluated against different audio.

**aggregate**: does no generation or evaluation. Reads every required cell checkpoint and
the corpus manifest, refuses on a missing shard, a corpus-identity mismatch across cells
(``aggregate.CorpusMismatchError``), or either of the two original hard refusals below, then
writes ``profile.json``. Run once, by hand, after every evaluate task has reported in.

Refusals, all hard errors:

1. any (backend, k) cell has fewer completed sessions than were required (``evaluate.py``);
2. a backend produced zero successful sessions at the smallest ``k`` swept -- that
   backend's row would otherwise be measuring the harness, not the backend (``evaluate.py``);
3. (``--mode aggregate``) a cell checkpoint or a shard's corpus manifest is absent -- a task
   that never finished, never ran, or died before writing anything (``aggregate.py``);
4. (``--mode aggregate``) two cells (or a cell and the corpus manifest) disagree on which
   corpus they came from (``aggregate.py``) -- see reason 1 above;
5. (``--mode evaluate``) the corpus at ``--corpus`` does not actually contain what its
   manifest claims for the requested counts (``aggregate.py``).

Usage::

    # Phase 1: generate the whole corpus (or shard it by k for speed -- see sweep_generate.sbatch):
    uv run python scripts/probe_speaker_ceilings.py --mode generate \\
        --counts 1 2 3 4 5 6 7 8 --sessions 20 --seed 17 --device cuda

    # One generation shard, run by hand (an array task relies on SLURM_ARRAY_TASK_ID instead):
    uv run python scripts/probe_speaker_ceilings.py --mode generate \\
        --counts 1 2 3 4 5 6 7 8 --sessions 20 --seed 17 --device cuda --shard-k 3

    # Phase 2: evaluate every backend against that corpus (--corpus is mandatory here):
    uv run python scripts/probe_speaker_ceilings.py --mode evaluate \\
        --counts 1 2 3 4 5 6 7 8 --sessions 20 --out artifacts/speaker_ceiling/<run> \\
        --corpus artifacts/speaker_ceiling/corpus/seed-17 --device cuda

    # Dry run on CPU, one backend, no GPU/venv needed (still two calls, generate then evaluate):
    uv run python scripts/probe_speaker_ceilings.py --mode generate \\
        --counts 1 2 --sessions 2 --seed 17 --corpus /tmp/ceiling-dry/corpus --device cpu
    uv run python scripts/probe_speaker_ceilings.py --mode evaluate \\
        --counts 1 2 --sessions 2 --out /tmp/ceiling-dry --corpus /tmp/ceiling-dry/corpus \\
        --device cpu --backends pyannote

    # Aggregate once every evaluate task has reported in:
    uv run python scripts/probe_speaker_ceilings.py --mode aggregate \\
        --counts 1 2 3 4 5 6 7 8 --sessions 20 --out artifacts/speaker_ceiling/<run> \\
        --corpus artifacts/speaker_ceiling/corpus/seed-17
"""

from __future__ import annotations

import argparse
import json
import os
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

import aggregate  # noqa: E402
from derive import (  # noqa: E402
    DEFAULT_ACCURACY_THRESHOLD,
    derive_ceiling,
    derive_structural_bound,
    format_structural_bound_evidence,
)
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
    cells_dir: Optional[Path] = None,
) -> Dict[str, Dict[int, List[SessionOutcome]]]:
    """Run every named backend over the corpus, printing progress as it goes.

    ``cells_dir``, when given, is threaded straight through to :func:`evaluate_backend` so
    every evaluate-mode call checkpoints per (backend, k) cell -- whether it is sharded to
    one backend/k or covers everything requested.
    """
    outcomes: Dict[str, Dict[int, List[SessionOutcome]]] = {}
    for name in backend_names:
        backend = BACKENDS_BY_NAME[name]
        print(f"evaluating {name} ({backend.model_id}) over k={list(counts)} ...", flush=True)
        outcomes[name] = evaluate_backend(backend, corpus_dir, manifest, counts, device, cells_dir=cells_dir)
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

    Alongside `ceiling` (an accuracy-based verdict from `derive.derive_ceiling`), every
    backend also gets `structural_bound` and `structural_bound_evidence` -- a structural
    verdict from `derive.derive_structural_bound`, applied to the confusion at the *largest*
    k in `counts` (see that function's docstring for why only the top of the sweep is
    trustworthy evidence of a plateau). The two are independent: a backend can have a low
    accuracy `ceiling` and no `structural_bound` at all (still trying and failing to track a
    high true count), or a hard `structural_bound` at a count its accuracy curve alone would
    not have flagged as a limit -- see `DiarizationCapabilities.max_speakers`'s docstring for
    why folding both into one field was the mistake this profile's schema now avoids.
    """
    seed = manifest.get("seed")
    probe_label = f"probe seed-{seed}" if seed is not None else "probe (seed unknown)"
    max_k = max(counts)

    backends_out: Dict[str, dict] = {}
    for name, by_k in outcomes.items():
        confusion = {str(k): confusion_from_outcomes(sessions) for k, sessions in sorted(by_k.items())}
        refusal_reasons = {
            str(k): reasons
            for k, sessions in sorted(by_k.items())
            if (reasons := refusal_reasons_from_outcomes(sessions))
        }
        curve = curve_from_outcomes(by_k)

        # The structural-bound rule needs the confusion at the top of *this backend's own*
        # sweep, which is `max_k` whenever every requested count was actually evaluated for
        # it -- true for every real run, but a defensively-missing cell (e.g. a hand-built
        # test fixture populating only some k) falls back to "unmeasured" rather than raising
        # a KeyError.
        max_k_confusion = confusion.get(str(max_k))
        if max_k_confusion is not None:
            structural_bound = derive_structural_bound(max_k_confusion, true_k=max_k)
            structural_bound_evidence = format_structural_bound_evidence(
                max_k_confusion, true_k=max_k, probe_label=probe_label
            )
        else:
            structural_bound = None
            structural_bound_evidence = "unmeasured"

        backends_out[name] = {
            "model_id": BACKENDS_BY_NAME[name].model_id,
            "confusion": confusion,
            "refusal_reasons": refusal_reasons,
            "accuracy_curve": {str(k): v for k, v in sorted(curve.items())},
            "ceiling": derive_ceiling(curve, threshold=threshold),
            "threshold": threshold,
            "structural_bound": structural_bound,
            "structural_bound_evidence": structural_bound_evidence,
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


def _resolve_shard_k(explicit: Optional[int], counts: Sequence[int]) -> Optional[int]:
    """Return the single k this process should handle, or None to cover every requested k.

    **Explicit only. ``SLURM_ARRAY_TASK_ID`` is deliberately NOT consulted**, and that is a
    correction rather than an omission. An earlier version fell back to it so ``--array=1-8``
    could map task ids straight onto k values, which is convenient exactly as long as every
    array is sharded by k. ``evaluate.sbatch`` is sharded by *backend* (``--array=0-5``), so on
    the real sweep task 0 resolved to ``--shard-k 0`` and died, while tasks 1-5 silently
    evaluated ``(backend_i, k=i)`` -- a diagonal, not a sweep -- skipping k=6,7,8 entirely and
    reporting COMPLETED. An implicit environment read that changes what an array index *means*
    is not worth the lookup table it saves, so callers now pass ``--shard-k`` themselves.

    Used by ``--mode generate`` (which k to generate) and ``--mode evaluate`` (which k to
    evaluate); never by ``--mode aggregate``, which always covers every requested k at once.

    Raises:
        ValueError: if the resolved shard k is not among the requested ``counts``.
    """
    if explicit is None:
        return None
    shard_k = explicit
    if shard_k not in counts:
        raise ValueError(f"--shard-k {shard_k} is not among the requested --counts {list(counts)}")
    return shard_k


def _default_corpus_dir(seed: int) -> Path:
    """Return the durable default corpus location for a generation run that omits ``--corpus``.

    Under ``artifacts/``, not a temp directory: Phase 2 (and any later re-evaluation,
    threshold change, or added backend) must be able to point ``--corpus`` back at exactly
    this and reuse it for free (see the module docstring's "Re-runs are free"). Keyed by
    seed so two sweeps with different seeds do not collide.
    """
    return Path("artifacts") / "speaker_ceiling" / "corpus" / f"seed-{seed}"


def _run_generate(args: argparse.Namespace, device: Optional[DeviceType]) -> int:
    """Phase 1: write a durable corpus, optionally sharded to one k for array parallelism.

    The only mode that ever writes under ``--corpus``. A sharded call
    (``--shard-k``/``SLURM_ARRAY_TASK_ID``) writes only that k's ``k=<k>/`` subdirectory and
    a same-named manifest fragment (``manifest.k<k>.json``) -- disjoint from every other
    shard's output, so concurrent array tasks never clobber each other (see
    ``generate.generate_corpus``'s ``manifest_name`` parameter).
    """
    try:
        shard_k = _resolve_shard_k(args.shard_k, args.counts)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    corpus_dir = args.corpus if args.corpus is not None else _default_corpus_dir(args.seed)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if shard_k is not None:
        counts, manifest_name, label = [shard_k], f"manifest.k{shard_k}.json", f"k={shard_k}"
    else:
        counts, manifest_name, label = list(args.counts), "manifest.json", f"k={list(args.counts)}"

    print(f"[generate {label}] writing corpus at {corpus_dir} (sessions={args.sessions}, seed={args.seed}) ...")
    generate_corpus(
        out_dir=corpus_dir,
        counts=counts,
        sessions_per_count=args.sessions,
        seed=args.seed,
        device=device,
        manifest_name=manifest_name,
    )
    print(f"[generate {label}] wrote {manifest_name} under {corpus_dir}")
    return 0


def _run_evaluate(args: argparse.Namespace, device: Optional[DeviceType], cells_dir: Path) -> int:
    """Phase 2: validate a durable corpus, then evaluate every requested backend against it.

    Refuses outright without ``--corpus`` -- no evaluation call may silently generate its
    own audio (see the module docstring's reason 1). Never writes ``profile.json``: even a
    call that covers every backend and every k does not know whether *this* was the only
    evaluate call for the sweep, so deriving a profile is left to ``--mode aggregate``,
    which reads back whatever every evaluate call actually wrote.
    """
    if args.corpus is None:
        print(
            "ERROR: --mode evaluate requires --corpus -- Phase 2 never generates its own audio. "
            "Point it at a corpus already written by --mode generate.",
            file=sys.stderr,
        )
        return 2
    corpus_dir = args.corpus

    try:
        shard_k = _resolve_shard_k(args.shard_k, args.counts)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    counts = [shard_k] if shard_k is not None else list(args.counts)
    label = f"k={shard_k}" if shard_k is not None else f"k={counts}"

    try:
        manifest = aggregate.merge_corpus_manifests(corpus_dir, counts)
        aggregate.validate_corpus_for_counts(corpus_dir, manifest, counts)
    except (aggregate.MissingShardError, aggregate.CorpusValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        evaluate_all(args.backends, corpus_dir, manifest, counts, device, cells_dir=cells_dir)
    except ValueError as exc:
        # Raised by evaluate_backend when a cached cell under cells_dir was computed
        # against a different corpus than `manifest` describes (see
        # evaluate.corpus_identity_from_manifest) -- a stale --out reused with a new
        # --corpus must refuse rather than silently mixing measurements from two corpora.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"[evaluate {label}] done -- cell checkpoints under {cells_dir}")
    return 0


def _run_aggregate(args: argparse.Namespace, cells_dir: Path) -> int:
    """Read every cell checkpoint and the corpus manifest, then write profile.json.

    Does no generation or evaluation. Refuses, in order: a missing shard (a checkpoint that
    never got written), a corpus-identity mismatch across cells, then both of the original
    ``evaluate.py`` refusals -- the same standard, and the same posture of refusing rather
    than emitting a weak profile, as ``scripts/calibrate_detection_margin.py``.
    """
    if args.corpus is None:
        print(
            "ERROR: --mode aggregate requires --corpus, to merge the corpus manifest and check every "
            "cell was evaluated against it.",
            file=sys.stderr,
        )
        return 2

    try:
        outcomes = aggregate.load_cells(cells_dir, args.backends, args.counts)
        manifest = aggregate.merge_corpus_manifests(args.corpus, args.counts)
        aggregate.check_cells_share_one_corpus(cells_dir, args.backends, args.counts, manifest)
    except (aggregate.MissingShardError, aggregate.CorpusMismatchError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        check_sweep_is_complete(outcomes, args.sessions, dump_dir=args.out)
        check_smallest_count_has_successes(outcomes, min(args.counts), dump_dir=args.out)
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. See the module docstring for the three modes and usage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["generate", "evaluate", "aggregate"],
        help="which phase to run -- see the module docstring's 'Three modes' section",
    )
    parser.add_argument("--counts", type=int, nargs="+", required=True, help="speaker counts (k) to sweep")
    parser.add_argument("--sessions", type=int, required=True, help="sessions required per speaker count")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="working directory for cell checkpoints and profile.json (required for --mode evaluate/aggregate)",
    )
    parser.add_argument("--seed", type=int, default=17, help="corpus generation seed (--mode generate only)")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"], help="device (default: auto)")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=[b.name for b in ALL_BACKENDS],
        choices=[b.name for b in ALL_BACKENDS],
        help="which backends to evaluate (default: all six; --mode evaluate/aggregate only)",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_ACCURACY_THRESHOLD, help="ceiling-derivation accuracy threshold"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "corpus directory: written by --mode generate (defaults under artifacts/ when omitted "
            "there); required (no default) for --mode evaluate/aggregate"
        ),
    )
    parser.add_argument(
        "--shard-k",
        type=int,
        default=None,
        help=(
            "restrict this task to one k (else SLURM_ARRAY_TASK_ID if set): which k to generate "
            "(--mode generate) or evaluate (--mode evaluate). Not used by --mode aggregate."
        ),
    )
    args = parser.parse_args(argv)

    device = DeviceType(args.device) if args.device else None

    if args.mode in ("evaluate", "aggregate") and args.out is None:
        print(f"ERROR: --out is required for --mode {args.mode}", file=sys.stderr)
        return 2

    if args.mode == "generate":
        return _run_generate(args, device)

    args.out.mkdir(parents=True, exist_ok=True)
    cells_dir = args.out / "cells"

    if args.mode == "evaluate":
        return _run_evaluate(args, device, cells_dir)
    return _run_aggregate(args, cells_dir)


if __name__ == "__main__":
    raise SystemExit(main())
