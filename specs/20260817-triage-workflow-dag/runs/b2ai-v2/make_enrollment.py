#!/usr/bin/env python3
"""Estimate one subject's target-speaker enrollment from that subject's own recordings.

    uv run python specs/20260817-triage-workflow-dag/runs/b2ai-v2/make_enrollment.py \
        --subject sub-17cee767-1864-457a-b2ec-446a058a81f8 \
        --dir /orcd/data/.../sub-17cee767-1864-457a-b2ec-446a058a81f8/ \
        --out enrollment/sub-17cee767.yaml

A thin driver over :func:`senselab.audio.tasks.speaker_embeddings.estimate_speaker_embedding_from_audios`,
which does the actual work: it windows every file, embeds each window with ECAPA, pools them and
describes the resulting distribution. This script adds only the three things that function has no
business deciding -- which files go in, what the subject is called, and the on-disk shape
``scripts/triage_audio.py --enrollment`` reads (``Enrollment``: ``subject_id``, ``vector``,
``provenance``, and optionally ``task`` and ``distribution``).

It refuses rather than guesses in three places, each of which produced a silently wrong enrollment
when it was left to a default:

* **The commit must agree with the campaign override.** ``SPEECH`` passes
  ``speech.enrollment_model.revision`` STRAIGHT into ``Enrollment.refusal_against`` as the probe
  commit -- it resolves nothing itself -- so an enrollment estimated at any other commit is refused
  on every file, and the run reports a refusal where a target should have been. ``--override``
  (default: the one beside this script) is read for the pinned commit, and the estimate is run at
  that commit rather than at ``main``.
* **The estimate must be told which files to trust.** This function reaches no verdict about
  whether its input was clean; a file that does not contain the target speaker is not an error, it
  is a row in the returned per-file statistics. ``--exclude`` and the printed per-file report are
  how a caller curates. Nothing here rejects contamination by default -- selecting a dominant group
  is a decision, and it is made with ``--reject-contamination`` and recorded in the provenance.
* **``created_at`` is not "now" unless asked.** A stamp nobody chose makes the output
  unreproducible; ``--created-at`` supplies one deliberately.

This is PHASE 2 of the campaign. Phase 1 runs unenrolled -- ``speech.target_match_cosine`` is only
read when an enrollment is supplied -- and its per-file diarization is what tells you whether a
subject's files are worth enrolling from at all. See README.md, "Phase 2: enrollment".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import yaml  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent
DEFAULT_OVERRIDE = HERE / "override.yaml"
_HEX = set("0123456789abcdef")


def pinned_model(override: Path) -> tuple[str, str]:
    """The model id and the pinned 40-hex commit the campaign's override declares.

    Args:
        override: The campaign override YAML.

    Returns:
        ``(model_id, commit_sha)``.

    Raises:
        ValueError: If the key is absent, if either half is missing, or if the revision is not a
            resolved 40-hex commit. A ref here is the failure this whole function exists to stop:
            ``refusal_against`` compares it for equality against the enrollment's resolved sha, so
            ``main`` refuses every enrollment while looking like a configuration.
    """
    values = yaml.safe_load(override.read_text()) or {}
    spec = (values.get("speech") or {}).get("enrollment_model")
    if not isinstance(spec, dict):
        raise ValueError(f"{override} carries no speech.enrollment_model mapping")
    model_id, revision = spec.get("model_id"), str(spec.get("revision") or "")
    if not model_id:
        raise ValueError(f"{override}: speech.enrollment_model.model_id is empty")
    if len(revision) != 40 or not set(revision.casefold()) <= _HEX:
        raise ValueError(
            f"{override}: speech.enrollment_model.revision is {revision!r}, not a resolved 40-hex "
            "commit. SPEECH compares this string for equality against the enrollment's commit, so a "
            "ref refuses every enrollment. Resolve it first: "
            '`python -c "from senselab.utils.model_revision import resolve_revision; '
            "print(resolve_revision('<model_id>', 'main'))\"` and pin the result in the override."
        )
    return str(model_id), revision.casefold()


def _sources(args: argparse.Namespace) -> list[Path]:
    """Every recording the enrollment is estimated from.

    Args:
        args: The parsed command line.

    Returns:
        The paths, sorted, with excluded basenames removed.

    Raises:
        ValueError: If no recording survives.
    """
    paths: list[Path] = [Path(name) for name in args.files]
    if args.dir is not None:
        paths += sorted(Path(args.dir).rglob("*.wav"))
    if args.list is not None:
        text = sys.stdin.read() if str(args.list) == "-" else Path(args.list).read_text()
        paths += [Path(line.strip()) for line in text.splitlines() if line.strip()]
    excluded = {name.casefold() for name in args.exclude}
    kept = sorted({path for path in paths if path.name.casefold() not in excluded})
    if not kept:
        raise ValueError("no recordings to enrol from: give paths, --dir or --list")
    return kept


def estimate(args: argparse.Namespace, model_id: str, commit: str) -> dict[str, object]:
    """Run the estimator and shape its result as an ``Enrollment`` mapping.

    Args:
        args: The parsed command line.
        model_id: The embedding model, from the override.
        commit: The pinned 40-hex commit, from the override.

    Returns:
        The mapping to write.

    Raises:
        ValueError: If the estimated commit does not match the pinned one -- which would be
            refused on every file at run time, silently, as a "not comparable" flag.
    """
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.preprocessing import resample_audios
    from senselab.audio.tasks.speaker_embeddings import estimate_speaker_embedding_from_audios
    from senselab.utils.data_structures import SpeechBrainModel

    sources = _sources(args)
    audios = resample_audios([Audio(filepath=str(path)) for path in sources], args.sample_rate)
    estimated = estimate_speaker_embedding_from_audios(
        audios,
        model=SpeechBrainModel(path_or_uri=model_id, revision=commit),
        window_s=args.window_s,
        hop_s=args.hop_s,
        aggregator=args.aggregator,
        reject_contamination=args.reject_contamination,
        created_at=args.created_at,
        file_ids=[path.name for path in sources],
    )
    provenance = estimated.provenance
    if (provenance.model_commit_sha or "").casefold() != commit:
        raise ValueError(
            f"the estimate resolved to commit {provenance.model_commit_sha} but the override pins "
            f"{commit}; SPEECH would refuse this enrollment on every file rather than compare it"
        )
    payload: dict[str, object] = {
        "subject_id": args.subject,
        "vector": [float(component) for component in estimated.vector],
        "provenance": provenance.model_dump(mode="json"),
    }
    if args.task is not None:
        payload["task"] = args.task
    if estimated.distribution is not None and not args.drop_distribution:
        payload["distribution"] = estimated.distribution.model_dump(mode="json")
    return payload


def report(payload: dict[str, object]) -> None:
    """Print what a caller needs to judge whether the estimate is well supported.

    Args:
        payload: The mapping about to be written.
    """
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    print(f"subject      {payload['subject_id']}")
    print(f"model        {provenance['model_id']}@{provenance['model_commit_sha']}")
    print(f"method       {provenance['method']}")
    print(f"files        {len(provenance['source_files'])}")
    print(f"windows      {provenance['n_windows_used']} used, {provenance['n_windows_dropped']} dropped")
    failures = provenance.get("extraction_failures") or {}
    if failures:
        print(f"FAILURES     {len(failures)} file(s) produced no window:")
        for file_id, reason in failures.items():
            print(f"  {file_id}: {reason}")
    distribution = payload.get("distribution")
    if isinstance(distribution, dict):
        print("distribution: read leave_one_file_out_cos and cross_file before trusting this vector.")


def build_parser() -> argparse.ArgumentParser:
    """The CLI.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Recordings to enrol from.")
    parser.add_argument("--subject", required=True, help="Subject id written into the enrollment.")
    parser.add_argument("--dir", type=Path, default=None, help="Directory to scan recursively for *.wav.")
    parser.add_argument("--list", default=None, help="File of one path per line, or '-' for stdin.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Basenames to leave out of the estimate.")
    parser.add_argument("--out", type=Path, required=True, help="Where the enrollment YAML is written.")
    parser.add_argument("--override", type=Path, default=DEFAULT_OVERRIDE, help="Campaign override (for the pin).")
    parser.add_argument("--task", default=None, help="The vocal task the enrollment was estimated over, if one.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Resample target (default: 16000).")
    parser.add_argument("--window-s", type=float, default=2.0, help="Embedding window (default: the profile 2.0 s).")
    parser.add_argument("--hop-s", type=float, default=1.0, help="Hop between windows (default: 1.0 s).")
    parser.add_argument(
        "--aggregator",
        default="spherical_mean",
        choices=["spherical_mean", "trimmed_mean", "medoid"],
        help="How the windows are pooled (default: spherical_mean).",
    )
    parser.add_argument(
        "--reject-contamination",
        action="store_true",
        help="Keep only the dominant window group. A DECISION; it is recorded in provenance.method.",
    )
    parser.add_argument("--created-at", default=None, help="ISO-8601 stamp. Omitted rather than defaulted to now.")
    parser.add_argument("--drop-distribution", action="store_true", help="Omit the distribution block from the file.")
    parser.add_argument("--json", action="store_true", help="Write JSON instead of YAML; the CLI reads either.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Estimate and write one subject's enrollment.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 on success, 2 when the arguments or the estimate could not be resolved.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        model_id, commit = pinned_model(args.override)
        payload = estimate(args, model_id, commit)
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1) if args.json else yaml.safe_dump(payload, sort_keys=False)
    args.out.write_text(text + ("\n" if not text.endswith("\n") else ""))
    report(payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
