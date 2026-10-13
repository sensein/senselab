#!/usr/bin/env python3
"""Estimate one subject's target-speaker enrollment from that subject's own recordings.

    uv run python specs/20260817-triage-workflow-dag/runs/b2ai-v2/make_enrollment.py \
        --subject sub-17cee767-1864-457a-b2ec-446a058a81f8 \
        --dir /orcd/data/.../sub-17cee767-1864-457a-b2ec-446a058a81f8/ \
        --out enrollment/sub-17cee767.yaml

    # which files the source policy would enrol from, without loading a model
    uv run python .../make_enrollment.py --list-sources --dir .../sub-<uuid>/
    uv run python .../make_enrollment.py --selftest

A thin driver over :func:`senselab.audio.tasks.speaker_embeddings.estimate_speaker_embedding_from_audios`,
which does the actual work: it windows every file, embeds each window with ECAPA, pools them and
describes the resulting distribution. This script adds only the three things that function has no
business deciding -- which files go in, what the subject is called, and the on-disk shape
``scripts/triage_audio.py --enrollment`` reads (``Enrollment``: ``subject_id``, ``vector``,
``provenance``, and optionally ``task`` and ``distribution``).

It refuses rather than guesses:

* **The commit must agree with the campaign override.** ``SPEECH`` passes
  ``speech.enrollment_model.revision`` STRAIGHT into ``Enrollment.refusal_against`` as the probe
  commit -- it resolves nothing itself -- so an enrollment estimated at any other commit is refused
  on every file, and the run reports a refusal where a target should have been. ``--override``
  (default: the one beside this script) is read for the pinned commit, and the estimate is run at
  that commit rather than at ``main``.
* **The sources are speech files only.** A candidate is enrolled from exactly when its task token
  resolves, through :mod:`make_hints`, to a hint whose ``metadata.speech_type`` is in
  :data:`SPEECH_TYPES` -- so every airway file and every non-lexical voice task is excluded, the
  exclusions are printed by category, and a subject with no speech file is refused rather than
  enrolled from what is left. This is the policy, not an option; see README.md, "Phase 2:
  enrollment", for the ruling of 2026-08-25 and the measurements behind it.
* **Within the selected speech, the estimator still reaches no verdict about its input.** A file
  that does not contain the target speaker is not an error there, it is a row in the returned
  per-file statistics. ``--exclude`` drops a named basename, and ``--reject-contamination`` keeps
  only the dominant window group and records that in ``provenance.method``.
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
from collections import Counter
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import make_hints
import yaml  # type: ignore[import-untyped]

HERE = Path(__file__).resolve().parent
DEFAULT_OVERRIDE = HERE / "override.yaml"
_HEX = set("0123456789abcdef")

SPEECH_TYPES: frozenset[str] = frozenset({"read", "elicited", "recall"})
"""The ``metadata.speech_type`` values :data:`make_hints.RULES` gives a speech task. A recording
whose task token resolves to any other value is not an enrollment source. See README.md, "Phase 2:
enrollment", for the ruling and the measurements behind it."""

UNCLASSIFIED = "unclassified"
"""The exclusion category for a filename whose task token no hint rule claims."""


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


class Selection(NamedTuple):
    """One subject's candidate recordings, partitioned by the enrollment source policy.

    Attributes:
        kept: The speech recordings the enrollment is estimated from, sorted.
        rejected: ``(path, category)`` for every candidate the policy excluded, sorted. The category
            is ``<speech_type>/<rule name>`` from :data:`make_hints.RULES`, or :data:`UNCLASSIFIED`.
    """

    kept: list[Path]
    rejected: list[tuple[Path, str]]


def select_speech(paths: Sequence[Path]) -> Selection:
    """Partition candidate recordings into speech sources and everything else.

    The classification is :func:`make_hints.resolve` over :func:`make_hints.task_token`, so a
    recording is a source exactly when the hint it would be given carries a
    ``metadata.speech_type`` in :data:`SPEECH_TYPES`.

    Args:
        paths: The candidate recordings.

    Returns:
        The partition.
    """
    kept: list[Path] = []
    rejected: list[tuple[Path, str]] = []
    for path in sorted(set(paths)):
        try:
            rule = make_hints.resolve(make_hints.task_token(path.name))
        except ValueError:
            rejected.append((path, UNCLASSIFIED))
            continue
        if rule.speech_type in SPEECH_TYPES:
            kept.append(path)
        else:
            rejected.append((path, f"{rule.speech_type}/{rule.name}"))
    return Selection(kept, rejected)


def _by_category(selection: Selection) -> list[tuple[str, int]]:
    """How many recordings each exclusion category dropped.

    Args:
        selection: The partition.

    Returns:
        ``(category, count)``, most-dropped first, ties broken by category name.
    """
    counts = Counter(category for _, category in selection.rejected)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _sources(args: argparse.Namespace) -> Selection:
    """Every recording the enrollment is estimated from, and every candidate the policy dropped.

    Args:
        args: The parsed command line.

    Returns:
        The partition of the candidates, with ``--exclude``d basenames removed before it is made.

    Raises:
        ValueError: If no candidate was named at all, or if no candidate is a speech recording.
    """
    paths: list[Path] = [Path(name) for name in args.files]
    if args.dir is not None:
        paths += sorted(Path(args.dir).rglob("*.wav"))
    if args.list is not None:
        text = sys.stdin.read() if str(args.list) == "-" else Path(args.list).read_text()
        paths += [Path(line.strip()) for line in text.splitlines() if line.strip()]
    excluded = {name.casefold() for name in args.exclude}
    candidates = sorted({path for path in paths if path.name.casefold() not in excluded})
    if not candidates:
        raise ValueError("no recordings to enrol from: give paths, --dir or --list")
    selection = select_speech(candidates)
    if not selection.kept:
        counted = ", ".join(f"{category}: {count}" for category, count in _by_category(selection))
        raise ValueError(
            f"no speech recording to enrol from: all {len(candidates)} candidate(s) are excluded by "
            f"the source policy ({counted}). The enrollment is estimated over speech only "
            f"(speech_type {', '.join(sorted(SPEECH_TYPES))}); a subject with no speech file cannot "
            "be enrolled from this session."
        )
    return selection


def report_selection(selection: Selection) -> None:
    """Print the source selection and every exclusion it made.

    Args:
        selection: The partition.
    """
    total = len(selection.kept) + len(selection.rejected)
    print(f"sources      {len(selection.kept)} speech file(s) of {total} candidate(s)")
    for path in selection.kept:
        print(f"  + {path.name}")
    if selection.rejected:
        counted = ", ".join(f"{category}: {count}" for category, count in _by_category(selection))
        print(f"excluded     {len(selection.rejected)} non-speech file(s) -- {counted}")
        for path, category in selection.rejected:
            print(f"  - {path.name}  [{category}]")


def estimate(args: argparse.Namespace, sources: Sequence[Path], model_id: str, commit: str) -> dict[str, object]:
    """Run the estimator and shape its result as an ``Enrollment`` mapping.

    Args:
        args: The parsed command line.
        sources: The speech recordings :func:`_sources` selected.
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


SELECTION_CASES: tuple[tuple[str, bool], ...] = (
    # Speech: the tasks whose hint speech_type is read, elicited or recall.
    ("Rainbow-Passage", True),
    ("Caterpillar-Passage", True),
    ("Reading-1", True),
    ("Harvard-Sentences-List-49-1", True),
    ("Cape-V-sentences-(v2)-3", True),
    ("Free-speech-1", True),
    ("Free-speech-(v2)-2", True),
    ("Picture-description", True),
    ("Picture-description-option2", True),
    ("Productive-Vocabulary-1", True),
    ("Random-Item-Generation-1", True),
    ("Word-color-Stroop-1", True),
    ("Story-recall", True),
    ("Story-recall-(v2)", True),
    ("Cinderella-Story", True),
    # Airway.
    ("Respiration-and-cough-Cough-1", False),
    ("Respiration-and-cough-(v2)-HardCough", False),
    ("Respiration-and-cough-Breath-1", False),
    ("Respiration-and-cough-FiveBreaths-3", False),
    ("Respiration-and-cough-(v2)-ThreeBreathsNose", False),
    # Non-lexical voice.
    ("Prolonged-vowel", False),
    ("Maximum-phonation-time-1", False),
    ("Maximum-phonation-time-(v2)-1", False),
    ("Glides-High-to-Low", False),
    ("Glides-Low-to-High", False),
    ("Loudness", False),
    ("Loudness-(v2)", False),
    ("Diadochokinesis-KA", False),
    ("Diadochokinesis-(v2)-puhtuhkuh", False),
    # Both limbs of the one token whose may_contain carries `speech` while its speech_type is
    # non-lexical: the speech_type governs the source policy, so it is excluded.
    ("Diadochokinesis-buttercup", False),
    ("Diadochokinesis-(v2)-buttercup", False),
)
"""Task token to whether the enrollment source policy selects it."""

B2AI_28_SPEECH = 6
V2_47_SPEECH = 32
"""How many of the two pinned task inventories in ``make_hints`` the policy selects."""


def _named(token: str) -> str:
    """A b2ai-shaped filename carrying one task token.

    Args:
        token: The task token.

    Returns:
        The filename.
    """
    return f"{make_hints.B2AI_28_SUBJECT}_{make_hints.B2AI_28_SESSION}_task-{token}.wav"


def selftest() -> int:
    """Check the enrollment source policy against the hint rules it reuses.

    Returns:
        0 when every case holds, 1 otherwise.
    """
    failures: list[str] = []
    for token, expected in SELECTION_CASES:
        selection = select_speech([Path(_named(token))])
        got = bool(selection.kept)
        if got is not expected:
            reason = selection.rejected[0][1] if selection.rejected else "selected"
            failures.append(f"  {token}: selected={got}, expected {expected} ({reason})")
    for inventory, expected_count, label in (
        (make_hints.B2AI_28, B2AI_28_SPEECH, "B2AI_28"),
        (make_hints.V2_47, V2_47_SPEECH, "V2_47"),
    ):
        selection = select_speech([Path(_named(token)) for token in inventory])
        if len(selection.kept) != expected_count:
            failures.append(f"  {label}: selected {len(selection.kept)} of {len(inventory)}, expected {expected_count}")
        if len(selection.kept) + len(selection.rejected) != len(inventory):
            failures.append(f"  {label}: {len(inventory)} tokens partitioned into a different total")
        for token in inventory:
            rule = make_hints.resolve(token)
            wanted = rule.speech_type in SPEECH_TYPES
            got = Path(_named(token)) in selection.kept
            if got is not wanted:
                failures.append(f"  {label}/{token}: speech_type {rule.speech_type!r} but selected={got}")
    unreadable = select_speech([Path("not-a-b2ai-name.wav")])
    if unreadable.kept or not unreadable.rejected:
        failures.append("  a filename carrying no 'task-' element must be excluded, not enrolled from")
    airway_only = [_named("Respiration-and-cough-Cough-1"), _named("Prolonged-vowel")]
    try:
        _sources(build_parser().parse_args(airway_only))
    except ValueError as error:
        if "speech" not in str(error):
            failures.append(f"  zero-speech refusal raised the wrong message: {error}")
    else:
        failures.append("  a subject with no speech file must be refused, not enrolled from non-speech material")
    speech_and_airway = [_named("Free-speech-1"), _named("Respiration-and-cough-Cough-1")]
    survivors = _sources(build_parser().parse_args(speech_and_airway))
    if [path.name for path in survivors.kept] != [_named("Free-speech-1")]:
        failures.append(f"  a mixed directory selected {[path.name for path in survivors.kept]}")
    if failures:
        print(f"selftest FAILED ({len(failures)}):", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(
        f"selftest ok: {len(SELECTION_CASES)} task tokens partitioned as ruled; "
        f"{B2AI_28_SPEECH}/{len(make_hints.B2AI_28)} and {V2_47_SPEECH}/{len(make_hints.V2_47)} selected, "
        "each agreeing with make_hints' speech_type; an unreadable name is excluded; "
        "a subject with no speech file is refused"
    )
    return 0


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
    parser.add_argument("--subject", default=None, help="Subject id written into the enrollment.")
    parser.add_argument("--dir", type=Path, default=None, help="Directory to scan recursively for *.wav.")
    parser.add_argument("--list", default=None, help="File of one path per line, or '-' for stdin.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Basenames to leave out of the estimate.")
    parser.add_argument("--out", type=Path, default=None, help="Where the enrollment YAML is written.")
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
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print the source selection and exit, without loading a model or embedding anything.",
    )
    parser.add_argument("--selftest", action="store_true", help="Check the source policy against the hint rules.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Estimate and write one subject's enrollment.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 on success, 2 when the arguments or the estimate could not be resolved.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.selftest:
        return selftest()
    if not args.list_sources and (args.subject is None or args.out is None):
        print("ERROR: --subject and --out are required unless --selftest or --list-sources", file=sys.stderr)
        return 2
    try:
        selection = _sources(args)
        report_selection(selection)
        if args.list_sources:
            return 0
        model_id, commit = pinned_model(args.override)
        payload = estimate(args, selection.kept, model_id, commit)
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
