#!/usr/bin/env python3
r"""Triage one recording through the senselab triage graph.

    uv run python scripts/triage_audio.py recording.wav [--out DIR] [--config OVERRIDE.yaml] \
        [--hint HINT.yaml] [--enrollment ENROLLMENT.yaml]

Two arguments, and three optional files. There are deliberately no per-knob flags: every number the
graph uses lives in one versioned file with its derivation written beside it,

    src/senselab/audio/workflows/triage/data/config/default.yaml

and ``--config`` deep-merges a partial YAML over it, so an override is a named, hashable object that
travels into every artifact's provenance rather than a shell line nobody kept. ``--hint`` is what the
recording was *declared* to contain — an assertion by an operator or a protocol, never a measurement;
it maps onto ``senselab.audio.data_structures.AudioHints``:

    may_contain: [cough, read-speech]
    targeted_speaker_count: 1
    environment: quiet-room
    expected_speech:
      - text: "The quick brown fox"
        prompt_id: harvard-01

``--enrollment`` is the target speaker's vector, estimated across that subject's recordings by
whatever produced it; SPEECH identifies the target by this and by nothing in the hint. It maps onto
``senselab.audio.workflows.triage.enrollment.Enrollment``, and YAML or JSON both read:

    subject_id: sub-01
    task: sustained-vowel
    vector: [0.021, -0.114, ...]
    provenance:
      model_id: speechbrain/spkrec-ecapa-voxceleb
      model_commit_sha: <the resolved 40-hex commit, never a ref>
      source_files: [sub-01_ses-1_task-vowel.wav, sub-01_ses-2_task-vowel.wav]

Layout under ``--out/<stem>_<utc-timestamp>/``:

    run/store.jsonl        the append-only provenance store: every node's measurements and verdicts
    run/streams/           the conditioned streams
    run/derivatives/       the sidecars measurements point at
    run/run.json           the runner's own record: per-node run state, and any node that raised
    summary/              REPORT's two products, on every file and every outcome: one page a reviewer
                          reads and one JSON a consumer does. A sibling of ``run/`` and never inside
                          ``released/``: both carry element ids, so both inherit the store's
                          sensitivity
    released/             REDACT's released pair, when it cleared one — a sibling of ``run/``, never
                          inside it, so the store and the release directory cannot be swept by one
                          publish step

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import yaml  # type: ignore[import-untyped]

from senselab.audio.data_structures import AudioHints, SpeakerEmbeddingProvenance
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.enrollment import Enrollment
from senselab.audio.workflows.triage.run import run_triage
from senselab.audio.workflows.triage.vocabulary import RunState

DEFAULT_OUT_DIR = Path("artifacts/triage")


def build_parser() -> argparse.ArgumentParser:
    """The CLI: a recording, where the run goes, and optionally a config override and a hint.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every other value lives in the triage config:\n"
            "  src/senselab/audio/workflows/triage/data/config/default.yaml\n"
            "Override with a YAML holding only the keys you are changing:\n"
            "  uv run python scripts/triage_audio.py recording.wav --config my.yaml"
        ),
    )
    parser.add_argument("audio", type=Path, help="Input audio file (.wav, .flac, .mp3, ...)")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory the run root is created in (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Triage config YAML, deep-merged over the packaged default. Name only the keys you change.",
    )
    parser.add_argument(
        "--hint",
        type=Path,
        default=None,
        help="AudioHints YAML: what the recording was declared to contain. An assertion, not a measurement.",
    )
    parser.add_argument(
        "--enrollment",
        type=Path,
        default=None,
        help="Enrollment YAML or JSON: one subject's target-speaker vector, with the model and the "
        "resolved commit it was estimated at. SPEECH identifies the target by this and nothing else.",
    )
    return parser


def load_hint(path: Path) -> AudioHints:
    """Read a hint YAML into the declared-content structure the nodes read.

    Args:
        path: The YAML file.

    Returns:
        The hint.

    Raises:
        ValueError: If the file is not a mapping, or carries a field the structure does not have.
    """
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a mapping of AudioHints fields, not {type(payload).__name__}")
    unknown = sorted(str(key) for key in payload if key not in AudioHints.model_fields)
    if unknown:
        raise ValueError(
            f"{path} carries keys AudioHints does not have: {', '.join(unknown)}. "
            "AudioHints ignores unknown keys, so this would have loaded as an empty hint and every "
            "absence would have read as a finding. A hints table keyed by filename needs the "
            "per-file entry extracted first; the fields available are: "
            f"{', '.join(sorted(AudioHints.model_fields))}."
        )
    return AudioHints.model_validate(payload)


def load_enrollment(path: Path) -> Enrollment:
    """Read an enrollment YAML or JSON into the structure SPEECH compares its speakers against.

    Args:
        path: The YAML or JSON file. Its shape is :class:`Enrollment`: ``subject_id``, ``vector``,
            a ``provenance`` block carrying ``model_id`` and the resolved ``model_commit_sha``, and
            optionally ``task`` and ``distribution``. ``sources`` is read off
            ``provenance.source_files`` and is not a field of its own.

    Returns:
        The enrollment.

    Raises:
        ValueError: If the file is not a mapping, or carries a key the structure does not have —
            at the top level or inside ``provenance``.
    """
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must hold a mapping of Enrollment fields, not {type(payload).__name__}")
    _refuse_unknown(path, payload, Enrollment.model_fields, "Enrollment")
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        _refuse_unknown(path, provenance, SpeakerEmbeddingProvenance.model_fields, "provenance")
    return Enrollment.model_validate(payload)


def _refuse_unknown(path: Path, payload: dict, fields: Mapping[str, object], what: str) -> None:
    """Refuse a mapping carrying a key the model does not have.

    Args:
        path: The file the mapping came from, for the message.
        payload: The mapping to check.
        fields: The model's fields.
        what: What the mapping is meant to be, for the message.

    Raises:
        ValueError: When any key is not a field. Pydantic ignores an extra key, so a misspelled
            field or a table keyed by subject would otherwise load as a different enrollment than
            the caller wrote — and an enrollment nobody notices is wrong identifies a target.
    """
    unknown = sorted(str(key) for key in payload if key not in fields)
    if not unknown:
        return
    raise ValueError(
        f"{path} carries {what} keys the structure does not have: {', '.join(unknown)}. "
        "Pydantic ignores unknown keys, so this would have loaded as a different enrollment than "
        "the one on disk. A table keyed by subject needs the per-subject entry extracted first; "
        f"the fields available are: {', '.join(sorted(fields))}."
    )


def main(argv: list[str] | None = None) -> int:
    """Triage one recording and print where the run went.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when VERDICT concluded and no node raised, 1 when a node raised or VERDICT never
        concluded — the verdict is still written and the run directory still holds everything the
        graph reached — and 2 when the arguments could not be resolved and nothing was measured.
        The code reports whether the graph ran, never what it concluded: a ``discard`` is a
        successful run.
    """
    args = build_parser().parse_args(argv)

    if not args.audio.exists():
        print(f"ERROR: audio file not found: {args.audio}", file=sys.stderr)
        return 2
    try:
        config = load_triage_config(args.config)
    except (OSError, ValueError, KeyError) as error:
        print(f"ERROR: invalid triage config {args.config}: {error}", file=sys.stderr)
        return 2
    hint = None
    if args.hint is not None:
        try:
            hint = load_hint(args.hint)
        except (OSError, ValueError) as error:
            print(f"ERROR: invalid hint {args.hint}: {error}", file=sys.stderr)
            return 2
    enrollment = None
    if args.enrollment is not None:
        try:
            enrollment = load_enrollment(args.enrollment)
        except (OSError, ValueError) as error:
            print(f"ERROR: invalid enrollment {args.enrollment}: {error}", file=sys.stderr)
            return 2

    print(f"Config: {config.name} v{config.version} ({config.config_hash})")
    print(f"Input:  {args.audio}")
    result = run_triage(args.audio, args.out, config, hint=hint, enrollment=enrollment)

    print(f"Run:    {result.run_dir}")
    print(f"Store:  {result.store_path}")
    for name, product in result.summary.items():
        print(f"Summary ({name}): {product}")
    for node, outcome in result.nodes.items():
        detail = outcome.verdict.outcome.value if outcome.verdict is not None else (outcome.error or "-")
        print(f"  {node:<11} {outcome.state.value:<10} {detail}")

    if result.file_verdict is not None:
        print(f"Triage:  {result.file_verdict.triage.value}")
        print(f"Release: {result.file_verdict.release.value}")
        for name, released in result.released.items():
            print(f"  {name}: {released}")
    else:
        print("Verdict: VERDICT itself did not run; see run.json", file=sys.stderr)

    errored = [node for node, outcome in result.nodes.items() if outcome.state is RunState.ERRORED]
    for node in errored:
        print(f"ERROR: {node} raised: {result.nodes[node].error}", file=sys.stderr)
    return 1 if errored or result.file_verdict is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
