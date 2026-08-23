#!/usr/bin/env python3
r"""Triage one recording through the senselab triage graph.

    uv run python scripts/triage_audio.py recording.wav [--out DIR] [--config OVERRIDE.yaml] [--hint HINT.yaml]

Two arguments, and two optional files. There are deliberately no per-knob flags: every number the
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

Layout under ``--out/<stem>_<utc-timestamp>/``:

    run/store.jsonl        the append-only provenance store: every node's measurements and verdicts
    run/streams/           the conditioned streams
    run/derivatives/       the sidecars measurements point at
    run/figures/           the aligned figures
    run/run.json           the runner's own record: per-node run state, and any node that raised
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

import yaml  # type: ignore[import-untyped]

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.run import run_triage

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
    return AudioHints.model_validate(payload)


def main(argv: list[str] | None = None) -> int:
    """Triage one recording and print where the run went.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when the graph ran, 2 when the arguments could not be resolved.
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

    print(f"Config: {config.name} v{config.version} ({config.config_hash})")
    print(f"Input:  {args.audio}")
    result = run_triage(args.audio, args.out, config, hint=hint)

    print(f"Run:    {result.run_dir}")
    print(f"Store:  {result.store_path}")
    for node, outcome in result.nodes.items():
        detail = outcome.verdict.outcome.value if outcome.verdict is not None else (outcome.error or "-")
        print(f"  {node:<11} {outcome.state.value:<10} {detail}")
    if result.file_verdict is None:
        print("Verdict: VERDICT itself did not run; see run.json", file=sys.stderr)
        return 0
    print(f"Triage:  {result.file_verdict.triage.value}")
    print(f"Release: {result.file_verdict.release.value}")
    for name, released in result.released.items():
        print(f"  {name}: {released}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
