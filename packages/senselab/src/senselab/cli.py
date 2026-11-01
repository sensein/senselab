"""The ``senselab`` command line; subcommands of later tasks are mounted here."""

import argparse
import sys
from collections.abc import Sequence

from senselab_core.cli import mount_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="senselab")
    groups = parser.add_subparsers(dest="group", required=True)
    mount_data(groups)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; return the process exit code (0 on success, 1 on a reported error)."""
    args = _build_parser().parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
