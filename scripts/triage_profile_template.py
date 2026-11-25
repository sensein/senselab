#!/usr/bin/env python3
"""Write a new dataset's triage profile as a template: every value it must supply blank, beside what it means.

    uv run python scripts/triage_profile_template.py OUT_DIR --name my-dataset
    uv run python scripts/triage_profile_template.py OUT_DIR --name my-dataset --extends none
    uv run python scripts/triage_profile_template.py OUT_DIR --name my-dataset --start-from OTHER_PROFILE_DIR

By default the profile extends senselab's reference profile for adult clinical speech recordings
(``senselab:adult_clinical_speech``), so only the dataset's own values are asked for: its task list, its policy,
the statistics of its own recordings and its per-task values. ``--extends none`` asks for every value; another
``senselab:<name>`` or a profile directory builds on that. ``--start-from`` copies another dataset's profile
instead. The printed count is what is still blank; a run names each one until it is filled.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from senselab.audio.workflows.triage.profile_template import REFERENCE_PROFILE, values_to_supply, write_template


def main(argv: list[str] | None = None) -> int:
    """Write the template and print what it asks for.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0, or 2 where the output exists or the profile to extend cannot be read.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None)
    parser.add_argument("out", type=Path, help="The new profile's directory (must not exist)")
    parser.add_argument("--name", required=True, help="The new profile's name")
    parser.add_argument("--extends", default=REFERENCE_PROFILE, help="senselab:<name>, a profile directory, or none")
    parser.add_argument("--start-from", type=Path, default=None, help="Copy this profile directory instead")
    args = parser.parse_args(argv)
    extends = None if str(args.extends).lower() == "none" else args.extends
    try:
        supply = values_to_supply(extends) if args.start_from is None else None
        root = write_template(args.out, name=args.name, extends=extends, start_from=args.start_from, supply=supply)
        blank = sum(len(entries) for entries in supply.values()) if supply is not None else None
    except (FileExistsError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Profile template: {root}")
    if blank is not None:
        print(f"Values to fill in: {blank}, plus tasks.yaml and the dataset files beside it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
