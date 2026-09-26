"""Build the single-file viewer for recording_vectors.parquet.

    uv run python scripts/triage_vectors_viewer.py            # refresh the committed page
    uv run python scripts/triage_vectors_viewer.py --out /tmp/viewer.html

Open the result from file:// and hand it the parquet. The page reads the parquet from your own
disk; it fetches nothing. It renders transcript text and detected PII, so it is not a page to host.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senselab.audio.workflows.triage.viewer import BUILT_PAGE, write  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Write the assembled page.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 always; a missing part raises instead.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help=f"Where to write; default {BUILT_PAGE}.")
    args = parser.parse_args(argv)

    target = write(args.out)
    print(f"{target}  {target.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
