"""The ``data`` command group, mounted by ``senselab.cli``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from senselab_core.runtime.fetch import DataManifest, FetchError, ManifestError, default_cache_root, fetch


def mount_data(groups: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add the ``data`` command group to a parser's subparsers."""
    data = groups.add_parser("data", help="versioned datasets")
    commands = data.add_subparsers(dest="command", required=True)
    fetch_cmd = commands.add_parser("fetch", help="download and verify the files of a data manifest")
    fetch_cmd.add_argument("manifest", type=Path, help="path to a data-manifest@2.0 YAML file")
    fetch_cmd.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="cache directory (default: $SENSELAB_DATA_CACHE or ~/.cache/senselab/data)",
    )
    fetch_cmd.add_argument(
        "--verify", action="store_true", help="re-hash every cached file instead of checking its size"
    )
    data.set_defaults(run=run_data)


def run_data(args: argparse.Namespace) -> int:
    """Run a parsed ``data fetch``; return the exit code (0 on success, 1 on a reported error)."""
    try:
        manifest = DataManifest.load(args.manifest)
        paths = fetch(manifest, cache_root=args.cache or default_cache_root(), verify="full" if args.verify else "size")
    except (ManifestError, FetchError) as err:
        print(f"senselab: error: {' '.join(str(err).split())}", file=sys.stderr)
        return 1
    except OSError as err:
        cause = " ".join(str(err).split())
        print(
            f"senselab: error: {cause}; check the paths exist and the cache directory is writable (--cache DIR)",
            file=sys.stderr,
        )
        return 1
    for name, path in paths.items():
        print(f"{name}\t{path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``data`` group alone; return the process exit code."""
    parser = argparse.ArgumentParser(prog="senselab")
    mount_data(parser.add_subparsers(dest="group", required=True))
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
