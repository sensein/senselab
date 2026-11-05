"""Fit a dataset profile's corpus statistics from a pilot run's corpus table, into the profile's ``fitted.yaml``.

The corpus statistics (kind ``b`` in ``data/profile_values.yaml``: fences, percentiles and medians over the dataset's
own recordings) are computed by each value's written method (``senselab.audio.workflows.triage.fit``) from the
``recording_vectors.parquet`` a pilot run's corpus pass wrote, and written with what they were fitted on. A value
the profile sets in its own files wins over its fitted one; this script lists each such value beside the fit.

The order for a new dataset (plan step 10):

1. a pilot run, with ``pilot: true`` in the profile's ``profile.yaml`` (the corpus statistics may stay empty);
2. the COHORT pass (``extend_cohort.py``) and the corpus table (``recording_vectors``);
3. this script;
4. a replay of the pilot runs (``extend_replay_decisions.py``), so their verdicts read the fitted values; then
   ``pilot: true`` comes out of the profile.

Usage:
    python scripts/fit_profile.py <recording_vectors.parquet> --profile <profile dir> [--out FILE] [--force]

Exit codes: 0 written; 2 refused (a table older than the fit columns, a fitted.yaml already there without
``--force``, or a profile that cannot be loaded).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.dataset_profile import FITTED_FILE, active_profile, data_values
from senselab.audio.workflows.triage.extend import add_config_arguments, driver_config
from senselab.audio.workflows.triage.fit import Fit, columns_read, fit_values, fitted_values

HEADER = (
    "# Written by scripts/fit_profile.py (plan step 10): this profile's corpus statistics, fitted on the corpus\n"
    "# table named under fitted_on by each value's method in senselab's data/profile_values.yaml. Refit rather\n"
    "# than edit; a value the profile sets in its own files wins over the one here.\n"
)


def _lookup(node: Any, path: Sequence[str]) -> Any:  # noqa: ANN401 -- a YAML value
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def own_value(fit: Fit) -> Any:  # noqa: ANN401 -- a YAML value
    """The value the active profile, or one it extends, sets for a fit in its own files, or None."""
    profile = active_profile()
    for layer in reversed(profile.chain() if profile is not None else ()):
        supplied = layer.supplies(fit.file)
        if supplied is not None and supplied.is_file():
            value = _lookup(yaml.safe_load(supplied.read_text(encoding="utf-8")) or {}, fit.path)
            if value is not None:
                return value
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def document(fits: Sequence[Fit], table: Path, rows: int, schema_version: str | None) -> dict[str, Any]:
    """The ``fitted.yaml`` document: what it was fitted on, the values, and each fit's count.

    Args:
        fits: The fits.
        table: The corpus table read.
        rows: Its row count.
        schema_version: Its schema version, from the parquet metadata.

    Returns:
        The document.
    """
    try:
        version = metadata.version("senselab")
    except metadata.PackageNotFoundError:
        version = None
    return {
        "fitted_on": {
            "table": str(table),
            "sha256": _sha256(table),
            "rows": rows,
            "schema_version": schema_version,
            "senselab": version,
            "date": datetime.date.today().isoformat(),
        },
        "values": fitted_values(fits),
        "fits": [
            {
                "value": fit.key,
                "fitted": fit.value,
                "n": fit.n,
                "statistic": fit.method["statistic"],
                "column": fit.method["column"],
            }
            for fit in fits
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Fit and write.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None)
    parser.add_argument("table", type=Path, help="The pilot run's recording_vectors.parquet")
    add_config_arguments(parser)
    parser.add_argument("--out", type=Path, default=None, help=f"Where to write; default <profile>/{FITTED_FILE}")
    parser.add_argument("--force", action="store_true", help="Replace a fitted.yaml already there")
    args = parser.parse_args(argv)
    if args.profile is None:
        print("refusing: --profile names the profile to fit", file=sys.stderr)
        return 2
    out = args.out or args.profile / FITTED_FILE
    if out.exists() and not args.force:
        print(f"refusing: {out} exists; pass --force to refit it", file=sys.stderr)
        return 2
    try:
        driver_config(args, require_values=False)
    except (OSError, ValueError) as error:
        print(f"refusing: the profile {args.profile} cannot be loaded: {error}", file=sys.stderr)
        return 2
    schema = pq.read_schema(args.table)
    absent = [name for name in columns_read() if name not in schema.names]
    if absent:
        print(
            f"refusing: {args.table} carries no {', '.join(absent)}; rebuild the corpus table (schema 32 on)",
            file=sys.stderr,
        )
        return 2
    rows = pq.read_table(args.table, columns=columns_read()).to_pylist()
    tasks = dict(data_values("tasks.yaml").get("tasks") or {})
    fits = fit_values(rows, tasks=tasks, setting=lambda relative, key: _lookup(data_values(relative), key))
    version = (schema.metadata or {}).get(b"senselab.recording_vectors.schema_version")
    held = document(fits, args.table, len(rows), version.decode() if version else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER + yaml.safe_dump(held, sort_keys=False), encoding="utf-8")
    print(f"{len(rows)} rows of {args.table} -> {out}")
    for fit in fits:
        own = own_value(fit)
        kept = "" if own is None else f"  (the profile sets {own!r} itself, which wins)"
        print(f"  {fit.key}: {fit.value!r} over {fit.n}{kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
