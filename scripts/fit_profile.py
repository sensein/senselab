"""Fit a dataset profile's corpus statistics from a pilot run's corpus table, into the profile's ``fitted.yaml``.

The corpus statistics (kind ``b`` in ``data/profile_values.yaml``: fences, percentiles and medians over the dataset's
own recordings) are computed by each value's written method (``senselab.audio.workflows.triage.fit``) from the
``recording_vectors.parquet`` a pilot run's corpus pass wrote, and written with what they were fitted on. A value
the profile sets in its own files wins over its fitted one; this script lists each such value beside the fit.

The order for a new dataset (plan step 10):

1. a pilot run, with ``pilot: true`` in the profile's ``profile.yaml`` (the corpus statistics may stay empty;
   until step 5 a voice task's count-in stays outside its extent and the checks that read them flag nothing);
2. the COHORT pass (``extend_cohort.py``) and the corpus table (``triage_recording_vectors.py``);
3. this script;
4. the COHORT pass again (``extend_cohort.py apply``): the session consistency outcome is decided when COHORT runs,
   so only a rerun reads the fitted bound;
5. a replay of the pilot runs (``extend_replay_decisions.py``), so their verdicts read the fitted values; then
   ``pilot: true`` comes out of the profile.

Usage:
    python scripts/fit_profile.py <recording_vectors.parquet> --profile <profile dir> [--out FILE] [--force]

Exit codes: 0 written; 2 refused (a table that is missing, unreadable or older than the fit columns, a fitted.yaml
already there without ``--force``, or a profile that cannot be loaded).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.dataset_profile import (
    FITTED_FILE,
    active_profile,
    data_values,
    lookup_value,
    run_data,
)
from senselab.audio.workflows.triage.extend import add_config_arguments, driver_config
from senselab.audio.workflows.triage.fit import Fit, columns_read, fit_values, fitted_values

PREVIOUS = ".previous"
"""The suffix a refit keeps the fit it replaces under."""

HEADER = (
    "# Written by scripts/fit_profile.py (plan step 10): this profile's corpus statistics, fitted on the corpus\n"
    "# table named under fitted_on by each value's method in senselab's data/profile_values.yaml. Refit rather\n"
    "# than edit; a value the profile sets in its own files wins over the one here.\n"
)


_ABSENT = object()


def _held(node: Any, path: Sequence[str]) -> Any:  # noqa: ANN401 -- a YAML value
    """The value at ``path``, None where it is written blank, :data:`_ABSENT` where it is not written."""
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return _ABSENT
        node = node[part]
    return node


def own_value(fit: Fit) -> Any:  # noqa: ANN401 -- a YAML value
    """What a layer merged after the profile's ``fitted.yaml`` writes for a fit.

    That is the run's data section, else the profile's own file; a profile it extends is merged before its fitted
    values, so never wins over them.

    Args:
        fit: The fit.

    Returns:
        The value written (None where it is written blank), or :data:`_ABSENT` where neither writes it.
    """
    held = _held(run_data().get(fit.file), fit.path)
    if held is not _ABSENT:
        return held
    profile = active_profile()
    supplied = profile.supplies(fit.file) if profile is not None else None
    if supplied is None or not supplied.is_file():
        return _ABSENT
    return _held(yaml.safe_load(supplied.read_text(encoding="utf-8")) or {}, fit.path)


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


def _fit(args: argparse.Namespace) -> tuple[list[Fit], dict[str, Any]]:
    """Load the profile, read the table and fit every corpus statistic.

    Args:
        args: The parsed options.

    Returns:
        The fits and the ``fitted.yaml`` document.

    Raises:
        ValueError: If the table lacks a column a fit reads (a corpus table older than schema 32).
    """
    driver_config(args, require_values=False)
    schema = pq.read_schema(args.table)
    absent = [name for name in columns_read() if name not in schema.names]
    if absent:
        raise ValueError(f"{args.table} carries no {', '.join(absent)}; rebuild the corpus table (schema 32 on)")
    rows = pq.read_table(args.table, columns=columns_read()).to_pylist()
    tasks = dict(data_values("tasks.yaml").get("tasks") or {})
    fits = fit_values(rows, tasks=tasks, setting=lambda relative, key: lookup_value(data_values(relative), key))
    version = (schema.metadata or {}).get(b"senselab.recording_vectors.schema_version")
    return fits, document(fits, args.table, len(rows), version.decode() if version else None)


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
    parser.add_argument(
        "--force", action="store_true", help=f"Replace a {FITTED_FILE} already there (kept as {FITTED_FILE}{PREVIOUS})"
    )
    args = parser.parse_args(argv)
    if args.profile is None:
        print("refusing: --profile names the profile to fit", file=sys.stderr)
        return 2
    loaded = args.profile / FITTED_FILE
    out = args.out or loaded
    if out.exists() and not args.force:
        print(f"refusing: {out} exists; pass --force to refit it", file=sys.stderr)
        return 2
    previous = None
    if out.exists() and out.resolve() == loaded.resolve():
        previous = loaded.with_name(FITTED_FILE + PREVIOUS)
        os.replace(loaded, previous)  # the fit reads the profile without the values it replaces
    try:
        fits, held = _fit(args)
    except (OSError, ValueError, yaml.YAMLError, pa.ArrowException) as error:
        if previous is not None:
            os.replace(previous, loaded)
        print(f"refusing: {error}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text(HEADER + yaml.safe_dump(held, sort_keys=False), encoding="utf-8")
    os.replace(partial, out)
    print(f"{held['fitted_on']['rows']} rows of {args.table} -> {out}")
    if out.resolve() != loaded.resolve():
        print(f"  the profile reads none of these until they are its {loaded}")
    for fit in fits:
        own = own_value(fit)
        note = (
            ""
            if own is _ABSENT
            else "  (the profile writes it blank, which hides the fit: remove that line)"
            if own is None
            else f"  (the profile sets {own!r} itself, which wins)"
        )
        print(f"  {fit.key}: {fit.value!r} over {fit.n}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
