#!/usr/bin/env python3
"""Profile any Python script or Jupyter notebook with Scalene.

A thin wrapper around Scalene 2.2's `run` and `view` subcommands that:
- Accepts .py or .ipynb targets (notebooks are auto-converted via nbconvert)
- Produces a standalone HTML report (default) or JSON profile
- Supports path-substring scoping (--scope, --no-thirdparty, --exclude)
- Writes reports to artifacts/scalene/ with a timestamp suffix

Install:
    uv sync --group profiling

Examples:
    uv run python scripts/profile_with_scalene.py path/to/script.py
    uv run python scripts/profile_with_scalene.py tutorials/audio/speech_to_text.ipynb
    uv run python scripts/profile_with_scalene.py --scope speech_to_text examples/run.py
    uv run python scripts/profile_with_scalene.py examples/run.py --- --input my.wav

See specs/20260503-235625-scalene-profiling/contracts/cli.md for the full CLI reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Exit codes (mirrors contracts/cli.md)
EXIT_SUCCESS = 0
EXIT_PROFILING_FAILED = 1
EXIT_SCALENE_MISSING = 3
EXIT_INVALID_ARGS = 4
EXIT_NBCONVERT_MISSING = 5


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments. Args after a literal `---` are forwarded to the target."""
    if "---" in argv:
        sep = argv.index("---")
        own_argv = argv[:sep]
        target_args = argv[sep + 1 :]
    else:
        own_argv = argv
        target_args = []

    parser = argparse.ArgumentParser(
        description="Profile a Python script or Jupyter notebook with Scalene.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Use `---` (three dashes) to separate target script arguments:\n"
            "  profile_with_scalene.py script.py --- --input my.wav --batch 4"
        ),
    )
    parser.add_argument("target", help="Path to a .py script or .ipynb notebook")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/scalene"),
        help="Directory where the report is written (default: artifacts/scalene/)",
    )
    parser.add_argument(
        "--format",
        choices=("html", "json"),
        default="html",
        help="Report format. JSON is Scalene's native run output; HTML adds a view step.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip memory profiling for a faster run",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU profiling (no-op on macOS; requires CUDA)",
    )
    parser.add_argument(
        "--exclude",
        metavar="SUBSTR",
        default=None,
        help="Exclude files whose path contains SUBSTR (substring match against file paths)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help=(
            "Retain the intermediate JSON profile (for --format html) and the converted "
            ".py file (for notebook targets) next to the final report"
        ),
    )
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument(
        "--scope",
        metavar="SUBSTR",
        default=None,
        help=(
            "Profile only files whose path contains SUBSTR (substring match against file "
            "paths, not function names). Mutually exclusive with --no-thirdparty."
        ),
    )
    scope_group.add_argument(
        "--no-thirdparty",
        action="store_true",
        help=("Restrict profiling to files with 'senselab' in their path. Mutually exclusive with --scope."),
    )

    ns = parser.parse_args(own_argv)
    ns.target_args = target_args
    return ns


def check_scalene_available() -> None:
    """Exit with code 3 and an install hint if Scalene is missing."""
    if importlib.util.find_spec("scalene") is None:
        print(
            "ERROR: Scalene is not installed in this environment.\nTo install: uv sync --group profiling",
            file=sys.stderr,
        )
        sys.exit(EXIT_SCALENE_MISSING)


def check_nbconvert_available() -> None:
    """Exit with code 5 and an install hint if nbconvert is missing."""
    if importlib.util.find_spec("nbconvert") is None:
        print(
            "ERROR: nbconvert is not installed; required to profile notebooks.\nTo install: uv sync --group profiling",
            file=sys.stderr,
        )
        sys.exit(EXIT_NBCONVERT_MISSING)


def validate_target(target: Path) -> str:
    """Return target kind ('python_script' or 'jupyter_notebook'); exit 4 if invalid."""
    if not target.exists() or not target.is_file():
        print(f"ERROR: target file does not exist or is not a file: {target}", file=sys.stderr)
        sys.exit(EXIT_INVALID_ARGS)

    suffix = target.suffix.lower()
    if suffix == ".py":
        return "python_script"
    if suffix == ".ipynb":
        return "jupyter_notebook"
    print(
        f"ERROR: unsupported target extension '{suffix}'. Expected .py or .ipynb.",
        file=sys.stderr,
    )
    sys.exit(EXIT_INVALID_ARGS)


_IPYTHON_STUB = """\
# Auto-injected by profile_with_scalene.py — stub IPython helpers so
# notebooks converted via nbconvert can run as plain Python under Scalene.
# This is best-effort: cells whose semantics depend on real IPython behavior
# (e.g., `%matplotlib inline` actually configuring backends, `!shell-cmd`
# side effects, `display(obj)` returning a value, widget event loops) will
# silently no-op. Profile the resulting script accordingly.
import sys as _scalene_wrapper_sys
import types as _scalene_wrapper_types

class _NoOpIPython:
    def __getattr__(self, name):
        return self
    def __call__(self, *args, **kwargs):
        return None

def get_ipython():  # noqa: N802
    return _NoOpIPython()

# Stub the IPython package + common submodules so `from IPython.display import display`
# and similar import statements (preserved verbatim by nbconvert) do not raise.
if "IPython" not in _scalene_wrapper_sys.modules:
    _ipy = _scalene_wrapper_types.ModuleType("IPython")
    _ipy.get_ipython = get_ipython
    _scalene_wrapper_sys.modules["IPython"] = _ipy
    for _sub in ("display", "core", "core.display", "core.magic"):
        _mod = _scalene_wrapper_types.ModuleType(f"IPython.{_sub}")
        # Any attribute access on these modules returns a no-op callable
        _mod.__getattr__ = lambda name: (lambda *a, **k: None)
        _scalene_wrapper_sys.modules[f"IPython.{_sub}"] = _mod
    # Convenience: `display` is the most-imported name; expose it directly.
    _scalene_wrapper_sys.modules["IPython.display"].display = lambda *a, **k: None
del _scalene_wrapper_sys, _scalene_wrapper_types
"""


def convert_notebook(notebook: Path, tmpdir: Path) -> Path:
    """Convert a notebook to .py via nbconvert; return the resulting .py path.

    nbconvert translates IPython magics (e.g., `%matplotlib inline`) into
    `get_ipython().run_line_magic(...)` calls. Those fail when the converted
    file is run as plain Python. We prepend a no-op `get_ipython` stub so
    magic calls become harmless no-ops during profiling.
    """
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "script",
        "--output-dir",
        str(tmpdir),
        str(notebook),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print("ERROR: nbconvert failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(EXIT_PROFILING_FAILED)
    # nbconvert may emit warnings to stderr even when returncode == 0 (e.g., missing
    # kernel spec, deprecation notices). Forward them so they do not silently disappear.
    if result.stderr.strip():
        print(result.stderr, end="", file=sys.stderr)
    # nbconvert decides the output extension from kernel metadata: .py for Python
    # notebooks, .txt as a fallback for non-Python kernels (or notebooks lacking
    # language_info). We accept either and rename .txt -> .py before profiling.
    candidates = [tmpdir / f"{notebook.stem}.py", tmpdir / f"{notebook.stem}.txt"]
    converted = next((p for p in candidates if p.exists()), None)
    if converted is None:
        produced = list(tmpdir.iterdir())
        print(
            f"ERROR: nbconvert did not produce expected file. Looked for: "
            f"{[str(c) for c in candidates]}; tmpdir contents: {[str(p) for p in produced]}",
            file=sys.stderr,
        )
        sys.exit(EXIT_PROFILING_FAILED)
    # Always run as a .py so Scalene's --program-path heuristics work consistently.
    if converted.suffix != ".py":
        py_path = converted.with_suffix(".py")
        converted.rename(py_path)
        converted = py_path
    original = converted.read_text(encoding="utf-8")
    converted.write_text(_inject_ipython_stub(original), encoding="utf-8")
    return converted


_FUTURE_IMPORT_RE = re.compile(r"^\s*from\s+__future__\s+import\b")


def _inject_ipython_stub(source: str) -> str:
    """Return ``source`` with ``_IPYTHON_STUB`` inserted at a syntactically valid spot.

    ``from __future__`` imports must be the first non-docstring/comment statement
    in a Python file, so the stub cannot be placed before them. Walk the source
    line-by-line, find the index just after any ``from __future__`` block, and
    insert the stub there. When no future imports are present, the stub goes at
    the very top.
    """
    lines = source.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if _FUTURE_IMPORT_RE.match(line):
            insert_at = i + 1
    return "".join(lines[:insert_at]) + _IPYTHON_STUB + "\n" + "".join(lines[insert_at:])


def build_run_command(
    target: Path,
    target_args: list[str],
    json_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Build the `scalene run` command list."""
    cmd = [sys.executable, "-m", "scalene", "run", "-o", str(json_path)]

    if args.cpu_only:
        cmd.append("--cpu-only")
    if args.gpu:
        cmd.append("--gpu")
    if args.scope is not None:
        cmd.extend(["--profile-only", args.scope])
    elif args.no_thirdparty:
        cmd.extend(["--profile-only", "senselab"])
    if args.exclude is not None:
        cmd.extend(["--profile-exclude", args.exclude])

    cmd.append(str(target))
    if target_args:
        cmd.append("---")
        cmd.extend(target_args)
    return cmd


def run_view_step(json_path: Path, work_dir: Path) -> Path:
    """Run `scalene view --standalone` to produce HTML; return the HTML path.

    Scalene's `view --standalone` writes to `scalene-profile.html` in the current
    working directory (not next to the input JSON). We run it inside `work_dir`
    and rename the result to a deterministic per-target name.
    """
    cmd = [
        sys.executable,
        "-m",
        "scalene",
        "view",
        "--standalone",
        str(json_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir, check=False)
    if result.returncode != 0:
        print("ERROR: scalene view failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(EXIT_PROFILING_FAILED)
    produced = work_dir / "scalene-profile.html"
    if not produced.exists():
        print(
            f"ERROR: scalene view did not produce expected file: {produced}",
            file=sys.stderr,
        )
        sys.exit(EXIT_PROFILING_FAILED)
    renamed = work_dir / json_path.with_suffix(".html").name
    produced.rename(renamed)
    return renamed


def main(argv: list[str] | None = None) -> int:
    """Run the wrapper end-to-end and return the wrapper's exit code."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    check_scalene_available()

    target = Path(args.target).resolve()
    kind = validate_target(target)
    if kind == "jupyter_notebook":
        check_nbconvert_available()

    if sys.platform == "darwin" and args.gpu:
        print(
            "Note: GPU profiling is not available on macOS; GPU columns will be empty.",
            file=sys.stderr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp + PID + short token: avoids collisions when two runs hit the
    # same second (e.g., concurrent CI invocations or rapid local re-runs).
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"{timestamp}-{os.getpid()}-{secrets.token_hex(2)}"
    json_filename = f"{target.stem}_{suffix}.json"
    final_json_path = args.output_dir / json_filename
    final_html_path = final_json_path.with_suffix(".html")

    with tempfile.TemporaryDirectory(prefix="scalene_profile_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        if kind == "jupyter_notebook":
            run_target = convert_notebook(target, tmpdir)
            if args.keep_intermediate:
                kept_py = args.output_dir / f"{target.stem}_{timestamp}.py"
                shutil.copy2(run_target, kept_py)
                print(f"Kept converted notebook: {kept_py}")
        else:
            run_target = target

        intermediate_json = tmpdir / json_filename
        run_cmd = build_run_command(run_target, args.target_args, intermediate_json, args)
        run_result = subprocess.run(run_cmd, text=True, check=False)
        if run_result.returncode != 0 or not intermediate_json.exists():
            print(
                "ERROR: scalene run failed (target script may have raised, or Scalene errored).",
                file=sys.stderr,
            )
            return EXIT_PROFILING_FAILED

        if args.format == "json":
            shutil.move(str(intermediate_json), str(final_json_path))
            report_path = final_json_path
        else:
            html_in_tmp = run_view_step(intermediate_json, tmpdir)
            shutil.move(str(html_in_tmp), str(final_html_path))
            report_path = final_html_path
            if args.keep_intermediate:
                shutil.move(str(intermediate_json), str(final_json_path))

    quoted_path = shlex.quote(str(report_path))
    print(f"Scalene profile written to: {report_path}")
    if sys.platform == "darwin":
        print(f"Open with: open {quoted_path}")
    elif sys.platform.startswith("linux"):
        print(f"Open with: xdg-open {quoted_path}")
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
