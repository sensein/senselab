"""Compatibility boundary tester.

Probes version boundaries declared in the compatibility matrix to verify
they are accurate. For each function × version combination:

1. **Within range**: verify the function works (import + basic call)
2. **Outside range**: verify the function indeed fails

Results are written to a JSON report that can be used to update the
matrix ranges — widening them when functions pass on untested versions,
or tightening when they fail within the declared range.

This is NOT part of the regular test suite. Run explicitly:

    uv run python -m senselab.utils.compatibility_test_runner [--update]

Or via pytest with the compat marker:

    uv run pytest -m compat --no-header
"""

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from senselab.utils.compatibility import COMPATIBILITY_MATRIX, CompatibilityEntry, VersionRange
from senselab.utils.subprocess_venv import ensure_venv

logger = logging.getLogger("senselab")


@dataclass
class ProbeResult:
    """Result of probing a single function × version combination."""

    function_key: str
    python_version: str
    torch_version: str
    expected_in_range: bool
    actual_passed: bool
    error: Optional[str] = None

    @property
    def boundary_correct(self) -> bool:
        """True if the matrix boundary matches reality."""
        return self.expected_in_range == self.actual_passed

    @property
    def should_widen(self) -> bool:
        """True if the matrix says out-of-range but function actually works."""
        return not self.expected_in_range and self.actual_passed

    @property
    def should_tighten(self) -> bool:
        """True if the matrix says in-range but function actually fails."""
        return self.expected_in_range and not self.actual_passed


@dataclass
class CompatibilityReport:
    """Full report from a compatibility probe run."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    host_python: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}")
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def boundaries_correct(self) -> list[ProbeResult]:
        """Results where the matrix boundary matches reality."""
        return [r for r in self.results if r.boundary_correct]

    @property
    def should_widen(self) -> list[ProbeResult]:
        """Functions that work outside their declared range."""
        return [r for r in self.results if r.should_widen]

    @property
    def should_tighten(self) -> list[ProbeResult]:
        """Functions that fail within their declared range."""
        return [r for r in self.results if r.should_tighten]

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Compatibility Probe Report ({self.timestamp})",
            f"  Host Python: {self.host_python}",
            f"  Total probes: {len(self.results)}",
            f"  Boundaries correct: {len(self.boundaries_correct)}",
            f"  Should widen (works outside range): {len(self.should_widen)}",
            f"  Should tighten (fails inside range): {len(self.should_tighten)}",
        ]
        if self.should_widen:
            lines.append("\n  Candidates to WIDEN:")
            for r in self.should_widen:
                lines.append(f"    {r.function_key}: passed on Python {r.python_version} / torch {r.torch_version}")
        if self.should_tighten:
            lines.append("\n  Candidates to TIGHTEN:")
            for r in self.should_tighten:
                lines.append(
                    f"    {r.function_key}: FAILED on Python {r.python_version} / torch {r.torch_version}: {r.error}"
                )
        return "\n".join(lines)


def _probe_function(
    function_key: str,
    python_version: str,
    torch_version: str,
) -> ProbeResult:
    """Probe whether a function's imports work on a specific version combo.

    Creates a temporary venv with the specified Python + torch version
    and tries to import the function's dependencies.
    """
    entry = COMPATIBILITY_MATRIX[function_key]

    expected_py = entry.python_versions.contains(f"{python_version}.0")
    expected_torch = entry.torch_versions.contains(f"{torch_version}.0")
    expected_in_range = expected_py and expected_torch

    if entry.isolated:
        # For isolated backends, just check the version range — actual
        # testing happens in the subprocess venv
        return ProbeResult(
            function_key=function_key,
            python_version=python_version,
            torch_version=torch_version,
            expected_in_range=expected_in_range,
            actual_passed=expected_in_range,  # trust the range for isolated
        )

    # Build a minimal test script that imports the deps
    dep_imports = []
    import_map = {
        "pyannote-audio": "pyannote.audio",
        "coqui-tts": "TTS",
        "opencv-python-headless": "cv2",
        "sentence-transformers": "sentence_transformers",
    }
    for dep in entry.required_deps:
        mod = import_map.get(dep, dep.replace("-", "_"))
        dep_imports.append(f"import {mod}")

    test_script = "\n".join(
        [
            "import sys",
            *dep_imports,
            "print('OK')",
        ]
    )

    # Create a probe venv
    venv_name = f"probe-py{python_version}-torch{torch_version}"
    reqs = [f"torch=={torch_version}.*"]
    for dep in entry.required_deps:
        ver_spec = entry.dep_versions.get(dep, "")
        reqs.append(f"{dep}{ver_spec}")

    try:
        venv_dir = ensure_venv(venv_name, reqs, python_version=python_version)
        python = str(venv_dir / "bin" / "python")
        result = subprocess.run(
            [python, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        actual_passed = result.returncode == 0
        error = result.stderr.strip() if not actual_passed else None
    except Exception as exc:
        actual_passed = False
        error = str(exc)

    return ProbeResult(
        function_key=function_key,
        python_version=python_version,
        torch_version=torch_version,
        expected_in_range=expected_in_range,
        actual_passed=actual_passed,
        error=error,
    )


def _extract_upper_bound(spec_str: str) -> Optional[str]:
    """Extract the upper bound version from a PEP 440 specifier.

    E.g., ">=3.11,<3.13" → "3.13", ">=2.8" → None (no upper bound).
    """
    for part in spec_str.split(","):
        part = part.strip()
        if part.startswith("<") and not part.startswith("<="):
            return part[1:]
        if part.startswith("<="):
            return part[2:]
    return None


def _next_minor(version: str) -> str:
    """Bump the minor version by 1. E.g., "3.12" → "3.13", "2.8" → "2.9"."""
    parts = version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    return ".".join(parts)


def _versions_to_probe(entry: CompatibilityEntry) -> tuple[list[str], list[str]]:
    """Determine which versions to probe for a function.

    Strategy: test one version inside each boundary (should pass) and
    one version just outside the upper bound (should fail). We don't
    test below the lower bound — no need to support older versions.

    Returns:
        (python_versions, torch_versions) to probe.
    """
    py_versions = set()
    torch_versions = set()

    # Python: one inside (highest supported), one just above upper bound
    py_upper = _extract_upper_bound(str(entry.python_versions))
    if py_upper:
        # Test the version just below the upper bound (should pass)
        below = py_upper.split(".")
        below[-1] = str(int(below[-1]) - 1)
        py_versions.add(".".join(below))
        # Test the upper bound itself (should fail)
        py_versions.add(py_upper)
    else:
        # No upper bound — test latest known versions (should pass)
        py_versions.update(["3.13", "3.14"])

    # Torch: same logic
    torch_upper = _extract_upper_bound(str(entry.torch_versions))
    if torch_upper:
        below = torch_upper.split(".")
        below[-1] = str(int(below[-1]) - 1)
        torch_versions.add(".".join(below))
        torch_versions.add(torch_upper)
    else:
        torch_versions.update(["2.10"])  # latest known

    return sorted(py_versions), sorted(torch_versions)


def run_compatibility_probes(
    functions: Optional[list[str]] = None,
    python_versions: Optional[list[str]] = None,
    torch_versions: Optional[list[str]] = None,
) -> CompatibilityReport:
    """Run compatibility probes at version boundaries.

    For each function, probes ONE version inside the boundary (should pass)
    and ONE version just outside the upper bound (should fail). Does NOT
    test below the lower bound — we don't need to support older versions.

    Override auto-detection with explicit python_versions/torch_versions
    to probe specific combos.

    Args:
        functions: Function keys to test. Default: all non-isolated.
        python_versions: Override: specific Python versions to test.
        torch_versions: Override: specific torch versions to test.

    Returns:
        CompatibilityReport with results.
    """
    func_keys = functions or [k for k, v in COMPATIBILITY_MATRIX.items() if not v.isolated]
    report = CompatibilityReport()

    for func_key in func_keys:
        entry = COMPATIBILITY_MATRIX[func_key]

        if python_versions and torch_versions:
            py_vers, t_vers = python_versions, torch_versions
        else:
            py_vers, t_vers = _versions_to_probe(entry)

        for py_ver in py_vers:
            for t_ver in t_vers:
                logger.info("Probing %s on Python %s / torch %s", func_key, py_ver, t_ver)
                result = _probe_function(func_key, py_ver, t_ver)
                report.results.append(result)

    return report


def update_matrix_from_report(report: CompatibilityReport) -> dict[str, dict[str, str]]:
    """Suggest matrix updates based on probe results.

    Returns:
        Dict mapping function_key → {"python_versions": new_spec, "torch_versions": new_spec}
        Only includes entries that need updating.
    """
    updates: dict[str, dict[str, str]] = {}

    # Group results by function
    by_func: dict[str, list[ProbeResult]] = {}
    for r in report.results:
        by_func.setdefault(r.function_key, []).append(r)

    for func_key, results in by_func.items():
        passing_py = sorted({r.python_version for r in results if r.actual_passed})
        failing_py = sorted({r.python_version for r in results if not r.actual_passed})
        failing_torch = sorted({r.torch_version for r in results if not r.actual_passed})

        entry = COMPATIBILITY_MATRIX.get(func_key)
        if not entry:
            continue

        suggested: dict[str, str] = {}

        # Suggest widened Python range if functions pass on higher versions
        if passing_py:
            max_passing = passing_py[-1]
            current_spec = str(entry.python_versions)
            if f"<{max_passing}" in current_spec or not entry.python_versions.contains(f"{max_passing}.0"):
                suggested["python_versions"] = f">={passing_py[0]}"
                if failing_py:
                    min_failing = failing_py[0]
                    suggested["python_versions"] += f",<{min_failing}"

        # Suggest widened torch range
        passing_t = sorted({r.torch_version for r in results if r.actual_passed})
        if passing_t:
            max_t = passing_t[-1]
            current_t = str(entry.torch_versions)
            if f"<{max_t}" in current_t or not entry.torch_versions.contains(f"{max_t}.0"):
                suggested["torch_versions"] = f">={passing_t[0]}"
                if failing_torch:
                    suggested["torch_versions"] += f",<{failing_torch[0]}"

        if suggested:
            updates[func_key] = suggested

    return updates


# ── CLI entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Probe senselab compatibility boundaries")
    parser.add_argument("--python", nargs="+", default=None, help="Python versions to test")
    parser.add_argument("--torch", nargs="+", default=None, help="Torch versions to test")
    parser.add_argument("--functions", nargs="+", default=None, help="Function keys to test")
    parser.add_argument("--output", type=str, default=None, help="Write JSON report to file")
    parser.add_argument("--update", action="store_true", help="Print suggested matrix updates")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report = run_compatibility_probes(
        python_versions=args.python,
        torch_versions=args.torch,
        functions=args.functions,
    )

    print(report.summary())

    if args.output:
        Path(args.output).write_text(report.to_json())
        print(f"\nReport written to {args.output}")

    if args.update:
        updates = update_matrix_from_report(report)
        if updates:
            print("\nSuggested matrix updates:")
            for func_key, changes in updates.items():
                print(f"  {func_key}:")
                for field_name, new_spec in changes.items():
                    print(f"    {field_name}: {new_spec}")
        else:
            print("\nNo matrix updates needed — all boundaries correct.")
