#!/usr/bin/env python3
"""Profile cold-start import times for senselab tutorial notebooks.

Extracts all import statements from tutorial notebooks, times each in an
isolated subprocess, identifies bottlenecks, and produces a Markdown report.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Modules known to be part of the Python standard library (common ones used in tutorials)
_STDLIB_MODULES = frozenset(
    {
        "os",
        "sys",
        "json",
        "re",
        "time",
        "typing",
        "pathlib",
        "platform",
        "urllib",
        "base64",
        "collections",
        "functools",
        "itertools",
        "math",
        "datetime",
        "io",
        "copy",
        "warnings",
        "abc",
        "contextlib",
        "dataclasses",
        "enum",
        "hashlib",
        "logging",
        "operator",
        "string",
        "textwrap",
        "threading",
        "multiprocessing",
        "subprocess",
    }
)

# Patterns to skip (platform-specific, unavailable outside their environment)
_SKIP_PATTERNS = [
    re.compile(r"google\.colab"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile import times for senselab tutorial notebooks.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Seconds above which an import is flagged as a bottleneck (default: 2.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/import_profile_report.md"),
        help="Path for the Markdown report (default: artifacts/import_profile_report.md)",
    )
    parser.add_argument(
        "--tutorials-dir",
        type=Path,
        default=Path("tutorials"),
        help="Path to the tutorials directory (default: tutorials/)",
    )
    return parser.parse_args()


def extract_imports_from_notebooks(tutorials_dir: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Extract import statements from all tutorial notebooks.

    Returns:
        A tuple of:
        - imports_to_notebooks: dict mapping each unique import line to a list of notebook paths
        - notebook_to_imports: dict mapping each notebook path to its ordered list of import lines
    """
    imports_to_notebooks: dict[str, list[str]] = {}
    notebook_to_imports: dict[str, list[str]] = {}

    notebook_paths = sorted(tutorials_dir.rglob("*.ipynb"))
    for nb_path in notebook_paths:
        # Skip checkpoint files
        if ".ipynb_checkpoints" in str(nb_path):
            continue

        rel_path = str(nb_path)
        notebook_imports: list[str] = []

        with open(nb_path, encoding="utf-8") as f:
            try:
                nb_data = json.load(f)
            except json.JSONDecodeError:
                continue

        for cell in nb_data.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            # Join all source fragments into one string, then split by newlines
            full_source = "".join(cell.get("source", []))
            source_lines = full_source.split("\n")

            i = 0
            while i < len(source_lines):
                line = source_lines[i].strip()
                # Match import statements
                if re.match(r"^(import |from \S+ import )", line):
                    # Handle multi-line imports with parentheses
                    if "(" in line and ")" not in line:
                        # Collect continuation lines until closing paren
                        parts = [source_lines[i].rstrip()]
                        i += 1
                        while i < len(source_lines) and ")" not in source_lines[i]:
                            parts.append(source_lines[i].rstrip())
                            i += 1
                        if i < len(source_lines):
                            parts.append(source_lines[i].rstrip())
                        # Join into a single import statement
                        line = " ".join(p.strip() for p in parts)
                        # Normalize whitespace inside parens
                        line = re.sub(r"\(\s+", "(", line)
                        line = re.sub(r"\s+\)", ")", line)
                        line = re.sub(r",\s+", ", ", line)

                    # Skip platform-specific imports
                    if any(pat.search(line) for pat in _SKIP_PATTERNS):
                        i += 1
                        continue
                    notebook_imports.append(line)
                    if line not in imports_to_notebooks:
                        imports_to_notebooks[line] = []
                    if rel_path not in imports_to_notebooks[line]:
                        imports_to_notebooks[line].append(rel_path)
                i += 1

        if notebook_imports:
            notebook_to_imports[rel_path] = notebook_imports

    return imports_to_notebooks, notebook_to_imports


def categorize_import(import_line: str) -> str:
    """Classify an import as senselab, third_party, stdlib, or platform_specific."""
    # Extract the top-level module name
    match = re.match(r"^(?:from\s+(\S+)\s+import|import\s+(\S+))", import_line)
    if not match:
        return "third_party"

    module = (match.group(1) or match.group(2)).split(".")[0]

    if module == "senselab":
        return "senselab"

    if module in _STDLIB_MODULES:
        return "stdlib"

    # Check if any skip pattern matches (platform-specific)
    if any(pat.search(import_line) for pat in _SKIP_PATTERNS):
        return "platform_specific"

    return "third_party"


def time_single_import(import_line: str, timeout: int = 120) -> tuple[float, str, str]:
    """Time a single import in an isolated subprocess.

    Returns:
        (wall_clock_seconds, status, error_message)
        status is one of: "success", "failed", "skipped"
    """
    script = f"import time as _t; _s = _t.perf_counter(); {import_line}; print(_t.perf_counter() - _s)"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            elapsed = float(result.stdout.strip())
            return (elapsed, "success", "")
        else:
            # Import failed
            err = result.stderr.strip().split("\n")[-1] if result.stderr.strip() else "unknown error"
            return (0.0, "failed", err)
    except subprocess.TimeoutExpired:
        return (float(timeout), "failed", f"timeout after {timeout}s")
    except (ValueError, IndexError) as e:
        return (0.0, "failed", f"parse error: {e}")


def profile_import_dependencies(import_line: str, timeout: int = 120) -> list[dict]:
    """Run -X importtime for a single import and parse the output.

    Returns a list of dicts with keys: module, self_time_us, cumulative_time_us, depth
    sorted by self_time_us descending.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-X", "importtime", "-c", import_line],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return []

    entries: list[dict] = []
    for line in result.stderr.splitlines():
        # Format: "import time:   self [us] |      cum [us] |   module_name"
        match = re.match(
            r"^import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\s*)(\S+)",
            line,
        )
        if match:
            self_us = int(match.group(1))
            cum_us = int(match.group(2))
            indent = len(match.group(3))
            module = match.group(4)
            depth = indent // 2  # Each nesting level is 2 spaces
            entries.append(
                {
                    "module": module,
                    "self_time_us": self_us,
                    "cumulative_time_us": cum_us,
                    "depth": depth,
                }
            )

    # Sort by self_time descending, keep top 15
    entries.sort(key=lambda e: e["self_time_us"], reverse=True)
    return entries[:15]


def time_tutorial_imports(import_lines: list[str], timeout: int = 120) -> tuple[float, str]:
    """Time all imports for a tutorial as a single block in one subprocess.

    Returns:
        (total_seconds, status) where status is "success" or "failed"
    """
    # Build a script that runs all imports sequentially, filtering out failures
    escaped_lines = "\n".join(import_lines)
    script = f"import time as _t\n_s = _t.perf_counter()\n{escaped_lines}\nprint(_t.perf_counter() - _s)\n"
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            elapsed = float(result.stdout.strip().split("\n")[-1])
            return (elapsed, "success")
        else:
            return (0.0, "failed")
    except subprocess.TimeoutExpired:
        return (float(timeout), "failed")
    except (ValueError, IndexError):
        return (0.0, "failed")


def generate_report(
    ranked_results: list[dict],
    tutorial_summaries: list[dict],
    breakdowns: dict[str, list[dict]],
    threshold: float,
    output_path: Path,
) -> None:
    """Generate the Markdown report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S UTC")
    lines: list[str] = []

    # Header
    lines.append("# Import Profile Report")
    lines.append(f"\n**Generated**: {now}")
    lines.append(f"**Threshold**: {threshold}s")
    lines.append(f"**Total distinct imports**: {len(ranked_results)}")

    bottleneck_count = sum(1 for r in ranked_results if r["is_bottleneck"])
    failed_count = sum(1 for r in ranked_results if r["status"] == "failed")
    skipped_count = sum(1 for r in ranked_results if r["status"] == "skipped")
    lines.append(f"**Bottlenecks (>{threshold}s)**: {bottleneck_count}")
    lines.append(f"**Failed**: {failed_count}")
    lines.append(f"**Skipped**: {skipped_count}")

    # Section 1: Ranked Imports
    lines.append("\n## Ranked Imports (slowest first)")
    lines.append("")
    lines.append("| # | Import | Time (s) | Category | Status | Bottleneck |")
    lines.append("|---|--------|----------|----------|--------|------------|")
    for i, r in enumerate(ranked_results, 1):
        import_display = r["raw_line"]
        if len(import_display) > 80:
            import_display = import_display[:77] + "..."
        bottleneck_flag = "**YES**" if r["is_bottleneck"] else ""
        lines.append(
            f"| {i} | `{import_display}` | {r['wall_clock_seconds']:.2f} "
            f"| {r['category']} | {r['status']} | {bottleneck_flag} |"
        )

    # Section 2: Per-Tutorial Summary
    if tutorial_summaries:
        lines.append("\n## Per-Tutorial Summary")
        lines.append("")
        lines.append("| Tutorial | Import Time (s) | # Imports | Status |")
        lines.append("|----------|----------------|-----------|--------|")
        for t in tutorial_summaries:
            lines.append(f"| {t['display_name']} | {t['total_seconds']:.2f} | {t['import_count']} | {t['status']} |")

    # Section 3: Dependency Breakdowns
    if breakdowns:
        lines.append("\n## Dependency Breakdowns")
        lines.append("")
        lines.append("For each bottleneck import, the top transitive dependencies by self-time:")
        for import_line, deps in breakdowns.items():
            import_display = import_line
            if len(import_display) > 80:
                import_display = import_display[:77] + "..."
            # Find the total time for this import
            total_time = next(
                (r["wall_clock_seconds"] for r in ranked_results if r["raw_line"] == import_line),
                0.0,
            )
            lines.append(f"\n### `{import_display}` ({total_time:.2f}s)")
            lines.append("")
            if deps:
                lines.append("| Module | Self Time (ms) | Cumulative (ms) |")
                lines.append("|--------|---------------|-----------------|")
                for d in deps:
                    lines.append(
                        f"| {d['module']} | {d['self_time_us'] / 1000:.1f} | {d['cumulative_time_us'] / 1000:.1f} |"
                    )
            else:
                lines.append("*No importtime data available*")

    # Section 4: Skipped/Failed
    problem_imports = [r for r in ranked_results if r["status"] in ("failed", "skipped")]
    if problem_imports:
        lines.append("\n## Skipped & Failed Imports")
        lines.append("")
        lines.append("| Import | Status | Error |")
        lines.append("|--------|--------|-------|")
        for r in problem_imports:
            import_display = r["raw_line"]
            if len(import_display) > 80:
                import_display = import_display[:77] + "..."
            error = r.get("error_message", "")
            if len(error) > 60:
                error = error[:57] + "..."
            lines.append(f"| `{import_display}` | {r['status']} | {error} |")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    print(f"=== Import Profile Tool ===")
    print(f"Tutorials dir: {args.tutorials_dir}")
    print(f"Threshold: {args.threshold}s")
    print(f"Output: {args.output}")
    print()

    # Phase 1: Extract imports
    print("Extracting imports from notebooks...")
    imports_to_notebooks, notebook_to_imports = extract_imports_from_notebooks(args.tutorials_dir)
    distinct_imports = list(imports_to_notebooks.keys())
    print(f"  Found {len(distinct_imports)} distinct imports across {len(notebook_to_imports)} notebooks")
    print()

    # Phase 2: Time each distinct import individually
    print(f"Timing {len(distinct_imports)} distinct imports (each in isolated subprocess)...")
    ranked_results: list[dict] = []
    for i, import_line in enumerate(distinct_imports, 1):
        category = categorize_import(import_line)
        # Skip stdlib imports (they're fast and uninteresting)
        if category == "stdlib":
            ranked_results.append(
                {
                    "raw_line": import_line,
                    "category": category,
                    "wall_clock_seconds": 0.0,
                    "status": "skipped",
                    "error_message": "stdlib (skipped)",
                    "is_bottleneck": False,
                    "source_notebooks": imports_to_notebooks[import_line],
                }
            )
            print(f"  [{i}/{len(distinct_imports)}] SKIP (stdlib): {import_line}")
            continue

        print(f"  [{i}/{len(distinct_imports)}] Timing: {import_line}", end="", flush=True)
        elapsed, status, error = time_single_import(import_line)
        is_bottleneck = status == "success" and elapsed >= args.threshold
        ranked_results.append(
            {
                "raw_line": import_line,
                "category": category,
                "wall_clock_seconds": elapsed,
                "status": status,
                "error_message": error,
                "is_bottleneck": is_bottleneck,
                "source_notebooks": imports_to_notebooks[import_line],
            }
        )
        if status == "success":
            flag = " ** BOTTLENECK **" if is_bottleneck else ""
            print(f" -> {elapsed:.2f}s{flag}")
        else:
            print(f" -> FAILED: {error}")

    # Sort by wall_clock_seconds descending
    ranked_results.sort(key=lambda r: r["wall_clock_seconds"], reverse=True)

    # Summary
    total = len(ranked_results)
    bottlenecks = sum(1 for r in ranked_results if r["is_bottleneck"])
    failed = sum(1 for r in ranked_results if r["status"] == "failed")
    skipped = sum(1 for r in ranked_results if r["status"] == "skipped")
    print(f"\n  Summary: {total} imports, {bottlenecks} bottlenecks, {failed} failed, {skipped} skipped")
    print()

    # Phase 3: Dependency breakdowns for bottleneck imports
    bottleneck_imports = [r for r in ranked_results if r["is_bottleneck"]]
    breakdowns: dict[str, list[dict]] = {}
    if bottleneck_imports:
        print(f"Profiling dependency breakdowns for {len(bottleneck_imports)} bottleneck imports...")
        for r in bottleneck_imports:
            import_line = r["raw_line"]
            print(f"  Profiling: {import_line}", end="", flush=True)
            deps = profile_import_dependencies(import_line)
            breakdowns[import_line] = deps
            print(f" -> {len(deps)} top dependencies captured")
        print()

    # Phase 4: Per-tutorial aggregate timing
    print(f"Timing aggregate imports for {len(notebook_to_imports)} tutorials...")
    tutorial_summaries: list[dict] = []
    for nb_path, imports in sorted(notebook_to_imports.items()):
        display_name = Path(nb_path).stem
        print(f"  [{display_name}]", end="", flush=True)
        total_sec, status = time_tutorial_imports(imports)
        tutorial_summaries.append(
            {
                "file_path": nb_path,
                "display_name": display_name,
                "total_seconds": total_sec,
                "import_count": len(imports),
                "status": status,
            }
        )
        if status == "success":
            print(f" -> {total_sec:.2f}s ({len(imports)} imports)")
        else:
            print(f" -> FAILED")

    # Sort tutorials by total time descending
    tutorial_summaries.sort(key=lambda t: t["total_seconds"], reverse=True)
    print()

    # Phase 5: Generate report
    print(f"Writing report to {args.output}...")
    generate_report(ranked_results, tutorial_summaries, breakdowns, args.threshold, args.output)
    print("Done!")

    # Print top 5 bottlenecks
    if bottleneck_imports:
        print(f"\n=== Top Bottlenecks (>{args.threshold}s) ===")
        for r in bottleneck_imports[:10]:
            print(f"  {r['wall_clock_seconds']:.2f}s  {r['raw_line']}")

    # Print top 5 slowest tutorials
    if tutorial_summaries:
        print(f"\n=== Top 5 Slowest Tutorials ===")
        for t in tutorial_summaries[:5]:
            print(f"  {t['total_seconds']:.2f}s  {t['display_name']} ({t['import_count']} imports)")


if __name__ == "__main__":
    main()
