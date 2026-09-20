"""Apply one mutation at a time, run the tests, revert, and report whether it was caught."""

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
AIRWAY = "src/senselab/audio/workflows/triage/nodes/airway.py"
BRANCHES = "src/senselab/audio/workflows/triage/nodes/branches.py"
AIRWAY_TESTS = "src/tests/audio/workflows/triage/nodes/airway_test.py"
BRANCH_TESTS = "src/tests/audio/workflows/triage/nodes/branches_test.py"

GUARD = """    if envelope is None or not windows:
        return instrument_absent(*absent_instruments(envelope, windows))
"""
GUARD_OLD = """    if envelope is None:
        return instrument_absent("energy_envelope")
"""

MUTANTS = [
    (
        "C1 event-series guard drops the classifier-window check",
        AIRWAY,
        GUARD,
        GUARD_OLD,
        1,
        AIRWAY_TESTS,
    ),
    (
        "C2 alternation guard drops the classifier-window check",
        AIRWAY,
        GUARD,
        GUARD_OLD,
        2,
        AIRWAY_TESTS,
    ),
    (
        "C3 detect guard drops the classifier-window check",
        AIRWAY,
        GUARD,
        GUARD_OLD,
        3,
        AIRWAY_TESTS,
    ),
    (
        "C4 the absence records only its first derivative",
        AIRWAY,
        "absent=list(names)",
        "absent=list(names)[:1]",
        1,
        AIRWAY_TESTS,
    ),
    (
        "B1 the coverage gate is reinstated at 0.5",
        AIRWAY,
        "    return Result(UNDETERMINED, components, findings)\n\n\n# ---",
        "    return Result(bool(coverage is not None and coverage >= 0.5), components, findings)\n\n\n# ---",
        1,
        AIRWAY_TESTS,
    ),
    (
        "B2 the denominator reverts to the whole stream",
        AIRWAY,
        "    if expectation.declared_duration_s is not None:\n"
        "        return float(expectation.declared_duration_s), DECLARED_EXTENT\n",
        "",
        1,
        AIRWAY_TESTS,
    ),
    (
        "B3 the fraction never reaches the report detail",
        AIRWAY,
        '        COVERAGE_FRACTION: None if coverage is None else coverage.evidence.get("value"),',
        "        COVERAGE_FRACTION: None,",
        1,
        AIRWAY_TESTS,
    ),
    (
        "A1 sounds_like reads a flat union of every classifier's spellings",
        BRANCHES,
        '        wanted = label_set.get(str(window.attributes.get("classifier")))',
        "        wanted = [name for names in label_set.values() for name in names]",
        1,
        [AIRWAY_TESTS, BRANCH_TESTS],
    ),
    (
        "A2 the AudioSet side mirrors the HeAR heads instead of resolving them",
        AIRWAY,
        "        audioset = {name for head in heads for name in corroborating.get(head, frozenset())}",
        "        audioset = set(heads)",
        1,
        AIRWAY_TESTS,
    ),
    (
        "A3 decided_label_sets ignores which classifier decided",
        AIRWAY,
        "                if str(label) in resolved.by_classifier().get(classifier, ()):",
        "                if any(str(label) in names for names in resolved.by_classifier().values()):",
        1,
        AIRWAY_TESTS,
    ),
    (
        "A4 the coverage pattern reads the AudioSet side against the HeAR sidecar",
        AIRWAY,
        "    labels = label_sets_by_classifier(params).get(kind, LabelSet((), ())).hear",
        "    labels = label_sets_by_classifier(params).get(kind, LabelSet((), ())).yamnet",
        1,
        AIRWAY_TESTS,
    ),
]


def nth_replace(text: str, old: str, new: str, n: int) -> str:
    """Replace only the nth occurrence.

    Args:
        text: The source.
        old: What to replace.
        new: What to put there.
        n: Which occurrence, 1-based.

    Returns:
        The mutated source.

    Raises:
        AssertionError: If there are fewer than n occurrences.
    """
    parts = text.split(old)
    assert len(parts) > n, f"only {len(parts) - 1} occurrence(s) of {old[:50]!r}"
    return old.join(parts[:n]) + new + old.join(parts[n:])


def main() -> None:
    """Run every mutant and print the table."""
    results = []
    for name, rel, old, new, nth, tests in MUTANTS:
        path = ROOT / rel
        original = path.read_text()
        try:
            path.write_text(nth_replace(original, old, new, nth))
            targets = tests if isinstance(tests, list) else [tests]
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", *targets],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            caught = proc.returncode != 0
            tail = [line for line in proc.stdout.splitlines() if "failed" in line or "passed" in line]
            results.append({"mutant": name, "caught": caught, "summary": tail[-1][-90:] if tail else "?"})
        finally:
            path.write_text(original)
        print(f"{'CAUGHT ' if results[-1]['caught'] else 'SURVIVED'}  {name}", flush=True)
    print()
    print(json.dumps(results, indent=1))
    survivors = [r["mutant"] for r in results if not r["caught"]]
    print("\nsurvivors:", survivors or "none")


if __name__ == "__main__":
    main()
