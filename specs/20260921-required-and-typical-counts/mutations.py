"""Mutation-test the required/typical count split: each mutation must make at least one test fail.

Each entry rewrites one line of the implementation into a plausible wrong version of itself, runs
the tests that should notice, and restores the file. A mutation that survives is a behaviour
nothing asserts. The results are in `design.md` in this directory.

    python specs/20260921-required-and-typical-counts/mutations.py
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRIAGE = ROOT / "src/senselab/audio/workflows/triage"
TESTS = "src/tests/audio/workflows/triage"
BRANCHES = f"{TESTS}/nodes/branches_test.py"
DDK = f"{TESTS}/nodes/ddk_test.py"
AIRWAY = f"{TESTS}/nodes/airway_test.py"
SPEECH = f"{TESTS}/nodes/speech_modes_test.py"
GATES = f"{TESTS}/nodes/gates_test.py"

MUTATIONS = [
    (
        "M1 the multi-syllable families declare syllables again, as the retired field did",
        TRIAGE / "nodes/branches.py",
        "        sequence=PATAKA,\n        typical_count=TypicalCount(10, CountUnit.REPETITIONS, DDK_MEDIANS),",
        "        sequence=PATAKA,\n        typical_count=TypicalCount(30, CountUnit.REPETITIONS, DDK_MEDIANS),",
        f"{BRANCHES} {DDK}",
    ),
    (
        "M2 /pa/ carries the underived ten rather than its measured eleven",
        TRIAGE / "nodes/branches.py",
        "        sequence=PA,\n        typical_count=TypicalCount(11, CountUnit.REPETITIONS, DDK_MEDIANS),",
        "        sequence=PA,\n        typical_count=TypicalCount(10, CountUnit.REPETITIONS, DDK_MEDIANS),",
        f"{BRANCHES} {SPEECH}",
    ),
    (
        "M3 a measured median is declared without citing the measurement",
        TRIAGE / "nodes/branches.py",
        'DDK_MEDIANS = "specs/20260817-triage-workflow-dag/measure-distributions.md"',
        'DDK_MEDIANS = ""',
        f"{BRANCHES} {DDK}",
    ),
    (
        "M4 a syllable-repetition row declares its median as a number the instruction gave",
        TRIAGE / "nodes/branches.py",
        "        sequence=PA,\n        typical_count=TypicalCount(11, CountUnit.REPETITIONS, DDK_MEDIANS),",
        "        sequence=PA,\n        required_count=RequiredCount(11, CountUnit.REPETITIONS),",
        f"{BRANCHES} {DDK}",
    ),
    (
        "M5 five breaths becomes a heuristic rather than the instruction's own number",
        TRIAGE / "nodes/branches.py",
        '        label_set="breath",\n        required_count=RequiredCount(5, CountUnit.EVENTS),',
        '        label_set="breath",\n        typical_count=TypicalCount(5, CountUnit.EVENTS, DDK_MEDIANS),',
        f"{BRANCHES} {AIRWAY}",
    ),
    (
        "M6 the median is written under the key a comparator reaches for",
        TRIAGE / "nodes/branches.py",
        '        "typical": typical.median,',
        '        "declared": typical.median,',
        f"{BRANCHES} {DDK}",
    ),
    (
        "M7 the required count is written without the unit it counts in",
        TRIAGE / "nodes/branches.py",
        '    evidence = {"found": found, "required": required.value, "unit": required.unit.value}',
        '    evidence = {"found": found, "required": required.value}',
        f"{BRANCHES} {AIRWAY} {SPEECH}",
    ),
    (
        "M8 a measured median becomes gateable",
        TRIAGE / "nodes/gates.py",
        "UNGATEABLE_READINGS = frozenset({TYPICAL_COUNT})",
        "UNGATEABLE_READINGS: frozenset[str] = frozenset()",
        f"{GATES} {BRANCHES}",
    ),
    (
        "M9 the gate table is assigned around its own refusal",
        TRIAGE / "nodes/gates.py",
        "GATE_SPECS: dict[str, GateSpec] = _gate_specs(",
        "GATE_SPECS: dict[str, GateSpec] = dict(",
        GATES,
    ),
    (
        "M10 a family whose instruction counts writes no count against it",
        TRIAGE / "nodes/airway.py",
        "    findings: list[Finding] = [] if required is None else "
        "[count_against_instruction(required, len(events), *carriers)]",
        "    findings: list[Finding] = []",
        AIRWAY,
    ),
    (
        "M11 the decoded count is compared in syllables against a median in repetitions",
        TRIAGE / "nodes/ddk.py",
        "        findings.append(count_beside_typical(typical, decode.count, *evidence))",
        "        findings.append(count_beside_typical(typical, decode.syllables, *evidence))",
        DDK,
    ),
    (
        "M12 the repetition measurement stops carrying the median beside it",
        TRIAGE / "nodes/ddk.py",
        "            typical_repetitions=None if typical is None else typical.median,",
        "            typical_repetitions=None,",
        DDK,
    ),
    (
        "M13 a required count of zero is admitted",
        TRIAGE / "nodes/branches.py",
        "        if self.value < 1:",
        "        if self.value < 0:",
        BRANCHES,
    ),
    (
        "M14 the speech token count is written against the wrong half of the row",
        TRIAGE / "nodes/speech.py",
        "    if expectation.required_count is not None:",
        "    if expectation.typical_count is not None:",
        SPEECH,
    ),
]


def run(paths: str) -> tuple[bool, str]:
    """Run one test selection.

    Args:
        paths: The pytest targets, space separated.

    Returns:
        Whether it passed, and the summary line.
    """
    completed = subprocess.run(
        ["uv", "run", "--no-sync", "python", "-m", "pytest", *paths.split(), "-q", "--no-header", "--tb=no"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    tail = [line for line in completed.stdout.splitlines() if "passed" in line or "failed" in line or "error" in line]
    return completed.returncode == 0, (tail[-1] if tail else "no summary")


def main() -> int:
    """Apply each mutation, run its tests, and report whether it was caught."""
    caught = 0
    for name, path, before, after, targets in MUTATIONS:
        original = path.read_text()
        if before not in original:
            print(f"SKIPPED  {name}: anchor not found in {path.name}")
            continue
        path.write_text(original.replace(before, after, 1))
        try:
            ok, summary = run(targets)
        finally:
            path.write_text(original)
        caught += 0 if ok else 1
        print(f"{'CAUGHT ' if not ok else 'SURVIVED'} {name}  ({summary})")
    print(f"\n{caught}/{len(MUTATIONS)} caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
