"""Mutation-test the gates-in-VERDICT change: each mutation must make at least one test fail.

Each entry rewrites one line of the implementation into a plausible wrong version of itself,
runs the tests that should notice, and restores the file. A mutation that survives is a
behaviour nothing asserts. The results are in `implementation.md` in this directory.

    python specs/20260921-gates-in-verdict/mutations.py
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRIAGE = ROOT / "src/senselab/audio/workflows/triage"
TESTS = "src/tests/audio/workflows/triage"

MUTATIONS = [
    (
        "M1 absent reading reads as a failure rather than as no answer",
        TRIAGE / "nodes/gates.py",
        "            UNDETERMINED if value is None or bound is None else _passes(value, bound, spec.op)",
        "            False if value is None or bound is None else _passes(value, bound, spec.op)",
        f"{TESTS}/nodes/gates_test.py {TESTS}/nodes/verdict_test.py",
    ),
    (
        "M2 a gate the group does not name is applied anyway",
        TRIAGE / "nodes/gates.py",
        "        if not bounds.names(name):\n            continue",
        "        if False:\n            continue",
        f"{TESTS}/nodes/gates_test.py {TESTS}/nodes/verdict_test.py",
    ),
    (
        "M3 a group that applied no gate reads as a pass",
        TRIAGE / "nodes/gates.py",
        "    if not applied or any(gate.passed == UNDETERMINED for gate in applied):\n"
        "        return UNDETERMINED, applied",
        "    if any(gate.passed == UNDETERMINED for gate in applied):\n        return UNDETERMINED, applied",
        f"{TESTS}/nodes/gates_test.py",
    ),
    (
        "M4 the spread gate compares the wrong way round",
        TRIAGE / "nodes/gates.py",
        '    "f0_spread_max_semitones": GateSpec("carrier_f0_spread_semitones", AT_MOST, float),',
        '    "f0_spread_max_semitones": GateSpec("carrier_f0_spread_semitones", AT_LEAST, float),',
        f"{TESTS}/nodes/gates_test.py {TESTS}/nodes/verdict_test.py {TESTS}/nodes/voice_test.py",
    ),
    (
        "M5 GLIDE is bound by the held vowel's spread",
        TRIAGE / "data/config/default.yaml",
        "        monotone_tolerance_semitones: 1.0  # reversal the sweep may contain and still count as monotone",
        "        f0_spread_max_semitones: 2.0\n"
        "        monotone_tolerance_semitones: 1.0  # reversal the sweep may contain and still count as monotone",
        f"{TESTS}/nodes/gates_test.py {TESTS}/nodes/verdict_test.py",
    ),
    (
        "M6 VERDICT gates an out-of-family report too",
        TRIAGE / "nodes/verdict.py",
        "    reported = next((report for report in reports if report.node == branch and report.in_family), None)",
        "    reported = next((report for report in reports if report.node == branch), None)",
        f"{TESTS}/nodes/verdict_test.py",
    ),
    (
        "M7 VERDICT leaves whatever the branch wrote on the report",
        TRIAGE / "nodes/verdict.py",
        "            conformance=outcome.conformance,",
        "            conformance=report.conformance,",
        f"{TESTS}/nodes/verdict_test.py",
    ),
    (
        "M8 the sustained carrier's readings are never written",
        TRIAGE / "nodes/voice.py",
        "    findings.extend(carrier_readings(carrier, *read_off))",
        "    findings.extend([])",
        f"{TESTS}/nodes/voice_test.py {TESTS}/nodes/verdict_test.py",
    ),
    (
        "M9 DDK writes a repetition reading even when neither instrument could look",
        TRIAGE / "nodes/ddk.py",
        "    if not readable_carrier and not readable_decode:\n        return []",
        "    if False:\n        return []",
        f"{TESTS}/nodes/ddk_test.py",
    ),
    (
        "M10 AIRWAY counts events as a reading even where the label cut is unmeasured",
        TRIAGE / "nodes/airway.py",
        '    if params.gate("score_min") is None:\n        return []',
        "    if False:\n        return []",
        f"{TESTS}/nodes/airway_test.py",
    ),
    (
        "M11 a recall is gated on how long it ran rather than on coverage",
        TRIAGE / "nodes/gates.py",
        "    if pattern is Pattern.FREE_RESPONSE and anti_pattern == VERBATIM_SOURCE:\n"
        "        return RECALL_CONFORMANCE_GATES",
        "    if False:\n        return RECALL_CONFORMANCE_GATES",
        f"{TESTS}/nodes/gates_test.py {TESTS}/nodes/speech_modes_test.py {TESTS}/nodes/speech_test.py",
    ),
    (
        "M12 a branch may write its own conformance onto its report again",
        TRIAGE / "nodes/voice.py",
        "        conformance=UNDETERMINED,",
        "        conformance=True,",
        f"{TESTS}/nodes/voice_test.py",
    ),
    (
        "M13 a family replaces its group's whole mapping instead of overriding key by key",
        TRIAGE / "nodes/gates.py",
        "    for layer in LAYERS:\n"
        "        for name, value in supplied[layer].items():\n"
        "            if name not in resolved:",
        "    for layer in reversed(LAYERS):\n"
        "        for name, value in supplied[layer].items():\n"
        "            if supplied[FAMILY_LAYER] and layer != FAMILY_LAYER:\n"
        "                continue\n"
        "            if True:",
        f"{TESTS}/nodes/gates_test.py",
    ),
    (
        "M14 the layers resolve least-specific-first, so a group beats its family",
        TRIAGE / "nodes/gates.py",
        "LAYERS = (FAMILY_LAYER, GROUP_LAYER, DEFAULT_LAYER)",
        "LAYERS = (DEFAULT_LAYER, GROUP_LAYER, FAMILY_LAYER)",
        f"{TESTS}/nodes/gates_test.py",
    ),
    (
        "M15 a gate is put on the count nobody gave",
        TRIAGE / "nodes/gates.py",
        '    "events_min": GateSpec("airway_events_found", AT_LEAST, int),',
        '    "events_min": GateSpec(TYPICAL_COUNT, AT_LEAST, int),',
        f"{TESTS}/nodes/gates_test.py {TESTS}/nodes/airway_test.py",
    ),
    (
        "M16 a family-specific bound records its group, so it reads as an inherited one",
        TRIAGE / "nodes/gates.py",
        "        if layer == FAMILY_LAYER:\n            return str(self.family)",
        "        if False:\n            return str(self.family)",
        f"{TESTS}/nodes/gates_test.py {TESTS}/nodes/verdict_test.py",
    ),
    (
        "M17 every applied gate records the default layer, whichever one supplied it",
        TRIAGE / "nodes/gates.py",
        "                layer=str(bounds.layer(name)),",
        "                layer=DEFAULT_LAYER,",
        f"{TESTS}/nodes/gates_test.py {TESTS}/nodes/verdict_test.py",
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
