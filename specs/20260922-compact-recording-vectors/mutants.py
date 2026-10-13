"""Mutate the encoder and the null handling, and report which mutations the suite catches.

    uv run --no-sync python specs/20260922-compact-recording-vectors/mutants.py

Round-tripping a known vector is not enough on its own: a mutation that drops a measure, or writes
zero where null belongs, round-trips perfectly. Each mutation below is a plausible wrong version of
the contract in `schema.md`; a mutation the suite does not kill is a hole in the tests, not a
finding about the data.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src" / "senselab" / "audio" / "workflows" / "triage" / "recording_vectors.py"
TESTS = ROOT / "src" / "tests" / "audio" / "workflows" / "triage" / "recording_vectors_test.py"

MUTATIONS: tuple[tuple[str, str, str], ...] = (
    (
        "drop a measure from the enumerated set",
        '    "phonation_onset_to_offset_s",\n',
        "",
    ),
    (
        "write zero instead of null for an unmeasured scalar",
        'row[f"m_{name}"] = sum(numbers) / len(numbers) if numbers else None',
        'row[f"m_{name}"] = sum(numbers) / len(numbers) if numbers else 0.0',
    ),
    (
        "report a never-scanned recording as clean rather than null",
        'row["pii_findings_n"] = len(pii_entities) if scanned else None',
        'row["pii_findings_n"] = len(pii_entities)',
    ),
    (
        "write an empty block where the producing node did not run",
        '        row["spans"] = row["span_labels"] = row["span_label_name"] = row["span_squim"] = None',
        '        row["spans"] = row["span_labels"] = row["span_squim"] = b""\n        row["span_label_name"] = []',
    ),
    (
        "pack the blocks big-endian",
        '\n    packer = struct.Struct("<" + layout)',
        '\n    packer = struct.Struct(">" + layout)',
    ),
    (
        "use 65536 as the time full scale",
        "TIME_SCALE = 65535",
        "TIME_SCALE = 65536",
    ),
    (
        "let an out-of-range value wrap instead of clamping",
        "return int(min(255, max(0, round((value - low) / (high - low) * 255))))",
        "return int(round((value - low) / (high - low) * 255)) % 256",
    ),
    (
        "swap the pesq and si_sdr ranges",
        'SQUIM_RANGES = (("stoi", (0.0, 1.0)), ("pesq", (1.0, 4.5)), ("si_sdr", (-10.0, 30.0)))',
        'SQUIM_RANGES = (("stoi", (0.0, 1.0)), ("pesq", (-10.0, 30.0)), ("si_sdr", (1.0, 4.5)))',
    ),
    (
        "decimate the envelope by mean instead of max",
        'row["env_dbfs"] = encode_trace(envelope, *ENVELOPE_DBFS_RANGE, how="max")',
        'row["env_dbfs"] = encode_trace(envelope, *ENVELOPE_DBFS_RANGE, how="mean")',
    ),
    (
        "decimate the continuity trace by max instead of mean",
        'row["continuity"] = encode_trace(continuity, *CONTINUITY_RANGE, how="mean")',
        'row["continuity"] = encode_trace(continuity, *CONTINUITY_RANGE, how="max")',
    ),
    (
        "emit 255 trace points instead of 256",
        "TRACE_POINTS = 256",
        "TRACE_POINTS = 255",
    ),
    (
        "index the label and squim blocks against the unfiltered span list",
        "    rowed = [e for e in general if _row_code(e) in SPAN_ROWS]\n    return rowed, len(general) - len(rowed)",
        "    rowed = [e for e in general if _row_code(e) in SPAN_ROWS]\n    return general, len(general) - len(rowed)",
    ),
    (
        "order the ASR lane by store order rather than by consensus index",
        'key=lambda e: int(e.attributes.get("index", 0)),',
        "key=lambda e: 0,",
    ),
    (
        "count a repeated measurement once instead of reporting how many readings there were",
        'row[f"m_{name}_n"] = len(values)',
        'row[f"m_{name}_n"] = min(1, len(values))',
    ),
    (
        "read invalidated entities as live",
        "        return [e for e in self.entities if e.prov_type == prov_type and e.id not in self.invalidated]",
        "        return [e for e in self.entities if e.prov_type == prov_type]",
    ),
    (
        "put the verdict column before the task column",
        '        pa.field("task", pa.string()),\n        pa.field("verdict", pa.string()),',
        '        pa.field("verdict", pa.string()),\n        pa.field("task", pa.string()),',
    ),
    (
        "drop the words from the ASR lane, keeping extents only",
        'row["asr_word_text"] = [str(e.attributes.get("text") or "") for e in words]',
        'row["asr_word_text"] = ["" for e in words]',
    ),
    (
        "drop the PII category, keeping the extent only",
        'row["pii_category"] = [str(e.attributes.get("category") or "") for e in pii_entities]',
        'row["pii_category"] = ["" for e in pii_entities]',
    ),
)


def run_suite() -> tuple[bool, str]:
    """Run the recording-vector tests once.

    Returns:
        Whether they passed, and the last line of pytest's summary.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-header", "-x"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    lines = [line for line in result.stdout.splitlines() if "passed" in line or "failed" in line]
    return result.returncode == 0, (lines[-1] if lines else "no summary").strip()


def main() -> int:
    """Apply each mutation in turn and report whether the suite killed it.

    Returns:
        0 when every mutation was killed, 1 otherwise.
    """
    original = MODULE.read_text()
    passed, summary = run_suite()
    print(f"baseline: {'pass' if passed else 'FAIL'} — {summary}\n")
    if not passed:
        return 1
    survivors: list[str] = []
    try:
        for description, before, after in MUTATIONS:
            if original.count(before) != 1:
                print(f"SKIPPED  {description} — its anchor appears {original.count(before)} times")
                survivors.append(description)
                continue
            MODULE.write_text(original.replace(before, after))
            still_passing, summary = run_suite()
            print(f"{'SURVIVED' if still_passing else 'killed  '} {description} — {summary}")
            if still_passing:
                survivors.append(description)
    finally:
        MODULE.write_text(original)
    print(f"\n{len(MUTATIONS) - len(survivors)}/{len(MUTATIONS)} killed")
    for description in survivors:
        print(f"  survived: {description}")
    return 0 if not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
