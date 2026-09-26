"""Write a synthetic recording-vectors parquet for the viewer's browser tests.

    uv run python scripts/triage_viewer_fixture.py --out artifacts/viewer_e2e

Every value is generated from a fixed seed through the producer's own :func:`schema` and
:func:`to_table`, so the file is schema-conformant by construction and byte-stable across hosts.
No byte of it comes from a recording: there is no transcript, no PII extent and no binary block.

The distributions are chosen so the facet panel has something to separate -- a long-tailed task
vocabulary, a three-term verdict, gates that pass, fail and stay undetermined, and columns whose
nulls are a large share of the corpus. The counts a test asserts are printed as JSON beside it.

**Every default axis must be populated here, or the page has a column it cannot paint.** The one
file serves both browser specs: ``facets.e2e`` over the panel and ``viewer.e2e`` over the rail and
the recording panel. They asserted against two fixtures until 2026-09-25, and the second one's
writer did not exist, so six of its checks could not run from a clean tree at all.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

from senselab.audio.workflows.triage.recording_vectors import (
    SCHEMA_VERSION,
    SPAN_ROWS,
    SPANS_LAYOUT,
    encode_records,
    quantise_time,
    to_table,
)
from senselab.audio.workflows.triage.vocabulary import (
    NO_TRANSCRIPT,
    NOTHING_BEYOND_STIMULUS,
    SCAN_FOUND_NOTHING,
)

PRIVATE_FILE_MODE = 0o600
COMPRESSION = "zstd"
ROW_GROUP_SIZE = 1024
SEED = 20260924

PARTICIPANTS = 120
PER_PARTICIPANT = 20

# name, weight. A long tail, because `task` is the facet with the most levels the panel must cope
# with and a flat vocabulary would not exercise the top-K cap.
TASKS = [
    ("story-recall", 26),
    ("free-speech", 20),
    ("maximum-phonation", 14),
    ("diadochokinesis-pa", 11),
    ("picture-description", 9),
    ("glide-up", 6),
    ("word-color-stroop", 5),
    ("harvard-sentences-list", 4),
    ("cough", 3),
    ("breath-hold", 2),
]
FAMILIES = ["speech", "voice", "airway", "ddk"]
VERDICTS = [("pass", 78), ("flag", 21), ("discard", 1)]
RELEASES = [
    ("release_without_redaction", 43),
    ("not_assessed", 31),
    ("release_with_redaction", 19),
    ("withheld", 7),
]
RELEASE_GROUNDS = [
    (SCAN_FOUND_NOTHING, 25),
    (NOTHING_BEYOND_STIMULUS, 30),
    (NO_TRANSCRIPT, 30),
    (None, 15),
]
ROUTE_STATES = [("routed", 70), ("declined", 22), ("unavailable", 5), ("empty", 3)]
CONFORMANCE = [("true", 55), ("false", 12), ("undetermined", 33)]
GATE_NAMES = ["train_min_s", "coverage_min", "dominant_speaker_share_min", "items_min"]
FLAG_NODES = ["VERDICT", "SPEECH", "VOICE", "AIRWAY", "REDACT"]


def _pick(rng: random.Random, weighted: Sequence[tuple[str | None, int]]) -> str | None:
    """One value from a weighted vocabulary.

    Args:
        rng: The generator.
        weighted: ``(value, weight)`` pairs; the value may be None for a null.

    Returns:
        The chosen value.
    """
    return rng.choices([v for v, _ in weighted], weights=[w for _, w in weighted], k=1)[0]


def rows(seed: int = SEED, participants: int = PARTICIPANTS) -> list[dict[str, object]]:
    """Every synthetic row, in a stable order.

    Args:
        seed: The generator's seed.
        participants: How many participants to write, at :data:`PER_PARTICIPANT` recordings each.

    Returns:
        One dict per recording, on the producer's schema.
    """
    rng = random.Random(seed)
    out: list[dict[str, object]] = []
    for p in range(participants):
        participant = f"sub-f{p:04d}"
        session = f"ses-{rng.choice(['s0', 's1'])}"
        for k in range(PER_PARTICIPANT):
            task = _pick(rng, TASKS)
            verdict = _pick(rng, VERDICTS)
            declined = rng.random() < 0.38
            row: dict[str, object] = {
                "stem": f"{participant}_{session}_task-{task}_{k:02d}",
                "participant": participant,
                "session": session,
                "task": task,
                "declared_family": rng.choice(FAMILIES),
                "verdict": verdict,
                "release": _pick(rng, RELEASES),
                "release_ground": _pick(rng, RELEASE_GROUNDS),
                "grounds": "content_absent" if verdict == "discard" else None,
                "route_state": _pick(rng, ROUTE_STATES),
                "route_airway": _pick(rng, [("routed", 40), ("declined", 55), ("unavailable", 5)]),
                "route_speech": _pick(rng, [("routed", 70), ("declined", 26), ("unavailable", 4)]),
                "route_voice": _pick(rng, [("routed", 36), ("declined", 60), ("unavailable", 4)]),
                "conformance_airway": None if declined else _pick(rng, CONFORMANCE),
                "conformance_speech": None if rng.random() < 0.30 else _pick(rng, CONFORMANCE),
                "conformance_voice": None if rng.random() < 0.64 else _pick(rng, CONFORMANCE),
                "conformance_quality": _pick(rng, [("undetermined", 85), ("true", 15)]),
                "duration_s": round(rng.lognormvariate(2.0, 0.9), 3),
                # A default axis, so it must be populated or the page has a column it cannot paint.
                # Null on the share of recordings enhancement left nothing to compare against.
                "enhanced_over_residual_rms_db": None if rng.random() < 0.12 else round(rng.gauss(9.0, 4.5), 3),
                "flags_n": 0 if verdict == "pass" else rng.randint(1, 4),
                "pii_findings_n": None if rng.random() < 0.30 else rng.choice([0, 0, 0, 1, 2, 5]),
                "schema_version": SCHEMA_VERSION,
                "malformed_store_lines": 0,
            }
            row["duration_conditioned_s"] = row["duration_s"]
            row["time_scale_s"] = row["duration_s"]
            row["flag_nodes"] = [] if verdict == "pass" else rng.sample(FLAG_NODES, rng.randint(1, 2))

            failed: list[str] = []
            undetermined: list[str] = []
            for gate in GATE_NAMES:
                applied = rng.random() < 0.72
                if not applied:
                    continue
                bound = {"train_min_s": 1.0, "coverage_min": 0.5, "dominant_speaker_share_min": 0.8, "items_min": 3.0}[
                    gate
                ]
                row[f"gate_{gate}_bound"] = bound
                # An applied gate whose reading nobody took answers UNDETERMINED: the bound stands,
                # the reading is absent, and the outcome is neither pass nor fail. The page must
                # carry all three terms, so the fixture has to produce all three.
                if rng.random() < 0.18:
                    row[f"gate_{gate}_passed"] = "undetermined"
                    undetermined.append(gate)
                    continue
                # A third of the applied readings are a genuine 0.0, which is the value the page
                # must place on the band rather than on the absent rail.
                reading = 0.0 if rng.random() < 0.33 else round(rng.uniform(0.0, 2.0 * bound), 4)
                passed = "true" if reading >= bound else "false"
                row[f"gate_{gate}"] = reading
                row[f"gate_{gate}_passed"] = passed
                if passed == "false":
                    failed.append(gate)
            row["gate_group"] = task
            row["gate_family"] = row["declared_family"]
            row["gate_node"] = "VERDICT"
            row["gate_applied_n"] = sum(1 for g in GATE_NAMES if f"gate_{g}_passed" in row)
            row["gate_flagging_n"] = 0
            row["gate_failed_n"] = len(failed)
            row["gate_undetermined_n"] = len(undetermined)
            row["gate_failed_names"] = sorted(failed)
            row["gate_flagged_names"] = []
            row["spans"], row["spans_unrowed_n"] = _spans(rng, row["duration_s"])
            out.append(row)
    return out


def _spans(rng: random.Random, duration_s: float) -> tuple[bytes, int]:
    """A synthetic span block, so the recording panel has lanes to draw and hits to report.

    The block is written through the producer's own encoder against the producer's own row codes,
    so it decodes under the shipped reader or it does not decode at all. The extents are generated,
    not measured: no byte of this comes from a recording.

    Args:
        rng: The generator's source of randomness.
        duration_s: The recording's length, which times quantise against.

    Returns:
        The packed block and how many spans carry no row code.
    """
    records: list[tuple[int, int, int]] = []
    for row_index in range(len(SPAN_ROWS)):
        for _ in range(rng.randint(1, 3)):
            start = rng.uniform(0.0, max(0.1, duration_s * 0.9))
            end = min(duration_s, start + rng.uniform(0.05, duration_s * 0.2))
            records.append((row_index, quantise_time(start, duration_s), quantise_time(end, duration_s)))
    records.sort(key=lambda record: (record[0], record[1]))
    return encode_records(records, SPANS_LAYOUT), rng.choice([0, 0, 0, 1])


def counts(built: list[dict[str, object]]) -> dict[str, object]:
    """The numbers a browser test can assert against without recomputing them itself.

    Args:
        built: The rows.

    Returns:
        A summary: the row count, and per-column value counts for the faceted columns.
    """
    faceted = [
        "task",
        "verdict",
        "release",
        "release_ground",
        "route_state",
        "conformance_airway",
        "conformance_speech",
        "gate_train_min_s_passed",
    ]
    summary: dict[str, object] = {"rows": len(built), "schema_version": SCHEMA_VERSION}
    for name in faceted:
        c = Counter("(absent)" if r.get(name) is None else str(r[name]) for r in built)
        summary[name] = dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))
    gate = "gate_train_min_s"
    summary["gate_train_min_s_zeros"] = sum(1 for r in built if r.get(gate) == 0.0)
    summary["gate_train_min_s_nulls"] = sum(1 for r in built if r.get(gate) is None)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Write the fixture and its counts.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("artifacts/viewer_e2e"), help="Where the fixture goes.")
    parser.add_argument("--seed", type=int, default=SEED, help="The generator's seed.")
    parser.add_argument(
        "--rows", type=int, default=PARTICIPANTS * PER_PARTICIPANT, help="Roughly how many rows to write."
    )
    args = parser.parse_args(argv)

    built = rows(args.seed, max(1, round(args.rows / PER_PARTICIPANT)))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "facets_fixture.parquet"
    pq.write_table(to_table(built), path, compression=COMPRESSION, row_group_size=ROW_GROUP_SIZE, write_page_index=True)
    os.chmod(path, PRIVATE_FILE_MODE)
    summary = counts(built)
    (args.out / "facets_fixture.counts.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"{summary['rows']} rows -> {path} ({path.stat().st_size / 1e3:.1f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
