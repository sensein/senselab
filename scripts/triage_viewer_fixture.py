"""Write a synthetic recording-vectors parquet for the viewer's browser tests.

    uv run python scripts/triage_viewer_fixture.py --out artifacts/viewer_e2e

Every value is generated from a fixed seed through the producer's own :func:`schema` and
:func:`to_table`, so the file is schema-conformant by construction and byte-stable across hosts.
No byte of it comes from a recording: there is no transcript, no PII extent, and the one binary
block it writes is a generated span layout.

The distributions are chosen so the facet panel has something to separate -- a long-tailed task
vocabulary, a three-term verdict, gates that pass, fail and stay undetermined, and columns whose
nulls are a large share of the corpus. The counts a test asserts are printed as JSON beside it.
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
    TIME_SCALE,
    encode_records,
    to_table,
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
RELEASES = [("not_assessed", 60), ("releasable", 26), ("withheld", 9), ("nothing_to_redact", 5)]
RELEASE_GROUNDS = [("pii_detected", 30), ("scan_declined", 40), (None, 30)]
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


def _uuid(rng: random.Random) -> str:
    """A v4-shaped UUID from the seeded generator, so the fixture is byte-stable."""
    h = "%032x" % rng.getrandbits(128)
    return "-".join((h[:8], h[8:12], "4" + h[13:16], h[16:20], h[20:32]))


def _bids_ids(rng: random.Random, participants: int) -> list[tuple[str, str]]:
    """One ``(sub-<uuid>, ses-<uuid>)`` pair per participant, 40 characters each.

    The first two participants are forced to share their first four hex digits, so a test can
    show that a four-character key would collide where the eight-character key the axis draws
    does not.

    Args:
        rng: The generator.
        participants: How many pairs to make.

    Returns:
        The pairs, in participant order.
    """
    out = [("sub-" + _uuid(rng), "ses-" + _uuid(rng)) for _ in range(participants)]
    if participants > 1:
        first = out[0][0]
        out[1] = ("sub-" + first[4:8] + out[1][0][8:], out[1][1])
    return out


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
    ids = _bids_ids(rng, participants)
    for p in range(participants):
        participant, session = ids[p]
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
                "flags_n": 0 if verdict == "pass" else rng.randint(1, 4),
                "pii_findings_n": None if rng.random() < 0.30 else rng.choice([0, 0, 0, 1, 2, 5]),
                "schema_version": SCHEMA_VERSION,
                "malformed_store_lines": 0,
            }
            row["duration_conditioned_s"] = row["duration_s"]
            row["time_scale_s"] = row["duration_s"]
            # The residual decomposition's levels. enhanced_over_residual_rms_db is a default
            # axis, so the fixture must carry a spread one rather than a column of nulls.
            residual_rms = round(rng.gauss(-54.0, 7.0), 3)
            over_residual = round(rng.gauss(26.0, 16.0), 3)
            row["residual_rms_dbfs"] = residual_rms
            row["residual_peak_dbfs"] = round(residual_rms + abs(rng.gauss(11.0, 3.0)), 3)
            row["enhanced_over_residual_rms_db"] = over_residual
            row["enhanced_rms_dbfs"] = round(residual_rms + over_residual, 3)
            row["enhanced_over_residual_rms_fitted_db"] = round(over_residual + rng.gauss(0.0, 2.0), 3)
            row["flag_nodes"] = [] if verdict == "pass" else rng.sample(FLAG_NODES, rng.randint(1, 2))

            failed: list[str] = []
            for gate in GATE_NAMES:
                applied = rng.random() < 0.72
                if not applied:
                    continue
                bound = {"train_min_s": 1.0, "coverage_min": 0.5, "dominant_speaker_share_min": 0.8, "items_min": 3.0}[
                    gate
                ]
                row[f"gate_{gate}_bound"] = bound
                if rng.random() < 0.15:
                    # Applied, and unanswerable: the bound stands, the reading does not.
                    row[f"gate_{gate}_passed"] = "undetermined"
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
            # One span block, so the recording view has a lane with hover targets. The offsets
            # are generated, not read off a recording: no waveform, transcript or PII is written.
            spans = []
            for r_i in range(len(SPAN_ROWS)):
                t0 = rng.uniform(0.0, 0.6)
                t1 = min(1.0, t0 + rng.uniform(0.05, 0.3))
                spans.append((r_i, round(t0 * TIME_SCALE), round(t1 * TIME_SCALE)))
            row["spans"] = encode_records(spans, SPANS_LAYOUT)
            row["spans_unrowed_n"] = 0

            row["gate_undetermined_n"] = sum(1 for g in GATE_NAMES if row.get(f"gate_{g}_passed") == "undetermined")
            row["gate_failed_names"] = sorted(failed)
            row["gate_flagged_names"] = []
            out.append(row)
    return out


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
