"""The tables in ``airway-flag-grounds.md``, from what the census wrote.

Reads only the JSONL under ``--out``; touches no store, no sidecar and no audio. Each section is
lettered as it is in the document.

Usage::

    python3 airway-flag-grounds-tables.py --out <dir> [--run <run dir>]

``--run`` is needed only for the control contrast, which reads the ``hear_scores`` sidecars of
families that elicit no breathing task.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

SCORE_MIN = 0.2
COVERAGE_MIN = 0.5
COVERAGE_FAMILIES = frozenset({"respiration-and-cough-breath", "respiration-and-cough-v2-breath"})
COUGH_FAMILIES = frozenset({"respiration-and-cough-cough", "respiration-and-cough-v2-hardcough", "voluntary-cough"})
LABEL_SET = {
    "respiration-and-cough-cough": "Cough",
    "respiration-and-cough-v2-hardcough": "Cough",
    "voluntary-cough": "Cough",
    "respiration-and-cough-fivebreaths": "Breathe",
    "respiration-and-cough-v2-threebreathsnose": "Breathe",
    "respiration-and-cough-v2-threebreathsmouth": "Breathe",
    "respiration-and-cough-threequickbreaths": "Breathe",
    "respiration-and-cough-v2-threebreaths": "Breathe",
    "respiration-and-cough-breath": "Breathe",
    "respiration-and-cough-v2-breath": "Breathe",
    "breath-sounds": "Breathe",
}
"""Each in-family row's ``label_set``, from ``AIRWAY_EXPECTATIONS``. Not inferable from the family
name: ``cough`` is a substring of every ``respiration-and-cough-*`` family, breath ones included."""

CONTROL_FAMILIES = (
    "maximum-phonation-time",
    "harvard-sentences-list",
    "prolonged-vowel",
    "free-speech",
    "glides-low-to-high",
    "word-color-stroop",
    "animal-fluency",
    "diadochokinesis-pataka",
    "rainbow-passage",
    "cinderella-story",
)
"""Ten families that elicit no breathing task, for the control contrast."""

HEAR_HEADS = ("Cough", "Baby Cough", "Throat Clear", "Breathe", "Sneeze", "Speech", "Laugh", "Snore")
SUSTAINED = ("respiration-and-cough-breath-1", "respiration-and-cough-breath-2", "respiration-and-cough-v2-breath")
COUNTED = (
    "respiration-and-cough-fivebreaths-1",
    "respiration-and-cough-fivebreaths-2",
    "respiration-and-cough-fivebreaths-3",
    "respiration-and-cough-fivebreaths-4",
    "respiration-and-cough-threequickbreaths-1",
    "respiration-and-cough-threequickbreaths-2",
    "respiration-and-cough-v2-threebreaths",
    "respiration-and-cough-v2-threebreathsmouth",
    "respiration-and-cough-v2-threebreathsnose",
)


def q(values: Sequence[Any], p: float) -> float | None:
    """A quantile, tolerant of None, for interpreters without ``statistics.quantiles``.

    Args:
        values: The values, Nones dropped.
        p: The quantile in [0, 1].

    Returns:
        The value, rounded, or None when nothing is left.
    """
    kept = sorted(v for v in values if v is not None)
    return round(kept[min(len(kept) - 1, int(len(kept) * p))], 4) if kept else None


def read(out: str, name: str) -> list[dict[str, Any]]:
    """Read one census JSONL.

    Args:
        out: The census directory.
        name: The file name.

    Returns:
        The rows.
    """
    with open(os.path.join(out, name)) as handle:
        return [json.loads(line) for line in handle]


# --------------------------------------------------------------------- the sections


def rates(index: list[dict[str, Any]]) -> None:
    """Sections 1-3: the corpus-wide rates, the per-family table, and the flag grounds."""
    print("### 1,2,3 -- the three rates (n = %d completed rows)" % len(index))
    for field in ("conf", "find", "route", "hint", "ran"):
        counts = collections.Counter(str(r[field]) for r in index if r[field] is not None)
        print("  %-6s %s" % (field, dict(counts)))
    print()
    print("  flag grounds")
    grounds: collections.Counter = collections.Counter()
    for row in index:
        for reason in row["reasons"]:
            if str(reason.get("outcome")) != "pass":
                grounds[str(reason.get("why"))[:78]] += 1
    for why, n in grounds.most_common(8):
        print("    %5d  %5.2f%%  %s" % (n, 100 * n / len(index), why))
    print()
    print("  per declared family")
    byfam: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in index:
        if row["conf"] is not None:
            byfam[str(row["declared_family"])][str(row["conf"])] += 1
    for family in sorted(byfam, key=lambda f: -byfam[f]["False"]):
        counts = byfam[family]
        n = sum(counts.values())
        if family not in LABEL_SET:
            continue
        reader = "coverage" if family in COVERAGE_FAMILIES else "event"
        print(
            "    %-44s n=%5d True=%5d False=%5d UND=%4d  pass=%5.1f%%  %s"
            % (family, n, counts["True"], counts["False"], counts["UNDETERMINED"], 100 * counts["True"] / n, reader)
        )
    print()
    print(
        "  hint x conformance:",
        collections.Counter((str(r["hint"]), str(r["conf"])) for r in index if r["conf"] is not None).most_common(),
    )
    print(
        "  UNDETERMINED by family:",
        collections.Counter(str(r["declared_family"]) for r in index if str(r["conf"]) == "UNDETERMINED").most_common(
            8
        ),
    )
    unavailable = [r for r in index if str(r["route"]) == "unavailable"]
    print(
        "  route unavailable: n=%d families=%s durations=%.2f..%.2f s"
        % (
            len(unavailable),
            collections.Counter(str(r["declared_family"]) for r in unavailable).most_common(),
            min(r["duration_s"] for r in unavailable),
            max(r["duration_s"] for r in unavailable),
        )
    )


def structural(infam: list[dict[str, Any]]) -> None:
    """Section 4: is every instrument AIRWAY reads present when it runs?"""
    print("### 4 -- the structural trap, pattern 1")
    for name, selector in (
        ("all in-family", lambda r: True),
        ("conformance False", lambda r: str(r["conf"]) == "False"),
        ("conformance UNDETERMINED", lambda r: str(r["conf"]) == "UNDETERMINED"),
    ):
        subset = [r for r in infam if selector(r)]
        if not subset:
            continue
        print("  %-26s n=%5d" % (name, len(subset)))
        for key in ("energy_envelope", "hear_scores", "spectrogram_wideband", "silence", "band_profile"):
            absent = sum(1 for r in subset if not (r.get("meas_present") or {}).get(key))
            print("    %-22s absent in %5d (%5.2f%%)" % (key, absent, 100 * absent / len(subset)))
        for key in ("n_span_hear", "n_span_yamnet"):
            zero = sum(1 for r in subset if not r.get(key))
            print("    %-22s zero  in %5d (%5.2f%%)" % (key, zero, 100 * zero / len(subset)))
        print(
            "    report notes:",
            collections.Counter(json.dumps((r.get("report") or {}).get("notes")) for r in subset).most_common(3),
        )
    live = sum(1 for r in infam if (r.get("hear_max") or {}).get("Breathe", 0) > 0)
    print("  rows whose span_hear raw_scores carry a nonzero 'Breathe'/'Cough': %d / %d" % (live, len(infam)))


def gate_separation(infam: list[dict[str, Any]]) -> None:
    """Defect A and the breath side: what the label gate saw, per family."""
    print("### Defect A / the label gate, per family and verdict")
    for family in sorted(LABEL_SET):
        label = LABEL_SET[family]
        for verdict in ("True", "False"):
            subset = [r for r in infam if r["declared_family"] == family and str(r["conf"]) == verdict]
            if not subset:
                continue
            maxima = [r["hear_max"][label] for r in subset if r.get("hear_max")]
            over = sum(1 for r in subset if (r.get("spans_over_min") or {}).get(label, 0) > 0)
            print(
                "  %-42s %-6s n=%4d  hear_max(%s) p10=%-8s p50=%-8s p90=%-8s"
                " | rows with a span >= %.1f: %4d (%5.1f%%)"
                % (
                    family,
                    verdict,
                    len(subset),
                    label,
                    q(maxima, 0.1),
                    q(maxima, 0.5),
                    q(maxima, 0.9),
                    SCORE_MIN,
                    over,
                    100 * over / len(subset),
                )
            )


def coverage_gate(infam: list[dict[str, Any]]) -> None:
    """Defect B: the coverage statistic, its shape, and what it discards."""
    print("### Defect B -- the SOUND_COVERAGE gate")
    for family in sorted(COVERAGE_FAMILIES):
        subset = [r for r in infam if r["declared_family"] == family and r.get("sc_Breathe")]
        print(" ", family, "n =", len(subset))
        histogram: collections.Counter = collections.Counter()
        for row in subset:
            value = row["sc_Breathe"]["coverage"]
            if value is not None:
                histogram[round(value, 1)] += 1
        print("   coverage histogram (0.1 bins):", " ".join("%.1f:%d" % (k, v) for k, v in sorted(histogram.items())))
        print(
            "   HeAR windows per recording:",
            collections.Counter(r["sc_Breathe"]["n_windows"] for r in subset).most_common(4),
        )
        for cut in (0.05, 0.10, 0.20, 0.30, 0.50):
            passed = sum(1 for r in subset if (r["sc_Breathe"]["coverage"] or 0) >= cut)
            print("   coverage >= %.2f -> %4d pass (%5.1f%%)" % (cut, passed, 100 * passed / len(subset)))
        false = [r for r in infam if r["declared_family"] == family and str(r["conf"]) == "False"]
        event = sum(1 for r in false if (r.get("spans_over_min") or {}).get("Breathe", 0) > 0)
        window = sum(1 for r in false if r.get("sc_Breathe") and r["sc_Breathe"]["n_over"] > 0)
        print(
            "   False n=%4d | >=1 span over score_min: %4d (%5.1f%%) | >=1 2 s window over %.1f: %4d (%5.1f%%)"
            % (len(false), event, 100 * event / len(false), SCORE_MIN, window, 100 * window / len(false))
        )


def paired(index: list[dict[str, Any]]) -> None:
    """Defect B's within-session control: the sustained task against the counted ones."""
    print("### Defect B -- within-session pairing")
    sessions: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    for row in index:
        store = row.get("store")
        if not isinstance(store, str):
            continue
        match = re.search(r"(sub-[0-9a-f-]+)/(ses-[0-9A-F-]+)/", store)
        if match:
            sessions[match.group(1) + "/" + match.group(2)][str(row["family"])] = row
    cells: collections.Counter = collections.Counter()
    for families in sessions.values():
        sustained = [families[f] for f in SUSTAINED if f in families and families[f]["conf"] is not None]
        counted = [families[f] for f in COUNTED if f in families and families[f]["conf"] is not None]
        if not sustained or not counted:
            continue
        passing = sum(1 for r in counted if str(r["conf"]) == "True")
        tag = "ALL counted pass" if passing == len(counted) else ("NONE pass" if passing == 0 else "some pass")
        for row in sustained:
            cells[(str(row["conf"]), tag)] += 1
    for key in sorted(cells):
        print("   %-14s | %-18s n=%5d" % (key[0], key[1], cells[key]))
    for tag in ("ALL counted pass", "NONE pass"):
        n = sum(v for k, v in cells.items() if k[1] == tag)
        f = cells[("False", tag)]
        print("  %-18s: %d of %d called False (%.1f%%)" % (tag, f, n, 100 * f / max(1, n)))


def controls(index: list[dict[str, Any]], workers: int) -> None:
    """Defect B's control contrast: the same statistic where no breathing task was asked for."""
    print("### Defect B -- control contrast")
    todo: list[dict[str, Any]] = []
    for family in CONTROL_FAMILIES:
        todo.extend([r for r in index if str(r["declared_family"]) == family][:400])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = [x for x in pool.map(_control_one, todo, chunksize=8) if x]
    byfam: dict[str, list] = collections.defaultdict(list)
    for row in got:
        byfam[row["fam"]].append(row)
    print("  %-34s %6s %8s %8s %8s %12s" % ("family", "n", "p50", "p90", "p99", ">= %.1f" % COVERAGE_MIN))
    for family in CONTROL_FAMILIES:
        subset = byfam.get(family) or []
        if not subset:
            continue
        values = [x["coverage"] for x in subset]
        over = sum(1 for x in subset if (x["coverage"] or 0) >= COVERAGE_MIN)
        print(
            "  %-34s %6d %8s %8s %8s %6d(%5.1f%%)"
            % (family, len(subset), q(values, 0.5), q(values, 0.9), q(values, 0.99), over, 100 * over / len(subset))
        )
    everything = [x["coverage"] for x in got]
    over = sum(1 for x in got if (x["coverage"] or 0) >= COVERAGE_MIN)
    print(
        "  %-34s %6d %8s %8s %8s %6d(%5.1f%%)"
        % (
            "ALL CONTROLS",
            len(got),
            q(everything, 0.5),
            q(everything, 0.9),
            q(everything, 0.99),
            over,
            100 * over / len(got),
        )
    )


def _control_one(row: dict[str, Any]) -> dict[str, Any] | None:
    """One control recording's breath-coverage fraction.

    Args:
        row: An ``index.jsonl`` row.

    Returns:
        The family and the coverage, or None when the sidecar is absent.
    """
    path = os.path.join(row["run_dir"], "derivatives", "hear_scores.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            windows = json.load(handle)
    except Exception:
        return None
    whole = row["duration_s"] or 0.0
    hit = []
    for window in windows:
        scores = {k: v for pair in window["label_scores"] for k, v in pair.items()}
        if float(scores.get("Breathe", 0.0)) >= SCORE_MIN:
            hit.append((float(window["start"]), float(window["end"])))
    merged: list[list[float]] = []
    for start, end in sorted(hit):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    total = sum(b - a for a, b in merged)
    return {"fam": row["declared_family"], "coverage": total / whole if whole else None}


def profile_rescue(profileset: list[dict[str, Any]]) -> None:
    """Pattern 1's cost: the packaged profile's corroboration set for ``Breathe``."""
    print("### Pattern 1 -- what a readable YAMNet breath vote covers")
    total_false = total_rescue = 0
    for family in sorted({r["declared_family"] for r in profileset} - COVERAGE_FAMILIES):
        subset = [r for r in profileset if r["declared_family"] == family and "hear_amp" in r]
        false = [r for r in subset if str(r["conf"]) == "False"]
        true = [r for r in subset if str(r["conf"]) == "True"]
        if not false:
            continue
        rescue = sum(1 for r in false if r["hear_amp"] == 0 and r["yam_only_amp"] > 0)
        total_false += len(false)
        total_rescue += rescue
        print(
            "  %-42s True n=%4d (yamnet breath fires on %5.1f%%) | False n=%4d | yamnet-only: %4d (%5.1f%%)"
            % (
                family,
                len(true),
                100 * sum(1 for r in true if r["yam_amp"] > 0) / max(1, len(true)),
                len(false),
                rescue,
                100 * rescue / len(false),
            )
        )
    print(
        "  TOTAL event-pattern breath False = %d, yamnet-only = %d (%.1f%%)"
        % (total_false, total_rescue, 100 * total_rescue / max(1, total_false))
    )
    labels: collections.Counter = collections.Counter()
    for row in profileset:
        if str(row["conf"]) == "False" and row.get("yam_only_amp", 0) > 0 and row.get("hear_amp") == 0:
            for label in row.get("labels") or {}:
                labels[label] += 1
    print("  rows in which each profile label fires:", dict(labels.most_common()))


def baby_cough(heads: list[dict[str, Any]]) -> None:
    """The ``Baby Cough`` head, checked for specificity and not claimed."""
    print("### Pattern 1 -- the Baby Cough head")
    for verdict in ("True", "False"):
        subset = [r for r in heads if str(r["conf"]) == verdict and r.get("hear_best")]
        print("  conformance %s (n=%d)" % (verdict, len(subset)))
        for head in HEAR_HEADS:
            values = [r["hear_best"].get(head, 0.0) for r in subset]
            over = sum(1 for r in subset if r["spans_over_amp"].get(head, 0) > 0)
            print(
                "    %-14s max p50=%-8s p90=%-8s"
                " | rows with a span >= %.1f on an amplitude carrier: %4d (%5.1f%%)"
                % (head, q(values, 0.5), q(values, 0.9), SCORE_MIN, over, 100 * over / len(subset))
            )


def instrument_absent(infam: list[dict[str, Any]]) -> None:
    """Defect C: the stores with no span at all, and the sub-2 s band."""
    print("### Defect C -- stores with no span at all")
    zero = [r for r in infam if not r.get("n_span_hear")]
    print("  n=%d  conformance=%s" % (len(zero), dict(collections.Counter(str(r["conf"]) for r in zero))))
    print("  n_spans:", collections.Counter(r.get("n_spans") for r in zero).most_common(3))
    print(
        "  duration p10/p50/p90:",
        q([r["duration_s"] for r in zero], 0.1),
        q([r["duration_s"] for r in zero], 0.5),
        q([r["duration_s"] for r in zero], 0.9),
    )
    print(
        "  by family x conformance:",
        collections.Counter((str(r["conf"]), r["declared_family"]) for r in zero).most_common(),
    )
    print()
    print("  conformance by duration band, all in-family")
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, float("inf"))):
        subset = [r for r in infam if lo <= (r["duration_s"] or 0) < hi]
        counts = collections.Counter(str(r["conf"]) for r in subset)
        print(
            "   %5.0f-%-6s n=%5d True=%5d False=%5d UND=%4d | hear_scores written %5d | any window %5d"
            % (
                lo,
                ("%.0fs" % hi) if hi < float("inf") else "inf",
                len(subset),
                counts["True"],
                counts["False"],
                counts["UNDETERMINED"],
                sum(1 for r in subset if r.get("hear_sidecar")),
                sum(1 for r in subset if r.get("n_span_hear")),
            )
        )


def nuisance(infam: list[dict[str, Any]]) -> None:
    """The nuisance trends: recording level and duration."""
    print("### nuisance variables")
    for name, families in (
        ("coverage families", COVERAGE_FAMILIES),
        ("event families", set(LABEL_SET) - COVERAGE_FAMILIES),
    ):
        subset = [
            r
            for r in infam
            if r["declared_family"] in families
            and (r["duration_s"] or 0) >= 2.0
            and r.get("level")
            and r["conf"] is not None
        ]
        if not subset:
            continue
        print("  %s (n=%d)" % (name, len(subset)))
        for key in ("lufs", "peak_dbfs"):
            ordered = sorted(subset, key=lambda r: r["level"][key])
            n = len(ordered)
            cells = []
            for i in range(5):
                block = ordered[i * n // 5 : (i + 1) * n // 5]
                false = sum(1 for r in block if str(r["conf"]) == "False")
                cells.append(
                    "[%6.1f,%6.1f] %4.1f%%"
                    % (block[0]["level"][key], block[-1]["level"][key], 100 * false / len(block))
                )
            print("    %-10s quintiles -> False rate: %s" % (key, "  ".join(cells)))
    print("  False rate by duration quartile, files of 2 s or more")
    for family in sorted(set(LABEL_SET) - COVERAGE_FAMILIES):
        subset = [
            r
            for r in infam
            if r["declared_family"] == family and (r["duration_s"] or 0) >= 2.0 and r["conf"] is not None
        ]
        if len(subset) < 100:
            continue
        subset.sort(key=lambda r: r["duration_s"])
        n = len(subset)
        cells = []
        for i in range(4):
            block = subset[i * n // 4 : (i + 1) * n // 4]
            false = sum(1 for r in block if str(r["conf"]) == "False")
            amp = q([r["span_measures"].get("amplitude", 0) for r in block], 0.5)
            cells.append(
                "%.0f-%.0fs:%.0f%%(amp%s)"
                % (block[0]["duration_s"], block[-1]["duration_s"], 100 * false / len(block), amp)
            )
        print("    %-42s n=%4d  %s" % (family, n, "  ".join(cells)))


def main(argv: list[str] | None = None) -> None:
    """Print every section.

    Args:
        argv: Command line, or None for ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="the census directory")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("NW", "10")))
    args = parser.parse_args(argv)

    index = read(args.out, "index.jsonl")
    infam = read(args.out, "infamily.jsonl")
    for section in (
        lambda: rates(index),
        lambda: structural(infam),
        lambda: gate_separation(infam),
        lambda: coverage_gate(infam),
        lambda: paired(index),
        lambda: controls(index, args.workers),
        lambda: profile_rescue(read(args.out, "profileset.jsonl")),
        lambda: baby_cough(read(args.out, "coughheads.jsonl")),
        lambda: instrument_absent(infam),
        lambda: nuisance(infam),
    ):
        section()
        print()


if __name__ == "__main__":
    main()
