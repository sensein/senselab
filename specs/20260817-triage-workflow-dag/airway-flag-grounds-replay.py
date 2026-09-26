"""Replay the three fixes' decision conditions over the corpus, read-only."""

import collections
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

BASE = "/orcd/scratch/bcs/002/satra/triage_design_20260919/airwaygrounds"
MIN = 0.2
# branch.label_sets, and the packaged profile's corroboration set for each head
HEAR = {"cough": ("Cough",), "breath": ("Breathe",)}
YAM = {"cough": ("Cough", "Throat clearing"), "breath": ("Breathing", "Gasp", "Pant", "Snoring", "Snort", "Wheeze")}
KIND = {
    "respiration-and-cough-cough": "cough",
    "respiration-and-cough-v2-hardcough": "cough",
    "voluntary-cough": "cough",
    "respiration-and-cough-fivebreaths": "breath",
    "respiration-and-cough-v2-threebreathsnose": "breath",
    "respiration-and-cough-v2-threebreathsmouth": "breath",
    "respiration-and-cough-threequickbreaths": "breath",
    "respiration-and-cough-v2-threebreaths": "breath",
    "respiration-and-cough-breath": "breath",
    "respiration-and-cough-v2-breath": "breath",
    "breath-sounds": "breath",
}
COVERAGE = {"respiration-and-cough-breath", "respiration-and-cough-v2-breath"}
ALTERNATION = {"voluntary-cough"}
CONTROL = (
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


def scan(row: dict[str, Any]) -> dict[str, Any] | None:
    """Which carriers each vocabulary fires on, and which instruments arrived.

    Args:
        row: An ``index.jsonl`` row from the census.

    Returns:
        The replay inputs for one recording, or None when its store is unreadable.
    """
    out = {
        "family": row["family"],
        "declared_family": str(row["declared_family"]),
        "before": str(row["conf"]),
        "duration_s": row["duration_s"],
    }
    store = row.get("store")
    if not isinstance(store, str) or not os.path.isfile(store):
        return None
    amp, allspans = set(), set()
    hit = {(c, k): set() for c in ("hear", "yamnet") for k in HEAR}
    n_windows = 0
    envelope = False
    for line in open(store):
        d = json.loads(line)
        if d.get("record") != "entity":
            continue
        a = d.get("attributes") or {}
        t = d.get("prov_type")
        if t == "measurement":
            name = a.get("name")
            if name in ("span_hear", "span_yamnet"):
                n_windows += 1
                which = "hear" if name == "span_hear" else "yamnet"
                table = HEAR if which == "hear" else YAM
                raw = a.get("raw_scores") or {}
                for kind, names in table.items():
                    if any(float(raw.get(n, 0.0)) >= MIN for n in names):
                        hit[(which, kind)].add(a.get("span_id"))
            elif name == "energy_envelope":
                envelope = True
        elif t == "span" and a.get("measure"):
            allspans.add(d["id"])
            if a["measure"] == "amplitude":
                amp.add(d["id"])
    out["envelope"] = envelope
    out["n_windows"] = n_windows
    carriers = allspans if out["declared_family"] in ALTERNATION else amp
    kind = KIND.get(out["declared_family"])
    out["before_fires"] = bool(hit[("hear", kind)] & carriers) if kind else None
    after = (hit[("hear", kind)] | hit[("yamnet", kind)]) & carriers if kind else set()
    out["after_fires"] = bool(after)
    return out


def verdict_after(r: dict[str, Any]) -> str:
    """What the three fixes make of one in-family recording.

    Args:
        r: One :func:`scan` result.

    Returns:
        The conformance the fixed branch would report.
    """
    if r["declared_family"] in COVERAGE:
        return "UNDETERMINED"
    if not r["envelope"] or r["n_windows"] == 0:
        return "UNDETERMINED"
    return "True" if r["after_fires"] else "False"


def control_scan(row: dict[str, Any]) -> dict[str, Any] | None:
    """Breath firing on a family that elicits no breathing task, before and after A.

    Args:
        row: An ``index.jsonl`` row.

    Returns:
        Whether each vocabulary fired, or None when the store is unreadable.
    """
    r = scan({**row, "declared_family": "respiration-and-cough-fivebreaths"})
    if r is None:
        return None
    return {
        "fam": str(row["declared_family"]),
        "before": r["before_fires"],
        "after": r["after_fires"],
        "n_windows": r["n_windows"],
    }


def main() -> None:
    """Print the before/after tables and the control contrast."""
    idx = [json.loads(line) for line in open(BASE + "/index.jsonl")]
    infam = [r for r in idx if str(r["declared_family"]) in KIND and r["conf"] is not None]
    with ProcessPoolExecutor(max_workers=10) as pool:
        rows = [r for r in pool.map(scan, infam, chunksize=16) if r]
    print("in-family rows replayed:", len(rows))

    groups = [
        ("event/alternation, cough", lambda f: KIND[f] == "cough"),
        ("event series, breath", lambda f: KIND[f] == "breath" and f not in COVERAGE),
        ("SOUND_COVERAGE, breath", lambda f: f in COVERAGE),
    ]
    print()
    print("%-28s %6s | %-24s | %-24s" % ("matcher", "n", "before", "after"))
    print("%-28s %6s | %6s %6s %6s | %6s %6s %6s" % ("", "", "True", "False", "UND", "True", "False", "UND"))
    for name, keep in groups:
        sub = [r for r in rows if keep(r["declared_family"])]
        b = collections.Counter(r["before"] for r in sub)
        a = collections.Counter(verdict_after(r) for r in sub)
        print(
            "%-28s %6d | %6d %6d %6d | %6d %6d %6d"
            % (name, len(sub), b["True"], b["False"], b["UNDETERMINED"], a["True"], a["False"], a["UNDETERMINED"])
        )
    b = collections.Counter(r["before"] for r in rows)
    a = collections.Counter(verdict_after(r) for r in rows)
    print(
        "%-28s %6d | %6d %6d %6d | %6d %6d %6d"
        % (
            "ALL IN-FAMILY",
            len(rows),
            b["True"],
            b["False"],
            b["UNDETERMINED"],
            a["True"],
            a["False"],
            a["UNDETERMINED"],
        )
    )

    print()
    print("per family")
    print("%-44s %6s | %6s %6s %6s | %6s %6s %6s" % ("family", "n", "True", "False", "UND", "True", "False", "UND"))
    for fam in sorted(KIND):
        sub = [r for r in rows if r["declared_family"] == fam]
        if not sub:
            continue
        b = collections.Counter(r["before"] for r in sub)
        a = collections.Counter(verdict_after(r) for r in sub)
        print(
            "%-44s %6d | %6d %6d %6d | %6d %6d %6d"
            % (fam, len(sub), b["True"], b["False"], b["UNDETERMINED"], a["True"], a["False"], a["UNDETERMINED"])
        )

    print()
    print("what moved, by cause (in-family rows whose verdict changed)")
    cause = collections.Counter()
    for r in rows:
        after = verdict_after(r)
        if after == r["before"]:
            continue
        if r["declared_family"] in COVERAGE:
            cause[("B coverage gate retired", r["before"] + " -> " + after)] += 1
        elif not r["envelope"] or r["n_windows"] == 0:
            cause[("C no instrument", r["before"] + " -> " + after)] += 1
        elif r["after_fires"] and not r["before_fires"]:
            cause[("A yamnet spelling readable", r["before"] + " -> " + after)] += 1
        else:
            cause[("unaccounted", r["before"] + " -> " + after)] += 1
    for k, v in sorted(cause.items()):
        print("  %-28s %-24s %5d" % (k[0], k[1], v))

    print()
    print("control contrast: breath firing on families that elicit no breathing task")
    todo = []
    for fam in CONTROL:
        todo.extend([r for r in idx if str(r["declared_family"]) == fam][:400])
    with ProcessPoolExecutor(max_workers=10) as pool:
        got = [x for x in pool.map(control_scan, todo, chunksize=8) if x]
    byfam = collections.defaultdict(list)
    for x in got:
        byfam[x["fam"]].append(x)
    print("  %-34s %6s %14s %14s" % ("family", "n", "before", "after"))
    for fam in CONTROL:
        s = byfam.get(fam) or []
        if not s:
            continue
        nb = sum(1 for x in s if x["before"])
        na = sum(1 for x in s if x["after"])
        print("  %-34s %6d %6d(%5.1f%%) %6d(%5.1f%%)" % (fam, len(s), nb, 100 * nb / len(s), na, 100 * na / len(s)))
    nb = sum(1 for x in got if x["before"])
    na = sum(1 for x in got if x["after"])
    print(
        "  %-34s %6d %6d(%5.1f%%) %6d(%5.1f%%)"
        % ("ALL CONTROLS", len(got), nb, 100 * nb / len(got), na, 100 * na / len(got))
    )
    ev = [r for r in rows if KIND[r["declared_family"]] == "breath" and r["declared_family"] not in COVERAGE]
    nb = sum(1 for r in ev if r["before_fires"])
    na = sum(1 for r in ev if r["after_fires"])
    print(
        "  %-34s %6d %6d(%5.1f%%) %6d(%5.1f%%)   <- the breath tasks themselves"
        % ("event-series breath families", len(ev), nb, 100 * nb / len(ev), na, 100 * na / len(ev))
    )


if __name__ == "__main__":
    main()
