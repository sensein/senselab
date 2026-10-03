"""Per-recording census of what AIRWAY actually had in hand.

Reads each finished run's store and its ``hear_scores`` sidecar, reconstructs the exact evidence
``airway.py`` reads -- the candidate spans, the ``span_hear`` / ``span_yamnet`` windows over them,
the envelope and coverage instruments -- and records what each gate could have decided on. Reads
only; writes one JSONL row per recording under ``--out``. No transcript text and no subject
identifier leaves: family names, gate names, counts and measurement values only.

Five steps, in order; each reads what the ones before it wrote.

    index     every completed row, with AIRWAY's four decision fields   -> index.jsonl
    infamily  the store of every in-family AIRWAY row                   -> infamily.jsonl
    spans     per-span classifier agreement, both classifiers together  -> spanagree.jsonl
    heads     every HeAR head's maximum on the cough families           -> coughheads.jsonl
    profile   the packaged ontology profile's corroboration set for     -> profileset.jsonl
              ``Breathe``, applied to ``span_yamnet``

Usage::

    python3 airway-flag-grounds-census.py --run <run dir> --out <dir> [step ...]

with no step meaning all five. ``--workers`` sets the reader pool size.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable

SCORE_MIN = 0.2
"""``branch.score_min`` in the packaged config: the cut ``sounds_like`` applies to ``raw_scores``."""

AIRWAY_FAMILIES = frozenset(
    {
        "respiration-and-cough-cough",
        "respiration-and-cough-v2-hardcough",
        "voluntary-cough",
        "respiration-and-cough-fivebreaths",
        "respiration-and-cough-v2-threebreathsnose",
        "respiration-and-cough-v2-threebreathsmouth",
        "respiration-and-cough-threequickbreaths",
        "respiration-and-cough-v2-threebreaths",
        "respiration-and-cough-breath",
        "respiration-and-cough-v2-breath",
        "breath-sounds",
    }
)
"""``AIRWAY_EXPECTATIONS``'s eleven keys: the families ``align_airway`` evaluates."""

COVERAGE_FAMILIES = frozenset({"respiration-and-cough-breath", "respiration-and-cough-v2-breath"})
"""The two families ``Pattern.SOUND_COVERAGE`` is the matcher for."""

COUGH_FAMILIES = frozenset({"respiration-and-cough-cough", "respiration-and-cough-v2-hardcough", "voluntary-cough"})
"""The three whose ``label_set`` is ``cough``."""

PROFILE_BREATH = ("Breathing", "Gasp", "Pant", "Snoring", "Snort", "Wheeze")
"""``corroboration_sets()["Breathe"]`` from ``data/classifier_ontology/2026-09-10.json``."""

PROFILE_COUGH = ("Cough", "Throat clearing")
"""``corroboration_sets()["Cough"]`` from the same profile."""

HEAR_HEADS = ("Cough", "Baby Cough", "Throat Clear", "Breathe", "Sneeze", "Speech", "Laugh", "Snore")
"""HeAR's eight event heads, as ``span_hear.raw_scores`` keys them."""


# --------------------------------------------------------------------- reading


def entities(store: str) -> Iterable[dict[str, Any]]:
    """Every entity record of one store.

    Args:
        store: Path to ``store.jsonl``.

    Yields:
        The parsed records whose ``record`` is ``entity``.
    """
    with open(store) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record") == "entity":
                yield record


def hear_windows(run_dir: str) -> list[dict[str, Any]] | None:
    """The ``hear_scores`` sidecar, on HeAR's own 2 s grid.

    Args:
        run_dir: The run directory the sidecar path is relative to.

    Returns:
        The windows, or None when the file is absent or unreadable.
    """
    path = os.path.join(run_dir, "derivatives", "hear_scores.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return None


def merge(extents: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Coalesce overlapping or touching extents.

    Args:
        extents: The extents.

    Returns:
        The merged extents, earliest first.
    """
    out: list[tuple[float, float]] = []
    for start, end in sorted(extents):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def coverage_of(windows: list[dict[str, Any]] | None, label: str, whole: float) -> dict[str, Any] | None:
    """``_airway_coverage``'s statistic, recomputed from the sidecar.

    Args:
        windows: The sidecar windows, or None.
        label: The HeAR head the label set names.
        whole: The stream duration the branch divides by.

    Returns:
        ``covered_s``, ``coverage``, the head's maximum, how many windows cleared the cut, the
        window count, and the hull of the covered runs. None when there is no sidecar.
    """
    if windows is None:
        return None
    hit: list[tuple[float, float]] = []
    best = 0.0
    for window in windows:
        scores = {name: value for pair in window["label_scores"] for name, value in pair.items()}
        value = float(scores.get(label, 0.0))
        best = max(best, value)
        if value >= SCORE_MIN:
            hit.append((float(window["start"]), float(window["end"])))
    merged = merge(hit)
    total = sum(end - start for start, end in merged)
    return {
        "covered_s": round(total, 3),
        "coverage": round(total / whole, 4) if whole else None,
        "max": round(best, 4),
        "n_over": len(hit),
        "n_windows": len(windows),
        "hull_s": round(merged[-1][1] - merged[0][0], 3) if merged else 0.0,
    }


# --------------------------------------------------------------------- step: index


def index_row(path: str) -> dict[str, Any] | None:
    """One corpus row, reduced to AIRWAY's decision fields.

    Args:
        path: Path to a ``*.row.json``.

    Returns:
        The reduced row, or None when the recording did not complete.
    """
    try:
        with open(path) as handle:
            row = json.load(handle)
    except Exception:
        return None
    if not row.get("ok"):
        return None
    decision = row.get("decision") or {}
    return {
        "family": row.get("family"),
        "declared_family": decision.get("declared_family"),
        "duration_s": row.get("duration_s"),
        "bytes": row.get("bytes"),
        "speech_type": row.get("speech_type"),
        "store": row.get("store"),
        "run_dir": row.get("run_dir"),
        "conf": (decision.get("conformance") or {}).get("AIRWAY"),
        "find": (decision.get("findings") or {}).get("AIRWAY"),
        "route": (decision.get("routes") or {}).get("AIRWAY"),
        "hint": (decision.get("hints") or {}).get("AIRWAY"),
        "ran": (decision.get("ran") or {}).get("AIRWAY"),
        "unmeasured": (decision.get("unmeasured") or {}).get("AIRWAY"),
        "dev": (decision.get("deviations") or {}).get("AIRWAY"),
        "reasons": [x for x in (decision.get("reasons") or []) if x.get("node") == "AIRWAY"],
    }


def step_index(run: str, out: str, workers: int) -> None:
    """Write ``index.jsonl``: every completed row.

    Args:
        run: The corpus ``run`` directory.
        out: The output directory.
        workers: Reader pool size.
    """
    files = glob.glob(os.path.join(run, "rows", "**", "*.row.json"), recursive=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [row for row in pool.map(index_row, files, chunksize=64) if row]
    _write(out, "index.jsonl", rows)
    print("index: %d completed of %d rows" % (len(rows), len(files)), file=sys.stderr)


# --------------------------------------------------------------------- step: infamily


def infamily_row(row: dict[str, Any]) -> dict[str, Any]:
    """What one in-family AIRWAY recording's store held.

    Args:
        row: An ``index.jsonl`` row.

    Returns:
        The branch report, the folded ``counts``, the coverage measurement, which derivatives
        reached the store, the span inventory, and the maxima of the two label-set heads.
    """
    out: dict[str, Any] = {
        key: row[key] for key in ("family", "declared_family", "conf", "find", "route", "hint", "duration_s")
    }
    windows = hear_windows(row["run_dir"])
    whole = row["duration_s"] or 0.0
    out["hear_sidecar"] = windows is not None
    for label in ("Breathe", "Cough"):
        out["sc_" + label] = coverage_of(windows, label, whole)
    store = row["store"]
    if not isinstance(store, str) or not os.path.isfile(store):
        out["store"] = False
        return out
    out["store"] = True

    names: collections.Counter = collections.Counter()
    by_measure: collections.Counter = collections.Counter()
    hear_max = {"Breathe": 0.0, "Cough": 0.0}
    yam_max: dict[str, float] = {}
    over: dict[str, set] = {"Breathe": set(), "Cough": set()}
    n_hear = n_yam = n_spans = 0
    for entity in entities(store):
        attributes = entity.get("attributes") or {}
        kind = entity.get("prov_type")
        if kind == "measurement":
            name = attributes.get("name")
            names[name] += 1
            if name == "span_hear":
                n_hear += 1
                raw = attributes.get("raw_scores") or {}
                for label in hear_max:
                    value = float(raw.get(label, 0.0))
                    hear_max[label] = max(hear_max[label], value)
                    if value >= SCORE_MIN:
                        over[label].add(attributes.get("span_id"))
            elif name == "span_yamnet":
                n_yam += 1
                raw = attributes.get("raw_scores") or {}
                for label in ("Cough", "Breathing"):
                    if label in raw:
                        yam_max[label] = max(yam_max.get(label, 0.0), float(raw[label]))
            elif name == "counts":
                out.setdefault("counts", []).append(attributes.get("entries"))
            elif name == "breath_coverage_fraction":
                out["breath_coverage_fraction"] = attributes.get("value")
                out["breath_covered_s"] = attributes.get("covered_s")
            elif name == "event_instrument":
                out["event_instrument_absent"] = attributes.get("absent")
            elif name == "level":
                out["level"] = {k: v for k, v in attributes.items() if k not in ("name", "signal")}
        elif kind == "span":
            by_measure[str(attributes.get("measure"))] += 1
            if attributes.get("measure"):
                n_spans += 1
        elif kind == "branch_report" and attributes.get("node") == "AIRWAY":
            out["report"] = {
                k: attributes[k]
                for k in (
                    "mode",
                    "task_family",
                    "labelled_n",
                    "by_label",
                    "contested_n",
                    "merged_n",
                    "spans_n",
                    "notes",
                    "unmeasured",
                    "conformance",
                    "deviations",
                )
                if k in attributes
            }
        elif kind == "stream" and attributes.get("name") == "recording":
            out["stream_extent"] = entity.get("extent")

    out["meas_present"] = {
        k: names[k]
        for k in ("energy_envelope", "hear_scores", "spectrogram_wideband", "silence", "band_profile")
        if names.get(k)
    }
    out["n_span_hear"] = n_hear
    out["n_span_yamnet"] = n_yam
    out["n_spans"] = n_spans
    out["hear_max"] = {k: round(v, 5) for k, v in hear_max.items()}
    out["yam_max"] = {k: round(v, 5) for k, v in yam_max.items()}
    out["spans_over_min"] = {k: len(v) for k, v in over.items()}
    out["span_measures"] = dict(by_measure)
    return out


def step_infamily(run: str, out: str, workers: int) -> None:
    """Write ``infamily.jsonl``: the store of every in-family AIRWAY row.

    Args:
        run: Unused; the stores are named by ``index.jsonl``.
        out: The output directory.
        workers: Reader pool size.
    """
    rows = _selected(out, AIRWAY_FAMILIES)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(infamily_row, rows, chunksize=16))
    _write(out, "infamily.jsonl", got)
    print("infamily: %d stores" % len(got), file=sys.stderr)


# --------------------------------------------------------------------- step: spans


def span_agreement(row: dict[str, Any]) -> dict[str, Any]:
    """Which spans each classifier put a breath-like or cough-like label over.

    ``shipped`` is what ``branch.label_sets`` actually reads today -- the literal spellings
    ``Breathe`` and ``Cough`` against both classifiers; the other keys are what each classifier's
    own vocabulary offers.

    Args:
        row: An ``index.jsonl`` row.

    Returns:
        Span counts per classifier and per label family, and the same restricted to ``amplitude``
        carriers, which is what ``_airway_event_series`` looks inside.
    """
    yam_breath = ("Breathing", "Gasp", "Pant", "Sniff", "Snort", "Wheeze", "Sigh", "Hiccup")
    yam_cough = ("Cough", "Throat clearing", "Sneeze")
    out: dict[str, Any] = {key: row[key] for key in ("family", "declared_family", "conf", "duration_s")}
    store = row["store"]
    match = re.search(r"(sub-[0-9a-f-]+)/(ses-[0-9A-F-]+)/", store or "")
    out["key"] = (match.group(1) + "/" + match.group(2)) if match else None
    if not isinstance(store, str) or not os.path.isfile(store):
        return out

    hear: dict[str, dict[str, float]] = collections.defaultdict(dict)
    yam: dict[str, dict[str, float]] = collections.defaultdict(dict)
    amplitude: set[str] = set()
    for entity in entities(store):
        attributes = entity.get("attributes") or {}
        kind = entity.get("prov_type")
        if kind == "measurement" and attributes.get("name") in ("span_hear", "span_yamnet"):
            target = hear if attributes["name"] == "span_hear" else yam
            span_id = attributes.get("span_id")
            for label, value in (attributes.get("raw_scores") or {}).items():
                if float(value) >= SCORE_MIN:
                    target[span_id][label] = max(target[span_id].get(label, 0.0), float(value))
        elif kind == "span" and attributes.get("measure") == "amplitude":
            amplitude.add(entity["id"])

    def having(table: dict[str, dict[str, float]], labels: Iterable[str]) -> set[str]:
        wanted = set(labels)
        return {span for span, seen in table.items() if wanted & set(seen)}

    hb, hc = having(hear, ("Breathe",)), having(hear, ("Cough", "Baby Cough"))
    yb, yc = having(yam, yam_breath), having(yam, yam_cough)
    out["n_amp"] = len(amplitude)
    out["breath"] = {
        "hear": len(hb),
        "yam": len(yb),
        "yam_not_hear": len(yb - hb),
        "hear_amp": len(hb & amplitude),
        "yam_amp": len(yb & amplitude),
        "yam_not_hear_amp": len((yb - hb) & amplitude),
        "yam_labels": sorted({lab for span in (yb - hb) for lab in yam[span] if lab in yam_breath}),
    }
    out["cough"] = {
        "hear": len(hc),
        "yam": len(yc),
        "yam_not_hear": len(yc - hc),
        "hear_amp": len(hc & amplitude),
        "yam_amp": len(yc & amplitude),
        "yam_not_hear_amp": len((yc - hc) & amplitude),
    }
    out["yam_label_vocab"] = sorted({lab for seen in yam.values() for lab in seen})[:60]
    return out


def step_spans(run: str, out: str, workers: int) -> None:
    """Write ``spanagree.jsonl``.

    Args:
        run: Unused.
        out: The output directory.
        workers: Reader pool size.
    """
    rows = _selected(out, AIRWAY_FAMILIES)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(span_agreement, rows, chunksize=16))
    _write(out, "spanagree.jsonl", got)
    print("spans: %d stores" % len(got), file=sys.stderr)


# --------------------------------------------------------------------- step: heads


def cough_heads(row: dict[str, Any]) -> dict[str, Any]:
    """Every HeAR head's maximum on one cough-family recording.

    Args:
        row: An ``index.jsonl`` row.

    Returns:
        Per-head maxima and per-head span counts over the cut, with and without restricting to
        ``amplitude`` carriers, plus which BIDS tree the recording came from.
    """
    out: dict[str, Any] = {key: row[key] for key in ("family", "declared_family", "conf", "duration_s")}
    store = row["store"]
    if not isinstance(store, str) or not os.path.isfile(store):
        return out
    best: dict[str, float] = {}
    spans: dict[str, set] = {}
    amplitude: set[str] = set()
    path = None
    for entity in entities(store):
        attributes = entity.get("attributes") or {}
        kind = entity.get("prov_type")
        if kind == "measurement" and attributes.get("name") == "span_hear":
            span_id = attributes.get("span_id")
            for label, value in (attributes.get("raw_scores") or {}).items():
                value = float(value)
                best[label] = max(best.get(label, 0.0), value)
                if value >= SCORE_MIN:
                    spans.setdefault(label, set()).add(span_id)
        elif kind == "span" and attributes.get("measure") == "amplitude":
            amplitude.add(entity["id"])
        elif kind == "stream" and attributes.get("name") == "recording":
            path = attributes.get("path")
    tree = re.search(r"/(bids_[a-z0-9_]+)/", path or "")
    out["path_tree"] = tree.group(1) if tree else None
    out["hear_best"] = {k: round(v, 4) for k, v in sorted(best.items())}
    out["spans_over"] = {k: len(v) for k, v in spans.items()}
    out["spans_over_amp"] = {k: len(v & amplitude) for k, v in spans.items()}
    out["n_amp"] = len(amplitude)
    return out


def step_heads(run: str, out: str, workers: int) -> None:
    """Write ``coughheads.jsonl``.

    Args:
        run: Unused.
        out: The output directory.
        workers: Reader pool size.
    """
    rows = _selected(out, COUGH_FAMILIES)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(cough_heads, rows, chunksize=16))
    _write(out, "coughheads.jsonl", got)
    print("heads: %d stores" % len(got), file=sys.stderr)


# --------------------------------------------------------------------- step: profile


def profile_set(row: dict[str, Any]) -> dict[str, Any]:
    """The packaged profile's ``Breathe`` corroboration set, applied to ``span_yamnet``.

    Args:
        row: An ``index.jsonl`` row.

    Returns:
        How many ``amplitude`` carriers HeAR's ``Breathe`` clears the cut on, how many YAMNet's
        corroborating set clears it on, and how many only YAMNet does.
    """
    out: dict[str, Any] = {key: row[key] for key in ("family", "declared_family", "conf")}
    store = row["store"]
    if not isinstance(store, str) or not os.path.isfile(store):
        return out
    hear_breath: set[str] = set()
    yam_breath: set[str] = set()
    amplitude: set[str] = set()
    labels: collections.Counter = collections.Counter()
    for entity in entities(store):
        attributes = entity.get("attributes") or {}
        kind = entity.get("prov_type")
        if kind == "measurement" and attributes.get("name") in ("span_hear", "span_yamnet"):
            raw = attributes.get("raw_scores") or {}
            span_id = attributes.get("span_id")
            if attributes["name"] == "span_hear":
                if float(raw.get("Breathe", 0.0)) >= SCORE_MIN:
                    hear_breath.add(span_id)
            else:
                for label in PROFILE_BREATH:
                    if float(raw.get(label, 0.0)) >= SCORE_MIN:
                        yam_breath.add(span_id)
                        labels[label] += 1
        elif kind == "span" and attributes.get("measure") == "amplitude":
            amplitude.add(entity["id"])
    out["hear_amp"] = len(hear_breath & amplitude)
    out["yam_amp"] = len(yam_breath & amplitude)
    out["yam_only_amp"] = len((yam_breath - hear_breath) & amplitude)
    out["labels"] = dict(labels)
    return out


def step_profile(run: str, out: str, workers: int) -> None:
    """Write ``profileset.jsonl``.

    Args:
        run: Unused.
        out: The output directory.
        workers: Reader pool size.
    """
    rows = _selected(out, AIRWAY_FAMILIES - COUGH_FAMILIES)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        got = list(pool.map(profile_set, rows, chunksize=16))
    _write(out, "profileset.jsonl", got)
    print("profile: %d stores" % len(got), file=sys.stderr)


# --------------------------------------------------------------------- driver


def _read(out: str, name: str) -> list[dict[str, Any]]:
    """Read one JSONL the census wrote.

    Args:
        out: The output directory.
        name: The file name.

    Returns:
        The rows.

    Raises:
        SystemExit: If the file is absent, naming the step that writes it.
    """
    path = os.path.join(out, name)
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is absent; run the earlier step first")
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def _selected(out: str, families: frozenset[str]) -> list[dict[str, Any]]:
    """The indexed rows of the named families on which AIRWAY wrote a conformance.

    Args:
        out: The output directory.
        families: The declared families to keep.

    Returns:
        The rows.
    """
    return [
        row for row in _read(out, "index.jsonl") if str(row["declared_family"]) in families and row["conf"] is not None
    ]


def _write(out: str, name: str, rows: list[dict[str, Any]]) -> None:
    """Write one JSONL.

    Args:
        out: The output directory.
        name: The file name.
        rows: The rows.
    """
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


STEPS = {
    "index": step_index,
    "infamily": step_infamily,
    "spans": step_spans,
    "heads": step_heads,
    "profile": step_profile,
}
"""Step name to the function that writes its JSONL, in the order they must run."""


def main(argv: list[str] | None = None) -> None:
    """Run the named steps.

    Args:
        argv: Command line, or None for ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="*", choices=list(STEPS), metavar="step")
    parser.add_argument("--run", required=True, help="the corpus run directory holding rows/ and out/")
    parser.add_argument("--out", required=True, help="where the census JSONL is written")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("NW", "10")))
    args = parser.parse_args(argv)
    for step in args.steps or list(STEPS):
        STEPS[step](args.run, args.out, args.workers)


if __name__ == "__main__":
    main()
