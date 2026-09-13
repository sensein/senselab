"""Before/after the frame-fill removal, recomputed from the ten already-run 20260908 stores.

The ten stores at ``~/Downloads/triage_10subj_20260908/out/`` were produced by the *old*
``span_yamnet_input`` (periodic fill for a span under YAMNet's 0.96 s native frame). "Before" reads
their ``span_yamnet`` measurements exactly as stored. "After" recomputes each filled span's score
under the new rule -- the overlap-weighted mean of the whole-file YAMNet windows covering it, via
the pipeline's own ``_covering_window_attribution`` -- and drops any span nothing covers, without
re-running YAMNet or any other model.

The whole-file windows come from ``derivatives/yamnet_scores.json`` rather than from a
``yamnet_window`` store entity: these ten stores were produced under the packaged default config
(``windows.yamnet.default_threshold: null``), under which ``_windows("yamnet")`` never ran and
wrote no ``yamnet_window`` entities at all -- confirmed empirically (none of the ten stores has
one). ``yamnet_scores.json`` is the same underlying per-window model output ``_windows`` would have
wrapped into entities; only the labelling threshold gate is skipped, which affects decision labels,
not the raw scores this recomputation reads.

Usage:
    uv run python frame_fill_removal_recompute.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.nodes.figure import _raster_rows
from senselab.audio.workflows.triage.nodes.preprocess import _covering_window_attribution
from senselab.audio.workflows.triage.nodes.taxonomy import _consolidate, _per_span_label_scores
from senselab.utils.prov_store import ProvStore

ARTEFACT_FAMILY = {
    "Synthesizer",
    "Keyboard (musical)",
    "Toothbrush",
    "Stomach rumble",
    "Heart sounds, heartbeat",
    "Throbbing",
    "Sidetone",
    "Effects unit",
}
AIRWAY_LABELS = {"Breathing", "Cough", "Gasp", "Snort", "Wheeze", "Sneeze", "Sigh", "Snoring"}
CONSOLIDATION_FLOOR = 0.2
RASTER_TOP_K = 4

ROOT = Path("~/Downloads/triage_10subj_20260908/out").expanduser()


def load_whole_file_windows(run_dir: Path) -> list[dict[str, Any]]:
    """``derivatives/yamnet_scores.json`` -- the raw, unpadded whole-file YAMNet windows."""
    path = run_dir / "derivatives" / "yamnet_scores.json"
    result: list[dict[str, Any]] = json.loads(path.read_text())
    return result


def span_yamnet_rows(store: ProvStore) -> dict[str, list[dict[str, Any]]]:
    """Every live ``span_yamnet`` measurement's attributes, grouped by the span it scored."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for entity in store.entities("measurement"):
        if entity.attributes.get("name") != "span_yamnet":
            continue
        span_id = entity.attributes.get("span_id")
        if span_id is None:
            continue
        rows.setdefault(str(span_id), []).append(entity.attributes)
    return rows


def _max_pool(rows: list[dict[str, Any]]) -> dict[str, float]:
    """One span's rows folded to the best score per label -- ``_per_span_label_scores``'s own rule."""
    slot: dict[str, float] = {}
    for row in rows:
        for label, score in (row.get("raw_scores") or {}).items():
            slot[str(label)] = max(slot.get(str(label), 0.0), float(score))
    return slot


def per_span_scores_before(rows_by_span: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """Exactly what the store holds: every row's ``raw_scores``, filled or not, pooled by max."""
    return {span_id: _max_pool(rows) for span_id, rows in rows_by_span.items()}


def per_span_scores_after(
    store: ProvStore,
    rows_by_span: dict[str, list[dict[str, Any]]],
    whole_file_windows: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """A filled span's rows replaced by covering-window attribution; native rows are untouched.

    A span with no covering window is dropped -- unmeasured, not scored with an invented number.
    """
    out: dict[str, dict[str, float]] = {}
    for span_id, rows in rows_by_span.items():
        if not rows[0].get("frame_filled"):
            out[span_id] = _max_pool(rows)
            continue
        span = store.get_entity(span_id)
        if span.extent is None:
            continue
        attribution = _covering_window_attribution(span.extent, whole_file_windows)
        if attribution is None:
            continue
        scores, _n, _seconds = attribution
        out[span_id] = scores
    return out


def top_k_at_floor(scores: dict[str, float], floor: float, k: int = 4) -> list[str]:
    """A span's own top ``k`` labels by score, restricted to those clearing ``floor``."""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
    return [label for label, score in ranked if score >= floor]


NOT_ARTEFACT = {"Speech", "Silence"} | AIRWAY_LABELS


def count_broad_non_speech_occurrences(per_span: dict[str, dict[str, float]], floor: float = 0.2) -> int:
    """Supplementary, wider check: anything but Speech/Silence/airway in a span's own top-4.

    ``ARTEFACT_FAMILY`` is the eight label names the brief named explicitly; this counts every
    *other* label too, since the fill manufactures far more names than those eight (``Noise``,
    ``Engine``, ``Hum``, ``Music``, ``Mechanisms``, ``Buzzer``, ... -- see the per-label breakdown
    this script's caller ran separately). It over-counts relative to a strict "artefact family"
    definition -- a real animal or vehicle sound would also land here -- so it is reported beside
    the named-family count, not instead of it.
    """
    return sum(
        1 for scores in per_span.values() for label in top_k_at_floor(scores, floor) if label not in NOT_ARTEFACT
    )


def count_family_occurrences(per_span: dict[str, dict[str, float]], family: set[str], floor: float) -> int:
    """How many times a member of ``family`` appears in some span's own top-4 at or above ``floor``."""
    return sum(1 for scores in per_span.values() for label in top_k_at_floor(scores, floor) if label in family)


def consensus_taxonomy_labels(store: ProvStore, yamnet_per_span: dict[str, dict[str, float]]) -> set[str]:
    """The label set ``consensus_taxonomy`` would hold, folding HeAR (unchanged) with the given YAMNet scores."""
    hear_consolidated = _consolidate(_per_span_label_scores(store, "span_hear"), CONSOLIDATION_FLOOR)
    yamnet_consolidated = _consolidate(yamnet_per_span, CONSOLIDATION_FLOOR)
    return set(hear_consolidated) | set(yamnet_consolidated)


def main() -> None:
    """Print the before/after table across the ten stores."""
    store_paths = sorted(ROOT.glob("*/*/run/store.jsonl"))
    if len(store_paths) != 10:
        raise SystemExit(f"expected 10 stores under {ROOT}, found {len(store_paths)}")

    totals = {
        "artefact_before": 0,
        "artefact_after": 0,
        "airway_before": 0,
        "airway_after": 0,
        "broad_before": 0,
        "broad_after": 0,
    }
    per_recording: list[dict[str, Any]] = []

    for store_path in store_paths:
        run_dir = store_path.parent
        subject = store_path.parts[-4]
        store = ProvStore.read_jsonl(store_path)
        whole_file_windows = load_whole_file_windows(run_dir)

        rows_by_span = span_yamnet_rows(store)
        before = per_span_scores_before(rows_by_span)
        after = per_span_scores_after(store, rows_by_span, whole_file_windows)

        artefact_before = count_family_occurrences(before, ARTEFACT_FAMILY, 0.2)
        artefact_after = count_family_occurrences(after, ARTEFACT_FAMILY, 0.2)
        airway_before = count_family_occurrences(before, AIRWAY_LABELS, 0.5)
        airway_after = count_family_occurrences(after, AIRWAY_LABELS, 0.5)
        broad_before = count_broad_non_speech_occurrences(before)
        broad_after = count_broad_non_speech_occurrences(after)

        tax_before = len(consensus_taxonomy_labels(store, before))
        tax_after = len(consensus_taxonomy_labels(store, after))

        raster_before = len(_raster_rows(before, RASTER_TOP_K, "file", CONSOLIDATION_FLOOR))
        raster_after = len(_raster_rows(after, RASTER_TOP_K, "file", CONSOLIDATION_FLOOR))

        totals["artefact_before"] += artefact_before
        totals["artefact_after"] += artefact_after
        totals["airway_before"] += airway_before
        totals["airway_after"] += airway_after
        totals["broad_before"] += broad_before
        totals["broad_after"] += broad_after

        n_short = sum(1 for rows in rows_by_span.values() if rows[0].get("frame_filled"))
        n_dropped = n_short - sum(
            1 for span_id in rows_by_span if rows_by_span[span_id][0].get("frame_filled") and span_id in after
        )

        per_recording.append(
            {
                "subject": subject,
                "n_spans": len(rows_by_span),
                "n_short_filled_before": n_short,
                "n_short_dropped_after": n_dropped,
                "artefact_before": artefact_before,
                "artefact_after": artefact_after,
                "airway_before": airway_before,
                "airway_after": airway_after,
                "broad_before": broad_before,
                "broad_after": broad_after,
                "tax_before": tax_before,
                "tax_after": tax_after,
                "raster_before": raster_before,
                "raster_after": raster_after,
            }
        )

    header = [
        "subject",
        "spans",
        "short(before)",
        "short->unmeasured",
        "artefact b/a",
        "airway b/a",
        "broad b/a",
        "tax b/a",
        "raster b/a",
    ]
    print(" | ".join(header))
    for row in per_recording:
        print(
            " | ".join(
                [
                    row["subject"],
                    str(row["n_spans"]),
                    str(row["n_short_filled_before"]),
                    str(row["n_short_dropped_after"]),
                    f"{row['artefact_before']}/{row['artefact_after']}",
                    f"{row['airway_before']}/{row['airway_after']}",
                    f"{row['broad_before']}/{row['broad_after']}",
                    f"{row['tax_before']}/{row['tax_after']}",
                    f"{row['raster_before']}/{row['raster_after']}",
                ]
            )
        )

    print()
    print(f"TOTAL artefact-family (8 named) top-4>=0.2:  before={totals['artefact_before']}  after={totals['artefact_after']}")
    print(f"TOTAL airway top-4>=0.5:                     before={totals['airway_before']}  after={totals['airway_after']}")
    print(f"TOTAL broad non-speech/silence/airway top-4>=0.2: before={totals['broad_before']}  after={totals['broad_after']}")
    tax_before_range = (min(r["tax_before"] for r in per_recording), max(r["tax_before"] for r in per_recording))
    tax_after_range = (min(r["tax_after"] for r in per_recording), max(r["tax_after"] for r in per_recording))
    print(f"consensus_taxonomy n_labels range: before={tax_before_range}  after={tax_after_range}")
    raster_before_range = (min(r["raster_before"] for r in per_recording), max(r["raster_before"] for r in per_recording))
    raster_after_range = (min(r["raster_after"] for r in per_recording), max(r["raster_after"] for r in per_recording))
    print(f"raster row count range:            before={raster_before_range}  after={raster_after_range}")


if __name__ == "__main__":
    main()
