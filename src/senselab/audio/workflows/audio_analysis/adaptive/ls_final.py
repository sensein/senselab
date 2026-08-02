"""Label Studio final tracks + resolved-disagreements index (FR-023, tasks.md T032).

Additive only: reads the run's LS bundle, appends ``final__*`` tracks for the fusion
stream's task, and writes copies under ``<out_dir>/final/`` — the original bundle is
never modified. Also emits ``disagreements_resolved.json``: the run's round-1
disagreements annotated with each bucket's final status and the interventions that
touched it — the before/after story of the loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.layout import belief_dir, final_dir

_CONF_BINS = (("high", 0.66), ("medium", 0.33), ("low", 0.0))


def _conf_bin(c: float) -> str:
    for name, lo in _CONF_BINS:
        if c >= lo:
            return name
    return "low"


def _region(
    rid: str, from_name: str, start: float, end: float, labels: list[str] | None = None, text: list[str] | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {"start": round(start, 4), "end": round(end, 4)}
    kind = "labels"
    if labels is not None:
        value["labels"] = labels
    if text is not None:
        value["text"] = text
        kind = "textarea"
    return {"id": rid, "from_name": from_name, "to_name": "audio", "type": kind, "value": value}


def _task_pass_label(task: dict[str, Any]) -> str | None:
    """Infer the pass a LS task belongs to from its region-id prefixes."""
    for pred in task.get("predictions") or []:
        for region in pred.get("result") or []:
            from_name = str(region.get("from_name") or "")
            if "__" in from_name:
                return from_name.split("__", 1)[0]
    return None


def build_final_ls_bundle(
    *,
    out_dir: Path,
    run_dir: Path,
    transcript: dict[str, Any],
    diarization: dict[str, Any],
    speech_presence_rows: list[dict[str, Any]],
    fusion_stream: str,
    iterations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write final/{labelstudio_tasks.json, labelstudio_config.xml, disagreements_resolved.json}."""
    final = final_dir(out_dir)
    final.mkdir(parents=True, exist_ok=True)
    rounds_dir = belief_dir(out_dir) / "rounds"
    report: dict[str, Any] = {}

    # The run bundle is the belief rendered for an annotator — per-pass uncertainty and scene
    # tracks — and it is *input* here: this stage appends the consensus tracks and writes the
    # deliverable next to them. So it lives under ``L2/``. While it lived in ``final/`` this stage
    # read it back out of the directory it was about to write, and in the integrated path the
    # bundle was not written until after the loop had already run, so the read always missed and
    # the stage silently produced nothing — "not found" being indistinguishable from "no bundle".
    tasks_path = belief_dir(run_dir) / "labelstudio_tasks.json"
    config_path = belief_dir(run_dir) / "labelstudio_config.xml"
    if tasks_path.exists() and config_path.exists():
        tasks = json.loads(tasks_path.read_text())
        config = config_path.read_text()

        regions: list[dict[str, Any]] = []
        for i, w in enumerate(transcript.get("words") or []):
            conf = float(w.get("confidence") or 0.0)
            regions.append(
                _region(
                    f"final_word_{i:04d}",
                    "final__consensus_transcript",
                    w["start"],
                    w["end"],
                    labels=[_conf_bin(conf)],
                )
            )
            spk = f" [{w['speaker']}]" if w.get("speaker") else ""
            regions.append(
                _region(
                    f"final_word_{i:04d}",
                    "final__consensus_transcript__text",
                    w["start"],
                    w["end"],
                    text=[f"{w['text']} ({conf:.2f}){spk}"],
                )
            )
        cluster_values = sorted({str(s["cluster_id"]) for s in diarization.get("segments") or []})
        for i, seg in enumerate(diarization.get("segments") or []):
            regions.append(
                _region(
                    f"final_diar_{i:04d}",
                    "final__diarization",
                    seg["start"],
                    seg["end"],
                    labels=[str(seg["cluster_id"])],
                )
            )
        # Presence status runs (merge consecutive equal statuses).
        run_start: float = 0.0
        run_status: str | None = None
        prev_end = 0.0
        status_regions: list[tuple[float, float, str]] = []
        for row in speech_presence_rows:
            status = str(row.get("status") or "open")
            if run_status is None:
                run_start, run_status = row["start"], status
            elif status != run_status or row["start"] > prev_end + 1e-6:
                status_regions.append((run_start, prev_end, run_status))
                run_start, run_status = row["start"], status
            prev_end = row["end"]
        if run_status is not None:
            status_regions.append((run_start, prev_end, run_status))
        for i, (s, e, status) in enumerate(status_regions):
            regions.append(_region(f"final_speech_presence_{i:04d}", "final__speech_presence", s, e, labels=[status]))

        # Attach to the fusion stream's task (fallback: first task).
        target = next((t for t in tasks if _task_pass_label(t) == fusion_stream), tasks[0] if tasks else None)
        if target is not None:
            preds = target.setdefault("predictions", [{"model_version": "adaptive_final", "score": 1.0, "result": []}])
            preds[0].setdefault("result", []).extend(regions)

        # Config: declare the three new tracks before the closing </View>.
        status_values = sorted({s for _, _, s in status_regions} | {"converged", "irreducible", "open"})
        blocks = ['<Labels name="final__consensus_transcript" toName="audio">']
        blocks += [f'  <Label value="{name}"/>' for name, _ in _CONF_BINS]
        blocks += [
            "</Labels>",
            '<TextArea name="final__consensus_transcript__text" toName="audio" perRegion="true" '
            'editable="false" placeholder="Fused word (confidence) [speaker]"/>',
            '<Labels name="final__diarization" toName="audio">',
        ]
        blocks += [f'  <Label value="{v}"/>' for v in cluster_values]
        blocks += ["</Labels>", '<Labels name="final__speech_presence" toName="audio">']
        blocks += [f'  <Label value="{v}"/>' for v in status_values]
        blocks += ["</Labels>"]
        config = config.replace("</View>", "\n".join(blocks) + "\n</View>")

        (final / "labelstudio_tasks.json").write_text(json.dumps(tasks, indent=2))
        (final / "labelstudio_config.xml").write_text(config)
        report["ls_tracks_added"] = ["final__consensus_transcript", "final__diarization", "final__speech_presence"]
        report["n_final_regions"] = len(regions)
    else:
        report["ls_tracks_added"] = []
        report["reason"] = f"run LS bundle not found under {belief_dir(run_dir)}"

    # ── disagreements_resolved.json ──────────────────────────────────────
    # The ranked index of contested buckets is a belief artifact — it says where the fold was
    # least sure — and this stage annotates it with what the loop did about each one. Input, so
    # ``L2/``; the annotated form is the deliverable and stays in ``final/``.
    dis_path = belief_dir(run_dir) / "disagreements.json"
    if dis_path.exists():
        dis = json.loads(dis_path.read_text())
        region_spans = _region_spans(rounds_dir)
        resolved = []
        for entry in dis.get("entries") or []:
            annotated = dict(entry)
            annotated["interventions"] = [
                e["intervention_id"]
                for e in iterations
                if e.get("status") == "fired"
                and e.get("region_id")
                and _overlaps(region_spans.get(e["region_id"]), entry)
                and e.get("axis") == entry.get("axis")
            ]
            annotated["resolution"] = _resolution_for(entry, rounds_dir)
            resolved.append(annotated)
        payload = {
            "source": str(dis_path),
            "entries": resolved,
        }
        (final / "disagreements_resolved.json").write_text(json.dumps(payload, indent=2))
        report["disagreements_resolved"] = len(resolved)
    return report


def _region_spans(rounds_dir: Path) -> dict[str, tuple[float, float]]:
    spans: dict[str, tuple[float, float]] = {}
    if not rounds_dir.is_dir():
        return spans
    for rd in rounds_dir.iterdir():
        f = rd / "regions.json"
        if f.exists():
            try:
                for reg in json.loads(f.read_text()):
                    spans[reg["region_id"]] = (float(reg["core_start"]), float(reg["core_end"]))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
    return spans


def _overlaps(span: tuple[float, float] | None, entry: dict[str, Any]) -> bool:
    if span is None:
        return False
    return float(entry.get("start", 0)) < span[1] and float(entry.get("end", 0)) > span[0]


def _final_belief_index(rounds_dir: Path) -> dict[tuple[str, float, float], dict[str, Any]]:
    """Last round's belief rows indexed by ``(axis, start, end)``.

    Not by stream, and no longer collapsed here: the belief file now holds one row per bucket,
    folded across passes by the writer under a policy it records. This function used to apply its
    own most-doubtful collapse, ``adaptive.plot`` filtered to the fusion stream, and ``evaluate``
    filtered to the transcript's — three answers from one file, only one of which was written
    down. The fold moved to the writer so there is one.
    """
    try:
        import pandas as pd

        last = max(int(p.name) for p in rounds_dir.iterdir() if p.name.isdigit())
    except (OSError, ValueError):
        return {}
    out: dict[tuple[str, float, float], dict[str, Any]] = {}
    for axis in ("speech_presence", "speaker", "asr"):
        f = rounds_dir / str(last) / "belief" / f"{axis}.parquet"
        if not f.exists():
            continue
        for _, row in pd.read_parquet(f).iterrows():
            key = (axis, round(float(row["start"]), 4), round(float(row["end"]), 4))
            out[key] = {
                "status": row.get("status"),
                "irreducible_reason": row.get("irreducible_reason"),
                "final_uncertainty": row.get("uncertainty"),
            }
    return out


def _resolution_for(
    entry: dict[str, Any],
    rounds_dir: Path,
    _cache: dict[str, dict[tuple[str, float, float], dict[str, Any]]] = {},  # noqa: B006 — per-call-site memo
) -> dict[str, Any]:
    # Memoized per rounds_dir + directory mtime so repeated in-process runs that
    # rewrite the same out_dir never read a stale final-belief index.
    try:
        stamp = f"{rounds_dir}|{rounds_dir.stat().st_mtime_ns}"
    except OSError:
        stamp = str(rounds_dir)
    if stamp not in _cache:
        _cache.clear()
        _cache[stamp] = _final_belief_index(rounds_dir)
    index = _cache[stamp]
    # No pass on a disagreements entry any more: an axis is a fold across passes, so the entry
    # names a span of the recording. The belief index is keyed the same way.
    row = index.get(
        (
            str(entry.get("axis")),
            round(float(entry.get("start", 0)), 4),
            round(float(entry.get("end", 0)), 4),
        )
    )
    if row is None:
        return {"status": "bucket_not_in_final_belief"}
    u0 = entry.get("triage_score")
    u1 = row.get("final_uncertainty")
    out = {
        "status": row.get("status"),
        "final_uncertainty": None if u1 is None or u1 != u1 else round(float(u1), 6),
        "delta_from_round1": None if (u0 is None or u1 is None or u1 != u1) else round(float(u1) - float(u0), 6),
    }
    if row.get("irreducible_reason") and row["irreducible_reason"] == row["irreducible_reason"]:
        out["irreducible_reason"] = row["irreducible_reason"]
    return out
