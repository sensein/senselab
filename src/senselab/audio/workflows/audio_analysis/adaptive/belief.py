"""Belief store: provenance-tagged votes + re-aggregation (prototype).

Implements the VoteStore / BeliefRow semantics of
``specs/20260723-225523-dynamic-uncertainty-workflow/contracts/belief-store.md``:

- one *vote* per (axis, bucket, source, stream, scope) with status
  ``active | shadowed | purged_hallucination``;
- region-scoped votes shadow file-scoped votes of the same (source, stream);
- aggregation is a pure function of the active votes, delegated to the
  existing per-axis aggregators (``aggregate.py``) — the harvest/aggregate
  split (research.md D8) demonstrated on real artifacts.

The prototype ingests a completed ``analyze_audio`` run directory: the six
per-pass uncertainty parquets are the round-1 vote population, and the stored
``within_pass_uncertainty`` doubles as a parity oracle for the re-aggregation
path (tasks.md T007).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.adaptive.types import AxisName
from senselab.audio.workflows.audio_analysis.aggregate import (
    aggregate_identity,
    aggregate_presence,
    aggregate_utterance,
    presence_p_voice,
)
from senselab.audio.workflows.audio_analysis.layout import pass_dir

AXES: tuple[AxisName, ...] = ("presence", "identity", "utterance")
"""The three uncertainty axes, typed so callers keep the narrowed literal."""

_META_COLUMNS = (
    "presence_confidence",
    "presence_uncertainty",
    "quality_snr",
    "quality_clip",
    "quality_reverb",
    "quality_bandwidth",
    "quality_uncertainty",
    "src_speech",
    "src_people",
    "src_machine",
    "src_environment",
    "src_dominant",
    "token_entropy",
    "scene_quality_coupling",
    "intensity_weight",
    "raw_within_pass_uncertainty",
    "comparison_status",
)


def bucket_key(start: float, end: float) -> tuple[float, float]:
    """Canonical (start, end) bucket key, rounded for float-stable dict use."""
    return (round(float(start), 6), round(float(end), 6))


@dataclass
class Vote:
    """One source's statement about one bucket on one axis (data-model.md)."""

    axis: str
    bucket: tuple[float, float]
    source: str
    stream: str
    scope: str  # "file" | "region:<id>"
    round: int
    payload: dict[str, Any]
    status: str = "active"  # active | shadowed | purged_hallucination
    shadowed_by: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def vote_id(self) -> str:
        """Deterministic id — same (axis, bucket, source, stream, scope) overwrites itself."""
        raw = f"{self.axis}|{self.bucket[0]}|{self.bucket[1]}|{self.source}|{self.stream}|{self.scope}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def to_record(self) -> dict[str, Any]:
        """Flat dict for parquet/JSON persistence."""
        return {
            "vote_id": self.vote_id,
            "axis": self.axis,
            "bucket_start": self.bucket[0],
            "bucket_end": self.bucket[1],
            "source": self.source,
            "stream": self.stream,
            "scope": self.scope,
            "round": self.round,
            "status": self.status,
            "shadowed_by": self.shadowed_by,
            "payload": json.dumps(self.payload, default=str),
            "provenance": json.dumps(self.provenance, default=str),
        }


class VoteStore:
    """All evidence for one run, indexed by (stream, axis, bucket)."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._votes: dict[str, Vote] = {}
        self._index: dict[tuple[str, str, tuple[float, float]], list[str]] = {}
        # Per-(stream, axis, bucket) row metadata from the ingested parquets
        # (quality / source-mass / presence columns + the stored aggregate used
        # as the round-1 parity oracle).
        self.row_meta: dict[tuple[str, str, tuple[float, float]], dict[str, Any]] = {}
        self._round_added: dict[int, list[str]] = {}

    # ── ingest ─────────────────────────────────────────────────────────

    @classmethod
    def from_run_dir(cls, run_dir: Path, passes: list[str]) -> "VoteStore":
        """Populate round-1 votes from ``<run_dir>/<pass>/uncertainty/<axis>.parquet``."""
        import pandas as pd

        store = cls()
        for stream in passes:
            for axis in AXES:
                pq = pass_dir(run_dir, stream) / "uncertainty" / f"{axis}.parquet"
                if not pq.exists():
                    continue
                df = pd.read_parquet(pq)
                for _, row in df.iterrows():
                    bk = bucket_key(row["start"], row["end"])
                    votes_raw = row.get("model_votes")
                    if isinstance(votes_raw, str):
                        try:
                            votes = json.loads(votes_raw)
                        except json.JSONDecodeError:
                            votes = {}
                    elif isinstance(votes_raw, dict):
                        votes = votes_raw
                    else:
                        votes = {}
                    for source, payload in (votes or {}).items():
                        if not isinstance(payload, dict):
                            continue
                        store.add_vote(
                            Vote(
                                axis=axis,
                                bucket=bk,
                                source=str(source),
                                stream=stream,
                                scope="file",
                                round=1,
                                payload=payload,
                            )
                        )
                    meta: dict[str, Any] = {
                        "stored_within_pass_uncertainty": _float_or_none(row.get("within_pass_uncertainty"))
                    }
                    for col in _META_COLUMNS:
                        if col in df.columns:
                            meta[col] = _json_safe(row.get(col))
                    store.row_meta[(stream, axis, bk)] = meta
        return store

    @classmethod
    def from_harvests(cls, harvests: dict[str, Any], *, round_idx: int = 1) -> "VoteStore":
        """Populate round-1 votes directly from ``compute.harvest_pass`` outputs (T009).

        ``harvests`` maps pass label → ``PassHarvest`` (duck-typed:
        ``presence_votes`` / ``identity_votes`` / ``utterance_votes`` bucket lists plus
        ``quality_by_bucket`` / ``source_by_bucket``). This is the in-process
        integration point for analyze_audio — no parquet round-trip; the parquet
        ingest path (:meth:`from_run_dir`) remains for artifact-driven runs.
        """
        store = cls()
        for stream, harvest in harvests.items():
            for axis, buckets in (
                ("presence", harvest.presence_votes),
                ("identity", harvest.identity_votes),
                ("utterance", harvest.utterance_votes),
            ):
                for bucket in buckets:
                    bk = bucket_key(bucket["start"], bucket["end"])
                    for source, payload in (bucket.get("votes") or {}).items():
                        if not isinstance(payload, dict):
                            continue
                        store.add_vote(
                            Vote(
                                axis=axis,
                                bucket=bk,
                                source=str(source),
                                stream=stream,
                                scope="file",
                                round=round_idx,
                                payload=payload,
                            )
                        )
                    if axis == "presence":
                        meta: dict[str, Any] = {"stored_within_pass_uncertainty": None}
                        # P2's second trigger reads this; it lives on the harvest
                        # bucket rather than in quality_by_bucket.
                        if bucket.get("frame_instability") is not None:
                            meta["frame_instability"] = float(bucket["frame_instability"])
                        q = harvest.quality_by_bucket.get(bk)
                        s = harvest.source_by_bucket.get(bk)
                        if q:
                            meta.update({k: _json_safe(v) for k, v in q.items() if k in _META_COLUMNS})
                        if s:
                            meta.update({k: _json_safe(v) for k, v in s.items() if k in _META_COLUMNS})
                        store.row_meta[(stream, axis, bk)] = meta
                    else:
                        store.row_meta.setdefault((stream, axis, bk), {"stored_within_pass_uncertainty": None})
        return store

    # ── mutation ───────────────────────────────────────────────────────

    def add_vote(self, vote: Vote) -> None:
        """Insert/overwrite a vote; region scope shadows same (source, stream) file scope."""
        vid = vote.vote_id
        self._votes[vid] = vote
        key = (vote.stream, vote.axis, vote.bucket)
        ids = self._index.setdefault(key, [])
        if vid not in ids:
            ids.append(vid)
        self._round_added.setdefault(vote.round, []).append(vid)
        if vote.scope.startswith("region:"):
            for other_id in ids:
                other = self._votes[other_id]
                if other_id != vid and other.source == vote.source and other.scope == "file":
                    if other.status == "active":
                        other.status = "shadowed"
                        other.shadowed_by = vid

    def purge_source_in_bucket(
        self, stream: str, bucket: tuple[float, float], source: str, *, reason: str, round_idx: int
    ) -> int:
        """Mark ``source``'s votes in ``bucket`` purged on presence + utterance axes (C10)."""
        n = 0
        for axis in ("presence", "utterance"):
            for vid in self._index.get((stream, axis, bucket), []):
                v = self._votes[vid]
                if v.source == source and v.status == "active":
                    v.status = "purged_hallucination"
                    v.provenance["purge_reason"] = reason
                    v.provenance["purge_round"] = round_idx
                    n += 1
        return n

    # ── reads ──────────────────────────────────────────────────────────

    def buckets(self, stream: str, axis: str) -> list[tuple[float, float]]:
        """All known buckets for (stream, axis), time-ordered."""
        got = {bk for (s, a, bk) in self._index if s == stream and a == axis}
        got |= {bk for (s, a, bk) in self.row_meta if s == stream and a == axis}
        return sorted(got)

    def active_votes(self, stream: str, axis: str, bucket: tuple[float, float]) -> dict[str, dict[str, Any]]:
        """Vote dict (source → payload) of active votes, as the aggregators expect."""
        out: dict[str, dict[str, Any]] = {}
        for vid in self._index.get((stream, axis, bucket), []):
            v = self._votes[vid]
            if v.status == "active":
                out[v.source] = v.payload
        return out

    def votes_added_in_round(self, round_idx: int) -> list[Vote]:
        """Votes first added in ``round_idx`` (for the append-only round files)."""
        return [self._votes[vid] for vid in self._round_added.get(round_idx, [])]

    # ── aggregation (pure; FR-006) ─────────────────────────────────────

    def reaggregate_bucket(
        self, stream: str, axis: str, bucket: tuple[float, float], *, aggregator: str
    ) -> dict[str, Any]:
        """Aggregate one bucket's active votes via the existing pure aggregators."""
        votes = self.active_votes(stream, axis, bucket)
        p_voice: float | None = None
        if axis == "presence":
            agg = aggregate_presence(votes)
            p_voice = presence_p_voice(votes)
        elif axis == "identity":
            agg = aggregate_identity(votes, raw_vs_enh=None, aggregator=aggregator)
        else:
            agg = aggregate_utterance(votes, aggregator=aggregator)
        return {
            "start": bucket[0],
            "end": bucket[1],
            "within_pass_uncertainty": agg,
            "p_voice": p_voice,
            "contributing_sources": sorted(votes.keys()),
        }

    def parity_check(self, passes: list[str], *, aggregator: str, tol: float = 1e-9) -> dict[str, Any]:
        """Re-aggregate every round-1 bucket and compare against the stored parquet values.

        This is the executable proof that aggregation is a pure function of the
        vote store (tasks.md T007): a nonzero mismatch count means the split
        missed an input.

        The comparison anchors on the **pre-coupling** scale: since FR-019
        (scene→utterance coupling, scene-quality-utterance US4) the parquet's
        ``within_pass_uncertainty`` may carry a scene multiplier that is not a
        function of the votes alone — the pure per-vote value is preserved on
        ``raw_within_pass_uncertainty``, which is what the belief store computes
        and compares (identical to ``within_pass_uncertainty`` on pre-FR-019
        artifacts and wherever coupling is 1.0).
        """
        report: dict[str, Any] = {}
        for stream in passes:
            for axis in AXES:
                n = mismatches = compared = 0
                max_abs = 0.0
                for bk in self.buckets(stream, axis):
                    n += 1
                    meta = self.row_meta.get((stream, axis, bk)) or {}
                    stored = meta.get("raw_within_pass_uncertainty")
                    if stored is None or stored != stored:  # NaN/missing → legacy column
                        stored = meta.get("stored_within_pass_uncertainty")
                    got = self.reaggregate_bucket(stream, axis, bk, aggregator=aggregator)["within_pass_uncertainty"]
                    if stored is None or got is None:
                        if (stored is None) != (got is None):
                            mismatches += 1
                        continue
                    compared += 1
                    diff = abs(float(stored) - float(got))
                    max_abs = max(max_abs, diff)
                    if diff > tol:
                        mismatches += 1
                report[f"{stream}/{axis}"] = {
                    "buckets": n,
                    "compared": compared,
                    "mismatches": mismatches,
                    "max_abs_diff": max_abs,
                }
        return report


class BeliefState:
    """Aggregated per-bucket state per (stream, axis), updated each round."""

    def __init__(self, aggregator: str) -> None:
        """Create an empty belief state using ``aggregator`` for identity/utterance."""
        self.aggregator = aggregator
        self.rows: dict[tuple[str, str], list[dict[str, Any]]] = {}

    @classmethod
    def from_store(cls, store: VoteStore, passes: list[str], *, aggregator: str) -> "BeliefState":
        """Round-1 belief: aggregate every bucket, attach meta + epistemic/aleatoric split."""
        state = cls(aggregator)
        for stream in passes:
            for axis in AXES:
                rows = []
                for bk in store.buckets(stream, axis):
                    row = store.reaggregate_bucket(stream, axis, bk, aggregator=aggregator)
                    meta = store.row_meta.get((stream, axis, bk)) or {}
                    row["meta"] = meta
                    _decompose(row, meta)
                    row["status"] = "open"
                    row["round"] = 1
                    row["history"] = [{"round": 1, "within_pass_uncertainty": row["within_pass_uncertainty"]}]
                    rows.append(row)
                state.rows[(stream, axis)] = rows
        return state

    def update_buckets(
        self, store: VoteStore, stream: str, axis: str, buckets: set[tuple[float, float]], round_idx: int
    ) -> list[dict[str, Any]]:
        """Incrementally re-aggregate only ``buckets`` (FR-006); returns changed rows."""
        changed = []
        for row in self.rows.get((stream, axis), []):
            bk = bucket_key(row["start"], row["end"])
            if bk not in buckets:
                continue
            new = store.reaggregate_bucket(stream, axis, bk, aggregator=self.aggregator)
            row["within_pass_uncertainty"] = new["within_pass_uncertainty"]
            if new["p_voice"] is not None:
                row["p_voice"] = new["p_voice"]
            row["contributing_sources"] = new["contributing_sources"]
            _decompose(row, row.get("meta") or {})
            row["round"] = round_idx
            row["history"].append({"round": round_idx, "within_pass_uncertainty": row["within_pass_uncertainty"]})
            changed.append(row)
        return changed

    def axis_rows(self, stream: str, axis: str) -> list[dict[str, Any]]:
        """Rows for one (stream, axis), time-ordered."""
        return self.rows.get((stream, axis), [])

    def uncertainty_mass(self, stream: str, axis: str, theta_low: float) -> float:
        """Σ max(0, u − θ_low) · width — the quantity interventions try to shrink."""
        total = 0.0
        for row in self.axis_rows(stream, axis):
            u = row.get("within_pass_uncertainty")
            if u is None:
                continue
            total += max(0.0, float(u) - theta_low) * (float(row["end"]) - float(row["start"]))
        return total


def _decompose(row: dict[str, Any], meta: dict[str, Any]) -> None:
    """Epistemic/aleatoric split (research.md D7).

    Floor = max(quality degradation, overlap posterior). The overlap term is
    populated by I4 when segmentation-3.0 per-class posteriors are available;
    otherwise the floor degrades to the quality-driven term only.
    """
    agg = row.get("within_pass_uncertainty")
    floor = 0.0
    for col in ("quality_snr", "quality_clip", "quality_reverb", "overlap_posterior"):
        v = meta.get(col)
        if v is not None:
            try:
                if v == v:  # NaN guard
                    floor = max(floor, min(1.0, max(0.0, float(v))))
            except (TypeError, ValueError):
                pass
    row["aleatoric_floor"] = floor
    row["epistemic"] = max(0.0, float(agg) - floor) if agg is not None else None


def _float_or_none(v: Any) -> float | None:  # noqa: ANN401
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _json_safe(v: Any) -> Any:  # noqa: ANN401
    """Coerce numpy scalars / NaN to plain JSON-safe python values."""
    if v is None or isinstance(v, (str, bool)):
        return v
    try:
        import numpy as np

        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:
        pass
    if isinstance(v, float) and v != v:
        return None
    return v
