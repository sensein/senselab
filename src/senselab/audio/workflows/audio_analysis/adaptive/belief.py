"""Belief store: provenance-tagged votes + re-aggregation (prototype).

Implements the VoteStore / BeliefRow semantics of
``specs/20260723-225523-dynamic-uncertainty-workflow/contracts/belief-store.md``:

- one *vote* per (axis, bucket, source, stream, scope) with status ``active | shadowed``;
- a vote is never removed from aggregation; what a rule may withdraw is *weight*
  (:meth:`VoteStore.attenuate_source_in_bucket`), floored so the claim stays visible and unable
  to win. Statistical aggregation has no notion of exclusion — only of weight — and a status is
  read as a filter, which is how "attenuate" turns back into "delete" one reader later;
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
from typing import Any, Sequence

from senselab.audio.workflows.audio_analysis.adaptive.types import AxisName
from senselab.audio.workflows.audio_analysis.aggregate import (
    aggregate_asr,
    aggregate_speaker,
    aggregate_speech_presence,
    speech_presence_p_voice,
)
from senselab.audio.workflows.audio_analysis.floors import MIN_EVIDENCE_WEIGHT
from senselab.audio.workflows.audio_analysis.layout import pass_dir
from senselab.audio.workflows.audio_analysis.support import (
    CORROBORATION_POOLING,
    EVIDENCE_WEIGHT_MAP,
    evidence_weight_from_corroboration,
)

AXES: tuple[AxisName, ...] = ("speech_presence", "speaker", "asr")
"""The three uncertainty axes, typed so callers keep the narrowed literal."""

ATTENUATED_AXES: tuple[str, ...] = ("speech_presence", "asr")
"""Axes an uncorroborated speech claim may be attenuated on.

The speaker axis is absent deliberately: evidence that no one spoke here is silent about *which*
speaker it was, and carrying the discount across would be an unmeasured leap.
"""

UNCORROBORATED_SPEECH_CLAIM = "uncorroborated_speech_claim"
"""Reason recorded when a speech claim is attenuated for want of independent corroboration.

Names what was observed. "Hallucination" would name a cause no measurement in this chain can
reach — a quiet, distant or overlapped speaker produces the identical signature.
"""

_META_COLUMNS = (
    "speech_presence_confidence",
    "speech_presence_uncertainty",
    "snr_brouhaha_db",
    "c50_brouhaha_db",
    "snr_spectral_gating_db",
    "snr_peak_db",
    "rolloff_95_hz",
    "proportion_clipped",
    "quality_snr",
    "quality_clip",
    "quality_reverb",
    "quality_bandwidth",
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
    status: str = "active"  # active | shadowed
    shadowed_by: str | None = None
    evidence_weight: float = 1.0
    """How far this vote's assertion is carried by evidence measured about it.

    ``1.0`` means *nothing was measured*, not "measured as fully corroborated" — a factor never
    gathered must not act as a discount. The factors that produced any other value are listed
    individually in ``provenance["evidence_weight_factors"]``; this field is their floored product.
    Separate from ``payload["weight"]``, which is what the link layer concluded about the voter's
    coarseness: multiplying them together in one field would make neither recoverable from the
    round parquet.
    """
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
            "evidence_weight": self.evidence_weight,
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
        # (quality / source-mass / speech_presence columns + the stored aggregate used
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
    def from_harvests(cls, harvests: dict[str, Any], *, round_idx: int = 1, policy: Any = None) -> "VoteStore":  # noqa: ANN401
        """Populate round-1 votes directly from ``compute.harvest_pass`` outputs (T009).

        ``harvests`` maps pass label → ``PassHarvest`` (duck-typed:
        ``speech_presence_evidence`` / ``speaker_votes`` / ``asr_votes`` bucket lists plus
        ``quality_by_bucket`` / ``source_by_bucket``). This is the in-process
        integration point for analyze_audio — no parquet round-trip; the parquet
        ingest path (:meth:`from_run_dir`) remains for artifact-driven runs.

        The store holds *votes*, so the speech-presence axis is linked from its L1 measurements
        under ``policy`` (defaults to the documented anchors) on the way in.
        """
        from senselab.audio.workflows.audio_analysis.speech_presence_link import votes_for_harvest

        store = cls()
        for stream, harvest in harvests.items():
            for axis, buckets in (
                ("speech_presence", votes_for_harvest(harvest, **({"policy": policy} if policy else {}))),
                ("speaker", harvest.speaker_votes),
                ("asr", harvest.asr_votes),
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
                    if axis == "speech_presence":
                        meta: dict[str, Any] = {"stored_within_pass_uncertainty": None}
                        # P2's second trigger reads this; it lives on the harvest
                        # bucket rather than in quality_by_bucket.
                        if bucket.get("frame_dispersion") is not None:
                            meta["frame_dispersion"] = float(bucket["frame_dispersion"])
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

    def attenuate_source_in_bucket(
        self,
        stream: str,
        bucket: tuple[float, float],
        source: str,
        *,
        corroboration: float,
        evidence_sources: Sequence[str],
        reason: str,
        round_idx: int,
        measured_on: tuple[str, tuple[float, float]],
        floor: float = MIN_EVIDENCE_WEIGHT,
        axes: Sequence[str] = ATTENUATED_AXES,
    ) -> list[dict[str, Any]]:
        """Withdraw weight from ``source``'s active votes in ``bucket``, never remove them.

        The withdrawal is proportional to what independent evidence measured there; the votes stay
        active and keep aggregating.

        The caller measures ``corroboration`` so that the quantity which triggered the withdrawal
        and the quantity that sizes it are the same number; the store records where it was taken.
        Because the evidence pool contains only signals that observe presence directly, the
        claimant is never in it, so the measurement does not move when the vote is attenuated —
        the fixed point is reached in one step and re-measuring in a later round returns the same
        number.

        The same evidence does contribute to the presence fold in its own right, so weighting a
        claimant by it does pull that fold toward the evidence a second time. That double use is
        bounded by the floor and by the trigger gate, and it is precisely why the map is the
        identity rather than something sharper: an exponent here would compound a term that is
        already counted twice.

        Args:
            stream: Pass label.
            bucket: The bucket whose votes are attenuated.
            source: The claimant.
            corroboration: Independent evidence for the claim, in ``[0, 1]``.
            evidence_sources: Which voters were asked. Empty means nothing was measured, and the
                caller must not have called at all.
            reason: What was observed — never a claimed cause.
            round_idx: Round the withdrawal happened in.
            measured_on: ``(axis, bucket)`` the corroboration was measured on. An asr vote is
                weighed by a *presence* bucket, so this is not always ``bucket``.
            floor: Minimum weight; see :func:`.support.evidence_weight_from_corroboration`.
            axes: Axes to act on; defaults to :data:`ATTENUATED_AXES`.

        Returns:
            One record per attenuated vote, with ``axis``, ``bucket``, ``source``, ``vote_id``,
            ``previous_weight``, ``evidence_weight`` and ``corroboration``.
        """
        factor = evidence_weight_from_corroboration(corroboration, floor=floor)
        records: list[dict[str, Any]] = []
        for axis in axes:
            for vid in self._index.get((stream, axis, bucket), []):
                v = self._votes[vid]
                if v.source != source or v.status != "active":
                    continue
                previous = float(v.evidence_weight)
                # Floored *after* composing, because a product of floored factors is not itself
                # floored: two rules each withdrawing to 0.05 would otherwise reach 0.0025 and,
                # with a third, effectively zero.
                v.evidence_weight = max(float(floor), previous * factor)
                v.provenance.setdefault("evidence_weight_factors", []).append(
                    {
                        "reason": reason,
                        "round": int(round_idx),
                        "corroboration": float(corroboration),
                        "corroboration_pooling": CORROBORATION_POOLING,
                        "evidence_sources": sorted(str(s) for s in evidence_sources),
                        "measured_on": {"axis": measured_on[0], "bucket": [measured_on[1][0], measured_on[1][1]]},
                        "weight_map": EVIDENCE_WEIGHT_MAP,
                        "floor": float(floor),
                        "factor": float(factor),
                        "evidence_weight_after": float(v.evidence_weight),
                    }
                )
                records.append(
                    {
                        "axis": axis,
                        "bucket": bucket,
                        "source": source,
                        "vote_id": vid,
                        "previous_weight": previous,
                        "evidence_weight": float(v.evidence_weight),
                        "corroboration": float(corroboration),
                    }
                )
        return records

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

    def evidence_weights(self, stream: str, axis: str, bucket: tuple[float, float]) -> dict[str, float]:
        """``{source → evidence_weight}`` for active votes carrying at least one measured factor.

        Sources with no factor are omitted rather than mapped to 1.0, so "unmeasured" stays
        distinguishable from "measured and fully corroborated" at every consumer — including the
        parquet, where an omission and a 1.0 would otherwise read the same.
        """
        out: dict[str, float] = {}
        for vid in self._index.get((stream, axis, bucket), []):
            v = self._votes[vid]
            if v.status == "active" and v.provenance.get("evidence_weight_factors"):
                out[v.source] = float(v.evidence_weight)
        return out

    def has_evidence_weight_factor(
        self, stream: str, axis: str, bucket: tuple[float, float], source: str, *, reason: str
    ) -> bool:
        """Has ``reason`` already been recorded against ``source``'s active vote here?

        The idempotence guard a rule needs now that attenuation no longer changes ``status``.
        Attenuation moves neither the claim nor its corroboration, so without this a rule's
        candidate set is stable across rounds: it re-fires forever for zero gain, ``epsilon``
        never admits it, and convergence C4 (``untried_actions``) never settles.
        """
        for vid in self._index.get((stream, axis, bucket), []):
            v = self._votes[vid]
            if v.source != source or v.status != "active":
                continue
            if any(f.get("reason") == reason for f in v.provenance.get("evidence_weight_factors") or []):
                return True
        return False

    def votes_added_in_round(self, round_idx: int) -> list[Vote]:
        """Votes first added in ``round_idx`` (for the append-only round files)."""
        return [self._votes[vid] for vid in self._round_added.get(round_idx, [])]

    # ── aggregation (pure; FR-006) ─────────────────────────────────────

    def reaggregate_bucket(
        self, stream: str, axis: str, bucket: tuple[float, float], *, aggregator: str
    ) -> dict[str, Any]:
        """Aggregate one bucket's active votes via the existing pure aggregators.

        Attenuated sources stay in ``contributing_sources`` — the record has to show who spoke up,
        and how far their claim was carried — with the withdrawn weights alongside in
        ``attenuated_sources``. An empty weight map is byte-identical to no map, which is what
        keeps :meth:`parity_check` comparing the same quantity it always did.
        """
        votes = self.active_votes(stream, axis, bucket)
        weights = self.evidence_weights(stream, axis, bucket)
        p_voice: float | None = None
        if axis == "speech_presence":
            agg = aggregate_speech_presence(votes, weights=weights)
            p_voice = speech_presence_p_voice(votes, weights=weights)
        elif axis == "speaker":
            agg = aggregate_speaker(votes, raw_vs_enh=None, aggregator=aggregator, evidence_weights=weights)
        else:
            agg = aggregate_asr(votes, aggregator=aggregator, weights=weights)
        return {
            "start": bucket[0],
            "end": bucket[1],
            "within_pass_uncertainty": agg,
            "p_voice": p_voice,
            "contributing_sources": sorted(votes.keys()),
            "attenuated_sources": {k: round(v, 6) for k, v in sorted(weights.items())},
        }

    def parity_check(self, passes: list[str], *, aggregator: str, tol: float = 1e-9) -> dict[str, Any]:
        """Re-aggregate every round-1 bucket and compare against the stored parquet values.

        This is the executable proof that aggregation is a pure function of the
        vote store (tasks.md T007): a nonzero mismatch count means the split
        missed an input.

        The comparison anchors on the **pre-coupling** scale: since FR-019
        (scene→asr coupling, scene-quality-asr US4) the parquet's
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
        """Create an empty belief state using ``aggregator`` for speaker/asr."""
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
            row["attenuated_sources"] = new["attenuated_sources"]
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
