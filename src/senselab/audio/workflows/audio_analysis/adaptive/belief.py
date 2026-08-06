"""Belief store: provenance-tagged votes + re-aggregation (prototype).

Implements the VoteStore / BeliefRow semantics of
``specs/20260723-225523-dynamic-uncertainty-workflow/contracts/belief-store.md``:

- one *vote* per (axis, bucket, source, stream, scope) with status ``active | shadowed``;
- a vote is never removed from aggregation; what a rule may withdraw is *weight*
  (:meth:`VoteStore.attenuate_source_in_bucket`), floored so the claim stays visible and unable
  to win. Statistical aggregation has no notion of exclusion — only of weight — and a status is
  read as a filter, which is how "attenuate" turns back into "delete" one reader later;
- region-scoped votes shadow file-scoped votes of the same (source, stream);
- aggregation is a pure function of the active votes.

**An axis is an aggregator across signals and across passes alike.** A vote is legitimately keyed
``(axis, bucket, source, pass, scope)`` — a signal measured on a pass is a per-pass measurement,
and it is exactly what makes perturbation stability computable. An *axis* may not be: a pass is an
input dimension to the fold, never an index on its output. So :meth:`VoteStore.reaggregate_bucket`
takes no stream, and the fold across passes is :func:`fuse.fuse_axis` — the same one
``compute_uncertainty_axes`` performs — rather than a second implementation living here. The two
L2 round tree (``L2/round/<n>/estimates/``) then answers the same question
with the same arithmetic, which :meth:`VoteStore.fused_parity` checks against what was written.

What that replaced was not a redundancy. The store held one belief row per (stream, axis, bucket),
so "converged on raw, open on enhanced" was a state a bucket could be in; every reader invented its
own collapse, and the writer's ``elected_stream`` — one pass's reading taken as the run's — was a
per-pass axis with the index moved into the value.

Ingest is the **linked evidence at the vote level** — ``L2/round/0/derivatives/votes/<axis>.parquet`` on the
artifact path, the ``PassHarvest`` objects on the in-process path. Re-derivability is proved by
:meth:`VoteStore.replay_check`, which rebuilds each bucket from what is persisted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.adaptive.types import AxisName
from senselab.audio.workflows.audio_analysis.aggregate import per_source_voice
from senselab.audio.workflows.audio_analysis.axes import ATTENUATED_AXES, AXIS_NAMES, HARVEST_SOURCES
from senselab.audio.workflows.audio_analysis.degradation import (
    DEFAULT_ANCHORS,
    SNR_PREFERENCE,
    clip_degradation,
    reverb_degradation,
    snr_degradation,
)
from senselab.audio.workflows.audio_analysis.estimates import control_doubt
from senselab.audio.workflows.audio_analysis.floors import MIN_EVIDENCE_WEIGHT
from senselab.audio.workflows.audio_analysis.fuse import SnrGate, fuse_axis
from senselab.audio.workflows.audio_analysis.layout import derivatives_dir, evidence_dir, perturbation_dir
from senselab.audio.workflows.audio_analysis.support import (
    CORROBORATION_POOLING,
    EVIDENCE_WEIGHT_MAP,
    evidence_weight_from_corroboration,
)

AXES: tuple[AxisName, ...] = AXIS_NAMES
"""Every active axis, from the one declaration. Re-exported here because half the loop imports it
from this module; it is not a second list, and it is deliberately not three long."""

_HARVEST_ACCESSORS: Final[frozenset[str]] = frozenset(HARVEST_SOURCES)
"""Axes :meth:`VoteStore.from_harvests` can read straight off a ``PassHarvest``.

Derived from :data:`~.axes.HARVEST_SOURCES` — what a reader can actually *find* — and not from the
``harvested`` flag, which is only what the axis claims. The distinction is the bug: the flag said
``background_mask`` was harvested, this method enumerated three axes in a literal tuple, and
``frozenset(HARVESTED_AXES)`` then reported the mask as covered. So the guard below could not fire,
the caller's ``unharvested`` entry was accepted instead, and the axis was rebuilt from one vote per
mask *region* — 1070 buckets at round 0, one by round 4. Keyed on the declaration a reader
dereferences, an axis nothing can read is *not* in this set and the guard raises."""

UNCORROBORATED_SPEECH_CLAIM = "uncorroborated_speech_claim"
"""Reason recorded when a speech claim is attenuated for want of independent corroboration.

Names what was observed. "Hallucination" would name a cause no measurement in this chain can
reach — a quiet, distant or overlapped speaker produces the identical signature.
"""

_META_COLUMNS = (
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
    "frame_dispersion",
)
"""Per-bucket measurements the store carries alongside its votes.

Deliberately *only* measurements, in their native units. The columns removed from this tuple —
``speech_presence_confidence``, ``speech_presence_uncertainty``, ``raw_within_pass_uncertainty``,
``comparison_status``, ``intensity_weight``, ``scene_quality_coupling`` — were not measurements:
the first four are the per-pass axis fold or a function of it, and the last two are L2 decisions
(a cross-axis reduction and a policy multiplier). Nothing in the adaptive subsystem read any of
them; they were carried and dropped.

``quality_snr`` and its siblings are gone for the same reason from the other side: they are
*scores*, anchored against a calibration profile, and neither ingest path ever carried one —
the harvest holds dB and the fused presence parquet holds neither. So ``aleatoric_floor`` read a
name that existed nowhere, took ``None``, and floored at ``0.0`` on every bucket of every run,
which is the confident claim "this audio imposes no floor". The scores are now derived here from
the dB columns under named anchors (:func:`_attach_floor`), which is where an anchored score
belongs anyway.
"""

_MEASUREMENT_SOURCES: dict[str, str | None] = {
    "__quality__": None,
    "__sources__": None,
    "__frame_dispersion__": "frame_dispersion",
}
"""Vote-file entries that are bucket measurements rather than a source's statement.

Keyed by name, not by payload shape. The shape test this replaces — "starts and ends with ``__``
and has a ``value`` key" — swallowed ``__cross_diar_label_disagreement__``, which is a *signal*
(the fraction of diarizer pairs that disagree here) and reports its reading under ``value`` like
any other. The artifact path therefore dropped it from the speaker fold while the in-process path
kept it, so the two ingests disagreed by exactly one signal and nothing said so.

The value is the meta key a single-valued payload lands under; ``None`` means merge the payload's
own keys (filtered to :data:`_META_COLUMNS`).
"""


_COUPLINGS_LEAVING_UNCERTAINTY_ALONE = frozenset({"scene_quality"})
"""Couplings that move a row's policy fold but not its entropy measure.

Named so :meth:`VoteStore.fused_parity` can still compare a row the scene coupling touched. The
coupling multiplies ``triage_score``, which exists to rank where budget goes; ``uncertainty`` has
no policy in it and is untouched.
"""


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


def snr_gate_from_run(run_dir: Path, *, floor_db: float) -> "SnrGate | None":
    """Rebuild the run's admission gate from what the run recorded about itself.

    The loop re-aggregates the same votes fusion folded, so it has to be gated **identically** or
    it is not a replay of the published axis — it is a second, differently-gated fold reported
    under the same name. That is not hypothetical: the gate reached round 0 only, and the loop's
    ungated re-aggregation folded the enhanced pass back in, so ``final/`` published 0.2267 on a
    recording whose round 0 read 0.0487. The axis appeared not to have changed at all.

    Both inputs come from the run's own artifacts rather than from a caller:

    - which perturbations are gated, from ``L1/perturbations.json`` — the register records each
      one's *declared transform*, which is exactly what ``SNR_GATED_TRANSFORMS`` is keyed on. Read
      from there rather than re-derived, so a standalone loop run on someone else's run directory
      cannot disagree with the fold that produced it.
    - identity-pass SNR per bucket, from ``L1/signals/scene_quality.parquet``.

    Returns ``None`` when the run has nothing to gate, which is the correct gate for a run whose
    only perturbation is the identity.
    """
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.perturbations import (
        IDENTITY_NAME,
        REGISTER_FILENAME,
        Perturbation,
    )

    register = evidence_dir(run_dir) / REGISTER_FILENAME
    if not register.exists():
        return None
    try:
        entries = (json.loads(register.read_text()).get("perturbations")) or []
    except (OSError, json.JSONDecodeError):
        return None
    gated = frozenset(str(e["name"]) for e in entries if Perturbation.from_json(e).admission_requires_low_snr)
    if not gated:
        return None

    snr: dict[tuple[float, float], float | None] = {}
    pq = evidence_dir(run_dir) / "signals" / "scene_quality.parquet"
    if pq.exists():
        frame = pd.read_parquet(pq)
        for row in frame[frame["perturbation"] == IDENTITY_NAME].itertuples():
            payload = row.measurement
            if isinstance(payload, str):
                payload = json.loads(payload)
            snr[(round(float(row.start), 6), round(float(row.end), 6))] = (payload or {}).get("snr_brouhaha_db")
    return SnrGate(floor_db=float(floor_db), snr_db_by_bucket=snr, gated_passes=gated)


class VoteStore:
    """All evidence for one run, indexed by (stream, axis, bucket)."""

    def __init__(self, *, snr_gate: SnrGate | None = None) -> None:
        """Create an empty store.

        Args:
            snr_gate: Which perturbations may contribute in which buckets when a bucket is
                re-aggregated. **Must be the same gate ``compute`` folded round 0 with**, or the
                loop's re-aggregation stops being a parity check on the published axis and starts
                being a second, differently-gated fold reported under the same name. ``None`` means
                no gating, which is correct for a run whose only perturbation is the identity.
        """
        self.snr_gate = snr_gate
        self._votes: dict[str, Vote] = {}
        self._index: dict[tuple[str, str, tuple[float, float]], list[str]] = {}
        # Per-(stream, axis, bucket) measurements that belong to the bucket rather than to any
        # one source: scene quality in native units, the L2 quality scores derived from them,
        # source-category masses, token entropy, frame dispersion. No axis value lives here.
        self.row_meta: dict[tuple[str, str, tuple[float, float]], dict[str, Any]] = {}
        self._round_added: dict[int, list[str]] = {}

    # ── ingest ─────────────────────────────────────────────────────────

    @classmethod
    def from_run_dir(cls, run_dir: Path, passes: list[str], *, snr_gate: "SnrGate | None" = None) -> "VoteStore":
        """Populate round-0 votes from ``<run_dir>/L2/round/0/derivatives/votes/<axis>.parquet``.

        Args:
            run_dir: The finished run to ingest.
            passes: The perturbations whose votes to take.
            snr_gate: The gate fusion folded round 0 with — build it with :func:`snr_gate_from_run`
                so this re-aggregation is a replay of the published axis rather than a second fold
                under the same name. ``None`` re-aggregates ungated.

        Ingests the **linked evidence at the vote level**, which is legitimately keyed
        ``(axis, bucket, source, pass, scope)``. It used to read
        ``L1/<pass>/uncertainty/<axis>.parquet`` — a per-pass axis fold, which is a quantity that
        cannot exist — and to keep that fold as a parity oracle against its own recomputation.
        Both are gone: this path now sees exactly what the in-process path
        (:meth:`from_harvests`) sees, which :meth:`fused_parity` is able to check because the two
        now fold identically.

        The per-bucket scene measurements travel in the vote file itself, under the reserved names
        of :data:`_MEASUREMENT_SOURCES`. They used to be joined from
        ``L2/round/0/estimates/speech_presence.parquet``, a file that carries none of them: the
        intersection was empty on every run, the join returned early, and every bucket's
        measurements were silently absent.

        Every vote row names the perturbation it was measured under, and a row naming one this run
        did not take is skipped. That is a real filter and it silently ate the fourth axis: the
        mask votes were written under a *fabricated* perturbation called ``"mask"``, which is in no
        run's perturbation set, so this path dropped every one of them. The mask is measured on the
        unmodified recording and now says so.
        """
        import pandas as pd

        store = cls(snr_gate=snr_gate)
        votes_dir = derivatives_dir(run_dir, 0) / "votes"
        for axis in AXES:
            pq = votes_dir / f"{axis}.parquet"
            if not pq.exists():
                continue
            frame = pd.read_parquet(pq)
            for _, row in frame.iterrows():
                bk = bucket_key(row["start"], row["end"])
                stream = str(row["perturbation"])
                if stream not in passes:
                    continue
                source = str(row["source"])
                try:
                    payload = json.loads(row["payload"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if source in _MEASUREMENT_SOURCES:
                    store._record_measurement(stream, axis, bk, source, payload)
                    continue
                store.add_vote(
                    Vote(axis=axis, bucket=bk, source=source, stream=stream, scope="file", round=0, payload=payload)
                )
                store.row_meta.setdefault((stream, axis, bk), {})
        return store

    def _record_measurement(
        self, stream: str, axis: str, bucket: tuple[float, float], source: str, payload: dict[str, Any]
    ) -> None:
        """File one reserved vote-file entry as a bucket measurement."""
        slot = self.row_meta.setdefault((stream, axis, bucket), {})
        key = _MEASUREMENT_SOURCES[source]
        if key is not None:
            slot[key] = _json_safe(payload.get("value"))
            return
        slot.update({k: _json_safe(v) for k, v in payload.items() if k in _META_COLUMNS})

    @classmethod
    def from_harvests(
        cls,
        harvests: dict[str, Any],
        *,
        round_idx: int = 0,
        policy: Any = None,  # noqa: ANN401
        unharvested: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]] | None = None,
        snr_gate: "SnrGate | None" = None,
    ) -> "VoteStore":
        """Populate baseline-round votes directly from ``compute.harvest_pass`` outputs (T009).

        ``snr_gate`` must be the gate fusion folded round 0 with (see :func:`snr_gate_from_run`), or
        every re-aggregation here silently folds a perturbation fusion excluded — which is how
        ``final/`` came to publish 0.2267 for an axis whose round 0 read 0.0487.

        ``harvests`` maps pass label → ``PassHarvest`` (duck-typed:
        ``speech_presence_evidence`` / ``speaker_votes`` / ``asr_votes`` bucket lists plus
        ``quality_by_bucket`` / ``source_by_bucket``). This is the in-process
        integration point for analyze_audio — no parquet round-trip; the parquet
        ingest path (:meth:`from_run_dir`) remains for artifact-driven runs.

        The store holds *votes*, so the speech-presence axis is linked from its L1 measurements
        under ``policy`` (defaults to the documented anchors) on the way in.

        Every axis is read through :func:`votes.buckets_for_axis` at the field its own declaration
        names, so the four axes L2 folds are the four this store carries. This method used to
        enumerate three of them in a literal tuple: ``background_mask`` entered no store on this
        path, and the caller compensated by handing its evidence in as ``unharvested`` — one vote
        per mask *region*, where L2 had folded one per bucket. A run's fourth axis therefore went
        from 1070 rows at round 0 to 1 by round 4, and an axis with a single bucket has nowhere to
        be uncertain, so it read as *settled*.

        ``unharvested`` carries ``{axis → {perturbation → buckets}}`` for an active axis with no
        vote harvest at all. **No active axis needs it today** — all four declare a
        :class:`~.axes.HarvestSource` — and it is kept because the guard below needs a remedy to
        name: ``task`` is declared ``harvested=False``, so activating it would make this the way its
        evidence arrives. An entry is still required for such an axis, and omitting one raises: an
        axis nobody hands in carries no belief through any round, proposes no region, and is
        reported by the convergence report as ``0 buckets, residual mass 0.0`` — *settled* rather
        than *never asked*. An empty mapping for an axis is a measurement ("nothing was found");
        a missing one is an omission, and only the second is an error.

        Raises:
            ValueError: When an active axis has neither a readable harvest source nor an
                ``unharvested`` entry — the fifth axis's version of the bug the fourth one shipped.
        """
        from senselab.audio.workflows.audio_analysis.votes import buckets_for_axis

        supplied = dict(unharvested or {})
        missing = [a for a in AXES if a not in _HARVEST_ACCESSORS and a not in supplied]
        if missing:
            raise ValueError(
                f"no evidence supplied for active axis/axes {missing}: they have no vote harvest, so the caller "
                "must pass their buckets as `unharvested` — an axis nobody hands in is an axis the loop carries "
                "no belief through, and it reports as settled rather than as unasked"
            )
        store = cls(snr_gate=snr_gate)
        for axis, by_stream in sorted(supplied.items()):
            for stream, buckets in sorted(by_stream.items()):
                store._ingest_buckets(axis, stream, buckets, round_idx)
        for stream, harvest in harvests.items():
            # Declaration order, not set order: votes are appended to the round's append-only file
            # in insertion order, so iterating a frozenset would make the artifact vary per process.
            for axis in HARVEST_SOURCES:
                buckets = buckets_for_axis(harvest, axis, policy=policy)
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
                        meta: dict[str, Any] = {}
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
                        store.row_meta.setdefault((stream, axis, bk), {})
        return store

    def _ingest_buckets(self, axis: str, stream: str, buckets: Sequence[Mapping[str, Any]], round_idx: int) -> None:
        """File one axis's bucket votes for one perturbation. No harvest, same store."""
        for bucket in buckets or ():
            bk = bucket_key(bucket["start"], bucket["end"])
            for source, payload in (bucket.get("votes") or {}).items():
                if isinstance(payload, dict):
                    self.add_vote(
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
            self.row_meta.setdefault((stream, axis, bk), {})

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

    def buckets(self, axis: str) -> list[tuple[float, float]]:
        """Every bucket this axis has evidence for, from any pass, time-ordered.

        Cross-pass because the axis is: a bucket one pass reported and the other did not is still
        one bucket of the recording, with one answer.
        """
        got = {bk for (_s, a, bk) in self._index if a == axis}
        got |= {bk for (_s, a, bk) in self.row_meta if a == axis}
        return sorted(got)

    def vote_buckets(self, stream: str, axis: str) -> list[tuple[float, float]]:
        """Buckets this pass has votes for, time-ordered — the vote-level enumeration.

        Separate from :meth:`buckets` and deliberately so: a caller reading *votes* works per
        pass, and one that folds an axis does not.
        """
        got = {bk for (s, a, bk) in self._index if s == stream and a == axis}
        got |= {bk for (s, a, bk) in self.row_meta if s == stream and a == axis}
        return sorted(got)

    def streams_for(self, axis: str, bucket: tuple[float, float]) -> list[str]:
        """Passes that reported this bucket on this axis, sorted."""
        got = {s for (s, a, bk) in self._index if a == axis and bk == bucket}
        got |= {s for (s, a, bk) in self.row_meta if a == axis and bk == bucket}
        return sorted(got)

    def votes_for(self, stream: str, axis: str, bucket: tuple[float, float]) -> list[Vote]:
        """Every vote on this bucket, active or shadowed — the persisted record, not the fold."""
        return [self._votes[vid] for vid in self._index.get((stream, axis, bucket), [])]

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

    def attenuation_detail(self, stream: str, axis: str, bucket: tuple[float, float]) -> list[dict[str, Any]]:
        """Every withdrawal recorded against this bucket's active votes, flattened for an artifact.

        The store already keeps this in ``provenance["evidence_weight_factors"]``, but provenance
        rides on the vote and the votes are only written for the round they were *added* in. An
        attenuation applied in round 3 to a round-1 vote therefore appeared in no file at all. This
        is the same information, reachable per bucket, so a writer can put it beside the aggregate
        it explains.

        Every factor is listed, not just the composed weight: two rules may each have something to
        say about one vote, and only the list shows that — the product cannot be decomposed after
        the fact.

        Args:
            stream: Pass label.
            axis: Axis name.
            bucket: The bucket to report on.

        Returns:
            One record per (source, withdrawal), source-ordered then in the order the withdrawals
            were applied. Empty when nothing here was attenuated, which is distinct from
            attenuated-to-1.0 and must stay so.
        """
        out: list[dict[str, Any]] = []
        for vid in sorted(
            self._index.get((stream, axis, bucket), []), key=lambda v: (self._votes[v].source, self._votes[v].round)
        ):
            v = self._votes[vid]
            if v.status != "active":
                continue
            for factor in v.provenance.get("evidence_weight_factors") or []:
                out.append(
                    {
                        "source": v.source,
                        "axis": axis,
                        # The composed weight the vote ended up with, alongside this factor's own
                        # contribution: with several factors the two differ, and the difference is
                        # the floor doing its job.
                        "evidence_weight": round(float(v.evidence_weight), 6),
                        "factor": round(float(factor.get("factor", 1.0)), 6),
                        "corroboration": factor.get("corroboration"),
                        "corroboration_pooling": factor.get("corroboration_pooling"),
                        "evidence_sources": factor.get("evidence_sources"),
                        "weight_map": factor.get("weight_map"),
                        "floor": factor.get("floor"),
                        "reason": factor.get("reason"),
                        "measured_on": factor.get("measured_on"),
                        "round": factor.get("round"),
                    }
                )
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

    def bucket_meta(self, axis: str, bucket: tuple[float, float]) -> dict[str, Any]:
        """This bucket's measurements, folded across the passes that reported it.

        Numeric measurements are averaged and labels take the modal value — "mean over passes
        reporting the bucket", the same fold ``compute._attach_scene_measurements`` applies, named
        the same way. The measurements describe the recording at that instant; the enhancement
        transform is a second look at it, not a second instant.

        Scene measurements ride the presence grid, so an axis with none of its own is filled in
        from the presence buckets it *overlaps*. Without that, ``aleatoric_floor`` is ``None`` on
        every speaker and asr bucket and the ``snr_floor`` verdict is unreachable on two axes out of
        three — which is how a noisy stretch got reported as
        ``no_reduction_under_available_interventions``: not "we looked and the scene does not
        explain it", but "we never asked".
        """
        collected = self._meta_for(axis, bucket)
        if not collected and axis != "speech_presence":
            collected = [
                m
                for bk in self.buckets("speech_presence")
                if bk[0] < bucket[1] and bk[1] > bucket[0]
                for m in self._meta_for("speech_presence", bk)
            ]
        per_name: dict[str, list[Any]] = {}
        for entry in collected:
            for name, value in entry.items():
                if value is not None:
                    per_name.setdefault(name, []).append(value)
        merged: dict[str, Any] = {}
        for name, values in per_name.items():
            numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if numbers:
                merged[name] = sum(numbers) / len(numbers)
            else:
                labels = [v for v in values if isinstance(v, str)]
                if labels:
                    merged[name] = max(sorted(set(labels)), key=labels.count)
        return merged

    def _meta_for(self, axis: str, bucket: tuple[float, float]) -> list[dict[str, Any]]:
        """Each pass's measurement dict for one (axis, bucket), empties dropped."""
        return [
            m
            for s in self.streams_for(axis, bucket)
            if (m := self.row_meta.get((s, axis, bucket)))  # noqa: RUF018
        ]

    def folded_evidence_weights(self, axis: str, bucket: tuple[float, float]) -> dict[str, float]:
        """``{source → evidence weight}`` for the cross-pass fold.

        A source's weight is the mean of its per-pass weights over the passes it voted in, with a
        pass that measured nothing about it contributing ``1.0`` — unmeasured must not act as a
        discount. Sources with no measured factor on any pass are omitted entirely, so "nobody
        looked" stays distinct from "looked and found full corroboration".
        """
        collected: dict[str, list[float]] = {}
        measured: set[str] = set()
        for stream in self.streams_for(axis, bucket):
            per_pass = self.evidence_weights(stream, axis, bucket)
            measured |= set(per_pass)
            for source in self.active_votes(stream, axis, bucket):
                collected.setdefault(source, []).append(float(per_pass.get(source, 1.0)))
        return {s: sum(v) / len(v) for s, v in sorted(collected.items()) if s in measured and v}

    def reaggregate_bucket(self, axis: str, bucket: tuple[float, float], *, aggregator: str) -> dict[str, Any]:
        """Fold one bucket's active votes into one axis value — across signals *and* passes.

        The fold is :func:`fuse.fuse_axis`, the same one ``compute_uncertainty_axes`` performs, so
        the belief the loop reasons over and the axis analyze_audio writes are one quantity rather
        than two estimators of it. Each signal contributes one reading — the mean of its readings
        across the passes it spoke in — so a signal's influence does not scale with how many passes
        happened to include it.

        Four quantities travel separately because they answer different questions:
        ``uncertainty`` (normalised entropy — what do we believe), ``epistemic_uncertainty`` (its
        reducible part), ``confidence`` (a probability) and ``triage_score`` (the policy fold under
        ``aggregator``, which exists to rank where budget goes). ``p_voice`` rides alongside for the
        presence axis: it is a probability about the *world*, not a fold of doubt, and the
        adjudication rules need it.

        Attenuated sources stay in ``contributing_sources`` — the record has to show who spoke up,
        and how far their claim was carried — with the withdrawn weights alongside in
        ``attenuated_sources`` and the measurements behind them in ``attenuation``.
        ``attenuated_sources`` answers "who, and how much"; ``attenuation`` answers "measured
        against what". Both are needed at the artifact boundary: a weight with no measurement
        beside it is an assertion a reader cannot check or disagree with.
        """
        streams = self.streams_for(axis, bucket)
        votes_by_pass = {s: self.active_votes(s, axis, bucket) for s in streams}
        weights = self.folded_evidence_weights(axis, bucket)
        rows = fuse_axis(
            {s: [{"start": bucket[0], "end": bucket[1], "votes": v}] for s, v in votes_by_pass.items()},
            weights=weights,
            aggregator=aggregator,
            weight_basis={s: {"evidence_weight": w} for s, w in weights.items()},
            snr_gate=self.snr_gate,
        )
        fused: dict[str, Any] = dict(rows[0]) if rows else {}
        sources = sorted({src for v in votes_by_pass.values() for src in v})
        attenuation: list[dict[str, Any]] = []
        for stream in streams:
            attenuation += [{**rec, "stream": stream} for rec in self.attenuation_detail(stream, axis, bucket)]
        return {
            "start": bucket[0],
            "end": bucket[1],
            "uncertainty": fused.get("uncertainty"),
            "epistemic_uncertainty": fused.get("epistemic_uncertainty"),
            "confidence": fused.get("confidence"),
            "variability": fused.get("variability"),
            "triage_score": fused.get("triage_score"),
            "p_voice": self._folded_p_voice(axis, bucket) if axis == "speech_presence" else None,
            "contributing_sources": sources,
            "contributing_signals": fused.get("contributing_signals") or [],
            "contributing_passes": fused.get("contributing_passes") or streams,
            # Not defaulted to ``streams``: an empty list here means "nothing was withheld", and
            # falling back to the stream set would claim every pass was gated out.
            "snr_gated_passes": fused.get("snr_gated_passes") or [],
            "signal_weights": fused.get("signal_weights") or {},
            # Carried so the loop's estimate writer can record *why* each weight was what it was,
            # the same account ``fuse`` writes for rounds 0-2.
            "weight_basis": fused.get("weight_basis") or {},
            "attenuated_sources": {k: round(v, 6) for k, v in sorted(weights.items())},
            "attenuation": attenuation,
        }

    def _folded_p_voice(self, axis: str, bucket: tuple[float, float]) -> float | None:
        """P(voice) for one bucket, folded across passes the way the axis is.

        Each voter's per-pass probability is averaged into one reading before the weighted mean
        across voters, so — exactly as in :func:`fuse.fuse_axis` — a voter present in both passes
        does not count twice. Computing a p_voice per pass and averaging *those* would instead
        weight each pass by how many voters it happened to have.
        """
        readings: dict[str, list[tuple[float, float]]] = {}
        for stream in self.streams_for(axis, bucket):
            per_pass = per_source_voice(
                self.active_votes(stream, axis, bucket), weights=self.evidence_weights(stream, axis, bucket)
            )
            for source, (p, w) in per_pass.items():
                readings.setdefault(source, []).append((p, w))
        num = den = 0.0
        for values in readings.values():
            p = sum(v[0] for v in values) / len(values)
            w = sum(v[1] for v in values) / len(values)
            num += p * w
            den += w
        return num / den if den > 0 else None

    def replay_check(self, *, aggregator: str, tol: float = 1e-9) -> dict[str, Any]:
        """Prove every value is re-derivable from the active evidence and the recorded decisions.

        Replays each bucket from a *fresh* store carrying only what is persisted — the votes, the
        record of which were shadowed, and the recorded evidence weights — and compares against
        this store's aggregation. Equality is the store's own contract ("aggregation is a pure
        function of the active votes"); a mismatch means a value depends on something not written
        down, which is exactly what makes an estimate unreproducible.

        Re-derivability and agreement-with-L2 are separate properties and are checked separately:
        this is the first, :meth:`fused_parity` is the second.
        """
        report: dict[str, Any] = {}
        for axis in AXES:
            n = mismatches = compared = 0
            max_abs = 0.0
            for bk in self.buckets(axis):
                n += 1
                first = self.reaggregate_bucket(axis, bk, aggregator=aggregator)
                replay = self._replay_bucket(axis, bk, aggregator=aggregator)
                a, b = first["uncertainty"], replay["uncertainty"]
                if a is None or b is None:
                    if (a is None) != (b is None):
                        mismatches += 1
                    continue
                compared += 1
                diff = abs(float(a) - float(b))
                max_abs = max(max_abs, diff)
                if diff > tol:
                    mismatches += 1
            report[axis] = {"buckets": n, "compared": compared, "mismatches": mismatches, "max_abs_diff": max_abs}
        return report

    def fused_parity(
        self,
        fused_rows_by_axis: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        aggregator: str,
        tol: float = 1e-9,
    ) -> dict[str, Any]:
        """Compare this store's fold against the axis L2 already wrote.

        The check the store should always have been held to. The old one compared re-aggregation
        against ``within_pass_uncertainty`` on an L1 parquet — an oracle of the wrong kind twice
        over: the quantity does not exist (a per-pass axis), and it came from a second
        implementation, so a mismatch could not distinguish "the store missed an input" from "the
        two folds disagree". Comparing against ``L2/round0/uncertainty/<axis>.parquet`` has neither
        problem: same arithmetic, so any difference is a difference in *evidence*, which is the
        thing worth catching — a signal the ingest dropped shows up here and nowhere else.

        ``uncertainty`` is the compared quantity because it is a pure function of the per-signal
        readings; ``triage_score`` and ``confidence`` are weighted, and L2's weights (measured
        stability and support) are not the store's (measured corroboration), so a difference there
        is expected rather than diagnostic.

        Buckets whose stored row was moved by *another axis* are counted as ``skipped_coupled``
        rather than compared: cross-axis coupling is an input the store does not have, and scoring
        it as a mismatch would report a difference the store could not have avoided.
        ``scene_quality`` is not such a case — it multiplies ``triage_score`` and leaves
        ``uncertainty`` alone — so treating every ``coupled_from`` alike skipped the whole asr axis
        and reported a vacuous zero.
        """
        report: dict[str, Any] = {}
        for axis in AXES:
            stored = {bucket_key(r["start"], r["end"]): r for r in (fused_rows_by_axis.get(axis) or []) if "start" in r}
            compared = mismatches = skipped = missing = 0
            max_abs = 0.0
            for bk in self.buckets(axis):
                row = stored.get(bk)
                if row is None:
                    missing += 1
                    continue
                # Tested for ``None`` explicitly: the column arrives from parquet as a numpy array,
                # and ``or ()`` on one raises rather than falling through to the empty case.
                raw_coupled: Any = row.get("coupled_from")
                coupled = {str(c) for c in (raw_coupled if raw_coupled is not None else ())}
                if coupled - _COUPLINGS_LEAVING_UNCERTAINTY_ALONE:
                    skipped += 1
                    continue
                # ``_float_or_none`` on both sides: parquet reads a missing value back as NaN and
                # the store holds ``None``, and comparing the two raw made 11 asr buckets where
                # *neither* had a value report as mismatches — a check that finds a difference
                # between two spellings of "nothing" is worse than no check.
                mine = _float_or_none(self.reaggregate_bucket(axis, bk, aggregator=aggregator)["uncertainty"])
                theirs = _float_or_none(row.get("uncertainty"))
                if mine is None or theirs is None:
                    if (mine is None) != (theirs is None):
                        mismatches += 1
                    continue
                compared += 1
                diff = abs(float(mine) - float(theirs))
                max_abs = max(max_abs, diff)
                if diff > tol:
                    mismatches += 1
            report[axis] = {
                "compared": compared,
                "mismatches": mismatches,
                "skipped_coupled": skipped,
                "not_in_l2": missing,
                "max_abs_diff": max_abs,
            }
        return report

    def _replay_bucket(self, axis: str, bucket: tuple[float, float], *, aggregator: str) -> dict[str, Any]:
        """Re-aggregate one bucket from a store rebuilt out of the persisted vote records."""
        replay = VoteStore()
        for stream in self.streams_for(axis, bucket):
            for vote in self.votes_for(stream, axis, bucket):
                record = vote.to_record()
                replay.add_vote(
                    Vote(
                        axis=record["axis"],
                        bucket=(record["bucket_start"], record["bucket_end"]),
                        source=record["source"],
                        stream=record["stream"],
                        scope=record["scope"],
                        round=record["round"],
                        payload=json.loads(record["payload"]),
                        status=record["status"],
                        shadowed_by=record["shadowed_by"],
                        evidence_weight=record["evidence_weight"],
                        provenance=json.loads(record["provenance"]),
                    )
                )
        return replay.reaggregate_bucket(axis, bucket, aggregator=aggregator)


class BeliefState:
    """One belief row per (axis, bucket), updated each round.

    Not per (stream, axis, bucket). The two passes are the same recording under a transform, so
    "converged on raw, open on enhanced" is not a state a recording can be in — and while the state
    held both, every consumer invented its own collapse: the writer elected the most doubtful pass,
    the plot filtered to the fusion stream, the evaluator to the transcript's. Three answers from
    one state.
    """

    def __init__(self, aggregator: str) -> None:
        """Create an empty belief state using ``aggregator`` for the policy fold."""
        self.aggregator = aggregator
        self.rows: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    def from_store(cls, store: VoteStore, *, aggregator: str, round_index: int) -> "BeliefState":
        """Baseline belief: fold every bucket, attach its measurements and its aleatoric floor.

        Args:
            store: The ingested vote store.
            aggregator: Aggregator for the policy fold.
            round_index: The round this baseline *is* — the one the loop adopts from fusion. Taken
                from the caller because only the caller knows it: this used to be a hardcoded ``1``
                while the loop adopted fusion's last round, so on a three-round fold every
                untouched bucket carried a round stamp naming a round the loop had never run.
        """
        state = cls(aggregator)
        for axis in AXES:
            rows = []
            for bk in store.buckets(axis):
                row = store.reaggregate_bucket(axis, bk, aggregator=aggregator)
                meta = store.bucket_meta(axis, bk)
                row["meta"] = meta
                _attach_floor(row, meta)
                row["status"] = "open"
                row["last_refolded_round"] = int(round_index)
                # ``doubt`` beside the entropy value: convergence measures round-over-round
                # improvement on the quantity its gate compares (``estimates.control_doubt``), and a
                # history in other units would judge "stalled" on a different scale from "converged".
                row["history"] = [
                    {"round": int(round_index), "uncertainty": row["uncertainty"], "doubt": control_doubt(row)}
                ]
                rows.append(row)
            state.rows[axis] = rows
        return state

    def update_buckets(
        self, store: VoteStore, axis: str, buckets: set[tuple[float, float]], round_idx: int
    ) -> list[dict[str, Any]]:
        """Incrementally re-fold only ``buckets`` (FR-006); returns changed rows."""
        changed = []
        for row in self.rows.get(axis, []):
            bk = bucket_key(row["start"], row["end"])
            if bk not in buckets:
                continue
            new = store.reaggregate_bucket(axis, bk, aggregator=self.aggregator)
            for key in (
                "uncertainty",
                "epistemic_uncertainty",
                "confidence",
                "variability",
                "triage_score",
                "contributing_sources",
                "contributing_signals",
                "contributing_passes",
                "signal_weights",
                "attenuated_sources",
                "attenuation",
            ):
                row[key] = new[key]
            if new["p_voice"] is not None:
                row["p_voice"] = new["p_voice"]
            _attach_floor(row, row.get("meta") or {})
            row["last_refolded_round"] = round_idx
            row["history"].append({"round": round_idx, "uncertainty": row["uncertainty"], "doubt": control_doubt(row)})
            changed.append(row)
        return changed

    def axis_rows(self, axis: str) -> list[dict[str, Any]]:
        """Rows for one axis, time-ordered."""
        return self.rows.get(axis, [])

    def uncertainty_mass(self, axis: str, theta_low: float) -> float:
        """Σ max(0, doubt − θ_low) · width — the quantity interventions try to shrink.

        Doubt (``estimates.control_doubt``), not the entropy column: ``theta_low`` is doubt-scaled,
        so measuring the mass above it in entropy units reported residual work where the evidence
        was already settled — 191 of 214 speaker buckets on a clean conversation.
        """
        total = 0.0
        for row in self.axis_rows(axis):
            u = control_doubt(row)
            if u is None:
                continue
            total += max(0.0, float(u) - theta_low) * (float(row["end"]) - float(row["start"]))
        return total


ALEATORIC_FLOOR_TERMS: tuple[str, ...] = ("quality_snr", "quality_reverb", "quality_clip", "overlap_posterior")
"""What may impose an irreducible floor on a bucket, folded by max.

Bandwidth is deliberately absent: a band-limited recording is missing frequencies, which is a
reason a *model* may be wrong, not evidence that the answer here cannot be recovered by looking
harder. The other three are conditions of the acoustic scene at that instant.
"""


def _attach_floor(row: dict[str, Any], meta: Mapping[str, Any]) -> None:
    """Attach the aleatoric floor, derived from measurements under named anchors.

    The floor is the largest degradation the scene imposes here. It was previously read from
    ``meta["quality_snr"]`` and siblings — *scores*, which neither ingest path has ever carried:
    the harvest holds dB, and the fused presence parquet holds neither. Every lookup missed, the
    floor was assigned ``0.0`` on every bucket of every run, and ``0.0`` is the confident claim
    "this audio imposes no floor" — so the ``snr_floor`` irreducibility verdict could not fire and
    a run could only ever report ``no_reduction_under_available_interventions``.

    So the scores are derived here, from the dB the store does carry, against
    :data:`degradation.DEFAULT_ANCHORS`. An anchored score is an L2 decision, so the anchors and
    the terms that survived travel on the row.

    **Absent is not zero.** With no measurement the floor is ``None``, and a ``None`` floor cannot
    explain a residual — which is the difference between "nothing constrains this bucket" and
    "nobody measured whether anything does".
    """
    snr_source = next((name for name in SNR_PREFERENCE if _float_or_none(meta.get(name)) is not None), None)
    derived: dict[str, float | None] = {
        "quality_snr": snr_degradation(_float_or_none(meta.get(snr_source)) if snr_source else None),
        "quality_reverb": reverb_degradation(_float_or_none(meta.get("c50_brouhaha_db"))),
        "quality_clip": clip_degradation(_float_or_none(meta.get("proportion_clipped"))),
        # Populated by I4 when segmentation-3.0 per-class posteriors are available; already a
        # posterior in [0, 1], so it needs no anchor.
        "overlap_posterior": _float_or_none(meta.get("overlap_posterior")),
    }
    terms = {
        name: max(0.0, min(1.0, float(value)))
        for name in ALEATORIC_FLOOR_TERMS
        if (value := derived.get(name)) is not None
    }
    row["aleatoric_floor"] = max(terms.values()) if terms else None
    row["aleatoric_floor_policy"] = {
        "terms": sorted(terms),
        "fold": "max",
        "anchors": dict(DEFAULT_ANCHORS),
        "snr_source": snr_source,
    }


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
