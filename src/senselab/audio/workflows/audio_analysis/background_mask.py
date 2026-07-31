"""Background mask: where no target activity happens (T031-T034, FR-031 to FR-045).

The mask marks regions free of **target activity** — activity from the near-microphone
participant being recorded — not regions free of speech. Two consequences follow, and both
matter more than the naming:

**Background claims are trustworthy in a target-free region without any suppression.**
There is no foreground there to leak, so the suppression-depth constraint that bounds
everything else simply does not apply. Since a 30 dB suppression baseline was measured to
leave residual foreground dominant, these regions may carry most of the trustworthy
background evidence in a recording.

**What counts as target activity depends on the task.** In a breathing or cough task the
target *is* a non-speech vocal event, and speech detection reports no activity while it is
happening. A mask built from speech activity alone would admit the target breaths — and
because AudioSet maps ``Breathing`` and ``Cough`` to the ``people`` category, they would be
reported as a background human-sound source. That is the collected signal misattributed as
an environmental finding, which is why :func:`requires_label_detection` exists and why
FR-033a forbids relying on voice activity alone.

Scope: lab-like collection with the microphone close to the source. A distant talker stays
*in* the mask and is reportable as a background source (FR-033c) — target-free is not
speech-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

MaskState = Literal["target_free", "target_active", "nontarget_active", "indeterminate"]

MASK_STATES: tuple[MaskState, ...] = ("target_free", "target_active", "nontarget_active", "indeterminate")
"""``nontarget_active`` is the actionable state: the task target is absent *and* there is
measurable other content. A pause with room tone and a pause with digital silence are both
target-free, but only the first is worth introspecting — conflating them is what made the mask
report a 21 s conversation as one uninformative region."""

TARGET_EVENT_LABELS: Mapping[str, tuple[str, ...]] = {
    "breath": ("Breathing", "Wheeze", "Snoring", "Gasp"),
    "cough": ("Cough", "Throat clearing", "Sneeze", "Sniff"),
    "mouth_noise": ("Chewing, mastication", "Lip smacking", "Whistling"),
    "throat_clear": ("Throat clearing",),
}
"""AudioSet labels that evidence each non-speech target event type.

Only classes the scene classifier can actually emit — a target type with no detectable
label would silently degrade to "never active", which is the failure mode this table
exists to prevent. ``speech`` is absent deliberately: speech targets are served by voice
activity and diarization, not by a label lookup.
"""

_SPEECH_TARGET = "speech"


@dataclass(frozen=True)
class BackgroundMaskRegion:
    """One contiguous run of buckets sharing a mask state."""

    region_id: str
    start: float
    end: float
    state: MaskState
    uncertainty: float
    guard_trimmed_s: float = 0.0
    contains_nontarget_speech: bool = False
    supports_long_window: bool = False

    @property
    def duration_s(self) -> float:
        """Region length in seconds."""
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class BackgroundMask:
    """The mask for one pass, with its provenance and coverage totals."""

    regions: list[BackgroundMaskRegion]
    task_type: str | None
    target_event_types: list[str]
    metadata_provenance: Literal["recognized", "fallback"]
    guard_interval_s: float
    duration_s: float
    negligible_threshold: float

    @property
    def total_masked_s(self) -> float:
        """Total duration of target-free regions (FR-038)."""
        return sum(r.duration_s for r in self.regions if r.state == "target_free")

    @property
    def masked_fraction(self) -> float:
        """Target-free duration as a fraction of the recording."""
        return (self.total_masked_s / self.duration_s) if self.duration_s > 0 else 0.0

    @property
    def is_empty(self) -> bool:
        """True when no target-free region exists (FR-040)."""
        return self.total_masked_s <= 0.0

    @property
    def negligible_fraction(self) -> bool:
        """True when the mask is too small to support conclusions."""
        return self.masked_fraction < self.negligible_threshold

    @property
    def regions_total(self) -> int:
        """Number of target-free regions."""
        return sum(1 for r in self.regions if r.state == "target_free")

    @property
    def nontarget_interest_s(self) -> float:
        """Total duration where the target is absent and other content is present.

        Reported separately from ``total_masked_s`` because a consumer asks two different
        questions: what is clean enough to rest a background claim on, and what is worth
        looking at. A target-free silent stretch answers the first and not the second.
        """
        return sum(r.duration_s for r in self.regions if r.state == "nontarget_active")

    @property
    def regions_of_interest(self) -> int:
        """Number of non-target-active regions."""
        return sum(1 for r in self.regions if r.state == "nontarget_active")

    @property
    def regions_supporting_long_window(self) -> int:
        """How many target-free regions can host an unpadded long-window decision."""
        return sum(1 for r in self.regions if r.state == "target_free" and r.supports_long_window)

    def to_json(self) -> dict[str, Any]:
        """Serialize per ``contracts/background-mask.md``."""
        return {
            "task_type": self.task_type,
            "target_event_types": list(self.target_event_types),
            "metadata_provenance": self.metadata_provenance,
            "guard_interval_s": self.guard_interval_s,
            "total_masked_s": round(self.total_masked_s, 6),
            "masked_fraction": round(self.masked_fraction, 6),
            "is_empty": self.is_empty,
            "negligible_fraction": self.negligible_fraction,
            "regions_supporting_long_window": self.regions_supporting_long_window,
            "regions_total": self.regions_total,
            "nontarget_interest_s": round(self.nontarget_interest_s, 6),
            "regions_of_interest": self.regions_of_interest,
            "requires_label_detection": requires_label_detection(self.target_event_types),
        }

    def to_rows(self) -> list[dict[str, Any]]:
        """Per-region rows for ``background_mask.parquet``."""
        types = ",".join(self.target_event_types)
        return [
            {
                "region_id": r.region_id,
                "start": r.start,
                "end": r.end,
                "state": r.state,
                "uncertainty": r.uncertainty,
                "guard_trimmed_s": r.guard_trimmed_s,
                "contains_nontarget_speech": r.contains_nontarget_speech,
                "supports_long_window": r.supports_long_window,
                "target_event_types": types,
            }
            for r in self.regions
        ]


def target_event_types_for(
    task_type: str | None,
    profile: Mapping[str, Any],
) -> tuple[list[str], Literal["recognized", "fallback"]]:
    """Resolve a task type to its target event types (FR-033, FR-033b).

    Args:
        task_type: Task name from metadata, or ``None`` when unavailable.
        profile: Detection-margin profile supplying ``mask.target_event_types_by_task``.

    Returns:
        ``(event_types, provenance)``. Provenance is ``"fallback"`` whenever the task type
        is missing or unrecognized — recorded rather than silently assumed, because a mask
        built without task context is a different object from one built with it, and the
        fallback deliberately over-excludes.
    """
    mask_cfg = profile.get("mask", {}) or {}
    by_task = mask_cfg.get("target_event_types_by_task", {}) or {}
    if task_type is not None and task_type in by_task:
        return list(by_task[task_type]), "recognized"
    return list(mask_cfg.get("fallback_target_event_types", [_SPEECH_TARGET])), "fallback"


def target_labels_for(event_types: Iterable[str]) -> tuple[str, ...]:
    """AudioSet labels evidencing the given non-speech target event types."""
    labels: list[str] = []
    for event in event_types:
        labels.extend(TARGET_EVENT_LABELS.get(event, ()))
    return tuple(dict.fromkeys(labels))


def requires_label_detection(event_types: Iterable[str]) -> bool:
    """True when the target includes a non-speech vocal event (FR-033a).

    Voice-activity detection reports nothing during a breath or a cough, so a mask built
    from it alone would admit the target signal and then report it as a background
    ``people`` source. When this returns ``True`` the caller **must** consult classifier
    labels for :func:`target_labels_for` rather than voice activity alone.
    """
    return any(event != _SPEECH_TARGET for event in event_types)


def _classify_bucket(
    confidence: float,
    uncertainty: float,
    *,
    active_at: float,
    free_at: float,
    max_free_uncertainty: float,
    nontarget_confidence: float | None = None,
    nontarget_at: float = 0.5,
) -> MaskState:
    """Assign one bucket's mask state.

    Both committed verdicts demand low uncertainty. ``target_free`` always did — "probably
    nothing there, but I cannot tell" is not a region background claims can rest on — while
    ``target_active`` demanded none, so a bucket at confidence 0.99 *and* uncertainty 0.99
    committed to "target active" on evidence that supported no verdict at all. Measured on a
    21.5 s conversation, that asymmetry produced a single whole-file ``target_active`` region
    at uncertainty 0.9997.

    ``nontarget_active`` requires the target to be absent as well as other content to be
    present: background content underneath active target speech is not a clean region to
    introspect, which is the leakage problem suppression-depth measurement exists for.
    """
    if uncertainty > max_free_uncertainty:
        return "indeterminate"
    if confidence >= active_at:
        return "target_active"
    if confidence <= free_at:
        if nontarget_confidence is not None and float(nontarget_confidence) >= nontarget_at:
            return "nontarget_active"
        return "target_free"
    return "indeterminate"


def build_mask(
    buckets: Sequence[Mapping[str, Any]],
    task_type: str | None,
    *,
    profile: Mapping[str, Any],
    long_window_s: float = 10.24,
) -> BackgroundMask:
    """Build the background mask from per-bucket target-activity evidence.

    Args:
        buckets: Rows with ``start``, ``end``, ``target_confidence``, ``uncertainty``, and
            optionally ``nontarget_speech``. The caller is responsible for computing
            ``target_confidence`` from the *right* detector for the task — see
            :func:`requires_label_detection`.
        task_type: Task name from metadata, or ``None``.
        profile: Detection-margin profile.
        long_window_s: Analysis window of the long-window classifier, used to decide
            whether a region can host an unpadded decision (FR-045).

    Returns:
        The assembled :class:`BackgroundMask`.
    """
    mask_cfg = profile.get("mask", {}) or {}
    guard_s = float(mask_cfg.get("guard_interval_s", 0.5))
    min_region_s = float(mask_cfg.get("min_region_s", 1.0))
    max_padding = float(mask_cfg.get("max_padding_fraction", 0.5))
    active_at = float(mask_cfg.get("target_active_confidence", 0.6))
    free_at = float(mask_cfg.get("target_free_confidence", 0.2))
    max_free_unc = float(mask_cfg.get("max_free_uncertainty", 0.5))
    negligible = float(mask_cfg.get("negligible_fraction", 0.05))
    nontarget_at = float(mask_cfg.get("nontarget_active_confidence", 0.5))

    event_types, provenance = target_event_types_for(task_type, profile)

    ordered = sorted(buckets, key=lambda b: float(b["start"]))
    duration = float(ordered[-1]["end"]) if ordered else 0.0

    states: list[MaskState] = [
        _classify_bucket(
            float(b.get("target_confidence") or 0.0),
            float(b.get("uncertainty") or 0.0),
            active_at=active_at,
            free_at=free_at,
            max_free_uncertainty=max_free_unc,
            nontarget_confidence=(
                float(b["nontarget_confidence"]) if b.get("nontarget_confidence") is not None else None
            ),
            nontarget_at=nontarget_at,
        )
        for b in ordered
    ]

    # Guard interval: the stretch *following* target activity is contaminated by the
    # reverberant tail, so it is not clean background even where no activity is detected
    # in it (FR-034). Forward-only by design — window-overlap contamination on the other
    # side is handled by the excision padding rule (FR-043), not by widening this guard,
    # which would eat short regions before they could be evaluated.
    guard_trim = [0.0] * len(ordered)
    last_active_end: float | None = None
    for i, bucket in enumerate(ordered):
        if states[i] == "target_active":
            last_active_end = float(bucket["end"])
            continue
        if last_active_end is None:
            continue
        if float(bucket["start"]) < last_active_end + guard_s and states[i] == "target_free":
            states[i] = "indeterminate"
            guard_trim[i] = min(float(bucket["end"]), last_active_end + guard_s) - float(bucket["start"])

    regions: list[BackgroundMaskRegion] = []
    idx = 0
    while idx < len(ordered):
        run_end = idx
        while run_end + 1 < len(ordered) and states[run_end + 1] == states[idx]:
            run_end += 1
        span = ordered[idx : run_end + 1]
        start, end = float(span[0]["start"]), float(span[-1]["end"])
        uncertainties = [float(b.get("uncertainty") or 0.0) for b in span]
        dur = max(0.0, end - start)
        padding_fraction = max(0.0, (long_window_s - dur) / long_window_s) if long_window_s > 0 else 0.0
        regions.append(
            BackgroundMaskRegion(
                region_id=f"m{len(regions)}",
                start=start,
                end=end,
                state=states[idx],
                uncertainty=max(uncertainties) if uncertainties else 0.0,
                guard_trimmed_s=round(sum(guard_trim[idx : run_end + 1]), 6),
                contains_nontarget_speech=any(bool(b.get("nontarget_speech")) for b in span),
                supports_long_window=(dur >= min_region_s and padding_fraction <= max_padding),
            )
        )
        idx = run_end + 1

    return BackgroundMask(
        regions=regions,
        task_type=task_type,
        target_event_types=event_types,
        metadata_provenance=provenance,
        guard_interval_s=guard_s,
        duration_s=duration,
        negligible_threshold=negligible,
    )


@dataclass(frozen=True)
class MaskedRegionIntrospection:
    """What one target-free region actually contains (FR-037)."""

    region_id: str
    start: float
    end: float
    is_noise_floor_only: bool
    floor_db_by_band: dict[str, float] = field(default_factory=dict)
    summary_a_weighted_db: float | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialize per ``contracts/background-mask.md``.

        ``summary_a_weighted_db`` is a human-readable summary only. The gate is per-band
        excess over a locally estimated floor; a broadband number would be set by the low
        bands and leave mid/high-band events ungated.
        """
        return {
            "region_id": self.region_id,
            "start": self.start,
            "end": self.end,
            "is_noise_floor_only": self.is_noise_floor_only,
            "floor_db_by_band": dict(self.floor_db_by_band),
            "summary_a_weighted_db": self.summary_a_weighted_db,
            "findings": list(self.findings),
        }


# ── target-activity evidence (T033, FR-033a) ───────────────────────────


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """True when two half-open intervals intersect."""
    return a_start < b_end and b_start < a_end


def _seg_bounds(seg: Any) -> tuple[float, float] | None:  # noqa: ANN401 — ScriptLine or dict
    """Return ``(start, end)`` for one diarization segment, or ``None`` if unreadable.

    Segments arrive as either Pydantic-like objects (in memory) or plain dicts (after a
    cache round-trip), so both shapes are handled. The attribute lookup deliberately does
    *not* use ``getattr(seg, "start", seg.get("start"))``: Python evaluates that default
    eagerly, so the dict access runs even for objects that already have the attribute — and
    raises on anything that is neither.
    """
    start = getattr(seg, "start", None)
    end = getattr(seg, "end", None)
    if start is None and isinstance(seg, Mapping):
        start, end = seg.get("start"), seg.get("end")
    if start is None or end is None:
        return None
    try:
        return float(start), float(end)
    except (TypeError, ValueError):
        return None


def _flatten_segments(result: Any) -> list[Any]:  # noqa: ANN401 — senselab outputs
    """Flatten a diarization result to a flat list of segments.

    ``diarize_audios`` returns one entry per input audio, so a single-audio call yields
    ``[[seg, seg, ...]]``. Iterating that without flattening hands *lists* where segments
    are expected. This only surfaced on a real run — every fixture had used the flat shape.
    """
    if not isinstance(result, list):
        return []
    flat: list[Any] = []
    for item in result:
        flat.extend(item) if isinstance(item, list) else flat.append(item)
    return flat


def _speech_activity_by_bucket(
    pass_summary: Mapping[str, Any],
    buckets: Sequence[tuple[float, float]],
) -> list[float | None]:
    """Per-bucket speech-activity confidence from diarization segments.

    ``None`` where no diarizer contributed, so "nobody looked" stays distinguishable from
    "everybody said no" — collapsing the two is what would let an unexamined bucket be
    reported as clean background.
    """
    diar = (pass_summary.get("diarization") or {}).get("by_model") or {}
    tracks: list[list[tuple[float, float]]] = []
    for outcome in diar.values():
        if not isinstance(outcome, Mapping) or outcome.get("status") != "ok":
            continue
        tracks.append([b for b in (_seg_bounds(seg) for seg in _flatten_segments(outcome.get("result"))) if b])
    if not tracks:
        return [None] * len(buckets)
    out: list[float | None] = []
    for b_start, b_end in buckets:
        votes = [1.0 if any(_overlaps(s, e, b_start, b_end) for s, e in track) else 0.0 for track in tracks]
        out.append(sum(votes) / len(votes) if votes else None)
    return out


def _label_score_by_bucket(
    pass_summary: Mapping[str, Any],
    buckets: Sequence[tuple[float, float]],
    labels: Sequence[str],
    *,
    absent_bound_below: float = 0.6,
) -> list[float | None]:
    """Per-bucket maximum score across ``labels``, from the scene classifiers.

    This is the mechanism FR-033a requires: for a breath or cough target, voice activity
    reports nothing while the target event is happening, so the evidence has to come from
    classifier labels instead.
    """
    from senselab.audio.workflows.audio_analysis.harvesters import classification_windows

    wanted = set(labels)
    if not wanted:
        return [None] * len(buckets)

    windows: list[Any] = []
    for key in ("ast", "yamnet"):
        block = pass_summary.get(key) or {}
        if block.get("status") == "ok":
            windows.extend(classification_windows(block.get("result")))
    if not windows:
        return [None] * len(buckets)

    out: list[float | None] = []
    for b_start, b_end in buckets:
        best: float | None = None
        for win in windows:
            if not isinstance(win, dict):
                continue
            if not _overlaps(float(win.get("start", 0.0)), float(win.get("end", 0.0)), b_start, b_end):
                continue
            scores = [float(x) for x in (win.get("scores") or [])]
            hit = next(
                (float(sc) for lb, sc in zip(win.get("labels") or [], scores) if str(lb) in wanted),
                None,
            )
            # A target label absent from a top-k window is bounded above by the smallest
            # score the window *did* report — it must rank below all of them. That bound is
            # usable evidence only when it lands below the active threshold: then the target
            # is definitely not active. When the bound sits above the threshold (a short
            # window reporting one confident label, say) it is true but uninformative, and
            # reporting it as the confidence would read a quiet bucket as target-active. In
            # that case there is genuinely nothing to say, so contribute nothing and let the
            # bucket fall to `indeterminate`.
            if hit is not None:
                value: float | None = hit
            elif scores and min(scores) < absent_bound_below:
                value = min(scores)
            else:
                value = None
            if value is not None:
                best = value if best is None else max(best, value)
        out.append(best)
    return out


def target_confidence_by_bucket(
    pass_summary: Mapping[str, Any],
    buckets: Sequence[tuple[float, float]],
    event_types: Sequence[str],
    *,
    active_threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Assemble per-bucket target-activity evidence for :func:`build_mask`.

    Combines two evidence sources, taking the **maximum** because either is sufficient to
    establish that the participant was active:

    - speech activity from diarization, when ``speech`` is a target type;
    - classifier label scores for every non-speech target type (FR-033a).

    Uncertainty is the *spread* between contributing sources — zero when they agree, wide
    when they disagree — and ``1.0`` when nothing contributed at all. That last case is the
    important one: a bucket no source examined must not read as confidently target-free.

    Args:
        pass_summary: The per-pass summary from ``run_pass``.
        buckets: ``(start, end)`` pairs on the analysis grid.
        event_types: Target event types from :func:`target_event_types_for`.
        active_threshold: The mask's target-active threshold. An absent target label's
            upper bound counts as evidence only when it falls below this; above it the
            bound is true but uninformative.

    Returns:
        Bucket rows ready for :func:`build_mask`.
    """
    sources: list[list[float | None]] = []
    if _SPEECH_TARGET in event_types:
        sources.append(_speech_activity_by_bucket(pass_summary, buckets))
    labels = target_labels_for(event_types)
    if labels:
        sources.append(_label_score_by_bucket(pass_summary, buckets, labels, absent_bound_below=active_threshold))

    rows: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(buckets):
        values: list[float] = [v for s in sources if (v := s[i]) is not None]
        if not values:
            rows.append({"start": start, "end": end, "target_confidence": 0.0, "uncertainty": 1.0})
            continue
        high, low = max(values), min(values)
        rows.append(
            {
                "start": start,
                "end": end,
                "target_confidence": high,
                # Spread between the sources, not a variance: with two sources their gap *is*
                # the disagreement, and a single source disagrees with nothing.
                "uncertainty": (high - low) if len(values) > 1 else 0.0,
            }
        )
    return rows


NONTARGET_EXCLUDED_CATEGORIES = ("speech",)
"""Scene categories that do not count as non-target content.

``speech`` is excluded because a speech-category detection during a target-free stretch is
most likely leaked or distant target speech, and the mask's job here is to point at content
worth introspecting rather than at the thing already being excluded.
"""


def nontarget_confidence_by_bucket(
    rows: Sequence[Mapping[str, Any]],
    source_by_bucket: Mapping[tuple[float, float], Mapping[str, Any]],
    *,
    excluded: Sequence[str] = NONTARGET_EXCLUDED_CATEGORIES,
) -> list[dict[str, Any]]:
    """Attach ``nontarget_confidence`` to target-activity rows from scene source mass.

    The mask needs a second quantity to distinguish "target absent, nothing there" from
    "target absent, something else is" — the second being the region worth introspecting. The
    scene classifiers already report per-category mass per bucket, so the evidence exists; it
    was simply never reaching the mask.

    Args:
        rows: Rows from :func:`target_confidence_by_bucket`.
        source_by_bucket: ``{(start, end) → {"src_machine": ..., "src_environment": ...}}``
            as harvested on the presence grid.
        excluded: Categories that do not count as non-target content.

    Returns:
        The same rows with ``nontarget_confidence`` added where scene mass was available. A
        bucket with no scene evidence is left without the field, so the mask keeps its prior
        verdict rather than being told there is nothing there.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        key = (round(float(row.get("start", 0.0)), 6), round(float(row.get("end", 0.0)), 6))
        mass = source_by_bucket.get(key)
        if isinstance(mass, Mapping):
            values = [
                float(v)
                for k, v in mass.items()
                if str(k).startswith("src_") and str(k)[4:] not in excluded and isinstance(v, (int, float))
            ]
            if values:
                enriched["nontarget_confidence"] = max(0.0, min(1.0, max(values)))
        out.append(enriched)
    return out
