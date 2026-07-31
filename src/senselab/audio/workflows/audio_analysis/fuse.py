"""Level 2: fuse per-signal uncertainties across signals and passes into the final maps.

The pipeline has two levels and they answer different questions.

**Level 1** (``harvest_*``) computes signals and each signal's *own* uncertainty. It must not
decide the answer. The per-pass fold it used to perform is a within-pass diagnostic — "what
did this pass think" — and folding early is precisely how one saturated sub-signal came to pin
an axis at 1.0 while two independent diarizers, both embedding models and the per-speaker
presence track all agreed nothing had changed: the fold ran before anything had been measured
about the signals, so there was no weight to apply.

**Level 2** (this module) aggregates across every signal and every pass, weighting each signal
by what was measured about it — perturbation stability and physical support — and iterates. Its
maps are the answer a consumer should read.

Once a fold has happened the weights can no longer be applied, which is why the ordering
matters rather than being a matter of taste.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.aggregators import apply_aggregator

__all__ = [
    "write_final_uncertainty",
    "fuse_axis",
    "per_signal_uncertainty",
]

# Fields carrying a signal's own uncertainty in [0, 1]. A signal may expose more than one
# (a same-label claim and a change claim); each counts separately, because they are distinct
# assertions that can disagree.
_UNCERTAINTY_FIELDS = (
    "same_label_uncertainty",
    "change_inconsistency_uncertainty",
    "value",
)


def per_signal_uncertainty(bucket: Mapping[str, Any]) -> dict[str, float]:
    """Each signal's own uncertainty in one bucket — the level-1 emission.

    Reported per signal rather than folded, so level 2 can weight them. A fold cannot be
    re-weighted after the fact, which is the whole reason for the split.

    A signal that said nothing is absent from the result rather than zero-filled: zero is a
    confident claim, and imputing it would manufacture confidence nobody expressed (FR-007).
    """
    votes = bucket.get("votes") or {}
    if not isinstance(votes, Mapping):
        return {}
    out: dict[str, float] = {}
    for name, entry in votes.items():
        if not isinstance(entry, Mapping):
            continue
        for field in _UNCERTAINTY_FIELDS:
            value = entry.get(field)
            if isinstance(value, (int, float)):
                out[str(name)] = max(0.0, min(1.0, float(value)))
                break
    return out


def fuse_axis(
    buckets_by_pass: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, float],
    aggregator: str = "mean",
) -> list[dict[str, Any]]:
    """Fuse one axis's per-signal uncertainties across signals and passes.

    Args:
        buckets_by_pass: ``{pass_label → per-bucket harvested votes}``.
        weights: ``{signal → measured weight}``. A signal absent from the mapping carries
            full weight: a factor never measured must not act as a discount.
        aggregator: How to combine the weighted per-signal uncertainties.

    Returns:
        One row per bucket, in time order, each carrying the fused ``uncertainty``, the
        signals and passes that contributed, and the weight each signal actually carried.
        Ordering is by time then pass so the maps stay byte-identical across runs (SC-004).

        ``uncertainty`` is ``None`` where no signal spoke. That is not the same as ``0.0``,
        which would assert confidence nobody expressed.
    """
    # (start, end) → signal → [uncertainty, ...]; a signal appearing in both passes
    # contributes both readings, because disagreement between passes is evidence too.
    collected: dict[tuple[float, float], dict[str, list[float]]] = {}
    passes_seen: dict[tuple[float, float], set[str]] = {}

    for pass_label in sorted(buckets_by_pass):
        for bucket in buckets_by_pass[pass_label] or []:
            if not isinstance(bucket, Mapping):
                continue
            key = (round(float(bucket.get("start", 0.0)), 6), round(float(bucket.get("end", 0.0)), 6))
            per_signal = per_signal_uncertainty(bucket)
            slot = collected.setdefault(key, {})
            passes_seen.setdefault(key, set()).add(str(pass_label))
            for signal, value in per_signal.items():
                slot.setdefault(signal, []).append(value)

    rows: list[dict[str, Any]] = []
    for start, end in sorted(collected):
        per_signal = collected[(start, end)]
        signals = sorted(per_signal)
        # One reading per signal: the mean across passes. Averaging here rather than treating
        # each pass as a separate voter keeps a signal's weight from scaling with how many
        # passes it happened to appear in.
        values = [sum(per_signal[s]) / len(per_signal[s]) for s in signals]
        applied = [float(weights.get(s, 1.0)) for s in signals]
        fused = (
            apply_aggregator([v for v in values], aggregator, weights=applied if signals else None) if signals else None
        )
        rows.append(
            {
                "start": start,
                "end": end,
                "uncertainty": fused,
                "contributing_signals": signals,
                "contributing_passes": sorted(passes_seen[(start, end)]),
                "signal_weights": {s: w for s, w in zip(signals, applied)},
            }
        )
    return rows


def write_final_uncertainty(
    out_dir: Any,  # noqa: ANN401 — Path
    *,
    harvests: Mapping[str, Any],
    weights_by_axis: Mapping[str, Mapping[str, float]],
    aggregator: str = "mean",
) -> dict[str, Any]:
    """Write ``final/uncertainty/{presence,identity,utterance}.parquet`` — the level-2 answer.

    These are the maps a consumer should read. The per-pass parquets under
    ``<pass>/uncertainty/`` remain, but as level-1 diagnostics: they record what each signal
    said and what a single pass would have concluded on its own, before anything was measured
    about the signals' reliability.

    Args:
        out_dir: Run directory.
        harvests: ``{pass_label → PassHarvest}``.
        weights_by_axis: ``{axis → {signal → measured weight}}``.
        aggregator: Aggregator for the cross-signal fold.

    Returns:
        ``{axis → written path}`` as strings, for the run summary.
    """
    from pathlib import Path

    import pandas as pd

    axis_field = {"presence": "presence_votes", "identity": "identity_votes", "utterance": "utterance_votes"}
    final = Path(out_dir) / "final" / "uncertainty"
    final.mkdir(parents=True, exist_ok=True)

    written: dict[str, Any] = {}
    for axis, field in axis_field.items():
        by_pass = {label: getattr(h, field, []) or [] for label, h in harvests.items()}
        rows = fuse_axis(by_pass, weights=weights_by_axis.get(axis, {}), aggregator=aggregator)
        frame = pd.DataFrame(
            [
                {
                    "start": r["start"],
                    "end": r["end"],
                    "axis": axis,
                    "uncertainty": r["uncertainty"],
                    "contributing_signals": r["contributing_signals"],
                    "contributing_passes": r["contributing_passes"],
                    "signal_weights": json.dumps(r["signal_weights"], sort_keys=True),
                }
                for r in rows
            ],
            columns=[
                "start",
                "end",
                "axis",
                "uncertainty",
                "contributing_signals",
                "contributing_passes",
                "signal_weights",
            ],
        )
        path = final / f"{axis}.parquet"
        frame.to_parquet(path, index=False)
        written[axis] = str(path)
    return written
