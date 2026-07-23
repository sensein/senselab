"""Label Studio region builders for the senselab backends.

Copied from ``scripts/analyze_audio.py`` (``_ls_label_region`` / ``_diarization_to_ls``) so
the backends do not import the batch CLI. The emitted region dicts use the exact same schema
in both directions -- an export task's ``predictions[].result`` and a served
``ModelResponse.result``. If these are later refactored into one shared module,
``analyze_audio.py`` and these backends should import that single implementation.

This module intentionally has no senselab import: it operates on plain segment objects
(``ScriptLine``) or JSON dicts via :func:`seg_attr`.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_SPEAKER_LABEL = "SPEAKER_UNKNOWN"
DEFAULT_TO_NAME = "audio"


def new_region_id(prefix: str, idx: int) -> str:
    """Return a stable per-region id for a Label Studio result entry.

    Args:
        prefix: A short track prefix (typically the control ``from_name``).
        idx: Zero-based index of the region within its track.

    Returns:
        A deterministic id such as ``"diarization_0003"``.
    """
    return f"{prefix}_{idx:04d}"


def seg_attr(seg: Any, name: str) -> Any:  # noqa: ANN401
    """Return ``seg.name`` whether ``seg`` is a Pydantic model or a plain dict.

    Args:
        seg: A ``ScriptLine`` (Pydantic) or a JSON-deserialized dict.
        name: Attribute/key to read.

    Returns:
        The attribute value, or ``None`` when absent.
    """
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


def ls_label_region(
    *,
    region_id: str,
    from_name: str,
    start: float,
    end: float,
    label: str,
    score: Optional[float] = None,
    to_name: str = DEFAULT_TO_NAME,
) -> dict[str, Any]:
    """Build one Label Studio ``labels`` result entry on the audio timeline.

    Args:
        region_id: Stable region id (see :func:`new_region_id`).
        from_name: The ``<Labels>`` control name this region belongs to.
        start: Region start in seconds.
        end: Region end in seconds.
        label: The label value to assign.
        score: Optional confidence, attached only when not ``None``.
        to_name: The object tag name (defaults to ``"audio"``).

    Returns:
        A Label Studio region dict of ``type: "labels"``.
    """
    entry: dict[str, Any] = {
        "id": region_id,
        "from_name": from_name,
        "to_name": to_name,
        "type": "labels",
        "value": {"start": float(start), "end": float(end), "labels": [label]},
    }
    if score is not None:
        entry["score"] = float(score)
    return entry


def diarization_to_ls(
    segments: Any,  # noqa: ANN401
    from_name: str,
    *,
    to_name: str = DEFAULT_TO_NAME,
) -> list[dict[str, Any]]:
    """Convert a per-audio list of diarization segments into Label Studio regions.

    Args:
        segments: The ``ScriptLine`` list for a single audio (or JSON dicts). A nested
            ``List[List[...]]`` (the raw ``diarize_audios`` shape) is unwrapped to its first
            element for convenience.
        from_name: The ``<Labels>`` control name to attach regions to.
        to_name: The object tag name (defaults to ``"audio"``).

    Returns:
        A list of ``type: "labels"`` region dicts, one per timed segment.
    """
    out: list[dict[str, Any]] = []
    if not segments:
        return out
    if isinstance(segments, list) and segments and isinstance(segments[0], list):
        segments = segments[0]
    for i, seg in enumerate(segments):
        start = seg_attr(seg, "start")
        end = seg_attr(seg, "end")
        speaker = seg_attr(seg, "speaker") or DEFAULT_SPEAKER_LABEL
        if start is None or end is None:
            continue
        out.append(
            ls_label_region(
                region_id=new_region_id(from_name, i),
                from_name=from_name,
                start=start,
                end=end,
                label=str(speaker),
                to_name=to_name,
            )
        )
    return out
