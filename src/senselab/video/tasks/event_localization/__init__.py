""".. include:: ./doc.md"""  # noqa: D415

from senselab.video.tasks.event_localization.api import (
    LocalizationResult,
    WindowReply,
    WindowTooLarge,
    localize_events,
)
from senselab.video.tasks.event_localization.merge import merge_windows, rasterize
from senselab.video.tasks.event_localization.metrics import (
    TIOUS,
    binary_map,
    event_pairs,
    event_prf,
    map_tal,
    mean_over,
    presence,
    seg_prf,
    shuffle_floor,
    tiou,
    variant_confusion,
    whole_video,
)
from senselab.video.tasks.event_localization.parsing import (
    build_prompt,
    parse_reply,
    segment_list_schema,
)
from senselab.video.tasks.event_localization.windows import (
    plan_windows,
    shift_segments,
    split_window,
)

__all__ = [
    "LocalizationResult",
    "TIOUS",
    "WindowReply",
    "WindowTooLarge",
    "binary_map",
    "build_prompt",
    "event_pairs",
    "event_prf",
    "localize_events",
    "map_tal",
    "mean_over",
    "merge_windows",
    "parse_reply",
    "plan_windows",
    "presence",
    "rasterize",
    "seg_prf",
    "segment_list_schema",
    "shift_segments",
    "shuffle_floor",
    "split_window",
    "tiou",
    "variant_confusion",
    "whole_video",
]
