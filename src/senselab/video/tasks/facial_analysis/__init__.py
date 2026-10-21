""".. include:: ./doc.md"""  # noqa: D415

from .api import (  # noqa: F401
    FacialAnalysis,
    NonverbalEvents,
    analyze_gallery_video,
    detect_gallery_tiles,
    detect_nonverbal_events,
    extract_face_geometry,
    extract_facial_affect,
    link_tiles_to_speakers,
    load_defaults,
    summarize_tiles,
)
from .results import (  # noqa: F401
    BlinkEvent,
    FrameTable,
    GalleryLayout,
    HeadGestureEvent,
    LayoutSegment,
    PersonSegment,
    PersonTrack,
    SpeakerLink,
    TileBox,
    TileSummary,
)
from .signals import segments_to_activity  # noqa: F401

__all__ = [
    "FacialAnalysis",
    "NonverbalEvents",
    "analyze_gallery_video",
    "detect_gallery_tiles",
    "detect_nonverbal_events",
    "extract_face_geometry",
    "extract_facial_affect",
    "link_tiles_to_speakers",
    "load_defaults",
    "summarize_tiles",
    "segments_to_activity",
    "BlinkEvent",
    "FrameTable",
    "GalleryLayout",
    "HeadGestureEvent",
    "LayoutSegment",
    "PersonSegment",
    "PersonTrack",
    "SpeakerLink",
    "TileBox",
    "TileSummary",
]
