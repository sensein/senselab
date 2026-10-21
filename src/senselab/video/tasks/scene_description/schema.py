"""The structured scene description a VLM returns for one frame, and its parser.

``SceneDescription`` is the fixed schema the prompt asks for. :func:`parse_scene_json` turns a raw
model completion into a validated ``SceneDescription`` or an error message suitable for a repair
prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Layout = Literal["gallery", "speaker_view", "screen_share", "single", "other"]
Setting = Literal["home", "office", "clinic", "vehicle", "outdoors", "classroom", "other", "unknown"]
BackgroundType = Literal["real", "virtual", "blurred", "unknown"]
Lighting = Literal["good", "dim", "backlit", "uneven", "unknown"]
HeadFraming = Literal["centered", "partial", "off_frame"]
Distraction = Literal[
    "pets",
    "children",
    "tv_screen",
    "phone_use",
    "eating",
    "moving_background",
    "noise_source_visible",
    "none",
]

LAYOUTS: Tuple[str, ...] = Layout.__args__  # type: ignore[attr-defined]
SETTINGS: Tuple[str, ...] = Setting.__args__  # type: ignore[attr-defined]
BACKGROUND_TYPES: Tuple[str, ...] = BackgroundType.__args__  # type: ignore[attr-defined]
LIGHTINGS: Tuple[str, ...] = Lighting.__args__  # type: ignore[attr-defined]
HEAD_FRAMINGS: Tuple[str, ...] = HeadFraming.__args__  # type: ignore[attr-defined]
DISTRACTIONS: Tuple[str, ...] = Distraction.__args__  # type: ignore[attr-defined]


def _normalise_token(value: Any) -> Any:  # noqa: ANN401
    """Lower-case a string enum value and map spaces and hyphens to underscores."""
    if isinstance(value, str):
        return re.sub(r"[\s\-]+", "_", value.strip().lower())
    return value


class TileDescription(BaseModel):
    """One participant tile of a video-call frame (the whole frame when there is one view).

    Attributes:
        camera_on: Whether the tile shows a live camera image rather than a name card or avatar.
        n_people_visible: People visible in the tile.
        setting: Kind of place the tile's camera is in.
        background_type: Whether the background is the real room, a virtual image, or blurred.
        lighting: Lighting on the participant.
        other_people_present: Whether anyone besides the tile's main participant is visible.
        distractions: Visible distractions from the closed vocabulary; ``["none"]`` when there are none.
        head_framing: Where the main participant's head sits in the tile.
    """

    model_config = ConfigDict(extra="forbid")

    camera_on: bool
    n_people_visible: int = Field(ge=0)
    setting: Setting
    background_type: BackgroundType
    lighting: Lighting
    other_people_present: bool
    distractions: List[Distraction]
    head_framing: HeadFraming

    @field_validator("setting", "background_type", "lighting", "head_framing", mode="before")
    @classmethod
    def _normalise_enum(cls, value: Any) -> Any:  # noqa: ANN401
        return _normalise_token(value)

    @field_validator("distractions", mode="before")
    @classmethod
    def _normalise_distractions(cls, value: Any) -> Any:  # noqa: ANN401
        if isinstance(value, list):
            items = [_normalise_token(v) for v in value]
            if len(items) > 1:
                items = [v for v in items if v != "none"]
            return list(dict.fromkeys(items)) or ["none"]
        return value


class SceneDescription(BaseModel):
    """The structured description of one sampled frame.

    Attributes:
        layout: The call layout visible in the frame.
        n_tiles: Number of participant tiles; 1 for a single view.
        tiles: One entry per tile, the largest (main) tile first.
        notes: Short free-text remarks.
    """

    model_config = ConfigDict(extra="forbid")

    layout: Layout
    n_tiles: int = Field(ge=0)
    tiles: List[TileDescription]
    notes: str = ""

    @field_validator("layout", mode="before")
    @classmethod
    def _normalise_layout(cls, value: Any) -> Any:  # noqa: ANN401
        return _normalise_token(value)

    @field_validator("notes", mode="before")
    @classmethod
    def _notes_default(cls, value: Any) -> Any:  # noqa: ANN401
        return "" if value is None else value


SCHEMA_TEXT = """{
  "layout": "gallery" | "speaker_view" | "screen_share" | "single" | "other",
  "n_tiles": <integer, number of participant video tiles; 1 if a single view>,
  "tiles": [
    {
      "camera_on": <true | false>,
      "n_people_visible": <integer>,
      "setting": "home" | "office" | "clinic" | "vehicle" | "outdoors" | "classroom" | "other" | "unknown",
      "background_type": "real" | "virtual" | "blurred" | "unknown",
      "lighting": "good" | "dim" | "backlit" | "uneven" | "unknown",
      "other_people_present": <true | false>,
      "distractions": [zero or more of "pets", "children", "tv_screen", "phone_use", "eating",
                       "moving_background", "noise_source_visible"; or ["none"]],
      "head_framing": "centered" | "partial" | "off_frame"
    }
  ],
  "notes": "<one short sentence>"
}"""

DESCRIBE_PROMPT = (
    "This image is one frame of a recorded video call or video interview. Describe the visible "
    "environment of each participant tile.\n"
    "Return STRICT JSON only, on a single line without indentation (no markdown, no code fences, no "
    "commentary), matching exactly this schema:\n"
    f"{SCHEMA_TEXT}\n"
    'Rules: list one entry in "tiles" per participant tile, the largest tile first; a frame showing '
    'one person or one camera view is layout "single" with one tile. A tile showing only a name, '
    "initials or an avatar has camera_on false, n_people_visible 0, setting, background_type and lighting "
    '"unknown", and head_framing "off_frame". "other_people_present" means someone besides the '
    "tile's main participant is visible, so it is true whenever n_people_visible is greater than 1. "
    "Use only the listed values."
)

REPAIR_PROMPT_TEMPLATE = (
    "The text below was supposed to be JSON matching this schema:\n{schema}\n"
    "It failed validation with this error:\n{error}\n"
    "Text:\n{text}\n"
    "Return only the corrected JSON object, no markdown and no commentary."
)


def repair_prompt(raw_text: str, error: str, max_chars: int = 4000) -> str:
    """Build the text-only prompt that asks the model to fix an invalid completion.

    Args:
        raw_text: The completion that failed to parse.
        error: The parse or validation error.
        max_chars: Truncation applied to ``raw_text`` and ``error``.

    Returns:
        The repair prompt.
    """
    return REPAIR_PROMPT_TEMPLATE.format(schema=SCHEMA_TEXT, error=error[:max_chars], text=raw_text[:max_chars])


def _extract_json_object(text: str) -> str:
    """Return the outermost ``{...}`` span of ``text``, with any code fences removed."""
    stripped = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the model output")
    return stripped[start : end + 1]


def parse_scene_json(text: str) -> Tuple[Optional[SceneDescription], Optional[str]]:
    """Parse and validate a raw completion into a ``SceneDescription``.

    Code fences and text around the outermost JSON object are ignored; enum values are matched
    case-insensitively with spaces and hyphens read as underscores. Nothing else is coerced.

    Args:
        text: The model completion.

    Returns:
        ``(description, None)`` on success, ``(None, error_message)`` otherwise.
    """
    try:
        payload = json.loads(_extract_json_object(text))
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "the top-level JSON value is not an object"
    try:
        return SceneDescription.model_validate(payload), None
    except ValidationError as exc:
        return None, "schema validation failed: " + "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
