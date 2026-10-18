"""Result models for gallery-video facial analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

Rect = Tuple[int, int, int, int]
"""``(x, y, width, height)`` in pixels."""


class TileBox(BaseModel):
    """One participant tile of a gallery layout.

    Attributes:
        tile_id: ``T0``, ``T1``, ... in reading order (top-to-bottom, then left-to-right).
        x: Left edge in source pixels, already inset past the tile border.
        y: Top edge in source pixels.
        width: Width in source pixels.
        height: Height in source pixels.
        masks: Tile-relative rectangles blanked before analysis (the name label).
        derivation: How the tile was found: ``"gutter"`` (bounded by dark canvas or an image seam),
            ``"aspect_split"`` (an equal split of a merged region, chosen by tile aspect ratio and
            face positions), ``"face_split"`` (a merged region cut between consistently separated
            face clusters, whatever its aspect ratio), ``"person_track"`` (a speaker-view person
            whose face is analysed wherever :attr:`GalleryLayout.persons` places it) or
            ``"whole_frame"`` (no layout was estimated).
        confidence: In ``[0, 1]``. For ``gutter`` tiles the fraction of sampled frames that show the
            tile's boundary; for ``aspect_split`` and ``face_split`` tiles the face-occupancy support
            of the split; for ``person_track`` tiles the fraction of identity samples showing the person.
    """

    tile_id: str
    x: int
    y: int
    width: int
    height: int
    masks: List[Rect] = Field(default_factory=list)
    derivation: Literal["gutter", "aspect_split", "face_split", "person_track", "whole_frame"] = "gutter"
    confidence: float = 1.0

    @property
    def box(self) -> Rect:
        """The tile as ``(x, y, width, height)``."""
        return (self.x, self.y, self.width, self.height)


class LayoutSegment(BaseModel):
    """A run of consecutive sampled frames that share one layout estimate.

    Attributes:
        start_s: Time of the first sample in the run.
        end_s: Time of the last sample in the run.
        n_samples: Sampled frames in the run.
        boxes: Tile rectangles estimated in this run.
        kind: ``"reference"`` when the run matches the reference layout; ``"subset"`` when every tile
            it found is a reference tile but some reference tiles are missing (a dark or camera-off
            tile); ``"different"`` otherwise (screen share, speaker view, a resized window).
        mode: ``"gallery"`` (participants in spatial tiles), ``"speaker_view"`` (one large tile
            showing one person at a time) or ``"other"`` (neither, e.g. screen share). Always
            ``"gallery"`` in a gallery-mode layout.
        persons: In a speaker-view layout, ``person_id -> region_id`` for the persons the run shows.
    """

    start_s: float
    end_s: float
    n_samples: int
    boxes: List[Rect]
    kind: Literal["reference", "subset", "different"]
    mode: Literal["gallery", "speaker_view", "other"] = "gallery"
    persons: Dict[str, str] = Field(default_factory=dict)


class PersonSegment(BaseModel):
    """An interval during which a person is shown in one region.

    Attributes:
        start_s: First identity sample of the interval (``0`` when it opens the recording).
        end_s: Last identity sample of the interval (the recording end when it closes it).
        region_id: The region (:attr:`GalleryLayout.regions`) the person's face is in.
    """

    start_s: float
    end_s: float
    region_id: str


class PersonTrack(BaseModel):
    """Where one identity-clustered person is visible over a speaker-view recording.

    Attributes:
        person_id: ``P0``, ``P1``, ... in order of first appearance.
        segments: Visible intervals, in time order; frames outside them belong to no row of this person.
        n_samples: Identity samples in which the person's face was found.
    """

    person_id: str
    segments: List[PersonSegment]
    n_samples: int


class GalleryLayout(BaseModel):
    """Fixed tile rectangles of a gallery recording and how the layout varied over time.

    Attributes:
        frame_width: Source frame width.
        frame_height: Source frame height.
        tiles: The reference layout's tiles.
        timeline: Per-run layout estimates over the sampled frames.
        reference_fraction: Fraction of sampled frames whose layout matches the reference.
        n_samples: Frames sampled.
        mode: ``"gallery"`` (tiles are fixed regions) or ``"speaker_view"`` (tiles are person tracks
            ``P0``, ``P1``, ... whose region changes over time, see ``persons``).
        regions: Physical regions of a speaker-view layout: ``S`` (the speaker tile) and ``G0``, ``G1``,
            ... (the tiles of gallery intervals). Empty in gallery mode.
        persons: Per person track, where and when the person is visible. Empty in gallery mode.
        identity: Face-identity provenance of a speaker-view layout (models, hashes, threshold).
    """

    frame_width: int
    frame_height: int
    tiles: List[TileBox]
    timeline: List[LayoutSegment]
    reference_fraction: float
    n_samples: int
    mode: Literal["gallery", "speaker_view"] = "gallery"
    regions: List[TileBox] = Field(default_factory=list)
    persons: List[PersonTrack] = Field(default_factory=list)
    identity: Dict[str, Any] = Field(default_factory=dict)

    def layout_changes(self) -> List[LayoutSegment]:
        """Runs whose layout differs from the reference (screen share, speaker view, resized window)."""
        return [segment for segment in self.timeline if segment.kind == "different"]

    def visible(self, tile_id: str, times: Sequence[float]) -> np.ndarray:
        """Whether a tile's face is shown at each time.

        Args:
            tile_id: A tile (gallery mode) or person track (speaker-view mode).
            times: Sample times in seconds.

        Returns:
            Boolean array; all True for a gallery tile, the person's segments for a person track.
        """
        t = np.asarray(times, dtype=np.float64)
        track = next((p for p in self.persons if p.person_id == tile_id), None)
        if track is None:
            return np.ones(t.shape, bool)
        out = np.zeros(t.shape, bool)
        for segment in track.segments:
            out |= (t >= segment.start_s) & (t <= segment.end_s)
        return out


class HeadGestureEvent(BaseModel):
    """A nod (pitch oscillation) or shake (yaw oscillation).

    Attributes:
        tile_id: The tile the face is in.
        kind: ``"nod"`` or ``"shake"``.
        start_s: Time of the first extremum.
        end_s: Time of the last extremum.
        n_extrema: Alternating extrema in the event.
        amplitude_deg: Median swing between consecutive extrema of the band-passed angle.
        frequency_hz: Oscillation rate implied by the extrema spacing.
    """

    tile_id: str
    kind: Literal["nod", "shake"]
    start_s: float
    end_s: float
    n_extrema: int
    amplitude_deg: float
    frequency_hz: float


class BlinkEvent(BaseModel):
    """One blink.

    Attributes:
        tile_id: The tile the face is in.
        start_s: First sample above the low threshold.
        end_s: Last sample above the low threshold.
        peak: Maximum mean eyeBlink coefficient in the event.
    """

    tile_id: str
    start_s: float
    end_s: float
    peak: float


class FrameTable(BaseModel):
    """Per-frame measurements of one analysis pass, one DataFrame per tile.

    Attributes:
        kind: ``"geometry"`` (MediaPipe) or ``"affect"`` (py-feat).
        fps: Sampling rate the frames were taken at.
        layout: The tile layout used.
        tiles: ``tile_id`` to a DataFrame with a ``time_s`` column and one column per measure.
        provenance: Backend name, version, model identity and pinned revision.
        timing: ``wall_s``, ``video_s`` and ``x_realtime`` of the pass.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["geometry", "affect"]
    fps: float
    layout: GalleryLayout
    tiles: Dict[str, pd.DataFrame]
    provenance: Dict[str, Any] = Field(default_factory=dict)
    timing: Dict[str, float] = Field(default_factory=dict)

    def on_grid(
        self, column: str, times: Sequence[float], tolerance_s: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """One column of every tile resampled onto a shared time grid.

        Person tracks have rows only where the person is shown, so their tables do not share a grid;
        this puts them on one for :func:`~senselab.video.tasks.facial_analysis.link_tiles_to_speakers`.

        Args:
            column: The column to read.
            times: The grid, in seconds.
            tolerance_s: Largest distance to the nearest row; half the frame step when ``None``.

        Returns:
            ``tile_id -> values``, NaN where no row lies within the tolerance.
        """
        grid = np.asarray(times, dtype=np.float64)
        tolerance = 0.5 / self.fps if tolerance_s is None else float(tolerance_s)
        out: Dict[str, np.ndarray] = {}
        for tile_id, frame in self.tiles.items():
            values = np.full(grid.shape, np.nan)
            if len(frame) and column in frame:
                row_times = frame["time_s"].to_numpy(np.float64)
                index = np.clip(np.searchsorted(row_times, grid), 1, max(len(row_times) - 1, 1))
                left = np.clip(index - 1, 0, len(row_times) - 1)
                right = np.clip(index, 0, len(row_times) - 1)
                nearest = np.where(np.abs(row_times[left] - grid) <= np.abs(row_times[right] - grid), left, right)
                ok = np.abs(row_times[nearest] - grid) <= tolerance
                values[ok] = frame[column].to_numpy(np.float64)[nearest[ok]]
            out[tile_id] = values
        return out

    def to_long(self) -> pd.DataFrame:
        """All tiles stacked into one Parquet-ready frame with a ``tile_id`` column."""
        frames = [df.assign(tile_id=tile_id) for tile_id, df in self.tiles.items()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        kind: Literal["geometry", "affect"],
        fps: float,
        layout: GalleryLayout,
        provenance: Optional[Dict[str, Any]] = None,
        timing: Optional[Dict[str, float]] = None,
    ) -> "FrameTable":
        """Inverse of :meth:`to_long`: split a stacked frame back into one DataFrame per tile.

        Args:
            df: A frame with a ``tile_id`` column, as returned by :meth:`to_long`.
            kind: ``"geometry"`` or ``"affect"``.
            fps: Sampling rate the frames were taken at.
            layout: The tile layout used.
            provenance: Backend identity; empty when ``None``.
            timing: Pass timing; empty when ``None``.

        Returns:
            A table whose tiles keep first-appearance order, with the ``tile_id`` column removed.
        """
        tiles = (
            {
                str(tile_id): group.drop(columns="tile_id").reset_index(drop=True)
                for tile_id, group in df.groupby("tile_id", sort=False)
            }
            if "tile_id" in df.columns
            else {}
        )
        return cls(
            kind=kind,
            fps=fps,
            layout=layout,
            tiles=tiles,
            provenance=dict(provenance or {}),
            timing=dict(timing or {}),
        )


class TileSummary(BaseModel):
    """Per-tile summary over a whole recording.

    Attributes:
        tile_id: The tile.
        duration_s: Span covered by the frames.
        face_present_fraction: Fraction of geometry frames with a face.
        blink_rate_per_min: Blinks per minute of face-present time.
        nods_per_min: Nods per minute of face-present time.
        shakes_per_min: Shakes per minute of face-present time.
        mouth_open_mean: Mean jawOpen over face-present frames.
        toward_camera_fraction: Fraction of face-present frames whose gaze proxy passes the threshold.
        motion_energy_mean: Mean frame-difference energy of the tile.
        affect_means: Mean of each affect column over frames with a face (empty without affect).
    """

    tile_id: str
    duration_s: float
    face_present_fraction: float
    blink_rate_per_min: Optional[float] = None
    nods_per_min: Optional[float] = None
    shakes_per_min: Optional[float] = None
    mouth_open_mean: Optional[float] = None
    toward_camera_fraction: Optional[float] = None
    motion_energy_mean: Optional[float] = None
    affect_means: Dict[str, float] = Field(default_factory=dict)


class SpeakerLink(BaseModel):
    """Assignment of tiles to diarized speakers from mouth motion.

    Attributes:
        correlations: ``tile_id -> speaker -> Pearson r`` of mouth activity with speaker activity.
        assignment: ``tile_id -> speaker`` maximising total correlation (Hungarian).
        margins: Per tile, its assigned correlation minus its best unassigned alternative.
        total_margin: Best total correlation minus the second-best complete assignment.
        p_value: Fraction of circular time shifts of the mouth series (at least ``min_shift_s``
            apart from the original) whose best total correlation reaches the observed one; the
            assignment's confidence, since a margin alone is inflated when speaker tracks are
            complements of each other.
        n_frames: Frames where both series were finite.
        visibility_activity_corr: ``tile_id -> speaker -> Pearson r`` of the tile's visibility (0/1)
            with speaker activity, when visibility was given. In active-speaker view the person shown
            is usually the one talking, so this is a link cue of its own.
    """

    correlations: Dict[str, Dict[str, float]]
    assignment: Dict[str, str]
    margins: Dict[str, float]
    total_margin: float
    p_value: float
    n_frames: int
    visibility_activity_corr: Dict[str, Dict[str, float]] = Field(default_factory=dict)
