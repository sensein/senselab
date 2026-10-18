"""Active-speaker view: person tracks from face identity instead of fixed spatial tiles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from senselab.video.tasks.facial_analysis.frame_stream import iter_frames, probe_video
from senselab.video.tasks.facial_analysis.identity import Face, cluster_identities, identity_provenance
from senselab.video.tasks.facial_analysis.results import (
    GalleryLayout,
    LayoutSegment,
    PersonSegment,
    PersonTrack,
    Rect,
    TileBox,
)
from senselab.video.tasks.facial_analysis.tiles import (
    _classify,
    _reading_order,
    _same_layout,
    estimate_frame_layout,
    face_position_split,
    modal_layout,
    tile_box,
)

Identifier = Callable[[np.ndarray], List[Face]]
"""``RGB frame -> faces`` with embeddings, boxes in that frame's pixels."""

SPEAKER_REGION = "S"


@dataclass
class IdentitySample:
    """One identity sample: the frame's layout estimate and its faces, in source pixels.

    Attributes:
        time_s: Frame time.
        rects: Per-frame tile estimate (:func:`estimate_frame_layout`).
        faces: Faces with identity embeddings.
    """

    time_s: float
    rects: List[Rect]
    faces: List[Face] = field(default_factory=list)


def is_speaker_view_candidate(layout: GalleryLayout, config: Dict[str, Any]) -> bool:
    """Whether a gallery layout could be active-speaker view: one detected tile covering much of the frame.

    Args:
        layout: A gallery-mode layout.
        config: The ``speaker_view`` section of the defaults.

    Returns:
        True when the layout has a single non-``whole_frame`` tile of at least
        ``min_tile_area_fraction`` of the frame.
    """
    if layout.mode != "gallery" or len(layout.tiles) != 1 or layout.tiles[0].derivation == "whole_frame":
        return False
    tile = layout.tiles[0]
    area = tile.width * tile.height / max(layout.frame_width * layout.frame_height, 1)
    return area >= float(config["min_tile_area_fraction"])


def sample_identities(
    path: str, tiles_config: Dict[str, Any], config: Dict[str, Any], identifier: Identifier
) -> Tuple[List[IdentitySample], float]:
    """Decode the recording sequentially at ``identity_fps`` and find faces with their embeddings.

    Args:
        path: The video file.
        tiles_config: The ``tiles`` section of the defaults.
        config: The ``speaker_view`` section of the defaults.
        identifier: Faces with embeddings per RGB frame.

    Returns:
        The samples and the recording duration in seconds.
    """
    info = probe_video(path)
    width, height = info["width"], info["height"]
    scale = min(1.0, float(config["identity_width"]) / width)
    samples: List[IdentitySample] = []
    last = 0.0
    for t, rgb in iter_frames(path, fps=float(config["identity_fps"]), scale=scale):
        sy, sx = height / rgb.shape[0], width / rgb.shape[1]
        rects = estimate_frame_layout(
            rgb.mean(axis=2).astype(np.uint8),
            dark_level=tiles_config["dark_level"],
            gutter_fraction=tiles_config["gutter_fraction"],
            min_tile_fraction=tiles_config["min_tile_fraction"],
        )
        faces = [
            Face(
                box=(int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy))),
                score=f.score,
                embedding=f.embedding,
            )
            for f in identifier(rgb)
            for x, y, w, h in [f.box]
        ]
        samples.append(
            IdentitySample(
                time_s=float(t),
                rects=[
                    (int(round(x * sx)), int(round(y * sy)), int(round(w * sx)), int(round(h * sy)))
                    for x, y, w, h in rects
                ],
                faces=faces,
            )
        )
        last = float(t)
    return samples, max(float(info["duration_s"]), last)


def _centre_in(box: Rect, rect: Rect) -> bool:
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    return rect[0] <= cx < rect[0] + rect[2] and rect[1] <= cy < rect[1] + rect[3]


def _gallery_regions(
    samples: Sequence[IdentitySample], tiles_config: Dict[str, Any], width: int, height: int
) -> List[Rect]:
    """Outer rectangles of the gallery intervals' tiles, split between face clusters where merged."""
    estimates = [(s.time_s, s.rects) for s in samples if s.rects]
    if not estimates:
        return []
    modal = modal_layout(estimates, width, height, tiles_config["match_iou"])
    faces = [[f.box for f in s.faces] for s in samples]
    out: List[Rect] = []
    for rect in modal:
        cells = face_position_split(faces, rect, tiles_config)
        out += [cell for cell, _, _ in cells] if cells else [rect]
    return _reading_order(out)


def speaker_view_layout(
    samples: Sequence[IdentitySample],
    base: GalleryLayout,
    tiles_config: Dict[str, Any],
    config: Dict[str, Any],
    duration_s: float,
) -> Optional[GalleryLayout]:
    """Person tracks of an active-speaker recording, or ``None`` when it is not one.

    Every face embedding is clustered into persons (:func:`cluster_identities`). A sample whose
    layout estimate matches the base layout's single tile is ``speaker_view`` and shows the person
    of its largest face there (region ``S``); a sample with two or more tiles or faces is
    ``gallery`` and maps each face to the gallery tile (``G0``, ``G1``, ...) holding its centre, by
    majority over each run of gallery samples; any other sample is ``other``. A person's segments are
    runs of consecutive samples showing them in one region, opened to the recording's start or end
    when the run touches it.

    Args:
        samples: Identity samples in time order.
        base: The gallery-mode layout, with one tile.
        tiles_config: The ``tiles`` section of the defaults.
        config: The ``speaker_view`` section of the defaults.
        duration_s: Recording duration.

    Returns:
        A ``speaker_view`` layout with tiles ``P0``, ``P1``, ..., or ``None`` when fewer than
        ``min_speaker_view_fraction`` of the samples show the speaker tile or fewer than two persons
        each appear in ``min_person_fraction`` of its face samples.
    """
    if not samples or len(base.tiles) != 1:
        return None
    speaker = base.tiles[0]
    width, height = base.frame_width, base.frame_height
    match_iou = tiles_config["match_iou"]
    threshold = float(config["cosine_threshold"])

    flat = [(i, j) for i, s in enumerate(samples) for j in range(len(s.faces))]
    labels = cluster_identities(
        [samples[i].faces[j].embedding for i, j in flat],
        threshold,
        min_fraction=float(config["min_person_fraction"]),
        max_samples=int(config["max_cluster_samples"]),
    )
    label_of = {key: int(label) for key, label in zip(flat, labels)}

    modes: List[str] = []
    for sample in samples:
        if sample.rects and _classify(sample.rects, [speaker.box], match_iou) == "reference":
            modes.append("speaker_view")
        elif len(sample.rects) >= 2 or len(sample.faces) >= 2:
            modes.append("gallery")
        else:
            modes.append("other")

    shown: List[Dict[str, str]] = [{} for _ in samples]
    for i, (sample, mode) in enumerate(zip(samples, modes)):
        if mode != "speaker_view":
            continue
        inside = [j for j, f in enumerate(sample.faces) if _centre_in(f.box, speaker.box)]
        if inside and label_of[(i, inside[0])] >= 0:
            shown[i] = {f"P{label_of[(i, inside[0])]}": SPEAKER_REGION}

    n_view = modes.count("speaker_view")
    view_counts = Counter(p for m, s in zip(modes, shown) if m == "speaker_view" for p in s)
    n_view_faces = sum(1 for m, s in zip(modes, shown) if m == "speaker_view" and s)
    switching = [p for p, c in view_counts.items() if c >= float(config["min_person_fraction"]) * max(n_view_faces, 1)]
    if n_view < float(config["min_speaker_view_fraction"]) * len(samples) or len(switching) < 2:
        return None

    gallery_samples = [s for s, m in zip(samples, modes) if m == "gallery"]
    gallery_rects = _gallery_regions(gallery_samples, tiles_config, width, height)
    tile_kwargs = {
        "label_mask_width_fraction": tiles_config["label_mask_width_fraction"],
        "label_mask_height_fraction": tiles_config["label_mask_height_fraction"],
        "border_inset_fraction": tiles_config["border_inset_fraction"],
    }
    regions = [speaker.model_copy(update={"tile_id": SPEAKER_REGION})]
    regions += [tile_box(f"G{k}", rect, "gutter", 1.0, **tile_kwargs) for k, rect in enumerate(gallery_rects)]

    i = 0
    while i < len(samples):
        if modes[i] != "gallery":
            i += 1
            continue
        end = i
        while end + 1 < len(samples) and modes[end + 1] == "gallery":
            end += 1
        votes: Dict[str, Counter] = {}
        for k in range(i, end + 1):
            for j, face in enumerate(samples[k].faces):
                label = label_of[(k, j)]
                where = next((g for g, rect in enumerate(gallery_rects) if _centre_in(face.box, rect)), None)
                if label >= 0 and where is not None:
                    votes.setdefault(f"P{label}", Counter())[f"G{where}"] += 1
        mapping: Dict[str, str] = {}
        taken: Dict[str, int] = {}
        for person, counter in sorted(votes.items(), key=lambda kv: -kv[1].most_common(1)[0][1]):
            region, count = counter.most_common(1)[0]
            if region not in taken:
                mapping[person] = region
                taken[region] = count
        for k in range(i, end + 1):
            shown[k] = dict(mapping)
        i = end + 1

    persons = sorted({p for s in shown for p in s}, key=lambda p: int(p[1:]))
    tracks: List[PersonTrack] = []
    for person in persons:
        segments: List[PersonSegment] = []
        run_start: Optional[int] = None
        for k in range(len(samples) + 1):
            region = shown[k].get(person) if k < len(samples) else None
            previous = shown[k - 1].get(person) if k > 0 and run_start is not None else None
            if run_start is not None and region != previous:
                start = 0.0 if run_start == 0 else samples[run_start].time_s
                stop = duration_s if k == len(samples) else samples[k - 1].time_s
                segments.append(PersonSegment(start_s=start, end_s=stop, region_id=previous or ""))
                run_start = None
            if run_start is None and region is not None:
                run_start = k
        n_seen = sum(
            1
            for k, sample in enumerate(samples)
            for j in range(len(sample.faces))
            if label_of[(k, j)] == int(person[1:])
        )
        tracks.append(PersonTrack(person_id=person, segments=segments, n_samples=n_seen))

    timeline: List[LayoutSegment] = []
    for sample, mode, mapping in zip(samples, modes, shown):
        kind = _classify(sample.rects, [speaker.box], match_iou)
        last = timeline[-1] if timeline else None
        if (
            last is not None
            and last.mode == mode
            and last.persons == mapping
            and last.kind == kind
            and _same_layout(sample.rects, last.boxes, match_iou)
        ):
            last.end_s = sample.time_s
            last.n_samples += 1
        else:
            timeline.append(
                LayoutSegment(
                    start_s=sample.time_s,
                    end_s=sample.time_s,
                    n_samples=1,
                    boxes=list(sample.rects),
                    kind=kind,  # type: ignore[arg-type]
                    mode=mode,  # type: ignore[arg-type]
                    persons=dict(mapping),
                )
            )

    tiles = [
        TileBox(
            tile_id=track.person_id,
            x=speaker.x,
            y=speaker.y,
            width=speaker.width,
            height=speaker.height,
            masks=list(speaker.masks),
            derivation="person_track",
            confidence=sum(1 for s in shown if track.person_id in s) / len(samples),
        )
        for track in tracks
    ]
    identity = identity_provenance(threshold)
    identity.update(
        {
            "identity_fps": float(config["identity_fps"]),
            "n_face_samples": len(flat),
            "n_unclustered": int((labels < 0).sum()),
            "mode_counts": dict(Counter(modes)),
        }
    )
    return GalleryLayout(
        frame_width=width,
        frame_height=height,
        tiles=tiles,
        timeline=timeline,
        reference_fraction=n_view / len(samples),
        n_samples=len(samples),
        mode="speaker_view",
        regions=regions,
        persons=tracks,
        identity=identity,
    )


def detect_speaker_view_in_file(
    path: str,
    base: GalleryLayout,
    tiles_config: Dict[str, Any],
    config: Dict[str, Any],
    identifier: Optional[Identifier] = None,
) -> GalleryLayout:
    """Replace a single-tile layout by person tracks when the recording is active-speaker view.

    Args:
        path: The video file.
        base: The gallery-mode layout from :func:`detect_gallery_tiles_in_file`.
        tiles_config: The ``tiles`` section of the defaults.
        config: The ``speaker_view`` section of the defaults.
        identifier: Faces with embeddings per RGB frame; YuNet + SFace on the pinned models when ``None``.

    Returns:
        The speaker-view layout, or ``base`` when the recording is not active-speaker view.
    """
    if not is_speaker_view_candidate(base, config):
        return base
    if identifier is None:
        from senselab.video.tasks.facial_analysis.identity import default_identifier

        identifier = default_identifier(config)
    samples, duration = sample_identities(path, tiles_config, config, identifier)
    return speaker_view_layout(samples, base, tiles_config, config, duration) or base
