#!/usr/bin/env python3
"""Throughput benchmark for ``senselab.video.tasks.facial_analysis``.

    uv run python scripts/bench_facial_analysis.py [--video PATH] [--workdir DIR] [--threads N]
        [--geometry-fps 8 16] [--affect-fps 2 4] [--device mps] [--out results.json]

Without ``--video`` it builds the benchmark recording itself: a 120 s, 1920x1080, 16 fps two-tile
gallery (dark canvas, name labels) from two public clips on Wikimedia Commons, downloaded into
``--workdir`` (default ``$TMPDIR/senselab-facial-bench``):

- "Science Speaks - Talking to Women in Science (Liz, Epidemiologist)", NIOSH, public domain
  (480p VP9 transcode, about 26 MB);
- "Interview with Barbara Firth", CC BY 3.0 (about 4 MB).

Configurations run one at a time, each in its own worker process. Every row records the 1-minute
load average before and after, because a timing taken on a busy host is an upper bound on cost. The
summary projects hours for a corpus of ``--corpus-hours`` (default 27) at each rate.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CLIPS = {
    "liz.webm": "https://upload.wikimedia.org/wikipedia/commons/transcoded/7/7b/"
    "Science_Speaks_-_Talking_to_Women_in_Science_%28Liz%2C_Epidemiologist%29.webm/"
    "Science_Speaks_-_Talking_to_Women_in_Science_%28Liz%2C_Epidemiologist%29.webm.480p.vp9.webm",
    "barbara.webm": "https://upload.wikimedia.org/wikipedia/commons/d/d0/Interview_with_Barbara_Firth.webm",
}
USER_AGENT = "senselab-bench (https://github.com/sensein/senselab)"


def _download(url: str, target: Path) -> None:
    if target.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        target.write_bytes(response.read())


def _fill(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to cover ``width x height`` and centre-crop, as Zoom fills a tile."""
    from PIL import Image

    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    resized = np.asarray(Image.fromarray(image).resize((int(round(w * scale)), int(round(h * scale)))))
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return resized[top : top + height, left : left + width]


def build_gallery(workdir: Path, seconds: float = 120.0) -> Path:
    """Download the two clips and compose the 1080p 16 fps two-tile gallery."""
    import av

    from senselab.video.tasks.facial_analysis.frame_stream import iter_frames

    out = workdir / "gallery_1080p16.mp4"
    if out.exists():
        return out
    for name, url in CLIPS.items():
        _download(url, workdir / name)
    streams = [iter_frames(str(workdir / name), fps=16.0) for name in CLIPS]
    tile_w, tile_h, top, xs = 952, 536, 272, (4, 964)
    with av.open(str(out), mode="w") as container:
        try:
            stream = container.add_stream("libx264", rate=16)
            stream.options = {"crf": "26", "preset": "veryfast"}
        except Exception:  # noqa: BLE001 - a PyAV build without libx264
            stream = container.add_stream("mpeg4", rate=16)
            stream.bit_rate = 8_000_000
        stream.width, stream.height, stream.pix_fmt = 1920, 1080, "yuv420p"
        for _ in range(int(seconds * 16)):
            canvas = np.full((1080, 1920, 3), 26, np.uint8)
            for x, frames in zip(xs, streams):
                _, frame = next(frames)
                tile = np.array(_fill(frame, tile_w, tile_h))
                tile[tile_h - 40 : tile_h - 8, 8:200] = (40, 40, 40)
                canvas[top : top + tile_h, x : x + tile_w] = tile
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out


def main() -> int:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", default=None)
    parser.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "senselab-facial-bench"))
    parser.add_argument("--threads", type=int, default=None, help="cap worker CPU threads")
    parser.add_argument("--geometry-fps", type=float, nargs="*", default=[8.0, 16.0])
    parser.add_argument("--affect-fps", type=float, nargs="*", default=[2.0, 4.0])
    parser.add_argument("--device", default=None)
    parser.add_argument("--corpus-hours", type=float, default=27.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from senselab.video.tasks.facial_analysis import detect_gallery_tiles, extract_face_geometry, extract_facial_affect

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    video = Path(args.video) if args.video else build_gallery(workdir)

    rows: List[Dict[str, Any]] = []
    began = time.monotonic()
    layout = detect_gallery_tiles(video)
    rows.append({"component": "tiles", "fps": None, "wall_s": time.monotonic() - began, "load": os.getloadavg()[0]})
    threads = {"threads": args.threads} if args.threads else {}
    for fps in args.geometry_fps:
        load = os.getloadavg()[0]
        table = extract_face_geometry(video, layout=layout, fps=fps, config={"geometry": threads})
        rows.append(
            {"component": "geometry", "fps": fps, **table.timing, "load": load, "load_after": os.getloadavg()[0]}
        )
    for fps in args.affect_fps:
        load = os.getloadavg()[0]
        table = extract_facial_affect(video, layout=layout, fps=fps, device=args.device, config={"affect": threads})
        rows.append(
            {
                "component": "affect",
                "fps": fps,
                **table.timing,
                "device": table.provenance.get("device"),
                "load": load,
                "load_after": os.getloadavg()[0],
            }
        )

    print(f"video: {video}  tiles: {[t.box for t in layout.tiles]}")
    print("| component | fps | tile-frames/s | x real time | hours for %.0f h | load 1m |" % args.corpus_hours)
    print("|---|---|---|---|---|---|")
    for row in rows:
        if row["component"] == "tiles":
            print(f"| tiles | - | - | - | - | {row['load']:.1f} |")
            continue
        x_rt = row["x_realtime"]
        hours = args.corpus_hours / x_rt if x_rt else float("inf")
        tile_fps = row["frames_per_s"] * len(layout.tiles)
        print(
            f"| {row['component']} | {row['fps']:g} | {tile_fps:.1f} | {x_rt:.2f} | {hours:.1f} | "
            f"{row['load']:.1f}->{row['load_after']:.1f} |"
        )
    if args.out:
        Path(args.out).write_text(json.dumps({"video": str(video), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
