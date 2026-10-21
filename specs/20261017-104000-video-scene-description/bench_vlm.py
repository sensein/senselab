"""Seconds per frame and peak memory of the MLX Qwen3-VL backend on synthetic video-call frames.

    uv run python specs/20261017-104000-video-scene-description/bench_vlm.py [--long-side 896 672 512] [--repeats 2]

Frames are composed from the repository's public tutorial images: a two-tile gallery, a 2x2 gallery with one
camera-off tile, a speaker view, a single full-frame participant, and a 3x3 gallery. Prints one
JSON line per long side.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from senselab.video.tasks.scene_description.backends import MLXVLMBackend
from senselab.video.tasks.scene_description.describe import describe_frames

REPO = Path(__file__).resolve().parents[2]
IMAGES = [REPO / "tutorial_images" / n for n in ("single_person.jpg", "three_people.jpg", "no_people.jpeg")]
W, H = 1920, 1080
BG = (26, 26, 26)


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    scale = min(w / img.shape[1], h / img.shape[0])
    resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    tile = np.empty((h, w, 3), dtype=np.uint8)
    tile[:] = BG
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return tile


def _camera_off(w: int, h: int, initials: str) -> np.ndarray:
    tile = np.empty((h, w, 3), dtype=np.uint8)
    tile[:] = (35, 35, 35)
    cv2.putText(tile, initials, (w // 2 - 60, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (230, 230, 230), 4)
    return tile


def _gallery(tiles: list, cols: int) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    tw, th = W // cols - 8, H // rows - 8
    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * (th + 8) + 4 : r * (th + 8) + 4 + th, c * (tw + 8) + 4 : c * (tw + 8) + 4 + tw] = (
            t if t.shape[:2] == (th, tw) else _fit(t, tw, th)
        )
    return canvas


def frames() -> list:
    """Five composed call frames."""
    single, three, room = (np.array(Image.open(p).convert("RGB")) for p in IMAGES)
    gallery4 = _gallery([single, three, room, _camera_off(100, 100, "JD")], cols=2)
    speaker = _fit(single, W, H)
    speaker[20:200, W - 340 : W - 20] = _fit(three, 320, 180)
    full = _fit(three, W, H)
    gallery9 = _gallery([single, three, room, single, _camera_off(100, 100, "AB"), three, room, single, three], 3)
    gallery2 = _gallery([single, room], cols=2)
    return [gallery2, gallery4, speaker, full, gallery9]


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-side", type=int, nargs="+", default=[896, 672, 512])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    t0 = time.perf_counter()
    backend = MLXVLMBackend()
    base = frames()
    warm = describe_frames([cv2.resize(base[0], (448, 252))], backend)
    load_s = time.perf_counter() - t0
    for side in args.long_side:
        imgs = [cv2.resize(f, (side, int(round(side * H / W / 2)) * 2), interpolation=cv2.INTER_AREA) for f in base]
        records = []
        wall0 = time.perf_counter()
        for _ in range(args.repeats):
            records += describe_frames(imgs, backend)
        wall = time.perf_counter() - wall0
        secs = [r.seconds for r in records]
        print(
            json.dumps(
                {
                    "long_side": side,
                    "n": len(records),
                    "wall_s_per_frame": wall / len(records),
                    "gen_s_median": statistics.median(secs),
                    "gen_s_max": max(secs),
                    "prompt_tokens_median": statistics.median(r.prompt_tokens for r in records),
                    "generation_tokens_median": statistics.median(r.generation_tokens for r in records),
                    "parse_ok": sum(r.parse_ok for r in records),
                    "repaired": sum(r.repaired for r in records),
                    "layouts": [r.description.layout if r.description else None for r in records[: len(base)]],
                    "n_tiles": [r.description.n_tiles if r.description else None for r in records[: len(base)]],
                    "camera_on": [
                        [t.camera_on for t in r.description.tiles] if r.description else None
                        for r in records[: len(base)]
                    ],
                    "stats": backend.stats(),
                    "startup_s": load_s,
                    "warm_parse_ok": warm[0].parse_ok,
                }
            ),
            flush=True,
        )
    backend.close()


if __name__ == "__main__":
    main()
