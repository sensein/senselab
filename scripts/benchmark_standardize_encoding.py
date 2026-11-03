# ruff: noqa: E501
"""Benchmark ``standardize_video_encoding`` profiles on a synthetic 720p clip (no data, no network).

``python scripts/benchmark_standardize_encoding.py [--seconds 60] [--ffmpeg PATH]`` prints one table: wall seconds per minute of
video, CPU seconds, size, SSIM against the source for archival (x264 slow), transient (x264 veryfast), x264 ultrafast, NVENC
(when available), and remux. The clip is ``testsrc2`` plus temporal noise, encoded once with B-frames and a long GOP (camera-like).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from senselab.video.tasks.standardize_encoding import standardize_video_encoding
from senselab.video.tasks.standardize_encoding.api import nvenc_available


def ssim(ffmpeg: str, a: Path, b: Path) -> float:
    """Mean SSIM (All) of ``a`` against reference ``b``."""
    done = subprocess.run(
        [ffmpeg, "-nostdin", "-i", str(a), "-i", str(b), "-lavfi", "ssim", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    m = re.search(r"All:([0-9.]+)", done.stderr)
    return float(m.group(1)) if m else float("nan")


def main() -> None:
    """Run the benchmark."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--ffmpeg", default="ffmpeg")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = tmp / "src.mp4"
        subprocess.run(
            [
                args.ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=s=1280x720:r=30:d={args.seconds}",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate=48000:duration={args.seconds}",
                "-vf",
                "noise=alls=12:allf=t",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-g",
                "250",
                "-bf",
                "2",
                "-c:a",
                "aac",
                str(src),
            ],
            check=True,
        )
        std = standardize_video_encoding(src, tmp / "std.mp4", profile="transient", verify=False)
        cases = [
            ("archival x264 slow", dict(profile="archival", allow_copy=False)),
            ("transient x264 veryfast", dict(profile="transient", allow_copy=False)),
            ("x264 ultrafast", dict(profile="transient", preset="ultrafast", allow_copy=False)),
            ("nvenc p4", dict(profile="transient", backend="h264_nvenc", allow_copy=False)),
        ]
        if not nvenc_available(args.ffmpeg):
            cases.pop()
        print(
            f"clip {args.seconds}s 1280x720 30 fps, source {src.stat().st_size / 1e6:.1f} MB; cpus via x264 threads=4"
        )
        print(
            f"{'case':26} {'wall s/min':>10} {'cpu s/min':>10} {'MB':>7} {'ratio':>6} {'SSIM':>7} {'ok':>3} {'mode':>8}"
        )
        for name, kw in cases:
            r = standardize_video_encoding(src, tmp / "o.mp4", **kw)  # type: ignore[arg-type]
            p, k = r.provenance, 60.0 / args.seconds
            print(
                f"{name:26} {p['encode_wall_s'] * k:10.1f} {p['encode_cpu_s'] * k:10.1f} {p['output']['size'] / 1e6:7.1f} "
                f"{p['output']['size_ratio']:6.2f} {ssim(args.ffmpeg, r.output_path, src):7.4f} {int(r.ok):3d} {p['mode']:>8}"
            )
        r = standardize_video_encoding(std.output_path, tmp / "r.mp4")
        p = r.provenance
        print(
            f"{'remux (compliant source)':26} {p['encode_wall_s'] * 60.0 / args.seconds:10.1f} {p['encode_cpu_s'] * 60.0 / args.seconds:10.1f} "
            f"{p['output']['size'] / 1e6:7.1f} {p['output']['size_ratio']:6.2f} {ssim(args.ffmpeg, r.output_path, std.output_path):7.4f} {int(r.ok):3d} {p['mode']:>8}"
        )


if __name__ == "__main__":
    main()
