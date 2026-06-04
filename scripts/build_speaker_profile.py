#!/usr/bin/env python
"""``build_speaker_profile`` — build one reusable speaker profile for a subject.

Thin CLI wrapper over :mod:`senselab.audio.workflows.speaker_profile.build`,
mirroring ``analyze_audio.py`` conventions (same ``--cache-dir`` / ``--device``,
model flags, JSON output). Pools per-window speaker embeddings across all of a
subject's files, clusters them, and persists the dominant cluster's centroid as
a contamination-tolerant profile artifact.

Exit codes:
  0  Profile written (any confidence, including ``insufficient``).
  2  Usage error (no files, missing ``--subject-id`` / ``--output``).
  1  Unrecoverable error (all files unreadable).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.input_output import read_audios
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.build import ProfileInput, build_speaker_profile
from senselab.utils.data_structures import DeviceType

TARGET_SR = 16000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define the CLI surface."""
    parser = argparse.ArgumentParser(
        prog="build_speaker_profile",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="*", type=Path, help="The subject's audio files (≥1, or use --files-from).")
    parser.add_argument(
        "--files-from",
        type=Path,
        default=None,
        help="Newline-delimited file list (optionally 'path\\tsession_id'). Alternative to positional FILEs.",
    )
    parser.add_argument("--subject-id", required=True, help="Subject identifier stamped into the artifact.")
    parser.add_argument("--output", required=True, type=Path, help="Where the profile JSON is written.")
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=list(C.DEFAULT_EMBEDDING_MODELS),
        help="Embedding consensus models (default: ECAPA + ResNet + WavLM). One model → single-model profile.",
    )
    parser.add_argument("--profile-window-s", type=float, default=C.PROFILE_WINDOW_S)
    parser.add_argument("--profile-hop-s", type=float, default=C.PROFILE_HOP_S)
    parser.add_argument("--min-confident-speech-s", type=float, default=C.MIN_CONFIDENT_SPEECH_S)
    parser.add_argument("--target-confident-speech-s", type=float, default=C.TARGET_CONFIDENT_SPEECH_S)
    parser.add_argument("--ambiguity-share-ratio", type=float, default=C.AMBIGUITY_SHARE_RATIO)
    parser.add_argument("--prefer-session", default=None, help="Up-weight windows from this session id.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/analyze_audio_cache"),
        help="Shared content-addressable cache. Reserved for cross-stage reuse with analyze_audio.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable cache lookup/store.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    return parser.parse_args(argv)


def pick_device(arg: str) -> DeviceType | None:
    """Resolve --device into a senselab DeviceType, or None for per-task auto."""
    if arg == "cuda":
        return DeviceType.CUDA
    if arg == "mps":
        return DeviceType.MPS
    if arg == "cpu":
        return DeviceType.CPU
    return None


def _resolve_inputs(args: argparse.Namespace) -> list[tuple[Path, str | None]]:
    """Merge positional files and ``--files-from`` into ``[(path, session_id)]``.

    De-duplicates by path (first occurrence wins) so a file listed both
    positionally and in ``--files-from`` is not ingested twice (which would
    double-count its windows in the dominant-cluster aggregation). If the first
    occurrence carried no session but a later duplicate does, the session is
    backfilled.
    """
    raw_inputs: list[tuple[Path, str | None]] = [(p, None) for p in args.files]
    if args.files_from is not None:
        for raw in args.files_from.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                path_str, session = line.split("\t", 1)
                raw_inputs.append((Path(path_str.strip()), session.strip() or None))
            else:
                raw_inputs.append((Path(line), None))

    out: list[tuple[Path, str | None]] = []
    index_by_path: dict[str, int] = {}
    for path, session in raw_inputs:
        key = str(path)
        if key in index_by_path:
            print(f"warn: duplicate input file ignored: {path}", file=sys.stderr)
            i = index_by_path[key]
            if out[i][1] is None and session is not None:
                out[i] = (out[i][0], session)
            continue
        index_by_path[key] = len(out)
        out.append((path, session))
    return out


def prepare_audio(path: Path) -> Audio:
    """Read audio, downmix to mono, resample to 16 kHz (same as analyze_audio)."""
    audio = read_audios([str(path)])[0]
    audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != TARGET_SR:
        audio = resample_audios([audio], resample_rate=TARGET_SR)[0]
    return audio


def main(argv: list[str] | None = None) -> int:
    """Build a profile from the CLI arguments and write it to ``--output``."""
    args = parse_args(argv)

    requested = _resolve_inputs(args)
    if not requested:
        print("error: no input files (pass FILE ... or --files-from)", file=sys.stderr)
        return 2

    device = pick_device(args.device)

    inputs: list[ProfileInput] = []
    for path, session in requested:
        try:
            audio = prepare_audio(path)
        except Exception as exc:  # noqa: BLE001 — per-file read failures are non-fatal
            print(f"warn: skipping unreadable file {path}: {exc!r}", file=sys.stderr)
            continue
        # ``file_id`` is the user-supplied path string — stable and human-readable.
        inputs.append(ProfileInput(audio=audio, file_id=str(path), session_id=session, pass_summary={}))

    if not inputs:
        print("error: all input files were unreadable", file=sys.stderr)
        return 1

    profile = build_speaker_profile(
        args.subject_id,
        inputs,
        embedding_models=args.embedding_models,
        profile_window_s=args.profile_window_s,
        profile_hop_s=args.profile_hop_s,
        min_confident_speech_s=args.min_confident_speech_s,
        target_confident_speech_s=args.target_confident_speech_s,
        ambiguity_share_ratio=args.ambiguity_share_ratio,
        prefer_session=args.prefer_session,
        device=device,
        output=args.output,
    )

    kept = sum(1 for s in profile.sources if s.kept)
    print(
        f"profile {args.subject_id}: confidence={profile.confidence} "
        f"models={len(profile.centroids)} "
        f"speech={profile.aggregate_speech_seconds:.1f}s "
        f"kept={kept}/{len(profile.sources)} files → {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
