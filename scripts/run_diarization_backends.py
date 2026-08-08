"""Run every diarization backend over a set of audio files and print what each produced.

Not a test — a look at real output. The suite proves the dispatch and the guards; this
shows what the four new backends actually say about a recording, side by side with the
incumbent Pyannote so the differences are visible rather than asserted.

Each backend runs in its own try/except on purpose. Three of the five have hard
requirements this machine may not meet (CUDA, a built subprocess venv, a large download),
and a backend that refuses is a result worth seeing, not a reason to abort the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, List, Optional, Sequence

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel

TARGET_SR = 16000

# (label, model_id, kind). `kind` decides which SenselabModel wraps the id — the same
# distinction model_for_task() makes internally.
BACKENDS: Sequence[tuple[str, str, str]] = (
    ("pyannote (incumbent)", "pyannote/speaker-diarization-community-1", "pyannote"),
    ("VibeVoice-ASR-HF", "microsoft/VibeVoice-ASR-HF", "hf"),
    ("MOSS-Transcribe-Diarize", "OpenMOSS-Team/MOSS-Transcribe-Diarize", "hf"),
    ("DiariZen", "BUT-FIT/diarizen-wavlm-large-s80-md", "hf"),
    ("child-adult (roles, CUDA-only)", "AlexXu811/whisper-child-adult", "hf"),
)


def _fmt_line(line: Any) -> str:
    """One ScriptLine as a single readable row."""
    start = getattr(line, "start", None)
    end = getattr(line, "end", None)
    speaker = getattr(line, "speaker", None) or "-"
    text = (getattr(line, "text", None) or "").strip()
    span = f"{start:7.2f}-{end:7.2f}s" if start is not None and end is not None else " " * 17
    row = f"    {span}  {speaker:<12}"
    if text:
        row += f"  {text[:88]}"
    return row


def _summarize(lines: List[Any], max_rows: int) -> None:
    if not lines:
        # An empty result is a real answer from some backends and a bug in others; say so
        # plainly rather than printing nothing and letting it read as success.
        print("    (no segments returned)")
        return
    speakers = sorted({(getattr(ln, "speaker", None) or "-") for ln in lines})
    have_text = sum(1 for ln in lines if (getattr(ln, "text", None) or "").strip())
    print(f"    segments={len(lines)}  speakers={len(speakers)} {speakers}  with_text={have_text}")
    for ln in lines[:max_rows]:
        print(_fmt_line(ln))
    if len(lines) > max_rows:
        print(f"    ... {len(lines) - max_rows} more")


def _field_report(lines: List[Any]) -> dict:
    """Which ScriptLine fields this backend actually populates, and with what.

    This is the evidence harmonisation needs: the five backends nominally share one
    return type, but they disagree about which fields carry meaning and — for
    ``speaker`` — about what the value *denotes*. A field that is always None is a
    field a downstream consumer cannot rely on.
    """
    fields = ("text", "speaker", "start", "end", "chunks")
    report: dict = {f: 0 for f in fields}
    for ln in lines:
        for f in fields:
            v = getattr(ln, f, None)
            if v is not None and (not isinstance(v, str) or v.strip()):
                report[f] += 1
    report["_n"] = len(lines)
    report["_speaker_values"] = sorted({(getattr(ln, "speaker", None) or "") for ln in lines})[:8]
    return report


def run_one(
    label: str,
    model_id: str,
    kind: str,
    audio: Audio,
    device: Optional[DeviceType],
    max_rows: int,
    collector: Optional[dict] = None,
    file_key: str = "",
) -> None:
    print(f"\n  --- {label} ---")
    print(f"      {model_id}")
    model = PyannoteAudioModel(path_or_uri=model_id) if kind == "pyannote" else HFModel(path_or_uri=model_id)
    t0 = time.time()
    try:
        results = diarize_audios(audios=[audio], model=model, device=device)
    except Exception as exc:  # noqa: BLE001 — a refusal is a result we want to see
        elapsed = time.time() - t0
        print(f"    RAISED after {elapsed:6.1f}s: {type(exc).__name__}: {exc}")
        if "--trace" in sys.argv:
            traceback.print_exc()
        if collector is not None:
            collector.setdefault(file_key, {})[label] = {
                "status": "raised",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(elapsed, 2),
            }
        return
    elapsed = time.time() - t0
    lines = results[0] if results else []
    print(f"    OK in {elapsed:6.1f}s")
    _summarize(lines, max_rows)
    if collector is not None:
        collector.setdefault(file_key, {})[label] = {
            "status": "ok",
            "elapsed_s": round(elapsed, 2),
            "fields": _field_report(lines),
            "lines": [
                {
                    "text": getattr(ln, "text", None),
                    "speaker": getattr(ln, "speaker", None),
                    "start": getattr(ln, "start", None),
                    "end": getattr(ln, "end", None),
                    "has_chunks": bool(getattr(ln, "chunks", None)),
                }
                for ln in lines
            ],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+", help="audio files to run every backend over")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--only", nargs="*", default=None, help="substring match on backend label")
    ap.add_argument("--max-rows", type=int, default=12)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json-out", default=None, help="write the full structured dump here")
    args = ap.parse_args()

    collector: dict = {}
    device = {"cuda": DeviceType.CUDA, "cpu": DeviceType.CPU, "mps": DeviceType.MPS}[args.device]

    try:
        import torch

        print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"(torch probe failed: {exc})")

    for path_str in args.audio:
        path = Path(path_str)
        audio = Audio(filepath=str(path))
        original_sr = audio.sampling_rate
        duration = audio.waveform.shape[-1] / original_sr
        if original_sr != TARGET_SR:
            audio = resample_audios([audio], resample_rate=TARGET_SR)[0]

        print("\n" + "=" * 100)
        print(f"FILE: {path.name}")
        print(f"      {duration:.2f}s, {original_sr} Hz -> {TARGET_SR} Hz, {audio.waveform.shape[0]}ch")
        print("=" * 100)

        for label, model_id, kind in BACKENDS:
            if args.only and not any(sub.lower() in label.lower() for sub in args.only):
                continue
            run_one(label, model_id, kind, audio, device, args.max_rows, collector, path.name)

    # Field-occupancy matrix. The point of the whole run for harmonisation purposes:
    # five backends share one return type and disagree about which fields carry meaning.
    print("\n" + "=" * 100)
    print("FIELD OCCUPANCY  (populated / total segments, per backend, across all files)")
    print("=" * 100)
    header = f"{'BACKEND':<34}{'STATUS':<10}{'SEGS':>6}{'text':>8}{'speaker':>9}{'start':>7}{'end':>7}{'chunks':>8}"
    print(header)
    print("-" * len(header))
    for _label, _mid, _kind in BACKENDS:
        rows = [per.get(_label) for per in collector.values() if per.get(_label)]
        if not rows:
            continue
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            print(f"{_label:<34}{'raised':<10}{'-':>6}{'-':>8}{'-':>9}{'-':>7}{'-':>7}{'-':>8}")
            continue
        agg = {k: sum(r["fields"][k] for r in ok) for k in ("text", "speaker", "start", "end", "chunks")}
        n = sum(r["fields"]["_n"] for r in ok)
        print(
            f"{_label:<34}{'ok':<10}{n:>6}"
            f"{agg['text']:>8}{agg['speaker']:>9}{agg['start']:>7}{agg['end']:>7}{agg['chunks']:>8}"
        )
    print("\nspeaker-label vocabularies (what the `speaker` field actually denotes):")
    for _label, _mid, _kind in BACKENDS:
        vals: set = set()
        for per in collector.values():
            r = per.get(_label)
            if r and r["status"] == "ok":
                vals.update(r["fields"]["_speaker_values"])
        if vals:
            print(f"  {_label:<34} {sorted(v for v in vals if v)}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(collector, indent=2))
        print(f"\nstructured dump -> {args.json_out}")

    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
