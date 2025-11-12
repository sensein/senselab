"""Worker script for running NVIDIA NeMo Sortformer diarization inside Docker.

Usage:
  python nemo_diarization_worker.py \
    --audio /app/input.wav \
    --model nvidia/diar_sortformer_4spk-v1 \
    --device cpu \
    --out /app/out.json
"""

import argparse
import json
import logging  # noqa: E402
import os
import sys

logging.getLogger("nemo_logger").setLevel(logging.ERROR)
os.environ["NEMO_LOG_LEVEL"] = "ERROR"

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
except Exception as e:  # ImportError or other
    # Always emit a JSON object (host reads from file or stdout)
    sys.stdout.write(json.dumps({"error": f"nemo_toolkit not available: {e}"}))
    sys.stdout.flush()
    sys.exit(1)


def _emit(obj: dict, out_path: str) -> None:
    """Write JSON either to a file path or stdout if out_path == '-'."""
    data = json.dumps(obj, separators=(",", ":"))
    if out_path == "-" or not out_path:
        sys.stdout.write(data)
        sys.stdout.flush()
        return
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(data)
    except Exception as err:
        # As a last resort, print a JSON error to stdout
        sys.stdout.write(json.dumps({"error": f"failed to write out file: {err}"}))
        sys.stdout.flush()


def main() -> None:
    """Run NeMo Sortformer diarization inside Docker."""
    parser = argparse.ArgumentParser(description="Run NeMo Sortformer diarization")
    parser.add_argument("--audio", type=str, required=True, help="Path to input WAV file")
    parser.add_argument("--model", type=str, default="nvidia/diar_sortformer_4spk-v1", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device (cuda or cpu)")
    parser.add_argument("--out", type=str, default="-", help="Path to write JSON output ('-' for stdout)")
    args = parser.parse_args()

    audio_path = args.audio
    model_name = args.model
    device = args.device
    out_path = args.out

    if not os.path.exists(audio_path):
        _emit({"error": f"Audio file not found: {audio_path}"}, out_path)
        sys.exit(1)

    try:
        model = SortformerEncLabelModel.from_pretrained(model_name).to(device)
        model.eval()

        # NeMo Sortformer diarize() returns sequence(s) of "start end speaker" strings
        diarization_segments = model.diarize(audio=audio_path)[0]
        results = []
        for seg in diarization_segments:
            parts = seg.strip().split()
            if len(parts) == 3:
                start, end, speaker = parts
                results.append({"start": float(start), "end": float(end), "speaker": str(speaker)})

        _emit({"segments": results}, out_path)
    except Exception as e:
        _emit({"error": str(e)}, out_path)
        sys.exit(1)


if __name__ == "__main__":
    main()
