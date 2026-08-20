"""Render a wideband spectrogram and ask a VLM to reason about it.

The image is produced here rather than supplied, so the figure a claim rests on is reproducible from
the recording. All energy in these recordings sits below 4 kHz, so the view is 0-8 kHz: showing the
empty 8-24 kHz band would spend most of the image on noise floor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

MODEL = "Qwen/Qwen3.8-27B"

PROMPTS = [
    (
        "describe",
        "This is a wideband spectrogram of an audio recording: time in seconds on the x axis, "
        "frequency in Hz on the y axis, brightness is energy. Describe what you see.",
    ),
    (
        "events",
        "From this spectrogram, list every distinct acoustic event you can identify, with its "
        "approximate start and end time in seconds. Say what each one is.",
    ),
    (
        "voicing",
        "Wideband spectrograms show voiced sound as regular vertical striations, one per glottal "
        "pulse. Are there striations here? Over which time ranges, and what fundamental "
        "frequency do they imply? Explain how you read it off the image.",
    ),
    (
        "production",
        "Is the sound in this spectrogram produced by a human vocal tract? If so, classify it: "
        "connected speech, sustained phonation on a vowel, a pitch glide, a cough, a breath, or "
        "something else. Justify the classification from features visible in the image.",
    ),
]


def render(audio: Path, out: Path, fmax: float, win_ms: float) -> Dict[str, Any]:
    """Write a wideband spectrogram of ``audio``.

    Args:
        audio: Recording to render.
        out: Destination PNG.
        fmax: Top of the frequency axis, Hz.
        win_ms: Analysis window in milliseconds.

    Returns:
        A record of what was rendered.
    """
    y, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    x = y.mean(axis=1)
    nfft = max(32, 2 ** round(np.log2(sr * win_ms / 1000)))
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.specgram(x, NFFT=nfft, Fs=sr, noverlap=int(nfft * 0.9), cmap="magma")
    ax.set_ylim(0, fmax)
    ax.set_xlim(0, x.size / sr)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return {
        "png": str(out),
        "sr": sr,
        "seconds": x.size / sr,
        "nfft": nfft,
        "window_ms": nfft / sr * 1000,
        "fmax": fmax,
    }


def main() -> int:
    """Render, then run every prompt against the image.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, default=Path("vlm.json"))
    ap.add_argument("--png", type=Path, default=Path("spectrogram.png"))
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--win-ms", type=float, default=4.0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    meta = render(args.audio, args.png, args.fmax, args.win_ms)
    print(
        f"rendered {meta['png']}: {meta['seconds']:.2f}s at {meta['sr']} Hz, "
        f"{meta['nfft']}-pt = {meta['window_ms']:.1f} ms window, 0-{meta['fmax']:.0f} Hz",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype="auto", device_map="auto")
    model.eval()
    print(f"loaded {MODEL} on {torch.cuda.device_count()} gpu(s)", flush=True)

    image = Image.open(meta["png"]).convert("RGB")
    rows: List[Dict[str, Any]] = []
    for name, prompt in PROMPTS:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        answer = processor.batch_decode(ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
        rows.append({"prompt": name, "question": prompt, "answer": answer})
        print(f"\n=== [{name}]\n{answer}", flush=True)
        args.out.write_text(json.dumps({"model": MODEL, "render": meta, "rows": rows}, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
