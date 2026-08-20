"""Ask a VLM to detect spans in a spectrogram and say what they contain.

The render mode is the load-bearing choice, not a detail. A previous run gave the model a wideband
image and it concluded there was no sustained vowel because it saw "no clear harmonic stacks" -- on a
recording measuring 87-88 Hz at periodicity 0.93. Wideband suppresses harmonics by construction, and at
a full-file zoom glottal striations span about two pixels, so neither voicing cue was available. The
model read the image correctly; the image could not carry the question. Narrowband is the default here
because harmonics are the cue this model demonstrably reasons with.

The colour scale is the second load-bearing render choice. ``specgram`` normalises over the full data
range, which on this recording is 165 dB (-211 to -46 dB): a noise floor 45 dB below the signal lands
at 61% of the palette and reads as structure. The run at that setting located both events correctly
and then made three claims with no measurable support -- harmonics to 6 kHz where 4-8 kHz holds 0.003%
of the energy, a "broadband sweep to 7.5 kHz" where the actual feature is a rising harmonic fan below
2.5 kHz, and a "click" at 8.38 s where a z=2.0 column of noise floor sits. Clipping to ``dyn_range``
dB below the loudest bin puts the floor at the bottom of the palette.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

MODEL = "Qwen/Qwen3.8-27B"

# Neutral: names the medium and what it may contain, asks for spans and reasoning, and takes no
# position on what is there. An earlier prompt asked the model to "explain how you read it off the
# image", which it answered by trying to infer the expected answer -- it wrote that the question looked
# like it came from an assignment.
PROMPT = (
    "This is a spectrogram of an audio recording that can contain vocal sounds. "
    "Detect the spans, and for each one give its start and end time in seconds and your reasoning "
    "for what it contains."
)

# Narrowband resolves harmonics (~35-40 Hz apart at 25 ms); wideband resolves glottal pulses but needs
# a span short enough for them to occupy more than a pixel or two.
MODES = {"narrowband": 25.0, "wideband": 4.0}


def render(audio: Path, out: Path, mode: str, fmax: float, dyn_range: float) -> Dict[str, Any]:
    """Write a spectrogram in the requested analysis mode.

    Args:
        audio: Recording to render.
        out: Destination PNG.
        mode: ``"narrowband"`` or ``"wideband"``.
        fmax: Top of the frequency axis, Hz.
        dyn_range: dB below the loudest bin to clip the colour scale at; ``<= 0`` leaves
            matplotlib's default full-range normalisation in place.

    Returns:
        What was rendered, including the resolved window in samples and milliseconds.
    """
    y, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    x = y.mean(axis=1)
    nfft = max(32, 2 ** round(np.log2(sr * MODES[mode] / 1000)))
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    spec, _, _, im = ax.specgram(x, NFFT=nfft, Fs=sr, noverlap=int(nfft * 0.9), cmap="magma")
    vmax = float(10 * np.log10(np.maximum(spec, 1e-30)).max())
    if dyn_range > 0:
        im.set_clim(vmax - dyn_range, vmax)
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
        "mode": mode,
        "nfft": nfft,
        "window_ms": nfft / sr * 1000,
        "fmax": fmax,
        "dyn_range": dyn_range,
        "clim": [vmax - dyn_range, vmax] if dyn_range > 0 else list(im.get_clim()),
    }


def main() -> int:
    """Render, ask once, and record the answer with its reasoning and whether it was cut off.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, default=Path("vlm.json"))
    ap.add_argument("--png", type=Path, default=Path("spectrogram.png"))
    ap.add_argument("--mode", choices=sorted(MODES), default="narrowband")
    ap.add_argument("--fmax", type=float, default=8000.0)
    ap.add_argument("--dyn-range", type=float, default=60.0)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    meta = render(args.audio, args.png, args.mode, args.fmax, args.dyn_range)
    print(
        f"rendered {meta['png']}: {meta['seconds']:.2f}s at {meta['sr']} Hz, {meta['mode']}, "
        f"{meta['nfft']}-pt = {meta['window_ms']:.1f} ms, 0-{meta['fmax']:.0f} Hz, "
        f"colour scale {meta['clim'][0]:.1f} to {meta['clim'][1]:.1f} dB",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype="auto", device_map="auto")
    model.eval()
    print(f"loaded {MODEL} on {torch.cuda.device_count()} gpu(s)", flush=True)

    image = Image.open(meta["png"]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False, enable_thinking=not args.no_thinking
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    with torch.no_grad():
        # The card's thinking-mode sampling. Greedy decoding is explicitly not what it asks for.
        ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
        )
    n_new = int(ids.shape[1]) - n_in
    raw = processor.batch_decode(ids[:, n_in:], skip_special_tokens=True)[0].strip()

    thinking, answer = "", raw
    if "</think>" in raw:
        head, _, tail = raw.partition("</think>")
        thinking, answer = head.replace("<think>", "").strip(), tail.strip()

    record = {
        "model": MODEL,
        "render": meta,
        "prompt": PROMPT,
        "new_tokens": n_new,
        "max_new_tokens": args.max_new_tokens,
        "hit_budget": n_new >= args.max_new_tokens,
        "thinking": thinking,
        "answer": answer,
    }
    args.out.write_text(json.dumps(record, indent=2))
    print(
        f"\ntokens generated: {n_new} of {args.max_new_tokens}"
        f"{'  ** HIT BUDGET, OUTPUT IS TRUNCATED **' if record['hit_budget'] else ''}",
        flush=True,
    )
    if thinking:
        print(f"\n--- reasoning ({len(thinking)} chars) ---\n{thinking}", flush=True)
    print(f"\n--- answer ---\n{answer}\n\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
