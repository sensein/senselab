"""Ask an audio LM what it hears, at native bandwidth and at 16 kHz.

The comparison is only meaningful if the model's own preprocessing does not already resample. This
prints what reaches the encoder in both cases, so a null result is distinguishable from a moot one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

PROMPTS = [
    ("open", "What do you hear in this audio? Describe every distinct sound and when it occurs."),
    (
        "vocal",
        "Is any sound in this audio produced by a human vocal tract? For each one, say what it is "
        "and give its start and end time in seconds.",
    ),
    ("airway", "List any breathing, coughing, throat clearing or other airway sounds, with times."),
    (
        "phonation",
        "Is anyone producing a sustained vowel, a pitch glide, or any voiced sound without words? "
        "If so, describe it and give its time range.",
    ),
]


def resample_to(path: Path, out: Path, rate: int) -> Dict[str, Any]:
    """Write ``path`` resampled to ``rate``.

    Args:
        path: Source recording.
        out: Destination path.
        rate: Target sampling rate.

    Returns:
        A record of what was written.
    """
    import librosa

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = y.mean(axis=1)
    if sr != rate:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=rate)
    sf.write(str(out), mono, rate, subtype="FLOAT")
    return {"path": str(out), "rate": rate, "samples": int(mono.size), "seconds": mono.size / rate}


def main() -> int:
    """Run every prompt against both bandwidths and dump the answers.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, default=Path("audiolm.json"))
    ap.add_argument("--work", type=Path, default=Path("."))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    native = sf.info(str(args.audio))
    print(
        f"input {args.audio.name}: {native.samplerate} Hz, {native.frames} frames, "
        f"{native.frames / native.samplerate:.3f}s, {native.channels}ch",
        flush=True,
    )

    variants = {
        "native": resample_to(args.audio, args.work / "native.wav", native.samplerate),
        "16k": resample_to(args.audio, args.work / "res16k.wav", 16000),
    }
    for k, v in variants.items():
        print(f"  variant {k}: {v['rate']} Hz, {v['samples']} samples", flush=True)

    # qwen_omni_utils hardcodes SAMPLE_RATE=16000 and calls librosa.load(sr=SAMPLE_RATE), so both
    # variants reach the encoder at 16 kHz. Kept as a pair so the identical fed sample counts
    # demonstrate that rather than the run asserting it.
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    fe = getattr(processor, "feature_extractor", None)
    fe_rate = getattr(fe, "sampling_rate", None)
    print(f"processor feature_extractor.sampling_rate = {fe_rate}", flush=True)

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL, dtype="auto", device_map="auto", attn_implementation="sdpa"
    )
    model.disable_talker()
    print(f"loaded; device_map spread over {torch.cuda.device_count()} gpu(s)", flush=True)

    rows: List[Dict[str, Any]] = []
    for vname, v in variants.items():
        for pname, prompt in PROMPTS:
            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": v["path"]},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
            # What actually reaches the encoder -- the whole point of the bandwidth comparison.
            fed = [np.asarray(a).size for a in (audios or [])]
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = inputs.to(model.device).to(model.dtype)
            ids, _ = model.generate(
                **inputs, return_audio=False, thinker_return_dict_in_generate=True, max_new_tokens=args.max_new_tokens
            )
            answer = processor.batch_decode(
                ids.sequences[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            rows.append(
                {
                    "variant": vname,
                    "input_rate": v["rate"],
                    "prompt": pname,
                    "samples_fed_to_encoder": fed,
                    "answer": answer,
                }
            )
            print(f"\n=== [{vname} @ {v['rate']} Hz | {pname}] fed={fed}\n{answer}", flush=True)
            args.out.write_text(
                json.dumps(
                    {"model": MODEL, "processor_sampling_rate": fe_rate, "variants": variants, "rows": rows}, indent=2
                )
            )
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
