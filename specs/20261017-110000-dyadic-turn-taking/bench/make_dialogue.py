"""Synthesize a two-speaker dialogue with macOS `say`, with known per-speaker timings.

Writes, under OUT/<name>/:
  a.wav, b.wav      clean per-speaker channels (16 kHz float32), the "clean stereo" reference
  mix.wav           a + b (+ optional noise), the mono input a Zoom recording would give
  truth.json        ipus [{speaker,start,end,kind}], turns, backchannels, diarization segments

Usage: python make_dialogue.py OUT --seed 0 --turns 40 [--noise-db -45]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
VOICES = {"A": "Samantha", "B": "Daniel"}
SENTENCES = [
    "I went to the market this morning and the line was out the door",
    "honestly I think the second option makes a lot more sense for us",
    "we should probably call them before the weekend",
    "the train was late again so I missed the first half of the meeting",
    "my sister just moved to a small town up north",
    "it rained all afternoon and the basement flooded a little",
    "did you ever finish that book about the lighthouse keeper",
    "I am not sure the numbers add up the way they said",
    "the new cafe on the corner has really good bread",
    "we could take the long road and stop at the lake on the way",
    "he told me the project would take about three more weeks",
    "I tried the recipe twice and it never came out right",
    "the kids want to go camping again this summer",
    "what time does the game start tonight",
    "I keep forgetting to renew my passport",
    "the garden is finally starting to look like something",
    "they changed the schedule without telling anyone",
    "I think we can make it work if we start early",
    "that reminds me of the trip we took years ago",
    "the printer at work broke down for the third time this month",
]
BACKCHANNELS = ["mm hm", "yeah", "right", "okay", "uh huh", "sure"]


def tts(text: str, voice: str) -> np.ndarray:
    """Render ``text`` with a `say` voice to 16 kHz float32, trimmed of leading/trailing silence."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.wav"
        subprocess.run(
            ["say", "-v", voice, "-o", str(path), "--file-format=WAVE", "--data-format=LEF32@16000", text], check=True
        )
        x, sr = sf.read(str(path), dtype="float32")
    assert sr == SR
    if x.ndim > 1:
        x = x.mean(axis=1)
    env = np.abs(x) > 1e-3
    idx = np.flatnonzero(env)
    return x[idx[0] : idx[-1] + 1] if idx.size else x


def build(seed: int, n_turns: int, noise_db: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Lay out turns of 1-3 IPUs, gaps from -0.3 s (overlap) to 1.0 s, backchannels in long turns."""
    rng = random.Random(seed)
    events: list[dict] = []  # {speaker, start, x, kind}
    t = 0.5
    speaker = "A"
    turns = []
    for _ in range(n_turns):
        n_ipu = rng.choice([1, 1, 2, 2, 3])
        turn_start = t
        for k in range(n_ipu):
            x = tts(rng.choice(SENTENCES), VOICES[speaker])
            events.append({"speaker": speaker, "start": t, "x": x, "kind": "ipu"})
            end = t + len(x) / SR
            if k < n_ipu - 1:
                pause = rng.uniform(0.25, 0.9)
                if rng.random() < 0.6:  # listener backchannels inside the pause, sometimes overlapping
                    other = "B" if speaker == "A" else "A"
                    bc = tts(rng.choice(BACKCHANNELS), VOICES[other])
                    bc_start = end + rng.uniform(-0.15, max(0.0, pause - len(bc) / SR))
                    events.append({"speaker": other, "start": bc_start, "x": bc, "kind": "backchannel"})
                t = end + pause
            else:
                t = end
        turns.append({"speaker": speaker, "start": turn_start, "end": t})
        speaker = "B" if speaker == "A" else "A"
        t += rng.choice([rng.uniform(-0.3, 0.0), rng.uniform(0.1, 0.5), rng.uniform(0.2, 1.0)])
    total = max(e["start"] + len(e["x"]) / SR for e in events) + 1.0
    n = int(total * SR)
    chan = {"A": np.zeros(n, np.float32), "B": np.zeros(n, np.float32)}
    for e in events:
        i = int(round(e["start"] * SR))
        chan[e["speaker"]][i : i + len(e["x"])] += e["x"][: n - i]
    peak = max(np.abs(chan["A"]).max(), np.abs(chan["B"]).max())
    for s in chan:
        chan[s] *= 0.5 / peak
    nrng = np.random.default_rng(seed)
    noise = (10 ** (noise_db / 20)) * nrng.standard_normal(n).astype(np.float32)
    ipus = sorted(
        (
            {
                "speaker": e["speaker"],
                "start": round(e["start"], 4),
                "end": round(e["start"] + len(e["x"]) / SR, 4),
                "kind": e["kind"],
            }
            for e in events
        ),
        key=lambda d: d["start"],
    )
    truth = {
        "sampling_rate": SR,
        "duration": total,
        "ipus": ipus,
        "turns": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in tr.items()} for tr in turns],
        "backchannels": [d for d in ipus if d["kind"] == "backchannel"],
    }
    return chan["A"], chan["B"], noise, truth


def main() -> None:
    """CLI."""
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--turns", type=int, default=40)
    p.add_argument("--noise-db", type=float, default=-50.0)
    args = p.parse_args()
    a, b, noise, truth = build(args.seed, args.turns, args.noise_db)
    out = Path(args.out) / f"dialogue_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "a.wav", a, SR, subtype="FLOAT")
    sf.write(out / "b.wav", b, SR, subtype="FLOAT")
    sf.write(out / "mix.wav", a + b + noise, SR, subtype="FLOAT")
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(out, f"{truth['duration']:.1f}s", len(truth["turns"]), "turns", len(truth["backchannels"]), "backchannels")


if __name__ == "__main__":
    main()
