"""Check that every AudioSet class the branch-1 draft names exists, and show the clean-run peaks."""

from __future__ import annotations

from pathlib import Path

import io_util
import numpy as np

RAW = Path(__file__).parent / "raw"

DRAFT_VOCAB = [
    "Silence",
    "Breathing",
    "Cough",
    "Gasp",
    "Sigh",
    "Throat clearing",
    "Sneeze",
    "Snoring",
    "Speech",
]

data = io_util.load(RAW / "yamnet.json.gz")["clean"]
labels = data["labels"]
scores = np.array(data["scores"])
index = {name: i for i, name in enumerate(labels)}

print("draft vocabulary presence:")
for name in DRAFT_VOCAB:
    print(f"  {name:18s} {'present' if name in index else 'MISSING'}")

print("\ntop-6 classes by peak posterior over the clean run:")
peaks = scores.max(axis=0)
for i in np.argsort(peaks)[::-1][:20]:
    print(f"  {labels[i]:30s} peak={peaks[i]:.4f}  at {data['starts'][int(scores[:, i].argmax())]:.2f}s")

print("\ncandidate mouth/body classes in the class map:")
needles = ["mouth", "lip", "smack", "chew", "click", "tongue", "swallow", "whistl", "hum", "grunt", "moan", "burp"]
for i, name in enumerate(labels):
    if any(n in name.lower() for n in needles):
        print(f"  {name:35s} peak={peaks[i]:.4f}")
