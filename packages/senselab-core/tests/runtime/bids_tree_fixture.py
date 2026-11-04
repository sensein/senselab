"""A generated BIDS tree of short synthetic recordings; no audio is committed."""

import json
from pathlib import Path

import numpy
import soundfile

RATE = 8_000


def write_audio(path: Path, seed: int, frames: int = 800) -> None:
    """Write `frames` samples of seeded noise, mono, to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    noise = numpy.random.default_rng(seed).uniform(-0.5, 0.5, frames).astype(numpy.float32)
    soundfile.write(path, noise, RATE, subtype="FLOAT")


def make_tree(root: Path) -> Path:
    """Build `root` with subjects sub-01 and sub-02 (two sessions for sub-01), participants, sidecars and strays."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "participants.tsv").write_text("participant_id\tage\nsub-01\t30\nsub-02\t41\n", encoding="utf-8")
    (root / "dataset_description.json").write_text(json.dumps({"Name": "synthetic"}), encoding="utf-8")
    (root / "task-read_audio.json").write_text(json.dumps({"Instruction": "read", "Gain": 1}), encoding="utf-8")
    seed = 0
    for sub, ses, run in (("01", "1", "1"), ("01", "1", "2"), ("01", "2", "1"), ("02", "1", "1")):
        base = root / f"sub-{sub}" / f"ses-{ses}" / "audio"
        write_audio(base / f"sub-{sub}_ses-{ses}_task-read_run-{run}_audio.wav", seed)
        seed += 1
    (root / "sub-01" / "ses-1" / "audio" / "sub-01_ses-1_task-read_run-1_audio.json").write_text(
        json.dumps({"Gain": 2}), encoding="utf-8"
    )
    write_audio(root / "stray.wav", 99)
    (root / "sub-01" / "notes.txt").write_text("not audio", encoding="utf-8")
    return root
