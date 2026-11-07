"""BIDS trees for the scale ladder: synthetic tones byte-identical on every host, or LibriSpeech test-clean utterances.

`synthetic(root, n, seed)` writes `n` triangle-wave tones as 16-bit PCM WAV, computed in integers and written by the
standard library, so a seed gives the same bytes on every host; `synthetic(root, 1, seed, seconds=s)` writes one tone
of `s` seconds, the ladder's long signal. `from_librispeech(root, manifest, n)` lays the first `n` utterances of the
fetched test-clean archive into `sub-<speaker>/ses-01/audio/sub-<speaker>_ses-01_task-read_run-<k>_audio.flac`.
"""

import json
import tarfile
import wave
from pathlib import Path

import numpy
from senselab_core.runtime.fetch import DataManifest, default_cache_root, fetch

RATE = 16_000
"""The sampling rate of a synthetic tone."""
SECONDS = (1, 4)
"""The whole seconds a synthetic tone's length is drawn from, both included."""
PERIODS = (16, 400)
"""The samples per period a tone's pitch is drawn from, both included: 1 kHz down to 40 Hz at `RATE`."""
AMPLITUDE = 8192
"""A tone's peak, a quarter of 16-bit full scale."""


def _description(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"Name": name, "BIDSVersion": "1.10.0"}, sort_keys=True)
    (root / "dataset_description.json").write_text(text + "\n", encoding="utf-8")


def _tone(frames: int, period: int) -> bytes:
    """Return `frames` samples of a triangle wave of `period` samples, little-endian 16-bit, by integer arithmetic."""
    phase = numpy.arange(frames, dtype=numpy.int64) % period
    samples = 4 * AMPLITUDE * numpy.abs(2 * phase - period) // (2 * period) - AMPLITUDE
    return samples.astype("<i2").tobytes()


def synthetic(root: Path, n: int, seed: int, *, seconds: int | None = None) -> Path:
    """Write `n` seeded tones as a BIDS tree at `root`, one subject each; `seconds` fixes every tone's length.

    Raises:
        ValueError: `n` is below 1.
    """
    if n < 1:
        raise ValueError(f"a tree holds at least one recording, got {n}")
    _description(root, f"synthetic tones, seed {seed}")
    rng = numpy.random.default_rng(seed)
    lengths = rng.integers(SECONDS[0], SECONDS[1], n, endpoint=True) if seconds is None else numpy.full(n, seconds)
    periods = rng.integers(PERIODS[0], PERIODS[1], n, endpoint=True)
    width = len(str(n))
    for index, (length, period) in enumerate(zip(lengths, periods, strict=True)):
        sub = f"sub-{index:0{width}d}"
        path = root / sub / "ses-01" / "audio" / f"{sub}_ses-01_task-tone_run-1_audio.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(RATE)
            out.writeframes(_tone(int(length) * RATE, int(period)))
    return root


def lay_librispeech(root: Path, archive: Path, n: int) -> Path:
    """Write the first `n` `.flac` utterances of the test-clean `archive`, in archive order, as a BIDS tree at `root`.

    Run `k` is an utterance's place among its speaker's. The bytes are copied unchanged.

    Raises:
        ValueError: the archive holds fewer than `n` utterances, or a member is not a file.
    """
    _description(root, "LibriSpeech test-clean")
    runs: dict[str, int] = {}
    laid = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if laid >= n:
                break
            if not member.name.endswith(".flac"):
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"{member.name} is not a file")
            speaker = Path(member.name).parts[-3]
            runs[speaker] = runs.get(speaker, 0) + 1
            sub = f"sub-{speaker}"
            path = root / sub / "ses-01" / "audio" / f"{sub}_ses-01_task-read_run-{runs[speaker]}_audio.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(handle.read())
            laid += 1
    if laid < n:
        raise ValueError(f"{archive.name} holds {laid} utterances, fewer than {n}")
    return root


def from_librispeech(root: Path, manifest: Path, n: int) -> Path:
    """Fetch the archive `manifest` names (`data-manifest@2.0`) into the data cache and lay `n` utterances at `root`."""
    [archive] = fetch(DataManifest.load(manifest), cache_root=default_cache_root()).values()
    return lay_librispeech(root, archive, n)
