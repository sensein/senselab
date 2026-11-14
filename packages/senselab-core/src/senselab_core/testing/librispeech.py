"""LibriSpeech test-clean items for real-weights tests and benchmarks, read from the fetched archive."""

import io
import tarfile
from pathlib import Path

import numpy
import soundfile

RATE = 16_000
LONG_FORM_SECONDS = 35
"""The length past which the long-form item ends: more than one 30 s window of a Whisper decode."""


def long_form(archive: Path, seconds: float = LONG_FORM_SECONDS) -> numpy.ndarray:
    """Return the first consecutive utterances of the first chapter of the test-clean `archive` over `seconds`.

    Raises:
        ValueError: if the chapter is shorter than `seconds`, or an utterance is not at `RATE`.
    """
    chapter, parts = None, []
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".flac"):
                continue
            if chapter is None:
                chapter = Path(member.name).parent
            if Path(member.name).parent != chapter:
                break
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"{member.name} is not a file")
            samples, rate = soundfile.read(io.BytesIO(handle.read()), dtype="float32")
            if rate != RATE:
                raise ValueError(f"{member.name} is at {rate} Hz, not {RATE}")
            parts.append(samples)
            if sum(len(p) for p in parts) > seconds * rate:
                return numpy.concatenate(parts)
    raise ValueError(f"the first chapter of test-clean holds less than {seconds} s")
