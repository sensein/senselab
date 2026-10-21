"""No code outside ``utils/portable_audio_io.py`` may decode an audio file itself.

Every read goes through :func:`~senselab.utils.portable_audio_io.read_audio`,
:func:`~senselab.utils.portable_audio_io.read_audio_blocks` or
:func:`~senselab.utils.portable_audio_io.audio_info`, in-process or -- for a subprocess-venv worker --
through the copy :func:`~senselab.utils.subprocess_venv.stage_portable_audio_io` stages beside it.
The sweep covers ``src/senselab`` and ``scripts/``, and recurses into worker scripts held in string
literals.

A worker that hands a file path to a third-party model API decodes nothing in senselab's own code,
so the sweep cannot see it. Every subprocess-venv backend is therefore classified in exactly one of
the three sets below, so a new one fails until someone decides which it is.

Fixing a failure means calling ``portable_audio_io``, not adding an allowlist entry.
``UNROUTED_AUDIO_READ`` exists for an admission with a reason and ships empty. Video frame decoding
(``torchcodec.decoders.VideoDecoder``) is out of scope and is not swept.

Design and the call sites converted: ``specs/20261010-portable-audio-reads/design.md``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

_SRC = Path(__file__).resolve().parents[2] / "senselab"
_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"

READ_LAYER = "utils/portable_audio_io.py"

# Decoders, as ``(receiver, attribute)``. The receiver is the last name before the attribute, so
# ``scipy.io.wavfile.read`` matches ``("wavfile", "read")``.
_DECODER_CALLS = {
    ("sf", "read"),
    ("sf", "info"),
    ("sf", "blocks"),
    ("sf", "SoundFile"),
    ("soundfile", "read"),
    ("soundfile", "info"),
    ("soundfile", "blocks"),
    ("soundfile", "SoundFile"),
    ("torchaudio", "load"),
    ("torchaudio", "info"),
    ("librosa", "load"),
    ("librosa", "stream"),
    ("wavfile", "read"),
}
_DECODER_NAMES = {"AudioDecoder"}
# ``from <module> import <name>`` spellings that would let a decoder be called bare.
_DECODER_IMPORTS = {
    ("soundfile", "read"),
    ("soundfile", "info"),
    ("soundfile", "blocks"),
    ("soundfile", "SoundFile"),
    ("torchaudio", "load"),
    ("torchaudio", "info"),
    ("librosa", "load"),
    ("scipy.io.wavfile", "read"),
    ("torchcodec.decoders", "AudioDecoder"),
}

# relpath -> why a direct decode is unavoidable there. Empty: every read goes through the layer.
UNROUTED_AUDIO_READ: dict[str, str] = {}

# Subprocess-venv backends, each in exactly one set (relative to ``src/senselab``).
WORKERS_READING_THROUGH_PORTABLE_AUDIO_IO = {
    "audio/tasks/classification/speech_emotion_recognition/api.py",
    "audio/tasks/classification/speech_emotion_recognition/emotion2vec.py",
    "audio/tasks/classification/yamnet.py",
    "audio/tasks/features_extraction/ppg.py",
    "audio/tasks/features_extraction/sparc.py",
    "audio/tasks/health_acoustics/hear.py",
    "audio/tasks/source_separation/unasdiff.py",
    "audio/tasks/speech_enhancement/driftse.py",
    "audio/tasks/ssl_embeddings/s3prl.py",
    "audio/tasks/voice_cloning/coqui.py",
    "audio/tasks/voice_cloning/sparc.py",
    "utils/clearvoice.py",
    "utils/subprocess_venv.py",
}
# The parent writes a WAV with ``Audio.save_to_file`` and the worker passes its path to a
# third-party API that decodes it with the library's own loader. ``vllm_server`` is the server
# form of this: its client sends the caller's audio file as bytes and vLLM decodes it.
WORKERS_PASSING_A_PARENT_WRITTEN_WAV_TO_A_LIBRARY = {
    "audio/tasks/features_extraction/clearvoice_speechscore.py",
    "audio/tasks/scene_quality/brouhaha.py",
    "audio/tasks/speaker_diarization/child_adult.py",
    "audio/tasks/speaker_diarization/diarizen.py",
    "audio/tasks/speaker_diarization/moss.py",
    "audio/tasks/speaker_diarization/nvidia.py",
    "audio/tasks/speech_to_text/canary_qwen.py",
    "audio/tasks/speech_to_text/crisperwhisper.py",
    "audio/tasks/speech_to_text/nemo.py",
    "audio/tasks/speech_to_text/qwen.py",
    "audio/tasks/text_to_speech/coqui.py",
    "utils/vllm_server.py",
}
WORKERS_WITH_NO_AUDIO_INPUT = {
    "video/tasks/scene_description/backends.py",
    "video/tasks/facial_analysis/backends.py",
    "audio/tasks/text_to_speech/qwen_tts.py",
    "text/tasks/pii_detection/redaction_review.py",
    "text/tasks/pii_detection/redaction_review_vllm.py",
    "text/tasks/pii_detection/subprocess_backend.py",
}
# Venv infrastructure: builds, locks or probes venvs and runs no backend of its own.
_VENV_INFRASTRUCTURE = {
    "utils/compatibility_test_runner.py",
    "utils/venv_lock.py",
}
_VENV_MARKERS = ("ensure_venv(", "venv_python(", "call_in_venv(")


def _is_decoder_call(node: ast.AST) -> bool:
    """Whether ``node`` calls an audio decoder directly."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _DECODER_NAMES
    if isinstance(func, ast.Attribute):
        if func.attr in _DECODER_NAMES:
            return True
        receiver = func.value
        name = receiver.id if isinstance(receiver, ast.Name) else getattr(receiver, "attr", None)
        if (name, func.attr) in _DECODER_CALLS:
            return True
        # ``parselmouth.Sound(path)`` decodes with Praat's reader; ``Sound(values, sampling_frequency=...)`` does not.
        if func.attr == "Sound" and name == "parselmouth" and len(node.args) == 1 and not node.keywords:
            return True
    return False


def _is_decoder_import(node: ast.AST) -> bool:
    """Whether ``node`` imports a decoder under a bare name."""
    if not isinstance(node, ast.ImportFrom) or node.module is None:
        return False
    return any((node.module, alias.name) in _DECODER_IMPORTS for alias in node.names)


def _mentions_a_decoder(text: str) -> bool:
    return (
        any(f"{recv}.{attr}(" in text for recv, attr in _DECODER_CALLS)
        or any(f"{name}(" in text for name in _DECODER_NAMES)
        or "parselmouth.Sound(" in text
        or any(f"from {module} import {name}" in text for module, name in _DECODER_IMPORTS)
    )


def _docstrings(tree: ast.AST) -> set:
    """``id()`` of every docstring constant, which is prose and not a worker script."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                found.add(id(first.value))
    return found


def _reads_in_source(source: str, origin: str) -> List[str]:
    """Direct decoder calls and imports in ``source``, recursing into worker-script string literals."""
    found: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [f"{origin}: does not parse, so its reads cannot be checked"]

    prose = _docstrings(tree)
    for node in ast.walk(tree):
        if _is_decoder_call(node) or _is_decoder_import(node):
            found.append(f"{origin}:{node.lineno}")  # type: ignore[attr-defined]
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in prose
            and _mentions_a_decoder(node.value)
        ):
            found.extend(_reads_in_source(node.value, f"{origin}:{node.lineno} (worker script)"))
    return found


def _swept_files() -> List[Tuple[str, Path]]:
    files = [(str(p.relative_to(_SRC)), p) for p in sorted(_SRC.rglob("*.py"))]
    if _SCRIPTS.is_dir():
        files += [(f"scripts/{p.relative_to(_SCRIPTS)}", p) for p in sorted(_SCRIPTS.rglob("*.py"))]
    return files


def _sweep() -> Tuple[List[str], List[str]]:
    """``(inside_layer, outside_layer)`` direct decoder uses across ``src/senselab`` and ``scripts/``."""
    inside: List[str] = []
    outside: List[str] = []
    for relpath, path in _swept_files():
        findings = _reads_in_source(path.read_text(), relpath)
        (inside if relpath == READ_LAYER else outside).extend(findings)
    return inside, outside


def test_no_audio_read_happens_outside_portable_audio_io() -> None:
    """The boundary: one decoder policy, so no caller can read a truncated FLAC as a short recording."""
    inside, outside = _sweep()
    assert inside, "the sweep found no decoder even inside portable_audio_io, so it is checking nothing"

    offenders = [r for r in outside if r.split(" ")[0].split(":")[0] not in UNROUTED_AUDIO_READ]
    assert not offenders, (
        "Audio read(s) outside portable_audio_io:\n"
        + "\n".join(f"  {r}" for r in offenders)
        + "\n\nCall senselab.utils.portable_audio_io.read_audio / audio_info / read_audio_blocks "
        "(in-process), or the staged module via stage_portable_audio_io (subprocess worker)."
    )


def test_the_allowlist_is_current_and_explained() -> None:
    """Every allowlist entry must name a real file, with a reason, that still decodes directly."""
    _, outside = _sweep()
    outside_files = {r.split(" ")[0].split(":")[0] for r in outside}
    for relpath, reason in UNROUTED_AUDIO_READ.items():
        assert reason.strip(), f"UNROUTED_AUDIO_READ entry {relpath} has no stated reason"
        assert relpath in outside_files, f"UNROUTED_AUDIO_READ entry {relpath} no longer decodes directly"


def test_the_detector_finds_each_decoder_and_ignores_the_layer_api() -> None:
    """Pin the detector's verdicts; the caught cases are the pre-migration spellings of real sites."""
    caught = [
        'sf.read(audio_path, dtype="float32")',
        'soundfile.read(path, dtype="float32", always_2d=True)',
        "sf.info(str(p))",
        'sf.SoundFile(stream_source, "r")',
        "torchaudio.load(str(path))",
        "librosa.load(p, sr=16000)",
        "scipy.io.wavfile.read(p)",
        "AudioDecoder(str(path))",
        "torchcodec.decoders.AudioDecoder(p)",
        "parselmouth.Sound(str(path))",
        "from soundfile import read",
        "from torchcodec.decoders import AudioDecoder",
    ]
    ignored = [
        "read_audio(p, always_2d=True)",
        "audio_info(p)",
        "Audio(filepath=p).waveform",
        "parselmouth.Sound(samples, sampling_frequency=16000.0)",
        "from torchcodec.decoders import VideoDecoder",
        "json.load(f)",
        "torch.load(p)",
        "np.load(p)",
    ]
    for source in caught:
        assert _reads_in_source(source, "<caught>"), f"{source} is a direct decode and must be caught"
    for source in ignored:
        assert not _reads_in_source(source, "<ignored>"), f"{source} is not a direct decode"

    worker = 'WORKER = """\nimport soundfile as sf\nx, sr = sf.read(p)\n"""\n'
    assert _reads_in_source(worker, "<worker>") == ["<worker>:1 (worker script):3"]


def _subprocess_backends() -> set:
    return {
        relpath
        for relpath, path in _swept_files()
        if not relpath.startswith("scripts/") and any(marker in path.read_text() for marker in _VENV_MARKERS)
    } - {READ_LAYER}


def test_every_subprocess_backend_is_classified_once() -> None:
    """A worker passing a path to a library is invisible to the sweep, so each must be classified."""
    sets = (
        WORKERS_READING_THROUGH_PORTABLE_AUDIO_IO,
        WORKERS_PASSING_A_PARENT_WRITTEN_WAV_TO_A_LIBRARY,
        WORKERS_WITH_NO_AUDIO_INPUT,
        _VENV_INFRASTRUCTURE,
    )
    classified = set().union(*sets)
    assert sum(len(s) for s in sets) == len(classified), "a backend is classified in more than one set"
    found = _subprocess_backends()
    assert not found - classified, f"unclassified subprocess-venv backend(s): {sorted(found - classified)}"
    assert not classified - found, f"classified file(s) that no longer run a venv: {sorted(classified - found)}"


def test_workers_classified_as_reading_through_the_layer_do() -> None:
    """The first set is a claim the sweep can check: the worker imports the staged module."""
    for relpath in sorted(WORKERS_READING_THROUGH_PORTABLE_AUDIO_IO):
        text = (_SRC / relpath).read_text()
        assert "from portable_audio_io import" in text or "portable_audio_io" in text, relpath
