"""No code outside the audio I/O layer may write an audio file itself.

The defect this guards is not one bad default argument. It is that audio writes happened in
thirteen places through four different APIs, so the range check and the subtype decision had
thirteen chances to be forgotten -- and were, repeatedly and silently. Enforcing an explicit
``subtype=`` at each site would have patched the instances; enforcing the boundary makes the class
of defect structurally unrepeatable.

The layer is two files with one policy: ``audio/data_structures/audio.py`` for in-process callers,
and ``utils/portable_audio_io.py`` for subprocess workers, which cannot import senselab at all and
are handed that file instead. Everything else goes through one of those.

Fixing a failure means routing the write through ``Audio.save_to_file`` (in-process) or
``portable_audio_io.write_audio`` (worker), not adding an allowlist entry. ``UNGUARDED_AUDIO_WRITE``
exists for an admission with a reason, on the ``revision_pinning_guard_test.py`` pattern, and ships
empty.

Reasoning and the guard's pre-fix output: ``specs/20260819-091500-wav-subtype-sweep/design.md``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

# The audio I/O layer, relative to ``src/senselab``. Only these may call a writer directly.
AUDIO_IO_LAYER = {
    "audio/data_structures/audio.py",
    "utils/portable_audio_io.py",
}

# Writers, as ``(receiver, attribute)``. A receiver of "" matches a bare call by that name.
_WRITER_CALLS = {
    ("sf", "write"),
    ("soundfile", "write"),
    ("torchaudio", "save"),
    ("wavfile", "write"),
    ("sf", "SoundFile"),
    ("soundfile", "SoundFile"),
}
_WRITER_NAMES = {"AudioEncoder"}

# relpath -> why a direct write is unavoidable there. Empty: every write goes through the layer.
UNGUARDED_AUDIO_WRITE: dict[str, str] = {}

# The one module allowed to define the subtype constant; everything else imports it.
_CONSTANT_HOME = "utils/portable_audio_io.py"

_STAGER = "stage_portable_audio_io"
_WORKER_IMPORT = "from portable_audio_io import"


def _src_root() -> Path:
    from tests.utils.hf_load_coverage_test import _SRC

    return _SRC


def _is_writer_call(node: ast.AST) -> bool:
    """Whether ``node`` calls one of the audio writers directly."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in _WRITER_NAMES
    if isinstance(func, ast.Attribute):
        receiver = func.value
        if isinstance(receiver, ast.Name) and (receiver.id, func.attr) in _WRITER_CALLS:
            return True
        # scipy.io.wavfile.write -- the receiver is itself an attribute chain.
        if isinstance(receiver, ast.Attribute) and (receiver.attr, func.attr) in _WRITER_CALLS:
            return True
    return False


def _opens_for_reading_only(node: ast.Call) -> bool:
    """Whether a ``SoundFile(...)`` call is a read, which is not a write and not guarded."""
    func = node.func
    if not (isinstance(func, ast.Attribute) and func.attr == "SoundFile"):
        return False
    modes: List[str] = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    modes += [
        k.value.value
        for k in node.keywords
        if k.arg == "mode" and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)
    ]
    return bool(modes) and all("w" not in mode and "x" not in mode for mode in modes)


def _writes_in_source(source: str, origin: str) -> List[str]:
    """Direct writer calls in ``source``, recursing into worker-script string literals.

    Returned as ``file:line`` labels; a worker-script finding carries both the literal's line in
    the parent and the line inside the worker. A worker string that mentions a writer and does not
    parse is reported rather than skipped.
    """
    found: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [f"{origin}: does not parse, so its writes cannot be checked"]

    for node in ast.walk(tree):
        if _is_writer_call(node):
            assert isinstance(node, ast.Call)
            if _opens_for_reading_only(node):
                continue
            found.append(f"{origin}:{node.lineno}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if not any(f"{recv}.{attr}(" in node.value for recv, attr in _WRITER_CALLS):
                if not any(f"{name}(" in node.value for name in _WRITER_NAMES):
                    continue
            found.extend(_writes_in_source(node.value, f"{origin}:{node.lineno} (worker script)"))
    return found


def _sweep() -> Tuple[List[str], List[str]]:
    """``(inside_layer, outside_layer)`` direct writer calls across ``src/senselab``."""
    root = _src_root()
    inside: List[str] = []
    outside: List[str] = []
    for py in sorted(root.rglob("*.py")):
        relpath = str(py.relative_to(root))
        findings = _writes_in_source(py.read_text(), relpath)
        (inside if relpath in AUDIO_IO_LAYER else outside).extend(findings)
    return inside, outside


def test_no_audio_write_happens_outside_the_io_layer() -> None:
    """The boundary. A write anywhere else has its own chance to forget the range check."""
    inside, outside = _sweep()

    assert inside, "the sweep found no writes even inside the layer, so it is checking nothing"

    offenders = [w for w in outside if w.split(" ")[0].split(":")[0] not in UNGUARDED_AUDIO_WRITE]
    assert not offenders, (
        "Audio write(s) outside the I/O layer, where neither the subtype resolution nor the "
        "out-of-range policy applies:\n"
        + "\n".join(f"  {w}" for w in offenders)
        + "\n\nRoute them through Audio.save_to_file (in-process) or portable_audio_io.write_audio "
        "(subprocess worker, via stage_portable_audio_io), or add the file to "
        "UNGUARDED_AUDIO_WRITE in this test with the reason a direct write is unavoidable."
    )


def test_the_allowlist_is_current_and_explained() -> None:
    """Every allowlist entry must name a real file, with a real reason, that still needs it."""
    root = _src_root()
    _, outside = _sweep()
    outside_files = {w.split(" ")[0].split(":")[0] for w in outside}

    for relpath, reason in UNGUARDED_AUDIO_WRITE.items():
        assert (root / relpath).is_file(), f"UNGUARDED_AUDIO_WRITE names a file that no longer exists: {relpath}"
        assert reason.strip(), f"UNGUARDED_AUDIO_WRITE entry {relpath} has no stated reason"
        assert relpath in outside_files, (
            f"UNGUARDED_AUDIO_WRITE entry {relpath} no longer writes directly -- remove it, so the "
            "allowlist keeps meaning 'reviewed and still needed'"
        )


def test_the_layer_files_exist_where_the_guard_expects_them() -> None:
    """A renamed layer file would silently exempt nothing and guard nothing."""
    root = _src_root()
    for relpath in sorted(AUDIO_IO_LAYER | {_CONSTANT_HOME}):
        assert (root / relpath).is_file(), f"the guard names a layer file that does not exist: {relpath}"


def test_the_detector_finds_each_api_and_ignores_reads() -> None:
    """Pin the detector's verdicts; the caught cases are the pre-fix source of the real sites."""
    caught = [
        "sf.write(p, x, sr)",
        'sf.write(p, x, sr, subtype="FLOAT")',  # explicit subtype is no longer enough
        "soundfile.write(p, x, sr)",
        'torchaudio.save(str(path), wf.cpu(), sr, format="flac")',
        "AudioEncoder(samples=w, sample_rate=sr).to_file(p)",
        "scipy.io.wavfile.write(p, sr, x)",
        'sf.SoundFile(p, "w")',
    ]
    ignored = [
        "write_audio(p, x, sr)",
        "audio.save_to_file(p)",
        'sf.read(p, dtype="float32")',
        "sf.info(p)",
        'sf.SoundFile(stream_source, "r")',
        "f.write(chunk)",
        "json.dump(result, f)",
    ]

    for source in caught:
        assert _writes_in_source(source, "<caught>"), f"{source} is a direct audio write and must be caught"
    for source in ignored:
        assert not _writes_in_source(source, "<ignored>"), f"{source} is not a direct audio write"

    worker = 'WORKER = """\nimport soundfile as sf\nsf.write(o, x, sr)\n"""\n'
    assert _writes_in_source(worker, "<worker>") == ["<worker>:1 (worker script):3"]

    unparsable = 'WORKER = "sf.write(o, x, sr"\n'
    found = _writes_in_source(unparsable, "<unparsable>")
    assert len(found) == 1 and "does not parse" in found[0]


def _worker_audio_io_files() -> Tuple[set, set]:
    """``(workers importing the staged module, parents calling the stager)``, by relpath."""
    root = _src_root()
    importers: set = set()
    stagers: set = set()
    for py in sorted(root.rglob("*.py")):
        text = py.read_text()
        relpath = str(py.relative_to(root))
        if _WORKER_IMPORT in text:
            importers.add(relpath)
        # The stager's own definition is not a call to it.
        if f"{_STAGER}(" in text and relpath != "utils/subprocess_venv.py":
            stagers.add(relpath)
        elif relpath == "utils/subprocess_venv.py" and text.count(f"{_STAGER}(") > 1:
            stagers.add(relpath)
    return importers, stagers


def test_every_worker_that_imports_the_policy_gets_it_staged() -> None:
    """A worker importing a module nobody copied fails only at runtime, in a subprocess.

    Both directions: a worker whose parent forgot to stage, and a parent that stages for a worker
    which no longer imports it (dead plumbing that would keep passing the first check forever).
    """
    importers, stagers = _worker_audio_io_files()

    assert importers, "no worker imports the staged policy, so this check is vacuous"
    unstaged = sorted(importers - stagers)
    unused = sorted(stagers - importers)
    assert not unstaged, (
        "Worker script(s) import portable_audio_io but their parent never calls "
        f"{_STAGER}(), so the import fails inside the subprocess:\n" + "\n".join(f"  {f}" for f in unstaged)
    )
    assert not unused, f"Parent(s) stage portable_audio_io for a worker that does not import it: {unused}"


def _subtype_constant_names(tree: ast.AST) -> List[str]:
    """Constant-cased ``*SUBTYPE*`` string assignments, i.e. second definitions of the policy.

    A *string* constant only. A lowercase local is a use of the policy, and a mapping keyed on
    someone else's vocabulary -- ``video/tasks/input_output.py``'s PyAV-codec-hint table -- is a
    translation into it, which belongs with the caller that owns that vocabulary. What must not
    exist twice is a bare name bound to a subtype literal, which is the shape that drifted.
    """
    names: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and "SUBTYPE" in target.id and target.id.isupper():
                names.append(f"{node.lineno} defines {target.id}")
    return names


def test_the_constant_home_check_tells_a_definition_from_a_use() -> None:
    """The pre-fix spellings must be caught; the post-fix ones must not be."""
    caught = ['_WAV_SUBTYPE = "FLOAT"', 'LOSSLESS_WAV_SUBTYPE = "FLOAT"', '_FLAC_SUBTYPE = "PCM_24"']
    ignored = [
        "subtype = resolve_subtype(fmt, dtype, subtype)",
        'wav_subtype = args["wav_subtype"]',
        "from senselab.utils.portable_audio_io import LOSSLESS_WAV_SUBTYPE",
        '_SUBTYPE_FOR_ACODEC = {"pcm_s16le": "PCM_16"}',  # a translation table, not a definition
    ]
    for source in caught:
        assert _subtype_constant_names(ast.parse(source)), f"{source} defines the constant again"
    for source in ignored:
        assert not _subtype_constant_names(ast.parse(source)), f"{source} uses the constant, not redefines it"


def test_the_subtype_constant_has_one_home() -> None:
    """The literal must be defined once; it previously lived under two names in three modules."""
    root = _src_root()
    duplicates: List[str] = []
    for py in sorted(root.rglob("*.py")):
        relpath = str(py.relative_to(root))
        if relpath in {_CONSTANT_HOME, "audio/data_structures/audio.py"}:
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        duplicates.extend(f"{relpath}:{found}" for found in _subtype_constant_names(tree))

    assert not duplicates, (
        f"Subtype constant(s) defined outside {_CONSTANT_HOME}:\n"
        + "\n".join(f"  {d}" for d in duplicates)
        + f"\n\nImport it from senselab.{_CONSTANT_HOME[:-3].replace('/', '.')} instead."
    )


# Formats that carry a float sample beyond +/-1 without clipping it. A subprocess hand-off is an
# internal intermediate, so it must be one of these; ``resolve_subtype`` gives them FLOAT.
_FLOAT_CAPABLE_FORMATS = {"wav", "aiff", "aifc", "w64", "caf", "rf64", "au", "nist", "sd2"}

# Hand-off sites admitted with a reason. Ships empty: a range-limited intermediate has no reason.
RANGE_LIMITED_HANDOFF: dict[str, str] = {}


def _handoff_formats(source: str, origin: str) -> List[str]:
    """Return ``save_to_file`` sites whose literal ``format=`` cannot carry the full range."""
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "save_to_file"):
            continue
        for kw in node.keywords:
            if kw.arg != "format" or not isinstance(kw.value, ast.Constant):
                continue
            fmt = str(kw.value.value).lower()
            if fmt not in _FLOAT_CAPABLE_FORMATS:
                offenders.append(f"{origin}:{node.lineno} (format={fmt!r})")
    return offenders


def test_no_internal_handoff_writes_a_format_that_cannot_carry_the_range() -> None:
    """A hand-off to a subprocess worker must not clip the signal the model then sees.

    FLAC tops out at PCM_24, so a waveform peaking above +/-1 -- which resampling routinely
    produces -- reached eight workers clipped, with the loss invisible to both sides.
    """
    offenders: List[str] = []
    for path in sorted(_src_root().rglob("*.py")):
        origin = path.relative_to(_src_root()).as_posix()
        if origin in RANGE_LIMITED_HANDOFF:
            continue
        offenders.extend(_handoff_formats(path.read_text(), origin))

    assert not offenders, (
        "save_to_file called with a format that clips samples beyond +/-1:\n    "
        + "\n    ".join(offenders)
        + "\nAn internal hand-off should omit ``format`` (or use one of "
        + f"{sorted(_FLOAT_CAPABLE_FORMATS)}) so the worker reads what the caller held."
    )


def test_the_handoff_detector_reads_the_format_argument() -> None:
    """The guard must key on the format, not on the presence of a call."""
    assert _handoff_formats('a.save_to_file(p, format="flac")', "x.py")
    assert _handoff_formats("a.save_to_file(p, format='mp3')", "x.py")
    assert not _handoff_formats('a.save_to_file(p, format="wav")', "x.py")
    assert not _handoff_formats("a.save_to_file(p)", "x.py")
    # A computed format is out of scope: only a literal choice is a decision made here.
    assert not _handoff_formats("a.save_to_file(p, format=fmt)", "x.py")
