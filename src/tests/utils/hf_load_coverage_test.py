"""Guard test: every HF-Hub model-load site must be caching-reviewed.

Loading a model from the HF Hub under a parallel batch (e.g. a SLURM array)
triggers a per-call Hub HEAD/revision check on every process at once, which the
Hub rate-limits (HTTP 429). ``senselab.utils.dependencies`` provides the fix:

- ``resolve_model`` / ``load_hf_resilient`` — for IN-PROCESS loads: resolve the
  ref to an immutable commit SHA once (download-once via the cross-process
  heartbeat lock) and pin the load to it, so cached files load with no HEAD.
- ``hf_subprocess_env`` — for SUBPROCESS-venv backends: stage the model once in
  the parent, then run the worker with ``HF_HUB_OFFLINE=1``.

This test statically inventories every model-load site in ``src/senselab`` and
fails when a new one appears that is not on the reviewed allowlists below. When
it fails, the fix is:

1. Route the new load through the appropriate helper (see above), then
2. Add its file to ``REVIEWED_INPROCESS`` or ``REVIEWED_SUBPROCESS`` here.

That second step is the point: it forces a human to consciously confirm the
new load is HF-cache-safe. Do not add a file to the allowlist without wiring it
through a helper first (the coverage assertions below check for that).

Known limitations (by design — this is a static, file-granular guard):
- It flags new *files* with loads, not a new raw load added *inside* an
  already-reviewed file (that file already references a helper, so the
  coverage assertion still passes). Review diffs to allowlisted files by hand.
- Subprocess detection assumes the worker script string and its
  ``subprocess.run``/``Popen`` launch live in the same module (true for all
  current backends). A worker factored into a separate file would evade it.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "senselab"

# In-process HF loaders (transformers / sentence-transformers / speechbrain /
# pyannote / huggingface_hub). Matched as attributes (``X.from_pretrained``) or
# bare names (``pipeline(...)``), in call OR argument position — the latter
# catches the wrapped form ``retry_on_transient_error(X.from_hparams, ...)`` and
# ``load_hf_resilient(pipeline, ...)``.
_ATTR_LOADERS = {"from_pretrained", "from_hparams"}
_NAME_LOADERS = {"pipeline", "snapshot_download", "hf_hub_download", "SentenceTransformer", "Inference"}

# Tokens that mark an HF load happening inside a subprocess worker *string*
# (invisible to the AST loader scan above, since it lives in a str constant).
_WORKER_TOKENS = (
    "from_pretrained",
    "pipeline(",
    "snapshot_download",
    "hf_hub_download",
    "GLiNER",
    "SALM",
    "ASRModel",
    "CrisperWhisper",
    "SentenceTransformer",
)

_INPROCESS_HELPERS = {"resolve_model", "load_hf_resilient", "ensure_hf_model"}
_SUBPROCESS_HELPER = "hf_subprocess_env"

# ---------------------------------------------------------------------------
# Reviewed allowlists — the human checkpoint this test enforces.
# ---------------------------------------------------------------------------

# In-process load sites that route through resolve_model / load_hf_resilient.
REVIEWED_INPROCESS = {
    "audio/tasks/classification/huggingface.py",
    "audio/tasks/classification/speech_emotion_recognition/api.py",
    "audio/tasks/forced_alignment/forced_alignment.py",
    "audio/tasks/speaker_diarization/pyannote.py",
    "audio/tasks/speaker_diarization/vibevoice.py",
    "audio/tasks/speaker_embeddings/speechbrain.py",
    "audio/tasks/speech_enhancement/speechbrain.py",
    "audio/tasks/speech_to_text/granite.py",
    "audio/tasks/speech_to_text/huggingface.py",
    "audio/tasks/ssl_embeddings/self_supervised_features.py",
    "audio/tasks/text_to_speech/huggingface.py",
    "text/tasks/embeddings_extraction/huggingface.py",
    "text/tasks/embeddings_extraction/sentence_transformers.py",
    # scene-quality / adaptive workflow (branch-only, #536)
    "audio/workflows/audio_analysis/adaptive/audio_io.py",
    "audio/workflows/audio_analysis/adaptive/backends.py",
}

# Subprocess-venv backends whose worker loads a model; the parent stages it via
# hf_subprocess_env and runs the worker offline.
REVIEWED_SUBPROCESS = {
    "audio/tasks/classification/speech_emotion_recognition/api.py",
    "audio/tasks/speaker_diarization/child_adult.py",
    "audio/tasks/speaker_diarization/diarizen.py",
    "audio/tasks/speaker_diarization/moss.py",
    "audio/tasks/speaker_diarization/nvidia.py",
    "audio/tasks/speech_to_text/canary_qwen.py",
    "audio/tasks/speech_to_text/nemo.py",
    "audio/tasks/speech_to_text/qwen.py",
    "audio/tasks/text_to_speech/qwen_tts.py",
    "text/tasks/pii_detection/subprocess_backend.py",
    # scene-quality / ASR (branch-only, #536)
    "audio/tasks/scene_quality/brouhaha.py",
    "audio/tasks/speech_to_text/crisperwhisper.py",
}

# Files that intentionally do a RAW load and are exempt from the helper check.
# Keep this set tiny and justify every entry.
RAW_LOAD_EXCEPTIONS = {
    # Defines the helpers; snapshot_download here IS the download-once primitive.
    "utils/dependencies.py",
    # Diagnostic CLI probe (`_probe`) that intentionally exercises a raw
    # AutoConfig.from_pretrained to report whether a model is loadable.
    "audio/tasks/classification/speech_emotion_recognition/__main__.py",
    # supported_speakers() resolves the ref to a commit SHA via resolve_revision
    # (the same manifest-backed resolver resolve_model uses internally) and only then
    # calls hf_hub_download(..., revision=<sha>) for a single small file
    # (config.json) -- a full commit hash triggers huggingface_hub's commit-hash
    # shortcut, so a cached file loads with zero network, same as resolve_model's
    # guarantee. It deliberately does not call resolve_model itself: that downloads
    # the *entire* multi-GB checkpoint snapshot, which would defeat the point of this
    # function (enumerate named speakers without paying for the weights).
    "audio/tasks/text_to_speech/qwen_tts.py",
    # DriftSE resolves the ref to a commit SHA via resolve_revision and then calls
    # hf_hub_download(..., revision=<sha>) for the single checkpoint file it reads. Upstream's
    # mirror is 2.4 GB -- two 1.14 GB checkpoint variants plus 1648 demo wavs -- so resolve_model
    # would download 1.3 GB no run reads. Same pinning guarantee as above: a full commit hash takes
    # huggingface_hub's commit-hash shortcut, so a cached file resolves with no network.
    "audio/tasks/speech_enhancement/driftse.py",
}


def _rel(p: Path) -> str:
    return str(p.relative_to(_SRC))


def _iter_src_files() -> list[Path]:
    return [p for p in sorted(_SRC.rglob("*.py")) if "/tests/" not in str(p)]


def _inprocess_load_files() -> dict[str, set[str]]:
    """Map relpath -> {helper names referenced} for files with in-process loads."""
    found: dict[str, set[str]] = {}
    for py in _iter_src_files():
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            continue
        loads = False
        helpers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _ATTR_LOADERS:
                loads = True
            elif isinstance(node, ast.Name):
                if node.id in _NAME_LOADERS:
                    loads = True
                elif node.id in _INPROCESS_HELPERS:
                    helpers.add(node.id)
        if loads:
            found[_rel(py)] = helpers
    return found


def _subprocess_worker_files() -> set[str]:
    """Files that launch a subprocess whose worker string loads an HF model."""
    found: set[str] = set()
    for py in _iter_src_files():
        txt = py.read_text()
        if "subprocess.run(" not in txt and "subprocess.Popen(" not in txt:
            continue
        try:
            tree = ast.parse(txt)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(tok in node.value for tok in _WORKER_TOKENS):
                    found.add(_rel(py))
                    break
    return found


def test_no_unreviewed_inprocess_hf_loads() -> None:
    """Fail if an in-process HF load appears in a file not on the reviewed allowlist."""
    detected = set(_inprocess_load_files())
    allowed = REVIEWED_INPROCESS | RAW_LOAD_EXCEPTIONS
    offenders = sorted(detected - allowed)
    assert not offenders, (
        "New in-process HF model-load site(s) in file(s) not reviewed for HF-cache safety:\n"
        + "\n".join(f"  {f}" for f in offenders)
        + "\n\nRoute the load through senselab.utils.dependencies "
        "(resolve_model + revision=<sha>, or load_hf_resilient), then add the file to "
        "REVIEWED_INPROCESS in this test. See the module docstring."
    )


def test_no_unreviewed_subprocess_hf_loads() -> None:
    """Fail if a subprocess worker loads an HF model from a file not on the allowlist."""
    offenders = sorted(_subprocess_worker_files() - REVIEWED_SUBPROCESS)
    assert not offenders, (
        "New subprocess backend(s) whose worker loads an HF model, not reviewed:\n"
        + "\n".join(f"  {f}" for f in offenders)
        + "\n\nStage the model in the parent via hf_subprocess_env(...) so the worker runs "
        "offline, then add the file to REVIEWED_SUBPROCESS in this test."
    )


def test_reviewed_inprocess_files_route_through_a_helper() -> None:
    """Fail if an allowlisted in-process file stopped referencing an HF-cache helper."""
    detected = _inprocess_load_files()
    missing = sorted(
        f for f in REVIEWED_INPROCESS if f in detected and not detected[f] and f not in RAW_LOAD_EXCEPTIONS
    )
    assert not missing, (
        "Allowlisted in-process file(s) no longer reference an HF-cache helper "
        "(resolve_model / load_hf_resilient) — a raw load may have crept back in:\n"
        + "\n".join(f"  {f}" for f in missing)
    )


def test_reviewed_subprocess_files_use_hf_subprocess_env() -> None:
    """Fail if an allowlisted subprocess file stopped calling hf_subprocess_env."""
    missing = sorted(f for f in REVIEWED_SUBPROCESS if _SUBPROCESS_HELPER not in (_SRC / f).read_text())
    assert not missing, "Allowlisted subprocess file(s) no longer call hf_subprocess_env:\n" + "\n".join(
        f"  {f}" for f in missing
    )


def test_allowlists_have_no_stale_entries() -> None:
    """Keep the allowlists honest: every entry must still be a real load site."""
    inproc = set(_inprocess_load_files())
    subproc = _subprocess_worker_files()
    stale_inproc = sorted((REVIEWED_INPROCESS | RAW_LOAD_EXCEPTIONS) - inproc - {"utils/dependencies.py"})
    stale_subproc = sorted(REVIEWED_SUBPROCESS - subproc)
    assert not stale_inproc, (
        f"Remove stale REVIEWED_INPROCESS/RAW_LOAD_EXCEPTIONS entries (no load found): {stale_inproc}"
    )
    assert not stale_subproc, f"Remove stale REVIEWED_SUBPROCESS entries (no subprocess load found): {stale_subproc}"
