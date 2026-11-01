"""USC-SAIL child-adult speaker-role classification via isolated subprocess venv.

Classifies audio into child / adult / overlap / silence at 20ms frame resolution
(usc-sail/child-adult-diarization, Whisper-base backbone + LoRA head, weights hosted
at ``AlexXu811/whisper-child-adult``). Unlike the other diarization backends, this
labels speaker *role* (child vs. adult), not speaker *identity* — useful for flagging
e.g. a parent prompting/assisting a child mid-recording in pediatric assessments,
which a generic diarizer's speaker-count vote can miss entirely.

The upstream repo has no installable package (no ``setup.py``/``pyproject.toml``,
just a loose ``requirements.txt`` and scripts run in place), so — unlike every other
subprocess-venv backend in senselab, which installs a real pip/git-installable
package into the isolated venv — the worker script itself clones the repo on first
use and adds it to ``sys.path``.

The upstream model also hardcodes ``.cuda()`` calls inside its forward pass
(``models/whisper.py``), with no clean CPU path (the authors' own README says CPU use
requires manually editing their source). Rather than patch upstream code at runtime,
this backend is **CUDA-only**: it raises immediately, before spawning the subprocess,
if CUDA isn't available.

What the ``speaker`` field means here
-------------------------------------
This backend's ``speaker`` values are ``CHILD`` / ``ADULT`` / ``OVERLAP``: a small
fixed vocabulary describing a *role*, and the same label means the same thing in
every file. That is a description of the tool's output, not a judgement about it —
:func:`~senselab.audio.tasks.speaker_diarization.api.capabilities_for` reports it as
``speaker_label_kind="role"`` so a caller can read it programmatically.

Every diarizer emits labels rather than identities; ``SPEAKER_00`` is no more an
identity than ``CHILD`` is. What differs between backends is only the vocabulary and
whether a label carries across files. Reconciling labels from different backends into
one namespace is a separate concern with its own utility, and no backend module
decides it. This one reports what it produced and stops there.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

CAPABILITIES = DiarizationCapabilities(
    populates_text=False,
    speaker_label_kind="role",  # CHILD/ADULT/OVERLAP name a role, not a speaker
    labels_stable_across_files=False,  # not measured; False is the conservative default
    # CHILD and ADULT are the only two *talkers* this backend distinguishes; OVERLAP
    # marks both talking at once, not a third speaker, so it does not raise the count.
    # (The worker does emit a literal "OVERLAP" label value — see the three-way split
    # below — but max_speakers counts distinguishable speakers, not label values.)
    # The seed-17 speaker-ceiling probe confirmed this structurally, not just from its
    # architecture: all 20 k=8 sessions counted exactly 2 speakers, never more.
    max_speakers=2,
    max_speakers_evidence="measured: saturates at 2 on 20/20 k=8 sessions (probe seed-17)",
    honors_speaker_hints=False,
)

# Upstream's WhisperWrapper() (default args) always loads the feature extractor
# from "openai/whisper-tiny" and the backbone from "openai/whisper-base" (the
# LoRA weights filename here is the "whisper-base" checkpoint) — staged
# alongside the LoRA weights repo so the worker can run fully offline.
_CHILD_ADULT_BACKBONE_REPOS = ("openai/whisper-tiny", "openai/whisper-base")

_CHILD_ADULT_VENV = "child-adult-diarization"
_CHILD_ADULT_REQUIREMENTS = [
    "transformers==4.30.2",
    "torch==2.3.0",
    "torchaudio==2.3.0",
    "loralib==0.1.2",
    "numpy==1.24.4",
    "huggingface_hub",
    "soundfile",
]
# The upstream repo's own README: "Python 3.10.9 was used originally and thus
# recommended" for this exact dependency set (transformers==4.30.2, torch==2.3.0) —
# deliberately NOT the "3.12" used by every other subprocess-venv backend in this
# package. Revisit if 3.12 turns out to work fine once this is actually exercised.
_CHILD_ADULT_PYTHON = "3.10"
# torch==2.3.0 only ships wheels on cu121/cu118 — cap the Stage-1 index so a
# CUDA >= 12.4 host doesn't select an index this pin has no wheel on.
_CHILD_ADULT_MAX_CUDA_VERSION = (12, 1)
_CHILD_ADULT_REPO_URL = "https://github.com/usc-sail/child-adult-diarization.git"
_CHILD_ADULT_HF_REPO = "AlexXu811/whisper-child-adult"
_CHILD_ADULT_WEIGHTS_FILENAME = "whisper-base_rank8_pretrained_50k.pt"

# Worker script — runs inside the isolated venv. Clones the (non-packaged) upstream
# repo on first use, downloads the fine-tuned head weights from HF, then reuses
# upstream's own `process_wav_file` (10 s chunking + majority-filter smoothing +
# frame-to-segment conversion + chunk-boundary merging) rather than reimplementing
# that logic here.
_WORKER_SCRIPT = r"""
import json
import subprocess as sp
import sys
from pathlib import Path

try:
    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    repo_url = args["repo_url"]
    hf_repo = args["hf_repo"]
    hf_revision = args["hf_revision"]
    weights_filename = args["weights_filename"]
    repo_dir = Path(args["repo_dir"])

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside the child-adult-diarization venv, but this "
            "backend requires it (the upstream model hardcodes .cuda() calls in its "
            "forward pass with no clean CPU path)."
        )

    repo_marker = repo_dir / "whisper-modeling"
    if not repo_marker.is_dir():
        import fcntl
        import os
        import shutil
        import tempfile as _tempfile

        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        # Clone under an exclusive flock so concurrent jobs sharing the same
        # venv/HOME don't race into the same repo_dir, and clone to a sibling
        # temp dir + atomic os.replace so a clone interrupted mid-way (SIGKILL,
        # the parent's subprocess timeout) never leaves repo_dir non-empty but
        # without whisper-modeling/ — which would wedge the guard above forever.
        with open(str(repo_dir) + ".lock", "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                if not repo_marker.is_dir():
                    if repo_dir.exists():
                        shutil.rmtree(repo_dir, ignore_errors=True)
                    tmp_clone_dir = Path(_tempfile.mkdtemp(prefix=".child-adult-clone-", dir=str(repo_dir.parent)))
                    try:
                        sp.run(["git", "clone", "--depth", "1", repo_url, str(tmp_clone_dir)], check=True)
                    except Exception:
                        shutil.rmtree(tmp_clone_dir, ignore_errors=True)
                        raise
                    if repo_dir.exists():
                        shutil.rmtree(repo_dir, ignore_errors=True)
                    os.replace(tmp_clone_dir, repo_dir)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    sys.path.insert(0, str(repo_dir / "whisper-modeling"))

    from huggingface_hub import hf_hub_download
    from models.whisper import WhisperWrapper
    from scripts.infer_wav_file import process_wav_file

    weights_path = hf_hub_download(repo_id=hf_repo, filename=weights_filename, revision=hf_revision)

    model = WhisperWrapper()
    model.backbone_model.encoder.embed_positions = (
        model.backbone_model.encoder.embed_positions.from_pretrained(model.embed_positions[:500])
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model = model.cuda()
    model.eval()

    all_results = []
    for audio_path in audio_paths:
        child, adult, overlap = process_wav_file(audio_path, model)
        segments = []
        for start, end in child:
            segments.append({"speaker": "CHILD", "start": start, "end": end})
        for start, end in adult:
            segments.append({"speaker": "ADULT", "start": start, "end": end})
        for start, end in overlap:
            segments.append({"speaker": "OVERLAP", "start": start, "end": end})
        segments.sort(key=lambda s: s["start"])
        all_results.append(segments)

    print(json.dumps({"results": all_results}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def diarize_audios_with_child_adult(
    audios: List[Audio],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Classify audios into child / adult / overlap / silence spans (USC-SAIL child-adult).

    Unlike the other diarization backends, ``speaker`` here is a **role** label
    (``"CHILD"``, ``"ADULT"``, or ``"OVERLAP"``; frames classified as silence produce
    no segment) rather than a speaker identity — this model doesn't distinguish
    between multiple children or multiple adults. It's aimed at a narrower question
    than "how many speakers": specifically, "was an adult voice present at all,"
    which matters for e.g. flagging a parent prompting/assisting a child mid-recording
    in pediatric assessments.

    **Requires CUDA.** The upstream model (usc-sail/child-adult-diarization) hardcodes
    ``.cuda()`` calls inside its forward pass (``models/whisper.py``) with no clean CPU
    path — the authors' own README says CPU use requires manually editing their
    source. Rather than patch upstream code at runtime, this raises immediately if
    CUDA isn't available, before spawning the subprocess.

    Audio is chunked into fixed 10-second windows (the model's positional embeddings
    were resized for exactly this length) with frame-level (20ms) predictions
    majority-filtered and merged into segments, reusing upstream's own
    ``process_wav_file`` rather than reimplementing that logic.

    **More audio is dropped than just a trailing partial window.** Upstream's own
    loop advances in 10s steps only while ``start + 10 < length`` (strict ``<``),
    so even clips longer than 10s always drop their final 10s block, e.g. a 20.0s
    clip only analyzes 0-10s, not 0-20s. A clip <= 10s would analyze **zero**
    windows under that same rule — indistinguishable from "no adult present" — so
    this raises ``ValueError`` up front for any such clip rather than returning a
    fabricated empty result (see ``Raises``).

    Args:
        audios (list[Audio]):
            Audio clips to classify. Each must be longer than 10s (see ``Raises``).
        model (HFModel | None):
            Defaults to ``HFModel(path_or_uri="AlexXu811/whisper-child-adult")``.
        device (DeviceType | None):
            Must resolve to CUDA; anything else raises.

    Returns:
        list[list[ScriptLine]]: One list per input audio; each `ScriptLine` carries
        `speaker` (`"CHILD"` / `"ADULT"` / `"OVERLAP"`), `start`, and `end`. An empty
        list means no adult/child speech was detected in an analyzed window.

    Raises:
        RuntimeError: If CUDA is not available/compatible.
        ValueError: If any input clip is <= 10s long (see the chunking caveat
            above) — a fabricated "no speech" result is worse than an explicit
            failure for a backend whose purpose is "was an adult voice present."

    Example (requires a CUDA-capable machine):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> lines = diarize_audios_with_child_adult([a1], device=DeviceType.CUDA)  # doctest: +SKIP
        >>> isinstance(lines[0], list)  # doctest: +SKIP
        True
    """
    if model is None:
        model = HFModel(path_or_uri=_CHILD_ADULT_HF_REPO)

    try:
        # Return value unused: this call exists only for its ValueError when CUDA
        # isn't available, same as diarizen.py's bare call.
        _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA])
    except ValueError as exc:
        raise RuntimeError(
            "USC-SAIL child-adult classifier requires CUDA: the upstream model "
            "(usc-sail/child-adult-diarization) hardcodes `.cuda()` calls inside its "
            "forward pass (models/whisper.py) with no clean CPU path. "
            f"Original error: {exc}"
        ) from exc

    venv_dir = ensure_venv(
        _CHILD_ADULT_VENV,
        _CHILD_ADULT_REQUIREMENTS,
        python_version=_CHILD_ADULT_PYTHON,
        max_cuda_version=_CHILD_ADULT_MAX_CUDA_VERSION,
    )
    python = venv_python(venv_dir)
    repo_dir = Path(venv_dir) / "child-adult-diarization-src"

    with tempfile.TemporaryDirectory(prefix="senselab-child-adult-") as tmpdir:
        tmp = Path(tmpdir)
        audio_paths = []
        for i, audio in enumerate(audios):
            # Upstream's chunking loop advances only while `start + 10 < length`
            # (strict `<`), so a clip <= 10s analyzes zero windows and this backend
            # would otherwise return `[]` — indistinguishable from "no child/adult
            # speech present." For a backend whose purpose is "was an adult voice
            # present," a fabricated "no" is worse than an explicit failure, so this
            # raises instead of silently producing an empty, misleading result.
            duration = audio.waveform.shape[-1] / audio.sampling_rate
            if duration <= 10:
                raise ValueError(
                    f"USC-SAIL child-adult classifier: audio at index {i} is {duration:.2f}s long, "
                    "but upstream's chunking loop only analyzes whole 10s windows under the strict "
                    "`start + 10 < length` rule, so a clip <= 10s produces zero analyzed windows. "
                    "Provide a clip longer than 10s."
                )
            path = str(tmp / f"audio_{i}.wav")
            audio.save_to_file(path)
            audio_paths.append(path)

        # Forward the resolved commit SHA to the worker, never the ref -- it has no senselab
        # install and cannot re-resolve, so a bare ref would load whatever this host's cache
        # resolves it to right now, which can disagree with the rest of a multi-node run.
        # commit_sha is already populated by HFModel's constructor-time resolution; the
        # resolve_revision fallback only matters if that somehow did not happen. Deferred
        # import (not at module top) keeps this monkeypatch-friendly at
        # senselab.utils.model_revision.resolve_revision, matching the rest of the codebase.
        from senselab.utils.model_revision import resolve_revision

        hf_repo = str(model.path_or_uri)
        hf_revision = model.commit_sha or resolve_revision(hf_repo, model.revision)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "repo_url": _CHILD_ADULT_REPO_URL,
                "hf_repo": hf_repo,
                "hf_revision": hf_revision,
                "weights_filename": _CHILD_ADULT_WEIGHTS_FILENAME,
                "repo_dir": str(repo_dir),
            }
        )

        # Stage the LoRA weights repo + the base Whisper feature-extractor/backbone
        # repos WhisperWrapper() loads internally (see _CHILD_ADULT_BACKBONE_REPOS)
        # once (cross-process, via the heartbeat lock) + run the worker offline so
        # from_pretrained/hf_hub_download calls make no per-call Hub version check.
        env = hf_subprocess_env(
            hf_repo,
            hf_revision,
            also=[(repo, "main") for repo in _CHILD_ADULT_BACKBONE_REPOS],
            base_env=_clean_subprocess_env(),
        )
        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
        )

        output = parse_subprocess_result(result, "USC-SAIL child-adult")

        results: List[List[ScriptLine]] = []
        for segments in output.get("results", []):
            script_lines = [
                ScriptLine(
                    speaker=str(seg.get("speaker", "")),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                )
                for seg in segments
            ]
            results.append(sorted(script_lines, key=lambda x: x.start or 0.0))

        return results
