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
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

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
    weights_filename = args["weights_filename"]
    repo_dir = Path(args["repo_dir"])

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available inside the child-adult-diarization venv, but this "
            "backend requires it (the upstream model hardcodes .cuda() calls in its "
            "forward pass with no clean CPU path)."
        )

    if not (repo_dir / "whisper-modeling").is_dir():
        sp.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)

    sys.path.insert(0, str(repo_dir / "whisper-modeling"))

    from huggingface_hub import hf_hub_download
    from models.whisper import WhisperWrapper
    from scripts.infer_wav_file import process_wav_file

    weights_path = hf_hub_download(repo_id=hf_repo, filename=weights_filename)

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
    ``process_wav_file`` rather than reimplementing that logic. A trailing partial
    window shorter than 10s is dropped by upstream's own chunking loop.

    Args:
        audios (list[Audio]):
            Audio clips to classify.
        model (HFModel | None):
            Defaults to ``HFModel(path_or_uri="AlexXu811/whisper-child-adult")``.
        device (DeviceType | None):
            Must resolve to CUDA; anything else raises.

    Returns:
        list[list[ScriptLine]]: One list per input audio; each `ScriptLine` carries
        `speaker` (`"CHILD"` / `"ADULT"` / `"OVERLAP"`), `start`, and `end`.

    Raises:
        RuntimeError: If CUDA is not available/compatible.

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
        resolved_device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA])
    except ValueError as exc:
        raise RuntimeError(
            "USC-SAIL child-adult classifier requires CUDA: the upstream model "
            "(usc-sail/child-adult-diarization) hardcodes `.cuda()` calls inside its "
            "forward pass (models/whisper.py) with no clean CPU path. "
            f"Original error: {exc}"
        ) from exc

    venv_dir = ensure_venv(_CHILD_ADULT_VENV, _CHILD_ADULT_REQUIREMENTS, python_version=_CHILD_ADULT_PYTHON)
    python = venv_python(venv_dir)
    repo_dir = Path(venv_dir) / "child-adult-diarization-src"

    with tempfile.TemporaryDirectory(prefix="senselab-child-adult-") as tmpdir:
        tmp = Path(tmpdir)
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.wav")
            audio.save_to_file(path)
            audio_paths.append(path)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "repo_url": _CHILD_ADULT_REPO_URL,
                "hf_repo": _CHILD_ADULT_HF_REPO,
                "weights_filename": _CHILD_ADULT_WEIGHTS_FILENAME,
                "repo_dir": str(repo_dir),
            }
        )

        env = _clean_subprocess_env()
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
