"""NVIDIA Sortformer diarization via isolated subprocess venv.

NeMo toolkit has dependency conflicts with the main senselab environment
(pins older transformers). Runs in an isolated subprocess venv managed by uv.
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
from senselab.utils.data_structures.logging import logger
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

# NeMo venv specification
_NEMO_VENV = "nemo-diarization"
_NEMO_REQUIREMENTS = [
    "nemo_toolkit[asr]",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "pyarrow<18",  # pyarrow 24+ removed PyExtensionType
    "matplotlib",
    "soundfile",
    # NeMo pulls librosa without a floor; pin numba so uv doesn't backtrack
    # librosa -> numba -> llvmlite 0.36.0 (no Python 3.12 support). See qwen.py.
    "numba>=0.60",
]
# NOTE: `lightning` 2.6.2/2.6.3 were pulled from PyPI as malware (quarantined,
# not merely removed) around April 2026. `lightning` itself is still on PyPI —
# 2.6.5 is current — so `lightning>=2.0.1` (NeMo's own pin) resolves cleanly;
# only those two specific releases are unavailable. Verified current as of this
# writing. DiariZen (diarizen.py) is the third venv in this repo depending on
# `lightning` (alongside this one and speech_to_text/nemo.py) — re-check this
# note there too before trusting it stale-dated.
_NEMO_PYTHON = "3.12"

CAPABILITIES = DiarizationCapabilities(
    populates_text=False,
    speaker_label_kind="index",
    labels_stable_across_files=False,  # not measured; False is the conservative default
    # The checkpoint's own name (diar_sortformer_4spk) claimed 4; the seed-17 speaker-ceiling
    # probe confirmed it structurally rather than by name alone: all 20 k=8 sessions predicted
    # exactly "4", never higher, regardless of the true count. Its counting accuracy is a
    # separate, much weaker fact — it never once reports k=1 correctly (0/20 across the probe).
    max_speakers=4,
    max_speakers_evidence="measured: saturates at 4 on 20/20 k=8 sessions (probe seed-17)",
    honors_speaker_hints=False,
)

# Worker script — runs inside the isolated venv
_WORKER_SCRIPT = r"""
import json
import sys
from pathlib import Path

try:
    import nemo.collections.asr as nemo_asr
    import torch

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_name = args["model_name"]
    device = args["device"]
    output_dir = args["output_dir"]

    # SortformerEncLabelModel.from_pretrained is nemo.core.classes.common.Model.from_pretrained,
    # which takes no revision argument at all: it resolves via try_to_load_from_cache /
    # HfApi.file_exists+hf_hub_download / snapshot_download, none of which is ever passed a
    # revision, so it always targets the mutable "main" ref regardless of what the parent
    # staged. Genuinely unpinnable at the loader; the parent still resolves+stages the SHA
    # so the run manifest and the download-once cache agree on a commit, even though this
    # call can't be pointed at it directly.
    model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(model_name)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    all_results = []
    for audio_path in audio_paths:
        with torch.no_grad():
            diar_output = model.diarize(
                audio=audio_path,
                batch_size=1,
                num_workers=0,
                verbose=False,
            )

        # diar_output is List[List[str]] — format: "start end speaker"
        segments = []
        if diar_output and diar_output[0]:
            for line in diar_output[0]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    speaker = parts[2]
                    segments.append({
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                    })
        all_results.append(segments)

    print(json.dumps({"results": all_results}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def diarize_audios_with_nvidia_sortformer(
    audios: List[Audio],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Diarize audios with NVIDIA Sortformer (NeMo) via subprocess venv.

    Args:
        audios: Audio clips to diarize (mono, correct sample rate).
        model: HF model to use (default: "nvidia/diar_sortformer_4spk-v1").
        device: CPU or CUDA.

    Returns:
        One list per input audio with (speaker, start, end), sorted by start time.
    """
    if model is None:
        model = HFModel(path_or_uri="nvidia/diar_sortformer_4spk-v1")
    elif model.revision != "main":
        # SortformerEncLabelModel.from_pretrained() has no revision parameter at all (see
        # the worker-script comment above) and always resolves the mutable "main" ref, so
        # a non-default revision here would otherwise be silently ignored.
        logger.warning(
            f"NVIDIA Sortformer ignores model.revision (got {model.revision!r}): the upstream "
            "SortformerEncLabelModel.from_pretrained() has no revision parameter and always "
            "resolves against the mutable 'main' ref."
        )
    model_name = str(model.path_or_uri)
    device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]

    venv_dir = ensure_venv(_NEMO_VENV, _NEMO_REQUIREMENTS, python_version=_NEMO_PYTHON)
    python = venv_python(venv_dir)

    with tempfile.TemporaryDirectory(prefix="senselab-nemo-") as tmpdir:
        tmp = Path(tmpdir)

        # Serialize audios to WAV
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.wav")
            audio.save_to_file(path)
            audio_paths.append(path)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "model_name": model_name,
                "device": device_type.value,
                "output_dir": str(tmp),
            }
        )

        # Stage via the *ref*, not a resolved SHA. This worker loads bare
        # (SortformerEncLabelModel.from_pretrained takes no revision and always resolves the "main"
        # ref inside the offline cache), so its only link to the run's commit is the refs/<ref>
        # pointer that hf_subprocess_env -> resolve_model -> _point_ref_at re-points at the
        # manifest-pinned commit. Passing a SHA here makes _point_ref_at return immediately -- its
        # ref argument is already a SHA -- so refs/<ref> is never re-pointed and keeps whatever
        # "main" resolved to when it was last written on this host.
        #
        # The severe form needs no unusual cache state. On node B of a multi-node sweep where the
        # manifest pins sha1 and upstream has since moved to sha2: the HFModel *field* validator
        # runs first and stages live "main" (refs/main = sha2), the *model* validator then sets
        # commit_sha = sha1 from the manifest, this backend stages sha1 by SHA, _point_ref_at
        # no-ops, and the bare worker loads sha2 while provenance records sha1 -- the
        # confidently-wrong provenance model_revision.py exists to prevent, in exactly the sweep
        # the manifest was built for. Where no refs/main exists at all (a cache reached only as
        # snapshots/<sha>), the same defect fails loudly instead: the bare offline load has no
        # pointer to resolve and dies under HF_HUB_OFFLINE=1 (reproduced on a cold cache).
        # resolve_model still pins through the run manifest either way, so the staged commit is
        # unchanged; only where refs/<ref> points differs. Matches text_to_speech/qwen_tts.py and
        # the invariant in revision_pinning_guard_test.LOADER_CANNOT_PIN_SUBPROCESS_FILES.
        env = hf_subprocess_env(model_name, model.revision, base_env=_clean_subprocess_env())
        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        output = parse_subprocess_result(result, "NeMo Sortformer")

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
