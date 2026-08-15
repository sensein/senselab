"""NeMo ASR via isolated subprocess venv.

NeMo toolkit has dependency conflicts with the main senselab environment
(pins older transformers). Runs in an isolated subprocess venv managed by uv,
reusing the same venv as NeMo diarization.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

# Reuse the same NeMo venv as diarization — it already has nemo_toolkit[asr]
_NEMO_VENV = "nemo-diarization"
# NOTE on the torch + torchaudio pins below: the version constraint here is
# necessary but not sufficient on newer-CUDA hosts. The shared ``ensure_venv``
# routes the install through the matching PyTorch wheel index
# (``cu128``/``cu126``/``cu124``/``cu121``/``cpu``) — that's what guarantees
# `torch` and `torchaudio` come from the same CUDA toolchain. Do not add a
# backend-local install path that bypasses ``ensure_venv``.
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
# NOTE: Same `lightning` package issue as diarization — see nvidia.py comment.
_NEMO_PYTHON = "3.12"

# Worker script — runs inside the isolated venv
_ASR_WORKER_SCRIPT = r"""
import json
import sys

try:
    import nemo.collections.asr as nemo_asr
    import torch

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_name = args["model_name"]
    device = args["device"]

    # ASRModel.from_pretrained is nemo.core.classes.common.Model.from_pretrained, which
    # takes no revision argument at all: it resolves via try_to_load_from_cache /
    # HfApi.file_exists+hf_hub_download / snapshot_download, none of which is ever passed
    # a revision, so it always targets the mutable "main" ref regardless of what the
    # parent staged. Genuinely unpinnable at the loader; the parent still resolves+stages
    # the SHA so the run manifest and the download-once cache agree on a commit, even
    # though this call can't be pointed at it directly.
    # NeMo auto-selects CTC vs RNNT vs hybrid based on model config
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    all_results = []
    with torch.no_grad():
        # NeMo transcribe accepts a list of file paths
        transcripts = model.transcribe(audio_paths, batch_size=1, verbose=False)

        # NeMo returns different formats depending on model type:
        # - CTC/RNNT models: list of strings or Hypothesis objects
        # - For Hypothesis objects, .text gives the string
        for transcript in transcripts:
            if hasattr(transcript, "text"):
                text = transcript.text
            else:
                text = str(transcript)
            all_results.append({"text": text.strip()})

    print(json.dumps({"results": all_results}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


class NeMoASR:
    """NeMo ASR transcription via isolated subprocess venv.

    NeMo models (e.g., ``nvidia/stt_en_conformer_ctc_large``) run in an
    isolated subprocess venv to avoid dependency conflicts with the main
    senselab environment.

    Supported model families:
        - CTC models (EncDecCTCModel / EncDecCTCModelBPE)
        - RNNT/Transducer models (EncDecRNNTModel / EncDecRNNTBPEModel)
        - Hybrid models

    The worker uses ``ASRModel.from_pretrained()`` which auto-selects the
    correct architecture based on the model config.
    """

    @classmethod
    def transcribe_with_nemo(
        cls,
        audios: List[Audio],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[ScriptLine]:
        """Transcribe audios with NeMo ASR via subprocess venv.

        Args:
            audios: Audio clips to transcribe (mono, correct sample rate).
            model: HF model to use (default: ``nvidia/stt_en_conformer_ctc_large``).
            device: CPU or CUDA.

        Returns:
            One ``ScriptLine`` per input audio with the transcript text.
        """
        if model is None:
            model = HFModel(path_or_uri="nvidia/stt_en_conformer_ctc_large")
        elif model.revision != "main":
            # ASRModel.from_pretrained() has no revision parameter at all (see the
            # worker-script comment above) and always resolves the mutable "main" ref,
            # so a non-default revision here would otherwise be silently ignored.
            logger.warning(
                f"NeMo ASR ignores model.revision (got {model.revision!r}): the upstream "
                "ASRModel.from_pretrained() has no revision parameter and always resolves "
                "against the mutable 'main' ref."
            )
        model_name = str(model.path_or_uri)
        device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]

        venv_dir = ensure_venv(_NEMO_VENV, _NEMO_REQUIREMENTS, python_version=_NEMO_PYTHON)
        python = venv_python(venv_dir)

        with tempfile.TemporaryDirectory(prefix="senselab-nemo-asr-") as tmpdir:
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
                }
            )

            # Stage via the *ref*, not a resolved SHA. This worker loads bare
            # (ASRModel.from_pretrained takes no revision and always resolves the "main" ref
            # inside the offline cache), so its only link to the run's commit is the refs/<ref>
            # pointer that hf_subprocess_env -> resolve_model -> _point_ref_at re-points at the
            # manifest-pinned commit. Passing a SHA here makes _point_ref_at return immediately --
            # its ref argument is already a SHA -- so refs/<ref> is never re-pointed and keeps
            # whatever "main" resolved to when it was last written on this host.
            #
            # The severe form needs no unusual cache state. On node B of a multi-node sweep where
            # the manifest pins sha1 and upstream has since moved to sha2: the HFModel *field*
            # validator runs first and stages live "main" (refs/main = sha2), the *model* validator
            # then sets commit_sha = sha1 from the manifest, this backend stages sha1 by SHA,
            # _point_ref_at no-ops, and the bare worker loads sha2 while provenance records sha1 --
            # the confidently-wrong provenance model_revision.py exists to prevent, in exactly the
            # sweep the manifest was built for. Where no refs/main exists at all (a cache reached
            # only as snapshots/<sha>), the same defect fails loudly instead: the bare offline load
            # has no pointer to resolve and dies under HF_HUB_OFFLINE=1 (reproduced on a cold
            # cache). resolve_model still pins through the run manifest either way, so the staged
            # commit is unchanged; only where refs/<ref> points differs. Matches
            # text_to_speech/qwen_tts.py and revision_pinning_guard_test's
            # LOADER_CANNOT_PIN_SUBPROCESS_FILES invariant.
            env = hf_subprocess_env(model_name, model.revision, base_env=_clean_subprocess_env())
            result = subprocess.run(
                [python, "-c", _ASR_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            output = parse_subprocess_result(result, "NeMo ASR")

            results: List[ScriptLine] = []
            for entry in output.get("results", []):
                results.append(ScriptLine(text=entry.get("text", "")))

            return results
