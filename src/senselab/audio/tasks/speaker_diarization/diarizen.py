"""DiariZen diarization via isolated subprocess venv.

DiariZen (BUTSpeechFIT, code MIT / **weights CC BY-NC 4.0 — non-commercial
only**) is a WavLM-Conformer EEND + clustering diarization toolkit built on
top of a *forked* pyannote-audio. The default checkpoint
(``BUT-FIT/diarizen-wavlm-large-s80-md``) uses VBx clustering
(``config.toml``'s ``[clustering.args] method = "VBxClustering"``), a
clustering algorithm that only exists in that fork
(``diarizen/pyannote-audio`` on GitHub) and is absent from the ``pyannote.audio``
package published on PyPI — so this backend installs the fork, not the
upstream package, from the same repo via pip's ``#subdirectory=`` VCS syntax.

The upstream ``diarizen`` and forked ``pyannote-audio`` packages declare no
runtime dependencies at all in their packaging metadata (an unmaintained
``setup.cfg`` for the fork; no ``[project.dependencies]`` for ``diarizen``
itself) — upstream's own install recipe relies on a separate
``requirements.txt`` instead. This backend pins that same dependency set
explicitly here rather than depending on a file fetched over the network at
install time.

No ``transformers`` dependency: unlike MOSS or VibeVoice, DiariZen's WavLM
encoder is its own vendored, torchaudio-style implementation
(``diarizen.models.module.wav2vec2``), not ``transformers.WavLMModel``.

Neither the diarization checkpoint nor its internal embedding-model dependency
(``pyannote/wespeaker-voxceleb-resnet34-LM``, downloaded automatically by
``DiariZenPipeline.from_pretrained``) is gated on Hugging Face, so — unlike
the Pyannote backend already in this package — no ``HF_TOKEN`` is required.

``DiariZenPipeline`` hardcodes its device selection internally (CUDA if
available, else CPU) with no override hook, so the ``device`` argument here
only validates the request isn't impossible before spawning the subprocess;
it can't force a specific device inside the pipeline itself.

Not wired into ``audio_analysis``
---------------------------------
This DiariZen backend is reachable through :func:`diarize_audios` and deliberately
**not** through ``scripts/analyze_audio.py --diarization-models``. Two hazard
classes motivate that split: a **role-label** backend, whose ``speaker`` output
names a role (e.g. ``CHILD``/``ADULT``/``OVERLAP``) rather than a speaker
identity, would build a per-role centroid blending distinct speakers under one
label and snap ambiguous frames to whichever centroid is nearest; a
**speaker-identity** backend with its own unreconciled labelling scheme would feed
those labels straight into cross-diarizer agreement and embedding clustering
before they are harmonized against the pass-wide cluster IDs those steps key on,
reading as spurious disagreement against every real diarization model. This
backend falls in the second class — VBx clustering assigns its own per-audio
speaker identities with no reconciliation against the pass-wide cluster IDs, so
wiring it into ``--diarization-models`` as-is would feed unreconciled labels
straight into cross-diarizer consensus and embedding clustering. The guards for
both hazard classes live in
``workflows/audio_analysis/{clustering,identity,presence}.py``, which this branch
does not carry. Port those guards from PR #537 before wiring any of the four new
backends into the workflow.
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

# Embedding model DiariZenPipeline.from_pretrained downloads internally (see
# module docstring) — staged alongside the main checkpoint so both are cached
# before the worker runs offline.
_DIARIZEN_EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

_DIARIZEN_VENV = "diarizen"
_DIARIZEN_REQUIREMENTS = [
    "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen.git",
    "pyannote-audio @ git+https://github.com/BUTSpeechFIT/DiariZen.git#subdirectory=pyannote-audio",
    "torch>=2.1,<2.9",
    "torchaudio>=2.1.1,<2.9",
    # The rest mirror pyannote-audio/requirements.txt in the DiariZen repo (that
    # fork's own packaging declares no install_requires, so these must be pinned
    # here explicitly rather than resolved transitively).
    "asteroid-filterbanks>=0.4",
    "einops>=0.6.0",
    "huggingface_hub>=0.13.0",
    "lightning>=2.0.1",
    "omegaconf>=2.1,<3.0",
    "pyannote.core==5.0.0",
    "pyannote.database==5.1.3",
    "pyannote.metrics==3.2.1",
    "pyannote.pipeline==3.0.1",
    "pytorch_metric_learning>=2.1.0",
    "rich>=12.0.0",
    "semver>=3.0.0",
    "soundfile>=0.12.1",
    "speechbrain>=0.5.14",
    "tensorboardX>=2.6",
    "torch_audiomentations>=0.11.0",
    "torchmetrics>=0.11.0",
    "scipy",
    "toml",
    "psutil",
    "accelerate",
    "numpy<2",
]
_DIARIZEN_PYTHON = "3.12"
_DIARIZEN_DEFAULT_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md"

# Worker script — runs inside the isolated venv.
_WORKER_SCRIPT = r"""
import json
import sys

try:
    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_name = args["model_name"]

    import torch

    # The DiariZen checkpoint was saved under an older PyTorch/Lightning, whose
    # pickled metadata embeds custom objects (torch's own TorchVersion,
    # pyannote-audio's Specifications, ...) beyond plain tensors. PyTorch >=2.6
    # defaults `torch.load` to `weights_only=True`, which rejects any global not
    # explicitly allowlisted, and pyannote-audio's own internal loading code
    # doesn't take a weights_only override we could pass through. Force
    # `weights_only=False` for this isolated subprocess venv only — reasonable
    # here since the checkpoint comes straight from the official BUT-FIT HF
    # repo for the exact backend being loaded, same trust level as this
    # package's own `trust_remote_code=True` usage elsewhere.
    _original_torch_load = torch.load

    def _torch_load_full(*load_args, **load_kwargs):
        load_kwargs["weights_only"] = False
        return _original_torch_load(*load_args, **load_kwargs)

    torch.load = _torch_load_full

    from diarizen.pipelines.inference import DiariZenPipeline

    pipeline = DiariZenPipeline.from_pretrained(model_name)

    all_results = []
    for audio_path in audio_paths:
        annotation = pipeline(audio_path)
        segments = [
            {"speaker": str(speaker), "start": float(turn.start), "end": float(turn.end)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        segments.sort(key=lambda s: s["start"])
        all_results.append(segments)

    print(json.dumps({"results": all_results}))
except Exception as exc:
    import traceback
    err = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=5),
    }
    print(json.dumps({"error": err}))
    sys.exit(1)
"""


def diarize_audios_with_diarizen(
    audios: List[Audio],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Diarize audios with **DiariZen**; returns per-speaker segments per audio.

    DiariZen (``BUT-FIT/diarizen-wavlm-large-s80-md`` by default) is a
    WavLM-Conformer EEND segmentation model followed by VBx clustering, run
    via ``DiariZenPipeline`` (from the ``diarizen`` package and its forked
    ``pyannote-audio`` dependency, both installed in this backend's isolated
    venv). Diarization only — no transcription (unlike MOSS/VibeVoice),
    so returned `ScriptLine`s carry `speaker`/`start`/`end` but no `text`.

    **License note**: the ``diarizen`` and forked ``pyannote-audio`` code is
    MIT, but the pretrained model weights are **CC BY-NC 4.0 — non-commercial
    use only**.

    Args:
        audios (list[Audio]):
            Audio clips to diarize.
        model (HFModel | None):
            Defaults to ``HFModel(path_or_uri="BUT-FIT/diarizen-wavlm-large-s80-md")``.
            Other checkpoints from the same org (e.g. ``BUT-FIT/diarizen-wavlm-large-s80-md-v2``,
            ``BUT-FIT/diarizen-meeting-base``) also work.
        device (DeviceType | None):
            Preferred device. ``DiariZenPipeline`` selects its device
            internally (CUDA if available, else CPU) with no override hook;
            this argument only validates the request isn't impossible before
            spawning the subprocess.

    Returns:
        list[list[ScriptLine]]: One list per input audio; each `ScriptLine` carries
        `speaker` (e.g. `"SPEAKER_00"`), `start`, and `end`.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> lines = diarize_audios_with_diarizen([a1], device=DeviceType.CPU)  # doctest: +SKIP
        >>> len(lines) == 1  # doctest: +SKIP
        True
    """
    if model is None:
        model = HFModel(path_or_uri=_DIARIZEN_DEFAULT_MODEL)
    elif model.revision != "main":
        # DiariZenPipeline.from_pretrained() takes no revision argument at all —
        # it always resolves via a plain snapshot_download() with none pinned —
        # so a non-default revision here would otherwise be silently ignored
        # rather than actually loading the requested snapshot.
        logger.warning(
            f"DiariZen ignores model.revision (got {model.revision!r}): the upstream "
            "DiariZenPipeline.from_pretrained() has no revision parameter and always "
            "resolves the latest snapshot."
        )

    _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])

    venv_dir = ensure_venv(_DIARIZEN_VENV, _DIARIZEN_REQUIREMENTS, python_version=_DIARIZEN_PYTHON)
    python = venv_python(venv_dir)

    with tempfile.TemporaryDirectory(prefix="senselab-diarizen-") as tmpdir:
        tmp = Path(tmpdir)
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.wav")
            audio.save_to_file(path)
            audio_paths.append(path)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "model_name": str(model.path_or_uri),
            }
        )

        # Stage both the main checkpoint and its internal embedding-model
        # dependency once (cross-process, via the heartbeat lock) + run the
        # worker offline so from_pretrained/snapshot_download calls make no
        # per-call Hub version check — the 429 source under parallel batch load.
        env = hf_subprocess_env(
            str(model.path_or_uri), "main", also=[(_DIARIZEN_EMBEDDING_MODEL, "main")], base_env=_clean_subprocess_env()
        )
        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
        )

        output = parse_subprocess_result(result, "DiariZen")

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
