"""ClearerVoice-Studio (ClearVoice) plumbing shared by the four task packages that expose it.

`ClearVoice <https://github.com/modelscope/ClearerVoice-Studio>`_ (Alibaba Speech Lab, Apache-2.0)
is one inference platform over four capabilities and six checkpoints. senselab exposes each
capability from the task package that owns it — enhancement beside SpeechBrain's and DriftSE's
enhancers, separation beside unasdiff's, and two new task packages for the two capabilities that had
no home — exactly as SpeechBrain is exposed across several tasks rather than one.

**This module holds no capability.** There is no enhancer, no separator and no scorer here: what is
here is the machinery all of them share, which is one pip distribution, one isolated venv, one
checkpoint-pinning rule, one device contract, one timeout derivation and one worker. It imports
nothing from ``senselab.audio`` or ``senselab.video`` and speaks only in file paths, so each task
module owns its own ``Audio``/``Video`` conversion and the layering stays one-way.

Capability set (read off ``clearvoice==0.1.2``'s own dispatch table in
``clearvoice/network_wrapper.py``, not off upstream's prose):

===============================  ==========================  =======  ===========================================
Upstream task                    Model                       Rate     senselab home
===============================  ==========================  =======  ===========================================
``speech_enhancement``           ``FRCRN_SE_16K``            16 kHz   ``audio/tasks/speech_enhancement``
``speech_enhancement``           ``MossFormerGAN_SE_16K``    16 kHz   ``audio/tasks/speech_enhancement``
``speech_enhancement``           ``MossFormer2_SE_48K``      48 kHz   ``audio/tasks/speech_enhancement``
``speech_separation``            ``MossFormer2_SS_16K``      16 kHz   ``audio/tasks/source_separation``
``speech_super_resolution``      ``MossFormer2_SR_48K``      48 kHz   ``audio/tasks/speech_super_resolution``
``target_speaker_extraction``    ``AV_MossFormer2_TSE_16K``  16 kHz   ``audio/tasks/target_speaker_extraction``
===============================  ==========================  =======  ===========================================

SpeechScore — upstream's fifth component — is not reachable through this module. It is a separate
directory of the git repository with no pip distribution of its own and a disjoint dependency set, so
it gets its own venv and its own worker; see
``senselab.audio.tasks.features_extraction.clearvoice_speechscore``.

The unpinnable loader
---------------------
``clearvoice/networks.py``'s ``SpeechModel.download_model`` calls
``snapshot_download(repo_id=..., local_dir=...)`` with **no revision argument at all**, so the loader
cannot be pinned through its own interface, and senselab's rule is a commit SHA or nothing
(``CLAUDE.md``; ``src/tests/utils/revision_pinning_guard_test.py``). Verifying the downloaded blobs
after the fact — which an earlier investigation did, matching four sha256 digests against the
``lfs.oid`` at a pinned commit — establishes what ran, but only after an unpinned network read has
already chosen it.

So the loader's download path is never reached. :func:`stage_clearvoice_checkpoints` resolves the ref
to a 40-hex commit through senselab's run-scoped resolver, downloads exactly the files that commit's
``last_best_checkpoint`` manifest names, and the worker points ``args.checkpoint_dir`` at the
resulting commit-addressed snapshot. ``load_model`` then finds ``last_best_checkpoint`` present and
never calls ``download_model`` — and the worker replaces ``download_model`` with a raiser, so a
staging bug surfaces as an error naming the model instead of as a silent unpinned download. The
commit is returned to every caller and recorded in each output's metadata, so a run always records
which weights produced its result.

One file has no revision-addressable home at all: the S3FD face detector the audio-visual pipeline
loads. It is absent from the wheel, and upstream fetches it from Google Drive by file id
(``gdown --id 1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt``) — unversioned, unverified, straight into
site-packages. :func:`stage_s3fd_weights` takes it from a pinned *commit* of the GitHub tree and
verifies its sha256 against a recorded digest, which is stronger than a revision: the bytes are
checked, not the pointer to them.

Design, the measurement behind the timeout term, and the capability→package argument:
``specs/20260819-clearvoice-integration/design.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype, device_run_opt
from senselab.utils.data_structures.logging import logger
from senselab.utils.file_lock import SharedFileLock
from senselab.utils.subprocess_venv import (
    _cache_dir_path,
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    stage_portable_audio_io,
    venv_python,
)

CLEARVOICE_HF_ORG = "alibabasglab"
"""Every ClearVoice checkpoint repository lives under this HuggingFace organisation."""

CLEARVOICE_VENV = "clearvoice"
CLEARVOICE_PYTHON = "3.11"
"""3.11, not 3.12: clearvoice pins ``opencv-python==4.10.0.84`` and ``librosa==0.10.2.post1``, and
3.11 is the interpreter the working reference install for this integration was built on."""

CLEARVOICE_VERSION = "0.1.2"
"""The distribution version this module's monkeypatches are written against.

Asserted inside the worker, because those patches reconstruct ``SpeechModel.__init__``'s
post-conditions field by field: a silent upstream change to that constructor must fail loudly rather
than run against a half-initialised object. ``clearvoice.__version__`` cannot be used for this — 0.1.2
still reports ``"0.1.0"`` — so the check goes through ``importlib.metadata``.
"""

# torch and torchaudio are named explicitly even though clearvoice depends on both, because
# ensure_venv decides whether to route Stage 1 through the CUDA-aware PyTorch wheel index by
# reading *this list*. Omit them and the pair can be resolved against plain PyPI and split across
# mismatched local-version tags. numpy<2 is clearvoice's own pin, repeated for the same reason: it
# constrains which torch wheel can be chosen.
CLEARVOICE_REQUIREMENTS = [
    f"clearvoice=={CLEARVOICE_VERSION}",
    "torch>=2.0.1",
    "torchaudio>=2.0.2",
    "numpy<2.0,>=1.24.3",
]

# ── Timeout terms ─────────────────────────────────────────────────────
#
# Measured once, on the shared development host: FRCRN_SE_16K on CPU decoded 21.48 s of 16 kHz
# speech in 18.6 s inside ``decode()`` (0.87 s per audio-second) after a 2.2 s checkpoint load. That
# is the *cheapest* of the five audio models — a convolutional recurrent net at 16 kHz — and the
# other four are a GAN generator, two 24-layer MossFormer2 stacks, and a MossFormer2 plus a HiFi-GAN
# vocoder at 48 kHz. Rather than invent four more per-model constants that would read as measured,
# one shared term sits an order of magnitude above the single measurement, and every entry point
# takes ``timeout_s`` to override it. This is a ceiling, not a budget: the failure it exists to
# prevent is a fixed 1800 s killing a legitimate long recording (see
# specs/20260818-071500-unasdiff-device-timeout-pcm16), so generosity is the correct bias.
_SECONDS_PER_AUDIO_SECOND = 8.0
_TIMEOUT_HEADROOM = 2.0
# Absorbs the fixed costs a per-second term cannot see: first import of torch inside the venv, and
# up to 734 MB of checkpoint read from a cold cache.
_TIMEOUT_FLOOR_S = 900.0

# The audio-visual pipeline's cost is dominated by per-frame S3FD face detection at 25 fps, plus
# three full ffmpeg passes over the video and one scene-detection pass. Unmeasured on this branch (no
# verified talking-face recording was available here), so the term is deliberately coarse and the
# floor generous; a CUDA host should be roughly an order of magnitude faster.
_TSE_SECONDS_PER_VIDEO_SECOND = 60.0
_TSE_TIMEOUT_FLOOR_S = 1800.0


@dataclass(frozen=True)
class ClearVoiceModelSpec:
    """One ClearVoice checkpoint and everything senselab needs in order to run it.

    Attributes:
        name: Upstream model name. Doubles as the HuggingFace repository name under
            :data:`CLEARVOICE_HF_ORG` and as the key ``network_wrapper`` dispatches on.
        upstream_task: The ``task`` string ``clearvoice.ClearVoice`` takes.
        sampling_rate: Rate the checkpoint was trained at. Inputs are resampled to it on the host,
            and outputs come back at it.
        expected_outputs: How many signals per input this checkpoint is *expected* to produce — 2 for
            the separator, 0 for the audio-visual extractor, whose count is the number of face tracks
            it finds. Used only to check what actually came back: the real count is read from the
            worker's output, never assumed from the model name. Seven SpeechBrain checkpoints
            returned interleaved sources as one signal and were believed on their name for months
            (PR #569), which is why this field is an expectation and not a contract.
        rms_normalises_input: Whether upstream's own reader RMS-normalises the input to -25 dBFS
            before decoding. Set for exactly the two models ``DataReader.extract_feature`` names
            (``FRCRN_SE_16K``, ``MossFormer2_SS_16K``); reproduced rather than corrected, so
            senselab's numbers agree with upstream's own tool for the same checkpoint.
        capability: Human-readable capability name, for log lines and error messages.
    """

    name: str
    upstream_task: str
    sampling_rate: int
    expected_outputs: int
    rms_normalises_input: bool
    capability: str

    @property
    def model_id(self) -> str:
        """The HuggingFace id a caller names this checkpoint by."""
        return f"{CLEARVOICE_HF_ORG}/{self.name}"


CLEARVOICE_MODELS: Dict[str, ClearVoiceModelSpec] = {
    spec.name: spec
    for spec in (
        ClearVoiceModelSpec("FRCRN_SE_16K", "speech_enhancement", 16000, 1, True, "speech enhancement"),
        ClearVoiceModelSpec("MossFormerGAN_SE_16K", "speech_enhancement", 16000, 1, False, "speech enhancement"),
        ClearVoiceModelSpec("MossFormer2_SE_48K", "speech_enhancement", 48000, 1, False, "speech enhancement"),
        ClearVoiceModelSpec("MossFormer2_SS_16K", "speech_separation", 16000, 2, True, "speech separation"),
        ClearVoiceModelSpec("MossFormer2_SR_48K", "speech_super_resolution", 48000, 1, False, "speech super-resolution"),
        ClearVoiceModelSpec(
            "AV_MossFormer2_TSE_16K",
            "target_speaker_extraction",
            16000,
            0,
            False,
            "audio-visual target speaker extraction",
        ),
    )
}

_TASK_OWNERS = {
    "speech_enhancement": "senselab.audio.tasks.speech_enhancement.enhance_audios",
    "speech_separation": "senselab.audio.tasks.source_separation.separate_audios",
    "speech_super_resolution": "senselab.audio.tasks.speech_super_resolution.super_resolve_audios",
    "target_speaker_extraction": (
        "senselab.audio.tasks.target_speaker_extraction.extract_target_speakers_from_videos"
    ),
}


def is_clearvoice_model_id(model_id: str) -> bool:
    """Whether ``model_id`` names a ClearVoice checkpoint.

    Args:
        model_id: A HuggingFace-style model id.

    Returns:
        True for any id under :data:`CLEARVOICE_HF_ORG`. Deliberately org-wide rather than a
        membership test against :data:`CLEARVOICE_MODELS`: a caller naming
        ``alibabasglab/<something else>`` has asked for ClearVoice and should get
        :func:`clearvoice_model_spec`'s message enumerating the six checkpoints, not a bare "no
        backend for this model" from a dispatcher that declined to recognise it.
    """
    return model_id.startswith(f"{CLEARVOICE_HF_ORG}/")


def clearvoice_model_spec(model_id: str, *, expected_task: Optional[str] = None) -> ClearVoiceModelSpec:
    """Resolve a model id (or bare upstream model name) to its spec.

    Args:
        model_id: ``"alibabasglab/FRCRN_SE_16K"`` or ``"FRCRN_SE_16K"``.
        expected_task: If given, the upstream task the calling task package owns. A checkpoint for a
            different capability is rejected here rather than run by the wrong entry point — which
            would otherwise hand two separated sources to a caller who asked to enhance one signal.

    Returns:
        The matching :class:`ClearVoiceModelSpec`.

    Raises:
        ValueError: If the name is not one of the six checkpoints, or names a different capability
            than ``expected_task``.
    """
    name = model_id.split("/")[-1]
    spec = CLEARVOICE_MODELS.get(name)
    if spec is None:
        raise ValueError(
            f"{model_id!r} is not a ClearVoice checkpoint. clearvoice=={CLEARVOICE_VERSION} ships "
            f"exactly six: {', '.join(sorted(CLEARVOICE_MODELS))}."
        )
    if expected_task is not None and spec.upstream_task != expected_task:
        raise ValueError(
            f"{spec.model_id!r} is a {spec.capability} checkpoint, not a "
            f"{expected_task.replace('_', ' ')} one. Call {_TASK_OWNERS[spec.upstream_task]} for it "
            "instead."
        )
    return spec


def clearvoice_models_for_task(upstream_task: str) -> List[ClearVoiceModelSpec]:
    """Return every ClearVoice checkpoint for one upstream task, in table order.

    Args:
        upstream_task: One of the four ``task`` strings ``clearvoice.ClearVoice`` accepts.

    Returns:
        The matching specs.
    """
    return [spec for spec in CLEARVOICE_MODELS.values() if spec.upstream_task == upstream_task]


# ── Checkpoint staging: the pinning mechanism ─────────────────────────


def stage_clearvoice_checkpoints(spec: ClearVoiceModelSpec, revision: str = "main") -> Tuple[Path, str]:
    """Download one checkpoint set at a resolved commit, returning its directory and that commit.

    File by file rather than a whole-repository snapshot, because ``MossFormer2_SR_48K`` carries a
    1.74 GB optimizer state (``do_03925000``) that no inference run reads. Which files to take is not
    guessed: the commit's own ``last_best_checkpoint`` manifest names them, and ``hf_hub_download``
    places every file of one repo/commit as siblings inside a single ``snapshots/<sha>/`` directory —
    which is therefore exactly the ``checkpoint_dir`` upstream's loader wants.

    Args:
        spec: The checkpoint to stage.
        revision: Ref or commit to resolve. Resolution goes through senselab's run-scoped resolver,
            so every task of one sweep binds to the same commit.

    Returns:
        ``(checkpoint_dir, commit_sha)`` — a directory holding ``last_best_checkpoint`` and the
        weight files it names, and the 40-hex commit those bytes came from.
    """
    from huggingface_hub import hf_hub_download

    from senselab.utils.model_revision import resolve_revision

    sha = resolve_revision(spec.model_id, revision)
    manifest = Path(hf_hub_download(spec.model_id, "last_best_checkpoint", revision=sha))
    # One name per line: MossFormer2_SR_48K's manifest names two files (the MossFormer stage and the
    # vocoder), which is why this reads every non-empty line rather than only the first.
    for filename in (line.strip() for line in manifest.read_text().splitlines()):
        if filename:
            hf_hub_download(spec.model_id, filename, revision=sha)
    return manifest.parent, sha


# The S3FD face detector the audio-visual pipeline loads: absent from the clearvoice wheel, with
# upstream falling back to an unversioned Google Drive id. Pinned to a commit of the GitHub tree
# *and* verified by digest, because a raw-URL fetch carries no revision guarantee once followed.
_S3FD_COMMIT = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61"
_S3FD_PATH_IN_REPO = "clearvoice/clearvoice/models/av_mossformer2_tse/faceDetector/s3fd/sfd_face.pth"
_S3FD_URL = f"https://raw.githubusercontent.com/modelscope/ClearerVoice-Studio/{_S3FD_COMMIT}/{_S3FD_PATH_IN_REPO}"
# Verified against the bytes at _S3FD_COMMIT (89,844,381 bytes). A mismatch is fatal: this is the one
# weight in the stack with no revision-addressable home, so its digest is its identity.
_S3FD_SHA256 = "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491"
_S3FD_SIZE_BYTES = 89844381


def stage_s3fd_weights() -> Path:
    """Fetch and verify the S3FD face-detector weights, returning the local path.

    Cached under ``~/.cache/senselab/clearvoice/s3fd/<sha256>/`` — content-addressed, so changing the
    digest constant is a cache miss rather than a stale hit. The download happens under a lock so
    concurrent jobs sharing one cache cannot race, and lands via a temporary file plus ``os.replace``
    so an interrupted fetch never leaves a short file that looks staged.

    Returns:
        Path to ``sfd_face.pth``.

    Raises:
        RuntimeError: If the downloaded bytes do not match :data:`_S3FD_SHA256`. Upstream's own path
            here is an unversioned Drive id with no digest at all, which is exactly why a mismatch is
            refused: a face detector that silently changed would move every face track, and the
            extraction would report success.
    """
    root = _cache_dir_path() / "clearvoice" / "s3fd" / _S3FD_SHA256
    weights = root / "sfd_face.pth"
    if weights.is_file():
        return weights

    root.mkdir(parents=True, exist_ok=True)
    with SharedFileLock(root, timeout=600):
        if weights.is_file():
            return weights
        logger.info(f"ClearVoice: fetching S3FD face-detector weights at commit {_S3FD_COMMIT[:12]}")
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=str(root), delete=False, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(_S3FD_URL) as response:  # noqa: S310 -- constant https URL
                while chunk := response.read(1 << 20):
                    digest.update(chunk)
                    tmp.write(chunk)
        actual = digest.hexdigest()
        if actual != _S3FD_SHA256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"S3FD weights fetched at commit {_S3FD_COMMIT} hashed {actual}, expected "
                f"{_S3FD_SHA256}. This file has no revision-addressable home upstream, so the digest "
                "is its only identity; refusing it rather than running an unknown face detector."
            )
        os.replace(tmp_path, weights)
    return weights


# ── Worker ────────────────────────────────────────────────────────────

# Runs inside the isolated venv. Two modes: "audio" for the five audio-only checkpoints, "tse" for
# the audio-visual one, which has no tensor-in/tensor-out path of its own.
#
# It reuses upstream's own model construction (``ClearVoice``), its own reader normalisation
# (``audio_norm``) and its own segmented decoders (``SpeechModel.decode`` -> ``decode_one_audio``),
# and replaces exactly three things:
#
#   1. ``SpeechModel.download_model`` -> a raiser. The unpinned ``snapshot_download`` must be
#      unreachable, not merely unused.
#   2. ``SpeechModel.__init__``'s device auto-detection -> the caller's device. Upstream selects MPS
#      whenever it is present, and otherwise polls ``nvidia-smi`` for the freest card — so it
#      discards the caller's choice on a multi-GPU host and silently picks an untested backend on
#      macOS.
#   3. File I/O. Upstream reads inputs through pydub, whose integer sample array it rescales by a
#      heuristic (``if max(samples) > 32768: /2**31 else: /2**15``) that mis-scales a quiet 32-bit
#      file by 65536x, and it writes outputs through pydub as 16-bit by default. senselab's staged
#      ``portable_audio_io`` is used on both sides instead, so the range policy and the subtype
#      decision are the same ones the in-process writer applies.
#
# For "tse" only, ``video_process.visualization`` is replaced too — see the comment at that patch.
_WORKER_SCRIPT = r"""
import json
import os
import sys
from pathlib import Path

try:
    args = json.loads(sys.stdin.read())
    sys.path.insert(0, args["io_dir"])
    from portable_audio_io import read_audio, write_audio

    staging = Path(args["staging_dir"])
    model_name = args["model_name"]

    # Upstream's inference configs give checkpoint_dir as the *relative* path "checkpoints/<MODEL>",
    # so a staging root with that layout is what points the loader at the commit-addressed snapshot.
    # A symlink, not a copy: the snapshot is up to 734 MB.
    (staging / "checkpoints").mkdir(parents=True, exist_ok=True)
    link = staging / "checkpoints" / model_name
    if not link.exists():
        os.symlink(args["checkpoint_dir"], link, target_is_directory=True)
    os.chdir(staging)

    from importlib.metadata import version as _dist_version

    installed = _dist_version("clearvoice")
    if installed != args["expected_version"]:
        raise RuntimeError(
            "clearvoice %s is installed but this worker's patches were written against %s"
            % (installed, args["expected_version"])
        )

    import numpy as np
    import torch

    import clearvoice.networks as cvnet

    def _blocked_download(self, name):
        raise RuntimeError(
            "clearvoice tried to fetch %s through its own unpinned snapshot_download, which means "
            "the staged checkpoint directory %s has no 'last_best_checkpoint'. Refusing: a download "
            "with no revision cannot be attributed to a commit."
            % (name, args["checkpoint_dir"])
        )

    cvnet.SpeechModel.download_model = _blocked_download

    requested = args["device"]
    if requested is None:
        requested = "cuda:%d" % torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "device %r was requested but torch.cuda.is_available() is False inside the clearvoice "
            "venv (CUDA_VISIBLE_DEVICES=%r). The host's torch and this venv's torch are separate "
            "builds, so only this answer counts." % (requested, os.environ.get("CUDA_VISIBLE_DEVICES"))
        )
    device = torch.device(requested)

    # Reconstructs SpeechModel.__init__'s post-conditions field for field, minus the device
    # auto-detection. The field list is pinned by the version assertion above.
    def _patched_init(self, a):
        a.use_cuda = 0 if device.type == "cpu" else 1
        self.args = a
        self.model = None
        self.name = None
        self.data = {}
        self.print = False
        self.device = device

    cvnet.SpeechModel.__init__ = _patched_init

    from clearvoice import ClearVoice

    if args["mode"] == "tse":
        import clearvoice.utils.video_process as vp

        # Upstream's visualization() does two things: it writes the extracted audio for each face
        # track, and it re-renders the entire source video once per track with a bounding box drawn
        # on every frame. senselab returns Audio, so the render is pure cost — and the write is
        # `sf.write(path, audio, 16000)` with no subtype, i.e. PCM_16, quantising the one output this
        # capability exists to produce. Replaced with the write alone, through the staged policy.
        def _write_only(tracks, est_sources, video_args):
            for idx, audio in enumerate(est_sources):
                write_audio(
                    os.path.join(video_args.pycropPath, "est_%s.wav" % idx),
                    np.asarray(audio, dtype=np.float32),
                    16000,
                    out_of_range="normalize",
                )

        vp.visualization = _write_only

        # The face detector loads from inside its own package directory and, when the file is absent,
        # shells out to gdown. Linking the verified copy into place makes that branch unreachable.
        s3fd_dir = Path(cvnet.__file__).parent / "models" / "av_mossformer2_tse" / "faceDetector" / "s3fd"
        s3fd_weights = s3fd_dir / "sfd_face.pth"
        if not s3fd_weights.exists():
            os.symlink(args["s3fd_weights"], s3fd_weights)

        cv = ClearVoice(task=args["task"], model_names=[model_name])
        outputs = []
        for video_path in args["video_paths"]:
            cv(input_path=video_path, online_write=True, output_path=args["output_dir"])
            stem = Path(video_path).name.split(".")[0]
            track_dir = Path(args["output_dir"]) / model_name / stem / "py_faceTracks"
            wavs = sorted(track_dir.glob("est_*.wav"), key=lambda p: int(p.stem.split("_")[-1]))
            outputs.append([str(p) for p in wavs])
        print(json.dumps({"output_paths": outputs, "device": str(device)}))
        sys.exit(0)

    cv = ClearVoice(task=args["task"], model_names=[model_name])
    net = cv.models[0]

    from clearvoice.dataloader.dataloader import audio_norm

    written = []
    scalars = []
    with torch.no_grad():
        for index, in_path in enumerate(args["in_paths"]):
            waveform, sample_rate = read_audio(in_path, always_2d=True, channels_first=True)
            waveform = waveform[0]
            scalar = 1.0
            if args["rms_normalise"]:
                # Upstream applies this for FRCRN_SE_16K and MossFormer2_SS_16K only, and returns the
                # inverse so a single-output task can restore the input's level.
                waveform, scalar = audio_norm(waveform)

            net.data = {
                "audio": [np.reshape(waveform.astype(np.float32), [1, waveform.shape[0]])],
                "audio_len": waveform.shape[0],
            }
            decoded = net.decode()

            if isinstance(decoded, list):
                # Multi-source. Upstream's own process() does NOT apply the inverse scalar on this
                # branch, so each source comes back RMS-matched to the -25 dBFS normalised input
                # rather than to the caller's input level. Reproduced, not corrected: the scalar is
                # reported back instead, so a caller can restore the level and can see that it had to.
                signals = [source[0, :] for source in decoded]
            else:
                signals = [decoded[0, :] * scalar]

            # The worker names the outputs from what the model actually produced, and the host counts
            # them. Nothing here presumes a source count from the model's name.
            paths = []
            for source_index, signal in enumerate(signals):
                out_path = os.path.join(args["out_dir"], "out_%d_s%d.wav" % (index, source_index))
                write_audio(out_path, np.asarray(signal, dtype=np.float32), sample_rate)
                paths.append(out_path)
            written.append(paths)
            scalars.append(float(scalar))

    print(json.dumps({"output_paths": written, "input_norm_scalars": scalars, "device": str(device)}))
except Exception as exc:
    import traceback

    print(
        json.dumps(
            {
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                }
            }
        )
    )
    sys.exit(1)
"""


def default_audio_timeout_s(total_audio_s: float) -> float:
    """Return the default worker ceiling for ``total_audio_s`` seconds of audio.

    Args:
        total_audio_s: Total duration the worker will decode, summed over every input.

    Returns:
        Seconds, never below :data:`_TIMEOUT_FLOOR_S`. Derivation: the comment block above these
        constants.
    """
    return max(_TIMEOUT_FLOOR_S, _TIMEOUT_HEADROOM * _SECONDS_PER_AUDIO_SECOND * total_audio_s)


def default_tse_timeout_s(total_video_s: float) -> float:
    """Return the default worker ceiling for ``total_video_s`` seconds of video.

    Args:
        total_video_s: Total video duration the worker will process.

    Returns:
        Seconds, never below :data:`_TSE_TIMEOUT_FLOOR_S`.
    """
    return max(_TSE_TIMEOUT_FLOOR_S, _TIMEOUT_HEADROOM * _TSE_SECONDS_PER_VIDEO_SECOND * total_video_s)


def resolve_worker_device(device: Optional[DeviceType]) -> Optional[str]:
    """Validate a caller's device and turn it into an explicit device string for the worker.

    ``DeviceType.CUDA`` becomes ``"cuda:<index>"`` — the index ``torch.cuda.current_device()`` reports
    in *this* process, which under a ``CUDA_VISIBLE_DEVICES`` mask is the allocated card rather than
    merely the first one on the host. A bare ``"cuda"`` would let the worker's torch pick its own
    default, which is how a caller's choice gets silently discarded.

    ``None`` stays ``None`` and the worker decides, because only the venv's torch can answer whether
    CUDA is usable there.

    MPS is not offered. Upstream selects it unconditionally whenever it is present, and none of the
    six checkpoints has been verified on it in this repository, so a caller passing
    ``DeviceType.MPS`` gets ``_select_device_and_dtype``'s error rather than an untested backend.

    Args:
        device: The caller's request, or ``None``.

    Returns:
        An explicit device string, or ``None`` to leave the choice to the worker.

    Raises:
        ValueError: If ``device`` is neither CUDA nor CPU, or names a device this host lacks.
    """
    if device is None:
        return None
    selected, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
    )
    return device_run_opt(selected)


def _run_worker(payload: Dict[str, object], timeout_s: float, label: str, on_timeout: str) -> dict:
    """Provision the venv, run the worker, and turn a timeout into an actionable failure.

    Args:
        payload: The worker's JSON input.
        timeout_s: Ceiling in seconds.
        label: Label for error messages.
        on_timeout: Sentence describing the work attempted, appended to a timeout message.

    Returns:
        The worker's parsed JSON output.

    Raises:
        RuntimeError: If the worker exceeds ``timeout_s`` or fails. ``parse_subprocess_result``
            preserves the upstream traceback in the latter case.
    """
    venv_dir = ensure_venv(CLEARVOICE_VENV, CLEARVOICE_REQUIREMENTS, python_version=CLEARVOICE_PYTHON)
    python = venv_python(venv_dir)
    try:
        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=_clean_subprocess_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{label} worker exceeded its {timeout_s:.10g}s ceiling with {on_timeout}. Nothing is "
            "returned: every output is discarded with the worker's temporary directory. Pass "
            "timeout_s to raise the ceiling, or select a CUDA device."
        ) from exc
    return parse_subprocess_result(result, venv_label=label)


def run_clearvoice_audio(
    spec: ClearVoiceModelSpec,
    in_paths: Sequence[str],
    out_dir: str,
    *,
    total_audio_s: float,
    device: Optional[DeviceType] = None,
    timeout_s: Optional[float] = None,
    revision: str = "main",
) -> Tuple[List[List[str]], List[float], str]:
    """Run one of the five audio-only checkpoints over already-prepared WAV files.

    The caller owns resampling, downmixing and writing the inputs (through ``Audio.save_to_file``);
    this function owns the pin, the venv, the device, the ceiling and the output naming.

    Args:
        spec: Checkpoint to run.
        in_paths: One mono WAV per input, already at ``spec.sampling_rate``.
        out_dir: Directory the worker writes outputs into. Must outlive the call until the returned
            paths have been read.
        total_audio_s: Total input duration, for the default ceiling.
        device: CUDA or CPU; ``None`` leaves the choice to the worker.
        timeout_s: Ceiling in seconds; ``None`` derives one from ``total_audio_s``.
        revision: Ref or commit for the checkpoint repository.

    Returns:
        ``(output_paths, input_norm_scalars, commit_sha)``. ``output_paths`` has one list per input,
        holding as many files as the checkpoint actually produced. ``input_norm_scalars`` has one
        entry per input: the inverse of upstream's RMS normalisation, ``1.0`` for the four checkpoints
        that do not normalise. It is already applied for single-output models and deliberately *not*
        applied for multi-source ones — see the worker's comment on that branch.

    Raises:
        ValueError: If ``timeout_s`` is not positive.
        RuntimeError: If the worker fails or exceeds its ceiling.
    """
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive number of seconds, got {timeout_s}")

    worker_device = resolve_worker_device(device)
    checkpoint_dir, sha = stage_clearvoice_checkpoints(spec, revision=revision)
    effective_timeout_s = default_audio_timeout_s(total_audio_s) if timeout_s is None else timeout_s

    logger.info(
        "ClearVoice %s (%s): %d input(s), %.10gs of audio, commit %s, device=%s, timeout=%.10gs",
        spec.name,
        spec.capability,
        len(in_paths),
        total_audio_s,
        sha[:12],
        worker_device or "worker's choice",
        effective_timeout_s,
    )

    with tempfile.TemporaryDirectory(prefix="senselab-clearvoice-") as staging:
        output = _run_worker(
            {
                "mode": "audio",
                "staging_dir": staging,
                "io_dir": stage_portable_audio_io(staging),
                "model_name": spec.name,
                "task": spec.upstream_task,
                "checkpoint_dir": str(checkpoint_dir),
                "expected_version": CLEARVOICE_VERSION,
                "device": worker_device,
                "rms_normalise": spec.rms_normalises_input,
                "in_paths": list(in_paths),
                "out_dir": out_dir,
            },
            timeout_s=effective_timeout_s,
            label=f"ClearVoice {spec.name}",
            on_timeout=f"{total_audio_s:.10g}s of audio over {len(in_paths)} input(s)",
        )
    return output["output_paths"], output["input_norm_scalars"], sha


def run_clearvoice_tse(
    spec: ClearVoiceModelSpec,
    video_paths: Sequence[str],
    output_dir: str,
    *,
    total_video_s: float,
    device: Optional[DeviceType] = None,
    timeout_s: Optional[float] = None,
    revision: str = "main",
) -> Tuple[List[List[str]], str]:
    """Run the audio-visual target-speaker extractor over video files.

    Args:
        spec: Must be the ``AV_MossFormer2_TSE_16K`` spec.
        video_paths: Video files, in a container upstream's reader accepts.
        output_dir: Directory the worker writes into. Must outlive this call, since the returned WAV
            paths point inside it.
        total_video_s: Total video duration, for the default ceiling.
        device: CUDA or CPU; ``None`` leaves the choice to the worker.
        timeout_s: Ceiling in seconds; ``None`` derives one from ``total_video_s``.
        revision: Ref or commit for the checkpoint repository.

    Returns:
        ``(wav_paths_per_video, commit_sha)`` — one 16 kHz WAV per detected face track, ordered by
        track index.

    Raises:
        ValueError: If ``timeout_s`` is not positive.
        RuntimeError: If the worker fails or exceeds its ceiling.
    """
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive number of seconds, got {timeout_s}")

    worker_device = resolve_worker_device(device)
    checkpoint_dir, sha = stage_clearvoice_checkpoints(spec, revision=revision)
    s3fd_weights = stage_s3fd_weights()
    effective_timeout_s = default_tse_timeout_s(total_video_s) if timeout_s is None else timeout_s

    logger.info(
        "ClearVoice %s (%s): %d video(s), %.10gs of video, commit %s, device=%s, timeout=%.10gs",
        spec.name,
        spec.capability,
        len(video_paths),
        total_video_s,
        sha[:12],
        worker_device or "worker's choice",
        effective_timeout_s,
    )

    with tempfile.TemporaryDirectory(prefix="senselab-clearvoice-tse-") as staging:
        output = _run_worker(
            {
                "mode": "tse",
                "staging_dir": staging,
                "io_dir": stage_portable_audio_io(staging),
                "model_name": spec.name,
                "task": spec.upstream_task,
                "checkpoint_dir": str(checkpoint_dir),
                "expected_version": CLEARVOICE_VERSION,
                "device": worker_device,
                "s3fd_weights": str(s3fd_weights),
                "video_paths": list(video_paths),
                "output_dir": output_dir,
            },
            timeout_s=effective_timeout_s,
            label=f"ClearVoice {spec.name}",
            on_timeout=f"{total_video_s:.10g}s of video over {len(video_paths)} file(s)",
        )
    return output["output_paths"], sha
