"""DriftSE one-step speech enhancement via an isolated subprocess venv.

DriftSE (Xu, Caviedes-Nozal, Kleijn, Yan & Olsson, *Speech Enhancement Based on Drifting Models*,
Interspeech 2026 oral, arXiv 2604.24199) reaches the clean-speech distribution in one network
evaluation, which is what makes it the only generative enhancer in senselab that runs on CPU.
Upstream code and weights are MIT (https://github.com/LiangXu123/DriftSE).

Usage:
    Pass an ``HFModel`` naming ``LIANGXU123/DriftSE`` to
    :func:`senselab.audio.tasks.speech_enhancement.enhance_audios`, or call
    :func:`enhance_audios_with_driftse` directly for control over ``sigma``, ``seed``, the
    checkpoint ``variant`` and the chunking. Inputs are resampled to 16 kHz mono; outputs are
    16 kHz with the input's sample count.

The worker clones upstream at a pinned commit into its venv on first use and downloads one pinned
checkpoint file from the Hub. ``SENSELAB_DRIFTSE_CHECKPOINT`` points it at a local directory holding
``last.ckpt`` + ``config.json`` instead, and then no Hub access happens at all.

Design, upstream history, the pins and the measurements behind every choice here:
``specs/20260818-083214-driftse-upstream-mit/design.md``. The device hand-off and the derivation of
the worker's default timeout: ``specs/20260818-235500-driftse-device-timeout/design.md``.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel
from senselab.utils.data_structures.logging import logger
from senselab.utils.subprocess_venv import (
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    stage_portable_audio_io,
    venv_python,
)

_DRIFTSE_VENV = "driftse"
_DRIFTSE_PYTHON = "3.11"

# Upstream's requirements.txt is a *training* set; only what the inference import chain touches is
# installed. pesq/pystoi are on that chain (util/other.py imports both at module scope) even though
# this backend computes no metric. torch and torchaudio are named explicitly so ensure_venv's
# CUDA-aware wheel routing triggers. Both findings are recorded in the spec named above.
_DRIFTSE_REQUIREMENTS = [
    "torch>=2.3",
    "torchaudio>=2.3",
    "numpy>=1.26",
    "scipy>=1.12",
    "librosa>=0.10.2",
    "soundfile>=0.12.1",
    "tqdm>=4.66",
    "pesq>=0.0.4",
    "pystoi>=0.3.3",
]

_DRIFTSE_REPO_URL = "https://github.com/LiangXu123/DriftSE.git"
# Pinned, not a branch: the repository is unpackaged, so this SHA is the version contract. It is at
# upstream HEAD, which is after both 70bb6ded (the paper-aligned sigma) and 60333a68 (the ema state
# dict); the previous pin, 695a64db, predates both and ran inference the author calls wrong.
_DRIFTSE_COMMIT = "0a489dadfa2778e86e4b4b0af03f6255d2de8c69"

# Upstream's own weights mirror, MIT-licensed and public, pinned so a re-upload cannot change what
# runs. It supersedes senselab's sensein/driftse-* mirror, whose files were byte-identical.
_DRIFTSE_HF_REPO = "LIANGXU123/DriftSE"
_DRIFTSE_HF_REVISION = "b99a25a637a9963d5c7557f0b70597fc54c7a0bb"
_DRIFTSE_CHECKPOINT_ENV = "SENSELAB_DRIFTSE_CHECKPOINT"

# variant -> (checkpoint file in the weights mirror, architecture config in the pinned clone). The
# mirror's own top-level config.json is HF download-tracking metadata, not an NCSN++ config.
_DRIFTSE_VARIANTS: Dict[str, Tuple[str, str]] = {
    "distillhubert_three_layers_with_z": (
        "logs/distillhubert_three_layers_with_z/last.ckpt",
        "config/with_z/v2_drift2_distillhubert_three_layers_adam.json",
    ),
    "distillhubert_three_layers_pesq_sisdr_ccmse_with_z": (
        "logs/distillhubert_three_layers_pesq_sisdr_ccmse_with_z/last.ckpt",
        "config/with_z/v2_drift2_distillhubert_three_layers_pesq_sisdr_ccmse.json",
    ),
}
_DRIFTSE_DEFAULT_VARIANT = "distillhubert_three_layers_with_z"

# Upstream enhancement.py's own constant for the Gaussian it adds to the model input.
_DRIFTSE_DEFAULT_SIGMA = 0.01

# Terms of the default worker ceiling: seconds of wall time per second of audio inside one window,
# a headroom multiplier, and a floor. Measurement and derivation:
# specs/20260818-235500-driftse-device-timeout/design.md.
_SECONDS_PER_WINDOW_SECOND = 1.1
_TIMEOUT_HEADROOM = 4.0
_TIMEOUT_FLOOR_S = 1800.0


def _window_count(n_samples: int, chunk_samples: int, hop_samples: int) -> int:
    """Return how many windows the worker will evaluate for one input.

    Mirrors the worker's own chunking: fixed-length windows on a regular hop, plus a final window
    anchored at the end of the signal when the regular ones do not reach it.

    Args:
        n_samples: Length of the (resampled) input in samples.
        chunk_samples: Window length in samples.
        hop_samples: Distance between the starts of adjacent windows, in samples.

    Returns:
        The number of windows, at least one.
    """
    if n_samples <= chunk_samples:
        return 1
    starts = list(range(0, n_samples - chunk_samples + 1, hop_samples))
    if starts[-1] + chunk_samples < n_samples:
        starts.append(n_samples - chunk_samples)
    return len(starts)


def _default_timeout_s(n_windows: int, chunk_s: float) -> float:
    """Return the default worker ceiling for ``n_windows`` windows of ``chunk_s`` seconds each.

    Args:
        n_windows: Total number of windows the worker will enhance, across every input.
        chunk_s: Window length in seconds.

    Returns:
        Seconds, never below ``_TIMEOUT_FLOOR_S``.
    """
    return max(_TIMEOUT_FLOOR_S, _TIMEOUT_HEADROOM * _SECONDS_PER_WINDOW_SECOND * n_windows * chunk_s)


# Worker script — runs inside the isolated venv. Clones the (non-packaged) upstream repo at a pinned
# commit on first use and adds it to sys.path, then reuses upstream's own backbone construction and
# spectral transforms rather than reimplementing them here.
_WORKER_SCRIPT = r"""
import json
import subprocess as sp
import sys
from pathlib import Path

try:
    args = json.loads(sys.stdin.read())
    repo_dir = Path(args["repo_dir"])
    repo_url, commit = args["repo_url"], args["commit"]
    ckpt_path = args["ckpt_path"]
    config_path, config_rel = args["config_path"], args["config_rel"]
    in_paths, out_paths = args["in_paths"], args["out_paths"]
    seed = int(args["seed"])
    sigma = float(args["sigma"])
    chunk_s, overlap_s = float(args["chunk_s"]), float(args["overlap_s"])
    sys.path.insert(0, args["io_dir"])
    from portable_audio_io import read_audio, write_audio
    requested_device = args.get("device")

    import fcntl, os, shutil, tempfile as _tempfile

    # Clone under an exclusive flock, to a sibling temp dir + atomic os.replace, so an interrupted
    # clone never leaves repo_dir present but incomplete (which would wedge the guard below
    # permanently) and concurrent jobs sharing $HOME cannot race into the same directory.
    marker = repo_dir / "enhancement.py"
    if not marker.is_file():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(str(repo_dir) + ".lock", "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not marker.is_file():
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)
                tmp_clone = Path(_tempfile.mkdtemp(prefix=".driftse-clone-", dir=str(repo_dir.parent)))
                try:
                    sp.run(["git", "init", "-q", str(tmp_clone)], check=True)
                    sp.run(["git", "-C", str(tmp_clone), "remote", "add", "origin", repo_url], check=True)
                    sp.run(["git", "-C", str(tmp_clone), "fetch", "-q", "--depth", "1", "origin", commit], check=True)
                    sp.run(["git", "-C", str(tmp_clone), "checkout", "-q", "FETCH_HEAD"], check=True)
                except Exception:
                    shutil.rmtree(tmp_clone, ignore_errors=True)
                    raise
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)
                os.replace(tmp_clone, repo_dir)

    sys.path.insert(0, str(repo_dir))

    import numpy as np
    import soundfile as sf
    import torch

    # NOTE: only util/other.py is imported; upstream's util/inference.py is never reached.
    from backbones.ncsnpp_v2 import NCSNpp_v2
    from backbones.ncsnpp_v2_drift import ncsnpp_v2_drift
    from util.other import pad_spec, set_torch_cuda_arch_list

    set_torch_cuda_arch_list()  # prints and returns when CUDA is absent

    def resolve_device(requested):
        # The host sends its caller's choice; None means "you decide". A bare "cuda" would take
        # whatever index torch defaults to, so an index is always chosen.
        if requested is None:
            return torch.device("cuda:%d" % torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        if not str(requested).startswith("cuda"):
            return torch.device(requested)
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DriftSE worker: device %r was requested but torch.cuda.is_available() is False "
                "inside the driftse venv (CUDA_VISIBLE_DEVICES=%r)"
                % (requested, os.environ.get("CUDA_VISIBLE_DEVICES"))
            )
        if ":" in str(requested):
            return torch.device(requested)
        return torch.device("cuda:%d" % torch.cuda.current_device())

    device = resolve_device(requested_device)
    torch.manual_seed(seed)

    # A local override supplies its own config.json; otherwise the architecture config comes from
    # the pinned clone, so weights and config are both commit-addressed.
    config_file = Path(config_path) if config_path else repo_dir / config_rel
    with open(config_file) as f:
        config = json.load(f)

    builder = ncsnpp_v2_drift if config["model"].lower() == "ncsnpp_v2_drift" else NCSNpp_v2
    model = builder(
        nf=config["nf"], ch_mult=config["ch_mult"], num_res_blocks=config["num_res_blocks"],
        attn_resolutions=config["attn_resolutions"], image_size=config["image_size"],
        fourier_scale=config["fourier_scale"], resamp_with_conv=config["resamp_with_conv"],
        fir=config["fir"], fir_kernel=config["fir_kernel"], skip_rescale=config["skip_rescale"],
        resblock_type=config["resblock_type"], progressive=config["progressive"],
        progressive_input=config["progressive_input"], progressive_combine=config["progressive_combine"],
        init_scale=config["init_scale"], embedding_type=config["embedding_type"],
        dropout=config["dropout"],
    ).to(device)

    # DEVIATION 1: weights_only=True. Upstream omits it; the checkpoint is a foreign pickle.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # ema first, then model, matching upstream's own priority at the pinned commit.
    if "ema" in ckpt:
        state_dict = ckpt["ema"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()

    n_fft, hop = config["n_fft"], config["hop_length"]
    if config["window_type"] == "sqrthann":
        window = torch.sqrt(torch.hann_window(n_fft, periodic=True, device=device))
    elif config["window_type"] == "hann":
        window = torch.hann_window(n_fft, periodic=True, device=device)
    else:
        raise NotImplementedError("Unsupported window_type: " + str(config["window_type"]))

    e, f = config["spec_abs_exponent"], config["spec_factor"]
    add_gaussian = str(config.get("train_add_gaussian", True)).lower() == "true"

    def enhance_window(y_win):
        # Upstream's whole-file procedure applied to one window: peak-normalise in, one network
        # evaluation, then rescale the output by its own peak back to the window's input peak.
        T_orig = y_win.shape[-1]
        norm = y_win.abs().max() + 1e-8
        Y = torch.stft(y_win / norm, n_fft=n_fft, hop_length=hop, window=window,
                       center=config["center"], return_complex=True)
        Y = (Y.abs() ** e * torch.exp(1j * Y.angle())) * f
        Y = pad_spec(Y.unsqueeze(0).unsqueeze(0), mode="zero_pad")
        with torch.no_grad():
            t = torch.ones(Y.shape[0], device=device)
            # DEVIATION 3: recorded RNG seed. With train_add_gaussian set -- which both released
            # checkpoints do -- this Gaussian sample makes the forward pass stochastic.
            out = model(Y + sigma * torch.randn_like(Y), t) if add_gaussian else model(Y, t)
        X = out.squeeze(0).squeeze(0) / f
        X = (X.abs() ** (1.0 / e)) * torch.exp(1j * X.angle())
        x = torch.istft(X, n_fft=n_fft, hop_length=hop, window=window,
                        center=config["center"], length=T_orig)
        out_peak = x.abs().max()
        return x / out_peak * norm if out_peak > 1e-8 else x * norm

    results = []
    for in_path, out_path in zip(in_paths, out_paths):
        y_np, sr = read_audio(in_path, always_2d=True, channels_first=False)
        y = torch.as_tensor(y_np[:, 0]).to(device)
        assert sr == 16000, "worker expects 16 kHz; the host resamples"

        # DEVIATION 2: overlap-add chunking. Upstream runs one STFT over an entire file, and the
        # NCSN++ backbone's attention makes memory grow superlinearly in duration.
        total = y.shape[-1]
        chunk = int(chunk_s * sr)
        hop_samples = int((chunk_s - overlap_s) * sr)
        if total <= chunk:
            x_hat = enhance_window(y)
        else:
            # Every window is exactly `chunk` long; the last one is anchored at the end of the file
            # rather than being a short remainder, so no tail is dropped and no window is too short
            # to transform.
            starts = list(range(0, total - chunk + 1, hop_samples))
            if starts[-1] + chunk < total:
                starts.append(total - chunk)
            base_taper = torch.hann_window(chunk, periodic=False, device=device)
            acc = torch.zeros_like(y)
            wsum = torch.zeros_like(y)
            for i, start in enumerate(starts):
                enhanced = enhance_window(y[start : start + chunk])
                taper = base_taper.clone()
                if i == 0:
                    taper[: chunk // 2] = 1.0  # no fade-in at the start of the file
                if i == len(starts) - 1:
                    taper[chunk // 2 :] = 1.0  # nor a fade-out at its end
                acc[start : start + chunk] += enhanced * taper
                wsum[start : start + chunk] += taper
            x_hat = acc / wsum.clamp(min=1e-8)

        write_audio(out_path, x_hat.detach().cpu().numpy(), sr)
        results.append(out_path)

    print(json.dumps({"output_paths": results, "seed": seed, "sigma": sigma}))
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


def enhance_audios_with_driftse(
    audios: List[Audio],
    model: HFModel,
    device: Optional[DeviceType] = None,
    seed: int = 0,
    sigma: float = _DRIFTSE_DEFAULT_SIGMA,
    variant: str = _DRIFTSE_DEFAULT_VARIANT,
    chunk_s: float = 20.0,
    overlap_s: float = 2.0,
    timeout_s: Optional[float] = None,
) -> List[Audio]:
    """Enhance each audio with DriftSE, one network evaluation per window.

    Resamples to 16 kHz mono on the way in — the checkpoints are trained at that rate and
    ``n_fft=510`` gives exactly the 256 frequency bins their ``image_size`` expects. Output is
    16 kHz with the same number of samples as the (resampled) input.

    Args:
        audios: Inputs. Resampled and downmixed to 16 kHz mono if needed.
        model: ``HFModel`` naming the weights repo and revision (``LIANGXU123/DriftSE``).
        device: Device the worker runs on. ``DeviceType.CUDA`` is resolved to an explicit
            ``"cuda:<index>"`` (the index ``torch.cuda.current_device()`` reports in this process,
            so under a ``CUDA_VISIBLE_DEVICES`` mask it is the allocated card) and sent to the
            worker; ``DeviceType.CPU`` is sent as ``"cpu"``. ``None`` leaves the choice to the
            worker, which takes ``cuda:<current index>`` when CUDA is available and CPU otherwise
            -- at 1 NFE, CPU is practical here. Only CUDA and CPU are accepted.
        seed: RNG seed for the Gaussian perturbation, which the released checkpoints make part of
            the forward pass. Output is stochastic without it, so it is recorded in the log line.
        sigma: Scale of that Gaussian. Upstream's own default is 0.01; 0 is equivalent within
            noise and 0.05 measurably degrades output (see the spec named in the module docstring).
        variant: Checkpoint to use, a key of ``_DRIFTSE_VARIANTS``. Both released checkpoints set
            ``train_add_gaussian``; ``distillhubert_three_layers_pesq_sisdr_ccmse_with_z`` scores
            higher on PESQ/SI-SDR and was trained with those metrics in its loss.
        chunk_s: Window length for long inputs. Each window is peak-normalised in and peak-matched
            out on its own, exactly as upstream treats a whole file.
        overlap_s: Overlap between windows, Hann-tapered and overlap-added.
        timeout_s: Ceiling on the worker subprocess, in seconds. ``None`` derives one from the
            work -- total windows across every input, times ``chunk_s``, times a per-window-second
            factor, with a floor covering the first-use venv build, clone and checkpoint load
            (:func:`_default_timeout_s`). Exceeding it raises ``RuntimeError`` and discards every
            output, finished or not.

    Returns:
        One enhanced ``Audio`` per input, in order.

    Raises:
        ValueError: if ``variant`` is not a known checkpoint, if ``timeout_s`` is not positive,
            or if ``device`` is neither CUDA nor CPU.
        RuntimeError: if the worker fails; the upstream traceback is included. Also if the worker
            exceeds ``timeout_s`` -- that message names the ceiling, the input, how far the worker
            had got, and the knob that raises it.
    """
    if variant not in _DRIFTSE_VARIANTS:
        raise ValueError(f"unknown DriftSE variant {variant!r}; known: {sorted(_DRIFTSE_VARIANTS)}")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError(f"timeout_s must be a positive number of seconds, got {timeout_s}")

    if not audios:
        return []

    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
    from senselab.utils.data_structures import _select_device_and_dtype, device_run_opt

    # None is forwarded as None rather than resolved here: the host interpreter and the venv have
    # separate torch builds, and only the venv's torch.cuda.is_available() governs where the worker
    # can actually run.
    worker_device: Optional[str] = None
    if device is not None:
        selected_device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        worker_device = device_run_opt(selected_device)

    checkpoint_file, config_rel = _DRIFTSE_VARIANTS[variant]

    checkpoint_override = os.environ.get(_DRIFTSE_CHECKPOINT_ENV)
    if checkpoint_override:
        override_dir = Path(checkpoint_override)
        ckpt_path = str(override_dir / "last.ckpt")
        config_path = str(override_dir / "config.json")
        config_rel = ""
    else:
        from huggingface_hub import hf_hub_download

        from senselab.utils.model_revision import resolve_revision

        # One file, not the snapshot: the mirror is 2.4 GB, of which one 1.14 GB checkpoint is
        # read. resolve_model would download all of it. Resolving the ref to a commit SHA first
        # keeps the pinning guarantee -- a full SHA takes huggingface_hub's commit-hash shortcut,
        # so a cached file resolves with no network.
        sha = model.commit_sha or resolve_revision(str(model.path_or_uri), model.revision)
        ckpt_path = hf_hub_download(str(model.path_or_uri), checkpoint_file, revision=sha)
        config_path = ""

    venv_dir = ensure_venv(_DRIFTSE_VENV, _DRIFTSE_REQUIREMENTS, python_version=_DRIFTSE_PYTHON)
    python = venv_python(venv_dir)
    # Cached alongside the venv rather than per-tempdir, so the pinned-commit clone in the worker
    # script happens once per host, not once per call.
    repo_dir = Path(venv_dir) / "driftse-src"

    mono_16k = downmix_audios_to_mono(resample_audios(audios, resample_rate=16000))

    chunk_samples = int(chunk_s * 16000)
    hop_samples = int((chunk_s - overlap_s) * 16000)
    total_windows = sum(_window_count(int(a.waveform.shape[-1]), chunk_samples, hop_samples) for a in mono_16k)
    effective_timeout_s = _default_timeout_s(total_windows, chunk_s) if timeout_s is None else timeout_s

    logger.info(
        f"DriftSE: enhancing {len(mono_16k)} audio(s) ({total_windows} window(s) total), "
        f"variant={variant}, seed={seed}, sigma={sigma}, chunk_s={chunk_s}, overlap_s={overlap_s}, "
        f"device={worker_device or 'worker-selected'}, timeout={effective_timeout_s:.10g}s"
    )

    with tempfile.TemporaryDirectory(prefix="senselab-driftse-") as tmpdir:
        tmp = Path(tmpdir)
        in_paths, out_paths = [], []
        for i, audio in enumerate(mono_16k):
            in_path = str(tmp / f"in_{i}.wav")
            out_path = str(tmp / f"out_{i}.wav")
            audio.save_to_file(in_path)
            in_paths.append(in_path)
            out_paths.append(out_path)

        input_json = json.dumps(
            {
                "repo_dir": str(repo_dir),
                "repo_url": _DRIFTSE_REPO_URL,
                "commit": _DRIFTSE_COMMIT,
                "ckpt_path": ckpt_path,
                "config_path": config_path,
                "config_rel": config_rel,
                "in_paths": in_paths,
                "out_paths": out_paths,
                "seed": seed,
                "sigma": sigma,
                "chunk_s": chunk_s,
                "overlap_s": overlap_s,
                "io_dir": stage_portable_audio_io(tmp),
                "device": worker_device,
            }
        )

        try:
            result = subprocess.run(
                [python, "-c", _WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=effective_timeout_s,
                env=_clean_subprocess_env(),
            )
        except subprocess.TimeoutExpired as exc:
            completed = sum(1 for path in out_paths if Path(path).is_file())
            total_input_s = sum(int(a.waveform.shape[-1]) for a in mono_16k) / 16000
            raise RuntimeError(
                f"DriftSE worker exceeded its {effective_timeout_s:.10g}s ceiling with "
                f"{completed}/{len(out_paths)} output(s) written, enhancing {len(mono_16k)} input(s) "
                f"({total_input_s:.1f}s of audio at 16000 Hz, {total_windows} window(s) of "
                f"{chunk_s:g}s) with variant={variant!r}, "
                f"device={worker_device or 'worker-selected'}. The written outputs are discarded "
                f"with the worker's temporary directory; pass timeout_s to raise the ceiling."
            ) from exc
        output = parse_subprocess_result(result, venv_label="DriftSE")

        # Read outputs back while the temp dir is still alive: Audio(filepath=...) lazy-loads on
        # first .waveform access, and that access must happen before this context manager deletes
        # the files it points at.
        enhanced_audios = []
        for out_path, original in zip(output["output_paths"], mono_16k):
            enhanced = Audio(filepath=out_path)
            _ = enhanced.waveform  # force the lazy load before the tempdir is gone
            enhanced.metadata = original.metadata
            enhanced_audios.append(enhanced)

    return enhanced_audios
