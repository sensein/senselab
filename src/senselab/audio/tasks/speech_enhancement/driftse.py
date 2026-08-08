"""DriftSE one-step speech enhancement via isolated subprocess venv.

DriftSE (Xu, Caviedes-Nozal, Kleijn, Yan & Olsson, *Speech Enhancement Based on
Drifting Models*, Interspeech 2026 oral, arXiv 2604.24199) formulates enhancement
as a distributional equilibrium problem and reaches the clean-speech distribution
in a **single** network evaluation, against 30 for SGMSE+ and 8 for UNIVERSE++.
On the DNS 2020 blind test set it reports WV-MOS 2.65 and SCOREQ 2.97.

Why inference is cheap
----------------------
The drifting field is computed in a frozen self-supervised latent space
(HuBERT / WavLM / DistilHuBERT) — but that is the **training** signal. Inference
is the backbone alone: one forward pass under ``no_grad``. Upstream's
``enhancement.py`` imports only ``backbones.ncsnpp_v2``,
``backbones.ncsnpp_v2_drift`` and ``util.other`` — no Lightning, no ``wandb``, no
``pesq``, and no SSL encoder. The ``latent_ckpt/`` archive its README requires for
training is therefore not needed here at all, and this is the first generative
enhancer in senselab that is genuinely CPU-viable.

Why a subprocess venv
---------------------
Not for dependency conflict — the inference dependency set would satisfy senselab
core. The upstream repository has no installable package and its top-level module
names are ``backbones``, ``util``, ``config`` and ``data``. Injecting a generic
``util`` onto the host interpreter's ``sys.path`` is the kind of hazard that
surfaces months later as an unrelated import resolving to the wrong module.

Licensing
---------
The upstream repository reports no license (no ``LICENSE`` file, no statement in
the README), and is itself built on SGMSE+ (MIT) without carrying that statement
forward. senselab therefore vendors none of it: the worker clones the repository
at a pinned commit into the user's own cache at first use. The checkpoint mirror
under ``sensein`` is private pending an upstream answer; see this module's
``doc.md`` for the status of that request.

Not wired into ``audio_analysis``
---------------------------------
Reachable through :func:`enhance_audios` by passing the model explicitly. The
workflow's default enhancer is unchanged. Deciding how a second enhancer's output
participates in the perturbation sample is a measurement, and it comes after this
backend exists.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel
from senselab.utils.data_structures.logging import logger
from senselab.utils.subprocess_venv import (
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    venv_python,
)

_DRIFTSE_VENV = "driftse"
_DRIFTSE_PYTHON = "3.11"

# Upstream's requirements.txt is a *training* dependency set. The inference path
# (enhancement.py -> backbones.ncsnpp_v2{,_drift} + util.other) imports none of
# pesq / pystoi / scoreq / torch-pesq / asteroid-filterbanks / wandb /
# pytorch-optimizer / torchinfo, so they are deliberately absent: pesq and scoreq
# in particular are slow and fragile to build for no benefit here.
#
# torch and torchaudio are named explicitly so ensure_venv's CUDA auto-detection
# triggers and routes Stage 1 through the matching PyTorch wheel index. Left
# transitive, the resolve skips that routing and can land a CPU-only wheel on a
# GPU host.
_DRIFTSE_REQUIREMENTS = [
    "torch>=2.3",
    "torchaudio>=2.3",
    "numpy>=1.26",
    "scipy>=1.12",
    "librosa>=0.10.2",
    "soundfile>=0.12.1",
    "tqdm>=4.66",
]

_DRIFTSE_REPO_URL = "https://github.com/LiangXu123/DriftSE.git"
# Pinned, not a branch: the repository is unlicensed and unpackaged, so this SHA
# is the only version contract available. An upstream force-push must not change
# what this backend runs.
_DRIFTSE_COMMIT = "695a64db187500fa0d7bae23912680bd5d4df613"

_DRIFTSE_HF_REPO = "sensein/driftse-distilhubert-three-layers"
# Pinned so a re-upload cannot change what this backend runs. The repo is private
# pending the upstream licence answer; callers without access use the env override.
_DRIFTSE_HF_REVISION = "76a9448aae12e4c232b1d52c24899d0835db5782"
_DRIFTSE_CHECKPOINT_ENV = "SENSELAB_DRIFTSE_CHECKPOINT"

# Worker script — runs inside the isolated venv. Clones the (non-packaged)
# upstream repo at a pinned commit on first use and adds it to sys.path, then
# reuses upstream's own backbone construction and spectral transforms rather
# than reimplementing them here.
_WORKER_SCRIPT = r"""
import json
import subprocess as sp
import sys
from pathlib import Path

try:
    args = json.loads(sys.stdin.read())
    repo_dir = Path(args["repo_dir"])
    repo_url, commit = args["repo_url"], args["commit"]
    ckpt_path, config_path = args["ckpt_path"], args["config_path"]
    in_paths, out_paths = args["in_paths"], args["out_paths"]
    seed = int(args["seed"])
    chunk_s, overlap_s = float(args["chunk_s"]), float(args["overlap_s"])

    import fcntl, os, shutil, tempfile as _tempfile

    # Clone under an exclusive flock, to a sibling temp dir + atomic os.replace,
    # so an interrupted clone never leaves repo_dir present but incomplete
    # (which would wedge the guard below permanently) and concurrent jobs
    # sharing $HOME cannot race into the same directory.
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

    # NOTE: only util/other.py is imported. util's inference module pulls in
    # pesq/pystoi, which are deliberately absent from this venv.
    from backbones.ncsnpp_v2 import NCSNpp_v2
    from backbones.ncsnpp_v2_drift import ncsnpp_v2_drift
    from util.other import pad_spec, set_torch_cuda_arch_list

    set_torch_cuda_arch_list()  # prints and returns when CUDA is absent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    with open(config_path) as f:
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

    # DEVIATION 1: weights_only=True. Upstream omits it. The checkpoint is a
    # foreign pickle from an unlicensed research repository; the unrestricted
    # unpickler is arbitrary code execution at enhancement time.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
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

    def enhance_window(y):
        T_orig = y.shape[-1]
        norm = y.abs().max() + 1e-8
        Y = torch.stft(y / norm, n_fft=n_fft, hop_length=hop, window=window,
                       center=config["center"], return_complex=True)
        Y = (Y.abs() ** e * torch.exp(1j * Y.angle())) * f
        Y = pad_spec(Y.unsqueeze(0).unsqueeze(0), mode="zero_pad")
        with torch.no_grad():
            t = torch.ones(Y.shape[0], device=device)
            # DEVIATION 3: recorded RNG seed (torch.manual_seed(seed) above). The
            # released checkpoint sets train_add_gaussian, so this Gaussian sample
            # makes the forward pass stochastic; an unseeded rerun would make any
            # cached artifact keyed on this output non-reproducible.
            out = model(Y + 0.05 * torch.randn_like(Y), t) if add_gaussian else model(Y, t)
        X = out.squeeze(0).squeeze(0) / f
        X = (X.abs() ** (1.0 / e)) * torch.exp(1j * X.angle())
        x = torch.istft(X, n_fft=n_fft, hop_length=hop, window=window,
                        center=config["center"], length=T_orig)
        return x * norm

    results = []
    for in_path, out_path in zip(in_paths, out_paths):
        y_np, sr = sf.read(in_path, dtype="float32", always_2d=True)
        y = torch.as_tensor(y_np[:, 0]).to(device)
        assert sr == 16000, "worker expects 16 kHz; the host resamples"

        # DEVIATION 2: overlap-add chunking. Upstream runs one STFT over an
        # entire file; the NCSN++ backbone carries attention layers, so memory
        # grows superlinearly in duration. Enhancement is per-segment consistent
        # (there is no cross-segment identity to preserve), so overlap-add is
        # safe here in a way it is not for separation.
        chunk = int(chunk_s * sr)
        hop_samples = int((chunk_s - overlap_s) * sr)
        if y.shape[-1] <= chunk:
            x_hat = enhance_window(y)
        else:
            acc = torch.zeros_like(y)
            wsum = torch.zeros_like(y)
            for start in range(0, y.shape[-1], hop_samples):
                seg = y[start : start + chunk]
                if seg.shape[-1] < n_fft:
                    break
                enhanced = enhance_window(seg)
                taper = torch.hann_window(seg.shape[-1], periodic=False, device=device)
                acc[start : start + seg.shape[-1]] += enhanced * taper
                wsum[start : start + seg.shape[-1]] += taper
            x_hat = acc / wsum.clamp(min=1e-8)

        sf.write(out_path, x_hat.detach().cpu().numpy(), sr)
        results.append(out_path)

    print(json.dumps({"output_paths": results, "seed": seed}))
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
    chunk_s: float = 20.0,
    overlap_s: float = 2.0,
) -> List[Audio]:
    """Enhance each audio with DriftSE, one network evaluation per window.

    Resamples to 16 kHz mono on the way in — the checkpoint is trained at that
    rate and ``n_fft=510`` gives exactly the 256 frequency bins its ``image_size``
    expects. Output is 16 kHz and the same number of samples as the (resampled)
    input.

    With ``train_add_gaussian`` true — which the released checkpoint sets — the
    forward pass consumes a Gaussian sample, so output is stochastic. ``seed``
    makes a run reproducible and is recorded in the log line. A caller wanting
    the deterministic formulation needs the ``no_z`` checkpoint, which is a
    different set of weights, not a flag.

    Args:
        audios: Inputs. Resampled and downmixed to 16 kHz mono if needed.
        model: ``HFModel`` naming the mirrored checkpoint repo and revision.
        device: Accepted for signature parity with the other enhancers. The
            worker selects CUDA when available and CPU otherwise; DriftSE is 1
            NFE, so CPU is practical here unlike every other generative enhancer
            in this package.
        seed: RNG seed for the Gaussian perturbation.
        chunk_s: Window length for long inputs.
        overlap_s: Overlap between windows, Hann-tapered and overlap-added.

    Returns:
        One enhanced ``Audio`` per input, in order.

    Raises:
        RuntimeError: if the worker fails; the upstream traceback is included.
    """
    if not audios:
        return []

    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
    from senselab.utils.data_structures import _select_device_and_dtype

    _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])

    checkpoint_override = os.environ.get(_DRIFTSE_CHECKPOINT_ENV)
    if checkpoint_override:
        override_dir = Path(checkpoint_override)
        ckpt_path = str(override_dir / "last.ckpt")
        config_path = str(override_dir / "config.json")
    else:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id=str(model.path_or_uri), filename="last.ckpt", revision=model.revision)
        config_path = hf_hub_download(repo_id=str(model.path_or_uri), filename="config.json", revision=model.revision)

    venv_dir = ensure_venv(_DRIFTSE_VENV, _DRIFTSE_REQUIREMENTS, python_version=_DRIFTSE_PYTHON)
    python = venv_python(venv_dir)
    # Cached alongside the venv rather than per-tempdir, so the pinned-commit
    # clone in the worker script happens once per host, not once per call.
    repo_dir = Path(venv_dir) / "driftse-src"

    mono_16k = downmix_audios_to_mono(resample_audios(audios, resample_rate=16000))

    logger.info(f"DriftSE: enhancing {len(mono_16k)} audio(s), seed={seed}, chunk_s={chunk_s}, overlap_s={overlap_s}")

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
                "in_paths": in_paths,
                "out_paths": out_paths,
                "seed": seed,
                "chunk_s": chunk_s,
                "overlap_s": overlap_s,
            }
        )

        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=1800,
            env=_clean_subprocess_env(),
        )
        output = parse_subprocess_result(result, venv_label="DriftSE")

        # Read outputs back while the temp dir is still alive: Audio(filepath=...)
        # lazy-loads on first .waveform access, and that access must happen
        # before this context manager deletes the files it points at.
        enhanced_audios = []
        for out_path, original in zip(output["output_paths"], mono_16k):
            enhanced = Audio(filepath=out_path)
            _ = enhanced.waveform  # force the lazy load before the tempdir is gone
            enhanced.metadata = original.metadata
            enhanced_audios.append(enhanced)

    return enhanced_audios
