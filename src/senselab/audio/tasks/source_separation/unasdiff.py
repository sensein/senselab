"""unasdiff unsupervised source separation via isolated subprocess venv.

unasdiff (Shi, Runwu et al., *Unsupervised Audio Source Separation using Diffusion
Priors*, https://github.com/RunwuShi/unasdiff) separates a mixture into speech and
one FSD50K-conditioned sound source without ever training on mixtures: it factors
the mixture likelihood into two independently-trained unconditional diffusion
priors (a speech prior and a sound prior) and runs posterior sampling at inference
time. This is what makes it usable for the off-target-speaker-detection problem
this repository is tracking (see ``project_off_target_speaker_detection`` in this
project's notes) -- there is no dataset of "target speech + arbitrary intruder"
mixtures to train a supervised separator on, but there are large unconditional
speech and sound corpora to train priors on separately.

Upstream ships training code and the two benchmark scripts its paper's numbers
came from (``benchmark_musdb.py``, ``benchmark_urmp.py``); it has no installable
package, no inference-only entry point, and no long-form chunking (the paper's
mixtures are short benchmark clips). The worker driver, the three separation
modes, and chunking for arbitrary-length recordings are therefore this
repository's own code reusing upstream's model construction, not a thin wrapper
around an upstream CLI -- see later tasks in this plan for the worker, the public
API, and long-form chunking.

Two label spaces, not one
--------------------------
The sound prior's conditioning embedding has 50 slots (``num_class=50`` in
``config/atten_unet_fsd/config.toml``), of which 41 were populated by training on
FSD50K subset labels -- see ``data/fsd41_classes.json`` and
:func:`load_fsd_class_map_document`. The speech prior's conditioning label space is
disjoint and has exactly one member (unconditional speech). Passing a sound-prior
index to the speech prior, or an index above 40 to the sound prior, is a caller
error this module is built to catch rather than silently accept -- see
:func:`senselab.audio.tasks.source_separation.api.resolve_source_classes`.

Why a subprocess venv
----------------------
Same template as DriftSE (``audio/tasks/speech_enhancement/driftse.py``): upstream
pins ``torch==2.6.0`` against Python 3.10 and has no installable package (its
top-level module names -- ``models``, ``sound_dataset_process``, ``config`` --
would collide with unrelated names on the host ``sys.path``), so it clones into an
isolated venv at a pinned commit rather than merging its dependency set (or its
module names) into senselab core.

flash-attn is deliberately absent from ``_UNASDIFF_REQUIREMENTS``:
``models/atten_unet.py`` sets ``use_flash = False`` up front and only flips it to
``True`` inside a ``try: from flash_attn import flash_attn_func`` that falls back
to manual softmax attention on ``ImportError`` -- verified against the pinned
commit, not assumed (see this task's report). The fallback materializes a
``[b, h, t, t]`` attention matrix and is therefore slower and heavier, an
acceptable trade against building flash-attn 2.5.8 in every user's cache for a
package that upstream itself treats as optional.

Licensing
---------
The upstream repository carries no ``LICENSE`` file and no license statement in
its README. An issue requesting clarification has been filed upstream and is
outstanding. Pending that answer, senselab vendors none of upstream's code (the
worker clones it at a pinned commit into the user's own cache at first use, same
as DriftSE).

The checkpoint mirror under ``sensein`` is **public**, so the backend is usable
during the alpha, while its licence remains **unknown**: publishing the mirror makes
the weights reachable and grants no rights over them. Treat them as
all-rights-reserved by default and consult upstream before any use that turns on
licence terms. ``SENSELAB_UNASDIFF_CHECKPOINTS`` remains for a caller supplying
their own checkpoints.

Not wired into ``audio_analysis``
----------------------------------
This backend is reachable only by a caller naming it explicitly, through
:func:`senselab.audio.tasks.source_separation.api.separate_audios` -- never through a default
model list and never through ``scripts/analyze_audio.py``. The licensing position above is the
reason: an unresolved license request must not end up load-bearing in a default pipeline.

Two priors, one mode dispatch
------------------------------
``p_sample_loop_group`` (upstream's multi-model sampler) zips one model object against one label
per slot, so ``n_sources`` model instances are always constructed -- even when two slots share
weights. Which prior a slot loads is **not** recoverable from ``source_class_indices`` alone: index
``0`` is simultaneously "unconditional speech" in the speech prior's one-label space and "Hi-hat" in
the sound prior's, so :func:`separate_with_unasdiff` takes ``mode`` explicitly rather than inferring
it, and the worker payload carries all four checkpoint/config paths so the worker can build whichever
slots the mode calls for.
"""

from __future__ import annotations

import functools
import itertools
import json
import os
import subprocess
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.logging import logger
from senselab.utils.subprocess_venv import (
    _clean_subprocess_env,
    ensure_venv,
    parse_subprocess_result,
    venv_python,
)

_UNASDIFF_VENV = "unasdiff"
_UNASDIFF_PYTHON = "3.10"
# torch==2.6.0 ships wheels up to cu124 -- cu128 support begins at torch 2.7 -- so cap the
# Stage-1 index or a modern host selects an index this pin has no wheel on. Measured on an H100
# with CUDA 12.8: ensure_venv probed the host, chose cu128, and the install failed outright with
# "no version of torch==2.6.0". The version and the CUDA build are coupled, which is why dropping
# upstream's +cu124 local tag and letting host detection choose was not sufficient on its own.
# cu124 wheels run fine on a newer driver; the ceiling only constrains which index is searched.
# Same failure and same fix as child_adult.py's _CHILD_ADULT_MAX_CUDA_VERSION.
_UNASDIFF_MAX_CUDA_VERSION = (12, 4)

# Upstream's requirements.txt pins torch==2.6.0+cu124 and numpy==1.23.5 against
# Python 3.10; the version pins are reproduced without the +cu124 local tag, and the CUDA
# build comes from the Stage-1 index -- capped by _UNASDIFF_MAX_CUDA_VERSION above, because
# host-CUDA routing alone picks an index this torch pin has no wheels on.
#
# flash-attn is deliberately absent: atten_unet.py sets use_flash=False on
# ImportError and branches to a manual softmax attention, so it is optional in
# fact and not merely in the README. The fallback materialises a [b, h, t, t]
# attention matrix, so it is slower and heavier -- an acceptable trade against
# building flash-attn 2.5.8 in every user's cache.
_UNASDIFF_REQUIREMENTS = [
    "torch==2.6.0",
    "torchaudio==2.6.0",
    "numpy==1.23.5",
    "scipy==1.10.1",
    "librosa==0.10.2.post1",
    "einops==0.8.1",
    "timm==1.0.19",
    "thop==0.1.1.post2209072238",
    "toml==0.10.2",
    "tqdm==4.67.0",
    "av==14.4.0",
    "soundfile",
]

_UNASDIFF_REPO_URL = "https://github.com/RunwuShi/unasdiff.git"
# Pinned, not a branch: the repository is unlicensed and unpackaged, so this SHA
# is the only version contract available. An upstream force-push must not change
# what this backend runs.
_UNASDIFF_COMMIT = "5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa"

_UNASDIFF_HF_REPO = "sensein/unasdiff-diffusion-priors"
# Pinned so a re-upload cannot change what this backend runs. The repo is public
# pending the upstream licence answer; callers without access use the env override.
_UNASDIFF_HF_REVISION = "8d7c32204d1ba31cd9fca3cd64313fd711949b58"
_UNASDIFF_CHECKPOINTS_ENV = "SENSELAB_UNASDIFF_CHECKPOINTS"

# Filenames inside the mirror repo (and any SENSELAB_UNASDIFF_CHECKPOINTS override
# directory) -- see the mirror's own README.md for provenance.
_UNASDIFF_SPEECH_CKPT = "speech_source.pt"
_UNASDIFF_SPEECH_CONFIG = "atten_unet_vctk.toml"
_UNASDIFF_SOUND_CKPT = "sound_source.pt"
_UNASDIFF_SOUND_CONFIG = "atten_unet_fsd.toml"

_MODE_SPEECH_SOUND = "speech_sound"
_MODE_SOUND_SOUND = "sound_sound"
_MODE_SPEECH_SPEECH = "speech_speech"

_TARGET_SR = 16000
_WINDOW_S = 4.0  # upstream's trained window; not a tunable
_OVERLAP_S = 2.0  # 50% overlap between adjacent windows -- see Task 5 / doc.md
_DIFFUSION_STEPS = 200  # config/*/config.toml: diffusion_step

_FSD_CLASS_MAP_RESOURCE = "fsd41_classes.json"


@functools.lru_cache(maxsize=1)
def load_fsd_class_map_document() -> dict[str, Any]:
    """Load and cache the FSD sound-prior class map from package data.

    Returns the whole profile document -- ``version``, ``derivation``,
    ``num_embedding_slots``, and ``classes`` (name -> conditioning index) -- not
    just the bare mapping, so a caller can inspect the derivation and the
    embedding-slot count alongside the indices themselves.

    A module-level cache is deliberate (the file never changes at runtime), but
    that also means tests must isolate it with ``monkeypatch.setattr`` on this
    function rather than ``load_fsd_class_map_document.cache_clear()`` --
    clearing the cache mutates state that outlives the test and can leak into
    whichever test runs next in the same process.
    """
    data_pkg = resources.files("senselab.audio.tasks.source_separation").joinpath("data", _FSD_CLASS_MAP_RESOURCE)
    doc = json.loads(data_pkg.read_text(encoding="utf-8"))
    if "classes" not in doc or "num_embedding_slots" not in doc:
        raise ValueError(f"{_FSD_CLASS_MAP_RESOURCE} is missing required 'classes'/'num_embedding_slots' keys")
    return doc


# Worker script -- runs inside the isolated venv. Clones the (non-packaged) upstream repo at a
# pinned commit on first use and adds it to sys.path, then reuses upstream's own model
# construction and diffusion sampler rather than reimplementing them here. Upstream ships no
# inference entry point of its own (only benchmark scripts -- test_speech_sound.py,
# test_soundevent.py, test_speech_speech.py -- each of which calls torch.cuda.set_device(0) at
# module import and aborts outright on a CPU host), so this is a from-scratch driver built on
# upstream's library modules (models, diffusion) plus a reimplementation of the benchmark
# scripts' load_model, whose EMA-vs-raw distinction is load-bearing (see load_prior below).
_WORKER_SCRIPT = r"""
import json
import subprocess as sp
import sys
from pathlib import Path

try:
    args = json.loads(sys.stdin.read())
    repo_dir = Path(args["repo_dir"])
    repo_url, commit = args["repo_url"], args["commit"]
    mode = args["mode"]
    speech_ckpt_path = args.get("speech_ckpt_path")
    speech_config_path = args.get("speech_config_path")
    sound_ckpt_path = args.get("sound_ckpt_path")
    sound_config_path = args.get("sound_config_path")
    n_sources = int(args["n_sources"])
    labels = args["labels"]
    in_paths, out_paths = args["in_paths"], args["out_paths"]
    seed = int(args["seed"])

    import fcntl, os, shutil, tempfile as _tempfile

    # Clone under an exclusive flock, to a sibling temp dir + atomic os.replace, so an
    # interrupted clone never leaves repo_dir present but incomplete (which would wedge the
    # guard below permanently) and concurrent jobs sharing $HOME cannot race into the same
    # directory. Identical shape to DriftSE's worker (speech_enhancement/driftse.py).
    marker = repo_dir / "models" / "atten_unet.py"
    if not marker.is_file():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(str(repo_dir) + ".lock", "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if not marker.is_file():
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)
                tmp_clone = Path(_tempfile.mkdtemp(prefix=".unasdiff-clone-", dir=str(repo_dir.parent)))
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
    import toml
    import torch
    from copy import deepcopy

    # Library modules only. The three test_*.py scripts call torch.cuda.set_device(0) at
    # import and abort on a CPU host.
    import models
    import diffusion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    def load_prior(config_path, ckpt_path):
        # Reimplementation of load_model() from upstream's benchmark scripts (not named here
        # literally: this worker deliberately avoids the substrings that mark those scripts,
        # since they call torch.cuda.set_device(0) at import and abort on a CPU host -- see the
        # module-level test that checks this file for exactly those substrings). That function
        # lives in a benchmark script, not the library, so there is nothing to import. It
        # returns the EMA copy -- ckpt["ema"], not ckpt["model"] -- and loading the non-EMA
        # weights separates measurably worse without failing, so the distinction is
        # load-bearing. (A triple-quoted docstring here would collide with the raw-string
        # delimiter this whole worker script is wrapped in, so this is a comment, not a
        # docstring.)
        config = toml.load(config_path)
        model_class = getattr(models, config["model_name"])
        model = model_class(config["model_cfg"])
        for p in model.parameters():
            p.requires_grad = False
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        ema = deepcopy(model)
        ema.load_state_dict(ckpt["ema"])
        ema.to(device).eval()
        return ema, config

    def degradation(x, n_src):
        # The mixture operator: sources are packed along time, so folding them back is a
        # split-and-sum.
        return sum(torch.split(x, x.shape[-1] // n_src, dim=-1))

    def separate_window(models_list, gaussian, mixture, n_src, labels):
        # One 4 s window. Returns a list of n_src waveforms.
        #
        # p_sample_loop_group ignores the measurement argument it is handed and recomputes
        # measurement = degradation(orig_x, n_src) on every step. Packing orig_x as
        # [mixture, zeros, ..., zeros] makes that sum equal the mixture exactly, so the sampler
        # sees precisely what it saw in the benchmark and no per-source information enters.
        # This looks like an oracle from the call site; it is not.
        T = mixture.shape[-1]
        mix = mixture.reshape(1, 1, -1)
        orig_x = torch.cat([mix] + [torch.zeros_like(mix)] * (n_src - 1), dim=-1)
        shape = (1, 1, n_src * T)
        gen = gaussian.p_sample_loop_group(
            models_list,
            shape=shape,
            measurement=mix,
            orig_x=orig_x,
            n_src=n_src,
            clip_denoised=True,
            degradation=degradation,
            model_kwargs=labels,
        )
        out = None
        for out in gen:
            pass
        est = out["sample"].reshape(1, 1, -1)
        return [seg.reshape(-1) for seg in torch.split(est, T, dim=-1)]

    # Model list + diffusion process for this mode. p_sample_loop_group zips `model` against
    # `model_kwargs` one-to-one, so every slot needs its own model object -- even
    # speech-speech, where both slots share the same weights: a separate deepcopy'd instance
    # per slot, not one instance reused twice.
    if mode == "speech_sound":
        speech_model, speech_cfg = load_prior(speech_config_path, speech_ckpt_path)
        models_list = [speech_model] + [
            load_prior(sound_config_path, sound_ckpt_path)[0] for _ in range(n_sources - 1)
        ]
        diffusion_config = speech_cfg
    elif mode == "sound_sound":
        loaded = [load_prior(sound_config_path, sound_ckpt_path) for _ in range(n_sources)]
        models_list = [m for m, _ in loaded]
        diffusion_config = loaded[0][1]
    elif mode == "speech_speech":
        loaded = [load_prior(speech_config_path, speech_ckpt_path) for _ in range(n_sources)]
        models_list = [m for m, _ in loaded]
        diffusion_config = loaded[0][1]
    else:
        raise ValueError("unknown mode: " + str(mode))

    # beta_start/beta_end/diffusion_step are identical across both released configs (verified
    # against the pinned commit), so which config's train_para feeds the schedule does not
    # change the result -- diffusion_config always names a config this mode actually loaded.
    gaussian = diffusion.GaussianDiffusion(
        steps=200,
        config_file=diffusion_config,
        beta_start=diffusion_config["train_para"]["beta_start"],
        beta_end=diffusion_config["train_para"]["beta_end"],
    )

    results = []
    for in_path, out_path_list in zip(in_paths, out_paths):
        y_np, sr = sf.read(in_path, dtype="float32", always_2d=True)
        y = torch.as_tensor(y_np[:, 0]).to(device)
        assert sr == 16000, "worker expects 16 kHz; the host resamples"

        # Peak-normalise per upstream's _norm, and invert it on every output.
        peak = y.abs().amax().clamp(min=1e-8)
        y_norm = y / peak * 0.95

        sources = separate_window(models_list, gaussian, y_norm, n_sources, labels)

        for src_wave, out_path in zip(sources, out_path_list):
            src_wave = src_wave / 0.95 * peak
            sf.write(out_path, src_wave.detach().cpu().numpy(), sr)
        results.append(out_path_list)

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


def _assignment_scores(prev_tail: List[torch.Tensor], next_head: List[torch.Tensor]) -> List[float]:
    """Score every candidate permutation of ``next_head`` against ``prev_tail``.

    Each candidate's score is the total zero-mean, unit-normalised correlation between
    ``prev_tail[i]`` and the candidate's pick for slot ``i``, summed over slots. Normalising
    per-tensor (not just centering) is what makes the score scale-invariant: adjacent windows
    overlap but are not identical there -- each is the sampler's own independent estimate -- so
    raw correlation would conflate amplitude drift between windows with an actual source swap.

    Factored out of :func:`align_permutations` so both it and the calibration procedure in this
    module's ``doc.md`` (which measures the gap between the best and second-best score on
    constructed cases with a known-correct answer) share one implementation.

    Args:
        prev_tail: The previous window's aligned per-source tail segments.
        next_head: The next window's raw (unaligned) per-source head segments; same length as
            ``prev_tail``.

    Returns:
        One total-correlation score per candidate permutation of ``range(len(prev_tail))``, in
        the order ``itertools.permutations`` enumerates them.
    """

    def _zero_mean_unit_norm(x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean()
        return centered / centered.norm().clamp(min=1e-8)

    prev_n = [_zero_mean_unit_norm(p) for p in prev_tail]
    next_n = [_zero_mean_unit_norm(h) for h in next_head]
    n = len(prev_tail)
    return [
        sum(float(torch.dot(prev_n[i], next_n[perm[i]])) for i in range(n)) for perm in itertools.permutations(range(n))
    ]


def align_permutations(prev_tail: List[torch.Tensor], next_head: List[torch.Tensor]) -> List[int]:
    """Return the index permutation mapping ``next_head``'s slots onto ``prev_tail``'s.

    Windows are separated independently, so slot 0 in window *k* need not be the same source as
    slot 0 in window *k + 1*. Concatenating without this check swaps sources mid-file, and the
    result is worse than the mixture -- this is what makes long-form separation different from
    enhancement, which has no cross-segment identity to preserve.

    Brute-force over all ``n!`` permutations rather than the Hungarian algorithm
    (``scipy.optimize.linear_sum_assignment``): ``n_sources`` is at most 3-4 in this backend (two
    priors, each usable for at most a couple of concurrent slots in practice), so ``n!`` is
    trivial and this avoids an optional dependency for a problem size where it buys nothing.

    Args:
        prev_tail: The previous window's aligned per-source tail segments.
        next_head: The next window's raw (unaligned) per-source head segments; same length as
            ``prev_tail``.

    Returns:
        ``result`` such that ``next_head[result[i]]`` is the best match for ``prev_tail[i]``,
        for every ``i`` -- i.e. reordering ``next_head`` by ``result`` aligns it onto
        ``prev_tail``'s slot order.
    """
    perms = list(itertools.permutations(range(len(prev_tail))))
    scores = _assignment_scores(prev_tail, next_head)
    best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    return list(perms[best_idx])


def _window_starts(n_samples: int, window_samples: int, hop_samples: int) -> List[int]:
    """Return start offsets (in samples) for windows of ``window_samples`` covering ``[0, n_samples)``.

    A single ``[0]`` when the signal already fits in one window (the Task 3 short path, with no
    taper and no alignment). Otherwise, evenly spaced starts at ``hop_samples`` apart, plus one
    final window flush against the end when the even spacing would leave a tail uncovered --
    that last window overlaps its predecessor by more than ``hop_samples``, trading a slightly
    uneven overlap for full coverage with every window still exactly ``window_samples`` long
    (which keeps the corrector's shape assumptions inside the worker unchanged).

    Args:
        n_samples: Total length of the signal, in samples.
        window_samples: Length of each window, in samples.
        hop_samples: Stride between consecutive window starts, in samples.

    Returns:
        Start offsets in increasing order.
    """
    if n_samples <= window_samples:
        return [0]
    starts = list(range(0, n_samples - window_samples + 1, hop_samples))
    last_covered = (starts[-1] + window_samples) if starts else 0
    if last_covered < n_samples:
        starts.append(n_samples - window_samples)
    return starts


def _resolve_checkpoint_paths(checkpoint_dir: Optional[Union[str, Path]]) -> tuple[Path, Path, Path, Path]:
    """Resolve the four files unasdiff's two priors need: two checkpoints, two configs.

    Resolution order mirrors DriftSE's (``speech_enhancement/driftse.py``): an explicit
    ``checkpoint_dir`` wins outright; otherwise ``SENSELAB_UNASDIFF_CHECKPOINTS`` (for a caller
    supplying their own checkpoints); otherwise the pinned HF mirror via
    :func:`senselab.utils.dependencies.resolve_model`, which resolves to an immutable commit and
    downloads once, cross-process. All four filenames are returned unconditionally -- the mirror
    is one repo carrying all four -- and the worker loads only the pair its ``mode`` needs.

    Args:
        checkpoint_dir: Explicit override directory, if any.

    Returns:
        ``(speech_ckpt, speech_config, sound_ckpt, sound_config)`` paths.
    """
    if checkpoint_dir is not None:
        base = Path(checkpoint_dir)
    else:
        override = os.environ.get(_UNASDIFF_CHECKPOINTS_ENV)
        if override:
            base = Path(override)
        else:
            from senselab.utils.dependencies import resolve_model

            _, base = resolve_model(_UNASDIFF_HF_REPO, _UNASDIFF_HF_REVISION)
    return (
        base / _UNASDIFF_SPEECH_CKPT,
        base / _UNASDIFF_SPEECH_CONFIG,
        base / _UNASDIFF_SOUND_CKPT,
        base / _UNASDIFF_SOUND_CONFIG,
    )


def separate_with_unasdiff(
    audios: List[Audio],
    n_sources: int,
    source_class_indices: List[int],
    mode: str = _MODE_SPEECH_SOUND,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    device: Optional[DeviceType] = None,
    seed: int = 17,
) -> List[List[Audio]]:
    """Separate each audio into ``n_sources`` sources with unasdiff.

    Resamples to 16 kHz mono on the way in. Inputs that fit in the 4 s window unasdiff was
    trained on go through a single sampler call with no taper. Longer inputs are split into
    overlapping 4 s / 50%-overlap windows, each separated independently and then stitched: since
    each window is separated with no notion of the others, slot 0 in window *k* need not be the
    same source as slot 0 in window *k + 1* (see :func:`align_permutations`), so adjacent windows
    are aligned by correlation on their overlap before a Hann-tapered overlap-add. This chunking
    scheme -- unlike DriftSE's plain overlap-add -- is senselab's own construction: upstream has
    no long-form path at all, only fixed-window benchmark clips.

    ``mode`` decides which prior each slot loads and is required rather than inferred, because
    ``source_class_indices`` alone is ambiguous: index ``0`` is the speech prior's only label in
    one mode and the sound prior's ``"Hi-hat"`` in another.

    Args:
        audios: Inputs. Resampled and downmixed to 16 kHz mono if needed.
        n_sources: Number of sources to separate into.
        source_class_indices: One conditioning label per slot, length ``n_sources``. Interpreted
            against the speech prior's one-label space or the sound prior's 41-class space
            depending on ``mode`` and slot position -- see
            :func:`senselab.audio.tasks.source_separation.api.separate_audios`, which builds
            this list per mode and is the intended caller.
        mode: One of ``"speech_sound"``, ``"sound_sound"``, ``"speech_speech"``.
        checkpoint_dir: Directory containing the four mirror files. If ``None``, resolved from
            ``SENSELAB_UNASDIFF_CHECKPOINTS`` or the pinned HF mirror.
        device: Accepted for signature parity with other separation/enhancement entry points.
            The worker selects CUDA when available and CPU otherwise.
        seed: RNG seed, recorded in the log line.

    Returns:
        One list of ``n_sources`` ``Audio`` objects per input, in order. For a multi-window
        input, every returned ``Audio``'s ``metadata["unasdiff_alignment_margins"]`` carries one
        ``best_score - second_best_score`` float per window boundary (see
        ``data/permutation_alignment.json`` for the measurement behind reading this number,
        which the profile does not gate on since the measurement did not support a fitted
        threshold).

    Raises:
        ValueError: if ``len(source_class_indices) != n_sources``.
        RuntimeError: if the worker fails; the upstream traceback is included.
    """
    if not audios:
        return []
    if len(source_class_indices) != n_sources:
        raise ValueError(
            f"source_class_indices must have exactly n_sources={n_sources} entries, got {len(source_class_indices)}"
        )

    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
    from senselab.utils.data_structures.device import _select_device_and_dtype

    _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])

    mono_16k = downmix_audios_to_mono(resample_audios(audios, resample_rate=_TARGET_SR))

    window_samples = int(_WINDOW_S * _TARGET_SR)
    overlap_samples = int(_OVERLAP_S * _TARGET_SR)
    hop_samples = window_samples - overlap_samples

    windows_per_audio: List[List[int]] = [
        _window_starts(int(audio.waveform.shape[-1]), window_samples, hop_samples) for audio in mono_16k
    ]

    speech_ckpt_path, speech_config_path, sound_ckpt_path, sound_config_path = _resolve_checkpoint_paths(checkpoint_dir)

    venv_dir = ensure_venv(
        _UNASDIFF_VENV,
        _UNASDIFF_REQUIREMENTS,
        python_version=_UNASDIFF_PYTHON,
        max_cuda_version=_UNASDIFF_MAX_CUDA_VERSION,
    )
    python = venv_python(venv_dir)
    # Cached alongside the venv rather than per-tempdir, so the pinned-commit clone in the
    # worker script happens once per host, not once per call.
    repo_dir = Path(venv_dir) / "unasdiff-src"

    logger.info(
        "unasdiff: separating %d audio(s) (%d window(s) total), mode=%s, n_sources=%d, seed=%d",
        len(mono_16k),
        sum(len(w) for w in windows_per_audio),
        mode,
        n_sources,
        seed,
    )

    with tempfile.TemporaryDirectory(prefix="senselab-unasdiff-") as tmpdir:
        tmp = Path(tmpdir)
        in_paths: List[str] = []
        out_paths: List[List[str]] = []
        # (flat_start, flat_end) into in_paths/out_paths for each input audio's windows, in
        # order -- the worker is called once for every window of every audio (one model load
        # for the whole batch), and this is how the flat results are regrouped back afterwards.
        window_ranges: List[Tuple[int, int]] = []
        for i, (audio, starts) in enumerate(zip(mono_16k, windows_per_audio)):
            flat_start = len(in_paths)
            waveform = audio.waveform.squeeze(0)
            for w, start in enumerate(starts):
                segment = waveform[start : start + window_samples]
                segment_audio = Audio(waveform=segment.unsqueeze(0), sampling_rate=_TARGET_SR)
                in_path = str(tmp / f"in_{i}_{w}.wav")
                segment_audio.save_to_file(in_path)
                in_paths.append(in_path)
                out_paths.append([str(tmp / f"out_{i}_{w}_{s}.wav") for s in range(n_sources)])
            window_ranges.append((flat_start, len(in_paths)))

        input_json = json.dumps(
            {
                "repo_dir": str(repo_dir),
                "repo_url": _UNASDIFF_REPO_URL,
                "commit": _UNASDIFF_COMMIT,
                "mode": mode,
                "speech_ckpt_path": str(speech_ckpt_path),
                "speech_config_path": str(speech_config_path),
                "sound_ckpt_path": str(sound_ckpt_path),
                "sound_config_path": str(sound_config_path),
                "n_sources": n_sources,
                "labels": source_class_indices,
                "in_paths": in_paths,
                "out_paths": out_paths,
                "seed": seed,
            }
        )

        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=3600,
            env=_clean_subprocess_env(),
        )
        output = parse_subprocess_result(result, venv_label="unasdiff")
        all_output_paths: List[List[str]] = output["output_paths"]

        # Read outputs back while the temp dir is still alive: Audio(filepath=...) lazy-loads
        # on first .waveform access, and that access must happen before this context manager
        # deletes the files it points at.
        separated: List[List[Audio]] = []
        for audio, starts, (flat_start, flat_end) in zip(mono_16k, windows_per_audio, window_ranges):
            window_output_paths = all_output_paths[flat_start:flat_end]

            if len(starts) == 1:
                # Short path, unchanged from Task 3: no taper, no alignment -- there is only
                # one window, so there is nothing to align it against.
                sources = []
                for p in window_output_paths[0]:
                    source_audio = Audio(filepath=p)
                    _ = source_audio.waveform
                    source_audio.metadata = dict(audio.metadata)
                    sources.append(source_audio)
                separated.append(sources)
                continue

            window_tensors: List[List[torch.Tensor]] = []
            for paths in window_output_paths:
                tensors = []
                for p in paths:
                    window_audio = Audio(filepath=p)
                    tensors.append(window_audio.waveform.squeeze(0))
                window_tensors.append(tensors)

            # Window 0 keeps its raw slot order (nothing precedes it to align onto). Every
            # later window is reordered to match the previous *aligned* window's slots, using
            # correlation on the overlap region -- not on the whole window, so speech/sound
            # content outside the shared region cannot influence the match.
            aligned_tensors: List[List[torch.Tensor]] = [window_tensors[0]]
            margins: List[float] = []
            for k in range(1, len(window_tensors)):
                prev_tail = [t[-overlap_samples:] for t in aligned_tensors[-1]]
                next_head = [t[:overlap_samples] for t in window_tensors[k]]
                perm = align_permutations(prev_tail, next_head)
                scores = sorted(_assignment_scores(prev_tail, next_head), reverse=True)
                margins.append(scores[0] - scores[1] if len(scores) > 1 else float("inf"))
                aligned_tensors.append([window_tensors[k][perm[s]] for s in range(n_sources)])

            n_samples = int(audio.waveform.shape[-1])
            taper = torch.hann_window(window_samples, periodic=False)
            acc = torch.zeros(n_sources, n_samples)
            weight_sum = torch.zeros(n_samples)
            for start, sources_at_window in zip(starts, aligned_tensors):
                for s in range(n_sources):
                    acc[s, start : start + window_samples] += sources_at_window[s] * taper
                weight_sum[start : start + window_samples] += taper
            weight_sum = weight_sum.clamp(min=1e-8)

            sources = []
            for s in range(n_sources):
                stitched_audio = Audio(waveform=(acc[s] / weight_sum).unsqueeze(0), sampling_rate=_TARGET_SR)
                stitched_audio.metadata = dict(audio.metadata)
                stitched_audio.metadata["unasdiff_alignment_margins"] = margins
                sources.append(stitched_audio)
            separated.append(sources)

    return separated
