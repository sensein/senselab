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
