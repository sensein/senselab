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
as DriftSE) and the checkpoint mirror under ``sensein`` is private; a caller
without access to the mirror uses the ``SENSELAB_UNASDIFF_CHECKPOINTS`` override.

Not wired into ``audio_analysis``
----------------------------------
This backend is reachable only by a caller naming it explicitly, once the public
API lands in a later task -- never through a default model list and never through
``scripts/analyze_audio.py``. The licensing position above is the reason: an
unresolved license request must not end up load-bearing in a default pipeline.
"""

from __future__ import annotations

import functools
import json
from importlib import resources
from typing import Any

_UNASDIFF_VENV = "unasdiff"
_UNASDIFF_PYTHON = "3.10"

# Upstream's requirements.txt pins torch==2.6.0+cu124 and numpy==1.23.5 against
# Python 3.10; the pins are reproduced (minus the +cu124 local tag, which
# ensure_venv supplies by routing Stage 1 through the index matching the host's
# CUDA). flash-attn is deliberately absent: atten_unet.py sets use_flash=False on
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
# Pinned so a re-upload cannot change what this backend runs. The repo is private
# pending the upstream licence answer; callers without access use the env override.
_UNASDIFF_HF_REVISION = "8d7c32204d1ba31cd9fca3cd64313fd711949b58"
_UNASDIFF_CHECKPOINTS_ENV = "SENSELAB_UNASDIFF_CHECKPOINTS"

_TARGET_SR = 16000
_WINDOW_S = 4.0  # upstream's trained window; not a tunable
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
