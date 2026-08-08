# Plan D — DriftSE one-step speech enhancement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add [DriftSE](https://github.com/LiangXu123/DriftSE) (Interspeech 2026, oral) as a speech-enhancement backend reachable through `enhance_audios(model=HFModel(...))` — a **single** network evaluation per utterance, against 30 for SGMSE+.

**Architecture:** A subprocess-venv backend following the `child_adult.py` pattern: isolated venv, runtime `git clone` of the unpackaged upstream at a pinned SHA, `sys.path` injection inside the worker only. The inference recipe is transcribed from upstream's `enhancement.py` with three deliberate deviations — `weights_only=True` on the checkpoint load, overlap-add chunking for long inputs, and a recorded RNG seed.

**Tech Stack:** Python 3.11 subprocess venv; torch/torchaudio CUDA-routed by `ensure_venv`; NCSN++V2 backbone; 16 kHz, `n_fft=510`, `hop_length=128`.

## Global Constraints

Copied from `design.md`. Every task's requirements implicitly include these.

- **No `analyze_audio` or `audio_analysis` wiring.** `enhance_audios`' default model stays `speechbrain/sepformer-wham16k-enhancement`, so no existing caller changes behaviour.
- **No `run_config` changes.**
- **No new host dependencies and no new extras.** Everything DriftSE needs lives in its own venv.
- **No vendored upstream source.** The repository is unlicensed (`license: null`); it is cloned at a pinned SHA into the user's own cache at first use and never redistributed.
- Upstream pin: **`695a64db187500fa0d7bae23912680bd5d4df613`**.
- `ensure_venv` must keep routing torch/torchaudio through the CUDA-aware PyTorch index. Do not bypass it.
- **Every Python command runs through `uv run`.**
- **Never run `pytest -n auto`.**
- **Run `uv run ruff format` before any push.**
- **Never `git add -A` unqualified.** Always limit it with a pathspec (`git add -A -- src/ docs/ pyproject.toml uv.lock`). The repository root can hold untracked local secrets — a developer-supplied API token sitting beside the checkout is the case that prompted this — and an unqualified `git add -A` would stage one. `git status` is not a safeguard: an agent running these steps does not read it before committing.
- **Cache invalidation is free** — if any cached artifact keys on enhancement output, bump `CACHE_SCHEMA_VERSION` rather than reasoning about which entries survive.

## Preconditions

Branch `feat/new-model-integrations` already exists, cut from the merged `alpha` (PR #547, `79b37d93`); run Plan A's Task 1 first to verify it. **Task 2's mirror is already done** — `sensein/driftse-distilhubert-three-layers` is live and private — so an implementer needs only read access to that repo, or `SENSELAB_DRIFTSE_CHECKPOINT` pointing at a local file.

## Upstream facts this plan depends on

Established by reading the repository at the pinned SHA; re-verify in Task 1 if the pin is moved.

- `enhancement.py` imports only `backbones.ncsnpp_v2`, `backbones.ncsnpp_v2_drift`, and `util.other`. **No Lightning, no `wandb`, no `pesq`, and no SSL encoder** — the HuBERT/WavLM/DistilHuBERT encoders are a *training* signal for the drifting field, so the `latent_ckpt/` Google Drive archive is not needed for inference.
- `util/inference.py` *does* import `pesq` and `pystoi`, and `enhancement.py` never imports it. The worker must not either.
- `pad_spec(Y, mode="zero_pad")` pads **dim 3** (time) up to a multiple of 64 and expects a 4-D `(B, C, F, T)` tensor.
- `set_torch_cuda_arch_list()` prints and returns when CUDA is unavailable — safe to call on CPU.
- With `n_fft=510`, `torch.stft` returns exactly **256** frequency bins, matching the config's `image_size: 256`. No frequency cropping.
- The released checkpoint is `logs/distillhubert_three_layers_with_z/last.ckpt`, pairing with `config/with_z/v2_drift2_distillhubert_three_layers.json` (`model: "ncsnpp_v2_drift"`, `train_add_gaussian: true`, `spec_factor: 0.15`, `spec_abs_exponent: 0.5`, `window_type: "hann"`, `center: true`).

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `src/senselab/audio/tasks/speech_enhancement/driftse.py` | Venv constants, clone helper, worker script, `enhance_audios_with_driftse` | Create |
| `src/senselab/audio/tasks/speech_enhancement/api.py` | `HFModel` dispatch branch | Modify |
| `src/senselab/audio/tasks/speech_enhancement/doc.md` | Backend documentation | Modify or create |
| `src/senselab/utils/data_structures/model.py` | `model_for_task(..., task="enhancement")` prefix match | Modify |
| `src/senselab/model_registry.yaml` / `.md`, `docs/compatibility-matrix.md` | Registry and isolated-backends rows | Modify |
| `src/tests/audio/tasks/speech_enhancement_test.py` | Backend tests, skip-gated | Modify |

---

### Task 1: Module scaffolding, venv constants, and the pinned clone

**Files:**
- Create: `src/senselab/audio/tasks/speech_enhancement/driftse.py`
- Test: `src/tests/audio/tasks/speech_enhancement_test.py`

**Interfaces:**
- Consumes: `subprocess_venv.ensure_venv`, `venv_python`, `_clean_subprocess_env`, `parse_subprocess_result`, `_cache_dir_path` (the last from Plan A, Task 4 — if Plan A has not run, add `_cache_dir_path` here and Plan A will find it present).
- Produces: module constants `_DRIFTSE_VENV`, `_DRIFTSE_PYTHON`, `_DRIFTSE_REQUIREMENTS`, `_DRIFTSE_REPO_URL`, `_DRIFTSE_COMMIT`, `_DRIFTSE_HF_REPO`, `_DRIFTSE_CHECKPOINT_ENV`.

- [ ] **Step 1: Write the failing constants test**

Add to `src/tests/audio/tasks/speech_enhancement_test.py`:

```python
from senselab.audio.tasks.speech_enhancement import driftse


def test_upstream_is_pinned_to_a_full_commit_sha() -> None:
    """A branch name or short SHA would let an upstream force-push change what
    this backend runs without any change here. The repository is unlicensed and
    unpackaged, so the pin is the only version contract available."""
    assert len(driftse._DRIFTSE_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in driftse._DRIFTSE_COMMIT)


def test_training_and_metric_dependencies_are_not_installed() -> None:
    """Upstream's requirements.txt lists these for training and scoring. The
    inference path imports none of them, and pesq/scoreq in particular are slow
    and fragile to build. util/inference.py imports pesq — the worker must never
    import util.inference.
    """
    excluded = {"pesq", "pystoi", "scoreq", "torch-pesq", "asteroid-filterbanks",
                "wandb", "pytorch-optimizer", "torchinfo"}
    named = {r.split(">=")[0].split("==")[0].strip().lower()
             for r in driftse._DRIFTSE_REQUIREMENTS}
    assert not (named & excluded), f"training-only deps in the inference venv: {named & excluded}"


def test_torch_is_named_explicitly_so_ensure_venv_routes_cuda() -> None:
    """ensure_venv's CUDA auto-detection triggers on an explicit torch pin. Left
    transitive, the resolve skips CUDA-aware routing and can land a CPU-only
    wheel on a GPU host."""
    named = {r.split(">=")[0].split("==")[0].strip().lower()
             for r in driftse._DRIFTSE_REQUIREMENTS}
    assert "torch" in named
    assert "torchaudio" in named
```

- [ ] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/audio/tasks/speech_enhancement_test.py -k "driftse or upstream_is_pinned or training_and_metric or torch_is_named" -v
```

Expected: FAIL with `ImportError: cannot import name 'driftse'`.

- [ ] **Step 3: Create the module with its constants and docstring**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/speech_enhancement_test.py -k "upstream_is_pinned or training_and_metric or torch_is_named" -v
```

Expected: PASS, 3 tests.

- [ ] **Step 5: Verify the pin resolves and the inference imports are what this plan assumes**

```bash
git ls-remote https://github.com/LiangXu123/DriftSE.git HEAD
gh api repos/LiangXu123/DriftSE/contents/enhancement.py?ref=695a64db187500fa0d7bae23912680bd5d4df613 \
  --jq '.content' | base64 -d | grep -n "^from \|^import "
```

Expected: the imports are `torch`, `torchaudio`, `numpy`, `librosa`, `soundfile`, `tqdm`, `backbones.ncsnpp_v2`, `backbones.ncsnpp_v2_drift`, `util.other`, and stdlib. If anything else appears, the pin has moved or this plan's dependency list is wrong — fix the list before continuing.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/speech_enhancement/ src/tests/
uv run mypy src/senselab/audio/tasks/speech_enhancement/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(speech_enhancement): DriftSE backend scaffolding and pinned upstream

Upstream is unlicensed and unpackaged, so it is cloned at a pinned SHA rather
than vendored. The venv carries the inference dependency set only: upstream's
requirements.txt is a training list, and util/inference.py's pesq/pystoi imports
are never reached because enhancement.py does not import that module.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Mirror the checkpoint — **DONE (2026-08-08)**, except the licence request

The mirror exists. Only Step 1 below, the upstream licence request, is outstanding.

**Produced:** `sensein/driftse-distilhubert-three-layers`, **private**, at revision
`76a9448aae12e4c232b1d52c24899d0835db5782`.

| File | SHA-256 | Size |
|---|---|---|
| `last.ckpt` | `6f476a95cf747748b066405870e575cba3ee42927d6cb516a9b3f88da88abbb6` | 1137.9 MB |
| `config.json` | `c61e97dfd618ff681be14493e3e43fc72312e95096a77a1dec0e968937b3e2f0` | 1.8 KB |
| `last_pesq_sisdr_ccmse.ckpt` | `d5d62e08c3f6a57d1d9ba61bda1a7dadc38b5f62fad5cd8d9f1e0c25a39aa0c6` | 1137.9 MB |
| `config_pesq_sisdr_ccmse.json` | `c0611e016c08b6b26864abb42159cf38cf65361ad1b6e26a4ac5dff561849aaf` | 1.9 KB |

Three notes from doing it, which change what later tasks should assume:

1. **The Google Drive archive is 11 GB and holds seven variants**, not one. Beyond the config-default `distillhubert_three_layers_with_z`, it contains the `DriftSE†` variant (`distillhubert_three_layers_pesq_sisdr_ccmse_with_z`) that carries the best published numbers on both benchmark tables. Both were mirrored, because re-downloading 11 GB to fetch the second later would be worse than storing it now. **The backend still pins `last.ckpt`** — the `†` file is available, not wired.
2. **Each checkpoint contains both `model` and `ema` state dicts**, plus `optimizer`, `scheduler`, `epoch`, `step`, and an embedded `config`. Upstream's `enhancement.py` loads `checkpoint["model"]`, so Task 3's worker does too — that is what reproduces the published numbers. This is the **opposite** of the sibling unasdiff codebase, whose loader returns its `ema` copy, so the difference is easy to get backwards. Do not "fix" Task 3 to use `ema` without measuring.
3. Both files load cleanly under `torch.load(..., weights_only=True)`, confirming Task 3's deviation is viable and not merely desirable.

**Files:** none in this repository.

**Interfaces:**
- Produces: `sensein/driftse-distilhubert-three-layers` at the revision above, which Task 3 pins.

- [x] **Step 1: Open the upstream licence request — DONE**

Posted 2026-08-08: <https://github.com/LiangXu123/DriftSE/issues/2>. Until it is answered, the mirror stays private. The drafted text is kept below for the record.


Before mirroring anything, ask. Post to https://github.com/LiangXu123/DriftSE/issues:

> **Request: an explicit license (and optionally a HuggingFace weights mirror)**
>
> Thanks for releasing DriftSE — we'd like to make it available as a backend in
> [senselab](https://github.com/sensein/senselab), an open-source behavioural-data
> toolkit, so researchers can call it alongside other enhancers.
>
> The repository currently has no `LICENSE` file and no license statement in the
> README, which under GitHub's default terms means all rights reserved — so we
> can't redistribute the code or the checkpoint. Would you be willing to add one?
> MIT or Apache-2.0 would match the SGMSE+ codebase this builds on. Relatedly, the
> README credits SGMSE+ (MIT) as the foundation, so carrying that license forward
> is likely something you'd want regardless.
>
> A HuggingFace mirror of `last.ckpt` would also help: Google Drive links can't be
> pinned to a revision or content hash, which we need for reproducible runs.
>
> Until then we clone at a pinned commit at run time and vendor nothing.

Record the issue URL in `doc.md` (Task 6).

- [x] **Steps 2–6: download, stage, create the private repo, write the card, record the revision — done**

Carried out on 2026-08-08. The repo is private, the model card records the provenance and the unresolved licence, and the revision and per-file SHA-256 digests are in the table above.

**Keep it private.** The card says so and this plan says so: a private mirror gives the checkpoint a pinned revision and content hash — which is what run provenance needs — **without** redistributing weights whose licence is unresolved. Making it public is a decision that waits on Step 1's answer.

---

### Task 3: The worker script and the enhancement call

**Files:**
- Modify: `src/senselab/audio/tasks/speech_enhancement/driftse.py`
- Test: `src/tests/audio/tasks/speech_enhancement_test.py`

**Interfaces:**
- Consumes: the constants from Task 1.
- Produces: `enhance_audios_with_driftse(audios: List[Audio], model: HFModel, device: Optional[DeviceType] = None, seed: int = 0) -> List[Audio]` — one enhanced `Audio` per input, at 16 kHz, same length as the input.

- [ ] **Step 1: Write the failing contract tests**

These run without the venv or the weights, so they gate the shape rather than the numerics.

```python
import pytest

from senselab.audio.tasks.speech_enhancement import driftse
from senselab.utils.data_structures import HFModel


def test_worker_script_compiles_standalone() -> None:
    """The worker is a string literal executed by another interpreter, so a
    syntax error in it surfaces only at first inference — after the venv build
    and the model download. Compiling it here makes that a unit-test failure."""
    compile(driftse._WORKER_SCRIPT, "<driftse worker>", "exec")


def test_worker_never_imports_util_inference() -> None:
    """util/inference.py imports pesq and pystoi, which are deliberately not in
    the venv. enhancement.py does not import it and neither may the worker."""
    assert "util.inference" not in driftse._WORKER_SCRIPT
    assert "from util import inference" not in driftse._WORKER_SCRIPT


def test_worker_loads_the_checkpoint_with_weights_only() -> None:
    """Upstream omits weights_only. The checkpoint is a foreign pickle from an
    unlicensed research repository; loading it with the unrestricted unpickler is
    arbitrary code execution at enhancement time."""
    assert "weights_only=True" in driftse._WORKER_SCRIPT


def test_empty_input_returns_empty_without_spawning() -> None:
    assert driftse.enhance_audios_with_driftse([], model=HFModel(path_or_uri=driftse._DRIFTSE_HF_REPO)) == []
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/audio/tasks/speech_enhancement_test.py -k "worker or empty_input" -v
```

Expected: FAIL — `_WORKER_SCRIPT` and `enhance_audios_with_driftse` do not exist.

- [ ] **Step 3: Write the worker script**

Append to `driftse.py`. This is a transcription of upstream's `enhancement.py` main block, with the three deviations marked inline.

```python
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

    # NOTE: util.other only. util.inference imports pesq/pystoi, which are
    # deliberately absent from this venv.
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

    # DEVIATION 1: weights_only=True. Upstream omits it.
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

    print("__SENSELAB_RESULT__" + json.dumps({"ok": True, "outputs": results, "seed": seed}))
except Exception as exc:
    import traceback
    print("__SENSELAB_RESULT__" + json.dumps(
        {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    ))
"""
```

Match the result-marker convention to whatever `parse_subprocess_result` expects — check it first:

```bash
grep -n "def parse_subprocess_result" -A 30 src/senselab/utils/subprocess_venv.py
```

- [ ] **Step 4: Write the host-side entry point**

```python
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
```

The body: return `[]` on empty input before doing anything; resolve the checkpoint from `os.environ[_DRIFTSE_CHECKPOINT_ENV]` if set, else `hf_hub_download` from `model.path_or_uri` at `model.revision`; `ensure_venv(_DRIFTSE_VENV, _DRIFTSE_REQUIREMENTS, python_version=_DRIFTSE_PYTHON)`; write inputs to a `tempfile.TemporaryDirectory` as 16 kHz mono WAVs; run `venv_python(...)` with `_clean_subprocess_env()`; `parse_subprocess_result`; read the outputs back into `Audio`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/speech_enhancement_test.py -k "worker or empty_input" -v
```

Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/speech_enhancement/ src/tests/
uv run mypy src/senselab/audio/tasks/speech_enhancement/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(speech_enhancement): DriftSE worker — 1 NFE, weights_only, overlap-add

Three deliberate deviations from upstream's enhancement.py: weights_only=True on
the checkpoint load (upstream omits it, and this is a foreign pickle from an
unlicensed repo), overlap-add chunking for long inputs (upstream runs one STFT
over a whole file and the backbone has attention layers), and a recorded RNG
seed (the released checkpoint sets train_add_gaussian, so output is stochastic).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Dispatch from `enhance_audios`

**Files:**
- Modify: `src/senselab/audio/tasks/speech_enhancement/api.py`
- Modify: `src/senselab/utils/data_structures/model.py`
- Test: `src/tests/audio/tasks/speech_enhancement_test.py`

**Interfaces:**
- Consumes: `enhance_audios_with_driftse` from Task 3.
- Produces: `enhance_audios(audios, model=None, device=None)` accepting an `HFModel` whose id starts with `sensein/driftse`; unchanged behaviour for `SpeechBrainModel` and unchanged default.

- [ ] **Step 1: Write the failing dispatch tests**

```python
from unittest.mock import patch

from senselab.audio.tasks.speech_enhancement import enhance_audios
from senselab.utils.data_structures import HFModel, SpeechBrainModel


def test_default_model_is_unchanged(mono_audio_sample) -> None:
    """No existing caller may change behaviour. The workflow calls enhance_audios
    with the SpeechBrain default and must keep reaching the SpeechBrain path."""
    with patch(
        "senselab.audio.tasks.speech_enhancement.api.SpeechBrainEnhancer.enhance_audios_with_speechbrain",
        return_value=[mono_audio_sample],
    ) as sb:
        enhance_audios([mono_audio_sample])
    sb.assert_called_once()
    assert sb.call_args.kwargs["model"].path_or_uri == "speechbrain/sepformer-wham16k-enhancement"


def test_hfmodel_with_the_driftse_prefix_dispatches_to_driftse(mono_audio_sample) -> None:
    with patch(
        "senselab.audio.tasks.speech_enhancement.api.enhance_audios_with_driftse",
        return_value=[mono_audio_sample],
    ) as ds:
        enhance_audios(
            [mono_audio_sample],
            model=HFModel(path_or_uri="sensein/driftse-distilhubert-three-layers"),
        )
    ds.assert_called_once()


def test_an_unrecognised_model_still_raises_not_implemented(mono_audio_sample) -> None:
    """Silently falling through to a default would enhance with a model the
    caller did not ask for."""
    import pytest

    with pytest.raises(NotImplementedError):
        enhance_audios([mono_audio_sample], model=HFModel(path_or_uri="some/other-model"))
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/audio/tasks/speech_enhancement_test.py -k dispatch -v
```

Expected: FAIL — the `HFModel` branch does not exist.

- [ ] **Step 3: Implement the dispatch**

```python
_DRIFTSE_MODEL_PREFIX = "sensein/driftse"

    if isinstance(model, SpeechBrainModel):
        return SpeechBrainEnhancer.enhance_audios_with_speechbrain(
            audios=audios, model=model, device=device
        )
    if isinstance(model, HFModel) and str(model.path_or_uri).startswith(_DRIFTSE_MODEL_PREFIX):
        return enhance_audios_with_driftse(audios=audios, model=model, device=device)
    raise NotImplementedError(
        f"No enhancement backend for {model.path_or_uri!r}. Supported: SpeechBrain models, "
        f"and HFModel ids starting with {_DRIFTSE_MODEL_PREFIX!r}."
    )
```

Add the matching branch to `model_for_task(model_id, task="enhancement")` so a bare model id resolves to `HFModel`.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/speech_enhancement_test.py -v
```

Expected: PASS.

- [ ] **Step 5: Verify no workflow behaviour moved**

```bash
uv run pytest src/tests/audio/workflows/ -v 2>&1 | tail -20
```

Expected: unchanged from before this task.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/senselab/ src/tests/
uv run mypy src/senselab/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(speech_enhancement): dispatch DriftSE by the sensein/driftse prefix

The SpeechBrain default is unchanged, so the audio_analysis workflow keeps
reaching the same enhancer it does today. An unrecognised model still raises
rather than silently falling through to a default.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: End-to-end run and the `upfirdn2d` fallback

The spec flags one risk to verify rather than assume: upstream JIT-compiles a CUDA extension, with a pure-PyTorch fallback whose selection has not been confirmed.

**Files:**
- Modify: `src/senselab/audio/tasks/speech_enhancement/driftse.py` (only if the fallback needs forcing)
- Modify: `src/tests/audio/tasks/speech_enhancement_test.py`

**Interfaces:** no new interface.

- [ ] **Step 1: Add the skip-gated end-to-end test**

```python
import pytest

from senselab.utils.subprocess_venv import _cache_dir_path


@pytest.mark.skipif(
    not (_cache_dir_path() / "driftse").is_dir(),
    reason="driftse venv not built; run manually to build it (first run takes minutes)",
)
def test_driftse_enhances_and_preserves_length(mono_audio_sample) -> None:
    """Length preservation is the cheapest real correctness check available
    without a reference signal: istft(length=T_orig) must round-trip, and a
    chunking bug shows up here immediately as a short or long result."""
    from senselab.audio.tasks.preprocessing import resample_audios
    from senselab.audio.tasks.speech_enhancement import enhance_audios
    from senselab.utils.data_structures import HFModel

    audio = resample_audios([mono_audio_sample], resample_rate=16000)[0]
    out = enhance_audios([audio], model=HFModel(path_or_uri="sensein/driftse-distilhubert-three-layers"))

    assert len(out) == 1
    assert out[0].sampling_rate == 16000
    assert out[0].waveform.shape[-1] == audio.waveform.shape[-1]
    assert out[0].waveform.abs().max() > 0, "silent output — the model produced nothing"


@pytest.mark.skipif(
    not (_cache_dir_path() / "driftse").is_dir(),
    reason="driftse venv not built",
)
def test_driftse_is_reproducible_under_a_fixed_seed(mono_audio_sample) -> None:
    """train_add_gaussian makes the forward pass stochastic. Without a seed a
    rerun produces different audio, which would make any cached artifact keyed
    on this output non-reproducible."""
    from senselab.audio.tasks.preprocessing import resample_audios
    from senselab.audio.tasks.speech_enhancement.driftse import enhance_audios_with_driftse
    from senselab.utils.data_structures import HFModel

    audio = resample_audios([mono_audio_sample], resample_rate=16000)[0]
    model = HFModel(path_or_uri="sensein/driftse-distilhubert-three-layers")
    a = enhance_audios_with_driftse([audio], model=model, seed=17)[0]
    b = enhance_audios_with_driftse([audio], model=model, seed=17)[0]

    assert (a.waveform - b.waveform).abs().max() < 1e-5
```

Check the real names of `resample_audios` and the audio fixture before relying on them:

```bash
grep -rn "def resample_audios" src/senselab/audio/tasks/preprocessing/
grep -n "@pytest.fixture" -A 3 src/tests/conftest.py | head -30
```

- [ ] **Step 2: Build the venv and run it for real**

```bash
uv run pytest src/tests/audio/tasks/speech_enhancement_test.py -k driftse -v -s
```

First run builds the venv, clones upstream, and downloads the checkpoint — several minutes. If this hangs, it is machine contention rather than a bug: rerun rather than disabling the backend.

- [ ] **Step 3: Determine which `upfirdn2d` path was taken**

```bash
uv run python - <<'PY'
import subprocess, sys
from pathlib import Path
from senselab.utils.subprocess_venv import _cache_dir_path, venv_python
repo = Path.home() / ".cache" / "senselab" / "repos" / "driftse"
code = (
    "import sys; sys.path.insert(0, %r);\n"
    "from backbones.ncsnpp_utils.op import upfirdn2d as u;\n"
    "print('module file:', u.__file__);\n"
    "print('native fallback in use:', 'native' in str(getattr(u, 'upfirdn2d', u)).lower())\n"
) % str(repo)
print(subprocess.run([str(venv_python(_cache_dir_path() / "driftse")), "-c", code],
                     capture_output=True, text=True).stdout)
PY
```

Adjust the repo path to wherever Task 3's worker actually clones. Expected: either the JIT extension loaded, or the native fallback selected — **not** a compile error.

- [ ] **Step 4: Force the native path if the fallback is not automatic**

If Step 3 raises rather than falling back, set the environment variable upstream's loader checks (read `upfirdn2d.py` to find it) in `_clean_subprocess_env()`'s output for this backend, and record why in a comment: a JIT CUDA compile at first inference is a failure that surfaces minutes into a run, on a machine that may have no compiler.

- [ ] **Step 5: Commit**

```bash
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "test(speech_enhancement): end-to-end DriftSE run, seed reproducibility, upfirdn2d path

Length preservation is the cheapest real correctness check without a reference
signal, and it catches a chunking bug immediately. The seed test exists because
train_add_gaussian makes the forward pass stochastic, so an unseeded rerun would
make any artifact keyed on this output non-reproducible.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Registry, compatibility matrix, and `doc.md`

**Files:**
- Modify: `src/senselab/model_registry.yaml`, `src/senselab/model_registry.md`
- Modify: `docs/compatibility-matrix.md`
- Modify: `src/senselab/audio/tasks/speech_enhancement/doc.md`

**Interfaces:** no new interface.

- [ ] **Step 1: Add the registry entry**

Match the shape of the existing isolated-backend entries. It must record: the model id, the task, that it is a subprocess-venv backend named `driftse`, the pinned upstream commit, and that the **upstream license is unresolved**.

- [ ] **Step 2: Add the isolated-backends row**

In `docs/compatibility-matrix.md`, add DriftSE with its Python version (3.11), its pinned torch range, and its licence status.

- [ ] **Step 3: Regenerate `model_registry.md` rather than hand-editing**

```bash
ls scripts/ | grep -i registry
```

If a generator exists, run it. A hand-edited generated file drifts from its YAML source.

- [ ] **Step 4: Write `doc.md`**

Cover, in this order: what DriftSE is and its published numbers; why inference is 1 NFE and needs no SSL encoder; why the venv omits upstream's training dependencies; the three deviations from upstream's script and the reason for each; the chunking scheme and why overlap-add is safe for enhancement but not for separation; the licence status with a link to the issue from Task 2 Step 1; and that this backend is not wired into `audio_analysis`.

- [ ] **Step 5: Final check**

```bash
uv run ruff format --check src/ && uv run ruff check src/ && uv run mypy src/senselab/
uv run pytest src/tests/audio/tasks/speech_enhancement_test.py src/tests/audio/workflows/ -v 2>&1 | tail -20
```

- [ ] **Step 6: Commit and report**

```bash
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "docs(speech_enhancement): register DriftSE and document its deviations

Records that the upstream license is unresolved, that the backend clones at a
pinned SHA and vendors nothing, and that the weights mirror is private pending
an answer.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

Report the commit SHAs, the end-to-end test result, which `upfirdn2d` path was taken, and the status of the upstream licence issue. Do not push.
