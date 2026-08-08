# Plan C — unasdiff single-channel source separation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add [unasdiff](https://github.com/RunwuShi/unasdiff) (AAAI 2026) as senselab's first **source separation** capability — speech–sound, sound–sound, and speech–speech — through a new `separate_audios()` task API.

**Architecture:** A subprocess-venv backend cloning the unpackaged, unlicensed upstream at a pinned SHA. Upstream ships only benchmark scripts, so senselab writes the inference driver against `GaussianDiffusion.p_sample_loop_group`, plus a long-form chunking scheme with cross-window permutation alignment that upstream has no equivalent of.

**Tech Stack:** Python 3.10 subprocess venv; `torch==2.6.0` CUDA-routed by `ensure_venv`; 16 kHz, fixed 4-second windows, `n_fft=510`, `hop=255`, 200 diffusion steps.

## Global Constraints

Copied from `design.md`. Every task's requirements implicitly include these.

- **No `analyze_audio` or `audio_analysis` wiring.**
- **No `run_config` changes.**
- **No new host dependencies and no new extras.**
- **No vendored upstream source.** The repository is unlicensed (`license: null`); it is cloned at a pinned SHA into the user's own cache at first use.
- Upstream pin: **`5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa`**.
- `ensure_venv` must keep routing torch/torchaudio through the CUDA-aware PyTorch index.
- **Every Python command runs through `uv run`.**
- **Never run `pytest -n auto`.**
- **Run `uv run ruff format` before any push.**
- **Never `git add -A` unqualified.** Always limit it with a pathspec (`git add -A -- src/ docs/ pyproject.toml uv.lock`). The repository root can hold untracked local secrets — a developer-supplied API token sitting beside the checkout is the case that prompted this — and an unqualified `git add -A` would stage one. `git status` is not a safeguard: an agent running these steps does not read it before committing.
- **Thresholds belong in `data/` with a written derivation, never as code literals.** The permutation-alignment threshold in Task 5 is the one number this plan introduces; it is derived by measurement there, not chosen.

## Preconditions

Branch `feat/new-model-integrations` already exists, cut from the merged `alpha` (PR #547, `79b37d93`); run Plan A's Task 1 first to verify it. **Task 2's mirror is already done** — `sensein/unasdiff-diffusion-priors` is live and private — so an implementer needs only read access to that repo, or `SENSELAB_UNASDIFF_CHECKPOINTS` pointing at a local directory.

## Upstream facts this plan depends on

Established by reading the repository at the pinned SHA. Re-verify in Task 1 if the pin moves.

- **`load_model` is defined in the benchmark script, not the library.** The worker must reimplement it — roughly:
  ```python
  model_class = getattr(models, config["model_name"])       # "Atten_unet"
  model = model_class(config["model_cfg"])
  for p in model.parameters():
      p.requires_grad = False
  ckpt = torch.load(config["ckpt_path"], map_location="cpu", weights_only=True)
  model.load_state_dict(ckpt["model"]); model.to(device).eval()
  ema_model = deepcopy(model); ema_model.load_state_dict(ckpt["ema"])
  ema_model.to(device).eval()
  return ema_model                                           # the EMA weights, not ckpt["model"]
  ```
  Returning `ckpt["model"]` instead of `ckpt["ema"]` silently gives worse separation. Upstream already passes `weights_only=True`, so unlike DriftSE no deviation is needed there.
- **`flash-attn` is genuinely optional.** `atten_unet.py` sets `use_flash = False` on `ImportError` and the attention forward branches to a manual softmax path. Omitting it is verified-safe, not hopeful — but the fallback materialises a `[b, h, t, t]` attention matrix, so it is slower and heavier than flash attention.
- **`p_sample_loop_group` ignores its own `measurement` argument** and recomputes `measurement = degradation(orig_x, n_src)` every step. `degradation` is supplied by the caller.
- **Per-model label spaces.** The speech prior (`config/atten_unet_vctk`) has `num_class = 1`; the sound prior (`config/atten_unet_fsd`) has `num_class = 50`, of which `sound_dataset_process/audio_infos/label.json` populates indices **0–40** — 41 FSDKaggle2018 classes. The remaining nine slots are unused headroom.
- Both configs share `hidden_size=72`, `emb_dim=128`, `num_heads=4`, `n_fft=510`, `hop_length=255`, `win_length=510`, `diffusion_step=200`, `beta_start=1e-4`, `beta_end=0.02`.
- Benchmark geometry: 16 kHz, 4-second windows, peak-normalised to 0.95.

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `src/senselab/audio/tasks/source_separation/__init__.py` | Public re-exports | Create |
| `src/senselab/audio/tasks/source_separation/api.py` | `separate_audios`; mode validation; class-name resolution | Create |
| `src/senselab/audio/tasks/source_separation/unasdiff.py` | Venv constants, clone helper, worker script, driver | Create |
| `src/senselab/audio/tasks/source_separation/data/fsd41_classes.json` | FSD class name → prior index, derived from upstream `label.json` | Create |
| `src/senselab/audio/tasks/source_separation/doc.md` | Module documentation | Create |
| `src/senselab/utils/data_structures/model.py` | `model_for_task(..., task="separation")` | Modify |
| `src/senselab/model_registry.yaml` / `.md`, `docs/compatibility-matrix.md` | Registry rows | Modify |
| `src/tests/audio/tasks/source_separation_test.py` | API, class resolution, worker contract, end-to-end | Create |

---

### Task 1: Module scaffolding, venv constants, and the FSD class map

**Files:**
- Create: `src/senselab/audio/tasks/source_separation/{__init__,api,unasdiff}.py`, `data/fsd41_classes.json`
- Test: `src/tests/audio/tasks/source_separation_test.py`

**Interfaces:**
- Consumes: `subprocess_venv` helpers.
- Produces: constants `_UNASDIFF_VENV`, `_UNASDIFF_PYTHON`, `_UNASDIFF_REQUIREMENTS`, `_UNASDIFF_REPO_URL`, `_UNASDIFF_COMMIT`, `_UNASDIFF_HF_REPO`, `_UNASDIFF_CHECKPOINTS_ENV`; and `load_fsd_class_map_document() -> dict[str, Any]` (the whole profile: `version`, `derivation`, `num_embedding_slots`, `classes`) and `resolve_source_classes(names: list[str]) -> list[int]`.

- [ ] **Step 1: Generate the class map from upstream**

```bash
mkdir -p src/senselab/audio/tasks/source_separation/data
gh api repos/RunwuShi/unasdiff/contents/sound_dataset_process/audio_infos/label.json?ref=5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa \
  --jq '.content' | base64 -d > /tmp/label.json
uv run python - <<'PY'
import json
labels = json.load(open("/tmp/label.json"))
doc = {
    "version": 1,
    "derivation": (
        "Copied from unasdiff's sound_dataset_process/audio_infos/label.json at commit "
        "5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa. These indices are the sound prior's "
        "own conditioning labels; they are NOT AudioSet ids and NOT shared with the "
        "speech prior, whose only label is 0. The prior's embedding has 50 slots "
        "(num_class=50 in config/atten_unet_fsd/config.toml) of which these 41 are "
        "populated -- the remaining nine are unused headroom, not classes."
    ),
    "num_embedding_slots": 50,
    "classes": labels,
}
json.dump(doc, open("src/senselab/audio/tasks/source_separation/data/fsd41_classes.json", "w"), indent=2)
print("classes:", len(labels), "max index:", max(labels.values()))
PY
```

Expected: `classes: 41 max index: 40`. If either differs, stop — this plan's class-space assumption is wrong.

- [ ] **Step 2: Write the failing tests**

```python
"""unasdiff source separation — API contract and class-space handling."""

import pytest

from senselab.audio.tasks.source_separation import unasdiff
from senselab.audio.tasks.source_separation.api import resolve_source_classes


def test_class_map_has_41_classes_in_50_slots() -> None:
    """The prior's embedding is 50-wide but only 41 labels were trained. Passing
    an index in 41..49 would condition on an untrained embedding row and produce
    plausible-looking noise rather than an error."""
    doc = unasdiff.load_fsd_class_map_document()
    assert len(doc["classes"]) == 41
    assert max(doc["classes"].values()) == 40
    assert doc["num_embedding_slots"] == 50


def test_resolve_source_classes_maps_names_to_indices() -> None:
    assert resolve_source_classes(["Applause", "Cello"]) == [
        unasdiff.load_fsd_class_map_document()["classes"]["Applause"],
        unasdiff.load_fsd_class_map_document()["classes"]["Cello"],
    ]


def test_an_unknown_class_raises_and_names_the_valid_options() -> None:
    """Silently falling back to index 0 would condition the prior on 'Hi-hat'
    while reporting the caller's own label — separation would be wrong and the
    output would claim otherwise."""
    with pytest.raises(ValueError) as exc:
        resolve_source_classes(["Helicopter"])
    assert "Helicopter" in str(exc.value)
    assert "Applause" in str(exc.value), "the error must enumerate the valid classes"


def test_upstream_is_pinned_to_a_full_commit_sha() -> None:
    assert len(unasdiff._UNASDIFF_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in unasdiff._UNASDIFF_COMMIT)


def test_flash_attn_is_not_required() -> None:
    """atten_unet.py sets use_flash=False on ImportError and branches to a manual
    softmax attention, so the venv can omit a package that is slow and fragile to
    build. Verified against upstream, not assumed."""
    named = {r.split(">=")[0].split("==")[0].strip().lower()
             for r in unasdiff._UNASDIFF_REQUIREMENTS}
    assert "flash-attn" not in named and "flash_attn" not in named


def test_torch_is_pinned_for_cuda_routing() -> None:
    named = {r.split(">=")[0].split("==")[0].strip().lower()
             for r in unasdiff._UNASDIFF_REQUIREMENTS}
    assert "torch" in named and "torchaudio" in named
```

- [ ] **Step 3: Run them and watch them fail**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -v
```

Expected: FAIL — the module does not exist.

- [ ] **Step 4: Create the module and constants**

`unasdiff.py` opens with a module docstring covering: what the paper does; that upstream ships only benchmark scripts so the driver is ours; that the two priors have separate label spaces; the licensing position; and that the backend is not wired into `audio_analysis`. Then:

```python
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
_UNASDIFF_COMMIT = "5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa"
_UNASDIFF_HF_REPO = "sensein/unasdiff-diffusion-priors"
# Pinned so a re-upload cannot change what this backend runs. The repo is private
# pending the upstream licence answer; callers without access use the env override.
_UNASDIFF_HF_REVISION = "8d7c32204d1ba31cd9fca3cd64313fd711949b58"
_UNASDIFF_CHECKPOINTS_ENV = "SENSELAB_UNASDIFF_CHECKPOINTS"

_TARGET_SR = 16000
_WINDOW_S = 4.0          # upstream's trained window; not a tunable
_DIFFUSION_STEPS = 200   # config/*/config.toml: diffusion_step
```

Add `load_fsd_class_map_document()` reading the JSON through `importlib.resources` with `functools.lru_cache(maxsize=1)`, and `resolve_source_classes` in `api.py` raising a `ValueError` that enumerates the valid names.

Note for any test that needs to vary the map: isolate it with `monkeypatch.setattr` on the loader, **never** `load_fsd_class_map_document.cache_clear()` — clearing a module-level cache mutates real state that outlives the test.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -v
```

Expected: PASS, 6 tests.

- [ ] **Step 6: Verify the pin and the flash-attn fallback against upstream**

```bash
gh api "repos/RunwuShi/unasdiff/contents/models/atten_unet.py?ref=5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa" \
  --jq '.content' | base64 -d | sed -n '14,24p'
```

Expected: `use_flash = False`, a `try: from flash_attn import flash_attn_func`, and `except ImportError` setting `use_flash = False`. If the fallback is gone at this pin, `flash-attn` must go back into the requirements and this plan's cost estimates change.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/source_separation/ src/tests/
uv run mypy src/senselab/audio/tasks/source_separation/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(source_separation): scaffolding, pinned upstream, FSD class map

The sound prior's 41 trained labels sit in a 50-wide embedding; the map records
that so an index in 41..49 cannot silently condition on an untrained row. The
speech prior's label space is separate and has exactly one member.

flash-attn is omitted: upstream sets use_flash=False on ImportError and branches
to a manual softmax attention, verified at the pinned commit.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Mirror the two priors — **DONE (2026-08-08)**, except the licence request

The mirror exists. Only Step 1 below, the upstream licence request, is outstanding.

**Produced:** `sensein/unasdiff-diffusion-priors`, **private**, at revision
`8d7c32204d1ba31cd9fca3cd64313fd711949b58`.

| File | SHA-256 | Size |
|---|---|---|
| `speech_source.pt` | `158060ea1e7c83a926063c15528e3f26e92f13d4ed32499158e59e4cebc36bb0` | 596.5 MB |
| `sound_source.pt` | `2f30c1178ab11e8f875e49589913fc0dc463d0a9f7bf6c290400fa66e304470b` | 597.0 MB |
| `atten_unet_vctk.toml` | `4a98c204d24c976f2fe05bde82c4b9b7c67a2608c247c5b90f8217e78f3f76e5` | 674 B |
| `atten_unet_fsd.toml` | `cf11d2a53bc9202418d0f9bcb3963b58b15fb6df9107790b43f56f976fb9d12b` | 672 B |

**Step 3's verification passed:** both priors load under `torch.load(..., weights_only=True)` and both contain `model`, `ema`, and `opt` keys. Task 3's loader uses `ema`, which is therefore present and correct — no fallback branch is needed. Note that Plan D's DriftSE checkpoints go the *other* way (upstream loads `model` there), so the two backends genuinely differ and neither should be made to match the other.

- [x] **Step 1: Open the upstream licence request — DONE**

Posted 2026-08-08: <https://github.com/RunwuShi/unasdiff/issues/1>. Until it is answered, the mirror stays private. The drafted text is kept below for the record.


Post to https://github.com/RunwuShi/unasdiff/issues:

> **Request: an explicit license (and optionally a HuggingFace weights mirror)**
>
> Thanks for releasing unasdiff — we'd like to offer it as a separation backend
> in [senselab](https://github.com/sensein/senselab), an open-source behavioural
> data toolkit.
>
> The repository has no `LICENSE` file and no license statement in the README,
> which under GitHub's default terms means all rights reserved, so we can't
> redistribute the code or the two priors. Would you be willing to add one?
>
> A HuggingFace mirror of `speech_source.pt` and `sound_source.pt` would also
> help — Google Drive links can't be pinned to a revision or content hash, which
> we need for reproducible runs.
>
> Until then we clone at a pinned commit at run time and vendor nothing.

- [x] **Steps 2–5: download, verify, create the private repo, write the card — done**

Carried out on 2026-08-08. The card records the provenance, the two separate label spaces, the `model`/`ema` distinction, and the authors' own same-class-separation caveat. Revision and digests are in the table above.

**Keep it private** until Step 1 is answered — a private mirror gives the priors a pinned revision and content hash without redistributing weights whose licence is unresolved.

---

### Task 3: The worker script and the single-window driver

**Files:**
- Modify: `src/senselab/audio/tasks/source_separation/unasdiff.py`
- Test: `src/tests/audio/tasks/source_separation_test.py`

**Interfaces:**
- Consumes: Task 1's constants and class map.
- Produces: `separate_with_unasdiff(audios, n_sources, source_class_indices, checkpoint_dir, device=None, seed=17) -> List[List[Audio]]`.

- [ ] **Step 1: Write the failing worker-contract tests**

```python
def test_worker_script_compiles_standalone() -> None:
    """The worker is a string literal run by another interpreter. A syntax error
    would otherwise surface only after the venv build and the model download."""
    compile(unasdiff._WORKER_SCRIPT, "<unasdiff worker>", "exec")


def test_worker_never_imports_the_benchmark_scripts() -> None:
    """test_speech_sound.py and its siblings call torch.cuda.set_device(0) at
    module import and abort on any CPU host. The worker imports the library
    modules directly."""
    for forbidden in ("test_speech_sound", "test_soundevent", "test_speech_speech"):
        assert forbidden not in unasdiff._WORKER_SCRIPT


def test_worker_uses_the_ema_weights() -> None:
    """load_model in the benchmark script returns the EMA copy, not ckpt['model'].
    Loading the non-EMA weights runs but separates measurably worse — a silent
    quality regression rather than a failure."""
    assert '"ema"' in unasdiff._WORKER_SCRIPT or "'ema'" in unasdiff._WORKER_SCRIPT


def test_worker_packs_the_mixture_so_degradation_reproduces_it() -> None:
    """p_sample_loop_group ignores its `measurement` argument and recomputes
    degradation(orig_x). Packing [mixture, zeros...] makes that sum equal the
    mixture exactly — which is what keeps this an inference call and not an
    oracle."""
    assert "zeros" in unasdiff._WORKER_SCRIPT
    assert "orig_x" in unasdiff._WORKER_SCRIPT
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -k worker -v
```

Expected: FAIL — `_WORKER_SCRIPT` does not exist.

- [ ] **Step 3: Write the worker**

The clone block is identical in shape to Plan D, Task 3 (flock, sibling temp dir, `os.replace`, fetch the pinned commit) with `repo_dir` named `unasdiff` and the marker file `models/atten_unet.py`. After `sys.path.insert(0, str(repo_dir))`:

```python
    import numpy as np
    import soundfile as sf
    import toml
    import torch
    from copy import deepcopy

    # Library modules only. The three test_*.py scripts call
    # torch.cuda.set_device(0) at import and abort on a CPU host.
    import models
    import diffusion

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    def load_prior(config_path, ckpt_path):
        """Reimplementation of load_model() from upstream's benchmark script.

        That function lives in test_speech_sound.py, not in the library, so there
        is nothing to import. It returns the EMA copy -- ckpt["ema"], not
        ckpt["model"] -- and loading the non-EMA weights separates measurably
        worse without failing, so the distinction is load-bearing.
        """
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
        """The mixture operator: sources are packed along time, so folding them
        back is a split-and-sum."""
        return sum(torch.split(x, x.shape[-1] // n_src, dim=-1))

    def separate_window(models_list, gaus, mixture, n_src, labels):
        """One 4 s window. Returns a list of n_src waveforms.

        p_sample_loop_group ignores the `measurement` argument it is handed and
        recomputes measurement = degradation(orig_x, n_src) on every step. Packing
        orig_x as [mixture, zeros, ..., zeros] makes that sum equal the mixture
        exactly, so the sampler sees precisely what it saw in the benchmark and no
        per-source information enters. This looks like an oracle from the call
        site; it is not.
        """
        T = mixture.shape[-1]
        mix = mixture.reshape(1, 1, -1)
        orig_x = torch.cat([mix] + [torch.zeros_like(mix)] * (n_src - 1), dim=-1)
        shape = (1, 1, n_src * T)
        gen = gaus.p_sample_loop_group(
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
```

Model list construction: index 0 is the speech prior when the mode includes speech, and each remaining slot loads the sound prior **again** (a separate instance — the sampler zips `model` against `model_kwargs`, so there must be `n_src` model objects). `labels` is `[0] + source_class_indices` for speech–sound, `source_class_indices` for sound–sound, and `[0, 0]` for speech–speech.

The `GaussianDiffusion` build, from the benchmark:

```python
    gaus = diffusion.GaussianDiffusion(
        steps=200,
        config_file=speech_config,
        beta_start=speech_config["train_para"]["beta_start"],
        beta_end=speech_config["train_para"]["beta_end"],
    )
```

Normalisation, per upstream's `_norm`: `peak = |x|.amax().clamp(min=1e-8)`, `x = x / peak * 0.95`, and the inverse applied to every output.

- [ ] **Step 4: Write the host-side driver for a single window**

`separate_with_unasdiff` resamples to 16 kHz mono, **rejects inputs longer than one window for now** with a clear `NotImplementedError` naming Task 5, resolves the checkpoint directory from `_UNASDIFF_CHECKPOINTS_ENV` or `hf_hub_download`, builds the venv, writes WAVs, runs the worker, reads back `n_sources` `Audio` objects per input.

Restricting to one window here is deliberate: it makes Task 5's chunking a separate, reviewable change rather than something entangled with getting the sampler call right.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -k worker -v
```

Expected: PASS, 4 tests.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/source_separation/ src/tests/
uv run mypy src/senselab/audio/tasks/source_separation/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(source_separation): unasdiff worker and single-window driver

Upstream ships only benchmark scripts, so the driver is ours. It reimplements
load_model (which lives in test_speech_sound.py, not the library) including the
EMA copy, and packs orig_x as [mixture, zeros...] so the sampler's internally
recomputed degradation(orig_x) equals the mixture exactly -- an inference call,
not the oracle its signature suggests.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The public API and its three modes

**Files:**
- Modify: `src/senselab/audio/tasks/source_separation/api.py`, `__init__.py`
- Modify: `src/senselab/utils/data_structures/model.py`
- Test: `src/tests/audio/tasks/source_separation_test.py`

**Interfaces:**
- Consumes: `separate_with_unasdiff` (Task 3), `resolve_source_classes` (Task 1).
- Produces:
  ```python
  def separate_audios(
      audios: list[Audio],
      model: SenselabModel | None = None,
      n_sources: int = 2,
      mode: str = "speech_sound",          # speech_sound | sound_sound | speech_speech
      source_classes: list[str] | None = None,
      device: DeviceType | None = None,
      seed: int = 17,
  ) -> list[list[Audio]]: ...
  ```

- [ ] **Step 1: Write the failing API tests**

```python
import pytest

from senselab.audio.tasks.source_separation import separate_audios


def test_sound_modes_require_source_classes(mono_audio_sample) -> None:
    """The sound prior is class-conditioned. Without a class there is no
    defensible default -- index 0 is 'Hi-hat', and silently choosing it would
    separate against the wrong prior while reporting success."""
    with pytest.raises(ValueError, match="source_classes"):
        separate_audios([mono_audio_sample], mode="speech_sound", n_sources=2)


def test_speech_speech_needs_no_source_classes(mono_audio_sample, monkeypatch) -> None:
    """Both slots use the speech prior, whose only label is 0."""
    captured = {}

    def fake(audios, n_sources, source_class_indices, **kwargs):
        captured["labels"] = source_class_indices
        return [[audios[0]] * n_sources]

    monkeypatch.setattr(
        "senselab.audio.tasks.source_separation.api.separate_with_unasdiff", fake
    )
    separate_audios([mono_audio_sample], mode="speech_speech", n_sources=2)
    assert captured["labels"] == [0, 0]


def test_speech_sound_prepends_the_speech_label(mono_audio_sample, monkeypatch) -> None:
    captured = {}

    def fake(audios, n_sources, source_class_indices, **kwargs):
        captured["labels"] = source_class_indices
        return [[audios[0]] * n_sources]

    monkeypatch.setattr(
        "senselab.audio.tasks.source_separation.api.separate_with_unasdiff", fake
    )
    separate_audios(
        [mono_audio_sample], mode="speech_sound", n_sources=2, source_classes=["Applause"]
    )
    assert captured["labels"][0] == 0, "slot 0 is the speech prior"
    assert len(captured["labels"]) == 2


def test_source_classes_length_must_match_the_sound_slots(mono_audio_sample) -> None:
    with pytest.raises(ValueError, match="n_sources"):
        separate_audios(
            [mono_audio_sample], mode="speech_sound", n_sources=3, source_classes=["Applause"]
        )


def test_an_unknown_mode_raises(mono_audio_sample) -> None:
    with pytest.raises(ValueError, match="mode"):
        separate_audios([mono_audio_sample], mode="music_speech", n_sources=2)
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -k "mode or source_classes or speech_" -v
```

Expected: FAIL.

- [ ] **Step 3: Implement `separate_audios`**

Validate `mode`; build the label list per mode; check `len(source_classes)` against the number of sound slots (`n_sources - 1` for `speech_sound`, `n_sources` for `sound_sound`, unused for `speech_speech`); delegate.

The `speech_speech` docstring must carry the authors' own caveat verbatim — a user will reach for it first, and the limitation is not discoverable from the output:

```python
    """...
    ``speech_speech`` ships with a caveat from unasdiff's own README:

        "The source-model-based separation approach is not well suited for
        same-class source separation (e.g. speech separation), because it lacks
        speaker-conditioning. In future work, we will attempt to address such
        issue."

    It is exposed because the alternative is a user rediscovering the limitation
    by measurement, but nothing downstream should treat its output as a reliable
    decomposition.
    """
```

Add `model_for_task(model_id, task="separation")` returning `HFModel` for the `sensein/unasdiff` prefix.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -v
```

Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/ src/tests/
uv run mypy src/senselab/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(source_separation): separate_audios with three modes

source_classes is required for any mode using the sound prior: index 0 is
'Hi-hat', so a silent default would separate against the wrong prior while
reporting success. speech_speech carries the authors' own caveat that the
approach lacks speaker conditioning.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Long-form chunking with cross-window permutation alignment

The one place this plan introduces a number. It is derived by measurement, not chosen.

**Files:**
- Modify: `src/senselab/audio/tasks/source_separation/unasdiff.py`
- Create: `src/senselab/audio/tasks/source_separation/data/permutation_alignment.json`
- Test: `src/tests/audio/tasks/source_separation_test.py`

**Interfaces:**
- Consumes: `separate_window` from Task 3.
- Produces: `align_permutations(prev_tail: list[Tensor], next_head: list[Tensor]) -> list[int]` — the index permutation mapping the next window's slots onto the previous window's.

- [ ] **Step 1: Write the failing alignment tests**

These test the alignment logic in isolation with synthetic signals, so they need neither the venv nor a GPU.

```python
import torch

from senselab.audio.tasks.source_separation.unasdiff import align_permutations


def test_identity_permutation_when_slots_already_match() -> None:
    a, b = torch.randn(1000), torch.randn(1000)
    assert align_permutations([a, b], [a, b]) == [0, 1]


def test_swapped_slots_are_detected() -> None:
    """Windows are separated independently, so slot 0 in window k need not be the
    same source as slot 0 in window k+1. Concatenating without this check swaps
    sources mid-file and the result is worse than the mixture."""
    a, b = torch.randn(1000), torch.randn(1000)
    assert align_permutations([a, b], [b, a]) == [1, 0]


def test_three_sources_resolve_to_a_full_permutation() -> None:
    a, b, c = torch.randn(1000), torch.randn(1000), torch.randn(1000)
    assert sorted(align_permutations([a, b, c], [c, a, b])) == [0, 1, 2]
    assert align_permutations([a, b, c], [c, a, b]) == [1, 2, 0]


def test_alignment_survives_scaling_and_noise() -> None:
    """Adjacent windows overlap but are not identical there — each is the
    sampler's own estimate. Correlation must be scale-invariant and tolerate the
    disagreement."""
    a, b = torch.randn(1000), torch.randn(1000)
    noisy_a = 0.7 * a + 0.05 * torch.randn(1000)
    noisy_b = 1.3 * b + 0.05 * torch.randn(1000)
    assert align_permutations([a, b], [noisy_b, noisy_a]) == [1, 0]
```

- [ ] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -k permutation -v
```

Expected: FAIL — `align_permutations` does not exist.

- [ ] **Step 3: Implement**

Zero-mean and unit-normalise each candidate over the overlap region, build the `n × n` correlation matrix, and choose the assignment maximising total correlation (`scipy.optimize.linear_sum_assignment` if available in the host, otherwise `itertools.permutations` — `n ≤ 4` in practice, so brute force is fine and avoids a host dependency). Scale-invariance comes from the normalisation, which is what makes the fourth test pass.

- [ ] **Step 4: Derive the confidence threshold by measurement, and record it**

Alignment can be ambiguous — two windows whose best and second-best assignments score almost identically. Report that rather than silently picking one.

Measure the distribution of `best_score - second_best_score` on constructed cases whose correct answer is known, so "ambiguous" has a reference:

```bash
uv run python - <<'PY'
import itertools, json, torch
from senselab.audio.tasks.source_separation.unasdiff import _assignment_scores

torch.manual_seed(0)
known_correct, ambiguous = [], []
for trial in range(200):
    a, b = torch.randn(2000), torch.randn(2000)
    # Known-correct: the two sources are distinct, so one assignment must win.
    noisy = [0.8 * b + 0.05 * torch.randn(2000), 1.2 * a + 0.05 * torch.randn(2000)]
    s = sorted(_assignment_scores([a, b], noisy), reverse=True)
    known_correct.append(s[0] - s[1])
    # Ambiguous: the two sources are near-identical, so no assignment is right.
    c = torch.randn(2000)
    near = [c + 0.01 * torch.randn(2000), c + 0.01 * torch.randn(2000)]
    s = sorted(_assignment_scores([c, c + 0.01 * torch.randn(2000)], near), reverse=True)
    ambiguous.append(s[0] - s[1])

q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
print("known-correct margins  p05/p50:", q(known_correct, 0.05), q(known_correct, 0.50))
print("ambiguous margins      p50/p95:", q(ambiguous, 0.50), q(ambiguous, 0.95))
PY
```

`_assignment_scores(prev_tail, next_head) -> list[float]` returns the total correlation of every candidate permutation; factor it out of `align_permutations` so both the function and this measurement use one implementation. Write the resulting margin into `data/permutation_alignment.json` with the measurement behind it, in the shape the repository's other profiles use:

```json
{
  "version": 1,
  "derivation": "...what was measured, on what, and what the numbers were...",
  "min_assignment_margin": 0.0
}
```

**If the measurement does not support a value, do not invent one.** Write the profile with the margin absent and have the code report every window's margin in its result rather than gating on an unfitted threshold — the repository has two prior defects from literals that were never fitted, and this plan is not adding a third. `scripts/calibrate_detection_margin.py` is the model for a calibration script that refuses to emit a profile from insufficient measurement.

- [ ] **Step 5: Wire chunking into the driver**

Replace Task 3's `NotImplementedError` with: 4 s windows at 50 % overlap; separate each; align window *k+1* onto window *k*; Hann-taper and overlap-add per aligned slot; carry the per-boundary alignment margins into the result so a caller can see where the assignment was close.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src/senselab/ src/tests/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(source_separation): long-form chunking with permutation alignment

Upstream trains on fixed 4 s windows and has no long-form path. Separating
windows independently swaps sources mid-file, so adjacent windows are aligned by
correlation on their overlap before overlap-add. Senselab's construction, not
upstream's, and doc.md says so. Per-boundary alignment margins are reported
rather than gated on an unfitted threshold.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: End-to-end run, registry, and `doc.md`

**Files:**
- Modify: `src/tests/audio/tasks/source_separation_test.py`
- Modify: `src/senselab/model_registry.yaml` / `.md`, `docs/compatibility-matrix.md`
- Create: `src/senselab/audio/tasks/source_separation/doc.md`

- [ ] **Step 1: Add the skip-gated end-to-end test**

```python
import pytest

from senselab.utils.subprocess_venv import _cache_dir_path


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.mark.skipif(
    not ((_cache_dir_path() / "unasdiff").is_dir() and _cuda_available()),
    reason="needs the unasdiff venv and CUDA; the sampler backprops through the "
    "priors at every one of 200 steps, so CPU is impractical",
)
def test_unasdiff_separates_a_mixture_into_n_sources(mono_audio_sample) -> None:
    """Shape and energy, not quality. A separation that returns the mixture in
    every slot passes a shape check, so the energy-difference assertion is the
    one that can actually fail."""
    from senselab.audio.tasks.preprocessing import resample_audios
    from senselab.audio.tasks.source_separation import separate_audios

    audio = resample_audios([mono_audio_sample], resample_rate=16000)[0]
    result = separate_audios(
        [audio], mode="speech_sound", n_sources=2, source_classes=["Applause"], seed=17
    )

    assert len(result) == 1 and len(result[0]) == 2
    for source in result[0]:
        assert source.sampling_rate == 16000
        assert source.waveform.shape[-1] == audio.waveform.shape[-1]

    a, b = result[0][0].waveform, result[0][1].waveform
    assert (a - b).abs().mean() > 1e-4, "both slots returned the same signal"
```

- [ ] **Step 2: Run it**

```bash
uv run pytest src/tests/audio/tasks/source_separation_test.py -k separates -v -s
```

On a CUDA host, first run builds the venv, clones upstream, and downloads both priors. Expect minutes per 4 s window: 200 steps × `n_sources` model evaluations, each with a backward pass through the prior. On a CPU host, expect a skip.

- [ ] **Step 3: Record the measured runtime in `doc.md`**

Time one 4-second, two-source separation and write the number down with the hardware it was measured on. "Impractical on CPU" is an assertion until someone has the number.

- [ ] **Step 4: Registry and compatibility matrix**

Add `sensein/unasdiff-diffusion-priors` with task `separation`, the venv name, Python 3.10, the pinned torch, the pinned upstream commit, and the unresolved licence.

- [ ] **Step 5: Write `doc.md`**

Cover, in this order: the paper and what it does; that senselab writes the driver because upstream ships only benchmark scripts; **why `p_sample_loop_group(orig_x=...)` is not an oracle** — this is the single most important paragraph, because the call reads as cheating; the per-model label spaces and the 41-in-50 embedding; the three modes and the speech–speech caveat; the chunking scheme, marked as senselab's construction with the failure mode it prevents; the measured runtime; the licence status with the issue link; and that the backend is not wired into `audio_analysis`.

- [ ] **Step 6: Final check, commit, report**

```bash
uv run ruff format --check src/ && uv run ruff check src/ && uv run mypy src/senselab/
uv run pytest src/tests/audio/tasks/source_separation_test.py -v 2>&1 | tail -20
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "docs(source_separation): register unasdiff and document the driver

doc.md records why p_sample_loop_group's ground-truth argument is not an oracle,
that the two priors have separate label spaces, and that long-form chunking is
senselab's addition rather than upstream's.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

Report the commit SHAs, whether the end-to-end test ran or skipped, the measured per-window runtime if it ran, and the status of the upstream licence issue. Do not push.
