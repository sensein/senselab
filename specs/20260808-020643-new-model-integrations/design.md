# New model integrations: diarization backends, standalone PII, DriftSE, unasdiff

Status: design approved, implementation plans written, not yet implemented.
Date: 2026-08-08.

Implementation plans, one per workstream — each independently landable, in this order:

| Plan | Workstream | Tasks |
|---|---|---|
| [`plan-a-diarization-backends.md`](plan-a-diarization-backends.md) | A — four diarization backends from #537 | 7 |
| [`plan-b-pii-detection-task.md`](plan-b-pii-detection-task.md) | B — standalone PII detection from #542 | 8 |
| [`plan-d-driftse-enhancement.md`](plan-d-driftse-enhancement.md) | D — DriftSE one-step enhancement | 6 |
| [`plan-c-unasdiff-separation.md`](plan-c-unasdiff-separation.md) | C — unasdiff source separation | 6 |

All four share the branch `feat/new-model-integrations`, which already exists off the merged `alpha`.
Plan A's Task 1 verifies it and records the baseline test state; the other three build on it.

Four independent additions to the task layer, delivered on one branch as four commits. None of
them touch `scripts/analyze_audio.py` or `audio/workflows/audio_analysis/`. Every one is a
standalone component callable directly on an `Audio` object — or, for PII, on text.

That constraint is the organising decision of this spec, and it is deliberate. Wiring a new model
into the uncertainty workflow means deciding how its output participates in a fold, which axes vote
on it, and what its disagreement with the incumbent models *means*. Those are measurements, not
integrations. Landing the backends first makes those measurements possible; making them at the same
time as the integration would mean publishing axis values derived from models nobody has run yet.

## Branch and ordering

Base: `alpha`. The `20260728-221507-per-speaker-identity-scene` refactor merged into it as PR #547
(`79b37d93`), so the branch `feat/new-model-integrations` is cut from the merged `alpha` and carries
the run-config refactor. Both cherry-pick sources (#537, #542) also target `alpha`, which is what
makes their hunks apply.

Commit order is risk-ascending, so that a workstream blocked on an upstream answer cannot hold back
the ones that are not:

| # | Workstream | Nature | Blocked on |
|---|---|---|---|
| A | Four diarization backends (#537) | Mechanical cherry-pick | nothing |
| B | Standalone PII detection (#542) | Refactor of existing code | nothing |
| D | DriftSE speech enhancement | New backend, cheap inference | upstream license, weights mirror |
| C | unasdiff source separation | New capability, expensive inference | upstream license, weights mirror |

---

## A. Four diarization backends — cherry-pick #537, task layer only

PR #537 (`feat/diarization-multi-speaker-uncertainty`, Evan Ng) adds four architecturally distinct
diarization backends. This branch takes the task layer and leaves the workflow layer behind.

### Taken

- `audio/tasks/speaker_diarization/{vibevoice,child_adult,moss,diarizen}.py`
- `audio/tasks/speaker_diarization/api.py` — dispatch, the ignored-speaker-hint warnings, and the
  role-label prefix list that #537 collapsed to one source of truth
- `utils/data_structures/model.py` — the diarization prefix match gains `microsoft/VibeVoice-ASR`,
  `AlexXu811/whisper-child-adult`, `OpenMOSS-Team/MOSS-Transcribe-Diarize`, `BUT-FIT/diarizen`
- `utils/compatibility.py` — the dispatch-paths note on the `diarize_audios` entry
- `utils/dependencies.py` — `hf_subprocess_env` warns instead of silently reverting to online Hub
  loading, which is the per-call 429 path that function exists to remove
- `utils/subprocess_venv.py` — `_cache_dir_path()`, side-effect-free, so a test's skip gate can
  check a venv location without creating a directory at import time
- `model_registry.yaml`, `model_registry.md`, `docs/compatibility-matrix.md`
- `pyproject.toml`, `uv.lock`
- `tests/audio/tasks/speaker_diarization_test.py`, `tests/utils/hf_load_coverage_test.py`

### Skipped

`audio/workflows/audio_analysis/{clustering,identity,presence,stage_context,stages}.py` and
`scripts/analyze_audio.py`.

### The consequence, and why it is safe

The four backends are reachable through `diarize_audios(model=...)` and **not** through
`analyze_audio.py --diarization-models`. That is correct rather than incomplete: the guards which
keep child-adult's `CHILD`/`ADULT`/`OVERLAP` role labels out of embedding clustering, out of the
identity axis's cross-diarizer agreement vote, and out of presence, live entirely in the skipped
files. Without them, a role-label backend in `--diarization-models` would build a `CHILD` centroid
blending two different children, snap `OVERLAP` to whichever centroid is nearest, and read as
spurious disagreement against every real diarization model.

Each backend's module docstring states this: wiring it into the workflow requires porting the guards
from #537 first. None of the four is in any default model list, so this is a guard against future
misconfiguration, not a live defect.

### Watch item: a core dependency bump

#537 raises core `transformers>=5.0` → `>=5.3`, needed for
`VibeVoiceAsrForConditionalGeneration`. That is every HuggingFace backend in the package, not just
diarization. The commit re-runs the HF-touching test directories (`speech_to_text`, `classification`,
`speaker_embeddings`, `ssl_embeddings`, `text`) rather than only `speaker_diarization`, and
regenerates `uv.lock`.

### Method

One squashed commit rather than eleven cherry-picks: with the workflow files stripped out, #537's
individual commits are not independently coherent (several are review-fix commits that touch both
layers). `Co-Authored-By: Evan Ng <evan.ng@sickkids.ca>`.

---

## B. Standalone PII detection over text and `ScriptLine`

PR #542 (Varun Thvar) is a 3,665-line self-contained `scripts/` tool answering two questions per
recording: does it contain PII, and did the participant perform the task. This branch takes the
first question only, as a senselab task.

### Scope, explicitly

**In:** PII detection over a string, a `ScriptLine`, or a list of either; the rule-cascade detection
engine from #542; a convenience path that runs the whole thing on an `Audio`.

**Out:** task-compliance verification, `task_reference.json` (797 Bridge2AI task definitions), the
Tier A/B/C modality routing, the Tier C LLM judge, and the `.pt`-folder batch driver. These were in
an earlier draft of this design and were cut on instruction. Nothing in the code left behind depends
on them.

### Module

`src/senselab/text/tasks/pii_detection/` — `api.py`, `subprocess_backend.py`, `rules.py`, `doc.md`.

Placed under `text/` because the input is a transcript. The `audio/` convenience entry point below
depends on `text/`, never the reverse.

A second, deliberately tiny module — `src/senselab/audio/tasks/pii_detection/api.py` — holds the
`Audio` entry point, for exactly that reason: it needs `transcribe_audios`, and a module under
`text/` importing from `audio/` would invert the layering the placement was chosen to establish.

### Public API

```python
detect_pii(
    inputs: str | ScriptLine | Sequence[str | ScriptLine],
    detectors: list[str] | None = None,
    ...,
) -> PiiReport | list[PiiReport]
```

A `ScriptLine` with nested `chunks` is flattened depth-first and its `text` fields joined, so a
word-level ASR result and a segment-level one produce the same scan. A `ScriptLine` carrying only
`speaker` and no `text` contributes nothing rather than erroring.

```python
detect_pii_in_audios(
    audios: list[Audio],
    asr_model: SenselabModel | None = None,
    ...,
) -> list[PiiReport]
```

Lives in `audio/tasks/pii_detection/api.py`, transcribes with `transcribe_audios`, and delegates to
`detect_pii`. This is what makes PII "runnable on an `Audio` object" without any workflow
involvement.

### Engines

All three run inside the **existing** `pii-detection` subprocess venv (Python 3.13), which already
hosts Presidio and GLiNER. `workflows/audio_analysis/pii_subprocess.py` moves here wholesale as
`subprocess_backend.py`.

1. **Presidio Analyzer** — regex + spaCy NER, unchanged.
2. **GLiNER PII** (`nvidia/gliner-pii`) — unchanged.
3. **The #542 rule cascade** — regex, gazetteers, self-disclosed demographics, rare roles, age > 90,
   and the combinatorial re-identification window. New, as `rules.py`.

`wordfreq` and `nltk` join `_PII_REQUIREMENTS`. **No new host dependency and no new extra** — the
`pii` extra proposed by #542 is not created, because PII already runs in a venv and the point of the
venv is that the host does not carry these packages.

Requirements change means existing hosts must rebuild the venv. The implementation plan verifies
that `ensure_venv` keys cache validity on the requirements list; if it does not, the venv name gets
a suffix so a stale tree cannot be silently reused.

### A defect this change surfaces

`_compute_detection_confidence` computes detector agreement as
`len(g["detectors"]) / len(_KNOWN_DETECTORS)`. Today `_KNOWN_DETECTORS` has two members, so a finding
both detectors agree on scores 1.0 and a single-detector finding scores 0.5.

Adding a third detector changes that denominator to 3, so **every confidence already published would
silently rescale** — a two-detector agreement drops from 1.0 to 0.67 with no change in evidence. The
denominator becomes the number of detectors that actually ran for that report. That also fixes a
pre-existing case the current code gets wrong: when GLiNER fails to load and only Presidio runs, a
Presidio finding is currently capped at 0.5 as though a second detector had declined to corroborate
it, when in fact no second detector was asked.

### Fixes from #542 carried forward, with their tests

- `torch.load(..., weights_only=True)` — scanning `.pt` files produced elsewhere with the
  unrestricted unpickler means code execution during a compliance scan.
- `_zipf` returns `None`, not `0.0`, when `wordfreq` is unavailable. `0.0` means "measured, maximally
  rare", which every caller reads as evidence *for* a PII hit, so the old sentinel inverted two
  guards rather than relaxing them.
- Structured-identifier format validation and the cross-engine corroboration requirement are not
  switchable by recall mode. Format validity is a correctness check under either posture.
- The optional localhost LLM engine is preserved, off by default, and talks only to `localhost`.

### The workflow keeps working

`workflows/audio_analysis/pii.py` becomes a thin adapter mapping `{asr_model_id → resolved_result}`
onto `detect_pii`, preserving `PiiPassReport`, its cross-ASR corroboration, and `report_to_dict`.
This is not new wiring — it is keeping what already runs from breaking when its implementation moves.
Per the repository's pre-alpha convention, the old module is replaced outright: no parallel fields,
no aliases, no deprecation shim.

### Tests

#542's 138 `--selftest` checks become real pytest modules under `src/tests/text/tasks/`. `--selftest`
itself is deleted; a self-test flag inside a shipped module is a second test framework with no
fixtures, no collection, and no CI.

---

## D. DriftSE speech enhancement

[DriftSE](https://github.com/LiangXu123/DriftSE) — *Speech Enhancement Based on Drifting Models*,
Xu, Caviedes-Nozal, Kleijn, Yan & Olsson, Interspeech 2026 (oral), arXiv 2604.24199. Formulates
enhancement as a distributional equilibrium problem and reaches the clean-speech distribution in a
**single network evaluation** (1 NFE), against 30 for SGMSE+ and 8 for UNIVERSE++. On the DNS 2020
blind test set it reports WV-MOS 2.65 and SCOREQ 2.97, above every listed baseline including SGMSE+.

### Why this one is cheap

The drifting field is computed in a frozen SSL latent space (HuBERT / WavLM / DistilHuBERT) — but
that is the **training** signal. Inference is the backbone alone: one forward pass under `no_grad`.
Confirmed by reading `enhancement.py`, which imports only `backbones.ncsnpp_v2`,
`backbones.ncsnpp_v2_drift` and `util.other`. No Lightning, no `wandb`, no `pesq`, and **no SSL
encoder**, so the `latent_ckpt/` Google Drive archive the README requires for training is not needed
here at all.

This makes DriftSE the first generative enhancer in the package that is genuinely CPU-viable.

### Module and dispatch

`audio/tasks/speech_enhancement/driftse.py`, alongside `speechbrain.py`.

`enhance_audios(...)` today accepts only a `SpeechBrainModel` and raises `NotImplementedError` for
anything else. It gains an `HFModel` branch matched on the `sensein/driftse` prefix — the same shape
#537 uses for the new diarizers, so there is one dispatch idiom in the package rather than two. The
default model stays `speechbrain/sepformer-wham16k-enhancement`, so no existing caller changes
behaviour; DriftSE is reached only by passing
`HFModel(path_or_uri="sensein/driftse-distilhubert-three-layers", revision=…)` explicitly.

### Isolation

Subprocess venv `driftse`, Python 3.11. Requirements: `torch`, `torchaudio` (CUDA-routed by
`ensure_venv`), `numpy`, `scipy`, `librosa`, `soundfile`, `tqdm`.

Deliberately **not** installed: `pesq`, `pystoi`, `scoreq`, `torch-pesq`, `asteroid-filterbanks`,
`wandb`, `pytorch-optimizer`, `torchinfo`. Upstream's `requirements.txt` lists them for training and
metric computation; the inference path imports none of them. `util/inference.py` does import `pesq`
and `pystoi`, but `enhancement.py` never imports `util.inference`, so the worker must not either.

Subprocess rather than in-process, even though the dependency set would satisfy senselab core. The
repository's top-level module names are `backbones`, `util`, `config`, and `data`; injecting a
generic `util` onto the host interpreter's `sys.path` is the sort of hazard that surfaces months
later as an unrelated import resolving to the wrong module.

### Upstream access

Runtime `git clone --depth 1` pinned to **`695a64db187500fa0d7bae23912680bd5d4df613`**, cloned to a
sibling temp directory and moved into place with `os.replace` under an `flock` — the `child_adult.py`
pattern, which exists because an interrupted clone otherwise wedges the existence guard permanently
and concurrent jobs sharing `$HOME` race into the same directory.

### Worker recipe

From `enhancement.py`, with senselab's corrections:

1. Resample to 16 kHz mono.
2. `norm_factor = |y|.max() + 1e-8`; `y /= norm_factor`.
3. `torch.stft` with the config's `n_fft`, `hop_length`, window type (`sqrthann` or `hann`), and
   `center`.
4. `spec_fwd`: `|S|^e · exp(i∠S) · f`, with `e = spec_abs_exponent`, `f = spec_factor`.
5. `pad_spec(..., mode="zero_pad")`.
6. **One** forward pass under `no_grad`: `model(Y + 0.05·z, t=1)` when `train_add_gaussian` is true,
   `model(Y, t=1)` otherwise, with `z ~ N(0, I)`.
7. `spec_bwd`, then `torch.istft(..., length=T_orig)`, then `× norm_factor`.

Model construction is `ncsnpp_v2_drift(**config)` or `NCSNpp_v2(**config)` selected by
`config["model"]`, then `load_state_dict(checkpoint["model"] if "model" in checkpoint else
checkpoint)`.

**Deviations from upstream, and why:**

- `torch.load(..., weights_only=True)`. Upstream omits it. The checkpoint is a foreign pickle from an
  unlicensed research repository; loading it with the unrestricted unpickler is arbitrary code
  execution at enhancement time.
- Long inputs are chunked with overlap-add. Upstream runs one STFT over an entire file; the NCSN++
  backbone carries attention layers at several resolutions, so memory grows superlinearly in
  duration. Enhancement is per-segment consistent — there is no cross-segment identity to preserve —
  so overlap-add is safe here, unlike the separation case in C.
- The determinism note: with `train_add_gaussian` true, the forward pass consumes a Gaussian sample,
  so output is stochastic. The worker seeds from a caller-visible parameter and records the seed, so
  a repeated run is reproducible and a caller who wants the deterministic `no_z` formulation knows
  that is a different checkpoint, not a flag.

### Weights

Mirror `logs/distillhubert_three_layers_with_z/last.ckpt` and its matching
`config/with_z/v2_drift2_distillhubert_three_layers.json` to
`sensein/driftse-distilhubert-three-layers`, **private** pending the license question below, fetched
by pinned revision so the checkpoint gets a content hash and provenance in the parquet output.
`SENSELAB_DRIFTSE_CHECKPOINT` overrides with a local directory.

### Risk to verify during implementation

`backbones/ncsnpp_utils/op/upfirdn2d.py` JIT-compiles a CUDA extension (`upfirdn2d_kernel.cu`).
Upstream ships `upfirdn2d_native.py` as a pure-PyTorch fallback. The plan confirms the native path is
actually selected when the extension will not build — on a CPU host, or on a CUDA host without a
matching toolchain — rather than assuming it, because the failure mode is a compile error at first
inference rather than at install.

---

## C. unasdiff source separation

[unasdiff](https://github.com/RunwuShi/unasdiff) — *Unsupervised Single-Channel Audio Separation with
Diffusion Source Priors*, Shi et al., AAAI 2026, arXiv 2512.07226. Two diffusion source priors
(speech, trained on VCTK; general sound, trained on FSDKaggle2018) drive a gradient-based
inverse-problem solver with a hybrid gradient update schedule and mixture-informed initialisation.

This is the package's first **separation** capability. `speech_enhancement` uses a SepFormer
checkpoint, but as a denoiser — it returns one signal, not a decomposition.

### Module

`audio/tasks/source_separation/` — `api.py`, `unasdiff.py`, `data/fsd41_classes.json`, `doc.md`.

```python
separate_audios(
    audios: list[Audio],
    model: SenselabModel | None = None,
    n_sources: int = 2,
    source_classes: list[str] | None = None,
    device: DeviceType | None = None,
) -> list[list[Audio]]
```

One list of `n_sources` `Audio` objects per input recording, all at 16 kHz. `model=None` selects the
mirrored unasdiff priors (`sensein/unasdiff-diffusion-priors`); it is the only backend, so the
argument exists to keep the signature consistent with the other task APIs rather than to offer a
choice today.

### Upstream ships no callable separation API

The repository contains three benchmark scripts (`test_speech_sound.py`, `test_soundevent.py`,
`test_speech_speech.py`) that synthesise mixtures from VCTK and an FSD test list and score SI-SNR
with permutation-invariant matching. There is no "separate this file" entry point. senselab writes
the driver.

### The driver, and why it is not an oracle

`GaussianDiffusion.p_sample_loop_group` takes an `orig_x` argument that the benchmark fills with
**ground-truth sources**, which reads as an oracle. It is not: the loop ignores its own `measurement`
parameter and recomputes `measurement = degradation(orig_x, n_src)` on every step, where
`degradation` splits the packed tensor into `n_src` equal segments along time and sums them. For the
benchmark's packing, that sum *is* the mixture; no per-source information survives it.

So the driver packs `orig_x = cat([mixture, zeros, …, zeros])`, for which `degradation(orig_x)`
equals the mixture exactly, and the sampler sees precisely what it saw in the benchmark. The module
docstring records this, because anyone reading the call signature will otherwise conclude the
backend cheats.

Steps:

1. Resample to 16 kHz mono. Peak-normalise to 0.95 (upstream's `_norm`), keeping the scale.
2. Pack `n_sources` slots along the time axis: shape `(1, 1, n_sources · T)`.
3. `degradation = lambda x, n_src: sum(torch.split(x, x.shape[-1] // n_src, dim=-1))`.
4. `GaussianDiffusion(steps=200, beta_start=1e-4, beta_end=0.02, config_file=…)`.
5. `p_sample_loop_group(models, shape, orig_x=cat([mixture, zeros…]), n_src=n_sources,
   clip_denoised=True, degradation=degradation, model_kwargs=labels)`, draining the 200-step
   generator and taking the final `out["sample"]`.
6. Split into `n_sources` segments, undo the peak scale, emit `Audio` at 16 kHz.

### Class conditioning is mandatory, and the label spaces are per-model

`model_kwargs` is one class label per source, zipped against a **list of `n_sources` model
instances**. The label spaces are not shared:

- The **speech prior** (`config/atten_unet_vctk`) has `num_class = 1`. Its only label is `0`.
- The **sound prior** (`config/atten_unet_fsd`) has `num_class = 50`, of which
  `sound_dataset_process/audio_infos/label.json` populates indices **0–40** — 41 FSDKaggle2018
  classes (`Hi-hat`, `Saxophone`, `Trumpet`, `Cello`, …). The remaining nine embedding slots are
  unused headroom, not classes.

So a 1-speech + 2-sound run holds three model instances resident (the sound prior loaded twice) and
passes labels `[0, fsd_idx_a, fsd_idx_b]`. `source_classes` is therefore **required** for any mode
involving the sound prior, and `data/fsd41_classes.json` — checked in, derived from upstream's
`label.json` — is the name-to-index map. An unknown class name is an error naming the 41 valid
options, never a silent fallback to index 0 (`Hi-hat`).

Memory note: `n_sources` model instances plus autograd graph, all resident simultaneously.

### Long-form is senselab's addition, marked as such

Upstream trains on fixed **4 s / 16 kHz** windows (`n_fft=510`, `hop=255`) and has no long-form path.
Separating windows independently is not merely suboptimal — output slot 0 in window *k* need not be
the same source as slot 0 in window *k+1*, so a naive concatenation swaps sources mid-file and the
result is worse than the mixture.

The backend chunks at 4 s with 50 % overlap and resolves the permutation between adjacent windows by
correlating candidate assignments on the shared region, then overlap-adds. This is senselab's
construction, not upstream's, and `doc.md` says so — the failure mode it fixes is the reason it
exists, and a later reader should be able to disagree with the approach rather than guess at it.

### Modes

All three ship.

- **speech–sound** — speech prior + sound prior(s). The most directly useful mode.
- **sound–sound** — sound prior instances with distinct FSD classes.
- **speech–speech** — the speech prior twice. Ships with the author's own caveat quoted in the
  docstring: *"The source-model-based separation approach is not well suited for same-class source
  separation (e.g. speech separation), because it lacks speaker-conditioning."* Shipping it with the
  caveat is better than omitting it, because the alternative is someone rediscovering the limitation
  by measurement; but nothing in senselab should treat its output as a reliable decomposition.

### Isolation and cost

Subprocess venv `unasdiff`, Python 3.10. Requirements: `torch==2.6.0`, `torchaudio==2.6.0`
(CUDA-routed by `ensure_venv`), `numpy==1.23.5`, `scipy`, `librosa`, `einops`, `timm`, `thop`,
`toml`, `tqdm`, `av`, `soundfile`. `flash-attn` is omitted — upstream calls it optional, and building
it is slow and fragile relative to what it buys at these tensor sizes.

Runtime `git clone --depth 1` pinned to **`5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa`**, same
flock-and-replace pattern as D. The worker imports `models`, `diffusion` and `utils` directly and
**never** the `test_*.py` scripts: those call `torch.cuda.set_device(0)` at module import and abort
on any CPU host.

Cost, in deliberate contrast to D: the sampler calls `x.requires_grad_()` and backpropagates through
the source models at **every one of 200 steps**, times `n_sources`. CUDA is strongly recommended;
CPU is supported but documented as impractical. Tests are skip-marked like the other heavy backends,
with the skip reason naming CUDA.

Weights: mirror `speech_source.pt` and `sound_source.pt` to `sensein/unasdiff-diffusion-priors`,
private pending licensing, fetched by pinned revision. `SENSELAB_UNASDIFF_CHECKPOINTS` overrides
with a local directory.

Registry entries in `model_registry.yaml` and the compatibility matrix, matching the other isolated
backends.

---

## Licensing — applies to both C and D

Both upstream repositories report `license: null` on the GitHub API: no `LICENSE` file, no license
statement in the README. Absent a license, the default is all rights reserved, so neither source tree
may be vendored into a published package.

The handling is the same for both, and is the reason the runtime-clone pattern was chosen over
vendoring in the first place:

1. **No vendored source.** Each backend clones upstream at a pinned SHA at first use, into the user's
   own cache. senselab distributes no upstream code.
2. **Weights mirrors stay private** under the `sensein` org until upstream answers. A private mirror
   gives the checkpoints a pinned revision and content hash — which is what provenance in the parquet
   output needs — without redistributing them.

   Both now exist, created 2026-08-08, both private, both with per-file SHA-256 recorded in their
   plans:
   - `sensein/driftse-distilhubert-three-layers` @ `76a9448aae12e4c232b1d52c24899d0835db5782`
   - `sensein/unasdiff-diffusion-priors` @ `8d7c32204d1ba31cd9fca3cd64313fd711949b58`

   **Neither may be made public before its upstream licence question is answered.** That is stated in
   each model card as well as here, because the decision will be made by whoever reads one of them,
   not necessarily by whoever created them.
3. **An upstream issue for each** — posted 2026-08-08, [DriftSE#2](https://github.com/LiangXu123/DriftSE/issues/2)
   and [unasdiff#1](https://github.com/RunwuShi/unasdiff/issues/1) — asking for an explicit license and, ideally, a HuggingFace weights
   mirror. For DriftSE the issue also notes that the codebase is built on SGMSE+ (MIT) without
   carrying that license statement forward, which the authors will likely want to fix regardless.
4. `doc.md` in each module records the status, so the answer is visible where the decision is.

If either author declines or does not respond, that backend stays behind an operator-supplied
checkpoint path (`SENSELAB_*_CHECKPOINT*`) with no mirror, and `doc.md` says so. The code path is
identical; only the default source of the weights changes.

---

## Testing

Per `CLAUDE.md`, the suite runs **serially** and scoped to the directories touched. `pytest -n auto`
is not used: each xdist worker duplicates 535 MB of frameworks plus its own model weights, and
`ensure_venv` takes no lock, so two workers wanting the same subprocess venv delete each other's tree
mid-install.

| Workstream | Directories |
|---|---|
| A | `src/tests/audio/tasks/speaker_diarization_test.py`, `src/tests/utils/`, plus the HF-touching dirs for the `transformers>=5.3` bump |
| B | `src/tests/text/`, `src/tests/audio/workflows/` (the adapter) |
| D | `src/tests/audio/tasks/speech_enhancement_test.py` |
| C | `src/tests/audio/tasks/source_separation_test.py` |

Heavy backends (C, D, and three of A's four) carry skip marks naming their requirement — CUDA, or a
built subprocess venv — resolved through `subprocess_venv._cache_dir_path()` so the gate honours
`SENSELAB_VENV_CACHE` and matches where the venv is actually built.

Tests that assert on backend output must use fixtures long enough for the backend's own windowing to
produce a non-empty result. #537 found this the hard way: a 4.9 s fixture sat under child-adult's 10 s
window, so both `all(...)` assertions passed vacuously over an empty list regardless of correctness.
C's 4 s window and D's overlap-add chunking both need a fixture longer than one window for the
long-form path to be exercised at all.

## Non-goals

- No `analyze_audio` or `audio_analysis` wiring for any of the four. Deciding how a new model's
  output participates in a fold is a measurement, and it comes after the backends exist.
- No `run_config` changes.
- No task-compliance verification (cut from B on instruction).
- No new host dependencies and no new extras. `wordfreq`/`nltk` go into the existing `pii-detection`
  venv; C and D carry their own venvs.
- No vendored upstream source for C or D.
