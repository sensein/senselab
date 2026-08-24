# Source separation

## unasdiff (unsupervised separation via diffusion priors)

[unasdiff](https://github.com/RunwuShi/unasdiff) (Shi, Runwu et al., *Unsupervised Audio Source
Separation using Diffusion Priors*, AAAI 2026) separates a mixture into its component sources
without ever training on mixtures. It factors the mixture likelihood into two independently
trained *unconditional* diffusion priors — a speech prior and a sound prior, each trained only on
single-source audio — and runs posterior sampling at inference time to pull each source back out
of the mixture. That is what makes it usable for the off-target-speaker-detection problem this
repository is tracking: there is no dataset of "target speech + arbitrary intruder" mixtures to
train a supervised separator on, but there are large unconditional speech and sound corpora to
train priors on separately.

### senselab writes the driver

Upstream ships training code and the two benchmark scripts its paper's numbers came from
(`benchmark_musdb.py`, `benchmark_urmp.py`); it has no installable package and no inference-only
entry point. Its three `test_*.py` scripts (`test_speech_sound.py`, `test_soundevent.py`,
`test_speech_speech.py`) call `torch.cuda.set_device(0)` at module import and abort outright on a
CPU host, so they are not usable as a library either. The worker script in
[`unasdiff.py`](unasdiff.py) is therefore senselab's own driver: it reuses upstream's model
construction (`models.py`) and diffusion sampler (`diffusion.py`) directly, and reimplements only
the benchmark scripts' `load_model` — whose EMA-vs-raw distinction is load-bearing, since loading
the non-EMA weights runs without error but separates measurably worse. The three separation modes,
the public `separate_audios` API, and long-form chunking (below) are equally senselab's
construction; nothing here is a thin wrapper around an upstream CLI.

### Why `p_sample_loop_group(orig_x=...)` is not an oracle

This is the paragraph to read before trusting the worker script, because the call site looks like
cheating. `separate_window` in the worker builds

```python
orig_x = torch.cat([mix] + [torch.zeros_like(mix)] * (n_src - 1), dim=-1)
```

and passes it to `gaussian.p_sample_loop_group(..., orig_x=orig_x, ...)` alongside the mixture
itself as `measurement`. Passing anything named `orig_x` — a name upstream itself uses for its
benchmark ground truth — reads as handing the sampler an answer key. It is not: upstream's
`p_sample_loop_group` **ignores** the `measurement` argument at every one of its 200 steps and
instead recomputes `measurement = degradation(orig_x, n_src)` from `orig_x`, where `degradation`
splits its input along time into `n_src` equal chunks and sums them. Because `orig_x` here is
`[mixture, zeros, ..., zeros]` laid out along that same time axis, `degradation(orig_x, n_src)`
sums to exactly the mixture — the same signal `measurement` already held. No per-source
information — no true `orig_x = [source_1, source_2, ...]`, which is what upstream's benchmark
scripts actually pass when scoring against ground truth — ever reaches the sampler. What looks like
a ground-truth argument is, at this call site, a mechanism for satisfying the diffusion process's
internal consistency check with the one signal separation is actually given: the mixture. Verified
by reading `degradation` and the `p_sample_loop_group` step loop at the pinned commit, not assumed.

### Two label spaces, not one

The sound prior's conditioning embedding has 50 slots (`num_class=50` in
`config/atten_unet_fsd/config.toml`), of which 41 were populated by training on FSD50K subset
labels. `senselab.audio.tasks.source_separation.unasdiff.load_fsd_class_map_document` loads this
map from `data/fsd41_classes.json`, and `api.resolve_source_classes` resolves a caller's class
names against it, raising (and enumerating the 41 valid names) rather than silently falling back to
index 0 on a typo — index 0 is `"Hi-hat"`, a real class, so a fallback there would condition on the
wrong sound silently.

The speech prior's conditioning label space is disjoint from the sound prior's and has exactly one
member: unconditional speech, index 0. The two label spaces sharing the integer 0 for unrelated
meanings ("Hi-hat" vs. "unconditional speech") is precisely why `separate_with_unasdiff` takes
`mode` as an explicit, required argument rather than inferring which prior a slot should load from
`source_class_indices` alone — the index alone is ambiguous.

`separate_with_unasdiff` validates every slot's label against the prior `mode` says that slot
loads before the worker ever starts (`_validate_source_class_indices`): a speech slot accepts only
`0`, a sound slot only `0..40`. This is a second, lower check than `api.resolve_source_classes` —
that one catches a typo in a class *name*, this one catches an out-of-range raw index reaching
`separate_with_unasdiff` directly, which bypasses the name lookup entirely.

### Three modes, and the speech–speech caveat

`separate_audios` exposes three modes, matching which of the two priors each slot loads:

- **`speech_sound`** (default): slot 0 is the speech prior (unconditional); the remaining
  `n_sources - 1` slots are the sound prior, each conditioned on one of `source_classes`.
- **`sound_sound`**: every slot is the sound prior, one class name per slot.
- **`speech_speech`**: every slot is the speech prior. No `source_classes` needed, since the speech
  prior's label space has exactly one member — but ship with upstream's own caveat from its README:

  > "The source-model-based separation approach is not well suited for same-class source separation
  > (e.g. speech separation), because it lacks speaker-conditioning. In future work, we will attempt
  > to address such issue."

  senselab exposes `speech_speech` anyway, because the alternative is a caller rediscovering the
  limitation by measurement rather than by reading this line. Nothing downstream should treat its
  output as a reliable decomposition of two overlapping speakers.

`p_sample_loop_group` zips one model object against one label per slot, so `n_sources` model
instances are always constructed, including in `speech_speech` where every slot shares the same
weights — a separate `deepcopy`'d instance per slot, never one instance reused across slots.

### Chunking: senselab's construction, not upstream's

unasdiff was trained and benchmarked on fixed 4 s clips (`_WINDOW_S = 4.0`, the config's own
diffusion window — not a tunable). Upstream has no path for longer inputs at all. senselab's
`separate_with_unasdiff` splits anything longer into 4 s windows at 50% overlap (`_OVERLAP_S =
2.0`), separates each window independently, and stitches the results back with a Hann-tapered
overlap-add.

The failure mode this scheme exists to prevent: each window is separated with no notion of any
other window, so slot 0 in window *k* need not be the same source as slot 0 in window *k + 1* — the
sampler has no memory across calls. Concatenating naively swaps sources mid-file, and the result is
worse than the mixture itself. `align_permutations` (scored by `_assignment_scores`, a zero-mean,
unit-normalised correlation over each window pair's overlap region) reorders every window after the
first to match the previous *aligned* window's slot order before the overlap-add runs. This is what
makes long-form separation different from single-model enhancement (DriftSE, `SepFormer`): those
have no cross-segment identity to preserve, so plain overlap-add is safe there in a way it is not
here.

The permutation-alignment metric is validated, not the operating threshold on top of it.
`data/permutation_alignment.json` records a 200-trial synthetic calibration (independent
`randn(2000)` sources, `torch.manual_seed(0)`): known-correct assignments separate from ambiguous
ones by more than five orders of magnitude in the assignment margin (`best_score - second_best_score`;
known-correct p05/p50 = 1.93 / 2.00, ambiguous p50/p95 = 2.3e-6 / 6.1e-6). That gap demonstrates the
scale-invariant correlation metric itself can tell a real permutation swap from a genuine tie — it
does **not** constitute a measurement of real unasdiff output, since both categories are independent
i.i.d.-Gaussian constructions, not the actual overlap-region statistics of two consecutive 4 s
diffusion-sampled windows.

**No margin gate is defined, and one does not belong here.** senselab ships unasdiff as a tool; it
does not tune the tool's output. Deciding that a margin is too low to trust is a decision, with its
own consequences and its own derivation, and it belongs to whatever layer is making that decision —
not to a wrapper whose job is to invoke the sampler and hand back what it produced. So
`separate_with_unasdiff` reports every window boundary's margin in
`metadata["unasdiff_alignment_margins"]` as **data**, and a caller who wants to distinguish a
confident alignment from a coin flip reads that list and sets its own bar.

Keeping the two apart is what makes the tool's parameters legible: `mode`, `n_sources`,
`source_class_indices`, `seed` and `diffusion_steps` all say *how to run unasdiff*, and nothing in
this module says *what to conclude* from what comes back.

### `diffusion_steps` re-specifies the schedule; it does not subsample it

Both released priors were trained at a fixed reverse-diffusion schedule length, `T=200`, with a
linear beta schedule from `1e-4` to `0.02` (`config/*/config.toml`). `diffusion_steps` is passed
straight through to `diffusion.GaussianDiffusion(steps=diffusion_steps, ...)` as that schedule's
*length*, not as a subsampling stride the way DDIM's step count is — so a value other than `200`
does not run the trained process faster or slower, it constructs a *different* process the priors
were never trained against. Concretely, at the pinned commit:

- **`steps <= 50` wraps silently.** `p_sample_loop`/`p_sample_loop_group` seed the augmented-mixture
  init at `t = t_last - 50` where `t_last = steps - 1`. For `steps <= 50` this index is negative,
  and indexing the precomputed alpha/beta arrays with a negative index does not raise — it wraps to
  the tail of the schedule and silently seeds from the wrong point.
- **`steps == 51` collapses the same init to `t = 0`** — one step above the wrap, so the
  augmented-mixture initialisation that is supposed to seed the reverse process partway through
  degenerates to (near) no noise at all.
- **`steps > 200` disables guidance for the first `steps - 200` iterations.** The DPS-style
  corrector both samplers call is gated `if i < 200 and i >= 0:`, a literal upstream hardcodes
  against its own `T=200` training value, not derived from `steps`. Reverse diffusion counts `i`
  down from `steps - 1` to `0`, so every iteration with `i >= 200` — the first `steps - 200` of
  them — runs with no measurement guidance at all.

None of this is a knob senselab can safely expose without vendoring and patching upstream's
sampler, which the licensing position (below) forbids. `api.separate_audios` therefore accepts only
`diffusion_steps=200` and raises `ValueError` naming the parameter for anything else.
`unasdiff.separate_with_unasdiff` is the one exception: it keeps accepting any positive integer,
because it is the layer a future retrained prior (trained at a different `T`) would use directly —
but today, against the checkpoints this backend ships, `200` is the only value with any published
or measured basis, and every other value hits one of the three mechanisms above.

Each of the 200 steps is one network evaluation per slot, and is this backend's dominant cost: on an
exclusive A100, 14.027 s of audio (7 windows) measured 560.71 s end to end, i.e. `560.71 / (7 x 200)
= 0.4 s` per window-step (see `specs/20260818-071500-unasdiff-device-timeout-pcm16/design.md`, D-2).
For contrast, DriftSE (`speech_enhancement/driftse.py`) reaches the clean-speech distribution in a
single step, and SGMSE+ takes 30 — 200 is a lot, by the standard of the other diffusion-based
backends in this repository.

### flash-attn is opt-in, not unconditional

flash-attn is absent from the venv's requirements by default (see the module docstring's "Why a
subprocess venv" section for the verification that upstream's `atten_unet.py` treats it as optional
in fact, not just in its README). Setting `SENSELAB_UNASDIFF_FLASH_ATTN` truthy (`1`/`true`/`yes`/`on`)
appends `flash-attn==2.5.8` to the venv's requirements on the next build, matching the operator-override
style of `SENSELAB_TORCH_INDEX_URL` and `SENSELAB_UNASDIFF_CHECKPOINTS`.

The decision is opt-in rather than unconditional because of a failure mode this branch already hit:
`av==14.4.0` had no wheel, fell back to a source build, and took the *entire* venv install down with
it — a training-only dependency the inference path never even imports. flash-attn is considerably
more build-fragile than that: it needs a matching CUDA toolkit, `--no-build-isolation`, and 10-30
minutes with `MAX_JOBS` tuning to avoid an out-of-memory compile. Upstream's own code already handles
a *missing* flash-attn gracefully (the `ImportError` fallback above); nothing handles the *install*
failing, so making it unconditional would convert that graceful runtime fallback into a hard
venv-creation failure on any host without a working `nvcc`.

**Toggling the flag forces a full venv rebuild.** `ensure_venv` (`subprocess_venv.py`) keys venv reuse
on a marker containing the requirements list, and the requirements list is exactly what this flag
changes — so flipping it from unset to set (or back) costs a 10-30 minute reinstall, not an
incremental change, the next time this backend runs. A failed opt-in build fails loudly for the
person who set the env var, which is the point: they are the one who can supply a working `nvcc` or
turn the flag back off.

### The default timeout scales with the device

`separate_with_unasdiff`'s derived ceiling (`_default_timeout_s`) multiplies the measured CUDA
per-window-step cost (`_SECONDS_PER_WINDOW_STEP_CUDA = 0.4`, the A100 figure above) by
`_CPU_TIMEOUT_MULTIPLIER = 45` whenever `device` names CPU or MPS: the review that produced this
fix measured a roughly 45x CPU/A100 wall-time ratio on this workload, and a ceiling sized for CUDA
killed every CPU run before its first window landed, discarding whatever had already completed
(`separate_with_unasdiff` has no salvage path — see the timeout section above). `device=None`
(the worker chooses) keeps the CUDA figure, matching the ceiling's pre-existing behaviour for the
common case where the worker is expected to find a GPU. An explicit `timeout_s` still overrides
this derivation outright.

### Measured runtime

Not yet measured in this repository: every prior task in this plan, and this one, ran on a host
with no CUDA device (`torch.cuda.is_available()` is `False` here), so the skip-gated end-to-end
test (`test_unasdiff_separates_a_mixture_into_n_sources` in `source_separation_test.py`) has only
ever been exercised as a skip. "Impractical on CPU" therefore remains an inference from the
sampler's shape rather than a measured number: 200 diffusion steps, each evaluating `n_sources`
model instances and backpropagating through the corresponding prior (the DPS-style guidance term
`p_sample_loop_group` computes needs a gradient through the network, not just a forward pass) —
a cost more like `n_sources` training-time backward passes per step than a single inference forward
pass. The runtime number a CUDA host produces for one 4 s, two-source `speech_sound` separation,
and the hardware it was measured on, belongs here as soon as that host exists; do not read the
number above as having already been supplied.

### Provenance metadata

Every returned `Audio` carries `metadata["unasdiff"]`: `mode`, `source_classes` (names, not indices
-- `None` for `speech_speech`), `n_sources`, `diffusion_steps`, `upstream_commit` (the pinned clone
commit), `checkpoint_revision`, and `device`. `checkpoint_revision` is the resolved 40-hex commit of
the checkpoint mirror -- `resolve_model` returns `(sha, path)`, and it used to be discarded, so
nothing downstream could tell which commit of the mirror actually produced a given separation, only
which ref was requested. It is `None` when the caller supplied checkpoints directly
(`checkpoint_dir` or `SENSELAB_UNASDIFF_CHECKPOINTS`) rather than through the pinned mirror -- there
is no commit to attribute those to. Same pattern as ClearVoice's `metadata["clearvoice"]` below.

### Licensing

The upstream repository carries no `LICENSE` file and no license statement in its README. A licence
request was opened upstream on 2026-08-08 and is unanswered:
<https://github.com/RunwuShi/unasdiff/issues/1>. Pending that answer, senselab vendors none of
upstream's code — the worker clones it at a pinned commit
(`5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa`) into the user's own cache at first use, the same
pattern as DriftSE (`speech_enhancement/driftse.py`).

The checkpoint mirror under `sensein` (`sensein/unasdiff-diffusion-priors`) is **public**, so the
backend is usable during the alpha, and its licence is **unknown** — those are two separate facts and
both matter. Publishing the mirror makes the weights reachable; it grants no rights over them. No
terms have been offered upstream, so treat the weights as all-rights-reserved by default, and consult
<https://github.com/RunwuShi/unasdiff> before any use that depends on licence terms.
`SENSELAB_UNASDIFF_CHECKPOINTS` remains for a caller supplying their own checkpoints. See the model
registry entry for the pinned revision.

### Not wired into `audio_analysis`

unasdiff is reachable only by calling `senselab.audio.tasks.source_separation.api.separate_audios`
directly (optionally passing an `HFModel` whose id starts with `sensein/unasdiff`, purely as a
containment check — there is exactly one backend today). It is not in any default model list and
`scripts/analyze_audio.py` never reaches it. The licensing position above is the reason: an
unresolved licence request must not end up load-bearing in a default pipeline.

## ClearVoice MossFormer2_SS_16K

[ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)'s two-speaker separator
(Apache-2.0), in an isolated subprocess venv. Upstream's SI-SNRi leads SepFormer on LRS2-2Mix
(15.5 against 13.5) and matches SPMamba on WHAM! (17.4).

```python
from senselab.audio.tasks.source_separation import separate_audios
from senselab.utils.data_structures import HFModel

sources = separate_audios(audios, model=HFModel(path_or_uri="alibabasglab/MossFormer2_SS_16K"))
```

Two things a caller needs to know, neither in upstream's documentation:

- **It separates speakers, not classes.** Measured on a recording with four verified cough bursts, it
  assigned each burst to whichever of its two slots was free rather than isolating cough as a class.
  Nothing downstream should read slot 0 as "speech" and slot 1 as "everything else".
- **Its sources are not at the input's level.** Upstream RMS-normalises this checkpoint's input to
  −25 dBFS and then, on the multi-source branch only, never applies the inverse. senselab reproduces
  that rather than silently correcting it, so numbers agree with upstream's own tool, and reports the
  scalar: every returned `Audio` carries `metadata["clearvoice"]["input_norm_scalar"]` and
  `input_norm_applied_to_output: false`. Multiply by the scalar to restore the input's level.

`n_sources` is fixed at 2 by the checkpoint, and `mode` / `source_classes` / `seed` /
`diffusion_steps` describe unasdiff's priors — passing any of them with this model raises rather than
being ignored. Each returned `Audio` records the resolved commit its weights came from; see
`specs/20260819-clearvoice-integration/design.md`.
