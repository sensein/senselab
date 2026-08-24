# Source separation

## unasdiff (unsupervised separation via diffusion priors)

[unasdiff](https://github.com/RunwuShi/unasdiff) (Shi, Runwu et al., *Unsupervised Single-Channel
Audio Separation with Diffusion Source Priors*, AAAI 2026, [arXiv:2512.07226](https://arxiv.org/abs/2512.07226))
separates a mixture into its component sources
without ever training on mixtures. It factors the mixture likelihood into two independently
trained *unconditional* diffusion priors — a speech prior and a sound prior, each trained only on
single-source audio — and runs posterior sampling at inference time to pull each source back out
of the mixture. That is what makes it usable for the off-target-speaker-detection problem this
repository is tracking: there is no dataset of "target speech + arbitrary intruder" mixtures to
train a supervised separator on, but there are large unconditional speech and sound corpora to
train priors on separately.

### senselab writes the driver

Upstream ships training code and the three benchmark scripts its paper's numbers came from
(`test_speech_sound.py`, `test_soundevent.py`, `test_speech_speech.py`); it has no installable
package and no inference-only entry point. Those same three scripts call
`torch.cuda.set_device(0)` at module import and abort outright on a CPU host, so they are not
usable as a library either. The worker script in [`unasdiff.py`](unasdiff.py) is therefore
senselab's own driver: it reuses upstream's model construction (`models/atten_unet.py`) and
diffusion sampler (`diffusion/gaussian_diffusion.py`) directly, and reimplements only the
benchmark scripts' `load_model` — whose EMA-vs-raw distinction is load-bearing, since loading
the non-EMA weights runs without error but separates measurably worse. The three separation modes,
the public `separate_audios` API, and long-form chunking (below) are equally senselab's
construction; nothing here is a thin wrapper around an upstream CLI.

### Why `orig_x=...` is not an oracle

This is the paragraph to read before trusting the worker script, because the call site looks like
cheating. `separate_window` in the worker builds

```python
orig_x = torch.cat([mix] + [torch.zeros_like(mix)] * (n_src - 1), dim=-1)
```

and passes it to whichever sampler the mode selects (`gaussian.p_sample_loop_group(...)` for
`speech_sound`, `gaussian.p_sample_loop(...)` for the other two modes — see "Two samplers, one mode
dispatch" below) alongside the mixture itself as `measurement`. Passing anything named `orig_x` — a
name upstream itself uses for its benchmark ground truth — reads as handing the sampler an answer
key. It is not: both samplers **ignore** the `measurement` argument at every one of their 200 steps
and instead recompute `measurement = degradation(orig_x, n_src)` from `orig_x`, where `degradation`
splits its input along time into `n_src` equal chunks and sums them. Because `orig_x` here is
`[mixture, zeros, ..., zeros]` laid out along that same time axis, `degradation(orig_x, n_src)`
sums to exactly the mixture — the same signal `measurement` already held. No per-source
information — no true `orig_x = [source_1, source_2, ...]`, which is what upstream's benchmark
scripts actually pass when scoring against ground truth — ever reaches the sampler. What looks like
a ground-truth argument is, at this call site, a mechanism for satisfying the diffusion process's
internal consistency check with the one signal separation is actually given: the mixture. Verified
by reading `degradation` and both samplers' step loops at the pinned commit, not assumed.

### Two label spaces, not one

The sound prior's conditioning embedding has 51 slots. `config/atten_unet_fsd/config.toml` sets
`num_class=50`, and `models/atten_unet.py`'s `LabelEmbedder` allocates `num_classes +
use_cfg_embedding` rows; `use_cfg_embedding` is `True` because `dropout_prob=0.1 > 0`. Row 50 is
therefore an untrained classifier-free-guidance null token (`token_drop`'s fallback when a label is
dropped for CFG training) — never a class, and never reachable, since upstream's own inference path
has no reachable CFG call (see "No configuration surface" below). Rows 0-40 were populated by
training on FSDKaggle2018 subset labels; rows 41-49 are untrained headroom. The model holds five
independent copies of this table (`y_embedder_1`..`y_embedder_5`), each initialised independently
and none trained past row 40. `senselab.audio.tasks.source_separation.unasdiff.load_fsd_class_map_document`
loads the 41 trained names from `data/fsd41_classes.json`, and `api.resolve_source_classes`
resolves a caller's class names against it, raising (and enumerating the 41 valid names) rather
than silently falling back to index 0 on a typo — index 0 is `"Hi-hat"`, a real class, so a
fallback there would condition on the wrong sound silently.

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

  **`sound_sound` carries the same caveat, for a different reason.** Label conditioning here is an
  unamplified embedding lookup: there is no guidance-scale knob (nothing exposes one, and upstream's
  own classifier-free-guidance path is dead code reachable through no code path this backend calls
  — the null row it would condition on, row 50, was never trained; see "Two label spaces, not one"
  above). Both sound slots start the reverse process from the *identical* noised mixture (the
  augmented-mixture init both samplers build is one tensor, tiled across slots), so the per-slot
  class embedding is the entire deterministic difference between what the two slots produce. The
  paper reports 11.5% failure on two-sound mixtures. Two sources summing to approximately the
  mixture with an arbitrary — not necessarily label-matched — assignment of content to slots is
  the expected degenerate outcome on a failure, not a bug: the labels choose which priors run, they
  do not reliably steer which slot collects which event.

Only `speech_sound` builds `n_sources` model instances: it needs two different priors in the same
call, which only upstream's multi-model sampler, `p_sample_loop_group`, supports (it zips one model
object against one label per slot). `sound_sound` and `speech_speech` use only one prior each and
now build exactly one model instance, run through upstream's single-model sampler, `p_sample_loop`,
which batches every slot into that one instance's forward pass instead — see "Two samplers, one
mode dispatch" below.

**`n_sources` is capped at 3.** Shi et al. (AAAI 2026) evaluate this method at up to three sources;
`separate_with_unasdiff` raises `ValueError` above that rather than extrapolating into a regime the
paper never measured.

### Two samplers, one mode dispatch

Upstream ships two reverse-diffusion loops with the same step logic and the same `degradation`/
`orig_x` mechanism (see above), differing only in how many priors they drive per call:

- `p_sample_loop_group` takes a **list** of model objects and a list of per-slot labels, zips them
  one-to-one, and runs one forward pass per slot per step — `n_sources` forward passes, and
  `n_sources` model instances, always.
- `p_sample_loop` takes a **single** model object and a single label tensor covering every slot,
  batches all `n_sources` slots into one forward pass per step, and needs only one model instance.
  This is what upstream's own single-prior benchmark scripts use.

`speech_sound` needs two different priors (speech and sound) in the same call, so it is the one mode
that must use `p_sample_loop_group`. `sound_sound` and `speech_speech` each use only one prior, so
the worker builds one model instance and calls `p_sample_loop` — reproducing upstream's own
single-prior scripts exactly, at `(n_sources - 1)` fewer model instances than the previous
`p_sample_loop_group`-for-everything shape used. `speech_speech` passes `model_kwargs=None` to
`p_sample_loop`, matching upstream's own script for that mode: the speech prior has no conditioning
to give it, and the sampler already tolerates `model_kwargs=None`. `sound_sound` passes the plain
list of per-slot sound-prior labels.

`test_the_sampler_choice_matches_upstream_per_mode` in `source_separation_test.py` is a structural
(AST) check on the worker source for this dispatch, not an execution of it: the mocked-subprocess
tests elsewhere in this file never run the real worker script, so they cannot observe which sampler
function it calls.

### Chunking: senselab's construction, not upstream's

unasdiff was trained and benchmarked on fixed 4 s clips (`_WINDOW_S = 4.0`, the paper's training
clip length and the fixed length upstream's `test_speech_sound.py`/`test_soundevent.py`/
`test_speech_speech.py` all hard-code as `tgt_len_sec` -- not a config field, and not a tunable).
Upstream has no path for longer inputs at all. senselab's
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

**Alignment is chained, so one ambiguous boundary corrupts everything after it.** Each window is
aligned onto the *previous aligned* window's slot order, not onto window 0's — a single boundary
whose margin lands in the ambiguous band (`~1e-6`, not the confident band's `~2.0`) commits to
whichever permutation happens to score marginally higher there, and every later window inherits
that choice: there is no mechanism that revisits or corrects it once made. The margins list is how
a reader detects this after the fact — a run of confident margins followed by one near-zero value
and then more confident-looking values downstream is the signature of a flip that was never
undone, not evidence the flip did not happen.

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

### No configuration surface for the scheduler, solver, or guidance

The beta schedule (linear, `1e-4..0.02`), the sampler (`p_sample_loop`/`p_sample_loop_group`), and
the DPS-style guidance term (`CorrectorVPConditional.update_fn_recons`, gated `i < 200`) are all
literals in upstream's `diffusion/gaussian_diffusion.py` at the pinned commit — not config fields,
not constructor arguments with a default, nothing `GaussianDiffusion(steps=...)` exposes a way to
override. Exposing any of them as a senselab-level knob would mean vendoring and patching that file,
and the licensing position below is exactly why this backend vendors nothing: upstream ships no
`LICENSE` and an unanswered clarification request, so patching and redistributing its sampler is not
available as a design option today. This is the same fact `diffusion_steps` runs into above — it is
not a scheduler knob, and there being no path to a real one is why it stays fixed at `200`.

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
(`separate_with_unasdiff` has no salvage path — completed windows are counted in the error message
but not preserved; see `specs/20260818-071500-unasdiff-device-timeout-pcm16/design.md`, D-2).
`device=None`
(the worker chooses) keeps the CUDA figure, matching the ceiling's pre-existing behaviour for the
common case where the worker is expected to find a GPU. An explicit `timeout_s` still overrides
this derivation outright.

### Measured runtime

The skip-gated end-to-end test (`test_unasdiff_separates_a_mixture_into_n_sources` in
`source_separation_test.py`) has been exercised as a skip on every host this plan has run on so
far (`torch.cuda.is_available()` is `False` here) — that test alone remains unmeasured. A separate
measurement exists, though: an exclusive A100 node ran 200 diffusion steps over 14.027 s of audio
(7 windows) in 560.71 s, i.e. `560.71 / (7 x 200) = 0.4 s` per window-step, RTF ≈ 40x
(`specs/20260818-071500-unasdiff-device-timeout-pcm16/design.md`, D-2). That measurement is the
`_SECONDS_PER_WINDOW_STEP_CUDA` this module's default timeout is built on. CPU is unmeasured
directly, but a separate review measured roughly 45x that A100 wall-time on this workload (see
"The default timeout scales with the device" above) — 200 diffusion steps, each evaluating
`n_sources` model instances and backpropagating through the corresponding prior (the DPS-style
guidance term `p_sample_loop_group`/`p_sample_loop` computes needs a gradient through the network,
not just a forward pass), is a cost more like `n_sources` training-time backward passes per step
than a single inference forward pass, which is consistent with CPU being far slower still than the
A100 figure above.

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
