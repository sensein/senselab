# DriftSE: upstream is MIT, and its inference code changed under us

Everything senselab's DriftSE backend knows that is not "what a caller types" lives here. The
module docstring, `speech_enhancement/doc.md` and the model registry describe the backend; this
document holds the measurements, the upstream history and the rejected alternatives.

Written 2026-08-18. Every upstream claim below was checked against the repository and the two
issues on that date, not carried over from an earlier note.

## 1. What changed upstream, verified

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | The licence is now MIT | **holds** | `LICENSE` at upstream commit `9ecfbac6` ("add MIT license", 2026-08-13) is the verbatim MIT text, `Copyright (c) 2026 Liang Xu, Diego Caviedes-Nozal, W. Bastiaan Kleijn, Longfei Felix Yan, Rasmus Kongsgaard Olsson`. Issue #2 is CLOSED with the maintainer's "MIT license added as requested". The HF mirror's card carries `license: mit`. GitHub's own licence API still reports `NOASSERTION`/"Other" — that is its detector reacting to the multi-line copyright header, not a second licence statement. |
| 2 | There is an official HF mirror, `LIANGXU123/DriftSE` | **holds** | Public, ungated, `license: mit`, 1659 files / 2.41 GB. Checkpoints at `logs/<variant>/last.ckpt`; the maintainer's own load recipe in issue #2 is `hf_hub_download("LIANGXU123/DriftSE", "logs/distillhubert_three_layers_with_z/last.ckpt")`. |
| 3 | `enhancement.py` now prioritises `ema` over `model` | **holds** | Commit `60333a68`: `if "ema" in checkpoint: load(checkpoint["ema"]) elif "model" in checkpoint: ...`. Maintainer's measured gap, 824 files: ema PESQ 3.00 / STOI 0.9333 / SI-SDR 15.8 against model PESQ 2.98 / STOI 0.9305 / SI-SDR 15.8. The presence of an `ema` key in the released checkpoint is the maintainer's statement, not something verified here — no local copy of the 1.14 GB file exists on this host. The worker therefore prefers `ema` and falls back to `model`, so an `ema`-less checkpoint still loads. |
| 4 | The inference code was misaligned with the paper and was fixed at `70bb6ded` | **holds, and it is behavioural** | The whole diff of `enhancement.py` from the previously pinned `695a64db` to upstream HEAD is two hunks: the `ema` branch above, and `model(Y_input + 0.05*z, t)` → `model(Y_input + 0.01*z, t)  # maximize test performance`, with the 0.05 line left commented as "better generalization". senselab's worker hardcoded `0.05`, so at the old pin it ran exactly the formulation the author now calls wrong. |
| 5 | `sigma` is an inference parameter with a measured effect | **holds, with one correction** | It is a *literal* upstream, not a flag: `enhancement.py` has no `--sigma` argument, so "the script default" means the constant on that line, 0.01. Independent reproduction by @julius-richter in issue #1, DistilHuBERT variant: σ=0 → PESQ 2.81 / SI-SDR 16.1, σ=0.01 → 2.80 / 16.1, σ=0.05 → 2.69 / 15.2. σ=0 and σ=0.01 are equivalent within noise; σ=0.05 costs about 0.11 PESQ and 0.9 dB SI-SDR. |
| 6 | Two checkpoint variants exist | **holds** | `distillhubert_three_layers_with_z` (PESQ 3.00 / SI-SDR 15.8) and `distillhubert_three_layers_pesq_sisdr_ccmse_with_z` (PESQ 3.50 / SI-SDR 20.2), both in the mirror, the second trained with PESQ/SI-SDR/CCMSE auxiliary losses (`pesq_weight` 1e-4, `sisdr_weight` 1e-4, `ccmse_weight` 1e-3 in its config against 0/0/0 in the first). Its advantage is therefore partly an optimisation of the metrics it is scored on, which is a reason to keep the first as the default rather than the higher number. |
| 7 | An unresolved integrity question, issue #1, still open | **holds** | @julius-richter evaluated the maintainer's own enhanced audio against a standard VoiceBank-DEMAND copy and got PESQ 2.99, against 3.50 with the maintainer's uploaded clean/noisy directories; a file-by-file comparison found **all 824 files differ** in both `clean` and `noisy`, some by one sample of length, others by max-abs 0.03–0.28 and not by a gain factor. Evaluating the maintainer's `noisy` against his `clean` gives PESQ 1.99 where VB-DMD noisy is normally 1.97. The maintainer explained the cause (he downsampled a 48 kHz copy with his own filter), re-scored against `JacobLinCool/VoiceBank-DEMAND-16k` — PESQ 2.59 for the plain variant, 2.98 for the † variant, both below the paper — and committed to retraining on that copy. So the published numbers rest on a test-set copy nobody else has reproduced, and the honest reading is that they are optimistic by roughly 0.4 PESQ. Nothing in this backend depends on them, but nothing here should quote them as settled either. |

Issue #3 (`torchaudio.load` fails with `ModuleNotFoundError: torchcodec`, CLOSED) touches upstream's
`requirements.txt` only. senselab's worker reads audio with `soundfile` and ignores that file, so it
is unaffected either way.

## 2. Pins

| | Old | New |
|---|---|---|
| Upstream code (`_DRIFTSE_COMMIT`) | `695a64db187500fa0d7bae23912680bd5d4df613` (2026-07-20) | `0a489dadfa2778e86e4b4b0af03f6255d2de8c69` (2026-08-18, upstream HEAD) |
| Weights (`_DRIFTSE_HF_REPO` @ `_DRIFTSE_HF_REVISION`) | `sensein/driftse-distilhubert-three-layers` @ `76a9448aae12e4c232b1d52c24899d0835db5782` | `LIANGXU123/DriftSE` @ `b99a25a637a9963d5c7557f0b70597fc54c7a0bb` |

The new code pin is at HEAD rather than at `70bb6ded` (the minimum that satisfies the requirement)
because `enhancement.py` is byte-identical from `60333a68` to HEAD, and the commits in between only
delete the stray `logs`/`out`/`data`/`latent_ckpt` symlinks from the tree and rename config files.
A clone at HEAD is the smallest one available.

**Switching weight sources is not a change of weights, and that is measured, not assumed.** The
two mirrors hold byte-identical files — sha256 `6f476a95cf747748…` (1 137 859 237 bytes) for
`distillhubert_three_layers_with_z` and `d5d62e08c3f6a57d…` (1 137 859 301 bytes) for the † variant,
identical on both repos. The only behavioural differences from this change are the ones intended:
`ema` in place of `model`, and σ 0.05 → 0.01.

The old checkpoint cache under `sensein` stays published; nothing points at it any more.

## 3. Where the architecture config now comes from

Upstream's mirror has one top-level `config.json` — `{"model_type": "DriftSE", "sample_rate": 16000}`
— which is HF download-tracking metadata, not the NCSN++ architecture config the worker needs. The
real configs live in the *code* repository, which the worker already clones at a pinned commit:

- `distillhubert_three_layers_with_z` → `config/with_z/v2_drift2_distillhubert_three_layers_adam.json`
- `distillhubert_three_layers_pesq_sisdr_ccmse_with_z` → `config/with_z/v2_drift2_distillhubert_three_layers_pesq_sisdr_ccmse.json`

Taking the config from the clone rather than mirroring it is what lets the whole config live at one
pinned commit instead of being copied into a weights repo. It is safe because every
inference-relevant key (`nf`, `ch_mult`, `num_res_blocks`, `attn_resolutions`, `image_size`,
`fourier_scale`, `resamp_with_conv`, `fir`, `fir_kernel`, `skip_rescale`, `resblock_type`,
`progressive*`, `init_scale`, `embedding_type`, `dropout`, `n_fft`, `hop_length`, `window_type`,
`center`, `spec_factor`, `spec_abs_exponent`, `model`, `train_add_gaussian`) is **identical**
between these repo configs and the `config.json` that shipped beside each checkpoint on the
`sensein` mirror — compared key by key, both variants, zero differences. They diverge only in
training-only keys (`SOAP`, `lr`, `data_dir`, `output_dir`, …). The `_adam` filename is upstream's
own rename at HEAD of the file that was `v2_drift2_distillhubert_three_layers.json`; it is the
config for the released `distillhubert_three_layers_with_z` checkpoint.

`SENSELAB_DRIFTSE_CHECKPOINT` still points at a directory holding `last.ckpt` + `config.json` and
still bypasses the Hub entirely; a caller supplying their own weights supplies their own config with
them.

## 4. Why the checkpoint is fetched file-by-file rather than as a snapshot

`resolve_model` downloads a whole snapshot. On the `sensein` mirror that was 2.3 GB of which
everything was wanted. On upstream's mirror it is **2.41 GB of which 1.14 GB is wanted**: the repo
also carries the second variant's checkpoint (another 1.14 GB) and 1648 demo wavs (133 MB).

So the backend now resolves the ref to a commit SHA with `resolve_revision` — the same
manifest-backed resolver `resolve_model` uses internally — and calls
`hf_hub_download(..., revision=<sha>)` for the single checkpoint file. A full 40-hex SHA triggers
`huggingface_hub`'s commit-hash shortcut, so a cached file resolves with no network at all, which is
the property `resolve_model` exists to guarantee. This is the pattern `qwen_tts.supported_speakers`
already uses and is allowlisted for; `driftse.py` joins it in
`hf_load_coverage_test.RAW_LOAD_EXCEPTIONS` for the same reason.

Rejected: keeping `resolve_model` and paying 1.3 GB per host for files no run reads. Also rejected:
mirroring the two files into a senselab-owned repo again, which is what upstream's mirror now makes
unnecessary.

## 5. The gate, and why it is gone

`api.py` used to carry: *"DriftSE's weights are on a private mirror pending an upstream licence
answer, so this backend must stay unreachable except by a caller naming it explicitly — no default
here and no entry in the audio_analysis workflow's model list may reference this prefix."*

Both halves of its premise are now false: the weights are on a public, MIT-licensed upstream mirror,
and the licence question is answered. The restriction is therefore lifted — DriftSE is a normal
selectable enhancement backend, nameable from a workflow config like any other model id.

What is *not* changed here, deliberately: DriftSE is still not the default enhancer, and no default
model list references it. Whether a one-step generative enhancer should displace SepFormer, or how a
second enhancer's output participates in the perturbation sample, is a measurement that has not been
made. A separate investigation is currently measuring whether the configured SepFormer default is
broken at all; this change leaves that decision to it.

## 6. Rationale moved out of the code

These four findings were multi-paragraph essays in `driftse.py`'s module docstring, in
`doc.md`, and in inline comments. They are facts about upstream, so they belong here.

### Why one network evaluation is enough (and why CPU is viable)

DriftSE (Xu, Caviedes-Nozal, Kleijn, Yan & Olsson, *Speech Enhancement Based on Drifting Models*,
Interspeech 2026 oral, arXiv 2604.24199) formulates enhancement as a distributional equilibrium
problem and reaches the clean-speech distribution in a single network evaluation, against 30 for
SGMSE+ and 8 for UNIVERSE++. On the DNS 2020 blind test set it reports WV-MOS 2.65 / SCOREQ 2.97.

The drifting field is computed in a frozen self-supervised latent space (HuBERT / WavLM /
DistilHuBERT), but that is the *training* signal. Inference is the backbone alone: one forward pass
under `no_grad`. Upstream's `enhancement.py` reaches only `backbones.ncsnpp_v2`,
`backbones.ncsnpp_v2_drift` and `util.other` — no Lightning, no `wandb`, no SSL encoder — so the
`latent_ckpt/` archive its README requires for training is not needed here at all. That is what
makes this the first generative enhancer in senselab that is genuinely CPU-viable.

### `pesq` and `pystoi` are inference dependencies, which only a real run revealed

`util/other.py` — the module the worker imports for `pad_spec` and `set_torch_cuda_arch_list` — does
`from pesq import pesq` and `from pystoi import stoi` at module scope (lines 7–8, still true at the
new pin). An earlier revision blamed `util/inference.py`, omitted both, and asserted their absence
in a test; the H100 run failed with `No module named 'pesq'`, and omitting `pystoi` would have
failed on the next line. The distinction that matters is between what the model computes and what
its import chain touches. Everything genuinely training-only — `scoreq`, `torch-pesq`,
`asteroid-filterbanks`, `wandb`, `pytorch-optimizer`, `torchinfo` — stays out, and a test asserts
that.

### The `upfirdn2d` path never compiles anything

`backbones/ncsnpp_utils/op/upfirdn2d.py` imports `torch.utils.cpp_extension.load` and never calls
it, hardcoding `upfirdn2d_op = None` under the comment *"Force PyTorch fallback to avoid CUDA_HOME
dependency"* — still true at the new pin. The dispatch
`if input.device.type == "cpu" or upfirdn2d_op is None` therefore always selects
`upfirdn2d_native` (plain `F.conv2d`), on CPU and CUDA alike. Confirmed on an H100: no `.so` beside
the source, empty `~/.cache/torch_extensions`, correct output on a 4.92 s clip.

Two consequences. The venv needs no build toolchain — no `nvcc`, no `CUDA_HOME` — which is why it
installs fast and portably. And the CUDA kernel's speed is not on the table, since that path is
unreachable. An upstream commit restoring the `load()` call would change both, which is one more
reason the code commit is pinned.

### Why a subprocess venv, given the deps would fit

Not for dependency conflict — the inference dependency set would satisfy senselab core. The upstream
repository has no installable package and its top-level module names are `backbones`, `util`,
`config` and `data`. Injecting a generic `util` onto the host interpreter's `sys.path` is the kind of
hazard that surfaces months later as an unrelated import resolving to the wrong module.

### The three deviations from upstream's script

1. **`torch.load(..., weights_only=True)`.** Upstream omits it. The checkpoint is a foreign pickle;
   the unrestricted unpickler is arbitrary code execution at enhancement time. (The repository being
   MIT changes the licence question, not the pickle question.)
2. **Overlap-add chunking for long inputs.** Upstream runs one STFT over an entire file. The NCSN++
   backbone carries attention layers, so memory grows superlinearly in duration. Enhancement is
   per-segment consistent — there is no cross-segment identity to preserve, unlike separation, where
   which-speaker-is-which must stay fixed across a boundary — so Hann-tapered overlap-add is safe
   here in a way it would not be for a separation backend.
3. **A recorded RNG seed.** `train_add_gaussian` is set in both released checkpoints' configs, so the
   forward pass consumes a Gaussian sample and is stochastic. An unseeded rerun would produce
   different audio and make any cached artifact keyed on that output non-reproducible.

## 7. Considered and dropped: `SpeechBrainModel(..., revision="main")` in `api.py`

This was raised as a pinning violation — the default enhancement model is constructed with a bare
ref, and `revision_pinning_guard_test.py` sweeps only subprocess-worker files, so no test covers it.
On reading the mechanism it is **not** a violation, and nothing was changed. Recorded here so it is
not re-filed:

- `SpeechBrainModel` subclasses `HFModel` (`utils/data_structures/model.py:166`), whose
  `_resolve_commit_sha` is a `model_validator(mode="after")`. Constructing with `revision="main"`
  resolves `commit_sha` there and then, reusing the SHA `check_hf_repo_exists` → `ensure_hf_model`
  already computed — no extra network call.
- The load goes through `resolve_model(repo_id, revision)` (`utils/dependencies.py:606`), which *is*
  the two-call pattern: it returns the immutable 40-hex SHA and the `snapshots/<sha>/` path.
- SpeechBrain's `from_hparams` takes no `revision`, so `speech_enhancement/speechbrain.py:68`
  pointing it at the snapshot path is the documented route for exactly that case.

A ref at this call site is therefore safe by construction. One residual observation, not acted on and
not a load defect: `SpeechBrainEnhancer._models`' key (`f"{path}-{model.revision}-{device}"`) and
`speechbrain_savedir(path, revision)` both key on the declared revision, so two upstream commits of
`speechbrain/sepformer-wham16k-enhancement` would share one in-process cache entry and one savedir
within a process whose manifest binding changed. Whether that is worth commit-addressing is a
question about those two keys, not about the pin.

For scale if anyone does take it up: an AST sweep for a literal, non-SHA `revision=` keyword in
executable code under `src/senselab/` (docstring examples excluded) finds 16 sites in 15 files, of
which three pass the ref to a loader directly — `forced_alignment.py:660,664` and
`audio_analysis/adaptive/backends.py:123`, all `load_hf_resilient(..., revision="main")` — and the
rest are default-model constructions of the kind analysed above. Widening the guard means classifying
all sixteen first; it is its own task, not a rider on this one.
