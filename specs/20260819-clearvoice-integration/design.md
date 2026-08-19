# ClearerVoice-Studio in senselab: all of it, and where each part lands

Upstream: <https://github.com/modelscope/ClearerVoice-Studio> (Alibaba Speech Lab, Apache-2.0).
Read at commit `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`; the inference platform read from the
installed `clearvoice==0.1.2` distribution rather than from the repository, because the wheel and the
tree differ in one load-bearing way (D-3).

## 1. What upstream actually ships

Established from `clearvoice/network_wrapper.py`'s dispatch table and the repository tree, not from
the README's prose, which is looser than the code.

Three components:

1. **ClearVoice** — the inference platform, on PyPI. Four tasks, six checkpoints:

   | Upstream task | Model | Rate | Outputs |
   |---|---|---|---|
   | `speech_enhancement` | `FRCRN_SE_16K` | 16 kHz | 1 |
   | `speech_enhancement` | `MossFormerGAN_SE_16K` | 16 kHz | 1 |
   | `speech_enhancement` | `MossFormer2_SE_48K` | 48 kHz | 1 |
   | `speech_separation` | `MossFormer2_SS_16K` | 16 kHz | 2 |
   | `speech_super_resolution` | `MossFormer2_SR_48K` | 48 kHz | 1 |
   | `target_speaker_extraction` | `AV_MossFormer2_TSE_16K` | 16 kHz | 1 per face track |

   All six repositories are `alibabasglab/<MODEL>` on HuggingFace, all Apache-2.0, all carrying a
   `last_best_checkpoint` text manifest naming the weight files.

2. **SpeechScore** — speech-quality assessment, 18 metric families. **Not on PyPI.** The
   `speechscore` distribution on PyPI is an unrelated package by another author (Gasser Elbanna,
   McDermott Lab); Alibaba's SpeechScore exists only as a directory of the git repository, with its
   model weights committed alongside the code (`nisqa.tar`, four DNSMOS ONNX graphs,
   `distill_mos_v7.pt`).

3. **train/** — training and finetuning recipes. The README advertises target-speaker extraction
   "conditioned on a reference speech (8 kHz)", "on body gestures", and "neuro-steered on EEG", and
   those are real, but they are `train/target_speaker_extraction/config/*.yaml` recipes with no
   ClearVoice inference entry point and no released checkpoint reachable through it. senselab
   integrates inference, so they are out of scope; they are recorded here so a later reader does not
   mistake their absence for an oversight.

### D-1: the capability set is six checkpoints and one scorer, not "speech enhancement"

The brief's description ("enhancement, separation, target-speaker extraction, super-resolution, and
SpeechScore") matched the code. The two corrections worth recording are that separation is 16 kHz
only in the shipped platform (the README's "8 kHz & 16 kHz" is about the training recipes), and that
SpeechScore's four non-intrusive metrics are usable with no reference at all — which is what makes it
worth integrating for this repository's uncertainty work, where a clean reference never exists.

## 2. Where each capability lands, and why

The SpeechBrain precedent is the governing one: senselab exposes SpeechBrain from
`speaker_embeddings/`, `speech_enhancement/`, `speaker_verification/` and others, never from a
`speechbrain/` package. A vendor is not a capability.

| Capability | Package | Argument |
|---|---|---|
| Enhancement | `audio/tasks/speech_enhancement/clearvoice.py` | Existing package, existing entry point, two existing backends (SpeechBrain, DriftSE). A third belongs beside them. |
| Separation | `audio/tasks/source_separation/clearvoice.py` | Existing package (unasdiff). Reinforced by PR #569: `enhance_audios` structurally cannot return N sources, so a 2-source checkpoint cannot live in enhancement even as a special case. |
| Super-resolution | **new** `audio/tasks/speech_super_resolution/` | D-2 |
| AV target-speaker extraction | **new** `audio/tasks/target_speaker_extraction/` | D-4 |
| SpeechScore | `audio/tasks/features_extraction/clearvoice_speechscore.py` | D-5 |
| Shared machinery | `utils/clearvoice.py` + `audio/tasks/clearvoice.py` | D-6 |

### D-2: super-resolution is its own task package, not a mode of enhancement

Rejected: folding `MossFormer2_SR_48K` into `enhance_audios`. It changes the output's sampling rate —
a 16 kHz input returns 48 kHz audio — so `enhance_audios` would sometimes preserve the input rate and
sometimes not, decided by which model id was passed. That is one function with two output contracts,
which is the shape the repository has been removing elsewhere (`_single_source`, PR #569).

Rejected: `preprocessing/`. Everything there is deterministic signal manipulation with no weights;
this is a two-stage generative model (MossFormer2 + a HiFi-GAN vocoder) with a 1.7 GB checkpoint set.

A new package under `audio/tasks/` is not a new top-level package: it is the house unit of
organisation, one per capability, and `health_acoustics/` (PR #568) is the current precedent for
adding one.

### D-3: the audio-visual extractor is reachable, and needs a file-backed video

Four things had to be true, and all four are:

1. **Video input.** `Video` carries `_file_path`, and upstream's reader accepts a single
   `.mp4`/`.avi`/`.mov`/`.webm` path (`dataloader/misc.py:read_and_config_file`). `.mkv` is not
   accepted, so the entry point validates the extension rather than letting ffmpeg fail obscurely.
   A frames-only `Video` cannot be used: the pipeline re-encodes the container to 25 fps and
   extracts the audio track with ffmpeg, so it needs the file, not the decoded frames.
2. **No senselab video extras.** The whole visual chain (opencv, scenedetect, torchvision,
   `python_speech_features`) is already in `clearvoice`'s own dependency set, so it comes with the
   isolated venv and senselab's `[video]` extra is not involved.
3. **ffmpeg.** Required, and already required by other senselab paths.
4. **The face detector's weights.** These are the problem, and D-8 is the answer.

So it lands in `audio/tasks/target_speaker_extraction/`, under **audio** rather than **video**,
because the capability's output is audio: the visual stream is a conditioning cue, and a caller
looking for "extract this speaker" will look where the other extraction and separation capabilities
are. `video/tasks/` holds capabilities whose *output* is visual (`pose_estimation`).

Not verified end to end: no talking-face recording with a known ground truth was available on this
host, so the extractor's numerical output is untested here. What is tested is dispatch, validation,
staging, the payload, and the timeout. This is stated in the module's `doc.md` as a limitation rather
than left for a user to discover.

### D-4: SpeechScore goes in `features_extraction/`, not `quality_control/`

`features_extraction/torchaudio_squim.py` is the near neighbour and the deciding one: it already
returns estimated PESQ/STOI/SI-SDR per audio as a dict of scalars, which is exactly SpeechScore's
shape with a wider metric set and a reference-optional interface.

`quality_control/` was rejected: it is a checks-and-verdicts framework (taxonomy, evaluations,
review) that *consumes* metrics to reach a pass/fail judgement. Putting a metric producer there would
invert that. A later change can add a QC check that consumes DNSMOS; that is a different change.

### D-5: two shared modules, neither of them a capability

`utils/clearvoice.py` holds the venv spec, the checkpoint-pinning rule, the device contract, the
timeout terms and the worker. It imports nothing from `senselab.audio`, so the layering stays
one-way, and it speaks in file paths.

`audio/tasks/clearvoice.py` holds the `Audio` ↔ path bridge, because that conversion is identical for
the three audio-only capabilities and the alternative is three copies of "and attach the commit that
produced it", one of which eventually lacks it. It deliberately does **not** decide the output count:
`run_clearvoice_over_audios` reports the sources it received, and each entry point enforces its own
contract on that (`single_source_per_input` for enhancement and super-resolution). This is PR #569's
lesson applied — derive the count from the output, never from the model's name.

## 3. The unpinnable loader

`clearvoice/networks.py:119`:

```python
snapshot_download(repo_id=f'alibabasglab/{model_name}', local_dir=checkpoint_dir)
```

No `revision`. CLAUDE.md requires a commit SHA and never a ref, and recording a SHA while loading
through a ref is worse than recording nothing.

### D-6: make the download path unreachable, rather than verify after it runs

Options considered:

| Option | Verdict |
|---|---|
| Let it download, then verify each blob's sha256 against the `lfs.oid` at a pinned commit | Rejected as the primary mechanism. It establishes what ran, but only *after* an unpinned network read has already chosen it; on a host where `main` has moved, the run has already loaded the new weights and the verification can only fail the run afterwards. It is also what the earlier investigation did — all four digests matched byte-for-byte — so it is proven but late. |
| Patch `snapshot_download` inside the worker to inject a revision | Rejected. It works by intercepting a call in a library we do not control, at a call site that may move. |
| Point `refs/<ref>` at the pinned commit and let the bare load resolve through it (`nvidia.py`, `nemo.py`, `diarizen.py` do this) | Rejected here. It works when the loader reads the HF cache, but this loader passes `local_dir=`, which bypasses the ref machinery entirely. |
| **Pre-stage at a resolved commit and hand the loader a local path** | **Chosen.** |

The chosen mechanism, in `stage_clearvoice_checkpoints`:

1. `resolve_revision(model_id, ref)` → a 40-hex commit, through the run-scoped manifest, so every
   task of one sweep binds to the same commit.
2. `hf_hub_download(model_id, "last_best_checkpoint", revision=sha)` — the commit's own manifest.
3. `hf_hub_download` for each weight file the manifest names, at the same SHA.
4. The worker symlinks that `snapshots/<sha>/` directory to `checkpoints/<MODEL>` in a staging root
   and chdirs there, because upstream's configs give `checkpoint_dir` as a relative path.
5. `load_model()` finds `last_best_checkpoint` present and never calls `download_model`.
6. `download_model` is replaced by a raiser anyway, so a staging bug is an error naming the model
   rather than a silent unpinned fetch.
7. The commit is returned and lands in every output's `metadata["clearvoice"]["commit"]`.

Verified (step 6): pointing `checkpoint_dir` at an empty directory produced
`RuntimeError: ... clearvoice tried to fetch FRCRN_SE_16K through its own unpinned snapshot_download`
rather than a download.

**File-by-file, not `snapshot_download`.** `MossFormer2_SR_48K` carries `do_03925000`, a 1.74 GB
optimizer state no inference run reads. The manifest names exactly the two files that are read.

**Guard-test classification.** `utils/clearvoice.py` belongs in
`revision_pinning_guard_test.LOADER_CANNOT_PIN_SUBPROCESS_FILES`: its upstream loader accepts no
revision, which is precisely what that list enumerates. It is *not* in
`REVISION_RESOLVED_SUBPROCESS_FILES`, because that list is coupled to a sweep over worker strings
containing an HF-load token, and this worker performs no HF load at all — the parent stages and the
worker reads a local path. It is in `hf_load_coverage_test.RAW_LOAD_EXCEPTIONS` for the parent's
`hf_hub_download`, on `driftse.py`'s precedent and for the same reason: `resolve_model` would
download 1.7 GB that is never read.

### D-7: the S3FD face detector, the one weight with no revision at all

`models/av_mossformer2_tse/faceDetector/s3fd/__init__.py` loads `sfd_face.pth` from inside its own
package directory and, when absent, runs:

```python
cmd = "gdown --id 1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt -O %s" % PATH_WEIGHT
```

The file is **not in the wheel** (confirmed against `clearvoice-0.1.2.dist-info/RECORD`), so on a pip
install this branch always runs: an unversioned Google Drive fetch, with no digest, writing into
site-packages.

Chosen: fetch it from a pinned *commit* of the GitHub tree and verify its sha256 against a recorded
digest, then symlink it into place so the gdown branch is unreachable. Measured at
`6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`: 89,844,381 bytes, sha256
`d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491`. It is stored under
`~/.cache/senselab/clearvoice/s3fd/<sha256>/`, so changing the constant is a cache miss rather than a
stale hit.

This is stronger than a revision, not a fallback from one: the bytes are checked, not the pointer to
them. A silently changed face detector would move every face track while the extraction reported
success.

## 4. Deviations from upstream's own inference path

Each is a correction to a defect, not a preference, and each is reproduced in the worker's comments.

### D-8: I/O does not go through pydub

`dataloader/dataloader.py:audioread` reads the input with `pydub.AudioSegment.from_file`, takes
`get_array_of_samples()` (integers), and rescales by:

```python
if max(data_array) > MAX_WAV_VALUE_16B:   # 32768
    audio_np = data_array / MAX_WAV_VALUE_32B
else:
    audio_np = data_array / MAX_WAV_VALUE_16B
```

A quiet 32-bit file whose peak sample is below 32768 is therefore divided by 2**15 instead of 2**31 —
amplified 65536x. The write side is `pydub` at the container's `sample_width`, defaulting to 16-bit.

senselab instead uses `Audio.save_to_file` on the host and the staged `portable_audio_io` in the
worker, so the subtype resolution and the out-of-range policy are the same ones every other senselab
write gets, and ffmpeg is not needed for the audio-only capabilities at all.

Consequence: because we bypass `DataReader`, the host owns resampling and downmixing
(`prepare_audios_for_clearvoice`), which is correct anyway — senselab's resampler is the one every
other task uses.

### D-9: `decode()`, not the tensor-to-tensor path

`ClearVoice.__call__` accepts a numpy array and routes to `decode_one_audio_batch`. That path is
broken for the super-resolution model with a single mono input: `decode_one_audio_mossformer2_sr_48k`
in `decode_batch.py` ends the short-audio branch with `outputs = generator_output.squeeze()`, which
for `b == 1` yields a 1-D array, and then indexes it as `outputs_pred[batch_idx, :]` → `IndexError`.
The file path's `decode.py` equivalent has no such bug.

So the worker drives `SpeechModel.decode()` — the same method upstream's own file API uses — with
`net.data` populated directly. This reuses upstream's segmented decoders unchanged and avoids the
batch path entirely.

### D-10: the device is chosen, not detected

`SpeechModel.__init__` selects MPS whenever `torch.backends.mps.is_available()`, else polls
`nvidia-smi` for the card with the most free memory. Both discard the caller's choice; the first
silently selects a backend none of these six checkpoints has been verified on (this host reports
`mps True`, so a naive integration would have run on MPS by default).

The worker replaces `SpeechModel.__init__` with one that reproduces its post-conditions field for
field (`args`, `model`, `name`, `data`, `print`, `device`) and takes the device from the payload.
`DeviceType.CUDA` is resolved on the host to `cuda:<torch.cuda.current_device()>` so a
`CUDA_VISIBLE_DEVICES` mask picks the allocated card. MPS is not offered: a caller passing it gets
`_select_device_and_dtype`'s error rather than an untested backend.

The field-by-field reconstruction is why the worker asserts the distribution version. Note
`clearvoice.__version__` reports `"0.1.0"` in 0.1.2, so the check goes through
`importlib.metadata.version`.

### D-11: the RMS normalisation is reproduced, including its asymmetry

`DataReader.extract_feature` applies `audio_norm` (to −25 dBFS, two-stage) for `FRCRN_SE_16K` and
`MossFormer2_SS_16K` only, and returns the inverse scalar. `SpeechModel.process` then applies that
inverse — but only on the single-output branch (`if not isinstance(output_audios, list)`). For the
separator, whose output *is* a list, the inverse is never applied, so its sources come back
RMS-matched to −25 dBFS rather than to the caller's input level.

Reproduced rather than corrected, so senselab's numbers agree with upstream's own tool for the same
checkpoint. The scalar is reported instead: every returned `Audio` carries
`metadata["clearvoice"]["input_norm_scalar"]` and `input_norm_applied_to_output`, so a caller can
restore the level and can see that they had to.

### D-12: the TSE visualisation step is replaced by its write

`utils/video_process.py:visualization` writes each face track's extracted audio *and* re-renders the
entire source video once per track with a bounding box drawn on every frame. senselab returns
`Audio`, so the render is pure cost — and the write is `sf.write(path, audio, 16000)` with no
subtype, i.e. PCM_16, quantising the one output the capability exists to produce. The worker replaces
the function with the write alone, through `portable_audio_io.write_audio` under
`out_of_range="normalize"` (upstream's own guard at that point divides by the peak when it exceeds 1,
so normalize reproduces its intent while reporting the gain).

Consequence: the `video_est_*.mp4` files upstream would leave behind are not produced. Documented in
the task's `doc.md`.

## 5. Timeouts

The defect being avoided (from `specs/20260818-071500-unasdiff-device-timeout-pcm16`): a hardcoded
ceiling that discards a legitimate long run.

**Measured, once**, on the shared development host: `FRCRN_SE_16K` on CPU decoded 21.48 s of 16 kHz
speech in 18.6 s inside `decode()` — 0.87 s per audio-second — after a 2.2 s checkpoint load.

That is the cheapest of the five audio models. The other four are a GAN generator, two 24-layer
MossFormer2 stacks, and a MossFormer2 plus a HiFi-GAN vocoder at 48 kHz. Rather than invent four more
per-model constants that would read as measured, one shared term sits an order of magnitude above the
single measurement:

```
default_audio_timeout_s(total_audio_s) = max(900, 2.0 * 8.0 * total_audio_s)
```

The floor absorbs what a per-second term cannot see: first import of torch in the venv, and up to
734 MB of checkpoint read cold. Every entry point takes `timeout_s`; exceeding the ceiling raises a
`RuntimeError` naming the ceiling, the work attempted, the fact that outputs are discarded, and the
two ways out (raise the ceiling, or select CUDA).

The audio-visual term is `max(1800, 2.0 * 60.0 * total_video_s)` and is **not** measured: the cost is
dominated by per-frame S3FD detection at 25 fps plus three ffmpeg passes, and no verified recording
was available here. Stated as coarse rather than presented as derived.

SpeechScore's is per (audio-second × metric), for the same reason its cost scales that way: 18
metrics over one file is 18 passes, three of them neural.

## 6. The model-specific parameter pathway

`utils/backend_parameters.py`.

**The measured defect.** `enhance_audios(audios, model, device)` has no channel for a
backend-specific parameter, so `enhance_audios_with_driftse`'s `variant` was never forwarded. DriftSE
ships two checkpoints; only the default was reachable through the public API, and it is the one that
suppresses a verified breath by 14.2 dB. The other was documented and unreachable.

**Why not `**kwargs` or a permissive dict.** A misspelled key would be dropped or ignored, the default
would run, and the run would report the parameter the caller believed they set. A permissive
dictionary is therefore worse than no pathway at all: the failure it produces is a confident wrong
result, which this repository already judges worse than no result (`RevisionResolutionError`).

**Chosen design.**

- Declared **from the backend callable's own signature** (`inspect.signature`), minus the
  dispatcher-owned names (`audios`, `model`, `device`, ...). A hand-maintained table of allowed keys
  is a second source of truth that goes stale the first time a backend gains a parameter.
- Validated against the **selected** backend, so a DriftSE key passed with a SpeechBrain model raises
  rather than being accepted before the dispatcher has decided where it is going.
- Unknown keys raise, with `difflib` near misses named — the failure mode is a typo, and a caller who
  wrote `variantt` should be told `variant`.
- A `**kwargs` backend declares nothing checkable and is treated as declaring nothing, rather than as
  "anything goes".
- The **effective** set — explicit values merged over declared defaults — is recorded on every
  returned object's `metadata["backend_parameters"]`, with `explicit` keeping the distinction between
  a deliberate choice and a default. Validation alone would not let a run say what produced it.

Only the caller's explicit keys are forwarded, so a backend default remains the backend's to change.

## 7. Measured behaviour worth a caller's attention

From a comparison against six human-verified events on a real recording (prior work on this branch;
recorded here because it is capability information absent from upstream's documentation, and it is
what a caller needs in order to choose a checkpoint):

- Every ClearVoice model **conserves energy**: output never exceeds input (−12.2 to −0.0 dB), in
  phase at zero lag, and clean speech is left essentially untouched. Every SepFormer checkpoint
  tested failed that check. Independently reproduced here: `FRCRN_SE_16K` on clean 16 kHz
  conversational speech returned −0.01 dB RMS with r = 1.0000 and peak 0.63.
- `MossFormer2_SE_48K` **destroys breaths** (−37 and −40 dB on two verified exhalations) while
  **keeping coughs** (−0.6, +0.2 dB).
- `FRCRN_SE_16K` keeps both (breath −2.0 / −5.5 dB, cough −1.0 / +0.1 dB).
- `MossFormerGAN_SE_16K` discards about 92% of input energy: breaths −51 dB, two of four coughs
  removed, speech kept. Useful when speech alone is wanted, destructive otherwise.
- `MossFormer2_SS_16K` is **not a class decomposer**. It assigns each cough burst to whichever of its
  two slots is free rather than isolating cough as a class.
- Peaks stayed ≤ 0.95, so no clipping exposure was observed for these models.

The operational consequence, and the reason this is in `doc.md` rather than only here: **which model
you choose determines which non-speech element survives**, so for any analysis that treats breaths or
coughs as signal, the enhancer is not a neutral preprocessing step.

## 8. What was and was not run

**Verified end to end on this host**, against the real `clearvoice==0.1.2` install and the real
checkpoint at commit `3766e6a64b0d8cb58f08d913d617bf129f11ed53`:

- `enhance_audios(..., model=HFModel("alibabasglab/FRCRN_SE_16K"), device=DeviceType.CPU,
  parameters={"timeout_s": 600})` over 21.48 s of 16 kHz speech. Output: 343,680 samples at 16 kHz
  (input's count preserved), FLOAT in and out, −0.01 dB RMS at r = 1.0000, peak 0.63, and
  `metadata["clearvoice"]` carrying the resolved commit plus `metadata["backend_parameters"]`
  carrying `{"timeout_s": 600.0}`.
- The blocked-downloader guard: pointing `checkpoint_dir` at an empty directory raised
  `RuntimeError: clearvoice reached SpeechModel.download_model for FRCRN_SE_16K …` rather than
  downloading.

**Not run.**

- The audio-visual extractor is not verified numerically (D-3): no talking-face recording with known
  ground truth was available, and fabricating one would have tested nothing.
- `MossFormer2_SS_16K`, `MossFormer2_SR_48K` and `AV_MossFormer2_TSE_16K` were not run end to end;
  their checkpoints total 1.6 GB and the host is shared. Dispatch, staging, validation, payload,
  device and ceiling are covered by tests that stub the worker.
- Per-model timeout costs for the four heavier checkpoints are extrapolated from FRCRN's measurement,
  not measured (§5).
- SpeechScore's NISQA/DNSMOS/DISTILL_MOS weights are committed in the upstream tree and arrive with
  the pinned sparse clone; they were not exercised here.

## 9. SpeechScore: pinning a component with no distribution, and two upstream traps

### D-13: a pinned sparse clone is the pin

SpeechScore has no pip distribution, and its metric weights are committed in the repository next to
the code (`scores/nisqa/weights/nisqa.tar`, four DNSMOS ONNX graphs, `scores/distill_mos/weights/
distill_mos_v7.pt`). So one pin covers both: a blobless, sparse clone of `/speechscore/` at
`6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61`, fetched with `--depth 1` under an exclusive lock into a
sibling temp dir and moved into place with `os.replace`, on `driftse.py`'s pattern. Sparse because the
rest of the studio carries checkpoints this never reads.

A separate venv from `clearvoice`'s: the dependency sets are disjoint (`museval`, `pysptk`, `pyworld`,
`gammatone`, `onnxruntime`, `xls_r_sqa`, `fastdtw`, `mir_eval`, plus pandas/matplotlib/tqdm, which
`NISQA_lib` imports at module scope). The requirements list is what the scores' own import chain
touches, not upstream's whole-studio `requirements.txt`. `gammatone` on PyPI (1.0.3) was checked to be
Jason Heeris' package and to contain the `fftweight` and `filters` modules SRMR imports.

### D-14: two things about running it that are not optional

1. **The working directory must be the `speechscore/` directory.** DNSMOS, NISQA and DISTILL_MOS
   address their weights relative to the cwd — `os.path.join('scores/dnsmos/DNSMOS', 'model_v8.onnx')`,
   `"scores/nisqa/weights/nisqa.tar"`, `os.path.join("scores/distill_mos/weights", "distill_mos_v7.pt")`.
2. **That directory, not its parent, must be on `sys.path`.** `speechscore/__init__.py` does
   `import absolute` and `import relative`, modules that do not exist in the tree, so importing
   `speechscore` *as a package* fails outright. With the directory itself on the path, `import
   speechscore` resolves to `speechscore.py` and the scores' `from scores.x import Y` imports resolve
   too — which is also how upstream's own `demo.py` works, since it sits inside that directory.

### D-15: the reference classification is senselab's, because upstream's is wrong

`ScoreBasis.intrusive` is never read by `basis.py` or anything else, and it disagrees with upstream's
own README and `demo.py`: `DNSMOS` and `SRMR` are marked `intrusive = True` and `MCD` is marked
`False`, all three the opposite of the truth. `SPEECHSCORE_METRICS` therefore carries senselab's own
`needs_reference`, taken from `demo.py`'s documented split (non-intrusive: NISQA, DNSMOS, DISTILL_MOS,
SRMR).

This matters because of what upstream does with a missing reference rather than what it says:
`ScoresList.audio_reader` zero-pads a single signal into the `audios` list, so a reference-requiring
metric called without one is computed against a copy of the test signal and returns a plausible
number. `resolve_speechscore_metrics` refuses instead.

### D-16: `window=None`, always

`basis.py`'s `scoring()` takes a `window` argument, and its windowed branch calls
`Framing(window * score_rate, window * score_rate, maxlen)` where `maxlen` is never assigned in that
scope — a `NameError` on any windowed call. Windowing is therefore not a capability this can expose,
and the worker passes `window=None` unconditionally rather than offering a parameter that cannot work.

## 10. One thing noticed, and since fixed under its own PR

`docs/compatibility-matrix.md` and `scripts/generate-compat-matrix.py` had diverged. The script wrote
that exact path from `generate_matrix_markdown()`, which emits a per-function table and a "Test Matrix"
section; the committed file contained neither, and was instead a hand-written document with sections
the generator does not produce ("Python Support", "Core Dependencies", "Isolated Backends", "System
Dependencies"). So running the generator would replace a maintained hand-written document with a
different one.

The two new venvs were therefore added to its "Isolated Backends" table by hand, which was the only
way that section could be maintained at the time, and `model_registry.md` — which *is* genuinely
generated — was regenerated with its script rather than edited.

**PR #572 resolves it**, and found a second half worth knowing about: `docs/` is gitignored as pdoc's
output directory, so the hand-written file was tracked only via a force-add — invisible to
`git status`, which is why an overwrite would have shown up as nothing at all. That document now lives
at `COMPATIBILITY.md` in the repo root, and the generated table stays in `docs/` uncommitted, produced
by both docs workflows. **The hand-edit described above therefore belongs in `COMPATIBILITY.md`**, and
moves there when the two branches meet.
