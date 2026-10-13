# HeAR as a senselab task: placement, TensorFlow, and the API the measurements dictate

Date: 2026-08-19. Supersedes PR #366.

## What was there before

PR #366 added `src/senselab/audio/tasks/speaker_embeddings/hear.py`: a 505-line research script
with an `argparse` CLI, directory scanning, `pandas` CSV checkpointing, `tqdm`, `notebook_login()`,
`transformers.AutoModel.from_pretrained("google/hear-pytorch")`, a comment reading "Try to import
senselab for future use", a `from utils import get_audio_files_from_directory` that cannot resolve
as a package module, and no tests. It is a script that happens to live in the package, not a task.

What it got right, and is reused here:

- **50% overlap over 2 s windows** as the default for a whole-file embedding. `hop_length=1.0`
  keeps it, and the reason is the same one PR #366 acted on: HeAR only takes 2 s, so a longer file
  has to be scanned.
- **Mean-pooling windows into one file-level vector.** Kept, as `HearEmbeddings.pooled()` — an
  explicit method rather than a hidden default, with the caveat that it is only comparable after
  centring and that averaging heterogeneous windows averages the events together.
- **Resample through senselab rather than librosa.** PR #366's `librosa` fallback and its manual
  peak normalisation are dropped: `resample_audios` is the house path, and HeAR is
  amplitude-invariant (gains ×0.1–×10 give cosine 1.0000), so normalising buys nothing and loses
  the recording's actual level.
- **`Audio.window_generator` as the windowing primitive** — considered and *rejected* here, for a
  reason PR #366 could not have known: it yields a **short final window** (`waveform[pos:end]`
  truncated at the signal end). For HeAR that is the one shape that must never reach the model, so
  windows are planned as explicit start offsets instead (`plan_scan_windows`).

Everything else — the CLI, the CSV, the directory walk, `notebook_login`, the `hear_pytorch`
backend — is gone.

## Decision 1: it lives in a new task package, `audio/tasks/health_acoustics/`

Rejected homes:

- **`speaker_embeddings/`** (PR #366's choice). HeAR describes *what the sound is*, not *who made
  it*; it was trained on non-semantic respiratory sounds and its own model card calls speaker
  identity out of scope. A caller browsing `speaker_embeddings/` for a verification model would
  find a health model, and vice versa.
- **`classification/` + `ssl_embeddings/`, split by capability.** The detector genuinely belongs
  with `yamnet.py` and the encoder genuinely resembles an SSL feature extractor. But the two
  capabilities are one repository, one gated licence, one pinned commit, one TensorFlow venv, and
  one hard 32000-sample window rule. Splitting them puts that rule — the single most load-bearing
  fact about this model — in two places to be maintained in agreement, and asks a reader of either
  half to discover the other.
- **`features_extraction/`** is the home of per-frame acoustic descriptors (openSMILE, Praat,
  torchaudio, SPARC), computed by DSP or small models with no gating and no isolation. A 300 M
  parameter gated foundation model would be the odd one out, and the event detector would have no
  home there at all.

Chosen: a task package named for the capability domain, `health_acoustics`, holding both entry
points, with **one dispatch alias** back into `classification.classify_audios` so that
`model="hear-event-detector"` works from the place a caller looks for classifiers. The alias is
seven lines and carries no implementation — the yamnet precedent is honoured for discoverability
without duplicating the window rule, the venv definition or the pin.

The package name also leaves room: senselab already has a `workflows/health_measurements`, and
future health-acoustic models (respiratory-rate estimators, cough classifiers) have somewhere
obvious to land that is not "classification, but medical".

## Decision 2: TensorFlow runs in an isolated subprocess venv; there is no usable transformers path

`google/hear` ships **TensorFlow SavedModels** (`library_name: tf-keras`): the repository root is
the encoder, and `event_detector/event_detector_{large,small}` plus a standalone
`spectrogram_frontend` sit beneath it. senselab is torch-based, so TF is not a core dependency.

A torch conversion exists — `google/hear-pytorch`, which PR #366 used — and it was checked rather
than assumed. Three findings, all verified against the Hub on 2026-08-19:

1. **It is separately gated and this account is not authorized for it.** `google/hear` resolves
   fine (SHA `9b2eb285…`); `google/hear-pytorch` answers
   `GatedRepoError: Access to model google/hear-pytorch is restricted and you are not in the
   authorized list`. So the claim that it produces the same embeddings as the SavedModel **cannot
   be verified here at all**, and an unverifiable numerical equivalence is not a foundation for a
   default backend. (Accepting its terms separately would settle this; until someone does, the
   position stands.)
2. **It takes a spectrogram, not a waveform.** Its Hub metadata is
   `pipeline_tag: image-feature-extraction`, `architectures: ViT`. The matching PCEN mel frontend
   is not in that repository — PR #366 imported `preprocess_audio` from a local clone of
   `github.com/google-health/hear` (`hear.python.data_processing.audio_utils`), i.e. an unpinned
   third source resolved through `sys.path.append`. The SavedModel has the frontend fused in:
   waveform in, embedding out, one artifact, one pin.
3. **It carries no event detector.** The detector exists only as a SavedModel. So TensorFlow is
   required for the second capability regardless of what the first one does, and routing the
   encoder through torch would mean two frameworks, two preprocessing paths and two provenance
   stories for one model family.

Therefore: `tf.saved_model.load` inside an isolated venv (`~/.cache/senselab/venvs/hear`,
`tensorflow>=2.16,<3`, Python 3.11), provisioned on first use by `ensure_venv` — the same pattern
and the same helpers (`_clean_subprocess_env`, `parse_subprocess_result`, `venv_python`) as
`classification/yamnet.py`, `speech_enhancement/driftse.py` and `source_separation/unasdiff.py`.
The venv declares no `torch`/`torchaudio`, so `ensure_venv` skips the CUDA probe and the PyTorch
wheel index entirely for it.

Two smaller choices inside that:

- **Its own venv, not yamnet's.** `ensure_venv` keys reuse on the exact requirements list, so two
  backends sharing a venv *name* with different lists would delete and rebuild each other's tree
  on alternate calls. Sharing would need the lists kept byte-identical forever, which couples two
  unrelated backends for one TF install.
- **The parent stages, the worker loads a path.** `resolve_model("google/hear", HEAR_REVISION,
  token=get_huggingface_token())` stages the pinned commit and returns
  `snapshots/<sha>/`; the payload carries that directory and nothing else identifying the model.
  The worker never imports `huggingface_hub`. This is `speech_to_text/crisperwhisper.py`'s shape,
  which is why `hear.py` is enumerated in `LOADER_CANNOT_PIN_SUBPROCESS_FILES` in
  `src/tests/utils/revision_pinning_guard_test.py` — `tf.saved_model.load` has no `revision`
  parameter, so the staged path *is* the pin.

Verified end-to-end on this host with the pinned commit: encoder returns `[4, 512]` over four
planned windows with `batch=4`; the large detector returns `[4, 8]` with the batch automatically
lowered to 1; an out-of-bounds window is refused by the worker with the intended `ValueError`.

## Decision 3: the API shape follows the measurements, not the model's tolerance

Signatures, probed rather than read off documentation:

```
encoder            serving_default   x: (None, 32000) float32   -> output_0: (None, 512)
event_detector_*   serving_default   audio_wav: (1, 32000)      -> mobilenetv3_{large,small}_model: (None, 8)
```

Note the detector's **pinned batch dimension** and the fact that the two detectors differ in their
output tensor's *name*. The worker reads the batch off `structured_input_signature` and takes the
sole output by position, so neither fact is a hardcoded constant that can silently rot.

### Enforced (a caller cannot get these wrong)

| Measurement | Enforcement |
|---|---|
| Detector rejects every length but 32000 samples (`InvalidArgumentError` at 0.5/1.0/1.5/3.0/4.0 s) | Window length is not a parameter anywhere in the API; only `hop_length` is |
| Detector batch dimension is pinned at 1 | Worker reads the static batch from the graph and lowers to 1; the detector path also declares `batch_size=1` in its payload |
| Padding destroys the representation (centred cosine 0.0–0.5 vs a ~0.9 class margin) | No code path pads. `plan_scan_windows` / `plan_centred_windows` only emit windows wholly inside the recording; the worker **re-checks** each window's bounds and raises. A test asserts window *values* equal the source slice, not merely that lengths are 32000 |
| Encoder silently accepts 160–64000 samples | Input shorter than 32000 samples (after resampling) raises `ValueError` naming the measurement and pointing at `extract_hear_embeddings_at_times` — the correct repair rather than padding |
| Usable length collapses below 2 s (+0.91 → +0.46 → +0.29) and 3 s is worse than 2 s | Same: 2 s is the only length the API can produce |
| Raw cosine is uninformative (0.977 vs 0.918) and centring fixes it (+0.653 vs −0.256, LOO-NN 0.846) | The only similarity helper, `centred_cosine_similarity`, centres by construction; no raw-cosine helper is offered, and it refuses to estimate a mean from a single vector |

### Documented, not enforced (judgement the caller must keep)

| Measurement | Where it is said |
|---|---|
| The detector is a **presence gate, not a locator**: 40 ms of cough crosses p > 0.5, so the response is a box-car of width (event + 2 s) and events closer than 2 s merge | `detect_health_acoustic_events`' docstring, `doc.md`, `classify_audios`' docstring, and the model registry's `recommended_for` |
| Scores are **independent probabilities**, not a distribution | Same places; `top_k` defaults to `None` (keep all eight) because dropping labels from a multi-label gate drops its negative evidence |
| Shift is benign (±50–200 ms → cosine 0.93–0.98) | The justification, in-code, for two policies that trade framing error against padding: the tail window snaps to end at the last sample, and an edge-adjacent centre slides inward. Both report where the window actually landed |
| Amplitude is irrelevant (gains ×0.1–×10 → 1.0000) | Why no normalisation is applied (and why PR #366's peak-normalise was dropped) |
| A hop wider than 2 s leaves audio no window sees | A `UserWarning`, not an error — sparse sampling of a long recording is legitimate, silently missing an event is not |

### Surface

```python
extract_hear_embeddings_from_audios(audios, model="hear", device=None, hop_length=1.0, batch_size=8) -> list[HearEmbeddings]
extract_hear_embeddings_at_times(audio, times, model="hear", device=None, batch_size=8) -> HearEmbeddings
detect_health_acoustic_events(audios, model="hear-event-detector", device=None, hop_length=0.25, top_k=None) -> list[list[dict]]
centred_cosine_similarity(embeddings, reference=None, mean=None) -> torch.Tensor
classify_audios(audios, model="hear-event-detector", hop_length=...)   # dispatch alias
```

`HearEmbeddings` carries `embeddings [n_windows, 512]`, `window_starts` (seconds, on the input's
timeline), `window_seconds`, `hop_seconds`, `model_id`, `revision`, and `pooled()`.

Detector output reuses the windowed shape `classification` already emits (`start`, `end`,
`label_scores` as descending single-key dicts, `win_length`, `hop_length`), so
`scene_results_to_segments` and the plotting helpers accept it unchanged — a test asserts that.

Model specs are **plain strings** (`"hear"`, `"hear-event-detector"`,
`"hear-event-detector-small"`), like `"yamnet"` and the S3PRL names, not `HFModel`. An `HFModel`
would validate and then mislead: nothing in transformers can load these SavedModels, so a spec
shaped like an HF spec invites exactly the `AutoModel.from_pretrained` call this backend replaces.

`device` accepts CPU/CUDA. CPU is honoured by setting `CUDA_VISIBLE_DEVICES=-1` in the worker's
environment, because TensorFlow has no per-call device argument here. MPS is not offered: it is not
a TensorFlow device without the separate `tensorflow-metal` plugin, which this venv does not
install.

## Gating and licence

`google/hear` is gated under the
[Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms);
acknowledging them while logged in grants access immediately. Access goes through
`get_huggingface_token()` → `resolve_model` — the same mechanism as `pyannote/brouhaha` and
DiariZen. A missing or unauthorized token surfaces `huggingface_hub`'s own `GatedRepoError`, which
already names the repository and the page to accept the terms on; wrapping it would only hide that.
PR #366's `notebook_login()` prompt is deliberately not carried over — it cannot work in a batch
job, which is where senselab runs.

## Tests

`src/tests/audio/tasks/health_acoustics/hear_test.py`, 43 passing + 3 skipped.

The 43 run with no TensorFlow, no venv and no Hub access: `ensure_venv`, `venv_python`,
`stage_hear_snapshot` and `subprocess.run` are monkeypatched by a stub that records the payload and
the exact audio each window was cut from, then fabricates arrays of the right shape. The properties
under test are the parent's — the pin, the refusals, the grid, the labels, the centring — so a stub
tests them honestly and a real model would only make them slow.

The audio fixture is a strictly increasing **ramp** rather than noise or a constant, so every
sample is unique and "this window is real audio from offset *k*" is checkable by value. That is
what makes the no-padding test a real test: a padded window is full-length too.

The 3 skips are the end-to-end ones (`requires_real_hear`): they run only when the `hear` venv is
already provisioned **and** the pinned commit is already in the HF cache, checked by file existence
rather than a Hub call. So they never provision a ~600 MB TF venv, never download 1.2 GB of gated
weights and never need a token — and they never fake a result either; if the model is not there,
they say so and skip. The third of them is the guard on the fixed window: it drives the worker with
`window_samples` set to 1 s and requires the graph to fail, so a future "make the window
configurable" change cannot look harmless in review.

### One guard interaction worth recording

`src/tests/utils/hf_load_coverage_test.py` discovers subprocess backends by sweeping every **string
constant** in a file — module docstring included — for HF loader names. An earlier draft of
`hear.py`'s docstring explained the pinning rule by naming `snapshot_download(revision="main")`, and
that mention alone made the sweep classify the file as a worker that loads from the Hub, which it is
not: the worker never imports `huggingface_hub`. The docstring now describes a "ref-addressed
download" instead of naming the function, and says explicitly that the worker imports no Hub client,
so the file's real classification (`LOADER_CANNOT_PIN_SUBPROCESS_FILES`) is the one a reader finds.
The alternative — adding it to `REVIEWED_SUBPROCESS`, whose companion assertion demands a
`hf_subprocess_env` call — would have meant asserting something false about the file and adding a
staging call the worker has no use for.

## Registry and docs

- `model_registry.yaml`: three entries under a new `health_acoustics` task (encoder, large
  detector, small detector), each declaring the gated licence and the pinned commit;
  `model_registry.md` regenerated with `scripts/generate_model_registry.py`.
- `docs/compatibility-matrix.md`: a `hear` row in the hand-maintained "Isolated Backends" table.
  Note for a future reader: `scripts/generate-compat-matrix.py` does **not** produce this file's
  current contents — the committed version has hand-written Python-support and dependency-version
  tables the generator knows nothing about, so running it would delete them. It was not run.
- `utils/compatibility.py`: two `isolated=True` entries, one per public function, both naming the
  `hear` venv. The flat schema has one entry per function and both capabilities share a venv, so
  the duplication is the schema's, not a modelling choice.
- `health_acoustics/doc.md` (pdoc-rendered, in the style of its siblings) carries the constraint
  table above, so the measurements sit next to the code they constrain.

## Not done, deliberately

- **No `audio_analysis` wiring.** HeAR is reachable only by a caller naming it, like DriftSE and
  unasdiff. It is a gated model under third-party terms and its uncertainty behaviour on this
  repository's own audio has not been characterised; a default pipeline is a separate decision.
- **No `spectrogram_frontend` exposure.** The repository ships it standalone for visualisation and
  custom datasets; nothing here needs it, since it is fused into all three models used.
- **No embedding cache.** `utils/tasks/cached_inference.py` is the workflow layer's mechanism, and
  this task is not in a workflow yet.
- **No `google/hear-pytorch` backend.** See Decision 2. If access is granted later, the first thing
  to do is measure cosine between its embeddings and the SavedModel's on the same windows — not to
  assume they match.
