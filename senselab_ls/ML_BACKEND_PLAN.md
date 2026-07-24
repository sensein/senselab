# Plan — senselab Label Studio ML backends (multi-model, per aspect)

## Context

`scripts/analyze_audio.py` already runs the full senselab audio stack (diarization, AST/YAMNet
scene classification, ASR/transcription, alignment, uncertainty) and emits **Label Studio
predictions** as parallel `from_name` tracks plus a dynamic labeling config. It is a batch CLI:
one process runs *every* analyzer and writes JSON tasks.

We want the same capability exposed through the **HumanSignal Label Studio ML SDK**
(`label-studio-ml`) so Label Studio can request predictions on demand — hosted on the
on-demand EC2 box (see `AWS_EC2_SETUP.md`). Per the user's steer, we **split the monolith into
one model call per aspect**: a diarization model, an ASR model, a scene-classification model,
each an independent `LabelStudioMLBase` backend that LS connects separately. This lets an
annotator enable/version/run each aspect on its own, and keeps each `predict()` small and fast.

## Why split into multiple backends (vs one mega-predict)

- Label Studio supports **multiple ML backends per project**; each returns its own
  `results`/track. Diarization latency shouldn't block ASR, and vice-versa.
- Each aspect maps to a distinct control tag already produced by analyze_audio:
  diarization → `<Labels>`, scene → windowed `<Labels>`, ASR → per-region `<TextArea>`.
- Independent `model_version` per backend → clean prediction provenance in LS.
- Matches the on-demand model: start only the backend(s) you need for a session.

## Target layout

```
senselab_ls/                   # package dir — named to NOT shadow the SDK's `label_studio_ml` import
  ML_BACKEND_PLAN.md           # this plan
  AWS_EC2_SETUP.md             # EC2 provisioning runbook
  common/
    audio_plus.py              # AudioPlus (+ recording_id) + build_audio_plus(ref, audio_loader, metadata_provider)
    b2ai_metadata.py           # B2AIMetadataProvider: concrete b2ai-voice v3.x metadata join
    audio_io.py                # load_audio(ref): s3:// via boto3 / http via LS / local -> senselab Audio
    ls_regions.py              # region builders (copied from analyze_audio; no senselab import)
    engine.py                  # thin senselab callers per aspect (diarize now; asr/scene next) + prepare_audio
  backends/
    diarization/model.py       # DiarizationBackend(LabelStudioMLBase)   :9090
    asr/model.py               # ASRBackend(LabelStudioMLBase)           :9091 (later)
    scene/model.py             # SceneBackend(LabelStudioMLBase)         :9092 (later)
  tests/
    ls_regions_test.py         # region-shape unit tests (write FIRST — TDD)
    audio_plus_test.py         # Audio+ construction from a ref (fake loader + mocked metadata)
    b2ai_metadata_test.py      # B2AIMetadataProvider against a synthetic b2ai-like fixture
    engine_smoke_test.py       # end-to-end on a synthetic clip, GPU-marked
  deploy/                      # systemd unit, env template, launcher, LS config, bring-up README
  requirements.txt            # backend-only extras (label-studio-ml/sdk/boto3)

**Dependencies (uv, two layers).** senselab itself is uv-locked (`uv pip install -e ".[audio]"`).
The backend extras are installed *on top* via `uv pip install -r senselab_ls/requirements.txt`,
kept OUT of senselab's `pyproject`/`uv.lock` (they break its locked resolution). Two gotchas
**verified against the installed SDK**, baked into `requirements.txt`:
- the API `model.py` uses (`ModelResponse`, `self.label_interface`) is the **2.x SDK, git-only** —
  PyPI's `label-studio-ml` is stuck at 1.0.9 and lacks it. Install
  `label-studio-ml @ git+https://github.com/HumanSignal/label-studio-ml-backend.git` (this pulls
  `label-studio-sdk` from git automatically).
- `redis` and `rq` are imported at module load but **not declared** by the SDK — list them
  explicitly or the backend won't import.

(If we later want full uv isolation, promote `senselab_ls/` to its own uv project with its own
`pyproject.toml` + lock, depending on senselab.)
```

## What is actually shared (the two scripts run in opposite directions)

`analyze_audio.py` **authors tasks** (creates `{data.audio, predictions[], config}` offline for
import); the ML backend **serves requests** (LS sends existing tasks, backend returns
predictions). So the reuse is a *middle layer*, not the whole script:

**Shared core → refactor into `common/` (single source of truth):**

| Helper | Location | Why it is direction-agnostic |
|---|---|---|
| `_ls_label_region` | `analyze_audio.py:1311` | emits one `labels` region dict — identical schema whether it lands in an export task's `predictions[].result` or a `ModelResponse.result` |
| `_ls_textarea_region` | `analyze_audio.py:1334` | same, for `textarea` regions |
| `_diarization_to_ls` | `analyze_audio.py:1363` | `ScriptLine[]` → speaker regions |
| `_classification_to_ls` | `analyze_audio.py:1387` | windowed scene labels |
| `_asr_to_ls` | `analyze_audio.py:1513` | transcript → textarea (+ 3-case timestamp branch) |
| `_safe`, `_seg_attr` | `analyze_audio.py:1632/1352` | id/label sanitization used by the converters |
| `prepare_audio` | `analyze_audio.py:896` | load → mono → 16 kHz |
| `pick_dispatch_model`, `pick_device` | `analyze_audio.py:579/598` | model/device resolution |

**NOT shared → stays in `analyze_audio.py` (export/authoring only):**

- `build_labelstudio_task` (`:1543`) — wraps regions in a `data`+`predictions` **task
  envelope**. The backend never builds tasks; LS supplies them and the backend only returns the
  inner `result` list inside a `ModelResponse`.
- `build_labelstudio_config` (`:1637`) / `_collect_classification_labels` (`:1697`) — generate
  labeling XML. The backend **reads** the project config via `self.label_interface`; it doesn't
  generate it. (Keep this available as a one-time *project-setup* helper, not part of the serve
  loop.)
- argparse, multi-pass orchestration, caching, uncertainty-track attachment — batch-only.

So the shared surface is the ~200 lines of region converters + audio prep + engine dispatch —
genuinely reused. The envelopes differ and correctly stay separate.

> `to_name` is hardcoded to `"audio"` in the converters — fine, since each backend's config
> uses `<Audio name="audio">`. Parameterize only if a project renames the object tag.

## Audio+ — the enriched input every backend builds first

A backend does **not** run models on the bare waveform. The **first step in every `predict`** is
to derive an **Audio+** object from the audio the SDK sends in:

```python
audio_plus = build_audio_plus(task["data"][value_key])   # common/audio_plus.py
```

`build_audio_plus(incoming_ref, *, audio_loader, metadata_provider)` takes the incoming audio
reference (`s3://…` key / path) and **generates-or-grabs** the enriched object:
- the **audio bytes** (via `common/audio_io.py`; see the S3 path below),
- **recording_id** — the dataset's stable id for the recording; the key a prediction is written
  back under when saved into an annotation,
- **task** name + content/prompt,
- **speaker** phenotype: **GSD (gold standard diagnosis)** + age,
- optionally the speaker's **related audios**, so a speaker-profile aspect can build a profile
  from them.

The metadata join is a pluggable `MetadataProvider`. The concrete implementation,
**`common/b2ai_metadata.py::B2AIMetadataProvider(dataset_root)`**, reads the standardized
b2ai-voice v3.x BIDS layout (specific, not a general BIDS parser):
- `recording_id`, `task_name`, `prompts` ← the recording's `_recording-metadata.json` sidecar,
- `age` ← `phenotype/demographics/demographics.tsv` (keyed by bare-UUID `participant_id`),
- **GSD** ← each `phenotype/diagnosis/<condition>.tsv` column ending in `gold_standard_diagnosis`
  (the condition-file stems where the participant is affirmative; `control` has none). The value
  vocab is documented in the diagnosis `.json` data dictionaries: `yes`/`no`/`notCertain` for
  most, plus `copd`/`asthma`/`bothCopdAsthma`/`neitherCopdAsthma`/`notCertain` for
  `copd_and_asthma`. Affirmative = present and not `no`/`notCertain`/`neitherCopdAsthma`,
- related audios ← the participant's other `ses-*/audio/*.wav`.

`dataset_root` may be a **local path or an `s3://bucket/prefix`** (read via boto3), since the
dataset lives on S3. Wired into the backend opt-in via the `B2AI_DATASET_ROOT` env var; unset →
bytes-only Audio+ (`NullMetadataProvider`). Analyzers then read what they need off Audio+ (diarization/ASR
need only the waveform; a profile aspect uses related audios; task/phenotype can condition or
annotate output).

> Audio+ is derived **on demand from the SDK input**, not pre-materialized and threaded through
> the pipeline. Its constructor is the shared entry point for `common/`. (Design note:
> `audio-plus-object`.)

## Backend contract (all three follow this shape)

```python
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

class DiarizationBackend(LabelStudioMLBase):
    def setup(self):
        self.set("model_version", "senselab-pyannote-community-1")
        # optional: warm the pipeline once (senselab caches per (uri,revision,device))

    def predict(self, tasks, context=None, **kwargs) -> ModelResponse:
        from_name, to_name, value_key = self.label_interface.get_first_tag_occurence(
            "Labels", "Audio")                       # ASR backend -> ("TextArea","Audio")
        preds = []
        for task in tasks:
            audio_plus = build_audio_plus(task["data"][value_key])  # bytes + task + GSD/age + related audios
            audio = prepare_audio(audio_plus.audio)                 # mono / 16 kHz
            regions = run_aspect(audio, from_name)                  # engine.diarize / asr / classify + *_to_ls
            preds.append({"result": regions,
                          "model_version": self.get("model_version"),
                          "score": 1.0})
        return ModelResponse(predictions=preds)
```

- **Diarization** — `engine.diarize(audio)` → `diarize_audios([audio], model=pyannote|sortformer, device=CUDA)` → `_diarization_to_ls`.
- **ASR** — `engine.transcribe(audio)` → `transcribe_audios(...)` (+ alignment when no native
  timestamps, mirroring the 3-case branch at `analyze_audio.py:1598`) → `_asr_to_ls`.
- **Scene** — `engine.classify(audio, win, hop)` → `classify_audios(...)` (AST/YAMNet) →
  `_classification_to_ls`.

### Audio fetch — prefer S3 direct read

Label Studio in this project pulls its audio from an **S3 bucket** (LS cloud-storage sync), so
`task["data"][value_key]` is typically an `s3://bucket/key` (or presigned `https`) URI.
`common/audio_io.py` (called by `build_audio_plus`) resolves the audio bytes in this order:

1. **`s3://…` → read directly via boto3** using the EC2 instance role (S3 read granted in
   `AWS_EC2_SETUP.md` step 4). Fastest; no round-trip through Label Studio, and works even when
   the LS API is slow/unavailable. Requires the backend's IAM role to have read on that bucket.
2. **`http(s)://…` (LS-hosted upload or presigned)** → `LabelStudioMLBase.get_local_path(url,
   task_id=…)`, which authenticates with `LABEL_STUDIO_URL` + `LABEL_STUDIO_API_KEY`.
3. **local path** → open directly (dev/testing with a synthetic clip).

Download to a temp file (or stream), then hand to `prepare_audio`. Because the S3 bucket is the
same source LS syncs from, the backend and LS always see identical bytes.

## Labeling config

Diarization/scene need their label set declared. Two options:
1. **Static per-project config** (recommended first): one control per connected backend, e.g.
   ```xml
   <View>
     <Audio name="audio" value="$audio"/>
     <Labels name="diarization" toName="audio">
       <Label value="SPEAKER_00"/>...<Label value="SPEAKER_UNKNOWN"/>
     </Labels>
     <Labels name="scene" toName="audio"><!-- AudioSet labels --></Labels>
     <TextArea name="asr" toName="audio" perRegion="true" editable="true"/>
   </View>
   ```
   Each backend's `from_name` must equal its control name (`diarization`/`scene`/`asr`).
2. **Dynamic labels** — reuse `build_labelstudio_config` (`analyze_audio.py:1637`) to generate
   the config from a dry run; import it into the project once.

> pyannote emits `SPEAKER_00…`; predeclare enough speaker labels or a project uses dynamic
> `value="$labels"` labeling.

## Running on EC2 (multiple backends)

Each backend is its own process/port (matches the SG ports in `AWS_EC2_SETUP.md`):

```bash
source /opt/lsml-venv/bin/activate
export HF_TOKEN=... LABEL_STUDIO_URL=https://<ls-host> LABEL_STUDIO_API_KEY=...
export PYTHONPATH=/opt/senselab:$PYTHONPATH   # so `senselab_ls` package is importable
label-studio-ml start senselab_ls/backends/diarization -p 9090 --host 0.0.0.0 &
label-studio-ml start senselab_ls/backends/asr         -p 9091 --host 0.0.0.0 &
label-studio-ml start senselab_ls/backends/scene       -p 9092 --host 0.0.0.0 &
```

Wrap in systemd units (or a single `docker-compose.yml` with three services) so start/stop of
the EC2 box brings them up cleanly. In Label Studio: project → Settings → Model → add each URL;
enable "Retrieve predictions when loading a task automatically."

## Build order (TDD — tests first, per project convention)

1. **Refactor** the reusable helpers out of `analyze_audio.py` into `common/ls_regions.py`;
   keep `analyze_audio.py` importing them (no behavior change — existing
   `global_summary`/tests must still pass: `cd src && uv run pytest`).
2. **`tests/test_ls_regions.py`** — assert region dict shapes (`type`, `from_name`,
   `value.start/end/labels|text`) for synthetic `ScriptLine`s. Write before wiring backends.
3. **`tests/test_audio_plus.py` + `common/audio_plus.py`** — `build_audio_plus(ref)` from a
   task-like reference with **mocked b2aiprep records**; assert Audio+ carries bytes + task +
   GSD/age + related audios. Tests first.
4. **`common/engine.py`** — the three thin senselab callers + `prepare_audio` reuse.
5. **`backends/diarization/model.py`** first (simplest, our near-term goal): `build_audio_plus`
   → `engine.diarize`; then ASR, scene.
6. **`tests/test_engine_smoke.py`** — GPU-marked, runs diarization on a **synthetic clip** and
   checks non-empty speaker regions (real validation runs against b2aiprep output, which is
   gated — see data-access rule).
7. Deploy per `AWS_EC2_SETUP.md`; run backends behind systemd.

## Verification

1. `cd src && uv run pytest && uv run ruff check .` — refactor keeps existing suite green.
2. **Local backend**: `label-studio-ml start backends/diarization -p 9090`; then
   `curl localhost:9090/health` → 200, and POST a task pointing at a local/synthetic wav to
   `/predict` → speaker `labels` regions returned. The scaffold's `test_api.py` covers this.
3. **EC2**: start instance, start backends, register URLs in a test LS project, import an audio
   task, open it → diarization/scene/ASR tracks appear pre-drawn; stop instance when done.

## Open decisions

- **Aspects for v1**: start with **diarization only** (near-term goal), add ASR + scene next?
  Or scaffold all three now.
- **Multi-backend vs single backend**: recommended multi-backend (one port per aspect). A
  single backend that returns all tracks in one `predict` is possible but couples latencies.
- **Engine defaults**: pyannote `speaker-diarization-community-1`; ASR model (Whisper variant);
  AST + YAMNet for scene — confirm which to enable by default.
- **Audio+ record lookup**: *resolved* for b2ai-voice v3.1 adult — `B2AIMetadataProvider`
  (`common/b2ai_metadata.py`) parses `sub-/ses-/task-` from the ref and reads the sidecar +
  `phenotype/` TSVs at `B2AI_DATASET_ROOT` (local path **or `s3://`** via boto3). GSD
  affirmative-detection is driven by the documented value vocab; raw values are preserved in
  `SpeakerInfo.metadata["gsd_details"]`. Remaining: confirm the exact S3 bucket/prefix and IAM
  read scope for the EC2 role.
