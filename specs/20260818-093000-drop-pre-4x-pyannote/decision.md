# Only 4.x-era pyannote checkpoints; the rest are dropped, not repaired

`pyannote-audio` 4.0.4 is what `pyproject.toml` requires (`pyannote-audio>=4.0`) and what is
installed. Only checkpoints published for 4.x are candidates. A pre-4.x checkpoint whose README
binds it to an earlier library version is **dropped rather than repaired**: fixing senselab's path
to it is work spent reaching a model we will not run.

**Kept**, both verified loading under 4.0.4:

- `pyannote/speaker-diarization-community-1` — the diarization backend, and the default VAD backend.
- `pyannote/brouhaha` — the frame-posterior extractor (VAD / SNR / C50 heads).

**Dropped:**

| model | why |
| --- | --- |
| `pyannote/voice-activity-detection` | config-only repo delegating to a pre-4.x weights repo; senselab's path to it cannot load under 4.x (below) |
| `pyannote/segmentation` (`Interspeech2021`) | pre-4.x weights; only reachable as the above pipeline's inner model |
| `pyannote/segmentation-3.0` | already unloaded in `5dd416f0`; nothing consumes its per-speaker channels |
| `pyannote/speaker-diarization-3.1` | superseded by community-1 |

## The dedicated VAD path could not have worked under 4.0.4

`pyannote/voice-activity-detection` is a **config-only repo** — three files, no weights:

```
.gitattributes  README.md  config.yaml
```

`config.yaml` names its weights by ref-suffixed id:

```yaml
pipeline:
  name: pyannote.audio.pipelines.VoiceActivityDetection
  params:
    segmentation: pyannote/segmentation@Interspeech2021
```

pyannote-audio 4.x rejects that syntax outright — `pyannote/audio/core/model.py:573`:

```python
if "@" in checkpoint:
    raise ValueError("Revisions must be passed with `revision` keyword argument.")
```

`pyannote_vad.py:79` pinned the **outer** pipeline to a resolved SHA and nothing rewrote the inner
reference, so the path raised before any inference. Reproduced on 2026-08-18 against 4.0.4:

```
detect_human_voice_activity_in_audios([audio],
    model=PyannoteAudioModel(path_or_uri="pyannote/voice-activity-detection", revision="main"))

pyannote/audio/pipelines/voice_activity_detection.py:109  get_model(segmentation, ...)
pyannote/audio/core/model.py:573
ValueError: Revisions must be passed with `revision` keyword argument.
```

The path was therefore never exercised against a real load. No test covered it either: the one
`pyannote/voice-activity-detection` test id in the tree was in the module docstring, not in
`src/tests/`.

### One piece of the reported evidence did not reproduce

The removal was proposed partly on `pyannote/segmentation` returning `GatedRepoError: 403` for this
account, with the note that `api.model_info()` succeeds on a gated repo and so cannot be used as an
access check. **The 403 did not reproduce on 2026-08-18.** For this account and token, both

```python
hf_hub_download("pyannote/segmentation", "pytorch_model.bin", revision="main")
hf_hub_download("pyannote/segmentation", "pytorch_model.bin", revision="Interspeech2021")
```

download successfully; `gated=auto` on `segmentation`, `segmentation-3.0`, `brouhaha` and
`community-1` alike, i.e. the gate is auto-approving and the account is through it. The observation
about `model_info()` remains true in general and is worth keeping in mind, but nothing here rests on
it. The decision rests on the `ValueError` above, which is independent of access: senselab's dedicated
VAD path cannot load the pipeline *even with the weights in hand*.

## What was removed

Code:

- `src/senselab/audio/tasks/voice_activity_detection/pyannote_vad.py` — deleted whole (the
  `PyannoteVAD` pipeline factory and `detect_voice_activity`).
- `voice_activity_detection/api.py` — the `_PYANNOTE_VAD_PREFIXES` tuple,
  `_is_pyannote_vad_model`, the dispatch branch that called `PyannoteVAD`, and the module and
  function docstring passages advertising a dedicated-VAD backend. Two backends remain (Pyannote
  diarization, NVIDIA Sortformer), both relabelling diarization segments as `"VOICE"`; the
  `NotImplementedError` message names them.
- `voice_activity_detection/frame_posteriors.py` — `SEGMENTATION_MODEL_ID`,
  `SEGMENTATION_REVISION`, `_get_inference` (the segmentation `Model` + `Inference` loader) and
  `_declared_classes`, none of which had a reader after `5dd416f0`; the three docstring paragraphs
  describing how `segmentation-3.0` loads, inside the docstring whose first paragraph said it is no
  longer loaded; and the `pyannote.audio` runtime import with its `PYANNOTEAUDIO_AVAILABLE` flag,
  which nothing read once the loader was gone. The module now loads no model at all: it is the
  `FramePosterior` container, the pooling, and the chunk stitching Brouhaha calls.
- `utils/compatibility.py` — the two `pyannote-audio` entries declared `>=3.0` while
  `pyproject.toml` requires `>=4.0` and the kept checkpoints need 4.x. Both now say `>=4.0`.

Registry and docs:

- `model_registry.yaml` — the `Pyannote VAD` entry. `model_registry.md` regenerated with
  `scripts/generate_model_registry.py` (never hand-edited). The generated file consequently has no
  **Voice Activity Detection** section at all, which is the honest rendering: the task has no models
  of its own, only diarization backends listed under Speaker Diarization.
- `voice_activity_detection/doc.md` — the Models paragraph said senselab integrates pyannote models
  "for VAD" and pointed at the pyannote org page. It now names the two diarization backends and
  states that there is no dedicated-VAD backend.
- `specs/20260728-221507-per-speaker-identity-scene/removal-ledger.md` — the note that
  `SEGMENTATION_MODEL_ID` was "deliberately left" is marked reversed, with a pointer here.
- `README.md:121` credited `HF_TOKEN` with enabling "the gated `pyannote/segmentation-3.0` overlap
  detector". Nothing has loaded that model since `5dd416f0`; the token's job in the adaptive loop is
  Brouhaha's frame posteriors, and overlap now comes from cross-diarizer occupancy with no model at
  all. Renamed to `pyannote/brouhaha`.
- `tutorials/audio/00_getting_started.ipynb` told readers to request access to
  `pyannote/segmentation-3.0` before running the VAD cell — which passes
  `pyannote/speaker-diarization-community-1`. That line is deleted; the community-1 line stays.

Tests:

- `src/tests/utils/hf_load_coverage_test.py` — `pyannote_vad.py` and `frame_posteriors.py` dropped
  from `REVIEWED_INPROCESS`. Neither file is a load site any more, and
  `test_allowlists_have_no_stale_entries` fails on an allowlisted non-load-site, which is the guard
  working as designed.
- No test exercised the removed VAD path, so none was deleted.
  `src/tests/audio/tasks/voice_activity_detection_test.py` already used
  `pyannote/speaker-diarization-community-1`, as does `scripts/profile_model_tiers.py`'s
  `setup_pyannote_vad`.

## Left alone deliberately

- `speaker_diarization/pyannote.py:140` describes `segmentation-3.0` as community-1's *internal*
  segmentation model. Still true: what is dropped is loading it *directly*, not the fact that a kept
  pipeline contains it.
- `pyannote/speaker-diarization-3.1` **was not in `model_registry.yaml`** — the entry the removal
  brief asked to delete does not exist, and the registry has carried only community-1 for
  diarization. The id survives in test fixtures (`model_test.py:200` routes it through
  `model_for_task`, `stages_test.py` and `speaker_identity_test.py` use it as a name in mocked
  stages) and in two prose examples (`workflows/audio_analysis/doc.md:294`,
  `speech_presence_link.py:464`). None of them loads it: they exercise id → provider-class routing
  and id-shaped naming. Left as-is to keep this change to model reachability rather than test
  rewriting; substituting community-1 throughout is a separate, mechanical edit.
- `utils/data_structures/model.py`'s `model_for_task` never referenced the VAD prefix — its
  diarization branch dispatches on the non-pyannote prefixes and falls through to
  `PyannoteAudioModel`. Nothing to clean.
- `docs/compatibility-matrix.md` is **not** regenerable from
  `scripts/generate-compat-matrix.py` any more: the script prints a per-function table, while the
  committed file is a hand-maintained Python/dependency-version document that already records
  `pyannote-audio >=4.0`. Running the generator would overwrite it, so it was left untouched. That
  divergence is a separate defect, unfiled here.

## One mention that was not descriptive after all

`scene_quality/brouhaha.py` was expected to hold a still-true description. It said `stitch_frames`
gives "native ~17 ms resolution (shared with the segmentation-3.0 extractor)". There has been no
segmentation-3.0 extractor since `5dd416f0`, so the sentence pointed at nothing — and it was also
garbled mid-clause ("Frames arrive as one continuous timelineand stitched back into"). Rewritten to
say what the worker does: chunk, then stitch.
