# Audio-visual target speaker extraction

## Task overview

Given a video of people talking, extract each visible speaker's voice from the mixed audio, using their
lip motion as the conditioning cue. Unlike blind separation, the outputs are *identified*: each one
belongs to a face the pipeline tracked, so there is no permutation ambiguity to resolve afterwards.

It lives under `audio/tasks/` rather than `video/tasks/` because the capability's output is audio — the
visual stream is a cue, not a result. `video/tasks/` holds capabilities whose output is visual.

## Model

One backend, in an isolated subprocess venv:
[`alibabasglab/AV_MossFormer2_TSE_16K`](https://huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K)
from [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) (Apache-2.0).

```python
from senselab.audio.tasks.target_speaker_extraction import extract_target_speakers_from_videos

per_video = extract_target_speakers_from_videos(["meeting.mp4"])
speakers = per_video[0]        # one 16 kHz Audio per detected face track
```

`Video` objects work too, provided they are file-backed (`Video(filepath=...)`).

## What it requires, and what it returns

- **A video file, not decoded frames.** The pipeline re-encodes the container to 25 fps and extracts
  its audio track with ffmpeg, so it needs the file. A frames-only `Video` is refused with that reason.
- **`.mp4`, `.avi`, `.mov` or `.webm`.** These are what upstream's reader matches; anything else
  (including `.mkv`) is refused rather than silently read as a list of paths. Remux first.
- **ffmpeg on PATH.**
- **One output per face track**, in track order. A video where no face is tracked long enough returns
  an empty list — an outcome, not an error. The whole visual chain (scene detection, S3FD face
  detection, tracking, cropping) runs inside the venv, so senselab's `[video]` extra is not involved.

Upstream also renders an annotated video per track. senselab does not: it returns `Audio`, and that
render is the most expensive step in the pipeline. The extracted audio is written through senselab's
write policy instead of upstream's default PCM_16.

## Two weights, pinned two different ways

The checkpoint is pinned by resolved commit, like every other ClearVoice model. The **S3FD face
detector** cannot be: it ships in no wheel, and upstream fetches it from an unversioned Google Drive
file id with no digest, writing it into site-packages. senselab fetches it from a pinned commit of the
GitHub tree and verifies its sha256 (`d54a87c2…`, 86 MB) before use, refusing a mismatch — a silently
changed detector would move every face track while the extraction reported success.

## Not verified here

**The extractor's numerical output is untested in this repository.** No talking-face recording with
known ground truth was available, and fabricating one would not have tested anything. What is tested is
every host-side decision: container validation, the frames-only refusal, the payload, the device, the
ceiling, provenance, and the empty-track outcome. Treat the extraction quality as upstream's claim
until measured. The ceiling's per-second term is likewise unmeasured and deliberately generous, since
per-frame face detection at 25 fps dominates the cost; raise it with
`parameters={"timeout_s": ...}` if a long video needs it.

Design and decisions: `specs/20260819-clearvoice-integration/design.md` (D-3, D-7, D-12).
