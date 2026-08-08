---
name: Cross-release model behavior drift
description: HF model_id alone is insufficient — model behavior can change between releases; pin revisions and include them in cache keys + parquet provenance
type: project
---

The same HuggingFace `model_id` can produce different output across model releases:
different label conventions (`SPEAKER_00` vs `0` vs `spk_0`), different speaker
counts, different segment boundaries, different feature dimensions. This is a
reproducibility hazard, not a runtime within-clip one.

**Where this bites senselab today**:

- `scripts/analyze_audio.py` constructs `HFModel(path_or_uri=model_id)` without
  pinning a HuggingFace revision (verified 2026-05-09 at `pick_dispatch_model` and
  `--ast-model` wiring).
- The analyze_audio cache key includes `(audio_signature, task, model_id, params,
  wrapper_hash, senselab_version, schema_version)` but NOT the resolved model SHA.
  A silent upstream model update produces different output that gets stored under
  the SAME cache key on subsequent runs.
- Cross-run comparison parquets become incoherent after an upstream model update —
  `SPEAKER_00` from a run today and `SPEAKER_00` from a run six months ago may refer
  to different speakers (or different speaker counts entirely).
- Breaks the reproducibility guarantee in FR-014 of
  `specs/20260508-173136-compare-uncertainty/spec.md`.

**Why:** This is a real surfaced concern from the higgs E2E validation — comes up
naturally once you start running comparators that read labels and embeddings across
models / passes / runs. The within-clip "online diarizer relabels mid-stream" case is
a separate issue not driven by this; this one is purely about deployment / time.

**How to apply** (mitigations to evaluate when this matters):

- Pin HF revisions explicitly: `HFModel(path_or_uri=model_id, revision="<sha>")`.
  HuggingFace lets you reference a specific commit SHA or a release tag.
- Include the resolved revision in the analyze_audio cache key tuple, alongside
  audio_signature / task / params / wrapper_hash / senselab_version / schema_version.
- Record the resolved revision in each parquet's `comparator_provenance` metadata so
  reviewers diffing two runs can see exactly which model version each came from.
- Surface a warning when running with an unpinned model_id (no revision specified)
  so the user knows the run is not reproducible.

**Out of scope here:** dedicated PR. Don't bundle into unrelated feature work. Affects
every per-task pipeline (diar / ASR / scene / embeddings / alignment / PPG), not just
the comparator stage.
