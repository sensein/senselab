# Quickstart: HuggingFace Model Cache & Version Consistency

How the shared mechanism is used once implemented. All commands via `uv` (Constitution II).

## Loading a model in a backend (the common case)

**Transformers loader (has a `revision` param):**
```python
from senselab.utils.hf_model_cache import load_hf_resilient
from transformers import pipeline

# resolves ref->sha once (coordinated, windowed), then loads sha-pinned + local-only
asr = load_hf_resilient(pipeline, "automatic-speech-recognition",
                        repo_id=model.path_or_uri, ref=model.revision or "main")
```

**Loader without a `revision` param (SpeechBrain / pyannote / NeMo) — use the local snapshot path:**
```python
from senselab.utils.hf_model_cache import resolve_model
from speechbrain.inference import EncoderClassifier

resolved = resolve_model(model.path_or_uri, model.revision or "main")
clf = EncoderClassifier.from_hparams(source=str(resolved.snapshot_path),
                                     savedir=speechbrain_savedir(model.path_or_uri, resolved.resolved_sha))
# pyannote: Pipeline.from_pretrained(str(resolved.snapshot_path))   # NOTE: do NOT also pass revision=
```

**Subprocess-venv worker (NeMo / Qwen / Granite / Sortformer / GLiNER):**
```python
env = hf_subprocess_env(model_name, revision or "main", base_env=_clean_subprocess_env())
subprocess.run([python, "-c", WORKER], env=env, ...)
```

## Reproducibility controls

**Freeze one long run (a multi-day batch) so no version shifts mid-run:**
```python
from senselab.utils.hf_model_cache import run_version_freeze
with run_version_freeze():
    analyze_audio(...)   # every model resolves its sha once, held for the whole block
```

**Freeze an entire environment (reproducible science) — pin versions, disable all re-checks:**
```bash
export SENSELAB_HF_FREEZE=1
uv run python -m senselab ...
```

**Tune the default freshness window (how often a coordinated re-check may happen):**
```bash
export SENSELAB_HF_FRESHNESS_DAYS=30   # default 7
```

## Verifying the guarantees (maps to Success Criteria)

```bash
cd src && uv run pytest tests/utils/hf_model_cache_test.py -q      # window/freeze/verify/coordination
cd src && uv run pytest tests/utils/dependencies_test.py -q        # retained primitives
cd src && uv run ruff check . && uv run mypy                       # quality gate
```

- **SC-001/SC-002**: launch ≥100 concurrent jobs loading one cached model → all succeed, zero Hub calls (verify on SLURM; unit test simulates concurrency with threads + a call counter).
- **SC-003**: many first-time jobs → one `snapshot_download`.
- **SC-004**: request a bogus ref → clear error naming repo+ref; never a silent substitution.
- **SC-008/SC-009**: run/system freeze → identical versions within a run / across runs.
- **SC-010**: `grep` for bespoke caching in backends → none remain; every HF-backed backend calls `resolve_model`/`load_hf_resilient`/`hf_subprocess_env`.

## Adding a new HF-backed backend (SC-006)

Implement only the model-specific load step: call `resolve_model(repo_id, ref)` (or `load_hf_resilient`), then either pass `revision=resolved_sha, local_files_only=True` or `resolved.snapshot_path`. No backend-specific caching, offline, or version code.

## Out of scope

TF-Hub (yamnet), s3prl, SPARC, Coqui backends have no HF Hub version concept and are excluded from HF version verification (they keep their existing `ensure_venv` download path).
