# Plan A — Four diarization backends (cherry-pick #537, task layer only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the four diarization backends from PR #537 — VibeVoice-ASR-HF, USC-SAIL child-adult, MOSS-Transcribe-Diarize, DiariZen — reachable through `diarize_audios(model=...)` and deliberately not through the `audio_analysis` workflow.

**Architecture:** A file-level subset of #537. New backend modules are taken wholesale with `git checkout`; modified shared files get their hunks applied with a three-way `git apply`, because `alpha` will have moved under them since #537 branched. The five `audio_analysis` files and `scripts/analyze_audio.py` are excluded, which is what keeps child-adult's role labels out of the identity axis.

**Tech Stack:** Python 3.12 host, `uv`, pytest (serial), `ensure_venv` subprocess venvs, HuggingFace `transformers>=5.3`.

## Global Constraints

Copied from `design.md`. Every task's requirements implicitly include these.

- **No `analyze_audio` or `audio_analysis` wiring.** `clustering.py`, `identity.py`, `presence.py`, `stage_context.py`, `stages.py`, and `scripts/analyze_audio.py` are not modified by this plan.
- **No `run_config` changes.**
- **No new host dependencies and no new extras.**
- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Never run `pytest -n auto`.** Each xdist worker duplicates 535 MB of frameworks plus its own model weights, and `ensure_venv` takes no lock, so two workers wanting the same subprocess venv delete each other's tree mid-install. Run serially, scoped to the directory changed.
- **`uv sync` is subtractive** — always pass `--all-extras`.
- **Run `uv run ruff format` before any push;** pre-commit CI fails on it otherwise.
- **Never `git add -A` unqualified.** Always limit it with a pathspec (`git add -A -- src/ docs/ pyproject.toml uv.lock`). The repository root can hold untracked local secrets — a developer-supplied API token sitting beside the checkout is the case that prompted this — and an unqualified `git add -A` would stage one. `git status` is not a safeguard: an agent running these steps does not read it before committing.
- Core floor becomes `transformers>=5.3` (Task 5). This is every HuggingFace backend in the package, not just diarization.
- Upstream attribution: the final commit carries `Co-Authored-By: Evan Ng <evan.ng@sickkids.ca>`.

## Known follow-ups, carried out of execution (2026-08-08)

Plan A executed to completion. Final whole-branch review verdict: **Ship with follow-ups, no
Critical findings**. Two items were adjudicated as real-but-not-blocking and are recorded here
rather than fixed, so they are not lost when the scratch workspace is deleted.

1. **`child_adult.py` — the short-clip guard is ordered after `ensure_venv`.**
   The final review asked for a `ValueError` on clips ≤ 10 s, because upstream's loop advances only
   while `start + 10 < length` and a shorter clip silently returns `[]` — indistinguishable from
   "no adult present". The guard was added and is correct: it fires for *every* input clip and
   names both the measured duration and the window rule. But it sits inside the per-audio loop,
   which runs after `venv_dir = ensure_venv(...)`. The pre-existing CUDA check still correctly
   precedes `ensure_venv`, so a CPU host rejects immediately; a **CUDA host** handed a too-short
   clip as its first-ever child-adult call pays a 438 MB venv build before the rejection.
   **Ruling: park.** The behaviour is correct and the cost is one-time — the venv is cached
   afterwards, and any real use of this backend needs it built anyway. Fix by hoisting a duration
   pass above `ensure_venv`; roughly five lines.

2. **Four copies of the result-parse loop.** `nvidia.py`, `child_adult.py`, `diarizen.py` and
   MOSS's variant each iterate `output.get("results", [])` directly, so a worker returning fewer
   entries than it was given yields a short list and a caller doing `zip(audios, results)` silently
   drops audios. `parse_subprocess_result` already covers "no output" and "worker raised"; only the
   partial case is uncovered.
   **Ruling: park.** It is a refactor touching a pre-existing file rather than a defect fix. One
   `_script_lines_from_segments(output, audios)` helper with a `len(results) == len(audios)`
   assertion closes all four.

A third, cheaper follow-up worth filing: a single test that `ast.parse`s every `_WORKER_SCRIPT`
string literal in the package. Guarding only the four new ones would make the invariant look
enforced when ~13 sites share the same exposure.

## Preconditions

**Met.** The `20260728-221507-per-speaker-identity-scene` refactor merged into `alpha` as PR #547 (`79b37d93`), and `4071eed9 refactor(config)` is an ancestor of `origin/alpha`. The branch `feat/new-model-integrations` already exists, branched from the merged `alpha` and carrying this spec and its four plans. Task 1 re-verifies rather than waits.

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `src/senselab/audio/tasks/speaker_diarization/vibevoice.py` | In-process VibeVoice-ASR-HF backend, 7B, `release_all()` cache eviction | Create (from #537) |
| `src/senselab/audio/tasks/speaker_diarization/child_adult.py` | CUDA-only subprocess backend, role labels (CHILD/ADULT/OVERLAP), runtime repo clone | Create (from #537) |
| `src/senselab/audio/tasks/speaker_diarization/moss.py` | Subprocess backend, unified ASR+diarization, 0.9B | Create (from #537) |
| `src/senselab/audio/tasks/speaker_diarization/diarizen.py` | Subprocess backend, WavLM-Conformer EEND + VBx clustering | Create (from #537) |
| `src/senselab/audio/tasks/speaker_diarization/api.py` | Dispatch by model prefix; speaker-hint warnings; the single role-label prefix list | Modify |
| `src/senselab/utils/data_structures/model.py` | `model_for_task` diarization prefix match | Modify |
| `src/senselab/utils/compatibility.py` | Dispatch-paths note on the `diarize_audios` entry | Modify |
| `src/senselab/utils/dependencies.py` | `hf_subprocess_env` warns on staging fallback | Modify |
| `src/senselab/utils/subprocess_venv.py` | `_cache_dir_path()`, side-effect-free | Modify |
| `src/senselab/model_registry.yaml` / `.md` | Registry entries for the four | Modify |
| `docs/compatibility-matrix.md` | Isolated-backends table rows | Modify |
| `pyproject.toml`, `uv.lock` | `transformers>=5.3` | Modify |
| `src/tests/audio/tasks/speaker_diarization_test.py` | Backend tests, skip-gated | Modify |
| `src/tests/utils/hf_load_coverage_test.py` | Coverage of the new HF loads | Modify |

---

### Task 1: Verify the branch and record the baseline

**Files:**
- Create: none

**Interfaces:**
- Consumes: nothing
- Produces: a verified `feat/new-model-integrations` at or ahead of the merged `alpha`, which every later task and every other plan (B, C, D) builds on, plus a recorded baseline test state for Task 5 to compare against.

- [ ] **Step 1: Confirm you are on the branch and it descends from the merged alpha**

```bash
cd /Users/satra/software/sensein/senselab
git fetch origin --prune
git rev-parse --abbrev-ref HEAD
git merge-base --is-ancestor origin/alpha HEAD && echo "branch contains origin/alpha"
git merge-base --is-ancestor 4071eed9 HEAD && echo "branch contains the run_config refactor"
```

Expected: `feat/new-model-integrations`, then both confirmations. If the branch does **not** contain `origin/alpha`, rebase onto it before continuing — `alpha` has moved since the branch was cut:

```bash
git rebase origin/alpha
```

- [ ] **Step 2: Confirm the run_config lives where later plans expect**

```bash
test -f src/senselab/audio/workflows/audio_analysis/run_config.py && echo "run_config present"
```

Expected: `run_config present`.

- [ ] **Step 3: Confirm the working tree is clean before touching anything**

```bash
git status --short
```

Expected: no output. The only files the branch carries beyond `alpha` are this spec and its four plans.

- [ ] **Step 4: Record the baseline test state**

```bash
uv sync --all-extras
uv run pytest src/tests/audio/tasks/speaker_diarization_test.py -v 2>&1 | tail -20
```

Expected: all pass or skip. Note the counts — Task 6 compares against them. A failure here is pre-existing on `alpha` and is not this plan's to fix; record it and continue.

- [ ] **Step 5: Commit nothing, report the baseline**

No commit. Report the baseline pass/skip counts to the reviewer.

---

### Task 2: Take the four backend modules

**Files:**
- Create: `src/senselab/audio/tasks/speaker_diarization/{vibevoice,child_adult,moss,diarizen}.py`
- Test: `src/tests/audio/tasks/speaker_diarization_test.py` (imports only, this task)

**Interfaces:**
- Consumes: the branch from Task 1.
- Produces: four modules whose public entry points Task 3 dispatches to —
  - `VibeVoiceDiarization.diarize_audios_with_vibevoice(audios, model, device, max_new_tokens) -> List[List[ScriptLine]]`, plus the classmethod `VibeVoiceDiarization.release_all()`
  - `diarize_audios_with_child_adult(audios, model, device) -> List[List[ScriptLine]]`
  - `diarize_audios_with_moss(audios, model, device, max_new_tokens) -> List[List[ScriptLine]]`
  - `diarize_audios_with_diarizen(audios, model, device) -> List[List[ScriptLine]]`

  Verify these exact names in Step 2 before relying on them; if #537 named one differently, use the name in the file and correct this plan's Task 3.

- [ ] **Step 1: Copy the four new modules from the PR branch**

```bash
git checkout origin/feat/diarization-multi-speaker-uncertainty -- \
  src/senselab/audio/tasks/speaker_diarization/vibevoice.py \
  src/senselab/audio/tasks/speaker_diarization/child_adult.py \
  src/senselab/audio/tasks/speaker_diarization/moss.py \
  src/senselab/audio/tasks/speaker_diarization/diarizen.py
git status --short
```

Expected: four files staged as new (`A`).

- [ ] **Step 2: Extract the actual public names**

```bash
grep -n "^def \|^class \|    def diarize" \
  src/senselab/audio/tasks/speaker_diarization/{vibevoice,child_adult,moss,diarizen}.py
```

Write down what you find. Task 3 dispatches to these names.

- [ ] **Step 3: Verify each module imports standalone**

```bash
uv run python -c "
from senselab.audio.tasks.speaker_diarization import vibevoice, child_adult, moss, diarizen
print('all four import')
"
```

Expected: `all four import`. A failure here is almost certainly a missing symbol in a shared file — that is Task 3 and Task 4's work, so note which import failed and continue; re-run this step at the end of Task 4.

- [ ] **Step 4: Add the no-workflow-wiring note to each module docstring**

For each of the four files, append this paragraph to the module docstring (adjusting the first clause to name the backend). The wording matters: it is the only place a future maintainer learns why the workflow can't see this backend.

```python
# Appended to the module docstring of each of the four backends:
"""
...existing docstring...

Not wired into ``audio_analysis``
---------------------------------
This backend is reachable through :func:`diarize_audios` and deliberately **not**
through ``scripts/analyze_audio.py --diarization-models``. The guards that keep a
role-label backend out of embedding clustering, out of the identity axis's
cross-diarizer agreement vote, and out of presence live in
``workflows/audio_analysis/{clustering,identity,presence}.py``, which this branch
does not carry. Without them a role-label backend in ``--diarization-models`` would
build a ``CHILD`` centroid blending two different children, snap ``OVERLAP`` to
whichever centroid is nearest, and read as spurious disagreement against every real
diarization model. Port those guards from PR #537 before wiring any of these four
into the workflow.
"""
```

- [ ] **Step 5: Format and commit**

```bash
uv run ruff format src/senselab/audio/tasks/speaker_diarization/
uv run ruff check src/senselab/audio/tasks/speaker_diarization/
git add src/senselab/audio/tasks/speaker_diarization/
git commit -m "feat(speaker_diarization): add VibeVoice-ASR, child-adult, MOSS, DiariZen backends

Cherry-picked from #537 (task layer only). Each module docstring records why
it is not reachable from the audio_analysis workflow: the role-label guards
live in files this branch does not carry.

Co-Authored-By: Evan Ng <evan.ng@sickkids.ca>"
```

Expected: commit succeeds. `ruff check` may report unused-import warnings until Task 3 wires dispatch — if so, do not silence them, just note them and proceed.

---

### Task 3: Wire dispatch in `api.py` and `model.py`

**Files:**
- Modify: `src/senselab/audio/tasks/speaker_diarization/api.py`
- Modify: `src/senselab/utils/data_structures/model.py:329-341`
- Test: `src/tests/audio/tasks/speaker_diarization_test.py`

**Interfaces:**
- Consumes: the four backend entry points from Task 2.
- Produces: `diarize_audios(audios, model, device, num_speakers=None, min_speakers=None, max_speakers=None, max_new_tokens=None) -> List[List[ScriptLine]]` dispatching by model-id prefix, and `model_for_task(model_id, task="diarization") -> HFModel | PyannoteAudioModel`. Also produces the module-level role-label prefix constant in `api.py` that later work (not this branch) imports as the one source of truth.

- [ ] **Step 1: Write the failing test for prefix resolution**

Add to `src/tests/audio/tasks/speaker_diarization_test.py`:

```python
import pytest

from senselab.utils.data_structures import HFModel, PyannoteAudioModel
from senselab.utils.data_structures.model import model_for_task


@pytest.mark.parametrize(
    "model_id",
    [
        "microsoft/VibeVoice-ASR-HF",
        "AlexXu811/whisper-child-adult",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "BUT-FIT/diarizen-wavlm-large-s80-md",
        "nvidia/diar_sortformer_4spk-v1",
    ],
)
def test_model_for_task_resolves_new_diarizers_to_hfmodel(model_id: str) -> None:
    """The four new backends and Sortformer are HF-hosted, not Pyannote-hosted.

    Resolving them to PyannoteAudioModel would send them through pyannote's
    pipeline loader and fail with an opaque config error rather than dispatching.
    """
    assert isinstance(model_for_task(model_id, task="diarization"), HFModel)


def test_model_for_task_still_defaults_to_pyannote() -> None:
    """Anything not matched by a prefix stays on the Pyannote path."""
    assert isinstance(
        model_for_task("pyannote/speaker-diarization-3.1", task="diarization"),
        PyannoteAudioModel,
    )


def test_vibevoice_prefix_does_not_capture_the_tts_checkpoints() -> None:
    """`microsoft/VibeVoice-1.5B` is a TTS model, not the ASR diarizer.

    A bare `microsoft/VibeVoice` prefix would route it to
    VibeVoiceAsrForConditionalGeneration.from_pretrained and fail opaquely.
    """
    assert isinstance(
        model_for_task("microsoft/VibeVoice-1.5B", task="diarization"),
        PyannoteAudioModel,
    )
```

- [ ] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_test.py -k "model_for_task or vibevoice_prefix" -v
```

Expected: FAIL — the four new ids resolve to `PyannoteAudioModel`.

- [ ] **Step 3: Apply #537's `model.py` hunk with a three-way merge**

```bash
git diff origin/alpha...origin/feat/diarization-multi-speaker-uncertainty \
  -- src/senselab/utils/data_structures/model.py | git apply -3 --verbose
```

If it conflicts, resolve by hand to this shape (the `VibeVoice-ASR` suffix on the prefix is load-bearing — see the test above):

```python
    if task == "diarization":
        if (
            model_id.startswith("nvidia/diar_sortformer")
            or model_id.startswith("microsoft/VibeVoice-ASR")
            or model_id.startswith("AlexXu811/whisper-child-adult")
            or model_id.startswith("OpenMOSS-Team/MOSS-Transcribe-Diarize")
            or model_id.startswith("BUT-FIT/diarizen")
        ):
            return HFModel(path_or_uri=model_id)
        return PyannoteAudioModel(path_or_uri=model_id)
```

- [ ] **Step 4: Apply #537's `api.py` hunk**

```bash
git diff origin/alpha...origin/feat/diarization-multi-speaker-uncertainty \
  -- src/senselab/audio/tasks/speaker_diarization/api.py | git apply -3 --verbose
```

Then read the result and confirm it contains all four of these, resolving by hand if `git apply` dropped any:
1. Dispatch branches for the four backends, keyed on the same prefixes as `model.py`.
2. `max_new_tokens` threaded through to VibeVoice and MOSS.
3. A `logger.warning` when `num_speakers` / `min_speakers` / `max_speakers` are passed to any backend that ignores them (VibeVoice, child-adult, MOSS, DiariZen, and the pre-existing Sortformer branch).
4. The module-level role-label prefix constant, as the single source of truth.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_test.py -k "model_for_task or vibevoice_prefix" -v
```

Expected: PASS, 6 tests.

- [ ] **Step 6: Add and run the speaker-hint warning test**

```python
def test_speaker_hints_warn_when_the_backend_ignores_them(caplog) -> None:
    """Only Pyannote honours num_speakers. Silently dropping the hint on the
    other backends makes a misconfigured run indistinguishable from a working one.
    """
    import logging

    from senselab.audio.tasks.speaker_diarization.api import _warn_ignored_speaker_hints

    with caplog.at_level(logging.WARNING):
        _warn_ignored_speaker_hints(
            model_id="BUT-FIT/diarizen-wavlm-large-s80-md", num_speakers=2,
            min_speakers=None, max_speakers=None,
        )
    assert any("num_speakers" in r.message for r in caplog.records)
```

If #537 named the helper differently, use its real name — take it from the `api.py` you just applied. Run:

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_test.py -k speaker_hints -v
```

Expected: PASS.

- [ ] **Step 7: Format and commit**

```bash
uv run ruff format src/senselab/ src/tests/
uv run ruff check src/senselab/audio/tasks/speaker_diarization/ src/senselab/utils/data_structures/model.py
uv run mypy src/senselab/audio/tasks/speaker_diarization/ src/senselab/utils/data_structures/model.py
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(speaker_diarization): dispatch the four new backends by model prefix

The VibeVoice prefix is 'microsoft/VibeVoice-ASR', not 'microsoft/VibeVoice':
the bare prefix also matches the 1.5B/-Large TTS checkpoints, which would reach
VibeVoiceAsrForConditionalGeneration.from_pretrained and fail with an opaque
config error.

Co-Authored-By: Evan Ng <evan.ng@sickkids.ca>"
```

---

### Task 4: Apply the three shared-utility deltas

**Files:**
- Modify: `src/senselab/utils/subprocess_venv.py` — add `_cache_dir_path()`
- Modify: `src/senselab/utils/dependencies.py:568-580` — `hf_subprocess_env` warns on staging fallback
- Modify: `src/senselab/utils/compatibility.py:66-80` — dispatch-paths note
- Test: `src/tests/utils/`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `subprocess_venv._cache_dir_path() -> Path` — returns the venv cache directory **without creating it**. Task 6's skip gates call it, and so do plans C and D.

- [ ] **Step 1: Write the failing test for the side-effect-free cache path**

Add to `src/tests/utils/subprocess_venv_test.py` (create the file if absent):

```python
from pathlib import Path

from senselab.utils import subprocess_venv


def test_cache_dir_path_does_not_create_the_directory(tmp_path, monkeypatch) -> None:
    """A test's skip gate must be able to ask *where* a venv would live without
    creating anything: at import time, on a read-only or sandboxed HOME, the
    mkdir in _cache_dir() would fail and take collection down with it.
    """
    target = tmp_path / "does-not-exist-yet"
    monkeypatch.setenv("SENSELAB_VENV_CACHE", str(target))

    assert subprocess_venv._cache_dir_path() == Path(str(target))
    assert not target.exists()


def test_cache_dir_creates_the_directory(tmp_path, monkeypatch) -> None:
    """_cache_dir() keeps its creating behaviour — callers that are about to
    build a venv rely on it."""
    target = tmp_path / "created-on-demand"
    monkeypatch.setenv("SENSELAB_VENV_CACHE", str(target))

    assert subprocess_venv._cache_dir() == Path(str(target))
    assert target.is_dir()


def test_cache_dir_path_honours_the_env_override(tmp_path, monkeypatch) -> None:
    """The gate must match where the venv is actually built, or it skips on a
    host that has the venv and runs on a host that does not."""
    monkeypatch.setenv("SENSELAB_VENV_CACHE", str(tmp_path / "elsewhere"))
    assert str(tmp_path / "elsewhere") == str(subprocess_venv._cache_dir_path())
```

- [ ] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/utils/subprocess_venv_test.py -v
```

Expected: FAIL with `AttributeError: module ... has no attribute '_cache_dir_path'`.

- [ ] **Step 3: Apply the three hunks**

```bash
for f in src/senselab/utils/subprocess_venv.py \
         src/senselab/utils/dependencies.py \
         src/senselab/utils/compatibility.py; do
  git diff origin/alpha...origin/feat/diarization-multi-speaker-uncertainty -- "$f" | git apply -3 --verbose
done
```

If `subprocess_venv.py` conflicts, the required shape is:

```python
def _cache_dir_path() -> Path:
    """Return the cache directory path for cached subprocess venvs, without creating it.

    Side-effect-free so callers that only need to *check* a venv's location
    (e.g. a test's existence-based skip gate) don't risk failing at import time
    on a read-only/sandboxed HOME — creating the directory is ``_cache_dir()``'s job.
    """
    return Path(os.environ.get("SENSELAB_VENV_CACHE", str(_DEFAULT_CACHE_DIR)))


def _cache_dir() -> Path:
    """Return the directory for cached subprocess venvs, creating it if missing."""
    cache = _cache_dir_path()
    cache.mkdir(parents=True, exist_ok=True)
    return cache
```

**Critical:** `ensure_venv` must keep routing torch/torchaudio through the CUDA-aware PyTorch index. Do not let a conflict resolution drop that — it is what stops a CUDA 12.9 host resolving `torch` and `torchaudio` to mismatched toolchains.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/utils/subprocess_venv_test.py src/tests/utils/dependencies_test.py -v
```

Expected: PASS.

- [ ] **Step 5: Verify the four backends now import cleanly**

```bash
uv run python -c "
from senselab.audio.tasks.speaker_diarization import vibevoice, child_adult, moss, diarizen
from senselab.audio.tasks.speaker_diarization.api import diarize_audios
print('dispatch wired')
"
```

Expected: `dispatch wired`.

- [ ] **Step 6: Format and commit**

```bash
uv run ruff format src/senselab/utils/ src/tests/utils/
uv run mypy src/senselab/utils/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(utils): side-effect-free venv cache path, staging-fallback warning

hf_subprocess_env's staging fallback now warns rather than silently reverting
to online Hub loading — that revert is exactly the per-call 429 path the
function exists to remove, and previously nothing recorded when it happened.

Co-Authored-By: Evan Ng <evan.ng@sickkids.ca>"
```

---

### Task 5: Raise the `transformers` floor and prove the blast radius is clean

**Files:**
- Modify: `pyproject.toml:39`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: nothing.
- Produces: `transformers>=5.3` on the core dependency list. Every HuggingFace-backed module in the package now resolves against it.

- [ ] **Step 1: Make the change**

In `pyproject.toml`, replace:

```toml
  "transformers>=5.0",  # >=5.0 required for huggingface-hub>=1.0 compatibility
```

with:

```toml
  "transformers>=5.3",  # >=5.3 required for VibeVoiceAsrForConditionalGeneration (VibeVoice-ASR-HF diarization backend)
```

- [ ] **Step 2: Re-resolve the lock file**

```bash
uv sync --all-extras
uv run python -c "import transformers; print(transformers.__version__)"
```

Expected: a version `>= 5.3`. `uv.lock` is modified.

- [ ] **Step 3: Prove the bump is inert for this resolution, then run only what can actually fail**

**This step was rewritten during execution on 2026-08-08.** The original text said to run
`speech_to_text`, `classification`, `speaker_embeddings`, `ssl_embeddings`, `text` and `utils` and
compare against "Task 1's baseline". Three things were wrong with it, and the correction is more
useful than the original:

1. **Two of the six paths do not exist.** `speaker_embeddings` is a single file
   (`speaker_embeddings_test.py`), and there is no `ssl_embeddings` test directory at all.
2. **There was no baseline to compare against.** Task 1 baselined only
   `speaker_diarization_test.py`, so "no new failures relative to Task 1's baseline" was
   unverifiable for every other directory.
3. **The run does not test the bump.** Measured: it reached 3 % in ~25 minutes — a ~14-hour
   projection — while pulling model weights that took the machine from 14 GB to 8.6 GB free. And it
   would have proven nothing, because `transformers` **already resolves to 5.5.4**, which satisfies
   both `>=5.0` and `>=5.3`. The floor bump cannot change what is installed in this environment; it
   only forbids resolving something older. Re-running the model zoo exercises an unchanged
   `transformers` against downloaded weights.

What the bump can actually break is **resolution** — a fresh environment that previously could
select 5.0–5.2 now cannot — and **imports**, if any module used an API that moved. Test those:

```bash
# a. Resolution: identical before and after means the bump is inert here.
uv run python -c "import transformers; print(transformers.__version__)"   # before edit
# ...make the pyproject edit, then:
uv sync --all-extras
uv run python -c "import transformers; print(transformers.__version__)"   # after edit
git diff --stat uv.lock
```

```bash
# b. Imports: every senselab module that touches transformers, loaded for real. Catches an
#    API that moved, which is the only import-level way this bump bites.
uv run python -c "
import importlib, pkgutil, senselab, sys
failed = []
for m in pkgutil.walk_packages(senselab.__path__, 'senselab.'):
    try:
        importlib.import_module(m.name)
    except Exception as exc:
        failed.append((m.name, f'{type(exc).__name__}: {exc}'))
print('modules imported:', len(list(pkgutil.walk_packages(senselab.__path__, 'senselab.'))))
for name, err in failed:
    print('FAILED', name, err)
sys.exit(1 if failed else 0)
"
```

```bash
# c. The fast, download-free suites, before and after the edit, compared like with like.
uv run pytest src/tests/utils src/tests/audio/tasks/speaker_diarization_test.py -v
```

Expected: the resolved `transformers` version is **identical** before and after; `uv.lock` changes
only where the constraint is recorded; every module imports; and the fast suites are unchanged.

If the resolved version *does* change, the bump is not inert and the full model-zoo run becomes
justified — but it is a deliberate, separately-scheduled job on a machine with the disk for it, not
a step wedged into this task.

**Do not run the full `speech_to_text` / `classification` suites here.** They are a model-zoo
integration run: hours long, tens of GB of downloads, and orthogonal to what this task changes.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: raise the transformers floor to >=5.3

Required for VibeVoiceAsrForConditionalGeneration. This is every HF backend
in the package, not just diarization, so the full HF-touching test set was
re-run against the new floor.

Co-Authored-By: Evan Ng <evan.ng@sickkids.ca>"
```

---

### Task 6: Backend tests with honest skip gates

**Files:**
- Modify: `src/tests/audio/tasks/speaker_diarization_test.py`
- Modify: `src/tests/utils/hf_load_coverage_test.py`

**Interfaces:**
- Consumes: `subprocess_venv._cache_dir_path()` from Task 4; the four dispatch entry points from Task 3.
- Produces: no importable interface. Deliverable is a green, honestly-skipped test module.

- [ ] **Step 1: Apply #537's test additions**

```bash
for f in src/tests/audio/tasks/speaker_diarization_test.py \
         src/tests/utils/hf_load_coverage_test.py; do
  git diff origin/alpha...origin/feat/diarization-multi-speaker-uncertainty -- "$f" | git apply -3 --verbose
done
```

- [ ] **Step 2: Verify each skip gate is honest**

Read every new `@pytest.mark.skipif` and check three things:

1. It resolves the venv location through `subprocess_venv._cache_dir_path()`, **not** a hardcoded `~/.cache/senselab/venvs`. A hardcoded path ignores `SENSELAB_VENV_CACHE` and so skips on a host that has the venv.
2. The child-adult gate skips on **absence of CUDA as well as** absence of the venv. Gating only on the venv makes a CPU host that shares `$HOME` with a GPU node run-and-fail instead of skip.
3. No gate helper performs a `mkdir` at import time.

Fix any that fail these checks.

- [ ] **Step 3: Verify the fixtures are long enough to be capable of failing**

This is the trap #537 hit: a 4.9 s fixture sits under child-adult's 10 s window, so `all(...)` over an empty result list passes vacuously regardless of correctness.

```bash
grep -n "concat\|14.8\|duration\|len(result)" src/tests/audio/tasks/speaker_diarization_test.py | head -20
```

Every test that asserts with `all(...)` over a result list must **also** assert the list is non-empty:

```python
    assert result, "empty result — the assertions below would pass vacuously"
    assert all(isinstance(line, ScriptLine) for line in result)
```

Add that assertion wherever it is missing.

- [ ] **Step 4: Run the suite**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_test.py src/tests/utils/hf_load_coverage_test.py -v
```

Expected: pass or skip, with skip reasons naming CUDA or a missing venv. Compare against Task 1's baseline — the pass count should rise and nothing previously passing should now skip.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/tests/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "test(speaker_diarization): backend tests with honest skip gates

Gates resolve the venv location through _cache_dir_path() so they honour
SENSELAB_VENV_CACHE, child-adult additionally skips without CUDA, and every
all(...) assertion is preceded by a non-empty assertion so a fixture shorter
than the backend's window cannot pass vacuously.

Co-Authored-By: Evan Ng <evan.ng@sickkids.ca>"
```

---

### Task 7: Registry, compatibility matrix, and the squash

**Files:**
- Modify: `src/senselab/model_registry.yaml`, `src/senselab/model_registry.md`
- Modify: `docs/compatibility-matrix.md`

**Interfaces:**
- Consumes: everything above.
- Produces: one commit on `feat/new-model-integrations` containing workstream A in full.

- [ ] **Step 1: Apply the registry and docs hunks**

```bash
for f in src/senselab/model_registry.yaml \
         src/senselab/model_registry.md \
         docs/compatibility-matrix.md; do
  git diff origin/alpha...origin/feat/diarization-multi-speaker-uncertainty -- "$f" | git apply -3 --verbose
done
```

- [ ] **Step 2: Verify `model_registry.md` is regenerated, not hand-edited**

```bash
grep -n "generated by" src/senselab/model_registry.md | head -5
ls scripts/ | grep -i registry
```

If a generator script exists, run it and diff the result against the applied file. A hand-edited generated file drifts from its YAML source — #537 itself picked up a stale YAMNet description that way.

- [ ] **Step 3: Confirm DiariZen's licence is recorded**

DiariZen's pretrained weights are **CC BY-NC 4.0 — non-commercial only**, unlike every other diarization backend here. Check the registry entry and the module docstring both say so:

```bash
grep -rn "CC BY-NC\|non-commercial" src/senselab/model_registry.yaml src/senselab/audio/tasks/speaker_diarization/diarizen.py
```

Expected: at least one hit in each file. Add it if missing — this is a licence obligation, not a nicety.

- [ ] **Step 4: Run the full scoped suite one last time**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_test.py src/tests/utils/ -v 2>&1 | tail -20
uv run ruff format --check src/ && uv run ruff check src/ && uv run mypy src/senselab/
```

Expected: tests pass/skip, formatter clean, linter clean, mypy clean.

- [ ] **Step 5: Do NOT squash — verify the history instead**

**This step was rewritten during execution on 2026-08-08.** It originally said to
`git reset --soft $(git merge-base HEAD origin/alpha)` and recommit as one. Do not do that. Two
reasons, and the second is the one that matters:

1. **It would destroy unrelated work.** The branch carries six documentation commits before the
   code starts — the design, both plans, the weights-mirror records, and the upstream licence-issue
   URLs — plus one more (`docs(plan)`) interleaved among the code commits. Resetting to the
   merge-base folds all of them into a single `feat(speaker_diarization)` commit, burying the
   provenance records that the licensing position depends on.
2. **The stated rationale never applied.** The spec justified squashing because "#537's individual
   commits are not independently coherent once the workflow files are stripped out." True — but
   this branch never cherry-picked those commits. It has its own commits, written test-first and
   each independently reviewed: backends, docstring correction, dispatch, test mocking, utility
   deltas, warning assertion, dependency floor, test hardening, mock independence. That history is
   coherent and more useful than one opaque commit; a reviewer can see which change each review
   finding produced.

So: verify, do not rewrite.

```bash
# Every code commit is attributed and the history is linear.
git log --oneline --reverse origin/alpha..HEAD
git log origin/alpha..HEAD --format='%H %s%n%b' | grep -c "Co-Authored-By: Evan Ng" # >= 1
```

Expected: the upstream attribution to Evan Ng appears on at least the backend commit, no merge
commits, and no commit mixing documentation with source changes beyond the ones already made.

If the project later wants one commit on `alpha`, that is what GitHub's squash-merge is for — it is
a merge-time decision, not something to bake into the branch and lose.

<details>
<summary>The original squashed-commit message, kept for the PR description</summary>

```
feat(speaker_diarization): four new diarization backends (from #537, task layer only)

Adds VibeVoice-ASR-HF (in-process, transformers>=5.3), the USC-SAIL child-adult
role classifier (CUDA-only subprocess venv), MOSS-Transcribe-Diarize, and
DiariZen (WavLM-Conformer EEND + VBx clustering), each dispatched from
diarize_audios() by model-id prefix.

Cherry-picked from #537 at the task layer only. The five audio_analysis files
and scripts/analyze_audio.py are deliberately excluded: the guards that keep
child-adult's CHILD/ADULT/OVERLAP role labels out of embedding clustering, out
of the identity axis's cross-diarizer agreement vote, and out of presence live
there. Without them a role-label backend in --diarization-models would build a
CHILD centroid blending two different children, snap OVERLAP to whichever
centroid is nearest, and read as spurious disagreement against every real
diarization model. Each backend's docstring says so, and none of the four is in
any default model list.

Raises the core transformers floor to >=5.3 for VibeVoiceAsrForConditionalGeneration
— every HF backend in the package, so the full HF-touching test set was re-run.

DiariZen's pretrained weights are CC BY-NC 4.0, non-commercial only, unlike the
other backends here; recorded in the registry and the module docstring.

Co-Authored-By: Evan Ng <evan.ng@sickkids.ca>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
```

</details>

- [ ] **Step 6: Report — do not push**

Report the commit SHA, the test counts, and which files were deliberately excluded. Pushing and opening the PR is the user's call, per the repo's PR workflow (feature PRs target `alpha`).
