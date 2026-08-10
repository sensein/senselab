# Speaker Ceiling Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace four unmeasured `max_speakers=None` values with numbers derived from a synthetic sweep against known ground truth, and test the two declared ceilings against reality.

**Architecture:** A `scripts/` measurement job in three separable pieces — corpus generation inside the existing `nemo-diarization` venv, backend evaluation, and profile derivation. The derivation is pure and unit-testable without a GPU; only generation and evaluation need one.

**Tech Stack:** Python 3.12 host; `nemo_toolkit[asr]` in the existing `nemo-diarization` subprocess venv; MIT ORCD (Engaging) H100 for the run.

## Global Constraints

- **No new host dependency and no new venv.** NeMo's simulator runs in the `nemo-diarization` venv this repo already builds (`nvidia.py:_NEMO_REQUIREMENTS`).
- **`max_speakers` values may only change in the final task**, and only to values this probe measured.
- **The profile records the full confusion, not just a verdict**, so the derivation rule can be re-applied without re-running the GPU sweep.
- **The 80 % threshold is a judgement and must be recorded as such** in the profile, beside the curve it was applied to — the convention `run_config`'s `⚠ UNDERIVED` marker sets.
- **Refuse rather than warn** on insufficient measurement, following `scripts/calibrate_detection_margin.py`.
- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Never `pytest -n auto`.** Serial, scoped. Check `uptime` before running locally — this box is shared.
- **Never `git add -A` unqualified.** Use `git add -A -- src/ scripts/ specs/`.
- **No test may construct an `HFModel`** without `monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)`.
- **`HF_HOME` on the cluster is redirected to scratch, which un-authenticates HuggingFace** unless `HF_TOKEN` is exported from `~/.cache/huggingface/token`. The cluster env file already does this; do not remove it or gated repos (pyannote) will 401.
- **Test docstrings in Google style**; ruff enforces `D205`/`D209`.

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `scripts/speaker_ceiling/generate.py` | Drive NeMo's `MultiSpeakerSimulator` in the nemo venv; emit sessions + RTTM per `k` | Create |
| `scripts/speaker_ceiling/evaluate.py` | Run each backend over the corpus, record predicted counts | Create |
| `scripts/speaker_ceiling/derive.py` | Pure: confusion → ceiling. No I/O, no GPU, fully unit-testable | Create |
| `scripts/probe_speaker_ceilings.py` | CLI tying the three together; refuses on insufficient measurement | Create |
| `src/senselab/audio/tasks/speaker_diarization/data/speaker_ceiling_profile.json` | The measured artifact | Create (final task) |
| `src/senselab/audio/tasks/speaker_diarization/{pyannote,vibevoice,moss,diarizen}.py` | `max_speakers` updated from `None` to measured | Modify (final task) |
| `src/tests/scripts/speaker_ceiling_derive_test.py` | Tests for the derivation rule | Create |

---

### Task 1: The derivation rule, pure and testable

Deliberately first and standalone: it is the only part carrying a judgement, and it needs no GPU, no NeMo, and no audio to test.

**Files:**
- Create: `scripts/speaker_ceiling/derive.py`
- Test: `src/tests/scripts/speaker_ceiling_derive_test.py`

**Interfaces:**
- Produces:
  ```python
  DEFAULT_ACCURACY_THRESHOLD = 0.8

  def exact_count_accuracy(predictions: list[int | None], true_k: int) -> float: ...
  def derive_ceiling(curve: dict[int, float], threshold: float = DEFAULT_ACCURACY_THRESHOLD) -> int | None: ...
  ```
  `curve` maps true speaker count → accuracy in `[0, 1]`. `derive_ceiling` returns the largest `k` such that every count from the minimum up to and including `k` meets the threshold, or `None` if even the smallest fails.

- [ ] **Step 1: Write the failing tests**

```python
"""The rule that turns a measured accuracy curve into a declared ceiling."""

import pytest

from scripts.speaker_ceiling.derive import (
    DEFAULT_ACCURACY_THRESHOLD,
    derive_ceiling,
    exact_count_accuracy,
)


def test_accuracy_counts_only_exact_matches() -> None:
    """Reporting 3 speakers when there are 4 is wrong, not partially right.

    A near-miss metric would let a backend that systematically undercounts look
    capable at high speaker counts, which is the exact error this probe exists to
    detect.
    """
    assert exact_count_accuracy([4, 4, 3, 4], true_k=4) == pytest.approx(0.75)


def test_a_refusal_counts_against_accuracy_but_is_not_a_wrong_answer() -> None:
    """None means the backend refused or crashed on that session.

    It cannot count as correct, but the caller still needs to distinguish it from
    a wrong number when reading the confusion — so it is preserved as None in the
    predictions and simply fails the exact-match test here.
    """
    assert exact_count_accuracy([2, None, 2, 2], true_k=2) == pytest.approx(0.75)


def test_empty_predictions_are_zero_not_a_crash() -> None:
    """A cell with no completed sessions scores zero, and the caller refuses on it."""
    assert exact_count_accuracy([], true_k=1) == 0.0


def test_ceiling_is_the_last_k_before_the_first_failure() -> None:
    """A backend good to 4 and poor at 5 has a ceiling of 4."""
    curve = {1: 1.0, 2: 1.0, 3: 0.95, 4: 0.85, 5: 0.30, 6: 0.10}
    assert derive_ceiling(curve) == 4


def test_a_later_recovery_does_not_raise_the_ceiling() -> None:
    """A ceiling a backend intermittently exceeds is not a ceiling.

    Scoring well at k=6 after failing k=4 means the k=6 successes are not
    dependable, so the honest ceiling is still 3.
    """
    curve = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.20, 5: 0.10, 6: 0.95}
    assert derive_ceiling(curve) == 3


def test_failing_the_smallest_count_yields_none() -> None:
    """A backend that cannot even do k=1 has no measurable ceiling.

    None here means 'the probe established nothing', which is the same meaning
    None already carries in DiarizationCapabilities.max_speakers.
    """
    assert derive_ceiling({1: 0.4, 2: 0.9, 3: 0.9}) is None


def test_threshold_is_a_parameter_not_a_constant() -> None:
    """The 80% rule is a judgement; a reader must be able to re-apply another.

    The profile records the full curve precisely so this can be recomputed
    without re-running 160 GPU sessions.
    """
    curve = {1: 1.0, 2: 0.85, 3: 0.5}
    assert derive_ceiling(curve, threshold=0.9) == 1
    assert derive_ceiling(curve, threshold=0.8) == 2


def test_default_threshold_is_the_documented_one() -> None:
    assert DEFAULT_ACCURACY_THRESHOLD == 0.8
```

- [ ] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/scripts/speaker_ceiling_derive_test.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.speaker_ceiling'`.

If the repo does not already make `scripts/` importable from tests, add `scripts/__init__.py` and `scripts/speaker_ceiling/__init__.py`. Check first: `ls src/tests/scripts/` and see how any existing test there imports.

- [ ] **Step 3: Implement**

```python
"""Turn a measured accuracy curve into a declared speaker ceiling.

Pure and I/O-free on purpose: this is the only part of the probe that carries a
judgement rather than a measurement, so it is the part that most needs to be
testable without a GPU, without NeMo, and without generating any audio.
"""

from __future__ import annotations

from typing import Dict, List, Optional

DEFAULT_ACCURACY_THRESHOLD = 0.8
"""Fraction of sessions at a given speaker count that must be exactly right.

A judgement, not a measurement — nothing was fitted to choose 0.8 over 0.75 or
0.9. It is recorded in the emitted profile beside the curve it was applied to, so
a reader who disagrees can re-derive from the same numbers rather than re-running
the sweep. This is the same posture ``run_config``'s ``snr_floor_db`` takes with
its explicit UNDERIVED marker.
"""


def exact_count_accuracy(predictions: List[Optional[int]], true_k: int) -> float:
    """Fraction of sessions where the backend reported exactly ``true_k`` speakers.

    Exact match only. A near-miss metric would let a backend that systematically
    undercounts look capable at high speaker counts, which is the error this probe
    exists to detect. ``None`` entries (refusal or crash) cannot count as correct;
    they are preserved upstream so the confusion can distinguish them from a wrong
    number.
    """
    if not predictions:
        return 0.0
    return sum(1 for p in predictions if p == true_k) / len(predictions)


def derive_ceiling(curve: Dict[int, float], threshold: float = DEFAULT_ACCURACY_THRESHOLD) -> Optional[int]:
    """Largest speaker count the backend handles dependably.

    Returns the largest ``k`` such that every count from the smallest measured up
    to and including ``k`` meets ``threshold``. A later recovery does not raise the
    ceiling: a backend that fails at 4 and succeeds at 6 cannot be depended on at
    6, so its honest ceiling is 3.

    Returns ``None`` when even the smallest measured count fails — meaning the
    probe established nothing, the same sense ``max_speakers=None`` already
    carries.
    """
    ceiling: Optional[int] = None
    for k in sorted(curve):
        if curve[k] < threshold:
            break
        ceiling = k
    return ceiling
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/scripts/speaker_ceiling_derive_test.py -q
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/ scripts/ && uv run ruff check src/ && uv run mypy src/senselab/
git add -A -- src/ scripts/
git commit -m "feat(probe): the speaker-ceiling derivation rule

Pure and GPU-free, because it is the only part of the probe carrying a judgement
rather than a measurement, and so the part that most needs testing in isolation.

A later recovery does not raise the ceiling: a backend that fails at 4 and
succeeds at 6 cannot be depended on at 6."
```

---

### Task 2: Corpus generation

**API confirmed on the cluster 2026-08-10 (job 20080436), against the nemo-diarization venv and
upstream source. Read this before writing code — it changes the task's prerequisites.**

```python
MultiSpeakerSimulator(cfg)                    # one OmegaConf object, not kwargs
  .generate_sessions(random_seed: int = None)
  .clean_up()
```

**The simulator does not synthesize speech.** Upstream's own class docstring: *"Simulates
multispeaker audio sessions using **single-speaker audio files and corresponding word
alignments**."* It samples from `data_simulator.manifest_filepath` and composes real utterances into
multi-speaker sessions, with `_min_alignment_count = 2` rejecting a manifest that lacks alignments.
The design doc called it a "synthetic data constructor", which is true of the *sessions* and not of
the *speech* — so this task needs an aligned single-speaker corpus as input, which was not costed.

Each manifest entry needs `audio_filepath`, `words`, and `alignments` (word end times in seconds,
parallel to `words`, **index 0 is always silence** — see upstream line 612). That is the standard
NeMo manifest format, and LibriSpeech with its published word alignments is the documented source.

**Prefer published alignments over generating them with senselab's own forced aligner.** The probe
measures senselab's diarizers against this ground truth; deriving that ground truth from another
senselab component would fold its alignment error into the measurement and make a poor result
ambiguous between "the diarizer miscounted" and "the alignments were wrong".

**Chosen approach: TTS-composed sessions, borrowing NeMo's session model rather than its
simulator.** Instead of sourcing an aligned corpus, synthesize the single-speaker material with
senselab's own `synthesize_texts` (Coqui exposes distinct voices via `tts.speakers[i]`, and
`speaker_wav` clones from reference clips when more identities are needed than the model ships), then
compose sessions ourselves.

Why this is better than LibriSpeech-plus-alignments, and not merely cheaper: **the ground truth
becomes constructive rather than estimated.** Placing a synthesized utterance at a chosen offset
means the RTTM is exact by definition — there is no alignment step whose error could be mistaken for
a diarizer miscounting. It also removes the corpus download, the published-alignment dependency, and
the need to run NeMo at all for generation.

What to borrow from `data_simulation.py` is its *session model*, which is the part carrying real
judgement: `turn_prob` (speaker switching), `dominance_var` / `min_dominance` (unequal speaking
time), `sentence_length_params` (negative-binomial utterance lengths), the overlap probability, and
`session_length`. A naive round-robin at equal dominance would make speaker counting far easier than
real conversation and inflate every ceiling.

**The caveat must ship with the numbers.** A ceiling measured on TTS-composed audio is a ceiling on
*clean, synthetically distinct voices*: no room acoustics, no channel variation, and vocoder
characteristics shared across speakers. That plausibly makes counting easier than real speech (more
separable identities) and could make it harder (shared synthesis artifacts). Either way the measured
value is an upper bound on well-conditioned audio, not a guarantee about a real recording — and the
profile records the generation method beside the curve so a reader can judge that for themselves,
exactly as it records the 80% threshold beside the distribution it was applied to.

The probe's knobs map onto the config directly: `session_config.num_speakers` is the *k* being
swept, `session_config.num_sessions` is sessions-per-count, and **`enforce_num_speakers` must be
true** — without it a session requested at *k*=8 may contain fewer speakers, which silently corrupts
the ground truth the whole probe rests on.


**Files:**
- Create: `scripts/speaker_ceiling/generate.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `generate_corpus(out_dir: Path, counts: Sequence[int], sessions_per_count: int, seed: int) -> Path` — writes `out_dir/k=<k>/session_<i>.wav` with sibling `.rttm`, plus a `manifest.json` recording the NeMo config used and the seed.

- [ ] **Step 1: Confirm NeMo's simulator API in the venv that already exists**

Do this before writing code against it — the API is the main unknown in this task.

```bash
source ~/orcd/scratch/senselab-diar-test/env.sh 2>/dev/null || true
uv run python - <<'PY'
from senselab.utils.subprocess_venv import ensure_venv, venv_python
from senselab.audio.tasks.speaker_diarization.nvidia import _NEMO_REQUIREMENTS, _NEMO_VENV, _NEMO_PYTHON
import subprocess
venv = ensure_venv(_NEMO_VENV, _NEMO_REQUIREMENTS, python_version=_NEMO_PYTHON)
code = (
    "from nemo.collections.asr.data.data_simulation import MultiSpeakerSimulator as S;"
    "import inspect; print(inspect.signature(S.__init__));"
    "print([m for m in dir(S) if not m.startswith('_')])"
)
print(subprocess.run([str(venv_python(venv)), "-c", code], capture_output=True, text=True).stdout)
PY
```

Record the real signature in your report. If `MultiSpeakerSimulator` is absent or renamed in the installed NeMo version, **stop and report BLOCKED** with what you found rather than substituting a different generator — the spec's ground-truth guarantee depends on this specific tool.

- [ ] **Step 2: Write the generator**

A worker script run inside the nemo venv, following the pattern in `nvidia.py`: build the `data_simulator.yaml`-equivalent config in Python, set the speaker count per batch, run the simulator, and collect the emitted audio plus RTTM.

Seed the simulator explicitly and record the seed in `manifest.json`. An unseeded corpus makes a re-run incomparable to the profile it produced.

- [ ] **Step 3: Generate one small corpus and inspect it by hand**

```bash
uv run python scripts/speaker_ceiling/generate.py --out /tmp/ceiling-smoke --counts 2 --sessions 2 --seed 17
find /tmp/ceiling-smoke -name '*.wav' -o -name '*.rttm' | head
```

Then verify the RTTM actually contains the requested number of distinct speakers — the ground truth is the foundation of every later number, so check it rather than assuming:

```bash
uv run python -c "
import pathlib, collections
for r in pathlib.Path('/tmp/ceiling-smoke').rglob('*.rttm'):
    spk = {line.split()[7] for line in r.read_text().splitlines() if line.strip()}
    print(r.name, 'distinct speakers:', len(spk), sorted(spk))
"
```

Expected: exactly 2 distinct speakers per session. If not, the corpus is wrong and everything downstream is meaningless — report BLOCKED.

- [ ] **Step 4: Commit**

```bash
uv run ruff format scripts/ && git add -A -- scripts/
git commit -m "feat(probe): synthetic multi-speaker corpus generation

Runs NeMo's MultiSpeakerSimulator in the nemo-diarization venv this repo already
builds. The RTTM ground truth is what turns 'how many speakers did it find' into a
measurement; the seed is recorded so a re-run is comparable to the profile it
produced."
```

---

### Task 3: Backend evaluation and the CLI

**Files:**
- Create: `scripts/speaker_ceiling/evaluate.py`, `scripts/probe_speaker_ceilings.py`

**Interfaces:**
- Consumes: `derive_ceiling`, `exact_count_accuracy` (Task 1); `generate_corpus` (Task 2).
- Produces: a profile JSON with, per backend: the full confusion (`{true_k: {predicted_count_or_"refused": n}}`), the accuracy curve, the derived ceiling, the threshold used, and the corpus manifest.

- [ ] **Step 1: Write the evaluation loop**

For each backend and each session: call `diarize_audios`, count distinct `speaker` values, record the integer. On any exception record `None` **and the exception type**, so a refusal (child-adult's `ValueError` under 10 s, or its CUDA requirement) stays distinguishable from a crash in the report.

Reuse `scripts/run_diarization_backends.py`'s model construction — including that it builds the model *inside* the try, because a gated repo raises at construction and would otherwise abort the whole sweep.

- [ ] **Step 2: Implement the refusals**

Hard-error, not warn, when:
1. any (backend, k) cell has fewer than the requested sessions completed;
2. a backend produced zero successful sessions at the smallest k.

Both messages must name the backend and the cell. Follow `scripts/calibrate_detection_margin.py`'s phrasing: state what was insufficient and what would fix it.

- [ ] **Step 3: Dry-run the whole pipeline at tiny scale on CPU**

```bash
uv run python scripts/probe_speaker_ceilings.py --counts 1 2 --sessions 2 --out /tmp/ceiling-dry --device cpu --backends pyannote
```

Expected: completes, emits a profile with a curve over `{1, 2}` for pyannote alone. This exercises generation → evaluation → derivation → refusal without a GPU. A tiny run should also *trip* the refusal if you ask for more sessions than you generated — verify that too, since an un-triggered refusal is an unverified one.

- [ ] **Step 4: Commit**

```bash
uv run ruff format scripts/ && git add -A -- scripts/
git commit -m "feat(probe): backend evaluation and CLI

Records the exception type alongside a None prediction so a refusal stays
distinguishable from a crash. Refuses on a partial sweep rather than emitting a
profile: a backend that crashed on the hard cases looks identical to one that
handled them badly, and that bias runs in the direction of a lower ceiling."
```

---

### Task 4: Run it on Engaging and fill in the declarations

**Files:**
- Create: `src/senselab/audio/tasks/speaker_diarization/data/speaker_ceiling_profile.json`
- Modify: `pyannote.py`, `vibevoice.py`, `moss.py`, `diarizen.py` — `max_speakers`
- Modify: `src/senselab/model_registry.yaml`, regenerate `model_registry.md`
- Test: `src/tests/audio/tasks/speaker_diarization_capabilities_test.py`

- [ ] **Step 1: Submit the full sweep**

`k` = 1…8, 20 sessions each, all six backends, on an H100. Use the `orcd-remote` skill's submit script and the existing `~/orcd/scratch/senselab-diar-test/env.sh`, which already exports `HF_TOKEN` from `~/.cache/huggingface/token` — without it, gated pyannote 401s and its entire row is lost.

Request enough wall-clock: MOSS took ~120 s on a single 21 s file, so budget generously and check the partition's `MaxTime`.

- [ ] **Step 2: Sanity-check before trusting any number**

Every backend should score near 1.0 at `k`=1. A backend that cannot count one speaker is measuring the harness, not itself — that is what the refusal in Task 3 catches, but check it by eye too.

Confirm child-adult's curve collapses after `k`=2 and Sortformer's after `k`=4. If either *exceeds* its declared ceiling, that is a real finding: report it rather than quietly updating the declaration.

- [ ] **Step 3: Update the four unmeasured declarations**

Replace `max_speakers=None` with the derived integer in `pyannote.py`, `vibevoice.py`, `moss.py`, `diarizen.py`, replacing the `# unmeasured — pending the NeMo synthetic-speaker probe` comment with one naming the profile and the measured accuracy at that `k`.

Leave child-adult's `2` and Sortformer's `4` alone unless the probe contradicted them — in which case stop and report, because that is a spec question, not a mechanical update.

- [ ] **Step 4: Update the registry and regenerate**

Update `max_speakers` in `model_registry.yaml` for the four, then `uv run python scripts/generate_model_registry.py`. Note the bare-key form (`max_speakers:`) exists only for unmeasured entries; a measured entry takes a plain integer and needs no comment.

- [ ] **Step 5: Verify the whole chain**

```bash
uptime
uv run pytest src/tests/audio/tasks/speaker_diarization_capabilities_test.py -q
uv run --with pre-commit pre-commit run --all-files 2>&1 | tail -20
uv run python scripts/generate_model_registry.py && git diff --exit-code src/senselab/model_registry.md
```

The capabilities test compares the registry against the code via `dataclasses.asdict`, so it fails if you update one and not the other — which is the intended behaviour, not an obstacle.

- [ ] **Step 6: Commit**

```bash
git add -A -- src/ scripts/
git commit -m "feat(probe): measured speaker ceilings replace four unmeasured Nones

Each number now carries the curve it was derived from, in
speaker_ceiling_profile.json alongside the full confusion, so the 80% rule can be
re-applied without re-running 160 GPU sessions."
```
