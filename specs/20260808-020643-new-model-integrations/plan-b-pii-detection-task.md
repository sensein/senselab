# Plan B — Standalone PII detection over text and `ScriptLine`

> **Verification status (2026-08-13, commit `ad4fffa2`):** every task below was verified complete against the code on branch `feat/diarization-backends`, except Task 7, whose unticked boxes are genuinely outstanding — see the note there. Boxes are ticked at *task-deliverable* granularity — the deliverable was confirmed present in the tree, not each TDD step observed independently.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn PII detection into a standalone senselab task callable on a string, a `ScriptLine`, or an `Audio` — adding PR #542's rule cascade as a third detector inside the venv that already hosts Presidio and GLiNER.

**Architecture:** The existing `workflows/audio_analysis/pii_subprocess.py` and the detection logic in `pii.py` move to `text/tasks/pii_detection/`, because their input is a transcript. #542's rule cascade is ported into the same `pii-detection` subprocess venv rather than the host, so no new host dependency and no new extra appear. A deliberately tiny `audio/tasks/pii_detection/` holds the `Audio` entry point, since it needs `transcribe_audios` and a module under `text/` importing from `audio/` would invert the layering. `workflows/audio_analysis/pii.py` survives as a thin adapter so the existing workflow keeps running.

**Tech Stack:** Python 3.12 host; the `pii-detection` subprocess venv on Python 3.13 (Presidio, spaCy `en_core_web_lg`, GLiNER `nvidia/gliner-pii`, and now `wordfreq` + `nltk`).

## Global Constraints

Copied from `design.md`. Every task's requirements implicitly include these.

- **No `analyze_audio` or `audio_analysis` wiring.** The only `audio_analysis` file this plan touches is `pii.py`, and only to keep what already runs working when its implementation moves.
- **No `run_config` changes.**
- **No new host dependencies and no new extras.** #542 proposed a `pii` extra; it is **not** created. `wordfreq` and `nltk` go into `_PII_REQUIREMENTS`, the subprocess venv's list.
- **Out of scope, cut on instruction:** task-compliance verification, `task_reference.json`, Tier A/B/C modality routing, the Tier C LLM judge, and the `.pt` batch driver.
- **Pre-alpha convention: rename and replace outright.** No parallel fields, no aliases, no deprecation shims. The old module is deleted, not deprecated.
- **Every Python command runs through `uv run`.**
- **Never run `pytest -n auto`.** Serial, scoped to the directory changed.
- **`uv sync` is subtractive** — always `--all-extras`.
- **Run `uv run ruff format` before any push.**
- **Never `git add -A` unqualified.** Always limit it with a pathspec (`git add -A -- src/ docs/ pyproject.toml uv.lock`). The repository root can hold untracked local secrets — a developer-supplied API token sitting beside the checkout is the case that prompted this — and an unqualified `git add -A` would stage one. `git status` is not a safeguard: an agent running these steps does not read it before committing.
- **Test cache isolation uses `monkeypatch.setattr`, never `.clear()`** on a module-level cache — clearing mutates real state that outlives the test.

## Preconditions

Branch `feat/new-model-integrations` already exists, cut from the merged `alpha` (PR #547, `79b37d93`). Run Plan A's Task 1 first to verify it and record the baseline. This plan can otherwise run before or after Plan A's remaining tasks; it touches no file Plan A touches.

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `src/senselab/text/tasks/pii_detection/__init__.py` | Public re-exports | Create |
| `src/senselab/text/tasks/pii_detection/api.py` | `detect_pii`; input normalisation; cross-detector and cross-source corroboration; confidence | Create (logic from `workflows/audio_analysis/pii.py`) |
| `src/senselab/text/tasks/pii_detection/subprocess_backend.py` | Venv constants, worker script, `detect_pii_via_subprocess` | Create (moved from `pii_subprocess.py`) |
| `src/senselab/text/tasks/pii_detection/rules.py` | #542's rule cascade, as worker-side source | Create |
| `src/senselab/text/tasks/pii_detection/local_llm.py` | Optional loopback-only LLM detector, off by default | Create |
| `src/senselab/text/tasks/pii_detection/doc.md` | Module documentation for pdoc | Create |
| `src/senselab/audio/tasks/pii_detection/api.py` | `detect_pii_in_audios` — transcribe, then delegate | Create |
| `src/senselab/audio/workflows/audio_analysis/pii.py` | Thin adapter preserving `PiiPassReport` / `report_to_dict` | Rewrite |
| `src/senselab/audio/workflows/audio_analysis/pii_subprocess.py` | — | **Delete** |
| `src/tests/text/tasks/pii_detection_test.py` | API, normalisation, corroboration, confidence | Create |
| `src/tests/text/tasks/pii_rules_test.py` | The rule cascade's precision guards | Create |
| `src/tests/text/tasks/pii_llm_test.py` | The LLM detector's loopback and failure semantics | Create |
| `src/tests/audio/workflows/pii_adapter_test.py` | The workflow adapter still produces `PiiPassReport` | Create |

---

### Task 1: Move the PII module to `text/tasks/pii_detection/`

A pure move. No behaviour changes — those come in Tasks 2–4. Doing it as its own task means the reviewer can see that nothing changed but the path.

**Files:**
- Create: `src/senselab/text/tasks/pii_detection/{__init__,api,subprocess_backend}.py`
- Delete: `src/senselab/audio/workflows/audio_analysis/pii_subprocess.py`
- Modify: `src/senselab/audio/workflows/audio_analysis/pii.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `text.tasks.pii_detection.subprocess_backend.detect_pii_via_subprocess(transcripts_by_source: dict[str, str], **kwargs) -> dict[str, Any]` with keys `spans_by_asr`, `failures`, `detectors_used`
  - `text.tasks.pii_detection.subprocess_backend.DETECTOR_PRESIDIO`, `DETECTOR_GLINER`, `_KNOWN_DETECTORS`
  - `text.tasks.pii_detection.api.PiiSpan`, `PiiReport`

- [x] **Step 1: Write a characterisation test against the current behaviour**

Before moving anything, pin what exists. Create `src/tests/text/tasks/pii_detection_test.py`:

```python
"""Characterisation tests for PII detection.

Written before the move from workflows/audio_analysis so the move can be shown
to change nothing but the import path.
"""

from senselab.text.tasks.pii_detection.api import PiiSpan, _compute_detection_confidence


def test_confidence_is_zero_when_no_spans() -> None:
    """'Detectors ran, found nothing' is 0.0. 'Detectors did not run' is None,
    and is carried on the report, not here."""
    assert _compute_detection_confidence([], n_asr_models=2) == 0.0


def test_two_detectors_agreeing_beats_one_detector_alone() -> None:
    """Cross-detector agreement is the strongest 'real entity vs hallucination'
    signal available at this layer, so it must dominate an equal raw score."""
    both = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9),
        PiiSpan(text="Jane Doe", category="PERSON", source="gliner/name", asr_model="w", score=0.9),
    ]
    one = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9),
    ]
    assert _compute_detection_confidence(both, n_asr_models=1) > _compute_detection_confidence(
        one, n_asr_models=1
    )


def test_cross_source_agreement_raises_confidence() -> None:
    """A span only one transcript contains is the prototypical ASR hallucination."""
    in_both = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="a", score=0.9),
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="b", score=0.9),
    ]
    in_one = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="a", score=0.9),
    ]
    assert _compute_detection_confidence(in_both, n_asr_models=2) > _compute_detection_confidence(
        in_one, n_asr_models=2
    )
```

- [x] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/text/tasks/pii_detection_test.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'senselab.text.tasks.pii_detection'`.

- [x] **Step 3: Perform the move**

```bash
mkdir -p src/senselab/text/tasks/pii_detection
git mv src/senselab/audio/workflows/audio_analysis/pii_subprocess.py \
       src/senselab/text/tasks/pii_detection/subprocess_backend.py
git mv src/senselab/audio/workflows/audio_analysis/pii.py \
       src/senselab/text/tasks/pii_detection/api.py
```

Then:
1. In `api.py`, change the import to `from senselab.text.tasks.pii_detection.subprocess_backend import (...)`.
2. Create `src/senselab/text/tasks/pii_detection/__init__.py`:

```python
"""Standalone PII detection over transcripts."""

from senselab.text.tasks.pii_detection.api import (
    PiiReport,
    PiiSpan,
    detect_pii,
    report_to_dict,
)
from senselab.text.tasks.pii_detection.subprocess_backend import (
    DETECTOR_GLINER,
    DETECTOR_PRESIDIO,
)

__all__ = [
    "DETECTOR_GLINER",
    "DETECTOR_PRESIDIO",
    "PiiReport",
    "PiiSpan",
    "detect_pii",
    "report_to_dict",
]
```

`detect_pii` and `PiiReport` do not exist yet — Task 2 adds them. Until then, comment those two names out of `__init__.py` and uncomment in Task 2. Leave a marker so it is not forgotten:

```python
# NOTE: detect_pii / PiiReport are added in Task 2 of plan-b. Until then this
# __init__ exports only what api.py actually defines.
```

3. Create a **temporary** `src/senselab/audio/workflows/audio_analysis/pii.py` re-exporting from the new location, so the workflow keeps importing. Task 6 replaces it with the real adapter:

```python
"""Temporary shim — replaced by the real adapter in Task 6 of plan-b."""

from senselab.text.tasks.pii_detection.api import (  # noqa: F401
    PiiPassReport,
    PiiSpan,
    detect_pii_in_pass,
    report_to_dict,
)
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/text/tasks/pii_detection_test.py src/tests/audio/workflows/ -v 2>&1 | tail -20
```

Expected: the three new tests PASS; the existing `audio_analysis` tests that touch PII still pass.

- [x] **Step 5: Commit**

```bash
uv run ruff format src/senselab/ src/tests/
uv run mypy src/senselab/text/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "refactor(pii): move PII detection to text/tasks/pii_detection

Pure move, no behaviour change. The input is a transcript, so the module
belongs under text/. workflows/audio_analysis/pii.py is a temporary re-export
shim, replaced by a real adapter later in this plan.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `detect_pii` over `str` and `ScriptLine`

**Files:**
- Modify: `src/senselab/text/tasks/pii_detection/api.py`
- Test: `src/tests/text/tasks/pii_detection_test.py`

**Interfaces:**
- Consumes: `detect_pii_via_subprocess` from Task 1.
- Produces:
  ```python
  @dataclass
  class PiiReport:
      contains_pii: bool
      n_spans: int
      categories: list[str]
      spans: list[PiiSpan]
      failures: dict[str, str]
      detector_used: str | None
      detection_confidence: float | None

  def detect_pii(
      inputs: str | ScriptLine | Sequence[str | ScriptLine],
      detectors: list[str] | None = None,
      presidio_score_threshold: float = 0.4,
      gliner_model: str | None = None,
      gliner_labels: list[str] | None = None,
      gliner_threshold: float = 0.5,
      require_cross_source_corroboration: bool = True,
  ) -> PiiReport | list[PiiReport]: ...

  def flatten_script_line(line: ScriptLine) -> str: ...
  ```
  A single input returns a single `PiiReport`; a sequence returns a list of the same length and order.

- [x] **Step 1: Write the failing tests**

Append to `src/tests/text/tasks/pii_detection_test.py`:

```python
import pytest

from senselab.text.tasks.pii_detection.api import detect_pii, flatten_script_line
from senselab.utils.data_structures import ScriptLine


def test_flatten_plain_script_line() -> None:
    assert flatten_script_line(ScriptLine(text="my name is Jane")) == "my name is Jane"


def test_flatten_joins_nested_chunks_depth_first() -> None:
    """A word-level ASR result and a segment-level one must scan identically.
    Whisper returns nested chunks; MMS alignment returns them too. If nesting
    changed what got scanned, PII coverage would silently depend on the backend.
    """
    line = ScriptLine(
        text=None,
        chunks=[
            ScriptLine(text="my name is"),
            ScriptLine(text="Jane Doe"),
        ],
    )
    assert flatten_script_line(line) == "my name is Jane Doe"


def test_flatten_ignores_a_speaker_only_line() -> None:
    """Diarization ScriptLines carry a speaker and no text. They contribute
    nothing rather than raising — a mixed list is a normal input."""
    assert flatten_script_line(ScriptLine(speaker="spk1")) == ""


def test_flatten_drops_whitespace_only_entries() -> None:
    line = ScriptLine(chunks=[ScriptLine(text="  "), ScriptLine(text="Jane")])
    assert flatten_script_line(line) == "Jane"


def test_detect_pii_with_no_detectors_short_circuits() -> None:
    """detectors=[] means 'the caller deliberately turned this off'. It must be
    distinguishable from 'the check failed' and from 'the check found nothing' —
    an auditor reading the report needs all three apart."""
    report = detect_pii("my name is Jane Doe", detectors=[])
    assert report.detector_used is None
    assert report.contains_pii is False
    assert report.detection_confidence is None
    assert "pii_disabled" in report.failures


def test_detect_pii_on_empty_text_does_not_spawn_a_subprocess() -> None:
    report = detect_pii("   ")
    assert report.n_spans == 0
    assert report.detector_used is None


def test_detect_pii_returns_one_report_per_input_in_order() -> None:
    reports = detect_pii(["", "  ", ""], detectors=[])
    assert isinstance(reports, list)
    assert len(reports) == 3


def test_detect_pii_single_input_returns_a_bare_report() -> None:
    assert not isinstance(detect_pii("", detectors=[]), list)
```

- [x] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/text/tasks/pii_detection_test.py -v
```

Expected: FAIL — `detect_pii` and `flatten_script_line` do not exist.

- [x] **Step 3: Implement**

In `api.py`, add `flatten_script_line`, rename `PiiPassReport` to `PiiReport` (dropping the `perturbation` field, which is workflow vocabulary — the adapter in Task 6 re-adds it), and add `detect_pii`. `flatten_script_line`:

```python
def flatten_script_line(line: ScriptLine) -> str:
    """Join a ScriptLine's text with its nested chunks', depth-first.

    Backends differ in where they put the words: Whisper returns segment text
    plus word-level ``chunks``, a forced aligner returns chunks with no parent
    text, and a diarization line carries a speaker and no text at all. Scanning
    only ``text`` would make PII coverage silently depend on which backend
    produced the transcript, so the whole tree is flattened.
    """
    parts: list[str] = []
    own = (line.text or "").strip()
    if own:
        parts.append(own)
    for child in line.chunks or []:
        nested = flatten_script_line(child)
        if nested:
            parts.append(nested)
    return " ".join(parts)
```

`detect_pii` normalises `inputs` to a list of strings, remembers whether the caller passed a scalar, and reuses the existing corroboration and confidence logic with one report per input. Keep the three early-return branches from the current `detect_pii_in_pass` — empty transcript, `detectors=[]`, subprocess failure — each leaving `detector_used=None` and `detection_confidence=None`, because a caller must be able to tell "did not run" from "ran and found nothing".

- [x] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/text/tasks/pii_detection_test.py -v
```

Expected: PASS, 11 tests.

- [x] **Step 5: Uncomment the `__init__.py` exports and commit**

```bash
uv run ruff format src/senselab/ src/tests/
uv run mypy src/senselab/text/
uv run pytest src/tests/text/ -v
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(pii): detect_pii over str and ScriptLine

Nested chunks are flattened depth-first so a word-level transcript and a
segment-level one scan identically — otherwise PII coverage would silently
depend on which ASR backend produced the input.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Fix the detector-agreement denominator

A live defect, fixed before Task 4 makes it worse. Its own task because a reviewer could reasonably approve this and reject the rule cascade.

**Files:**
- Modify: `src/senselab/text/tasks/pii_detection/api.py` — `_compute_detection_confidence`
- Test: `src/tests/text/tasks/pii_detection_test.py`

**Interfaces:**
- Consumes: `PiiSpan` from Task 1.
- Produces: `_compute_detection_confidence(spans: list[PiiSpan], n_asr_models: int, n_detectors_run: int) -> float` — note the **new third parameter**. Task 2's call site and Task 6's adapter both pass it.

- [x] **Step 1: Write the failing tests**

```python
def test_agreement_denominator_is_detectors_that_ran_not_detectors_that_exist() -> None:
    """When GLiNER fails to load and only Presidio runs, a Presidio finding is
    the *best available* evidence, not half-corroborated evidence. Dividing by
    the number of known detectors caps it at 0.5 as though a second detector
    had declined to confirm it — when in fact none was asked.
    """
    spans = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9)
    ]
    assert _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=1) == pytest.approx(0.9)


def test_a_third_detector_does_not_rescale_two_detector_agreement() -> None:
    """Adding the rule cascade must not silently move every previously published
    confidence. Two detectors agreeing out of two that ran is still full agreement.
    """
    two = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.8),
        PiiSpan(text="Jane Doe", category="PERSON", source="gliner/name", asr_model="w", score=0.8),
    ]
    assert _compute_detection_confidence(two, n_asr_models=1, n_detectors_run=2) == pytest.approx(0.8)


def test_partial_agreement_among_three_scores_between() -> None:
    three_ran_two_agree = [
        PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9),
        PiiSpan(text="Jane Doe", category="PERSON", source="gliner/name", asr_model="w", score=0.9),
    ]
    all_three = three_ran_two_agree + [
        PiiSpan(text="Jane Doe", category="PERSON", source="rules/gazetteer", asr_model="w", score=0.9),
    ]
    partial = _compute_detection_confidence(three_ran_two_agree, n_asr_models=1, n_detectors_run=3)
    full = _compute_detection_confidence(all_three, n_asr_models=1, n_detectors_run=3)
    assert 0.0 < partial < full == pytest.approx(0.9)


def test_denominator_never_divides_by_zero() -> None:
    """n_detectors_run=0 cannot reach here — the caller short-circuits — but a
    ZeroDivisionError in a scoring function is a bad failure mode regardless."""
    spans = [PiiSpan(text="x", category="PERSON", source="presidio", asr_model="w", score=0.5)]
    assert _compute_detection_confidence(spans, n_asr_models=1, n_detectors_run=0) >= 0.0
```

- [x] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/text/tasks/pii_detection_test.py -k "denominator or rescale or partial_agreement" -v
```

Expected: FAIL — `_compute_detection_confidence() got an unexpected keyword argument 'n_detectors_run'`.

- [x] **Step 3: Implement**

```python
def _compute_detection_confidence(
    spans: list[PiiSpan], n_asr_models: int, n_detectors_run: int
) -> float:
    ...
    denom_detectors = max(1, n_detectors_run)
    denom_asrs = max(1, n_asr_models)
    risks: list[float] = []
    for g in groups.values():
        detector_agreement = min(1.0, len(g["detectors"]) / denom_detectors)
        asr_agreement = min(1.0, len(g["asrs"]) / denom_asrs)
        risks.append(g["max_score"] * detector_agreement * asr_agreement)
    return max(risks) if risks else 0.0
```

Update the docstring's "cross-detector agreement" bullet: the denominator is the number of detectors that **ran for this report**, not the number the module knows about. Record why — dividing by the known set caps a single-detector finding at `1/len(_KNOWN_DETECTORS)` even when no other detector was asked, and it would silently rescale every published confidence each time a detector is added.

Update the call site in `detect_pii` to pass `n_detectors_run=len(detectors_used)`.

- [x] **Step 3b: Update Task 1's characterisation tests for the new signature**

Task 1 wrote three tests calling `_compute_detection_confidence(spans, n_asr_models=...)` with the old two-argument signature. They will now fail with a `TypeError`, which is correct — the signature changed deliberately. Add `n_detectors_run=2` to each (two detectors existed when those tests were written, which is what they were characterising), and check that their **assertions still hold** rather than only that they run:

```python
def test_confidence_is_zero_when_no_spans() -> None:
    assert _compute_detection_confidence([], n_asr_models=2, n_detectors_run=2) == 0.0
```

If an assertion no longer holds, stop — the fix has changed behaviour beyond the denominator and that needs explaining, not patching.

- [x] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/text/tasks/pii_detection_test.py -v
```

Expected: PASS, 15 tests.

- [x] **Step 5: Commit**

```bash
uv run ruff format src/senselab/ src/tests/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "fix(pii): divide detector agreement by the detectors that ran

len(_KNOWN_DETECTORS) as the denominator caps a Presidio-only finding at 0.5
when GLiNER failed to load — as though a second detector had declined to
corroborate it, when none was asked. It also means every detector added to the
module silently rescales every confidence already published.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Port #542's rule cascade as a third detector

**Files:**
- Create: `src/senselab/text/tasks/pii_detection/rules.py`
- Modify: `src/senselab/text/tasks/pii_detection/subprocess_backend.py`
- Test: `src/tests/text/tasks/pii_rules_test.py`

**Source:** `git show origin/pii-compliance-pipeline:scripts/pii_compliance_pipeline.py`. Port these, preserving their docstrings and the reasoning in them:

| Source lines | Symbols |
|---|---|
| 406–449 | `STRONG_RIGID`, `CONTEXTUAL_RIGID`, `WEAK_RIGID`, `MISC`, `CATEGORY_WEIGHTS`, `CANONICAL_LABEL`, `rigidity_tier`, `_entity` |
| 484–575 | `_zipf`, `_wordfreq_available`, `_name_hard_gate_eligible`, `_valid_structured_identifier`, `postprocess_entities` |
| 578–806 | `load_name_gazetteer`, `load_place_gazetteer`, `gazetteer_scan`, `regex_scan`, `ner_scan`, `honorific_scan`, `profession_scan`, `rareword_scan`, `_phrase_min_zipf`, `rare_role_scan` |
| 807–982 | `demographic_scan`, `_age_word_to_int`, `age_scan` |
| 1065–1101 | `combinatorial_scan` |
| 1306–1419 | `merge_pii`, `build_masked_preview` |

**Do not port:** anything from `word_error_rate` onward (compliance), `ComplianceChecker`, `Pipeline`, `process_folder`, the `.pt` loader, the report writers, or `llm_judgement_score` (which scores task completion, not PII). Those are out of scope.

`LocalLLM` itself is **in** scope but belongs to Task 4b, not here — it is a fourth detector, independent of the cascade, and a reviewer may reasonably want the cascade without it.

**Interfaces:**
- Consumes: `_KNOWN_DETECTORS` from Task 1, the confidence signature from Task 3.
- Produces: `DETECTOR_RULES = "rules"` in `subprocess_backend.py`, added to `_KNOWN_DETECTORS`; and worker-side `rules_scan(text: str) -> list[dict]` returning the same span shape as the Presidio and GLiNER scans (`{"text", "category", "source", "score"}`), with `source` prefixed `rules/<method>`.

- [x] **Step 1: Write the failing precision-guard tests**

Create `src/tests/text/tasks/pii_rules_test.py`. These are the guards #542's review found were broken; they are the reason this cascade is worth porting rather than reimplementing.

```python
"""The rule cascade's precision guards.

Every test here corresponds to a defect found in review of PR #542. They run on
the host against pure-Python helpers — the cascade's engine-dependent parts
(spaCy NER, gazetteer downloads) live in the subprocess worker and are covered
separately.
"""

import pytest

from senselab.text.tasks.pii_detection import rules


def test_zipf_returns_none_not_zero_without_wordfreq(monkeypatch) -> None:
    """0.0 means 'measured, maximally rare', and every caller reads rarity as
    evidence FOR a PII hit. Returning 0.0 on a missing dependency inverts the
    precision guards instead of relaxing them.
    """
    monkeypatch.setattr(rules, "_WORDFREQ_IMPORT", None)
    assert rules._zipf("the") is None


def test_a_word_only_device_identifier_is_dropped() -> None:
    """An ASR'd cough token flagged as a 'device identifier' has no digits.
    Real MRNs, SSNs, and account numbers always do."""
    assert rules._valid_structured_identifier("cough", "IDNUM") is False
    assert rules._valid_structured_identifier("MRN 4417829", "IDNUM") is True


def test_contact_requires_an_at_sign_or_seven_digits_or_an_ip() -> None:
    assert rules._valid_structured_identifier("jane@example.com", "CONTACT") is True
    assert rules._valid_structured_identifier("617 555 0134", "CONTACT") is True
    assert rules._valid_structured_identifier("192.168.1.10", "CONTACT") is True
    assert rules._valid_structured_identifier("telephone", "CONTACT") is False


def test_url_requires_a_url_shape() -> None:
    assert rules._valid_structured_identifier("https://example.com", "URL") is True
    assert rules._valid_structured_identifier("website", "URL") is False


def test_format_validation_is_not_switchable_by_recall_mode() -> None:
    """Format validity is a correctness check under either posture. Tying it to
    the precision flag let high-recall promote a word-only 'device identifier'
    straight to confirmed hard-gate PII — the opposite of what recall mode
    promises."""
    entities = [rules._entity(0, 5, "IDNUM", 0.99, "gliner")]
    for precision in (True, False):
        kept = rules.postprocess_entities(entities, "cough", precision_mode=precision)
        assert kept == [], f"word-only IDNUM survived with precision_mode={precision}"


def test_a_holiday_is_reclassified_as_a_date_not_a_name() -> None:
    entities = [rules._entity(0, 9, "NAME", 0.9, "ner")]
    kept = rules.postprocess_entities(entities, "Christmas", precision_mode=True)
    assert kept and kept[0]["category"] == "DATE_PARTIAL"


def test_a_lone_common_word_name_is_not_hard_gate_eligible() -> None:
    """Will / May / Grant / Mark are the classic NER false positives. They drop
    to needs_review rather than failing a file."""
    assert (
        rules._name_hard_gate_eligible(
            span_text="Will", start=0, source_text="Will you read this",
            methods=set(), engines={"gliner"}, score=0.7,
        )
        is False
    )


def test_a_multitoken_name_is_hard_gate_eligible() -> None:
    assert (
        rules._name_hard_gate_eligible(
            span_text="Jane Doe", start=0, source_text="Jane Doe speaking",
            methods=set(), engines={"gliner"}, score=0.7,
        )
        is True
    )


def test_unknown_word_frequency_takes_the_precision_safe_branch(monkeypatch) -> None:
    """Without wordfreq, 'is this a common word?' is unknown. Treating unknown as
    'rare' would make every lone token hard-gate eligible."""
    monkeypatch.setattr(rules, "_zipf", lambda _word: None)
    assert (
        rules._name_hard_gate_eligible(
            span_text="Will", start=0, source_text="Will you read this",
            methods=set(), engines={"gliner"}, score=0.7,
        )
        is False
    )


def test_rigidity_tiers_cover_every_weighted_category() -> None:
    """A category with a weight but no tier scores without being classified,
    which reads as a tier-less finding downstream."""
    for category in rules.CATEGORY_WEIGHTS:
        assert rules.rigidity_tier(category) != "misc" or category in rules.MISC


def test_age_over_ninety_is_flagged_and_under_is_not() -> None:
    """HIPAA Safe Harbor: ages over 89 are identifiers. Ages below are not."""
    assert rules.age_scan("I am ninety four years old", over_years=90)
    assert not rules.age_scan("I am forty two years old", over_years=90)
```

- [x] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/text/tasks/pii_rules_test.py -v
```

Expected: FAIL with `ModuleNotFoundError: ... 'rules'`.

- [x] **Step 3: Port the cascade**

```bash
git show origin/pii-compliance-pipeline:scripts/pii_compliance_pipeline.py \
  > "$CLAUDE_JOB_DIR/tmp/pii542.py"
```

Create `rules.py` from the line ranges in the table above. Four required adaptations:

1. **`PRECISION_MODE` becomes a parameter, not a module global.** `postprocess_entities(entities, source_text, *, precision_mode: bool = True)`. A module-level mutable posture flag is untestable in parallel and was the mechanism behind two of #542's review findings.
2. **`_zipf` imports `wordfreq` through a module-level indirection** so tests can `monkeypatch.setattr` it without touching an import cache:

```python
try:  # wordfreq lives in the pii-detection venv, not the host
    from wordfreq import zipf_frequency as _WORDFREQ_IMPORT
except ImportError:
    _WORDFREQ_IMPORT = None


def _zipf(word: str) -> float | None:
    """Zipf word frequency, or None when wordfreq is unavailable.

    None is NOT 0.0. 0.0 means "measured, maximally rare", and every caller reads
    rarity as evidence FOR a PII hit — so returning 0.0 on a missing dependency
    silently INVERTS the precision guards instead of relaxing them: a common-word
    NAME false positive becomes hard-gate eligible, and the soft rare-role
    qualifier fires on every phrase. Callers must treat None as "unknown" and take
    their precision-safe branch.
    """
    if _WORDFREQ_IMPORT is None:
        return None
    return _WORDFREQ_IMPORT(word.lower(), "en")
```
3. **Gazetteer loading must not `nltk.download` at import time.** Keep it behind the existing lazy loader and make the loader's failure a logged warning that disables the gazetteer method, not an exception.
4. **Type annotations throughout** — the repo runs `mypy` with the pydantic plugin, and `scripts/` was exempt where `src/` is not.

- [x] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/text/tasks/pii_rules_test.py -v
```

Expected: PASS, 12 tests.

- [x] **Step 5: Register the cascade as the third detector**

In `subprocess_backend.py`:

```python
DETECTOR_RULES = "rules"
_KNOWN_DETECTORS = (DETECTOR_PRESIDIO, DETECTOR_GLINER, DETECTOR_RULES)
```

Add `rules_scan(text)` to the worker script, emitting spans with `source="rules/<method>"` so the `source.split("/", 1)[0]` grouping in `_compute_detection_confidence` buckets them as one detector. Add `wordfreq` and `nltk` to `_PII_REQUIREMENTS` with a comment recording that they are here — not in a host extra — because the host must not carry them.

The worker needs `rules.py`'s source. Read it with `importlib.resources` and pass it into the worker's stdin payload alongside the transcripts, rather than duplicating the cascade as a string literal — a second copy of 400 lines will drift.

- [x] **Step 6: Verify the venv rebuilds on the changed requirements**

```bash
uv run python -c "
from senselab.utils.subprocess_venv import ensure_venv
import inspect
src = inspect.getsource(ensure_venv)
print('requirements' in src and ('hash' in src or 'digest' in src or 'marker' in src))
"
```

If `ensure_venv` does **not** key cache validity on the requirements list, a host with the old venv would silently run without `wordfreq` — and `_zipf` returning `None` means the guards quietly take their precision-safe branch, so nothing would look broken. In that case append a version suffix to the venv name (`_PII_VENV = "pii-detection-v2"`) and record why in a comment.

- [x] **Step 7: Run an end-to-end detection and commit**

```bash
uv run python -c "
from senselab.text.tasks.pii_detection import detect_pii
r = detect_pii('Hi, my name is Jane Doe, you can reach me at jane.doe@example.com or 617-555-0134.')
print('detectors:', r.detector_used)
print('categories:', r.categories)
print('confidence:', r.detection_confidence)
"
```

Expected: `detectors: presidio,gliner,rules` (order may vary), categories including `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, and a confidence near 1.0. **First run builds the venv and downloads GLiNER — allow several minutes.** If this hangs, it is machine contention, not a bug: rerun rather than disabling detectors.

```bash
uv run ruff format src/senselab/ src/tests/
uv run mypy src/senselab/text/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(pii): add #542's rule cascade as a third detector

Regex, gazetteers, self-disclosed demographics, rare roles, age > 90, and the
combinatorial re-identification window, running inside the existing
pii-detection venv alongside Presidio and GLiNER. wordfreq and nltk join the
venv's requirements; no new host dependency and no new extra.

PRECISION_MODE becomes a parameter rather than a module global — it was the
mechanism behind two review findings, because format validation and cross-engine
corroboration were switching off with it.

Co-Authored-By: Varun Thvar <77816253+TheSerperiorOne@users.noreply.github.com>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4b: The optional local-LLM PII engine

#542 runs four PII engines, not three: the rule cascade, GLiNER, Presidio, and an optional local LLM. The fourth is in scope — it is off by default and talks only to `localhost`.

**Files:**
- Create: `src/senselab/text/tasks/pii_detection/local_llm.py`
- Modify: `src/senselab/text/tasks/pii_detection/{api,subprocess_backend}.py`
- Test: `src/tests/text/tasks/pii_llm_test.py`

**Source:** `pii_compliance_pipeline.py` lines 1130–1277 (`class LocalLLM`), **excluding** `llm_judgement_score` at 1278, which scores task completion.

**Interfaces:**
- Consumes: `DETECTOR_RULES` and `_KNOWN_DETECTORS` from Task 4.
- Produces: `DETECTOR_LLM = "llm"` added to `_KNOWN_DETECTORS`; `LocalLlmConfig(base_url: str = "http://localhost:11434", model: str = "...", timeout_s: float = 60.0)`; and worker-side `llm_scan(text, config) -> list[dict]` in the same span shape as the other scans, `source="llm/<model>"`.

- [x] **Step 1: Write the failing tests**

```python
"""The optional local-LLM PII engine.

Never contacted in these tests: an engine that reaches the network during a unit
test is an engine that will reach the network during a compliance scan.
"""

import pytest

from senselab.text.tasks.pii_detection import api, local_llm
from senselab.text.tasks.pii_detection.subprocess_backend import _KNOWN_DETECTORS


def test_llm_is_off_by_default() -> None:
    """Default-on would mean a scan silently depends on whether a local server
    happens to be listening — the same corpus would score differently on two
    machines."""
    assert "llm" not in api.default_detectors()


def test_llm_is_a_known_detector_so_it_counts_in_the_agreement_denominator() -> None:
    assert "llm" in _KNOWN_DETECTORS


def test_base_url_must_be_loopback() -> None:
    """Transcript text never leaves the machine. A remote base_url would send
    clinical speech to a third party, which is the one thing this module
    promises not to do."""
    with pytest.raises(ValueError, match="localhost|loopback|127.0.0.1"):
        local_llm.LocalLlmConfig(base_url="https://api.example.com")


def test_loopback_forms_are_accepted() -> None:
    for url in ("http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"):
        assert local_llm.LocalLlmConfig(base_url=url).base_url == url


def test_an_unreachable_server_is_a_recorded_failure_not_a_clean_pass() -> None:
    """If the LLM cannot be reached, the report must say the detector did not
    run. Returning no spans would read as 'the LLM found no PII'."""
    result = local_llm.scan_or_fail("some text", local_llm.LocalLlmConfig(base_url="http://127.0.0.1:1"))
    assert result.spans == []
    assert result.failure is not None
    assert "llm" in result.failure.lower() or "connect" in result.failure.lower()
```

- [x] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/text/tasks/pii_llm_test.py -v
```

Expected: FAIL — `local_llm` does not exist.

- [x] **Step 3: Port `LocalLLM` with the loopback guard**

Two required changes to #542's class:

1. **Validate `base_url` is loopback in `__post_init__`.** #542 documents localhost-only as a property; making it a checked invariant is what stops a future edit quietly turning it into a remote call. Accept `localhost`, `127.0.0.0/8`, and `::1`.
2. **Distinguish "did not run" from "found nothing".** `scan_or_fail` returns a small result object carrying `spans` and an optional `failure` string; a connection error, a timeout, or an unparsable response all populate `failure`, and `detect_pii` records it in `report.failures` and leaves the detector out of `detectors_used`.

- [x] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/text/tasks/pii_llm_test.py -v
```

Expected: PASS, 5 tests.

- [x] **Step 5: Confirm the agreement denominator moved correctly**

Adding a fourth known detector is exactly the case Task 3 fixed. Verify it did:

```bash
uv run pytest src/tests/text/tasks/pii_detection_test.py -k "denominator or rescale" -v
```

Expected: PASS unchanged — those tests pass `n_detectors_run` explicitly, so growing `_KNOWN_DETECTORS` cannot move them. If they now fail, the fix regressed to using the known-detector count.

- [x] **Step 6: Commit**

```bash
uv run ruff format src/senselab/ src/tests/
uv run mypy src/senselab/text/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(pii): optional local-LLM detector, off by default, loopback-only

The loopback restriction is a checked invariant rather than a documented
property — transcript text never leaving the machine is this module's central
promise, and a future edit should have to defeat an assertion to break it.
An unreachable server is a recorded failure, never a clean pass.

Co-Authored-By: Varun Thvar <77816253+TheSerperiorOne@users.noreply.github.com>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: The `Audio` entry point

**Files:**
- Create: `src/senselab/audio/tasks/pii_detection/__init__.py`, `api.py`
- Test: `src/tests/audio/tasks/pii_detection_test.py`

**Interfaces:**
- Consumes: `detect_pii` from Task 2.
- Produces: `detect_pii_in_audios(audios: list[Audio], asr_model: SenselabModel | None = None, device: DeviceType | None = None, **detect_kwargs) -> list[PiiReport]`.

- [x] **Step 1: Write the failing test**

```python
"""PII detection run directly on an Audio object."""

from unittest.mock import patch

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.pii_detection import detect_pii_in_audios
from senselab.utils.data_structures import ScriptLine


def test_detect_pii_in_audios_transcribes_then_detects(mono_audio_sample: Audio) -> None:
    """The Audio entry point is a two-step composition, not a new engine. It must
    pass the transcript through unchanged — a transcription bug and a detection
    bug should stay distinguishable."""
    with patch(
        "senselab.audio.tasks.pii_detection.api.transcribe_audios",
        return_value=[[ScriptLine(text="my name is Jane Doe")]],
    ) as mock_asr:
        reports = detect_pii_in_audios([mono_audio_sample], detectors=[])
    mock_asr.assert_called_once()
    assert len(reports) == 1
    assert "pii_disabled" in reports[0].failures


def test_one_report_per_audio(mono_audio_sample: Audio) -> None:
    with patch(
        "senselab.audio.tasks.pii_detection.api.transcribe_audios",
        return_value=[[ScriptLine(text="a")], [ScriptLine(text="b")]],
    ):
        reports = detect_pii_in_audios(
            [mono_audio_sample, mono_audio_sample], detectors=[]
        )
    assert len(reports) == 2
```

Use whatever audio fixture `src/tests/conftest.py` already provides; check its name first with `grep -n "def mono_audio_sample\|@pytest.fixture" src/tests/conftest.py`.

- [x] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/audio/tasks/pii_detection_test.py -v
```

Expected: FAIL — module does not exist.

- [x] **Step 3: Implement**

```python
"""PII detection over audio: transcribe, then scan the transcript.

This module exists under ``audio/`` rather than beside the detection logic in
``text/tasks/pii_detection`` for one reason: it needs ``transcribe_audios``. A
module under ``text/`` importing from ``audio/`` would invert the layering that
put detection under ``text/`` in the first place.
"""

from typing import Any, List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.text.tasks.pii_detection.api import PiiReport, detect_pii, flatten_script_line
from senselab.utils.data_structures import DeviceType, SenselabModel


def detect_pii_in_audios(
    audios: List[Audio],
    asr_model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
    **detect_kwargs: Any,
) -> List[PiiReport]:
    """Transcribe each audio and scan its transcript for PII.

    One report per input audio, in order. Transcription uses whatever
    ``transcribe_audios`` defaults to unless ``asr_model`` says otherwise.

    Detection quality is bounded by transcription quality, and the two failure
    modes are different: an ASR that drops a spoken name produces a clean report
    on a recording that does contain PII. Callers who need corroboration across
    ASR backends should call :func:`detect_pii` directly with one transcript per
    backend, which is what the cross-source agreement term is for.
    """
    transcripts = transcribe_audios(audios=audios, model=asr_model, device=device)
    texts = [
        " ".join(flatten_script_line(line) for line in result).strip()
        for result in transcripts
    ]
    reports = detect_pii(texts, **detect_kwargs)
    return reports if isinstance(reports, list) else [reports]
```

Verify `transcribe_audios`' real signature and return shape before relying on it:

```bash
grep -n "def transcribe_audios" -A 25 src/senselab/audio/tasks/speech_to_text/api.py
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/pii_detection_test.py -v
```

Expected: PASS, 2 tests.

- [x] **Step 5: Commit**

```bash
uv run ruff format src/senselab/ src/tests/
uv run mypy src/senselab/audio/tasks/pii_detection/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "feat(pii): detect_pii_in_audios — run PII detection on an Audio

Lives under audio/ because it needs transcribe_audios; a module under text/
importing from audio/ would invert the layering that put detection under text/.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Move the pass-level API out of the task layer into a workflow adapter

**This task's original text is superseded and kept below only as a record.** It said "replace the
shim" in `audio_analysis/pii.py`. That shim no longer exists: it was deleted and its single
consumer (`scripts/analyze_audio.py`) repointed at the task module, because this repository's
pre-alpha convention forbids re-export shims. Following the original text literally would
re-introduce one.

The defect it was aiming at is still real, and is now precisely locatable. `detect_pii_in_pass`
and `PiiPassReport` live in `src/senselab/text/tasks/pii_detection/api.py`, and that file
mentions `perturbation` ten times. "Pass" and "perturbation" are `audio_analysis` vocabulary. The
task API is supposed to be standalone — that was the whole point of the relocation — and it
currently carries workflow concepts inside it.

So the direction is the inverse of what was written: **move the pass-level API out of the task
layer**, leaving `text/tasks/pii_detection` free of workflow vocabulary.

**Files:**
- Create: `src/senselab/audio/workflows/audio_analysis/pii.py` — the adapter, owning
  `PiiPassReport`, `detect_pii_in_pass`, and `report_to_dict`.
- Modify: `src/senselab/text/tasks/pii_detection/api.py` — remove those three and every
  `perturbation` reference; `detect_pii`, `PiiReport`, `PiiSpan` and the scoring helpers stay.
- Modify: `scripts/analyze_audio.py` — repoint its import at the adapter.
- Test: `src/tests/audio/workflows/pii_adapter_test.py`

**Interfaces:**
- Consumes: `detect_pii` and `PiiReport` (Task 2), `_compute_detection_confidence` and
  `corroboration_family` (Task 3 and its follow-up).
- Produces: `PiiPassReport`, `detect_pii_in_pass(perturbation, asr_resolved, **kwargs) ->
  PiiPassReport`, and `report_to_dict(report) -> dict` — **the same shapes they have today**, so
  `scripts/analyze_audio.py` changes only its import line and no artifact format moves.

**Two things the adapter exists to do**, and they are the reason it is not just a re-export:
1. **Re-attach `perturbation`**, which the workflow keys artifacts on and the task API
   deliberately does not carry.
2. **Own the multi-ASR ensemble.** `asr_resolved` maps model id → scriptlines, and cross-ASR
   corroboration is only meaningful with more than one transcript. The standalone path scans one
   transcript and cannot have it; that asymmetry belongs on the workflow side.

**A test must prove the layering, not just the behaviour:** assert that
`src/senselab/text/tasks/pii_detection/` contains no occurrence of `perturbation` and no import
from `senselab.audio.workflows`. Without that, the vocabulary drifts back one function at a time.

<details><summary>Superseded original text</summary>

**Files:**
- Rewrite: `src/senselab/audio/workflows/audio_analysis/pii.py`
- Test: `src/tests/audio/workflows/pii_adapter_test.py`

**Interfaces:**
- Consumes: `detect_pii` (Task 2), `_compute_detection_confidence` (Task 3).
- Produces: `PiiPassReport` and `detect_pii_in_pass(perturbation, asr_resolved, **kwargs) -> PiiPassReport` and `report_to_dict(report) -> dict`, all with the **same shape they have today**, so no `audio_analysis` caller changes.

</details>

- [x] **Step 1: Write the failing test**

```python
"""The workflow adapter preserves the contract audio_analysis already depends on."""

from senselab.audio.workflows.audio_analysis.pii import (
    PiiPassReport,
    detect_pii_in_pass,
    report_to_dict,
)


def test_adapter_returns_a_pass_report_carrying_the_perturbation() -> None:
    """`perturbation` is workflow vocabulary that the task API deliberately does
    not carry. The adapter re-attaches it, because the workflow keys artifacts on it."""
    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"openai/whisper-tiny": [{"text": "hello"}]},
        detectors=[],
    )
    assert isinstance(report, PiiPassReport)
    assert report.perturbation == "raw"


def test_adapter_report_to_dict_is_json_serializable() -> None:
    import json

    report = detect_pii_in_pass(
        perturbation="raw", asr_resolved={"m": [{"text": "hello"}]}, detectors=[]
    )
    json.dumps(report_to_dict(report))


def test_cross_asr_corroboration_survives_the_move() -> None:
    """A span in only one of several ASR transcripts is the prototypical
    hallucination. The workflow relies on this gate; the task API's own
    corroboration is per-input, so the adapter must keep doing it across inputs."""
    report = detect_pii_in_pass(
        perturbation="raw",
        asr_resolved={"a": [{"text": ""}], "b": [{"text": ""}]},
        detectors=[],
    )
    assert report.contains_pii is False
```

- [x] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/audio/workflows/pii_adapter_test.py -v
```

Expected: FAIL — the shim re-exports `detect_pii_in_pass` from a module where Task 2 renamed it away.

- [x] **Step 3: Write the adapter**

Replace the shim with a real module: `PiiPassReport` (a `PiiReport` plus `perturbation`), `detect_pii_in_pass` building `{asr_model → flattened text}`, calling `detect_pii` once per ASR transcript, then applying the **cross-ASR** corroboration that the task API cannot do (it sees each input independently), and `report_to_dict`.

Keep the docstring's explanation of why there is no category-severity weighting: in pediatric and clinical voice data the nominally most severe Presidio categories (`US_SSN`, `CREDIT_CARD`) have near-zero true-positive rate and are dominated by ASR digit hallucinations, so weighting them up inflates exactly the hits a reviewer should de-prioritise.

- [x] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/workflows/pii_adapter_test.py -v
uv run pytest src/tests/audio/workflows/ -v 2>&1 | tail -20
```

Expected: PASS, and no regression in the rest of the workflow tests.

- [x] **Step 5: Confirm nothing still imports the deleted module**

```bash
grep -rn "pii_subprocess" src/ scripts/ || echo "no references to the old module"
```

Expected: `no references to the old module`. Per the pre-alpha convention the old path is gone outright — no alias, no shim.

- [x] **Step 6: Commit**

```bash
uv run ruff format src/senselab/ src/tests/
uv run mypy src/senselab/
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "refactor(pii): workflow adapter over the standalone task

detect_pii_in_pass keeps its shape so no audio_analysis caller changes, and
keeps the cross-ASR corroboration the task API cannot do — it sees each input
independently. The old pii_subprocess module is deleted outright, per the
pre-alpha rename-and-replace convention.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Port the remaining `--selftest` checks and write `doc.md`

> **NOT DONE as of 2026-08-13 (commit `ad4fffa2`).** Verified against the tree, not assumed.
> Part of this task's coverage arrived early, under Task 4, when the cascade itself landed:
> the precision guards, holiday reclassification, common-word-NAME lever and
> structured-identifier format validation all have named tests in `pii_rules_test.py`.
> Three pieces are genuinely outstanding:
>
> 1. **GLiNER windowing was never ported.** `_gliner_chunks` does not exist, so
>    `subprocess_backend.py` calls `predict_entities` on the whole string and a long
>    transcript is silently truncated at the model's token limit. This is a correctness
>    bug in shipped code, not only a missing test — fix it before writing the test the
>    step below specifies.
> 2. **The `--flag-all-pii` interlock was never ported.** `postprocess_entities` has no
>    `flag_all` parameter, which is exactly the defect the step below was written to
>    guard: in #542 the flag was read only inside `if HIGH_RECALL:`, so on its own it did
>    nothing, silently.
> 3. **`doc.md` does not exist.** Every other task module in this branch has one.
>
> Task 4b was in the same state until 2026-08-13 and is now complete. Both were missed
> because no checkbox in this plan was ticked during execution, so nothing recorded which
> tasks had shipped.

**Files:**
- Modify: `src/tests/text/tasks/pii_rules_test.py`
- Create: `src/senselab/text/tasks/pii_detection/doc.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no importable interface. Deliverable is coverage parity with #542's self-test on the PII half, plus rendered module documentation.

- [ ] **Step 1: Enumerate #542's PII-side checks**

```bash
grep -n "def _selftest\|check(\|Part [A-K]" "$CLAUDE_JOB_DIR/tmp/pii542.py" | head -60
```

#542 reports 138 checks across Parts A–K. Parts covering Tier A calibration, task→reference resolution, and the composite compliance bands are **out of scope** — compliance is not being ported. Port the rest: the precision guards, the GLiNER windowing offsets, the common-word-NAME lever, the structured-identifier format validation, the holiday reclassification, and the `--flag-all-pii` interlock.

- [ ] **Step 2: Write the ported tests**

For each check, write a named pytest function whose docstring states the defect it guards. Do not write one parametrised test over an opaque table — a failing row should name the guard it broke. Task 4 established the shape; follow it:

```python
def test_gliner_window_offsets_are_absolute_not_window_relative() -> None:
    """GLiNER is run over overlapping word windows. A span's offsets come back
    relative to its window, so without re-basing they point at the wrong
    characters — and the masked preview then redacts the wrong text.
    """
    text = "one two three four five six seven eight nine ten Jane Doe"
    windows = list(rules._gliner_chunks(text, max_words=5, overlap_words=2))
    for window_text, offset in windows:
        assert text[offset : offset + len(window_text)] == window_text


def test_flag_all_pii_takes_effect_independently_of_recall_mode() -> None:
    """In #542 this flag was read only inside `if HIGH_RECALL:`, so on its own it
    did nothing — silently. The two are separate levers: 'report every candidate'
    is not the same request as 'lower the detection thresholds'.
    """
    entities = [rules._entity(0, 4, "AGE", 0.2, "age")]
    assert rules.postprocess_entities(
        entities, "I am 42", precision_mode=True, flag_all=True
    ) == entities
    assert rules.postprocess_entities(
        entities, "I am 42", precision_mode=True, flag_all=False
    ) == []
```

Work down the list from Step 1 and check each off. When a check has no meaningful analogue after the compliance code was dropped, say so in a one-line comment in the test module rather than leaving it silently absent — the count is the evidence that the port was complete.

- [ ] **Step 3: Run the whole text suite**

```bash
uv run pytest src/tests/text/ -v 2>&1 | tail -30
```

Expected: PASS. Note the count; #542's PII-side subset is roughly 60–80 checks.

- [ ] **Step 4: Write `doc.md`**

Cover: the three detectors and why each exists; why detection runs in a subprocess venv (host Python versions without spaCy wheels); why the GLiNER label list must stay flat and must not grow beyond the HIPAA-18 (the competing-claim interference measured on `john.doe@example.com` — port that comment from `subprocess_backend.py`, it is the most useful paragraph in the module); the confidence formula and its denominator; and the deliberate absence of category-severity weighting.

- [ ] **Step 5: Verify pdoc renders it**

```bash
grep -rn "doc.md" src/senselab/audio/workflows/audio_analysis/__init__.py \
  src/senselab/audio/tasks/speech_enhancement/__init__.py | head -5
```

Match whatever mechanism the existing `doc.md` files use to get picked up, and mirror it in `text/tasks/pii_detection/__init__.py`.

- [ ] **Step 6: Final check and commit**

```bash
uv run ruff format --check src/ && uv run ruff check src/ && uv run mypy src/senselab/
uv run pytest src/tests/text/ src/tests/audio/workflows/ src/tests/audio/tasks/pii_detection_test.py -v 2>&1 | tail -20
git add -A -- src/ docs/ pyproject.toml uv.lock
git commit -m "test(pii): port #542's PII-side self-test checks to pytest; add doc.md

A self-test flag inside a shipped module is a second test framework with no
fixtures, no collection, and no CI. Each check becomes a named test whose
docstring states the defect it guards.

Co-Authored-By: Varun Thvar <77816253+TheSerperiorOne@users.noreply.github.com>
Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```
