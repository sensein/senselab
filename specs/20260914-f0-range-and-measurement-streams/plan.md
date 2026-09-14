# F0 Range and Measurement Streams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the right signal with a range derived from it — replace the binary sex-typed F0 range with a per-recording narrowing, and move the Praat scalars and the phonetic posteriorgram off the speech-enhanced stream onto the unprocessed one.

**Architecture:** Three independent defects share one consequence. `extract_pitch_values` picks one of two hardcoded (floor, ceiling) pairs from a threshold on trimmed mean F0, so every measure downstream of `derive_f0_range` carries a step discontinuity at a sex-typed boundary. Separately, `preprocess.py` hands the FRCRN-enhanced stream to both the Praat extractor and the PPG worker, so forty acoustic scalars and one routing gate are computed on a denoiser's output. The fix is one function replacement and two `resolve_stream` argument changes, plus a re-run path in the extend driver that currently skips every store that already holds the measurements.

**Tech Stack:** Python 3.12, `uv`, pytest, numpy, parselmouth (Praat bindings), `ProvStore` (append-only W3C PROV-shaped provenance), Slurm array jobs on ORCD for corpus passes.

**Spec:** `specs/20260817-triage-workflow-dag/praat-instrument-audit.md` — findings 0, 1 and 8, and remediation steps 1, 1b and 2. Branch consumers: `specs/20260817-triage-workflow-dag/branch-voice.md` (V1, V4), `branch-ddk.md` (D1, D2), `branch-quality.md` (Q2). The contract is `specs/20260913-branch-contract-and-hints/design.md`.

## Global Constraints

- **No threshold may be fitted against this corpus.** Its only labels are declared task names, so fitting against them fits the declaration. Every new value must be parameter-free, or a declared convention with its reasoning recorded in `specs/`, or marked owed. The six kinds of owed are enumerated in `specs/20260817-triage-workflow-dag/branch-listening-sample.md`.
- Every stream is resampled to **16 kHz mono** before any measurement (`resample.target_hz: 16000`, `src/senselab/audio/workflows/triage/data/config/default.yaml:19`).
- **Pre-alpha: rename and replace outright.** No parallel fields, no aliases, no deprecation shims.
- **Rationale goes in `specs/`, never in code comments or docstrings.** Docstrings say what a thing is and how to call it.
- Google-style docstrings, line length 120, full type hints, `from __future__ import annotations`.
- All Python through `uv run`. **Never `pytest -n auto`** — run the directory you changed.
- `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean on every file touched.
- `CACHE_SCHEMA_VERSION` is **not** the invalidation lever here. It lives in `src/senselab/utils/tasks/cached_inference.py` and belongs to the `audio_analysis` workflow; `grep -r 'cached_inference\|cache_dir' src/senselab/audio/workflows/triage/` returns nothing. Re-derivation is an extend driver, which is why Task 6 exists.

## Scope

**In scope:** audit steps 2 (replace `derive_f0_range`), 1 (Praat scalars onto `plain`), 1b (PPG onto `plain`), and the extend driver's re-run path.

**Explicitly out of scope, each getting its own plan:** the CPPS reimplementation (step 4); jitter and shimmer from the `PeriodMark` sequence (step 3); withholding the scalars already written into 62,547 stores (step 0 — an extend driver of its own); instrument coverage per instrument (step 5); every branch capability (V1–V8, A1–A7, D1–D6, S1–S9, Q1–Q8); the SCREEN merge; and the nine pieces in the contract's decomposition.

**One thing this plan deliberately does not resolve.** `branch-voice.md` V3 records a task-shaped tension: narrowing buys octave-error robustness on stationary material and is wrong on a glide, where the derived ceiling is set by how high the speaker went, making V3's "did F0 reach the derived limit" conformance flag partly circular. This plan implements the narrowing and leaves that open, as the design leaves it.

---

## File Structure

| file | responsibility after this plan |
| --- | --- |
| `src/senselab/audio/tasks/features_extraction/praat_parselmouth.py` | `extract_pitch_values` returns a per-recording narrowed range from robust log-Hz percentiles of the wide-search contour, and reports whether it failed or found nothing. Modify `:358-446`. |
| `src/senselab/audio/tasks/phonation/api.py` | `derive_f0_range` distinguishes a parselmouth failure from a genuine absence. Modify `:45-69`. |
| `src/senselab/audio/workflows/triage/nodes/preprocess.py` | `ppg_input` and the Praat block read `plain`. Modify `:758`, `:809`, `:874`, `:880-895`. |
| `src/tests/audio/tasks/features_extraction_test.py` | the narrowing's own behaviour: monotonicity, no discontinuity, low and high sources. |
| `src/tests/audio/tasks/phonation_test.py` | `derive_f0_range`'s contract. **Rewrite `TestDeriveF0Range` at `:109-131` — it currently asserts the bin's exact output.** |
| `src/tests/audio/workflows/triage/nodes/preprocess_test.py` | both measurements record `signal="plain"`. **Rename the test at `:2210`, which asserts the enhanced stream in its own name.** |
| `scripts/extend_ppg_praat.py` | gains `--force` so a completed slice can be re-derived. Modify `:94-115`, `:187-190`, `:235-238`, and the docstring at `:26-27`. |
| `src/tests/scripts/extend_ppg_praat_test.py` | `--force` re-derives on a store that already holds both measurements. |
| `specs/20260817-triage-workflow-dag/praat-instrument-audit.md` | steps 1, 1b and 2 marked landed, with the new convention's reasoning. |

### Two existing tests pin the defects

Both must be rewritten, not extended, and finding them is why this plan starts with tests rather than code:

- `src/tests/audio/tasks/phonation_test.py:112-119` asserts `low == (60.0, 250.0)` and `high == (100.0, 500.0)` under the docstring *"The narrowing is what the standardization method does; a fixed corpus range cannot."* It asserts the bin while describing the thing the bin is not.
- `src/tests/audio/workflows/triage/nodes/preprocess_test.py:2210` is named `test_praats_scalars_are_attributes_and_the_stream_is_the_enhanced_one`.

### What does not need changing, and why

`routing_analysis/features.py` finds the PPG measurement **by name** and resolves its sidecar from the measurement's own `path` attribute — it does not filter on `signal`. So Task 5 changes the DDK gate feature's **value**, not its availability. Verified: no `signal`-keyed PPG lookup exists in that module.

---

## Task 1: The narrowing replaces the bin

**Files:**
- Modify: `src/senselab/audio/tasks/features_extraction/praat_parselmouth.py:358-446`
- Test: `src/tests/audio/tasks/features_extraction_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `extract_pitch_values(snd, search_floor_hz: float = 50.0, search_ceiling_hz: float = 600.0) -> Dict[str, float]` returning keys `pitch_floor`, `pitch_ceiling` (both `float`, both `np.nan` when no pitch was placed) and `pitch_frames` (`float`, the count of voiced frames the narrowing rested on, `0.0` when none). Task 2 reads all three.

The current body, verified at `:414-436`: it takes the wide-search contour, drops zeros, trims at ±2 SD **in linear Hz**, takes the mean, and returns one of two hardcoded pairs on `mean_pitch < 170`. Three things change — the trim moves to log-Hz, the two pairs become percentiles of the contour, and the frame count becomes an output so a caller can see how much signal the range rests on.

- [ ] **Step 1: Write the failing test for monotonicity**

Add to `src/tests/audio/tasks/features_extraction_test.py`:

```python
class TestPitchRangeNarrowing:
    """The range is this recording's own, narrowed off a wide search — not one of two presets."""

    def test_the_derived_range_rises_monotonically_with_source_f0(self) -> None:
        """A preset bin is a step function of F0; a narrowing is monotone in it."""
        ceilings = [
            extract_pitch_values(_buzz(f0), search_floor_hz=50.0, search_ceiling_hz=600.0)["pitch_ceiling"]
            for f0 in (90.0, 130.0, 180.0, 260.0, 380.0)
        ]
        assert ceilings == sorted(ceilings), f"ceiling must not decrease as F0 rises: {ceilings}"
        assert ceilings[0] < ceilings[-1], "a 90 Hz and a 380 Hz voice cannot share a ceiling"
```

Use the `_buzz` helper already in `src/tests/audio/tasks/phonation_test.py:24-28` — copy it into this module, or import it if the file already has an equivalent; check before duplicating.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/tests/audio/tasks/features_extraction_test.py::TestPitchRangeNarrowing -v`
Expected: FAIL. The 90 Hz and 130 Hz sources both land in the `< 170` bin and return ceiling 250.0, so `ceilings[0] == ceilings[1]` and the strict inequality on the ends may hold while monotonicity is satisfied only by accident — read the actual failure before proceeding.

- [ ] **Step 3: Write the failing test for the absent discontinuity**

```python
    def test_no_discontinuity_at_the_retired_170_hz_boundary(self) -> None:
        """The retired rule stepped floor/ceiling from 60/250 to 100/500 across mean F0 = 170 Hz."""
        below = extract_pitch_values(_buzz(165.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        above = extract_pitch_values(_buzz(175.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert below["pitch_ceiling"] == pytest.approx(above["pitch_ceiling"], rel=0.15)
        assert below["pitch_floor"] == pytest.approx(above["pitch_floor"], rel=0.15)
```

- [ ] **Step 4: Write the failing tests for the two clipped populations**

```python
    def test_a_sub_60_hz_source_is_not_floored_at_60(self) -> None:
        """Vocal fry and Parkinsonian creak sat below the retired 60 Hz floor and yielded no pulses."""
        derived = extract_pitch_values(_buzz(55.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_floor"] < 60.0, "the floor must follow the voice below the retired bin"

    def test_a_420_hz_source_is_not_clipped_at_250(self) -> None:
        """A child or high-F0 speaker whose trimmed mean fell below 170 was tracked to 250 Hz."""
        derived = extract_pitch_values(_buzz(420.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_ceiling"] > 420.0, "the ceiling must admit the source it was derived from"

    def test_the_range_reports_the_frames_it_rests_on(self) -> None:
        """A range derived from four voiced frames is not the same claim as one from four hundred."""
        derived = extract_pitch_values(_buzz(150.0, seconds=2.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_frames"] > 0.0
```

- [ ] **Step 5: Run all four and confirm they fail for the stated reasons**

Run: `uv run pytest src/tests/audio/tasks/features_extraction_test.py::TestPitchRangeNarrowing -v`
Expected: FAIL. `pitch_frames` raises `KeyError`; the 55 Hz case returns floor 60.0 exactly; the 420 Hz case returns ceiling 250.0 if its trimmed mean lands below 170 and 500.0 otherwise — record which, because it tells you whether the wide search is tracking the source at all.

- [ ] **Step 6: Replace the bin with the narrowing**

In `praat_parselmouth.py`, replace `:420-438` (from `pitch_values = pitch_values[pitch_values != 0]` through `return {"pitch_floor": pitch_floor, "pitch_ceiling": pitch_ceiling}`) with:

```python
        pitch_values = pitch_values[pitch_values != 0]
        if pitch_values.size == 0:
            return {"pitch_floor": np.nan, "pitch_ceiling": np.nan, "pitch_frames": 0.0}

        log_values = np.log2(pitch_values)
        median_log, spread_log = np.median(log_values), np.std(log_values)
        kept = pitch_values[np.abs(log_values - median_log) <= 2.0 * spread_log] if spread_log > 0 else pitch_values
        if kept.size == 0:
            return {"pitch_floor": np.nan, "pitch_ceiling": np.nan, "pitch_frames": 0.0}

        low, high = np.percentile(kept, [PITCH_RANGE_LOW_PERCENTILE, PITCH_RANGE_HIGH_PERCENTILE])
        pitch_floor = max(float(search_floor_hz), float(low) / PITCH_RANGE_MARGIN)
        pitch_ceiling = min(float(search_ceiling_hz), float(high) * PITCH_RANGE_MARGIN)
        return {
            "pitch_floor": pitch_floor,
            "pitch_ceiling": pitch_ceiling,
            "pitch_frames": float(kept.size),
        }
```

And add the three module constants beside the other module-level names near the top of the file:

```python
PITCH_RANGE_LOW_PERCENTILE = 5.0
PITCH_RANGE_HIGH_PERCENTILE = 95.0
PITCH_RANGE_MARGIN = 1.5
```

Three things to note while editing. The trim is now in **log2 Hz**, because an octave error is a factor and a linear trim about a mean that sits between two octave-separated modes removes neither. The margin is multiplicative for the same reason. And the `except Exception` at `:439-445` must also gain `"pitch_frames": 0.0` so every return path carries the same keys — Task 2 depends on that.

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `uv run pytest src/tests/audio/tasks/features_extraction_test.py::TestPitchRangeNarrowing -v`
Expected: PASS, all five.

- [ ] **Step 8: Record the convention in the spec**

The three constants are **declared conventions, not fits** — they are not derived from this corpus and must say so. Add to `specs/20260817-triage-workflow-dag/praat-instrument-audit.md` under step 2: the 5th/95th percentiles and the 1.5× margin are Hirst's two-pass method as conventionally parameterised, the trim is in log-Hz because octave errors are multiplicative, and none of the three was fitted against the corpus. Mark the glide tension as owed, cross-referencing `branch-voice.md` V3.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff format src/senselab/audio/tasks/features_extraction/praat_parselmouth.py src/tests/audio/tasks/features_extraction_test.py
uv run ruff check src/senselab/audio/tasks/features_extraction/praat_parselmouth.py src/tests/audio/tasks/features_extraction_test.py
uv run mypy src/senselab/audio/tasks/features_extraction/praat_parselmouth.py
git add -A && git commit -m "fix(praat): derive the F0 range per recording instead of binning on sex"
```

---

## Task 2: A parselmouth failure stops reading as an absence

**Files:**
- Modify: `src/senselab/audio/tasks/phonation/api.py:45-69`
- Test: `src/tests/audio/tasks/phonation_test.py`

**Interfaces:**
- Consumes: `extract_pitch_values`'s three keys from Task 1.
- Produces: `F0RangeUnavailable` for a genuine absence (unchanged); a new `F0RangeFailed(RuntimeError)` for a parselmouth failure. `derive_f0_range`'s signature and return type are unchanged. Both must be exported from `src/senselab/audio/tasks/phonation/__init__.py`.

`extract_pitch_values` has **two** identical NaN returns: the non-finite guard at `:426`, which is a real absence, and the bare `except Exception` at `:445`, which is a crash. `derive_f0_range:63-68` checks only `np.isfinite` and raises `F0RangeUnavailable` for both — the same crash/absence conflation the audit condemns for CPPS at finding 2.

- [ ] **Step 1: Write the failing test**

```python
class TestDeriveF0RangeDistinguishesFailureFromAbsence:
    """A crashed analysis and a recording with no pitch are not the same finding."""

    def test_silence_is_an_absence(self) -> None:
        """No pitch placed over the wide search is an absence, attributable as one."""
        silence = Audio(waveform=np.zeros((1, SR), dtype=np.float32), sampling_rate=SR)
        with pytest.raises(F0RangeUnavailable):
            derive_f0_range(silence, search_floor_hz=50.0, search_ceiling_hz=600.0)

    def test_a_parselmouth_failure_is_not_an_absence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A crash must not be attributed as 'this recording has no pitch'."""

        def _boom(*_args: object, **_kwargs: object) -> dict[str, float]:
            raise RuntimeError("parselmouth exploded")

        monkeypatch.setattr("senselab.audio.tasks.phonation.api.extract_pitch_values", _boom)
        with pytest.raises(F0RangeFailed):
            derive_f0_range(_buzz(150.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
```

Import `F0RangeFailed` alongside `F0RangeUnavailable` at the top of the test module.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py::TestDeriveF0RangeDistinguishesFailureFromAbsence -v`
Expected: FAIL on the import — `F0RangeFailed` does not exist.

- [ ] **Step 3: Add the second typed error and stop swallowing the crash**

In `api.py`, beside `F0RangeUnavailable` at `:41-42`:

```python
class F0RangeFailed(RuntimeError):
    """The pitch analysis itself failed; this is an operational fault, never an absence."""
```

Then in `derive_f0_range`, replace the body from `values = extract_pitch_values(...)` through the `raise`:

```python
    try:
        values = extract_pitch_values(audio, search_floor_hz=search_floor_hz, search_ceiling_hz=search_ceiling_hz)
    except Exception as error:
        raise F0RangeFailed(f"the pitch analysis failed on this recording: {error}") from error
    floor, ceiling = float(values["pitch_floor"]), float(values["pitch_ceiling"])
    if not np.isfinite(floor) or not np.isfinite(ceiling):
        raise F0RangeUnavailable(
            f"no F0 range could be derived from this recording over [{search_floor_hz}, {search_ceiling_hz}] Hz"
        )
    return floor, ceiling
```

Add `F0RangeFailed` to the `Raises:` block of the docstring, and export it from `__init__.py` beside `F0RangeUnavailable` in both the import list and `__all__`.

- [ ] **Step 4: Run and confirm both pass**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py -v`
Expected: PASS for the new class. **`TestDeriveF0Range` at `:109-131` will now FAIL** — it asserts `low == (60.0, 250.0)`. That is correct and is Step 5.

- [ ] **Step 5: Rewrite the test that pinned the bin**

Replace `test_a_low_voice_and_a_high_voice_get_different_ranges` at `:112-119` with:

```python
    def test_a_low_voice_and_a_high_voice_get_ranges_that_follow_them(self) -> None:
        """The range follows the recording; it is not one of two presets chosen by a threshold."""
        low = derive_f0_range(_buzz(110.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        high = derive_f0_range(_buzz(230.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert low[0] < 110.0 < low[1], f"110 Hz must sit inside its own derived range, got {low}"
        assert high[0] < 230.0 < high[1], f"230 Hz must sit inside its own derived range, got {high}"
        assert low[1] < high[1], "a higher voice must get a higher ceiling"
```

Its old docstring claimed the narrowing while asserting the bin; the new one asserts what it claims.

- [ ] **Step 6: Run the whole phonation and features suites**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py src/tests/audio/tasks/features_extraction_test.py -q`
Expected: PASS. Any other failure here is a caller that depended on the bin's exact output — read it rather than adjusting the assertion.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format src/senselab/audio/tasks/phonation/ src/tests/audio/tasks/phonation_test.py
uv run ruff check src/senselab/audio/tasks/phonation/ src/tests/audio/tasks/phonation_test.py
uv run mypy src/senselab/audio/tasks/phonation/
git add -A && git commit -m "fix(phonation): a failed pitch analysis is a fault, not an absence"
```

---

## Task 3: The narrowing's consumers still pass

**Files:**
- Test: `src/tests/audio/workflows/triage/nodes/preprocess_test.py`, `src/tests/audio/workflows/triage/nodes/voice_test.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: nothing. This task is a gate, not a change.

`derive_f0_range` has two production callers: `preprocess.py:941` (on `plain`, feeding `f0_track` and `formant_track`) and `voice.py:71-73` (on `plain`, feeding `hnr_track` and `period_marks`). Both now receive a range that varies continuously.

- [ ] **Step 1: Run both consumer suites**

Run: `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py src/tests/audio/workflows/triage/nodes/voice_test.py -q`
Expected: PASS. If a test fails, it asserted a window length or a track shape that followed from the bin's fixed floor — Praat's window is `periods_per_window / floor`, so a changed floor changes frame counts.

- [ ] **Step 2: Commit only if something needed changing**

If nothing failed, skip. If something did:

```bash
git add -A && git commit -m "test(triage): the F0 range varies continuously, so window lengths follow the voice"
```

---

## Task 4: The Praat scalars move to `plain`

**Files:**
- Modify: `src/senselab/audio/workflows/triage/nodes/preprocess.py:874-895`
- Test: `src/tests/audio/workflows/triage/nodes/preprocess_test.py:2210`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: the `praat_features` measurement recording `signal="plain"` and `derived_from=(plain_id,)`.

Verified at `:880`: `enhanced_id, audio = resolve_stream(store, run_dir, "enhanced")`, then `:882` mints the activity with `(enhanced_id,)`, `:892` writes `signal="enhanced"`, `:895` writes `derived_from=(enhanced_id,)`, and the docstring `Raises:` at `:874` names the enhanced stream. All five change together — this is not one argument.

- [ ] **Step 1: Rename and invert the test that asserts the enhanced stream**

At `preprocess_test.py:2210`, rename `test_praats_scalars_are_attributes_and_the_stream_is_the_enhanced_one` to `test_praats_scalars_are_attributes_and_the_stream_is_the_unprocessed_one`, and add to its body:

```python
        assert attrs["signal"] == "plain", "FRCRN removes aperiodic energy, which is the breathiness signal"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -k praats_scalars -v`
Expected: FAIL with `assert 'enhanced' == 'plain'`.

- [ ] **Step 3: Change all five sites**

In `preprocess.py`, in the Praat block: change `:880` to `plain_id, audio = resolve_stream(store, run_dir, "plain")`, `:882`'s activity tuple to `(plain_id,)`, `:892` to `signal="plain"`, `:895` to `derived_from=(plain_id,)`, and the docstring `Raises:` at `:874` to name the `plain` stream.

- [ ] **Step 4: Run and confirm it passes**

Run: `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -k praats_scalars -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/senselab/audio/workflows/triage/nodes/preprocess.py src/tests/audio/workflows/triage/nodes/preprocess_test.py
uv run ruff check src/senselab/audio/workflows/triage/nodes/preprocess.py src/tests/audio/workflows/triage/nodes/preprocess_test.py
uv run mypy src/senselab/audio/workflows/triage/nodes/preprocess.py
git add -A && git commit -m "fix(preprocess): measure the Praat scalars on plain, not on a denoiser's output"
```

---

## Task 5: The PPG moves to `plain`, and DDK routing moves with it

**Files:**
- Modify: `src/senselab/audio/workflows/triage/nodes/preprocess.py:758`, `:809`
- Test: `src/tests/audio/workflows/triage/nodes/preprocess_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: the `ppg_posteriorgram` measurement recording `signal="plain"`.

This is a different event from Task 4. The PPG feeds `ppg.segment_rate_per_s`, which is a DDK routing gate, so this changes **which branch runs** on recordings across the corpus — not just what a number reads. `routing_analysis/features.py` finds the measurement by name and resolves the sidecar from its `path` attribute, with no `signal` filter, so the feature's availability is unaffected and only its value moves.

- [ ] **Step 1: Write the failing test**

```python
    def test_the_posteriorgram_is_computed_on_the_unprocessed_stream(
        self, residual_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The PPG feeds a DDK routing gate, so a denoiser in its path moves which branch runs."""
        store, run_dir = _run_preprocess(residual_config, tmp_path, monkeypatch)
        ppg = find_measurement(store, "ppg_posteriorgram")
        assert ppg is not None
        assert ppg.attributes["signal"] == "plain"
```

Match the fixture and helper names the neighbouring tests in this file already use — read the PPG tests near `:2259` and reuse their setup rather than inventing one.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -k posteriorgram_is_computed -v`
Expected: FAIL with `assert 'enhanced' == 'plain'`.

- [ ] **Step 3: Change both sites**

In `ppg_input`: change `:758` to `plain_id, audio = resolve_stream(store, run_dir, "plain")`, rename the local through its uses in that function, and change `:809` to `signal="plain"`. Update the function's docstring where it names the enhanced stream.

- [ ] **Step 4: Run and confirm**

Run: `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -k posteriorgram -v`
Expected: PASS.

- [ ] **Step 5: Assert the enhanced stream has no measurement consumers left**

Add to the same test class:

```python
    def test_no_measurement_reads_the_enhanced_stream(self) -> None:
        """The audit's scope bound is that exactly two sites read it; both have moved."""
        source = Path("src/senselab/audio/workflows/triage/nodes/preprocess.py").read_text(encoding="utf-8")
        assert 'resolve_stream(store, run_dir, "enhanced")' not in source
```

This is the regression guard for the audit's scope claim. The enhanced stream is still *written* by the enhancement step and still read by the residual computation — this asserts only that no measurement resolves it.

- [ ] **Step 6: Run the whole triage node suite**

Run: `uv run pytest src/tests/audio/workflows/triage -q`
Expected: PASS. A failure in `live_evidence_test.py` or `routing_analysis_test.py` means a fixture hardcodes `"signal": "enhanced"` for a PPG measurement — `live_evidence_test.py:216` and `routing_analysis_test.py:949`/`:959` do. Update those fixtures to `"plain"`; they are constructing a measurement the node no longer writes that way.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format src/senselab/audio/workflows/triage/ src/tests/audio/workflows/triage/
uv run ruff check src/senselab/audio/workflows/triage/ src/tests/audio/workflows/triage/
uv run mypy src/senselab/audio/workflows/triage/nodes/preprocess.py
git add -A && git commit -m "fix(preprocess): compute the posteriorgram on plain, which moves DDK routing"
```

---

## Task 6: The extend driver gains a re-run path

**Files:**
- Modify: `scripts/extend_ppg_praat.py:26-27`, `:94-115`, `:187-190`, `:235-238`
- Test: `src/tests/scripts/extend_ppg_praat_test.py`

**Interfaces:**
- Consumes: Tasks 4 and 5 — without them there is nothing to re-derive.
- Produces: `--force` on the CLI, and a `force: bool` parameter on whatever function `run_slice` is called; match the module's existing signature style.

Verified: the docstring at `:26-27` says *"A recording whose store already holds both measurements is skipped"*; `pending()` at `:119-128` returns `(ppg_pending, praat_pending)` from `find_measurement(...) is None`; the skip is `:187-190`; and `:235` re-checks `pending` before the Praat block. There is **no `--force` flag** — the CLI is `manifest`, `--slice-index`, `--slice-count`, `--batch-size`, `--device`, `--log-dir`, `--config`. So re-running this driver after Tasks 4 and 5 changes nothing on all 62,547 stores, and would look like success.

- [ ] **Step 1: Write the failing test**

```python
class TestForceReDerives:
    """After a stream switch the stored measurements are stale, and skipping them looks like success."""

    def test_force_re_derives_a_store_that_already_holds_both(self, tmp_path: Path) -> None:
        """Without --force the driver skips, so a corpus-wide re-derivation silently does nothing."""
        run_root = _seeded_run_with_both_measurements(tmp_path)
        summary = run_slice(_manifest_for(run_root), slice_index=0, slice_count=1, force=True, **_defaults())
        assert summary["counts"].get("skipped", 0) == 0, "force must not skip a complete store"
        assert summary["counts"].get("ok", 0) == 1
```

Build `_seeded_run_with_both_measurements` from the fixtures already in this test module — read them first and reuse, since they know how to make a store the driver will open.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/tests/scripts/extend_ppg_praat_test.py::TestForceReDerives -v`
Expected: FAIL — `run_slice` has no `force` parameter.

- [ ] **Step 3: Add the flag and thread it through**

Add to `build_parser` beside `--config`:

```python
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-derive both measurements even where the store already holds them",
    )
```

Thread `force` into `run_slice` and change the skip at `:187-190` to:

```python
        ppg_pending, praat_pending = pending(store)
        if force:
            ppg_pending = praat_pending = True
        if not ppg_pending and not praat_pending:
```

Apply the same override at the `:235` re-check. Pass `args.force` at the `main` call site, and correct the docstring at `:26-27` to say the skip is the default and `--force` overrides it.

**Do not** make `--force` write a second measurement beside the first. The store is append-only and a second live `praat_features` would assert two readings of the same thing — the driver must supersede, which is what `senselab.audio.workflows.triage.extend.supersede` exists for. Read how `extend_withdraw_clips.py` uses it and follow that pattern.

- [ ] **Step 4: Run and confirm**

Run: `uv run pytest src/tests/scripts/extend_ppg_praat_test.py -v`
Expected: PASS, and the existing skip tests still pass without `--force`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format scripts/extend_ppg_praat.py src/tests/scripts/extend_ppg_praat_test.py
uv run ruff check scripts/extend_ppg_praat.py src/tests/scripts/extend_ppg_praat_test.py
uv run mypy scripts/extend_ppg_praat.py
git add -A && git commit -m "feat(extend): --force re-derives a store the stream switch made stale"
```

---

## Task 7: The spec records what landed, and what the corpus now owes

**Files:**
- Modify: `specs/20260817-triage-workflow-dag/praat-instrument-audit.md`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: nothing in code.

- [ ] **Step 1: Mark steps 1, 1b and 2 landed**

Record for each: what changed, the new convention and that it was not fitted, and what remains. Steps 0, 3, 4 and 5 stay open and must still read as open.

- [ ] **Step 2: Record the two corpus passes this plan creates but does not run**

Tasks 4 and 5 make every `praat_features` and `ppg_posteriorgram` measurement in all 62,547 stores stale. Two consequences to write down, because neither is in the code:

- `extend_ppg_praat.py --force` over the corpus re-derives both. That is a GPU-bearing pass for the PPG; size it against the original `ppg_20260911` run rather than guessing.
- The PPG switch changes `ppg.segment_rate_per_s` and therefore DDK routing. **The design requires a before-and-after routing count** — record that it is owed, and that until it is run, no DDK routing figure in any document describes the shipped pipeline.

- [ ] **Step 3: Run the full suite and commit**

```bash
uv run pytest src/tests/audio/tasks src/tests/audio/workflows/triage src/tests/scripts -q
git add -A && git commit -m "docs(audit): steps 1, 1b and 2 landed; the corpus re-derivation is owed"
```

---

## Self-Review

**Spec coverage.** Step 2 → Tasks 1, 2, 3. Step 1 → Task 4. Step 1b → Task 5. The re-run path → Task 6. The spec update → Task 7. Every testing requirement in the brief maps to a step: the 170 Hz discontinuity (Task 1 Step 3), the sub-60 Hz source (Task 1 Step 4), the 420 Hz source (Task 1 Step 4), crash versus absence (Task 2 Step 1), `signal="plain"` on both measurements (Tasks 4 and 5), zero `enhanced` resolve sites (Task 5 Step 5), and `--force` re-deriving (Task 6 Step 1).

**Placeholders.** None. Every code step carries the actual test or the actual replacement text. Three steps name a fixture to read and reuse rather than quoting it — Task 1 Step 1 (`_buzz`), Task 5 Step 1 (the preprocess run helper), Task 6 Step 1 (`_seeded_run_with_both_measurements`) — because duplicating a fixture this suite already has would be the wrong instruction.

**Type consistency.** `extract_pitch_values` returns three `float`-valued keys on every path including the `except`. `derive_f0_range` keeps `tuple[float, float]` and raises one of two typed errors. `run_slice` gains `force: bool`.

**Known gap, deliberate.** Task 6's supersession pattern is named rather than written out, because the correct form depends on how `extend_ppg_praat` structures its writes and the implementer must read `extend_withdraw_clips.py` to match it. Every other step is literal.
