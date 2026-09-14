# F0 Range and Measurement Streams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the right signal with a range derived from it — replace the binary sex-typed F0 range with a per-recording narrowing, and move the Praat scalars and the phonetic posteriorgram off the speech-enhanced stream onto the unprocessed one.

**Architecture:** Three independent defects share one consequence. `extract_pitch_values` picks one of two hardcoded (floor, ceiling) pairs from a threshold on trimmed mean F0, so every measure downstream of `derive_f0_range` carries a step discontinuity at a sex-typed boundary. Separately, `preprocess.py` hands the FRCRN-enhanced stream to both the Praat extractor and the PPG worker, so forty acoustic scalars and one routing gate are computed on a denoiser's output. The fix is one function replacement and two stream changes, plus a re-run path in the extend driver that currently skips every store already holding the measurements.

**Tech Stack:** Python 3.12, `uv`, pytest, numpy, parselmouth (Praat bindings), `ProvStore` (append-only W3C PROV-shaped provenance), Slurm array jobs on ORCD for corpus passes.

**Spec:** `specs/20260817-triage-workflow-dag/praat-instrument-audit.md` — findings 0, 1 and 8, and remediation steps 1, 1b and 2. Branch consumers: `specs/20260817-triage-workflow-dag/branch-voice.md` (V1, V4), `branch-ddk.md` (D1, D2), `branch-quality.md` (Q2). The contract is `specs/20260913-branch-contract-and-hints/design.md`.

## Global Constraints

- **No threshold may be fitted against this corpus.** Its only labels are declared task names, so fitting against them fits the declaration. Every new value must be parameter-free, or a declared convention with its reasoning recorded in `specs/`, or marked owed. The six kinds of owed are in `specs/20260817-triage-workflow-dag/branch-listening-sample.md`.
- Every stream is resampled to **16 kHz mono** before any measurement (`resample.target_hz: 16000`, `src/senselab/audio/workflows/triage/data/config/default.yaml:19`).
- **Pre-alpha: rename and replace outright.** No parallel fields, no aliases, no deprecation shims. This governs Task 5's `enhanced_id` parameter rename.
- **Rationale goes in `specs/`, never in code comments or docstrings.**
- Google-style docstrings, line length 120, full type hints, `from __future__ import annotations`.
- All Python through `uv run`. **Never `pytest -n auto`** — run the directory you changed.
- `uv run ruff format`, `uv run ruff check`, `uv run mypy` clean on every file touched.
- `CACHE_SCHEMA_VERSION` is **not** the invalidation lever here. It lives in `src/senselab/utils/tasks/cached_inference.py` and belongs to `audio_analysis`; `grep -r 'cached_inference\|cache_dir' src/senselab/audio/workflows/triage/` returns nothing. Re-derivation is an extend driver, which is why Task 7 exists.

## Scope

**In scope:** audit step 2 (replace `derive_f0_range`, and make a crash distinguishable from an absence), step 1 (Praat scalars onto `plain`), step 1b (PPG onto `plain`), and the extend driver's re-run path.

**Explicitly out of scope, each getting its own plan:** the CPPS reimplementation (step 4); jitter and shimmer from the `PeriodMark` sequence (step 3); withholding the scalars already written into 62,547 stores (step 0); instrument coverage per instrument (step 5); **step 2b — tracking F0 on the same signal the range was derived on** (`preprocess.py:941` derives on `plain`, `:951` tracks on `sharp`); the owed widening of `voice.f0_search_range_hz`'s 600 Hz ceiling against the CPPS band's 700 Hz; every branch capability; the SCREEN merge; and the contract's nine pieces.

**One thing this plan deliberately does not resolve.** `branch-voice.md` V3 records that narrowing buys octave-error robustness on stationary material and is wrong on a glide, where the derived ceiling is set by how high the speaker went, making V3's "did F0 reach the derived limit" flag partly circular. This plan implements the narrowing and leaves that open.

---

## File Structure

| file | responsibility after this plan |
| --- | --- |
| `src/senselab/audio/tasks/features_extraction/praat_parselmouth.py` | `extract_pitch_values` narrows per recording from robust log-Hz percentiles, and **reports a failed analysis as data** rather than as an indistinguishable NaN. Modify `:358-446`. |
| `src/senselab/audio/tasks/phonation/api.py` | `derive_f0_range` raises `F0RangeFailed` on a reported failure and `F0RangeUnavailable` on a genuine absence. Modify `:41-69`. |
| `src/senselab/audio/workflows/triage/nodes/preprocess.py` | `ppg_input`, `write_ppg_posteriorgram` and the Praat block read `plain`. Modify `:742-763`, `:766-826`, `:839-844`, `:859-895`. |
| `src/tests/audio/tasks/features_extraction_test.py` | the narrowing's behaviour and the failure signal. |
| `src/tests/audio/tasks/phonation_test.py` | `derive_f0_range`'s two typed outcomes. **Rewrite `TestDeriveF0Range` `:109-131`.** |
| `src/tests/audio/workflows/triage/nodes/preprocess_test.py` | both measurements record `signal="plain"`. **Six stale assertions plus one whole test — enumerated in Task 6.** |
| `src/tests/scripts/extend_reprocessed_outputs_test.py` | a reported failure is a failed row, not an escaped exception. |
| `scripts/extend_ppg_praat.py` | gains `--force`, which supersedes rather than appends. Modify `:26-27`, `:94-115`, `:187-190`, `:197`, `:219`, `:235-238`. |
| `src/tests/scripts/extend_ppg_praat_test.py` | `_seed_run` seeds `plain`; `--force` re-derives and leaves exactly one live measurement. |
| `specs/20260817-triage-workflow-dag/praat-instrument-audit.md` | steps 1, 1b and 2 landed; the corpus passes and the noise-robustness statement recorded as owed. |

### Two existing tests pin the defects

- `phonation_test.py:112-119` asserts `low == (60.0, 250.0)` and `high == (100.0, 500.0)` under the docstring *"The narrowing is what the standardization method does; a fixed corpus range cannot."* It asserts the bin while describing what the bin is not.
- `preprocess_test.py:2238` is named `test_both_read_the_enhanced_stream_back_out_of_the_store`, and its class docstring at `:2172` opens *"Both run on ``enhanced``"*.

### What does not need changing, and why

`routing_analysis/features.py` finds the PPG measurement **by name** and resolves its sidecar from the measurement's `path` attribute — it applies no `signal` filter. So Task 5 changes the DDK gate feature's **value**, not its availability. This also means `live_evidence_test.py:216` and `routing_analysis_test.py:949`/`:959`, which hardcode `"signal": "enhanced"` in PPG fixtures, will **keep passing**. Update them for accuracy, but do not expect a failure to prompt you.

---

## Task 1: The narrowing replaces the bin, and a failed analysis says so

**Files:**
- Modify: `src/senselab/audio/tasks/features_extraction/praat_parselmouth.py:358-446`
- Test: `src/tests/audio/tasks/features_extraction_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `extract_pitch_values(snd, search_floor_hz: float = 50.0, search_ceiling_hz: float = 600.0) -> Dict[str, float]` returning **five** `float` keys on all **three** return paths: `pitch_floor`, `pitch_ceiling` (both `np.nan` when no range could be derived), `pitch_frames` (voiced frames the range rests on, `0.0` when none), `pitch_failed` (`1.0` when the analysis itself raised), and `pitch_range_fell_back` (`1.0` when the contour was pinned near the search floor and the wide range was used instead). Task 2 reads `pitch_failed` and the two range values; Task 8 carries all five onto the measurement.

**The narrowing must not be able to exclude the speaker's F0, and a single pass can.** Measured on a synthesised **120 Hz voice with a 60 Hz hum at ~−22 dB**: the wide search locks to the subharmonic across the whole contour and a one-pass narrowing returns **`[50.0, 90.0]`** — a range the speaker's F0 never enters. Every measure downstream is then computed over a range that excludes the voice.

**The bin was accidentally robust here**: a 60 Hz trimmed mean selects `(60, 250)`, which still contains 120 Hz. So a one-pass narrowing would be a **regression** on this configuration — and 60 Hz mains against a ~120 Hz male voice, or 50 Hz against ~100 Hz, is a common clinical recording. It interacts adversely with Task 5: FRCRN suppresses stationary low-frequency noise and `plain` does not, so moving to `plain` makes the hum case *more* frequent in the same pass that introduces the vulnerability.

**An earlier draft of this plan proposed a second-pass median comparison, and it cannot fire.** Re-running at the narrowed range and widening back when the two medians differ by an octave is unreachable by construction: the narrowed range is `[p5/1.5, p95·1.5]` with the first median near its centre, so **the second median can deviate by at most a factor of 1.5, which is 0.585 octave.** A `>= 1.0` octave test never triggers. Measured, with Task 1 implemented: the hum case returns `pitch_range_fell_back = 0.0` and the range `[50.0, 90.0]` unchanged, and bimodal 120/240 gives `[80, 360]` with m₂ = 120.6, bimodal 90/400 gives `[60, 600]` with m₂ = 399.7. Any threshold small enough to catch the hum case would be a fitted operating point, so "introduces no fitted value" and "closes the regression" cannot both hold with that mechanism.

**The rule that works applies the octave to the right comparison.** The hum signature is not that a median moved — it is that **the whole contour is pinned within an octave of the search floor**. So: after the wide pass, if **p95 is below twice the search floor**, the contour is pinned at the bottom of the search range, the narrowing is not trustworthy, and the **wide search range is used instead**. One octave above the search floor is a natural unit applied where it can actually fire.

Measured on every case, with the wide pass at `to_pitch_ac(0.005, 50.0, pitch_ceiling=600.0)`:

| source | p95 | fires? | resulting range |
| --- | --- | --- | --- |
| normal 120 Hz buzz | 120.00 | no | `[80.0, 180.0]` |
| **120 Hz + 60 Hz hum at −22 dB** | **60.00** | **yes** | **`[50.0, 600.0]` — contains 120** |
| 120 Hz at 0 dB broadband SNR | 122.22 | no | `[78.6, 183.3]` — contains 120 |
| 55 Hz fry | 55.01 | yes | `[50.0, 600.0]` — wide, and safe |
| 45 Hz fry | — | 0 frames | absence, unchanged |
| glide 100→400 | 366.33 | no | `[72.8, 549.5]` |
| child 300 Hz | 300.00 | no | `[200.0, 450.0]` |
| 420 Hz | 420.00 | no | `[280.0, 600.0]` |

The 55 Hz fry falling back to the wide range is the right outcome: contamination pushing the range **wider** is the safe direction, and a wide range for a fry voice costs octave-error robustness, not the voice. Verified separately that `extract_jitter` and `extract_shimmer` return finite values at floor 50, so the 55 Hz test in Step 3 still passes.

**Why `pitch_failed` exists, and why this task and not Task 2 carries it.** The current `except Exception` at `:439-445` returns `{pitch_floor: nan, pitch_ceiling: nan}` — byte-identical to the legitimate no-pitch return at `:426`. A caller cannot tell a parselmouth crash from a silent recording. The audit's step 2 (`praat-instrument-audit.md:158-159`) requires that distinction, and **it can only be made here**, because this is the only frame that sees the exception. A `try/except` in `derive_f0_range` around this call would be unreachable in production: the crash never propagates past this `except`. We keep the swallow — other callers depend on the batch extractor not aborting — and add the signal as data.

- [ ] **Step 1: Write the failing test — the discontinuity, which is what actually fails today**

Add to `src/tests/audio/tasks/features_extraction_test.py`. Copy `_buzz` from
`src/tests/audio/tasks/phonation_test.py:24-28` if this module has no equivalent; check first.

**Imports this module does not currently have** and every test below needs: `numpy as np`, `parselmouth`,
`pytest`, `Audio`, and `derive_f0_range` / `extract_jitter` / `extract_shimmer`. `phonation_test.py` needs
`F0RangeFailed` added to its existing `senselab.audio.tasks.phonation` import. `extend_reprocessed_outputs_test.py`
imports only three names from `extend` and needs `attempt_derivation`, plus `F0RangeFailed` and
`F0RangeUnavailable` from `senselab.audio.tasks.phonation`. Add them as you go — a missing import reads as a
collection error, not as a failing assertion, and wastes a cycle.

```python
class TestPitchRangeNarrowing:
    """The range is this recording's own, narrowed off a wide search — not one of two presets."""

    def test_no_discontinuity_at_the_retired_170_hz_boundary(self) -> None:
        """The retired rule stepped floor/ceiling from 60/250 to 100/500 across mean F0 = 170 Hz."""
        below = extract_pitch_values(_buzz(165.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        above = extract_pitch_values(_buzz(175.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert below["pitch_ceiling"] == pytest.approx(above["pitch_ceiling"], rel=0.15)
        assert below["pitch_floor"] == pytest.approx(above["pitch_floor"], rel=0.15)
```

This is deliberately first. The monotonicity test in Step 3 **passes under the bin** — ceilings for `(90, 130, 180, 260, 380)` are `[250, 250, 500, 500, 500]`, which is sorted with `250 < 500` — so starting there would begin the TDD loop green.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/tests/audio/tasks/features_extraction_test.py::TestPitchRangeNarrowing -v`
Expected: FAIL. 165 Hz returns ceiling 250.0 and 175 Hz returns 500.0 — a factor of two, far outside `rel=0.15`.

- [ ] **Step 3: Add the remaining behaviour tests**

```python
    def test_the_derived_range_rises_monotonically_with_source_f0(self) -> None:
        """A preset bin is a step function of F0; a narrowing is monotone in it."""
        ceilings = [
            extract_pitch_values(_buzz(f0), search_floor_hz=50.0, search_ceiling_hz=600.0)["pitch_ceiling"]
            for f0 in (90.0, 130.0, 180.0, 260.0, 380.0)
        ]
        assert ceilings == sorted(ceilings), f"ceiling must not decrease as F0 rises: {ceilings}"
        assert len(set(ceilings)) == len(ceilings), f"each voice gets its own ceiling, not a bin's: {ceilings}"

    def test_a_55_hz_source_yields_finite_perturbation(self) -> None:
        """The retired 60 Hz floor placed zero pulses here, so jitter and shimmer were NaN.

        Asserting the outcome, not the floor: ``pitch_floor < 60`` is satisfied by ``max(50.0, …)``
        for any voice whose p5 is under 90 Hz, so it would pass without this source being tracked.
        """
        audio = _buzz(55.0)
        floor, ceiling = derive_f0_range(audio, search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert np.isfinite(extract_jitter(audio, floor=floor, ceiling=ceiling)["local_jitter"])
        assert np.isfinite(extract_shimmer(audio, floor=floor, ceiling=ceiling)["local_shimmer"])

    def test_a_45_hz_source_records_where_the_exclusion_moved_to(self) -> None:
        """The exclusion moved from 60 Hz to the search floor; it did not go away.

        45 Hz places no pitch at a 50 Hz search floor, so the range is an absence and VOICE reads
        'no phonation found' for a voice that is plainly phonating. The binding constraint is now
        ``voice.f0_search_range_hz[0]``, and this test is the record of that.
        """
        derived = extract_pitch_values(_buzz(45.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_frames"] == 0.0
        assert derived["pitch_failed"] == 0.0, "an out-of-search-range voice is an absence, not a crash"

    def test_a_420_hz_source_reaches_the_search_ceiling_rather_than_the_retired_250(self) -> None:
        """Fixed to about 400 Hz and then saturating: 380 Hz gives 570, 420 Hz gives the 600 clamp."""
        assert extract_pitch_values(_buzz(380.0), search_floor_hz=50.0, search_ceiling_hz=600.0)[
            "pitch_ceiling"
        ] == pytest.approx(570.0, rel=0.05)
        assert extract_pitch_values(_buzz(420.0), search_floor_hz=50.0, search_ceiling_hz=600.0)[
            "pitch_ceiling"
        ] == pytest.approx(600.0), "the search ceiling binds above ~400 Hz, not the retired 250"

    def test_a_hum_does_not_capture_the_range_away_from_the_voice(self) -> None:
        """A one-pass narrowing returned [50, 90] for this source — a range the voice never enters.

        The retired bin was accidentally robust here, so this is the one case where the replacement
        would have been worse than what it replaced.
        """
        voice = _buzz(120.0, seconds=2.0).waveform.numpy()[0]
        t = np.arange(voice.size) / 16000
        hum = (10 ** (-22 / 20)) * np.sin(2 * np.pi * 60.0 * t)
        audio = Audio(waveform=(voice + hum).astype(np.float32)[None, :], sampling_rate=16000)

        derived = extract_pitch_values(audio, search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_floor"] <= 120.0 <= derived["pitch_ceiling"], (
            f"the speaker's F0 must lie inside its own derived range, got "
            f"[{derived['pitch_floor']}, {derived['pitch_ceiling']}]"
        )
        assert derived["pitch_range_fell_back"] == 1.0, "the pinned-contour fallback is what caught it"

    def test_a_noisy_source_does_not_capture_the_range(self) -> None:
        """Measured at 0 dB broadband SNR: 389 frames, median 120.06, range [78.6, 183.3]."""
        voice = _buzz(120.0, seconds=2.0).waveform.numpy()[0]
        rng = np.random.default_rng(0)
        noisy = voice + rng.standard_normal(voice.size).astype(np.float32) * float(np.sqrt((voice**2).mean()))
        audio = Audio(waveform=noisy.astype(np.float32)[None, :], sampling_rate=16000)

        derived = extract_pitch_values(audio, search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_frames"] > 0.0, "measured 389 frames; a guard here would hide a real change"
        assert derived["pitch_floor"] <= 120.0 <= derived["pitch_ceiling"]
        assert derived["pitch_range_fell_back"] == 0.0, "p95 = 122 clears twice the floor, so no fallback"

    def test_a_glide_is_bracketed_rather_than_clipped(self) -> None:
        """An exponential 100 to 400 Hz sweep measured [72.8, 549.5] against produced extremes 102/392."""
        t = np.arange(2 * 16000) / 16000
        f0 = 100.0 * (4.0 ** (t / t[-1]))
        wave = np.sin(2 * np.pi * np.cumsum(f0) / 16000).astype(np.float32)
        audio = Audio(waveform=wave[None, :], sampling_rate=16000)

        derived = extract_pitch_values(audio, search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_floor"] < 100.0 and derived["pitch_ceiling"] > 400.0

    def test_the_range_reports_the_frames_it_rests_on(self) -> None:
        """A range derived from four voiced frames is not the same claim as one from four hundred."""
        derived = extract_pitch_values(_buzz(150.0, seconds=2.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_frames"] > 0.0
        assert derived["pitch_failed"] == 0.0

    def test_silence_is_an_absence_and_not_a_failure(self) -> None:
        """No pitch placed is a real answer about the recording; the analysis did not fail."""
        silence = Audio(waveform=np.zeros((1, 16000), dtype=np.float32), sampling_rate=16000)
        derived = extract_pitch_values(silence, search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert np.isnan(derived["pitch_floor"]) and np.isnan(derived["pitch_ceiling"])
        assert derived["pitch_frames"] == 0.0
        assert derived["pitch_failed"] == 0.0, "silence is an absence, not a crash"

    def test_a_failed_analysis_is_reported_rather_than_returned_as_an_absence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The crash return was byte-identical to the no-pitch return; a caller could not tell them apart."""

        def _boom(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("parselmouth exploded")

        monkeypatch.setattr(parselmouth.Sound, "to_pitch_ac", _boom, raising=False)
        derived = extract_pitch_values(_buzz(150.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert derived["pitch_failed"] == 1.0
        assert np.isnan(derived["pitch_floor"])
```

The last test patches the parselmouth call the function makes, not the function itself — so it exercises the real `except` path. Check the exact call at `:414-418` and patch whichever method it uses.

- [ ] **Step 4: Run all six and record which fail and why**

Run: `uv run pytest src/tests/audio/tasks/features_extraction_test.py::TestPitchRangeNarrowing -v`
Expected: the discontinuity, monotonicity-uniqueness, 55 Hz and 420 Hz cases FAIL on the bin; the two `pitch_frames`/`pitch_failed` cases FAIL with `KeyError`.

- [ ] **Step 5: Replace the body**

Replace **from `pitch_values = pitch_values[pitch_values != 0]` (`:420`) through `return {"pitch_floor": pitch_floor, "pitch_ceiling": pitch_ceiling}` (`:438`) inclusive** — and nothing after it, so the existing `except` block at `:439` stays where it is:

```python
        pitch_values = pitch_values[pitch_values != 0]
        if pitch_values.size == 0:
            return _no_pitch_range()

        low, high = np.percentile(pitch_values, [PITCH_RANGE_LOW_PERCENTILE, PITCH_RANGE_HIGH_PERCENTILE])
        if float(high) < PITCH_PINNED_OCTAVES * float(search_floor_hz):
            floor, ceiling, fell_back = float(search_floor_hz), float(search_ceiling_hz), 1.0
        else:
            floor = max(float(search_floor_hz), float(low) / PITCH_RANGE_MARGIN)
            ceiling = min(float(search_ceiling_hz), float(high) * PITCH_RANGE_MARGIN)
            fell_back = 0.0

        return {
            "pitch_floor": floor,
            "pitch_ceiling": ceiling,
            "pitch_frames": float(pitch_values.size),
            "pitch_failed": 0.0,
            "pitch_range_fell_back": fell_back,
        }
```

**There is no second pitch pass, and the earlier draft's one was a hard bug as well as an unreachable one.** It called `get_sound(snd)`, but by that point `snd` is already a `parselmouth.Sound` and `get_sound` accepts only `Path` or `Audio` (`:50`, isinstance chain `:70-77`) — so its local is never bound and it raises `RuntimeError: cannot access local variable`, which the outer `except Exception` swallows into `_no_pitch_range(failed=1.0)`. Every recording would then return a NaN range and `derive_f0_range` would raise `F0RangeFailed` **corpus-wide**. mypy cannot catch it — `import parselmouth  # type: ignore` makes the object `Any` — and `features_extraction_test.py:234-242` cannot either, since it asserts only `isinstance(..., float)` and `np.nan` satisfies that. The fallback rule needs no second pass at all, which removes the whole class of error.

Add a module-level helper so all four absence and failure paths stay identical:

```python
def _no_pitch_range(*, failed: float = 0.0) -> Dict[str, float]:
    """The five-key shape every ``extract_pitch_values`` path returns when no range was derived."""
    return {
        "pitch_floor": np.nan,
        "pitch_ceiling": np.nan,
        "pitch_frames": 0.0,
        "pitch_failed": failed,
        "pitch_range_fell_back": 0.0,
    }
```

The `except` block's return at `:445` becomes `return _no_pitch_range(failed=1.0)`. **There are then exactly three return paths — the empty-contour guard, the main dict, and the `except` — and all three carry all five keys.** Task 2 and Task 8 depend on that. Note the replacement in this step **deletes** the old non-finite `mean_pitch` guard at `:426` along with the `mean_pitch` computation it guarded, so there is no third guard to convert; do not look for one.

Add beside the module's other constants:

```python
PITCH_RANGE_LOW_PERCENTILE = 5.0
PITCH_RANGE_HIGH_PERCENTILE = 95.0
PITCH_RANGE_MARGIN = 1.5
PITCH_PINNED_OCTAVES = 2.0
```

**There is no trim, and that is deliberate.** An earlier draft kept a log-Hz MAD trim before the percentiles. Two reasons it is gone. It does not do the job it was added for — an octave is 1.0 in log₂ and a bimodal contour widens the MAD, so a 2-MAD window keeps *both* modes and the 95th percentile still lands in the doubled one; the refinement above is what actually catches that. And it corrupts `pitch_frames`: measured on a clean 110 Hz buzz, the trim discarded **36 of 188** voiced frames, so the percentiles were of the post-trim set (about p7/p93) and `pitch_frames` was not the voiced-frame count its consumers would read it as. The 5th/95th percentiles already trim 10%.

Check the exact first-pass call at `:414-418` and mirror its argument style in the refinement call — if the first pass passes a time step, the second must too, or the two medians are not comparable.

- [ ] **Step 6: Run and confirm all six pass**

Run: `uv run pytest src/tests/audio/tasks/features_extraction_test.py::TestPitchRangeNarrowing -v`
Expected: PASS, all six.

- [ ] **Step 7: Update the docstrings this changes**

`extract_pitch_values`' `Returns:` block (`:373-380`) documents two keys and must document four. Its `Examples:` (`:402`) prints the old pair. And `src/senselab/audio/tasks/features_extraction/api.py:375-376` shows the same pair. Fix all three — each is a doctest-shaped example that now misdescribes the return.

- [ ] **Step 8: Record the convention in the spec, without a false citation**

Add to `praat-instrument-audit.md` under step 2:

- **Cite Hirst for the two-pass structure, and nothing else.** Hirst's rule is a first pass at **50–700 Hz**, then `floor = 0.75 × q1` and `ceiling = 1.5 × q3` — **the quartiles, with a deliberately asymmetric pair**. De Looze & Hirst's variant uses q15/q65. This plan's 5th/95th with a symmetric ±1.5 is none of those, and its first pass is 50–600. Write that the **structure** is Hirst's two-pass and the **parameterisation is a senselab convention**. A claim of "Hirst as conventionally parameterised" is checkable in five minutes and would be the second wrong derivation in this document family.
- **Say why the numbers are deliberately wider.** ±1.5 on p5/p95 is ±7.02 semitones and yields a range strictly *wider* than Hirst's at both ends. That is the right direction for a corpus enriched for pathological voices, where a narrow range is the failure that matters. Record the reason, not just the values.
- **The pinned-contour fallback is this project's own, and must be cited as such.** De Looze & Hirst's variant iterates the quantile narrowing to convergence; **it has no widen-back or fallback step**, so citing them for this would be the second wrong derivation in this family — which this plan twice warns against. Record the rule, the measured table above, and that one octave above the search floor is a natural unit rather than a fitted cut. An earlier draft proposed a second-pass median test and it is provably unreachable (max deviation 0.585 octave against a 1.0 test); record that too, so nobody re-proposes it.
- **There is no trim, and say so.** An earlier draft carried a log-Hz MAD trim; it is removed because it does not remove an octave-split mode (an octave is 1.0 in log₂ and a bimodal contour widens the MAD, so both modes survive) and because it corrupted `pitch_frames` — measured, 36 of 188 voiced frames discarded on a clean 110 Hz buzz, making the percentiles p7/p93 of the original set. Octave robustness comes from the refinement, not from a trim.
- The range ratio widens: the bin always gave 4.17 or 5.0, the narrowing measured **7.5** on a glide and can exceed 11. `voice.f0_range_ratio_max` is null so nothing fires, but `voice.py:77-80` **raises** rather than flags once it is set. Record it.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff format src/senselab/audio/tasks/features_extraction/ src/tests/audio/tasks/features_extraction_test.py
uv run ruff check src/senselab/audio/tasks/features_extraction/ src/tests/audio/tasks/features_extraction_test.py
uv run mypy src/senselab/audio/tasks/features_extraction/praat_parselmouth.py
git add -A && git commit -m "fix(praat): derive the F0 range per recording, and report a failed analysis"
```

---

## Task 2: `derive_f0_range` raises two different errors

**Files:**
- Modify: `src/senselab/audio/tasks/phonation/api.py:41-69`, `src/senselab/audio/tasks/phonation/__init__.py`
- Test: `src/tests/audio/tasks/phonation_test.py`

**Interfaces:**
- Consumes: `pitch_failed` from Task 1.
- Produces: `F0RangeFailed(ValueError)` exported from `senselab.audio.tasks.phonation`. `derive_f0_range`'s signature and return type are unchanged.

**`F0RangeFailed` subclasses `ValueError`, and that choice is load-bearing at both consumers.** Verified:

- `extend.py:120-126` lists `F0RangeUnavailable` in `UNAVAILABLE`; `attempt_derivation:160-165` then catches `UNAVAILABLE` as a non-failure and `(OSError, ValueError, LookupError)` as a **failed row**. A `ValueError` subclass therefore records the failure and lets the array task continue. A `RuntimeError` subclass would escape both handlers and **kill the task** — which is the defect `extend_reprocessed_outputs_test.py:504-519` exists to prevent, after 613 rows of the last corpus pass did exactly that.
- `preprocess.py:2619-2626`: `except (ValueError, LookupError)` records a cascading absence; `except Exception` appends to `hard_failures` and the node then **raises** at `:2631`. A `ValueError` subclass keeps one crashed recording from aborting a node whose other blocks still need to run.

So `F0RangeFailed` must **not** be added to `UNAVAILABLE` — it is a failure, not an absence, and `attempt_derivation`'s `ValueError` branch is where it belongs.

- [ ] **Step 1: Write the failing test**

Add to `phonation_test.py`. Do not duplicate the silence case — `:120-124` already covers it.

```python
class TestDeriveF0RangeSeparatesFailureFromAbsence:
    """A crashed analysis and a recording with no pitch are different findings."""

    def test_a_reported_failure_raises_f0_range_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """extract_pitch_values reports the crash as data; derive_f0_range must not call it an absence."""
        monkeypatch.setattr(
            "senselab.audio.tasks.phonation.api.extract_pitch_values",
            lambda *_a, **_k: {
                "pitch_floor": np.nan,
                "pitch_ceiling": np.nan,
                "pitch_frames": 0.0,
                "pitch_failed": 1.0,
            },
        )
        with pytest.raises(F0RangeFailed):
            derive_f0_range(_buzz(150.0), search_floor_hz=50.0, search_ceiling_hz=600.0)

    def test_f0_range_failed_is_a_value_error(self) -> None:
        """extend.attempt_derivation records a ValueError as a failed row; a RuntimeError escapes it."""
        assert issubclass(F0RangeFailed, ValueError)
        assert not issubclass(F0RangeFailed, RuntimeError)
```

Both assertions matter: the first is what `attempt_derivation` dispatches on, the second is what a future edit to `RuntimeError` would break.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py::TestDeriveF0RangeSeparatesFailureFromAbsence -v`
Expected: FAIL on import — `F0RangeFailed` does not exist.

- [ ] **Step 3: Add the error and read the flag**

Beside `F0RangeUnavailable` at `:41-42`:

```python
class F0RangeFailed(ValueError):
    """The pitch analysis itself failed. A ValueError so ``extend.attempt_derivation`` records a
    failed row rather than letting it escape, and so PREPROCESS records an absence rather than
    aborting the node."""
```

That docstring states a *what*, not a rationale — the reasoning goes in the spec per the global constraints. Trim it to one sentence if it reads as rationale.

In `derive_f0_range`, replace `:63-68`:

```python
    values = extract_pitch_values(audio, search_floor_hz=search_floor_hz, search_ceiling_hz=search_ceiling_hz)
    if values.get("pitch_failed", 0.0):
        raise F0RangeFailed("the pitch analysis failed on this recording")
    floor, ceiling = float(values["pitch_floor"]), float(values["pitch_ceiling"])
    if not np.isfinite(floor) or not np.isfinite(ceiling):
        raise F0RangeUnavailable(
            f"no F0 range could be derived from this recording over [{search_floor_hz}, {search_ceiling_hz}] Hz"
        )
    return floor, ceiling
```

Add `F0RangeFailed` to the `Raises:` block, and export it from `__init__.py` in both the import and `__all__`.

- [ ] **Step 4: Rewrite the test that pinned the bin**

Replace `test_a_low_voice_and_a_high_voice_get_different_ranges` at `:112-119`:

```python
    def test_a_low_voice_and_a_high_voice_get_ranges_that_follow_them(self) -> None:
        """The range follows the recording; it is not one of two presets chosen by a threshold."""
        low = derive_f0_range(_buzz(110.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        high = derive_f0_range(_buzz(230.0), search_floor_hz=50.0, search_ceiling_hz=600.0)
        assert low[0] < 110.0 < low[1], f"110 Hz must sit inside its own derived range, got {low}"
        assert high[0] < 230.0 < high[1], f"230 Hz must sit inside its own derived range, got {high}"
        assert low[1] < high[1], "a higher voice must get a higher ceiling"
```

- [ ] **Step 5: Run the phonation and features suites**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py src/tests/audio/tasks/features_extraction_test.py -q`
Expected: PASS. Any other failure is a caller that depended on the bin's exact output — read it rather than adjusting the assertion.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format src/senselab/audio/tasks/phonation/ src/tests/audio/tasks/phonation_test.py
uv run ruff check src/senselab/audio/tasks/phonation/ src/tests/audio/tasks/phonation_test.py
uv run mypy src/senselab/audio/tasks/phonation/
git add -A && git commit -m "fix(phonation): a failed pitch analysis is a failure, not an absence"
```

---

## Task 3: Both error paths behave correctly at their consumers

**Files:**
- Test: `src/tests/scripts/extend_reprocessed_outputs_test.py`, `src/tests/audio/workflows/triage/nodes/preprocess_test.py`

**Interfaces:**
- Consumes: `F0RangeFailed` from Task 2.
- Produces: nothing. This task exists because Task 2's choice of base class is only correct if asserted.

- [ ] **Step 1: Write the extend-driver test**

Add to `extend_reprocessed_outputs_test.py`, beside the existing typed-absence tests at `:504-519`:

```python
    def test_a_failed_f0_analysis_is_a_failed_row_and_not_an_escape(self) -> None:
        """A RuntimeError would leave attempt_derivation and kill the array task."""
        outcome = attempt_derivation(lambda: (_ for _ in ()).throw(F0RangeFailed("boom")))
        assert outcome.failed is True
        assert "F0RangeFailed" in outcome.detail

    def test_an_absent_f0_range_is_not_a_failed_row(self) -> None:
        """The absence stays in UNAVAILABLE; only the failure is a failure."""
        outcome = attempt_derivation(lambda: (_ for _ in ()).throw(F0RangeUnavailable("none")))
        assert outcome.failed is False
        assert outcome.detail.startswith("absent")
```

Match the module's existing import and helper style before writing these — read `:504-519` first.

- [ ] **Step 2: Write the PREPROCESS test**

```python
    def test_a_failed_f0_analysis_is_an_absence_and_does_not_abort_the_node(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A RuntimeError would reach hard_failures and raise, taking every other block with it."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch)
        monkeypatch.setattr(
            preprocess_module, "derive_f0_range", lambda *_a, **_k: (_ for _ in ()).throw(F0RangeFailed("boom"))
        )
        result = preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        assert "phonation_tracks" in result.absent

```

Confirm the block's registered name against the `blocks` list at `preprocess.py:2596-2616` — use whatever name that list gives the phonation-tracks block, not a guess. And follow `preprocess_test.py:1090`, which already templates the monkeypatch-a-block-into-raising pattern; do not invent a second shape for it.

- [ ] **Step 3: Run both and confirm they pass**

Run: `uv run pytest src/tests/scripts/extend_reprocessed_outputs_test.py src/tests/audio/workflows/triage/nodes/preprocess_test.py -q`
Expected: PASS. If the PREPROCESS test raises `RuntimeError: PREPROCESS: 1 block(s) failed unexpectedly`, `F0RangeFailed` is not a `ValueError` — go back to Task 2 Step 3.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "test(triage): a failed F0 analysis is a failed row, and never a node abort"
```

---

## Task 4: The narrowing's existing consumers still pass

**Files:**
- Test: `src/tests/audio/workflows/triage/nodes/preprocess_test.py`, `voice_test.py`

**Interfaces:** consumes Tasks 1–3; produces nothing.

`derive_f0_range` has two production callers: `preprocess.py:941` (feeding `f0_track` and `formant_track`) and `voice.py:71-73` (feeding `hnr_track` and `period_marks`). Both now receive a continuously varying range, and Praat's window is `periods_per_window / floor`, so a changed floor changes frame counts.

- [ ] **Step 1: Run both suites**

Run: `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py src/tests/audio/workflows/triage/nodes/voice_test.py -q`
Expected: PASS. A failure asserting a track length or window followed from the bin's fixed floor — fix the assertion to the new derivation, not the other way round.

- [ ] **Step 2: Commit if anything needed changing**

```bash
git add -A && git commit -m "test(triage): window lengths follow the derived range, not a bin"
```

---

## Task 5: Both measurements move to `plain`

**Files:**
- Modify: `src/senselab/audio/workflows/triage/nodes/preprocess.py:742-763`, `:766-826`, `:839-844`, `:859-895`
- Modify: `scripts/extend_ppg_praat.py:219`
- Test: `src/tests/audio/workflows/triage/nodes/preprocess_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: both measurements recording `signal="plain"` and `derived_from=(plain_id,)`. `write_ppg_posteriorgram`'s `enhanced_id` keyword becomes **`stream_id`**.

Tasks 5 and 6 are one change split by concern: this task moves the code, the next fixes every test it breaks. The PPG move is the consequential half — it changes `ppg.segment_rate_per_s`, hence **DDK routing corpus-wide**, not merely what a number reads.

- [ ] **Step 1: Move the Praat block**

In `preprocess.py`, in `_praat_features`: `:880` → `plain_id, audio = resolve_stream(store, run_dir, "plain")`; `:882`'s activity tuple → `(plain_id,)`; `:892` → `signal="plain"`; `:895` → `derived_from=(plain_id,)`. The docstring names `enhanced` at `:859`, `:868` and `:874` — all three change.

- [ ] **Step 2: Move the PPG block and rename the parameter**

`ppg_input` (`:742-763`): `:758` → `resolve_stream(store, run_dir, "plain")`, rename the local, and fix the docstring at `:743`, `:749` and `:756`.

`write_ppg_posteriorgram` (`:766-822`): rename the **keyword parameter** `enhanced_id` (`:770`) to `stream_id`, its docstring (`:783`), and its two uses at `:794` and `:821`. Fix the docstring at `:822`.

Then both call sites, which pass it **by name**: `preprocess.py:844` and **`scripts/extend_ppg_praat.py:219`**. Missing the second leaves a `TypeError` the triage suite will not catch. Pre-alpha says rename outright — no alias.

Note `:809` is `signal="enhanced"` inside `write_ppg_posteriorgram`, not `ppg_input`. Change it to `"plain"`.

- [ ] **Step 3: Run and expect failures, not success**

Run: `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -q`
Expected: **FAIL**, in the tests Task 6 enumerates. That is the point of splitting the tasks — do not adjust assertions here.

- [ ] **Step 4: Lint and commit a deliberately red suite**

This commit leaves `preprocess_test.py` failing, on purpose — Task 6 is the fix, and splitting them keeps
the code move reviewable separately from the twelve test edits. Say so in the commit message so a bisect
does not read it as a break.

```bash
uv run ruff format src/senselab/audio/workflows/triage/nodes/preprocess.py scripts/extend_ppg_praat.py
uv run ruff check src/senselab/audio/workflows/triage/nodes/preprocess.py scripts/extend_ppg_praat.py
uv run mypy src/senselab/audio/workflows/triage/nodes/preprocess.py scripts/extend_ppg_praat.py
git add -A && git commit -m "fix(preprocess): measure the Praat scalars and the posteriorgram on plain"
```

---

## Task 6: Every test the move breaks, and what each becomes

**Files:**
- Test: `src/tests/audio/workflows/triage/nodes/preprocess_test.py`, `src/tests/audio/workflows/triage/live_evidence_test.py`, `routing_analysis_test.py`

**Interfaces:** consumes Task 5; produces nothing.

Six stale assertions, two stale names, one test whose whole premise dissolves. Each is listed with what it becomes.

- [ ] **Step 1: Retarget the class that asserts the enhanced stream**

In the class whose docstring is at `:2172`:

| line | now | becomes |
| --- | --- | --- |
| `:2172` | docstring "Both run on ``enhanced``…" | "Both run on ``plain``…" |
| `:2210` | `test_praats_scalars_are_attributes_and_the_stream_is_the_enhanced_one` | `..._the_unprocessed_one`. **A name, so nothing will flag it** — the assertion inside it is a separate row below |
| `:2193` | `assert attrs["signal"] == "enhanced"` | `== "plain"` |
| `:2204` | `resolve_stream(store, tmp_path, "enhanced")` for the first measurement | `"plain"`, and rename the local |
| `:2205` | `derived_from(measurement.id) == [enhanced_id]` | the renamed local |
| `:2226` | `assert attrs["signal"] == "enhanced"` | `== "plain"` |
| `:2235-2236` | `resolve_stream(...)` + `derived_from == [enhanced_id]` | `"plain"`, rename the local |
| `:2238` | `test_both_read_the_enhanced_stream_back_out_of_the_store` | `..._the_plain_stream_...` |

- [ ] **Step 2: Decide what the no-enhanced-stream test becomes**

`test_both_are_absent_when_no_enhanced_stream_was_written` at `:2289-2305` asserts both measurements land in `result.absent` with `"enhanced"` in the reason. That absence came **only** from `resolve_stream(..., "enhanced")` raising. After the move `plain` always exists, both blocks always run, and the test's premise is gone.

**Replace it with the same contract over the stream that can now be missing**, keeping the cascading-absence property it was protecting:

```python
    def test_both_are_absent_when_no_plain_stream_was_written(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing measurement stream is a cascading absence, not a failure of the node."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch)
        monkeypatch.setattr(
            preprocess_module,
            "resolve_stream",
            lambda *_a, **_k: (_ for _ in ()).throw(LookupError("no live stream named 'plain'")),
        )
        result = preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)

        assert PPG_MEASUREMENT in result.absent
        assert PRAAT_MEASUREMENT in result.absent
```

Patching `resolve_stream` is blunt and will make other blocks absent too — assert only the two this test is about. If the module's fixtures offer a way to seed a run without a `plain` stream, prefer that and drop the monkeypatch.

- [ ] **Step 3: Replace the false regression guard with a true one**

Do **not** write `test_no_measurement_reads_the_enhanced_stream`. It would be false: `enhanced_yamnet_scores`, `enhanced_ast_scores` and `enhanced_hear_scores` **are** measurements written with `signal="enhanced"` — `_stream_classifier_scores` writes `name = f"{prefix}_{classifier}_scores"` (`:2528`, `:2503`), so the *block* is `enhanced_yamnet` (`:2604`) but the *measurement* carries the `_scores` suffix. `find_measurement(store, f"enhanced_{classifier}")` is `None` for all three. They pass a string guard only because they read `state["enhanced_audio"]` instead of calling `resolve_stream` with a literal. A guard whose name asserts something untrue is worse than none.

Scope it to the audit's actual claim — that these two measurements no longer read the denoised stream:

```python
    def test_neither_praat_nor_the_posteriorgram_reads_the_enhanced_stream(
        self, residual_config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: ProvStore,
        wav_writer: Callable[..., Path],
    ) -> None:
        """The audit's bound is about these two measurements, not about the stream having no readers."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), residual_config, run_dir=tmp_path)
        for name in (PPG_MEASUREMENT, PRAAT_MEASUREMENT):
            measurement = find_measurement(store, name)
            assert measurement is not None
            assert measurement.attributes["signal"] == "plain"
        assert any(
            find_measurement(store, f"enhanced_{classifier}_scores") is not None
            for classifier in ("yamnet", "ast", "hear")
        ), "the enhanced stream still has its own classifier measurements; that is not what moved"
```

Match the fixture set the neighbouring tests use rather than the list above if they differ.

- [ ] **Step 4: Run the whole triage suite**

Run: `uv run pytest src/tests/audio/workflows/triage -q`
Expected: PASS.

- [ ] **Step 5: Update the two PPG fixtures for accuracy**

`live_evidence_test.py:216` and `routing_analysis_test.py:949`/`:959` hardcode `"signal": "enhanced"` on constructed PPG measurements. Nothing filters on `signal`, so **they pass either way** — change them to `"plain"` because they now describe a measurement the node never writes that way, not because a test fails.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format src/tests/audio/workflows/triage/
uv run ruff check src/tests/audio/workflows/triage/
git add -A && git commit -m "test(triage): the measurement stream is plain, and the guard says what is true"
```

---

## Task 7: The extend driver re-derives, and supersedes when it does

**Files:**
- Modify: `scripts/extend_ppg_praat.py:26-27`, `:94-115`, `:187-190`, `:197`, `:235-238`
- Test: `src/tests/scripts/extend_ppg_praat_test.py`

**Interfaces:**
- Consumes: Task 5 — without it there is nothing stale to re-derive.
- Produces: `--force` on the CLI; a `force: bool` keyword threaded to **the function containing the skip at `:187`** (read whether that is `process_batch` or its caller before writing the signature — the plan does not guess).

**The integrity consequence, which is why `--force` cannot simply append.** `write_ppg_posteriorgram` writes `derivatives/ppg_posteriorgram.npz` at a fixed path. A forced re-derivation **overwrites that file**, so the old measurement entity's `checksum_sha256` no longer matches the bytes it names. The store is append-only, so the old entity must be **superseded**, not left live beside the new one — otherwise the store asserts two readings of the same thing and one of them is provably stale.

- [ ] **Step 1: Seed `plain` in the fixture, or every test in the module fails**

`_seed_run` (`:61-97`) writes exactly one stream entity, `name: "enhanced"` (`:88`). After Task 5, both blocks resolve `plain` and raise `LookupError`, so essentially every test here fails — including the new one. Add a second stream entity named `plain`, written the same way, pointing at a `streams/plain.flac` written alongside `streams/enhanced.flac`.

- [ ] **Step 2: Run the module and confirm it is green again before adding anything**

Run: `uv run pytest src/tests/scripts/extend_ppg_praat_test.py -q`
Expected: PASS. If not, the fixture is not seeding `plain` the way `resolve_stream` reads it — fix that before proceeding.

- [ ] **Step 3: Write the failing `--force` test**

The module drives everything through `cli.main([...])` with the `provisioned` and `stub_ppgs` fixtures. Follow that:

```python
class TestForceReDerives:
    """After a stream switch the stored measurements are stale, and skipping them looks like success."""

    def test_force_re_derives_a_store_that_already_holds_both(
        self, corpus: Callable[[int], tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """Without --force the driver skips, so a corpus-wide re-derivation silently does nothing."""
        manifest, roots = corpus(1)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        before = _store_of(roots[0]).fingerprint()

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"]) == 0
        store = _store_of(roots[0])
        assert store.fingerprint() != before, "--force must re-derive, not skip"

    def test_force_leaves_exactly_one_live_measurement_of_each(
        self, corpus: Callable[[int], tuple[Path, list[Path]]], provisioned: None, stub_ppgs: None
    ) -> None:
        """The store is append-only, so the superseded reading must not stay live beside the new one."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"])

        store = _store_of(roots[0])
        for name in (PPG_MEASUREMENT, PRAAT_MEASUREMENT):
            live = [e for e in live_entities(store, "measurement") if e.attributes.get("name") == name]
            assert len(live) == 1, f"{name}: {len(live)} live measurements after --force"
```

Use `senselab.audio.workflows.triage.nodes.common.live_entities` rather than reinventing the liveness filter — `extend_withdraw_clips_test.py` is the model for both this and the `supersede` call in Step 6.

- [ ] **Step 4: Run and watch both fail**

Run: `uv run pytest src/tests/scripts/extend_ppg_praat_test.py::TestForceReDerives -v`
Expected: FAIL — `--force` is not a recognised argument.

- [ ] **Step 5: Add the flag**

In `build_parser`, beside `--config` (`:115`):

```python
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-derive both measurements even where the store already holds them, superseding the old",
    )
```

Thread `force` from `args.force` through to the function holding the skip. At `:187-190`:

```python
        ppg_held, praat_held = (not p for p in pending(store))
        ppg_pending, praat_pending = pending(store)
        if force:
            ppg_pending = praat_pending = True
        if not ppg_pending and not praat_pending:
```

**Capture what the store held before the override, or Step 6 has nothing to supersede.** Overriding both
to `True` destroys exactly the information the supersession needs — which measurements already existed.
Thread `ppg_held` / `praat_held` alongside the pending flags. Apply the same capture and override at the
`:235` re-check, which recomputes `praat_pending` and would otherwise skip the Praat block on a forced run.

**And decide what happens when the replacement write fails.** If supersession runs first and the write then
raises at the driver's `except` (`:223`), the store carries an invalidated old measurement and no live
replacement — worse than either end state. Either supersede only after the new bytes and the new entity
both exist, or record the half-state as an accepted outcome with its reason. The plan does not choose for
you, but it must be chosen before this ships. Correct the docstring at `:26-27` — the skip is the default and `--force` overrides it — and the comment at `:197`, which becomes false.

- [ ] **Step 6: Supersede the old measurement before writing the new one**

Add a step constant beside `extend.py:77-81`'s `WORD_SUPERSEDED` / `CLIP_SPAN_SUPERSEDED` family:

```python
PPG_MEASUREMENT_SUPERSEDED = "ppg_posteriorgram_superseded"
PRAAT_MEASUREMENT_SUPERSEDED = "praat_features_superseded"
```

and a reason string in the same idiom:

```python
_STREAM_SWITCH_REASON = "it was measured on the enhanced stream; the measurement now reads plain"
```

Then, in the forced path, for each measurement the store already holds, call `extend.supersede` **before** the replacement is written — the sidecar is overwritten in place, so once the new bytes exist the old entity's checksum is already wrong and the store is briefly inconsistent. Read `extend_withdraw_clips.py`'s use of `supersede` and follow its argument shape exactly (`node=`, `step=`, `reason=`, `software=`).

- [ ] **Step 7: Run and confirm both pass**

Run: `uv run pytest src/tests/scripts/extend_ppg_praat_test.py -v`
Expected: PASS, including the existing skip tests without `--force` and `test_a_rerun_converges_on_the_same_graph` (`:276`), which asserts the no-force path is still idempotent.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff format scripts/extend_ppg_praat.py src/tests/scripts/extend_ppg_praat_test.py
uv run ruff check scripts/extend_ppg_praat.py src/tests/scripts/extend_ppg_praat_test.py
uv run mypy scripts/extend_ppg_praat.py
git add -A && git commit -m "feat(extend): --force re-derives and supersedes what the stream switch made stale"
```

---

## Task 8: The derived range is recorded, not just used

**Files:**
- Modify: `src/senselab/audio/tasks/features_extraction/praat_parselmouth.py:1302` and the `feature_data` assembly around it
- Modify: `src/senselab/audio/workflows/triage/nodes/preprocess.py:886-895`
- Test: `src/tests/audio/tasks/features_extraction_test.py`, `src/tests/audio/workflows/triage/nodes/preprocess_test.py`

**Interfaces:** consumes Tasks 1 and 5; produces `pitch_floor`, `pitch_ceiling`, `pitch_frames`, `pitch_failed` and `pitch_range_fell_back` in the 40-scalar dict and on the `praat_features` measurement's attributes.

**Without this task the plan is a net loss in auditability.** `_extract_one` (`:1302`) calls `extract_pitch_values` and threads the floor and ceiling into every subsequent extractor, but puts **neither into `feature_data`**. Under the bin a reader could at least infer which of two ranges applied. After Task 1 every recording's forty scalars are conditioned on a **recording-specific** range recorded nowhere, with no support count — which is audit finding 5 (no function returns a support count) reproduced on the very instrument this plan repairs. It is also what would have surfaced the hum regression: a `pitch_range_fell_back` of 1.0, or a floor and ceiling that do not bracket the voice, is visible in the measurement's attributes and invisible anywhere else.

- [ ] **Step 1: Write the failing tests**

```python
    def test_the_forty_scalars_carry_the_range_they_were_measured_under(self) -> None:
        """Per-recording ranges make two recordings' scalars incomparable; the range must travel."""
        [features] = extract_praat_parselmouth_features_from_audios([_buzz(150.0, seconds=2.0)])
        for key in ("pitch_floor", "pitch_ceiling", "pitch_frames", "pitch_range_fell_back"):
            assert key in features, f"{key} must travel with the scalars it conditioned"
        assert features["pitch_floor"] < 150.0 < features["pitch_ceiling"]
```

And in `preprocess_test.py`, inside the class retargeted in Task 6:

```python
        for key in ("pitch_floor", "pitch_ceiling", "pitch_frames", "pitch_range_fell_back"):
            assert key in attrs["features"], f"{key} must be on the measurement, not only in the call"
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/features_extraction_test.py -k range_they_were_measured -v`
Expected: FAIL with `KeyError` or the `in` assertion — `feature_data` has no such keys.

**Two consequences of this task that must be stated, not discovered.** The five keys reach `record.praat`
through `features.py:726`, so `pitch_failed` and `pitch_range_fell_back` become **two new routing indicator
features** — harmless today because no gate names them, but they are now in the feature vector and the
ruleset's own audit should know. And `_extract_one` calls `extract_pitch_values(snd=snd)` with **no search
range**, so the 40-scalar path uses the signature defaults (50/600) rather than `voice.f0_search_range_hz`.
That is two sources for one range, the Praat one being a code literal — in tension with this plan's own
constraints, and it is the literal Task 8 will record as provenance. Record it as owed; do not fix it here,
because threading config into that function is a separate change with its own callers.

- [ ] **Step 3: Thread the five keys into `feature_data`**

In `_extract_one`, after the `extract_pitch_values` call at `:1302`, add its five keys to `feature_data` under their own names. Do not rename them — they must match what `extract_pitch_values` returns so a reader grepping one finds the other. The `praat_features` measurement already carries `features=scalars` (`preprocess.py:893`), so they reach the store with no change there; confirm that and only touch `preprocess.py` if it filters keys.

- [ ] **Step 4: Run and confirm**

Run: `uv run pytest src/tests/audio/tasks/features_extraction_test.py src/tests/audio/workflows/triage/nodes/preprocess_test.py -q`
Expected: PASS. The `n_features` count on the measurement rises by five — if a test asserts an exact count, update it to the new number rather than excluding the new keys.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src/senselab/audio/tasks/features_extraction/ src/senselab/audio/workflows/triage/nodes/preprocess.py
uv run ruff check src/senselab/audio/tasks/features_extraction/ src/senselab/audio/workflows/triage/nodes/preprocess.py
uv run mypy src/senselab/audio/tasks/features_extraction/praat_parselmouth.py
git add -A && git commit -m "fix(praat): the derived range travels with the scalars it conditioned"
```

---

## Task 9: The spec records what landed and what is owed

**Files:** modify `specs/20260817-triage-workflow-dag/praat-instrument-audit.md`

- [ ] **Step 1: Mark steps 1, 1b and 2 landed**

For each: what changed, the convention and that it was not fitted, and what remains. Steps 0, 3, 4 and 5 stay open and must still read as open. Add that **step 2b is untouched** — `preprocess.py:941` derives the range on `plain` while `:951` tracks F0 on `sharp`, and this plan does not close that.

- [ ] **Step 2: Answer the question step 1 asks and this plan did not**

The audit's step 1 requires saying **what carries noise robustness once FRCRN leaves the measurement path**. It is now unanswered in the shipped design. Either name what does (HeAR, YAMNet and SQUIM still read `plain` and never read `enhanced`, so nothing downstream of them changes; the `enhanced` stream keeps its own classifier measurements) or record it as owed. Do not leave it unstated.

- [ ] **Step 3: Record what the stream switch trades, which is not the same as what it fixes**

Three statements, none currently in the audit:

- **The bias flips rather than disappearing.** FRCRN inflated HNR and CPPS and deflated perturbation, worst on dysphonic voices — an *endogenous* confound that inverted sensitivity. On `plain`, additive noise is indistinguishable from aperiodicity, so it deflates HNR and CPPS and inflates perturbation in proportion to environment, device and level. That is the better trade **because the new confound is exogenous and covariable — but only if the covariates ride along.** They exist: SQUIM is already computed per span on `plain`, plus `clipping/` and `scene_quality/`. Record that requirement, and that the **raw-vs-enhanced pilot** the contract requires for the structurally identical diarization decision is owed here too.
- **`plain` is not raw.** It is resampled to 16 kHz, so energy above 8 kHz — a breathiness correlate — is already gone. That caps how much of V4's sensitivity this switch can recover, and the phrase "measured on the unprocessed stream" invites the wrong assumption.
- **The PPG's justification is not the Praat scalars'.** The breathiness argument does not transfer: the PPG is a **trained phoneme classifier**, so noisy `plain` moves it *away* from its training domain while FRCRN moved it toward. The defensible reason is D1's, and it belongs at this site — **FRCRN is out of domain on a DDK train and can smear the transients whose timing is the measurement.** Put D1's reason in the audit beside step 1b.

- [ ] **Step 4: Correct the glide entry, which is owed for a different reason than stated**

Measured: an exponential 100→400 Hz glide yields **`[72.8, 549.5]`** against produced extremes of 102 and 392 Hz. The ±7-semitone margin brackets the sweep with room to spare, so **narrowing does not clip a glide** — and it is strictly better than the bin, which capped a low-binned upward glide at 250 Hz. What degenerates is **V3's conformance flag**: after narrowing it can only fire when the margin pushes past the 50/600 clamps, so it becomes near-**vacuous** rather than circular. Record that, and drop the "actively wrong on a glide" framing — it overstates the cost and would justify a task-conditioned range the measurement does not support.

- [ ] **Step 5: Record the three consequences this plan creates and does not close**

- **Within-participant contrasts are improved, not restored.** A participant's shouted and comfortable productions still get *different* instruments — continuously now rather than in a 1.67× step. The right instrument for a within-participant contrast is one range shared across the productions compared. Out of scope here; record it as owed so it does not read as closed.
- **Coverage now shifts F0-dependently.** The window formulas `3/floor` (period marks), `3.2/floor` (intensity) and `4.5/floor` (harmonicity) vary per recording: a 420 Hz speaker gets an 11 ms intensity window and a 55 Hz speaker gets 64 ms — **worse than the bin's 53 ms**. So the audit's step 5 coverage figure becomes conditioned on a per-recording window and is *harder* to interpret, not easier — unless the floor is recorded, which Task 8 is what fixes.
- **`voice.f0_range_ratio_max` intuition changes.** The bin always gave 4.17 or 5.0; the narrowing measured **7.5** on a glide and can exceed 11. Whoever populates that key should know that bin-era intuition will refuse glides, and that `_f0_range` **raises** rather than flags.

- [ ] **Step 6: Record the two corpus passes this plan creates but does not run**

- `extend_ppg_praat.py --force` over the corpus re-derives both measurements in all 62,547 stores. That is a GPU-bearing pass for the PPG — size it against the original `ppg_20260911` run rather than guessing.
- The PPG switch changes `ppg.segment_rate_per_s` and therefore DDK routing. **The design requires a before-and-after routing count**, and alongside it the **`PpgsPosteriorgramUnavailable` rate before and after**, since the switch may change how often the model fails outright — Task 6's note about availability concerns the lookup path, not the model. Until both are run, no DDK routing figure in any document describes the shipped pipeline.
- **Attribution.** Landing Tasks 1 and 5 together moves the forty scalars for two reasons at once, making any corpus-level change unattributable. A **two-arm sample re-derivation** — range-only on `enhanced`, stream-only with the bin — over a few hundred recordings separates them. That is attribution, not fitting, so the no-fitting rule does not reach it.

- [ ] **Step 7: State plainly that no scalar becomes publishable from this plan**

This is the sentence most likely to be needed and least likely to be written. After this lands the scalars are on the right stream with a principled range — and `cepstral_peak_prominence_mean` is still cut at `> 4` dB, still peak-searched 60–330 Hz, still averaged unweighted over voiced intervals inflated ~70%; `range_ratio_intensity_db` is still dimensionally invalid; no function returns a support count; and step 0's withholding has not happened. **Write in the audit that no Praat scalar becomes publishable from this plan**, or "we fixed the stream and the range" will be read as "the numbers are now good."

Pre-empt two specific misreadings:

- **The jitter and shimmer NaN rate will fall for two opposite reasons at once** — more low voices admitted (good) and noise inflating perturbation into measurable territory (bad). Without step 5's coverage measurement nobody can separate them, so a falling NaN rate is not evidence of improvement.
- **A 45 Hz creak reading as `F0RangeUnavailable` becomes VOICE "no phonation found"** — a false clinical statement about a phonating voice. Attribute that absence as **bounded by the declared search range**, not by the voice.

- [ ] **Step 8: Run the full suite and commit**

```bash
uv run pytest src/tests/audio/tasks src/tests/audio/workflows/triage src/tests/scripts -q
git add -A && git commit -m "docs(audit): steps 1, 1b and 2 landed; the corpus re-derivation is owed"
```

---

## Self-Review

**Spec coverage.** Step 2 → Tasks 1, 2, 3, 4. Step 1 → Task 5 (Praat half), Task 6. Step 1b → Task 5 (PPG half), Task 6. The re-run path → Task 7. The derived range's auditability → Task 8. The spec update, the noise-robustness statement, the corrected glide entry and the not-publishable headline → Task 9.

**The one place this plan is worse than what it replaces, and how it is closed.** A single-pass narrowing
returns `[50, 90]` for a 120 Hz voice under a 60 Hz hum — a range excluding the speaker — where the bin
returned `(60, 250)`, which contains it. **The pinned-contour fallback in Task 1 Step 5 closes it**: if p95
falls below twice the search floor the contour is at the bottom of the search range and the wide range is
used instead. Measured on eight sources, it fires on the hum and the 55 Hz fry and on nothing else.

Two earlier attempts at this are recorded in the plan so they are not re-proposed. A second-pass median
comparison **cannot fire** — the narrowed range is `[p5/1.5, p95·1.5]`, so the second median moves at most
0.585 octave against a 1.0 test — and its snippet also called `get_sound` on an object that function does not
accept, which the outer `except` would have swallowed into a corpus-wide `F0RangeFailed`. Both were found by
implementing Task 1 and running its tests rather than by reading it.

**Every test the changes break is named.** `phonation_test.py:112-119`; `preprocess_test.py:2172`, `:2193`, `:2194`, `:2204-2206`, `:2226`, `:2235-2236`, `:2238`, `:2289-2305`; `extend_ppg_praat_test.py:61-97` (the fixture, which breaks the whole module). The two `signal: enhanced` fixtures in `live_evidence_test.py` and `routing_analysis_test.py` are named as **not** breaking, with the reason.

**Placeholders.** No step describes work without showing it. Four steps name an existing fixture or API to read and match rather than quoting it — Task 1 Step 1 (`_buzz`), Task 3 Steps 1–2 (the two suites' fixture sets), Task 6 Step 3 (the neighbours' fixtures), Task 7 Step 3 (`ProvStore`'s liveness API) — because this suite already has them and duplicating would be the wrong instruction. Task 7 Step 6 names `supersede`'s argument shape as something to copy from `extend_withdraw_clips.py` rather than transcribing it, for the same reason.

**Type consistency.** `extract_pitch_values` returns **five** `float` keys on all **three** return paths — the empty-contour guard, the main dict, and the `except`. `derive_f0_range` keeps `tuple[float, float]` and raises one of two `ValueError` subclasses. `write_ppg_posteriorgram`'s keyword is `stream_id` at the definition and both call sites. `force: bool` threads to the function holding the skip.

**The TDD loop starts red.** Task 1's first test is the discontinuity case, which fails under the bin. The monotonicity test is deliberately second, because `[250, 250, 500, 500, 500]` is sorted and would pass on current code.

**Known limits, stated rather than hidden.** The 55 Hz test passes via the search-floor clamp, not by the floor following the voice — its docstring says so. The log-Hz trim does not remove an octave-split mode; the 1.5× margin and the ceiling clamp bound that error, and octave robustness is marked owed. The percentile and margin values are this project's convention, not Hirst's parameterisation. Task 7's supersede call names its constants and its liveness assertion but defers `supersede`'s exact signature to the existing driver.
