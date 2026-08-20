# Review of `plan-foundation.md`

Reviewed against `senselab` at `f4865db4` (branch `feat/triage-phase2-defects`), the node documents in
this directory, `capability-map.md`, and `benchmarks/open.md`. Nothing was implemented and no file
outside this one was modified.

**Method note.** Where a claim is numeric I extracted the plan's implementation verbatim into
`/tmp/planrev/` and ran it under `uv run` against the real `senselab.audio.data_structures.Audio`.
Results marked "ran" below are measured, not reasoned. Everything else names a file and line I read.

Three findings (F-1, F-2, F-9) are test failures the plan will hit at the step where it says "all
PASS". Two (F-3, F-6) are the failure mode this review was commissioned to hunt: a type the codebase
already has. One (F-9) blocks every task.

---

## High severity

### F-1 — Task 4, Step 1: the plan's own two tests contradict each other, and one contradicts `preprocess.md`

**What the plan says.** `TestGate::test_a_peak_below_k_is_not_proposed` builds an envelope with one
event at −45 dB over a flat −55 dB floor and asserts `propose_spans(..., k_db=18.0) == []`.
`TestNoContrast::test_no_peak_anywhere_is_no_contrast_not_an_empty_list` asserts `NoContrast` for a
constant envelope.

**What actually happens.** Ran the plan's `propose_spans` verbatim on the plan's own fixture:

```
gate below k -> == []?  False
  NoContrast(reason='no peak rose 18.0 dB above the local floor; the largest rose 10.0 dB')
```

Both fixtures have zero peaks clearing `k_db`, so they are indistinguishable to the function. The
first test fails. Step 4's "Expected: all PASS" is unreachable.

**The design agrees with the implementation, not the test.** `preprocess.md`, `## spans`: "If no peak
anywhere reaches `K` above the local floor, the node reports **`no_contrast`** rather than an empty
list."

**Correction.** Make `test_a_peak_below_k_is_not_proposed` assert `NoContrast`. Get `[]` from the only
path the design leaves for it: a peak that *clears* `k_db` but whose derived span falls under
`min_duration_ms` (e.g. a 20 ms event at −20 dB). Separately, `TestNoContrast`'s constant-array
fixture yields no peaks for *any* `k_db`, so it never exercises the gate — give it a real peak below
`K`.

### F-2 — Task 7, Step 1: the periodicity test fails, and the estimator cannot reach the interval `voice.md` measured

**What the plan says.** `test_a_buzz_is_periodic_and_noise_is_not` asserts
`float(np.median(p_voiced)) > 0.9` for a 100 Hz harmonic buzz.

**What actually happens.** Ran the plan's `periodicity_track` verbatim:

```
median voiced periodicity 0.8   > 0.9 ?  False
median noise  periodicity 0.085 < 0.5 ?  True
median f0 120.3 (~120 ±4) True
```

`_autocorr_peak` normalises a **biased** `np.correlate` by `ac[0]`, so the value carries the taper
`1 − lag/N`. With `frame_len = 3·sr/f0_min` and `lag = sr/f0`: `1 − 160/800 = 0.80` exactly. The test
threshold is unreachable by construction, not by a near miss.

**The consequence is larger than the test.** For the reference voice at 87 Hz with
`f0_min_hz = 60`, the ceiling is `1 − 184/800 = 0.77` — **below** `benchmarks/voice.md`'s measured
0.933 and 0.934. So `voice.md`'s gate interval `(0.44, 0.933)` is partly unreachable, and the numbers
recorded there are not the quantity this function returns. A VOICE gate thresholded on `voice.md`
against this estimator would be comparing two different scales.

**Correction.** Use a normalised cross-correlation denominator
(`ac[lag] / sqrt(Σx[0:N−lag]² · Σx[lag:N]²)`), which is what Praat's `cc` method does, and record in
`benchmarks/voice.md` which estimator produced 0.933 — otherwise the interval cannot be checked
against anything.

### F-3 — Task 7: `period_marks` reinvents Praat's point process, and cannot deliver the jitter it exists for

**What senselab has.** `src/senselab/audio/tasks/features_extraction/praat_parselmouth.py:1115`
builds a real glottal period point process:

```python
return parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
```

It is consumed by `extract_jitter` (`praat_parselmouth.py:1101`) and `extract_shimmer` (`:1157`), and
`f0min`/`f0max` arrive as required arguments there — the same no-default discipline Task 7 wants.
`extract_pitch_values` (`:358`) documents at `:386` that cross-correlation is the right pitch route
"such as jitter and shimmer, whereas auto-correlation is better at finding intended intonation
contours" — the plan chose the one that docstring names as the wrong route for this purpose.

**Why the plan's version cannot work.** `period_marks` sets `time_s = cursor / sr` and advances
`cursor += int(sr / f0)`. Marks are therefore integer multiples of a frame-wise argmax lag, not
waveform landmarks. Consecutive "periods" can differ only when the argmax lag changes, so the only
period-to-period variation available is lag quantisation — 62.5 µs at 16 kHz. `benchmarks/voice.md`
("Jitter, and shimmer from the amplitudes, are defined *between consecutive periods* and are
unrecoverable from a resampled contour — so the primary product is a point process") is the entire
reason the point process was specified, and this implementation forecloses it.

**Correction.** Expose the pulse times from the existing Praat point process (`Get number of points`
/ `Get time from index`) and derive `period_s` and `amplitude` from consecutive pulses — the gap
senselab actually has is that `extract_jitter`/`extract_shimmer` return only aggregated numbers, not
the point process. If Praat's is unsuitable, the plan must carry the measurement showing so.

### F-4 — Task 7: `periodicity_floor` is defaulted, and it is the first parameter the plan's own Global Constraints forbid defaulting

**What the plan says.** Global Constraints: "Parameters the design leaves undecided must be
keyword-only with no default … These are: **the phonation gate floors**, SQUIM thresholds, the
redaction padding margin, the word-gap threshold, `min_families` per kind."

**What the plan writes.** `PERIODICITY_FOR_MARKS = 0.5`, and
`period_marks(..., periodicity_floor: float = PERIODICITY_FOR_MARKS)`.

0.5 sits inside the interval three separate documents refuse to collapse:

- `benchmarks/open.md`: "**The phonation gate's floors** … VOICE's gate carries an interval —
  periodicity `(0.44, 0.933)`, RMS `(0.0007, 0.0161)` — and no value".
- `benchmarks/voice.md`: "the config records the interval and leaves the derivation empty. **A
  midpoint would be an invented decision.**"
- `capability-map.md` §5.1, row 3.

The plan enforced no-default on `f0_min_hz`/`f0_max_hz` instead — also required
(`capability-map.md` §5.1, last row) but a different parameter.

**Also missing entirely.** The **RMS floor** `(0.0007, 0.0161)`. `voice.md`: "the
energy-and-periodicity conjunction excludes them. A gate on energy alone would admit every breath in
the recording." `period_marks` has no energy gate, so half the conjunction is absent.

**Correction.** `periodicity_floor` keyword-only, no default. Add `rms_floor`, keyword-only, no
default. Both documented as intervals with the derivation slot empty.

### F-5 — Task 4, Step 5: the reproduction step names the wrong script, an unrunnable path, and a figure the benchmark does not contain

**What the plan says.** "Run: `uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/floor.py`
Expected: five spans at 2.32–3.29, 5.32–6.22, 7.92–8.51, 9.61–9.96, 11.75–13.16 s, matching
`benchmarks/spans.md`. If they differ, the implementation diverges from the measured rules — fix the
implementation, not the benchmark."

Three problems, each independently fatal to the step:

1. **The path is machine-local.** `benchmarks/scripts/floor.py:3-4` loads
   `/Users/satra/.claude/jobs/295c3f8a/tmp/stage1.npz` and `.../yamnet.npz`, and `:39` writes back
   into the same job scratch dir. Thirteen of the sixteen scripts under `benchmarks/scripts/` do the
   same. No other host, and no future session, can run this.
2. **It is not the span producer, and it uses a different rule.** `floor.py:20-37` is a *floor
   comparison*: it prints three span sets under three **scalar** floors (whole-file 10th percentile,
   YAMNet-silence median, YAMNet-silence p95). The script whose output shape matches Task 4's `Span`
   (it carries `Edb[p]-floor` as a third column) is `benchmarks/scripts/s2b.py`, also on a scalar
   floor. Task 4 implements `find_peaks(envelope − floor_track, height=k_db)` against a **rolling**
   floor. Neither script exercises that rule, so neither can confirm or refute the implementation.
3. **The expected figure is not in the benchmark.** `2.32` and `11.75` appear nowhere in
   `benchmarks/spans.md`. That file's `K` table (spans.md, "Propose threshold `K`") says K = 18 dB
   gives **6** spans on the clean file. The "5" the plan quotes looks like the onset hit-rate from
   spans.md's "## Onset" table — "**5 / 6** (5/5 excluding speech)" — which is a hit count, not a
   span count.

**Correction.** Delete Step 5, or replace it with a reproducible fixture: commit the envelope and
floor arrays (or the reference WAV) into the repo, write the expected span list into `spans.md`
first, and assert against that.

### F-6 — Task 2: `ModelProvenance` duplicates two existing provenance types and inverts this codebase's meaning of `revision`

**What senselab has.** Two shapes, both with written rationale:

- `src/senselab/audio/data_structures/audio_hints.py:47` — `SpeakerEmbeddingProvenance`, a **pydantic
  v2 model** with `model_id`, `model_commit_sha` (40-hex validated at `:88-107`: "Never a ref:
  recording `"main"` here would be provenance that is confidently wrong, which is worse than
  recording none"), and `unresolved_reason` ("Required in that case, so an absent commit is always
  explained rather than merely missing").
- `src/senselab/audio/workflows/audio_analysis/signal.py:56` — `SignalProvenance`, carrying **both**
  `revision` and `commit_sha`, with the reason spelled out at `:67-70`: "'revision' is the ref the
  model was pinned to, 'commit_sha' is the immutable commit that ref resolved to … A consumer that
  sees only 'revision' cannot tell a deliberate pin from a tracked ref that happened to resolve
  there on the day". `StageContext.provenance_for`
  (`audio_analysis/stage_context.py:303-332`) emits the same pair.

**What the plan writes.** `ModelProvenance(model_id: str, revision: str)` where `revision` must be a
40-hex SHA. In this codebase `revision` means the ref; the resolved commit is `commit_sha` or
`model_commit_sha`. Naming the SHA field `revision` is exactly the parallel-name problem `CLAUDE.md`
forbids pre-alpha, in the one field where getting it wrong makes provenance "confidently wrong".

**Second problem: no unresolved state.** `signal.resolved_commit_sha` (`signal.py:96-112`)
deliberately degrades to `None` on a Hub outage, and `audio_hints` pairs `None` with a required
`unresolved_reason`. `ModelProvenance.__post_init__` raises, so a node running when the Hub is
unreachable cannot write *any* element — the store's availability is coupled to
`huggingface.co`.

**Third problem: a fourth copy of the regex.** `_SHA = re.compile(r"^[0-9a-f]{40}$")` already exists
verbatim at `utils/model_revision.py:30`, `utils/dependencies.py:301` and `audio_hints.py:28`.

`store.md` does say "the model id and resolved revision", so the plan is not inventing the concept —
it is inventing a third incompatible shape for it.

**Correction.** Name the field `model_commit_sha`; allow `None` with a required `unresolved_reason`;
import the predicate rather than re-declaring the regex. Make it a pydantic model so it and
`SpeakerEmbeddingProvenance` are one shape (or subclass it). Note also that `store.md` requires
"An embedding comparison additionally carries the model that produced the target" — a single
`model_id` cannot express that, which the node plan will hit.

### F-7 — Task 2: the store overwrites, contradicting its central claim, because `model` is not in the id

`add_element` digests `{'r': run_id, 'k': kind, 'x': extent, 'a': author, 'e': evidence}` — `model` is
excluded — then executes `self._elements[eid] = Element(...)`. That is a dict assignment, i.e. an
overwrite, in a class whose docstring says "Nothing is deleted and nothing is overwritten".

Concretely: two models producing the same evidence over the same extent for the same author collide,
and **the second silently replaces the first, losing its provenance**. `assert_over` has the same
omission but appends to a list, so two such assertions carry an *identical* `aid`. That breaks two
things the plan tests for: `assertions_for` returns two records `confirm`/`contest` cannot
disambiguate (the invariant `test_an_assertion_carries_its_own_id_so_confirm_can_name_it` exists to
protect), and `merge`'s `seen: set[str]` drops one of them, so merge is not content-preserving even
though `fingerprint()` reports it as identical (see F-14).

**Correction.** Include `model` in both digests, and have `add_element` raise on an id already
present with different content (or no-op on an exact duplicate), so "nothing is overwritten" is
enforced rather than asserted.

### F-8 — Task 6: `span_to_hear_buffer` duplicates existing window planners, ships two unmeasured placements, and names a test file that does not exist

**What senselab has.** `hear.plan_centred_windows` (`audio/tasks/health_acoustics/hear.py:354`) and
`api.extract_hear_embeddings_at_times` (`audio/tasks/health_acoustics/api.py:174`) already answer "I
have a 0.3 s cough at t = 9.58 s", by taking a 2 s window **from the real recording**, clamped inward
rather than padded. `hear._require_two_seconds` (`hear.py:308-322`) carries the measurement:
zero-padding a 0.3 s event to 2 s "moves its embedding as far as substituting unrelated audio
(centred cosine 0.0-0.5 against a class margin of ~0.9)".

**What is genuinely missing, and what is not.** `capability-map.md` §4.1 (lines 454-468) is right
that both can be true — the refusal was measured on the **encoder**, `benchmarks/hear-yamnet.md`
measured the **detector** on padded spans (Breathe 0.989 / Cough 0.996), and only
`detect_health_acoustic_events` (a sliding scan) exists for the detector, so "classify one span with
the detector" is a real gap. But:

- The plan's signature is `Audio -> Audio` with nothing tying it to the detector. The obvious next
  call is `extract_hear_embeddings_at_times(span_to_hear_buffer(...))` — the **encoder**, where the
  measurement says padding is wrong — and the docstring does not warn against it. It should say so
  explicitly and live beside `plan_scan_windows`/`plan_centred_windows` with the same caveat block.
- `placement: str` accepts `"start"` and `"end"`. `capability-map.md` §5.2: "Centred, left-aligned
  and right-aligned are three different inputs … **Pick one, name it, and record which the benchmark
  numbers were measured under.**" Only centred was measured (`benchmarks/scripts/spaninput.py`) and
  only centred is tested. Two unmeasured options behind a public parameter is precisely what
  `CLAUDE.md`'s no-per-knob-flags rule exists to stop. Make it `Literal["centre"]` or drop it.
- `want = 2 * sr` and the error text's "the 2 s the detector accepts" re-derive constants that are
  already named at `hear.py:126` (`HEAR_WINDOW_SAMPLES = 32000`) and `hear.py:130`
  (`HEAR_WINDOW_SECONDS`).

**Wrong test file.** The plan targets `src/tests/audio/tasks/health_acoustics_test.py`. What exists is
the package `src/tests/audio/tasks/health_acoustics/` containing `__init__.py` and `hear_test.py`.
Step 2's expected `ImportError` and Step 4's "the existing ones still pass" both assume the plan's
path is the existing file. Put the tests in `src/tests/audio/tasks/health_acoustics/hear_test.py`.

### F-9 — `ruff check` will fail on every test file in the plan, and on two production functions

`pyproject.toml`:

```toml
[tool.ruff.lint]
select = ["ANN", "D", "E", "F", "I"]
ignore = ["F401"]

[tool.ruff.lint.per-file-ignores]
"src/tests/**/*.py" = []
```

Tests get **no** exemptions. Every test the plan writes is `def test_x():` with no return annotation
(ANN201) and no docstring (D103); its test classes (`class TestGate:`, `class TestAppendOnly:`, …)
have no docstrings (D101) and no method docstrings (D102); and
`test_jsonl_survives_a_round_trip(self, tmp_path)` leaves `tmp_path` unannotated (ANN001). That is
roughly sixty violations across the ten tasks. The house style is visible at
`src/tests/audio/tasks/plotting_test.py:18-22` — class docstring, `-> None` on every method, docstring
on every method.

Production code, same ruleset: Task 2's `_digest(payload: dict[str, Any]) -> str:` has no docstring
(D103), and `assert_over`'s `value: Any` argument is ANN401 (the codebase's own precedent for that is
`# noqa: ANN401` at `audio_analysis/signal.py:130` and `audio_analysis/pii.py:55`).

Task 9, Step 5 runs `uv run ruff check src/senselab src/tests`, so this is a hard stop at the end of
the plan rather than a nit.

### F-10 — Task 1, Step 5: the call-site edit does not fit the code it edits

**What the plan says.**

```python
        line = item if isinstance(item, ScriptLine) else None
        spans = _materialize_spans(raw, source_id=source_id, line=line)
```

**What is actually there.** `src/senselab/text/tasks/pii_detection/api.py:503-511` is a list
comprehension over `range(len(texts))`, with no `item`, no `raw` and no `source_id` in scope:

```python
    return _finish(
        [
            PiiScan(
                spans=_materialize_spans(spans_by_index_raw.get(str(i), []), str(i)),
                detectors_used=list(detectors_used),
                failures=dict(failures),
            )
            for i in range(len(texts))
        ]
    )
```

**Correction.** Replace the comprehension with a loop over `enumerate(items)` — both `items` and
`texts` are in scope from `api.py:421-427` — and take
`line = item if isinstance(item, ScriptLine) else None` there.

### F-11 — Task 1, Step 5: the rewritten `_materialize_spans` drops two behaviours the current one has

Current, `api.py:346-357`: dedupe key is `(category, text.strip().lower(), source)`;
`if not normalized or dedup_key in seen: continue` skips empty and whitespace-only findings; `score`
is coerced `float(score) if score is not None else None`; every key is read as `.get(...) or <default>`.

The plan's version uses `key = (raw["category"], raw["text"], raw["source"])` and direct indexing.
Three regressions:

1. **Case and whitespace dedupe is gone.** "Jane Doe" from Presidio and "jane doe " from the rules
   cascade now count as two findings, which changes `_compute_detection_confidence`'s agreement
   numerator and denominator — the value `pii_detection_test.py:27-56` exists to protect.
2. **The empty-text guard is gone**, and it matters *more* now: `ScriptLine` accepts `text=""` (I
   constructed one), so an empty finding is materialisable and inflates `n_spans`.
3. **`raw["text"] is None` now raises.** Verified: `PiiSpan(category=…, source=…, asr_model=…)` with
   neither `text` nor `speaker` raises `ValidationError` (`script_line.py:140-141`). The old code
   skipped such a row. All current producers do set both keys
   (`subprocess_backend.py:430,466,511`, `local_llm.py:166`), so this is latent rather than live —
   but it converts a skip into a crash inside `scan_for_pii`.

Keep the normalisation, the guard and the `.get` defaults.

---

## Medium severity

### F-12 — Task 5 max-normalises its output, the exact property Task 3's tests forbid two tasks earlier

`gammatone_filterbank` returns `db - db.max()`; the docstring says "``energy_db`` is relative to the
bank's own maximum."

Task 3 ships `TestEnvelopeIsAbsolute::test_scaling_the_input_shifts_the_envelope_by_the_same_amount`
with the message `"dBFS is absolute, not max-normalised"`. `benchmarks/spans.md` locates a rejected
design's fault in the same place: "The last row locates the fault: **normalising the envelope by its
own maximum**, which lets one loud sample rescale everything." And
`audio_analysis/acoustic.py:1-18` is an entire module docstring about per-recording normalisation
turning a level into a rank. `preprocess.md`'s gammatone row specifies "40 ERB channels, 80–7800 Hz,
5 ms hop" and says nothing about normalisation.

**Correction.** Return absolute dBFS; let a renderer normalise. (Task 5's four tests do pass as
written — ran: 40 channels, `cf[0]` 80.0, `cf[-1]` 7800.0, loudest channel 1049 Hz for a 1 kHz tone,
shape `(24, 200)`.)

### F-13 — Task 9: `fold_file_verdict` misses two of `verdict.md`'s rows and invents a third

1. **`state is None` is not handled.** `verdict.md`: "present or undecided | **never ran** | **flag**
   — a kind the graph was asked about has no answer". The code flags only when
   `ran.get(node) in (RunState.SKIPPED, RunState.ERRORED)`. A node simply absent from `ran` gives
   `state = None`, falls through the `if verdict is None` branch and contributes nothing. This is the
   exact collapse `verdict.md` warns about ("a graph that skipped a node for an operational reason …
   would otherwise be indistinguishable from one that looked and found nothing"). No test covers it,
   and `test_every_kind_absent_is_a_different_fail_from_admit` passes `ran={}` and depends on the
   silence.
2. **A branch verdict for an unpredicted kind is discarded.** The loop iterates `kind_predictions`,
   so a `NodeVerdict` whose `kind` TAXONOMY never mentioned produces no contradiction, no
   resolution, and never enters `kinds` — while `verdict.md`'s Product declares all three kinds.
3. **`undecided` + `flag` silently resolves to `ABSENT`.**
   `elif predicted is KindState.UNDECIDED: kinds[kind] = PRESENT if verdict.outcome is PASS else ABSENT`.
   `verdict.md`'s table gives undecided+pass → present and undecided+fail → absent, and says nothing
   about undecided+flag. Resolving a *flagged* branch's kind to absent is an invented decision, and
   `kinds` is a reported product.

### F-14 — Task 9: the release fold is not implemented and not tested

`verdict.md`'s "## The release fold" derives `release` from REDACT: did not run → `not_assessed`;
returned `fail` → `withheld`; returned `pass` → `releasable`. `fold_file_verdict` takes
`release: Release = Release.NOT_ASSESSED` and passes it through untouched, so the fold lives in the
caller and **none of the nine tests exercises `release` at all**. It is also the only parameter with
a default among four, so a caller who forgets it gets `not_assessed` silently.

**Correction.** Derive it from the REDACT `NodeVerdict` and `ran["REDACT"]`, and test all three rows —
in particular `verdict.md`'s "`not_assessed` is not `releasable`".

### F-15 — Task 9: `NodeVerdict` and `FileVerdict` omit `view`, which both design documents require

`store.md`, "A node's product is a verdict and a view": `view | the element ids the node authored or
asserted over, so a consumer need not scan the store`. `verdict.md`'s Product:
`view: the verdict element id, and the node verdict ids it folded`. Neither dataclass has a field for
it, so the traceability the append-only store exists to provide cannot be carried through the fold.
Add `view: tuple[str, ...]`.

### F-16 — Task 1's PII location is line-granular; REDACT needs word-granular

`scan_for_pii` flattens the whole `ScriptLine` tree via `flatten_script_line` (`api.py` docstring at
`:186`: "Scanning only `text` would make PII coverage silently depend on which backend produced the
transcript, so the whole tree is flattened"). Step 5 then copies the **line's** `start`/`end` onto
every finding. A name inside a 6 s Whisper segment therefore gets the segment's extent.

`redact.md`: "Word edges come from `alignment` and carry error. A boundary off by 100 ms either leaves
a fragment of a name audible or clips the neighbouring word" — and Task 8's `padding_ms` is sized
against *alignment edge error*, not segment length. Padding a whole segment by 100 ms and silencing
it is a very different artifact from padding a word.

**Correction.** Either locate the finding within the chunk tree (`ScriptLine.iter_leaves`,
`script_line.py:224`, is already the consolidated word walk) and carry the leaf's extent, or state in
the docstring that the extent is the *scanned line's*, so REDACT does not silently redact segments
while its verdict reports word redactions.

### F-17 — Task 8's Interfaces section contradicts Task 1

"Consumes: `PiiSpan` (Task 1) for its `start_s`/`end_s`." Task 1's
`test_a_located_finding_reports_its_extent_and_speaker_natively` asserts
`not hasattr(span, "start_s"), "no parallel name for a field that already exists"`. The fields are
`start`/`end`. Stale text from the pre-subclass draft.

### F-18 — Task 10 is a third clipping detector, with a third threshold and a second run rule

- `src/senselab/audio/tasks/quality_control/metrics.py:146`
  `proportion_clipped_metric(audio, clip_threshold=1.0)`, whose inner
  `is_likely_clipped(channel, min_consecutive=3)` (`:166`) tests `max_consecutive > min_consecutive`,
  i.e. **4 or more**.
- `src/senselab/audio/workflows/audio_analysis/level.py:209`
  `clipped_fraction(waveform, *, threshold=0.999_9)`.
- Task 10: `clip_headroom = 0.999`, `min_clip_run = 3`, tested `>= min_clip_run`, i.e. **3 or more**.

Three headrooms (1.0 / 0.9999 / 0.999) and two run rules for one physical phenomenon. `CLAUDE.md`:
"Thresholds belong in `data/` with a written derivation, never as code literals." The plan's own
docstring says the four values are "conventional rather than fitted", which is both an admission and
(per `CLAUDE.md`) rationale that belongs in `specs/`.

`quality_control/checks.py:205,226` (`high_proportion_clipped_check`, `clipping_present_check`) are
the existing consumers, so a divergent fourth definition will produce two different answers about the
same span.

**Correction.** One headroom, one run rule, in a `data/` profile with the derivation. Task 10's twelve
tests do pass as written — ran: saturated tone 400 runs / 0.675 s; single full-scale sample 0 runs;
4000-sample zero run → 1 dropout / 0.25 s; 20-sample run → 0; step → 1 discontinuity; DC 0.2000;
span scoping correct.

### F-19 — Task 8's merged category string cannot produce `redact.md`'s `by_category{}`

`plan_redactions` joins merged categories as `"PERSON+DATE"`, and the plan tests for exactly that
string. `redact.md`'s verdict is `{ redactions_n, by_category{}, padding_ms, verified: bool,
survived[] }`. A `"+"`-joined string has to be re-split to build `by_category`, and a category
containing `+` breaks it. Use `categories: tuple[str, ...]`.

### F-20 — Task 4: `MIN_SEPARATION_MS` is the one rule parameter that is neither an argument nor exported

`preprocess.md`'s `spans` block lists "minimum separation 150 ms" alongside the four rules Task 4 does
expose as keyword arguments. In the plan it is a module constant read from inside `propose_spans`, and
`__init__.py`'s `__all__` exports `HANGOVER_MS`, `MIN_DURATION_MS`, `OFFSET_FRACTION` and
`ONSET_DROP_DB` but not `MIN_SEPARATION_MS`. Make it a keyword argument like the others.

### F-21 — Task 4 silently resolves an ambiguity `capability-map.md` flags as open

`capability-map.md` §5.2, last item: "**How is a span's `peak_over_floor_db` defined once the floor is
a track?** The floor varies over the span. At the peak, over the span, or at the onset are three
different numbers, and AIRWAY reports it as 'what a reader needs to discount' a span."

The plan takes it at the peak (`peak - float(floor_db[p])`), and the merge branch takes `max()` of two
spans' values — a fourth definition, for the merged span. Record the choice in `benchmarks/spans.md`
rather than leaving it in the code.

### F-22 — Task 4: an event at the very start or end of the recording is invisible

`scipy.signal.find_peaks` never reports index 0 or the last index. A recording that begins mid-cough
yields no span for it, and if it is the only event the function returns `NoContrast` — "the recording
is unmeasurable" — for a recording containing a loud event. Document the limitation or clamp-pad the
`above` array by one sample at each end.

---

## Low severity and test quality

### F-23 — Task 2's two round-trip/merge tests would pass against a store that keeps only ids

`fingerprint()` returns `_digest({"e": sorted(element ids), "a": sorted(assertion ids)})` — ids and
nothing else. So `test_jsonl_survives_a_round_trip` and `test_merging_in_any_order_gives_the_same_store`
— the only coverage for `write_jsonl`, `read_jsonl` and `merge` — cannot detect a round trip that
lost `evidence`, `extent`, `author`, `verb`, `value` or `model`. That matters concretely because
`write_jsonl` serialises with `json.dumps(..., default=str)`, which turns a numpy float in `evidence`
into the string `"31.4"` rather than failing. Assert on the reconstructed records.

The merge test is weak for a second reason: `fingerprint()` sorts, so it tests `sorted()` rather than
merge semantics. The real order-dependence is F-7's duplicate-`aid` case, which this test cannot see.

### F-24 — `test_nothing_can_be_deleted` asserts nothing about behaviour

`assert not hasattr(s, "delete_element")` / `"remove_assertion"` passes against an empty class. The
invariant it is reaching for — an existing id is never replaced — is F-7, and is untested.

### F-25 — `test_a_model_authored_element_requires_a_resolved_sha` does not test the store

The `ValueError` is raised in `ModelProvenance.__post_init__` while the argument list is being
evaluated, before `add_element` is entered. It is a `ModelProvenance` test under a store-shaped name.

Nothing tests `store.md`'s "**Provenance is not optional.** An element or assertion authored by a
model carries the model id and resolved revision" — `model` defaults to `None` and nothing in
`ElementStore` enforces it.

### F-26 — `KIND` and `VERB` are unenforced at runtime

They are `Literal` annotations on frozen dataclasses, so `add_element(kind="cough")` is accepted at
runtime and only mypy would object; `read_jsonl` reconstructs from JSON with no check at all. If the
closed vocabularies from `store.md`'s two tables are load-bearing — and the whole point of a closed
`TimestampSource` at `script_line.py:49-52` is that they are — validate them. A pydantic model would
do it for free, which is also F-6's suggestion.

### F-27 — Task 1's two `scan_for_pii` tests spawn the real subprocess venv

`test_scanning_a_script_line_carries_its_timing_onto_every_finding` and
`test_scanning_a_bare_string_leaves_the_finding_unlocated` call
`scan_for_pii(..., detectors=["rules"])`, which reaches `detect_pii_via_subprocess` and, on a cold
host, builds a Python 3.13 venv with `presidio-analyzer` + `spacy` + `gliner`.

Every existing test in the tree avoids this deliberately: `pii_detection_test.py:160` uses
`detectors=[]`, `:169` uses whitespace input, and `pii_adapter_test.py:104` monkeypatches
(`monkeypatch.setattr(pii_api_module, "detect_pii_via_subprocess", fake_subprocess)` — patched where
`api` resolves it, per that file's module docstring). Do the same; the behaviour under test is the
timing copy, which a fake subprocess exercises exactly as well.

### F-28 — Task 1, Step 6's stated risk does not exist; a different one does

The plan says `pii_adapter_test.py` "is the one that might" assert the exact key set of a serialised
span. It does not: `src/tests/audio/workflows/pii_adapter_test.py:102-104` asserts only that
`payload["spans"]` is truthy and that every span carries `perturbation` and `asr_model`. No test in
the tree asserts an exact key set. I checked every reader of the serialised shape.

The real change is that `model_dump(exclude_none=True)` **drops** `score` when it is `None`, where
`asdict` always emitted `"score": None`. `global_summary.py:359-382` reads `s.score` as an attribute
and is unaffected, but any consumer of `pii.json` that indexes `span["score"]` would now `KeyError`.
Either say so in the step or use `exclude_none=False`.

### F-29 — Task 2's `read_jsonl` will not pass Task 9 Step 5's mypy run

```python
store._elements[rec["id"]] = Element(extent=tuple(extent) if extent else None, model=prov, **rec)
```

`tuple(extent)` is `tuple[Any, ...]`, not `tuple[float, float] | None`. Write
`(float(extent[0]), float(extent[1]))`. (`Iterable` is imported and never used; that is only not an
error because `ignore = ["F401"]`.)

### F-30 — Rationale prose in code, which the plan's own Global Constraints forbid

`CLAUDE.md`: "Docstrings and comments say what a thing is and how to call it; the measurement behind a
choice, the failure that drove it and the rejected alternatives go in `specs/`." The plan restates
this and then violates it in eight places:

- Task 2 — `ModelProvenance.__post_init__` error: "Recording a ref makes the provenance confidently
  wrong."; `merge` docstring: "Append-only means no element is ever mutated, so the union is
  order-independent."; module docstring: "so merging two stores is a set union and is
  order-independent."
- Task 4 — `NoContrast`: "Distinct from an empty span list: an unmeasurable recording must not read as
  a quiet one."; `k_db`: "No default: the value is per-reader and unmeasured across readers."
- Task 6 — docstring: "A span shorter than that is placed in a silent buffer so the model sees the
  span and silence, never a neighbouring event."
- Task 7 — `f0_min_hz`: "No default: no single range serves both low adult and infant voices."
- Task 8 — `padding_ms`: "it must exceed the worst edge error of whatever produced the extents, and
  that error is unquantified."
- Task 10 — the module docstring's second paragraph, and `detect_disruptions`'s "The four parameters
  are conventional rather than fitted: a single sample at full scale is not clipping, which is why
  ``min_clip_run`` exists, and the values are the usual ones rather than values derived from labelled
  verdicts."

An error *message* that tells the caller what to do next is legitimate. The justification for the
design decision is not — it belongs in `benchmarks/`. Note that the tree still carries a lot of this
(`acoustic.py:1-18`, `sources.py:275-301`); `CLAUDE.md` says to move it out when editing, not to
match it.

### F-31 — Four code lines exceed 120 characters

Plan lines 520 (124), 956 (124), 1939 (126), 1944 (122). `ruff format`, which every task runs before
committing, will rewrap all four, so this only bites if that step is skipped.

### F-32 — Task 3's `FLOOR_PERCENTILE` collides with an existing exported constant of the same name and different correction

`audio_analysis/acoustic.py:116` exports `FLOOR_PERCENTILE = 10.0` (same value, and in `__all__` at
`:34`), used by `level_above_floor_track` at `:165`:

```python
floor_db = float(np.percentile(levels_db, float(floor_percentile))) + _FLOOR_BIAS_DB
```

with `_FLOOR_BIAS_DB = 9.8` at `:119` — a bias correction for taking a low percentile of a noise
distribution. Task 3's `rolling_floor_dbfs` omits any such correction, so `envelope − floor` from
Task 3 and `excess_db` from `acoustic.level_above_floor_track` differ by ~9.8 dB despite carrying the
same name and the same percentile. Task 3 is the one consistent with `benchmarks/spans.md` and
`floor.py` (both uncorrected), so the fix is naming and a note, not a value change — but two
exported `FLOOR_PERCENTILE`s on different scales is a trap.

### F-33 — Task 4's `k_db` docstring overstates the design's silence

`preprocess.md`'s spans parameter table gives `K` = **18 dB**, scope "per reader; AIRWAY reads at this
setting", and `branch-airway.md`'s element table reads spans "at `K` = 18 dB".
`benchmarks/spans.md`'s "Propose threshold `K`" table measures 18, 12 and 8 dB and concludes "18 dB
for AIRWAY … and 12 dB for SPEECH". No-default is still correct (it is per-reader), but "the value is
per-reader **and unmeasured across readers**" reads as "nobody has a value", which is not what the
benchmark says. `K` is also not on `open.md`'s undecided list, and correctly not in the plan's
Global Constraints list.

### F-34 — Inherited `ScriptLine` surface on `PiiSpan`, two minor footguns

Verified by construction:

- `PiiSpan.from_dict({"text": "x"})` raises `ValidationError` — the inherited classmethod cannot
  supply `category`/`source`/`asr_model`. `ScriptLine.from_dict` is used widely
  (`script_line.py:249`), so this is a live footgun; override or document it.
- `repr()` is inherited: `repr(span)` renders `SPEAKER_00: Jane [1.00 - 1.30]`, dropping `category`
  and `source`. Test failure output on a PII assertion will not say which category was found.

---

## Checked and genuinely fine

Everything below I verified directly; the controller can treat it as cleared.

**Task 1's central move works.** I constructed `class PiiSpan(ScriptLine)` with the plan's three added
fields against the real `senselab.utils.data_structures.ScriptLine`:
`isinstance(span, ScriptLine)` True; `start`/`end` default `None`; `hasattr(span, "start_s")` False;
`model_dump(exclude_none=True)` → `{'text','score','category','source','asr_model'}`, and with timing
→ `{'text','speaker','start','end','timestamp_source','timestamp_model','category','source','asr_model'}`.
`ScriptLine`'s "at least text or speaker" validator (`script_line.py:117-142`) accepts every
construction the plan makes, and rejects the one it should. `ScriptLine` does carry `text`, `speaker`,
`start`, `end`, `score`, `chunks`, `timestamp_source` and `timestamp_model`
(`script_line.py:91-115`), so the subclass-not-compose decision is right and the fields are genuinely
not duplicated.

**Every `PiiSpan(...)` construction in the tree is keyword-only** and passes only inherited-or-kept
fields, exactly as the plan claims: `text/tasks/pii_detection/api.py:358`;
`src/tests/text/tasks/pii_detection_api_test.py:27,99,142,162,200`;
`src/tests/text/tasks/pii_detection_test.py:33,34,37,47,48,51,64,74,75,83,84,87,101,199,200`.

**`asdict` on a `PiiSpan` happens in exactly one place**, `audio_analysis/pii.py:275`. The only other
`asdict` calls in the tree are `utils/compatibility_test_runner.py:89` and
`audio_analysis/votes.py:513`, neither on PII. So one line plus removing the `asdict` import from
`pii.py:32` is genuinely enough. (I confirmed the failure mode: `dataclasses.asdict` on a dataclass
holding pydantic models leaves the models untouched, so `json.dumps` would fail — the plan's fix is
the right one.)

**`global_summary.py:359-382`** reads span fields by attribute (`s.category`, `s.text`, `s.asr_model`,
`s.score`, `s.source`), so it survives Task 1 with no edit.

**`Audio` interop.** `waveform` is always 2-D `(channels, samples)` — `convert_to_tensor`
(`audio.py:184-206`) unsqueezes 1-D input and casts to float32. Each of the plan's access patterns
works, run against the real class: `np.asarray(audio.waveform, dtype=np.float64)`,
`np.asarray(audio.waveform, dtype=np.float32)`, `audio.waveform.numpy().copy()`, and
`Audio(waveform=<2-D float32 ndarray>, sampling_rate=…)`.

**Task 3 (envelope) passes all five tests as written.** Ran: half-scale tone median −6.021 dBFS
(window −7.5…−4.5); ×0.1 gives a 20.000 dB shift; the injected 30 ms click moves the distant median by
**0.045 dB** against a 0.5 dB tolerance; the rolling floor tracks −60 → −30 with `window_s=1.0`; one
floor value per sample. The `filtfilt`/`hilbert`/`butter` composition and the `np.interp` step
reconstruction are all correct.

**Task 5 (gammatone) passes all four tests as written.** Ran: `erb_space` gives 40 ascending
centres, `cf[0]` 79.9999, `cf[-1]` 7800.0002, top spacing 643 Hz against 5× bottom spacing 135 Hz;
a 1 kHz tone excites the channel at 1049 Hz; shape `(24, 200)` for 2 s at a 10 ms hop.
`scipy.signal.gammatone` is available in the pinned environment. There is no in-tree gammatone to
reuse — the only hit, `utils/clearvoice.py:102`, is the `gammatone` PyPI package listed as a
ClearVoice subprocess-venv requirement, not a senselab implementation.

**Task 8 (redaction) is correct as written.** Padding (1.0, 1.2) → (0.9, 1.3); (0.02, 0.1) clamps to
0.0; (0.9,1.2) + (1.15,1.45) merges to one; silence-in-place with duration preserved (verified the
numpy/torch round trip on a real `Audio`). And the gap is real: nothing in senselab redacts audio
today — `capability-map.md` line 119's MISSING verdict is accurate, and
`data_augmentation.api.augment_audios` does not do targeted extents. `padding_ms` is correctly
keyword-only with no default.

**Task 10 (disruptions) passes all twelve tests as written** (numbers in F-18). The `_runs` helper is
correct at both boundaries, including all-True and alternating masks, which I checked by hand.

**Task 9's nine tests all pass** against the implementation as written — I traced each one through
the fold by hand and confirmed the branch each takes. The `VERB` and `KIND` literals match
`store.md`'s two tables term for term. `Outcome`, `KindState`, `RunState` and `Release` are genuinely
new: the only enums in reach are `DeviceType` (`utils/data_structures/device.py:26`), `Arity`
(`audio_analysis/keys.py:214`), `GridRelation` (`audio_analysis/shapes.py:58`), `Stage`
(`audio_analysis/stage_io.py:103`) and `Label` (`quality_control/review.py:36`, an IntEnum of review
labels). None is a pass/flag/fail verdict, and nothing in `quality_control` carries one either.

**Module placement.** `utils/element_store.py` does avoid the fan-in the plan cites:
`src/senselab/utils/__init__.py` is a bare docstring, while `utils/data_structures/__init__.py`
eagerly imports dataset, device, estimate, file, language, logging and the whole `model` module.
Test paths mirror correctly for Tasks 2, 3, 4, 5, 7, 8, 9 and 10 (only Task 6's is wrong — F-8).

**`src/tests/audio/tasks/task_layer_guard_test.py:52`** ("nothing under `audio/tasks/` may import
from `audio/workflows/`") is not tripped: none of the five new task packages imports a workflow, and
the triage vocabulary correctly lives under `workflows/`.

**Undecided parameters — five of six correct.** `benchmarks/open.md` names six. `padding_ms` is
keyword-only with no default ✓. The disruption *tolerance* is correctly absent — Task 10 emits counts
and extents only, which is what `branch-speech.md` (lines 174-193) specifies ✓. SQUIM thresholds, the
word-gap threshold and `min_families` are not implemented in this plan at all, so nothing sneaks a
default in ✓. The F0 search range is keyword-only with no default ✓ (`capability-map.md` §5.1, last
row). The one violation is `periodicity_floor` — F-4.

**`CLAUDE.md` traps.** No `pytest -n auto` anywhere; every Python command is `uv run`; `uv sync` is
never invoked (so the `--all-extras` trap is not reachable); no task loads a model, so there is no
ref-vs-SHA load to get wrong and no `CACHE_SCHEMA_VERSION` bump owed. The revision-pinning guard
(`src/tests/utils/revision_pinning_guard_test.py`) sweeps subprocess-worker files only, and the plan
adds none. `ModelProvenance`'s SHA check is a *recording* discipline, not a load — the load-time trap
does not apply, though F-6's naming does.

**Prior art I checked and judged adjacent rather than duplicated** (worth knowing so nobody
re-litigates it):

- `audio_analysis/acoustic.py:127` `level_above_floor_track` and `audio_analysis/l1_plot.py:44`
  `rms_dbfs_track` are per-frame dBFS / level-above-floor tracks. `preprocess.md` specifies the
  analytic-signal envelope at sample resolution, which neither provides, so Task 3 is not a
  duplicate — but see F-32 on the constant name.
- `audio_analysis/sources.py:421` `resolve_extent` is the closest thing to Task 4: a strict margin
  decides presence, a looser one grows the extent, and it returns `None` when nothing establishes
  presence. It returns exactly **one** extent, anchored on the single strongest frame, so it cannot
  serve multi-event span proposal. Not a duplicate. The parameter vocabulary should still align —
  `speech_presence_margin_db` ≈ `k_db`, `extent_margin_db` ≈ `onset_drop_db`.
- `audio_analysis/background_mask.py:60` `BackgroundMaskRegion` and `sources.py:259`
  `ExcisedSegment` both use `start`/`end` field names, which `Span` and `RedactionExtent` match ✓.
  `sources.py:275` `plan_excision` plans padding for a fixed classifier window, which is close to
  Task 6's job but keyed on mask rows rather than a span — not reusable, but the naming precedent
  (`plan_*`) supports putting Task 6's function beside `plan_centred_windows`.
- `audio/tasks/preprocessing/silence_segmentation.py:51` `_rms_envelope` plus `_pause_candidates`
  (`:77`) is a percentile-gated RMS envelope segmenter, but for pause-aware chunking, not event
  proposal. Not a duplicate.
- `audio_analysis/noise_floor.py:134` `estimate_band_floor_db` is per-band with bias correction;
  Task 3's is broadband. `noise_floor.py:1-30`'s own docstring draws that boundary. Not a duplicate.

**Categories with nothing to report.** *Tests asserting on a mock rather than on behaviour:* none —
the plan uses no mocks at all, which is itself F-27's problem (it needs one). *A model load passing a
ref, `pytest -n auto`, `uv sync` without `--all-extras`, a missed cache-schema bump:* none, as above.

---

## Where I am unsure, and what would settle it

- **F-2's downstream reach.** I measured that the plan's estimator caps at `1 − lag/N` and that this
  puts 0.933 out of reach at `f0_min_hz = 60`. What I could not determine is which estimator produced
  `voice.md`'s 0.933/0.934 — `benchmarks/scripts/` is unrunnable (F-5), and `voice.md` does not say.
  Settled by: naming the estimator in `voice.md`, or re-measuring the two 200 ms regions with a
  normalised cross-correlation and recording both numbers.
- **F-8's severity.** Whether zero-padding is acceptable depends entirely on which HeAR head consumes
  the buffer, and `benchmarks/hear-yamnet.md`'s detector numbers (Breathe 0.989 / Cough 0.996) say it
  is fine there. I did not run HeAR, so I am relying on `capability-map.md` §4.1 for that reading.
  Settled by: restricting the function's documented use to the detector, or measuring the encoder on
  padded spans directly.
- **F-16's practical impact.** How coarse the line-level extent actually is depends on
  CrisperWhisper's segment lengths on the reference file, which I did not measure. Settled by:
  reporting the segment-duration distribution for one real run.
- **F-18's correct headroom.** I established that three values exist and disagree; I did not
  establish which is right. Settled by: one measurement of a known-clipped recording against all
  three.
