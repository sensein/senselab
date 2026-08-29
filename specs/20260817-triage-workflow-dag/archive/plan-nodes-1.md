# Triage Node Implementation Plan — nodes 1–4: ADMIT, PREPROCESS, TAXONOMY, AIRWAY

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first four nodes of the audio-triage workflow — ADMIT, PREPROCESS, TAXONOMY,
AIRWAY — over the merged foundation (`ProvStore`, `TriageConfig`, the vocabulary, and the DSP tasks
`envelope`, `spans`, `gammatone`, `disruptions`).

**Scope split:** A sibling plan (`plan-nodes-2.md`) covers SPEECH, VOICE, REDACT and VERDICT. This plan
does not describe those nodes beyond the store schema they consume, which §"What the sibling plan's
nodes read" states precisely. (An earlier dead single-file attempt, `plan-nodes.md`, has since been
removed from this directory; nothing depends on it.)

**Architecture:** Each node is one module in `src/senselab/audio/workflows/triage/nodes/` (a new
package), taking the store, an input (a file path, an `Audio`, or a store-held stream name), the
config, and an optional hint; writing entities/activities/agents/relations to the store; and returning
a `NodeResult` carrying a `NodeVerdict` plus whatever the node's design product section names — never a
copy of the store's content. No orchestrator is built here — orchestration lives in
`specs/20260817-triage-workflow-dag/nextflow/`.

**Tech Stack:** Python 3.12, pydantic v2, numpy, scipy, soundfile, pytest. uv for everything.

**Design source of truth:** `specs/20260817-triage-workflow-dag/` — `store.md` first, then `admit.md`,
`preprocess.md`, `taxonomy.md`, `branch-airway.md`. `capability-map.md` maps requirements to existing
code (evidence, not gospel — §"Corrections to capability-map.md" below lists where this plan found it
stale). `benchmarks/open.md` lists what is deliberately unmeasured; **supplying a value for any item in
it is wrong.**

## Prerequisite, now satisfied: HeAR is on the merged tree

The `triage` merge has happened. The tree this plan executes on (commit `33bf65ad`) carries
`senselab.audio.tasks.health_acoustics` in full, **including the whole-span buffer helper**
`span_to_hear_buffer` and the model-imposed constant `HEAR_WINDOW_SECONDS`
(`src/senselab/audio/tasks/health_acoustics/hear.py:387,130`), so **Task 4 calls the module function
and inlines nothing**. Verified against the merged tree:

- `senselab.audio.tasks.health_acoustics.api.detect_health_acoustic_events(audios: List[Audio], model: str = "hear-event-detector", device: Optional[DeviceType] = None, hop_length: float = 0.25, top_k: Optional[int] = None) -> List[List[Dict[str, Any]]]` — per audio, per-window dicts with `start`, `end`, `label_scores` (descending single-key dicts over the eight `HEAR_EVENT_LABELS`), `win_length` (2.0), `hop_length`. Raises `ValueError` on audio shorter than 2 s at 16 kHz. `top_k=None` keeps all eight (`health_acoustics/api.py:228`).
- `senselab.audio.tasks.health_acoustics.hear.HEAR_MODEL_ID = "google/hear"`, `HEAR_REVISION` = a 40-hex commit literal, `HEAR_EVENT_LABELS = ("Cough", "Snore", "Baby Cough", "Breathe", "Sneeze", "Throat Clear", "Laugh", "Speech")`.
- `span_to_hear_buffer(audio: Audio, start_s: float, end_s: float, *, placement: str = "centre") -> Audio` — places the whole span in a silent buffer of exactly `HEAR_WINDOW_SECONDS` at the input's rate; **raises `ValueError` on a span longer than the window** ("Split it or classify a sub-span") and on an unknown placement (`"centre"`, `"start"`, `"end"` exist). For the event detector only, never for embeddings.
- The existing `src/tests/audio/workflows/triage/config_test.py` already pins `hear.window_s == HEAR_WINDOW_SECONDS`, so the config value and the model constant cannot drift.

## Global Constraints

Copied verbatim from the foundation plan; every one is binding here too.

- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Never run `pytest -n auto`.** Each xdist worker duplicates ~535 MB of frameworks. Run the directory you changed.
- Tests live in `src/tests/` mirroring the package, named `*_test.py`.
- Google-style docstrings; line length 120; type hints required (mypy with the pydantic plugin). ruff applies to test code, so **every test class and function needs a docstring and every test function `-> None`**.
- **Rationale does not go in code.** Docstrings say what a thing is and how to call it. Measurements and rejected alternatives go in `specs/20260817-triage-workflow-dag/benchmarks/`.
- **No numeric constant appears in code.** Not as a signature default, not as a module-level constant. Every number lives in `data/config/default.yaml` with its derivation beside it, per `CLAUDE.md`: "Thresholds belong in `data/` with a written derivation, never as code literals." Functions take the values they need as arguments; the caller reads them from the config.
- **A value nobody has measured is `null` in the config, and reading it raises.** The loader names the parameter and points at `benchmarks/open.md`. This replaces keyword-only-without-default as the mechanism for making an unmeasured value impossible to use by accident.
- A model load passes a **resolved 40-hex commit SHA, never a ref**. There is an AST-sweep guard test (`src/tests/utils/revision_pinning_guard_test.py`).
- `uv sync` is subtractive — always pass `--all-extras`.
- Run `ruff format` before every commit.
- **Pre-alpha: rename and replace outright.** No parallel fields, no aliases, no deprecation shims.
- **A plan (or an implementation) supplying a value for anything in `benchmarks/open.md` is wrong.** Those keys stay `null`; tests exercise them through explicit YAML overrides, which is the intended production mechanism too.

Two clarifications of the no-literal rule, applied throughout this plan:

- **Definitional constants are not thresholds.** `20 · log10` (the definition of dB), full scale `1.0`
  (the definition of dBFS), and floor clamps like `1e-12` already appear in the merged foundation
  (`envelope/api.py`, `gammatone/api.py`); this plan uses them the same way and nowhere else.
- **Test fixtures may use literals.** The rule binds production code; a test constructing a 150 ms
  burst at 440 Hz is describing its fixture, not deciding a threshold.

## The node contract — defined here in Task 1, reused by every node in both plans

```python
def <node>(
    store: ProvStore,
    source: <per-node input — see table>,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> <Node>Result: ...
```

| node | `source` | returns |
| --- | --- | --- |
| ADMIT | `str | Path` — the recording as supplied | `AdmitResult(verdict, view, verdict_entity_id, audio)` |
| PREPROCESS | `Audio` — the audio ADMIT returned | `PreprocessResult(verdict, view, verdict_entity_id, absent)` |
| TAXONOMY | `str` — a store-held stream name, `"plain"` | `TaxonomyResult(verdict, view, verdict_entity_id, kinds)` |
| AIRWAY | `str` — a store-held stream name, `"plain"` | `AirwayResult(verdict, view, verdict_entity_id, figure_path)` |

- `verdict` is a `vocabulary.NodeVerdict` (`node`, `Outcome`, `kind`, `why`). The design's richer
  per-node verdict mappings (`by_label{}`, `kinds{}`, …) are written to the store as attributes of one
  `verdict` entity per node, which is what the sibling plan's VERDICT node reads.
- `view` is a tuple of store entity ids the node wrote or asserted over — the design's "named view".
- `run_dir` is where sidecars live: `streams/` (WAV), `derivatives/` (npz/json), `figures/` (png).
  `ProvStore` serialises attributes to JSON, so tracks and spectrograms are sidecar files; the entity
  carries the run_dir-relative path. **The caller persists the store itself** (`store.write_jsonl`)
  — into a directory that is never a release directory, because word and transcript entities carry
  PII (store.md §"It holds PII"; the consequence lands on the sibling's REDACT, the obligation to
  keep transcript text out of error messages and figures lands on every node here too).
- **Mocking boundary (binding for every task):** each node module imports its model-calling functions
  **by name at module top**; tests monkeypatch **on the node module**
  (`monkeypatch.setattr(node_module, "transcribe_audios", fake)`), following
  `src/tests/audio/workflows/pii_adapter_test.py`'s rule of patching where the callee resolves the
  name. Model *constructors* that resolve a commit over the network (`HFModel(...)`) are reached only
  through module-level factory functions (`_crisperwhisper_model()`, …) so tests can monkeypatch those
  too and never touch the Hub. No test loads YAMNet, AST, HeAR, CrisperWhisper, Qwen, an aligner or
  SQUIM. Pure DSP (envelope, spans, gammatone, resample, `fuse_word_streams`, `integrated_lufs`,
  spectrogram extraction, plotting) runs real.

## What the sibling plan's nodes read — the store schema contract

SPEECH, VOICE and VERDICT consume what these four nodes write. This section is the contract; the
sibling plan must not re-derive it, and a change to it is a change to both plans. All paths are
run_dir-relative unless stated. Every entity below `wasGeneratedBy` its step activity and
`wasAttributedTo` the agent answerable for it (a model agent for model output, the software agent
otherwise); every store read is recorded with `used`.

**`stream` entities** (written by ADMIT and PREPROCESS):

| `name` | attributes | notes |
| --- | --- | --- |
| `recording` | `path` (absolute, the supplied file), `sampling_rate`, `channels` | as supplied; ADMIT writes it on `pass` |
| `plain` | `path` = `streams/plain.wav`, `sampling_rate` = `resample.target_hz`, `channels` = 1, `peak_scale` (float; `1.0` when no overshoot) | mono-averaged, resampled; the signal every model reads |
| `preemphasised` | `path` = `streams/preemphasised.wav`, `sampling_rate`, `channels` = 1, `coefficient` | absent when `preemphasis.enabled` is false |

All carry `extent = (0.0, duration_s)`. `plain` is `wasDerivedFrom` `recording`; `preemphasised` from `plain`.

**`measurement` entities** (PREPROCESS derivatives; each carries `name` and `signal`):

| `name` | attributes beyond `name`/`signal` |
| --- | --- |
| `energy_envelope` | `path` (npz, keys `envelope_dbfs`, `floor_dbfs`), `sampling_rate` |
| `yamnet_windows` | `path` (json: the exact `classify_audios(..., model="yamnet", top_k=<yamnet.top_k>)` window list — `start`, `end`, `label_scores`, `win_length`, `hop_length`), `n_windows` |
| `silence` | `threshold`, `windows`: `[{start, end, score, is_silence}, ...]` |
| `level` | `peak_dbfs`, `rms_dbfs`, `lufs` — file-level only |
| `asr_crisperwhisper` | `recognizer`, `transcript` (**PII**), `word_ids`, `timestamp_source` = `"native"` |
| `asr_qwen` | `recognizer`, `transcript` (**PII**), `word_ids`, `timestamp_source` = `"bundled_aligner"`, `timestamp_model` = `"Qwen/Qwen3-ForcedAligner-0.6B"` |
| `asr_agreement` | `words`: the verbatim `fuse_word_streams` output (`text`, `start`, `end`, `confidence`, `existence_confidence`, `temporal_confidence`, `coverage`, `corroboration`, `member_agreement`, `member_corroboration`, `sources`, `alternates`, `flags`, `speaker?` — `speech_to_text_ensemble/api.py:283-289,452-472`), `systems` |
| `alignment` | `path` (json: aligned `ScriptLine.model_dump()` list), `language`, `transcript_source` = `"asr_agreement"` |
| `spectrogram_wideband`, `spectrogram_narrowband` | `path` (npz, key `spectrogram`), `win_length`, `hop_length`, `n_fft` (samples) |
| `gammatone` | `path` (npz, keys `centre_frequencies_hz`, `energy_db`), `hop_s` |
| `spans_no_contrast` | `k_db`, `reason` — written **instead of** span entities when `propose_spans` returns `NoContrast`; a reader checks `k_db` against its own `K` |

A derivative that could not be computed is **absent** — no error entity, no placeholder. The
PREPROCESS verdict entity's `absent` attribute lists the missing names with the exception **class**
only (never the message: an ASR error message is not controlled vocabulary and may quote audio content).

**`span` entities** (PREPROCESS): `extent = (start, end)`, attributes `{peak_over_floor_db, k_db,
signal}`. **Never a label key.** AIRWAY labels them by assertion; SPEECH derives its own spans from
word timings and does not read these (`spans.k_db` has only the `airway` entry).

**`word` entities** (PREPROCESS, one per recognizer word): `extent = (start, end)`, attributes
`{text (PII), score, recognizer, timestamp_source, timestamp_model?}`. SPEECH invents no words: the
consensus `word` entities it authors (per `branch-speech.md`'s product table) are `wasDerivedFrom`
the `asr_agreement` measurement — their confidences come from the fusion, never from a third
recognizer — and its withdrawals of diarizer segments reference ids, not text.

**`kind` entities** (TAXONOMY, exactly three): attributes `{kind, state, families, min_families}`
where `kind ∈ {"airway", "speech", "voice_no_words"}`, `state ∈ {"present", "absent", "undecided",
"not_screened"}`, `families` maps family name → `{state, members: {detector: evidence}}`, and
`min_families` is the configured integer or the string `"unmeasured"`.

**`assertion` entities** (AIRWAY, plus PREPROCESS's `measure`): attribute `verb` plus per-verb fields —

| `verb` | derived from | attributes |
| --- | --- | --- |
| `label` | the `span` entity | `label`, `score`, `scores` (all labels of interest), `input` (`"buffered"` \| `"sliding"`), `in_certified_silence` (bool \| None) |
| `confirm` / `contest` | the `label` assertion **and** the span | `winner`, `coverage`, `n_windows`, `mapped_to` |
| `abstain` | the `label` assertion and the span | `best_coverage`, `n_windows` — a label with neither confirm, contest nor abstain does not exist; abstain is recorded so "single-source" is a store fact, not an inference from silence |
| `flag` | the `interval` entity | `reason` = `"lexical_contamination"`, `word_ids` (ids only — **never word text**) |
| `measure` | the `span` entity | `name` = `"squim"` with `stoi`, `pesq`, `si_sdr`, or `unmeasured` = exception class (written by PREPROCESS, one per span) |

**`interval` entity** (AIRWAY): `name` = `"airway_labelled_interval"`, `extent` = first labelled span
start → last labelled span end.

**`verdict` entities** (one per node): `{node, outcome, kind, why}` plus the node's design-named
verdict fields (`ADMIT`: `stream`; `PREPROCESS`: `absent`, `derivatives`; `TAXONOMY`: `kinds`;
`AIRWAY`: `labelled_n`, `by_label`, `contested_n`, `flags`). VERDICT reads these; it never re-reads
elements to reconstruct an outcome.

**Agent conventions:** `HFModel`-loaded models → `agent(agent_type="model", model_id=<path_or_uri>,
commit_sha=<model.commit_sha>)` (resolved at construction). HeAR → `commit_sha = HEAR_REVISION`.
YAMNet → `model_id="https://tfhub.dev/google/yamnet/1"`, `unresolved_reason="TF-Hub URL pin; no
commit exists to resolve"`. SQUIM → `model_id="torchaudio SQUIM_OBJECTIVE"`,
`unresolved_reason="bundled torchaudio weights, version <torchaudio version>"`. The wav2vec2 aligner →
`model_id` from `forced_alignment.constants.DEFAULT_ALIGN_MODELS_HF[<language>]`,
`unresolved_reason="align_transcriptions loads its aligner internally; the commit is not reported to
the caller"`. Everything else → the software agent (`agent_type="software"`,
`version="senselab <installed version>"`).

## Under-specified points, resolved by this plan

Each is a **decision this plan makes** where the design admits more than one implementation
(`capability-map.md` §5) or where two design files pull apart. An implementer must not silently
re-decide one; changing one means changing the plan (and the sibling plan where the store schema is
touched).

| # | point | decision |
| --- | --- | --- |
| N1 | `preprocess.md` resamples but says nothing about channels; every downstream model is mono | PREPROCESS mean-averages channels to mono before resampling, recorded in the conditioning activity's parameters |
| N2 | "guard against overshoot past full scale" names no mechanism | after resampling, a peak above full scale scales the waveform down by that peak; the scalar is recorded as `peak_scale` on the `plain` stream entity (`1.0` otherwise) |
| N3 | `preprocess.md` lists `alignment` ("forced alignment of the **agreed** transcript") as a PREPROCESS derivative, but agreement is a SPEECH concern | PREPROCESS runs `fuse_word_streams` over both recognizers' words, writes the fused list as derivative `asr_agreement`, and aligns the fused text — the derivative exists where the design puts it; SPEECH reads `asr_agreement` rather than re-fusing |
| N4 | the alignment's language is unstated | English (`Language(language_code="en")`), recorded on the alignment activity and entity. When a corpus needs another, it arrives as a config key with a derivation — not a guess here |
| N5 | `squim` is "per span" before speech spans exist | measured over PREPROCESS's own envelope spans (what `benchmarks/squim.md` measured), one `measure` assertion per span; a span SQUIM rejects (it re-raises on short input) records `unmeasured` with the exception class — **no padding**, padding changes the measurement |
| N6 | `preprocess.md` names only `silence` from YAMNet, but TAXONOMY and AIRWAY both need YAMNet's full native windows | PREPROCESS stores the full windowed output as derivative `yamnet_windows` (one model run, three consumers — admitted by `store.md`'s rule that a derivative needs provenance, not a declared consumer); `silence` is a projection of it |
| N7 | spectrogram `n_fft` is undetermined by the design | `n_fft = win_length` — zero-padding adds rendering density, not resolution; a derived identity, not a constant |
| N8 | `no_contrast` is a property of a `(K, recording)` pair, not of PREPROCESS | the `spans_no_contrast` measurement carries the `k_db` it was found at; AIRWAY's `fail` reads it only at its own `K` |
| N9 | `taxonomy.min_families.*` is `null` (unmeasured, `benchmarks/open.md`) yet the node must run | **absence needs no value** (unanimity of eligible families, per the design). While `min_families[kind]` is null, presence is declared only on **unanimity of eligible families** — the one condition every legal value (`1 ≤ v ≤ n_eligible`) agrees on — and the kind entity records `min_families: "unmeasured"`. Any other split is `undecided`. With a config override the design's rule runs verbatim; an override outside `[1, n_eligible]` raises `ValueError`. The node never calls `require()` on these keys, so it runs on the packaged config and reports honestly rather than erroring |
| N10 | TAXONOMY's within-family fold for family A (YAMNet + AST) | members agree → that state; disagree → the family is unsure; a member whose presence floor is `null` (AST ships unmeasured) **abstains** and is recorded, leaving the family to its voting member; both unavailable → unsure |
| N11 | which labels can express each kind is nowhere written | config **lists** (semantic vocabularies, not thresholds): `taxonomy.audioset_airway_labels`, `taxonomy.audioset_speech_labels`, `taxonomy.hear_airway_labels`, `taxonomy.lexical_airway_tokens` — each with a derivation naming it a vocabulary mapping read off the detectors' label inventories, overridable |
| N12 | `taxonomy.md` gives airway **three** eligible families, but `benchmarks/taxonomy.md` says CrisperWhisper's non-lexical labels are unreliable | the lexical family votes airway via bracketed non-lexical tokens anyway — the design's family counts govern, and the fold's agreement requirements are exactly the protection against one unreliable family; the token vocabulary is `taxonomy.lexical_airway_tokens` |
| N13 | HeAR buffer placement (centre / left / right) is unstated in `branch-airway.md` | **centred**, via config `hear.placement` — the benchmark numbers (`benchmarks/hear-yamnet.md`) were measured under centred placement. AIRWAY calls `span_to_hear_buffer` (merged; `hear.py:387`), which implements all three placements; the node accepts only `"centre"` from the config and raises on any other value, because the other two are unmeasured |
| N14 | a span longer than `hear.window_s` cannot be placed in the buffer | `span_to_hear_buffer` raises `ValueError` on such a span, and that refusal is the routing signal: AIRWAY classifies it over its own sliced audio with the sliding detector (the function's own default hop), label score = max over windows; the assertion records `input: "sliding"`. Unmeasured path, recorded as such |
| N15 | AIRWAY step 3 says "any ASR word"; its read table names `asr_crisperwhisper` only | CrisperWhisper words only, **excluding** bracketed non-lexical tokens (`[cough]`, `[breath]`): an airway annotation inside the airway interval is not lexical contamination |
| N16 | the YAMNet "coverage winner" is undefined when several labels have coverage | winner = the label with the highest coverage among windows overlapping the span; ties broken by the highest single-window score; **abstain** when no label reaches `yamnet.coverage_threshold` in any overlapping window |
| N17 | "whether it lies inside certified silence" — intersection or containment? | containment: `in_certified_silence` is true when **every** YAMNet window overlapping the span is certified silent; `None` when the `silence` derivative is absent |
| N18 | what makes a hint "declare airway content" | any `hint.may_contain` tag equal, lower-cased, to a label of interest lower-cased or to the literal `"airway"` |
| N19 | which signal the figure's waveform panel renders (`capability-map.md` §5.2: pre-emphasis is not gain-neutral) | the **plain** waveform; the envelope/floor panel is the pre-emphasised quantity and its panel label says so. A figure failure never changes the verdict — the figure is an artifact, not a product of the store |
| N20 | `store.md` names `label`/`confirm`/`contest`/`measure` as assertion verbs; `branch-airway.md`'s product records confirm/contest/**abstain** per span | `abstain` is written as an assertion too (attributes `best_coverage`, `n_windows`), so "single-source" is a store fact rather than an inference from the absence of a record. This is a deliberate one-verb extension of the store vocabulary, stated here |
| N21 | `taxonomy(store)` in the design takes no hint; the shared node shape has one | TAXONOMY accepts `hint` for the shared shape and does not read it; AIRWAY reads it only in its step 4 (an absence-meaning change, nothing else) |
| N22 | where model identifiers live | module-level string constants in the node modules (`CRISPERWHISPER_ID`, `QWEN_ID`, `AST_ID`) — identity strings, not numbers; the no-literal rule binds numbers. Reached only through `_*_model()` factories so tests never construct an `HFModel` |
| N23 | how an uncomputable derivative is recorded | absent from the store (per `preprocess.md`), and named in the PREPROCESS verdict entity's `absent` attribute with the exception **class only** — an exception message may quote audio content and the store's controlled vocabulary must not |

New config keys these tasks add to `data/config/default.yaml` (allowed: overrides may not introduce
keys, the packaged file may — each arrives with a derivation): `yamnet.top_k` (Task 2),
`taxonomy.audioset_airway_labels`, `taxonomy.audioset_speech_labels`, `taxonomy.hear_airway_labels`,
`taxonomy.lexical_airway_tokens`, `taxonomy.presence_floor.{yamnet,ast,hear}` (Task 3, `ast: null` —
unmeasured), `airway.labels_of_interest`, `airway.confirmation_map`, `hear.placement` (Task 4).
**None of these supplies a value for anything in `benchmarks/open.md`** — `taxonomy.min_families.*`,
`speech.word_gap_ms`, `phonation.*`, `quality.*` and `redaction.padding_ms` stay `null`.

---

### Task 1: The node package, the shared contract, and ADMIT

**Scope:** `src/senselab/audio/workflows/triage/nodes/` (new package): `__init__.py`, `common.py`
(the `NodeResult` type and the four store helpers every node uses), `admit.py`. Tests at
`src/tests/audio/workflows/triage/nodes/` (new package): `__init__.py`, `conftest.py`,
`admit_test.py`. One task: the helpers have no independent consumer until ADMIT exercises them.

**Design points this task must not get wrong (from `admit.md`):**

- ADMIT rejects **only** decode failure, all-zero, constant. There is **no threshold** anywhere in the
  node — no "too quiet" row — and **no `flag` outcome**: its conditions admit no doubt, so `pass` |
  `fail` is the whole vocabulary.
- No models, no speech test, no enhancement, no second version of the audio, no level/clip/band
  tracks: the port list is `fail(reason) | pass(audio)` and nothing else.
- The admitted audio is **the recording as supplied** — no resampling, no channel reduction.
- ADMIT reads nothing from the config: it holds no numbers at all.

**Steps:**

- [ ] **Step 1 — write the failing tests.**

`src/tests/audio/workflows/triage/nodes/__init__.py`:

```python
"""Tests for the triage workflow's nodes."""
```

`src/tests/audio/workflows/triage/nodes/conftest.py`:

```python
"""Shared fixtures for the triage node tests. Nothing here loads a model."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.utils.prov_store import ProvStore


@pytest.fixture
def config() -> TriageConfig:
    """The packaged configuration, unmodified."""
    return load_triage_config()


@pytest.fixture
def store() -> ProvStore:
    """An empty store for one test run."""
    return ProvStore(run_id="test-run")


@pytest.fixture
def wav_writer(tmp_path: Path) -> Callable[..., Path]:
    """A writer for mono or stereo float32 WAV fixtures under this test's tmp dir."""

    def _write(name: str, samples: np.ndarray, sampling_rate: int = 16000) -> Path:
        path = tmp_path / name
        sf.write(str(path), samples.astype(np.float32), sampling_rate)
        return path

    return _write


def burst_samples(duration_s: float = 3.0, sampling_rate: int = 16000) -> np.ndarray:
    """A quiet noise bed with one loud 150 ms tone burst at 1.5 s.

    The burst stands far more than 18 dB over the bed, so `propose_spans` at the airway `K`
    proposes exactly one span over it.
    """
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(duration_s * sampling_rate)) * 1e-4).astype(np.float32)
    i0 = int(1.5 * sampling_rate)
    i1 = i0 + int(0.15 * sampling_rate)
    t = np.arange(i1 - i0) / sampling_rate
    x[i0:i1] += (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return x
```

`src/tests/audio/workflows/triage/nodes/admit_test.py`:

```python
"""ADMIT rejects only decode failure, all-zero and constant. No thresholds, no flag, no models."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.admit import AdmitResult, admit
from senselab.audio.workflows.triage.nodes.common import resolve_stream
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


def _sine(duration_s: float = 1.0, amplitude: float = 0.5, sampling_rate: int = 16000) -> np.ndarray:
    """A mono sine fixture."""
    t = np.arange(int(duration_s * sampling_rate)) / sampling_rate
    return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


class TestRejections:
    """The three degenerate conditions fail, without exceptions escaping."""

    def test_a_file_that_does_not_decode_fails(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """A text file with a .wav name is a decode failure, not a crash."""
        path = tmp_path / "not_audio.wav"
        path.write_text("this is not a wav file")
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "decode" in result.verdict.why
        assert result.audio is None

    def test_a_missing_file_fails_rather_than_raising(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path
    ) -> None:
        """A path that does not exist is a decode failure."""
        result = admit(store, tmp_path / "absent.wav", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL

    def test_all_zero_fails(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Every sample exactly zero is unmeasurable."""
        path = wav_writer("zeros.wav", np.zeros(16000, dtype=np.float32))
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "zero" in result.verdict.why

    def test_constant_dc_fails(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A constant nonzero value has no variance and is unmeasurable."""
        path = wav_writer("dc.wav", np.full(16000, 0.25, dtype=np.float32))
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "constant" in result.verdict.why

    def test_zero_frames_fails(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A zero-frame file fails, whether the decoder raises or returns nothing."""
        path = wav_writer("empty.wav", np.zeros(0, dtype=np.float32))
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL


class TestAdmission:
    """Everything non-degenerate passes; there is no level threshold and no flag."""

    def test_a_sine_passes_and_returns_the_decoded_audio(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The pass port carries the decoded audio."""
        path = wav_writer("sine.wav", _sine())
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.audio is not None
        assert result.audio.sampling_rate == 16000

    def test_a_very_quiet_recording_passes_because_there_is_no_level_threshold(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """Quiet is not empty: room-tone-level signal is admitted."""
        path = wav_writer("quiet.wav", _sine(amplitude=1e-4))
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS

    def test_the_admitted_audio_is_the_recording_as_supplied(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """No resampling and no channel reduction happen here: 48 kHz stereo stays 48 kHz stereo."""
        stereo = np.stack([_sine(sampling_rate=48000), _sine(sampling_rate=48000)], axis=1)
        path = wav_writer("stereo48k.wav", stereo, sampling_rate=48000)
        result = admit(store, path, config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.audio is not None
        assert result.audio.sampling_rate == 48000
        assert result.audio.waveform.shape[0] == 2

    def test_admit_never_flags(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The outcome vocabulary is pass or fail; flag does not exist for ADMIT."""
        fixtures = [
            wav_writer("a.wav", _sine()),
            wav_writer("b.wav", np.zeros(16000, dtype=np.float32)),
            wav_writer("c.wav", _sine(amplitude=1e-4)),
            tmp_path / "missing.wav",
        ]
        for path in fixtures:
            result = admit(ProvStore(run_id=f"never-flag-{path.name}"), path, config, run_dir=tmp_path)
            assert result.verdict.outcome in (Outcome.PASS, Outcome.FAIL)


class TestStoreWrites:
    """What ADMIT writes to the store, and what it does not."""

    def test_pass_writes_a_recording_stream_with_provenance(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """The recording enters the store as a stream entity, generated and attributed."""
        path = wav_writer("sine.wav", _sine())
        result = admit(store, path, config, run_dir=tmp_path)
        [stream] = store.entities("stream")
        assert stream.attributes["name"] == "recording"
        assert stream.attributes["sampling_rate"] == 16000
        assert stream.attributes["channels"] == 1
        assert Path(stream.attributes["path"]).is_absolute()
        assert store.generated_by(stream.id) is not None
        assert stream.id in result.view
        entity_id, audio = resolve_stream(store, tmp_path, "recording")
        assert entity_id == stream.id
        assert audio.waveform.shape[-1] == 16000

    def test_fail_writes_only_a_verdict(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """A rejected file leaves no stream behind — nothing else is claimed about it."""
        path = wav_writer("zeros.wav", np.zeros(16000, dtype=np.float32))
        admit(store, path, config, run_dir=tmp_path)
        assert store.entities("stream") == []
        [verdict] = store.entities("verdict")
        assert verdict.attributes["outcome"] == "fail"

    def test_the_verdict_entity_names_the_node_and_outcome(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, wav_writer: Callable[..., Path]
    ) -> None:
        """VERDICT reads verdict entities; theirs is the shape that must hold."""
        path = wav_writer("sine.wav", _sine())
        result = admit(store, path, config, run_dir=tmp_path)
        verdict = store.get_entity(result.verdict_entity_id)
        assert verdict.prov_type == "verdict"
        assert verdict.attributes["node"] == "ADMIT"
        assert verdict.attributes["outcome"] == "pass"
        [agent_id] = store.associated_with(store.generated_by(result.verdict_entity_id) or "")
        assert store.get_agent(agent_id).agent_type == "software"

    def test_resolve_stream_raises_on_an_unknown_name(self, store: ProvStore, tmp_path: Path) -> None:
        """A missing stream is a LookupError naming the stream, not a silent None."""
        with pytest.raises(LookupError, match="plain"):
            resolve_stream(store, tmp_path, "plain")
```

- [ ] **Step 2 — run them; expect failure.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/admit_test.py -x -q`
  Expected: `ModuleNotFoundError: No module named 'senselab.audio.workflows.triage.nodes'`.

- [ ] **Step 3 — implement.**

`src/senselab/audio/workflows/triage/nodes/__init__.py`:

```python
"""The triage workflow's nodes. Each writes to the provenance store and returns a NodeResult."""
```

`src/senselab/audio/workflows/triage/nodes/common.py`:

```python
"""The shape every triage node shares: its result type and its store conventions."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.vocabulary import NodeVerdict, Outcome
from senselab.utils.prov_store import Entity, ProvStore


@dataclass(frozen=True)
class NodeResult:
    """What every node returns.

    Attributes:
        verdict: The node's conclusion, in the graph's shared vocabulary.
        view: Ids of the store entities this node wrote or asserted over.
        verdict_entity_id: The verdict entity this node wrote to the store.
    """

    verdict: NodeVerdict
    view: tuple[str, ...]
    verdict_entity_id: str


def software_agent(store: ProvStore) -> str:
    """The agent for work senselab itself performed, at the installed version.

    Args:
        store: The provenance store.

    Returns:
        The agent's id.
    """
    return store.agent(agent_type="software", version=f"senselab {version('senselab')}")


def write_verdict(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    node: str,
    outcome: Outcome,
    kind: str | None,
    why: str,
    detail: dict[str, Any],
) -> tuple[str, NodeVerdict]:
    """Write one node's verdict entity.

    Args:
        store: The provenance store.
        activity_id: The activity that concluded.
        agent_id: The agent answerable for the verdict.
        node: The node's name.
        outcome: What it concluded.
        kind: The kind the node screens, or None.
        why: The reason, in controlled vocabulary — never transcript text.
        detail: The node's design-named verdict fields.

    Returns:
        The verdict entity's id and the vocabulary verdict.
    """
    entity_id = store.entity(
        prov_type="verdict",
        extent=None,
        attributes={"node": node, "outcome": outcome.value, "kind": kind, "why": why, **detail},
    )
    store.was_generated_by(entity_id, activity_id)
    store.was_attributed_to(entity_id, agent_id)
    return entity_id, NodeVerdict(node=node, outcome=outcome, kind=kind, why=why)


def find_measurement(store: ProvStore, name: str) -> Entity | None:
    """The latest measurement entity carrying this name, or None.

    Args:
        store: The provenance store.
        name: The measurement's ``name`` attribute.

    Returns:
        The entity, or None when nothing carries the name.
    """
    found = [e for e in store.entities("measurement") if e.attributes.get("name") == name]
    return found[-1] if found else None


def resolve_stream(store: ProvStore, run_dir: Path, name: str) -> tuple[str, Audio]:
    """Load a stream the graph wrote earlier, by its name.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.
        name: The stream entity's ``name`` attribute.

    Returns:
        The stream entity's id and its audio, loaded lazily from the sidecar.

    Raises:
        LookupError: If no stream entity carries that name.
    """
    for entity in store.entities("stream"):
        if entity.attributes.get("name") == name:
            path = Path(entity.attributes["path"])
            if not path.is_absolute():
                path = run_dir / path
            return entity.id, Audio(filepath=str(path))
    raise LookupError(f"no stream named {name!r} in the store; the node that writes it has not run")
```

`src/senselab/audio/workflows/triage/nodes/admit.py`:

```python
"""ADMIT — is this recording measurable at all.

The only rejections are decode failure, all samples zero, and a constant signal. No thresholds, no
``flag`` outcome, no models, no derived audio. The measurements behind the threshold-free rule are in
``specs/20260817-triage-workflow-dag/admit.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import NodeResult, software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

NODE = "ADMIT"


@dataclass(frozen=True)
class AdmitResult(NodeResult):
    """ADMIT's result.

    Attributes:
        audio: The decoded recording, as supplied, on ``pass``; None on ``fail``.
    """

    audio: Audio | None


def admit(
    store: ProvStore,
    source: str | Path,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> AdmitResult:
    """Decide whether the recording is measurable at all.

    Rejects only decode failure, all-zero samples and a constant signal. Everything else passes, as
    supplied — no resampling, no channel reduction, no models, no quality judgement. ``config``,
    ``hint`` and ``run_dir`` belong to the shared node shape and are not read: ADMIT holds no
    numbers, no hint changes whether a file decodes, and it writes no sidecars.

    Args:
        store: The provenance store.
        source: The recording, as supplied.
        config: The triage configuration. Unread.
        hint: What the recording was declared to contain. Unread.
        run_dir: The run directory. Unused.

    Returns:
        The verdict and, on ``pass``, the decoded audio.
    """
    activity_id = store.activity(node=NODE, step=None, parameters={"audio_file": str(source)})
    agent_id = software_agent(store)
    store.was_associated_with(activity_id, agent_id)

    def _fail(why: str) -> AdmitResult:
        entity_id, verdict = write_verdict(
            store, activity_id, agent_id, node=NODE, outcome=Outcome.FAIL, kind=None, why=why, detail={}
        )
        return AdmitResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id, audio=None)

    try:
        audio = Audio(filepath=str(source))
        waveform = audio.waveform
    except Exception as err:  # noqa: BLE001 — every decode failure is the same finding
        return _fail(f"decode failure: {type(err).__name__}")
    if waveform.shape[-1] == 0:
        return _fail("decode returned zero frames")
    if not bool(torch.any(waveform != 0)):
        return _fail("every sample is zero")
    if bool(torch.all(waveform == waveform.reshape(-1)[0])):
        return _fail("constant value; no variance")

    duration_s = waveform.shape[-1] / audio.sampling_rate
    stream_id = store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={
            "name": "recording",
            "path": str(Path(source).resolve()),
            "sampling_rate": int(audio.sampling_rate),
            "channels": int(waveform.shape[0]),
        },
    )
    store.was_generated_by(stream_id, activity_id)
    store.was_attributed_to(stream_id, agent_id)
    verdict_id, verdict = write_verdict(
        store,
        activity_id,
        agent_id,
        node=NODE,
        outcome=Outcome.PASS,
        kind=None,
        why="the file decodes and its samples vary",
        detail={"stream": stream_id},
    )
    return AdmitResult(verdict=verdict, view=(stream_id, verdict_id), verdict_entity_id=verdict_id, audio=audio)
```

- [ ] **Step 4 — run the tests; expect all PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/admit_test.py -x -q`

- [ ] **Step 5 — lint and type-check.**
  `uv run ruff format src/senselab/audio/workflows/triage/nodes src/tests/audio/workflows/triage/nodes`
  `uv run ruff check src/senselab/audio/workflows/triage/nodes src/tests/audio/workflows/triage/nodes`
  `uv run mypy src/senselab/audio/workflows/triage/nodes`

- [ ] **Step 6 — commit.** `git add -A && git commit -m "feat(triage): the node contract, and ADMIT"`

**Interfaces:**

*Consumed (verified against this branch):*
- `ProvStore(run_id)`, `.entity(prov_type=..., extent=..., attributes=...) -> str`, `.activity(node=..., step=..., parameters=...) -> str`, `.agent(agent_type=..., model_id=None, commit_sha=None, unresolved_reason=None, version=None) -> str`, the six relation methods, `.entities(prov_type=None) -> list[Entity]`, `.get_entity`, `.get_agent`, `.generated_by`, `.associated_with` — `src/senselab/utils/prov_store.py`.
- `Audio(filepath=...)` — lazy decode; the decoder raises on first `.waveform` access, which is why ADMIT touches `.waveform` inside the `try` (`audio/data_structures/audio.py:125`).
- `NodeVerdict`, `Outcome` — `src/senselab/audio/workflows/triage/vocabulary.py`.
- `TriageConfig` — accepted, unread (ADMIT is threshold-free by design).

*Produced (consumed by every later task and the sibling plan):*
- `nodes.common.NodeResult`, `software_agent`, `write_verdict`, `find_measurement`, `resolve_stream` — exactly as above.
- `admit(store, source, config, hint=None, *, run_dir) -> AdmitResult` with `AdmitResult.audio`.
- Store writes: one `stream` entity `name="recording"` on pass; one `verdict` entity always (schema table above).

*What the sibling plan may rely on:* `resolve_stream`/`find_measurement` as the way its nodes read
PREPROCESS's writes; the verdict-entity shape (`node`, `outcome`, `kind`, `why`, …) as what VERDICT
folds over.

---

### Task 2: PREPROCESS

**Scope:** `src/senselab/audio/workflows/triage/nodes/preprocess.py`; one config addition
(`yamnet.top_k`); tests at `src/tests/audio/workflows/triage/nodes/preprocess_test.py`.

**Design points this task must not get wrong (from `preprocess.md` and the store):**

- **Two signals, and every derivative names which one it reads.** ASR (CrisperWhisper + Qwen),
  alignment, SQUIM, `level` and `silence` (YAMNet) run on the **plain** resampled signal;
  `energy_envelope`, `spans`, both spectrograms and `gammatone` run on the **pre-emphasised** signal.
- **`spans` uses `k_db = spans.k_db.airway`** — the config's only `k_db` entry. SPEECH does not read
  these spans (it derives its own from word timings), so no second `K` exists to leak in.
- **A derivative that cannot be computed is absent, not an error.** PREPROCESS has no `fail` and no
  `flag`; the verdict lists what is absent with the exception class only (N23).
- **`no_contrast` is not an empty span list.** `propose_spans` returns `NoContrast` and the node
  writes a `spans_no_contrast` measurement carrying the `k_db` it was found at (N8).
- **Span entities carry `peak_over_floor_db` and no label** — ever.
- **YAMNet must be read with the full label space** (`top_k` = `yamnet.top_k` = 521): the function's
  windowed default of 5 can drop `Silence` entirely, which reads as a zero score (deferred finding,
  capability-map §4.2).
- **Everything model-authored gets an Agent** with a resolved commit or an honest
  `unresolved_reason` (YAMNet, SQUIM, the aligner).

**Steps:**

- [ ] **Step 1 — add the config key.** In
`src/senselab/audio/workflows/triage/data/config/default.yaml`, extend the `yamnet` block:

```yaml
yamnet:
  silence_threshold: 0.5
  coverage_threshold: 0.5
  top_k: 521
```

and append to the `derivation` block (it is a config value; the hash will change, which is correct):

```
  YAMNet top_k 521 -- the full label space, which is a size, not a threshold. classify_audios
  defaults windowed top_k to 5 and Silence is not always in the top 5 (capability-map 4.2), so a
  truncated read silently reports zero for a label the model actually emitted.
```

- [ ] **Step 2 — write the failing tests.**

`src/tests/audio/workflows/triage/nodes/preprocess_test.py`:

```python
"""PREPROCESS writes every derivative to the store with provenance; an uncomputable one is absent.

Every model call is monkeypatched on the node module (the pii_adapter_test pattern); the DSP —
resample, envelope, spans, spectrograms, gammatone, fuse_word_streams — runs real. No test here
loads weights or touches the network.
"""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import preprocess as node
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.preprocess import PreprocessResult, preprocess
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore

from tests.audio.workflows.triage.nodes.conftest import burst_samples


class _FakeModel:
    """A model spec stub carrying exactly what the node reads: path_or_uri and commit_sha."""

    def __init__(self, path_or_uri: str) -> None:
        """Stub a resolved model."""
        self.path_or_uri = path_or_uri
        self.commit_sha = "a" * 40


@pytest.fixture
def calls() -> dict[str, list]:
    """Captured model-call arguments, per mocked function."""
    return {"classify": [], "transcribe": [], "align": [], "squim": []}


@pytest.fixture
def mock_models(monkeypatch: pytest.MonkeyPatch, calls: dict[str, list]) -> None:
    """Replace every model call PREPROCESS makes; payload shapes mirror the real returns.

    CrisperWhisper/Qwen: a ScriptLine whose ``chunks`` are word ScriptLines with text/start/end/score
    (crisperwhisper.py builds exactly that). YAMNet: windowed dicts with start/end/label_scores/
    win_length/hop_length (classification/api.py). SQUIM: one dict with stoi/pesq/si_sdr
    (torchaudio_squim.py). Alignment: a list per input of ScriptLine | None (forced_alignment.py).
    """
    monkeypatch.setattr(node, "_crisperwhisper_model", lambda: _FakeModel(node.CRISPERWHISPER_ID))
    monkeypatch.setattr(node, "_qwen_model", lambda: _FakeModel(node.QWEN_ID))

    def fake_classify(audios: list, model: object, top_k: int | None = None, **kwargs: object) -> list:
        """YAMNet-shaped windows over the input's real duration."""
        calls["classify"].append({"model": model, "top_k": top_k})
        duration = audios[0].waveform.shape[-1] / audios[0].sampling_rate
        windows, start = [], 0.0
        while start + 0.96 <= duration:
            windows.append(
                {
                    "start": round(start, 2),
                    "end": round(start + 0.96, 2),
                    "label_scores": [{"Silence": 0.7}, {"Speech": 0.1}],
                    "win_length": 0.96,
                    "hop_length": 0.48,
                }
            )
            start += 0.48
        return [windows]

    def fake_transcribe(audios: list, model: object, **kwargs: object) -> list:
        """Two words inside the burst, for either recognizer."""
        calls["transcribe"].append({"model": model.path_or_uri, "audio": audios[0], "kwargs": kwargs})
        chunks = [
            ScriptLine(text="hello", start=1.50, end=1.58, score=0.9),
            ScriptLine(text="doctor", start=1.60, end=1.66, score=0.9),
        ]
        return [ScriptLine(text="hello doctor", start=1.50, end=1.66, chunks=chunks, score=0.9)]

    def fake_align(items: list, levels_to_keep: dict | None = None, aligner_model: str | None = None) -> list:
        """One aligned line per input tuple."""
        calls["align"].append({"n": len(items)})
        return [[ScriptLine(text="hello doctor", start=1.50, end=1.66)] for _ in items]

    def fake_squim(audios: list, device: object = None) -> list:
        """One objective-head dict per input."""
        calls["squim"].append({"n_samples": int(audios[0].waveform.shape[-1])})
        return [{"stoi": 0.91, "pesq": 1.8, "si_sdr": 7.5} for _ in audios]

    monkeypatch.setattr(node, "classify_audios", fake_classify)
    monkeypatch.setattr(node, "transcribe_audios", fake_transcribe)
    monkeypatch.setattr(node, "align_transcriptions", fake_align)
    monkeypatch.setattr(node, "extract_objective_quality_features_from_audios", fake_squim)


def _run(
    store: ProvStore,
    config: TriageConfig,
    tmp_path: Path,
    samples: np.ndarray | None = None,
    sampling_rate: int = 16000,
) -> PreprocessResult:
    """Admit a fixture recording, then preprocess it."""
    path = tmp_path / "input.wav"
    sf.write(str(path), (burst_samples() if samples is None else samples).astype(np.float32), sampling_rate)
    admitted = admit(store, path, config, run_dir=tmp_path)
    assert admitted.audio is not None
    return preprocess(store, admitted.audio, config, run_dir=tmp_path)


def _measurement(store: ProvStore, name: str) -> Any:
    """The one measurement entity with this name."""
    [entity] = [e for e in store.entities("measurement") if e.attributes.get("name") == name]
    return entity


class TestConditioning:
    """The two retained signals and the overshoot guard."""

    def test_plain_stream_is_mono_16k_with_provenance(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """A 48 kHz stereo input becomes one mono 16 kHz plain stream derived from the recording."""
        stereo = np.stack([burst_samples(sampling_rate=48000)] * 2, axis=1)
        _run(store, config, tmp_path, samples=stereo, sampling_rate=48000)
        [plain] = [e for e in store.entities("stream") if e.attributes.get("name") == "plain"]
        assert plain.attributes["sampling_rate"] == 16000
        assert plain.attributes["channels"] == 1
        assert plain.attributes["peak_scale"] == 1.0
        data, rate = sf.read(str(tmp_path / plain.attributes["path"]))
        assert rate == 16000
        [recording] = [e for e in store.entities("stream") if e.attributes.get("name") == "recording"]
        assert recording.id in store.derived_from(plain.id)

    def test_preemphasised_stream_is_the_first_difference(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """y[n] = x[n] - c * x[n-1], with the coefficient from the config, derived from plain."""
        _run(store, config, tmp_path)
        [plain] = [e for e in store.entities("stream") if e.attributes.get("name") == "plain"]
        [sharp] = [e for e in store.entities("stream") if e.attributes.get("name") == "preemphasised"]
        c = float(config.require("preemphasis.coefficient"))
        assert sharp.attributes["coefficient"] == c
        x, _ = sf.read(str(tmp_path / plain.attributes["path"]))
        y, _ = sf.read(str(tmp_path / sharp.attributes["path"]))
        assert np.allclose(y[1:], x[1:] - c * x[:-1], atol=1e-6)
        assert plain.id in store.derived_from(sharp.id)

    def test_disabled_preemphasis_routes_envelope_to_plain(
        self, store: ProvStore, tmp_path: Path, mock_models: None
    ) -> None:
        """With preemphasis.enabled false there is no second stream and derivatives read plain."""
        override = tmp_path / "override.yaml"
        override.write_text("preemphasis:\n  enabled: false\n")
        config = load_triage_config(override)
        _run(store, config, tmp_path)
        assert [e for e in store.entities("stream") if e.attributes.get("name") == "preemphasised"] == []
        assert _measurement(store, "energy_envelope").attributes["signal"] == "plain"


class TestEnvelopeAndSpans:
    """The pre-emphasised derivatives and the span proposals."""

    def test_envelope_reads_the_preemphasised_signal(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """The envelope names its signal and its sidecar holds envelope and floor tracks."""
        _run(store, config, tmp_path)
        envelope = _measurement(store, "energy_envelope")
        assert envelope.attributes["signal"] == "preemphasised"
        sidecar = np.load(tmp_path / envelope.attributes["path"])
        assert sidecar["envelope_dbfs"].shape == sidecar["floor_dbfs"].shape

    def test_spans_carry_the_airway_k_and_no_label(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """The burst yields at least one span at K = spans.k_db.airway, unlabelled."""
        _run(store, config, tmp_path)
        spans = store.entities("span")
        assert spans, "the burst fixture must propose at least one span"
        for span in spans:
            assert span.attributes["k_db"] == float(config.require("spans.k_db.airway"))
            assert "label" not in span.attributes
            assert span.attributes["peak_over_floor_db"] >= float(config.require("spans.k_db.airway"))

    def test_no_contrast_is_recorded_with_its_k(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """A burst-free recording writes spans_no_contrast, not an empty span list."""
        rng = np.random.default_rng(1)
        _run(store, config, tmp_path, samples=(rng.standard_normal(48000) * 1e-4))
        assert store.entities("span") == []
        no_contrast = _measurement(store, "spans_no_contrast")
        assert no_contrast.attributes["k_db"] == float(config.require("spans.k_db.airway"))
        assert "reason" in no_contrast.attributes


class TestModelDerivatives:
    """The plain-signal derivatives: YAMNet, level, SQUIM, the recognizers, agreement, alignment."""

    def test_yamnet_is_read_with_the_full_label_space(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None, calls: dict
    ) -> None:
        """top_k comes from the config, never the function's windowed default of 5."""
        _run(store, config, tmp_path)
        [call] = calls["classify"]
        assert call["model"] == "yamnet"
        assert call["top_k"] == int(config.require("yamnet.top_k"))
        windows_entity = _measurement(store, "yamnet_windows")
        windows = json.loads((tmp_path / windows_entity.attributes["path"]).read_text())
        assert windows_entity.attributes["n_windows"] == len(windows) > 0
        silence = _measurement(store, "silence")
        assert all(row["is_silence"] == (row["score"] >= 0.5) for row in silence.attributes["windows"])

    def test_asr_runs_on_the_plain_signal_not_the_preemphasised_one(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None, calls: dict
    ) -> None:
        """Both recognizers receive the plain waveform (pre-emphasis changes the peak measurably)."""
        _run(store, config, tmp_path)
        [plain] = [e for e in store.entities("stream") if e.attributes.get("name") == "plain"]
        reference, _ = sf.read(str(tmp_path / plain.attributes["path"]), dtype="float32")
        assert len(calls["transcribe"]) == 2
        for call in calls["transcribe"]:
            received = call["audio"].waveform.squeeze(0).numpy()
            assert np.allclose(received, reference, atol=1e-4)

    def test_words_become_word_entities_with_recognizer_provenance(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """One word entity per recognizer word, stamped with who timed it, attributed to the model."""
        _run(store, config, tmp_path)
        words = store.entities("word")
        assert len(words) == 4  # two words from each recognizer
        recognizers = {w.attributes["recognizer"] for w in words}
        assert recognizers == {node.CRISPERWHISPER_ID, node.QWEN_ID}
        sources = {w.attributes["recognizer"]: w.attributes["timestamp_source"] for w in words}
        assert sources[node.CRISPERWHISPER_ID] == "native"
        assert sources[node.QWEN_ID] == "bundled_aligner"
        for word in words:
            [agent_id] = [
                a for a in store.associated_with(store.generated_by(word.id) or "")
            ]
            agent = store.get_agent(agent_id)
            assert agent.agent_type == "model"
            assert agent.commit_sha == "a" * 40

    def test_agreement_and_alignment_follow_the_recognizers(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None, calls: dict
    ) -> None:
        """asr_agreement fuses both streams; alignment aligns the fused transcript on plain."""
        _run(store, config, tmp_path)
        agreement = _measurement(store, "asr_agreement")
        assert agreement.attributes["systems"] == [node.CRISPERWHISPER_ID, node.QWEN_ID]
        assert {w["text"] for w in agreement.attributes["words"]} == {"hello", "doctor"}
        alignment = _measurement(store, "alignment")
        assert alignment.attributes["transcript_source"] == "asr_agreement"
        payload = json.loads((tmp_path / alignment.attributes["path"]).read_text())
        assert payload, "the aligned transcript is serialised to the sidecar"
        assert calls["align"] == [{"n": 1}]

    def test_squim_is_measured_per_span_as_a_measure_assertion(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None, calls: dict
    ) -> None:
        """One measure assertion per span, derived from it, on the sliced plain signal."""
        _run(store, config, tmp_path)
        spans = store.entities("span")
        measures = [a for a in store.entities("assertion") if a.attributes.get("name") == "squim"]
        assert len(measures) == len(spans) > 0
        for measure in measures:
            assert measure.attributes["verb"] == "measure"
            assert measure.attributes["stoi"] == 0.91
            assert any(s.id in store.derived_from(measure.id) for s in spans)
        assert all(c["n_samples"] > 0 for c in calls["squim"])

    def test_a_span_squim_refuses_is_unmeasured_not_padded(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SQUIM re-raises on short input; the node records unmeasured and never pads."""

        def refusing_squim(audios: list, device: object = None) -> list:
            """The real function's failure mode for a too-short span."""
            raise RuntimeError("input too short")

        monkeypatch.setattr(node, "extract_objective_quality_features_from_audios", refusing_squim)
        result = _run(store, config, tmp_path)
        measures = [a for a in store.entities("assertion") if a.attributes.get("name") == "squim"]
        assert measures, "the refusal is recorded per span, not dropped"
        assert all(m.attributes["unmeasured"] == "RuntimeError" for m in measures)
        assert result.verdict.outcome is Outcome.PASS


class TestAbsenceIsNotAnError:
    """A derivative that cannot be computed is absent from the store; the node still passes."""

    def test_a_failing_model_leaves_its_derivatives_absent(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """YAMNet failing removes yamnet_windows and silence; nothing raises; outcome stays pass."""

        def broken_classify(*args: object, **kwargs: object) -> list:
            """A backend crash."""
            raise RuntimeError("subprocess venv failed")

        monkeypatch.setattr(node, "classify_audios", broken_classify)
        result = _run(store, config, tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert "yamnet_windows" in result.absent
        assert "silence" in result.absent
        names = {e.attributes.get("name") for e in store.entities("measurement")}
        assert "yamnet_windows" not in names and "silence" not in names
        verdict = store.get_entity(result.verdict_entity_id)
        assert verdict.attributes["absent"]["yamnet_windows"] == "RuntimeError"

    def test_one_missing_recognizer_takes_agreement_and_alignment_with_it(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Agreement needs both recognizers; its absence is recorded, not raised."""

        def qwen_only_fails(audios: list, model: object, **kwargs: object) -> list:
            """Qwen's venv fails; CrisperWhisper still answers."""
            if model.path_or_uri == node.QWEN_ID:
                raise RuntimeError("qwen venv failed")
            chunks = [ScriptLine(text="hello", start=1.50, end=1.58, score=0.9)]
            return [ScriptLine(text="hello", start=1.50, end=1.58, chunks=chunks, score=0.9)]

        monkeypatch.setattr(node, "transcribe_audios", qwen_only_fails)
        result = _run(store, config, tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert {"asr_qwen", "asr_agreement", "alignment"} <= set(result.absent)
        assert _measurement(store, "asr_crisperwhisper") is not None
```

- [ ] **Step 3 — run them; expect failure.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -x -q`
  Expected: `ImportError` (no `nodes.preprocess` module).

- [ ] **Step 4 — implement.**

`src/senselab/audio/workflows/triage/nodes/preprocess.py`:

```python
"""PREPROCESS — one conditioning pass, every shared derivative written to the store.

The recognizers, the aligner, SQUIM, level and YAMNet silence read the plain resampled signal; the
envelope, spans, spectrograms and gammatone read the pre-emphasised one. A derivative that cannot be
computed is absent from the store, not an error. Every parameter's derivation is in
``data/config/default.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.tasks.envelope.api import hilbert_envelope_dbfs, rolling_floor_dbfs
from senselab.audio.tasks.features_extraction.torchaudio import extract_spectrogram_from_audios
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
)
from senselab.audio.tasks.forced_alignment.constants import DEFAULT_ALIGN_MODELS_HF
from senselab.audio.tasks.forced_alignment.forced_alignment import align_transcriptions
from senselab.audio.tasks.gammatone.api import gammatone_filterbank
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.spans.api import NoContrast, propose_spans
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.audio.tasks.speech_to_text_ensemble.api import fuse_word_streams, iter_word_leaves
from senselab.audio.workflows.audio_analysis.level import integrated_lufs
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import NodeResult, software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import HFModel, Language, ScriptLine
from senselab.utils.prov_store import ProvStore

NODE = "PREPROCESS"
CRISPERWHISPER_ID = "nyralabs/CrisperWhisper2.0_turbo"
QWEN_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_TIMESTAMP_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNMENT_LANGUAGE = "en"


def _crisperwhisper_model() -> HFModel:
    """The CrisperWhisper model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=CRISPERWHISPER_ID, revision="main")


def _qwen_model() -> HFModel:
    """The Qwen3-ASR model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=QWEN_ID, revision="main")


@dataclass(frozen=True)
class PreprocessResult(NodeResult):
    """PREPROCESS's result.

    Attributes:
        absent: Names of derivatives that could not be computed and are absent from the store.
    """

    absent: tuple[str, ...]


def _measurement(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    name: str,
    signal: str,
    attributes: dict[str, Any],
    derived_from: tuple[str, ...] = (),
) -> str:
    """Write one derivative measurement entity with its provenance."""
    entity_id = store.entity(
        prov_type="measurement", extent=None, attributes={"name": name, "signal": signal, **attributes}
    )
    store.was_generated_by(entity_id, activity_id)
    store.was_attributed_to(entity_id, agent_id)
    for source_id in derived_from:
        store.was_derived_from(entity_id, source_id)
    return entity_id


def preprocess(  # noqa: C901 — one block per derivative, each independent
    store: ProvStore,
    source: Audio,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> PreprocessResult:
    """Condition the admitted audio and write every derivative to the store.

    Args:
        store: The provenance store, already holding ADMIT's ``recording`` stream.
        source: The audio ADMIT returned, as supplied.
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read.
        run_dir: Where the streams and sidecars are written.

    Returns:
        A pass verdict (PREPROCESS has no fail and no flag), the view over what was written, and the
        names of derivatives that are absent.
    """
    software = software_agent(store)
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)

    recording_ids = [e.id for e in store.entities("stream") if e.attributes.get("name") == "recording"]
    target_hz = int(config.require("resample.target_hz"))
    preemph_enabled = bool(config.require("preemphasis.enabled"))
    coefficient = float(config.require("preemphasis.coefficient"))

    condition = store.activity(
        node=NODE,
        step="condition",
        parameters={
            "target_hz": target_hz,
            "downmix": "mean",
            "preemphasis_enabled": preemph_enabled,
            "coefficient": coefficient,
        },
    )
    store.was_associated_with(condition, software)
    for recording_id in recording_ids:
        store.used(condition, recording_id)

    mono = Audio(waveform=source.waveform.mean(dim=0, keepdim=True), sampling_rate=source.sampling_rate)
    [plain] = resample_audios([mono], target_hz)
    peak = float(plain.waveform.abs().max())
    peak_scale = 1.0 if peak <= 1.0 else 1.0 / peak
    if peak_scale != 1.0:
        plain = Audio(waveform=plain.waveform * peak_scale, sampling_rate=target_hz)
    duration_s = plain.waveform.shape[-1] / target_hz
    plain.save_to_file(str(run_dir / "streams" / "plain.wav"))
    plain_id = store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={
            "name": "plain",
            "path": "streams/plain.wav",
            "sampling_rate": target_hz,
            "channels": 1,
            "peak_scale": peak_scale,
        },
    )
    store.was_generated_by(plain_id, condition)
    store.was_attributed_to(plain_id, software)
    for recording_id in recording_ids:
        store.was_derived_from(plain_id, recording_id)

    if preemph_enabled:
        x = plain.waveform
        emphasised = torch.cat([x[:, :1], x[:, 1:] - coefficient * x[:, :-1]], dim=1)
        sharp = Audio(waveform=emphasised, sampling_rate=target_hz)
        sharp.save_to_file(str(run_dir / "streams" / "preemphasised.wav"))
        sharp_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "preemphasised",
                "path": "streams/preemphasised.wav",
                "sampling_rate": target_hz,
                "channels": 1,
                "coefficient": coefficient,
            },
        )
        store.was_generated_by(sharp_id, condition)
        store.was_attributed_to(sharp_id, software)
        store.was_derived_from(sharp_id, plain_id)
        sharp_signal = "preemphasised"
    else:
        sharp, sharp_id, sharp_signal = plain, plain_id, "plain"

    absent: dict[str, str] = {}
    derivatives: dict[str, Any] = {}
    view: list[str] = [plain_id] + ([sharp_id] if sharp_id != plain_id else [])
    state: dict[str, Any] = {}

    def _step(step: str, parameters: dict[str, Any], reads: tuple[str, ...], agent_id: str) -> str:
        """One sub-activity, associated and with its reads recorded."""
        activity_id = store.activity(node=NODE, step=step, parameters=parameters)
        store.was_associated_with(activity_id, agent_id)
        for entity_id in reads:
            store.used(activity_id, entity_id)
        return activity_id

    def _envelope() -> None:
        """`energy_envelope` and its floor, over the pre-emphasised signal, to one npz sidecar."""
        parameters = {
            "lowpass_hz": float(config.require("envelope.lowpass_hz")),
            "filter_order": int(config.require("envelope.filter_order")),
            "floor_window_s": float(config.require("floor.window_s")),
            "floor_percentile": float(config.require("floor.percentile")),
            "floor_eval_grid_s": float(config.require("floor.eval_grid_s")),
        }
        activity = _step("envelope", parameters, (sharp_id,), software)
        envelope = hilbert_envelope_dbfs(
            sharp, lowpass_hz=parameters["lowpass_hz"], filter_order=int(parameters["filter_order"])
        )
        floor = rolling_floor_dbfs(
            envelope,
            target_hz,
            window_s=parameters["floor_window_s"],
            percentile=parameters["floor_percentile"],
            eval_grid_s=parameters["floor_eval_grid_s"],
        )
        np.savez(run_dir / "derivatives" / "energy_envelope.npz", envelope_dbfs=envelope, floor_dbfs=floor)
        entity_id = _measurement(
            store,
            activity,
            software,
            name="energy_envelope",
            signal=sharp_signal,
            attributes={"path": "derivatives/energy_envelope.npz", "sampling_rate": target_hz},
            derived_from=(sharp_id,),
        )
        derivatives["energy_envelope"] = entity_id
        view.append(entity_id)
        state.update(envelope=envelope, floor=floor, envelope_id=entity_id)

    def _spans() -> None:
        """Span proposals at the airway K; `NoContrast` becomes a measurement, never an empty list."""
        if "envelope" not in state:
            raise LookupError("energy_envelope is absent")
        k_db = float(config.require("spans.k_db.airway"))
        parameters = {
            "k_db": k_db,
            "onset_drop_db": float(config.require("spans.onset_drop_db")),
            "offset_fraction": float(config.require("spans.offset_fraction")),
            "hangover_ms": int(config.require("spans.hangover_ms")),
            "min_duration_ms": int(config.require("spans.min_duration_ms")),
            "min_separation_ms": int(config.require("spans.min_separation_ms")),
        }
        activity = _step("spans", parameters, (state["envelope_id"],), software)
        proposed = propose_spans(
            state["envelope"],
            state["floor"],
            target_hz,
            k_db=k_db,
            onset_drop_db=parameters["onset_drop_db"],
            offset_fraction=parameters["offset_fraction"],
            hangover_ms=parameters["hangover_ms"],
            min_duration_ms=parameters["min_duration_ms"],
            min_separation_ms=parameters["min_separation_ms"],
        )
        if isinstance(proposed, NoContrast):
            entity_id = _measurement(
                store,
                activity,
                software,
                name="spans_no_contrast",
                signal=sharp_signal,
                attributes={"k_db": k_db, "reason": proposed.reason},
                derived_from=(state["envelope_id"],),
            )
            derivatives["spans_no_contrast"] = entity_id
            view.append(entity_id)
            return
        span_ids: list[str] = []
        for span in proposed:
            span_id = store.entity(
                prov_type="span",
                extent=(span.start, span.end),
                attributes={"peak_over_floor_db": span.peak_over_floor_db, "k_db": k_db, "signal": sharp_signal},
            )
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, software)
            store.was_derived_from(span_id, state["envelope_id"])
            span_ids.append(span_id)
        derivatives["spans"] = span_ids
        view.extend(span_ids)
        state["span_ids"] = span_ids

    def _yamnet() -> None:
        """The full YAMNet native windows, to a json sidecar."""
        top_k = int(config.require("yamnet.top_k"))
        agent = store.agent(
            agent_type="model",
            model_id="https://tfhub.dev/google/yamnet/1",
            unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
        )
        activity = _step("yamnet", {"top_k": top_k}, (plain_id,), agent)
        [windows] = classify_audios([plain], model="yamnet", top_k=top_k)
        (run_dir / "derivatives" / "yamnet_windows.json").write_text(json.dumps(windows))
        entity_id = _measurement(
            store,
            activity,
            agent,
            name="yamnet_windows",
            signal="plain",
            attributes={"path": "derivatives/yamnet_windows.json", "n_windows": len(windows)},
            derived_from=(plain_id,),
        )
        derivatives["yamnet_windows"] = entity_id
        view.append(entity_id)
        state.update(yamnet_windows=windows, yamnet_windows_id=entity_id)

    def _silence() -> None:
        """The Silence projection of the YAMNet windows."""
        if "yamnet_windows" not in state:
            raise LookupError("yamnet_windows is absent")
        threshold = float(config.require("yamnet.silence_threshold"))
        activity = _step("silence", {"threshold": threshold}, (state["yamnet_windows_id"],), software)
        rows = []
        for window in state["yamnet_windows"]:
            score = 0.0
            for pair in label_scores(window):
                if "Silence" in pair:
                    score = float(pair["Silence"])
                    break
            rows.append(
                {"start": window["start"], "end": window["end"], "score": score, "is_silence": score >= threshold}
            )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="silence",
            signal="plain",
            attributes={"threshold": threshold, "windows": rows},
            derived_from=(state["yamnet_windows_id"],),
        )
        derivatives["silence"] = entity_id
        view.append(entity_id)

    def _level() -> None:
        """File-level peak dBFS, RMS dBFS and LUFS on the plain signal."""
        activity = _step("level", {}, (plain_id,), software)
        x = plain.waveform.squeeze(0).numpy()
        peak_dbfs = float(20.0 * np.log10(max(float(np.abs(x).max()), 1e-12)))
        rms_dbfs = float(20.0 * np.log10(max(float(np.sqrt(np.mean(x**2))), 1e-12)))
        lufs = float(integrated_lufs(x, target_hz))
        entity_id = _measurement(
            store,
            activity,
            software,
            name="level",
            signal="plain",
            attributes={"peak_dbfs": peak_dbfs, "rms_dbfs": rms_dbfs, "lufs": lufs},
            derived_from=(plain_id,),
        )
        derivatives["level"] = entity_id
        view.append(entity_id)

    def _squim() -> None:
        """One objective-head measure assertion per envelope span; refusals recorded, never padded."""
        if not state.get("span_ids"):
            raise LookupError("spans are absent")
        agent = store.agent(
            agent_type="model",
            model_id="torchaudio SQUIM_OBJECTIVE",
            unresolved_reason=f"bundled torchaudio weights, version {_dist_version('torchaudio')}",
        )
        activity = _step("squim", {}, tuple(state["span_ids"]), agent)
        assertion_ids: list[str] = []
        for span_id in state["span_ids"]:
            span = store.get_entity(span_id)
            start, end = span.extent or (0.0, 0.0)
            segment = Audio(
                waveform=plain.waveform[:, int(start * target_hz) : int(end * target_hz)],
                sampling_rate=target_hz,
            )
            try:
                [scores] = extract_objective_quality_features_from_audios([segment])
                attributes: dict[str, Any] = {
                    "verb": "measure",
                    "name": "squim",
                    "stoi": float(scores["stoi"]),
                    "pesq": float(scores["pesq"]),
                    "si_sdr": float(scores["si_sdr"]),
                }
            except Exception as err:  # noqa: BLE001 — a span SQUIM refuses is unmeasured, not padded
                attributes = {"verb": "measure", "name": "squim", "unmeasured": type(err).__name__}
            assertion_id = store.entity(prov_type="assertion", extent=span.extent, attributes=attributes)
            store.was_generated_by(assertion_id, activity)
            store.was_attributed_to(assertion_id, agent)
            store.was_derived_from(assertion_id, span_id)
            assertion_ids.append(assertion_id)
        derivatives["squim"] = assertion_ids
        view.extend(assertion_ids)

    def _asr(name: str, factory: Callable[[], HFModel], source_kind: str, timing_model: str | None,
             **kwargs: Any) -> None:
        """One recognizer: transcript measurement plus one word entity per timed word."""
        model = factory()
        agent = store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
        activity = _step(name, {"model": str(model.path_or_uri), **{k: str(v) for k, v in kwargs.items()}},
                         (plain_id,), agent)
        [line] = transcribe_audios([plain], model=model, **kwargs)
        word_ids: list[str] = []
        for chunk in line.chunks or []:
            attributes = {
                "text": chunk.text,
                "score": chunk.score,
                "recognizer": str(model.path_or_uri),
                "timestamp_source": source_kind,
            }
            if timing_model is not None:
                attributes["timestamp_model"] = timing_model
            word_id = store.entity(
                prov_type="word",
                extent=(float(chunk.start or 0.0), float(chunk.end or 0.0)),
                attributes=attributes,
            )
            store.was_generated_by(word_id, activity)
            store.was_attributed_to(word_id, agent)
            word_ids.append(word_id)
        meta: dict[str, Any] = {
            "recognizer": str(model.path_or_uri),
            "transcript": line.text or "",
            "word_ids": word_ids,
            "timestamp_source": source_kind,
        }
        if timing_model is not None:
            meta["timestamp_model"] = timing_model
        entity_id = _measurement(
            store, activity, agent, name=name, signal="plain", attributes=meta, derived_from=(plain_id,)
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        view.extend(word_ids)
        state[name] = line
        state[name + "_id"] = entity_id

    def _agreement() -> None:
        """The fused word list over both recognizers — the derivative SPEECH reads."""
        if "asr_crisperwhisper" not in state or "asr_qwen" not in state:
            raise LookupError("both recognizers are needed")
        activity = _step(
            "agreement",
            {"systems": [CRISPERWHISPER_ID, QWEN_ID]},
            (state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
            software,
        )
        streams = {
            CRISPERWHISPER_ID: iter_word_leaves([state["asr_crisperwhisper"].model_dump()]),
            QWEN_ID: iter_word_leaves([state["asr_qwen"].model_dump()]),
        }
        fused = fuse_word_streams(streams)
        entity_id = _measurement(
            store,
            activity,
            software,
            name="asr_agreement",
            signal="plain",
            attributes={"words": fused, "systems": [CRISPERWHISPER_ID, QWEN_ID]},
            derived_from=(state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
        )
        derivatives["asr_agreement"] = entity_id
        view.append(entity_id)
        state.update(fused=fused, asr_agreement_id=entity_id)

    def _alignment() -> None:
        """Forced alignment of the agreed transcript, on the plain signal."""
        if not state.get("fused"):
            raise LookupError("asr_agreement is absent or empty")
        fused = state["fused"]
        agent = store.agent(
            agent_type="model",
            model_id=str(DEFAULT_ALIGN_MODELS_HF[ALIGNMENT_LANGUAGE]),
            unresolved_reason="align_transcriptions loads its aligner internally; the commit is not reported",
        )
        activity = _step("alignment", {"language": ALIGNMENT_LANGUAGE}, (state["asr_agreement_id"],), agent)
        transcript = ScriptLine(
            text=" ".join(word["text"] for word in fused),
            start=min(word["start"] for word in fused),
            end=max(word["end"] for word in fused),
        )
        [aligned] = align_transcriptions([(plain, transcript, Language(language_code=ALIGNMENT_LANGUAGE))])
        payload = [line.model_dump() for line in aligned if line is not None]
        (run_dir / "derivatives" / "alignment.json").write_text(json.dumps(payload, default=str))
        entity_id = _measurement(
            store,
            activity,
            agent,
            name="alignment",
            signal="plain",
            attributes={
                "path": "derivatives/alignment.json",
                "language": ALIGNMENT_LANGUAGE,
                "transcript_source": "asr_agreement",
            },
            derived_from=(state["asr_agreement_id"],),
        )
        derivatives["alignment"] = entity_id
        view.append(entity_id)

    def _spectrogram(name: str, window_key: str) -> None:
        """One STFT magnitude, window and hop from the config, n_fft = win_length (decision N7)."""
        window_ms = float(config.require(window_key))
        hop_ms = float(config.require("spectrogram.hop_ms"))
        win_length = int(target_hz * window_ms / 1000.0)
        hop_length = int(target_hz * hop_ms / 1000.0)
        parameters = {"win_length": win_length, "hop_length": hop_length, "n_fft": win_length}
        activity = _step(name, parameters, (sharp_id,), software)
        [result] = extract_spectrogram_from_audios(
            [sharp], n_fft=win_length, win_length=win_length, hop_length=hop_length
        )
        np.savez(run_dir / "derivatives" / f"{name}.npz", spectrogram=result["spectrogram"].numpy())
        entity_id = _measurement(
            store,
            activity,
            software,
            name=name,
            signal=sharp_signal,
            attributes={"path": f"derivatives/{name}.npz", **parameters},
            derived_from=(sharp_id,),
        )
        derivatives[name] = entity_id
        view.append(entity_id)

    def _gammatone() -> None:
        """The ERB-spaced filterbank energies, to one npz sidecar."""
        parameters: dict[str, Any] = {
            "n_channels": int(config.require("gammatone.n_channels")),
            "low_hz": float(config.require("gammatone.low_hz")),
            "high_hz": float(config.require("gammatone.high_hz")),
            "hop_s": float(config.require("gammatone.hop_s")),
        }
        activity = _step("gammatone", parameters, (sharp_id,), software)
        centre_frequencies, energy_db = gammatone_filterbank(
            sharp,
            n_channels=parameters["n_channels"],
            low_hz=parameters["low_hz"],
            high_hz=parameters["high_hz"],
            hop_s=parameters["hop_s"],
        )
        np.savez(
            run_dir / "derivatives" / "gammatone.npz",
            centre_frequencies_hz=centre_frequencies,
            energy_db=energy_db,
        )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="gammatone",
            signal=sharp_signal,
            attributes={"path": "derivatives/gammatone.npz", "hop_s": parameters["hop_s"]},
            derived_from=(sharp_id,),
        )
        derivatives["gammatone"] = entity_id
        view.append(entity_id)

    blocks: list[tuple[str, Callable[[], None]]] = [
        ("energy_envelope", _envelope),
        ("spans", _spans),
        ("yamnet_windows", _yamnet),
        ("silence", _silence),
        ("level", _level),
        ("squim", _squim),
        ("asr_crisperwhisper", lambda: _asr("asr_crisperwhisper", _crisperwhisper_model, "native", None)),
        (
            "asr_qwen",
            lambda: _asr("asr_qwen", _qwen_model, "bundled_aligner", QWEN_TIMESTAMP_MODEL, return_timestamps=True),
        ),
        ("asr_agreement", _agreement),
        ("alignment", _alignment),
        ("spectrogram_wideband", lambda: _spectrogram("spectrogram_wideband", "spectrogram.wideband_window_ms")),
        (
            "spectrogram_narrowband",
            lambda: _spectrogram("spectrogram_narrowband", "spectrogram.narrowband_window_ms"),
        ),
        ("gammatone", _gammatone),
    ]
    for name, block in blocks:
        try:
            block()
        except Exception as err:  # noqa: BLE001 — an uncomputable derivative is absent, not an error
            absent[name] = type(err).__name__

    verdict_id, verdict = write_verdict(
        store,
        condition,
        software,
        node=NODE,
        outcome=Outcome.PASS,
        kind=None,
        why="conditioning complete; absent derivatives are listed",
        detail={"absent": dict(sorted(absent.items())), "derivatives": derivatives},
    )
    view.append(verdict_id)
    return PreprocessResult(
        verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, absent=tuple(sorted(absent))
    )
```

- [ ] **Step 5 — run the tests; expect all PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -x -q`
  Also re-run Task 1: `uv run pytest src/tests/audio/workflows/triage/nodes -x -q`

- [ ] **Step 6 — lint and type-check** (same commands as Task 1, plus
  `uv run pytest src/tests/audio/workflows/triage -q` for the config tests, since default.yaml changed
  — `config_test.py` hashes the packaged mapping and must still pass; if it pins an exact hash, update
  the pin in the same commit).

- [ ] **Step 7 — commit.** `git add -A && git commit -m "feat(triage): PREPROCESS writes the shared derivatives to the store"`

**Interfaces:**

*Consumed (verified against this branch):*
- `resample_audios(audios: List[Audio], resample_rate: int, lowcut=None, order=4) -> List[Audio]` — `preprocessing/preprocessing.py:30`. Its anti-alias filter quirk (capability-map §4.3) is left alone; the overshoot guard is this node's (N2) because the function reports none.
- `hilbert_envelope_dbfs(audio, *, lowpass_hz, filter_order) -> np.ndarray`, `rolling_floor_dbfs(envelope_db, sampling_rate, *, window_s, percentile, eval_grid_s) -> np.ndarray` — `tasks/envelope/api.py`.
- `propose_spans(envelope_db, floor_db, sampling_rate, *, k_db, onset_drop_db, offset_fraction, hangover_ms, min_duration_ms, min_separation_ms) -> list[Span] | NoContrast` — `tasks/spans/api.py`; `Span.start/.end/.peak_over_floor_db`, `NoContrast.reason`.
- `classify_audios(audios, model="yamnet", top_k=<config>) -> List[List[Dict]]` — windowed dicts `{start, end, label_scores, win_length, hop_length}`; `label_scores(window) -> list[dict[str, float]]` — `tasks/classification/`.
- `transcribe_audios(audios, model: SenselabModel, language=None, device=None, **kwargs) -> List[ScriptLine]` — routes on `path_or_uri` prefix (`_CRISPER_PREFIXES = ("nyralabs/CrisperWhisper2.0",)`, `_QWEN_ASR_PREFIXES = ("Qwen/Qwen3-ASR",)`, `speech_to_text/api.py:32,38`); `return_timestamps` reaches only the Qwen backend (`api.py:137-142`); word chunks are `ScriptLine(text, start, end, score)`.
- `fuse_word_streams(word_streams: dict[str, list[dict]], *, ...) -> list[dict]` and `iter_word_leaves(node) -> list[dict]` — `tasks/speech_to_text_ensemble/api.py:197/119`. Note the pre-existing coupling: this task module imports from `workflows/audio_analysis/floors` (capability-map §4.9); this plan uses it as-is and adds no second such edge beyond `integrated_lufs` below.
- `align_transcriptions(audios_and_transcriptions_and_language: List[Tuple[Audio, ScriptLine, Language]], levels_to_keep=..., aligner_model=None) -> List[List[ScriptLine | None]]` — `forced_alignment.py:685`; `DEFAULT_ALIGN_MODELS_HF` from `forced_alignment/constants.py`. The input `ScriptLine` carries `start`/`end` so the whole-recording-alignment stderr warning path never fires.
- `extract_spectrogram_from_audios(audios, n_fft=1024, win_length=None, hop_length=None) -> List[Dict[str, Tensor]]` — arguments in **samples**; return key `"spectrogram"` (`features_extraction/torchaudio.py:20,54`).
- `gammatone_filterbank(audio, *, n_channels, low_hz, high_hz, hop_s) -> tuple[np.ndarray, np.ndarray]` — `tasks/gammatone/api.py`.
- `extract_objective_quality_features_from_audios(audios, device=None) -> List[Dict[str, Any]]` (`stoi`, `pesq`, `si_sdr`) — refuses non-mono/non-16 kHz; **re-raises** on short spans (capability-map §4.7), which is why `_squim` catches per span.
- `integrated_lufs(waveform: np.ndarray, sampling_rate: int) -> float` — `workflows/audio_analysis/level.py:138`; the one deliberate cross-workflow import (capability-map §3.3 recommends lifting it to a task later; not this plan's scope).
- `HFModel(path_or_uri=..., revision="main")` — resolves `.commit_sha` (40-hex) at construction; reached only through `_crisperwhisper_model`/`_qwen_model` so tests never construct one.
- `Audio.save_to_file(file_path, format=None, subtype=None, out_of_range="raise") -> AudioWriteReport` — **the merged write layer** (`audio/data_structures/audio.py:371`): a plain `.wav` write resolves to the `FLOAT` subtype and round-trips float samples bit-exactly, including values beyond ±1, so PREPROCESS's stream sidecars are exact; the peak-scale guard (N2) still runs because downstream consumers read `plain` as a ≤ full-scale signal.

*Produced (the sibling plan's read surface — schema table in the preamble):* `stream` entities
`plain`/`preemphasised`; `measurement` entities `energy_envelope`, `yamnet_windows`, `silence`,
`level`, `asr_crisperwhisper`, `asr_qwen`, `asr_agreement`, `alignment`, `spectrogram_wideband`,
`spectrogram_narrowband`, `gammatone`, `spans_no_contrast`; `span` entities at `spans.k_db.airway`;
`word` entities per recognizer; `measure` assertions named `squim`; the PREPROCESS `verdict` entity
with `absent` and `derivatives`. `preprocess(store, source, config, hint=None, *, run_dir) ->
PreprocessResult` with `PreprocessResult.absent`.

---

### Task 3: TAXONOMY

**Prerequisite:** none outstanding — this node imports `detect_health_acoustic_events` and
`HEAR_MODEL_ID`/`HEAR_REVISION` at module top; both are on the merged tree (see the prerequisite
section above).

**Scope:** `src/senselab/audio/workflows/triage/nodes/taxonomy.py`; config additions under
`taxonomy`; the `seed_store` test fixture (mirrors Task 2's schema); tests at
`src/tests/audio/workflows/triage/nodes/taxonomy_test.py`.

**Design points this task must not get wrong (from `taxonomy.md`):**

- **TAXONOMY is advisory.** It predicts, gates nothing; every branch runs regardless. The node writes
  kind entities and a verdict on every path — including `fail` — and never raises on an outcome.
- **Eligibility before thresholds.** A detector votes only where its label space can express the
  kind; an ineligible detector is not a vote for absence.
- **Families, not detectors:** A = AudioSet = {YAMNet, AST}; B = lexical = {CrisperWhisper};
  C = health = {HeAR}. Airway has three eligible families, speech two.
- **HeAR is barred from the speech kind** — structurally: the speech fold never sees family C.
- **Presence needs `min_families[kind]` agreement; absence needs unanimity.** `min_families` is
  `null` in the config and stays null (`benchmarks/open.md`). The honest choice (N9): the node uses
  `config.get`, not `require`, and while the count is unmeasured declares presence only on
  **unanimity of eligible families** — the one sufficient condition every legal value agrees on —
  recording `min_families: "unmeasured"` on the kind entity; anything short of unanimity that is not
  unanimous absence is `undecided`. An override runs the design rule verbatim. `require()` would make
  the node unrunnable on the packaged config, which would gate the graph on an unmeasured number —
  worse than reporting honestly.
- **It localises nothing.** Every detector answers presence on its own grid; no grid is shared.
- **Model-call arguments are explicit** (deferred findings): AST gets `function_to_apply="sigmoid"`
  (AudioSet is multi-label; the softmax default makes 527 scores sum to 1) and `top_k=None`; HeAR
  gets `top_k=None` (dropping labels drops negative evidence). YAMNet is **not** re-run: its native
  windows come from the store (`yamnet_windows`, N6), which is also what makes its vote reproducible
  against PREPROCESS's.
- **Outcome:** `fail` = every screened kind absent; `flag` = any undecided; `pass` otherwise.
  `voice_no_words` is not screened and is written as a kind entity with state `"not_screened"`.

**Steps:**

- [ ] **Step 1 — add the config keys.** In `default.yaml`, replace the `taxonomy` block:

```yaml
taxonomy:
  min_families:
    airway: null
    speech: null
  presence_floor:
    yamnet: 0.5
    ast: null
    hear: 0.5
  audioset_airway_labels: [Cough, Throat clearing, Sneeze, Sniff, Breathing, Wheeze, Snoring, Gasp, Sigh]
  audioset_speech_labels: [Speech]
  hear_airway_labels: [Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear]
  lexical_airway_tokens: [cough, breath, breathe, throat, sniff, snore, sneeze, wheeze, gasp, sigh]
```

and append to the `derivation` block:

```
  Taxonomy presence floors -- yamnet 0.5 from the bimodal gaps benchmarks/hear-yamnet.md records
  (Cough 0.84 -> 0.27, Speech 0.92 -> 0.14, Breathing 0.59 -> 0.36: a threshold inside an empty
  interval); hear 0.5 from the same file's whole-span winner/runner-up gap (0.940-0.996 against
  0.02-0.41), a transfer from whole-span inputs to sliding windows that has not been separately
  measured. ast is null: AST's sigmoid scores have never been measured on the reference recording,
  so its member abstains and is recorded until someone measures a floor.

  Taxonomy label vocabularies -- semantic mappings, not thresholds: which of each detector's labels
  can express each kind, read off the label inventories (AudioSet's 521, HEAR_EVENT_LABELS' eight,
  CrisperWhisper's bracketed non-lexical tokens). Not fitted; overridable. benchmarks/taxonomy.md
  records why no single AudioSet roll-up label exists.
```

- [ ] **Step 2 — add the `seed_store` fixture** to
`src/tests/audio/workflows/triage/nodes/conftest.py` (append; it mirrors the Task 2 store schema,
which Task 2's tests hold PREPROCESS to, so drift is caught there):

```python
@pytest.fixture
def seed_store(tmp_path: Path) -> Callable[..., dict]:
    """A builder writing PREPROCESS-shaped entities into a store — the Task 2 schema, seeded.

    Writes a real plain-stream WAV so nodes that slice audio can, and entities for spans, words,
    YAMNet windows, silence and no_contrast as requested.
    """

    def _seed(
        store: ProvStore,
        *,
        spans: tuple = (),
        words: tuple = (),
        yamnet_windows: list | None = None,
        silence_windows: list | None = None,
        no_contrast_k: float | None = None,
        asr_available: bool = True,
        k_db: float = 18.0,
        duration_s: float = 4.0,
    ) -> dict:
        (tmp_path / "streams").mkdir(exist_ok=True)
        (tmp_path / "derivatives").mkdir(exist_ok=True)
        sf.write(str(tmp_path / "streams" / "plain.wav"), burst_samples(duration_s=duration_s), 16000)
        activity = store.activity(node="PREPROCESS", step="seed", parameters={})
        agent = store.agent(agent_type="software", version="senselab test-seed")
        store.was_associated_with(activity, agent)
        ids: dict = {"spans": [], "words": []}

        plain_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "plain",
                "path": "streams/plain.wav",
                "sampling_rate": 16000,
                "channels": 1,
                "peak_scale": 1.0,
            },
        )
        store.was_generated_by(plain_id, activity)
        ids["plain"] = plain_id

        for start, end, contrast in spans:
            span_id = store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"peak_over_floor_db": contrast, "k_db": k_db, "signal": "preemphasised"},
            )
            store.was_generated_by(span_id, activity)
            ids["spans"].append(span_id)

        if no_contrast_k is not None:
            nc_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "spans_no_contrast",
                    "signal": "preemphasised",
                    "k_db": no_contrast_k,
                    "reason": "seeded",
                },
            )
            store.was_generated_by(nc_id, activity)
            ids["no_contrast"] = nc_id

        if yamnet_windows is not None:
            path = tmp_path / "derivatives" / "yamnet_windows.json"
            path.write_text(json.dumps(yamnet_windows))
            yw_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "yamnet_windows",
                    "signal": "plain",
                    "path": "derivatives/yamnet_windows.json",
                    "n_windows": len(yamnet_windows),
                },
            )
            store.was_generated_by(yw_id, activity)
            ids["yamnet_windows"] = yw_id

        if silence_windows is not None:
            s_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={"name": "silence", "signal": "plain", "threshold": 0.5, "windows": silence_windows},
            )
            store.was_generated_by(s_id, activity)
            ids["silence"] = s_id

        if asr_available:
            asr_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "asr_crisperwhisper",
                    "signal": "plain",
                    "recognizer": "nyralabs/CrisperWhisper2.0_turbo",
                    "transcript": " ".join(str(w["text"]) for w in words),
                    "word_ids": [],
                    "timestamp_source": "native",
                },
            )
            store.was_generated_by(asr_id, activity)
            ids["asr"] = asr_id

        for word in words:
            word_id = store.entity(
                prov_type="word",
                extent=(float(word["start"]), float(word["end"])),
                attributes={
                    "text": str(word["text"]),
                    "score": 0.9,
                    "recognizer": "nyralabs/CrisperWhisper2.0_turbo",
                    "timestamp_source": "native",
                },
            )
            store.was_generated_by(word_id, activity)
            ids["words"].append(word_id)
        return ids

    return _seed
```

with `import json` added to conftest's imports.

- [ ] **Step 3 — write the failing tests.**

`src/tests/audio/workflows/triage/nodes/taxonomy_test.py`:

```python
"""TAXONOMY predicts kinds by family agreement. Advisory: it gates nothing and runs on every path.

AST and HeAR are monkeypatched on the node module; YAMNet's windows come from the seeded store.
"""

from pathlib import Path
from typing import Any, Callable

import pytest

from senselab.audio.data_structures import AudioClassificationResult
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import taxonomy as node
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


class _FakeModel:
    """A model spec stub carrying path_or_uri and a resolved commit."""

    def __init__(self, path_or_uri: str) -> None:
        """Stub a resolved model."""
        self.path_or_uri = path_or_uri
        self.commit_sha = "b" * 40


def _yamnet_window(start: float, end: float, scores: dict[str, float]) -> dict[str, Any]:
    """One YAMNet-shaped window."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {
        "start": start,
        "end": end,
        "label_scores": [{label: score} for label, score in ranked],
        "win_length": 0.96,
        "hop_length": 0.48,
    }


@pytest.fixture
def detector_calls() -> dict[str, list]:
    """Captured AST and HeAR call arguments."""
    return {"ast": [], "hear": []}


@pytest.fixture
def mock_detectors(monkeypatch: pytest.MonkeyPatch, detector_calls: dict[str, list]) -> dict[str, Any]:
    """Replace AST and HeAR with controllable fakes; return the mutable score dicts.

    AST's payload mirrors classify_audios' whole-audio return (AudioClassificationResult with
    parallel labels/scores); HeAR's mirrors detect_health_acoustic_events (windowed dicts with
    descending single-key label_scores over the eight labels).
    """
    scores = {"ast": {"Cough": 0.1, "Speech": 0.1}, "hear": {"Cough": 0.1, "Speech": 0.01}}
    monkeypatch.setattr(node, "_ast_model", lambda: _FakeModel(node.AST_ID))

    def fake_ast(audios: list, model: object, top_k: int | None = None, **kwargs: object) -> list:
        """AST, whole-audio mode."""
        detector_calls["ast"].append({"top_k": top_k, **kwargs})
        labels = list(scores["ast"])
        return [AudioClassificationResult(labels=labels, scores=[scores["ast"][label] for label in labels])]

    def fake_hear(
        audios: list,
        model: str = "hear-event-detector",
        device: object = None,
        hop_length: float = 0.25,
        top_k: int | None = None,
    ) -> list:
        """HeAR's sliding detector."""
        detector_calls["hear"].append({"top_k": top_k, "hop_length": hop_length})
        ranked = sorted(scores["hear"].items(), key=lambda kv: -kv[1])
        window = {
            "start": 0.0,
            "end": 2.0,
            "label_scores": [{label: score} for label, score in ranked],
            "win_length": 2.0,
            "hop_length": hop_length,
        }
        return [[window] for _ in audios]

    monkeypatch.setattr(node, "classify_audios", fake_ast)
    monkeypatch.setattr(node, "detect_health_acoustic_events", fake_hear)
    return scores


def _kind(store: ProvStore, name: str) -> Any:
    """The one kind entity for this kind."""
    [entity] = [e for e in store.entities("kind") if e.attributes["kind"] == name]
    return entity


class TestEligibility:
    """Who may vote, per kind."""

    def test_hear_is_barred_from_the_speech_kind(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """A strong HeAR Speech score contributes nothing: speech is folded from families A and B."""
        mock_detectors["hear"]["Speech"] = 0.99
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Speech": 0.1})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        speech = _kind(store, "speech")
        assert "C_health" not in speech.attributes["families"]
        assert result.kinds["speech"] == "absent"

    def test_lexical_airway_vote_reads_bracketed_tokens_only(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """[cough] votes airway-present; the plain word "cough" is lexical content, not an event."""
        seed_store(
            store,
            yamnet_windows=[_yamnet_window(0.0, 0.96, {"Speech": 0.1})],
            words=({"text": "[cough]", "start": 1.0, "end": 1.2},),
        )
        taxonomy(store, "plain", config, run_dir=tmp_path)
        airway = _kind(store, "airway")
        assert airway.attributes["families"]["B_lexical"]["state"] == "present"
        speech = _kind(store, "speech")
        assert speech.attributes["families"]["B_lexical"]["state"] == "absent"


class TestTheFold:
    """Presence needs agreement, absence needs unanimity, and the unmeasured count stays honest."""

    def test_unanimous_presence_is_present_while_min_families_is_unmeasured(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """All three eligible families agree, so any legal min_families would agree too."""
        mock_detectors["hear"]["Cough"] = 0.9
        seed_store(
            store,
            yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})],
            words=({"text": "[cough]", "start": 1.0, "end": 1.2},),
        )
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds["airway"] == "present"
        assert _kind(store, "airway").attributes["min_families"] == "unmeasured"

    def test_disagreement_without_min_families_is_undecided_and_flags(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """One family present and two absent cannot be adjudicated without the count."""
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds["airway"] == "undecided"
        assert result.verdict.outcome is Outcome.FLAG

    def test_a_min_families_override_applies_the_design_rule(
        self,
        store: ProvStore,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """With min_families.airway = 2, two present families out of three decide presence."""
        override = tmp_path / "override.yaml"
        override.write_text("taxonomy:\n  min_families:\n    airway: 2\n")
        config = load_triage_config(override)
        mock_detectors["hear"]["Cough"] = 0.9
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds["airway"] == "present"
        assert _kind(store, "airway").attributes["min_families"] == 2

    def test_an_out_of_range_override_raises(
        self,
        store: ProvStore,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """min_families beyond the eligible family count is a configuration error, not a fold."""
        override = tmp_path / "override.yaml"
        override.write_text("taxonomy:\n  min_families:\n    airway: 5\n")
        config = load_triage_config(override)
        seed_store(store, yamnet_windows=[], words=())
        with pytest.raises(ValueError, match="min_families"):
            taxonomy(store, "plain", config, run_dir=tmp_path)

    def test_absence_needs_unanimity_and_all_absent_fails(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """Every eligible family says absent for both screened kinds: the prediction is fail."""
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Speech": 0.1, "Cough": 0.1})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds == {"airway": "absent", "speech": "absent", "voice_no_words": "not_screened"}
        assert result.verdict.outcome is Outcome.FAIL

    def test_speech_present_with_airway_absent_passes(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """Present + absent with no undecided kind is a pass."""
        seed_store(
            store,
            yamnet_windows=[_yamnet_window(0.0, 0.96, {"Speech": 0.9})],
            words=({"text": "hello", "start": 1.0, "end": 1.2},),
        )
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds["speech"] == "present"
        assert result.kinds["airway"] == "absent"
        assert result.verdict.outcome is Outcome.PASS


class TestMembersAndArguments:
    """Member-level honesty and the explicit model arguments."""

    def test_ast_abstains_while_its_floor_is_null(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """AST's presence floor ships unmeasured; its member abstains and the record says why."""
        mock_detectors["ast"]["Cough"] = 0.99
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        taxonomy(store, "plain", config, run_dir=tmp_path)
        family_a = _kind(store, "airway").attributes["families"]["A_audioset"]
        assert family_a["members"]["ast"]["state"] == "abstained"
        assert family_a["state"] == "present"  # YAMNet's vote carries the family

    def test_model_arguments_are_explicit(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
        detector_calls: dict[str, list],
    ) -> None:
        """AST runs with sigmoid and no top-k truncation; HeAR keeps all eight labels."""
        seed_store(store, yamnet_windows=[], words=())
        taxonomy(store, "plain", config, run_dir=tmp_path)
        [ast_call] = detector_calls["ast"]
        assert ast_call["function_to_apply"] == "sigmoid"
        assert ast_call["top_k"] is None
        [hear_call] = detector_calls["hear"]
        assert hear_call["top_k"] is None

    def test_advisory_on_fail_everything_is_still_written(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """fail is a prediction, not a gate: three kind entities and a verdict exist regardless."""
        seed_store(store, yamnet_windows=[], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert len(store.entities("kind")) == 3
        assert _kind(store, "voice_no_words").attributes["state"] == "not_screened"
        assert store.get_entity(result.verdict_entity_id).attributes["kinds"] == result.kinds
```

- [ ] **Step 4 — run them; expect failure** (`ImportError`, then assertion failures as the node grows).
  `uv run pytest src/tests/audio/workflows/triage/nodes/taxonomy_test.py -x -q`

- [ ] **Step 5 — implement.**

`src/senselab/audio/workflows/triage/nodes/taxonomy.py`:

```python
"""TAXONOMY — which kinds are in the recording. Advisory: it predicts, and gates nothing.

Each detector answers presence on its own grid; families vote, not detectors. Presence needs
``min_families[kind]`` agreement — unanimity while that count is unmeasured — and absence needs
unanimity of the eligible families. Every branch runs regardless of the outcome here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.health_acoustics.hear import HEAR_MODEL_ID, HEAR_REVISION
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import HFModel
from senselab.utils.prov_store import ProvStore

NODE = "TAXONOMY"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SCREENED_KINDS = ("airway", "speech")


def _ast_model() -> HFModel:
    """The AST model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=AST_ID, revision="main")


@dataclass(frozen=True)
class TaxonomyResult(NodeResult):
    """TAXONOMY's result.

    Attributes:
        kinds: Predicted state per kind — the design's verdict mapping.
    """

    kinds: dict[str, str]


def _windowed_max(windows: list[dict[str, Any]], labels: set[str]) -> tuple[float, str | None]:
    """The highest score any of these labels reaches in any window, and which label reached it."""
    best, best_label = 0.0, None
    for window in windows:
        for pair in label_scores(window):
            for label, score in pair.items():
                if label in labels and float(score) > best:
                    best, best_label = float(score), label
    return best, best_label


def _is_bracketed(text: str) -> bool:
    """Whether a recognizer token is a non-lexical annotation like ``[cough]``."""
    return text.startswith("[") and text.endswith("]")


def taxonomy(  # noqa: C901 — one member per detector, one fold per kind
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> TaxonomyResult:
    """Predict which kinds are in the recording. Nothing downstream is gated on the answer.

    Args:
        store: The provenance store, holding PREPROCESS's derivatives.
        source: The store-held stream name to classify, ``"plain"``.
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read (the design's signature has none).
        run_dir: The run directory sidecar paths are relative to.

    Returns:
        The verdict, the kind entity ids as the view, and the per-kind states.

    Raises:
        ValueError: If a ``taxonomy.min_families`` override lies outside ``[1, n_eligible]``.
    """
    software = software_agent(store)
    stream_id, plain = resolve_stream(store, run_dir, source)

    floors = {
        "yamnet": config.get("taxonomy.presence_floor.yamnet"),
        "ast": config.get("taxonomy.presence_floor.ast"),
        "hear": config.get("taxonomy.presence_floor.hear"),
    }
    audioset_labels = {
        "airway": {str(label) for label in config.require("taxonomy.audioset_airway_labels")},
        "speech": {str(label) for label in config.require("taxonomy.audioset_speech_labels")},
    }
    hear_labels = {str(label) for label in config.require("taxonomy.hear_airway_labels")}
    lexical_tokens = [str(token).lower() for token in config.require("taxonomy.lexical_airway_tokens")]

    yamnet_meas = find_measurement(store, "yamnet_windows")
    yamnet_windows: list[dict[str, Any]] | None = None
    if yamnet_meas is not None:
        yamnet_windows = json.loads((run_dir / yamnet_meas.attributes["path"]).read_text())

    ast_scores: dict[str, float] | None = None
    ast_error: str | None = None
    try:
        model = _ast_model()
        ast_agent = store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
        ast_activity = store.activity(
            node=NODE,
            step="classify_ast",
            parameters={"model": str(model.path_or_uri), "function_to_apply": "sigmoid", "top_k": None},
        )
        store.was_associated_with(ast_activity, ast_agent)
        store.used(ast_activity, stream_id)
        [ast_result] = classify_audios([plain], model=model, function_to_apply="sigmoid", top_k=None)
        ast_scores = {label: float(score) for label, score in zip(ast_result.labels, ast_result.scores)}
    except Exception as err:  # noqa: BLE001 — an unavailable detector abstains; it is not absence evidence
        ast_error = type(err).__name__

    hear_windows: list[dict[str, Any]] | None = None
    hear_error: str | None = None
    try:
        hear_agent = store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION)
        hear_activity = store.activity(node=NODE, step="classify_hear", parameters={"model": HEAR_MODEL_ID})
        store.was_associated_with(hear_activity, hear_agent)
        store.used(hear_activity, stream_id)
        [hear_windows] = detect_health_acoustic_events([plain], top_k=None)
    except Exception as err:  # noqa: BLE001 — same rule as AST
        hear_error = type(err).__name__

    words = [w for w in store.entities("word") if w.attributes.get("recognizer") == CRISPERWHISPER_ID]
    crisper_available = find_measurement(store, "asr_crisperwhisper") is not None

    def _yamnet_member(kind: str) -> dict[str, Any]:
        """Family A's first member, read from the store's native windows."""
        if yamnet_windows is None:
            return {"state": "unavailable", "why": "yamnet_windows absent from the store"}
        if floors["yamnet"] is None:
            return {"state": "abstained", "why": "presence floor unmeasured"}
        best, best_label = _windowed_max(yamnet_windows, audioset_labels[kind])
        state = "present" if best >= float(floors["yamnet"]) else "absent"
        return {"state": state, "max_score": best, "label": best_label}

    def _ast_member(kind: str) -> dict[str, Any]:
        """Family A's second member, file-level over its own 10.24 s frame."""
        if ast_scores is None:
            return {"state": "unavailable", "why": ast_error or "no scores"}
        if floors["ast"] is None:
            return {"state": "abstained", "why": "presence floor unmeasured"}
        best, best_label = 0.0, None
        for label, score in ast_scores.items():
            if label in audioset_labels[kind] and score > best:
                best, best_label = score, label
        state = "present" if best >= float(floors["ast"]) else "absent"
        return {"state": state, "max_score": best, "label": best_label}

    def _lexical_member(kind: str) -> dict[str, Any]:
        """Family B: words for speech, bracketed non-lexical tokens for airway."""
        if not crisper_available:
            return {"state": "unavailable", "why": "asr_crisperwhisper absent from the store"}
        if kind == "speech":
            lexical = [
                w for w in words if w.attributes.get("text") and not _is_bracketed(str(w.attributes["text"]))
            ]
            return {"state": "present" if lexical else "absent", "n_words": len(lexical)}
        matched = [
            w.id
            for w in words
            if w.attributes.get("text")
            and _is_bracketed(str(w.attributes["text"]))
            and any(token in str(w.attributes["text"]).lower() for token in lexical_tokens)
        ]
        return {"state": "present" if matched else "absent", "word_ids": matched}

    def _hear_member() -> dict[str, Any]:
        """Family C, airway only: the detector's own sliding grid."""
        if hear_windows is None:
            return {"state": "unavailable", "why": hear_error or "no windows"}
        if floors["hear"] is None:
            return {"state": "abstained", "why": "presence floor unmeasured"}
        best, best_label = _windowed_max(hear_windows, hear_labels)
        state = "present" if best >= float(floors["hear"]) else "absent"
        return {"state": state, "max_score": best, "label": best_label}

    def _family_a(kind: str) -> dict[str, Any]:
        """AudioSet family: members must agree; an abstaining member leaves it to the other."""
        members = {"yamnet": _yamnet_member(kind), "ast": _ast_member(kind)}
        votes = [m["state"] for m in members.values() if m["state"] in ("present", "absent")]
        if votes and all(v == votes[0] for v in votes):
            state = votes[0]
        else:
            state = "unsure"
        return {"state": state, "members": members}

    def _single(member_name: str, member: dict[str, Any]) -> dict[str, Any]:
        """A one-member family: the member's vote, unsure when it cannot vote."""
        state = member["state"] if member["state"] in ("present", "absent") else "unsure"
        return {"state": state, "members": {member_name: member}}

    def _fold_kind(kind: str, families: dict[str, dict[str, Any]]) -> tuple[str, Any]:
        """The design's presence/absence/undecided fold, honest about the unmeasured count.

        An override is validated before any state is read, so a bad count raises whatever the
        recording contains.
        """
        states = [family["state"] for family in families.values()]
        min_families = config.get(f"taxonomy.min_families.{kind}")
        if min_families is not None:
            min_int = int(min_families)
            if not 1 <= min_int <= len(states):
                raise ValueError(
                    f"taxonomy.min_families.{kind} = {min_int} lies outside [1, {len(states)}] eligible families"
                )
            if all(state == "absent" for state in states):
                return "absent", min_int
            if sum(1 for state in states if state == "present") >= min_int:
                return "present", min_int
            return "undecided", min_int
        if states and all(state == "absent" for state in states):
            return "absent", "unmeasured"
        if states and all(state == "present" for state in states):
            return "present", "unmeasured"
        return "undecided", "unmeasured"

    airway_families = {
        "A_audioset": _family_a("airway"),
        "B_lexical": _single("crisperwhisper", _lexical_member("airway")),
        "C_health": _single("hear", _hear_member()),
    }
    # HeAR is barred from the speech kind: family C is not eligible and never enters this fold.
    speech_families = {
        "A_audioset": _family_a("speech"),
        "B_lexical": _single("crisperwhisper", _lexical_member("speech")),
    }

    fold = store.activity(node=NODE, step="fold", parameters={"kinds": list(SCREENED_KINDS)})
    store.was_associated_with(fold, software)
    store.used(fold, stream_id)
    if yamnet_meas is not None:
        store.used(fold, yamnet_meas.id)
    for word in words:
        store.used(fold, word.id)

    kinds_out: dict[str, str] = {}
    view: list[str] = []
    for kind, families in (("airway", airway_families), ("speech", speech_families)):
        state, min_recorded = _fold_kind(kind, families)
        kinds_out[kind] = state
        kind_id = store.entity(
            prov_type="kind",
            extent=None,
            attributes={"kind": kind, "state": state, "families": families, "min_families": min_recorded},
        )
        store.was_generated_by(kind_id, fold)
        store.was_attributed_to(kind_id, software)
        view.append(kind_id)

    residual_id = store.entity(
        prov_type="kind",
        extent=None,
        attributes={"kind": "voice_no_words", "state": "not_screened", "families": {}, "min_families": None},
    )
    store.was_generated_by(residual_id, fold)
    store.was_attributed_to(residual_id, software)
    view.append(residual_id)
    kinds_out["voice_no_words"] = "not_screened"

    screened = [kinds_out[kind] for kind in SCREENED_KINDS]
    if all(state == "absent" for state in screened):
        outcome, why = Outcome.FAIL, "every screened kind is absent; nothing is predicted present"
    elif any(state == "undecided" for state in screened):
        undecided = [kind for kind in SCREENED_KINDS if kinds_out[kind] == "undecided"]
        outcome, why = Outcome.FLAG, "undecided: " + ", ".join(undecided)
    else:
        outcome, why = Outcome.PASS, "every screened kind is decided, and at least one is present"

    verdict_id, verdict = write_verdict(
        store, fold, software, node=NODE, outcome=outcome, kind=None, why=why, detail={"kinds": kinds_out}
    )
    view.append(verdict_id)
    return TaxonomyResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, kinds=kinds_out)
```

- [ ] **Step 6 — run the tests; expect all PASS**, then the whole node directory:
  `uv run pytest src/tests/audio/workflows/triage/nodes -x -q`

- [ ] **Step 7 — lint, type-check, config tests** (default.yaml changed again — same note as Task 2
  Step 6), **commit**: `git add -A && git commit -m "feat(triage): TAXONOMY predicts kinds by family agreement"`

**Interfaces:**

*Consumed:*
- `classify_audios(audios, model: HFModel, function_to_apply="sigmoid", top_k=None) -> List[AudioClassificationResult]` — whole-audio mode; `AudioClassificationResult.labels/.scores` are parallel lists (`audio/data_structures/audio_classification_result.py:8`). The two explicit arguments are the deferred findings (capability-map §4.2) — a test pins them.
- `detect_health_acoustic_events(audios, model="hear-event-detector", device=None, hop_length=0.25, top_k=None) -> List[List[Dict]]` — post-merge; windowed dicts as in the prerequisite section.
- `HEAR_MODEL_ID`, `HEAR_REVISION` (a 40-hex literal) — `health_acoustics/hear.py`, post-merge.
- Store reads: `yamnet_windows` measurement (its json sidecar), `asr_crisperwhisper` measurement, `word` entities (`recognizer`, `text`), the `plain` stream via `resolve_stream`.
- `TriageConfig.get` for `taxonomy.min_families.*` and `taxonomy.presence_floor.*` (null-tolerant by design, N9/N10); `require` for the label vocabularies.

*Produced (the sibling plan's read surface):* three `kind` entities exactly as the schema table
states — VERDICT maps `not_screened` onto its fold's undecided semantics, and reads `state` per kind;
SPEECH and VOICE read nothing from TAXONOMY (it is advisory). The TAXONOMY `verdict` entity carries
`kinds`. `taxonomy(store, source, config, hint=None, *, run_dir) -> TaxonomyResult` with `.kinds`.

---

### Task 4: AIRWAY

**Prerequisite:** none outstanding — the `triage` merge landed HeAR, and `span_to_hear_buffer` +
`HEAR_WINDOW_SECONDS` are on the merged tree (`health_acoustics/hear.py:387,130`). The node calls the
module function; nothing is inlined (N13).

**Scope:** `src/senselab/audio/workflows/triage/nodes/airway.py`; config additions (`hear.placement`,
`airway.*`); tests at `src/tests/audio/workflows/triage/nodes/airway_test.py`.

**Design points this task must not get wrong (from `branch-airway.md`):**

- **AIRWAY proposes nothing.** It labels, confirms and contests the `span` elements PREPROCESS wrote
  at `K` = `spans.k_db.airway`; a `no_contrast` finding counts only at that same `K` (N8).
- **HeAR classifies the whole span placed in a 2 s silent buffer** (`hear.window_s`) containing
  nothing else, centred (`hear.placement`, N13). The buffer comes from
  `span_to_hear_buffer(plain, start, end, placement=...)` — a buffer of exactly the model's window
  makes the detector score exactly one window; a span the function refuses (longer than the window)
  takes the sliding path (N14).
- **YAMNet confirms from its own native windows by coverage** — never from a padded span, never from
  the span as an input. Structurally enforced: the module has no import of `classify_audios` at all;
  its YAMNet evidence is the store's `yamnet_windows`.
- **Labels are restricted to `labels_of_interest`**, default `{Cough, Breathe}`; a best label below
  `hear.label_floor` leaves the span with **no label assertion and no substitute record** — an
  unlabelled span is a span without a `label` assertion, and nothing downstream reads it.
- **A contest flags, never relabels.**
- **Lexical contamination** is any CrisperWhisper lexical word intersecting
  `[first labelled span start, last labelled span end]` — gaps included, unlabelled spans never
  extending it, bracketed tokens excluded (N15). The flag carries **word ids only**.
- **A hint changes only what an absence means.** It never creates a span, relabels one, alters a
  threshold, or promotes a `fail` to a `pass`.
- **The figure is an artifact, not a product of the store** — a rendering failure changes no verdict.

**Steps:**

- [ ] **Step 1 — add the config keys.** In `default.yaml`, extend `hear` and add `airway`:

```yaml
hear:
  window_s: 2.0
  label_floor: 0.5
  placement: centre

airway:
  labels_of_interest: [Cough, Breathe]
  confirmation_map:
    Cough: [Cough]
    Breathe: [Breathing, Sigh, Gasp]
```

and append to the `derivation` block:

```
  HeAR placement centre -- benchmarks/hear-yamnet.md's whole-span numbers were measured with the
  span centred in the buffer (benchmarks/scripts/spaninput.py). span_to_hear_buffer implements
  start and end placements too, but those are different inputs and have not been measured, so the
  node accepts only centre.

  Airway labels_of_interest {Cough, Breathe} -- branch-airway.md's default, from HeAR's eight. The
  confirmation map Cough -> {Cough}, Breathe -> {Breathing, Sigh, Gasp} is branch-airway.md's step 2
  table: which AudioSet labels corroborate each HeAR label. Vocabulary, not thresholds.
```

- [ ] **Step 2 — write the failing tests.**

`src/tests/audio/workflows/triage/nodes/airway_test.py`:

```python
"""AIRWAY interprets PREPROCESS's spans: HeAR labels whole spans in a silent buffer, YAMNet
confirms from its own native windows, ASR words are presence-only evidence, a hint changes only
what an absence means. HeAR is monkeypatched on the node module; nothing here loads weights."""

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes import airway as node
from senselab.audio.workflows.triage.nodes.airway import airway
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

matplotlib.use("Agg", force=True)


def _yamnet_window(start: float, end: float, scores: dict[str, float]) -> dict[str, Any]:
    """One YAMNet-shaped window."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {
        "start": start,
        "end": end,
        "label_scores": [{label: score} for label, score in ranked],
        "win_length": 0.96,
        "hop_length": 0.48,
    }


@pytest.fixture
def hear_calls() -> list[dict[str, Any]]:
    """Captured HeAR call payloads, in call order."""
    return []


@pytest.fixture
def hear_scores() -> dict[str, float]:
    """The mutable label scores the fake detector returns for every window."""
    return {"Cough": 0.97, "Breathe": 0.2, "Speech": 0.01}


@pytest.fixture
def mock_hear(
    monkeypatch: pytest.MonkeyPatch, hear_calls: list[dict[str, Any]], hear_scores: dict[str, float]
) -> None:
    """Replace the HeAR detector; the payload mirrors detect_health_acoustic_events' return."""

    def fake_hear(
        audios: list,
        model: str = "hear-event-detector",
        device: object = None,
        hop_length: float = 0.25,
        top_k: int | None = None,
    ) -> list:
        """One window per 2 s of each input, all carrying the fixture scores."""
        hear_calls.append(
            {
                "hop_length": hop_length,
                "lengths": [int(a.waveform.shape[-1]) for a in audios],
                "waveforms": [a.waveform.clone() for a in audios],
            }
        )
        ranked = sorted(hear_scores.items(), key=lambda kv: -kv[1])
        window = {
            "start": 0.0,
            "end": 2.0,
            "label_scores": [{label: score} for label, score in ranked],
            "win_length": 2.0,
            "hop_length": hop_length,
        }
        return [[dict(window)] for _ in audios]

    monkeypatch.setattr(node, "detect_health_acoustic_events", fake_hear)


def _labels(store: ProvStore) -> list:
    """Every label assertion in the store."""
    return [a for a in store.entities("assertion") if a.attributes.get("verb") == "label"]


def _answers(store: ProvStore, verb: str) -> list:
    """Every confirm/contest/abstain assertion of one verb."""
    return [a for a in store.entities("assertion") if a.attributes.get("verb") == verb]


class TestHearClassification:
    """Step 1: the whole span, buffered by the module's own function."""

    def test_the_whole_span_is_buffered_by_the_module_function(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
        hear_calls: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The buffer comes from span_to_hear_buffer at the configured placement; one window scored."""
        from senselab.audio.tasks.health_acoustics.hear import span_to_hear_buffer

        buffer_calls: list[dict[str, Any]] = []

        def recording_buffer(audio: object, start_s: float, end_s: float, *, placement: str) -> object:
            """The real function, with its arguments captured."""
            buffer_calls.append({"start_s": start_s, "end_s": end_s, "placement": placement})
            return span_to_hear_buffer(audio, start_s, end_s, placement=placement)

        monkeypatch.setattr(node, "span_to_hear_buffer", recording_buffer)
        ids = seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, run_dir=tmp_path)
        [buffer_call] = buffer_calls
        assert buffer_call == {"start_s": 1.5, "end_s": 1.65, "placement": "centre"}
        window_s = float(config.require("hear.window_s"))
        [call] = hear_calls
        assert call["hop_length"] == window_s
        assert call["lengths"] == [int(window_s * 16000)]  # the function's whole-window buffer
        [label] = _labels(store)
        assert label.attributes["label"] == "Cough"
        assert label.attributes["input"] == "buffered"
        assert ids["spans"][0] in store.derived_from(label.id)
        assert result.verdict.outcome in (Outcome.PASS, Outcome.FLAG)

    def test_a_span_longer_than_the_window_takes_the_sliding_path(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
        hear_calls: list[dict[str, Any]],
    ) -> None:
        """span_to_hear_buffer refuses a 3 s span; its own audio is scanned and the assertion says so."""
        seed_store(store, spans=((0.5, 3.5, 40.0),), yamnet_windows=[])
        airway(store, "plain", config, run_dir=tmp_path)
        [call] = hear_calls
        assert call["hop_length"] == 0.25  # the detector's own default, not the buffer hop
        assert call["lengths"] == [int(3.0 * 16000)]
        [label] = _labels(store)
        assert label.attributes["input"] == "sliding"

    def test_a_best_label_below_the_floor_leaves_the_span_unlabelled(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
        hear_scores: dict[str, float],
    ) -> None:
        """No label assertion, no substitute record, and the branch flags."""
        hear_scores.update({"Cough": 0.3, "Breathe": 0.2})
        seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert _labels(store) == []
        assert result.verdict.outcome is Outcome.FLAG
        verdict = store.get_entity(result.verdict_entity_id)
        assert verdict.attributes["labelled_n"] == 0

    def test_airway_has_no_path_to_yamnet_as_a_model(self) -> None:
        """YAMNet is read from the store's native windows; the module cannot classify with it."""
        assert not hasattr(node, "classify_audios")


class TestYamnetConfirmation:
    """Step 2: coverage over native windows; confirm, contest or abstain — never relabel."""

    def test_matching_coverage_confirms(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """Cough windows over a Cough span confirm it, with the coverage recorded."""
        windows = [
            _yamnet_window(1.44, 2.40, {"Cough": 0.9}),
            _yamnet_window(0.96, 1.92, {"Cough": 0.8}),
        ]
        seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=windows)
        result = airway(store, "plain", config, run_dir=tmp_path)
        [confirm] = _answers(store, "confirm")
        assert confirm.attributes["winner"] == "Cough"
        assert confirm.attributes["coverage"] == 1.0
        assert confirm.attributes["n_windows"] == 2
        [label] = _labels(store)
        assert label.id in store.derived_from(confirm.id)
        assert result.verdict.outcome is Outcome.PASS

    def test_a_confident_outside_label_contests_without_relabelling(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """Speech coverage against a Cough label contests and flags; the label stands."""
        windows = [_yamnet_window(1.44, 2.40, {"Speech": 0.9, "Cough": 0.1})]
        seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=windows)
        result = airway(store, "plain", config, run_dir=tmp_path)
        [contest] = _answers(store, "contest")
        assert contest.attributes["winner"] == "Speech"
        [label] = _labels(store)
        assert label.attributes["label"] == "Cough"
        assert result.verdict.outcome is Outcome.FLAG
        assert store.get_entity(result.verdict_entity_id).attributes["contested_n"] == 1

    def test_nothing_confident_anywhere_abstains_single_source(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """No window reaches the coverage threshold: the label stands, marked single-source."""
        windows = [_yamnet_window(1.44, 2.40, {"Cough": 0.2, "Speech": 0.1})]
        seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=windows)
        result = airway(store, "plain", config, run_dir=tmp_path)
        [abstain] = _answers(store, "abstain")
        assert abstain.attributes["best_coverage"] == 0.0
        assert abstain.attributes["n_windows"] == 1
        assert result.verdict.outcome is Outcome.PASS

    def test_breathe_is_confirmed_by_sigh(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
        hear_scores: dict[str, float],
    ) -> None:
        """The confirmation map sends Breathe to {Breathing, Sigh, Gasp}."""
        hear_scores.update({"Cough": 0.1, "Breathe": 0.95})
        windows = [_yamnet_window(1.44, 2.40, {"Sigh": 0.8})]
        seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=windows)
        result = airway(store, "plain", config, run_dir=tmp_path)
        [confirm] = _answers(store, "confirm")
        assert confirm.attributes["winner"] == "Sigh"
        assert confirm.attributes["mapped_to"] == "Breathe"
        assert result.verdict.outcome is Outcome.PASS


class TestLexicalContamination:
    """Step 3: the interval spans the gaps; brackets and out-of-interval words do not count."""

    def test_a_word_in_the_gap_between_labelled_spans_flags_by_id_only(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """The interval covers first-start to last-end; the flag names word ids, never text."""
        ids = seed_store(
            store,
            spans=((1.0, 1.2, 40.0), (2.5, 2.7, 40.0)),
            yamnet_windows=[],
            words=(
                {"text": "Marisol", "start": 1.8, "end": 1.9},
                {"text": "[cough]", "start": 1.85, "end": 1.95},
                {"text": "later", "start": 3.5, "end": 3.6},
            ),
        )
        result = airway(store, "plain", config, run_dir=tmp_path)
        [flag] = [a for a in store.entities("assertion") if a.attributes.get("verb") == "flag"]
        assert flag.attributes["reason"] == "lexical_contamination"
        assert flag.attributes["word_ids"] == [ids["words"][0]]
        assert "Marisol" not in json.dumps(flag.attributes)
        [interval] = store.entities("interval")
        assert interval.extent == (1.0, 2.7)
        assert result.verdict.outcome is Outcome.FLAG

    def test_a_word_outside_the_interval_does_not_flag(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """Unlabelled spans never extend the interval and later words never enter it."""
        seed_store(
            store,
            spans=((1.0, 1.2, 40.0),),
            yamnet_windows=[],
            words=({"text": "later", "start": 3.5, "end": 3.6},),
        )
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert [a for a in store.entities("assertion") if a.attributes.get("verb") == "flag"] == []
        assert result.verdict.outcome is Outcome.PASS


class TestOutcomeAndHint:
    """Step 4: a hint conditions only what an absence means."""

    def test_no_spans_is_fail_and_a_hint_makes_it_flag(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """Nothing proposed: fail without a hint, flag with one — never a pass."""
        seed_store(store, spans=(), yamnet_windows=[], no_contrast_k=18.0)
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "no_contrast" in result.verdict.why
        hinted_store = ProvStore(run_id="hinted")
        seed_store(hinted_store, spans=(), yamnet_windows=[], no_contrast_k=18.0)
        hinted = airway(
            hinted_store, "plain", config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path
        )
        assert hinted.verdict.outcome is Outcome.FLAG

    def test_no_contrast_at_another_k_is_not_this_readers_no_contrast(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """no_contrast is a (K, recording) finding; a 12 dB finding says nothing at 18 dB."""
        seed_store(store, spans=(), yamnet_windows=[], no_contrast_k=12.0)
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "no_contrast" not in result.verdict.why

    def test_a_hint_changes_nothing_when_spans_are_labelled(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """With labelled spans the hint is inert: same pass either way."""
        seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS


class TestFigure:
    """The figure is an artifact; its failure changes no verdict."""

    def test_the_figure_is_written(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """One aligned figure per recording, under run_dir/figures."""
        seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.figure_path is not None
        assert result.figure_path.exists()

    def test_a_figure_failure_changes_no_verdict(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_hear: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rendering raising leaves the same outcome with figure_path None."""

        def broken_plot(*args: object, **kwargs: object) -> object:
            """A renderer crash."""
            raise RuntimeError("no display")

        monkeypatch.setattr(node, "plot_aligned_panels", broken_plot)
        seed_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.figure_path is None
```

- [ ] **Step 3 — run them; expect failure.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/airway_test.py -x -q`

- [ ] **Step 4 — implement.**

`src/senselab/audio/workflows/triage/nodes/airway.py`:

```python
"""AIRWAY — interpret PREPROCESS's spans. It proposes nothing: it labels, confirms and contests.

HeAR classifies each whole span placed in a silent buffer of exactly the model's window, via
``span_to_hear_buffer``; YAMNet confirms from its own native windows by coverage, never from a
padded span; ASR words are read for presence only. A hint changes only what an absence means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.health_acoustics.hear import HEAR_MODEL_ID, HEAR_REVISION, span_to_hear_buffer
from senselab.audio.tasks.plotting.plotting import plot_aligned_panels
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "AIRWAY"


@dataclass(frozen=True)
class AirwayResult(NodeResult):
    """AIRWAY's result.

    Attributes:
        figure_path: The aligned figure, or None when rendering failed. An artifact, not store
            content.
    """

    figure_path: Path | None


def _hint_declares_airway(hint: AudioHints | None, labels_of_interest: list[str]) -> bool:
    """Whether the caller declared airway content (decision N18)."""
    if hint is None:
        return False
    declared = {tag.lower() for tag in hint.may_contain}
    return bool(declared & ({label.lower() for label in labels_of_interest} | {"airway"}))


def _inside_certified_silence(span: Entity, silence_windows: list[dict[str, Any]] | None) -> bool | None:
    """Whether every silence-graded window overlapping the span was certified silent (N17)."""
    if silence_windows is None:
        return None
    start, end = span.extent or (0.0, 0.0)
    overlapping = [w for w in silence_windows if float(w["start"]) < end and float(w["end"]) > start]
    if not overlapping:
        return None
    return all(bool(w["is_silence"]) for w in overlapping)


def _max_score(windows: list[dict[str, Any]], label: str) -> float:
    """The label's highest score across these windows."""
    best = 0.0
    for window in windows:
        for pair in window.get("label_scores", []):
            score = pair.get(label)
            if score is not None and float(score) > best:
                best = float(score)
    return best


def _best_of_interest(windows: list[dict[str, Any]], labels_of_interest: list[str]) -> dict[str, float]:
    """Each label of interest's highest score over these windows."""
    return {label: _max_score(windows, label) for label in labels_of_interest}


def airway(  # noqa: C901 — the branch's four steps, in order
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> AirwayResult:
    """Label, confirm and contest the spans PREPROCESS proposed at the airway K.

    Args:
        store: The provenance store, holding PREPROCESS's spans and derivatives.
        source: The store-held stream name HeAR's buffers are cut from, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain; read only to condition an absence.
        run_dir: The run directory; the figure goes under ``figures/``.

    Returns:
        The verdict, the view over the spans and assertions touched, and the figure path.

    Raises:
        ValueError: If ``hear.placement`` names an unimplemented placement.
    """
    software = software_agent(store)
    stream_id, plain = resolve_stream(store, run_dir, source)
    sr = int(plain.sampling_rate)

    k_db = float(config.require("spans.k_db.airway"))
    labels_of_interest = [str(label) for label in config.require("airway.labels_of_interest")]
    label_floor = float(config.require("hear.label_floor"))
    window_s = float(config.require("hear.window_s"))
    placement = str(config.require("hear.placement"))
    if placement != "centre":
        raise ValueError(f"hear.placement {placement!r} is not implemented; only 'centre' is")
    coverage_threshold = float(config.require("yamnet.coverage_threshold"))
    confirmation_map = {
        str(hear_label): {str(v) for v in yamnet_labels}
        for hear_label, yamnet_labels in config.require("airway.confirmation_map").items()
    }

    spans = [e for e in store.entities("span") if e.attributes.get("k_db") == k_db]
    spans.sort(key=lambda e: e.extent or (0.0, 0.0))
    hint_declares = _hint_declares_airway(hint, labels_of_interest)
    silence = find_measurement(store, "silence")
    silence_windows = silence.attributes.get("windows") if silence is not None else None

    if not spans:
        no_contrast = find_measurement(store, "spans_no_contrast")
        at_this_k = no_contrast is not None and no_contrast.attributes.get("k_db") == k_db
        reason = "PREPROCESS reported no_contrast at this K" if at_this_k else "no span was proposed at this K"
        activity = store.activity(node=NODE, step="classify", parameters={"k_db": k_db, "n_spans": 0})
        store.was_associated_with(activity, software)
        store.used(activity, stream_id)
        if at_this_k and no_contrast is not None:
            store.used(activity, no_contrast.id)
        if hint_declares:
            outcome = Outcome.FLAG
            why = reason + "; a hint declares airway content not found"
        else:
            outcome, why = Outcome.FAIL, reason
        verdict_id, verdict = write_verdict(
            store,
            activity,
            software,
            node=NODE,
            outcome=outcome,
            kind="airway",
            why=why,
            detail={
                "labelled_n": 0,
                "by_label": {},
                "contested_n": 0,
                "flags": [why] if outcome is Outcome.FLAG else [],
            },
        )
        return AirwayResult(verdict=verdict, view=(verdict_id,), verdict_entity_id=verdict_id, figure_path=None)

    # Step 1 — HeAR labels each span: the whole span, buffered by span_to_hear_buffer; a span the
    # function refuses (longer than the window) is scanned over its own audio instead.
    hear_agent = store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION)
    classify = store.activity(
        node=NODE,
        step="classify",
        parameters={
            "k_db": k_db,
            "labels_of_interest": labels_of_interest,
            "label_floor": label_floor,
            "window_s": window_s,
            "placement": placement,
            "n_spans": len(spans),
        },
    )
    store.was_associated_with(classify, hear_agent)
    store.used(classify, stream_id)
    for span in spans:
        store.used(classify, span.id)
    if silence is not None:
        store.used(classify, silence.id)

    buffered: list[tuple[Entity, Audio]] = []
    sliding: list[tuple[Entity, Audio]] = []
    for span in spans:
        start, end = span.extent or (0.0, 0.0)
        try:
            buffered.append((span, span_to_hear_buffer(plain, start, end, placement=placement)))
        except ValueError:  # the function refuses a span longer than the window (N14)
            segment = plain.waveform[:, int(start * sr) : int(end * sr)]
            sliding.append((span, Audio(waveform=segment, sampling_rate=sr)))

    scored: list[tuple[Entity, dict[str, float], str]] = []
    if buffered:
        outputs = detect_health_acoustic_events([audio for _, audio in buffered], hop_length=window_s)
        for (span, _), windows in zip(buffered, outputs):
            scored.append((span, _best_of_interest(windows, labels_of_interest), "buffered"))
    if sliding:
        outputs = detect_health_acoustic_events([audio for _, audio in sliding])
        for (span, _), windows in zip(sliding, outputs):
            scored.append((span, _best_of_interest(windows, labels_of_interest), "sliding"))

    label_ids: dict[str, str] = {}
    span_labels: dict[str, str] = {}
    by_label: dict[str, int] = {}
    for span, scores, input_kind in scored:
        best_label = max(scores, key=lambda label: scores[label])
        if scores[best_label] < label_floor:
            continue
        assertion_id = store.entity(
            prov_type="assertion",
            extent=span.extent,
            attributes={
                "verb": "label",
                "label": best_label,
                "score": scores[best_label],
                "scores": scores,
                "input": input_kind,
                "in_certified_silence": _inside_certified_silence(span, silence_windows),
            },
        )
        store.was_generated_by(assertion_id, classify)
        store.was_attributed_to(assertion_id, hear_agent)
        store.was_derived_from(assertion_id, span.id)
        label_ids[span.id] = assertion_id
        span_labels[span.id] = best_label
        by_label[best_label] = by_label.get(best_label, 0) + 1

    # Step 2 — YAMNet answers each label from its own native windows, by coverage.
    yamnet_meas = find_measurement(store, "yamnet_windows")
    yamnet_windows: list[dict[str, Any]] | None = None
    if yamnet_meas is not None:
        yamnet_windows = json.loads((run_dir / yamnet_meas.attributes["path"]).read_text())
    yamnet_agent = store.agent(
        agent_type="model",
        model_id="https://tfhub.dev/google/yamnet/1",
        unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
    )
    confirm_activity = store.activity(
        node=NODE, step="confirm", parameters={"coverage_threshold": coverage_threshold}
    )
    store.was_associated_with(confirm_activity, yamnet_agent)
    if yamnet_meas is not None:
        store.used(confirm_activity, yamnet_meas.id)

    contested_n = 0
    flags: list[str] = []
    answers: list[str] = []
    for span in spans:
        label_id = label_ids.get(span.id)
        if label_id is None:
            continue
        start, end = span.extent or (0.0, 0.0)
        overlapping = (
            [w for w in yamnet_windows if float(w["start"]) < end and float(w["end"]) > start]
            if yamnet_windows is not None
            else []
        )
        coverage_counts: dict[str, int] = {}
        for window in overlapping:
            for pair in window.get("label_scores", []):
                for label, score in pair.items():
                    if float(score) >= coverage_threshold:
                        coverage_counts[label] = coverage_counts.get(label, 0) + 1
        if not coverage_counts:
            attributes: dict[str, Any] = {
                "verb": "abstain",
                "best_coverage": 0.0,
                "n_windows": len(overlapping),
            }
        else:
            winner = max(
                coverage_counts,
                key=lambda label: (coverage_counts[label], _max_score(overlapping, label)),
            )
            verb = "confirm" if winner in confirmation_map.get(span_labels[span.id], set()) else "contest"
            attributes = {
                "verb": verb,
                "winner": winner,
                "coverage": coverage_counts[winner] / len(overlapping),
                "n_windows": len(overlapping),
                "mapped_to": span_labels[span.id],
            }
            if verb == "contest":
                contested_n += 1
                flags.append(f"yamnet contests {span_labels[span.id]} with {winner}")
        answer_id = store.entity(prov_type="assertion", extent=span.extent, attributes=attributes)
        store.was_generated_by(answer_id, confirm_activity)
        store.was_attributed_to(answer_id, yamnet_agent)
        store.was_derived_from(answer_id, label_id)
        store.was_derived_from(answer_id, span.id)
        answers.append(answer_id)

    # Step 3 — lexical contamination over the airway-labelled interval only.
    interval_id: str | None = None
    flag_id: str | None = None
    if span_labels:
        labelled_extents = [store.get_entity(span_id).extent or (0.0, 0.0) for span_id in span_labels]
        interval = (min(e[0] for e in labelled_extents), max(e[1] for e in labelled_extents))
        lexical = store.activity(node=NODE, step="lexical", parameters={"interval": list(interval)})
        store.was_associated_with(lexical, software)
        interval_id = store.entity(
            prov_type="interval", extent=interval, attributes={"name": "airway_labelled_interval"}
        )
        store.was_generated_by(interval_id, lexical)
        store.was_attributed_to(interval_id, software)
        contaminating: list[str] = []
        for word in store.entities("word"):
            if word.attributes.get("recognizer") != CRISPERWHISPER_ID:
                continue
            text = str(word.attributes.get("text") or "")
            if text.startswith("[") and text.endswith("]"):
                continue
            word_start, word_end = word.extent or (0.0, 0.0)
            if word_start < interval[1] and word_end > interval[0]:
                store.used(lexical, word.id)
                contaminating.append(word.id)
        if contaminating:
            flag_id = store.entity(
                prov_type="assertion",
                extent=interval,
                attributes={"verb": "flag", "reason": "lexical_contamination", "word_ids": contaminating},
            )
            store.was_generated_by(flag_id, lexical)
            store.was_attributed_to(flag_id, software)
            store.was_derived_from(flag_id, interval_id)
            flags.append("lexical_contamination")

    # Step 4 — the outcome. A hint conditions only what an absence means.
    if not span_labels:
        why = "no span carries a label of interest"
        if hint_declares:
            why += "; a hint declares airway content not found"
        flags.append(why)
        outcome = Outcome.FLAG
    elif flags:
        outcome, why = Outcome.FLAG, "; ".join(flags)
    else:
        outcome = Outcome.PASS
        why = "at least one span carries a label of interest and nothing contests it"

    verdict_id, verdict = write_verdict(
        store,
        classify,
        software,
        node=NODE,
        outcome=outcome,
        kind="airway",
        why=why,
        detail={"labelled_n": len(span_labels), "by_label": by_label, "contested_n": contested_n, "flags": flags},
    )

    figure_path: Path | None = None
    try:
        figure_path = _render_figure(store, plain, spans, span_labels, silence_windows, run_dir, config)
    except Exception:  # noqa: BLE001 — the figure is an artifact; failing to draw it changes no verdict
        figure_path = None

    view = (
        [span.id for span in spans]
        + list(label_ids.values())
        + answers
        + ([interval_id] if interval_id else [])
        + ([flag_id] if flag_id else [])
        + [verdict_id]
    )
    return AirwayResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, figure_path=figure_path)


def _render_figure(
    store: ProvStore,
    plain: Audio,
    spans: list[Entity],
    span_labels: dict[str, str],
    silence_windows: list[dict[str, Any]] | None,
    run_dir: Path,
    config: TriageConfig,
) -> Path:
    """One aligned figure: plain waveform, envelope with floor, spans, silence, spectrogram."""
    panels: list[dict[str, Any]] = [{"type": "waveform"}]
    envelope = find_measurement(store, "energy_envelope")
    if envelope is not None:
        sidecar = np.load(run_dir / envelope.attributes["path"])
        rate = int(envelope.attributes["sampling_rate"])
        stride = max(1, int(rate * float(config.require("gammatone.hop_s"))))
        times = (np.arange(len(sidecar["envelope_dbfs"])) / rate)[::stride]
        panels.append(
            {
                "type": "features",
                "data": [
                    (times.tolist(), sidecar["envelope_dbfs"][::stride].tolist(), "envelope dBFS (pre-emphasised)", "tab:blue"),
                    (times.tolist(), sidecar["floor_dbfs"][::stride].tolist(), "floor dBFS", "tab:gray"),
                ],
            }
        )
    segments = [
        {
            "label": span_labels.get(span.id, "unlabelled"),
            "start": (span.extent or (0.0, 0.0))[0],
            "end": (span.extent or (0.0, 0.0))[1],
        }
        for span in spans
    ]
    if segments:
        panels.append({"type": "segments", "segments": segments})
    if silence_windows:
        panels.append(
            {
                "type": "segments",
                "segments": [
                    {"label": "Silence" if w["is_silence"] else "sound", "start": w["start"], "end": w["end"]}
                    for w in silence_windows
                ],
            }
        )
    panels.append({"type": "spectrogram", "mel": False})
    figure = plot_aligned_panels(plain, panels, title="AIRWAY")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    path = run_dir / "figures" / "airway.png"
    figure.savefig(path)
    return path
```

- [ ] **Step 5 — run the tests; expect all PASS**, then the whole directory:
  `uv run pytest src/tests/audio/workflows/triage/nodes -x -q`

- [ ] **Step 6 — lint, type-check, config tests, commit**:
  `git add -A && git commit -m "feat(triage): AIRWAY labels, confirms and contests PREPROCESS's spans"`

**Interfaces:**

*Consumed:*
- `span_to_hear_buffer(audio, start_s, end_s, *, placement="centre") -> Audio` —
  `health_acoustics/hear.py:387`. Places the whole span in a silent buffer of exactly
  `HEAR_WINDOW_SECONDS` at the input's rate; raises `ValueError` on a span longer than the window
  (AIRWAY's routing signal to the sliding path) or an unknown placement. `hear.window_s` is pinned
  equal to `HEAR_WINDOW_SECONDS` by the existing config test.
- `detect_health_acoustic_events` — signature in the prerequisite section. Two call shapes:
  buffered spans with `hop_length=window_s` (the function's whole-window buffer makes
  `plan_scan_windows` return one window, so the whole span is scored once and no padding check
  fires), and over-length spans with the function's own default hop.
- `HEAR_MODEL_ID`, `HEAR_REVISION` — the agent's identity.
- `plot_aligned_panels(audio, panels, title="", figsize=None, spectrogram_params=None, context="auto") -> Figure` — panel dicts `{"type": "waveform"}`, `{"type": "features", "data": [(times, values, label, color), ...]}`, `{"type": "segments", "segments": [{"label", "start", "end"}]}`, `{"type": "spectrogram", "mel": bool}` (`tasks/plotting/plotting.py:488-508`). Requires mono input — the plain stream is mono by construction.
- Store reads: `span` entities filtered to `k_db == spans.k_db.airway`; `spans_no_contrast` (only at
  this `K`); `silence` windows; `yamnet_windows` (json sidecar); `word` entities
  (`recognizer == CRISPERWHISPER_ID`, bracket-filtered); the `plain` stream; `energy_envelope`
  sidecar (figure only).
- `AudioHints.may_contain` — the hint port; nothing else on the hint is read here.

*Produced (the sibling plan's read surface):* `label`/`confirm`/`contest`/`abstain`/`flag`
assertions and the `airway_labelled_interval` interval entity exactly as the schema table states.
SPEECH's withdrawal step reads the **labelled spans**: a span is airway-labelled iff a `label`
assertion is `wasDerivedFrom` it (`store.derived_from(assertion.id)` contains the span id); the
label itself, its confirmation state and `in_certified_silence` ride on those assertions. VOICE's
residual subtracts the same labelled spans. The AIRWAY `verdict` entity carries `labelled_n`,
`by_label`, `contested_n`, `flags`. `airway(store, source, config, hint=None, *, run_dir) ->
AirwayResult` with `.figure_path`.

---

## Corrections to capability-map.md, found while verifying this plan

The map remains the best index into senselab. Two of its stale-on-the-design-branch rows were
**resolved by the `triage` merge** and two corrections still stand:

1. **Resolved:** `health_acoustics` (with `span_to_hear_buffer` and `HEAR_WINDOW_SECONDS`,
   commit `4788ffeb`) and the ClearVoice separation backend both exist on the merged tree
   (commit `33bf65ad`); the map's §1.3/§1.4/§1.9 rows now describe the tree this plan executes on.
2. **Resolved:** `Audio.save_to_file` on the merged tree is the
   `subtype`/`out_of_range`/`AudioWriteReport` write layer §1.7 describes
   (`audio/data_structures/audio.py:371`): a plain `.wav` write resolves to the `FLOAT` subtype and
   round-trips float samples bit-exactly. The ASR-input test's quantisation-tolerant comparison is
   kept as slack, not as a requirement.
3. **The merged foundation diverges from the map's §2 task specs**, and the merged code governs:
   `envelope/` has no `preemphasise_audios` and no Nyquist/multichannel refusals (PREPROCESS inlines
   the one-line pre-emphasis; a multichannel input was already downmixed by then);
   `spans/api.py` returns `list[Span] | NoContrast` rather than the map's `SpanProposal`, and `Span`
   carries only `start`/`end`/`peak_over_floor_db` (no `peak_time`/parameter echo);
   `gammatone_filterbank` does not refuse `high_hz` above Nyquist — at the 16 kHz working rate with
   the configured 7800 Hz ceiling the case cannot arise, and if it ever does the absent-derivative
   path records it.
4. **§1.4's "the aligned figure — every panel the figure needs exists" is overstated.**
   `plot_aligned_panels` has no panel type for an image-valued bank: the gammatone view and the
   "HeAR channels in use" panel that `branch-airway.md`'s figure section names cannot be rendered
   with the existing vocabulary. See the self-review below.

## What this plan does not build

The SPEECH, VOICE, REDACT and VERDICT nodes (`plan-nodes-2.md`); any orchestrator (the DAG order —
ADMIT → PREPROCESS → TAXONOMY → AIRWAY → SPEECH → {REDACT ∥ VOICE} → VERDICT — lives in `nextflow/`);
the `audio/tasks/level/` lift capability-map §3.3 recommends; store persistence policy beyond "the
caller writes the JSONL outside any release directory".

## Self-review against the four node documents — gaps, stated rather than silent

1. **The AIRWAY figure is incomplete.** `branch-airway.md` wants waveform, envelope+floor+spans,
   YAMNet Silence, the wideband spectrogram, **the gammatone view and the HeAR channels in use** on
   one axis. The last two have no `plot_aligned_panels` panel type (an image-valued bank and
   per-label score tracks); Task 4 renders the first four and returns the figure. Closing the gap
   means a new panel type in `tasks/plotting/` — a task-layer change this plan deliberately does not
   smuggle in. The spectrogram panel is also recomputed by the plotting module at its own settings
   rather than read from the `spectrogram_wideband` sidecar — rendering density only, per
   capability-map §5.2 — and the envelope panel is decimated to the 5 ms analysis hop
   (`gammatone.hop_s`) for drawing, a density choice borrowing an existing config value rather than
   inventing a new number.
2. **AST sees at most its first 10.24 s frame of the recording** (`benchmarks/taxonomy.md`: the
   extractor pads or truncates to exactly 10.24 s). `taxonomy.md`'s window table says "file-level",
   which is true only for recordings up to that length. The member evidence records the score AST
   actually produced; a per-frame sweep for long recordings is unmeasured and not invented here.
3. **The AIRWAY confirmation winner's tie-break** (highest single-window score, N16) and the
   **containment reading of "lies inside certified silence"** (N17) are choices the design under-
   determines; both are in the decisions table so they are contestable, but neither is measured.
4. **`hint` is accepted and ignored by TAXONOMY** (N21) — the design's `taxonomy(store)` has no hint
   port; the shared node shape carries one. If the sibling plan's nodes give the hint a TAXONOMY
   meaning later, this is where it would leak.
5. **ADMIT writes a `recording` stream entity**, which `admit.md`'s port table does not name. The
   store model needs the recording to exist as an entity for `used`/`wasDerivedFrom` to point at;
   the design point "no second version of the audio" is preserved (the entity carries the supplied
   path, not a copy). Stated in Task 1; flagged here because it is an extension of the port list.
6. **Language is fixed to English for alignment** (N4) and no `language` is passed to either
   recognizer. A non-English corpus needs a config key and a derivation before these nodes are
   honest about it.
7. **Float equality joins `k_db` across nodes.** AIRWAY selects spans by
   `attributes["k_db"] == config value`. Exact-equality on a float survives the JSON round-trip for
   values like 18.0, but a future computed `K` would break the join silently; the store schema
   section states the convention so the sibling does not invent a second one.
8. **`test_zero_frames_fails` accepts either failure reason** (decoder raises vs. returns empty)
   because the installed decoder's behaviour on a zero-frame WAV is environment-dependent. The
   outcome — `fail` — is pinned; the reason string is not.
9. **PREPROCESS's verdict is unconditionally `pass`**, even when conditioning itself fails and every
   derivative is absent (`preprocess.md`: "No fail, no flag"). VERDICT will see a pass with an
   `absent` list covering everything; whether that deserves a file-level flag is the sibling's
   VERDICT decision, noted here so it is not lost.
10. **The unmeasured keys are exercised exactly as the constraints require**: no task supplies a
    value for `taxonomy.min_families.*` (the only `benchmarks/open.md` item these four nodes
    touch); tests reach the measured-rule path through explicit YAML overrides, and the packaged
    null path is itself under test (unanimity fold, `"unmeasured"` recorded).
