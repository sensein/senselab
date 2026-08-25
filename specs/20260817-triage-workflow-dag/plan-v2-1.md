# Triage v2 Implementation Plan — foundation: PREPROCESS, TAXONOMY, ROUTING

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `src/senselab/audio/workflows/triage/`'s first three nodes up to the **v2 specs**, which
were revised on 2026-08-17..24 to encode the owner's rulings. The v2 specs — not the code that exists
today, and not `plan-nodes-1.md`/`plan-nodes-2.md`, which implemented v1 — are the source of truth.
Where this plan and the shipped behaviour disagree, the shipped behaviour is **replaced outright**
(pre-alpha; no aliases, no shims, no parallel fields), and each superseded test is deleted with the
ruling that justifies it named beside the deletion.

**Scope split:** This file covers **T1 PREPROCESS v2**, **T2 TAXONOMY v2**, **T3 ROUTING** (a new node)
and the conditional execution in `run.py` that honours it. The sibling `plan-v2-2.md` covers the
branches (SPEECH, VOICE, AIRWAY), REDACT, VERDICT, REPORT and one bounded diagnostic. The two files
share the store schema stated in §"The v2 store contract" below; a change to it is a change to both.

**Architecture:** unchanged from v1 — one module per node under
`src/senselab/audio/workflows/triage/nodes/`, each taking `(store, source, config, hint=None, *, run_dir)`
and returning a `NodeResult`. What changes in v2 is **who computes what**: every whole-file model moves
into PREPROCESS, TAXONOMY becomes a pure fold over stored derivatives, and a new `routing` node stands
between TAXONOMY and the branches.

**Tech stack:** Python 3.12, pydantic v2, numpy, scipy, praat-parselmouth, pytest. uv for everything.

**Design source of truth:** `specs/20260817-triage-workflow-dag/` — `store.md` first, then
`preprocess.md`, `taxonomy.md`, `routing.md`. `branch-*.md`, `redact.md`, `verdict.md` and `report.md`
are read here only for the store schema they consume.

## Global Constraints

Binding on every task in **both** plan files.

- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Never run `pytest -n auto`.** Each xdist worker duplicates ~535 MB of frameworks plus its own model
  weights. Run the directory you changed, serially.
- `uv sync` is subtractive — always pass `--all-extras`.
- Tests live in `src/tests/` mirroring the package, named `*_test.py`. ruff applies to test code, so
  **every test class and function needs a docstring and every test function is annotated `-> None`**.
- Google-style docstrings; line length 120; type hints required (mypy with the pydantic plugin).
- **Rationale does not go in code.** Docstrings say what a thing is and how to call it. Measurements,
  rejected alternatives and the failures that drove a choice go in
  `specs/20260817-triage-workflow-dag/benchmarks/` — except a `derivation:` block in
  `data/config/default.yaml`, which is a config *value* and stays there.
- **No numeric or string tunable appears in production code.** Not as a signature default, not as a
  module-level constant. Every such value is a key in `data/config/default.yaml` with its derivation
  beside it, read through `config.require()` (mandatory) or `config.get()` (optional-with-a-recorded-absence).
  Definitional constants are the only exemption: `20·log10` (the definition of dB), full scale `1.0`
  (the definition of dBFS), `1e-12` floor clamps, `1000.0` ms-per-second, `1200.0` cents-per-octave, and
  a *vocabulary token* that is a controlled string the store must round-trip (`"not_evaluated"`,
  `"bounded"`, `"unavailable"`).
- **A value nobody has measured is `null` in the config, and reading it raises.** The v2 specs'
  "Open derivations (v2)" sections list **26 rows across eight files**, expanding to the **33 concrete
  YAML keys** enumerated in §"The 33 open keys" below. This plan creates every one of them, each `null`,
  each carrying the spec's own words for what it is owed. **Supplying a value for any of them is wrong.**
  Tests exercise them through explicit YAML overrides, which is the intended production mechanism too.
- **Append-only `ProvStore`.** Nothing is modified or deleted; a superseded claim is
  `wasInvalidatedBy`, a refined one `wasDerivedFrom`. Every read of the store goes through
  `nodes/common.py`'s helpers (`find_measurement`, `resolve_stream`, and the two added in T1), which
  apply the store's shared rule: **an invalidated entity is never read, and of the survivors asserting
  the same thing the latest write wins.** A node that hand-rolls `store.entities(...)` without that
  filter is a defect.
- **Nodes never import sibling nodes**, with exactly one sanctioned exception: the model-identity
  constants re-exported from `nodes/preprocess.py` (`CRISPERWHISPER_ID`, `QWEN_ID`, `AST_ID`,
  `YAMNET_MODEL_URI`). Those are names, not behaviour. Importing a *function* from a sibling node is a
  defect; read the store instead.
- A model load passes a **resolved 40-hex commit SHA, never a ref**.
  `src/tests/utils/revision_pinning_guard_test.py` sweeps the subprocess-worker files for this.
- **TDD, red first.** Every task's step 1 writes the tests and step 2 **runs them and records the
  failure text**. A task whose tests passed before the implementation was written did not test the
  change.
- **Behavioural tests.** A node test seeds a store with the entities its predecessors would have
  written, calls the node, and asserts on the verdict and the entities that came out. Models are mocked
  **at the node module's boundary** (`monkeypatch.setattr(node_module, "classify_audios", fake)`), never
  deeper; model *constructors* that resolve a commit over the network are reached only through
  module-level factory functions so those can be patched too. **No test loads YAMNet, AST, HeAR,
  CrisperWhisper, Qwen, pyannote, an aligner, SQUIM, an embedder or a separator.** Pure DSP (envelope,
  spans, gammatone, resample, `fuse_word_streams`, `fuse_consensus_words`, Praat, `plan_redactions`)
  runs real.
- Run `uv run ruff format` before every commit, then `uv run ruff check` and `uv run mypy` on the paths
  touched.
- **Pre-alpha: rename and replace outright.** The v1 TAXONOMY (detector committee, `min_families`,
  `voice_no_words` residual), the v1 VOICE (residual subtraction), the v1 REDACT (re-transcribing
  verification) and the v1 file-verdict vocabulary (`Outcome` on the file axis) are **replaced**. Delete
  the tests that pinned them, and name in the deletion commit the v2 spec sentence that superseded each.

## The 33 open keys — every one `null`, created by this plan

The table below is the whole set the v2 specs owe a measurement, with the file that owes it and the
spec's own words for what is owed. **T1 creates every one of these keys in
`data/config/default.yaml` in a single edit** — including the ones only the sibling plan's tasks read —
so that no task in either file has to invent a key shape, and so that `_merge`'s "overrides may not
introduce keys" rule accepts every test override on day one.

| key | spec | owed |
| --- | --- | --- |
| `windows.yamnet.default_threshold` | preprocess | the score at which a YAMNet label is confident enough to enter a window's set |
| `windows.yamnet.label_thresholds` | preprocess | the same, per label, overriding the default |
| `windows.ast.default_threshold` | preprocess | the same for AST |
| `windows.ast.label_thresholds` | preprocess | the same for AST, per label |
| `windows.ast.hop_s` | preprocess | the hop between AST's frames (see C1: the window itself is **not** open — it ships 0.96 s) |
| `windows.hear.default_threshold` | preprocess | the same for HeAR |
| `windows.hear.label_thresholds` | preprocess | the same for HeAR, per label |
| `windows.hear.hop_s` | preprocess | the hop, fitted on spans HeAR's 2 s input does not have to be padded to fill |
| `phonation_spans.f0_stability_cents` | preprocess | how little F0 may move across a hop for the interval to be a stable-F0 sustain |
| `phonation_spans.formant_stability_hz` | preprocess | the same for F1 and F2, for a stable-formant sustain |
| `phonation_spans.glide_min_excursion_cents` | preprocess | the monotone excursion over an interval that makes it a glide rather than drift |
| `phonation_spans.hangover_ms` | preprocess | how long the criterion must fail continuously before a span closes |
| `phonation_spans.voicing_strength_floor` | preprocess | the Praat pitch strength above which a frame counts as voiced |
| `phonation_spans.mixed_voiced_fraction` | preprocess | the voiced-frame fraction separating `voiced` from `mixed` from `unvoiced` |
| `words.onomatopoeic_tokens` | preprocess | the token set normalised into bracketed non-words; a vocabulary, owed the corpus it was drawn from |
| `taxonomy.presence_floor.speech.acoustic` | taxonomy | how many windows of speech-family evidence the acoustic line needs |
| `taxonomy.presence_floor.speech.lexical` | taxonomy | how many consensus words the lexical line needs |
| `taxonomy.presence_floor.airway.health_acoustic` | taxonomy | how many HeAR cough/breath windows the line needs |
| `taxonomy.presence_floor.airway.acoustic` | taxonomy | how many AudioSet airway windows the line needs |
| `taxonomy.voice_min_duration_s` | taxonomy | the duration at which a phonation or glide span makes the voice kind present, across voiced, unvoiced and mixed |
| `taxonomy.voice_uncertain_duration_s` | taxonomy | the shorter floor separating `uncertain` from `absent` |
| `taxonomy.speech_labels` | taxonomy | the AudioSet speech family, beyond the single `Speech` label the earlier list carried |
| `routing.hint_kind_map` | routing | which hint tags and `speech_type` values force which kind's branch |
| `airway.k_db` | branch-airway | the span gate for airway, with quiet breaths and coughs both represented in the fit |
| `airway.k_db_by_task` | branch-airway | the same, per declared task |
| `airway.k_margin_db` | branch-airway | how close to the gate a labelled span must sit to flag |
| `airway.contest_labels` | branch-airway | the declared YAMNet labels that may contest a HeAR label, disjoint from the airway evidence labels |
| `speech.enrollment_model` | branch-speech | which speaker-embedding model and revision enrollment is estimated with |
| `speech.separation_backend` | branch-speech | `unasdiff` or `MossFormer2_SS_16K`, once the two are ranked on this corpus |
| `speech.separation_sound_class` | branch-speech | **not a measurement**: `speech_sound` refuses to run without a conditioning class, so this is owed either an unconditioned sound slot upstream or a defensible FSD class — see V17 in the sibling plan |
| `speech.nontarget.level_db` | branch-speech | the proximity leg's level threshold |
| `speech.nontarget.tilt_db_per_octave` | branch-speech | the proximity leg's spectral-tilt threshold |
| `speech.nontarget.d_to_r_db` | branch-speech | the proximity leg's direct-to-reverberant threshold |

**The rest of the ledger, stated once so the two plan files cannot disagree about it (I8).**

*Already `null` on this branch, kept as-is, re-created by nobody* — `speech.word_gap_ms`,
`speech.target_match_cosine`, `speech.speech_test_stoi_floor`, `speech.speech_test_si_sdr_floor`,
`phonation.hnr_floor_db`, `phonation.rms_floor`, `redaction.padding_ms`.

*Created `null` by **T1**, beside the 33 above, because T1 owns the one config edit that renames the
two scalar F0 keys away* — **`voice.f0_range_hz`**. It is not in the 33 because it is a *rename*, not
a new open key: `phonation.f0_min_hz` and `phonation.f0_max_hz` were already null and it replaces
both. Sibling T5 **reads** it and does not create it.

*Created `null` by sibling tasks, against shapes fixed here* — `voice.f0_range_by_population`,
`voice.f0_range_ratio_max`, `voice.task_duration_ranges` (T5); `redaction.fill` (T7);
`report.format` — **which ships a declared default `"png"`, not null**, per the ruling in the sibling
plan's I4 row (T9).

*Created with a **measured or conventional value**, not null, and therefore in no open ledger* —
`windows.ast.win_length_s` = 0.96 and `windows.ast.top_k` = 527 (T1, this task);
`phonation_spans.{hop_s, formant_max_hz, max_formants, formant_window_s, formant_preemphasis_hz}`
(T1); `redaction.bleep_hz` = 1000.0 (T7).

*Deleted outright with the code that read them* — `hear.label_floor`, `hear.placement`,
`speech.agreement_flag_floor`, `taxonomy.min_families`, `taxonomy.ast_frame_s`,
`taxonomy.audioset_speech_labels`, `taxonomy.lexical_airway_tokens`,
`taxonomy.presence_floor.{yamnet,ast,hear}`, `phonation.f0_min_hz`, `phonation.f0_max_hz`.

**Consequence, stated loudly:** with the packaged config unmodified, `windows.*.default_threshold` is
null, so **no classifier's label sets can be folded**, every TAXONOMY line reads `unavailable`, every
kind is `uncertain`, ROUTING runs all three branches, and the file flags. That is the honest v2 state of
an unfitted graph and it is what the packaged config must report. Every test that needs a label set
supplies an override.

## The v2 store contract

Additions and replacements over the v1 contract in `plan-nodes-1.md` §"What the sibling plan's nodes
read". Anything not named here is unchanged.

**New `PROV_TYPE` members** (T1 adds `event`, T3 adds `branch_decision`, sibling T4 adds `enrollment`;
`src/senselab/utils/prov_store.py`):

```python
PROV_TYPE = Literal[
    "span", "word", "event", "speaker", "interval", "measurement", "kind", "stream",
    "pii", "verdict", "assertion", "target_match", "branch_decision", "enrollment",
]
```

**Window classifications — the T1→T2 contract.** Each of the three classifiers writes **two kinds of
record**, because a model's scores and the threshold fold over them fail independently:

1. **The scores.** One `measurement` per classifier, `name` ∈ `{"yamnet_scores", "ast_scores",
   "hear_scores"}`, attributes `{classifier, path, n_windows, win_length_s, hop_s}` where `path` is a
   run_dir-relative JSON sidecar holding the verbatim per-window dicts (`start`, `end`, `label_scores`,
   `win_length`, `hop_length`). Written whenever the model ran. **No threshold is read to write this.**
2. **The label sets.** One `measurement` per classifier, `name` ∈ `{"yamnet_windows", "ast_windows",
   "hear_windows"}`, plus one `measurement` **per window** of `name` `"<classifier>_window"`:

```
# the pooled record — presence
measurement(name="yamnet_windows"): {
  classifier:        "yamnet",
  labels:            [label, ...],                       # the set-union across windows, sorted
  windows_by_label:  { label: [window_entity_id, ...] }, # retained windows, per label
  n_windows:         int,                                # the denominator, including empty-set windows
  win_length_s:      float,
  hop_s:             float,
  default_threshold: float,
  label_thresholds:  { label: float },                   # only the entries that fired
}

# one per window — extent
measurement(name="yamnet_window", extent=(start, end)): {
  classifier: "yamnet",
  labels:     [label, ...],       # the confident set; MAY BE EMPTY
  scores:     { label: score },   # the score behind each member, members only
}
```

Three properties are binding. **A window's product is a set, never a winner** — no `argmax`, no
`top_label`, no ranking by how often a label won. **Pooling is set-union, and the windows are
retained** — `labels` is the union and `windows_by_label` is the index a consumer reads for extent. **A
window whose set is empty is still written**, because a window nobody's threshold cleared and a window
that was never classified are different facts and the store must be able to say which.

**`span` entities** (PREPROCESS, envelope family): `extent`, `{peak_over_floor_db, k_db, signal,
merged_proposals}`, never a `label` key. `merged_proposals` is new in v2 and is written by
`propose_spans` (T1 step 9b), **not** by the node: sibling T6 reports the merge rate and must read
what production writes, not what a fixture supplies.

**`phonation_span` — the T1→T5 contract.** Written as `span` entities carrying a `family` attribute, so
a reader can tell them from envelope spans without consulting the generating activity:

```
span(extent=(start, end)): {
  family:             "phonation",
  member:             "sustained" | "glide",
  duration_s:         float,               # the primary feature; TAXONOMY and routing read this
  production:         "voiced" | "unvoiced" | "mixed",
  voiced_fraction:    float,
  f0_median_hz:       float | None,        # None for a span with no voiced frame
  f0_start_hz:        float | None,
  f0_end_hz:          float | None,
  glide_direction:    "rising" | "falling" | None,
  glide_extent_cents: float | None,
  offset_criterion:   "f0_stability" | "formant_stability" | "monotonicity" | "stream_end",
  signal:             "preemphasised",
  hop_s:              float,
}
```

and one `measurement` per phonation span, `name="formant_tracks"`, `extent` the span's, attributes
`{f1_hz, f2_hz, f3_hz, f4_hz, f1_bw_hz, f2_bw_hz, f3_bw_hz, f4_bw_hz, times_s, hop_s, signal}` — each
of the nine a list on the analysis hop, sliced from tracks computed once over the whole stream.

**`consensus_transcript`, `word` and `event` — the T1→T4→T7 contract.** PREPROCESS is the **only**
author of `word` entities, and they are the **consensus** words:

```
measurement(name="consensus_transcript"): {
  words:      [ <verbatim fuse_consensus_words element>, ... ],
  provenance: { <verbatim fuse_consensus_words provenance dict> },
  systems:    [ CRISPERWHISPER_ID, QWEN_ID ],
  word_ids:   [ entity_id, ... ],     # positionally aligned with the kept subset of `words`
  event_ids:  [ entity_id, ... ],
  text:       str,                    # PII — the space-joined word texts; the only text downstream reads
}

word(extent=(start, end)): {
  text, confidence, existence_confidence, temporal_confidence, coverage,
  recognizers:    [ model_id, ... ],   # `sources` from the fusion
  timing_sources: int | None,
  index:          int,                 # position in `words`
}

event(extent=(start, end)): {
  bracketed:   str,        # e.g. "[COUGH]" — the normalised form
  raw:         str,        # the token the recognizer emitted, so the normalisation is legible
  origin:      "bracketed" | "onomatopoeic",
  recognizers: [ model_id, ... ],
}
```

The per-recognizer hypotheses stay as `measurement` entities (`asr_crisperwhisper`, `asr_qwen`) whose
`words` attribute holds that recognizer's own word list. **They are no longer `word` entities.** This
replaces v1, where PREPROCESS wrote one `word` per recognizer word and SPEECH wrote a second set for the
consensus — two populations of `word` that every consumer had to disambiguate by generating activity.

**`kind` entities** (TAXONOMY, exactly three): `{kind, state, lines}` where `kind ∈ {"airway", "speech",
"voice"}` — **`voice_no_words` is gone** — `state ∈ {"present", "absent", "uncertain"}` — **`undecided`
and `not_screened` are gone** — and `lines` maps line name → `{state, evidence, floor, element_ids}`
with `state ∈ {"present", "absent", "unavailable"}`.

**`branch_decision` entities** (ROUTING, exactly three) — the T3→T8 contract:

```
branch_decision: {
  branch:         "AIRWAY" | "SPEECH" | "VOICE",
  kind:           "airway" | "speech" | "voice",
  will_run:       bool,
  kind_state:     "present" | "uncertain" | "absent",
  forced_by_hint: bool,
  hint_tags:      [ str, ... ],       # the tags that forced it, if any
  unmapped_tags:  [ str, ... ],       # tags present on the hint that map to no kind
  why:            str,                # controlled vocabulary
}
```

`used(routing_activity, kind_entity)` records which classification each decision rests on, and
`wasDerivedFrom(branch_decision, kind_entity)` ties the two together.

## Under-specified points, resolved by this plan

Numbered `V*` so they do not collide with `plan-nodes-1.md`'s `N1..N27`. `V15`..`V22` live in the
sibling plan. An implementer must not silently re-decide one.

| # | point | decision |
| --- | --- | --- |
| V1 | `preprocess.md` says "each window is written as an element" but a 10-minute file yields ~1250 YAMNet windows | every window is written, including one whose label set is empty. The absence-vs-zero distinction this design keeps making is exactly what an unwritten empty window would destroy, and JSONL absorbs the volume |
| V2 | `preprocess.md`'s window table gives AST "10.24 s (its native frame)" | **that value is retracted in this repository and the plan does not implement it.** `audio_analysis/data/run_config/default.yaml` retracts the "native frame: 1024 mel frames at 10 ms" reasoning by name: 1024 frames is a fixed *input shape*, not an *analysis resolution* — `ASTFeatureExtractor` zero-pads a shorter window with rectangular padding and no taper, so AST slides at any hop. Measured there on a 21.48 s clip, 10.24 s / 10.24 s gave 3 windows scoring 0.473 / 0.449 / 0.195 on speech while 0.96 s / 0.48 s gave 45 windows scoring 0.75–0.92, and on a 4.9 s recording the coarse window exceeded the clip and returned one flat value for every bucket. AST's window is therefore the config key **`windows.ast.win_length_s`, defaulting to 0.96**, with the retraction quoted in its derivation; the owner's 10 s figure is reachable as an override and named as such. **There is no `AST_FRAME_S` literal.** YAMNet's hop remains absent from config for a different reason: `classify_audios(model="yamnet")` ignores `win_length`/`hop_length` entirely and returns its own grid, which is recorded on the pooled measurement as a fact |
| V2b | `classify_audios` does `top_k=top_k or 5` on the windowed path (`classification/api.py:135`), so `top_k=None` silently truncates every window to its top five labels | that is a **ranking over a vocabulary**, which `preprocess.md` forbids in the same paragraph that defines the set rule, and it would have made "the set of labels over threshold" mean "the set of the top five labels that are also over threshold". Every windowed classifier passes its vocabulary size explicitly: YAMNet `yamnet.top_k` = 521, AST `windows.ast.top_k` = 527. HeAR reaches `detect_health_acoustic_events` on a different path where `top_k=None` does keep all eight, and is left as `None` |
| V3 | a null threshold would lose the model output too | the classifier is **two steps**: `<name>_scores` runs the model and writes the raw windows (no threshold read), `<name>_windows` folds the thresholds and writes the label sets (raises while they are null). The expensive output survives a null; the fold is honestly absent |
| V4 | `preprocess.md` gives no F0 search range for the Praat pass | it is `voice.f0_range_hz`, the range the VOICE branch declares per population. `phonation.f0_min_hz` / `phonation.f0_max_hz` are **renamed away** into it (pre-alpha), so PREPROCESS and VOICE cannot hold two ranges that drift |
| V5 | the phonation continuity criterion is "a stable F0 / stable formant interval" with no operational form | a frame satisfies the criterion when `abs(dF0)` across one hop is under `phonation_spans.f0_stability_cents` **or** both `abs(dF1)` and `abs(dF2)` are under `phonation_spans.formant_stability_hz`. A maximal run of satisfying frames, closed by `phonation_spans.hangover_ms` of continuous failure, is a **sustained** span. A run that fails stability but whose F0 (or F1 where F0 is absent) is monotone with a total excursion over `phonation_spans.glide_min_excursion_cents` is a **glide** span. No periodicity floor opens or closes a span |
| V6 | "the detector may not require a periodicity floor" leaves production mode undefined for an unvoiced sustain | a frame is voiced when Praat's pitch `strength` clears `phonation_spans.voicing_strength_floor`. A span's `voiced_fraction` above `phonation_spans.mixed_voiced_fraction` is `voiced`, below `1 - mixed_voiced_fraction` is `unvoiced`, between is `mixed`. F0 statistics are reported over voiced frames only, and are `None` when there are none |
| V7 | `preprocess.md` names `fuse_consensus_words` but the routine takes `{model → resolved ASR result}` | PREPROCESS passes `{CRISPERWHISPER_ID: line, QWEN_ID: line}` — the two `ScriptLine`s it already holds — and stores the returned `(words, provenance)` verbatim. `policy=None`, so the task API's own defaults apply and are recorded in the provenance the routine returns |
| V8 | "an onomatopoeic cough- or breath-like token is normalised into a bracketed non-word" leaves the bracketed form unstated | the bracketed form is `"[" + token.upper() + "]"` where `token` is the matched vocabulary entry, not the raw text; the raw text travels as `raw`. Matching is on `casefold()` with edge punctuation stripped, against `words.onomatopoeic_tokens` |
| V9 | `disruptions_file` is listed as a PREPROCESS derivative but nothing computes it | PREPROCESS runs `detect_disruptions` over `[0, duration]` of the **`recording`** stream, at its original rate and level, and writes `measurement(name="disruptions_file")` |
| V10 | on a wordless file the store carries no disruption reading at all, which reads as "clean" | that is a real defect and V9 fixes its file-level half. SPEECH's per-span readings stay span-scoped and are legitimately absent when there are no spans, because a span nobody measured must not report zero — sibling T4 verifies the scoping and adds nothing |
| V11 | TAXONOMY's `presence_floor.<kind>.<line>` unit is unstated | a **count of windows** for an acoustic or health-acoustic line, and a **count of words** for the lexical line. Both are integers, both null, and each key's derivation says which |
| V12 | `taxonomy.audioset_speech_labels` (v1) versus `taxonomy.speech_labels` (v2) | renamed outright to `taxonomy.speech_labels` and set to `null`, because the v2 spec owes it "the AudioSet speech family, beyond the single `Speech` label the earlier list carried" and the v1 single-member list was never that family |
| V13 | routing's `hint_kind_map` must serve both `may_contain` tags and `metadata.speech_type` | one map, `{tag_or_speech_type_value: kind}`, matched `casefold()`ed against every `may_contain` tag and against `hint.metadata.get("speech_type")`. A tag matching no key is recorded in `unmapped_tags` and forces nothing |
| V14 | `routing.md` says the unit is encapsulated over one input stream but the current target runs it once | every element ROUTING and TAXONOMY write carries `stream: "plain"`. No second pass is built |

---

### Task 1: PREPROCESS v2 — every whole-file model, phonation spans, the consensus, bracket-aware words

**Scope:**

- `src/senselab/utils/prov_store.py` — add `"event"` to `PROV_TYPE`.
- `src/senselab/audio/workflows/triage/data/config/default.yaml` — create **all 33 open keys** from
  §"The 33 open keys" plus the non-open keys T1 needs; rename `phonation.f0_min_hz`/`f0_max_hz` into
  `voice.f0_range_hz`; rename `taxonomy.audioset_speech_labels` into `taxonomy.speech_labels`; delete
  `taxonomy.presence_floor.{yamnet,ast,hear}`, `taxonomy.min_families`, `taxonomy.ast_frame_s`,
  `taxonomy.lexical_airway_tokens`, `hear.label_floor`, `hear.placement` and
  `speech.agreement_flag_floor` (M3: deleted outright with the code that read them — `hear.placement`
  is read only by `airway.py`'s `span_to_hear_buffer` call, which sibling T6 deletes, and
  `agreement_flag_floor` only by `speech.py`'s aggregate-agreement row, which sibling T4 deletes).
  `yamnet.silence_threshold` and `hear.window_s` are **kept**.
- `src/senselab/audio/tasks/phonation/api.py` — add `formant_track` and `propose_phonation_spans`.
- `src/senselab/audio/tasks/spans/api.py` — add `merged_proposals` to `Span` (C4).
- `src/senselab/audio/workflows/triage/nodes/common.py` — add `find_measurements` and `live_entities`.
- `src/senselab/audio/workflows/triage/nodes/preprocess.py` — the node.
- Tests: `src/tests/audio/workflows/triage/nodes/preprocess_test.py` (rewritten),
  `src/tests/audio/workflows/triage/nodes/conftest.py` (fixtures added),
  `src/tests/audio/workflows/triage/config_test.py` (updated key assertions),
  `src/tests/audio/tasks/phonation/api_test.py` (extended).

**Design points this task must not get wrong (from `preprocess.md`):**

- **PREPROCESS runs every model that answers a whole-file question.** YAMNet, AST and HeAR all run
  here. No later node re-runs one. TAXONOMY and the branches read the stored windows.
- **A window's product is a SET of labels over per-label thresholds, not a winner.** Set-union pooling,
  windows retained. Nothing counts windows into a score or takes an argmax over a vocabulary.
- **YAMNet on its native 0.96 s / 0.48 s grid; AST on a configured window defaulting to 0.96 s; HeAR on
  its fixed 2 s window.** YAMNet's grid is model-imposed; AST's window **and** hop are config keys;
  HeAR's window is model-imposed and only its hop is config. **AST's window is not 10.24 s** — see V2.
- **Phonation spans admit voiced, unvoiced and mixed production.** A detector that required periodicity
  would measure exactly the voices least in need of measurement. `duration_s` is the primary feature.
- **Tracks are computed once on the stream and then sliced.** No criterion is ever renormalised to a
  fragment's own maximum.
- **The consensus transcript comes from `fuse_consensus_words` in
  `senselab.audio.workflows.audio_analysis.asr`** — called, not reimplemented.
- **A bracketed token is not a word**, and an onomatopoeic cough/breath token is normalised into one
  before any `word` entity is written.
- **`disruptions_file` is measured on the `recording` stream**, before any rate or level change.
- **A derivative that cannot be computed is absent from the store, not an error.** The existing
  block-per-derivative loop with its `absent` dict is the mechanism and stays.

**Steps:**

- [ ] **Step 1 — add `"event"` to `PROV_TYPE`.**

In `src/senselab/utils/prov_store.py`, replace the `PROV_TYPE` literal with:

```python
PROV_TYPE = Literal[
    "span",
    "word",
    "event",
    "speaker",
    "interval",
    "measurement",
    "kind",
    "stream",
    "pii",
    "verdict",
    "assertion",
    "target_match",
    "branch_decision",
    "enrollment",
]
```

(`branch_decision` and `enrollment` are added here too, in one edit, so T3 and sibling T4 need no
second change to this file.)

- [ ] **Step 2 — write the failing config test.**

Append to `src/tests/audio/workflows/triage/config_test.py`:

```python
class TestTheV2OpenKeys:
    """Every key the v2 specs owe a measurement exists and is null."""

    OPEN_KEYS = (
        "windows.yamnet.default_threshold",
        "windows.yamnet.label_thresholds",
        "windows.ast.default_threshold",
        "windows.ast.label_thresholds",
        "windows.ast.hop_s",
        "windows.hear.default_threshold",
        "windows.hear.label_thresholds",
        "windows.hear.hop_s",
        "phonation_spans.f0_stability_cents",
        "phonation_spans.formant_stability_hz",
        "phonation_spans.glide_min_excursion_cents",
        "phonation_spans.hangover_ms",
        "phonation_spans.voicing_strength_floor",
        "phonation_spans.mixed_voiced_fraction",
        "words.onomatopoeic_tokens",
        "taxonomy.presence_floor.speech.acoustic",
        "taxonomy.presence_floor.speech.lexical",
        "taxonomy.presence_floor.airway.health_acoustic",
        "taxonomy.presence_floor.airway.acoustic",
        "taxonomy.voice_min_duration_s",
        "taxonomy.voice_uncertain_duration_s",
        "taxonomy.speech_labels",
        "routing.hint_kind_map",
        "airway.k_db",
        "airway.k_db_by_task",
        "airway.k_margin_db",
        "airway.contest_labels",
        "speech.enrollment_model",
        "speech.separation_backend",
        "speech.separation_sound_class",
        "speech.nontarget.level_db",
        "speech.nontarget.tilt_db_per_octave",
        "speech.nontarget.d_to_r_db",
    )

    def test_every_open_key_exists_and_is_null(self) -> None:
        """A key that does not exist is a typo; a key with a value is an unmeasured decision shipped.

        Both halves are checked through the public API: ``require`` distinguishes the two failures by
        message — "unknown configuration key" for a typo, "has no value" for a null — so asserting on
        which message fires is what tells "the key is missing" from "the key is null".
        """
        config = load_triage_config()
        for path in self.OPEN_KEYS:
            with pytest.raises(ValueError, match="has no value") as raised:
                config.require(path)
            assert "unknown configuration key" not in str(raised.value), path
            assert config.get(path, "SENTINEL") == "SENTINEL", path

    def test_the_v1_keys_the_v2_specs_replaced_are_gone(self) -> None:
        """Pre-alpha: a replaced key is deleted, not left beside its replacement."""
        config = load_triage_config()
        for path in (
            "phonation.f0_min_hz",
            "phonation.f0_max_hz",
            "taxonomy.audioset_speech_labels",
            "taxonomy.min_families",
            "taxonomy.ast_frame_s",
            "taxonomy.lexical_airway_tokens",
            "taxonomy.presence_floor.yamnet",
            "hear.label_floor",
        ):
            with pytest.raises(ValueError, match="unknown configuration key"):
                config.require(path)

    def test_the_f0_range_replaces_the_two_scalar_keys(self) -> None:
        """One range, read by PREPROCESS and VOICE alike, so the two cannot drift."""
        config = load_triage_config()
        with pytest.raises(ValueError, match="has no value"):
            config.require("voice.f0_range_hz")
        assert config.get("voice.f0_range_hz", "SENTINEL") == "SENTINEL"
```

- [ ] **Step 3 — run it; expect FAIL** (`unknown configuration key 'windows.yamnet.default_threshold'`).
  `uv run pytest src/tests/audio/workflows/triage/config_test.py -x -q`

- [ ] **Step 4 — edit `data/config/default.yaml`.**

Delete the `phonation.f0_min_hz`, `phonation.f0_max_hz`, `hear.label_floor`,
`taxonomy.min_families`, `taxonomy.ast_frame_s`, `taxonomy.audioset_speech_labels`,
`taxonomy.lexical_airway_tokens` and `taxonomy.presence_floor.{yamnet,ast,hear}` keys, and add the
blocks below. Prose in `derivation:` is a config value, so add these paragraphs to it verbatim rather
than as `#` comments.

Add to `derivation:`:

```
  v2 window classifications -- preprocess.md's "sets, not accumulators" rule. A window's product is the
  set of labels each clearing ITS OWN threshold, and the file-level product is the set-union with the
  windows retained per label. The thresholds are windows.<classifier>.default_threshold with a
  per-label override map; all six are null because no ROC over this corpus exists. The v1
  taxonomy.presence_floor.{yamnet,ast,hear} 0.5 values are RETRACTED and deleted rather than carried
  over: they were read off bimodal gaps in one reference recording's whole-file scores, and a
  whole-file gap is not a per-window threshold. YAMNet's grid is not a key at all -- classify_audios
  ignores win_length/hop_length for YAMNet and returns its own 0.96 s / 0.48 s frames, so the grid is
  recorded as a fact on the pooled measurement.

  windows.ast.win_length_s 0.96 -- NOT 10.24. The "native frame: 1024 mel frames at 10 ms" reasoning
  is RETRACTED in this repository, and the retraction is quoted here because a plan that reintroduced
  the value would be reintroducing a defect this codebase already measured and removed. From
  audio_analysis/data/run_config/default.yaml: "That conflates the model's required input size with
  its temporal precision. 1024 frames is a fixed input shape, not an analysis resolution: a shorter
  window is zero-padded to 1024 frames by ASTFeatureExtractor -- rectangular padding, no taper -- so
  the content is unattenuated and AST can be slid at any hop." Measured there on a 21.48 s
  conversation clip: 10.24 s / 10.24 s gave 3 windows scoring 0.473, 0.449 and 0.195 on speech, while
  0.96 s / 0.48 s gave 45 windows scoring 0.75-0.92 -- confidence ROSE, because a 10.24 s window of a
  conversation spreads its softmax mass over speech plus silence plus everything else while a 0.96 s
  window of speech is unambiguous. And on a 4.9 s recording the coarse window exceeded the whole clip,
  so AST returned one flat value for every bucket while carrying the largest weight on its axis. The
  "kaldi-fbank refuses chunks below ~1 s" constraint is real and does not bind: 0.96 s is ~96 mel
  frames. 0.96 s also puts AST on YAMNet's grid, which is what lets the acoustic evidence line count
  windows from either classifier without reconciling two frame rates. A coarser window -- the owner's
  10 s figure among them -- remains reachable as an override, and a run that takes it declares it in
  config_hash. windows.ast.hop_s stays null because the v2 spec owes it a fit; 0.48 is the value the
  retraction above measured at, and is what an override should start from.

  windows.ast.top_k 527 -- the full AudioSet label space AST was fine-tuned on. A SIZE, not a
  threshold, and the same reason yamnet.top_k is 521: classify_audios does `top_k=top_k or 5` on its
  windowed path (classification/api.py:135), so passing None does not mean "keep everything", it means
  "keep five". Five of 527 is a RANKING over the vocabulary, which is the one operation
  preprocess.md's set rule forbids -- it would make "the set of labels over threshold" silently mean
  "the set of the top five labels that are also over threshold", and a label the model emitted at 0.9
  in a busy window would vanish. HeAR is unaffected: detect_health_acoustic_events takes a different
  path on which top_k=None does keep all eight labels.

  HeAR's 2 s window is model-imposed and already lives at hear.window_s; only its hop is a key, and it
  is owed a fit on spans HeAR's input does not have to be padded to fill. hear.placement and
  hear.label_floor are DELETED with the code that read them: placement was only ever an argument to
  span_to_hear_buffer, which branch-airway.md removes from the graph by confining HeAR to PREPROCESS,
  and label_floor is replaced by windows.hear.default_threshold. speech.agreement_flag_floor is
  deleted likewise: branch-speech.md replaces the aggregate-agreement flag with per-word recognizer
  membership, so there is no aggregate for a floor to gate.

  phonation_spans -- preprocess.md's sustained-phonation and glide detector. Every parameter is null.
  f0_stability_cents and formant_stability_hz are the two limbs of the continuity criterion (a frame
  continues a sustain when F0 moves less than the first across one hop, OR F1 and F2 both move less
  than the second); glide_min_excursion_cents is the monotone excursion separating a glide from drift;
  hangover_ms is how long the criterion must fail continuously before the span closes.
  voicing_strength_floor is the Praat pitch strength above which a frame counts as voiced, and
  mixed_voiced_fraction is the voiced-frame fraction separating voiced from mixed from unvoiced -- both
  are needed because a disordered voice sustains with little or no periodicity and a detector that
  required a periodicity floor would measure exactly the voices least in need of measurement.
  formant_max_hz 5000.0, max_formants 5, formant_window_s 0.025 and formant_preemphasis_hz 50.0 are
  praat_parselmouth.py:813's own to_formant_burg defaults -- conventional, not fitted here. hop_s 0.01
  is Praat's documented time_step default, the same value phonation.hop_s already carries.

  voice.f0_range_hz replaces phonation.f0_min_hz and phonation.f0_max_hz. One range, as [min, max],
  read by both PREPROCESS's phonation pass and the VOICE branch, so the two cannot hold ranges that
  drift. It is null for the reason the two scalars were: no single search range serves both a low adult
  male fundamental and an infant voice, so the caller must state which population it is measuring.

  words.onomatopoeic_tokens -- the vocabulary of cough- and breath-like renderings a recognizer emits
  as ordinary words ("khh", "ahem", "uh-huh-huh"). Null: it is owed the corpus it was drawn from, and
  seeding it from three remembered examples would be a vocabulary nobody fitted. While null, only
  already-bracketed tokens become events, and an onomatopoeic rendering is counted as a word -- which
  is the honest state, not a safe default.

  taxonomy v2 -- taxonomy.md. TAXONOMY runs no models and folds stored evidence only, so the v1
  min_families committee and its per-detector floors are deleted rather than re-derived. Each kind now
  has named evidence LINES with their own floors: speech has acoustic (window count) and lexical (word
  count); airway has health_acoustic (HeAR window count) and acoustic (AudioSet window count); voice has
  neither, being classified from phonation-span duration alone. presence_floor values are counts, not
  scores, and all four are null. voice_min_duration_s and voice_uncertain_duration_s are the two
  duration cutoffs, both null, both owed a fit across voiced, unvoiced and mixed production.
  taxonomy.speech_labels replaces audioset_speech_labels and is null because the v2 spec owes it the
  AudioSet speech FAMILY and the v1 list carried one member. lexical_airway_tokens is deleted: airway's
  lexical evidence line no longer exists, because a bracketed event is not a word and carries no
  lexical evidence at all.

  routing.hint_kind_map -- which may_contain tags and which metadata.speech_type values force which
  kind's branch. A vocabulary, null, owed the corpus it was drawn from. A tag matching no entry forces
  nothing and is recorded as unmapped; forcing only ever ADDS a branch.

  airway v2 -- branch-airway.md. airway.k_db and airway.k_db_by_task override spans.k_db for this
  branch and are null: an airway event is level-limited and one value fitted on coughs does not serve
  quiet breaths, so the 18.0 dB spans.k_db.airway value is retained as the shared default while the
  branch's own override waits for a fit that represents both. airway.k_margin_db is how close to the
  gate a labelled span must sit to flag, null. airway.contest_labels is the declared set of YAMNet
  labels that may contest a HeAR label; it is null and, when supplied, is refused at load if it
  intersects taxonomy.audioset_airway_labels -- a label cannot be both airway evidence and a contest of
  airway evidence.

  speech v2 -- branch-speech.md. speech.enrollment_model names the speaker-embedding model AND its
  revision that enrollment is estimated with; null, and while null an enrollment is refused rather than
  compared. speech.separation_backend chooses between unasdiff in speech_sound mode and
  MossFormer2_SS_16K; null until the two are ranked on this corpus, and while null separation does not
  run. speech.separation_sound_class is the FSD class name unasdiff's sound slot is conditioned on.
  It is null for a DIFFERENT reason from every other null here, and the distinction matters: nobody
  needs to measure anything for it. branch-speech.md says the slot stands for any background and
  should not be conditioned on a class, and separate_audios refuses speech_sound without one ("index 0
  is 'Hi-hat'"), so THE CAPABILITY IS ABSENT UPSTREAM. It is settled by adding an unconditioned sound
  slot to unasdiff, or by someone naming a defensible class and saying why -- not by a ROC, a corpus
  or a fit. Until then the unasdiff option cannot run. speech.nontarget.{level_db,tilt_db_per_octave,d_to_r_db} are the proximity
  leg's three thresholds, each null; until all three exist the legs are measured and reported per span
  and nontarget_speech_s is written as null rather than zero.
```

Add the key blocks:

```yaml
windows:
  yamnet:
    default_threshold: null
    label_thresholds: null
  ast:
    default_threshold: null
    label_thresholds: null
    win_length_s: 0.96      # NOT 10.24 -- the "native frame" reasoning is retracted; see the derivation
    hop_s: null             # owed a fit; the retraction measured at 0.48
    top_k: 527              # AudioSet's full label space; a size, not a threshold (classify_audios: top_k or 5)
  hear:
    default_threshold: null
    label_thresholds: null
    hop_s: null

phonation_spans:
  hop_s: 0.01
  f0_stability_cents: null
  formant_stability_hz: null
  glide_min_excursion_cents: null
  hangover_ms: null
  voicing_strength_floor: null
  mixed_voiced_fraction: null
  formant_max_hz: 5000.0
  max_formants: 5
  formant_window_s: 0.025
  formant_preemphasis_hz: 50.0

words:
  onomatopoeic_tokens: null
```

Replace the `taxonomy:` block with:

```yaml
taxonomy:
  presence_floor:
    speech:
      acoustic: null
      lexical: null
    airway:
      health_acoustic: null
      acoustic: null
  voice_min_duration_s: null
  voice_uncertain_duration_s: null
  speech_labels: null
  audioset_airway_labels: [Cough, Throat clearing, Sneeze, Sniff, Breathing, Wheeze, Snoring, Gasp, Sigh]
  hear_airway_labels: [Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear]
```

Add to the `airway:` block (keeping `labels_of_interest` and `confirmation_map`):

```yaml
  k_db: null
  k_db_by_task: null
  k_margin_db: null
  contest_labels: null
```

Add to the `speech:` block:

```yaml
  enrollment_model: null
  separation_backend: null
  separation_sound_class: null
  nontarget:
    level_db: null
    tilt_db_per_octave: null
    d_to_r_db: null
```

Add a `routing:` block and a `voice.f0_range_hz` key (the rest of the `voice:` block is sibling T5's):

```yaml
routing:
  hint_kind_map: null

voice:
  hint_tags: [phonation, humming, sustained-vowel, voice]   # N25 -- vocabulary, not fitted
  f0_range_hz: null
```

Delete `phonation.f0_min_hz` and `phonation.f0_max_hz` from the `phonation:` block; the rest of that
block (`hnr_floor_db`, `rms_floor`, `hop_s`, `silence_threshold`, `periods_per_window`,
`period_doubling_factor`, `hnr_floor_interval_db`, `rms_floor_interval`) stays for VOICE.

- [ ] **Step 5 — run the config test; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/config_test.py -x -q`

- [ ] **Step 6 — write the failing `formant_track` test.**

`src/tests/audio/tasks/phonation/api_test.py`, appended:

```python
class TestFormantTrack:
    """Formants over the whole stream, on the analysis hop, four per frame with bandwidths."""

    def test_a_synthetic_vowel_yields_four_formants_per_frame(self) -> None:
        """Every returned array is the same length and carries F1-F4 with their bandwidths."""
        sr = 16000
        t = np.arange(int(0.5 * sr)) / sr
        wave = sum(np.sin(2 * np.pi * f * t) for f in (120.0, 700.0, 1200.0, 2600.0, 3400.0))
        audio = Audio(waveform=torch.tensor(wave, dtype=torch.float32).unsqueeze(0), sampling_rate=sr)
        track = formant_track(
            audio,
            hop_s=0.01,
            max_formants=5,
            formant_max_hz=5000.0,
            window_s=0.025,
            preemphasis_hz=50.0,
        )
        lengths = {
            len(track.times_s),
            len(track.f_hz[0]),
            len(track.f_hz[3]),
            len(track.bandwidth_hz[0]),
            len(track.bandwidth_hz[3]),
        }
        assert len(lengths) == 1
        assert len(track.f_hz) == 4
        assert len(track.bandwidth_hz) == 4

    def test_tracking_a_slice_and_slicing_the_track_are_not_the_same_measurement(self) -> None:
        """The whole point of tracking once on the stream: a fragment renormalises to its own maximum.

        The fixture is a stream whose second half is 20 dB louder than its first, so a track computed
        on the quiet fragment alone sees a different dynamic range from the same interval sliced out
        of the stream's track. Both are compared here explicitly, which is what makes this test say
        something — a test that only checked the sliced track against itself would pass under either
        implementation.
        """
        sr = 16000
        t = np.arange(int(2.0 * sr)) / sr
        wave = sum(np.sin(2 * np.pi * f * t) for f in (120.0, 700.0, 1200.0))
        wave = np.where(t < 1.0, wave * 0.1, wave)
        audio = Audio(waveform=torch.tensor(wave, dtype=torch.float32).unsqueeze(0), sampling_rate=sr)
        params = {
            "hop_s": 0.01, "max_formants": 5, "formant_max_hz": 5000.0,
            "window_s": 0.025, "preemphasis_hz": 50.0,
        }
        whole = formant_track(audio, **params)
        quiet = Audio(waveform=audio.waveform[:, : int(1.0 * sr)], sampling_rate=sr)
        fragment = formant_track(quiet, **params)
        sliced = whole.f_hz[0][(whole.times_s >= 0.0) & (whole.times_s < 1.0)]
        n = min(len(sliced), len(fragment.f_hz[0]))
        assert n > 50
        assert np.nanmedian(sliced[:n]) == pytest.approx(np.nanmedian(fragment.f_hz[0][:n]), rel=0.15)
        assert len(whole.times_s) > len(fragment.times_s)
```

- [ ] **Step 7 — run it; expect FAIL** (`ImportError: cannot import name 'formant_track'`).
  `uv run pytest src/tests/audio/tasks/phonation/api_test.py -x -q`

- [ ] **Step 8 — implement `formant_track`.**

Append to `src/senselab/audio/tasks/phonation/api.py`:

```python
@dataclass(frozen=True)
class FormantTrack:
    """Formant frequencies and bandwidths over one stream, on a fixed hop.

    Attributes:
        times_s: Frame times, in seconds.
        f_hz: Four arrays, F1 to F4, each one value per frame. NaN where Praat placed none.
        bandwidth_hz: Four arrays, the corresponding 3 dB bandwidths. NaN where the formant is NaN.
    """

    times_s: np.ndarray
    f_hz: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    bandwidth_hz: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def formant_track(
    audio: Audio,
    *,
    hop_s: float,
    max_formants: int,
    formant_max_hz: float,
    window_s: float,
    preemphasis_hz: float,
) -> FormantTrack:
    """F1-F4 and their bandwidths over the whole stream, by Praat's Burg method.

    Computed once over the stream so a consumer slices the track rather than re-tracking a fragment.

    Args:
        audio: The recording. ``get_sound`` handles channel merging and resampling.
        hop_s: Praat's ``time_step``. Read it from ``phonation_spans.hop_s``.
        max_formants: Praat's ``max_number_of_formants``. Read it from ``phonation_spans.max_formants``.
        formant_max_hz: Praat's ``maximum_formant``. Read it from ``phonation_spans.formant_max_hz``.
        window_s: Praat's ``window_length``. Read it from ``phonation_spans.formant_window_s``.
        preemphasis_hz: Praat's ``pre_emphasis_from``. Read it from
            ``phonation_spans.formant_preemphasis_hz``.

    Returns:
        The track. A frame where Praat placed no formant carries NaN in both arrays, so a missing
        formant is absent rather than zero.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    formants = snd.to_formant_burg(
        time_step=hop_s,
        max_number_of_formants=max_formants,
        maximum_formant=formant_max_hz,
        window_length=window_s,
        pre_emphasis_from=preemphasis_hz,
    )
    times = np.asarray(formants.xs(), dtype=np.float64)
    values: list[np.ndarray] = []
    bandwidths: list[np.ndarray] = []
    for order in (1, 2, 3, 4):
        values.append(
            np.asarray(
                [formants.get_value_at_time(order, t, unit=parselmouth.FormantUnit.HERTZ) for t in times],
                dtype=np.float64,
            )
        )
        bandwidths.append(
            np.asarray(
                [formants.get_bandwidth_at_time(order, t, unit=parselmouth.FormantUnit.HERTZ) for t in times],
                dtype=np.float64,
            )
        )
    return FormantTrack(
        times_s=times,
        f_hz=(values[0], values[1], values[2], values[3]),
        bandwidth_hz=(bandwidths[0], bandwidths[1], bandwidths[2], bandwidths[3]),
    )
```

Export it: `src/senselab/audio/tasks/phonation/__init__.py` gains `FormantTrack` and `formant_track` in
both the import and `__all__`.

- [ ] **Step 9 — run it; expect PASS.**
  `uv run pytest src/tests/audio/tasks/phonation/api_test.py -x -q`

- [ ] **Step 9b — make `propose_spans` count what it merged (C4).**

`branch-airway.md` requires "the merge rate is reported… so a span covering several events is legible
as one", and nothing in the tree records it: `propose_spans` merges overlapping proposals
(`spans/api.py:97-108`) and discards how many it absorbed. A plan whose test seeded the count from a
fixture would pass while production never wrote it, so the count gets a real owner here.

Failing test first, appended to `src/tests/audio/tasks/spans/api_test.py`:

```python
class TestTheMergeRate:
    """A span records how many proposals it absorbed, so several events in one span are legible."""

    def test_an_unmerged_span_absorbed_one_proposal(self) -> None:
        """One proposal in, one span out: the count is 1, not 0 — a span is its own proposal."""
        envelope, floor, rate = _one_burst()
        [span] = propose_spans(envelope, floor, rate, **_SPAN_PARAMS)
        assert span.merged_proposals == 1

    def test_two_overlapping_proposals_report_two(self) -> None:
        """The offset rule merged them; the survivor says so."""
        envelope, floor, rate = _two_overlapping_bursts()
        spans = propose_spans(envelope, floor, rate, **_SPAN_PARAMS)
        assert len(spans) == 1
        assert spans[0].merged_proposals == 2

    def test_two_separated_proposals_each_report_one(self) -> None:
        """Nothing was absorbed, and neither span claims otherwise."""
        envelope, floor, rate = _two_separated_bursts()
        spans = propose_spans(envelope, floor, rate, **_SPAN_PARAMS)
        assert [span.merged_proposals for span in spans] == [1, 1]
```

Run it; expect `AttributeError: 'Span' object has no attribute 'merged_proposals'`.

Then `Span` gains the field and the merge loop accumulates it:

```python
@dataclass(frozen=True)
class Span:
    """One proposed span.

    Attributes:
        start: Onset in seconds.
        end: Offset in seconds.
        peak_over_floor_db: The span's peak, referenced to the local floor.
        merged_proposals: How many proposals this span absorbed. One for a span the merge rule left
            alone — a span is its own proposal — so zero is never a valid value, and a span covering
            several events is legible as one rather than indistinguishable from a single event.
    """

    start: float
    end: float
    peak_over_floor_db: float
    merged_proposals: int = 1
```

```python
    merged: list[Span] = []
    for span in found:
        if merged and span.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Span(
                start=last.start,
                end=max(last.end, span.end),
                peak_over_floor_db=max(last.peak_over_floor_db, span.peak_over_floor_db),
                merged_proposals=last.merged_proposals + span.merged_proposals,
            )
        else:
            merged.append(span)
    return merged
```

Run it; expect PASS. PREPROCESS's `_spans` block then writes `"merged_proposals": span.merged_proposals`
into each `span` entity's attributes, beside `peak_over_floor_db`, and sibling T6 reads **that**
rather than a fixture value.

- [ ] **Step 10 — add the two store-read helpers.**

Append to `src/senselab/audio/workflows/triage/nodes/common.py`:

```python
def find_measurements(store: ProvStore, name: str) -> list[Entity]:
    """Every live measurement entity carrying this name, in write order.

    The plural of :func:`find_measurement`, for a name one node writes many of — the per-window
    classifications, the per-span formant tracks. Reads by the store's shared rule: an invalidated
    entity is never returned.

    Args:
        store: The provenance store.
        name: The measurement's ``name`` attribute.

    Returns:
        The entities, oldest first. Empty when nothing live carries the name.
    """
    return [
        e for e in store.entities("measurement") if e.attributes.get("name") == name and not store.is_invalidated(e.id)
    ]


def live_entities(store: ProvStore, prov_type: PROV_TYPE) -> list[Entity]:
    """Every non-invalidated entity of one type, in write order.

    The store's shared read rule in its simplest form, so no node re-derives the filter and forgets
    the invalidation check.

    Args:
        store: The provenance store.
        prov_type: The entity type to read.

    Returns:
        The live entities, oldest first.
    """
    return [e for e in store.entities(prov_type) if not store.is_invalidated(e.id)]
```

`common.py` gains `from senselab.utils.prov_store import PROV_TYPE, Entity, ProvStore`.

- [ ] **Step 11 — write the failing PREPROCESS tests.**

Replace `src/tests/audio/workflows/triage/nodes/preprocess_test.py` wholesale. Keep the existing
fixtures in `conftest.py` and add these two:

```python
@pytest.fixture
def windows_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with every window threshold and hop supplied, so the folds can run."""
    override = tmp_path / "windows.yaml"
    override.write_text(
        "windows:\n"
        "  yamnet:\n"
        "    default_threshold: 0.5\n"
        "    label_thresholds: {Speech: 0.4}\n"
        "  ast:\n"
        "    default_threshold: 0.3\n"
        "    label_thresholds: {}\n"
        "    hop_s: 0.48\n"
        "  hear:\n"
        "    default_threshold: 0.5\n"
        "    label_thresholds: {}\n"
        "    hop_s: 1.0\n"
    )
    return load_triage_config(override)


def window(start: float, end: float, scores: dict[str, float]) -> dict[str, Any]:
    """One classifier window in the shape ``label_scores`` reads."""
    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    return {
        "start": start,
        "end": end,
        "label_scores": [{label: score} for label, score in ordered],
        "win_length": end - start,
        "hop_length": end - start,
    }
```

The test file:

```python
"""PREPROCESS v2: every whole-file model here, sets not winners, phonation spans, bracket-aware words."""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage import nodes
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import preprocess as preprocess_module
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import find_measurement, find_measurements, live_entities
from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID, QWEN_ID, preprocess
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore


class TestWindowClassificationsAreSets:
    """A window carries every label over its own threshold, and pooling is set-union."""

    def test_a_window_may_carry_several_labels(
        self, store: ProvStore, windows_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two labels clearing their thresholds in one window are both members; nothing wins."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(
            monkeypatch,
            yamnet=[window(0.0, 0.96, {"Speech": 0.9, "Cough": 0.7, "Music": 0.1})],
        )
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        pooled = find_measurement(store, "yamnet_windows")
        assert pooled is not None
        assert pooled.attributes["labels"] == ["Cough", "Speech"]
        per_window = find_measurements(store, "yamnet_window")
        assert len(per_window) == 1
        assert sorted(per_window[0].attributes["labels"]) == ["Cough", "Speech"]
        assert set(per_window[0].attributes["scores"]) == {"Cough", "Speech"}

    def test_a_per_label_threshold_overrides_the_default(
        self, store: ProvStore, windows_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Speech at 0.45 clears its own 0.4 while Cough at 0.45 misses the 0.5 default."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, yamnet=[window(0.0, 0.96, {"Speech": 0.45, "Cough": 0.45})])
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert find_measurements(store, "yamnet_window")[0].attributes["labels"] == ["Speech"]

    def test_an_empty_window_is_still_written(
        self, store: ProvStore, windows_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A window nobody's threshold cleared is not the same fact as a window never classified."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(
            monkeypatch,
            yamnet=[window(0.0, 0.96, {"Speech": 0.9}), window(0.48, 1.44, {"Speech": 0.01})],
        )
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        per_window = find_measurements(store, "yamnet_window")
        assert len(per_window) == 2
        assert per_window[1].attributes["labels"] == []
        assert find_measurement(store, "yamnet_windows").attributes["n_windows"] == 2

    def test_pooling_is_union_and_the_windows_are_retained(
        self, store: ProvStore, windows_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The union names the labels; windows_by_label names where each one was."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(
            monkeypatch,
            yamnet=[window(0.0, 0.96, {"Speech": 0.9}), window(0.48, 1.44, {"Cough": 0.9})],
        )
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        pooled = find_measurement(store, "yamnet_windows")
        per_window = find_measurements(store, "yamnet_window")
        assert pooled.attributes["labels"] == ["Cough", "Speech"]
        assert pooled.attributes["windows_by_label"]["Speech"] == [per_window[0].id]
        assert pooled.attributes["windows_by_label"]["Cough"] == [per_window[1].id]

    def test_the_scores_survive_a_null_threshold(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The packaged config folds nothing, but the model output is still in the store (V3)."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, yamnet=[window(0.0, 0.96, {"Speech": 0.9})])
        result = preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert find_measurement(store, "yamnet_scores") is not None
        assert find_measurement(store, "yamnet_windows") is None
        assert "yamnet_windows" in result.absent

    def test_ast_runs_at_the_configured_window_not_at_10_24_s(
        self, store: ProvStore, windows_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The 'native frame' reasoning is retracted in this repo; the window is a key defaulting to 0.96."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, ast=[window(0.0, 0.96, {"Speech": 0.9})], record=seen)
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert seen["ast"]["win_length"] == pytest.approx(0.96)
        assert seen["ast"]["hop_length"] == pytest.approx(0.48)

    def test_ast_is_asked_for_its_whole_vocabulary(
        self, store: ProvStore, windows_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """classify_audios does `top_k or 5`, so None would rank 527 labels down to five (C2)."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, ast=[window(0.0, 0.96, {"Speech": 0.9})], record=seen)
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert seen["ast"]["top_k"] == 527

    def test_a_truncating_top_k_would_lose_a_confident_label(
        self, store: ProvStore, windows_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A window carrying six labels over threshold keeps all six; a top-5 rank would drop one."""
        _seed_admit(store, tmp_path, wav_writer)
        scores = {f"L{i}": 0.9 - i * 0.01 for i in range(6)}
        _stub_models(monkeypatch, ast=[window(0.0, 0.96, scores)])
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        ast_window = find_measurements(store, "ast_window")[0]
        assert len(ast_window.attributes["labels"]) == 6

    def test_hear_runs_on_its_fixed_window_at_the_configured_hop(
        self, store: ProvStore, windows_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """HeAR's 2 s window is model-imposed; hop_s is the only key."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, hear=[window(0.0, 2.0, {"Cough": 0.9})], record=seen)
        preprocess(store, _audio(tmp_path), windows_config, run_dir=tmp_path)
        assert seen["hear"]["hop_length"] == pytest.approx(1.0)
        assert find_measurement(store, "hear_windows").attributes["labels"] == ["Cough"]


class TestPhonationSpans:
    """Sustains and glides, voiced, unvoiced or mixed, with duration_s as the primary feature."""

    def test_a_sustained_vowel_yields_a_span_carrying_its_duration(
        self, store: ProvStore, phonation_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 1.5 s steady tone is one sustained span whose duration_s is its extent."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert spans
        best = max(spans, key=lambda e: e.attributes["duration_s"])
        assert best.attributes["member"] == "sustained"
        assert best.attributes["duration_s"] == pytest.approx(best.extent[1] - best.extent[0])
        assert best.attributes["duration_s"] > 1.0

    def test_an_unvoiced_sustain_is_a_span_like_any_other(
        self, store: ProvStore, phonation_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Steady band-limited noise sustains with no periodicity and is not refused."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_noise())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert spans
        assert any(e.attributes["production"] in ("unvoiced", "mixed") for e in spans)

    def test_a_glide_is_a_span_with_a_direction_and_an_excursion(
        self, store: ProvStore, phonation_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rising sweep is a glide, not a sustain, and carries where it went."""
        _seed_admit(store, tmp_path, wav_writer, samples=_rising_glide())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        glides = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") == "phonation" and e.attributes["member"] == "glide"
        ]
        assert glides
        assert glides[0].attributes["glide_direction"] == "rising"
        assert glides[0].attributes["glide_extent_cents"] > 0.0

    def test_formant_tracks_are_written_per_span_and_sliced_from_the_stream(
        self, store: ProvStore, phonation_config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One formant_tracks measurement per span, each derived from the span it covers."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        tracks = find_measurements(store, "formant_tracks")
        assert len(tracks) == len(spans)
        assert set(store.derived_from(tracks[0].id)) & {e.id for e in spans}
        assert len(tracks[0].attributes["f1_hz"]) == len(tracks[0].attributes["times_s"])

    def test_a_null_criterion_leaves_the_spans_absent(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The packaged config fits nothing, so the pass is absent rather than run on invented floors."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        result = preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert "phonation_spans" in result.absent
        assert not [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]


class TestTheConsensusTranscript:
    """fuse_consensus_words is called, and its output is what every text consumer reads."""

    def test_the_consensus_comes_from_fuse_consensus_words(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The routine is called with both recognizers' resolved results, and its provenance stored."""
        seen: dict[str, Any] = {}
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"), record=seen)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        assert sorted(consensus.attributes["systems"]) == sorted([CRISPERWHISPER_ID, QWEN_ID])
        assert consensus.attributes["provenance"]["operator"] == "consensus_words/resample"
        assert consensus.attributes["text"] == "hello world"

    def test_word_entities_are_the_consensus_words_only(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two recognizers agreeing on two words yield two word entities, not four."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert len(live_entities(store, "word")) == 2

    def test_the_per_recognizer_hypotheses_stay_as_measurements(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The evidence the consensus was fused from is retained, but not as word entities."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello world"), qwen=_line("hello there"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        for name in ("asr_crisperwhisper", "asr_qwen"):
            measurement = find_measurement(store, name)
            assert measurement is not None
            assert len(measurement.attributes["words"]) == 2


class TestWordsAreBracketAware:
    """A bracketed or onomatopoeic token is an event, and carries no lexical evidence."""

    def test_a_bracketed_token_is_an_event_not_a_word(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """[COUGH] between two words leaves two words and one event."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(
            monkeypatch,
            crisper=_line("hello [COUGH] world"),
            qwen=_line("hello [COUGH] world"),
        )
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert [e.attributes["text"] for e in live_entities(store, "word")] == ["hello", "world"]
        events = live_entities(store, "event")
        assert len(events) == 1
        assert events[0].attributes["bracketed"] == "[COUGH]"
        assert events[0].attributes["origin"] == "bracketed"

    def test_an_onomatopoeic_token_is_normalised_into_an_event(
        self, store: ProvStore, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the vocabulary supplied, 'khh' becomes [KHH] and the raw token travels with it."""
        override = tmp_path / "tokens.yaml"
        override.write_text("words:\n  onomatopoeic_tokens: [khh, ahem]\n")
        config = load_triage_config(override)
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello khh world"), qwen=_line("hello khh world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert [e.attributes["text"] for e in live_entities(store, "word")] == ["hello", "world"]
        events = live_entities(store, "event")
        assert events[0].attributes["bracketed"] == "[KHH]"
        assert events[0].attributes["raw"] == "khh"
        assert events[0].attributes["origin"] == "onomatopoeic"

    def test_a_null_vocabulary_leaves_an_onomatopoeic_token_a_word(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The honest unfitted state: nobody drew the vocabulary, so nothing is normalised."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line("hello khh world"), qwen=_line("hello khh world"))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert [e.attributes["text"] for e in live_entities(store, "word")] == ["hello", "khh", "world"]


class TestDisruptionsAreMeasuredOnTheOriginal:
    """The file-level reading exists whatever the transcript says (V9, V10)."""

    def test_a_wordless_file_still_carries_a_file_level_disruption_reading(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No words is not no measurement; that confusion is what this row exists to remove."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch, crisper=_line(""), qwen=_line(""))
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        measurement = find_measurement(store, "disruptions_file")
        assert measurement is not None
        assert measurement.attributes["clipped_runs"] == 0
        assert "zero_crossing_rate" in measurement.attributes

    def test_the_reading_names_the_original_recording_stream(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path,
        wav_writer: Callable[..., Path], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Peak normalisation and resampling destroy the defects, so the stream must be the original."""
        _seed_admit(store, tmp_path, wav_writer)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        assert find_measurement(store, "disruptions_file").attributes["signal"] == "recording"
```

`_seed_admit`, `_audio`, `_line`, `_stub_models`, `_steady_vowel`, `_steady_noise` and `_rising_glide`
are module-private helpers in **this test file**; `_stub_models` monkeypatches, **on
`preprocess_module`**, `classify_audios`, `detect_health_acoustic_events`, `transcribe_audios`,
`align_transcriptions`, `extract_objective_quality_features_from_audios`, `_crisperwhisper_model`,
`_qwen_model` and `_ast_model`, recording each call's kwargs into `record` when one is given.
`windows_config` and `phonation_config` are `conftest.py` fixtures layering overrides — the first
supplying every window threshold and hop, the second `voice.f0_range_hz: [75, 500]` and every
`phonation_spans.*` null.

**The fixture ownership rule (I1), binding on both plan files.** Six tasks need a seeded store, and a
single `seeded_store` fixture taking a different `words=` type per task would be three incompatible
contracts wearing one name. Following the tree's existing pattern, **each task owns its own seeder,
and `conftest.py` carries exactly one shared one**, defined here:

```python
@pytest.fixture
def seed_preprocess_store(tmp_path: Path) -> Callable[..., None]:
    """Write the entities PREPROCESS would have left behind, for a node test downstream of it.

    Every argument defaults to ``None``, which writes **nothing** for that derivative — that is how a
    test sets up an ``unavailable`` line, and it is a different state from passing an empty list,
    which writes the derivative and records that it found nothing.

    Args:
        store: The store to seed.
        stream_hz: The ``plain`` stream's rate. A silent mono WAV of ``duration_s`` is written under
            ``tmp_path`` and both the ``recording`` and ``plain`` stream entities point at it.
        duration_s: The streams' duration.
        yamnet_labels: One label list per YAMNet window, on a 0.96 s / 0.48 s grid. ``None`` writes no
            YAMNet measurement at all.
        ast_labels: The same on a 0.96 s / 0.48 s grid, for AST.
        hear_labels: The same on a 2 s / 1 s grid, for HeAR.
        words: The consensus words, as ``[text, ...]`` or ``[(text, (start, end)), ...]``. **An empty
            list still writes a ``consensus_transcript`` measurement carrying no words** — PREPROCESS
            fusing to nothing is not PREPROCESS never having run, and TAXONOMY's lexical line reads
            ``absent`` in the first case and ``unavailable`` in the second. ``None`` writes neither.
        events: Bracketed or onomatopoeic non-words, same shapes as ``words``.
        phonation: ``[(start, end, production), ...]`` phonation spans, plus the
            ``PREPROCESS``/``phonation_spans`` activity that says the pass ran. ``[]`` writes the
            activity and no spans; ``None`` writes neither, which is the ``unavailable`` case.
        spans: ``[(start, end, peak_over_floor_db), ...]`` envelope spans at ``span_k_db``.
        span_k_db: The ``k_db`` those spans were proposed at.
        disruptions_file: Whether to write the file-level disruption measurement.

    Returns:
        A callable taking ``(store, **the above)`` and writing them. It returns None; a test reads
        what it needs back out of the store, which is what makes these tests behavioural.
    """
```

Every other task's seeder is **module-private in that task's own test file**, built by calling
`seed_preprocess_store` first and then writing that branch's own predecessors:
`_seed_speech_store` (T4: adds diarizer segments, `pii` findings, a target speaker),
`_seed_voice_store` (T5: adds speech spans and airway labels, to prove neither is subtracted),
`_seed_airway_store` (T6: adds per-window `hear_window`/`yamnet_window` entities and merge counts),
`_seed_redact_store` (T7: adds `pii` entities and the `pii_scan` measurement),
`_seed_report_store` (T9: adds every branch's verdict). **No task after T1 edits `conftest.py`**, so
T4–T7 can be dispatched in parallel without colliding on one file.

- [ ] **Step 12 — run them; expect FAIL** (`KeyError: 'yamnet_window'` / `AssertionError`).
  `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -x -q`

- [ ] **Step 13 — rewrite `preprocess.py`.**

Keep the module's existing skeleton (the `_step` sub-activity helper, the `_measurement` writer, the
`blocks` list with its per-derivative `try` that records an uncomputable derivative in `absent`). The
changes are:

**Module constants** gain the two model identities that move here from TAXONOMY:

```python
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
YAMNET_MODEL_URI = "https://tfhub.dev/google/yamnet/1"


def _ast_model() -> HFModel:
    """The AST model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=AST_ID, revision="main")
```

**The shared threshold fold**, one function serving all three classifiers:

```python
def _confident_labels(
    window: dict[str, Any], default_threshold: float, label_thresholds: dict[str, float]
) -> dict[str, float]:
    """The labels this window is confident of, each with the score behind it.

    A label is a member iff its score clears its own threshold — ``label_thresholds[label]`` where
    one exists, ``default_threshold`` otherwise. The result may be empty, which is a window nobody's
    threshold cleared and is a different fact from a window that was never classified.

    Args:
        window: A classifier window, in the shape ``label_scores`` reads.
        default_threshold: The threshold for a label with no entry of its own.
        label_thresholds: Per-label thresholds.

    Returns:
        ``{label: score}`` over the members, in descending score order.
    """
    members: dict[str, float] = {}
    for pair in label_scores(window):
        for label, score in pair.items():
            if float(score) >= float(label_thresholds.get(label, default_threshold)):
                members[label] = float(score)
    return dict(sorted(members.items(), key=lambda item: -item[1]))
```

**The two-step classifier block**, instantiated three times:

```python
    def _scores(name: str, agent_id: str, activity_step: str, run: Callable[[], list[dict[str, Any]]]) -> None:
        """Run one classifier and store its verbatim windows; no threshold is read here (V3)."""
        activity = _step(activity_step, {}, (plain_id,), agent_id)
        windows = run()
        path = f"derivatives/{name}.json"
        (run_dir / path).write_text(json.dumps(windows))
        entity_id = _measurement(
            store,
            activity,
            agent_id,
            name=name,
            signal="plain",
            attributes={
                "classifier": name.removesuffix("_scores"),
                "path": path,
                "n_windows": len(windows),
                "win_length_s": float(windows[0]["win_length"]) if windows else None,
                "hop_s": float(windows[0]["hop_length"]) if windows else None,
            },
            derived_from=(plain_id,),
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        state[name] = windows
        state[name + "_id"] = entity_id

    def _windows(classifier: str) -> None:
        """Fold the thresholds over one classifier's stored scores into per-window label sets."""
        scores_name = f"{classifier}_scores"
        if scores_name not in state:
            raise LookupError(f"{scores_name} is absent")
        default_threshold = float(config.require(f"windows.{classifier}.default_threshold"))
        label_thresholds = {
            str(label): float(value)
            for label, value in (config.require(f"windows.{classifier}.label_thresholds") or {}).items()
        }
        activity = _step(
            f"{classifier}_windows",
            {"default_threshold": default_threshold, "label_thresholds": label_thresholds},
            (state[scores_name + "_id"],),
            software,
        )
        raw = state[scores_name]
        window_ids: list[str] = []
        windows_by_label: dict[str, list[str]] = {}
        fired: dict[str, float] = {}
        for raw_window in raw:
            members = _confident_labels(raw_window, default_threshold, label_thresholds)
            window_id = store.entity(
                prov_type="measurement",
                extent=(float(raw_window["start"]), float(raw_window["end"])),
                attributes={
                    "name": f"{classifier}_window",
                    "classifier": classifier,
                    "signal": "plain",
                    "labels": list(members),
                    "scores": members,
                },
            )
            store.was_generated_by(window_id, activity)
            store.was_attributed_to(window_id, software)
            store.was_derived_from(window_id, state[scores_name + "_id"])
            window_ids.append(window_id)
            for label in members:
                windows_by_label.setdefault(label, []).append(window_id)
                if label in label_thresholds:
                    fired[label] = label_thresholds[label]
        entity_id = _measurement(
            store,
            activity,
            software,
            name=f"{classifier}_windows",
            signal="plain",
            attributes={
                "classifier": classifier,
                "labels": sorted(windows_by_label),
                "windows_by_label": windows_by_label,
                "n_windows": len(raw),
                "win_length_s": float(raw[0]["win_length"]) if raw else None,
                "hop_s": float(raw[0]["hop_length"]) if raw else None,
                "default_threshold": default_threshold,
                "label_thresholds": fired,
            },
            derived_from=(state[scores_name + "_id"],),
        )
        derivatives[f"{classifier}_windows"] = entity_id
        view.append(entity_id)
        view.extend(window_ids)
```

The three classifier blocks, each a **named closure** so the `blocks` list at the end of this step
names it rather than repeating it — the two listings are one listing, and an implementer who finds
them disagreeing has found a plan defect, not a choice:

```python
    def _yamnet_scores() -> None:
        """YAMNet on its own native grid; `win_length`/`hop_length` are ignored by this backend."""
        _scores(
            "yamnet_scores",
            store.agent(
                agent_type="model",
                model_id=YAMNET_MODEL_URI,
                unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
            ),
            "yamnet",
            lambda: classify_audios([plain], model="yamnet", top_k=int(config.require("yamnet.top_k")))[0],
        )

    def _ast_scores() -> None:
        """AST at the configured window and hop, over its whole label space (C1, C2)."""
        model = _ast_model()
        _scores(
            "ast_scores",
            store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha),
            "ast",
            lambda: classify_audios(
                [plain],
                model=model,
                win_length=float(config.require("windows.ast.win_length_s")),
                hop_length=float(config.require("windows.ast.hop_s")),
                top_k=int(config.require("windows.ast.top_k")),
                function_to_apply="sigmoid",
            )[0],
        )

    def _hear_scores() -> None:
        """HeAR at its model-imposed 2 s window and the configured hop; `top_k=None` keeps all eight."""
        _scores(
            "hear_scores",
            store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION),
            "hear",
            lambda: detect_health_acoustic_events(
                [plain], hop_length=float(config.require("windows.hear.hop_s")), top_k=None
            )[0],
        )
```

**`silence` reads the stored YAMNet scores**, unchanged in behaviour, but from `state["yamnet_scores"]`
and `derived_from=(state["yamnet_scores_id"],)`.

**`_phonation_spans`**, the new block, implementing V5 and V6:

```python
    def _phonation_spans() -> None:
        """Sustained-phonation and glide spans, from tracks computed once over the whole stream."""
        f0_range = config.require("voice.f0_range_hz")
        f0_min_hz, f0_max_hz = float(f0_range[0]), float(f0_range[1])
        parameters: dict[str, Any] = {
            "hop_s": float(config.require("phonation_spans.hop_s")),
            "f0_stability_cents": float(config.require("phonation_spans.f0_stability_cents")),
            "formant_stability_hz": float(config.require("phonation_spans.formant_stability_hz")),
            "glide_min_excursion_cents": float(config.require("phonation_spans.glide_min_excursion_cents")),
            "hangover_ms": float(config.require("phonation_spans.hangover_ms")),
            "voicing_strength_floor": float(config.require("phonation_spans.voicing_strength_floor")),
            "mixed_voiced_fraction": float(config.require("phonation_spans.mixed_voiced_fraction")),
            "max_formants": int(config.require("phonation_spans.max_formants")),
            "formant_max_hz": float(config.require("phonation_spans.formant_max_hz")),
            "formant_window_s": float(config.require("phonation_spans.formant_window_s")),
            "formant_preemphasis_hz": float(config.require("phonation_spans.formant_preemphasis_hz")),
            "f0_min_hz": f0_min_hz,
            "f0_max_hz": f0_max_hz,
        }
        activity = _step("phonation_spans", parameters, (sharp_id,), software)
        times, f0_hz, strength = f0_track(
            sharp, f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz, hop_s=parameters["hop_s"]
        )
        formants = formant_track(
            sharp,
            hop_s=parameters["hop_s"],
            max_formants=parameters["max_formants"],
            formant_max_hz=parameters["formant_max_hz"],
            window_s=parameters["formant_window_s"],
            preemphasis_hz=parameters["formant_preemphasis_hz"],
        )
        proposals = propose_phonation_spans(
            times=times,
            f0_hz=f0_hz,
            strength=strength,
            formants=formants,
            hop_s=parameters["hop_s"],
            f0_stability_cents=parameters["f0_stability_cents"],
            formant_stability_hz=parameters["formant_stability_hz"],
            glide_min_excursion_cents=parameters["glide_min_excursion_cents"],
            hangover_ms=parameters["hangover_ms"],
            voicing_strength_floor=parameters["voicing_strength_floor"],
            mixed_voiced_fraction=parameters["mixed_voiced_fraction"],
        )
        span_ids: list[str] = []
        for proposal in proposals:
            span_id = store.entity(
                prov_type="span",
                extent=(proposal.start, proposal.end),
                attributes={
                    "family": "phonation",
                    "member": proposal.member,
                    "duration_s": proposal.end - proposal.start,
                    "production": proposal.production,
                    "voiced_fraction": proposal.voiced_fraction,
                    "f0_median_hz": proposal.f0_median_hz,
                    "f0_start_hz": proposal.f0_start_hz,
                    "f0_end_hz": proposal.f0_end_hz,
                    "glide_direction": proposal.glide_direction,
                    "glide_extent_cents": proposal.glide_extent_cents,
                    "offset_criterion": proposal.offset_criterion,
                    "signal": sharp_signal,
                    "hop_s": parameters["hop_s"],
                },
            )
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, software)
            store.was_derived_from(span_id, sharp_id)
            span_ids.append(span_id)
            inside = (formants.times_s >= proposal.start) & (formants.times_s < proposal.end)
            track_id = _measurement(
                store,
                activity,
                software,
                name="formant_tracks",
                signal=sharp_signal,
                attributes={
                    "times_s": formants.times_s[inside].tolist(),
                    "hop_s": parameters["hop_s"],
                    **{
                        f"f{order + 1}_hz": formants.f_hz[order][inside].tolist() for order in range(4)
                    },
                    **{
                        f"f{order + 1}_bw_hz": formants.bandwidth_hz[order][inside].tolist()
                        for order in range(4)
                    },
                },
                derived_from=(span_id,),
            )
            view.append(track_id)
        derivatives["phonation_spans"] = span_ids
        view.extend(span_ids)
```

`propose_phonation_spans` is a **new pure function** in
`src/senselab/audio/tasks/phonation/api.py`, taking the tracks and every parameter as arguments and
returning a `list[PhonationSpan]` (a frozen dataclass with exactly the fields the entity above reads).
Its rule is V5 verbatim: a frame *continues* when `abs(1200*log2(f0[i]/f0[i-1]))` is under
`f0_stability_cents` (NaN F0 never satisfies this limb) **or** both
`abs(f1[i]-f1[i-1])` and `abs(f2[i]-f2[i-1])` are under `formant_stability_hz`; a maximal run of
continuing frames closed by `hangover_ms / 1000.0` seconds of continuous non-continuation is a
`"sustained"` span; a maximal run that does *not* continue but over which the defined F0 values (or
F1 where none is defined) are monotone with `abs(1200*log2(last/first))` at or over
`glide_min_excursion_cents` is a `"glide"` span. `voiced_fraction` is the fraction of the span's frames
whose `strength` clears `voicing_strength_floor`; `production` is `"voiced"` above
`mixed_voiced_fraction`, `"unvoiced"` below `1 - mixed_voiced_fraction`, `"mixed"` between.
`offset_criterion` names what closed the span, and the assignment is total: `"f0_stability"` when the
F0 limb was the one holding and stopped, `"formant_stability"` when the formant limb was, `"both"`
when both limbs held and both stopped in the same frame, **`"monotonicity"` for every glide span** —
a glide is opened and closed by its monotone run, so the criterion that ends it is the frame where
monotonicity fails, never a stability limb — and `"stream_end"` when the span runs to the end of the
stream. Sibling T5 reports this string verbatim as `longest_span_criterion`, so a value outside those
five is a defect in this block, not in that one.

**`_consensus`** replaces `_agreement`:

```python
    def _consensus() -> None:
        """The consensus over both recognizers, by the audio-analysis routine, plus its word entities."""
        if "asr_crisperwhisper" not in state or "asr_qwen" not in state:
            raise LookupError("both recognizers are needed")
        activity = _step(
            "consensus",
            {"systems": [CRISPERWHISPER_ID, QWEN_ID], "routine": "fuse_consensus_words"},
            (state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
            software,
        )
        fused, provenance = fuse_consensus_words(
            {CRISPERWHISPER_ID: state["asr_crisperwhisper"], QWEN_ID: state["asr_qwen"]}
        )
        # ``fuse_consensus_words`` returns ``([], {})`` when no recognizer produced a readable word,
        # so ``provenance`` is empty on the wordless path and every later read of a named key would
        # raise. The measurement is still written — a fold that ran and found nothing is a fact —
        # with the operator recorded so a reader can tell it from a fold that never ran.
        if not provenance:
            provenance = {"operator": "consensus_words/resample", "sources": [], "n_words": 0}
        onomatopoeic = {
            _norm_token(str(token)) for token in (config.get("words.onomatopoeic_tokens") or [])
        }
        word_ids: list[str] = []
        event_ids: list[str] = []
        kept: list[dict[str, Any]] = []
        for entry in fused:
            span = _bound_to_duration(float(entry["start"]), float(entry["end"]), duration_s)
            if span is None:
                continue
            text = str(entry.get("text") or "")
            recognizers = [str(s) for s in (entry.get("sources") or [])]
            bracketed, origin = _as_non_word(text, onomatopoeic)
            if bracketed is not None:
                event_id = store.entity(
                    prov_type="event",
                    extent=span,
                    attributes={
                        "bracketed": bracketed,
                        "raw": text,
                        "origin": origin,
                        "recognizers": recognizers,
                    },
                )
                store.was_generated_by(event_id, activity)
                store.was_attributed_to(event_id, software)
                event_ids.append(event_id)
                continue
            word_id = store.entity(
                prov_type="word",
                extent=span,
                attributes={
                    "text": text,
                    "confidence": entry.get("confidence"),
                    "existence_confidence": entry.get("existence_confidence"),
                    "temporal_confidence": entry.get("temporal_confidence"),
                    "coverage": entry.get("coverage"),
                    "recognizers": recognizers,
                    "timing_sources": entry.get("timing_sources"),
                    "index": len(kept),
                },
            )
            store.was_generated_by(word_id, activity)
            store.was_attributed_to(word_id, software)
            word_ids.append(word_id)
            kept.append({**entry, "start": span[0], "end": span[1]})
        entity_id = _measurement(
            store,
            activity,
            software,
            name="consensus_transcript",
            signal="plain",
            attributes={
                "words": kept,
                "provenance": provenance,
                "systems": [CRISPERWHISPER_ID, QWEN_ID],
                "word_ids": word_ids,
                "event_ids": event_ids,
                "text": " ".join(str(entry.get("text") or "") for entry in kept),
            },
            derived_from=(state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
        )
        derivatives["consensus_transcript"] = entity_id
        view.append(entity_id)
        view.extend(word_ids)
        view.extend(event_ids)
        state.update(consensus=kept, consensus_id=entity_id)
```

with the two token helpers:

```python
def _norm_token(token: str) -> str:
    """A token normalised for vocabulary matching: casefolded, edge punctuation stripped."""
    return token.casefold().strip(".,;:!?\"'()")


def _as_non_word(text: str, onomatopoeic: set[str]) -> tuple[str | None, str | None]:
    """The bracketed form of a non-lexical token, or ``(None, None)`` when the token is a word.

    Args:
        text: The token as the recognizer produced it.
        onomatopoeic: The normalised ``words.onomatopoeic_tokens`` vocabulary; empty while it is null.

    Returns:
        ``(bracketed, origin)`` where ``origin`` is ``"bracketed"`` or ``"onomatopoeic"``, or
        ``(None, None)``.
    """
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped, "bracketed"
    normalised = _norm_token(stripped)
    if normalised and normalised in onomatopoeic:
        return f"[{normalised.upper()}]", "onomatopoeic"
    return None, None
```

**`_asr`** no longer writes `word` entities. It keeps its transcript measurement and adds a `words`
attribute holding that recognizer's own word list (`[{text, start, end, score}]`), so the evidence the
consensus was fused from is still in the store. It stores the `ScriptLine` in `state[name]` unchanged,
which is what `fuse_consensus_words` consumes.

**`_alignment`** reads `state["consensus"]` instead of `state["fused"]` and records
`transcript_source: "consensus_transcript"`.

**`_disruptions_file`**, the new block (V9):

```python
    def _disruptions_file() -> None:
        """Clipping, dropouts, discontinuities, DC and ZCR over the whole ORIGINAL recording."""
        if not recording_ids:
            raise LookupError("no recording stream in the store")
        parameters = {
            "clip_headroom": float(config.require("disruptions.clip_headroom")),
            "min_clip_run": int(config.require("disruptions.min_clip_run")),
            "min_dropout_ms": float(config.require("disruptions.min_dropout_ms")),
            "discontinuity_local_factor": float(config.require("disruptions.discontinuity_local_factor")),
            "discontinuity_window_ms": float(config.require("disruptions.discontinuity_window_ms")),
        }
        activity = _step("disruptions_file", parameters, (recording_ids[-1],), software)
        original_duration = source.waveform.shape[-1] / int(source.sampling_rate)
        found = detect_disruptions(source, 0.0, original_duration, **parameters)
        counts = {key: value for key, value in asdict(found).items() if key not in ("start", "end")}
        entity_id = _measurement(
            store,
            activity,
            software,
            name="disruptions_file",
            signal="recording",
            attributes={**counts, "sampling_rate": int(source.sampling_rate)},
            derived_from=(recording_ids[-1],),
        )
        derivatives["disruptions_file"] = entity_id
        view.append(entity_id)
```

**The `blocks` list**, in full and in order:

```python
    blocks: list[tuple[str, Callable[[], None]]] = [
        ("energy_envelope", _envelope),
        ("spans", _spans),
        ("yamnet_scores", _yamnet_scores),
        ("yamnet_windows", lambda: _windows("yamnet")),
        ("silence", _silence),
        ("ast_scores", _ast_scores),
        ("ast_windows", lambda: _windows("ast")),
        ("hear_scores", _hear_scores),
        ("hear_windows", lambda: _windows("hear")),
        ("level", _level),
        ("disruptions_file", _disruptions_file),
        ("squim", _squim),
        ("asr_crisperwhisper", lambda: _asr("asr_crisperwhisper", _crisperwhisper_model, "native", None)),
        (
            "asr_qwen",
            lambda: _asr("asr_qwen", _qwen_model, "bundled_aligner", QWEN_TIMESTAMP_MODEL, return_timestamps=True),
        ),
        ("consensus_transcript", _consensus),
        ("alignment", _alignment),
        ("phonation_spans", _phonation_spans),
        ("spectrogram_wideband", lambda: _spectrogram("spectrogram_wideband", "spectrogram.wideband_window_ms")),
        (
            "spectrogram_narrowband",
            lambda: _spectrogram("spectrogram_narrowband", "spectrogram.narrowband_window_ms"),
        ),
        ("gammatone", _gammatone),
    ]
```

- [ ] **Step 14 — run the PREPROCESS tests; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/preprocess_test.py -x -q`

- [ ] **Step 15 — lint, type-check, and run the suites this task owns.**
  `uv run ruff format src/senselab/audio/workflows/triage src/senselab/audio/tasks/phonation src/senselab/audio/tasks/spans src/tests/audio/workflows/triage src/tests/audio/tasks/phonation src/tests/audio/tasks/spans`
  `uv run ruff check` and `uv run mypy` over the same production paths.
  `uv run pytest src/tests/audio/tasks/phonation src/tests/audio/tasks/spans src/tests/audio/workflows/triage/nodes/preprocess_test.py src/tests/audio/workflows/triage/config_test.py -q`

  **The rest of the triage suite is red after this task, and stays red until T4–T7.** T1 changes the
  store schema every other node reads, so `taxonomy_test.py`, `airway_test.py`, `speech_test.py`,
  `voice_test.py`, `redact_test.py`, `verdict_test.py` and `run_test.py` all fail here. **T2 repairs
  TAXONOMY and T3 repairs routing + `run.py`; AIRWAY, SPEECH, VOICE and REDACT are repaired by T6, T4,
  T5 and T7 in the sibling plan, and VERDICT by T8.** Record the failing list at the end of this step
  and hand it forward; do not patch another task's module here.

**Dispatch order for both plan files** — stated once, here, because a fresh subagent taking a task in
isolation needs it:

```
T1  (foundation)                 ── must land first; everything reads its store schema
 └─ T2  (TAXONOMY)               ── sequential after T1
     └─ T3  (routing + run.py)   ── sequential after T2
         ├─ T4  (SPEECH)   ─┐
         ├─ T5  (VOICE)     │    ── parallel after T3; each owns its own test file and
         ├─ T6  (AIRWAY)    │       seeder, and none edits conftest.py, so they do not collide
         └─ T7  (REDACT)   ─┘
             └─ T8  (VERDICT)    ── after T3 and all of T4-T7 (it folds their verdicts)
                 └─ T9  (REPORT) ── after T8

T10 (CrisperWhisper diagnostic)  ── independent of every task above; any time
```

T4–T7 may also run sequentially; the parallelism is available, not required. What is **not** available
is running any of them before T3, because `run.py` calls `speech(..., enrollment=...)` only after T3
threads the parameter.

- [ ] **Step 16 — commit.**
  `git commit -m "feat(triage/preprocess): every whole-file model here, as label sets over per-label thresholds"`

**Interfaces:**

*Consumed (verified against this branch):*
- `classify_audios(audios, model, device=None, win_length=None, hop_length=None, top_k=None, **kwargs) -> list[AudioClassificationResult] | list[list[dict]]` — windowed when `win_length` is given; YAMNet ignores `win_length`/`hop_length` and always returns its native grid (`classification/api.py:40`).
- `label_scores(window) -> list[dict[str, float]]` — `[{label: score}, ...]` in rank order (`classification/label_scores.py:21`).
- `detect_health_acoustic_events(audios, model="hear-event-detector", device=None, hop_length=0.25, top_k=None) -> list[list[dict]]` (`health_acoustics/api.py:189`); `HEAR_MODEL_ID`, `HEAR_REVISION` (`health_acoustics/hear.py`).
- `fuse_consensus_words(asr_resolved: Mapping[str, Any], *, policy=None) -> tuple[list[dict], dict]` — returns `([], {})` when no model produced a readable word; each fused element carries `text, start, end, confidence, existence_confidence, temporal_confidence, coverage, corroboration, member_agreement, member_corroboration, sources, alternates, flags, timing_sources, speaker?` (`workflows/audio_analysis/asr.py:194`, `speech_to_text_ensemble/api.py:283`).
- `detect_disruptions(audio, start_s, end_s, *, clip_headroom, min_clip_run, min_dropout_ms, discontinuity_local_factor, discontinuity_window_ms) -> Disruptions` with fields `start, end, clipped_runs, clipped_s, dropout_runs, dropout_s, discontinuities, dc_offset, zero_crossing_rate` (`disruptions/api.py:101`).
- `f0_track(audio, *, f0_min_hz, f0_max_hz, hop_s) -> (times_s, f0_hz, strength)`; unvoiced frames carry NaN F0 with `strength` retained (`phonation/api.py:124`).

*Produced (the contract T2, T3 and every sibling task read):*
- `preprocess(store, source, config, hint=None, *, run_dir) -> PreprocessResult` — signature unchanged; `absent` now includes `yamnet_windows`, `ast_windows`, `hear_windows`, `phonation_spans` under the packaged config.
- Store: `yamnet_scores`/`ast_scores`/`hear_scores`, `yamnet_windows`/`ast_windows`/`hear_windows` and their per-window `*_window` measurements, `phonation` `span` entities with `formant_tracks`, `consensus_transcript` with `word` and `event` entities, `disruptions_file` — all exactly as §"The v2 store contract" states.
- `senselab.audio.tasks.phonation.formant_track`, `FormantTrack`, `propose_phonation_spans`, `PhonationSpan`.
- `senselab.audio.workflows.triage.nodes.common.find_measurements`, `live_entities`.
- **The one shared test fixture, `seed_preprocess_store`** — the single signature every downstream
  task's own seeder builds on, and the only fixture any task after T1 may assume exists in
  `conftest.py`:

  ```python
  seed_preprocess_store(
      store: ProvStore,
      *,
      stream_hz: int = 16000,
      duration_s: float = 5.0,
      yamnet_labels: list[list[str]] | None = None,
      ast_labels: list[list[str]] | None = None,
      hear_labels: list[list[str]] | None = None,
      words: list[str] | list[tuple[str, tuple[float, float]]] | None = None,
      events: list[str] | list[tuple[str, tuple[float, float]]] | None = None,
      phonation: list[tuple[float, float, str]] | None = None,
      spans: list[tuple[float, float, float]] | None = None,
      span_k_db: float = 18.0,
      span_merged: int = 1,
      disruptions_file: bool = False,
  ) -> None
  ```

  `None` writes nothing for that derivative; `[]` writes the derivative and records that it found
  nothing. `words=[]` still writes a `consensus_transcript` measurement (I7).
- `preprocess.AST_ID`, `YAMNET_MODEL_URI`, `CRISPERWHISPER_ID`, `QWEN_ID` — the sanctioned cross-node constants. **There is no `AST_FRAME_S`**; AST's window is `windows.ast.win_length_s`.

**Superseded tests, deleted with the ruling that justifies each:**

| deleted | ruling |
| --- | --- |
| every `preprocess_test.py` assertion on one `word` entity per recognizer word | preprocess.md: "`word` entities are written here, and only here" — over the consensus, which is one population, not two |
| `preprocess_test.py`'s `asr_agreement` assertions | preprocess.md names `consensus_transcript` produced by `fuse_consensus_words`; `asr_agreement` was v1's local `fuse_word_streams` call |
| `config_test.py`'s `hear.label_floor` and `taxonomy.presence_floor.yamnet` pins | preprocess.md: thresholds are now per-classifier, per-label, and null; the 0.5 whole-file values do not transfer to a per-window set rule |

---

### Task 2: TAXONOMY v2 — a fold over stored derivatives, three kinds, set-based evidence

**Scope:** `src/senselab/audio/workflows/triage/nodes/taxonomy.py` (rewritten);
`src/tests/audio/workflows/triage/nodes/taxonomy_test.py` (rewritten). No config change — T1 created
every key this task reads.

**Design points this task must not get wrong (from `taxonomy.md`):**

- **TAXONOMY runs no models.** Every piece of evidence it reads was written by PREPROCESS. The AST and
  HeAR calls that lived here in v1 are **deleted**, not moved and not conditionally kept.
- **Hints are not an input.** The `hint` parameter stays in the signature for the shared node shape and
  is documented as unread. A classification that reads the declaration cannot disagree with it.
- **It localises nothing.** No span is written, no extent is claimed.
- **Three kinds: `speech`, `airway`, `voice`.** There is no `not_screened` state and no residual kind.
  `voice_no_words` is deleted.
- **Set-based evidence, never an accumulator.** A line's evidence is a **count of windows whose label
  set intersects the kind's family** (or a count of consensus words), compared against that line's own
  floor. Nothing reads a score here; the scores were compared in PREPROCESS.
- **A missing derivative is not absence evidence.** A line whose derivative is absent from the store is
  `unavailable`, and a kind whose only line is unavailable is `uncertain`, never `absent`.
- **States are `present` | `absent` | `uncertain`.** `undecided` is renamed to `uncertain` outright.
- **Outcome:** `fail` when every kind is absent; `flag` when any kind is uncertain; `pass` when every
  kind is present or absent and at least one is present.

**Steps:**

- [ ] **Step 1 — write the failing tests.**

Replace `src/tests/audio/workflows/triage/nodes/taxonomy_test.py` wholesale:

```python
"""TAXONOMY v2: a fold over PREPROCESS's stored derivatives. No models, no hints, no localisation."""

from pathlib import Path
from typing import Any, Callable

import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import taxonomy as taxonomy_module
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.taxonomy import SCREENED_KINDS, taxonomy
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


def _floors(tmp_path: Path, **extra: str) -> TriageConfig:
    """The packaged config with every TAXONOMY floor supplied and the speech family named."""
    body = (
        "taxonomy:\n"
        "  presence_floor:\n"
        "    speech: {acoustic: 1, lexical: 1}\n"
        "    airway: {health_acoustic: 1, acoustic: 1}\n"
        "  voice_min_duration_s: 1.0\n"
        "  voice_uncertain_duration_s: 0.3\n"
        "  speech_labels: [Speech, Narration, monologue, Conversation]\n"
    )
    path = tmp_path / "floors.yaml"
    path.write_text(body + "".join(extra.values()))
    return load_triage_config(path)


class TestItRunsNoModels:
    """Every classifier call belongs to PREPROCESS; this node folds what is already there."""

    def test_the_module_imports_no_classifier(self) -> None:
        """A model function reachable from this module is a boundary violation, not a convenience."""
        for name in ("classify_audios", "detect_health_acoustic_events", "transcribe_audios"):
            assert not hasattr(taxonomy_module, name)

    def test_it_writes_no_activity_that_names_a_model(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The only activity is the fold."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Speech"]], words=2)
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert [a.step for a in store.activities("TAXONOMY")] == ["fold"]
        assert not [
            agent
            for activity in store.activities("TAXONOMY")
            for agent in store.associated_with(activity.id)
            if store.get_agent(agent).agent_type == "model"
        ]


class TestTheThreeKinds:
    """speech, airway and voice, each with its own rule and its own evidence."""

    def test_speech_needs_both_lines(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Acoustic windows and lexical words both clearing their floors makes speech present."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Speech"]], words=3)
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "present"

    def test_speech_with_windows_but_no_words_is_uncertain(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """One line present and one absent is disagreement, which is uncertain, not present."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Speech"]], words=0)
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "uncertain"

    def test_speech_with_neither_line_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Both lines below their floors is absence."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Music"]], words=0)
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "absent"

    def test_a_bracketed_event_carries_no_lexical_evidence(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """PREPROCESS wrote it as an event, so nothing here counts it toward the word floor."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Speech"]], words=0, events=3)
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "uncertain"

    def test_airway_needs_hear_and_audioset(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The health-acoustic and acoustic lines both carrying evidence makes airway present."""
        seed_preprocess_store(store, tmp_path, hear_labels=[["Cough"]], yamnet_labels=[["Cough"]])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["airway"] == "present"

    def test_ast_windows_serve_the_acoustic_line_beside_yamnet(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The acoustic line reads either grid; a label on AST alone is still acoustic evidence."""
        seed_preprocess_store(store, tmp_path, hear_labels=[["Cough"]], ast_labels=[["Cough"]])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["airway"] == "present"

    def test_voice_is_classified_from_phonation_span_duration_alone(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A 2 s sustain makes voice present, whatever else is in the recording."""
        seed_preprocess_store(store, tmp_path, phonation=[(0.0, 2.0, "voiced")])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["voice"] == "present"

    def test_an_unvoiced_sustain_makes_voice_present_too(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A disordered voice sustaining without periodicity is phonation."""
        seed_preprocess_store(store, tmp_path, phonation=[(0.0, 2.0, "unvoiced")])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["voice"] == "present"

    def test_a_short_span_is_uncertain_and_a_shorter_one_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Between the two floors is uncertain; below the shorter floor there is nothing to be sure of."""
        seed_preprocess_store(store, tmp_path, phonation=[(0.0, 0.5, "voiced")])
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "uncertain"
        other = ProvStore(run_id="short")
        seed_preprocess_store(other, tmp_path, phonation=[(0.0, 0.1, "voiced")])
        assert taxonomy(other, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "absent"

    def test_no_phonation_span_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The pass ran and found nothing, which is absence."""
        seed_preprocess_store(store, tmp_path, phonation=[])
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "absent"

    def test_no_phonation_pass_at_all_is_uncertain(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A pass that did not run leaves the line unavailable; that is not evidence of absence."""
        seed_preprocess_store(store, tmp_path, phonation=None)
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "uncertain"


class TestAMissingDerivativeIsNotAbsence:
    """A line whose derivative never reached the store is unavailable, and unavailable is uncertain."""

    def test_a_null_threshold_leaves_every_kind_uncertain(
        self, store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The packaged config folds no windows, so nothing can be absent."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=None, hear_labels=None, ast_labels=None, phonation=None)
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert set(result.kinds.values()) == {"uncertain"}
        assert result.verdict.outcome is Outcome.FLAG

    def test_the_unavailable_line_says_so_on_the_kind_element(
        self, store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A reader must see why a kind is uncertain, not only that it is."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=None, hear_labels=None, ast_labels=None, phonation=None)
        taxonomy(store, "plain", config, run_dir=tmp_path)
        speech = next(e for e in live_entities(store, "kind") if e.attributes["kind"] == "speech")
        assert speech.attributes["lines"]["acoustic"]["state"] == "unavailable"


class TestHintsAreNotAnInput:
    """A classification that reads the declaration cannot disagree with it."""

    def test_a_hint_declaring_speech_does_not_move_the_classification(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The same store classifies the same way with and without a hint."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Music"]], words=0)
        hinted = taxonomy(store, "plain", _floors(tmp_path), AudioHints(may_contain=["speech"]), run_dir=tmp_path)
        other = ProvStore(run_id="unhinted")
        seed_preprocess_store(other, tmp_path, yamnet_labels=[["Music"]], words=0)
        plain = taxonomy(other, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert hinted.kinds == plain.kinds


class TestTheOutcome:
    """fail on all-absent, flag on any-uncertain, pass otherwise."""

    def test_all_absent_fails(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Nothing is classified present."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Music"]], hear_labels=[[]], ast_labels=[[]], words=0, phonation=[])
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.FAIL

    def test_any_uncertain_flags(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """One kind the lines disagree about is enough."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Speech"]], hear_labels=[[]], ast_labels=[[]], words=0, phonation=[])
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.FLAG

    def test_present_and_absent_together_pass(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Speech present, airway and voice absent, nothing uncertain."""
        seed_preprocess_store(
            store, tmp_path, yamnet_labels=[["Speech"]], hear_labels=[[]], ast_labels=[["Speech"]], words=3, phonation=[]
        )
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.PASS

    def test_exactly_three_kind_elements_and_no_residual(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """voice_no_words and not_screened are gone; nothing is a kind by virtue of the others."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Speech"]], words=3)
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        kinds = {e.attributes["kind"] for e in live_entities(store, "kind")}
        assert kinds == set(SCREENED_KINDS) == {"airway", "speech", "voice"}
        assert not [e for e in live_entities(store, "kind") if e.attributes["state"] == "not_screened"]

    def test_it_localises_nothing(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """No span, no interval, no extent-bearing element is authored by this node."""
        seed_preprocess_store(store, tmp_path, yamnet_labels=[["Speech"]], words=3, phonation=[(0.0, 2.0, "voiced")])
        before = {e.id for e in live_entities(store, "span")}
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert {e.id for e in live_entities(store, "span")} == before
        assert not live_entities(store, "interval")
```

`seed_preprocess_store` is the shared fixture T1 defined; this task **adds nothing to
`conftest.py`**. Every `seed_preprocess_store(store, tmp_path, ...)` call above is
`seed_preprocess_store(store, ...)`, and the two behaviours these tests hinge on are the ones its
docstring states: passing `None` writes nothing for that derivative (the `unavailable` rows), and
passing `words=[]` still writes a `consensus_transcript` measurement carrying no words (I7), so
`test_speech_with_windows_but_no_words_is_uncertain` reads a lexical line that is `absent` rather than
`unavailable`. A test that got those backwards would pass for the wrong reason.

- [ ] **Step 2 — run them; expect FAIL** (`ImportError` on `SCREENED_KINDS` shape / `KeyError: 'voice'`).
  `uv run pytest src/tests/audio/workflows/triage/nodes/taxonomy_test.py -x -q`

- [ ] **Step 3 — rewrite `taxonomy.py`.**

The whole module. Imports drop `classify_audios`, `label_scores`, `detect_health_acoustic_events`,
`HEAR_MODEL_ID`, `HEAR_REVISION`, `HFModel` and `CRISPERWHISPER_ID`; they gain
`find_measurement`, `live_entities`.

```python
"""TAXONOMY — which kinds are in the recording, folded from PREPROCESS's stored derivatives.

It runs no model, reads no hint and localises nothing. Each kind's rule reads named evidence lines,
each line counts stored elements against its own configured floor, and a line whose derivative never
reached the store is ``unavailable`` — which makes its kind uncertain, never absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    live_entities,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "TAXONOMY"

SCREENED_KINDS = ("airway", "speech", "voice")

PRESENT = "present"
ABSENT = "absent"
UNCERTAIN = "uncertain"
UNAVAILABLE = "unavailable"

_PHONATION_FAMILY = "phonation"


@dataclass(frozen=True)
class TaxonomyResult(NodeResult):
    """TAXONOMY's result.

    Attributes:
        kinds: The classified state per kind — ``present``, ``absent`` or ``uncertain``.
    """

    kinds: dict[str, str]


def _window_evidence(store: ProvStore, classifier: str, family: set[str]) -> dict[str, Any]:
    """One acoustic line's evidence: how many of this classifier's windows carry a family member.

    Args:
        store: The provenance store.
        classifier: ``"yamnet"``, ``"ast"`` or ``"hear"``.
        family: The kind's label family for this classifier.

    Returns:
        ``{available, n_windows, element_ids}``. ``available`` is False when the classifier's pooled
        measurement is absent, which is the state a null threshold leaves.
    """
    pooled = find_measurement(store, f"{classifier}_windows")
    if pooled is None:
        return {"available": False, "n_windows": 0, "element_ids": []}
    windows_by_label: dict[str, list[str]] = pooled.attributes.get("windows_by_label") or {}
    matched = {
        window_id for label, ids in windows_by_label.items() if label in family for window_id in ids
    }
    return {"available": True, "n_windows": len(matched), "element_ids": sorted(matched), "pooled": pooled.id}


def _acoustic_line(store: ProvStore, family: set[str]) -> dict[str, Any]:
    """The AudioSet line, over YAMNet and AST together: either grid's windows are acoustic evidence."""
    yamnet = _window_evidence(store, "yamnet", family)
    ast = _window_evidence(store, "ast", family)
    if not yamnet["available"] and not ast["available"]:
        return {"available": False, "n_windows": 0, "element_ids": []}
    return {
        "available": True,
        "n_windows": yamnet["n_windows"] + ast["n_windows"],
        "element_ids": yamnet["element_ids"] + ast["element_ids"],
    }


def _lexical_line(store: ProvStore) -> dict[str, Any]:
    """The lexical line: consensus ``word`` entities. Bracketed and onomatopoeic events are not words."""
    if find_measurement(store, "consensus_transcript") is None:
        return {"available": False, "n_words": 0, "element_ids": []}
    words = live_entities(store, "word")
    return {"available": True, "n_words": len(words), "element_ids": [w.id for w in words]}


def _phonation_spans(store: ProvStore) -> list[Entity] | None:
    """PREPROCESS's phonation spans, or None when the pass left nothing in the store at all.

    Args:
        store: The provenance store.

    Returns:
        The live phonation spans, possibly empty; None when no phonation activity ran, so a reader
        can tell "the pass found nothing" from "the pass did not happen".
    """
    if not [activity for activity in store.activities("PREPROCESS") if activity.step == "phonation_spans"]:
        return None
    return [e for e in live_entities(store, "span") if e.attributes.get("family") == _PHONATION_FAMILY]


def _line_state(available: bool, evidence: int, floor: Any) -> str:
    """One line's state from its evidence and its floor.

    Args:
        available: Whether the derivative the line reads is in the store.
        evidence: The count the line measured.
        floor: The configured floor, or None while it is unmeasured.

    Returns:
        ``unavailable`` when the derivative is missing or the floor is unmeasured — a line that
        cannot be judged has said nothing, which is not the same as saying absent — else
        ``present`` or ``absent``.
    """
    if not available or floor is None:
        return UNAVAILABLE
    return PRESENT if evidence >= int(floor) else ABSENT


def _fold_two_lines(lines: dict[str, dict[str, Any]]) -> str:
    """The two-line rule: present when both carry evidence, absent when neither does, else uncertain."""
    states = [line["state"] for line in lines.values()]
    if all(state == PRESENT for state in states):
        return PRESENT
    if all(state == ABSENT for state in states):
        return ABSENT
    return UNCERTAIN


def taxonomy(
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> TaxonomyResult:
    """Classify which kinds are in the recording, from the store alone.

    Args:
        store: The provenance store, holding PREPROCESS's derivatives.
        source: The stream every element it writes names, ``"plain"``.
        config: The triage configuration.
        hint: Accepted for the shared node shape and **not read**. A classification that reads the
            declaration cannot disagree with it; forcing a branch is ``routing``'s job.
        run_dir: Accepted for the shared node shape; this node writes no sidecars.

    Returns:
        The verdict, the three kind element ids as the view, and the state per kind.
    """
    software = software_agent(store)
    speech_family = {str(label) for label in (config.get("taxonomy.speech_labels") or [])}
    audioset_airway = {str(label) for label in config.require("taxonomy.audioset_airway_labels")}
    hear_airway = {str(label) for label in config.require("taxonomy.hear_airway_labels")}
    floors = {
        ("speech", "acoustic"): config.get("taxonomy.presence_floor.speech.acoustic"),
        ("speech", "lexical"): config.get("taxonomy.presence_floor.speech.lexical"),
        ("airway", "health_acoustic"): config.get("taxonomy.presence_floor.airway.health_acoustic"),
        ("airway", "acoustic"): config.get("taxonomy.presence_floor.airway.acoustic"),
    }
    voice_min_s = config.get("taxonomy.voice_min_duration_s")
    voice_uncertain_s = config.get("taxonomy.voice_uncertain_duration_s")

    speech_acoustic = _acoustic_line(store, speech_family) if speech_family else {
        "available": False, "n_windows": 0, "element_ids": []
    }
    speech_lexical = _lexical_line(store)
    airway_health = _window_evidence(store, "hear", hear_airway)
    airway_acoustic = _acoustic_line(store, audioset_airway)
    spans = _phonation_spans(store)

    lines: dict[str, dict[str, dict[str, Any]]] = {
        "speech": {
            "acoustic": {
                "state": _line_state(
                    speech_acoustic["available"], speech_acoustic["n_windows"], floors[("speech", "acoustic")]
                ),
                "evidence": speech_acoustic["n_windows"],
                "unit": "windows",
                "floor": floors[("speech", "acoustic")],
                "element_ids": speech_acoustic["element_ids"],
            },
            "lexical": {
                "state": _line_state(
                    speech_lexical["available"], speech_lexical["n_words"], floors[("speech", "lexical")]
                ),
                "evidence": speech_lexical["n_words"],
                "unit": "words",
                "floor": floors[("speech", "lexical")],
                "element_ids": speech_lexical["element_ids"],
            },
        },
        "airway": {
            "health_acoustic": {
                "state": _line_state(
                    airway_health["available"], airway_health["n_windows"], floors[("airway", "health_acoustic")]
                ),
                "evidence": airway_health["n_windows"],
                "unit": "windows",
                "floor": floors[("airway", "health_acoustic")],
                "element_ids": airway_health["element_ids"],
            },
            "acoustic": {
                "state": _line_state(
                    airway_acoustic["available"], airway_acoustic["n_windows"], floors[("airway", "acoustic")]
                ),
                "evidence": airway_acoustic["n_windows"],
                "unit": "windows",
                "floor": floors[("airway", "acoustic")],
                "element_ids": airway_acoustic["element_ids"],
            },
        },
    }

    longest_s = max((float(e.attributes["duration_s"]) for e in spans), default=0.0) if spans else 0.0
    if spans is None or voice_min_s is None or voice_uncertain_s is None:
        voice_line_state, voice_state = UNAVAILABLE, UNCERTAIN
    elif longest_s >= float(voice_min_s):
        voice_line_state, voice_state = PRESENT, PRESENT
    elif longest_s >= float(voice_uncertain_s):
        voice_line_state, voice_state = PRESENT, UNCERTAIN
    else:
        voice_line_state, voice_state = ABSENT, ABSENT
    lines["voice"] = {
        "phonation": {
            "state": voice_line_state,
            "evidence": longest_s,
            "unit": "seconds",
            "floor": voice_min_s,
            "uncertain_floor": voice_uncertain_s,
            "element_ids": [e.id for e in spans] if spans else [],
        }
    }

    states = {
        "speech": _fold_two_lines(lines["speech"]),
        "airway": _fold_two_lines(lines["airway"]),
        "voice": voice_state,
    }

    fold = store.activity(node=NODE, step="fold", parameters={"kinds": list(SCREENED_KINDS), "stream": source})
    store.was_associated_with(fold, software)
    read_ids = {
        element_id
        for kind_lines in lines.values()
        for line in kind_lines.values()
        for element_id in line["element_ids"]
    }
    for element_id in sorted(read_ids):
        store.used(fold, element_id)

    view: list[str] = []
    for kind in SCREENED_KINDS:
        kind_id = store.entity(
            prov_type="kind",
            extent=None,
            attributes={"kind": kind, "state": states[kind], "lines": lines[kind], "stream": source},
        )
        store.was_generated_by(kind_id, fold)
        store.was_attributed_to(kind_id, software)
        view.append(kind_id)

    if all(state == ABSENT for state in states.values()):
        outcome, why = Outcome.FAIL, "every kind is absent"
    elif any(state == UNCERTAIN for state in states.values()):
        uncertain = [kind for kind in SCREENED_KINDS if states[kind] == UNCERTAIN]
        outcome, why = Outcome.FLAG, "uncertain: " + ", ".join(uncertain)
    else:
        outcome, why = Outcome.PASS, "every kind is present or absent, and at least one is present"

    verdict_id, verdict = write_verdict(
        store, fold, software, node=NODE, outcome=outcome, kind=None, why=why, detail={"kinds": states}
    )
    view.append(verdict_id)
    return TaxonomyResult(verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, kinds=states)
```

- [ ] **Step 4 — run them; expect PASS.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/taxonomy_test.py -x -q`

- [ ] **Step 5 — lint, type-check.**
  `uv run ruff format src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run ruff check src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run mypy src/senselab/audio/workflows/triage`

- [ ] **Step 6 — commit.**
  `git commit -m "feat(triage/taxonomy): classify three kinds from stored evidence, and run no model"`

**Interfaces:**

*Consumed:* T1's `yamnet_windows` / `ast_windows` / `hear_windows` pooled measurements (their
`windows_by_label` index and nothing else), the `consensus_transcript` measurement's existence, live
`word` entities, live `span` entities with `family == "phonation"` and their `duration_s`, and the
`PREPROCESS`/`phonation_spans` activity's existence — which is how "the pass did not run" is told from
"the pass found nothing". `common.find_measurement`, `common.live_entities`.

*Produced (the T2→T3 and T2→T8 contract):*
- `taxonomy(store, source, config, hint=None, *, run_dir) -> TaxonomyResult` with
  `kinds: dict[str, str]` over exactly `{"airway", "speech", "voice"}`, values in
  `{"present", "absent", "uncertain"}`.
- `taxonomy.SCREENED_KINDS == ("airway", "speech", "voice")`, and the module constants `PRESENT`,
  `ABSENT`, `UNCERTAIN`, `UNAVAILABLE` — the controlled strings T3 and sibling T8 compare against.
- Three `kind` entities, `{kind, state, lines, stream}`, exactly as §"The v2 store contract" states.

**Superseded tests, deleted with the ruling that justifies each:**

| deleted | ruling |
| --- | --- |
| `TestEligibility::test_hear_is_barred_from_the_speech_kind` and every other family-eligibility test | taxonomy.md replaces the family committee with named per-kind lines; HeAR is confined to airway by the *rule*, not by an eligibility table |
| `TestTheFold` in full (`min_families`, unanimity, the out-of-range override) | taxonomy.md: "Present when both lines carry evidence at or above their configured floors" — there is no family count and no `min_families` key |
| `TestMembersAndArguments::test_ast_abstains_while_its_floor_is_null`, `test_model_arguments_are_explicit` | taxonomy.md: "TAXONOMY runs no models" — there are no model arguments here to be explicit about |
| `test_advisory_on_fail_everything_is_still_written` | taxonomy.md is no longer advisory: routing.md gates the branches on its output |
| every `voice_no_words` / `not_screened` assertion | taxonomy.md: "There is no `not_screened` state and no residual kind" |

---

### Task 3: ROUTING — the branch gate, and the runner that honours it

**Scope:** `src/senselab/audio/workflows/triage/nodes/routing.py` (new);
`src/senselab/audio/workflows/triage/run.py` (conditional branch execution);
`src/tests/audio/workflows/triage/nodes/routing_test.py` (new);
`src/tests/audio/workflows/triage/run_test.py` (extended). `PROV_TYPE` already carries
`branch_decision` from T1 step 1; `routing.hint_kind_map` already exists as null from T1 step 4.

**Design points this task must not get wrong (from `routing.md`):**

- **It measures nothing and classifies nothing.** It reads `kind` elements and the hint, and writes one
  `branch_decision` per branch.
- **`present` runs, `uncertain` runs, `absent` does not run unless a hint forces it.** Uncertain runs
  because the branch is the more precise instrument and an unsettled kind is what a branch exists to
  settle.
- **Hints force execution; they never alter the classification.** The `kind` element TAXONOMY wrote is
  **not** rewritten. Forcing **adds** a branch — never removes one, never relaxes a threshold, never
  makes a branch's own conclusion more or less likely.
- **A tag with no entry in `routing.hint_kind_map` forces nothing and is recorded as unmapped.**
- **No `fail`.** A file every branch declines is a `flag` carrying which kinds were absent, not a
  failure and not a discard — the fold decides what an empty execution set means.
- **The decisions are written before any branch runs**, which is what lets VERDICT tell a branch that
  found nothing from a branch that never looked.
- **REDACT is inside SPEECH.** ROUTING writes no decision for it.

**Steps:**

- [ ] **Step 1 — write the failing tests.**

`src/tests/audio/workflows/triage/nodes/routing_test.py`:

```python
"""ROUTING: which branches run, why, and the record that lets VERDICT tell 'nothing' from 'never looked'."""

from pathlib import Path
from typing import Callable

import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.routing import BRANCH_FOR_KIND, routing
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


def _map(tmp_path: Path) -> TriageConfig:
    """The packaged config with a hint map supplied, covering tags and one speech_type value."""
    path = tmp_path / "routing.yaml"
    path.write_text(
        "routing:\n"
        "  hint_kind_map:\n"
        "    speech: speech\n"
        "    read-speech: speech\n"
        "    cough: airway\n"
        "    phonation: voice\n"
        "    prolonged-vowel: voice\n"
    )
    return load_triage_config(path)


def _kinds(store: ProvStore, **states: str) -> None:
    """Write one kind element per named kind, as TAXONOMY would."""
    activity = store.activity(node="TAXONOMY", step="fold", parameters={})
    for kind, state in states.items():
        entity_id = store.entity(
            prov_type="kind", extent=None, attributes={"kind": kind, "state": state, "lines": {}, "stream": "plain"}
        )
        store.was_generated_by(entity_id, activity)


class TestTheRule:
    """present runs, uncertain runs, absent does not."""

    def test_present_runs(self, store: ProvStore, tmp_path: Path) -> None:
        """A kind the classification found runs its branch."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.runs == ("SPEECH",)

    def test_uncertain_runs(self, store: ProvStore, tmp_path: Path) -> None:
        """A kind the classification could not settle is exactly what a branch exists to settle."""
        _kinds(store, speech="uncertain", airway="absent", voice="absent")
        assert routing(store, None, _map(tmp_path), run_dir=tmp_path).runs == ("SPEECH",)

    def test_absent_does_not_run(self, store: ProvStore, tmp_path: Path) -> None:
        """With no hint, an absent kind's branch is skipped and the decision says why."""
        _kinds(store, speech="absent", airway="present", voice="absent")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.runs == ("AIRWAY",)
        assert set(result.skipped) == {"SPEECH", "VOICE"}


class TestHintsForceAndNothingElse:
    """A hint adds a branch. It never rewrites a classification and never removes a branch."""

    def test_a_hint_forces_an_absent_kinds_branch(self, store: ProvStore, tmp_path: Path) -> None:
        """The branch runs against an absent classification, which is the mismatch VERDICT detects."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ("AIRWAY",)
        assert result.forced == ("AIRWAY",)

    def test_speech_type_metadata_forces_too(self, store: ProvStore, tmp_path: Path) -> None:
        """routing.md names both may_contain and the task's speech_type as forcing inputs."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        hint = AudioHints(metadata={"speech_type": "read-speech"})
        assert routing(store, None, _map(tmp_path), hint, run_dir=tmp_path).runs == ("SPEECH",)

    def test_forcing_does_not_rewrite_the_kind_element(self, store: ProvStore, tmp_path: Path) -> None:
        """The disagreement between decision and classification is the product, not a thing to erase."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        airway = next(e for e in live_entities(store, "kind") if e.attributes["kind"] == "airway")
        assert airway.attributes["state"] == "absent"
        decision = next(
            e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "AIRWAY"
        )
        assert decision.attributes["kind_state"] == "absent"
        assert decision.attributes["forced_by_hint"] is True

    def test_forcing_never_removes_a_branch(self, store: ProvStore, tmp_path: Path) -> None:
        """A hint naming only cough leaves a present speech kind's branch running."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert set(result.runs) == {"SPEECH", "AIRWAY"}

    def test_an_unmapped_tag_forces_nothing_and_is_recorded(self, store: ProvStore, tmp_path: Path) -> None:
        """A tag with no entry is data about the hint, not a silent no-op."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["birdsong"]), run_dir=tmp_path)
        assert result.runs == ()
        decision = live_entities(store, "branch_decision")[0]
        assert decision.attributes["unmapped_tags"] == ["birdsong"]

    def test_a_null_map_forces_nothing(self, store: ProvStore, config: TriageConfig, tmp_path: Path) -> None:
        """While the vocabulary is unmeasured, every tag is unmapped and nothing is forced."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, config, AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ()
        assert result.verdict.outcome is Outcome.FLAG


class TestTheEmptyExecutionSet:
    """A file that enters no branch is flagged, not failed and not discarded."""

    def test_no_branch_flags(self, store: ProvStore, tmp_path: Path) -> None:
        """routing has no fail; whether an empty set discards the file is the fold's decision."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert result.empty_set is True
        assert "absent" in result.verdict.why

    def test_any_branch_running_passes(self, store: ProvStore, tmp_path: Path) -> None:
        """A non-empty execution set is a pass; nothing here is a judgement about the recording."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        assert routing(store, None, _map(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.PASS


class TestTheStoreContract:
    """One decision per branch, before any branch runs, tied to the classification it rests on."""

    def test_three_decisions_and_none_for_redact(self, store: ProvStore, tmp_path: Path) -> None:
        """REDACT is a step of SPEECH, not a branch beside it."""
        _kinds(store, speech="present", airway="present", voice="present")
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        branches = {e.attributes["branch"] for e in live_entities(store, "branch_decision")}
        assert branches == set(BRANCH_FOR_KIND.values()) == {"AIRWAY", "SPEECH", "VOICE"}

    def test_each_decision_is_derived_from_its_kind_element(self, store: ProvStore, tmp_path: Path) -> None:
        """wasDerivedFrom ties the decision to the classification, and used records the read."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        speech_kind = next(e for e in live_entities(store, "kind") if e.attributes["kind"] == "speech")
        decision = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert speech_kind.id in store.derived_from(decision.id)
        activity = store.get_activity(store.generated_by(decision.id))
        assert speech_kind.id in store.uses_of(activity.id)

    def test_a_kind_taxonomy_never_wrote_is_uncertain_and_runs(self, store: ProvStore, tmp_path: Path) -> None:
        """A classification that is not in the store is not an absence; the branch is asked."""
        _kinds(store, speech="present")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert set(result.runs) == {"SPEECH", "AIRWAY", "VOICE"}
        airway = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "AIRWAY")
        assert airway.attributes["kind_state"] == "uncertain"
```

Append to `src/tests/audio/workflows/triage/run_test.py`. **These use that file's own existing
helpers** — the `graph` fixture, `_fakes` and `_tone` — and **not** `nodes/conftest.py`'s `wav_writer`,
which is one directory down and does not apply here. `_fakes` gains two keyword arguments (`kinds`,
which the fake TAXONOMY writes as `kind` entities, and `routing_outcome`) and one more fake, `_routing`,
which calls the real `routing` node over the store the fake TAXONOMY seeded — so these tests exercise
the real gate against fake branches, which is the behaviour under test:

```python
class TestConditionalExecution:
    """run.py runs the branches routing selected, and records the rest as skipped."""

    def test_a_skipped_branch_is_not_called_and_is_recorded_skipped(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A branch with will_run false never runs, and RunState.SKIPPED says so."""
        calls = graph(kinds={"speech": "present", "airway": "absent", "voice": "absent"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert "AIRWAY" not in calls
        assert result.ran["AIRWAY"] is RunState.SKIPPED
        assert result.ran["SPEECH"] is RunState.COMPLETED

    def test_redact_runs_only_when_speech_ran_and_found_pii(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """REDACT is a step of SPEECH; no speech branch means no REDACT verdict at all."""
        calls = graph(kinds={"speech": "absent", "airway": "present", "voice": "absent"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert "REDACT" not in calls
        assert result.ran["REDACT"] is RunState.SKIPPED

    def test_speech_running_without_a_finding_still_skips_redact(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """redact.md: SPEECH ran and found no PII, so the release axis reads not_assessed."""
        calls = graph(kinds={"speech": "present", "airway": "absent", "voice": "absent"}, pii=False)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert "SPEECH" in calls and "REDACT" not in calls
        assert result.file_verdict is not None
        assert result.file_verdict.release is Release.NOT_ASSESSED

    def test_speech_running_with_a_finding_reaches_redact(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """One live pii entity is the whole gate."""
        calls = graph(kinds={"speech": "present", "airway": "absent", "voice": "absent"}, pii=True)
        run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert "REDACT" in calls

    def test_an_empty_execution_set_still_reaches_verdict(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The file reaches the fold with no branch conclusions, which is what routing.md requires."""
        calls = graph(kinds={"speech": "absent", "airway": "absent", "voice": "absent"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.ran["VERDICT"] is RunState.COMPLETED
        assert {"AIRWAY", "SPEECH", "VOICE"}.isdisjoint(calls)

    def test_a_raising_routing_runs_every_branch_and_is_recorded_errored(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The degradation is designed, not a default: the fold sees a node that was asked and was silent."""
        calls = graph(routing_outcome="raise")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert {"AIRWAY", "SPEECH", "VOICE"} <= set(calls)
        assert result.ran["routing"] is RunState.ERRORED
```

`GRAPH`, the module-level tuple `run_test.py` asserts graph order against, gains `"routing"` between
`"TAXONOMY"` and `"AIRWAY"`, and `TestHappyPath::test_calls_all_eight_nodes_in_graph_order` is renamed
to `test_calls_all_nine_nodes_in_graph_order` — its body is unchanged, since it compares against
`GRAPH`.

- [ ] **Step 4 — make `run.py` honour the decisions.**

`GRAPH_ORDER` gains `"routing"` between `"TAXONOMY"` and `"AIRWAY"`:

```python
GRAPH_ORDER = ("ADMIT", "PREPROCESS", "TAXONOMY", "routing", "AIRWAY", "SPEECH", "VOICE", "REDACT", "VERDICT")
```

`_drive_branches` becomes:

```python
def _drive_branches(
    store: ProvStore,
    audio: Audio,
    config: TriageConfig,
    hint: AudioHints | None,
    *,
    run_dir: Path,
    artifacts_dir: Path,
    outcomes: dict[str, NodeOutcome],
    enrollment: Enrollment | None,
) -> dict[str, Path]:
    """Run PREPROCESS, TAXONOMY and routing, then exactly the branches routing selected.

    A branch routing declined is recorded ``SKIPPED`` and never called, which is what lets VERDICT
    tell a branch that found nothing from a branch that never looked. A branch that raises is still
    recorded ``ERRORED`` and its siblings still run: none of them reads another's output.

    Args:
        store: The provenance store, already holding ADMIT's ``recording`` stream.
        audio: The audio ADMIT decoded.
        config: The triage configuration.
        hint: What the recording was declared to contain.
        run_dir: The run directory sidecar paths are relative to.
        artifacts_dir: The release directory handed to REDACT.
        outcomes: The per-node record each call is added to.
        enrollment: The target speaker's enrollment, when the caller supplied one.

    Returns:
        REDACT's released pair, empty unless it cleared one.
    """
    _attempt(outcomes, "PREPROCESS", lambda: preprocess(store, audio, config, hint, run_dir=run_dir))
    _attempt(outcomes, "TAXONOMY", lambda: taxonomy(store, _CONDITIONED_STREAM, config, hint, run_dir=run_dir))
    routed = _attempt(outcomes, "routing", lambda: routing(store, None, config, hint, run_dir=run_dir))
    selected = set(routed.runs) if routed is not None else set(_BRANCHES)
    branches: dict[str, Callable[[], NodeResult]] = {
        "AIRWAY": lambda: airway(store, _CONDITIONED_STREAM, config, hint, run_dir=run_dir),
        "SPEECH": lambda: speech(
            store, _CONDITIONED_STREAM, config, hint, run_dir=run_dir, enrollment=enrollment
        ),
        "VOICE": lambda: voice(store, _CONDITIONED_STREAM, config, hint, run_dir=run_dir),
    }
    for branch in _BRANCHES:
        if branch in selected:
            _attempt(outcomes, branch, branches[branch])
        else:
            outcomes[branch] = NodeOutcome(node=branch, state=RunState.SKIPPED)
    if "SPEECH" in selected and _speech_found_pii(store):
        redacted = _attempt(
            outcomes,
            "REDACT",
            lambda: redact(store, _SOURCE_STREAM, config, hint, run_dir=run_dir, artifacts_dir=artifacts_dir),
        )
        return dict(redacted.artifacts) if redacted is not None else {}
    outcomes["REDACT"] = NodeOutcome(node="REDACT", state=RunState.SKIPPED)
    return {}
```

with the two module additions:

```python
_BRANCHES = ("AIRWAY", "SPEECH", "VOICE")


def _speech_found_pii(store: ProvStore) -> bool:
    """Whether SPEECH's scan over the consensus transcript found anything.

    REDACT is a step of SPEECH and runs only on a finding, so a file with no PII has no REDACT
    verdict at all and its release axis reads ``not_assessed``.

    Args:
        store: The provenance store.

    Returns:
        True when at least one live ``pii`` entity is in the store.
    """
    return bool([finding for finding in store.entities("pii") if not store.is_invalidated(finding.id)])
```

`run_triage` gains an `enrollment: Enrollment | None = None` parameter (sibling T4 defines
`Enrollment`; until it lands, type it `Any` and the sibling task narrows it), forwards it, and its
ADMIT-fail branch marks `GRAPH_ORDER[1:-1]` skipped as before — which now includes `routing`.

**`routing`'s absence must not silently run everything.** When `routed is None` — the node raised —
`selected` falls back to every branch, and the `_attempt` record for `routing` is already
`RunState.ERRORED`, so VERDICT sees a node that was asked and left no answer and flags. That is the
designed degradation; it is not a default.

- [ ] **Step 5 — run them; expect PASS — over routing, taxonomy and the runner only.**
  `uv run pytest src/tests/audio/workflows/triage/nodes/routing_test.py src/tests/audio/workflows/triage/nodes/taxonomy_test.py src/tests/audio/workflows/triage/run_test.py -x -q`
  `airway_test.py`, `speech_test.py`, `voice_test.py`, `redact_test.py` and `verdict_test.py` are
  **still red** at the end of the foundation half and are repaired by T6, T4, T5, T7 and T8. Do not
  run the whole triage directory here and do not repair another task's module.

- [ ] **Step 6 — lint, type-check.**
  `uv run ruff format src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run ruff check src/senselab/audio/workflows/triage src/tests/audio/workflows/triage`
  `uv run mypy src/senselab/audio/workflows/triage`

- [ ] **Step 7 — commit.**
  `git commit -m "feat(triage/routing): the branch gate, and a runner that runs only what it selects"`

**Interfaces:**

*Consumed:* T2's `kind` entities and its `PRESENT`/`ABSENT`/`UNCERTAIN` strings (compared as literals
here rather than imported, because a node importing a sibling node is a defect — the strings are the
store's vocabulary and are pinned by `routing_test.py` and `taxonomy_test.py` together);
`AudioHints.may_contain` and `AudioHints.metadata`; `common.live_entities`, `write_verdict`.

*Produced (the T3→T8 contract, and the runner contract):*
- `routing(store, source, config, hint=None, *, run_dir) -> RoutingResult` with `runs`, `skipped`,
  `forced`, `empty_set`.
- `routing.BRANCH_FOR_KIND == {"airway": "AIRWAY", "speech": "SPEECH", "voice": "VOICE"}` — the mapping
  sibling T8 joins branch verdicts to decisions with.
- Three `branch_decision` entities, exactly as §"The v2 store contract" states, each
  `wasDerivedFrom` its `kind` element where one exists.
- `run.GRAPH_ORDER` including `"routing"`; `run._BRANCHES`; `run_triage(..., enrollment=None)`.
- `run_test.py`'s `_fakes` gains `kinds`, `pii` and `routing_outcome` keyword arguments and a
  `_routing` entry that calls the **real** `routing` node over the store the fake TAXONOMY seeded, so
  the gate is exercised against fake branches. `GRAPH` gains `"routing"`. Sibling T9 extends the same
  `_fakes`, and is the only other task that touches `run_test.py`.

**Superseded tests, deleted with the ruling that justifies each:**

| deleted | ruling |
| --- | --- |
| `run_test.py`'s assertions that every branch runs unconditionally | routing.md: "a branch runs only if its decision says so" |
| `run_test.py`'s assertion that REDACT runs after SPEECH whatever it concluded | redact.md: "Only when SPEECH's PII scan over the consensus transcript found something" |

---

## What this plan file does not build

- The branches, REDACT, VERDICT and REPORT — `plan-v2-2.md`.
- Any second pass of `PREPROCESS → TAXONOMY → routing` over a suppressed-foreground stream. The store
  contract makes one expressible (every element names its stream); nothing here invokes one.
- Any orchestration beyond `run.py`'s single-file drive. Multi-file orchestration lives in
  `specs/20260817-triage-workflow-dag/nextflow/`.
- Any value for any of the 33 open keys.

## Self-review (second pass, after the review fixes)

### Spec coverage — every v2 spec section this file owns maps to a task

| spec section | task | where |
| --- | --- | --- |
| preprocess.md §Conditioning | — | unchanged from v1; T1 keeps the two-signal split and the `recording` retention |
| preprocess.md §Derivatives (table) | T1 | `blocks` list, step 13; `disruptions_file` added (V9), `consensus_transcript` replaces `asr_agreement`, `phonation_spans` and `formant_tracks` added |
| preprocess.md §Window classifications — sets, not accumulators | T1 | `_confident_labels`, `_scores`/`_windows`, steps 11 & 13; `TestWindowClassificationsAreSets` |
| preprocess.md §`spans` | — | unchanged from v1 |
| preprocess.md §`phonation_spans` | T1 | `_phonation_spans`, `propose_phonation_spans` (V5, V6); `TestPhonationSpans` |
| preprocess.md §`consensus_transcript` | T1 | `_consensus` calling `fuse_consensus_words` (V7); `TestTheConsensusTranscript` |
| preprocess.md §Words are bracket-aware | T1 | `_as_non_word`, `words.onomatopoeic_tokens` (V8); `TestWordsAreBracketAware` |
| preprocess.md §Working rate | T1 | `disruptions_file` on the `recording` stream is the named exception; `TestDisruptionsAreMeasuredOnTheOriginal` |
| preprocess.md §Open derivations (v2), 5 rows | T1 | 15 keys, step 4 |
| taxonomy.md §Signature, §What it reads | T2 | `taxonomy()` and its line readers |
| taxonomy.md §The three kinds and their rules | T2 | `_acoustic_line`, `_lexical_line`, `_phonation_spans`, the voice duration ladder; `TestTheThreeKinds` |
| taxonomy.md §States, §"A missing derivative is not absence evidence" | T2 | `_line_state`'s `UNAVAILABLE` row; `TestAMissingDerivativeIsNotAbsence` |
| taxonomy.md §"Hints are not an input" | T2 | documented-unread `hint`; `TestHintsAreNotAnInput` |
| taxonomy.md §"TAXONOMY runs no models" | T2 | imports deleted; `TestItRunsNoModels` |
| taxonomy.md §"It localises nothing" | T2 | `test_it_localises_nothing` |
| taxonomy.md §Outcome, §Product, §"All three kinds are screened" | T2 | the outcome ladder; `TestTheOutcome` |
| taxonomy.md §Open derivations (v2), 4 rows | T1 | 7 keys, step 4 |
| routing.md §Signature, §What it reads, §The rule | T3 | `routing()`; `TestTheRule` |
| routing.md §"Hints force execution; they never alter the classification" | T3 | `forced_by_hint`, no kind rewrite; `TestHintsForceAndNothingElse` |
| routing.md §"REDACT is inside the speech branch" | T3 | no decision for REDACT; `run._speech_found_pii` |
| routing.md §"A file that enters no branch is flagged" | T3 | `empty_set`; `TestTheEmptyExecutionSet` |
| routing.md §"The pass is encapsulated" | T3 (V14) | `stream: "plain"` on every element; one pass, stated |
| routing.md §Store contract, §Product | T3 | `branch_decision` schema; `TestTheStoreContract` |
| routing.md §Open derivations (v2), 1 row | T1 | `routing.hint_kind_map`, step 4 |
| store.md §PROV model, §Ordering is declared by what a node reads | T1, T3 | `PROV_TYPE` additions; `used`/`wasDerivedFrom` on every read |

**Spec sections deliberately not covered here**, each named with the sibling task that owns it:
`branch-airway.md` → T6; `branch-speech.md` → T4, T7; `branch-voice.md` → T5; `redact.md` → T7;
`verdict.md` → T8; `report.md` → T9. Their open-derivation keys are nonetheless created by T1, so the
sibling tasks read keys that already exist.

### Placeholder scan

Searched this file for `TBD`, `TODO`, `FIXME`, `XXX`, `...` used as an ellipsis-in-code, "add
validation", "similar to task", "as above", "etc." in a step body, and "handle appropriately".

- **`TBD`/`TODO`/`FIXME`/`XXX`: none.**
- **`...` appears only inside store-schema *illustrations*** (`[label, ...]`, `[ entity_id, ... ]`) in
  §"The v2 store contract", where it names a repeated element of a documented list, never a step's
  code. Every code block a step tells an implementer to type is complete Python or complete YAML.
- **"similar to task N": none.** The one deliberate cross-reference, T3's `Enrollment` type, names the
  sibling task **and** states the interim type (`Any`) so T3 is executable before T4 lands.
- **Two named forward dependencies, both discharged:** `propose_phonation_spans` and `PhonationSpan`
  are specified in T1 step 13 by their full rule and field list rather than by a code block, because
  the rule is three paragraphs of interval logic and a literal block would have been longer than the
  prose without being more precise. An implementer has everything needed: the inputs, the two limbs of
  the continuity test, the closing rule, the glide test, the production ladder, and the exact
  attribute names the entity writer reads from the returned dataclass. **This is the one place in the
  file where a step describes rather than shows, and it is flagged here so a reviewer can judge it.**

### Type-consistency scan

| type | fixed in | every reader agrees |
| --- | --- | --- |
| kind state strings | T2 (`PRESENT`/`ABSENT`/`UNCERTAIN`/`UNAVAILABLE`) | T3 compares the same literals; sibling T8 maps them to `KindState` |
| `TaxonomyResult.kinds` | `dict[str, str]` over `{"airway","speech","voice"}` | T3 reads `kind` **entities**, not this dict, so a runner that skipped TAXONOMY degrades to `uncertain` rather than `KeyError` |
| `branch_decision.will_run` | `bool` | T3 writes, `run.py` reads via `RoutingResult.runs`, sibling T8 reads the entity |
| branch names | `"AIRWAY"`, `"SPEECH"`, `"VOICE"` — uppercase, matching `run.GRAPH_ORDER` | T3's `BRANCH_FOR_KIND`, `run._BRANCHES`, sibling T8's `branches` product |
| node name `"routing"` | lowercase, as `routing.md`'s signature spells it | `NODE = "routing"`, `run.GRAPH_ORDER`, and sibling T8's `_GRAPH_ORDER` must use the same casing — **a mismatch here is the one cross-file type risk in this plan and is called out in T3's Interfaces** |
| window label set | `list[str]` on the entity, `dict[str, float]` for `scores` | T1 writes, T2 reads `windows_by_label` only; sibling T6 reads per-window `labels` |
| `phonation` span `duration_s` | `float`, seconds | T1 writes, T2 compares to `taxonomy.voice_min_duration_s` (seconds), sibling T5 reports `longest_span_s` (seconds) |
| consensus word | `dict` with the `fuse_consensus_words` keys, stored verbatim | T1 writes `words`; sibling T4 and T7 read the same list and the `word` entities' `index` to join |
| `Outcome` | still the three-member node-level enum | unchanged; the **file**-level axis becomes `Triage` in sibling T8, and no node returns a `Triage` |

One inconsistency found and fixed while writing: T2's `_phonation_spans` originally returned `[]` for
both "no pass ran" and "the pass found nothing", which would have made a missing derivative read as
absence — the exact failure `taxonomy.md` §States forbids. It now returns `None` for the first case,
distinguished by the presence of the `PREPROCESS`/`phonation_spans` activity, and
`test_no_phonation_pass_at_all_is_uncertain` pins the distinction.

### Second-pass results, after the review

**Spec coverage — the delta.** No task changed hands and no spec section became unassigned. Three
rows in the coverage table above now cover more than they did:

| spec section | change | task |
| --- | --- | --- |
| preprocess.md §Window classifications | AST's window is now a config key defaulting to 0.96 s, not a 10.24 s literal, and every windowed classifier passes its own vocabulary size so no window is silently ranked to its top five | T1 (C1, C2) |
| preprocess.md §`spans` | `merged_proposals` now has a producer — `propose_spans` counts what it absorbed — where the plan previously reported a rate nothing computed | T1 step 9b (C4) |
| preprocess.md §`phonation_spans` | the glide's offset criterion is assigned (`"monotonicity"`) and the assignment is total over five values, closing the one resolution hole | T1 |

**One spec line amended**, which the plan is otherwise not permitted to touch:
`preprocess.md`'s window table said "AST | 10.24 s (its native frame)". That figure is retracted in
this repository by measurement, so the row now names `windows.ast.win_length_s` (default 0.96 s) and
the Open-derivations row says why. **Flagged for the owner**: this is a spec edit, not a plan edit,
and it is the only one in this file.

**Placeholder scan — re-run.** `TBD`/`TODO`/`FIXME`/`XXX`: none (the two matches are in this scan's
own text). `...` still appears only in store-schema illustrations and, in one code block, marking
unchanged surrounding lines. The forward dependency the first pass flagged —
`propose_phonation_spans` described rather than shown — is unchanged and still flagged; it is now the
**only** such place in this file, because the two listings of PREPROCESS's `blocks` were merged into
one listing of named closures (M8) and the fixture contract was written out in full rather than
gestured at (I1).

**Type-consistency scan — re-run, with three new rows.**

| type | fixed in | every reader agrees |
| --- | --- | --- |
| `windows.ast.win_length_s` | `float`, seconds, default 0.96 | T1's config, T1's `_ast_scores`, T1's tests. **No `AST_FRAME_S` exists**, so a stale 10.24 cannot survive anywhere |
| `windows.ast.top_k` | `int` = 527, passed explicitly | T1's `_ast_scores`. YAMNet's 521 is `yamnet.top_k`, unchanged; HeAR keeps `top_k=None` because its path does not apply `or 5` |
| `Span.merged_proposals` | `int`, ≥ 1, defaulted to 1 on the dataclass | `propose_spans` (T1 step 9b) → the `span` entity's attributes (T1's `_spans`) → sibling T6's `merged_n`. **Zero is never valid**: a span is its own proposal |
| `seed_preprocess_store` | one signature, stated in T1's Interfaces | the only fixture any task may assume; every branch task's seeder is private to its own test file, so the three incompatible `words=` contracts the review found cannot recur |

**Two defects the second pass found and fixed**, beyond the reviewed list:

1. `windows_config`'s AST hop was 5.0 s, left over from the 10.24 s window. Against a 0.96 s window
   that is a hop five times the window — the classifier would have skipped four fifths of the file
   while the test asserted only that *a* hop reached it. It is now 0.48, the value the retraction
   measured at.
2. The `_scores`/`blocks` split named `_yamnet_scores` in one listing and inlined a lambda in the
   other. An implementer following the second would have written a `blocks` entry calling a function
   the first listing never defined. Both are now one listing of three named closures.
