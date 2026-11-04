# Triage as parts: stages, components and a workflow that is data

Status: draft for walkthrough (2026-11-04). Not agreed. Relates to `specs/20261022-triage-dataset-profile/plan.md`
(steps 6-11) and supersedes the shape of its step 11.

## Why this belongs to "moving B2AI out", not after it

Steps 1-5 moved B2AI's *data* out of senselab: its identities, its task list, its fitted values and its policy now
come from a dataset profile, and senselab ships a reference profile for adult clinical speech. What is still B2AI's
and still in senselab's code:

| What | Where it is today | Why it is B2AI's |
|---|---|---|
| The workflow's shape: which stages run, in what order, under what condition | hand-written calls in `run.py:490-565` and `drive_decisions` (`run.py:324-413`); `GRAPH_ORDER` only labels them | B2AI's protocol needs airway, voice and speech analysis, LLM review of free speech, a cohort comparison |
| The models | constants in `nodes/preprocess.py:156-161` (CrisperWhisper, Qwen3-ASR, Qwen3 aligner, AST, YAMNet, FRCRN), task-module constants (HeAR, PPGS); some in config (diarizer, enrollment, separator, LLM reviewer) | chosen and checked on adult B2AI recordings |
| What the verdict knows | `nodes/verdict.py:1429` reads about 25 store names from every stage and folds them with B2AI's release rules | the fold encodes which stages exist and what their absence means |
| The corpus passes' order | docstrings ("run replay afterwards", `extend_session_floor.py`; "run refold afterwards", `extend_cohort.py:33`) | B2AI's campaign practice |

The pediatric B2AI release shows why these matter: a different task list (no breathing tasks), children's voices
on which most adult models and cut-offs fail (speaker embeddings, recognizers, pitch ranges, DDK rates, classifier
thresholds), and likely no use for some stages. With today's code a pediatric run can change values and tasks but
not which recognizers run, which branches exist, or what the verdict requires. So the generalization the plan set out
to do is not finished until the workflow's shape and its components are declarable too. senselab `alpha` already
gained a child-speech recognizer (`drivendata/pasketti-word`, #631), which nothing in triage can select.

## What triage already has that helps (from the code, 2026-11-04)

- **A blackboard.** Stages do not call each other; each reads and writes named entities and measurements in one
  ProvStore. That is the natural seam for parts: a stage's contract is the names it reads and writes.
- **Rules as data.** The routing gates (`taxonomy.ruleset.gates`, each a feature path, an operator and a
  threshold), the decision reasons and their order (`decision_reasons.yaml`), the verdict gates (`verdict.gates`) and
  the task list are already data.
- **A plug-in shape in the branches.** `AlignMode` / `DetectMode` and `dispatch` (`nodes/branches.py:950-1022`).
- **A block list in PREPROCESS.** 40+ `(name, callable)` blocks (`nodes/preprocess.py:3167-3212`) with a de facto
  switch: a block whose config value is null records `absent`.
- **senselab's task layer is a component layer.** `transcribe_audios`, `classify_audios`, `diarize_audios`,
  `extract_speaker_embeddings_from_audios`, `enhance_audios`, `separate_audios`, `scan_for_pii` each take a model.

What works against it:
- **Settings arrive two ways.** `TriageConfig` is passed to most stages, but about 40 data-file loaders read a
  module-level active profile and run data (`dataset_profile._ACTIVE`, `_RUN_DATA`), so a stage cannot be called as a
  plain function with its settings; a few caches ignore the profile (`classifier_ontology`, `detectors`,
  `residue.FUNCTION_WORDS` at import).
- **Contracts are implicit.** Store names are scattered strings, some built with f-strings; nothing checks that a
  stage's inputs exist before it runs, except where it raises (`SPEECH` without `consensus_transcript`, `REDACT`
  without `pii_scan`).
- **PREPROCESS is a monolith**: most models and half the measurements live in one function with an internal
  `state` dict and an order-dependent block list.
- **The verdict is the integration point**: it re-derives which stages ran (`nodes/verdict.py:1384-1426`) and reads
  every stage's outputs by name.

Other senselab engines, for comparison: `audio_analysis` has `stage_*` functions with a `StageContext`, a `PassPlan`
where an absent model means "skip", a hashed run config, and one name-keyed registry (`adaptive/interventions.RULES`,
each rule with `enabled`). `utils/reproduce` runs whole analysis programs as sandboxed subprocess steps ordered by
`needs` (`dag.toposort`, `spec.StepSpec`): a process-level engine, not an in-process one.

## The three depths

### Depth 1: stages on and off

A profile or run names the stages it runs; values are required only for the stages that run; the verdict records a
stage that was off as "not run", distinct from failed.

```yaml
workflow:
  stages: {AIRWAY: off, COHORT: off, SECOND_OPINION: off}
```

Needs: each stage's dependencies written down (REDACT needs SPEECH's transcript and PII scan; REVIEW needs REDACT;
the release axis needs SPEECH), so turning a stage off either turns its dependents off or is refused; each profile
value tagged with the stage that reads it (`data/profile_values.yaml` gains `stage:`); a verdict policy for "not run"
(e.g. REDACT off means no audio is released, not that it is released unmasked).

### Depth 2: components chosen by name

Within a stage, a slot names its implementation; a registry maps a name to a senselab task, a model, a pinned
revision and its parameters. A reference profile picks a set; a pediatric reference profile picks another.

```yaml
components:
  asr: [crisperwhisper_turbo, qwen3_asr_1_7b]     # adult reference
  aligner: qwen3_forced_aligner
  sound_events: [yamnet, hear, ast]
  enhancer: frcrn_se_16k
  speaker_embedding: ecapa_voxceleb
  diarizer: pyannote_community_1
# pediatric reference: asr: [pasketti_word, ...], speaker_embedding: <a child-speech model>, ...
```

Needs: a slot interface (what the component reads, what it writes: e.g. an `asr` component writes one
`asr_<name>` measurement in the consensus format); the PREPROCESS blocks split into components with declared
inputs; the model constants moved behind names; the consensus and arbitration steps reading "the ASR components
that ran" rather than two fixed names.

### Depth 3: the workflow is data

A workflow spec lists its stages by registered name, each stage's components and parameters, what it needs and
when it runs; the runner is generic; triage, pediatric triage and `audio_analysis` are each one spec.

```yaml
workflow: triage
version: 1
stages:
  - {name: ADMIT}
  - {name: PREPROCESS, components: {asr: [crisperwhisper_turbo, qwen3_asr_1_7b], enhancer: frcrn_se_16k}}
  - {name: BACKGROUND, needs: [PREPROCESS]}
  - {name: routing, needs: [TAXONOMY], rules: taxonomy.ruleset}
  - {name: SPEECH, needs: [routing], when: {routed: SPEECH}}
  - {name: REDACT, needs: [SPEECH], when: {found: pii}}
  - {name: VERDICT, needs: all, rules: decision_reasons}
corpus:                       # the corpus passes, ordered instead of documented
  - {name: session_floor, then: replay}
  - {name: cohort, then: refold}
```

Needs everything depth 1 and 2 need, plus: a generic in-process DAG runner; every stage callable with one context
argument; the verdict as a fold over the contributions of the stages that ran; and, for the corpus passes, either
the same runner or `utils/reproduce` (whose step granularity, one program per step with `needs`, matches the
`extend_*` scripts well).

## Two ways to build toward depth 3

**(A) Parts first, composed in code.** Write each stage's contract down as a `Stage` descriptor, register the stages,
and make the runner iterate the registry instead of hand-written calls. Settings arrive in one explicit context.
`run_triage` becomes one composition; a pediatric composition is another. The spec format of depth 3 is then a
serialization of the registry, designed once there are two or three real workflows to test it against.

**(B) The spec first.** Design the workflow YAML now and write today's triage as its first spec, with the runner built
to read it. Reaches depth 3 directly, but fixes the format against one workflow, and the format would be designed
before the contracts that it serializes exist.

**Recommendation: (A), with the registry designed as the in-memory form of the spec.** A stage descriptor:

```python
@dataclass(frozen=True)
class Stage:
    name: str                          # "BACKGROUND"
    run: Callable[[StageContext], StageResult]
    needs: tuple[str, ...]             # stages whose outputs it requires
    reads: tuple[StoreKey, ...]        # store names it reads (required / optional)
    writes: tuple[StoreKey, ...]
    settings: tuple[str, ...]          # config sections and data files it reads
    slots: tuple[str, ...] = ()        # component slots it fills (asr, enhancer, ...)
    optional: bool = True              # may a workflow turn it off
    when: Condition | None = None      # e.g. REDACT: SPEECH routed and PII found

@dataclass(frozen=True)
class StageContext:
    store: ProvStore
    run_dir: Path
    config: TriageConfig               # merged config, the profile chain and the run's data section bound in
    hint: AudioHints | None
    components: ComponentSet           # the slot -> component resolution for this run
```

A spec loader later builds the same descriptors from YAML, so depth 3 adds a format, not a second architecture.

## Increments, each checked on B2AI as steps 4 and 5 were

| # | What | B2AI check | Unlocks |
|---|---|---|---|
| M1 | **Contracts and an explicit context.** `Stage` descriptors for the 15 stages and REPORT; the runner iterates them; store names as constants per stage; data-file values reached through the context instead of the module globals (the globals go; no compatibility kept, as agreed for B2AI); a check that a stage's required inputs exist before it runs. | identical decisions and stores | everything below; stages callable as plain functions |
| M2 | **Components.** A component registry; the PREPROCESS blocks split along slots (asr, aligner, sound_events, enhancer, diarizer, speaker_embedding, separator, pii_detectors, llm_reviewer, ppg); every model constant behind a name; the reference profile names B2AI's set. | identical | a pediatric model set |
| M3 | **Stages on and off.** The `workflow` section; required values per stage; dependents follow their stage; the verdict folds over the stages that ran, each stage contributing its own evidence and grounds, with a policy for "not run". | identical with every stage on | a pediatric workflow without airway |
| M4 | **The pediatric case as acceptance.** A pediatric reference profile (components and the values that can be named without a fit) and a small pediatric run end to end; what cannot be fitted yet is listed, not guessed. | B2AI adult unchanged | proof that the parts generalize |
| M5 | **Depth 3.** The spec format, written against triage adult, triage pediatric and `audio_analysis`; the corpus passes as a declared corpus workflow (own runner or `utils/reproduce`). | identical | senselab-wide workflows |

The plan's remaining steps move around them: step 6 (PII policy, task guidance and second opinion as caller-supplied
inputs) becomes the settings and components of REDACT, REVIEW and SECOND_OPINION in M1-M2; step 9 (target speaker,
recording id, corpus-pass inputs on the replay marker) fits M1's contracts and M5's corpus workflow; step 7 (hand-over
to b2aiprep) waits for M3, since the profile gains `workflow` and `components` sections; step 10 (fitting from a pilot
run) is independent and can come any time after M1; step 11 is replaced by M1-M4.

## Hard parts, named now

1. **The verdict.** It is a fold written against every stage. Making it a fold over contributions means each stage
   declaring the evidence and grounds it adds, and a written policy for a stage that did not run. This is the largest
   single piece and the one most likely to move B2AI's outputs if done carelessly; its check is the 192-run replay.
2. **PREPROCESS.** Its blocks share an internal state dict and depend on their order. Splitting it into components
   with declared inputs is a refactor of its own, and the measurements it writes are what every later stage reads.
3. **The consensus transcript.** It is built from exactly two named recognizers today; with an `asr` slot it must be
   built from however many ran, and a pediatric recognizer set may not include either.
4. **Corpus passes.** They write into stores that per-recording stages then reread, and their order lives in prose.
5. **Settings without globals.** About 40 loaders and their caches change signature; caches keyed by the context.

## Not in scope

A new branch or a new matcher (`Pattern` stays a closed set); fitting pediatric values (step 10 and later work);
changing what triage measures or decides for B2AI.

## Questions for the walkthrough

1. (A) parts first with the registry as the in-memory spec, or (B) the spec first?
2. Settings through an explicit context, removing the module globals?
3. The component slots: which ten, and which PREPROCESS blocks each covers?
4. The verdict as a fold over stage contributions: what "not run" means for each stage, especially REDACT and the
   release axis.
5. Corpus passes: their own runner, or `utils/reproduce`?
6. `audio_analysis`: brought onto the same contracts in M5, or earlier?
7. The pediatric acceptance case in M4: which recordings, and which stages and components it must run.

## Decisions (walkthrough)

1. **(A) parts first** (user, 2026-11-04): stages become descriptors in a registry that the runner iterates, the
   registry designed as the in-memory form of the eventual workflow spec; the spec format comes in M5, written
   against adult triage, pediatric triage and `audio_analysis`.
2. **Settings through an explicit context** (user, 2026-11-04): every stage receives one context carrying its
   config resolved through the profile chain and the run's data section, and the run's components; loaders read
   from it; the module-level active profile and run data are removed, with no compatibility layer.
3. **Scope narrowed** (user, 2026-11-04): plan step 11 does only what moving B2AI out needs, without reinventing
   a workflow engine (pydra does that) and without reworking how stages are written:
   - the model choices that are still constants in code become named settings (the speech recognizers, the
     sound-event classifier AST, the enhancer);
   - a `workflow.stages` setting lists the stages that run; a stage left off is recorded skipped, noted as off in this
     workflow, and a stage that needs it is skipped too;
   - the runner drives a small table of stages (name, call, what it needs, whether it may be off) instead of
     hand-written calls;
   - each profile value names the stage or stages that read it, and is required only when one of them runs.

   Deferred to a separate modular-design effort that starts from this note: stage descriptors and the workflow spec
   format (M1, M5), sub-workflows, corpus passes as workflows (M3), pydra, PREPROCESS as tasks, the verdict as a
   fold over stage contributions, and **decision 2 (the explicit settings context), which is withdrawn for step 11:
   loaders keep reading the active profile.**
