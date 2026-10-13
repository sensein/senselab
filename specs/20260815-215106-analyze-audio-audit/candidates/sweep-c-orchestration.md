# Sweep C — Orchestration-layer candidates

Audited the 20 orchestration files (5,423 lines) that import a senselab task, read in full:
`compute.py`, `labelstudio.py`, `stages.py`, `adaptive/plot.py`, `background_mask.py`,
`embeddings.py`, `adaptive/fusion.py`, `harvesters.py`, `l1_plot.py`, `speech_presence.py`,
`quality.py`, `adaptive/backends.py`, `asr.py`, `stage_context.py`, `adaptive/evaluate.py`,
`sound_sources.py`, `perturbations.py`, `adaptive/audio_io.py`, `foreground.py`, `aggregate.py`.

Cross-checked against sweep A (`sweep-a-prose.md`) and sweep B (`sweep-b-computation.md`); no
finding below duplicates a B-n item. Two candidate mechanisms were investigated in depth via
background research agents and **ruled out** after tracing the actual runtime behavior (recorded
under "Checked and clean" rather than reported as findings, per the instruction not to invent a
failure mechanism):

- `StageContext._commit_sha_for` (stage_context.py) resolves a commit SHA for the cache key/
  provenance only; the actual model *load* goes through `model_for_task` → `HFModel`, which
  independently re-runs `resolve_revision(model_id, "main")` in its own pydantic validator. The two
  resolutions are **not data-connected** — they agree only because `resolve_revision` memoizes
  per-process/per-run and both call sites happen to use the same ref string (`"main"`). This is a
  latent fragility (already half-anticipated in `stage_context.py`'s own comment on
  `_DEFAULT_REVISION_REF`), but not a live defect: verified today's behavior is correct, so it is
  not reported as a finding.
- `adaptive/backends.py::_transcribe_crop_pipeline`'s `revision="main"` literal passed to
  `load_hf_resilient` looked like the exact "hardcoded ref beneath a resolved commit" trap the audit
  warns about. Traced `load_hf_resilient` (`utils/dependencies.py:699-737`): `revision`/`repo_id` are
  the function's *own* named parameters (input to resolution), not forwarded kwargs — it resolves the
  SHA internally and injects it into the actual `pipeline(...)` call via `kwargs.setdefault`. The
  literal `"main"` never reaches the HF loader. Not a defect.

### C-1
- kind: call-site-mismatch
- location: `compute.py:433` (`harvest_pass`)
- defect: `fuse_consensus_words(asr_resolved)` is called with no `policy=` argument, even though
  `harvest_pass` already has `speech_presence_policy` bound in scope (parameter at `compute.py:109`,
  used three lines later in the same function for the mask harvest, `compute.py:401`:
  `policy=speech_presence_policy`). `fuse_consensus_words`'s own signature and docstring
  (`asr.py:194-226`) exist precisely to thread the run's configured slot-alignment parameters
  through: `slot_overlap = float(getattr(policy, "asr_slot_overlap", 0.3)) if policy is not None else
  0.3` and the same for `asr_slot_mid_tol_s` (`0.15`). `default.yaml` declares both as
  run-configurable (`linking.asr_slot_overlap: 0.3`, `linking.asr_slot_mid_tol_s: 0.15`,
  `data/run_config/default.yaml:309-310`). A repo-wide grep for `fuse_consensus_words(` finds four
  call sites total: this one; `asr.py:287`'s internal fallback inside `_consensus_word_doubt`
  (zero production callers — its only caller is `asr_word_resampling_test.py:271`); `asr.py:466`'s
  internal fallback inside `harvest_asr_votes` (documented at asr.py:453-455 as "what a standalone
  caller wants" when no fold is supplied); and a direct call from
  `asr_word_resampling_test.py:298`. None of the four passes `policy`. Of the three non-test sites,
  only `compute.py:433` is reachable in a real run: `compute.py:447-452` always calls
  `harvest_asr_votes(..., fused=consensus_fold)`, so `asr.py:466`'s fallback branch is never taken
  in production, exactly like `asr.py:287`'s. So the parameter, the docstring's threading story,
  and the config keys are dead in every real run — not because there is only one call site, but
  because the one call site that runs (`compute.py:433`) is the one that drops `policy`, and the
  two call sites that would honor `policy` if reached (the fallbacks) are structurally unreachable
  given how the real caller always supplies `fused=`.
- failure: A user setting `linking.asr_slot_overlap: 0.5` (or any value other than the 0.3 default)
  in their run config sees no change in the published `asr` axis or in the `speaker` axis's
  word-location doubt — both are downstream of this one fold (`compute.py:424-452`: the same
  `consensus_fold` feeds `harvest_speaker_votes(..., fused_words=consensus_fold[0])` and
  `harvest_asr_votes(..., fused=consensus_fold)`). The fold silently always runs at 0.3/0.15,
  and the recorded provenance (`word_doubt_provenance["slot_overlap"]`) will report 0.3 regardless
  of what the config asked for — the exact "recorded value and used value cannot drift" guarantee
  the docstring claims is not actually being exercised, because the recorded value is always the
  hardcoded default.
- callers: `compute.py:433` (production path, live, drops `policy`); `asr.py:287`
  (`_consensus_word_doubt`'s fallback, no policy available there either, and no production caller —
  only exercised by `asr_word_resampling_test.py:271`); `asr.py:466` (`harvest_asr_votes`'s fallback,
  same policy-less shape, and also never reached in production since `compute.py:447-452` always
  passes `fused=consensus_fold`); `asr_word_resampling_test.py:298` (direct test call, no policy,
  not a production path). Four call sites of `fuse_consensus_words(` total; zero pass `policy`; one
  is live.

### C-2
- kind: model-in-control-flow
- location: `compute.py:890-1009` (`_speech_window_mask`), mirrored at `stages.py:763-806`
  (`_scene_source_mass`), `sound_sources.py:193` (`window_label_mass`/category harvesting), and
  `background_mask.py:534` (label-mass evidence for FR-033a targets)
- defect: Whether an embedding-clustering window counts as "speech" (and therefore participates in
  speaker-count/cluster estimation) is decided by a hardcoded backend-priority ladder keyed on the
  literal dict keys `"yamnet"` (`compute.py:920`) and `"ast"` (`compute.py:919`): "YAMNet is
  authoritative when available" (`compute.py:967`), falling back to AST only when YAMNet is absent
  (`compute.py:975`), falling back to openSMILE loudness only when both are absent. This is a
  *decision* (which scene classifier's verdict is trusted) gated on which specific named model ran,
  not on anything measured about the two classifiers' relative confidence in that window, and there
  is no config knob to change the trust order or add a third scene classifier without editing this
  function. The identical hardcoded two-classifier assumption recurs in three more places, all
  in-scope orchestration files: `stages.py:785`'s `for classifier in ("ast", "yamnet"):` loop in
  `_scene_source_mass` (background-mask's non-target-content evidence); `sound_sources.py:193`'s
  `for key in ("ast", "yamnet"):` loop that builds `per_classifier` for sound-source categorization;
  and `background_mask.py:534`'s `for key in ("ast", "yamnet"):` loop that gathers classifier
  windows for FR-033a label-mass evidence. Four call sites, not one, hardcode the same closed pair.
  All are structural: `PassPlan` (`stage_context.py:353-354`) only ever exposes
  `ast_model`/`yamnet_model` as named fields, so the whole pipeline supports exactly two
  interchangeable-in-name-only scene classifiers, unlike the ASR/diarization/embedding-model lists
  elsewhere in this same package, which take arbitrary model-id lists through config. Adding a third
  scene classifier (or dropping to one) means editing all four sites, not a config change.
- failure: On a window where AST's top-1 is `Speech` at high confidence and YAMNet's top-1 happens
  to be `Music` or `Singing` (a documented YAMNet confusion on child/sung voices — the function's own
  docstring names this exact tradeoff, `compute.py:906-913`, and prescribes tuning
  `speech_presence_labels` as the only available mitigation), the window is unconditionally marked
  non-speech and excluded from `cluster_pass_speakers`'s embedding-clustering input — regardless of
  AST's disagreement, and with no way to reweight or disable the YAMNet-first rule from a run config.
  This is an acknowledged, documented tradeoff rather than a hidden bug, but it is exactly the
  "adding/reordering a model requires editing control flow rather than config" pattern the
  model-pluggability sweep is scoped to surface.
- callers: N/A (both are terminal decision points, not helpers with a mismatched caller).

## Checked and clean

- **compute.py**: `harvest_pass`/`compute_uncertainty_axes` traced in full — one grid shared across
  every axis (D-24), `passes_for_axis` correctly restricts `IDENTITY_ONLY_AXES` to the identity pass,
  `SnrGate.build` shared by both the round-0 fold here and `fuse.write_final_uncertainty` so the two
  folds of one harvest cannot gate differently, `_signal_multiplicity` correctly maxes rather than
  sums across passes. Brouhaha is hardcoded as the sole VAD/SNR/C50 model with no config override,
  but senselab wraps no alternative backend with that exact triple output, so this is not reported as
  a pluggability gap (nothing to select between).
- **stages.py**: `run_pass`'s stage order is fixed by literal call sequence in one function body
  (diarization → scene → features → asr → alignment → background_mask/sources), so the "alignment
  after asr" and "mask/sources after scene+diarization" dependencies are enforced, not implicit.
  `stage_asr`'s `model_id.startswith("Qwen/Qwen3-ASR")` branch and `stage_alignment`'s
  `aligner: Literal["qwen","mms"]` dispatch are legitimate backend-specific parameter toggles per the
  audit's own stated exception (prefix dispatch to a backend-specific worker), not control-flow
  decisions gated on which model ran. `background_sources` is only ever invoked nested under
  `background_mask and variant == "unmodified"` — read as a real, intentional coupling (background
  source claims need the mask to know where they can be trusted), not an unenforced ordering gap.
- **stage_context.py**: `_commit_sha_for`'s three-way branch (non-Hub id / definitive not-found /
  propagate) traced and correct; see the ruled-out item above for why it does not create a load-time
  mismatch today. `STAGE_VERSIONS`/`stage_code_version` raise on an undeclared stage rather than
  defaulting silently. `PassPlan`'s `asr_language: str | None = None` fix (documented inline) verified
  against `stage_alignment`'s `language or "en"` resolution — consistent.
- **asr.py / harvesters.py**: `fuse_consensus_words`/`resample_word_doubt`/`resample_member_doubt`
  traced against their docstrings' worked examples; `aligned_columns`'s star-alignment fallback to
  time-overlap grouping verified; `_as_plain`'s ScriptLine/dict/duck-type normalization checked
  against both cache-read and live-object shapes. `harvest_asr_votes`'s `fused=` out-parameter is
  correctly the full `(words, provenance)` tuple, distinct from `harvest_speaker_votes`'s
  `fused_words=consensus_fold[0]` (just the word list) — verified both call sites in `compute.py` pass
  the right shape to each.
- **embeddings.py**: `cluster_pass_speakers`'s spectral→k-means fallback, `_sequential_calibration_band`
  preferred over `_empirical_calibration_band`, and the three calibration-band helpers' derivations
  checked against their own measured numbers; no hardcoded speaker-embedding model selection (models
  come from the caller's list).
- **speech_presence.py**: Every harvested signal traced to a single native-unit measurement with no
  threshold/inversion/ranking, matching the module's own stated contract; frame-posterior naming
  (`frame_brouhaha_vad`) confirmed structural (keyed on `frame_mean`, not a model name) per
  `compute.py`'s comment on the removed segmentation-3.0 frame voter.
- **quality.py**: `harvest_quality_measurements`/`quality_series` produce measurements only, no
  degradation scores (matches the module's stated L1/L2 split, cross-checked against sweep A/B's
  notes on the same boundary); Brouhaha hardcoding same as above, not reported for the same reason.
- **sound_sources.py**: `AUDIOSET_SCORE_FUNCTION = "sigmoid"` is a math-transform constant applied
  uniformly to both classifiers via `stage_scene`'s `function_to_apply` cache-keyed parameter, not a
  per-model branch; `window_label_mass`/`_window_category_masses` verified against their docstrings.
- **background_mask.py**: `_classify_bucket`/`build_mask`/`combine_target_evidence`/
  `margin_uncertainty` traced against the register's stated FR- numbers; `apply_span_evidence`'s
  raise-only, graded-by-coverage contribution verified against its own docstring's account of the
  saturating bug it replaced.
- **foreground.py / perturbations.py**: `suppress_foreground`'s default enhancement model id is a
  genuine overridable default (`model: str | None = None` parameter), not a hardcoded-and-unreachable
  one, matching the same default used in `perturbations.apply` and `adaptive/audio_io._enhance` — all
  three take an explicit override. `Perturbation.apply`'s exhaustive branch on the *declared* transform
  (not on `name`) verified against `TRANSFORMS`' registration-required construction.
- **aggregate.py**: `per_source_voice`/`speech_presence_p_voice` verified against their docstrings'
  weighting rules; absent-vs-zero discipline for `weights` and `native_confidence` checked.
- **labelstudio.py**: Track-attachment functions traced for the `axis_task`/`by_pass_task` join keys;
  `_scene_rows`'s L1/L2 join on rounded `(start, end)` keys checked against both producers' rounding
  (6 decimals, matching `compute.py`'s own bucket-key rounding convention throughout).
- **adaptive/backends.py / adaptive/audio_io.py**: Every function's `(result, reason)` failure
  envelope traced; `get_stream_wav`'s dispatch on the *declared* transform (not perturbation name) via
  `_declared_transform` verified to require no per-perturbation code edit, matching `perturbations.py`'s
  own design intent. `consensus_align`'s qwen/mms backend switch is a documented, configurable
  parameter (`backend: str = "qwen"`), not a hidden branch.
- **adaptive/fusion.py**: `build_final_outputs`'s `boundary_confidence` fabricated-0.5 fallback for
  the non-refined-identity path (`fusion.py:297,324`) is the same defect already reported as B-18's
  consumer — not re-reported here.
- **adaptive/evaluate.py**: `evaluate_against_ground_truth` reads only from `final/`, matching its own
  stated contract ("the evaluator scores the deliverable and nothing else"); the boundary-F1 and
  word-speaker-accuracy computations traced against the docstring's stated method.
- **adaptive/plot.py / l1_plot.py**: Both are read-only rendering sidecars over already-written
  artifacts; `_fused_axis`'s stale docstring line was already flagged as A-4/A-98 (not re-reported).
  No control-flow model selection, no producer/consumer contract crossed that isn't already covered by
  the axis-declaration checks (`AXIS_NAMES` completeness) noted in both files' own comments.
