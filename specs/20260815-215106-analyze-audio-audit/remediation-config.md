# Remediation: unfitted thresholds → config

Companion to `register.md`. That audit named 7 `unfitted-threshold` findings (F-144, F-145,
F-149, F-139, F-140, F-143, F-157) among its 46 gated defects. The directive here is broader:
**every** unfitted numeric literal that gates a verdict, classification, mask, or report-inclusion
decision anywhere in `audio_analysis` moves to config, named for what it decides. This document
is the inventory and the design; it changes no code.

**Method.** Four parallel sweeps covered the package's ~95 files (`adaptive/`, the
fuse/speaker/compute core, the quality/acoustic/global-summary group, and the remaining
infra/plumbing files), independently of `register.md`, then cross-checked against it. A fifth,
targeted pass (below) chased one lead the sweeps surfaced but didn't fully chase down: whether
`RunConfig`'s `*_policy` `Mapping` fields are actually read anywhere. They mostly aren't.

## Counts

- **Unfitted decision-gating parameters found: 60**, counting each ramp anchor / clamp bound /
  named constant separately (a 2-anchor ramp is 2 rows because either anchor alone changes the
  verdict), and the 8-member `mask.*` family (2.4) as 8. Of these, **7 are the register's own
  named `unfitted-threshold` set**. Of the remaining 53: **7 are `register-adjacent`** — the same
  literal appears in `register.md` but filed under a different defect class (`misnamed-statistic`,
  `adult-speech-assumption`) that never called it out as *unfitted*, so its magnitude was never
  scrutinized; **46 are entirely new** — no finding id anywhere in `register.md`'s 176 names this
  literal.
- **Dead config keys found: 14.** The register named 2 (`linking.asr_slot_overlap` /
  `asr_slot_mid_tol_s` counted as one pair behind F-162, and `speaker_count.multimodal_threshold`
  behind F-144). **12 are new**, and they are not scattered — 9 of the 12 share one root cause:
  four whole `RunConfig` fields (`rounds_policy`, `quality_policy`, `labelstudio_policy`,
  `support_policy`) plus two orphaned keys inside `speaker_policy` are built by `_build()` in
  `run_config.py` and then **never read again by anything** — not by name, not through `.raw`.
  Confirmed by grepping the whole tree for each field name outside `run_config.py` itself.

---

## Part 1 — Dead config keys (config exists, nothing consumes it)

Worse than a hardcoded literal per the task brief, because it advertises control that isn't there.
Verified by grepping every consumer candidate (direct field access, `.raw[...]` fallback, and the
constant each section is supposed to override) outside `run_config.py`.

| # | key | `RunConfig` field | what it was supposed to override | actual consumer | status |
|---|---|---|---|---|---|
| D1 | `linking.asr_slot_overlap` / `asr_slot_mid_tol_s` | `linking["asr_slot_overlap"]` etc. | `asr.fuse_consensus_words`'s word-slot join | `compute.py:433`'s `harvest_pass` calls `fuse_consensus_words(asr_resolved)` with no `policy=` — the only production call site | **register-known (F-162)** |
| D2 | `speaker_count.multimodal_threshold: 0.15` | not parsed at all — `run_config.py` has no `speaker_count` section reader | `speaker_identity.speaker_count_posterior`'s multimodal test | `speaker_identity.py:121` `_SUPPORTED_THRESHOLD`-style default, never receives the YAML value from any call site including `scripts/analyze_audio.py:1241` | **register-known (F-144)** |
| D3 | `speaker.supported_threshold: 0.15`→0.5 (yaml says 0.5) | `speaker_policy["supported_threshold"]` | `speaker_identity._SUPPORTED_THRESHOLD` (`speaker_identity.py:60`), used at `:274` for `has_supported_evidence` | nothing — `RunConfig.speaker_policy` is built (`run_config.py:481`) and never read anywhere else in the tree | new |
| D4 | `speaker.centroid_min_similarity: 0.5` | `speaker_policy["centroid_min_similarity"]` | H2/D-6's centroid matcher in label harmonisation | nothing — `harmonize.py:45`'s own `MIN_CENTROID_SIMILARITY=0.5` is never read either; the real call site (`speaker.py:240`) passes `min_similarity=cluster_cosine_threshold` instead, so a *different* config value silently substitutes for this one | new |
| D5 | `quality.floor_percentile: 10.0` | `quality_policy["floor_percentile"]` | `acoustic.level_above_floor_track`'s percentile choice | nothing — `speech_presence.py:244` calls `level_above_floor_track(...)` with no override; `acoustic.py:116`'s own `FLOOR_PERCENTILE=10.0` constant is what runs | new |
| D6 | `quality.max_gain_cap_db: 10.0` | `quality_policy["max_gain_cap_db"]` | scene-quality gain-normalisation cap | nothing found anywhere in the tree | new |
| D7 | `support.min_evidence_spread: 0.15` | `support_policy["min_evidence_spread"]` | `support.informative_evidence`'s admission gate | nothing — both production call sites (`compute.py:728`, `adaptive/corroboration.py:60`) invoke it with zero keyword overrides; `support.py:270`'s `MIN_EVIDENCE_SPREAD` constant is what runs | new |
| D8 | `support.evidence_low_threshold: 0.20` | `support_policy["evidence_low_threshold"]` | same function, "low evidence" classification | same as D7 — `support.py:273`'s `EVIDENCE_LOW_THRESHOLD` constant runs instead | new |
| D9 | `support.min_low_fraction: 0.02` | `support_policy["min_low_fraction"]` | same function, pool-admission fraction | same as D7 — `support.py:276`'s `MIN_LOW_FRACTION` constant runs instead. **Compounds F-143**: not only is 0.02 unfitted, the YAML slot meant to let someone override it while re-deriving does nothing | new (compounds register-known F-143) |
| D10 | `labelstudio.low_threshold: 0.33` | `labelstudio_policy["low_threshold"]` | `labelstudio.LOW_THRESHOLD` binning | nothing — module constant is bound at import time, `RunConfig.labelstudio_policy` never read | new |
| D11 | `labelstudio.high_threshold: 0.66` | `labelstudio_policy["high_threshold"]` | `labelstudio.HIGH_THRESHOLD`, and (via import) `disagreements.py`'s inclusion gate for `disagreements.json` | nothing — same as D10, and `disagreements.py:18,95` imports the *Python constant* directly, one hop further from any config path | new |
| D12 | `rounds.epistemic_tolerance: 0.001` | `rounds_policy["epistemic_tolerance"]` | round-to-round convergence tolerance | nothing — `rounds.py` has its own `EPISTEMIC_TOLERANCE=1e-3` (same value, coincidentally) never sourced from `rounds_policy` | new |
| D13 | `rounds.cycle_window: 4` | `rounds_policy["cycle_window"]` | oscillation-detection window | nothing — `rounds.py:255`'s `cycle_window: int = DEFAULT_CYCLE_WINDOW` default is what every caller uses; no caller passes `cfg.rounds_policy["cycle_window"]` | new |
| D14 | *(structural)* `RunConfig.rounds_policy`, `.quality_policy`, `.labelstudio_policy`, `.support_policy` | — | — | grepped for each field name (`\.rounds_policy`, `\.quality_policy`, `\.labelstudio_policy`, `\.support_policy`) outside `run_config.py`: **zero hits**. These four dataclass fields exist only to be assigned, never to be read. | new — root cause of D5–D13 |

`rounds.max_rounds` is the one exception in the `rounds:` section — it has its own scalar
`RunConfig.max_rounds` field (not routed through `rounds_policy`) and **is** genuinely threaded to
`fuse.py`. That split (one live scalar, two dead map entries, in the same YAML block) is what made
D12/D13 easy to miss by inspection alone.

---

## Part 2 — Unfitted decision-gating parameters

Grouped by area. Columns: location, literal, what it gates, derivation, config reachability.
"Register" column: the finding id if any tag names this literal, `adjacent` if a register finding
discusses the same code but under a different defect class, `—` if new.

### 2.1 Register's own 7 (confirmed present, included for completeness)

| location | literal | gates | derivation | config? | register |
|---|---|---|---|---|---|
| `speaker_identity.py:121` | `multimodal_threshold=0.15` | speaker-count posterior "multimodal" verdict | none | dead (D2) | F-144 |
| `speaker_identity.py:60` | `_SUPPORTED_THRESHOLD=0.5` | `has_supported_evidence` | none | dead (D3) | F-145 |
| `global_summary.py:209` | `ramp(pesq, low=2.0, high=3.5)` | `quality.uncertainty` PESQ term | claimed "literature-derived," uncited, contradicts own docstring | none | F-149 |
| `global_summary.py:210` | `ramp(stoi, low=0.5, high=0.85)` | `quality.uncertainty` STOI term | same claim, same problem | none | F-149 |
| `global_summary.py:211` | `ramp(sisdr, low=0.0, high=15.0)` | `quality.uncertainty` SI-SDR term | same claim; docstring itself says "below 5 dB poor" vs. code's 0.0 anchor | none | F-149 |
| `fuse.py:559` | `settled_below=0.35` (`derive_mask_from_axes`) | bucket → `target_free` mask region | none | none | F-139 |
| `fuse.py:831` | `unsettled_above=0.6` (`fuse_axes`) | bucket offered to `remeasure` (D-10), counted for C4 convergence | none | none | F-140 |
| `support.py:276` | `MIN_LOW_FRACTION=0.02` | evidence-pool admission | measured under a since-disowned bug | dead (D9) | F-143 |
| `interventions.py:939` | `_p2_trigger`: `mean_instability > 0.0` | whether P2 fine-posterior re-analysis fires | none; contradicts own docstring ("high value" implied, code checks `>0`) | live via `ctx["policy"]`, but the comparator itself (`>0.0`) has no threshold slot to move | F-157 |

### 2.2 Adaptive loop (`adaptive/`) — new

| location | literal | gates | derivation | config? | register |
|---|---|---|---|---|---|
| `adaptive/policy.py:14` | `_COST_WEIGHT={"light":1.0,"medium":4.0,"heavy":16.0}` | denominator of `priority=gain/_COST_WEIGHT[cost]`, ranks every intervention candidate for budget admission | none | none | adjacent (F-158, filed as misnamed-statistic about the numerator; the weight ladder itself is untouched by that finding) |
| `adaptive/convergence.py:149` | `window: int=3` (`build_convergence_report`) | oscillation/stagnation window feeding `termination_reason` in `final/convergence.json` | docstring gives a rationale (loop rounds are expensive) not a fit | none — sole caller in `loop.py` never overrides | — |
| `adaptive/evaluate.py:99` | `pred = float(pv) >= 0.5` | binarizes `speech_presence_confidence` for `eval.json` precision/recall | none | none | — |
| `adaptive/evaluate.py:157` | `tol=0.25` (boundary match) | whether a predicted diarization boundary "matches" ground truth → `boundary_f1` | none | none | — |
| `adaptive/fusion.py:128` | `gap <= 0.5` | whether two same-speaker word spans merge into one rollup segment | none | none | — |
| `adaptive/fusion.py:297` | fallback `{"start":0.5,"end":0.5}` | `boundary_confidence` written into `final/diarization.json` when I2 didn't compute one | none | none | adjacent (F-156 covers the *same pattern* at `identity_repair.py:227-231`; this is an independent literal at a second call site that can drift from it) |
| `adaptive/fusion.py:324` | fallback `{"start":0.5,"end":0.5}` (second code path) | same as above, no-refined-identity branch | none | none | adjacent (F-156, third instance) |
| `adaptive/fusion.py:310` | `pv < 0.5: continue` | whether a bucket is emitted as a diarization segment at all | none | none | — |
| `adaptive/fusion.py:349` | `overlap_posterior >= 0.5` | `seg["overlap"]` boolean published per segment | none | none | — |
| `adaptive/identity_repair.py:219` | `coassoc[i,j] >= 0.5` | cross-model consensus merge for I2 re-clustering | none | none, and independent of the configured `recluster_cosine_threshold` that gates the per-model step | — |
| `adaptive/interventions.py:250` | `raw_pres < 0.2` | S1's "revert to raw" transform-artifact guard | comment explains the guard exists, not the magnitude | none | — |
| `adaptive/interventions.py:740` | `len(windows) >= 4` | whether stored window-embeddings are usable for I1/I2 at all | none | none | — |
| `adaptive/interventions.py:1004` | `2.0` scale factor (`frame_dispersion`) | dispersion value that feeds `_p2_trigger`'s `mean_instability` next round | none | none | — |
| `adaptive/ls_final.py:27` | `_CONF_BINS=(("high",0.66),("medium",0.33),("low",0.0))` | LS-export per-word confidence-bin label | none in this file; matches `theta_high`/`theta_low`'s comment "matches labelstudio HIGH/LOW" | dead — third independent hardcoding of 0.66/0.33 (after `labelstudio.py` and `adaptive.thresholds`), not read from `ctx["policy"]` | — |
| `adaptive/triage.py:31` | `aggregate_win_s=0.1` | frame-posterior aggregation window feeding `speech_present`/`needs_enhancement` | none | none — sole call site never overrides | arguable (window-shaped, but see note below) |
| `adaptive/triage.py:125` | `nonspeech_threshold=0.35` (`dsp_snr_series`) | which frames count "non-speech" for the DSP-fallback noise floor, feeding `needs_enhancement` | none | none — call site never overrides | — |

Note on `triage.py:31`: listed because it feeds two further gated decisions (`speech_present`,
`needs_enhancement`), even though a window length is normally operational; the distinction is
arguable, flagged per the task's own instruction rather than silently dropped.

### 2.3 Speaker / embeddings / clustering / consensus — new

| location | literal | gates | derivation | config? | register |
|---|---|---|---|---|---|
| `fuse.py:825` | `tolerance=1e-3` (`fuse_rounds`) | whether a round counts as "no change" (C1 convergence criterion) | none | none | — |
| `embeddings.py:100` | `n_clusters_max=6` | hard cap on discoverable speaker count | none | none | — |
| `embeddings.py:101` | `min_windows_for_clustering=4` | whether the multi-cluster sweep runs at all vs. defaulting to 1 speaker | none | none | — |
| `embeddings.py:102` | `coherent_silhouette_threshold=0.10` | single- vs. multi-speaker classification of a pass | none — same unfitted-silhouette-as-probability pattern the register flags elsewhere for a different value | none | adjacent (same defect class as F-141's silhouette misuse, different site) |
| `embeddings.py:270-271` | `min_cluster_fraction=0.10` | rejects a candidate k-way partition as having an unrealistic cluster; changes final `n_speakers` | worked example, not fitted | none | — |
| `embeddings.py:324` | `merge_threshold=0.55` | whether two sub-clusters merge into one speaker | literature rationale (ECAPA ranges), explicitly not fit to this pipeline | none | adjacent (F-164 discusses this literal under adult-speech-assumption; not previously flagged as simply *unfitted*) |
| `embeddings.py:466,529,585` | `min_pairs=5` (3 functions) | whether a per-pass empirical calibration band is trusted vs. falling back to config `same_floor`/`diff_floor` | none | none | — |
| `embeddings.py:518-520,574-576` | clamp `[0.05,0.95]`, min gap `0.05` | whether a measured calibration band is accepted or discarded | none | none | — |
| `embeddings.py:584` | `fallback_diff_floor=0.70`, `same_floor+0.20` push | diff-floor used for single-cluster passes | none | none — independently duplicates config's `diff_floor` | — |
| `clustering.py:90` | `cosine_threshold=0.5` (`assign_unified_clusters_with_seed_phase`) | whether an `other_items` label snaps to a seed cluster (feeds timeline colouring/labels) | rationale comment, not fit | none — `plot.py:171`'s caller has its own hardcoded `0.5`, independent of `speaker.cluster_cosine_threshold` | — |
| `clustering.py:91` | `cross_group_threshold=0.75` | cross-pass seed-group merge (e.g. raw-Peter ↔ enhanced-Peter) | rationale comment, not fit | none — no config key exists, no caller overrides | — |
| `sources.py:174` | `tolerance=0.05` (`matches_floor_signature`) | rejects a window outright as "classifier floor-response" (highest-priority screen) | none | none | — |
| `sources.py:371,375` | `max_confident_uncertainty=0.3` (`discount_for_mask_uncertainty`) | tier downgrade for a source finding in an uncertain mask region | none | none, **and function has no production caller** (test-only) | — |
| `votes.py:146-148` | `mask_from_pvoice` 0.5 breakpoint | weight assigned to a query bucket | none | none, **and function is unused** (no caller anywhere) | — |
| `compute.py:960` | `len(vals) >= 100` | whether openSMILE-loudness fallback (3rd tier of speech-window mask) computes at all | comment gives approximate justification | none | — |

### 2.4 Background-mask calibration (`data/detection_margin/…json`) — new, one family

The 8 `mask.*` constants below live in the *right place* (a `data/` profile, resolved through
`calibration.py`) but, unlike that same profile's `margins_db`/`level`/`noise_floor` blocks (each
with a `human_basis`/`machine_basis` derivation `calibration.validate_detection_margin_profile`
actually enforces), **the `mask` block carries no derivation requirement at all** — the validator
never checks it the way it checks the other three.

| location | literal | gates |
|---|---|---|
| `calibration.py`/`background_mask.py:293` | `target_active_confidence=0.6` | bucket → `target_active` |
| `calibration.py`/`background_mask.py:294` | `target_free_confidence=0.2` | bucket → `target_free` |
| `calibration.py`/`background_mask.py:295` | `max_free_uncertainty=0.5` | `target_free` vs. `indeterminate` |
| `calibration.py`/`background_mask.py:296` | `negligible_fraction=0.05` | whole-mask `negligible_fraction: True` verdict |
| `calibration.py`/`background_mask.py:297` | `nontarget_active_confidence=0.5` | `nontarget_active` (the "worth introspecting" verdict) |
| `calibration.py`/`background_mask.py:290` | `guard_interval_s=0.5` | reverberant-tail trim after target activity |
| `calibration.py`/`background_mask.py:291` | `min_region_s=1.0` | region long enough to host an unpadded decision |
| `calibration.py`/`background_mask.py:292` | `max_padding_fraction=0.5` | how much of a short region padding may consume |

Reachable via the profile (so not "dead config" in the D1–D14 sense), but undocumented in the
sense CLAUDE.md itself requires ("a written derivation... regenerate from measured verdicts").
Also triple-duplicated as Python-side fallback literals in `background_mask.py`, `calibration.py`,
and `stages.py:545-546` — three independent copies of the same 8 numbers, any one of which could
drift from the shipped profile without the others noticing.

### 2.5 Quality / acoustic / global summary — new

| location | literal | gates | derivation | config? | register |
|---|---|---|---|---|---|
| `acoustic.py:41` | `SILENCE_LUFS=-60.0` | anchor of `loudness_confidence` ramp | qualitative only | none, **and function has no production caller** | — |
| `acoustic.py:45` | `SPEECH_LUFS=-20.0` | other anchor, same ramp | qualitative only | none; also **disagrees** with `linking.lufs_speech=-30.0`, the config value used by the module that actually runs (`speech_presence_link.py`) — two hardcoded "speech loudness" anchors, -20 and -30, for what should be one decision | — |
| `acoustic.py:116` | `FLOOR_PERCENTILE=10.0` | percentile taken as noise floor | standard-convention percentile | dead (D5) | — |
| `acoustic.py:119` | `_FLOOR_BIAS_DB=9.8` | bias correction on the floor estimate | has a real derivation elsewhere (`quantile_bias_correction_db`), but hardcoded as a second, independently-drifting copy here instead of calling that function | none | — |
| `degradation.py:39` | `snr_floor_db=5.0` (1-anchor, `DEFAULT_ANCHORS`) | SNR degradation ramp's "fully degraded" end | none named (register's F-169 discusses the 0-anchor `snr_clean_db=25.0` only) | overridable via a fitted `calibration` profile; this literal is what runs absent one | adjacent (F-169 names the sibling anchor) |
| `degradation.py:42` | `c50_floor_db=-5.0` (1-anchor) | reverb ramp's "fully degraded" end | none named (F-169 names `c50_clean_db=30.0` only) | same as above | adjacent (F-169) |
| `level.py:209` | `clipped_fraction(threshold=0.9999)` | what counts as a "clipped" sample → `quality_clip` score | none | none | — |
| `global_summary.py:114,142` | `no_speech_threshold=0.5` | ASR-chunk "likely hallucination" classification feeding `transcript_accuracy_uncertainty` | none ("high" asserted) | none — **independent hardcoded duplicate** of `linking.no_speech_threshold=0.5`, which feeds a different module (`speech_presence_link.py`) entirely; the two could drift and nothing would notice | — |
| `floors.py:13` | `MIN_EVIDENCE_WEIGHT=0.05` | floor under every withdrawn evidence weight — shapes every fused axis's uncertainty | qualitative argument only, magnitude (0.05 vs 0.02/0.1) not derived | none for its 7 real consumers (see Part 3, threading problem #1); a *separate*, hand-synced copy exists at `adaptive.adjudication.min_evidence_weight` for one unrelated rule | — |
| `aggregate.py:96` | `p_voter=0.1` (hallucinated-vote substitute) | vote value substituted for any voter flagged hallucinated, feeding `speech_presence_p_voice` | none ("vote against voice" asserted, magnitude not) | none | — |
| `disagreements.py:95` / `labelstudio.py:82-83` | `HIGH_THRESHOLD=0.66` | whether a bucket enters `disagreements.json`'s high-uncertainty index at all | none in `labelstudio.py`; matches `labelstudio.high_threshold`/`adaptive.thresholds.theta_high` | dead (D11) | — |

### 2.6 PII — new, clean case

| location | literal | gates | derivation | config? |
|---|---|---|---|---|
| `pii.py:82` | `presidio_score_threshold=0.4` | Presidio spans below this never reach the PII verdict | docstring gives an engineering guess ("permissive enough...") | **no `pii:` section exists in `default.yaml` at all** |
| `pii.py:85` | `gliner_threshold=0.5` | same, for GLiNER spans | same | same |
| `pii.py:241-243` | `count >= 2` (cross-model corroboration) | the `contains_pii` verdict itself, written into every `pii.json` | rationale given (hallucination filtering) for *having* a count gate, not for why 2 | same |

`detect_pii_in_pass` (the sole call site, `scripts/analyze_audio.py:858`) passes no overrides —
these three defaults are what every run uses, and there is no path to change any of them short of
editing Python.

### Arguable / excluded across all four sweeps

Buffer/window/hop sizes chosen for memory or DSP correctness (STFT/COLA windows, `level.py`'s
BS.1770/EBU-Tech-3342-mandated oversampling and percentile gates, `noise_floor.py`'s convergence
iteration tolerances), retry/iteration counts, plot-only cosmetics (colours, alpha blending,
`plot.py`'s speaker-cluster colouring, `l1_plot.py`'s display floors), report-length knobs
(`summary.py`'s `top_n=5`), numerical-stability epsilons (`1e-9`/`1e-12` norm floors and tie
breaks), and `stages.py:545-546`'s mask-threshold fallback (a duplicate of an already-calibrated
profile value, not a fresh guess — flagged in 2.4 instead as a drift risk, not counted as its own
finding). `rounds.py`'s `DEFAULT_MAX_ROUNDS=10` disagreeing with `default.yaml`'s `max_rounds: 3`
is noted as a drift risk but not a live decision gate (the real value is threaded via the separate
`RunConfig.max_rounds` scalar).

---

## Part 3 — Config design

### 3.1 Naming convention

Every new key names **what crossing it decides**, not the module or variable it happened to live
in — `mask.target_free_doubt_ceiling`, not `settled_below`; `speaker_count.multimodal_posterior_floor`,
not `multimodal_threshold` (kept close to the original only where the original name already says
the decision, e.g. `presence.supported_evidence_floor`). Sections group by *decision domain*
(mask, speaker identity, speaker clustering, quality ramps, PII, adaptive-loop internals), mirroring
`linking:`'s existing "grouped by the question they answer" convention, not by source file.

### 3.2 Proposed YAML — the register's 7, plus the highest-value new ones

```yaml
# ── mask: when a bucket counts as target-active / target-free / worth flagging ──
# Moved out of fuse.py's bare defaults. `settled_below` and `unsettled_above` used to be
# undocumented function parameters with no caller override anywhere in the tree.
mask:
  target_free_doubt_ceiling: 0.35
    # was fuse.py:559 `derive_mask_from_axes(settled_below=0.35)`. Below this doubt, a bucket
    # becomes a target_free mask region (discounts later-round signals there).
    # derivation: unfitted — bare default, register F-139. A bucket at 0.34 gets the discount,
    # one at 0.36 does not, with no measured distinction between them.
  remeasure_doubt_floor: 0.6
    # was fuse.py:831 `fuse_axes(unsettled_above=0.6)`. Above this doubt, a bucket is offered to
    # the D-10 remeasure hook and counted toward C4 convergence.
    # derivation: unfitted — register F-140.

# ── speaker_identity: posterior-level verdicts about "how many speakers" and "is it corroborated" ──
speaker_identity:
  multimodal_posterior_floor: 0.15
    # was speaker_identity.py:121 `multimodal_threshold`. Posterior mass above which a speaker
    # count reads as a second supported mode ("multimodal" / not-converged).
    # derivation: unfitted — register F-144. THREADING NOTE: a `speaker_count.multimodal_threshold`
    # key already exists in default.yaml at this name and is *not* the fix — it is decorative
    # (see D2); this key replaces it and must actually reach speaker_count_posterior (see Part 4).
  supported_evidence_floor: 0.5
    # was speaker_identity.py:60 `_SUPPORTED_THRESHOLD`. Posterior mass at/above which a source's
    # claim counts as "corroborated by the audio" (has_supported_evidence).
    # derivation: unfitted — register F-145. Same decorative-config trap: default.yaml's
    # `speaker.supported_threshold: 0.5` already exists and is dead (D3); this key replaces it.

# ── quality: acceptance ramps for the headline quality/uncertainty verdict ──
quality:
  ramps:
    pesq: {low: 2.0, high: 3.5}
      # was global_summary.py:209. derivation: unfitted — claimed "literature-derived
      # acceptance thresholds," register F-149, no citation found anywhere in module/config/specs.
    stoi: {low: 0.5, high: 0.85}
      # was global_summary.py:210. derivation: unfitted — same claim, same problem, F-149.
    sisdr: {low: 0.0, high: 15.0}
      # was global_summary.py:211. derivation: unfitted — F-149; module's own docstring says
      # "below 5 dB poor," contradicting the 0.0 low-anchor in the code.

# ── support: evidence-pool admission for corroborating a signal's claim ──
support:
  min_low_evidence_fraction: 0.02
    # was support.py:276 MIN_LOW_FRACTION. Fraction of a signal's claims that must land in
    # low-evidence buckets before its support is discounted at all.
    # derivation: unfitted — register F-143; the only numbers offered were measured under a bug
    # the module's own docstring disowns ("must be re-measured before cited again").
    # THREADING NOTE: default.yaml's existing support.min_low_fraction key is dead (D9) — parsed
    # into RunConfig.support_policy, which nothing reads. This key must actually reach
    # support.informative_evidence's call sites (compute.py:728, adaptive/corroboration.py:60).

# ── adaptive.adjudication: unchanged shape, one addition ──
adaptive:
  adjudication:
    p2_fine_posterior_instability_floor: 0.0
      # was interventions.py:939 `_p2_trigger`'s bare `mean_instability > 0.0`. Above this,
      # a bucket's coarse-classifier share alone no longer explains its instability, and P2 fires.
      # derivation: unfitted — register F-157; the module's own docstring implies "a high value,"
      # but the code checks strictly >0, which is true of nearly every real-valued posterior.
      # Left at 0.0 here rather than silently changed, so the config key's default matches
      # current behavior exactly; the number to re-derive is this one, not a new default.
```

### 3.3 Additional sections for the highest-value *new* findings (illustrative, not exhaustive —
Part 2's full inventory is the actual to-do list)

```yaml
pii:
  # New section — none existed. Three call-site defaults with no config path at all (2.6).
  presidio_score_threshold: 0.4      # derivation: unfitted — engineering guess ("permissive
                                      # enough to catch standard phone-number formats").
  gliner_threshold: 0.5              # derivation: unfitted.
  cross_model_corroboration_count: 2 # derivation: unfitted — rationale exists for requiring
                                      # cross-model agreement at all, not for requiring exactly 2.

speaker_clustering:
  # New section, distinct from `speaker:` (same/diff/cluster floors) — these gate cluster COUNT
  # and MERGE decisions, a different question than same/different-speaker cosine floors.
  max_speakers: 6                    # was embeddings.py:100 n_clusters_max. derivation: unfitted.
  min_windows_for_multi_speaker: 4   # was embeddings.py:101. derivation: unfitted.
  single_speaker_silhouette_ceiling: 0.10  # was embeddings.py:102 coherent_silhouette_threshold.
                                      # derivation: unfitted — same silhouette-as-probability
                                      # pattern the register flags elsewhere (F-141) for a
                                      # different aggregator; here it decides single- vs.
                                      # multi-speaker instead of a disagreement score.
  min_cluster_size_fraction: 0.10    # was embeddings.py:270. derivation: unfitted.
  cluster_merge_cosine: 0.55         # was embeddings.py:324 merge_threshold. derivation:
                                      # unfitted for the magnitude; register F-164 separately
                                      # flags this same literal's adult-speech-corpus provenance.
  seed_group_merge_cosine: 0.75      # was clustering.py:91 cross_group_threshold. derivation:
                                      # unfitted; currently has NO config key at all, not even
                                      # a decorative one.
```

---

## Part 4 — Threading problems

For each parameter, the question is not "does a config key exist" (Part 1 answers that) but
**how many call-graph hops separate the read site from the RunConfig/policy object**, because
that hop count is the actual cost of this change.

### Shallow (0–1 hops: the variable holding the config value already exists in scope)

- **F-162 / D1 (`compute.py:433`)** — `speech_presence_policy` is already a local variable in
  `harvest_pass`, built from config three lines before the broken call. The fix is
  `fuse_consensus_words(asr_resolved, policy=speech_presence_policy)`. No new plumbing.
- **F-157 (`interventions.py:939`)** — `ctx["policy"]` is already the function's argument; every
  sibling threshold in the same file (`corroboration_low`, `low_native_confidence`, etc.) already
  reads from it. Adding `ctx["policy"]["adjudication"]["p2_fine_posterior_instability_floor"]` is
  a one-line change in a function already wired for config.
- **`adaptive/identity_repair.py:219`, `adaptive/fusion.py:297/310/324/349`** — all inside
  functions that already receive `ctx`/`policy` as an argument for *other* parameters in the same
  block (`cp_k`, `cp_floor`, `recluster_cosine_threshold` are already threaded two lines away).
  These are 4-5 separate one-line additions, not a design problem — the wiring exists, it just
  wasn't extended to every literal in the function.

### Medium (2–4 hops: config value must be threaded through 1-2 intermediate function signatures)

- **F-144/D2 (`speaker_identity.speaker_count_posterior`)** — `build_speaker_identity` (the one
  caller, from `scripts/analyze_audio.py:1241`) does not currently accept or forward a
  `multimodal_threshold`/`RunConfig` at all. Needs: (1) `RunConfig` gains a field, (2)
  `analyze_audio.py`'s call site passes it to `build_speaker_identity`, (3) that function forwards
  it to `speaker_count_posterior`. Three call-graph layers, none of them deep, but all three must
  move together or the config key becomes decorative again (exactly D2's current failure mode).
- **`support.informative_evidence`** — two independent call sites (`compute.py:728`,
  `adaptive/corroboration.py:60`), neither currently passing overrides. Fixing D7-D9 means
  updating both call sites consistently — a smaller version of the same "two callers must agree"
  problem as embeddings below.
- **`pii.detect_pii_in_pass`** — one call site (`scripts/analyze_audio.py:858`), but zero existing
  config surface (`RunConfig` has no `pii_*` fields, `default.yaml` has no `pii:` section). Shallow
  in call depth (one hop) but requires a full vertical slice: new YAML section, new `_validate`
  checks, new `RunConfig` fields, and the call-site edit — more design work than depth.

### Hard (5+ hops, or multiple independent call sites that must be reconciled, or no existing plumbing at all)

**1. `floors.MIN_EVIDENCE_WEIGHT` (`floors.py:13`).** By design (its own docstring) this is a leaf
module imported by `reliability.py`, `support.py`, `rounds.py`, `influence.py`, `invariance.py`,
`adaptive/belief.py`, and `adaptive/identity_repair.py` — **7 independent consumers across the
L1/L2/adaptive boundary**, none of which currently accept the value as a parameter; all import the
constant directly. `RunConfig` has no field for it at all. A single, separately-synced copy exists
at `adaptive.adjudication.min_evidence_weight` for one unrelated rule (P3), guarded only by a test
that fails if the two numbers drift (`evidence_weight_test.py`). Making this a real config value
means either (a) threading a config-sourced float through 7 modules that were deliberately kept
import-free of the package's own config system (the module's docstring explicitly argues for
leaf-module purity — "a shared constant must not be the thing that inverts their dependency
direction"), which means the fix fights the module's own stated design, or (b) accepting that
`floors.py` becomes the one leaf module `run_config.py` may import into, which is a small
architecture decision, not a mechanical thread-through. This is the hardest case in the inventory.

**2. Background-mask `mask.*` family (2.4, 8 constants).** Currently resolved through
`calibration.py`'s bundled detection-margin profile, with fallback literals **independently
duplicated in three files** (`background_mask.py`, `calibration.py`'s own defaults, and
`stages.py:545-546`). Any fix must (a) add the missing derivation requirement to
`validate_detection_margin_profile` for the `mask` block — currently the only one of four blocks
the validator doesn't check — and (b) collapse three independent fallback copies into one, without
breaking the two other blocks' existing (working) validation. This isn't deep in call-chain terms,
but it's wide: three files' worth of literal fallbacks have to move in lockstep or the "recorded
value cannot drift from used value" guarantee (the same guarantee F-162 already showed can silently
fail) breaks a second time in the same package.

**3. `embeddings.py`'s clustering-decision family (2.3, `n_clusters_max`, `merge_threshold`, etc.).**
Two independent production call sites — `compute.py:184` and `speech_presence_link.py:409` — plus
a *third*, unrelated hardcoded copy of the same cosine-threshold concept in `clustering.py:90-91`,
consumed by `plot.py`, that doesn't even share a name with the `speaker:` config section it
conceptually belongs to. Unifying these requires: deciding whether `compute.py` and
`speech_presence_link.py`'s two call sites should share one config value or need independent ones
(they currently silently share Python defaults, which may or may not be intentional), then
reconciling `clustering.py`'s plotting-only threshold against whichever `speaker_clustering:`
value is chosen — three files, two of them production paths that must not regress mid-change, one
of them display-only and easy to silently leave stale if the audit stops at the two "real" paths.
