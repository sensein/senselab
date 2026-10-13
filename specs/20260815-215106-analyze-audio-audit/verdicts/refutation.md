# Refutation verdicts — Sweep A stale-or-false findings (F-1..F-8)

### F-1
- verdict: SURVIVED
- evidence: `src/senselab/audio/workflows/audio_analysis/__init__.py:1-6` claims "three per-bucket uncertainty time series — speech_presence, speaker, and asr" and a "5-row timeline plot". `src/senselab/audio/workflows/audio_analysis/axes.py:1-18` (module docstring) and `axes.py:160-` (`AXES` tuple) confirm `AXES` has 4 members including `background_mask`, and the module's own docstring names "any list of three axes is wrong" as the defect this file exists to fix.
- note: top-level package docstring was never updated after `axes.py` was written; genuinely stale and false, not just imprecise.

### F-2
- verdict: SURVIVED
- evidence: `src/senselab/audio/workflows/audio_analysis/io.py:150` docstring says `write_linked_votes` writes `"L2/round0/votes/<axis>.parquet"`. The real call site, `scripts/analyze_audio.py:830-833`, writes to `derivatives_dir(run_dir, 0) / "votes" / f"{axis_name}.parquet"`, and `contracts.py:566-568,842` declares the artifact as `L2/round/{n}/derivatives/votes/*.parquet`. Neither `round0` (no slash) nor the missing `derivatives/` segment appears in the real path.
- note: docstring path is both missing a path segment and misspelled; matches claim exactly.

### F-3
- verdict: SURVIVED
- evidence: `stage_context.py:91-94` justifies bumping `STAGE_VERSIONS["ast"]`/`["yamnet"]` by saying "the classifiers (attach phoneme labels)". `grep -rn phoneme` across the package shows `phoneme_similarity`/`g2p_phonemes` live entirely in `asr.py`/`harvesters.py`; `stages.py:196-260` shows `ast`/`yamnet` only ever populate AudioSet scene-classification fragments (`ast_result`, `yamnet_result`, `scene_agreement`) — no phoneme output anywhere near them.
- note: the stated justification for bumping those two stages' versions describes a behavior that belongs to a different module entirely.

### F-4
- verdict: SURVIVED
- evidence: `adaptive/plot.py:711` (`_fused_axis` docstring) says "the belief store ingests L1's per-pass axis folds". `adaptive/belief.py:288-294` (`VoteStore.from_run_dir`) says explicitly: "It used to read `L1/<pass>/uncertainty/<axis>.parquet` — a per-pass axis fold, which is a quantity that cannot exist... Both are gone: this path now sees exactly what the in-process path... sees," ingesting `L2/round/0/derivatives/votes/<axis>.parquet` instead.
- note: `belief.py` documents its own removal of exactly the path `plot.py` still claims is live.

### F-5
- verdict: SURVIVED
- evidence: `adaptive/provenance.py:1-22` module docstring claims "every state change in a mutually-influencing loop is attributable ... via `RevisionRecord`/`classify_resolution`". `grep -rn "RevisionRecord|classify_resolution" src/senselab/audio/workflows/audio_analysis/` finds definitions only in `provenance.py` itself and a prose mention (not a call) in `corroboration.py:13`. No call sites in `loop.py`, `interventions.py`, or `belief.py`. The only call sites at all are in `src/tests/audio/workflows/audio_analysis/influence_test.py`, i.e. the mechanism is unit-tested in isolation but never wired into a real run.
- note: dead module as described; claim's mechanism is correct, not just its conclusion.

### F-6
- verdict: SURVIVED
- evidence: `adaptive/loop.py:3-6` module docstring says "round 1 is the ingested analyze_audio run". `_baseline_round` (`loop.py:735-742`) actually returns `last_round(out_dir) or 0` and its own docstring says this replaced a scheme where "the adaptive loop used to call its ingest 'round 1' while the fusion loop called the same iteration 'round 0'" — i.e. the baseline round is adopted, not fixed at 1. Also, `run_adaptive_loop` (`loop.py:61-76`) accepts `harvests=`, `unharvested_votes=`, `summary=` for a fully supported in-memory ingest path, entirely unmentioned in the docstring's "Artifact-driven" framing.
- note: two independent staleness points in the same docstring, both confirmed against code 730 lines later in the same file.

### F-7
- verdict: SURVIVED
- evidence: `interventions.py:22-24` lists "Still deferred: `P2_fine_posteriors`". `_p2_trigger`/`_p2_guard`/`_p2_execute` are defined at lines 895/949/962 and registered in the `RULES` table entry at lines 1209-1215 (`"id": "P2_fine_posteriors"`, wired to trigger/guard/execute).
- note: straightforwardly false as written — the rule is fully implemented and live in `RULES`.

### F-8
- verdict: SURVIVED
- evidence: `interventions.py:19-20` describes `I4_overlap_detection` as "segmentation-3.0 per-class posteriors (gated model...)". `_i4_execute` (`interventions.py:1087-1102`) calls `backends.overlap_track_from_spans`, whose own docstring (`backends.py:237-257`) says it derives overlap "from **cross-diarizer** spans rather than one model's channels" and produces a 1.0/0.0 **decision**, not a posterior ("Overlap here is a decision, not a posterior... A soft probability would need a model that reports one").
- note: the module-level summary describes the pre-refactor mechanism (direct segmentation-3.0 posteriors); the real mechanism is cross-diarizer span agreement, confirmed by a comment in the same file the finding cites.

---

Summary: 8/8 SURVIVED, 0 REFUTED, 0 SURVIVED-CORRECTED.

---

# Refutation verdicts — Sweep C (orchestration) + Sweep D (assumptions), F-162..F-176

### F-162
- verdict: SURVIVED-CORRECTED
- evidence: `compute.py:433` (`harvest_pass`) calls `consensus_fold = fuse_consensus_words(asr_resolved)` with no `policy=`, even though `speech_presence_policy` is a bound parameter of the same function and is passed to `harvest_background_mask_evidence(..., policy=speech_presence_policy, ...)` three lines earlier (`compute.py:401`). `fuse_consensus_words`'s own `policy` param (`asr.py:194-226`) is the only route by which `linking.asr_slot_overlap`/`asr_slot_mid_tol_s` (`speech_presence_link.py:105-106`, read via `policy_from_params`) can reach `slot_overlap`/`slot_mid_tol_s` — a caller passing `policy=None` always gets the 0.3/0.15 literals. Grepping every call of `fuse_consensus_words(` in the repo (`asr.py:287`, `asr.py:466`, `compute.py:433`), none passes `policy=` — the parameter is dead everywhere, not just at this one site.
- corrected-mechanism: the claim's "the two other call sites are structurally unreachable given the real caller always supplies `fused=`" is false: `adaptive/interventions.py:595` (`_reharvest_asr`, wired to the registered `U1_region_reasr` rule in `RULES`, `interventions.py:1181-1189`) calls `harvest_asr_votes(pass_summary=..., grid=..., alignment_by_model=...)` with no `fused=`, which falls through to `asr.py:466`'s own `fuse_consensus_words(asr_resolved)` call — also with no policy threaded through (`harvest_asr_votes` doesn't even have a `policy` parameter to forward). So the U1 live-re-ASR intervention path is reachable in production and independently drops the same config.
- note: core defect survives and is broader than claimed — every production path silently ignores `linking.asr_slot_overlap`/`asr_slot_mid_tol_s`, not just the one named call site.

### F-163
- verdict: REFUTED
- evidence: `compute.py:890-917` (`_speech_window_mask` docstring) gives an explicit, reasoned justification for YAMNet-over-AST priority ("YAMNet is trained on the AudioSet hierarchy with explicit speech labels; AST is broader-coverage but noisier on speech specifically"), names the exact tradeoff (child-voice-as-Music/Singing), and names a mitigation (tune `speech_presence_labels`). The finding's own `failure` line concedes: "Acknowledged/documented tradeoff, not a hidden bug." Checking the other three cited call sites: `stages.py:763-786` (`_scene_source_mass`) takes `max()` across both classifiers' scores per label — no priority order, both always contribute when present; `sound_sources.py:193-197` builds a `per_classifier` dict for both and unions whichever are available; `background_mask.py:520-535` (`_target_activity_by_label`) concatenates both classifiers' windows into one list and takes the best score across all of them, again no veto/priority. Only `compute.py:890` implements an actual priority ladder.
- note: legitimate, documented backend-routing decision (not an unprincipled dependency on which model happened to run) at the one site that does implement a ladder; the other three of the "four call sites" aggregate rather than replicate the ladder, so "the same...assumption is hardcoded at four call sites" overstates a real (but narrower) closed-two-classifier limitation as a 4x-repeated bug.

### F-164
- verdict: SURVIVED-CORRECTED
- evidence: `compute.py:405-418` (`harvest_pass`) shows `same_speaker_floor=0.30`/`diff_speaker_floor=0.70` are *not* applied uniformly — per-embedder `empirical_same_speaker_floor`/`empirical_diff_speaker_floor` are computed per pass from that pass's own clustering (`embeddings.py:196-213`, `_within_cluster_band`) and preferred over the fixed floors, which are used only as a fallback ("the honest reading of 'not measured'"). But `cluster_cosine_threshold=0.5` (`compute.py:105,533`, used at `speaker.py:240` as `min_similarity`) and `merge_threshold=0.55` (`embeddings.py:324-325`) are fixed literals with no empirical-override path found anywhere in `embeddings.py`/`speaker.py`. The `merge_threshold` comment (`embeddings.py:312-323`) explicitly names the exact failure mode the claim describes ("different speakers with similar timbre (same gender+age, children...) can sit at cos_sim ~0.30") and picks 0.55 as a fixed compromise, not a re-derived, population-specific value.
- corrected-mechanism: the defect survives for `cluster_cosine_threshold`/`merge_threshold` (uniform adult-derived constants, acknowledged-but-unresolved child-voice risk, no population caveat carried forward), but is weaker than claimed for `same_speaker_floor`/`diff_speaker_floor`, which the pass already re-measures per embedder when clustering succeeds.

### F-165
- verdict: SURVIVED
- evidence: `speaker.py:517-526` (`harvest_speaker_votes`): `if fused_words and coverage[key] <= 0.0: ... bucket_dict["votes"] = {}; continue`. The `votes` dict at that point (built over `speaker.py:380-493`) holds per-`diar::embedding` cosine/calibration entries, `__cross_diar_label_disagreement__`, per-embedder change-point evidence (J2), and `overlap_count` (J1) — described in the same file's own comments as measurements that "stay" for other consumers (e.g. "it stays because `identity_repair` reads it to place boundaries"). The wordless gate wipes the entire dict, including all of these, not just the two attribution voters that logically need words.
- note: matches the earlier reviewer's D-2 confirmation; independently re-verified against the actual `votes` contents built earlier in the function.

### F-166
- verdict: SURVIVED
- evidence: `speech_presence_link.py:249-273` (`_link_asr`): `hallucinated = bool(said_something and nsp is not None and nsp >= policy.no_speech_threshold)`; `speaks = said_something and not hallucinated`. A confident hallucination (low `nsp`) is indistinguishable from genuine transcription — `speaks=True` with `confidence = _pool_confidence(logprobs, ...)`, i.e. as high as the model's own (fabricated) token confidence. No cross-check against family agreement or task type exists in this function.
- note: no downstream mitigation found; claim stands as written.

### F-167
- verdict: SURVIVED
- evidence: `speaker_identity.py:469-492` (`evidence_from_passes`) builds evidence purely from `_distinct_speakers(outcome)` per diarizer per pass, and `build_speaker_identity` (`:495-528`) folds this through cross-pass stability and `support` weighting only. `grep -rn "population|age_band|pediatric|infant|toddler|child"` over `speaker_identity.py` and `global_summary.py` returns nothing — no population signal anywhere in the count-posterior path.
- note: claim is accurately descriptive; the structural indistinguishability of "child-diarizer failure" from "ordinary disagreement" follows directly.

### F-168
- verdict: SURVIVED
- evidence: `background_mask.py:41-46` (`TARGET_EVENT_LABELS`) has only `breath`/`cough`/`mouth_noise`/`throat_clear` keys. `calibration.py:230-236` and `data/detection_margin/2026-07-29.json:69-89` (`target_event_types_by_task`) define only `speech`/`breath`/`cough` tasks, falling back to `["speech","breath","cough","mouth_noise"]` for anything else, including a `cry`/`babble` task type. `data/audioset_source_map.json:32,142` confirms `"Baby cry, infant cry"→"people"` and `"Crying, sobbing"→"people"` exist in the *source-category* map used elsewhere, but neither reaches `TARGET_EVENT_LABELS`/`target_event_types_by_task`.
- note: confirmed exactly as claimed — the AudioSet vocabulary for cry/babble exists in the package but is wired to the wrong lookup table for target-activity purposes.

### F-169
- verdict: SURVIVED
- evidence: `degradation.py:33-43` (`DEFAULT_ANCHORS`, `snr_clean_db=25.0`, `c50_clean_db=30.0`) and `scene_degradation` (`degradation.py:129-152`) take only `measurements`/`sampling_rate`/`calibration`, no `task_type`. The one call site, `votes.py:459,474`, computes `anchors = _quality_anchors(params)` once per run (`votes.py:243-258`, reading `params["calibration"]`, a single dict of `snr_clean_db`/`c50_clean_db`/etc.) — applied identically to every bucket regardless of task.
- note: confirmed; no task-conditioning path exists for this anchor, unlike `background_mask.py`'s task-keyed machinery.

### F-170
- verdict: SURVIVED
- evidence: `compute.py:898-913` (`_speech_window_mask` docstring) states YAMNet's top-1 is authoritative/veto over AST and loudness, names the child-voice-as-Music/Singing confusion explicitly, and states "the mitigation is upstream: tune `speech_presence_labels` ... yourself" — a manual, per-run operator action with no automatic default. `compute.py:963-981` implements exactly this veto in the mask-building loop feeding `_cluster_pass_speakers`.
- note: independently re-confirmed; consistent with the earlier reviewer's finding.

### F-171
- verdict: SURVIVED
- evidence: `compute.py:101,529,591-592`: `embedding_window_s: float = 2.0` documented as "ECAPA's recommended minimum", a single global default with no task/population conditioning found anywhere in `compute.py`/`run_config.py` (`run_config.py:169,464` just forwards one float from config).
- note: confirmed fixed, non-adaptive window sizing as claimed.

### F-172
- verdict: SURVIVED
- evidence: `global_summary.py:289-306`: `elif n_speakers == 1: ... = 0.0` / `elif n_speakers == 0: ... 1.0 if expects_speech else 0.0` / `else: single_speaker_uncertainty = 1.0` — any count ≥2 is unconditionally scored 1.0. `compute_run_global_summary`'s only population/expectation knob is `expects_speech: bool = True` (`:226-249`), which governs only the `n_speakers==0` branch; no analogous parameter exists for the ≥2 case.
- note: confirmed exactly as claimed, independently re-verified against the earlier reviewer's D-9 confirmation.

### F-173
- verdict: SURVIVED
- evidence: `identity_repair.py:169` (`min_seg = float(cfg.get("min_segment_s", 0.25))`) and `:199` (`thr = float(cfg.get("recluster_cosine_threshold", 0.45))`), defaults mirrored in `data/run_config/default.yaml:485-487` with no derivation comment or `data/`-file citation (unlike this codebase's stated convention for fitted thresholds) and no population-specific note anywhere nearby.
- note: confirmed fixed, undocumented-derivation defaults as claimed.

### F-174
- verdict: SURVIVED-CORRECTED
- evidence: `interventions.py:1059-1122` (`_i4_execute`) calls `overlap_track_from_spans(by_model, span=...)` (`backends.py:237-278`), whose docstring says overlap is computed "from **cross-diarizer** spans" but the actual computation (`backends.py:265-277`) is `track.append(1.0 if max(count_at(s, t) for s in spans_by_tool.values()) > 1 else 0.0)` — i.e. overlap is 1.0 if **any single diarizer's own** (`exclusive=False`) segmentation reports ≥2 concurrent spans at that instant, OR'd across diarizers. `belief.py:1099` confirms `overlap_posterior` feeds `ALEATORIC_FLOOR_TERMS`/`_attach_floor`.
- corrected-mechanism: the claim's mechanism ("purely from inter-diarizer disagreement") is wrong — it is an OR across each diarizer's own overlap-aware output, not a disagreement measure between diarizers. The underlying population concern survives under the corrected mechanism: if every diarizer (trained mainly on adult multi-speaker corpora) individually fails to register a child's simultaneous non-turn-taking vocalization as a second concurrent span, `overlap_posterior` still reads falsely low, feeding `aleatoric_floor`/`irreducible_reason` with no population caveat.

### F-175
- verdict: SURVIVED
- evidence: `interventions.py:430-446` (`_missed_speech_candidates`): builds `families` from ASR votes with `payload.get("text")` in the bucket, then `if len(families) >= 2: out.append(...)` — confirmed exact gate as claimed.
- note: non-lexical vocalization (crying/babbling) structurally cannot converge ≥2 ASR families on word text, so C9 cannot fire regardless of true vocalization presence, exactly as claimed.

### F-176
- verdict: SURVIVED-CORRECTED
- evidence: `evaluate.py:85-87` (`transcribed`/`untranscribed` split by GT `text` presence) and `:117-124` (`_score_words` restricts both ref and hyp tokens to `transcribed` spans). `evaluate.py:192-195` does record `"untranscribed_gt_spans": untranscribed` (raw span list) and `"fused_confidence_in_untranscribed"` under `localization` — so "no record of the untranscribed-span fraction" overstates the gap slightly, since the raw spans are present and a fraction is derivable from them.
- corrected-mechanism: no field computes a duration/fraction figure (e.g. `untranscribed_fraction`) alongside the headline `transcript.wer`/`wer_normalized`; a consumer reading only the headline WER gets no coverage caveat, so the substantive claim (identical-looking headline WER for an all-adult vs. substantially-untranscribable corpus) still holds.

---

Summary (F-162..F-176): 10 SURVIVED, 4 SURVIVED-CORRECTED, 1 REFUTED.

---

# Refutation verdicts — Sweep B computation findings (F-139..F-161, raised-by B-1..B-22 + B-shadow)

### F-139
- verdict: SURVIVED
- evidence: `fuse.py:559` (`derive_mask_from_axes`) has `settled_below: float = 0.35` as a bare default, used at `fuse.py:598` (`if max(settled) > float(settled_below): continue`). `grep`ed every call site of `derive_mask_from_axes(` in the package: none passes `settled_below=`. `data/run_config/default.yaml` has no `settled_below` key. `keys.py:144` itself names `settled_below=0.35` as the paradigm case of an unfitted default.
- note: confirms the earlier reviewer's finding; no override path exists anywhere.

### F-140
- verdict: SURVIVED
- evidence: `fuse.py:831` (`fuse_axes`) has `unsettled_above: float = 0.6` as a bare default, used at `fuse.py:988` to build `_pending` (offered to `remeasure` and counted toward C4 convergence). No call site of `fuse_axes(` passes `unsettled_above=`; not present in `default.yaml`.
- note: same unfitted-default pattern as F-139, independently confirmed.

### F-141
- verdict: SURVIVED
- evidence: `aggregators.py:86-91`. Algebraically: `mean_conf = sum(confidences)/len(confidences) = 1 - mean(u)`, so `(1.0 - mean_conf) * max_u == mean(u) * max(u)` exactly. Worked the two examples: five signals at 0.9 → `mean=0.9, max=0.9` → `0.81`; four at 0.0 + one at 1.0 → `mean=0.2, max=1.0` → `0.2`. `"disagreement_weighted"` is confirmed selectable via `data/run_config/default.yaml:236`.
- note: exactly as claimed — the aggregator is a level statistic, not a spread statistic.

### F-142
- verdict: SURVIVED
- evidence: `level.py:129-289` (`apply_gain_db`, `integrated_lufs`, `loudness_range_lu`, `true_peak_dbtp`, `clipped_fraction`, `normalization_gain_db`, `peak_limited_gain_db`) all take only `(waveform, sampling_rate, ...)` numpy/pyloudnorm/scipy args, no `audio_analysis`-specific types. `senselab/audio/tasks/quality_control/` exists (checked its files) and has no BS.1770 loudness/gain/clipping implementation to duplicate against.
- note: genuine promotion-candidate — no workflow coupling found.

### F-143
- verdict: SURVIVED
- evidence: `support.py:276-298` (`MIN_LOW_FRACTION` docstring) explicitly states the only numbers ever offered for `MIN_LOW_FRACTION=0.02` were measured under `native_confidence` read undirected, and closes with "the per-voter verdicts above must be re-measured before they are cited again" — the docstring disowns its own evidence. Used at `support.py:301-353` (`informative_evidence`) to gate evidence-pool admission feeding `speaker_count_posterior`/`reliability.measured_weights`.
- note: matches earlier-confirmed B-5 read; the threshold's only justification is self-disowned.

### F-144
- verdict: SURVIVED
- evidence: `speaker_identity.py:121` (`multimodal_threshold: float = 0.15`), used at line 177 (`modes = [c for c, p in probabilities.items() if p >= multimodal_threshold]`). `data/run_config/default.yaml:550` carries the same value with only a restated-semantics comment ("a count above this counts as a supported mode"), no derivation. Confirmed `is_multimodal` gates `converged` at `speaker_identity.py:585` (`converged=not posterior.is_multimodal and doubt < 0.5`), so the threshold does flip a stopping decision. Also note (not part of the original claim but relevant): the config value is never actually threaded to `speaker_count_posterior(claims, gates=gates)` at line 524 — the call site omits `multimodal_threshold=`, so the YAML entry is currently decorative and the hardcoded 0.15 always governs.
- note: confirmed unfitted; comparison to `calibration.py`'s cited detection-margin ladder holds up.

### F-145
- verdict: SURVIVED
- evidence: `speaker_identity.py:60` (`_SUPPORTED_THRESHOLD = 0.5`), used at `has_supported_evidence` (lines 267-274, `any(v >= _SUPPORTED_THRESHOLD ...)`). No derivation comment. Confirmed the contrastive claim: `floors.py:13` `MIN_EVIDENCE_WEIGHT = 0.05` is reused with a derivation note across `invariance.py`, `influence.py`, `reliability.py`, `rounds.py`, `support.py`, unlike this bare midpoint.
- note: exactly as claimed.

### F-146
- verdict: SURVIVED
- evidence: `identity_binding.py:98-147` (`per_speaker_presence`), `binding_agreement = (len(bound)/eligible) if eligible else 0.0` (line 145). `bind_labels`'s own docstring (line 42-44) says it returns `None` "when a tool produced no spans" (not a failure to bind, nothing to bind to) — and at line 129-130 `if binding is None: continue` drops that tool from bound/unbound/eligible entirely, same as a tool `is_censored_at` capacity (also excluded from eligible). So `eligible==0` covers both "everybody explicitly rejected" and "nothing was checked" identically, and the docstring (lines 112-119) only describes the rejection case. Confirmed unwired: `grep`ed all non-test call sites of `per_speaker_presence`/`binding_agreement` in the package — none; only `identity_binding_test.py` exercises it.
- note: mechanism confirmed exactly as stated, including the "unwired into production" detail.

### F-147
- verdict: SURVIVED
- evidence: `speaker.py:636` (`n_models = len(clusters)`), `:640` (`share = len(sources)/n_models`), where `clusters = _bucket_clusters(bucket)` (line 577-589) builds its dict only from diar models actually present in `votes` for that bucket — a crashed/never-run model is simply absent, shrinking `n_models` rather than counting as a silent vote. Confirmed the sibling fix exists: `speaker.py:556` (`speaker_assignment` voter) explicitly carries `"n_sources": len(clusters)` alongside the value specifically to avoid this same collapse (comment there names the exact failure mode).
- note: matches earlier-confirmed B-9 read.

### F-148
- verdict: SURVIVED
- evidence: `statistics.py:51,81,112,132` (`confidence`, `variability`, `entropy_uncertainty`, `epistemic_uncertainty`) — read the whole file; imports are only `math`/`typing`, signatures are generic `Sequence[bool|float]`/`Mapping[str,float]`, zero `audio_analysis` type coupling.
- note: genuine promotion-candidate.

### F-149
- verdict: SURVIVED
- evidence: `global_summary.py:195-211` (`ramp`, `pesq_unc/stoi_unc/sisdr_unc`) — `ramp(2.6, low=2.0, high=3.5) = 1 - (2.6-2.0)/(3.5-2.0) = 0.6`, confirmed by direct arithmetic. Docstring at lines 160-172 claims "literature-derived acceptance thresholds" and says SI-SDR "below 5 dB poor," but the ramp's actual low anchor is `0.0` (line 211: `ramp(sisdr_mean, low=0.0, high=15.0)`) — confirmed contradiction. `grep`ed `specs/` and `data/` trees for any PESQ/STOI/SI-SDR citation backing 2.0/3.5, 0.5/0.85, or 0.0/15.0 specifically for this ramp: none found (only unrelated model-comparison numbers in `research-models-2026.md`). Traced `combined_uncertainty` to `global_summary.py:398` (`combined = max(components)`, with `quality_block.get("uncertainty")` one of the components) — confirmed the headline metric can indeed be driven by this ramp.
- note: as claimed; the docstring's own numbers contradict its own code.

### F-150
- verdict: SURVIVED
- evidence: `disagreements.py:152` (`"high_uncertainty_rate": (high_count / total_rows) if total_rows else 0.0`). Confirmed downstream: `scripts/check_layering.py:121` prints `d['totals']['high_uncertainty_rate']` directly against a hardcoded baseline comment `(was 0.9941)`.
- note: exactly as claimed.

### F-151
- verdict: REFUTED
- evidence: `noise_floor.py:294` (`margin = float(cfg.get("recorder_margin_db", 3.0))`), consumed by `binding_floor` (`noise_floor.py:358-373`). The claim says no measured/cited number backs this, unlike every other margin in `data/detection_margin/2026-07-29.json`. But `calibration.py:238-240` (the same file/profile's `"derivation"` block) lists `{"claim": "minimum measurability ~+3 dB", "source": "ISO 1996-2:2017", "status": "verified"}` — an exact match to `recorder_margin_db`'s value (3.0 dB), and the module's own scoping comment at `calibration.py:142-144` states this derivation block explicitly governs "the noise-floor estimator settings" (which is where `recorder_margin_db` lives, nested under `noise_floor` in the same profile). The concepts align directly: ISO 1996-2's "minimum measurability" margin and `recorder_margin_db`'s "is the measured floor indistinguishable from the known recorder self-noise floor" are the same underlying question (minimum dB gap before a difference is not reliably real).
- note: weaker-than-ideal derivation (no inline comment ties the ISO citation to this specific key by name, and `specs/.../tasks.md:155` only restates "within a few dB" qualitatively), so this is a borderline call — but the file's own stated scope and the exact numeric match are enough to say a derivation exists that the claim missed, rather than that none exists anywhere.

### F-152
- verdict: SURVIVED
- evidence: `acoustic.py:50-76` (`lufs_track`), `:127-167` (`level_above_floor_track`) — both take only `(waveform, sampling_rate, hop_s=...)`, pure numpy + optional pyloudnorm, no workflow types.
- note: genuine promotion-candidate.

### F-153
- verdict: SURVIVED
- evidence: `occupancy.py:133-157` (`occupancy`), `:160-169` (`_union_length`) — `_union_length` is pure interval-union arithmetic on `list[tuple[float,float]]`; `occupancy`'s only workflow coupling is the `Spans`/`Span` dataclass signature, as claimed.
- note: genuine promotion-candidate.

### F-154
- verdict: SURVIVED
- evidence: `adaptive/loop.py:315-317` (`run_state = "converged" if not not_admitted else "no_runnable_interventions"`, gated on `if not fired`). Traced `not_admitted`: only populated by `policy.py:plan_round` for rule *candidates whose trigger already fired* but were blocked by guard/disable/budget (`policy.py:213-225`). A bucket sitting between `theta_low` and `theta_high` never gets a region seeded at all (`regions.py:46`, seed requires `_u(rows[i]) >= theta_high`), so no rule trigger ever runs against it, so it can never appear in either `admitted` or `not_admitted` — confirming the self-contradiction the claim describes (per-bucket status can read "open" while the run-level status reads "converged").
- note: confirms the earlier reviewer's finding; matches the exact mechanism claimed.

### F-155
- verdict: SURVIVED
- evidence: I1 (`interventions.py:798-812`) writes payload `{"change_point_times", "change_point_confidence"}`; I2 (`:850-865`) writes `{"speaker_label", "cluster_id", "speaker_changed_from_prev"}`. `fuse.py:54-77` (`_UNCERTAINTY_FIELDS`/`_CONFIDENCE_FIELDS`/`_LOGPROB_FIELDS` = the full `_SCORED_FIELDS` set `per_signal_uncertainty` reads) contains none of those keys. `belief.py:818` (`sources = sorted({src for v in votes_by_pass.values() for src in v})`) builds `contributing_sources` from every vote source key present, regardless of whether `fuse_axis` scored it.
- note: exactly as claimed — a schema mismatch that silently produces "no-op" votes counted as contributors.

### F-156
- verdict: SURVIVED
- evidence: `identity_repair.py:227-231` — `seg["boundary_confidence"] = {"start": cp_conf.get(..., 0.5), "end": cp_conf.get(..., 0.5)}`, the literal `0.5` fallback for any edge with no genuine change-point. `adaptive/fusion.py:285-297` writes this into `final/diarization.json` with the comment "real boundary confidences from change-point prominence" verbatim, and line 324's `else` branch also defaults to `{"start": 0.5, "end": 0.5}` for the non-repaired path.
- note: exactly as claimed.

### F-157
- verdict: SURVIVED
- evidence: `interventions.py:939` — `fires = coarse_share >= threshold or mean_instability > 0.0`, comparing a continuous `frame_dispersion`-derived mean against exactly `0.0`. Docstring at lines 905-907 ("a high value means the bucket straddles an onset") confirms `mean_instability` is meant to be read as graded, not boolean, contradicting the `>0.0` gate.
- note: exactly as claimed.

### F-158
- verdict: SURVIVED
- evidence: `policy.py:14` (`_COST_WEIGHT = {"light":1.0,"medium":4.0,"heavy":16.0}`), `:196` (`priority = round(gain / _COST_WEIGHT[rule["cost"]], 9)`), sorted globally at `policy.py:211`. Three incompatible gain functions found: `interventions.py:1134` (`_mass_gain`, bounded region mass), `:1138` (`_n_candidates_gain`, raw unbounded count), `:502-503` (`_u2_gain`, `uncertainty_mass * epistemic * 10.0`, arbitrary x10). All three are assigned to different `RULES` entries (lines 1149-1223) feeding the same sorted candidate list.
- note: exactly as claimed — no shared normalization across gain families.

### F-159
- verdict: SURVIVED
- evidence: `convergence.py:75-77` — `improvement = float(prev_u) - float(last_u)`, the raw two-point delta, feeding `stalled = improvement is not None and improvement < epsilon` (line 78), which gates `irreducible`/convergence marking. `provenance.py:1-22`'s docstring (confirmed dead per F-5) explicitly describes exactly this ambiguity ("uncertainty can fall for two completely different reasons... both look identical in the number alone") as the risk `classify_resolution` exists to close, and it is never called from `convergence.py` or anywhere in the live loop.
- note: exactly as claimed; ties correctly to the same dead module as F-5/B-shadow-adjacent findings.

### F-160
- verdict: SURVIVED
- evidence: `identity_repair.py:46-50` (`_l2`), `:53-78` (`change_point_trajectory`), `:125-149` (`_agglomerative_cosine`) — all take plain numpy arrays / `list[dict]` with generic `vector`/`start_s`/`end_s` keys; no `Region`, `VoteStore`, or policy-dict coupling anywhere in the three functions.
- note: genuine promotion-candidate; leading-underscore naming is indeed the only obstacle to moving them.

### F-161
- verdict: SURVIVED
- evidence: Reproduced live: `cd src/senselab/audio/workflows/audio_analysis && uv run python -c "import ast"` raises `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types' (most likely due to a circular import) (.../audio_analysis/types.py)` — output matches the claim verbatim, including the misleading intermediate frame through `weakref`/`_weakrefset`.
- note: directly reproduced, not just read.

---

Summary (Sweep B, F-139..F-161): 22/23 SURVIVED, 1/23 REFUTED (F-151/B-13), 0 SURVIVED-CORRECTED.
