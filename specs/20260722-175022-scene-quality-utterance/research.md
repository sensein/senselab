# Research: Scene-aware presence axis + improved utterance uncertainty

**Feature**: `20260722-175022-scene-quality-utterance`
**Date**: 2026-07-22
**Inputs**: `spec.md`; handoff note `SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md`; three codebase research passes (quality plumbing, scene/source plumbing, presence/utterance internals).

This document resolves every unknown needed to plan implementation. Each decision lists rationale and the alternatives rejected.

---

## D1 — Frame-level speech posteriors (presence)

**Decision**: Add a new function in `src/senselab/audio/tasks/voice_activity_detection/` that returns **continuous per-frame speech probability arrays + frame hop** (not `ScriptLine` segments), sourced from `pyannote/segmentation-3.0` raw scores via `pyannote.audio.Model.from_pretrained(...)` + `pyannote.audio.Inference(model)` (max over the speaker/powerset axis → P(speech)). Load through `ensure_hf_model` + token + `local_files_only` when cached.

**Rationale**: The current VAD path uses the high-level `Pipeline` (`pyannote_vad.py:73`), which returns thresholded segments and discards the posterior — exactly the smoothing the handoff note flags. `segmentation-3.0` is not referenced anywhere in the repo today, so this is a genuinely new low-level extractor. `Inference` yields a `SlidingWindowFeature` at ~16.9 ms/frame that we aggregate within each reporting bucket.

**Alternatives rejected**: (a) reuse `detect_human_voice_activity_in_audios` — returns segments only, smoothing lost. (b) TEN VAD / Silero — new dependency; the note ranks pyannote raw scores highest-leverage and zero-dep. Out of scope per spec.

**Constitution**: VI — use `ensure_hf_model`/`hf_local_files_only` (`dependencies.py:329,356`) rather than the direct-token load the pipeline path uses, so cached runs skip the Hub.

---

## D2 — Scene/quality model: `pyannote/brouhaha`

**Decision**: Add `pyannote/brouhaha` as a new scene/quality model loaded via the same `Model.from_pretrained` + `Inference` pattern as D1 (it is a `pyannote-audio` multitask model). One forward pass yields per-frame **(VAD, SNR dB, C50 dB)**. Gated; reuse the existing `HF_TOKEN` flow; **no new pip dependency, no subprocess venv**.

- `quality_reverb` ← Brouhaha C50 (dB → `[0,1]` degradation via calibrated normalization).
- `quality_snr` primary ← Brouhaha frame SNR (dB → `[0,1]`).
- Brouhaha VAD head → a second frame-posterior presence voter alongside D1.

**Rationale**: Confirmed via HF (`pyannote/brouhaha`, library `pyannote-audio`, gated, trained on LibriSpeech+AudioSet+EchoThief+MIT-reverb, arXiv 2210.13248). It coexists with the existing pyannote-audio dependency; it is the only source in-repo for room-acoustics (C50) — no DSP reverb routine exists anywhere in senselab.

**Alternatives rejected**: DSP C50 proxy (energy-decay/kurtosis) — lower fidelity, and the maintainer explicitly chose "add the model." Brouhaha via the `brouhaha-vad` pip package — redundant; the pyannote-audio load path is already available.

**Constraint**: Brouhaha inference is per-pass (once on the whole 16 kHz mono audio), then bucketed — not per-bucket model calls.

---

## D3 — Quality signal sourcing + the grid-resolution split

**Decision**: Compute the four quality degradation scores on a **coarse internal analysis window (0.5 s / 0.25 s hop)** and broadcast each presence bucket's value from its containing analysis window. Sources:

| Signal | Source | Notes |
|---|---|---|
| `quality_snr` | Brouhaha frame SNR (primary) cross-checked vs `spectral_gating_snr_metric`, `peak_snr_from_spectral_metric` (existing DSP) | agreement spread → `quality_uncertainty` |
| `quality_clip` | existing `proportion_clipped_metric` (cheap, slice-safe) | per bucket directly |
| `quality_reverb` | Brouhaha C50 | learned |
| `quality_bandwidth` | **new** minimal estimator: `librosa.feature.spectral_rolloff` (85% rolloff vs Nyquist) | see D8 |
| `quality_uncertainty` | normalized spread across the independent SNR estimators (Brouhaha vs the two DSP SNRs) in the window | high when estimators disagree |

**Rationale**: Research confirmed the STFT-based SNR metrics need a ≥~128 ms slice and SQUIM needs mono/16 kHz and is only meaningful at ≥~0.5 s; they are unreliable at the 0.1 s presence grid. Decoupling *analysis resolution* (0.5 s) from *reporting grid* (presence 0.1 s) is precisely the handoff note's central point. Broadcasting a 0.5 s quality value across the finer presence buckets is honest (the value's true resolution is recorded in provenance).

**`voice_signal_to_noise_power_ratio_metric` is NOT used per-bucket** — it runs VAD internally; if used at all it is computed once per pass.

**SQUIM**: available (`extract_objective_quality_features_from_audios` → stoi/pesq/si_sdr) but **optional** — it is a heavier secondary cross-check for `quality_snr`/intelligibility, batched over analysis windows, gated on `torchaudio_available()`. Not required for the P1 slice; include only if cheap enough on the validation clip.

**Alternatives rejected**: per-bucket quality at 0.1 s (unreliable numerics); `get_windowed_evaluation` as the driver (a single `None` window nukes the whole series — we slice + call metrics directly following the `embeddings.py:_slice_audio` + tail-anchored `_window_starts` pattern instead).

---

## D4 — Sound-source categorization

**Decision**:
1. **Raise `top_k`** for the AST and YAMNet windowed classification calls to the full label space (AST 527, YAMNet 521) via a new `--scene-top-k` CLI parameter (default: full), so per-window **full class-score vectors** persist. Top-1 consumers (`speech_presence_labels`, YAMNet veto) are unaffected — they index `labels[0]`.
2. Author a **checked-in, versioned JSON map** `{AudioSet display_name → category}` over the union of the AST + YAMNet vocab, category ∈ `{speech, people, machine, environment}`, with a documented default (`environment`) and logging on unmapped classes.
3. New harvester (mirrors `classification_top1_in_window`) sums per-window `scores` into the four category masses, normalizes to sum ~1, projects onto presence buckets by the existing center→nearest-window logic (`presence.py:335`).

**Rationale**: Research found only top-5 is persisted today (`_classify_windowed` truncates), and no AudioSet ontology exists in the repo or deps — so the map is hand-authored from the flat display-name list. Raising `top_k` is the mandatory unlock; the HF pipeline already computes the full distribution, only the slice is truncated.

**Alternatives rejected**: vendoring the full AudioSet `ontology.json` tree (heavier, and a flat keyword/label→category dict is sufficient for 4 coarse buckets); a dedicated model (PANNs/BEATs/HeAR) — out of scope (interface left open per FR-010).

**Coverage test** (SC-003): an automated test loads the AST `id2label` (527) and the YAMNet class CSV (521) and asserts every class maps to exactly one category. YAMNet class names live in the model's bundled CSV read inside the subprocess — the test obtains the canonical list from the model asset or a vendored copy.

---

## D5 — Per-axis grids

**Decision**: Add a `presence_grid: BucketGrid | None` parameter to `compute_uncertainty_axes`, mirroring the existing `utterance_grid` plumbing (param at `compute.py:57`, resolve at `:371`, thread into the presence harvest at `:246`, record in provenance at `:301`). Defaults: presence `0.1 s / 0.02 s`, utterance `1.0 s / 0.5 s`, identity/shared `0.5 s`. Quality analysis window fixed at `0.5 s / 0.25 s` (D3), recorded in provenance.

**Rationale**: Per-axis grids are already half-implemented (`utterance_grid` is a real, CLI-wired parameter); presence/identity currently share one grid. Adding `presence_grid` is a mechanical mirror.

**Alternatives rejected**: a single global grid (can't satisfy the phone-scale presence vs word-scale utterance requirement); a full per-axis grid registry object (premature abstraction — YAGNI per constitution VII; two named optional params suffice).

---

## D6 — Presence confidence/uncertainty split + coarse-voter demotion

**Decision**:
- Surface **`presence_confidence`** (= `presence_p_voice`, already computed at `compute.py:259` but never written) and **`presence_uncertainty`** (= existing `aggregate_presence` output) as two new columns. `aggregated_uncertainty` stays = `aggregate_presence` for backward compatibility.
- Add the D1/D2 frame-posterior voters to the per-bucket vote dict.
- **Demote coarse voters** (AST/YAMNet whole-window tags, `no_speech_prob`, sentence-level ASR overlap) at fine grids: instead of each casting an equal binary `speaks` vote per bucket, they contribute a single down-weighted "context prior" term. Implemented as a per-voter weight in the mean, keyed on whether the voter's native resolution is coarser than the reporting grid (native grid already recovered via `_native_classification_grid`, `presence.py:124`).

**Rationale**: The presence aggregator is already a weighted mean of per-voter p, not Shannon entropy (the docstring is stale). Confidence and uncertainty are both already computed; the rework is (a) exposing them and (b) adding continuous voters + demoting coarse ones so fine-grid agreement isn't inflated by voters that repeat one value across many buckets.

**Alternatives rejected**: replacing the aggregator wholesale with a new entropy/dispersion formula — unnecessary; the mean-of-p already yields both quantities. Dropping coarse voters entirely — loses useful low-frequency context; demotion to a prior is the note's recommendation.

---

## D7 — Token-level utterance uncertainty (highest risk)

**Decision**: Plumb Whisper token logits through the HF speech-to-text path and derive per-token entropy:
1. In `speech_to_text/huggingface.py`, request `generate_kwargs={"return_dict_in_generate": True, "output_scores": True}` (Whisper) and compute per-token softmax entropy + `avg_logprob`/`no_speech_prob` from the returned scores.
2. Add **optional fields** to `ScriptLine` (`avg_logprob`, `no_speech_prob`, `token_entropy: list[float] | float | None`) — currently `extra="ignore"` drops them, so they must be declared.
3. New utterance sub-signal in `aggregate_utterance` folding mean token entropy per bucket; new vote field harvested in `utterance.py`. Degrades gracefully (None) for backends that don't expose token scores (Granite/Canary/Qwen subprocess/text-only).

**Bonus finding**: this also **revives the currently-dead `avg_logprob` signal** — research showed `avg_logprob`/`no_speech_prob` are read via `seg_attr` but return `None` in production because `ScriptLine` never carries them (only test fixtures do). So plumbing token scores fixes an existing latent gap in both presence (Whisper `no_speech_prob` voter) and utterance (Whisper native confidence).

**Rationale**: FR-017 explicitly requires a token-level sub-signal; no backend exposes it today, so it must be generated. Whisper's HF generate loop is the only backend that naturally surfaces token logits.

**Risk / sequencing**: This is the one change touching a **core data structure** (`ScriptLine`) and a **non-workflow module** (`speech_to_text`), with broad blast radius (many tasks construct `ScriptLine`). It is sequenced last (phase 4), behind an additive, defaulted-`None` field change, with focused tests. If it destabilizes, it can be split to a follow-up PR without blocking phases 1–3 (flagged in Complexity Tracking).

**Alternatives rejected**: n-best/beam disagreement (needs `num_beams>1`, changes decoding cost and output) — entropy from a single greedy pass is cheaper and sufficient; MC-dropout (separate future primitive per existing memory `project_mc_dropout_optional`).

---

## D8 — Effective-bandwidth estimator + librosa dependency

**Decision**: Implement `quality_bandwidth` from `librosa.feature.spectral_rolloff` (85% roll-off frequency) normalized against Nyquist → high degradation when energy is confined to a low band (telephone/codec). **Add `librosa` as an explicit `pyproject.toml` dependency** (it is currently only transitive via `audiomentations`, pinned `0.11.0`).

**Rationale**: No rolloff/bandwidth/centroid routine exists in senselab. librosa is already importable and used in `quality_control/metrics.py`, so this adds no real install weight, but relying on a transitive dep is fragile — make it explicit. Rolloff-vs-Nyquist cleanly separates full-band from band-limited signals.

**Alternatives rejected**: Praat `extract_spectral_moments` (`spectral_gravity`/`spectral_std_dev`) — returns whole-slice scalars, needs parselmouth, and centroid/spread is a weaker band-limit signal than rolloff. Kept as a documented fallback only.

---

## D9 — Synthetic calibration harness

**Decision**: A `scripts/` helper that, from a clean tutorial clip:
- mixes added noise (white/pink, generated with numpy — no new dep) at a target SNR sweep;
- convolves a **synthetic exponential-decay RIR** parameterized to a target RT60 (numpy; C50 derived analytically) — no `pyroomacoustics` dependency;
- runs the quality estimators on the mixtures, fits the dB→`[0,1]` normalization (and a temperature-scaling hook for confidences) against the known SNR/RT60, persists the fitted parameters to a checked-in calibration profile, and emits a reported-vs-true validation plot/table.

**Rationale**: No labeled per-bucket SNR/reverb dataset exists (spec assumption). Synthetic mixtures give exact ground truth cheaply. Generating noise and a decaying-envelope RIR in numpy keeps it dependency-free (constitution VII).

**Alternatives rejected**: `pyroomacoustics` image-source RIRs (new dep for marginal realism gain); real RIR corpora (EchoThief/MIT) — download weight and licensing; the synthetic decay RIR gives a known RT60/C50 which is what calibration needs.

**Constitution VIII**: clip path, SNR sweep, RT60 targets, output path all CLI parameters with sensible defaults.

---

## D10 — Additive output & backward compatibility

**Decision**: Add new columns to `UncertaintyRow` (slots dataclass) **with defaults**, and matching `pa.array` columns in `write_axis_parquet`:
`presence_confidence`, `presence_uncertainty`, `quality_snr`, `quality_clip`, `quality_reverb`, `quality_bandwidth`, `quality_uncertainty`, `src_speech`, `src_people`, `src_machine`, `src_environment`, `src_dominant`, plus the utterance `token_entropy` and `scene_quality_coupling`. Per-vote detail (raw estimator values, category sub-scores) rides in the existing JSON `model_votes` — no schema churn there. `aggregated_uncertainty` semantics unchanged.

**Rationale**: Precedent exists (`intensity_weight`/`raw_aggregated_uncertainty` were added after the original parquet contract). Column-projecting readers ignore unknown columns; the LS bundle / plot / disagreements consumers keep working (regression tests guard this — SC-008).

**Constitution VII note**: new signals are **always-on additive columns** (null when a model is unavailable), **not** feature-flagged — simpler than a toggle and constitution-compliant. SC-008's "unchanged when disabled" is realized as "the existing `aggregated_uncertainty` computation is untouched by the new columns," verified by the existing regression suite.

---

## Resolved unknowns summary

| Unknown | Resolution |
|---|---|
| Frame posteriors source | `segmentation-3.0` + Brouhaha via `Model`+`Inference` (D1, D2) |
| Reverb estimator | Brouhaha C50 (D2) |
| SNR estimator | Brouhaha SNR + existing DSP SNR metrics (D3) |
| Quality at fine grid | coarse 0.5 s analysis window broadcast to presence buckets (D3) |
| Source class vectors | raise `top_k` to full; hand-authored category map (D4) |
| AudioSet ontology | none in repo → static display-name map (D4) |
| Per-axis grids | add `presence_grid` mirroring `utterance_grid` (D5) |
| Confidence/uncertainty split | surface `presence_p_voice` + `aggregate_presence` (D6) |
| Token entropy | plumb Whisper `output_scores` + extend `ScriptLine` (D7) |
| Bandwidth + librosa | librosa `spectral_rolloff`; make librosa explicit dep (D8) |
| Calibration data | synthetic noise + decay-RIR in numpy (D9) |
| Output additivity | new defaulted columns + JSON votes (D10) |
