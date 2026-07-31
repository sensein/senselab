# SOTA Review: Speech Detection, Timing, Scene Analysis & Off-Target Speaker Detection (2024–2026)

**Purpose.** Durable reference for evolving the senselab `audio_analysis` workflow
(`src/senselab/audio/workflows/audio_analysis/`) toward two goals:

1. **Detect speech and its timing as accurately as possible** (presence axis + word/turn boundaries).
2. **Catch speakers who are not the intended recording target** — in recordings that are
   *supposed* to be a single person doing **read speech** (reading a passage) or **free speech**
   (spontaneous monologue), flag any segment that is a different / additional voice
   (interviewer, bystander, background talker).

**Key operating constraint (from the study).** There is **no reliable target-speaker anchor** —
recordings vary and often ship without a clean enrollment clip of the intended subject.
Therefore off-target detection must be **unsupervised**: estimate the file's *dominant* identity
internally and flag **novelty / deviation** from it, rather than verifying against an enrollment.

**Scope of the literature.** 2024–early 2026, prioritizing ICASSP, Interspeech, ICLR, ICML,
DCASE, ASRU/SLT, WASPAA. Every arXiv/HF/DOI id below was verified against a primary source by
the research pass; anything not confirmed to a primary number is marked **unverified**.

> Cross-paper metric comparisons are only indicative — datasets and splits differ
> (VoiceBank+DEMAND vs DNS vs URGENT for enhancement; TIMIT vs Buckeye for alignment;
> AudioSet-2M vs AudioSet-Strong vs DESED for tagging/SED).

---

## 0. TL;DR — priorities for *this* objective

Ranked by expected impact × availability, adapted to the **no-anchor, mostly-single-target,
read+free-speech** setting:

| # | Action | Serves | Available today | Effort |
|---|--------|--------|-----------------|--------|
| 1 | **Unsupervised off-target axis**: per-window distance from the file's dominant ECAPA centroid + overlap + scene-babble, computed on **raw** audio | Off-target detection | ✅ (senselab already has ECAPA + powerset seg-3.0) | Med |
| 2 | **PretrainedSED** frame posteriors replace AST/YAMNet clip tags | Scene sources w/ timing, speech-vs-babble, cleaner speech mask | ✅ `fschmid56/PretrainedSED` (MIT) | Med |
| 3 | **Boundary-aware inference + collar/PSDS metrics** on VAD posteriors | Speech-timing boundary accuracy | ✅ (own posteriors) | Med |
| 4 | **MWA aligner** (learned-DP on MMS reps) replaces MMS/WhisperX alignment | Word-boundary precision | ✅ `MLSpeech/Multilingual-Word-Aligner` (CC-BY-4.0) | Med |
| 5 | **MarbleNet v2.0 / TEN VAD** frame VAD (20/16 ms, noise-hardened) | Presence + sharp offsets | ✅ NeMo HF / `pip ten-vad` | Low |
| 6 | **Per-file contextual biasing** (name/vocab lists → ASR) | Short-word recovery ("Josh"), free-speech OOV | ✅ (config) | Low |
| 7 | **Read-speech transcript-deviation** signal (words not in reference passage) | Off-target + extraneous speech | ✅ (uses existing ASR + reference text) | Low-Med |
| 8 | **VAP** gap-arbitration stream | Presence dips at short inter-turn gaps | ⚠️ research code | Med-High |

**Two counterintuitive rules for the off-target objective specifically:**

- **Run off-target detection on RAW audio.** Speech enhancement / denoising / target-speaker
  extraction all exist to *suppress* background voices — i.e. they erase the exact evidence you
  want to catch. An enhanced pass will bias toward a false "clean, target-only" verdict.
  Enhancement stays useful for *transcription/quality* axes, but must not gate off-target detection.
- **Do not raise no-speech / silence-suppression thresholds.** The same knob that curbs ASR
  hallucination is what deletes short, quiet, off-target utterances (and short target words like a
  dropped name). Favor verbatim ASR + max-recall union instead.

---

## 1. Voice Activity Detection & speech-boundary timing

**Problem we measured.** On a hard validation clip, frame-posterior VADs (pyannote
`segmentation-3.0` + `brouhaha`) fire continuously through a ~0.4 s inter-turn gap and never dip;
onset/offset boundary MAE ≈ 2.6 s with high recall only. This is the **known** failure mode of
frame-posterior VADs that (a) lack a boundary-localization objective and (b) use
hysteresis / min-duration smoothing that bridges short silences.

### Frame-level VAD backbones

| Model | Venue / Year | Idea | Metric | Frame hop | Availability | Fit |
|-------|--------------|------|--------|-----------|--------------|-----|
| **Frame-VAD Multilingual MarbleNet v2.0** | NVIDIA NeMo release 2024–25 (arch ICASSP 2021) | 1D TC-separable CNN, per-20 ms speech prob, noise+confuser augmentation (cough/laugh/breath) | ROC-AUC: VoxConverse 96.7/97.6, AMI 96.3, Earnings21 97.1, AVA-Speech 95.3 | 20 ms | HF `nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0` (91.5K params, NVIDIA Open Model License) | ★★★ noise-hardened, tiny, CPU |
| **TEN VAD** | Open-source 2025 (no paper) | Lightweight ONNX VAD tuned to cut speech→non-speech transition latency | Beats WebRTC/Silero on PR curves; **no scalar F1/AUC** (unverified) | 10/16 ms | HF `TEN-framework/ten-vad`; pip `ten-vad`; Apache-2.0 (+LPCNet BSD) | ★★★ sharp offsets — targets the "rides through the gap" symptom |
| **SincQDR-VAD** | arXiv 2508.20885 (Aug 2025) | Learnable Sinc front-end + Quadratic Disparity Ranking loss; 8.0K params | AVA-Speech AUROC 0.914 / F₂ 0.911; −10 dB AUROC 0.709 (vs MarbleNet 0.620); ACAM changing-env avg AUROC 0.97 | — | GitHub `JethroWangSir/SincQDR-VAD` (license unverified) | ★★★ best low-SNR / per-environment; ranking loss adaptable |
| **Silero VAD v5** | OSS 2024 | Small RNN/CNN frame VAD | 87.7% TPR; ~4× fewer errors than WebRTC @5% FPR (vendor) | — | GitHub `snakers4/silero-vad`, MIT | ★ *is* the class that produces the gap-bridging symptom; baseline only |
| **Quail VAD** (ai-coustics) | Vendor Nov 2025 | VAD + real-time enhancement front-end | Claims F1 wins on MSDWild (no numbers) | — | Proprietary SDK, no HF | ✗ closed, not library-usable |

### Boundary / onset–offset accuracy — the core fix

- **Boundary-Aware Optimization & Inference for SED** — arXiv **2601.04178** (Jan 2026, Meta
  Reality Labs + JKU). Treat onset and offset as **separate** detection targets with dedicated
  losses + a **boundary-aware inference** procedure that replaces naive hysteresis/threshold
  post-processing on any frame-wise model. Improves PSDS1 + collar-F1 (deltas not machine-
  extractable — unverified numbers, but the *right* metrics). **This is the paradigm the current
  VAD is missing.** Adaptable to existing posteriors — no new model required.
- **Metrics to adopt** (stop reporting boundary-MAE alone): **collar-based event F1** (onset collar
  ~200 ms; offset collar = max(200 ms, 20% of duration) — DCASE standard); **PSDS1** (threshold-
  independent, onset/offset-strict; def. arXiv 2201.13148); diarization **collar 250 ms/side**.
  The "high recall, 2.6 s MAE" result is a smoothing artifact these per-operating-point metrics expose.

### Short inter-turn gap disambiguation

- **Voice Activity Projection (VAP)** — real-time VAP IWSDS 2024 (arXiv **2401.04868**);
  multilingual EN/ZH/JA LREC-COLING 2024 (arXiv **2403.06487**); triadic 2025 (arXiv 2507.07518).
  Transformer predicts each speaker's activity over the next 2 s in bins
  **0–200 / 200–600 / 600–1200 / 1200–2000 ms** — i.e. it models gap structure directly and decides
  *hold* (within-turn pause) vs *shift* (real offset). Public code, CPU real-time. Run as a
  second stream to **arbitrate** boundaries the frame VAD smooths over. ★★★ for the 0.4 s gap.
- **Next-Turn duration-aware endpointing** — arXiv 2606.18094 (2026): regress time-to-next-onset
  instead of thresholding silence → avoids false endpoints on brief pauses. (metrics unverified)

### Noise / changing-environment robustness (recipes, mostly adaptable not drop-in)

- **DN-APC noise-robust TS-VAD** — arXiv **2501.03184** (Jan 2025, Aalborg/Oticon): causal
  denoising-APC SSL pretraining (predict clean features k=3 ahead from noise+reverb input) →
  +~2% mAP, largest at low SNR (+2.63% @ −5 dB), targets **unseen** babble.
- **SSL pretraining for robust personal VAD in adverse conditions** — arXiv 2312.16613.
- **Robust personal VAD for domain mismatch** — Interspeech 2025 (ISCA `lin25_interspeech`)
  (numbers unverified — PDF not machine-extractable).
- **WavLM** (`microsoft/wavlm-base-plus`) as a noise-invariant frame front-end (denoising SSL
  pretraining) — pip/HF, adds compute.

---

## 2. Auditory scene analysis / ASC / SED — background-source detection with timing

**Why central to off-target.** Background human voices (`Babble`, `Crowd`, `Conversation`,
`Hubbub`, `Chatter`) are a *direct, enrollment-free* signal that a non-target speaker may be
present — even when they don't cleanly overlap the target. Current stack uses **clip-level** AST +
YAMNet posteriors; the upgrade is **frame-level** posteriors from AudioSet-Strong models.

### Foundation models (tagging plateau ≈ 49–51 mAP AudioSet-2M)

| Model | Venue / Year | mAP (AS-2M) | Frame-level? | Availability |
|-------|--------------|-------------|--------------|--------------|
| **BEATs** | ICLR 2023 (arXiv 2212.09058) | 50.6 (audio-only SOTA) | backbone (frame embeddings) | GitHub `microsoft/unilm`, MIT — common DCASE Task-4 encoder |
| **CED-base** | ICASSP 2024 (arXiv 2308.11957) | ~49.0 @ 86M | no (clip / 10-s) | **HF `mispeech/ced-base` (Apache-2.0, ONNX)** — drop-in AST replacement |
| **Dasheng-1.2B** | Interspeech 2024 (arXiv 2406.06992) | >50 (first open) | backbone (~25 Hz) | HF `mispeech/dasheng-1.2B` (Apache-2.0) |
| **EAT** | IJCAI 2024 (arXiv 2401.03497) | base 48.6 / large 49.5 | frame+utterance heads | GitHub `cwx-worst-one/EAT` |
| **M2D / M2D-X** | TASLP 2024 (arXiv 2404.06095) | 49.0 | backbone | GitHub `nttcslab/m2d` |
| **ATST-Frame** | TASLP 2024 (arXiv 2306.04186) | ~mid-40s clip | **yes** (designed dense) | GitHub `Audio-WestlakeU/audiossl` |
| Audio-MAE | NeurIPS 2022 | ~47.3 | backbone | `facebookresearch/AudioMAE` — **CC-BY-NC (non-commercial)** |

### SED with accurate timing (DCASE Task 4 lineage)

- **PretrainedSED** — DCASE 2024 Workshop / arXiv **2409.09546**, GitHub **`fschmid56/PretrainedSED`**
  (MIT). Pretrains ATST-F / BEATs / M2D on **AudioSet-Strong** frame labels →
  **native dense posteriors over 447 classes**. DCASE Task-4 **PSDS1 0.587 / PSDS2 0.812** (SOTA);
  per-encoder AudioSet-Strong PSDS1 ≈ 0.458–0.465. Tiny frame-MobileNet heads (`frame_mn06` 1.6M,
  `frame_mn10` 3.8M) for CPU. **This is the single biggest upgrade over YAMNet/AST:** sum frame
  posteriors over ontology sub-trees for per-bucket people/machine/environment time series, and use
  distinct `Speech` vs `Babble/Crowd/Hubbub` frame posteriors to (a) gate speech-vs-speech-like and
  (b) supply a **clean speech mask** to the speaker-clustering step.
- **DCASE 2024 Task-4 winner** — Multi-Iteration Multi-Stage fine-tuning, arXiv **2407.12997**,
  PSDS1 **0.692** DESED public eval (same repo family).
- **Onset-and-Offset-Aware SED** — Interspeech 2025 (ISCA `yoshinaga25`): explicit boundary-collar
  training.
- **DCASE 2025/2026 Task 4 pivoted to Spatial Semantic Segmentation (S5)** — arXiv 2506.10676 /
  2604.00776: separate + label sources from Ambisonic mixtures. Directly the "isolate then label
  background sources" problem but **multi-channel**; single-channel relevance partial. Strongest
  *single-channel* SED SOTA to benchmark against remains the 2024 PretrainedSED numbers.

### Open-vocabulary, frame-level querying (for speech-like edge cases: TV dialogue, PA, distant chatter)

| Method | Venue / Year | Metric | Availability |
|--------|--------------|--------|--------------|
| **FLAM** | **ICML 2025** (arXiv 2505.05335) | CLAP + calibrated frame-wise contrastive; strong open-vocab localization | GitHub `adobe-research/openflam` |
| **DASM** | ACM MM 2025 (arXiv 2507.16343) | +7.8 PSDS over CLAP open-vocab; DESED zero-shot **PSDS1 42.2** (beats supervised CRNN) | arXiv (code referenced) |
| **FlexSED** | 2025 (arXiv 2509.18606) | SSL encoder + CLAP text; beats vanilla SED on AudioSet-Strong | GitHub `JHU-LCAP/FlexSED` |
| **MGA-CLAP** | ACM MM 2024 (arXiv 2408.07919) | zero-shot DESED PSDS1 13.1→26.4 | GitHub (authors) |

### Near-free side-channel

- **Whisper-AT** — Interspeech 2023 (arXiv 2307.03183), `pip install whisper-at`,
  GitHub `YuanGongND/whisper-at`. Frozen Whisper + Time-and-Layer transformer emits AudioSet tags at
  segment resolution for <1% extra cost; Whisper features correlate with background non-speech type.
  Coarse timing (~seconds); a cheap corroborating vote since Whisper already runs.

---

## 3. Speech enhancement / background suppression — and its downstream cost

> **For the off-target objective, this section is a set of *risks* to avoid, not tools to adopt.**
> Enhancement suppresses the background voices we want to detect. It remains relevant only for the
> transcription/quality axes and must never gate off-target detection.

**The dominant finding.** Enhancement that improves signal quality routinely **degrades** downstream
ASR (distribution mismatch), and any model capable of *recovering* masked/quiet content is the same
class that **hallucinates**:

- **When Denoising Hinders** — arXiv **2603.04710** (2026): SAM-Audio denoising raised PSNR
  32.3→36.0 dB yet Whisper WER **worsened across every size/language** (EN Whisper-base 10.5→21.7%;
  BN large-v3 65.8→77.4%). Cause: enhanced audio leaves Whisper's pretraining distribution.
- **URGENT 2024 / 2025 challenges** — arXiv 2506.01611, **2505.23212** (Interspeech 2025), P.808
  2507.11306: a purely **discriminative** sub-band RNN won overall; **purely generative models
  hallucinated content at low SNR** and were language-biased (worse on unseen langs). Content
  accuracy (ASR-based CAcc) + Levenshtein Phone Similarity were explicit metrics.

### Enhancer taxonomy (for the transcription/quality passes only)

- **Discriminative (safe for timing, ~zero invention):** DeepFilterNet3 (`pip deepfilternet`, MIT —
  deterministic mask, phase/onset preserved, real-time CPU); MP-SENet (Interspeech 2023, arXiv
  2308.08926, explicit phase decoders); GTCRN (ICASSP 2024, 23.7K params); FRCRN (ICASSP 2022);
  xLSTM-SENet / MambAttention (arXiv 2507.00966, better OOD).
- **Generative / diffusion (fidelity ceiling, hallucination risk):** SGMSE+/StoRM
  (HF `sp-uhh/speech-enhancement-sgmse`, MIT; StoRM's regenerate-from-estimate cuts invention);
  GenSE (ICLR 2025, arXiv 2502.02942); Genhancer (Interspeech 2024); DiTSE (arXiv 2504.09381);
  **PASE** (arXiv 2511.13300) — WavLM phonological prior *constrains* generation → low hallucination;
  LLaSE-G1 (ACL 2025, arXiv 2503.00493).
- **Restoration / universal (max recovery, max invention):** Miipher-2 (WASPAA 2025, arXiv
  2505.04457, RTF 0.0078; **no official weights**, community `yukara-ikemiya/Open-Miipher-2`);
  `ResembleAI/resemble-enhance` (MIT); Demucs denoiser (`facebookresearch/denoiser`, MIT).

### Target-speaker extraction / separation (relevant only if you ever want per-target *cleanup*, not detection)

USEF-TSE / USEF-TFGridNet (arXiv 2409.02615, SI-SDRi 23.2 dB SOTA); MTSE (Interspeech 2025, multi-
target); onset-prompted Listen-to-Extract (arXiv 2505.05114, no enrollment DB); TargetVoice
(Interspeech 2025, low-latency). **Note:** TSE outputs are aggressively masked → drop low-energy
frames and shift boundaries; and for our objective, extracting "the target" presupposes an anchor
we don't have.

**Open literature gap:** no primary source quantifies enhancement-induced **word-boundary shift** at
scale. Treat as a measurement to make, not a number to cite. Practical rule: **align on the same
audio you report timings from**, and keep raw-vs-enhanced boundary disagreement as an uncertainty signal.

---

## 4. Word/turn timestamps, forced alignment & short-word recovery

**Anti-omission (the dropped-name / "Josh" problem).**

- **CrisperWhisper** — Interspeech 2024, arXiv **2408.16589** (DOI 10.21437/Interspeech.2024-731).
  Verbatim fine-tune + DTW over cross-attention with punctuation/whitespace tokens removed, pauses
  capped 160 ms, <50 ms tokens deleted. **Insertion/omission-proxy IER 2.26 vs vanilla Whisper
  11.77** (AMI); word-seg F1 84.7 / mIoU 63.4 @0.2 s collar (> WhisperX 76.7/61.5). HF weights
  `nyralabs/CrisperWhisper` are large-v3-based, **CC-BY-NC-4.0** (non-commercial — flag for any
  commercial deployment). **Verbatim training is itself a strong anti-omission fix.**
- **Contextual biasing / hotword lists** (the direct fix): BR-ASR (Interspeech 2025, up to 200K
  bias entries); OWSM-Biasing (arXiv 2506.09448); cheapest path = Qwen3-ASR context field +
  Whisper `initial_prompt` with a per-file name/vocab list. Especially valuable for **free speech**
  (open vocabulary). Post-hoc NE recovery: arXiv 2506.10779; EMNLP 2025 `2025.emnlp-main.1052`.
- **Max-recall union** across CrisperWhisper + Qwen + Canary (all already run) before alignment —
  recovers words any single model drops. **Do not** raise silence/no-speech thresholds.

**Forced-alignment SOTA (hard numbers, word-level % within tolerance):**

| Aligner | @≤10 ms | @≤50 ms | @≤100 ms | Notes |
|---------|---------|---------|----------|-------|
| **MWA** (arXiv **2606.10675**, `MLSpeech/Multilingual-Word-Aligner`, CC-BY-4.0) | **58.0** | **91.6** | 97.8 | Learned DP on **MMS reps we already load**; no G2P; best measured |
| MFA | 41.6 | 89.4 | 97.4 | HMM-GMM; still competitive (survey arXiv 2606.18466) |
| WhisperX | 22.4 | 82.4 | 94.2 | *below* MMS-DP at tight tol.; known issue #1247 |
| MMS (current) | 18.6 | 75.7 | 94.7 | our present path |
| Canary-1B | 9.2 | 44.2 | 72.8 | native timestamps are **coarse** — weight below MMS/MWA |

(TIMIT; Buckeye shows the same ordering.) **FALCON** (arXiv 2606.25460, same lab, CC-BY-4.0) gives
end-to-end differentiable **phone-level** boundaries (TIMIT @≤50 ms ≈ 94.9% vs MFA 81.1%) —
complements the PPG-vs-ASR PER axis.

**Timestamp-native ASR & joint diar+ASR:** Word-level timestamp distillation into Canary
(Interspeech 2025, arXiv 2505.15646, 20–120 ms error); Canary-1B-v2 / Parakeet-TDT-v3 (arXiv
2509.14128, NFA+CTC timestamps); **Sortformer** (ICML 2025, arXiv 2409.06656) + **Streaming
Sortformer** (Interspeech 2025, arXiv 2507.18446), HF `nvidia/diar_streaming_sortformer_4spk-v2`
(CC-BY-4.0) — ms-level, arrival-ordered turn boundaries; **TagSpeech** (arXiv 2601.06896) E2E
multi-speaker ASR+diar with word-level who-said-what-when (metrics unverified).

> Caveat that matters here: our best measured turn boundary (Qwen ~68 ms) is already competitive at
> ≤100 ms, but **all methods degrade sharply below ~50 ms** — only MWA/FALCON materially help there.
> If the goal is *not dropping words*, biasing + verbatim + union dominate; alignment choice is secondary.

---

## 5. Design direction — unsupervised off-target-speaker axis (no anchor)

Because there is **no enrollment**, the axis estimates the target *internally* and scores novelty.
All signals computed on **raw** audio; output a per-window `off_target ∈ [0,1]`.

**Estimating the de-facto target (within-file, unsupervised):**
- Cluster per-window ECAPA embeddings (senselab already does this); take the **dominant cluster**
  (largest total duration) as the pseudo-target. For a genuine single-speaker file this is the
  whole recording and `off_target ≈ 0` everywhere — the clean, expected case.
- **Mis-anchor risk:** if an intruder speaks a lot, the "dominant" cluster may not be the intended
  subject. Mitigation: report *presence of >1 speaker* as the primary flag and dominant-deviation as
  secondary; surface the raw cluster structure so a human can adjudicate. (Without an anchor,
  "someone other than the majority speaker is present" is the honest, defensible claim.)

**Fused per-window signals (enrollment-free):**
1. **Embedding novelty** — `1 − cosine(window, dominant_centroid)`, with a running/global centroid;
   the short **0.5 s** embedding window (already the default after this branch's tuning) is correct
   here — it catches *brief* intrusions a 1.0 s window would smear into the target.
2. **Overlap** — pyannote powerset `segmentation-3.0` overlap posterior (already wired): target +
   other simultaneously.
3. **Scene background-voice** — PretrainedSED frame `Babble/Crowd/Conversation/Chatter` posteriors:
   flags background humans even without clean overlap; **no anchor needed**.
4. **Speaker-change/novelty points** — embedding-distance change detection between adjacent windows
   (elevated identity-uncertainty already localizes these; verified on the validation clip: near-
   boundary uncertainty 0.84 vs 0.55 baseline once the window was 0.5 s).
5. **[Read speech only] transcript deviation** — words/regions not matching the known reference
   passage, or alignment-confidence collapse, independently flag extraneous speech. Free speech
   drops this signal but keeps 1–4.

**What NOT to do for this axis:** don't run it on the enhanced pass; don't use TSE "target
extraction" (presupposes an anchor); don't rely on the dedicated diarizers alone on very short files
(pyannote and sortformer both collapsed to 1 speaker on the 5 s validation clip — they need longer
context; read/free-speech files are typically minutes, where they behave far better, so re-evaluate
them at realistic durations before discounting).

---

## 6. Validation-clip findings recap → which SOTA lever addresses each

Measured on `audio_48khz_mono_16bits.wav` (5 s, 4 speakers, noisy — a deliberate stress test, *not*
the typical single-target file) against Label-Studio ground truth:

| Observation | Root cause | SOTA lever |
|-------------|-----------|------------|
| Presence never dips in the 0.4 s gap (MAE ~2.6 s) | Frame-VAD hysteresis/smoothing, no boundary objective | Boundary-aware inference (2601.04178) + collar/PSDS metrics; VAP (2401.04868) gap arbitration; MarbleNet v2.0 / TEN VAD sharper offsets |
| Embedding clustering over-splits recurring speaker (5 vs 4) | Silhouette k-selection + noisy VAD speech-mask | Cleaner **frame-level** speech mask from PretrainedSED `Speech` vs `Babble`; (for off-target, count matters less than "≥1 non-dominant speaker present") |
| Identity axis was **inverted** at 1.0 s window | Window wider than turns → adjacent buckets look *more* similar at change | Fixed this branch: 0.5 s window → near-boundary 0.84 vs 0.55, peaks within 14–16 ms |
| Both neural diarizers collapse to 1 speaker | Clip far shorter than their training context | Re-evaluate at realistic (minutes-long) read/free-speech durations; Streaming Sortformer for short turns |
| Qwen drops short quiet word ("Josh") in raw mode | Omission (silence/LM prior); enhancement can't safely recover | Contextual biasing + verbatim CrisperWhisper + max-recall union; **not** generative recovery |

---

## 7. License & availability notes

- **Clean license, ready today:** CED-base, Dasheng (Apache-2.0); PretrainedSED, BEATs, Silero,
  DeepFilterNet3, MP-SENet, Whisper-AT (MIT); MWA, FALCON, Sortformer, Canary (CC-BY-4.0).
- **Non-commercial — flag before commercial use:** CrisperWhisper weights (CC-BY-NC-4.0);
  Audio-MAE (CC-BY-NC).
- **Restricted / bespoke:** MarbleNet v2.0 (NVIDIA Open Model License — verify terms); FLAM
  (Adobe research), DASM/FlexSED/TF-Locoformer (verify).
- **Not library-usable:** Quail VAD (closed SDK); pyannote hosted precision-2 (API); Miipher-2
  (no official weights).

## 8. Verified-source index

VAD/timing: 2601.04178 · 2401.04868 · 2403.06487 · 2508.20885 · 2501.03184 · 2312.16613 ·
MarbleNet v2.0 (HF) · TEN VAD (HF/pip) · Silero (GitHub) · 2606.18094 · 2201.13148.
Scene/SED: 2212.09058 · 2308.11957 (HF `mispeech/ced-base`) · 2406.06992 (HF `mispeech/dasheng-1.2B`) ·
2401.03497 · 2404.06095 · 2306.04186 · 2409.09546 (GitHub `fschmid56/PretrainedSED`) · 2407.12997 ·
2406.08056 · 2506.10676 · 2505.05335 (GitHub `adobe-research/openflam`) · 2507.16343 · 2509.18606
(GitHub `JHU-LCAP/FlexSED`) · 2408.07919 · 2307.03183 (GitHub `YuanGongND/whisper-at`).
Enhancement: 2603.04710 · 2505.23212 · 2506.01611 · 2507.11306 · 2505.04457 · 2502.02942 ·
2504.09381 · 2511.13300 · 2503.00493 · 2308.08926 · 2507.00966 · 2409.02615 · 2505.05114 ·
HF `sp-uhh/speech-enhancement-sgmse`, `ResembleAI/resemble-enhance`, `facebookresearch/denoiser`.
Timing/alignment/ASR: 2408.16589 · 2505.15646 · 2606.10675 (GitHub `MLSpeech/Multilingual-Word-Aligner`) ·
2606.25460 (GitHub `MLSpeech/FALCON`) · 2606.18466 · 2506.09448 · 2506.10779 · 2409.06656 ·
2507.18446 (HF `nvidia/diar_streaming_sortformer_4spk-v2`) · 2509.14128 (HF `nvidia/canary-1b-v2`) ·
2601.06896 · 2501.11378.

**Unverified / flagged:** TEN VAD & Quail scalar metrics; ISCA `lin25`/`yoshinaga25`/`wang25b` numbers;
A-JEPA / PMAM / USE / OmniSep / MiDashengLM / TagSpeech numbers; PASE & Open-Miipher-2 weight
fidelity; enhancement-induced word-boundary-shift magnitude (no primary source — measure it yourself).
