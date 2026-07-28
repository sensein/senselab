# Model landscape refresh (mid-2024 → mid-2026) — scene/events, diarization/VAD, enhancement

**Date**: 2026-07-24 · **Method**: three parallel research agents (web + HF Hub verification of every
model id/license/gated flag; claims marked *unverified* where a primary source wasn't confirmed) ·
**Anchors being reconsidered**: AST (2021) + YAMNet (2019); pyannote community-1 + Sortformer v1;
segmentation-3.0 + Brouhaha; ECAPA/ResNet embeddings; SepFormer-WHAM16k (2021).

## Cross-cutting literature findings (they validate the adaptive-loop design)

1. **"When De-noising Hurts" (arXiv 2512.17562, Dec 2025)**: enhancement preprocessing degraded WER in
   **40/40** model×noise configurations (+1.1 to +46.6 abs). Modern ASR is internally noise-robust;
   enhancement deletes acoustically useful detail. This is exactly the failure we measured on the test
   clip (SepFormer deleted words) — and confirms three choices already implemented: the raw pass stays
   authoritative; the fusion stream is chosen by utterance-evidence consistency, not by quality scores;
   enhancement is a gated auxiliary stream (`--enhancement auto`, triage FR-003).
2. **URGENT 2025 (arXiv 2505.23212)**: best universal enhancer was *discriminative*; generative SE
   shows hallucination/language-dependency risk. → Diffusion/LM restorers (Resemble, Sidon, LLaSE-G1,
   SGMSE+) must never feed ASR by default; they belong behind an explicit restoration flag.
3. **ETH diarization benchmark (arXiv 2509.26177, Sep 2025)**: sub-0.5 s segments account for <5% of
   DER; the dominant failures are boundary imprecision and **speaker-counting collapse** — precisely
   our 4-speakers→1-cluster failure. Fix = better counting models, not finer frames alone.
4. **SPEECH_PRESENCE_CERTAINTY_ANALYSIS.md** (repo handoff note) aligns: frame posteriors over
   segmentized VAD, ~100 ms reporting grid, coarse taggers as priors only — the triage round
   implemented today follows it; its "add TEN VAD backend" recommendation is seconded below.

## Front 1 — scene analysis / event tagging (replace AST + YAMNet)

| Rank | Model (verified) | Why | License | Surface |
|---|---|---|---|---|
| 1 | **`mispeech/ced-base`** (+ `-small/-mini/-tiny`) | 527-class AudioSet posteriors, mAP **50.0** (vs AST 45.9), **variable-length input** — scores 0.5–1 s windows directly, removing AST's 10.24 s floor (the D2 constraint "AST never on crops" disappears). CPU-fast at `-mini/-tiny`. | Apache-2.0 | HF transformers (`trust_remote_code`) |
| 2 | **`fschmid56/PretrainedSED`** (BEATs/ATST-Frame "strong"; `frame_mn10` 3.8M, `frame_mn06` 1.6M) | True **40 ms frame-level** event posteriors (447 AudioSet-Strong classes, PSDS1 46.5). The MobileNet heads are the YAMNet replacement (finer, better, MIT). | MIT | pip-from-GitHub, pinned release |
| 3 | `topel/ConvNeXt-Tiny-AT` (0.471 mAP, MIT) / EfficientAT `mn40_as` (0.483, license *unverified*) | Light alternates | MIT / ? | pip |

Excluded: LALMs (Qwen-Audio etc.) — free text, not calibrated tag distributions. Nothing gated.

**senselab integration**: add CED to the classification dispatcher (HF pipeline path, like AST);
PretrainedSED as a new backend module; keep the AudioSet→source-mass map (527-class compatible; the
447-class strong set needs a map extension). Adaptive-loop payoff: scene re-checks on crops at any
length; region-level `src_*` re-computation becomes a legal intervention.

## Front 2 — diarization, VAD/OSD, speaker embeddings

Diarization (short-turn/counting weighted):

| Rank | Model (verified) | Key numbers | License / gated |
|---|---|---|---|
| 1 | **`BUT-FIT/diarizen-wavlm-large-s80-md-v2`** (DiariZen) | Best open speaker-counting (ETH: 4-spk 12.7 DER); AMI 14.0 / DIHARD3 14.5 | **CC-BY-NC-4.0** (weights; research OK, no commercial), ungated |
| 2 | **`nvidia/diar_streaming_sortformer_4spk-v2`** | ~4× better 4-spk than v1 (13.2 vs 21.3); exposes **T×4 @ 0.08 s frame posteriors** via `include_tensor_outputs=True` | **CC-BY-4.0**, ungated |
| 3 | `pyannote-community/speaker-diarization-community-1` | Current default — **ungated mirror exists** (the canonical repo is soft-gated); CPU-feasible | CC-BY-4.0 |
| — | `nvidia/diar_sortformer_4spk-v1` (current) | The model that failed our clip; also CC-BY-**NC** | retire from defaults |

VAD / speech presence (triage-relevant):

| Rank | Model | Why | License / gated |
|---|---|---|---|
| 1 | **`nvidia/Frame_VAD_Multilingual_MarbleNet_v2.0`** | **20 ms frame posteriors, 91.5K params, ONNX-on-CPU recipe, ungated** — the drop-in ungated alternative to gated segmentation-3.0 for the triage speech gate | NVIDIA OML (commercial per card), ungated |
| 2 | **`TEN-framework/ten-vad`** | 10/16 ms hop, per-hop probability, ~306 KB, arm64 — the backend the SPEECH_PRESENCE note asked for | Apache-2.0 |
| 3 | `snakers4/silero-vad` v6 | Ubiquitous cross-check; hangover lag makes it a co-voter, not primary | MIT |
| — | `pyannote/segmentation-3.0` (current) | Keep for **overlap** (no ungated frame-level OSD exists — verified gap) | MIT weights, **gated** |

Speaker embeddings (change-point / short-segment — our 0.38 s failure):

| Rank | Model | Why | License |
|---|---|---|---|
| 1 | **`IDRnD/ReDimNet`** (b0–b2, non-LM) | 1–4.7M params, Vox1-O 0.52% (b2); cheap enough for 0.1 s-hop sliding windows on CPU | MIT (torch.hub) |
| 2 | **ERes2NetV2** (`iic/speech_eres2netv2_sv_zh-cn_16k-common`) | The only model with *published* short-duration EER (0.98%@3s, 1.48%@2s) | Apache-2.0 (ModelScope/ONNX) |
| 3 | `speechbrain/spkrec-ecapa-voxceleb` (current) | Keep as A/B baseline | Apache-2.0 |

Caveats (verified): **no embedding model is benchmarked <1 s** — 0.1–0.25 s hops are out-of-distribution
everywhere; rely on relative distances (as `identity_repair.py` does) and validate on labeled clips.
Avoid `-LM` checkpoints for short windows (tuned >3 s).

## Front 3 — enhancement / separation / target-speaker extraction

Enhancement (feeding ASR/diarization; identity-preservation weighted):

| Rank | Model | Key numbers | License / surface |
|---|---|---|---|
| 1 | **`alibabasglab/MossFormerGAN_SE_16K`** | PESQ 3.57 / SI-SDR 20.6 (DNS2020) — highest signal-fidelity of the verified set; native 16 kHz | Apache-2.0, `pip install clearvoice` |
| 2 | **DeepFilterNet3** | Real-time CPU (RTF ~0.04); *filtering* (no resynthesis) → architecturally safest on clean speech | MIT/Apache dual, pip |
| 3 | `alibabasglab/FRCRN_SE_16K` | Light, proven (PESQ 3.24 DNS2020) | Apache-2.0, clearvoice |
| 4 | `wyz/tfgridnet_for_urgent24` | Trained/ranked on WAcc + SpkSim + phoneme similarity — the only family optimized against our exact axes | ESPnet |
| — | `speechbrain/sepformer-wham16k-enhancement` (current) | SI-SNR 13.8/PESQ 2.20; measured deleting words on clean speech | **retire as default** |

Separation/TSE: `alibabasglab/MossFormer2_SS_16K` (2-spk separation, Apache-2.0);
`pyannote/speech-separation-ami-1.0` (PixIT — diarization-*aligned* per-speaker streams; MIT, gated);
`JusperLee/TIGER-speech` (0.82M params, CPU-feasible overlap repair); WeSep toolkit for
enrollment-based extraction (pairs with our unified clusters). Generative restorers (Resemble, Sidon,
LLaSE-G1) only behind an explicit restoration flag; Miipher-2 weights not released.

**Do-no-harm gate (new, from the literature + our measurement)**: per-bucket
`SI-SDR(raw, enhanced)` + RMS-deletion detector inside VAD speech regions + DNSMOS delta
(ClearerVoice `SpeechScore` packages these) — route the enhanced stream per bucket only when it
demonstrably improves. This formalizes the S1 guard and the fusion-stream rule; optional
observation-adding (mix ~20–30% raw back) further reduces artifact damage (Iwamoto et al. 2022).

## Adoption plan (phased, additive, registry/policy only — no behavior change until enabled)

1. **Triage hardening (now)**: MarbleNet v2 / TEN-VAD as ungated frame-posterior backends in
   `voice_activity_detection` beside segmentation-3.0 (SPEECH_PRESENCE item 1); triage prefers
   whichever is available — removes the HF-token dependency from round 0.
2. **Enhancement swap (high value)**: add MossFormerGAN_SE_16K + DeepFilterNet3 to the enhancement
   dispatcher; policy `enhancement_model`; SepFormer stays available for reproducibility. Implement the
   SI-SDR/deletion do-no-harm gate as S1's quantitative guard.
3. **Diarization defaults**: add Streaming Sortformer v2 (CC-BY-4.0 + frame posteriors → replaces v1
   and feeds the identity axis directly); DiariZen behind a research flag (NC license); keep
   community-1 via the ungated mirror.
4. **Embeddings for I1/I2**: ReDimNet-b0/b2 + ERes2NetV2 as additional `speaker_embeddings` backends;
   policy `identity.fine_hop_s` path benefits immediately (targets the unresolved 0.38 s speaker).
5. **Scene refresh**: CED (all sizes) into classification; PretrainedSED frame heads as the new
   fine-grid event voter; retire the AST-crops prohibition in contracts/region-reprocessing.md.
6. **Overlap repair (v2/U4)**: MossFormer2_SS_16K or TIGER for separation-based re-ASR; PixIT where
   diarization-consistent streams are needed.

Each addition follows the existing backend pattern (dispatcher + registry entry + pinned revision +
`ensure_hf_model`/local-files-only, constitution VI). License review flags: DiariZen weights (CC-BY-NC),
Sortformer v1 (CC-BY-NC — already in defaults today, worth an explicit review), MarbleNet (NVIDIA OML),
EfficientAT/M2D (license text unverified).
