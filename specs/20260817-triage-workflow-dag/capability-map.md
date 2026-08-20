# Capability map — the triage DAG against senselab as it stands

What the nine nodes of this spec need, what senselab already provides for it, and what has to be
built. Every "exists" row names a function whose source was read; where reading it left a question
that only a measurement or a decision can close, the row says so instead of guessing.

Scope: senselab-internal capability. Orchestration is out of scope and lives in `nextflow/`.

Read with [`store.md`](store.md) — several rows are "the measurement exists, the *provenance* the
store requires does not", and that distinction is where most of the work is.

---

## 1. Capability map

Legend: **OK** — usable as-is. **OK\*** — the function exists but the caller must pass a
non-default argument or perform a wrapping step, named in the row. **PARTIAL** — some of what the
node needs. **MISSING** — nothing in senselab does this.

### 1.1 ADMIT

| the design needs | senselab | status |
| --- | --- | --- |
| decode, or fail | `senselab.audio.data_structures.audio.Audio(filepath=...)`; decode is lazy, so the decoder raises on first `.waveform` access, not at construction (`Audio._lazy_load_data_from_filepath`) | OK\* — ADMIT must touch `.waveform` to make the decode happen, and catch there |
| zero frames | `Audio.waveform.shape[1] == 0` | OK |
| every sample zero | no named function. `quality_control.metrics.proportion_silent_metric(audio, silence_threshold=0.01)` is a *threshold*, not an exact-zero test, and ADMIT is threshold-free by design | MISSING (one line, but it must not be `proportion_silent_metric`) |
| constant value / DC with no variance | `senselab.audio.tasks.quality_control.metrics.signal_variance_metric(audio) -> float` | OK |
| **no** level track, clip track, band floor | — | ADMIT emits none of these; nothing to map |

`amplitude_headroom_metric` looks like the right tool for "is this file degenerate" and is not: it
**raises `ValueError`** when any sample exceeds ±1.0. A file that overshot full scale is exactly a
file ADMIT must admit and describe, not one it can measure with this.

### 1.2 PREPROCESS

| the design needs | senselab | status |
| --- | --- | --- |
| resample to 16 kHz | `senselab.audio.tasks.preprocessing.preprocessing.resample_audios(audios, resample_rate, lowcut=None, order=4) -> List[Audio]` | OK\* — but see the trap in §4.3: its anti-alias filter is designed at the *target* rate and applied to the *source*-rate signal, and it reports no overshoot |
| overshoot guard past full scale | `workflows.audio_analysis.level.clipped_fraction(waveform, threshold=0.9999) -> float`; `workflows.audio_analysis.level.true_peak_dbtp(waveform, sampling_rate) -> float` | PARTIAL — both exist but live in another **workflow**, not a task (§3.3) |
| pre-emphasis `y[n] = x[n] − 0.97·x[n−1]`, switchable | nothing | MISSING |
| `energy_envelope`: `\|x + jH{x}\|`, 4th-order Butterworth 40 Hz, `filtfilt`, dBFS | `quality_control.metrics.amplitude_modulation_depth_metric` computes `np.abs(scipy.signal.hilbert(...))` internally and **returns a scalar modulation depth** — the envelope is discarded. No function returns an envelope track | MISSING |
| rolling local percentile floor (10th pct, 3 s window, dBFS) | nothing on an amplitude envelope. Closest prior art is `workflows.audio_analysis.noise_floor.estimate_band_floor_db(...)` — a per-third-octave-band **spectral power** floor, bias-corrected, patch-aggregated. Different quantity; read it for the two lessons in §4.6 | MISSING |
| `spans`: propose / onset / offset / discard / merge | nothing. `preprocessing.silence_segmentation.pause_aware_boundaries` is a pause-*cut* planner for chunking, not a peak-picked event proposer | MISSING |
| `no_contrast` outcome when no peak reaches `K` | — | MISSING (part of the spans task) |
| `silence`: YAMNet `Silence` per 0.96 s / 0.48 s | `senselab.audio.tasks.classification.api.classify_audios(audios, model="yamnet", top_k=521)` → `List[List[Dict]]`, each window `{start, end, label_scores, win_length, hop_length}`; `label_scores` is `[{label: score}, ...]` descending. Read a score with `senselab.audio.tasks.classification.label_scores.label_scores(window)` | OK\* — **`top_k` defaults to 5** and `Silence` is not always in the top 5. `top_k=521` is required and is what `benchmarks/scripts/spaninput.py` passes |
| `level`: peak dBFS, RMS dBFS, LUFS | LUFS: `workflows.audio_analysis.level.integrated_lufs(waveform, sampling_rate) -> float` (`-inf` for silence, pads to one 400 ms gating block). RMS **dBFS** track: `workflows.audio_analysis.l1_plot.rms_dbfs_track(waveform, sampling_rate, hop_s=0.02) -> (times, levels_dbfs)`. File-level RMS in **linear** units: `quality_control.metrics.root_mean_square_energy_metric`. True peak in dBTP: `level.true_peak_dbtp` | PARTIAL — plain **peak dBFS** and file-level **RMS dBFS** have no named function; and everything usable lives in `workflows/audio_analysis/`, not a task |
| `squim`: STOI, PESQ, SI-SDR, objective head only | `senselab.audio.tasks.features_extraction.torchaudio_squim.extract_objective_quality_features_from_audios(audios, device=None) -> List[Dict[str, Any]]` with keys `stoi`, `pesq`, `si_sdr`. Refuses non-mono and non-16 kHz with `ValueError` | OK\* — **per span**, so the caller slices first (see the `extract_segments` row) and must handle short spans (§4.7) |
| the subjective head must **not** be used | `extract_subjective_quality_features_from_audios(audios, non_matching_references)` exists and requires the reference the design refuses. Simply do not call it | OK |
| `asr_crisperwhisper`: transcript + word and token edges | `senselab.audio.tasks.speech_to_text.api.transcribe_audios(audios, model=HFModel(path_or_uri="nyralabs/CrisperWhisper2.0_turbo"), language=..., device=...) -> List[ScriptLine]`. Routes on the `nyralabs/CrisperWhisper2.0` prefix to `crisperwhisper.CrisperWhisperASR.transcribe_with_crisperwhisper`. Word chunks are `ScriptLine(text, start, end, score)` in `.chunks`, sorted by start | OK\* — see §4.4: the returned chunks carry **no `timestamp_source` / `timestamp_model`**, which `fuse_word_streams` needs, and the worker is fed via `audio.save_to_file()` |
| `asr_qwen`: transcript + word timings | same entry point with `HFModel(path_or_uri="Qwen/Qwen3-ASR-...")`; routes to `qwen.QwenASR.transcribe_with_qwen(..., return_timestamps=bool, forced_aligner=...)` | OK\* — timings come from the bundled `Qwen/Qwen3-ForcedAligner`, which is a *shared* timing source with Canary-Qwen; irrelevant here since Canary is not in the design, but the field that records it matters (§4.4) |
| `alignment`: forced alignment of the agreed transcript | `senselab.audio.tasks.forced_alignment.forced_alignment.align_transcriptions(audios_and_transcriptions_and_language: List[Tuple[Audio, ScriptLine, Language]], levels_to_keep={"asr": False, "word": False, "char": False}, aligner_model=None) -> List[List[ScriptLine | None]]`. Raises if `audio.sampling_rate != 16000`. Warns on stderr when the input `ScriptLine` has no `start`/`end` | OK |
| `spectrogram_wb` 5 ms window, 5 ms hop | `features_extraction.torchaudio.extract_spectrogram_from_audios(audios, n_fft=1024, win_length=None, hop_length=None) -> List[Dict[str, Tensor]]`. Arguments are **samples**, not seconds: at 16 kHz the design's 5 ms/5 ms is `win_length=80, hop_length=80`, and `n_fft` must be chosen (≥ 80) | OK\* — `n_fft` is undetermined by the design (§5) |
| `spectrogram_nb` 20 ms window, 5 ms hop | same, `win_length=320, hop_length=80` | OK\* |
| a dB-scaled spectrogram for rendering | `workflows.audio_analysis.l1_plot.spectrogram_db(waveform, sampling_rate, n_fft=1024, hop_s=0.01, floor_db=-80.0)` — **peak-normalised**, Hann, its own FFT | OK for figures; not the analysis product |
| `gammatone`: 40 ERB channels, 80–7800 Hz, 5 ms hop | `scipy.signal.gammatone` is present (scipy 1.17.1, verified: `gammatone(1000, "iir", fs=16000)` returns a 5/9-tap IIR). No senselab function builds the bank, the ERB spacing, or the 5 ms-hop energy readout | MISSING |

### 1.3 TAXONOMY

| the design needs | senselab | status |
| --- | --- | --- |
| YAMNet, 521 labels, presence per kind | `classify_audios(audios, model="yamnet", top_k=521)` | OK\* (`top_k`) |
| AST, a second AudioSet opinion, file-level over its 10.24 s frame | `classify_audios(audios, model=HFModel(path_or_uri="MIT/ast-finetuned-audioset-10-10-0.4593"))` with `win_length=None` → `_classify_whole` → `classification.huggingface.HuggingFaceAudioClassifier.classify_audios_with_transformers(audios, model, top_k=None, function_to_apply="softmax", batch_size=16, device=None) -> List[AudioClassificationResult]` | OK\*\* — **two required overrides**: `function_to_apply="sigmoid"` (AudioSet is multi-label; the default softmax is wrong for it) and `top_k=None` to keep all 527. Both go through `classify_audios`'s `**kwargs`. And the return shape differs from every other classifier path (§4.2) |
| CrisperWhisper, the only source of words | as PREPROCESS above | OK\* |
| HeAR, strongest on breath, **barred from speech** | `senselab.audio.tasks.health_acoustics.api.detect_health_acoustic_events(audios, model="hear-event-detector", device=None, hop_length=0.25, top_k=None) -> List[List[Dict]]`; labels are `health_acoustics.hear.HEAR_EVENT_LABELS` (8, independent presence probabilities). `top_k=None` keeps all eight | OK — the bar on speech is the caller's rule, and nothing in the module enforces it |
| per-kind `min_families`, eligibility, presence/absence/undecided fold | nothing | MISSING (and the values are undecided by design — `benchmarks/open.md`) |
| write one `kind` element per kind with per-family evidence | nothing | MISSING (store, §3) |

### 1.4 AIRWAY

| the design needs | senselab | status |
| --- | --- | --- |
| classify **the whole span** with HeAR, in a 2 s buffer containing nothing else | `detect_health_acoustic_events`. It works only because a buffer of *exactly* 32000 samples makes `hear.plan_scan_windows` return `[0]`. Anything shorter is refused by `hear._require_two_seconds`, whose error text argues **against** the padding this node requires. `benchmarks/scripts/spaninput.py` does it by building a **centred** zero buffer and calling `detect_health_acoustic_events(auds, hop_length=2.0)` | PARTIAL / **conflict** — see §4.1. The mechanism is reachable; the module documents a measurement against it, and the design does not say centred vs. right-aligned. The benchmark chose centred |
| YAMNet **on its own native windows overlapping the span**, never on the span as input | `classify_audios(..., model="yamnet", top_k=521)` over the whole recording, then the caller selects overlapping windows | OK\* — the "never feed YAMNet a padded span" rule is the caller's; nothing stops it |
| coverage aggregation: fraction of overlapping windows ≥ 0.5 | nothing | MISSING (trivial, but it is the node's decision rule and belongs somewhere named) |
| the `Breathe` → {`Breathing`, `Sigh`, `Gasp`} confirmation mapping | nothing | MISSING |
| lexical contamination: any ASR word inside `[first labelled span start, last labelled span end]` | word list from `ScriptLine.chunks`, or `speech_to_text_ensemble.api.iter_word_leaves(node)` for a serialised tree | OK\* |
| `label` / `confirm` / `contest` / abstain, all retained | nothing | MISSING (store, §3) |
| the aligned figure | `senselab.audio.tasks.plotting.plotting.plot_aligned_panels(audio, panels, title="", figsize=None, spectrogram_params=None, context="auto") -> Figure` — panel types `waveform`, `spectrogram`, `features`, `segments`, `overlay_on_spectrogram`. `classification.api.scene_results_to_segments(results)` converts windowed classifier output to the `segments` shape | OK — every panel the figure needs exists |

### 1.5 SPEECH

| the design needs | senselab | status |
| --- | --- | --- |
| word agreement between two recognizers → per-word confidence | `senselab.audio.tasks.speech_to_text_ensemble.api.fuse_word_streams(word_streams: dict[str, list[dict]], *, weights=None, slot_overlap=0.3, slot_mid_tol_s=0.15, winner_margin=0.66, alternate_min_share=0.15, min_corroboration=MIN_CORROBORATION, speaker_at=None, calibrator=None, text_similarity=None, columns=None) -> list[dict]`. Returns per word `existence_confidence`, `temporal_confidence` (per edge, `None` for a single timing source), `member_agreement`, `coverage`, `alternates`, `flags`, optional `speaker`. `iter_word_leaves` builds the input streams from ScriptLine trees | **OK, and this is the strongest existing fit in the whole map.** Two caveats: it needs `timestamp_model` on each word (§4.4) and it imports from `workflows/audio_analysis/floors` (§4.9) |
| the fabrication test — a word over no energy and no periodicity | nothing. Needs the envelope + local floor (MISSING above) and a periodicity measure (MISSING, §1.6) | MISSING |
| speech spans from word timings, grouped by a gap | nothing — and **the gap threshold is undecided by design** (`benchmarks/open.md`) | MISSING |
| YAMNet `Speech` coverage per span | as AIRWAY | OK\* |
| SQUIM per span, as a test of *whether the span is speech* | `extract_objective_quality_features_from_audios` on the sliced span | OK\* |
| slice a span out of a recording | `preprocessing.preprocessing.extract_segments(data: List[Tuple[Audio, List[Tuple[float, float]]]]) -> List[List[Audio]]` — validates `start >= 0` and `end <= duration`, raises otherwise | OK |
| pyannote restricted to `[first word start, last word end]` | `senselab.audio.tasks.speaker_diarization.api.diarize_audios(audios, model=None, num_speakers=None, min_speakers=None, max_speakers=None, device=None, exclusive=True, max_new_tokens=None) -> List[List[ScriptLine]]`. `model=None` gives `PyannoteAudioModel("pyannote/speaker-diarization-community-1", revision="main")`; `pyannote.py` resolves that ref to a SHA before loading. There is **no interval argument** — the caller slices with `extract_segments` and must add the interval offset back onto the returned `ScriptLine.start`/`.end` | OK\* — the offset-shift is caller work and is where an off-by-one interval bug would live |
| a second diarizer when count ≠ 1 | same entry point: `HFModel("BUT-FIT/diarizen-...")` (weights CC BY-NC 4.0, non-commercial only), `HFModel("microsoft/VibeVoice-ASR...")`, `HFModel("nvidia/diar_sortformer_4spk-v1")` (≤ 4 speakers, fixed). `speaker_diarization.api.capabilities_for(model_id) -> DiarizationCapabilities` reports what each honours | OK — note `exclusive=` is silently ignored by every non-pyannote backend |
| MossFormer separation, one stream per speaker | `senselab.audio.tasks.source_separation.api.separate_audios(audios, model=HFModel("alibabasglab/MossFormer2_SS_16K"), n_sources=2, device=None, parameters=None, timeout_s=None) -> List[List[Audio]]`. `n_sources` must equal 2 or it raises; `mode`/`source_classes`/`seed`/`diffusion_steps` are unasdiff-only and are **refused** rather than ignored. Each output `Audio` carries `metadata["clearvoice"]` naming the model, resolved commit, source index, and the RMS scalar | OK\* — and see §4.5: this checkpoint has `rms_normalises_input=True`, so **absolute level is destroyed** |
| refuse to separate when the count is ≥ 3 | the count is fixed at 2 by `utils.clearvoice.CLEARVOICE_MODELS["MossFormer2_SS_16K"].expected_outputs`; `separate_audios` raises on any other `n_sources` | OK — the refusal is the node's, and the constant to check against is named |
| words → speakers by timing; mark a straddling word | `fuse_word_streams(..., speaker_at=callable)` attributes a speaker per word; it does not mark straddling | PARTIAL |
| speaker embeddings with model provenance | `senselab.audio.tasks.speaker_embeddings.api.extract_speaker_embeddings_from_audios(audios, model=None, device=None) -> List[torch.Tensor]` (default `SpeechBrainModel("speechbrain/spkrec-ecapa-voxceleb", revision="main")`, 192-d). Provenance: `SpeechBrainModel` inherits `HFModel.commit_sha`, resolved at construction — so `model.commit_sha` is the 40-hex commit **before** the call | OK |
| a target embedding carrying the model and revision that produced it, refused without provenance | `senselab.audio.data_structures.audio_hints.AudioHints` with `.target_speaker: TargetSpeakerEmbedding{vector, provenance: SpeakerEmbeddingProvenance, distribution}`. `SpeakerEmbeddingProvenance.model_commit_sha` has a field validator that **rejects anything not 40-hex** — a ref cannot be stored there — and `unresolved_reason` explains a `None`. `AudioHints.may_contain`, `.targeted_speaker_count`, `.environment`, `.expected_speech` cover the rest of the `hint?` port | **OK, and it is exactly the type the design asks for.** Also: `speaker_embeddings.api.estimate_speaker_embedding_from_audios(...) -> TargetSpeakerEmbedding` builds one |
| PII scan of the transcript, decision imposed by this branch | `senselab.text.tasks.pii_detection.api.scan_for_pii(inputs, detectors=None, presidio_score_threshold=0.4, gliner_model=None, gliner_labels=None, gliner_threshold=0.5, local_llm_config=None) -> PiiScan | list[PiiScan]` and `decide_pii(scans, require_cross_source_corroboration=True, n_sources=1) -> PiiReport | list[PiiReport]`. `PiiScan{spans, detectors_used, failures}` — `failures` is populated and honoured exactly as the design requires: empty `spans` + populated `failures` is "could not check" | OK for scanning and for the failure signal |
| a PII finding's **extent**, and the `word` elements it marks | **`PiiSpan` has no offsets and no times.** It is `{text, category, source, asr_model, score}` — `text` is the matched string, and `asr_model` is filled by `_materialize_spans(raw, source_id)` with the **batch index as a string** (`"0"`, `"1"`), not a model id | MISSING — locating a finding in the transcript and mapping it onto word elements is the gap, and it is load-bearing for both SPEECH's speaker-scoped rule and REDACT |
| verdict carries category and extent, **never** matched text | `PiiSpan.text` and `PiiReport.spans` both carry it; nothing strips it | MISSING (a projection, not a detector) |
| scan **both** recognizers' transcripts, each finding tagged with which one | `scan_for_pii` accepts a sequence and returns one `PiiScan` per input, so two transcripts is one batched call; `decide_pii(scans, n_sources=2)` folds them. The per-finding model tag needs the caller's index→model map | OK\* |
| SQUIM over the target speaker's spans, on the separated stream when separation ran | `extract_objective_quality_features_from_audios` on the sliced stream | OK\* — record which stream (§4.5) |

### 1.6 VOICE

| the design needs | senselab | status |
| --- | --- | --- |
| the residual as a fold over what other branches asserted | nothing | MISSING (store, §3) |
| normalised autocorrelation with an RMS floor, as a **track** | nothing. `features_extraction.torchaudio.extract_pitch_from_audios(audios, freq_low=80, freq_high=500)` returns a torchaudio `detect_pitch_frequency` contour — a pitch track, not a normalised-autocorrelation periodicity track, and it carries no periodicity value. Praat's `extract_harmonicity_descriptors(snd, floor, frame_shift)` returns `{hnr_db_mean, hnr_db_std_dev}` — **summary statistics only** | MISSING |
| `period_marks`: an ordered point process of glottal period boundaries, each with duration, amplitude and the placing peak | nothing. Praat's `PointProcess` is reachable through `parselmouth` but senselab exposes only `extract_jitter(snd, floor, ceiling)` and `extract_shimmer(snd, floor, ceiling)`, both of which return **means and std devs** — precisely the resampled summary the design says is unrecoverable | MISSING |
| `energy_track`, `periodicity_track`, `f0_candidates` on the analysis hop | F0 partially, via `extract_pitch_from_audios` or Praat `to_pitch_ac`; periodicity and energy tracks not at all | PARTIAL |
| an F0 search range that flags rather than resolves an ambiguous run | `features_extraction.praat_parselmouth.extract_pitch_values(snd)` picks the range with a **hard-coded rule**: mean pitch < 170 Hz → floor 60 / ceiling 250, else 100 / 500. It resolves the ambiguity the design says must be flagged | **actively contrary** — do not reuse it for this |
| the gate's two floors | undecided by design: periodicity in `(0.44, 0.933)`, RMS in `(0.0007, 0.0161)`, no fitted value. Nothing to map, and nothing may default one | MISSING **and must stay unset** |
| runs are elementary, never merged | — | MISSING (part of the gate task) |

### 1.7 REDACT

| the design needs | senselab | status |
| --- | --- | --- |
| mute / silence a padded time extent in audio | **nothing in senselab redacts anything.** `grep -rn 'redact'` finds only comments in `workflows/audio_analysis/global_summary.py` and `text/tasks/pii_detection/rules.py`. `data_augmentation.api.augment_audios` wraps audiomentations and does not do targeted extents | MISSING |
| pad an extent outward by a margin | nothing — and **the margin is undecided by design** (it must exceed the *worst* alignment edge error, which is unquantified) | MISSING and must stay unset |
| merge overlapping padded extents | nothing | MISSING |
| produce a redacted transcript | nothing | MISSING |
| re-run ASR and the PII scan on the node's own output | `transcribe_audios` + `scan_for_pii` again | OK\* |
| write the redacted audio | `Audio.save_to_file(file_path, format=None, subtype=None, out_of_range="raise") -> AudioWriteReport`. Default subtype for `.wav` is `FLOAT` (`utils.portable_audio_io.resolve_subtype`, `LOSSLESS_WAV_SUBTYPE = "FLOAT"`), so float samples round-trip exactly and an out-of-range write **refuses** rather than clipping | OK |
| a released artifact sharing **no element ids** with the store | nothing enforces it | MISSING (§3.4) |

### 1.8 VERDICT

| the design needs | senselab | status |
| --- | --- | --- |
| read every node's verdict, fold on two axes, record contradictions | nothing | MISSING (store, §3) |

### 1.9 Provenance — the cross-cutting row

The store requires "an element or assertion authored by a model carries the model id and resolved
revision". Where that is available today:

| model | how a caller gets the resolved commit | status |
| --- | --- | --- |
| CrisperWhisper, Qwen3-ASR, AST, wav2vec2/MMS aligners, pyannote, DiariZen, ECAPA, MossFormer2_SS_16K | `HFModel` (and its subclasses `SpeechBrainModel`, `PyannoteAudioModel`) resolve `commit_sha` in a `model_validator(mode="after")` at **construction**, from `utils.model_revision.resolve_revision`. So `model.commit_sha` is a 40-hex commit before any inference runs. Backends re-resolve internally via `utils.dependencies.resolve_model(repo_id, revision) -> (sha, snapshot_path)` | OK |
| HeAR | `health_acoustics.hear.HEAR_REVISION = "9b2eb2853c426676255cc6ac5804b7f1fe8e563f"` — a literal SHA in source. `HearEmbeddings.revision` carries it | OK |
| **YAMNet** | none. `classification/yamnet.py` loads `https://tfhub.dev/google/yamnet/1` from TF-Hub. There is no HF repo, no commit, and `classify_audios` returns no provenance field. The store would have to record the URL and the installed `tensorflow-hub` version | MISSING as structured provenance |
| **SQUIM** | none. `torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()` carries bundled weights. Provenance is the installed `torchaudio` version (2.11.0 here) | MISSING as structured provenance |
| **any of them, returned from the call** | none of `classify_audios`, `transcribe_audios`, `diarize_audios`, `align_transcriptions`, `extract_speaker_embeddings_from_audios`, `extract_objective_quality_features_from_audios` returns the model id or revision alongside its result. Only `HearEmbeddings` and `separate_audios`' `metadata["clearvoice"]` do | **PARTIAL, and it is the store's main integration cost**: the node must construct the model object, read `.commit_sha`, and stamp it itself. Nothing prevents a node from stamping a revision it did not actually load |

---

## 2. New-task specifications

Convention followed throughout: `src/senselab/audio/tasks/<name>/` with `api.py` holding the public
functions and one module per backend or per concern; Google docstrings; type hints; line length 120;
tests at `src/tests/audio/tasks/<name>/<name>_test.py`. **No rationale prose in the code** — the
measurement behind each parameter is already in `benchmarks/` and stays there.

None of these need a subprocess venv. Every subprocess-venv dependency the DAG has is reached
through an existing task (`yamnet`, `hear`, `crisperwhisper`, `qwen`, `clearvoice`, `pii_detection`),
each of which owns its own venv already. That is a deliberate property worth preserving: the new
tasks are pure DSP and pure folds.

### 2.1 `src/senselab/audio/tasks/envelope/`

Serves PREPROCESS's `energy_envelope`, its floor, and the pre-emphasis switch.

```python
# api.py
def preemphasise_audios(audios: List[Audio], coefficient: float = 0.97) -> List[Audio]: ...

def extract_energy_envelope(
    audio: Audio,
    cutoff_hz: float = 40.0,
    order: int = 4,
) -> EnergyEnvelope: ...

def rolling_percentile_floor(
    envelope_dbfs: np.ndarray,
    sampling_rate: int,
    window_s: float = 3.0,
    percentile: float = 10.0,
) -> np.ndarray: ...
```

`EnergyEnvelope` (a pydantic model or frozen dataclass) carries `values_dbfs: np.ndarray`,
`sampling_rate: int`, `cutoff_hz`, `order`, `zero_phase: bool`, and `floor_dbfs_value` — the
`1e-12` clamp, so a reader can tell a floored sample from a measured one.

Must refuse rather than guess:
- a `cutoff_hz` at or above Nyquist — `scipy.signal.butter` would accept a normalised value > 1
  and produce garbage;
- a multi-channel `Audio` — the envelope of a downmix is not the envelope of a channel, and
  choosing which is a decision;
- `window_s` shorter than one envelope sample, or a `percentile` outside `[0, 100]`.

Must **not** silently choose causal filtering. `filtfilt` is what the 63.5 ms-vs-90.1 ms measurement
bought, and it makes the envelope offline-only; a `zero_phase=False` option would be a second,
unmeasured product under one name.

### 2.2 `src/senselab/audio/tasks/spans/`

Serves PREPROCESS's `spans`, and is the only place `K`, the onset drop, the offset fraction and the
hangover appear.

```python
# api.py
def propose_spans(
    envelope_dbfs: np.ndarray,
    floor_dbfs: np.ndarray,
    sampling_rate: int,
    *,
    k_db: float,
    onset_drop_db: float = 15.0,
    offset_fraction: float = 0.7,
    hangover_ms: float,
    min_duration_ms: float = 50.0,
    min_separation_ms: float = 150.0,
) -> SpanProposal: ...
```

`SpanProposal` carries `spans: list[ProposedSpan]` and `no_contrast: bool`. `ProposedSpan` carries
`start`, `end`, `peak_time`, `peak_dbfs`, `floor_at_peak_dbfs`, `peak_over_floor_db`, and the
parameter set that produced it. **No label field** — the design is explicit that a span carries no
label, and a nullable label on the proposal type is an invitation to fill it.

`k_db` and `hangover_ms` are **keyword-only and have no defaults**, because both are per-consumer:
AIRWAY reads at `K` = 18 dB and SPEECH at 12 dB, and the hangover must be shorter than the shortest
event the consumer intends to bound. A default here would silently make one consumer's setting the
other's.

Must refuse rather than guess:
- `floor_dbfs` whose length differs from `envelope_dbfs` — the floor is a track, not a scalar, and a
  broadcast scalar reintroduces the global floor the measurement rejected;
- `hangover_ms` at or above `min_duration_ms` — the 250 ms hangover overshooting a 202 ms click by
  418 ms is a measured failure, and the rule "must observe more silence than the event lasts" is
  checkable;
- `offset_fraction` outside `(0, 1)`.

Return `no_contrast=True` rather than an empty list when no peak clears `K` anywhere. An empty list
and "there is no contrast in this recording" are different findings and the outcome tables in
`branch-airway.md` depend on telling them apart.

### 2.3 `src/senselab/audio/tasks/gammatone/`

Serves PREPROCESS's `gammatone`, which only the figure and short-transient detection read.

```python
# api.py
def extract_gammatone_energies(
    audio: Audio,
    n_channels: int = 40,
    low_hz: float = 80.0,
    high_hz: float = 7800.0,
    hop_s: float = 0.005,
) -> GammatoneBank: ...
```

Backend `scipy_gammatone.py` builds the bank with `scipy.signal.gammatone(freq, "iir", fs=...)` on
ERB-spaced centre frequencies and reads out per-channel energy on the hop grid. `GammatoneBank`
carries `energies: np.ndarray` shaped `[n_channels, n_frames]`, `centre_frequencies_hz`, `hop_s`,
`erb_spacing: bool`.

Must refuse `high_hz` above Nyquist — at 16 kHz the design's 7800 Hz ceiling is 200 Hz clear of it,
and a narrowband 8 kHz input would put it past. The refusal is the signal that
`preprocess.md`'s "a narrowband input with a 4 kHz ceiling restricts what the airway branch can
conclude" has actually happened.

### 2.4 `src/senselab/audio/tasks/phonation/`

Serves VOICE steps 2–4. This is the task where the undecided parameters live, and it must carry them
as **absent**, not as midpoints.

```python
# api.py
def extract_periodicity_track(
    audio: Audio,
    hop_s: float = 0.005,
    window_s: float = 0.040,
    f0_min_hz: float,
    f0_max_hz: float,
) -> PeriodicityTrack: ...

def gate_voiced_runs(
    periodicity: PeriodicityTrack,
    *,
    periodicity_floor: float,
    rms_floor: float,
) -> list[VoicedRun]: ...

def extract_period_marks(audio: Audio, run: VoicedRun) -> PeriodMarks: ...
```

`PeriodicityTrack` carries `periodicity`, `rms`, `f0_candidates_hz`, `hop_s`, and the search range —
F0 travelling with the periodicity that placed it, so a reader cannot separate them.

`PeriodMarks` carries an ordered `boundaries_s: list[float]` plus, per interval, `duration_s`,
`amplitude`, `placing_peak`. It has **no jitter and no shimmer field**: those are what a consumer
computes from the marks, and the design puts them out of scope precisely because a summary statistic
is not recoverable back into a point process.

`VoicedRun` names its offset criterion explicitly — `onset_kind="observed_period"`,
`offset_kind="gate_release"` — because the design insists the two edges are not the same kind of
quantity.

Must refuse rather than guess:
- `periodicity_floor` and `rms_floor` are **keyword-only with no defaults**. There is no fitted value
  and a midpoint of `(0.44, 0.933)` would be an invented decision. The function signature is where
  that stays visible;
- `f0_min_hz` / `f0_max_hz` likewise have no defaults — one range cannot serve a low adult male
  fundamental and an infant voice, and `praat_parselmouth.extract_pitch_values`' hard-coded
  `mean < 170 → 60/250` rule is the exact mistake not to repeat;
- merging adjacent runs: not offered at all. No merge criterion has been measured, so there is no
  parameter to expose.

The gate should additionally report, per run, whether it sits near the interval's edge — VOICE's
`flag` condition "the gate's parameters are still un-derived and a run sits near the interval's edge"
needs that to be computable.

### 2.5 `src/senselab/audio/tasks/redaction/`

Serves REDACT.

```python
# api.py
def plan_redactions(
    extents: Sequence[tuple[float, float]],
    *,
    padding_ms: float,
    duration_s: float,
) -> list[tuple[float, float]]: ...

def redact_audio(audio: Audio, extents: Sequence[tuple[float, float]], mode: str = "silence") -> Audio: ...

def redact_transcript(words: Sequence[Mapping[str, Any]], redacted: Sequence[tuple[float, float]]) -> list[dict]: ...
```

`plan_redactions` pads outward and merges overlaps — the "audible sliver between two separately
redacted words" failure the design names. `padding_ms` is **keyword-only with no default**: it must
exceed the worst measured alignment edge error, and that distribution has not been measured
(`benchmarks/open.md`).

Must refuse rather than guess:
- a `padding_ms` of zero or negative — the whole point is a conservative outward margin;
- an extent outside `[0, duration_s]`, and an extent whose end precedes its start;
- `mode` values other than the one that is actually implemented. "Silence" and "noise fill" and
  "beep" are three different claims about what a listener can infer from a redacted region, and only
  one should exist until someone has a reason for a second.

`redact_audio` must return audio whose provenance says it is redacted, and must **not** be given a
path back to the store (§3.4).

### 2.6 Extensions to existing tasks, rather than new tasks

Three gaps are better closed inside the module that owns the concept than in a new task:

**`text/tasks/pii_detection`: character offsets on `PiiSpan`.** The subprocess backend already knows
where in the text it matched (`rules.py`'s comment about window-relative offsets redacting the wrong
thing says so explicitly). Adding `start_char` / `end_char` to `PiiSpan`, and a helper that maps
character offsets onto a `ScriptLine` word tree, is the single highest-leverage change in this map:
without it neither SPEECH's speaker-scoped rule nor REDACT can locate a finding. Pre-alpha, so this
is a field addition, not a parallel type.

**`text/tasks/pii_detection`: rename or repurpose `PiiSpan.asr_model`.** It currently receives the
batch index as a string. The design requires the recognizer identity per finding; a caller-supplied
`source_ids: Sequence[str]` on `scan_for_pii` would put the real value there.

**`audio/tasks/speech_to_text`: populate `timestamp_source` and `timestamp_model`.** The fields exist
on `ScriptLine`, `fuse_word_streams` groups by them, and `crisperwhisper.py` sets neither. Until it
does, every CrisperWhisper word is treated as its own independent timing source (the conservative
reading, but not the true one).

---

## 3. The element store

### 3.1 It is not a task

senselab's task convention is a stateless function over `Audio` plus a model spec, returning a typed
result. The store is the opposite of that in every dimension: it is mutable-by-append, it outlives a
single call, it is read by nodes that did not write it, and it has no model. Making it a task would
mean a task whose `api.py` exposes `write` and `read`, which no other task does and which would make
`tasks/` mean two different things.

### 3.2 Split it: mechanism in `utils/`, vocabulary in the workflow

**Mechanism → `src/senselab/utils/element_store.py`.** `Element`, `Assertion`, `ModelProvenance`,
the append-only writer, the fold. None of it touches `Audio`; all of it is domain-neutral, and the
same primitives would serve a video or text DAG unchanged.

Put it directly under `utils/`, not under `utils/data_structures/`. Every leaf utility with no
senselab-internal dependencies already lives there — `compatibility.py`, `portable_audio_io.py`,
`clearvoice.py`, `subprocess_venv.py`, `model_revision.py` — and `utils/data_structures/__init__.py`
pulls in enough that importing through it has already caused circular-import trouble in this repo.

**Vocabulary → `src/senselab/audio/workflows/triage/`.** The `kind` values (`span`, `word`,
`speaker`, `interval`, `measurement`, `kind`, `stream`, `pii`, `target_match`), the verbs' meanings
for *this* graph, each node's `verdict` type, and the folds each node intends. These are this
design's vocabulary, not senselab's, and `verdict.md`'s resolution table is a decision only this
workflow makes.

Precedent for exactly this split already exists: `audio/data_structures/audio_hints.py` imports
`EmbeddingDistribution` from `utils/tasks/embedding_distribution.py` and its comment explains why —
the leaf is generic, the composition is domain-specific.

`ModelProvenance` should reuse the constraint `SpeakerEmbeddingProvenance` already enforces: a
`model_commit_sha` field validated as 40-hex, plus a required `unresolved_reason` when it is `None`.
That validator is the mechanism that makes "recording a SHA while loading through a ref" —
`CLAUDE.md`'s stated worst outcome — unrepresentable rather than merely discouraged.

### 3.3 A note on where the level and floor code currently lives

`integrated_lufs`, `true_peak_dbtp`, `clipped_fraction`, `rms_dbfs_track`, `spectrogram_db` and
`estimate_band_floor_db` all sit in `src/senselab/audio/workflows/audio_analysis/`. A new workflow
importing from a sibling workflow is a coupling nobody wants, and copying the functions is worse.
The clean move is to lift the level primitives into a task — `audio/tasks/level/` — and have both
workflows import that. That is a refactor of existing code, so it is called out here rather than
buried in §2, but it is on the critical path for PREPROCESS's `level` derivative.

### 3.4 It holds PII, and that changes how it is persisted

Once SPEECH writes a transcript the store is a sensitive artifact, permanently, because it is
append-only. Consequences for persistence, all of them stated in `redact.md` or following directly
from it:

**Format.** JSONL, one element or assertion per line, appended and flushed. Append-only is the
data structure's whole contract, and a format that requires rewriting the file to add a record
(a single JSON object, a parquet file with a fixed schema) breaks it. It also gives crash-safety for
free: a truncated final line is a detectable partial write.

**Location.** The store must be written to a directory that is **not** the one REDACT writes its
artifacts to, and the two must not be siblings that a single "publish this folder" step would sweep
up together. `release: releasable` never applies to the store.

**Never in a content-addressable cache.** `artifacts/analyze_audio_cache/` entries are keyed by
content and reused across runs and hosts. A transcript there is PII in a shared cache with no owner
and no expiry. Whatever caching the new workflow adopts, transcript-bearing elements must be outside
it — or the cache must be per-subject and disposed with the subject's data.

**No id sharing with released artifacts.** REDACT's artifacts must carry their own id namespace,
because an id that indexes both the store and a released artifact is a join key back to the PII. This
is easy to get wrong precisely because id-carrying is *correct* for the figure and the view inside
the store. Practically: the store writer and the artifact writer should not be the same object, and
the artifact writer should not be able to reach a store id.

**Projection, not filtering, on the way out.** A `pii` element must be constructed by taking
`category` and extent from a `PiiSpan` and dropping `.text`. A "redact on read" filter over a record
that still contains the text is one code path away from leaking it; a type that never had a text
field cannot.

**The figure is in scope for this.** SPEECH's figure renders words. A PII marking on a `word`
element has to reach the renderer, or the figure republishes what the scan just found.

---

## 4. Traps

### 4.1 HeAR refuses the padded input AIRWAY is specified to give it

`health_acoustics.hear._require_two_seconds` raises on anything under 32000 samples, with an error
that argues zero-padding a 0.3 s event to 2 s "moves its embedding as far as substituting unrelated
audio (centred cosine 0.0-0.5 against a class margin of ~0.9)". `branch-airway.md` specifies exactly
that padding. Both can be true — the refusal was measured on the **encoder**'s embeddings, and
`benchmarks/hear-yamnet.md` measured the **detector**'s labels on padded spans and got
`Breathe` 0.989 / `Cough` 0.996 — but an implementer reading only the module will conclude the design
is wrong, and an implementer reading only the design will not know the module disagrees.

What actually works, from `benchmarks/scripts/spaninput.py`: build a 32000-sample buffer, place the
span **centred** in it, wrap in `Audio(waveform=buf[None, :], sampling_rate=16000)`, and call
`detect_health_acoustic_events(auds, hop_length=2.0)`. At exactly 32000 samples `plan_scan_windows`
returns `[0]`, so one window is scored and no padding check fires. This is a length coincidence
holding the path open; it deserves a named function rather than being rediscovered per call site.
The design does not say centred, and the benchmark chose centred — see §5.

### 4.2 Two incompatible classifier return shapes, and the wrong activation for AudioSet

`classify_audios` returns `List[AudioClassificationResult]` in whole-audio mode and
`List[List[Dict]]` with `label_scores` in windowed / YAMNet / HeAR mode. So AST (whole-audio,
file-level, as the design requires) comes back in the shape with parallel `labels` and `scores`
lists — which is the shape `classification/label_scores.py`'s module docstring exists to argue
against. A caller reading all four detectors must handle both.

Worse: `classify_audios_with_transformers` defaults to `function_to_apply="softmax"`. AudioSet is
multi-label and AST's head is trained for independent presence; softmax makes 527 scores sum to 1 and
makes the design's 0.5 threshold meaningless. Pass `function_to_apply="sigmoid"` explicitly.

And `top_k`: `None` keeps everything for the HF path, but the YAMNet path is `top_k or 5`, so an
omitted `top_k` silently truncates to five labels — and `Silence` is not always in the top five.
`top_k=521`.

### 4.3 `resample_audios` is not the resampler PREPROCESS describes

`preprocessing.resample_audios` designs its anti-alias filter with
`signal.butter(order, _lowcut, btype="low", output="sos", fs=resample_rate)` — `fs` is the **target**
rate — then applies it with `sosfiltfilt` to the signal at its **source** rate, before handing off to
`speechbrain.augment.time_domain.Resample`. For 48 kHz → 16 kHz the default `lowcut` of 7900 Hz is
therefore realised at roughly 23.7 kHz on the input, i.e. the explicit pre-filter is close to
inert; whatever band-limiting happens comes from SpeechBrain's own sinc interpolation. That may be
adequate in practice, and I have not measured it. What settles it: run a 48 kHz sweep or a 48 kHz
band-limited-noise input through `resample_audios(..., 16000)` and look for energy folded back below
8 kHz. Until then, PREPROCESS's "integer decimation from 48 kHz" is not what this function does.

Separately, it reports no overshoot. `preprocess.md` asks for a guard against overshoot past full
scale, and `benchmarks/preprocess-params.md` records 0.9648 → 0.9593 on the reference file — i.e. the
guard is worth having and was not needed there. `resample_audios` returns a bare `Audio`, so the
caller must measure the peak itself (`level.true_peak_dbtp`, `level.clipped_fraction`) and must not
use `quality_control.amplitude_headroom_metric`, which raises on the very condition being tested.

### 4.4 Word timing provenance is declared but not populated

`ScriptLine` has `timestamp_source` (`native` / `bundled_aligner` / `external_aligner`) and
`timestamp_model`, and `fuse_word_streams._temporal_agreement` groups members by `timestamp_model`
falling back to `timestamp_source`, "treating a member declaring neither as its own source". But
`crisperwhisper.CrisperWhisperASR.transcribe_with_crisperwhisper` constructs its word `ScriptLine`s
with `text`, `start`, `end`, `score` and **nothing else**. So today the two recognizers group apart
by default — which happens to be correct for this design (CrisperWhisper native vs. Qwen's bundled
aligner) but for the wrong reason, and it will stop being correct the moment a third stream arrives
whose timings come from `align_transcriptions`.

Also: `crisperwhisper.py` writes its worker input with `audio.save_to_file(path)`. That now resolves
to `FLOAT` for `.wav` (`portable_audio_io.LOSSLESS_WAV_SUBTYPE`), so the stale comment in
`classification/yamnet.py` claiming that path writes `PCM_16` is wrong. But
`model-to-branch.md` records an unexplained discrepancy — two runs of the same pinned
CrisperWhisper revision on nominally the same audio producing different non-speech token sets, one
via "senselab-resampled 16 kHz copy" and one via a "raw recording" path. The load/resample/write path
is the prime suspect and it is unresolved. Do not treat CrisperWhisper's non-lexical tokens as stable
until it is.

### 4.5 `MossFormer2_SS_16K` destroys absolute level

`utils.clearvoice.CLEARVOICE_MODELS["MossFormer2_SS_16K"]` has `rms_normalises_input=True`: upstream's
reader RMS-normalises the input to −25 dBFS before decoding, and senselab reproduces that faithfully.
`benchmarks/separation.md`'s "identical RMS to two decimals (−36.94 dB) is a global-normalisation
coincidence" is this. Consequences:

- any dBFS-referenced measurement on a separated stream — the envelope, its floor, `level`, clipping
  — is **not** comparable to the same measurement on the recording;
- SQUIM's SI-SDR is scale-invariant and STOI is largely so, so §8's quality readings survive; the
  level readings do not;
- the scalar is recoverable: each returned `Audio` carries `metadata["clearvoice"]` naming the model,
  the resolved commit, the source index, and "the RMS scalar that was **not** applied to it".

The design's rule that "every measurement taken on a stream records which stream it came from" is
therefore not bookkeeping — without it a level comparison between a stream and the recording is
silently a comparison of two different normalisations.

### 4.6 Two lessons from `noise_floor.py` that apply to the envelope floor

`workflows/audio_analysis/noise_floor.py` computes a different quantity (per-band spectral power) but
its module docstring records two results the design's rolling 10th-percentile envelope floor should be
checked against:

- **a `q`-quantile of exponentially distributed noise power sits a calculable factor below the
  mean — about 9.8 dB for a tenth percentile.** Uncorrected, every relative-dB gate built on it is
  that much more permissive, and "the failure looks like generosity rather than a bug". The design's
  envelope is *amplitude*, not power, so the factor differs; that it is nonzero does not.
- **a single time-frequency bin's log-power has a ~5.6 dB spread**, so a few-dB threshold on one bin
  is meaningless; over a ~1 s patch it falls below a few tenths of a dB. The design's 3 s window is
  comfortably in the aggregated regime, which is a point in its favour worth knowing.

Also relevant: `benchmarks/preprocess-params.md` records the envelope reaching **−225 dB** inside
YAMNet-certified silence (exact zero samples), so a min or a mean over that region is meaningless and
the `1e-12` clamp in the envelope is load-bearing, not cosmetic.

### 4.7 SQUIM has no minimum-length guard

`extract_objective_quality_features_from_audios` catches `RuntimeError`, assigns `nan` to all three
metrics, and then **re-raises**. So a span too short for the model produces an exception, not a
`nan`. `benchmarks/scripts/speech2.py` worked around it by padding anything under `fs // 2` to 0.5 s
before calling `SQUIM_OBJECTIVE` directly. The design's spans run 350–1410 ms, so most are fine, but
SPEECH derives its own spans from word timings and a single-word span can be far shorter. Decide the
minimum-length policy in the caller and record it, because the padding changes the measurement.

Its two hard refusals are useful and should not be worked around: non-mono and non-16 kHz both raise
`ValueError`.

### 4.8 Revision pinning, and the guard test that enforces it

`src/tests/utils/revision_pinning_guard_test.py` is an AST sweep, not a text search, and it is
adversarial about it: `test_revision_resolved_subprocess_files_still_resolve_before_sending` requires
an *executable* `resolve_revision(...)` call or `.commit_sha` access, because comments and string
literals are invisible to `ast` and a reverted implementation with an intact explanatory comment
would otherwise pass.

Two allowlists, and every subprocess backend must be in exactly one:

- `REVISION_RESOLVED_SUBPROCESS_FILES` — the worker payload carries a resolved SHA. Includes
  `text/tasks/pii_detection/subprocess_backend.py` and `audio/tasks/speech_to_text/qwen.py`, both on
  this DAG's path.
- `LOADER_CANNOT_PIN_SUBPROCESS_FILES` — the upstream loader takes no revision, so the parent stages
  the commit and hands over a `snapshots/<sha>` path. Includes
  `audio/tasks/speech_to_text/crisperwhisper.py`, `audio/tasks/health_acoustics/hear.py`,
  `audio/tasks/speaker_diarization/diarizen.py`, `utils/clearvoice.py`.

`test_no_unreviewed_subprocess_revision_payload` fails on any **new** subprocess file whose
parent-side dict literal carries a `revision`-ish key until it is reviewed and allowlisted. So a new
subprocess backend added for this DAG will break the suite until someone classifies it — which is the
point. The sweep's blind spot, documented in the test: it only sees files that *carry* a revision key,
which is why the second list is enumerated rather than left empty.

`classification/yamnet.py` is in neither list and does not trip the sweep, because
`hf_load_coverage_test._subprocess_worker_files` only discovers workers that load an **HF** model and
YAMNet loads a TF-Hub URL. Its "pin" is the `/1` in `https://tfhub.dev/google/yamnet/1`. Do not read
its absence from the allowlists as review.

### 4.9 Two existing couplings to be careful not to extend

`speech_to_text_ensemble/api.py` — a **task** — imports `MIN_EVIDENCE_WEIGHT` from
`senselab.audio.workflows.audio_analysis.floors`, and its docstring defends the choice at length. The
new workflow will want `fuse_word_streams`, which means the new workflow depends on a task that
depends on the *old* workflow. It works, and it is worth noticing before adding a second such edge.

`workflows/audio_analysis/` is also where `integrated_lufs`, `true_peak_dbtp`, `clipped_fraction`,
`rms_dbfs_track` and `spectrogram_db` live. See §3.3.

### 4.10 Miscellaneous

- **`pytest -n auto` is prohibited.** Each xdist worker imports torch + transformers + speechbrain
  independently (535 MB resident before any test runs) plus its own copy of any weights it loads; it
  has exhausted a 32 GB machine, and on macOS CI three workers stalled simultaneously and the job hung
  for 5.5 hours. Run the directory you changed, serially.
- **`uv sync` is subtractive** — always pass `--all-extras`.
- **Cache schema.** `utils/tasks/cached_inference.CACHE_SCHEMA_VERSION` is currently `23`, and
  bumping it is the intended way to invalidate rather than reasoning about which
  `artifacts/analyze_audio_cache/` entries survive. Keys have been commit-aware since schema 23. If
  the triage workflow reuses that cache, note §3.4: it must not carry transcripts.
- **`requires_compatibility`.** `classify_audios`, `transcribe_audios`, `diarize_audios`,
  `align_transcriptions`, `extract_speaker_embeddings_from_audios`,
  `detect_health_acoustic_events` and `extract_features_from_audios` are wrapped in
  `@requires_compatibility(...)`, which runs `check_compatibility` **at call time** and can raise on
  a dependency or version mismatch. `extract_objective_quality_features_from_audios` and
  `separate_audios` are not wrapped. A node's failure mode therefore differs by which function it
  calls.
- **`SENSELAB_STRICT_HEAD_LOAD`.** `classification/huggingface.py` raises `RuntimeError` by default
  when a checkpoint leaves classifier-head weights randomly initialised — the ~uniform-softmax silent
  failure. Do not set it to `0` to make AST load; if AST trips it, the head is not loading.
- **`diarize_audios(exclusive=...)` is silently ignored** by every non-pyannote backend, and no
  warning is raised because it defaults to `True` and an explicit pass is indistinguishable from the
  default. SPEECH's second-diarizer step must not assume it got a partition.
- **DiariZen weights are CC BY-NC 4.0**, non-commercial only. The code is MIT.
- **`align_transcriptions` warns on stderr, and prints rather than logs**, when the input
  `ScriptLine` carries no `start`/`end` — and then aligns over the whole recording. On a 14 s file
  with 1.6 s of speech that is a bad alignment, not an error.
- **The MMS uppercase bug is fixed.** `forced_alignment._preprocess_segments` now probes the model
  dictionary for ASCII case (`dict_uses_uppercase = has_upper and not has_lower`) and defaults to
  lower-casing for non-Latin adapters. Any note saying MMS-eng alignment yields punctuation-only
  chunks is stale.

---

## 5. Ambiguities

Questions the design does not answer, separated from the parameters `benchmarks/open.md` deliberately
leaves unset. Neither list is resolved here.

### 5.1 Undecided by design — must stay unset

Restated so an implementer does not fill them in. Every one of these should appear as a
keyword-only argument with **no default**, or as a config key with an empty derivation slot.

| parameter | node | what would settle it |
| --- | --- | --- |
| SQUIM thresholds over speech spans | SPEECH §8 | labelled quality verdicts on speech spans, from more than one recording |
| the phonation gate's periodicity floor in `(0.44, 0.933)` and RMS floor in `(0.0007, 0.0161)` | VOICE §2 | labelled voiced/unvoiced verdicts on more than one file |
| the redaction padding margin | REDACT | the alignment edge-error **distribution**, not its median |
| the word-gap threshold grouping words into speech spans | SPEECH §2 | unspecified entirely; any value is a claim about what makes one utterance |
| `min_families` per kind (airway has 3 eligible families, speech 2) | TAXONOMY | the asymmetry is stated, the values are not |
| the F0 search range | VOICE §3 | no single range serves a low adult male fundamental and an infant voice |

### 5.2 Genuinely ambiguous — the design admits more than one implementation

**How is the span placed in HeAR's 2 s buffer?** `branch-airway.md` says "a 2 s buffer containing
nothing else" and stops. `benchmarks/scripts/spaninput.py` centred it. Centred, left-aligned and
right-aligned are three different inputs, and `hear.plan_centred_windows`' own docstring records that
a 50–200 ms framing shift costs centred cosine 0.93–0.98 on the encoder — small but not zero. Pick
one, name it, and record which the benchmark numbers were measured under.

**What `n_fft` for the two spectrograms?** The design fixes window and hop in milliseconds (5/5 and
20/5) and `extract_spectrogram_from_audios` takes samples plus an `n_fft` the design never mentions.
At 16 kHz, `win_length=80` with `n_fft=80` gives 41 bins; `n_fft=256` zero-pads to 129 bins without
improving true resolution. `benchmarks/preprocess-params.md` quotes "300 Hz frequency resolution,
Hann" for the 5 ms window, which is a property of the window, not of `n_fft` — so the figure is
unaffected either way and the choice is about rendering density. It still has to be written down,
because `plot_aligned_panels` defaults to a third setting (`n_fft=256, hop_length=80,
win_length=160`) and `l1_plot.spectrogram_db` to a fourth (`n_fft=1024, hop_s=0.01`).

**Which signal does the figure render?** `preprocess.md` sends both spectrograms and the gammatone
view to the pre-emphasised signal, and the figure is specified to carry "the waveform". Pre-emphasis
is not gain-neutral — peak 0.9593 → 0.4199, RMS −6.2 dB — so a pre-emphasised waveform panel under a
dBFS envelope panel would be two different scales on one axis.

**How is a span's `peak_over_floor_db` defined once the floor is a track?** The floor varies over the
span. At the peak, over the span, or at the onset are three different numbers, and AIRWAY reports it
as "what a reader needs to discount" a span.

**When SPEECH `refine`s a PREPROCESS span, what is the overlap criterion?** "`refine`s these where
they overlap" — any overlap, a fraction, a containment? A word-derived span and an envelope span at
`K` = 12 dB will frequently overlap partially (`benchmarks/snr.md`: IoU falls to 0.17 at +10 dB SNR
while the verdict is still confidently speech), so "overlap" is doing real work.

**Does the AIRWAY lexical-contamination interval use words from one recognizer or both?** Step 3 says
"any ASR word", and the read table names `asr_crisperwhisper` only. `asr_qwen` also produces words.

**Is `no_contrast` from PREPROCESS a per-`K` finding?** `K` is per reader, so a recording can have no
contrast at 18 dB and contrast at 12 dB. AIRWAY's `fail` condition reads `no_contrast` as a property
of PREPROCESS; it is a property of a `(K, recording)` pair.

**What consumes fabrication candidates?** `benchmarks/open.md` names this: SPEECH detects them,
nothing acts on them, and the `flag` condition says "fabrication candidates survive" — survive what?

**Where do the `stream` element's measurements sit relative to the recording's?** SPEECH's product
table lists `stream` as a kind and requires every measurement to record its stream. Given §4.5, a
level measurement on a stream and one on the recording are not on the same scale. Does the store
normalise, refuse to compare, or record both with their scalars?

**Does REDACT's re-scan use the same two recognizers as PREPROCESS?** "Re-runs ASR" — one model or
both. Since a clean re-scan is explicitly the weaker check, the answer changes how weak.

**How does the target-speaker match threshold work?** SPEECH §6 says a match happens "only when the
hint supplies a target embedding" with provenance, and `flag`s when "a target was given with
provenance and no speaker matches". No similarity threshold appears anywhere, and
`estimate_speaker_embedding_from_audios` returns a distribution precisely so a caller can set one.

**How does VERDICT read a node that did not run?** The resolution table covers `fail` / `pass` /
`flag` against each TAXONOMY state. A branch that was never invoked — because a derivative it needs
is absent from the store, which `preprocess.md` says is a legitimate state — is a fourth case, and it
is not "expected" in the sense the `absent` + `fail` row means.
