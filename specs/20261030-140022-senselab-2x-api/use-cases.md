# senselab 2.x: use cases, models and the signal taxonomy

Research note for `specs/20261030-140022-senselab-2x-api/design.md` (sections 4.1, 4.2, 4.10, 4.12).
Licences were checked on 2026-10-30 against the GitHub licence API or LICENSE text, Hugging Face card
metadata and PyPI. Anything not confirmed from a primary source is marked **unverified**.

## 1. Use cases

| Use case | Information extracted | Signals | Task chain | Key QC needs |
|---|---|---|---|---|
| Voice as a health biomarker (B2AI-style protocols) | Phonation (f0, jitter, shimmer, HNR, CPP), articulation, prosody, speech rate, embeddings | Audio, plus a protocol's expected text | read → QC → VAD → diarize → ASR/align → features (openSMILE/Praat) → embeddings (HeAR, SSL) → per-task summary | Clipping, SNR, off-target speaker, a task that does not match its protocol; masked or redacted segments |
| Neurological conditions (PD, ALS, ataxia, stroke) | Dysarthria measures, DDK rate, pauses; tremor and gait | Audio, accelerometry, video pose | audio chain + peak/syllable detection; IMU gait and tremor spectra; pose kinematics | Sustained phonation that is *disordered* is data, not failure; device placement |
| Psychiatric conditions (depression, psychosis, mania) | Prosody variability, latency, lexical/semantic coherence, sentiment, facial affect, sleep regularity, mobility | Audio, text, video, actigraphy, phone sensing | diarize → ASR → text embeddings/coherence → turn-taking; face AUs; circadian metrics; GPS/phone features | Speaker attribution (interviewer vs participant), ASR error on atypical speech, missing-data patterns in passive sensing |
| Respiratory conditions | Cough/breath events, breathing rate, voice quality | Audio, respiration belt, PPG/ECG-derived respiration | event detection (HeAR event detector) → counts/rates; ECG-/PPG-derived respiration | Background-noise confounds; microphone type |
| Child development and child–adult interaction | Child vs adult speech, vocal maturity, turn counts, motor milestones | Day-long audio, video, actigraphy | VAD → child/adult classification → turn-taking; pose | Long-form audio, overlap, distance to microphone |
| Conversation and turn-taking dynamics | Turns, gaps and overlaps, backchannels, entrainment, dialogue acts | Audio (multi-channel), text, video | diarize → align → turn-taking (VAP/MAAI) → `text/conversation_analysis`; active-speaker detection | Cross-channel alignment, crosstalk, consistent speaker IDs across modalities |
| Affect and arousal | Categorical/dimensional emotion; physiological arousal | Audio, video, text, EDA, ECG/HRV | SER; face AUs; sentiment; EDA tonic/phasic decomposition and SCRs; HRV | Model trained on acted data; EDA motion artefacts; the same timeline across sensors |
| Speech and language assessment | WER against a prompt, disfluencies, lexical diversity, syntax, phonetic accuracy | Audio, text | ASR ensemble → forced alignment → PPGs → text metrics | Prompt mismatch; ASR uncertainty; accent |
| Multimodal clinical interviews | All of the above, fused per speaker | Video + audio + text (+ wearable) | demux → diarize + face tracks → association → per-speaker features | Clock alignment, face–voice association, PII redaction |
| Free-living monitoring | Physical activity, steps, sleep/wake, circadian rhythm, HR/HRV, mobility, phone use | Accelerometer, PPG, ECG patch, EDA, phone sensors | read device file → calibrate → non-wear → epoch → activity/sleep classifiers → daily summaries | Non-wear, calibration, timezone/DST, gaps, device clock drift |
| Sleep studies | Sleep stages, arousals, apneas, HRV during sleep | EEG/PSG (EDF), ECG, actigraphy, respiration | YASA staging; sleepecg; asleep (wrist) | Electrode impedance/flatline, lights-off markers |
| Cognitive and attention tasks | Fixations, saccades, pupil dilation | Eye tracking, EEG, events | I-VT/REMoDNaV/I2MC event detection; pupil preprocessing; ERP epochs | Blinks, calibration drift, synchronisation with stimulus events |

Three common threads across these:

1. Almost every use case ends in *per-interval summaries on one timeline*: per task, per speaker, per
   night, per day.
2. QC is not one task but a family:
   - signal-level QC, which is generic;
   - content-level QC, which is modality-specific;
   - protocol-level QC, which compares content against a dataset hint.
3. Several use cases need two or more devices whose clocks differ.

## 2. Models and algorithms per modality

Flags: **NC** = non-commercial, **CL-S/W** = strong/weak copyleft, **G** = gated.

### 2.1 Audio and speech

The current registry already covers speaker embeddings, SSL, ASR, diarization, SER, enhancement,
separation, HeAR, PPGs, SPARC and openSMILE.

| Need | Tool / model | Licence | Flags |
|---|---|---|---|
| DSP and features | librosa ([repo](https://github.com/librosa/librosa)) | ISC | — |
| DSP and features | torchaudio ([repo](https://github.com/pytorch/audio)) | BSD-2-Clause | — |
| Voice measures | praat-parselmouth ([repo](https://github.com/YannickJadoul/Parselmouth)) | GPL-3.0 | CL-S |
| Voice measures | openSMILE / opensmile-python ([repo](https://github.com/audeering/opensmile)) | LicenseRef-audEERING research licence | NC |
| Pathological speech | DisVoice ([repo](https://github.com/jcvasquezc/DisVoice)) | MIT | — |
| Pathological speech | surfboard ([repo](https://github.com/novoic/surfboard)) | GPL-3.0 | CL-S, archived: avoid |
| VAD | Silero VAD ([repo](https://github.com/snakers4/silero-vad)) | MIT | — |
| ASR | Whisper code and large-v3-turbo weights ([card](https://huggingface.co/openai/whisper-large-v3-turbo)) | MIT | — |
| ASR | parakeet-tdt-0.6b-v3 ([card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)) | CC-BY-4.0 | — |
| ASR | canary-qwen-2.5b ([card](https://huggingface.co/nvidia/canary-qwen-2.5b)) | CC-BY-4.0 | — |
| ASR | Qwen3-ASR ([card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)) | Apache-2.0 | — |
| ASR | WhisperX ([repo](https://github.com/m-bain/whisperX)) | BSD-2-Clause | — |
| Diarization | pyannote.audio code ([repo](https://github.com/pyannote/pyannote-audio)) | MIT | — |
| Diarization | speaker-diarization-community-1 weights ([card](https://huggingface.co/pyannote/speaker-diarization-community-1)) | CC-BY-4.0 | G |
| Emotion | audeering MSP-dim ([card](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)) | CC-BY-NC-SA-4.0 | NC |
| Emotion | emotion2vec+ ([card](https://huggingface.co/emotion2vec/emotion2vec_plus_large)) | FunASR model licence (terms **unverified**) | ? |
| Health acoustics | HeAR ([card](https://huggingface.co/google/hear)) | LicenseRef-HAI-DEF | G, use-restricted |
| Phonetic and articulatory features | ppgs ([repo](https://github.com/interactiveaudiolab/ppgs)) | MIT | — |
| Phonetic and articulatory features | SPARC ([repo](https://github.com/cheoljun95/Speech-Articulatory-Coding)) | MIT on PyPI only; no repo LICENSE; weights **unverified** | ? |
| Turn-taking | MAAI / VAP ([repo](https://github.com/maai-kyoto/maai)) | MIT (code); weights **unverified** | ? |
| Audio–text embeddings | LAION CLAP ([repo](https://github.com/LAION-AI/CLAP)) | CC0-1.0 code; Apache-2.0 HF weights | — |
| Audio SSL | BEATs ([unilm](https://github.com/microsoft/unilm)) | MIT repo; checkpoints **unverified** | ? |

### 2.2 Video (face, pose, scene)

| Need | Tool / model | Licence | Flags |
|---|---|---|---|
| Landmarks, face mesh, pose | MediaPipe ([repo](https://github.com/google-ai-edge/mediapipe)) | Apache-2.0 | — |
| Action units | py-feat ([repo](https://github.com/cosanlab/py-feat)) | MIT code; bundled models vary | check per model |
| Action units | LibreFace ([repo](https://github.com/ihp-lab/LibreFace)) | LicenseRef-USC-Research | NC |
| Action units | OpenFace 2.0 ([repo](https://github.com/TadasBaltrusaitis/OpenFace)) | academic licence | NC, stale |
| Pose | MMPose/RTMPose ([repo](https://github.com/open-mmlab/mmpose)) | Apache-2.0 | — |
| Pose | ViTPose ([repo](https://github.com/ViTAE-Transformer/ViTPose)) | Apache-2.0 | — |
| Pose | Sapiens ([repo](https://github.com/facebookresearch/sapiens)) | CC-BY-NC-4.0 | NC |
| Pose | OpenPose ([repo](https://github.com/CMU-Perceptual-Computing-Lab/openpose)) | CMU academic licence | NC |
| Pose | Ultralytics YOLO ([repo](https://github.com/ultralytics/ultralytics)) | AGPL-3.0 | CL-S, network |
| Pose | DeepLabCut ([repo](https://github.com/DeepLabCut/DeepLabCut)) | LGPL-3.0 | CL-W |
| Face identity | InsightFace ([repo](https://github.com/deepinsight/insightface)) | MIT code; NC weights | NC |
| Face identity | deepface ([repo](https://github.com/serengil/deepface)) | MIT wrapper; wrapped models vary | check per model |
| Gaze from video | L2CS-Net ([repo](https://github.com/Ahmednull/L2CS-Net)) | MIT | stale |
| Video SSL | VideoMAE ([repo](https://github.com/MCG-NJU/VideoMAE)) | CC-BY-NC-4.0 | NC |

### 2.3 Text and language

| Need | Tool / model | Licence |
|---|---|---|
| Embeddings | sentence-transformers ([repo](https://github.com/UKPLab/sentence-transformers)) | Apache-2.0 |
| Embeddings | all-MiniLM-L6-v2 ([card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) | Apache-2.0 |
| Parsing, POS, syntax | spaCy ([repo](https://github.com/explosion/spaCy)) | MIT (models vary) |
| Parsing, POS, syntax | Stanza ([repo](https://github.com/stanfordnlp/stanza)) | Apache-2.0 |
| Readability and lexical metrics | TextDescriptives ([repo](https://github.com/HLasse/TextDescriptives)) | Apache-2.0 |
| PII | Presidio ([repo](https://github.com/microsoft/presidio)) | MIT |
| Sentiment | cardiffnlp twitter-roberta-base-sentiment-latest ([card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)) | CC-BY-4.0 |
| Topics | BERTopic ([repo](https://github.com/MaartenGr/BERTopic)) | MIT |
| Psycholinguistic dictionaries | LIWC ([site](https://www.liwc.app)) | proprietary (**unverified**); not suitable for a core backend |

### 2.4 New modalities

| Modality | Canonical algorithms | Libraries | Pretrained / foundation models |
|---|---|---|---|
| **Actigraphy / IMU** | <ul><li>Autocalibration against local gravity</li><li>ENMO/MAD/counts</li><li>Non-wear (SD/range windows)</li><li>Epoching</li><li>Sleep: Cole-Kripke, Sadeh, van Hees HDCZA</li><li>Circadian: IS, IV, RA, L5/M10, cosinor</li><li>Step counting</li><li>Gait (SKDH)</li></ul> | <ul><li>GGIR (R, Apache-2.0, [repo](https://github.com/wadpac/GGIR))</li><li>SKDH (MIT, [repo](https://github.com/pfizer-opensource/scikit-digital-health))</li><li>pyActigraphy (GPL-3.0, CL-S, [repo](https://github.com/ghammad/pyActigraphy))</li><li>actipy (Oxford Academic Use licence, **NC**, [repo](https://github.com/OxWearables/actipy))</li></ul> | Oxford stepcount, asleep and ssl-wearables/harnet ([repo](https://github.com/OxWearables/ssl-wearables)). All are under the Oxford Academic Use licence (**NC**, click-through). |
| **ECG** | <ul><li>Band-pass, powerline notch</li><li>R-peak detection (Pan–Tompkins and others)</li><li>Delineation</li><li>Signal quality indices</li><li>Ectopic correction</li><li>HRV (time, frequency, nonlinear)</li><li>ECG-derived respiration</li></ul> | <ul><li>NeuroKit2 (MIT, [repo](https://github.com/neuropsychology/NeuroKit))</li><li>BioSPPy (BSD-3-Clause, [repo](https://github.com/scientisst/BioSPPy))</li><li>pyHRV (BSD-3-Clause, [repo](https://github.com/PGomes92/pyhrv))</li><li>hrv-analysis (GPL-3.0, CL-S, [repo](https://github.com/Aura-healthcare/hrv-analysis))</li><li>Systole (Apache-2.0, [repo](https://github.com/embodied-computation-group/systole))</li><li>sleepecg (BSD-3-Clause, [repo](https://github.com/cbrnr/sleepecg))</li><li>wfdb-python (MIT, [repo](https://github.com/MIT-LCP/wfdb-python))</li></ul> | <ul><li>ECG-FM (MIT, [repo](https://github.com/bowang-lab/ecg-fm))</li><li>ECGFounder (MIT, [repo](https://github.com/PKUDigitalHealth/ECGFounder))</li><li>HuBERT-ECG (CC-BY-NC-4.0, **NC**, [repo](https://github.com/Edoar-do/HuBERT-ECG))</li></ul> |
| **PPG** | <ul><li>Band-pass</li><li>Systolic-peak and onset detection</li><li>Fiducials (pyPPG)</li><li>Signal quality (template matching)</li><li>Pulse rate and PRV</li><li>Motion-artefact handling with accelerometer reference</li></ul> | <ul><li>NeuroKit2</li><li>HeartPy (MIT, stale, [repo](https://github.com/paulvangentcom/heartrate_analysis_python))</li><li>pyPPG (GPL-3.0 plus an "academic only" statement that conflicts with it, [repo](https://github.com/godamartonaron/GODA_pyPPG))</li></ul> | <ul><li>PaPaGei (BSD-3-Clause code; repo archived; weights licence **unverified**, [repo](https://github.com/Nokia-Bell-Labs/papagei-foundation-model))</li><li>Pulse-PPG (MIT code, [repo](https://github.com/maxxu05/pulseppg))</li></ul> |
| **EDA** | <ul><li>Low-pass</li><li>Tonic/phasic decomposition (cvxEDA, sparse methods)</li><li>SCR detection (onset, amplitude, rise time)</li><li>Artefact detection</li></ul> | NeuroKit2; FLIRT (MIT, stale, [repo](https://github.com/im-ethz/flirt)) | none verified |
| **Respiration** | <ul><li>Breath detection (peak/trough or zero-crossing)</li><li>Rate, volume proxies, variability</li><li>Derived from ECG/PPG/audio</li></ul> | NeuroKit2 | HeAR (audio events) |
| **EEG / PSG** | <ul><li>Re-reference, filtering, ICA</li><li>Epoching, ERPs, spectra</li><li>Sleep staging, spindle and slow-wave detection</li></ul> | <ul><li>MNE-Python (BSD-3-Clause, [repo](https://github.com/mne-tools/mne-python))</li><li>YASA (BSD-3-Clause, [repo](https://github.com/raphaelvallat/yasa))</li><li>braindecode (BSD-3-Clause, [repo](https://github.com/braindecode/braindecode))</li><li>pyEDFlib (BSD-3-Clause, [repo](https://github.com/holgern/pyedflib))</li></ul> | <ul><li>LaBraM (MIT repo; BSD-3-Clause braindecode HF copy, [repo](https://github.com/935963004/LaBraM))</li><li>BIOT (MIT, stale)</li><li>CBraMod (MIT)</li><li>EEGPT (Apache-2.0)</li><li>The last three have separate weight terms (**unverified**).</li></ul> |
| **Eye tracking** | <ul><li>Blink detection and interpolation</li><li>I-VT/I-DT fixation and saccade detection</li><li>Adaptive classifiers (REMoDNaV, I2MC)</li><li>Pupil preprocessing</li><li>AOI mapping</li></ul> | <ul><li>pymovements (MIT, [repo](https://github.com/pymovements/pymovements))</li><li>REMoDNaV (MIT, [repo](https://github.com/psychoinformatics-de/remodnav))</li><li>I2MC_Python (MIT, [repo](https://github.com/dcnieho/I2MC_Python))</li><li>Pupil Core (LGPL-3.0, CL-W)</li><li>PyGaze (GPL-3.0 in the LICENSE file vs MIT on PyPI: conflict)</li></ul> | — |
| **Passive phone sensing** | <ul><li>GPS trajectories (stay points, home time, radius of gyration)</li><li>Call/text/screen counts</li><li>Missingness modelling</li></ul> | <ul><li>Beiwe forest (BSD-3-Clause, [repo](https://github.com/onnela-lab/forest))</li><li>LAMP-cortex (BSD-3-Clause, [repo](https://github.com/BIDMCDigitalPsychiatry/LAMP-cortex))</li><li>RAPIDS (AGPL-3.0, CL-S network, [repo](https://github.com/carissalow/rapids))</li></ul> | — |

### 2.5 General time-series libraries and foundation models

Libraries:

| Library | Licence |
|---|---|
| sktime ([repo](https://github.com/sktime/sktime)) | BSD-3-Clause |
| aeon ([repo](https://github.com/aeon-toolkit/aeon)) | BSD-3-Clause |
| tsfresh ([repo](https://github.com/blue-yonder/tsfresh)) | MIT |
| tslearn ([repo](https://github.com/tslearn-team/tslearn)) | BSD-2-Clause |
| TSFEL ([repo](https://github.com/fraunhoferportugal/tsfel)) | BSD-3-Clause |
| STUMPY ([repo](https://github.com/stumpy-dev/stumpy)) | BSD-3-Clause |
| ruptures ([repo](https://github.com/deepcharles/ruptures)) | BSD-2-Clause |

Foundation models:

| Model | Licence (code / weights) | Role |
|---|---|---|
| Chronos, Chronos-Bolt, Chronos-2 ([repo](https://github.com/amazon-science/chronos-forecasting)) | Apache-2.0 / Apache-2.0 | forecasting; embeddings |
| TimesFM ([repo](https://github.com/google-research/timesfm)) | Apache-2.0 / Apache-2.0 | forecasting |
| Moirai, uni2ts ([repo](https://github.com/SalesforceAIResearch/uni2ts)) | Apache-2.0 / **CC-BY-NC-4.0** | forecasting (weights NC) |
| MOMENT ([repo](https://github.com/moment-timeseries-foundation-model/moment)) | MIT / MIT | classification, anomaly, imputation, embeddings |
| Lag-Llama ([repo](https://github.com/time-series-foundation-models/lag-llama)) | Apache-2.0 / Apache-2.0 | probabilistic forecasting (stale) |
| Toto ([repo](https://github.com/DataDog/toto)) | Apache-2.0 / Apache-2.0 | forecasting (observability data) |
| Mantis ([repo](https://github.com/vfeofanov/mantis)) | Apache-2.0 / Apache-2.0 | classification embeddings |

For behavioural research, the embedding/classification models (MOMENT, Mantis) and anomaly detection
are more useful than forecasting. These models sit in the generic layer: their input is any regular
`Signal`.

**Licence summary for new modalities.** Most core physiological tooling is permissive (MIT/BSD).
The traps are:

- the Oxford wearables stack (academic only);
- GPL tools: hrv-analysis, pyActigraphy, pyPPG, PyGaze;
- AGPL RAPIDS;
- NC weights: HuBERT-ECG, Moirai.

All of them fit the design's `License` record (4.12), which needs a `LicenseRef-Oxford-Academic-Use`
entry and a "conflicting sources" note for pyPPG and PyGaze.

## 3. Data standards

| Standard | Covers | Status | Metadata that matters | Source |
|---|---|---|---|---|
| BIDS core | dataset layout, entities, sidecars | v1.11.2 is the latest release | `scans.tsv` `acq_time` ("the moment the acquisition … started"; identical for files from one recording) | [releases](https://github.com/bids-standard/bids-specification/releases), [scans.tsv](https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/data-summary-files.html) |
| BIDS physio | `_physio.tsv.gz` (no header row) + JSON; `_physioevents.tsv.gz` | stable; EMG added in 1.11.1 | Required: `SamplingFrequency`, `StartTime` (seconds relative to the parent recording), `Columns`. `PhysioType`. A file is split with `recording-<label>` when rate or device differs. | [spec](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/physiological-recordings.html) |
| BIDS eye tracking (BEP020) | gaze and pupil as `PhysioType: eyetrack` | merged in 1.11.0 | `RecordedEye`, `SampleCoordinateSystem`, screen geometry; columns `timestamp, x_coordinate, y_coordinate[, pupil_size]`; blinks and saccades as physioevents | same |
| BIDS motion (BEP029) | IMU and motion capture: `_tracksys-<label>_motion.tsv` | in the stable spec (first release not confirmed) | `SamplingFrequency` (+ `SamplingFrequencyEffective`). `channels.tsv`: `name, component, type (ACCEL/GYRO/…), tracked_point, units` | [spec](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html) |
| EEG-BIDS | EEG | stable | Formats: EDF(+), BDF(+), BrainVision, EEGLAB .set. Required JSON: `EEGReference`, `SamplingFrequency`, `PowerLineFrequency`, `SoftwareFilters`, `TaskName`. `channels.tsv` `units` is required. | [spec](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html) |
| BIDS audio/video (BEP047 behavioural, BEP044 stimuli) | `audio`, `video`, `audiovideo` media | **proposal**; PR #2367 closed, work continues in the BEP | `AudioSampleRate`, `AudioChannelCount`, `VideoFrameRate`, `RecordingDuration`, codec | [PR](https://github.com/bids-standard/bids-specification/pull/2367) |
| BEP028 provenance | processing provenance | **not merged** (review closed May 2026) | — | [BEP028](https://bids.neuroimaging.io/extensions/beps/bep_028.html) |
| EDF / EDF+ / BDF | polygraphy, PSG, EEG | stable | Samples are 16-bit (BDF: 24-bit). Physical/digital min and max scale each signal. EDF+D is discontinuous. TAL annotations (onset, duration, text). Start date has a 1985 clipping point. | [EDF+](https://www.edfplus.info/specs/edfplus.html) |
| WFDB | ECG and PhysioNet records | stable | `.hea`: fs, base time and date, per-signal ADC gain, baseline, units (default mV), resolution. Separate annotation files (beat labels). | [header spec](https://physionet.org/physiotools/wag/header-5.htm) |
| ActiGraph .gt3x | accelerometry | open spec | zip of `info.txt` (serial, start, download) + binary log; older NEO/MRA layout | [spec](https://github.com/actigraph/GT3X-File-Format) |
| Axivity .cwa, GENEActiv .bin | accelerometry | vendor formats; readers in actipy and GGIR | device ID, calibration, temperature, light | [actipy](https://pypi.org/project/actipy/) (primary specs **unverified**) |
| Empatica EmbracePlus | wearable raw data | vendor Avro, files ≤30 min, organised by UTC | per-sensor timestamps | [PMC11644628](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11644628/) |
| XDF / LSL | multi-stream lab recording | spec 1.x | `nominal_srate` (0 = irregular), `channel_format`; per-stream ClockOffset chunks every 5–10 s, aligned by robust linear fit | [XDF spec](https://github.com/sccn/xdf/wiki/Specifications) |
| NWB | neurophysiology time series | active (pynwb BSD-style) | timestamps or a rate per TimeSeries (schema details **unverified**) | [pynwb](https://github.com/NeurodataWithoutBorders/pynwb) |
| IEEE 1752.1-2021 / Open mHealth | mobile health: sleep, activity | published 2021; P1752.2 in development | units, time frame, data source | [IEEE](https://standards.ieee.org/ieee/1752.1/6982/) |

Metadata every reader must normalise into the core type:

- **Physical units** per channel. WFDB gain/baseline and EDF physical min/max must be applied on
  read; BIDS `channels.tsv` `units` is the target.
- **Sampling.** Nominal and effective rate. XDF uses `nominal_srate = 0` for irregular streams.
  EDF+D files have gaps.
- **Device.** Manufacturer, model, serial, firmware, placement (`tracked_point`) and calibration.
- **Absolute start time with a timezone**, plus each stream's offset to a reference clock:
  - BIDS gives `acq_time` plus a relative `StartTime`;
  - XDF gives clock-offset samples;
  - wearables give UTC or local time with DST hazards.
- **Annotations.** EDF TALs, WFDB annotations, BIDS `events.tsv`/`physioevents` (onset in seconds,
  duration), all mapped onto one `Span`/`Annotation` shape.

## 4. Taxonomy of algorithms

**Type vocabulary used below:**

- **Signal**: regular, multichannel, with units.
- **IrregularSignal**: sample timestamps.
- **Events**: point annotations.
- **Intervals**: span annotations, with labels or values.
- **Track**: a time-indexed value series. It is a Signal by another name; see 5.
- **Embedding**.
- **Table**: per-interval summary rows.

### (a) Generic: any time series

| Algorithm | In → Out | Notes |
|---|---|---|
| Resample (rational, anti-aliased); regularise irregular → regular | Signal/IrregularSignal → Signal | Today: `audio/preprocessing` |
| Filter (FIR/IIR band, notch at line frequency, detrend, high-pass baseline) | Signal → Signal | Line frequency is a dataset parameter |
| Rescale and unit conversion (ADC → physical, g ↔ m/s²) | Signal → Signal | Driven by units metadata |
| Channel ops: select, re-reference, vector magnitude, PCA/ICA | Signal → Signal | Vector magnitude covers ENMO and multi-mic downmix |
| Windowing / epoching (fixed, sliding, event-locked) | Signal (+Events) → Signal[] or windows | ERP epochs, actigraphy epochs, audio frames |
| Spectral: PSD (Welch), STFT, band power, spectral entropy, dominant frequency | Signal → Track / Table | `band_profile` and `spectral_continuity` generalise here |
| Envelope: Hilbert, RMS, rectify + smooth | Signal → Track | `audio/envelope` |
| Activity / event detection: energy threshold, hysteresis, change-point (ruptures), matrix profile (STUMPY) | Signal/Track → Intervals / Events | VAD-by-energy, active vs rest, non-wear |
| Segmentation and span algebra: union, intersect, merge gaps, min duration | Intervals → Intervals | `audio/spans`, generic |
| Signal QC: flatline, clipping/saturation, dropouts/gaps, SNR, line-noise power, out-of-range values | Signal → Track + Intervals (measure, not verdict) | `audio/clipping` and parts of `quality_control` generalise |
| Synchronisation: offset/drift estimation by cross-correlation, shared events or clock-offset fits; timeline mapping | Signal×Signal or ClockOffsets → TimeMap | New; needed for any multi-device study |
| Generic features (tsfresh/TSFEL catalogues, statistics over windows) | Signal/Track (+Intervals) → Table | Per-interval aggregation |
| Embeddings from time-series foundation models (MOMENT, Mantis, Chronos encoder) | Signal → Embedding (per window) | `space` = model identity |
| Dimensionality reduction, clustering, statistics | Embedding[]/Table → Embedding[]/labels/Table | Already in `utils/tasks` |
| Imputation, interpolation of gaps | Signal + Intervals(gaps) → Signal | Must record what was imputed |

### (b) Shared families with modality-specific parameters

| Family | Instances | In → Out | What differs |
|---|---|---|---|
| Peak / onset detection | ECG R-peaks; PPG systolic peaks; respiration breaths; syllable nuclei from the speech envelope; SCR onsets; steps; saccade velocity peaks | Signal → Events (+ amplitude) | Band, refractory period, prominence, physiological plausible-rate prior |
| Inter-event interval analysis | HRV/PRV; breathing variability; speech rate / DDK; step cadence; inter-turn gaps | Events → Track / Table | Artefact/ectopic rules, frequency bands (HRV LF/HF vs speech rhythm) |
| Periodicity / f0 tracking | Voice f0 (autocorrelation/YIN); heart rate; breathing rate; cadence; tremor frequency | Signal → Track | Frequency range, window length |
| Signal quality index | ECG/PPG template correlation; audio SNR/HNR; EEG impedance and flatness; actigraphy non-wear | Signal → Track / Intervals | Template and thresholds (thresholds belong in `data/` profiles with a derivation) |
| Decomposition | EDA tonic/phasic; source separation (audio); ICA (EEG); gravity vs body acceleration | Signal → Signal (multiple) | Model of the components |
| Event classification over windows | Audio scene/SER; activity type (HAR); sleep stage per 30-s epoch; cough vs speech | Signal → Intervals (label, score) | Window length, label set, model |
| Turn/state sequencing | Conversational turns; sleep cycles; fixation→saccade sequences | Intervals → Intervals / Table | Grammar of states |
| Circadian / rhythm statistics | Activity IS/IV/L5/M10; HR daily rhythm; phone use rhythm | Track (multi-day) → Table | Period, epoch |
| Cross-signal coupling | Entrainment (prosody); cardiorespiratory coupling; synchrony in dyads | Track×Track → Track / Table | Lags, measures |

### (c) Strictly modality-specific

| Modality | Tasks | Out types |
|---|---|---|
| Audio | ASR, forced alignment, diarization, speaker embeddings/verification, TTS/cloning, enhancement, PPGs, articulatory coding, voice quality (jitter/shimmer/CPP), SER | Transcript, Intervals(speaker), Embedding, Audio, Track |
| Text | Tokenisation, parsing, sentiment, PII, coherence, dialogue acts, lexical metrics | Annotations on text spans (which map to time via alignment), Table |
| Video | Face detection/tracking, AUs, landmarks, head pose, gaze from video, body pose, scene description, active-speaker detection | Track (per entity), Intervals, Embedding, Text |
| Actigraphy | Autocalibration, ENMO/counts, non-wear, sleep/wake (Cole-Kripke, van Hees, asleep), step count, gait | Signal, Intervals, Table |
| ECG | Delineation (P/QRS/T), arrhythmia classification, QT, ECG-FM embeddings | Events, Intervals, Embedding |
| PPG | Fiducials, SpO2 (needs red+IR), blood-pressure proxies | Events, Track |
| EDA | SCR features with fixed physiological conventions | Events, Table |
| EEG/PSG | Re-reference montages, ICA artefacts, sleep staging (AASM), spindles, ERPs | Signal, Intervals, Events |
| Eye tracking | Blink handling, fixation/saccade/PSO classification, AOI hits, pupil foreshortening correction | Intervals, Events, Table |
| Phone sensing | GPS stay points, mobility metrics, missingness | IrregularSignal → Intervals, Table |

## 5. Implications for the design

### 5.1 Data model

1. **Add `Signal` as the base type; make `Audio` a subclass or a profile of it.** Section 4.1
   defines `Audio` as `(channels, samples)` float32 plus `sampling_rate`. Generalise it as follows:
   - `Signal{data: (channels, samples), sampling_rate, channels: [ChannelInfo{name, type, units,
     component?, tracked_point?, reference?}], start: TimeAnchor, device: DeviceInfo, identity}`.
   - `Audio` becomes `Signal` with `type="audio"` and units = full-scale.
   - Make 4.10's `Track[T]` the same `Signal` (regular) or `IrregularSignal`, so f0, HR, ENMO and
     facial-AU intensity series all flow into the generic tasks of 4(a).
   - Keep dtypes per channel: actigraphy and ADC data arrive as int16/int32 with gain. Allow
     float64 where precision matters (timestamps, EDA).
2. **Add `IrregularSignal`**: per-sample timestamps for phone sensing, LSL streams with
   `nominal_srate=0`, RR-interval series and eye trackers with dropped samples.
   - Regularisation is an explicit generic task, never an implicit read step.
   - Discontinuous records (EDF+D, wearable gaps) are a `Signal` plus a `gaps: Intervals`, or a
     list of segments. They are never zero-filled silently.
3. **Units are first-class.** Readers apply ADC gain/baseline (WFDB) and physical min/max (EDF).
   Tasks declare which units they accept: an HRV task requires Events in seconds and an ECG in mV
   or "arbitrary". Unit metadata must enter `identity`.
4. **One timeline needs a clock model, not only `Span{start, end, source}`.**
   - Each source carries `TimeAnchor{absolute_start (tz-aware, possibly date-shifted), clock_id}`.
   - A **`TimeMap`** (offset + drift, or piecewise-linear, with uncertainty) maps `clock_id → reference
     clock`. It is produced by the synchronisation task in 4(a).
   - Spans stay on their source clock and are projected through the TimeMap when two sources are
     compared. Provenance records which TimeMap was used.
   - The model covers BIDS `StartTime` relative to its parent, XDF ClockOffsets, video-frame vs
     audio-sample drift, and wearable UTC/DST.
5. **Events vs intervals.** `Annotation` with `duration=0` covers events, but peak detectors emit
   thousands per hour. A columnar `Events`/`Intervals` container (arrays of onset, duration, label,
   value, score) is needed for performance; it serialises as Parquet or as BIDS `events.tsv` /
   `physioevents`.
6. **Multi-day and long-form scale.** Actigraphy at 100 Hz for 7 days is about 60M samples × 3
   channels; day-long child audio is similar.
   - `Signal` must load lazily *by time range* (memory-mapped or chunked: Zarr or HDF5 for derived
     data).
   - Tasks must be able to declare that they are chunk-able (streaming filter, windowed features), so
     the planner shards along time, not just across records.
7. **Record views generalise.** A Record holds named Signals by modality and device (`ecg`, `ppg`,
   `acc@wrist`, `audio`, `video`, `eyetrack`) sharing entities. The BIDS `recording-<label>` /
   `tracksys-<label>` entities map to view names.

### 5.2 Task catalogue

- **Organise by input type first, modality second.**
  - `sl.signal` (or `sl.timeseries`) holds the generic tasks of 4(a). They accept any `Signal` and
    are reused by `sl.audio`, `sl.ecg`, etc.
  - The current audio DSP that is really generic moves there: preprocessing, spans, envelope,
    band_profile, clipping, spectral continuity.
  - `sl.audio`, `sl.video` and `sl.text` keep 4(c).
  - Add `sl.physio` (ECG, PPG, EDA, respiration), `sl.motion` (actigraphy/IMU), `sl.eeg` (thin,
    delegating to MNE/YASA) and `sl.eyetrack`. Possibly add `sl.mobile`.
- **Families in 4(b) are one task with a modality-keyed parameter profile**, not N tasks. For
  example `detect_peaks(signal, profile="ecg_rpeak")`, where the profile lives in `data/` with its
  derivation (P-17), and the catalogue default depends on the channel `type`. Wrappers like
  `sl.physio.ecg_rpeaks` are sugar over the family task, as 4.2 already does for typed functions.
- **Type-driven dispatch.** A task's `inputs` declare `Signal[type in {...}, units]`, so the
  planner and `sl.describe` can tell an agent "these 14 tasks accept your ACCEL channel".
- **Bridges (4.10) gain physio entries**, for example:
  - ECG/PPG → respiration;
  - accelerometer + PPG → motion-corrected pulse;
  - video → respiration (rPPG);
  - audio + actigraphy → speaking-while-moving.
- **Measure/decide split holds.** QC tasks emit SQI tracks and gap intervals. Non-wear or
  "unusable epoch" verdicts are decision tasks with a fitted profile.
- **Licences.** Add LicenseRef entries for Oxford Academic Use, audEERING, OpenFace/OpenPose
  academic, USC research and HAI-DEF. Many physio defaults can be permissive (NeuroKit2, MNE, YASA,
  SKDH, GGIR via R, ECG-FM, MOMENT), so a `commercial_use: required` policy remains satisfiable for
  most chains. Actigraphy sleep/steps is the main gap: asleep and stepcount are NC, so the permissive
  path is SKDH or GGIR.
- **Environments.** R (GGIR) and Java (actipy) are new non-Python runtimes. The env layer (4.10)
  must declare non-pip requirements, as it already does for TF.

## Unverified items to resolve before adding them to the catalogue

- emotion2vec FunASR licence terms.
- Full HeAR HAI-DEF terms.
- Weight licences for:
  - SPARC;
  - MAAI;
  - BEATs;
  - PaPaGei;
  - BIOT, CBraMod and EEGPT;
  - Pulse-PPG.
- The Axivity/GENEActiv primary specs.
- The NWB schema details.
- Which BIDS release first merged motion.
- The pyPPG and PyGaze licence conflicts.
- LIWC terms.
