# Public benchmarks for senselab 2.x

Checked 2026-10-30. "Verified" means the licence/access statement was read on the dataset's own page,
card, README or paper (URL given). "Unverified" means only secondary sources were found; resolve
before a dataset enters the suite. Access tiers used below:

- **open**: anonymous download.
- **click**: account or click-through terms.
- **request**: a form or EULA reviewed by a person.
- **DUA**: a signed data-use agreement.
- **LDC**: paid LDC licence.

"Derived results" means publishing scores and per-item metrics. In general, publishing aggregate
metrics is fine under every licence below. **Per-item outputs** (transcripts, embeddings, crops) are
derivatives of the content. They are allowed under CC BY/ODC-By with attribution and are blocked
under ND/NC/DUA terms.

## 1. Per-task benchmarks

### Speech and audio

| Task | Benchmark | Size | Metric(s) | Licence / access | Smoke subset |
|---|---|---|---|---|---|
| ASR | LibriSpeech test-clean/test-other ([openslr 12](https://www.openslr.org/12)) | ~5.4 h each | WER (normalised) | CC BY 4.0, open (verified) | 50 utts of test-clean |
| ASR | Open ASR Leaderboard sets ([repo](https://github.com/huggingface/open_asr_leaderboard), Apache-2.0 code): AMI, Earnings22, multilingual FLEURS/MCV/MLS | varies | WER + RTFx | per-dataset terms; some ESB members are gated on HF | 1 shard of `hf-audio/esb-datasets-test-only-sorted` |
| ASR | Common Voice | per language | WER/CER | data CC0. Since Oct 2025 distributed **only via Mozilla Data Collective**, which needs an account and acceptance of its ToS ([MF README](https://huggingface.co/spaces/mozilla-foundation/README/blob/main/README.md), [MDC post](https://discourse.mozilla.org/t/common-voice-dataset-segments-now-available-on-mdc/149060)) → click | 100 clips/language |
| Forced alignment | TIMIT (LDC93S1) | 630 spk | boundary error ≤20 ms %, mean abs. error | LDC (paid; from LDC catalogue, not re-checked) | 20 utts |
| Forced alignment | Buckeye ([licence](https://buckeyecorpus.osu.edu/License.pdf)) | 40 spk conversational | same | free **non-commercial**, registration, **no derivative works**, lab-scoped, server must be password-protected (verified) → request | 1 talker |
| Forced alignment (open) | LibriSpeech + published MFA alignments | — | agreement with reference aligner | CC BY 4.0. **Silver standard only.** | 50 utts |
| Diarization | VoxConverse v0.3 ([repo](https://github.com/joonson/voxconverse)) | 448 YouTube recordings (dev+test) | DER, JER | CC BY 4.0, video copyright stays with owners; use v0.3 RTTMs (verified) | 5 test files |
| Diarization | AMI Mix-Headset ([corpus](https://groups.inf.ed.ac.uk/ami/corpus/)) | 100 h meetings | DER (pyannote "only_words" split) | CC BY 4.0 for signals and transcripts, "some" annotations (verified via official page summary) | 2 meetings |
| Diarization | DIHARD III eval | 33 h, 11 domains | DER/JER, collar 0 | LDC fee for non-members ([LDC newsletter](https://www.ldc.upenn.edu/node/12925); catalog id unverified) | — |
| VAD | AVA-Speech ([download](https://research.google.com/ava/download.html)) | 160 movies still on YouTube × 15 min | frame F1/AUC per condition (clean / +music / +noise) | labels CC BY 4.0 ([Google AVA](https://sites.research.google/gr/ava/download/)). Audio must be fetched from YouTube, so the set shrinks over time. | 5 movies |
| VAD (open, stable) | AMI or VoxConverse RTTM → speech/non-speech | as above | detection error rate (pyannote) | CC BY 4.0 | as above |
| Speaker verification | VoxCeleb1-O ([KAIST mirror](https://mm.kaist.ac.kr/datasets/voxceleb/), [licence](https://mm.kaist.ac.kr/datasets/voxceleb/files/license.txt)) | 37,720 trials, 40 spk | EER, minDCF (p=0.01 or 0.05) | CC BY 4.0, request/download. VGG host was unreachable today. Trial lists are on [openslr 49](https://openslr.org/49/), marked "not copyrighted". | 1,000 trials |
| Speech enhancement | VoiceBank+DEMAND ([DataShare 2791](https://datashare.ed.ac.uk/handle/10283/2791)) | 824 test utts | PESQ, ESTOI, SI-SDR, DNSMOS | predecessor record CC BY 4.0. **Rights field on 2791 unconfirmed**; open. | 50 utts |
| Speech enhancement | DNS Challenge blind test ([repo](https://github.com/microsoft/DNS-Challenge)) | ~600 clips/round | DNSMOS P.835 (non-intrusive) | repo CC BY 4.0 docs + MIT code; datasets "under the original terms" per source (partly unverified) | 50 clips |
| Separation | LibriMix ([paper](https://arxiv.org/abs/2005.11262)) | Libri2Mix test 3,000 mixtures | SI-SDRi | LibriSpeech CC BY 4.0 + WHAM! noise **CC BY-NC 4.0**, so the noisy variant is NC; Libri2Mix-clean is not | 50 mixtures (clean) |
| SER | CREMA-D ([repo](https://github.com/CheyneyComputerScience/CREMA-D)) | 7,442 clips, 91 actors, 6 emotions | UA/WA recall, macro-F1 | ODbL + DbCL, open (verified) | 1 actor-disjoint fold, 300 clips |
| SER | RAVDESS ([Zenodo](https://zenodo.org/records/1188976)) | 24 actors, 8 emotions | UA recall | **CC BY-NC-SA 4.0**; commercial licence sold (verified) | 2 actors |
| SER | MSP-Podcast ([lab](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)), IEMOCAP ([form](https://sail.usc.edu/iemocap/release_form.php)) | 100+ h / 12 h | UA; CCC for arousal/valence/dominance | MSP: free academic, institution-signed; commercial is paid. IEMOCAP: academic email, manual review, no commercial use. → request | — |
| Audio events | ESC-50 | 2,000 clips, 50 classes, 5 folds | accuracy (5-fold) | **CC BY-NC 3.0** per mirrors/loaders ([soundata](https://soundata.readthedocs.io/en/latest/_modules/soundata/datasets/esc50.html)); ESC-10 subset CC BY; GitHub LICENSE not read directly | ESC-10 |
| Audio events | FSD50K ([paper](https://arxiv.org/pdf/2010.00475)) | eval 10,231 clips / 27.9 h, 200 classes | mAP, d′ | dataset CC BY 4.0; per-clip CC0/BY/BY-NC/Sampling+ (≈15% NC) | 500 CC0/BY eval clips |
| Representations (all audio) | HEAR ([paper](https://arxiv.org/abs/2203.03022), [Zenodo](https://zenodo.org/record/5887964)) | 19 tasks | task-specific, frozen-embedding probe | mixed per-task open licences | 3 small tasks |
| f0 / voicing | PTDB-TUG ([SPSC](https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html)) | 20 spk, 4,720 sentences, laryngograph | gross pitch error (>20%), fine pitch error, V/UV error | **ODbL/DbCL**, open (verified) | 100 sentences |
| Voice quality (HNR, jitter, CAPE-V) | PVQD ([Mendeley](https://data.mendeley.com/datasets/9dz247gnyb)) | 296 speakers, /a/ /i/ + CAPE-V sentences, expert VAS ratings | Spearman ρ of measure vs. rating; agreement with Praat (Parselmouth) values | CC BY 4.0 per Voice Foundation and York catalogue (Mendeley field not seen directly) | 30 files |
| Cough / health | COUGHVID v3 ([Zenodo](https://zenodo.org/records/7024894)) | >30k recordings, >2k expert-labelled; 500-item private test held back | cough-detection AUROC; expert-label agreement | CC BY 4.0, open | 200 expert-labelled |
| Cough / breath | Coswara ([repo](https://github.com/iiscleap/Coswara-Data)) | 2,635 individuals, 9 sound types | AUROC | CC BY 4.0 (README verified). One mirror says BY-NC-ND, so trust the official repo. | 50 subjects |
| Respiratory sounds | ICBHI 2017 ([site](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)) | 920 recordings, 6,898 cycles, 126 subj | ICBHI score (mean of Se, Sp) | "freely available for research", no named licence; use the corrected file set | 20 recordings |
| Turn-taking | AMI (word timings) | as above | turn-shift vs. hold F1 at pauses, gap/overlap distributions | CC BY 4.0 | 2 meetings |
| Turn-taking | CANDOR ([BetterUp](https://www.betterup.com/research/candor-research), [ConvoKit](https://convokit.cornell.edu/documentation/candor.html)) | ~1,650 video calls | same + backchannel F1 | request; licence not stated publicly (a CC BY-NC 4.0 notice appears in paper context) | — |
| Turn-taking | Switchboard / Fisher | 2,400 calls | standard VAP-style shift/hold | LDC | — |

### Text

| Task | Benchmark | Size | Metric | Licence / access | Smoke |
|---|---|---|---|---|---|
| PII detection | gretelai/synthetic_pii_finance_multilingual ([HF](https://huggingface.co/datasets/gretelai/synthetic_pii_finance_multilingual)) | 29 PII types, 7 languages | span-level precision/recall/F1 (exact + partial), per-type recall | Apache-2.0 tag (README not read) | 200 docs |
| PII detection | ai4privacy pii-masking-300k ([HF](https://huggingface.co/datasets/ai4privacy/pii-masking-300k)) | 225k rows, 6 languages | same | **custom `license.md`, could not be read (404)**. Treat as restricted until read. | — |
| PII (clinical) | n2c2 2014 de-identification ([n2c2](https://n2c2.dbmi.hms.harvard.edu/data-sets)) | 288 patients, 2-5 records each | token- and entity-level F1, PHI recall | DUA, per-person portal access | — |
| Sentiment | TweetEval sentiment ([card](https://huggingface.co/datasets/cardiffnlp/tweet_eval)) | test 12,284 | macro-recall (official), macro-F1 | sentiment subset CC BY 3.0 + Twitter ToS; other subsets differ | 500 tweets |
| Sentiment | SST-2 (GLUE) | 872 dev | accuracy | licence field not visible on [HF card](https://huggingface.co/datasets/stanfordnlp/sst2); **unverified** | 200 |

Transcripts as text input: the speech-aware text tasks (PII on ASR output) should also be scored on
**ASR output of a speech set**, not only on gold text, so error propagation is measured.

### Video

| Task | Benchmark | Size | Metric | Licence / access | Smoke |
|---|---|---|---|---|---|
| 2D pose | COCO val2017 keypoints | 5k images, 17 kp | OKS-mAP (pycocotools) | annotations CC BY 4.0; images under Flickr terms ([HF COCO card](https://huggingface.co/datasets/HuggingFaceM4/COCO/blob/refs%2Fpr%2F5/README.md); official terms page not opened) | 200 images |
| 2D pose | MPII Human Pose ([download](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download)) | 25k images | PCKh@0.5 | annotations Simplified BSD; **images non-commercial** (verified) | 200 |
| Face detection | WIDER FACE val | 3,226 images, easy/medium/hard | AP per subset | non-commercial research; HF card says CC BY-NC-ND 4.0 ([card](https://huggingface.co/datasets/CUHK-CSE/wider_face)) | 100 images |
| Action units | DISFA / DISFA+ | 27 subj, 12 AU intensities | per-AU F1, ICC | request; terms **not found publicly** | — |
| AU / VA / expression | Aff-Wild2 / ABAW ([iBUG](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)) | 564 videos, ~2.8M frames | AU F1, CCC (V/A), expression F1 | EULA; academic route limited to permanent staff; student and industry routes exist | — |
| Affect (face) | AffectNet | ~0.4M images | accuracy, CCC | request; terms **not verified** | — |
| Scene description | COCO Captions val2017 | 5k × 5 captions | CIDEr, SPICE, CLIPScore | as COCO | 100 images |
| Video description | MSR-VTT test | 2,990 clips | CIDEr, retrieval R@k | licence **not found** ("Other") | — |

### New modalities

| Task | Benchmark | Size | Metric | Licence / access | Smoke |
|---|---|---|---|---|---|
| ECG R-peak | MIT-BIH Arrhythmia ([PhysioNet](https://physionet.org/content/mitdb/1.0.0/)) | 48 × 30 min, 360 Hz | Se, PPV, F1 at 150 ms tolerance; also at sample-level tolerance | ODC-By 1.0, open | 3 records × 5 min |
| ECG R-peak (wearable, exercise) | GUDB ([Glasgow](https://researchdata.gla.ac.uk/716)) | 25 subj × 5 tasks × 2 min, 250 Hz | same, per activity | CC BY 4.0. Sample-exact annotations; noisy records have none. | sitting + jogging, 5 subj |
| HRV | MIT-BIH NSR / GUDB with reference RR | — | abs. error in RMSSD, SDNN, HF power vs. RR from reference peaks | ODC-By / CC BY | — |
| ECG classification probe (FM embeddings) | PTB-XL ([PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)) | 21,799 10-s 12-lead | macro-AUROC on fold 10 | CC BY 4.0, open (verified) | 500 records |
| PPG HR | PPG-DaLiA ([UCI](https://archive.ics.uci.edu/dataset/495/ppg+dalia)) | 15 subj × ~2.5 h, daily activities, ECG reference | MAE (bpm) on 8-s windows / 2-s shift | CC BY 4.0, open | 1 subject |
| PPG / respiration | BIDMC ([PhysioNet](https://physionet.org/content/bidmc/)) | 53 × 8 min, 2 annotators' breaths | breath-detection F1, RR MAE (bpm) | ODC-By 1.0, open | 5 records |
| Respiration rate | CapnoBase TBME RR benchmark ([Borealis](https://borealisdata.ca/dataverse/capnobase)) | 42 × 8 min | RR MAE, 2SD limits of agreement | EULA sign-in; **test-only, must not be used for tuning** | — |
| EDA | WESAD ([UCI](https://archive.ics.uci.edu/dataset/465/wesad)) | 15 subj, chest 700 Hz + wrist 4 Hz | stress vs. baseline vs. amusement F1 (feature-level); chest-vs-wrist SCR agreement | UCI page gives no licence; secondary sources say academic/non-commercial → **unverified** | 2 subjects |
| EDA (open alternative) | PPG-DaLiA E4 EDA channel | as above | signal-quality and SCR-rate agreement only | CC BY 4.0 | — |
| Activity / HAR | Capture-24 ([ORA](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001)) | 151 participants, ~24 h, 100 Hz wrist, 2,562 h labelled | macro-F1 (Walmsley labels), Cohen κ, subject-wise CV | CC BY 4.0 + **no re-identification** clause, open (verified), 6.4 GB zip | 5 participants |
| Steps | OxWalk ([ORA](https://ora.ox.ac.uk/objects/uuid:19d3cb34-e2b3-4177-91b6-1bad0e0163e7)) | 39 participants, wrist + hip, video-labelled heel strikes | step count MAPE; per-step F1 | "free for use" per papers; **licence not seen** | 3 participants |
| Sleep (actigraphy) | Apple Watch sleep-accel ([PhysioNet](https://physionet.org/content/sleep-accel/)) | 31 subj (count from paper, unverified), PSG labels | epoch sleep/wake acc., κ, TST error | ODC-By 1.0, open | 3 nights |
| Sleep staging (EEG) | Sleep-EDF Expanded ([PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)) | 197 nights (153 SC + 44 ST) | 5-class macro-F1, Cohen κ, per-stage F1 | ODC-By 1.0, open | 3 SC nights |
| Sleep staging (large) | NSRR (SHHS, MESA) | thousands of nights | same | DUA (from NSRR policy, not re-checked) | — |
| EEG (BCI / ERD) | EEGMMIDB ([PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/)) | 109 subj (subject count from MOABB, unverified), 64 ch, 160 Hz | motor-imagery accuracy, within/cross-subject | ODC-By 1.0, open | 3 subj |
| Eye-tracking events | Lund2013 hand-labelled ([critique and description](https://ar5iv.labs.arxiv.org/html/1908.11436)) | 34 files, fixation/saccade/PSO/pursuit | event-level F1, Cohen κ (sample), onset/offset error | licence **unverified**; 6 files are 200 Hz, not 500 Hz | 5 files |
| Eye-tracking events | Hollywood2 EM ([GIN](https://gin.g-node.org/ioannis.agtzidis/hollywood2_em)) | ~2 h, manual labels incl. pursuit | same | licence **unverified** | — |
| Eye-tracking robustness | GazeBase ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8285447/)) | 322 subj, 12,334 recordings, 1 kHz | no event labels; use for throughput, task-level stats, cross-session ID | CC BY 4.0 (verified) | 10 recordings |
| TS-FM embeddings | UCR (128) / UEA (30) archives ([UEA paper](https://arxiv.org/pdf/1811.00075)) | 158 problems | linear-probe accuracy, critical-difference rank | **no licence stated**; site footer says "All rights reserved", citation requested. Do not redistribute; download in the job. | 10 UCR sets |
| TS-FM embeddings (behavioural) | probes on Capture-24, PTB-XL, Sleep-EDFx, PPG-DaLiA | — | frozen-embedding probe macro-F1 / MAE, subject-wise splits | as above | reuse smoke subsets |

## 2. Reference harnesses to align metric code with

| Area | Reference | What to match |
|---|---|---|
| ASR | [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) (Apache-2.0) | its `normalizer/` (Whisper English normaliser for English, separate multilingual normaliser), WER via `jiwer`/`evaluate`, RTFx definition. Port the normaliser verbatim and pin its commit. |
| Diarization / VAD | [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/reference.html) | `DiarizationErrorRate(collar=…, skip_overlap=False)`; `collar` defaults to 0, overlap is scored by default. Use `JaccardErrorRate` for JER. Cross-check against `md-eval.pl`/`dscore` for DIHARD. |
| SV | VoxCeleb/ASVspoof minDCF scripts (SpeechBrain, WeSpeaker) | P_target and C_miss/C_fa settings must be stated per number |
| Enhancement | `pesq` (ITU P.862 wideband), `pystoi`, DNSMOS from the DNS repo | sample rate (PESQ-WB requires 16 kHz), P.835 model version |
| Representations | [hear-eval-kit](https://pypi.org/project/heareval/) | frozen embedding + shallow MLP protocol, HEAR 2021.2 datasets |
| SER | none canonical; follow EmoBox / SUPERB ER fold definitions (not verified here) | speaker-independent folds |
| Pose / caption | `pycocotools` (OKS sigmas), `pycocoevalcap` | official val2017 split and annotation JSON |
| ECG / PPG / EDA | [NeuroKit2 benchmarks](https://rpanderson-neurokit2.readthedocs.io/en/latest/benchmarks/ecg_preprocessing.html) (GUDB + MIT-BIH + NSR); Porr & Howell's [JF-score methodology](https://www.biorxiv.org/content/10.1101/722397v3.full.txt); `wfdb.processing.compare_annotations` | detector tolerance window, beat-type exclusions |
| PPG HR / RR | PPG-DaLiA protocol (8-s/2-s windows); [RRest](https://rrest.readthedocs.io/en/latest/datasets/summary), [ppg-beats](https://ppg-beats.readthedocs.io/en/latest/datasets/capnobase/) | window and reference definitions |
| Sleep | YASA validation protocol, PhysioNet/CinC 2018 scorer, NSRR conventions | epoch alignment (30 s), how lights-off/on is handled |
| Actigraphy | Oxford [stepcount](https://pypi.org/project/stepcount/) / asleep evaluation on OxWalk / Capture-24 | subject-wise CV. Their **models** are NC (see use-cases.md), but the **protocol** can be reused. |
| EEG | MOABB evaluation classes (WithinSession/CrossSession/CrossSubject) | the paradigm definition and splits |
| Eye tracking | pymovements dataset loaders; REMoDNaV / I2MC papers' event-matching rules | event-matching (overlap-based) definition |
| TS-FM | aeon/sktime benchmarking, UCR critical-difference plots | default train/test splits |

## 3. Recommended suites

### (a) Minimal suite: open access only, every modality

Criterion: anonymous or account-only download, licence verified permissive (CC BY, CC0, ODC-By,
ODbL, BSD). NC sets are excluded even when downloadable.

| Modality | Benchmark (smoke → full) |
|---|---|
| ASR | LibriSpeech test-clean/other |
| Forced alignment | LibriSpeech vs. reference alignments (silver; mark the score as agreement, not accuracy) |
| Diarization + VAD + turn-taking | VoxConverse v0.3 test; AMI Mix-Headset (diarization, VAD from RTTM, shift/hold at pauses) |
| Speaker verification | VoxCeleb1-O |
| Enhancement / separation | VoiceBank+DEMAND (once rights field confirmed); Libri2Mix **clean** |
| SER | CREMA-D |
| Audio events / embeddings | FSD50K eval (CC0/CC BY clip subset) |
| f0 / voice quality | PTDB-TUG; PVQD |
| Health acoustics | COUGHVID v3; Coswara |
| Text PII / sentiment | gretelai synthetic PII (after reading README); TweetEval-sentiment (CC BY 3.0) |
| Pose / scene | COCO val2017 keypoints + captions (annotations only; images fetched at run time) |
| Face | **gap**: no permissively licensed face detection/AU set was verified. Use COCO person crops for a detection sanity check and flag this. |
| ECG / HRV | MIT-BIH Arrhythmia, GUDB, PTB-XL |
| PPG / respiration | PPG-DaLiA, BIDMC |
| EDA | PPG-DaLiA E4 EDA (agreement and signal quality only, no gold SCRs) |
| Actigraphy / sleep | Capture-24, sleep-accel, Sleep-EDFx |
| EEG | EEGMMIDB, Sleep-EDFx |
| Eye tracking | GazeBase (throughput / statistics). **Gap for labelled events** until Lund2013 or Hollywood2 EM licence is read. |
| TS-FM | probes on Capture-24 / PTB-XL / Sleep-EDFx (UCR/UEA excluded: no licence) |

### (b) Extended suite (adds request/DUA/LDC/NC sets; results kept private unless terms allow)

Common Voice (MDC), DIHARD III, TIMIT, Buckeye, AVA-Speech, DNS blind test, LibriMix-noisy (NC),
RAVDESS (NC), MSP-Podcast, IEMOCAP, ESC-50 (NC), HEAR full, ICBHI, CANDOR, Switchboard/Fisher,
ai4privacy, n2c2 2014, MPII (NC images), WIDER FACE (NC-ND), DISFA, Aff-Wild2, AffectNet, MSR-VTT,
CapnoBase (test only), WESAD, OxWalk, NSRR SHHS/MESA, Lund2013/Hollywood2 EM, UCR/UEA.

## 4. Pitfalls

- **WER normalisation.** Report raw and normalised WER and name the normaliser commit. The Whisper
  English normaliser maps numbers, spellings and currencies, and it moves WER by several points. It is
  English-only; use CER or the leaderboard's multilingual normaliser elsewhere. Disfluency and
  partial-word handling (AMI, Buckeye, CANDOR) must be stated.
- **DER collar and overlap.** pyannote defaults to collar 0 with overlap scored. Older papers use a
  0.25 s collar and drop overlap, and `md-eval` applies the collar on both sides of each boundary
  (2t). Always store `(collar, skip_overlap, uem)` with the number. Use VoxConverse v0.3, not v0.2.
  The AMI "only_words" and "word_and_vocalsounds" references differ.
- **VAD on AVA-Speech** shrinks as YouTube videos disappear. Record the actual file list and hash.
- **SV minDCF** has incompatible parameterisations (P_target 0.01 vs 0.05). VoxCeleb1-O, -E and -H
  are different lists. Use the cleaned lists.
- **Enhancement metrics** are sample-rate bound: VoiceBank is 48 kHz and PESQ-WB is 16 kHz. DNSMOS
  is non-intrusive and drifts across model versions. CapnoBase forbids training/tuning on the
  benchmark, and the same discipline should apply to every test set here.
- **SER taxonomies.** Label sets differ: CREMA-D has 6 emotions, RAVDESS 8, IEMOCAP is usually cut
  to 4 with excitement merged into happy, and MSP-Podcast uses both categorical and dimensional
  labels. Splits must be speaker-independent. Acted corpora overstate in-the-wild accuracy.
- **Audio event taxonomies.** AudioSet ontology (FSD50K) vs. ESC-50 classes. Use a mapping table
  and do not report cross-taxonomy zero-shot numbers as accuracy.
- **f0 conventions.** Gross-error threshold (20% vs 50 cents), octave errors, and how V/UV
  disagreements enter the score. PTDB-TUG's reference is RAPT on the laryngograph signal, so it is a
  reference, not truth. Jitter/shimmer/HNR have no ground truth; benchmark them as agreement with
  Praat (via Parselmouth) and rating correlation (PVQD). Do not report them as accuracy.
- **Health sets.** COUGHVID labels are crowdsourced plus SSL, with expert labels on >2k recordings
  and a 500-item private test. COVID labels are self-reported. Report detection, not diagnosis.
- **PII.** Synthetic PII sets overstate recall on real transcripts. Span-matching rule (exact vs
  partial vs token) and entity-type mapping must be fixed. n2c2's DUA forbids sharing outputs.
- **Pose/face.** COCO image rights are Flickr's. WIDER's ND clause forbids publishing crops. AU
  datasets differ in AU sets and intensity coding.
- **ECG.** Detector tolerance changes the ranking. 150 ms tolerances hide errors that sample-exact
  GUDB annotations reveal (Porr & Howell). Exclude non-beat annotations and paced records (MIT-BIH
  102, 104, 107, 217) consistently.
- **PPG/EDA.** The PPG-DaLiA reference is ECG-derived HR, so lag and windowing matter. No public
  gold-standard SCR annotations were found, so EDA can be benchmarked only as agreement and
  downstream classification.
- **Actigraphy/sleep.** Subject-wise splits are mandatory. Epoch length and alignment (30 s)
  matter, as do N3/N4 merging in Sleep-EDF (R&K → AASM) and the lights-off window. Capture-24
  forbids re-identification.
- **Eye tracking.** Sampling-rate mismatches (Lund2013), and event matching by sample vs. by event.
- **Licences of test sets.** The NC sets here are RAVDESS, ESC-50, WHAM! noise, MPII images, WIDER,
  part of FSD50K, and Buckeye (also ND). They can be run, but per-item outputs must not be
  published, and a commercial user of senselab cannot reproduce them. UCR/UEA carry no licence.
  Common Voice moved behind MDC terms in Oct 2025, so an old HF loader path will break.
- **Model licence ≠ data licence.** A benchmark run of an NC model (Oxford stepcount, HuBERT-ECG,
  Moirai) is research use. Tag the result with both licences (design.md 4.12 `License` record).

## 5. Cadence and hardware

| Tier | Contents | When | Hardware | Budget |
|---|---|---|---|---|
| Smoke | the "Smoke" column above, fixed item lists + checksums, cached data | every PR touching a task (path-filtered), nightly on `alpha` | CI CPU runner / laptop CPU, serial (no `-n auto`) | <15 min total; physio/text/eye <1 min each |
| Core | full open minimal suite (a), CPU-feasible parts: physio, actigraphy, sleep, EEG, f0/VQ, text, SER on CREMA-D | weekly + every release candidate | laptop/workstation CPU | 1–3 h |
| GPU | ASR (LibriSpeech, ESB subset), diarization, SV, enhancement/separation, FSD50K, COCO pose/captions, TS-FM probes | every release; weekly on `alpha` for ASR/diarization | one GPU (A100/H100 class) | 3–8 h |
| Extended | suite (b), full ESB, HEAR, NSRR, multilingual ASR | per minor release, and when a backend or model revision changes | cluster (ORCD array, typed H100), data under credentialed storage | overnight |

Each run records dataset version/DOI/hash, item list, model id + resolved commit SHA,
normaliser/scorer commit, metric parameters (collar, tolerance, P_target), hardware and wall time,
in line with the cache-key provenance rules. Timing figures are valid only from idle, dedicated
nodes. Publish aggregate metrics for all tiers. Publish per-item outputs only for CC BY / ODC-By /
CC0 / ODbL sets.

## Reviewer note (2026-10-30)

Spot-checked against the sources: ESC-50 is CC BY-NC 3.0, but its ESC-10 subset is CC BY 3.0, so it is a permissive option for the open smoke suite; PTB-XL is CC BY 4.0; VoxConverse is CC BY 4.0, with video copyright kept by the original owners.
