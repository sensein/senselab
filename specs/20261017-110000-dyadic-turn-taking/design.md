# Dyadic turn-taking projection for mono two-person recordings

Task: `src/senselab/audio/tasks/turn_taking_projection/`. Target corpus: two-person Zoom
conversations recorded as **mono** mixes, 16 kHz, 27-98 min each, about 27 h in all.

## API

```python
separate_dyad(audio, diarization, method="masking", speakers=None, device=None) -> (Audio, Audio)
project_voice_activity(stereo_or_pair, model="vap_en_kyoto_mc", backchannel_model="vap_bc_en",
                       device=DeviceType.CPU, speakers=None, num_workers=3) -> VAPResult
summarize_projection(vap, turns, ipus, backchannels=None) -> dict
```

`VAPResult.frames` is a DataFrame at 10 Hz: `time, p_now_a/b, p_future_a/b, vad_a/b, p_bc_a/b`.
`time` is the end of the audio the frame has seen: frame `i` sits at `(i + 1) / 10` s. Measured,
not assumed: the correlation of `vad_a` with the true activity of speaker A peaks at that alignment
(0.985) against 0.940 one half-frame earlier and 0.955 one half-frame later.

## Decisions

**D-1 MaAI, not ErikEkstedt/VoiceActivityProjection.** MaAI (`maai==0.2.24`, MIT) ships English
stereo VAP (`maai-kyoto/vap_en_kyoto`, MIT) and an English backchannel fine-tune
(`maai-kyoto/vap_bc_en`, MIT) and was usable offline, so the fallback was not needed.

**D-2 Own venv, with the GUI and audio-device dependencies excluded.** `maai` declares PyQt5,
pyqtgraph, pygame-ce, PyAudio (needs PortAudio headers to build; no macOS wheel) and onnxruntime(-gpu)
for microphones, GUIs and the Mimi encoder. The offline CPC path imports none of them: `maai.input`
imports `pyaudio`/`pygame` at module top only for microphone classes, so the worker registers empty
stand-in modules for those two names. `ensure_venv` gained `excluded_packages`: the lock is compiled
from the full requirement set and the named distributions are dropped from the body
(`uv pip compile --no-emit-package`), recorded in an `# exclude:` header line and in the
requirements digest (only when non-empty, so no existing lock's digest changed). The venv is 707 MB.

**D-3 Drive MaAI's own streaming step offline.** `Maai.process(x1, x2)` is called once per 100 ms
frame over the whole recording with no threads or real-time input classes, collecting each result.
The frames are exactly what MaAI's real-time path emits: CPC encoder state and the 20 s KV cache
carry over every step, so long files need no chunking or overlap. Audio is read in 60 s blocks;
memory does not grow with duration. A batched re-implementation of the encoder was considered and
not done: it would reproduce MaAI's per-chunk convolution boundaries only by copying its internals.

**D-4 CPU, one thread per run, runs in parallel.** MaAI accepts only `cpu`/`cuda`. Forcing the
model onto MPS after construction ran at 5.5x real time against 11.4x for one CPU thread and 9.4x
for four (206.7 s dialogue, load average about 9-12): per-frame tensors are too small for the GPU
and thread fan-out. The backchannel model predicts channel 2's backchannels while channel 1 speaks,
so it runs twice (listener B: `(a, b)`; listener A: `(b, a)`); the VAP run and the two backchannel
runs go concurrently.

**D-5 Pinning.** Checkpoints are fetched by the parent with
`hf_hub_download(repo, file, revision=<literal 40-hex SHA>)` (`vap_en_kyoto@8450d1d8…`,
`vap_bc_en@3ff203ce…`) and handed to the worker as files (`Maai(local_model=...)`). The CPC encoder
comes from `dl.fbaipublicfiles.com` and is checked against SHA-256 `d0f474de…17f1` (the digest
prefix is in its file name). The worker makes no Hub call.

**D-6 Masking is the default way to split a mono mix.** Measured below.

**D-7 Separators are windowed and re-assigned per window.** 60 s windows, 2 s linear cross-fades.
Each window's sources are matched to the caller's speakers by energy share on frames where only one
speaker talks (`shares[k, s]` = fraction of all sources' energy in `s`'s exclusive frames that is in
source `k`; best pair of distinct sources). A separator's arbitrary output order then cannot swap
channels for the rest of the file, and a window where only one speaker talks still assigns.
Both channels are rescaled together so their sum has the mix's RMS (separators renormalise).

**D-8 Pauses holding a backchannel are not scored for hold/shift.** On the clean synthetic stereo,
scoring them inverted the hold/shift AUC to 0.15: a listener's "mm-hm" inside a within-turn pause
pulls `p_now` towards the listener. They are reported with `backchannel_in_pause=True`.

## Measurements

All on an M1 Pro (10 cores, 32 GB) shared with other jobs: the 1-minute load average was 7-12
throughout and is recorded beside every timing in `results/`. **Timings are upper bounds.** Re-run
everything with one command: `bash specs/20261017-110000-dyadic-turn-taking/bench/run_all.sh`
(three threads, one benchmark process at a time).

### Data

Three synthetic dialogues (`bench/make_dialogue.py`, seeds 0-2; 207, 235, 209 s) made with macOS
`say` (Samantha and Daniel), 40 turns each of 1-3 IPUs, gaps from -0.3 s (overlap) to 1 s, and
14-20 listener backchannels placed in within-turn pauses, sometimes overlapping. Clean per-speaker
channels are kept; the mono input is their sum plus white noise at -50 dBFS. Truth: every IPU,
turn and backchannel with its time.

### Separation vs masking (VAP on each input against VAP on the clean channels)

`bench/evaluate_inputs.py`, `bench/rescore.py`; per-seed rows in `results/rescored.json`.
Diarization: `oracle` = the true IPUs; `pyannote` = `speaker-diarization-community-1` on the mix
with two speakers (overlap-aware output), labels mapped to A/B by overlap. Shift events: changes of
`argmax p_now` held for 2 frames, matched in direction within 0.5 s.

| input | `p_now` corr | `p_now` MAE | VAD corr | `p_bc` corr | shift-event F1 |
| --- | --- | --- | --- | --- | --- |
| masking, oracle diarization | **0.990** | 0.023 | 0.989 | 0.951 | **0.987** |
| masking, pyannote diarization | **0.982** | 0.030 | 0.970 | 0.907 | **0.915** |
| MossFormer2, either diarization (seed 0 only) | 0.057 | 0.421 | 0.148 | 0.368 | 0.407 |
| mix duplicated into both channels | -0.017 | 0.425 | 0.305 | 0.523 | 0.278 |

Means over the three seeds (MossFormer2: seed 0). Per seed, masking-oracle `p_now` correlation is
0.992 / 0.990 / 0.990 and masking-pyannote 0.983 / 0.983 / 0.981. Masking with pyannote diarization
loses shift-event recall (0.82-0.87; precision 1.0): diarization misses some short IPUs, so some
model shifts happen late or not at all.

`MossFormer2_SS_16K` did not separate these conversations. On a 10 s excerpt of seed 0, source 0
correlated 0.99 with the mix (0.86 with speaker A, 0.50 with B) and source 1 0.16 with the mix: the
model left both speakers in one source. It is trained on fully overlapped mixtures; a conversation
is mostly one speaker at a time. Both diarizations gave identical results because the separator's
per-window output was identical and the assignment chose the same sources. Speed: 1449 s for
207 s (0.14x real time, CPU, three threads).

`pyannote/speech-separation-ami-1.0` **was not measured**: the repository (and
`pyannote/separation-ami-1.0`, which it loads) is gated and this account has not accepted its terms
(`GatedRepoError` 403). The code path is implemented (`method="pyannote"`, `resolve_model` + pinned
SHA `9486b106…`) and unit-tested only through the shared windowing/assignment with a stand-in
separator.

**Default: masking.** It is the only method measured to reproduce the clean-stereo projection, it
is free (8000x real time), and it needs no gated model. Its cost is the dependency on diarization
quality, quantified in the table.

### What the synthetic dialogues say about the measures (and their limits)

- On clean stereo (trajectories from seed 0) the model does not anticipate turn ends in `say` speech: the next speaker's
  `p_now` averaged 0.03 over the 0.5 s before a shift and rose only once the pause opened (0.23 at
  +0.1 s, 0.46 at +0.3 s, 0.65 at +0.5 s). Shift and hold pauses look the same for the first 300 ms
  (0.24 vs 0.26 at +0.2 s) and diverge after (0.55 vs 0.16 at +0.5 s), so the 50-250 ms hold/shift
  read gives balanced accuracy 0.50 and AUC 0.45 on clean input. Synthetic TTS carries no
  turn-yielding prosody; these numbers test the plumbing, not VAP's turn-end prediction on people.
- "Mix in both channels" gets a higher hold/shift balanced accuracy (0.59) than clean input, from
  `p_now` hovering at 0.50 +/- 0.01 around the threshold: noise, not information. Read the
  continuous scores and AUC, not only the 0.5 call.

### Throughput (`bench/bench_speed.py`, 10 min tiled from seed 0; `results/speed_10min.json`)

| component | x real time | load avg |
| --- | --- | --- |
| masking | ~8000 | 7.0 |
| VAP run (one thread) | 8.6-9.3 (64-70 s for 600 s) | 7-8 |
| one backchannel run (one thread) | 8.3-9.7 | 7-8 |
| VAP + both backchannel runs, 3 in parallel (`num_workers=3`) | 8.1 end to end | 8.0 |
| the same, serially (`num_workers=1`) | 3.0 end to end | 8.0 |
| MossFormer2 separation (3 threads) | 0.14 | 9 |
| pyannote community-1 diarization (bench only; caller supplies diarization) | 0.7-1.1 | 7-11 |

Estimate for the 27 h corpus, masking + VAP + both backchannel runs: 27 h x 3 runs / ~9x real time
= **~9 core-hours**; one file at a time with `num_workers=3` takes ~3.3 h wall; three files at once
(9 cores) ~1.1-1.5 h. VAP alone: ~3 core-hours. Separation with MossFormer2 would add ~190 h on CPU.
These are load-contaminated upper bounds; an idle-machine pass is pending.

## Licences

| artefact | licence | used |
| --- | --- | --- |
| MaAI package (`maai-kyoto/MaAI`) | MIT | yes |
| `maai-kyoto/vap_en_kyoto` (VAP, standard and multi-condition) | MIT | yes, default `_mc` |
| `maai-kyoto/vap_bc_en` (backchannel) | MIT | yes |
| `maai-kyoto/vap_en` (VAP incl. mono models) | **CC-BY-NC-ND-4.0, non-commercial** | **no**, deliberately not registered |
| CPC encoder `60k_epoch4-d0f474de.pt` (facebookresearch/CPC_audio, Libri-light) | CPC_audio code MIT; checkpoint has no separate licence statement | yes |
| `alibabasglab/MossFormer2_SS_16K` | Apache-2.0 | method `mossformer2` |
| `pyannote/speech-separation-ami-1.0` | MIT, gated | method `pyannote`, not measured |
| `pyannote/speaker-diarization-community-1` | CC-BY-4.0 | benchmark only |

## Limits of mono input

- VAP was trained on one speaker per channel. A mono mix carries no channel identity, so every
  dyadic reading depends on the diarization that splits it. With masking, a channel is exactly
  silent outside its speaker's segments, so `vad_*` partly echoes the diarization; missed short
  IPUs (often backchannels) become missing VAP input, and a speaker confusion becomes a channel swap
  for that segment.
- Overlapped speech is duplicated into both channels under masking; VAP then hears both speakers
  in both channels for the overlap.
- The "mix in both channels" row is what VAP makes of mono with no speaker information: `p_now`
  sits at 0.5. VAP cannot be applied to mono without a split.
- MaAI's single-channel (`vap_mono`) models exist only in the non-commercial `vap_en` repository
  and predict one speaker's own activity, not the dyadic `p_now`, so they were not used.
- Only synthetic `say` dialogues were measured: no codec, no cross-talk, no real prosody. The
  separation-vs-masking ordering is clear on them; the absolute measure values are not
  evidence about real conversations.
