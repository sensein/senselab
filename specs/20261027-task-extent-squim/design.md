# SQUIM over the task extent

Issue: [sensein/senselab#614](https://github.com/sensein/senselab/issues/614).

## Problem

`triage.tsv` published `squim_pesq`, `squim_stoi` and `squim_si_sdr` as the median of every live
`squim` assertion in the store. Those assertions are PREPROCESS's per-span readings, which exist as a
*speech test* ([benchmarks/squim.md](../20260817-triage-workflow-dag/benchmarks/squim.md): "usable as a
test of whether a span is speech, and unusable as a quality measure anywhere else"). A recording carries
about 84-96 spans with a median length of 0.27-0.32 s, and about 4 in 10 are `measure="gap"` spans —
silence, which SQUIM scores near its floor. The #614 reporter reproduced the release columns exactly
(Pearson 1.000 on 50 and on 60 recordings) and found:

| | release (all spans) | 10 s windows, same audio |
| --- | --- | --- |
| PESQ | 1.22 | 2.94 |
| STOI | 0.72 | 0.98 |
| SI-SDR | -6.8 dB | 17.4 dB |
| Spearman with `snr_db` | -0.37 / -0.20 / -0.27 | +0.12 to +0.27 |

The release values fall as SNR rises because louder, cleaner recordings have more distinct gaps. The
owner's decision (issue comment): recompute the columns over the detected task extent and publish a patch
release with the updated table.

## Decision

1. **A new measurement, `task_squim`, on the `plain` stream**, written by
   `task_squim.measure_task_squim` and replayed over finished stores by `scripts/extend_task_squim.py`.
   It reads senselab's existing objective-SQUIM function
   (`features_extraction/torchaudio_squim.extract_objective_quality_features_from_audios`, the call
   PREPROCESS makes), so there is no second model path.
2. **The extent** is the one `extend_task_audio.py` cuts: the live `task_audio` measurement's padded
   bounds (`task_audio.padding_s` = 0.25 s), or, where no cut stands (VERDICT discarded the recording, so
   the cut was retired), the same definition — `task_audio.task_extent` over the live task-extent spans —
   recomputed. `extent_source` records which. Discards therefore get a value, which the verification
   uses to check that discards read lower. `triage.tsv`'s `task_start_s`/`task_end_s` are the unpadded
   hull, so the measured extent is up to 0.5 s longer than `task_duration_s`.
3. **Windows** (`data/task_squim.yaml`, version 1): non-overlapping 10 s windows from the extent's start;
   a trailing remainder of at least 3 s is a window, a shorter one is dropped; an extent under 3 s is one
   window over the whole extent; an extent under 1 s gets no value. The medians over the windows read are
   the values; `windows_n` and `covered_s` say how much was read. Per-window readings are kept in the
   measurement's `windows`.
4. **Branches.** AIRWAY (breath, cough) and VOICE (sustained vowels, glides, maximum phonation time)
   families get no value; the driver records them `absent` with reason `airway family` / `voice family`,
   and retires any `task_squim` a recording carries from before. The columns therefore describe speech
   quality only. `no_value_branches` in `data/task_squim.yaml` holds the rule (v1 excluded AIRWAY only;
   v2 adds VOICE), and `triage_table.NO_SQUIM_BRANCHES` blanks the same two branches in the table as a
   second guard. See "Pilot" for why VOICE was dropped.
5. **Columns.** `recording_vectors` replaces `q_plain_squim_{pesq,stoi,si_sdr}` with
   `q_task_squim_{pesq,stoi,si_sdr}` and adds `q_task_squim_windows_n` (schema 28 → 29; `decode.js`
   and the viewer follow). `triage.tsv` keeps the published names `squim_*`, with new descriptions, and
   its schema goes 3 → 4. The per-span `squim` assertions are untouched and still feed the summary
   figure's span strip (`span_squim`).

## Provenance

One `TASK_SQUIM`/`measure` activity per pass, associated with the senselab software agent and a model
agent `torchaudio SQUIM_OBJECTIVE`. The weights are torchaudio-hosted, not a git repository, so there is
no 40-hex commit; the agent's `unresolved_reason` names the asset, its SHA-256 and the torchaudio
version, and `verify_weights` refuses a cached `squim_objective_dns2020.pth` whose SHA-256 is not the
pinned `2c54586f…e828`. The measurement is derived from the plain stream (and the `task_audio`
measurement where one stood) and carries `measure_key`: plain-stream digest, extent, window config,
weights digest and process version. A pass whose key matches the live measurement writes nothing; any
other pass retires the old measurement with an invalidation edge and writes a new one.

## Window derivation

Not fitted. 10 s is the window the #614 reporter measured with, so the corpus values are comparable with
theirs. 1 s is the shortest region in the squim benchmark, where a 0.9-1.6 s speech span already reads as
speech (STOI 0.95, SI-SDR +12 to +17 dB) and a 1.00 s silence reads at the floor; below it no reading in
the benchmark was taken. 3 s keeps a short trailing remainder (an utterance boundary and a pause) from
contributing a full-weight window, while still giving short tasks one value. All three are to be revisited
against the corpus VERIFY report.

## Cost

Local check (Apple silicon, 2 torch threads): 0.7 s per 10 s window, model load and weight check
0.02 s once cached. Most task extents are under a minute, so a recording costs a few seconds of SQUIM
plus the store read, write and BEP028 re-export; CPU is sufficient. The pilot's slice logs record
`elapsed_s` per recording, and the VERIFY report gives its median.

## Verification plan

`scripts/verify_task_squim.py --old <r37 triage.tsv> --new <new triage.tsv> --logs <slice logs>` writes
a markdown report, aggregates only (no participant, session, stem or transcript; groups under 20 rows
pooled out):

- old against new: median and IQR of each head, overall, by branch and by family;
- against SNR: Spearman of old and new with `snr_db` and `speech_over_residual_db` (expected: new
  positive, old negative), per branch, and the median per SNR decile (expected: new rises);
- against verdict and reason: by verdict, by primary reason, and weak-event or low-SNR reasons against
  the rest (expected: lower);
- against #614: clean read speech (SPEECH, pass, a *passage* or *sentences* family, `snr_db` at or over
  the corpus median) near PESQ 2.9, STOI 0.98, SI-SDR 17 dB;
- coverage: rows with a value by branch, driver outcomes by branch, windows per recording, seconds
  covered over extent seconds, extent source;
- SNR-decile medians for SPEECH rows alone, beside the non-airway deciles, so a decile pattern can be
  attributed to speech rather than to the mix of branches;
- sanity (exit 1 if violated): no value on an airway or voice row; none where the task extent is missing.

The run order is: a stratified pilot of about 200 recordings written into a mirrored tree
(`--out-root`), its vectors and table, and VERIFY against the r37 table; then the full manifest in
place, the vectors and `triage.tsv` rebuilt, and VERIFY again.

## Pilot (ORCD job 27069320, 2026-10-27)

About 200 recordings, stratified across task families and r37 SNR deciles, were measured into a mirrored
tree at commit 62f4b922 (windows v1, VOICE still read). Its `squim38/pilot/verify.md` showed SPEECH
(n = 121) moving to the #614 range: PESQ 1.21 → 2.80, STOI 0.68 → 0.98, SI-SDR −8.4 → 17.8 dB. Its
Spearman with `snr_db` went from negative to positive (PESQ −0.17 → +0.29, STOI −0.27 → +0.23). Clean
read speech (n = 11) read PESQ 2.80, STOI 0.985, SI-SDR 23.5 dB. Both sanity checks held. VOICE barely moved: PESQ 1.18 → 1.26 and SI-SDR −10.6 →
−9.7 dB (per-span → task-extent medians), with no relation to SNR. SQUIM is trained on speech (DNS
2020) and a sustained phonation is outside that distribution, so the owner decided on 2026-10-27 that
VOICE gets no value. That decision is `data/task_squim.yaml` v2.

The pilot's top `snr_db` decile (45.6–75.6 dB, n = 16) read lower (PESQ 1.77, SI-SDR 5.1 dB) than the
deciles below it. With VOICE rows in that decile the pilot cannot tell whether the drop is real in
speech. The SPEECH-only decile table in the corpus report answers that.
