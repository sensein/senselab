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

## Full run (squim38, 62,550 recordings, commit 451af99c)

The numbers are from `squim38/verify.md`. "Old" is r37's per-span medians and "new" the task-extent
windows. They are medians, with the IQR in brackets.

### Old against new

| rows | head | old n | old | new n | new |
| --- | --- | --- | --- | --- | --- |
| all with a value | PESQ | 49,056 | 1.21 (1.18–1.25) | 39,656 | 2.99 (2.06–3.60) |
| all with a value | STOI | 49,056 | 0.669 (0.579–0.768) | 39,656 | 0.986 (0.958–0.995) |
| all with a value | SI-SDR dB | 49,056 | −8.63 (−11.6 to −3.64) | 39,656 | 18.0 (9.36–23.6) |
| SPEECH | PESQ | 40,763 | 1.21 (1.18–1.26) | 39,656 | 2.99 (2.06–3.60) |
| SPEECH | STOI | 40,763 | 0.687 (0.599–0.784) | 39,656 | 0.986 (0.958–0.995) |
| SPEECH | SI-SDR dB | 40,763 | −7.80 (−11.0 to −2.55) | 39,656 | 18.0 (9.36–23.6) |

The old VOICE values (n = 8,293; PESQ 1.19, STOI 0.581, SI-SDR −11.7 dB) have no new counterpart.

### Spearman with SNR

| SNR column | rows | head | old | new |
| --- | --- | --- | --- | --- |
| `snr_db` | all with a value | PESQ / STOI / SI-SDR | −0.238 / −0.167 / −0.114 | +0.265 / +0.149 / +0.342 |
| `snr_db` | SPEECH | PESQ / STOI / SI-SDR | −0.262 / −0.230 / −0.184 | +0.265 / +0.149 / +0.342 |
| `speech_over_residual_db` | all with a value | PESQ / STOI / SI-SDR | +0.049 / +0.218 / +0.147 | +0.467 / +0.377 / +0.404 |

### SPEECH by `snr_db` decile (new medians)

| decile | dB | n | PESQ | STOI | SI-SDR dB |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.6–20.5 | 4,120 | 1.78 | 0.963 | 6.36 |
| 2 | 20.5–25.3 | 4,119 | 2.47 | 0.980 | 12.4 |
| 3 | 25.3–28.3 | 4,116 | 2.76 | 0.982 | 16.1 |
| 4 | 28.3–30.8 | 4,115 | 2.94 | 0.985 | 18.3 |
| 5 | 30.8–33.2 | 4,116 | 3.13 | 0.987 | 20.1 |
| 6 | 33.2–35.6 | 4,133 | 3.17 | 0.989 | 20.8 |
| 7 | 35.6–38.3 | 4,117 | 3.29 | 0.990 | 22.0 |
| 8 | 38.3–41.3 | 4,125 | 3.38 | 0.992 | 22.7 |
| 9 | 41.3–45.4 | 4,123 | 3.36 | 0.991 | 22.1 |
| 10 | 45.4–115.6 | 4,123 | 3.08 | 0.981 | 19.2 |

The old PESQ stays between 1.19 and 1.24 in every decile, and the old SI-SDR falls from −6.1 to −9.9 dB.

### SPEECH by `speech_over_residual_db` decile (new medians)

| decile | dB | n | PESQ | STOI | SI-SDR dB |
| --- | --- | --- | --- | --- | --- |
| 1 | −52.9–12.8 | 3,981 | 1.77 | 0.937 | 8.22 |
| 2 | 12.8–21.6 | 3,985 | 2.19 | 0.966 | 11.6 |
| 3 | 21.6–27.6 | 3,982 | 2.54 | 0.976 | 14.3 |
| 4 | 27.6–32.8 | 3,984 | 2.84 | 0.983 | 16.2 |
| 5 | 32.8–37.2 | 3,990 | 3.03 | 0.987 | 17.7 |
| 6 | 37.2–41.2 | 3,984 | 3.22 | 0.990 | 19.4 |
| 7 | 41.2–44.9 | 3,971 | 3.33 | 0.991 | 20.7 |
| 8 | 44.9–48.6 | 3,996 | 3.41 | 0.992 | 21.3 |
| 9 | 48.6–52.8 | 3,985 | 3.49 | 0.992 | 22.1 |
| 10 | 52.8–73.0 | 3,986 | 3.61 | 0.993 | 22.9 |

### Clean read speech

This is SPEECH rows with verdict pass, a *passage* or *sentences* family, and `snr_db` at or above the
corpus median of 31.2 dB. Old (n = 12,202): PESQ 1.19, STOI 0.648, SI-SDR −8.86 dB. New (n = 12,189):
PESQ 3.30 (2.42–3.75), STOI 0.991 (0.974–0.997), SI-SDR 22.0 dB (13.7–25.3). #614 reported 2.9, 0.98
and 17 dB on 10 s windows of a 110-recording sample.

### By verdict (new medians, rows with a value)

pass (n = 35,974): PESQ 3.02, SI-SDR 18.4 dB. review (n = 3,642): 2.64, 14.7 dB. discard (n = 40): 1.22,
1.06 dB. Rows with a weak-event reason (n = 18) read PESQ 1.34 and SI-SDR −1.05 dB.

### Coverage

| branch | rows | old with a value | new with a value | driver outcomes |
| --- | --- | --- | --- | --- |
| AIRWAY | 13,017 | 0 | 0 | 13,017 airway family |
| SPEECH | 41,226 | 40,763 | 39,656 | 39,656 measured; 1,468 no task extent; 102 under the 1 s floor |
| VOICE | 8,307 | 8,293 | 0 | 8,307 voice family |

- Windows per measured recording: 1 window for 28,900 recordings, 2–5 for 7,758, 6–10 for 2,607, and
  more than 10 for 391.
- The median padded extent is 4.37 s (IQR 3.23–17.5 s). The median fraction of it covered is 1.0.
- 39,616 extents came from the `task_audio` cut and 40 from the spans (discards).
- The driver took a median of 0.6 s per recording.
- Sanity checks: no airway row, voice row, or row without a task extent carries a value.

### The top-SNR-decile dip

In SPEECH, the new medians rise through decile 8 or 9 and then fall in decile 10 (45.4–115.6 dB): PESQ
3.36 → 3.08, SI-SDR 22.1 → 19.2 dB. The same dip appears in the non-airway decile table, so it does not
come from VOICE rows, which now carry no value. It does not appear against `speech_over_residual_db`,
where all three heads rise through the top decile. *Hypothesis, not tested:* `snr_db` is the plain
stream's active level over its noise floor, so a very high value is often a near-digital-silence floor,
for example a gated or noise-suppressed capture device. That processing lowers SQUIM's estimate even
though the measured SNR is high. Testing it needs the floor level and the device or capture-path
metadata for the top-decile rows.
