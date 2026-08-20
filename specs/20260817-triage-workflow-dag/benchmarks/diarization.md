# Diarization

## pyannote gets the count right and two thirds of its evidence wrong

`pyannote/speaker-diarization-community-1` on the reference recording returns three segments, all
`SPEAKER_00`:

| segment | what is actually there |
| --- | --- |
| 7.95–9.01 s | **cough 1** |
| 9.51–10.19 s | **cough 2** |
| 11.62–13.06 s | the speech |

**1.74 s of the 3.18 s it calls speech is cough.** So the count is usable while anything derived from
its spans — speech duration, turn count, speaking rate — is not, unless the non-speech is excluded first.

## Restricting the interval removes them, rather than withdrawing them

Applying pyannote only to `[first word start, last word end]` — the speech branch's rule — is measured
against applying it to the whole file:

| input to pyannote | segments | speakers | onset error | offset error |
| --- | --- | --- | --- | --- |
| whole file, 14.03 s | **3** — two of them coughs | 1 | **0 ms** | −140 ms |
| speech span only, 1.41 s | **1** | 1 | +160 ms | **0 ms** |
| span ± 0.5 s, 2.41 s | 1 | 1 | −70 ms | +310 ms |
| span ± 1.0 s, 3.41 s | 1 | 1 | −30 ms | +130 ms |
| label extent, 1.58 s | 1 | 1 | +30 ms | +50 ms |

pyannote runs without complaint on 1.41 s, so short intervals are not a risk on this material. The
restriction eliminates the cough segments outright rather than requiring them to be withdrawn afterwards,
and it trades boundary errors rather than improving both: the whole file gives an exact onset and a
−140 ms offset, the restricted interval an exact offset and a +160 ms onset. Padding makes both worse.

`airway_spans` withdrawal is still needed for an airway event that falls *inside* the speech interval,
which this recording does not contain. Segments are withdrawn rather than relabelled, because a withdrawn
segment is what a reader needs to see when the count and the spans disagree.

Its speech segment: onset **11.62 s against a label of 11.62 s** — exact — and offset 13.06 against
13.20, **−140 ms early**. The onset-sharp / offset-early asymmetry that holds across every instrument
here.

## Escalation is a second opinion, not a replacement

No alternative diarizer is established as better. The available figures compare diarizers at different
speaker counts — VibeVoice 95% at k=2, DiariZen 75–90% at k=2–3, pyannote 85% — which is not a
comparison. An earlier claim that "every alternative was worse than pyannote" generalised k=1 figures and
was wrong.

## ASR edges

CrisperWhisper on the verified speech span: onset **−13 ms**, offset **−27 ms**, coverage **98.3%** —
best of six instruments. Whisper large-v3-turbo on the same span: onset **+187 ms**, coverage 87.5%, so
it is not a member of the fold for edges.

CrisperWhisper token edges on four verified airway windows: +20, +32, −26, −10 ms. Two instruments
agreeing to ~30 ms would be the strongest onset evidence available, but no disagreement rule has been
derived, so the airway branch reads word *presence* only.
