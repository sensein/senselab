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

**RETRACTED — "`airway_spans` withdrawal is still needed for an airway event that falls *inside* the
speech interval."** The rule was written for a case this reference recording does not contain, and the
campaign contained it 10 times. On b2ai-28 a diarizer segment overlapping an AIRWAY label assertion was
invalidated, so a single `Breathe` label of score 0.629 inside a 38 s narrative zeroed the file's
speaker count: **count 0 on 10 of 28 files**. Every word in those files went unattributed, and the
unattributed words cascaded into false PII withholds, because a finding whose speaker cannot be
resolved is treated as the target's.

Diarization is about speech. A diarizer segment is no longer withdrawn for overlapping an airway
event, and the speaker count is the live segment count. Restricting the interval to
`[first word start, last word end]` remains the only thing that removes cough segments, and it does so
by not showing them to pyannote at all.

**The speaker-count codomain {1, 2, ≥3} held only after that rule was removed.** The design states the
count is one of those three, with 2 triggering separation and ≥3 reported. Under the withdrawal rule
the observed codomain was {0, 1, 2, ≥3}, and 0 was the modal value on more than a third of the corpus —
a value the design had no branch for, which is why it read as "no speakers" rather than as a defect.
With withdrawal gone the count cannot be 0 for this reason: a file with words has an interval, an
interval yields at least one segment, and no segment is withdrawn. What is still unmeasured is whether
pyannote can return zero segments over a non-empty interval on this material; nothing here rules it out.

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
