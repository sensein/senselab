# Taxonomy screening set

## HeAR is barred from speech

On verified speech HeAR reports `Snore` 0.88 and `Speech` 0.01, across six measurements. Over the whole
labelled recording its `Speech` channel maxes at **0.088** while its `Snore` channel reaches 0.864 — and
over the speech region alone it returns `Snore` 0.864, `Throat Clear` 0.732, `Cough` 0.372. Those are
airway labels on speech, so the bar is not about a weak speech score; it is about strong wrong ones.

## YAMNet has no human-vocalic roll-up

`Human sounds`, `Human voice` and `Respiratory sounds` are all absent from the 521 AudioSet labels. Only
a union of specific labels is available, which is why a kind cannot be read off one node. YAMNet's
`Speech` is nonetheless the cleanest gate measured anywhere in this design — a 0.77-wide empty interval
below its confident scores (0.919 → 0.145).

## AST disagrees usefully, and shares a corpus

AST called one verified cough `Throat clearing` 0.96 where YAMNet said `Cough` 1.000. Useful
disagreement — but the two share AudioSet's corpus and label space, so they can be wrong together, which
is why they are **one family** and not two.

AST takes 1024 frames at a 10 ms hop and its extractor pads or truncates every clip to exactly 10.24 s,
so feeding it less buys nothing: 0.96 s in is 9% real audio and 91% padding.

## CrisperWhisper's labels are unreliable where its timings are not

It bounded one verified cough to onset −26 ms and offset −14 ms, and split the other into `[UH]` plus
`[breath]` covering 440 of its 640 ms. So its non-lexical *labels* cannot be trusted while its *timings*
can — and it is the only source of words, so the speech kind rests on it.

## Family counts are asymmetric

Airway has three eligible families (AudioSet, lexical, health-acoustic); speech has two, since HeAR is
barred. A single `min_families` would therefore mean different things for the two kinds, which is why the
parameter is per kind.

## Absence needs unanimity

A low score is ambiguous between "not present" and "present but quiet or masked". Masked events are the
case this workflow exists to catch — the mouth click at 12.65 dB of contrast, missed by every model
instrument, is the example — so no single family may retire a kind on its own.
