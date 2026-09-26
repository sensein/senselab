# SQUIM — per span, never per file

| region | duration | STOI | PESQ | SI-SDR |
| --- | --- | --- | --- | --- |
| whole file | 14.03 s | 0.864 | 1.34 | **−12.92** |
| the speech label alone | 1.58 s | **0.950** | **1.69** | **+12.40** |
| cough 1 | 0.57 s | 0.443 | 1.12 | −13.68 |
| exhalation 1 | 1.22 s | 0.388 | 1.10 | −15.84 |
| silence | 1.00 s | 0.397 | 1.23 | −13.51 |

**SI-SDR flips sign — a 25 dB swing** between the file and its speech. The speech is 11% of this
recording's duration, so the file-level figure is mostly a measurement of coughs and silence, and a gate
reading it would reject a recording whose speech is clean.

**A low score on non-speech is not a quality finding.** Pure silence scores STOI 0.397, so the estimator
returns a confident number for input it cannot interpret. That fact is what makes SQUIM usable as a *test*
of whether a span is speech, and unusable as a quality measure anywhere else.

## As the speech test in step 1

Over PREPROCESS's five spans:

| span | STOI | SI-SDR | YAMNet `Speech` max | coverage | verdict |
| --- | --- | --- | --- | --- | --- |
| 2.32–3.29 s | 0.404 | −15.73 | 0.145 | 0% | not speech |
| 5.32–6.22 s | 0.498 | −15.10 | 0.020 | 0% | not speech |
| 7.92–8.51 s | 0.313 | −14.52 | 0.030 | 0% | not speech |
| 9.61–9.96 s | 0.535 | −14.69 | 0.036 | 0% | not speech |
| **11.75–13.16 s** | **0.954** | **+16.90** | **0.993** | **80%** | **speech** |

| separation | speech | the rest | gap |
| --- | --- | --- | --- |
| STOI | 0.954 | 0.313–0.535 | **+0.419** |
| SI-SDR | +16.90 dB | −15.73…−14.52 dB | **+31.4 dB** |
| YAMNet max | 0.993 | 0.020–0.145 | **+0.848** |
| coverage | 80% | 0% throughout | binary |

Three independent measures separate the one speech span from the four airway spans, and coverage does it
binarily. The decision therefore rests on **agreement between two instruments** rather than a threshold
in either: with gaps this wide, a span landing between the populations is outside everything measured,
and a cut point there would fit a constant to a case never observed.

## Pre-emphasis takes SQUIM off distribution

Pre-emphasised, STOI rises 0.8635 → 0.9683 while SI-SDR falls −12.917 → −20.676 dB. One signal cannot be
materially more intelligible *and* far more distorted; the heads disagree because neither is being asked
about a signal like its training data. A STOI gate would be inflated by 0.10 — enough to flip a verdict
on a filter nobody intended as a quality change. Hence SQUIM reads the plain signal.

## Subjective head refused

It needs a non-matching reference, so the MOS it returns is a comparison against whatever recording
someone chose. Measured: **MOS 4.259 on a stream containing one isolated cough**, against 3.058 for the
input holding the actual sentence. If MOS is ever wanted, the reference becomes a declared config
artifact with a derivation.
