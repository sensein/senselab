# Labelled material for testing the flow

Running note, extended as the design proceeds. Each entry names what it would exercise and what
label it carries. **Nothing here is verified yet** — availability, licence and label granularity all
need checking before any of it is relied on, and several are listed because they are the only
candidate rather than because they are known to be good.

The purpose is not a benchmark. It is to have, for each element and each decision in the flow, at
least one recording whose correct answer is known, so a change can be shown to break something.

## By taxonomy element

| element | candidate source | label it carries | exercises |
| --- | --- | --- | --- |
| inhalation, exhalation | Coswara (breathing-deep, breathing-shallow) | which manoeuvre was requested | airway spans, hysteresis edges, count prior |
| cough | Coswara (cough-heavy, cough-shallow), COUGHVID | cough present, and elicited vs spontaneous | event onsets, bout vs single |
| sustained vowel | Coswara vowels, Saarbruecken Voice Database, PVQD | vowel identity, and pathology rating in SVD/PVQD | phonation span, the offset criterion, MPT |
| syllable repetition | dysarthria corpora carrying DDK (TORGO and similar) | syllable sequence | ~10 ms onset precision, rate |
| word production | MyST, CSLU Kids, OGI Kids | orthography per utterance | child speech, short utterances |
| connected read speech | LibriSpeech, any Harvard-sentence corpus | reference text | alignment against a known script |
| connected spontaneous | Switchboard, CallHome, AMI | transcript, speaker turns | no-reference ASR, discourse timing |
| singing | NUS-48E, MIR-1K | lyrics and pitch | pitched-lexical material, the singing/music confusion |
| vocal imitation | ESC-50 animal classes as the *imitated target*, paired with human imitations | the imitated class | the D11 hazard: can the acoustic family contradict a correct classifier |
| other-speaker speech | AMI, CHiME | speaker turns, overlap | unaccounted-voice detection, expected vs intruding |
| laughter | AudioSet, FSD50K | laughter | non-lexical vocal, not-a-defect |
| crying | ESC-50 crying_baby, donateacry | infant cry | F-165's population end to end |
| environmental | ESC-50, UrbanSound8K, FSD50K, AudioSet eval | event class | confound separation, the source map's replacement |
| device and handling noise | none known — likely synthesised | — | gap: no labelled source found |
| silence | trivially constructed | — | the ADMIT rejection path |

## Why Coswara is worth checking first

One protocol, one participant per session, carrying breathing, cough, sustained vowels and counting.
That spans the airway branch, the phonation branch and part of the speech branch with the *same*
speaker, which is what makes it useful for the branch-selection decision rather than for any single
detector. If the hint-free gate routes a Coswara vowel recording to the phonation branch and its
cough recording to the airway branch, branch selection works.

## Decisions that still have no test material

- **The three-way gate.** No corpus carries "this recording should have been discarded" or "this one
  should have been flagged", which is why D6 leaves the thresholds liberal and underived. Synthesising
  this is the separate benchmark task already noted.
- **Task match.** Needs recordings labelled with the task that was *asked for* alongside what was
  actually produced, including mismatches. The b2ai protocol produces exactly this pairing; no public
  corpus does.
- **Unaccounted voice.** Needs the protocol's expected-speaker structure, not just speaker labels.
  AMI gives overlap and turns but has no notion of an expected speaker.
