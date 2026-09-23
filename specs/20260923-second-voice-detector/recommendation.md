# Recommendation

The owner decides. This says what the measurements support and what they rule out.

## Ruled out, with the measurement that rules it out

**Retuning the incumbent's clustering threshold.** `params.clustering.threshold` is the only
count-related knob `speaker-diarization-community-1`'s config exposes. Swept over 13 values
from 0.05 to 0.95 on the known case and over 96 recordings at six and three values — **not
one recording changed its output at any value**, while the instrumentation confirms the
parameter reaches the `VBxClustering` object. There is nothing to tune here.

**Recording-level clustering of voice embeddings.** Measured properly: ECAPA, 2.0 s windows
on a 1.0 s hop over the diarizer's own speech spans, every window identical in length so the
duration confound cannot express itself; five clusterers; 26 derived statistics; positives
and negatives matched for length and speech type. **None reaches 0.70 AUC.** The best
catches the known case only by also calling **a third of genuine single-speaker
connected-speech recordings** two-speaker — roughly 14,000 false flags across the corpus to
win one true one.

This is the route the owner proposed, so it is worth being exact about *why* it fails. It is
not that the embedding cannot tell the two speakers apart — with complete linkage or Ward it
recovers the true partition at ARI 0.70. It is that the *between-speaker* distance
(centroid cosine 0.7285) is smaller than the *within-speaker* spread of the longer talker
(mean 0.7133), so the quantity that would have to carry the decision is the same size as
ordinary variation within one person's speech. There is no threshold in there.

One narrower finding worth keeping even so: `select_dominant_vectors` uses **average**
linkage, which on this recording peels three outlier windows off one speaker and returns
ARI **−0.063**, where complete linkage returns **0.697** and Ward **0.708** on the same
vectors. If that function is ever used for anything other than contamination rejection, the
linkage is the wrong one for finding a second talker.

## What to pursue: MOSS-Transcribe-Diarize in the `speech.second_diarizer` slot

`OpenMOSS-Team/MOSS-Transcribe-Diarize` is **the only backend that found the second voice
without being told the answer**. On the known case it returned two speakers, gave the second
9.37 s, and placed 9.33 of those seconds inside the 15.13 s ground-truth span — recall
0.617, precision 0.996 — recovering the turn change at 15.68 s against a true 15.69 s.

Sortformer, VibeVoice and the incumbent all returned one. DiariZen and child-adult returned
two but their second label holds 1.12 s and 0.18 s: they found a fragment, not the turn, and
crediting them would be taking a right answer from a wrong reason.

False-positive rate on the matched negative arm is the number that decides this, and it is
still accruing (jobs 23559301 and 23558701). The interim reading is **0 of 5** for MOSS on
long single-speaker connected speech.

Cost, amortized on an A100: MOSS runs at **2–4× realtime**, so a pass over the corpus's
~418 hours of SPEECH audio is roughly **100–200 GPU-hours**. Real, but an order of magnitude
below the 132 CPU-hours-per-pass the separation instrument already spends, and it replaces
nothing.

### Two changes the slot needs before it can hold this

1. **The gate is the wrong shape.** The archived design runs the second diarizer *only when
   pyannote's count is not 1*. On the known case pyannote's count **is** 1 — that is the
   entire failure. A corroborating detector that only runs on disagreement cannot see this
   class of error. The slot has to run when the incumbent says one, which is 47,500 of
   62,518 recordings, and that is where the cost above comes from.
2. **`_second_diarizer_model` cannot express a pyannote second diarizer.** It returns a
   plain `HFModel`; `diarize_audios` dispatches to pyannote on
   `isinstance(model, PyannoteAudioModel)`, and `PyannoteAudioModel` is a *subclass* of
   `HFModel`, not the reverse. So any `pyannote/...` id configured there falls through every
   prefix branch to `NotImplementedError`. The tests miss it because they monkeypatch
   `_second_diarizer_model`. MOSS is HF-prefixed so it would work through that path today,
   but the defect should be fixed rather than dodged.

## A cheap corroborating signal that already exists and is never read

`speaker_activity()` computes `max_concurrent_speakers` and `write_diarization` stores it on
every `<stream>_diarization` measurement. Nothing in the tree reads it back — verified by
grep, whose only other hits are assertions in `preprocess_test.py`. It is the overlap
statistic, the one reading that separates two people talking over each other from one
person, and it has never been consulted. On the known case it reads 1, so it would not have
caught this either, but it costs nothing and is already on disk.

## What the owner should also know about the existing positives

The population this pipeline currently calls two-speaker is largely not two people.

* Corpus-wide, 2,764 of 62,518 recordings (4.42 %) get ≥ 2 from the incumbent.
* In the 24 sampled two-speaker recordings with ≥ 20 s of speech, the second speaker holds a
  **median of 0.30 s**.
* **`task-maximum-phonation-time-v2-1` — one person sustaining a single vowel — gets two
  speakers on 10.1 % of 700 recordings.**

So the incumbent reliably splits off a tenth of a second and reliably does not split off
fifteen seconds. Whatever is done about the known case, that asymmetry is worth fixing in
how `speaker_count` is consumed, independently of which diarizer produces it.

## What was not run, and why

* No model was retrained or fine-tuned; the brief asked for a detector selection, not a
  training run.
* The `single` arm (sustained vowel, DDK) is reported but is not the operating comparison:
  at a median 6.2 s of audio only 3 of 24 recordings yield enough 2.0 s windows to cluster.
  The 40-recording `speech_single` arm replaced it.
* DiariZen's weights are **CC BY-NC 4.0, non-commercial only**. It was measured, and it lost
  on the merits anyway, but the licence should be settled before anyone revisits it.
