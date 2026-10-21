# Turn-taking projection

Voice Activity Projection (VAP) for two-person conversations, and the dyadic turn-taking measures
read from it.

```python
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.turn_taking_projection import (
    project_voice_activity, separate_dyad, summarize_projection,
)

mix = Audio(filepath="call.wav")                          # mono, both speakers
channel_a, channel_b = separate_dyad(mix, diarization)    # list[ScriptLine] with speaker/start/end
vap = project_voice_activity((channel_a, channel_b))      # VAPResult, 10 frames/s
summary = summarize_projection(vap, turns, ipus)          # dict: per speaker, overall, events
```

A stereo recording with one speaker per channel goes straight to `project_voice_activity(stereo)`.

## Steps

**`separate_dyad`** turns a mono mix into one 16 kHz channel per speaker; the caller's diarization
defines who is who.

| `method` | A channel is | Model, licence |
| --- | --- | --- |
| `masking` (default) | the mix under that speaker's diarization segments, 50 ms raised-cosine ramps outside each segment; overlap goes to both | none |
| `mossformer2` | a `MossFormer2_SS_16K` source matched to the speaker | `alibabasglab/MossFormer2_SS_16K`, Apache-2.0 |
| `pyannote` | a `speech-separation-ami-1.0` source matched to the speaker | `pyannote/speech-separation-ami-1.0`, MIT, gated |

Separators run on 60 s windows with 2 s cross-fades; each window's sources are matched to the
caller's speakers by energy share on frames where only one of them talks, so a separator's
arbitrary output order cannot swap the channels for the rest of a recording.

On synthetic dialogues, masking reproduced the clean-stereo projection (`p_now` correlation 0.99
with the true segments, 0.98 with pyannote diarization); `MossFormer2_SS_16K` left most of both
speakers in one source (0.06) and ran at 0.14x real time on CPU. `pyannote` is implemented but was
not measured. Numbers and method: `specs/20261017-110000-dyadic-turn-taking/`.

**`project_voice_activity`** runs MaAI's stereo VAP (`maai-kyoto/vap_en_kyoto`, multi-condition
training by default) and its backchannel model (`maai-kyoto/vap_bc_en`) in an isolated `maai` venv,
on CPU. Both checkpoints are MIT and pinned to commits; the CPC encoder they build on is checked by
SHA-256. The backchannel model predicts the second channel's backchannels, so it runs once per
listener; the three runs go in parallel, one thread each.

`VAPResult.frames` columns: `time` (end of the audio the frame has seen), `p_now_a/b` and
`p_future_a/b` (probability that each speaker holds the floor 0-0.6 s / 0.6-2 s ahead; `a + b = 1`),
`vad_a/b`, and `p_bc_a/b` (probability that the speaker backchannels now).

**`summarize_projection`** reads the frames at the caller's events:

- *shifts* (consecutive turns by different speakers): the next speaker's mean `p_now` over the 0.5 s
  before the floor changed, whether it exceeded 0.5, and for how long it had (`lead_s`);
- *pauses* (mutual silences of at least 0.25 s between IPUs): the other speaker's `p_now` 50-250 ms
  into the pause, called shift or hold and scored against what happened (accuracy, balanced
  accuracy, AUC). A pause holding a backchannel is reported but not scored;
- *backchannel opportunities* (peaks of a listener's `p_bc` above 0.5, at least 1 s apart): matched
  to the backchannels the listener produced (-0.5 to +1 s): responsiveness (opportunities answered)
  and the fraction of the listener's backchannels timed to one.

Backchannels can be passed in; otherwise a listener IPU of at most 1 s inside the other speaker's
turn is taken as one. Every constant is in `data/parameters.json` with its derivation, and every
event row is returned with its continuous score.

## Limits

- Mono input loses what VAP was trained on, each speaker's own channel. Under masking a channel is
  silent wherever the diarization says that speaker is silent, so diarization errors (missed short
  backchannels, wrong speaker at overlaps) become VAP input errors, and overlapped speech is in both
  channels.
- The checkpoints are English and were trained on dialogue recorded one speaker per channel.
  Telephone/Zoom codecs and cross-talk were not measured here.
- The measures are model readings at annotated moments, not ground truth about the speakers.
