# Glides-Low-to-High: SPEECH raised on a degenerate diarization interval

Diagnosis written before the fix, from the failing run's own store. **The degenerate-interval
mechanism is confirmed**, and the earlier draft that asserted it was right about this run; the
b2ai-28 Glides runs did not contradict it so much as show it is not universal — a glide file can
also produce healthy word timings, and its sibling here does.

## The failure

| | |
| --- | --- |
| run | `1f4ea26f` |
| file | `sub-1f4ea26f-9764-4f89-a41a-66e248b9386f_ses-D987B8B0-A963-49B9-ABBE-2FD8FCC83E81_task-Glides-Low-to-High.wav` |
| duration | 6.4401875 s, 16 kHz mono |
| error | `SPEECH -> ValueError: 'waveform' must be provided as a (channel, time) torch Tensor.` |
| knock-on | `REDACT -> ValueError: no PII scan measurement in the store (N15)` — SPEECH died before writing one |

Reproduced in **both** campaigns, `out/` (`20260824-062502`) and `out-v3/` (`20260824-224801`),
byte-for-byte the same store facts.

## What the store says

The run's own `store.jsonl` holds exactly one `word` entity:

```json
{"attributes": {"recognizer": "Qwen/Qwen3-ASR-1.7B", "score": null, "text": "Ee",
                "timestamp_model": "Qwen/Qwen3-ForcedAligner-0.6B",
                "timestamp_source": "bundled_aligner"},
 "extent": [0.72, 0.72], "prov_type": "word"}
```

`n_words: 1`, extent span `(0.72, 0.72)`, degenerate words `1`. CrisperWhisper contributed no word
at all. The `SPEECH`/`diarize` activity records what followed from it:

```json
{"node": "SPEECH", "step": "diarize",
 "parameters": {"interval": [0.72, 0.72], "model": "pyannote/speaker-diarization-community-1"}}
```

Three SPEECH activities opened — `transcript`, `corroborate`, `diarize` — and none after, so the
raise came from the call that follows `diarize`'s parameters being recorded.

## The mechanism, end to end

1. The interval is `(min word start, max word end)` over the surviving words. With one word whose
   start equals its end, the interval is `(0.72, 0.72)`.
2. `extract_segments` (`src/senselab/audio/tasks/preprocessing/preprocessing.py:261-269`) checks
   `start < 0` and `end > dur` and **not** `start >= end`. It computes `s = e = 11520` and returns
   `Audio(waveform=<shape (1, 0)>)` without complaint.
3. `diarize_audios` hands pyannote `{"waveform": <(1, 0)>, "sample_rate": 16000}`. `pyannote`'s
   `Audio.validate_file` (`pyannote/audio/core/io.py:172-176`) rejects it on
   `waveform.shape[0] > waveform.shape[1]` — `1 > 0` — with the exact string the cluster recorded.

Nothing between steps 1 and 3 said anything was wrong. The empty tensor was manufactured by a helper
whose sibling `chunk_audios` has refused the same request since it was written
(`preprocessing.py:199-201`).

## The controlled comparison

The same subject and session, the other glide direction, in the same two campaigns:

| file | n_words | word extents | diarize interval | SPEECH |
| --- | --- | --- | --- | --- |
| `task-Glides-High-to-Low` | 2 | `[0.56, 6.08]`, `[0.56, 6.08]` | `[0.56, 6.08]` | completed |
| `task-Glides-Low-to-High` | 1 | `[0.72, 0.72]` | `[0.72, 0.72]` | raised |

A 5.52 s interval diarizes; a 0 s interval does not. The variable is the interval, not the task, the
host, the config hash (`e7893648350055d1` for all four) or the campaign.

## Which candidate mechanism the evidence supports

**The degenerate interval.** Not "short but non-empty": the interval is exactly zero-width, and the
raise is pyannote's shape check on an empty tensor, not a model failing on too little audio. No
minimum-length floor is implied by this evidence, and none is added — see the plan's step 5b.

## The fix

`extract_segments` gains the `start >= end` refusal its sibling already had, so a zero-length request
raises at the helper rather than producing an `Audio` every model-facing caller then passes on. In
`SPEECH` the refusal is caught around the crop and becomes a finding about the recording —
`diarization: "interval_selects_no_samples"`, `speaker_count: null`, and a flag — rather than an
uncaught error that costs the branch its PII scan and the file its release assessment.

## What is still open

Why the Qwen forced aligner placed `"Ee"` at a single instant on this file is not established here
and is a question for PREPROCESS, not SPEECH. It is upstream of the guard and the guard is correct
either way: a consensus that places every word at one instant is a fact about the transcript, and
this branch should report it rather than crash on it.
