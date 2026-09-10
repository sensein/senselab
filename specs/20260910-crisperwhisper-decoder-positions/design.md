# CrisperWhisper's 448-position overrun

## Incidence

A 62,550-recording triage run produced three failures. Two of them were lost entirely — not just
their `asr_crisperwhisper` derivative — to the same error:

```
RuntimeError: PREPROCESS: 1 block(s) failed unexpectedly:
  asr_crisperwhisper: RuntimeError: No position encodings are defined for positions >= 448,
  but got position 448
```

The two recordings (listed in `/orcd/scratch/bcs/002/satra/triage_full_20260908/retry_asr.txt`):

| recording | duration |
| --- | --- |
| `sub-5638512b…_task-rainbow-passage` | 77.44 s |
| `sub-bdf1f48d…_task-random-item-generation-v2` | 59.81 s |

`preprocess.py:1955` classifies `ValueError`/`LookupError` as a cascading absence and continues;
anything else is appended to `hard_failures`, and `preprocess.py:1963-1965` raises once every block
has run — discarding the recording's already-computed `PreprocessResult`, including the other
recognizer, YAMNet, HeAR, spans and SQUIM. A bare `RuntimeError` out of CTranslate2 is in the
second class.

## Duration is not the discriminator

Measured duration distributions from the same run:

| family | n | sampled median | sampled max | failures |
| --- | --- | --- | --- | --- |
| rainbow-passage | 898 | 27.70 s | 81.97 s | 1 (at 77.44 s) |
| random-item-generation | 473 | 56.66 s | 120.91 s | 1 (at 59.81 s) |
| story-recall | 1,549 | 51.18 s | 154.76 s | 0 |
| free-speech | 5,194 | 30.01 s | 43.00 s | 0 |

A 323 s rainbow passage
(`sub-c9b77a28…_ses-2B071659…_task-rainbow-passage.wav`, 10,340,202 B of 16 kHz mono PCM16) is in
the corpus and transcribes fine. Four times the length of either failure.

That control is not merely longer, it is *more* dysfluent: its transcript carries
`h- what h- what h- what h- what h- what h- what` and `in is p- p- p- a- s- p- sik- sik- sik- abo-`.
It runs 13 windows and every prompt stays between 26 and 39 tokens (longest context word: 21
characters). A repetition loop is harmless as long as it has interior spaces, because the 12-word
cap then does bound it. What the two failures have is a loop with **no** spaces.

## Where 448 comes from

448 is Whisper's `max_target_positions`: the decoder holds 448 learned position encodings, indexed
0…447. CTranslate2's C++ decoder raises when a step indexes past the table. The string is not in
any Python source — it is compiled into `libctranslate2`, which is why the condition surfaces as a
bare `RuntimeError` with no type to key on.

The bound is on the **absolute** decoder position, so it covers prompt **and** generated tokens:

    len(prompt_tokens) + n_generated  must stay under 448

Measured directly against `ctranslate2.models.Whisper.generate` (stock ctranslate2 4.8.2, macOS
arm64, `nyralabs/CrisperWhisper2.0_turbo` converted to CT2 float32), with EOT suppressed so the
decode cannot stop early and `max_length=512` — the value `crisperwhisper.engine.CT2Engine.generate`
computes for the shipped `max_new_tokens=256`:

```
prompt_len=4    max_length=512 -> generated 256
prompt_len=104  max_length=512 -> generated 256
prompt_len=189  max_length=512 -> generated 256
prompt_len=193  max_length=512 -> generated 256
prompt_len=194  max_length=512 -> RuntimeError: No position encodings are defined for positions >= 448, but got position 448
```

CTranslate2 caps the run at 256 new tokens regardless of the prompt (`max_length=512` yields
`min(512 // 2, 512 - prompt_len + 1)`), so the failure condition reduces to a single inequality:

    prompt_len >= 194

Nothing in the decode adapts to a long prompt. The generated budget stays at 256 and the prompt is
simply added on top.

## The mechanism: a repetition loop becomes the next chunk's prompt

Both recordings are longer than 30 s, so `crisperwhisper.model.CrisperWhisperModel.transcribe`
routes them through the default `longform_strategy="continuation"`: 30 s windows on a 26 s stride,
each window prompted with the last `context_words=12` **whitespace-separated words** of the
confirmed transcript so far, wrapped as `{mode_tags} <ctx> … <ectx>`
(`crisperwhisper/prompt.py:103-119`, `crisperwhisper/longform/continuation.py:466-494`).

`context_words` caps *words*. Nothing caps *tokens*. The library's own docstring states the
assumption — "at ~2.8 words/s a 4 s overlap is ~11 words" — which puts a normal 12-word context at
roughly 20-40 tokens. Measured over the 30 continuation prompts of the three control recordings
below: 26-41 tokens, every one of them.

A decode that falls into an **unbroken, space-free repetition loop** violates that assumption by
one to two orders of magnitude, because the whole loop is a single whitespace-separated word.

Traced on ORCD with the CT2 fork (`ctranslate2-crisperwhisper` 4.7.1.post3, the build the run
used, on the run's own snapshot `de0369c8…`), instrumenting `PromptBuilder._build` and every
`ctranslate2.models.Whisper` decoder entry point:

**rainbow-passage.** The speaker stutters severely on "division of twi-". Chunk 1's decode loops.
`crisperwhisper.hallucination.generate_with_repair_and_attention` fires its rewind-and-escape
repair three times — `prefix_len=41 max_new_tokens=224`, then `42/223`, then `43/222` — exhausts
`max_repairs=3`, and returns the looped tokens anyway. Chunk 2's prompt:

```
PROMPT mode=verbatim tokens=247 ctx_words=12 longest_word_chars=314
  ctx='The the rainbow is a division of twi- Twitwit- Twitwit- Twitwit- Twitwitwitwitwitwit…witw'
!! generate_greedy_with_attention RAISED prefix_len=247 max_new_tokens=256:
   No position encodings are defined for positions >= 448, but got position 448
```

One of the twelve "words" is 314 characters long. 247 + 256 = 503.

**random-item-generation-v2.** The task is reciting random letters; CrisperWhisper renders a letter
chain as one hyphenated word. Chunk 1's decode loops on `A-W-Z-`:

```
PROMPT mode=verbatim tokens=269 ctx_words=1 longest_word_chars=256
  ctx='K-N-O-Z-D-Q-A-W-Z-A-D-A-W-Z-A-D-A-W-Z-A-D-A-W-Z-A-D-A-W-Z-A-W-Z-A-W-Z-A-W-Z-A-W-Z-…-A-W-Z-'
!! generate_greedy_with_attention RAISED prefix_len=269 max_new_tokens=256:
   No position encodings are defined for positions >= 448, but got position 448
```

The entire twelve-word context is **one** word, 256 characters. 269 + 256 = 525.

Both failures are on chunk 2. Both prompts clear the measured 194-token threshold with room to
spare. Neither recording would fail at any length if its first window had decoded normally.

## What distinguishes these two recordings

Not duration, and not dysfluency on its own — the 323 s control above is more dysfluent than
either and passes. Traced context-prompt lengths, same instrumentation, same fork, same snapshot:

| recording | duration | windows | prompt tokens (chunks 2…n) | longest context word |
| --- | --- | --- | --- | --- |
| rainbow-passage (failing) | 77.4 s | 3 | 9, **247** → raise | 314 chars |
| random-item-generation-v2 (failing) | 59.8 s | 3 | 9, **269** → raise | 256 chars |
| rainbow-passage (control) | 323 s | 13 | 28-39 | 21 chars |
| story-recall (control) | 308 s | 12 | 26-29 | 14 chars |
| random-item-generation (control) | 121 s | 5 | 32-41 | 10 chars |

(A second 121 s random-item-generation control also completed without a raise.)

The discriminating property is a **space-free repetition loop in a window's decode**. Its two
observed sources are a prolonged stutter on a single syllable and a spelled letter chain — both
near-periodic acoustic sequences that drive the verbatim decoder into a loop the hallucination
repair cannot break, and both of which render as a single unspaced token run. Fluent connected
speech does not produce them; nor, as the control shows, does even severe dysfluency when the
repeated unit carries a space.

No level, clipping, silence or noise anomaly was found in either file; both are ordinary 16 kHz
mono PCM16 b2ai recordings.

## Backend dependence

The transformers backend (the non-Linux-x86 path, `crisperwhisper.py:70`) does not fail on either
recording: run locally it transcribes both, because its own decode of chunk 1 breaks the loop where
the CT2 decode does not (its context words stay at 43 and 40 tokens). HuggingFace's Whisper also
clamps rather than raising on a position overrun. The condition is therefore only reachable on the
CT2 path — which is exactly the Linux x86_64 GPU path every cluster run takes.

## The fix

The root cause is upstream: `crisperwhisper`'s continuation prompt is bounded in words and unbounded
in tokens. The correct upstream fix is to cap the context at a token budget when the prompt is
built — `PromptBuilder._build` already encodes the context, so it can drop leading context words
until `len(prompt_ids) + max_new_tokens < 448`. That is a change to a pinned third-party package
(`crisperwhisper[…]==2.0.1`) and is not made here.

What senselab can do is make the condition cost the derivative rather than the recording. The
knobs it does control do not fix it and are not used:

- `context_words` caps words, which is the broken invariant itself; a single 314-character word
  still overruns at `context_words=1`.
- `max_new_tokens` lower than 256 raises the prompt ceiling but does not bound it, and truncates
  legitimately dense chunks — the healthy chunks here use the full 256.
- `longform_strategy="chunked_lcs"` / `"token_lcs"` build no continuation prompt at all and would
  avoid the condition, but change transcription for all 62,550 recordings on an unmeasured basis.

So `crisperwhisper.py` classifies the condition instead, following `AudioTooShortForAST` /
`SpanTooShortForYAMNet` (`specs/20260909-ast-too-short-guard/`):
`CrisperWhisperDecoderPositionsExceeded` subclasses `ValueError`, and the host maps the worker's
`RuntimeError` to it when the message carries CTranslate2's position-limit signature.
`preprocess.py:1955`'s existing `except (ValueError, LookupError)` then records it as an absence
with the reason attached — no widening of that clause, and the recording keeps its Qwen transcript
and every non-ASR measurement.

Matching on the message text is the only signal available: CTranslate2 raises the condition from C++
with no distinguishable Python type. `test_other_worker_failures_stay_hard` pins that an unrelated
worker `RuntimeError` is still a hard failure, so the match cannot silently widen.
