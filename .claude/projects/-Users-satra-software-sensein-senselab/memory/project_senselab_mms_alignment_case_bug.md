---
name: senselab MMS alignment case-handling bug
description: _preprocess_segments uppercases the transcript before dict lookup; MMS-eng tokenizer vocab is lowercase, so every alphabet character misses and alignment returns punctuation-only chunks
type: project
---

`senselab.audio.tasks.forced_alignment.forced_alignment._preprocess_segments` uppercases
each character of the transcript (`char_ = char.upper()`) before checking
`char_ in model_dictionary.keys()`. This works for the per-language wav2vec2 aligners
(uppercase A–Z vocab) but **silently corrupts alignment for MMS** (`facebook/mms-1b-all`),
whose vocab is lowercase a–z.

Symptom: alignment runs without error but the resulting ScriptLine chunks contain only
punctuation characters (apostrophes, dots, exclamations) — every alphabetic character
gets dropped from `clean_char` because the uppercase form never matches the lowercase
dict. Verified empirically on 2026-05-09 against `tutorial_audio_files/english_conversation_higgs_audio_v2.wav`
with Granite Speech 3.3 8B + Canary-Qwen 2.5B (both text-only ASR backends that need
post-MMS alignment).

**Verification** (uv run python):
```python
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/mms-1b-all", target_lang="eng")
sorted(k for k in processor.tokenizer.get_vocab() if k.isalpha())[:5]
# ['a', 'b', 'c', 'd', 'e']  — lowercase
```

**Failed first attempt:** lowercasing the transcript in the caller doesn't help —
`_preprocess_segments` does `char.upper()` regardless, so the lowercased letters get
re-uppercased before dict lookup and still miss MMS's lowercase vocab.

**Real fix in senselab/forced_alignment.py** (committed on branch
`20260508-173136-compare-uncertainty`): probe the dictionary's case once per call
(`"A" in dict_keys and "a" not in dict_keys`); apply `.upper()` only when the dict
is uppercase, otherwise `.lower()`. Apply the same case-folding to the per-word check
(`clean_wdx`).

**How to apply:** if you're touching `_preprocess_segments` or related code, preserve
this dict-case detection. Don't reintroduce unconditional `.upper()` — it'll silently
re-break MMS alignment.
