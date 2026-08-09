# Declared capabilities for diarization backends

Status: design approved 2026-08-09, not yet implemented.

Six backends now reach `diarize_audios()` — Pyannote, NeMo Sortformer, VibeVoice-ASR-HF,
USC-SAIL child-adult, MOSS-Transcribe-Diarize, DiariZen. They share a return type and disagree
about almost everything else, and today the only way to learn how is to run one.

Follows [`../20260808-020643-new-model-integrations/design.md`](../20260808-020643-new-model-integrations/design.md),
which added four of the six.

## The problem, as measured

Running all six over three recordings on an H100 (21.5 s two-speaker conversation, 4.92 s clip,
11.3 s streaming capture) produced this field-occupancy matrix — populated fields per backend,
summed across files:

```
BACKEND                    SEGS   text  speaker  start    end  chunks
VibeVoice-ASR-HF              7      7        7      7      7       0
MOSS-Transcribe-Diarize       6      6        6      6      6       0
DiariZen                     10      0       10     10     10       0
child-adult                  19      0       19     19     19       0
```

and these speaker-label vocabularies:

```
VibeVoice-ASR-HF     ['0', '1', 'None']      <- 'None' was a defect, since fixed
MOSS                 ['S01', 'S02']
DiariZen             ['0', '1', '2']
child-adult          ['ADULT', 'CHILD']
Pyannote             ['SPEAKER_00']
```

Four observations follow, and each is a field in the design below:

1. **`text` is populated by exactly two of six.** A consumer reading a `ScriptLine` with
   `text=None` cannot tell "this backend does not transcribe" from "this segment had no words".
2. **`speaker` denotes different things.** Five emit a speaker *identity*; child-adult emits a
   *role*. The `audio_analysis` guards already branch on this, by matching model-id prefixes
   against a hand-maintained list.
3. **Labels are not stable across files.** DiariZen's VBx clustering numbers per audio: the same
   run produced `['1','2']` for one file and `['0','0','1','0']` for another. `'1'` in one file
   and `'1'` in another are unrelated.
4. **Speaker-count ceilings exist and are undeclared.** child-adult can only emit CHILD/ADULT, so
   it is a 2-speaker diarizer. Sortformer's is in its own name (`diar_sortformer_4spk`). The rest
   are undocumented — and on the 4.92 s clip VibeVoice and MOSS reported 1 speaker while DiariZen
   reported 2, which nothing in the codebase can currently adjudicate.

## Design

A frozen `DiarizationCapabilities` record per backend, declared as a module constant beside the
backend it describes, and surfaced through `model_registry.yaml`.

```python
@dataclass(frozen=True)
class DiarizationCapabilities:
    populates_text: bool
    speaker_label_kind: Literal["identity", "role"]
    labels_stable_across_files: bool
    max_speakers: int | None
    honors_speaker_hints: bool
```

### Why `ScriptLine` does not change

`ScriptLine` already provides a uniform key set — eleven fields, the same object from every
backend. "Harmonise the keys" cannot mean adding fields, because structurally they are already
identical. What is missing is the *declaration* of which of those keys carry meaning for a given
backend, and that is not per-segment data.

`ScriptLine` is also shared by ASR, forced alignment, and the workflow's harvesters. Changing a
type with that many consumers to solve a diarization-specific gap would be a far larger blast
radius than the problem justifies.

### Why the declaration is static, not per-call

A caller must be able to ask "can this backend give me more than two speakers?" **before** paying
for a 16 GB download and a GPU-minute. A per-call metadata block answers only after the cost is
sunk. Senselab already has both precedents — `COMPATIBILITY_MATRIX` in `compatibility.py` for
static per-function facts, `model_registry.yaml` for static per-model facts — and this is the same
kind of fact.

The rejected alternative is declaring statically *and* echoing per call. It is more informative and
creates two sources of truth that drift silently. Not worth it.

### The fields

| Field | Question it answers | Value today |
|---|---|---|
| `populates_text` | Is `text=None` a backend limitation or an empty segment? | VibeVoice, MOSS: `True`. Pyannote, Sortformer, DiariZen, child-adult: `False` |
| `speaker_label_kind` | Is this label an identity or a role? | child-adult: `"role"`. All others: `"identity"` |
| `labels_stable_across_files` | Can I compare `'1'` in file A with `'1'` in file B? | DiariZen: `False`. Others: `False` until measured — see below |
| `max_speakers` | What is the ceiling? | child-adult: `2`. Sortformer: `4`. Others: `None` until probed |
| `honors_speaker_hints` | Does `num_speakers`/`min`/`max` do anything? | Pyannote: `True`. All others: `False` |

**Count and kind are deliberately separate.** child-adult is, as a matter of speaker count, a
2-speaker diarizer — the reframing that dissolves its apparent special-case status. But its labels
still *denote* roles, which is what decides whether they may be fed to embedding clustering. A
2-speaker identity diarizer and a 2-speaker role classifier have the same `max_speakers` and must
be treated differently, so one field cannot carry both.

**`labels_stable_across_files` defaults to `False`.** Only DiariZen is measured; for the rest the
honest value is "not established". `False` is the safe default: a consumer that assumes instability
is merely conservative, whereas one that wrongly assumes stability silently merges two speakers.

**`max_speakers=None` means unmeasured, not unlimited.** Two of six have a known ceiling. The rest
get their value from the NeMo synthetic-speaker probe (separate spec), which scores each backend
against known ground truth from 1 to 8 speakers. This follows the repository's convention that a
number is written down with the measurement behind it — a ceiling asserted from a model card is
exactly the kind of unfitted literal that CLAUDE.md warns about.

## What this does not do

- It does **not** rewrite speaker labels into a common vocabulary. Values stay as each model emits
  them; only the key structure and its declared meaning are harmonised.
- It does **not** wire anything into `audio_analysis`. The existing role-label prefix list keeps
  working; migrating those guards to read `speaker_label_kind` instead of matching model-id prefixes
  is an obvious follow-up, and a separate change with its own review.
- It does **not** add a runtime check that a backend's output conforms to its declaration. Tempting,
  but it would run on every call to catch an error that can only be introduced at edit time; the
  tests below cover it once.

## Testing

- Every model id dispatchable from `diarize_audios` has a `DiarizationCapabilities` record. This is
  the test that stops a seventh backend being added without declaring itself.
- One test per measured claim in the table above, using the fake-processor fixtures the existing
  diarization suite already uses. No test may construct an `HFModel` without monkeypatching
  `check_hf_repo_exists` — an unmocked construction triggers a real `snapshot_download`.
- A test that `speaker_label_kind == "role"` exactly for the backends in the existing
  `ROLE_LABEL_ONLY_PREFIXES` list, so the new declaration and the old prefix list cannot disagree
  while both exist.

## Open item

`max_speakers` for Pyannote, VibeVoice, MOSS and DiariZen is `None` until the NeMo probe runs. The
probe is a separate spec; this design ships with `None` rather than guessing, and the probe fills
the values in.
