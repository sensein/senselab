# A compatibility key that drifts fails silently

## The hazard

`check_compatibility(function_key)` looks the key up and returns **True** when it is absent:

```python
entry = COMPATIBILITY_MATRIX.get(function_key)
if entry is None:
    return True
```

For a caller probing an arbitrary key that is defensible. For `@requires_compatibility` it is not:
the matrix entry is what declares a task's dependencies **and its isolated subprocess venv**. A
decorated task whose key has drifted therefore **skips both, in silence**. Nothing raises at
import, nothing raises at call. The task runs without the environment it declared and fails later,
somewhere less obvious — or appears to work while doing something subtly different.

## How it surfaced

Audio-visual target-speaker extraction moved from `audio/tasks/target_speaker_extraction/` to
`video/tasks/`. Its matrix key is a **dotted module path matched as a string**, so nothing in the
language ties the key to the module's location. Had the key been left at `audio.tasks...`, the
move would have looked completely successful: imports fine, tests collect, task callable — and the
ClearerVoice venv silently never provisioned.

That is the worst shape a defect can take. A rename that breaks loudly costs minutes.

## The fix

`requires_compatibility` refuses an unknown key **at decoration time**, which is import time:

```
KeyError: 'audio.tasks.target_speaker_extraction.extract_target_speakers_from_videos' is not in
COMPATIBILITY_MATRIX. A decorated task whose key is absent would silently skip its dependency
check and its isolated-venv provisioning, because check_compatibility returns True for a key it
does not know. Add the entry, or fix the key. Did you mean:
video.tasks.target_speaker_extraction.extract_target_speakers_from_videos?
```

`difflib.get_close_matches` supplies the suggestion, which is what makes it useful for the moved-
module case: the old and new keys differ by one segment, so the near match is exactly the answer.

**`check_compatibility` itself is unchanged.** Its permissive behaviour is right for a query
function, and narrowing it would break callers legitimately probing keys the matrix does not hold.
The guard belongs where the declaration is made, not where a question is asked.

## Verified

- All **19** decorated keys in the package resolve today, so the guard is strict without breaking
  anything.
- The exact stale key from the move is refused, and the suggestion names the new path.
- Six matrix entries carry no decorator. That direction is legitimate — an entry may exist for a
  task reached another way — so it is not enforced.

## The test, and a guard on the guard

`src/tests/utils/compatibility_key_test.py` scans the package for decorated keys and asserts each
is in the matrix, then asserts the decorator refuses an unknown one.

A scan-based test has its own failure mode: if the regex or the source path is wrong it finds
nothing, and a parametrized test over an empty list **passes**. So it first asserts the scan found
at least 15 keys. That assertion earned itself immediately — the first version resolved its source
path to the repo root rather than `src/`, found zero, and would otherwise have reported a clean
run while checking nothing.
