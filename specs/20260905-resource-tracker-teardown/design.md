# The resource-tracker teardown hang

## What happened

A cluster array task finished its work in 154 s and then sat for **18 minutes** doing nothing
before exiting: a defunct `python3` child, `uv` idle at 0% CPU. Every array task would have run to
its time limit after completing. The workaround was `os._exit` in the driver after results were
written, which has been carried in every cluster driver since.

Locally the same defect shows only as noise, printed after every test run:

```
Exception ignored in: <function ResourceTracker.__del__>
  File ".../multiprocess/resource_tracker.py", line 80, in __del__
  File ".../multiprocess/resource_tracker.py", line 89, in _stop
  File ".../multiprocess/resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
```

It was read as harmless teardown noise for a long time. It is the same defect.

## The mechanism

`multiprocess` is the dill fork that `datasets` depends on (`multiprocess<0.70.20`); we do not
import it ourselves. Version 0.70.19 copies a newer CPython's `resource_tracker`, whose
`_stop_locked` opens with:

```python
if self._lock._recursion_count() > 1:
    return self._reentrant_call_error()
```

`self._lock` is `threading.RLock()`, i.e. `_thread.RLock`. **CPython gained
`_thread.RLock._recursion_count()` after 3.12.0.** On 3.12.0 the attribute does not exist, so:

1. `ResourceTracker.__del__` calls `_stop(use_blocking_lock=False)`,
2. which calls `_stop_locked`,
3. which raises `AttributeError` on its first line.

Because `__del__` raised, the rest of `_stop_locked` never runs — and the rest is what closes the
"alive" file descriptor and `waitpid`s the tracker child. The child is never reaped. The parent
waits on it. That is the 18 minutes.

CPython's own `multiprocessing.resource_tracker` is unaffected: its lock is a plain
`threading.Lock` and it never calls `_recursion_count`.

## Measured

| interpreter | `_thread.RLock._recursion_count` |
| --- | --- |
| 3.11.9 | present |
| **3.12.0** | **missing** |
| 3.12.7 | present |
| 3.12.11 | present |
| 3.13.1 | present |

So the defect is confined to early 3.12 patch releases.

## Why it survived so long

Two reasons, both worth keeping in mind.

`.python-version` said `3.12` with no patch, so the venv resolved to whatever was installed —
3.12.0 here. And **CI runs on 3.11** (`tests.yaml`, with 3.12 commented out), which has the
attribute, so no CI job could ever see it. A defect that only appears on a developer's and a
cluster's interpreter, and only at teardown, is close to invisible.

## The fix

Pin `.python-version` to `3.12.11`. No source change: the bug is upstream, in a dependency we do
not import, and it is already fixed in every interpreter after 3.12.0.

A runtime shim patching `multiprocess`'s tracker was considered and rejected. Monkeypatching a
third-party library's teardown to paper over an interpreter we can simply not use trades a
one-line pin for a permanent maintenance burden, and would go on lying about the cause.

`src/tests/utils/resource_tracker_teardown_test.py` pins both halves: that the interpreter exposes
the attribute, and that the exact call `__del__` makes returns cleanly. Both were confirmed to fail
on 3.12.0 before the pin.

## What this retires

The `os._exit` workaround in the cluster drivers is no longer load-bearing once those runs use a
pinned interpreter. It is harmless to keep, but it should not be copied into anything new as though
it were a requirement.
