"""The interpreter must be one whose RLock `multiprocess` can introspect.

`multiprocess` (the dill fork `datasets` depends on) copies a newer CPython's resource tracker,
whose `_stop_locked` calls `self._lock._recursion_count()`. `_thread.RLock` gained that method
after 3.12.0, so on 3.12.0 the call raises inside `ResourceTracker.__del__` at interpreter
teardown. Because `__del__` raises, `_stop_locked` never reaps the tracker child, and a process
that has finished its work sits waiting on it -- measured at ~18 minutes per task on a cluster
array, which had been worked around with `os._exit` in the driver rather than fixed.

The derivation is in `specs/20260905-resource-tracker-teardown/`.
"""

import threading

import multiprocess.resource_tracker


class TestTheResourceTrackerCanStop:
    """The teardown path `multiprocess` takes must not raise."""

    def test_this_interpreter_exposes_the_rlock_recursion_count(self) -> None:
        """The attribute `multiprocess` needs is the one 3.12.0 lacks."""
        assert hasattr(threading.RLock(), "_recursion_count"), (
            "this interpreter predates _thread.RLock._recursion_count, so multiprocess's "
            "ResourceTracker.__del__ will raise at teardown and leak its child process"
        )

    def test_stopping_the_tracker_without_the_blocking_lock_does_not_raise(self) -> None:
        """`__del__` takes this path, and it is the one that raised on 3.12.0."""
        tracker = multiprocess.resource_tracker._resource_tracker
        tracker.ensure_running()
        tracker._stop(use_blocking_lock=False)
