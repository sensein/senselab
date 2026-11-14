"""`Env.host`: a backend's `handle` called in this process, with its state in the process's model cache."""

import contextlib
import importlib
from collections.abc import Callable, Iterator
from typing import Any

from senselab_worker.codec import validate_message
from senselab_worker.device_peak import torch_meter
from senselab_worker.faults import PASS_THROUGH, FaultClass, ProtocolError, error_record
from senselab_worker.model_cache import Loan, ModelCache, ModelKey, check_model_key
from senselab_worker.serve import call_handle

from senselab_core.runtime import request_reports
from senselab_core.runtime.worker_client import WorkerFatal, item_error


class HostWorker:
    """`Env.host`: the backend module's `handle` called in this process, its state held in a `ModelCache`.

    `run` returns what `WorkerClient.run` returns for the same module, `init`, items and params, except that values
    are passed as they are rather than through their wire form. The state is loaded at construction and
    reloaded by `run` if the cache has evicted it.
    """

    def __init__(
        self,
        module: str,
        *,
        init: dict[str, Any],
        key: ModelKey,
        cache: ModelCache,
        size_of: Callable[[object], int],
    ) -> None:
        """Import `module` and load `init` under `key` in `cache`, sized by `size_of`.

        Raises:
            ValueError: if `init` uses the reserved key `array_dir`, or `key` is not a `ModelKey`.
            WorkerFatal: if importing `module` or loading fails, as a worker reports it.
        """
        if "array_dir" in init:
            raise ValueError("init key 'array_dir' is reserved")
        check_model_key(key)
        self._init = dict(init)
        self._key = key
        self._cache = cache
        self._size_of = size_of
        with _fatal():
            backend = importlib.import_module(module)
            self._load = backend.load
            self._handle = backend.handle
            self._loan()

    def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
        """Run one request in this process and return one result per item, in order, as `WorkerClient.run` does.

        Raises:
            ValueError: if `items` or `params` are malformed.
            WorkerFatal: if reloading an evicted state fails.
        """
        try:
            validate_message({"id": "in-process", "items": items, "params": params}, "request")
        except ProtocolError as exc:
            raise ValueError(f"malformed request: {exc}") from exc
        with _fatal():
            loan = self._loan()
        entries, device = torch_meter(lambda: call_handle(self._handle, loan.state, items, params))
        self._cache.answered(self._key)
        request_reports.record(len(items), since_load=loan.since_load, load_s=loan.load_s, device=device)
        return [
            item_error(item["id"], error_record(entry["error"])) if "error" in entry else {"id": item["id"], **entry}
            for item, entry in zip(items, entries)
        ]

    def _loan(self) -> Loan:
        return self._cache.get_or_load(self._key, lambda: self._load(dict(self._init)), self._size_of)


@contextlib.contextmanager
def _fatal() -> Iterator[None]:
    """Raise a failure to import or load the backend as the `WorkerFatal` a worker reports for it."""
    try:
        yield
    except PASS_THROUGH:
        raise
    except BaseException as exc:  # noqa: BLE001 - every import or load failure is fatal, as in a worker
        record = error_record(exc)
        raise WorkerFatal({**record, "fault": FaultClass(record["fault"])}) from exc
