"""The per-process model cache: loaded backend state keyed by model, LRU under a memory budget."""

import collections
import threading
from collections.abc import Callable

from senselab_worker.digest import COMMIT_HEX, SHA256_HEX
from senselab_worker.faults import ResourceFault

ModelKey = tuple[str, str, str, str]
"""`(backend, model_identity, device, dtype)`; `model_identity` is a 40-hex commit or a 64-hex weights digest."""


class ModelExceedsBudget(ResourceFault):
    """A loaded model alone is larger than the cache's budget."""


class ModelCache:
    """Backend state keyed by `ModelKey`, holding at most `budget_bytes` by the sizes `size_of` reports.

    One instance serves a process: the host's in-process backends or a worker's. Loads are serialised under a
    reentrant lock, so a loader may read the cache. The size of every key ever loaded is kept, one int per key.
    """

    def __init__(self, budget_bytes: int) -> None:
        """Hold at most `budget_bytes`.

        Raises:
            ValueError: if `budget_bytes` is not a non-negative int.
        """
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError(f"budget_bytes must be a non-negative int, got {budget_bytes!r}")
        self.budget_bytes = budget_bytes
        self._entries: collections.OrderedDict[ModelKey, tuple[object, int]] = collections.OrderedDict()
        self._sizes: dict[ModelKey, int] = {}
        self._lock = threading.RLock()

    def get_or_load(self, key: ModelKey, loader: Callable[[], object], size_of: Callable[[object], int]) -> object:
        """Return the state cached under `key`, else `loader()` cached at `size_of(state)` bytes.

        A hit becomes the most recently used entry. A key loaded before, whose size is known, first evicts least
        recently used entries until it fits, and then loads; a new key loads and then evicts until it fits. A failed
        load caches nothing.

        Raises:
            ValueError: if `key` is not four non-empty strings with a commit or weights digest as the identity,
                or `size_of` returns anything but a non-negative int.
            ModelExceedsBudget: if the loaded state alone exceeds the budget; nothing is evicted.
        """
        check_model_key(key)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key][0]
            known = self._sizes.get(key)
            if known is not None and known <= self.budget_bytes:
                self._evict_for(known)
            state = loader()
            size = size_of(state)
            if type(size) is not int or size < 0:
                raise ValueError(f"size_of must return a non-negative int, got {size!r}")
            if size > self.budget_bytes:
                raise ModelExceedsBudget(f"{key[0]} model {key[1]} needs {size} bytes; budget {self.budget_bytes}")
            self._evict_for(size)
            self._entries[key] = (state, size)
            self._sizes[key] = size
            return state

    def _evict_for(self, size: int) -> None:
        """Evict least recently used entries until `size` more bytes fit the budget."""
        while self._entries and self.used_bytes + size > self.budget_bytes:
            self._entries.popitem(last=False)

    @property
    def used_bytes(self) -> int:
        """The summed sizes of the cached entries."""
        with self._lock:
            return sum(size for _, size in self._entries.values())

    def clear(self) -> None:
        """Drop every cached state; the sizes of the keys loaded stay known."""
        with self._lock:
            self._entries.clear()

    def keys(self) -> list[ModelKey]:
        """The cached keys, least recently used first."""
        with self._lock:
            return list(self._entries)


def check_model_key(key: object) -> None:
    """Raise `ValueError` unless `key` is a `ModelKey` whose identity is a commit or weights digest, never a ref."""
    if not (type(key) is tuple and len(key) == 4 and all(type(k) is str and k for k in key)):  # noqa: PLR2004
        raise ValueError(f"a model key is (backend, model_identity, device, dtype) as non-empty strings, got {key!r}")
    identity = key[1]
    if not (COMMIT_HEX.fullmatch(identity) or SHA256_HEX.fullmatch(identity)):
        raise ValueError(f"model identity must be a 40-hex commit or 64-hex weights digest, never a ref: {identity!r}")
