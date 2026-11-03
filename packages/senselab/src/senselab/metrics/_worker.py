"""The worker every metric backend shares: `load` and the per-item loop of `handle`.

A metric module's `handle` calls `serve` with its params model and its scorer; the scorer takes the item's input
values by port name, as kernel values, and returns a `Metric`.
"""

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel
from senselab_core.kernel._base import Frozen
from senselab_core.kernel.declare import Env
from senselab_core.kernel.results import Metric
from senselab_core.kernel.wire import from_wire_form, wire_form
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault

from senselab.item_faults import raise_injected

_INIT_KEYS = frozenset({"array_dir", "faults", "device", "model"})

HOST_BACKEND: dict[str, Any] = {
    "version": "1",
    "stack": "metric",
    "load_stub": {},
    "batch_invariant": True,
    "devices": ("cpu",),
    "env": Env.host(),
    "requirements": ("numpy",),
}
"""The declaration fields every metric backend shares; each adds `task`, `name` and `worker`."""

Scorer = Callable[[Mapping[str, Any], Any], Metric]
"""A metric: the item's inputs by port and the validated params, to the `Metric`."""


def load(init: dict[str, Any]) -> FakeFaults:
    """Return the worker's state, the faults to inject; `init` may hold `array_dir`, `faults`, `device` and `model`.

    Raises:
        ValueError: on any other key.
    """
    unknown = sorted(init.keys() - _INIT_KEYS)
    if unknown:
        raise ValueError(f"a metric backend's init takes only {sorted(_INIT_KEYS)}, got {unknown}")
    return FakeFaults.from_json(init.get("faults", {}))


def serve(
    state: FakeFaults, items: list[dict[str, Any]], params: dict[str, Any], model: type[BaseModel], score: Scorer
) -> list[dict[str, Any]]:
    """Return one `{"outputs": {"metric": <metric@2.0 wire form>}}` per item, or its fault."""
    validated = model.model_validate({k: v for k, v in params.items() if k != "backend_params"})
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            raise_injected(item["id"], state)
            inputs = {port: _kernel(value) for port, value in item["inputs"].items()}
            results.append({"outputs": {"metric": wire_form(score(inputs, validated))}})
        except Exception as exc:  # noqa: BLE001 - every per-item failure is reported as that item's typed fault
            results.append({"error": exc})
    return results


def _kernel(value: object) -> Frozen:
    return value if isinstance(value, Frozen) else from_wire_form(value)  # type: ignore[arg-type]


def require_same(what: str, first: object, second: object) -> None:
    """Raise a content fault unless `first == second`; `what` names the compared property."""
    if first != second:
        raise ContentFault(f"the two inputs differ in {what}: {first!r} and {second!r}")
