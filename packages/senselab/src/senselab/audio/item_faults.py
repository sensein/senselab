"""Typed per-item faults and audio checks shared by the isolated backends; imports only numpy and `senselab_worker`."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy
from senselab_worker.digest import COMMIT_HEX
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault, FaultClass, InfrastructureFault, ResourceFault, classify


def typed(exc: Exception) -> Exception:
    """Return `exc` as a typed fault: resource when it says so, else the item's own (content)."""
    if isinstance(exc, (ContentFault, ResourceFault, InfrastructureFault)):
        return exc
    if classify(exc) is FaultClass.RESOURCE:
        return ResourceFault(f"{type(exc).__name__}: {exc}")
    return ContentFault(f"{type(exc).__name__}: {exc}")


def raise_injected(item_id: str, faults: FakeFaults) -> None:
    """Raise the content or resource fault `faults` injects for `item_id`, if any."""
    if item_id in faults.fail_items:
        raise ContentFault(f"injected failure for item {item_id}")
    if item_id in faults.oom_items:
        raise ResourceFault(f"injected out of memory for item {item_id}")


def mono_samples(item: dict[str, Any], faults: FakeFaults, rate: int, who: str) -> tuple[numpy.ndarray, str]:
    """Return the item's mono float32 samples and its clock id, or raise its content or resource fault.

    `who` names the backend in the message; `rate` is the sampling rate it takes.
    """
    raise_injected(item["id"], faults)
    wire = item["inputs"]["audio"]
    group = wire["groups"]["audio"]
    samples = numpy.asarray(group["data"])
    if samples.shape[0] != 1:
        raise ContentFault(f"{who} takes mono audio, got {samples.shape[0]} channels")
    if float(group["sampling_rate"]) != rate:
        raise ContentFault(f"{who} takes {rate} Hz audio, got {group['sampling_rate']} Hz; resample it first")
    return samples[0].astype(numpy.float32), str(wire["anchor"]["clock_id"])


def init_model(init: Mapping[str, Any], keys: Collection[str], who: str) -> tuple[FakeFaults, Mapping[str, Any] | None]:
    """Return the faults and the pinned hub model a worker's `init` names; the model is None under `"stub": True`.

    `keys` are the init keys the backend `who` takes.

    Raises:
        ValueError: on an unknown key, a model that is not a hub model pinned by its 40-hex commit, or both `stub`
            and `model`.
    """
    unknown = sorted(init.keys() - set(keys))
    if unknown:
        raise ValueError(f"{who} init takes only {sorted(keys)}, got {unknown}")
    faults = FakeFaults.from_json(init.get("faults", {}))
    model = init.get("model")
    if init.get("stub"):
        if model is not None:
            raise ValueError(f"{who} init takes `stub` or `model`, not both")
        return faults, None
    if not isinstance(model, dict) or model.get("source") != "hf":
        raise ValueError(f"{who} loads a hub model, got {model!r}")
    if not COMMIT_HEX.fullmatch(str(model.get("identity"))):
        raise ValueError(f"a model is loaded by its 40-hex commit, never a ref: {model.get('identity')!r}")
    return faults, model
