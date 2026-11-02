"""Injected per-item faults and audio checks shared by the isolated backends; imports numpy and `senselab_worker`."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy
from senselab_worker.digest import COMMIT_HEX
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault, ResourceFault

NS_PER_S = 1_000_000_000


def clamp_ns(start_s: float | None, end_s: float | None, floor_ns: int, duration_ns: int) -> tuple[int, int]:
    """Return `(start, end)` in ns of a timed stamp in seconds, inside `[floor_ns, duration_ns]` and ordered.

    A missing start is `floor_ns`; a missing end is `duration_ns`.
    """
    start = min(duration_ns, max(floor_ns, floor_ns if start_s is None else round(start_s * NS_PER_S)))
    end = min(duration_ns, max(start, duration_ns if end_s is None else round(end_s * NS_PER_S)))
    return start, end


def raise_injected(item_id: str, faults: FakeFaults) -> None:
    """Raise the content or resource fault `faults` injects for `item_id`, if any."""
    if item_id in faults.fail_items:
        raise ContentFault(f"injected failure for item {item_id}")
    if item_id in faults.oom_items:
        raise ResourceFault(f"injected out of memory for item {item_id}")


def mono_samples(item: dict[str, Any], faults: FakeFaults, rate: int, who: str) -> tuple[numpy.ndarray, str]:
    """Return the item's mono float32 samples and its clock id, or raise its content or resource fault.

    The item's audio is an `Audio` (a host backend) or its wire form (an isolated one).

    `who` names the backend in the message; `rate` is the sampling rate it takes.
    """
    raise_injected(item["id"], faults)
    samples, sampling_rate, clock_id = _audio(item["inputs"]["audio"])
    if samples.shape[0] != 1:
        raise ContentFault(f"{who} takes mono audio, got {samples.shape[0]} channels")
    if sampling_rate != rate:
        raise ContentFault(f"{who} takes {rate} Hz audio, got {sampling_rate} Hz; resample it first")
    return samples[0].astype(numpy.float32), clock_id


def _audio(value: Any) -> tuple[numpy.ndarray, float, str]:  # noqa: ANN401 - an `Audio` or its wire form
    """Return `(samples (channels, n), sampling_rate, clock_id)` of an `Audio` (host backends) or its wire form."""
    if isinstance(value, dict):
        group = value["groups"]["audio"]
        return numpy.asarray(group["data"]), float(group["sampling_rate"]), str(value["anchor"]["clock_id"])
    from senselab_core.runtime.audio_io import READER  # noqa: PLC0415 - only a host backend is handed an object

    return value.read(reader=READER), float(value.sampling_rate), str(value.anchor.clock_id)


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
