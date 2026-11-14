"""Injected per-item faults and audio checks shared by the isolated backends; imports numpy and `senselab_worker`."""

from collections.abc import Collection, Mapping
from typing import Any

import numpy
from senselab_worker.digest import COMMIT_HEX
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault, ResourceFault
from senselab_worker.units import NS_PER_S


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


def mono_samples(
    item: dict[str, Any], faults: FakeFaults, rate: int, who: str, min_input_ns: int | None = None
) -> tuple[numpy.ndarray, str]:
    """Return the item's mono float32 samples and its clock id, or raise its content or resource fault.

    The item's audio is a wire form. `who` names the backend in the message; `rate` is the sampling rate it takes and
    `min_input_ns` the shortest input it accepts, its declared `BackendSpec.min_input_ns`.
    """
    raise_injected(item["id"], faults)
    return mono_check(*wire_audio(item["inputs"]["audio"]), rate, who, min_input_ns)


def wire_audio(wire: Mapping[str, Any]) -> tuple[numpy.ndarray, float, str]:
    """Return `(samples (channels, n), sampling_rate, clock_id)` of an `audio@3.0` wire form."""
    group = wire["groups"]["audio"]
    return numpy.asarray(group["data"]), float(group["sampling_rate"]), str(wire["anchor"]["clock_id"])


def mono_check(  # noqa: PLR0913 - the audio's three parts, the rate and name the backend takes, and its minimum
    samples: numpy.ndarray, sampling_rate: float, clock_id: str, rate: int, who: str, min_input_ns: int | None = None
) -> tuple[numpy.ndarray, str]:
    """Return the mono float32 samples and clock id, or raise a content fault.

    The fault is for stereo, a rate other than `rate`, or fewer than `min_input_ns` of audio.
    """
    if samples.shape[0] != 1:
        raise ContentFault(f"{who} takes mono audio, got {samples.shape[0]} channels")
    if sampling_rate != rate:
        raise ContentFault(f"{who} takes {rate} Hz audio, got {sampling_rate} Hz; resample it first")
    if min_input_ns is not None and samples.shape[1] * NS_PER_S < min_input_ns * rate:
        raise ContentFault(
            f"{who} needs at least {min_input_ns} ns of audio, got {samples.shape[1] * NS_PER_S // rate}"
        )
    return samples[0].astype(numpy.float32), clock_id


def init_model(init: Mapping[str, Any], keys: Collection[str], who: str) -> Mapping[str, Any] | None:
    """Return the pinned hub model a worker's `init` names; None under `"stub": True`.

    `keys` are the init keys the backend `who` takes.

    Raises:
        ValueError: on an unknown key, a model that is not a hub model pinned by its 40-hex commit, or both `stub`
            and `model`.
    """
    unknown = sorted(init.keys() - set(keys))
    if unknown:
        raise ValueError(f"{who} init takes only {sorted(keys)}, got {unknown}")
    model = init.get("model")
    if init.get("stub"):
        if model is not None:
            raise ValueError(f"{who} init takes `stub` or `model`, not both")
        return None
    if not isinstance(model, dict) or model.get("source") != "hf":
        raise ValueError(f"{who} loads a hub model, got {model!r}")
    if not COMMIT_HEX.fullmatch(str(model.get("identity"))):
        raise ValueError(f"a model is loaded by its 40-hex commit, never a ref: {model.get('identity')!r}")
    return model
