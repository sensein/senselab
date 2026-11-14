"""A backend whose output for each item is one interval spanning its `audio` input, on that input's clock."""

from typing import Any

from senselab_core.kernel.results import Intervals


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return each item's whole span as one `speech` interval on its audio's clock."""
    out = []
    for item in items:
        audio = item["inputs"]["audio"]
        end = audio.frames * 10**9 // int(audio.sampling_rate)
        spans = Intervals(start_ns=[0], end_ns=[end], label=["speech"], clock_id=audio.anchor.clock_id)
        out.append({"outputs": {"spans": spans}})
    return out
