"""Typed per-item faults shared by the `speech_to_text` backends; imports only `senselab_worker`."""

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
