"""The packaged measured equivalence classes, read from `runtime/data/equivalence.yaml` (the kernel does no I/O)."""

from importlib import resources

from senselab_core.kernel.identity import Equivalence, parse_equivalence


def load_equivalence() -> Equivalence:
    """Return the packaged equivalence table, read-only; pass it, or another measured table, as `equivalence`."""
    text = resources.files("senselab_core.runtime").joinpath("data", "equivalence.yaml").read_text("utf-8")
    return parse_equivalence(text)


EQUIVALENCE: Equivalence = load_equivalence()
"""The packaged measured equivalence classes; the default of `ResultStore(equivalence=)`."""
