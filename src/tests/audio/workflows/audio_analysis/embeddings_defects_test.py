"""Guards for two defects this change fixes.

Both are the kind that reappear: a silhouette coefficient renamed back into a probability, and a
docstring drifting from the signature it describes.
"""

import ast
import inspect
import re
from pathlib import Path

from senselab.audio.workflows.audio_analysis import embeddings as emb

_SOURCE = Path(emb.__file__).read_text(encoding="utf-8")


def test_no_silhouette_is_rescaled_into_a_probability() -> None:
    """0.5*(s+1) turns a clustering-geometry index into something that reads as a probability.

    CLAUDE.md names this defect class, and the L1 register documents its cost here: the signal
    produced 0.4022-0.4996 doubt across 214 buckets with stdev 0.0227 and earned the highest
    fusion weight of fifteen signals *because* it was near-constant; removing its consumer moved
    published presence doubt from 0.0682 to 0.0385.
    """
    assert "p_voice" not in _SOURCE, "p_voice reads as a probability; name it what it measures"


def test_the_module_docstring_agrees_with_the_signature() -> None:
    r"""A docstring claiming different defaults than the code is worse than none.

    A reader trusts it and passes nothing, expecting the documented behaviour. The original
    version of this guard (``doc.split("\\n")[2]`` plus a loose ``in`` check) could not fail: it
    would pass whether the docstring's numbers matched the signature's or not, because it never
    checked that the *specific pair of numbers* it found were the ones being claimed as defaults.
    This version extracts the "default X s with Y s hop" claim from the docstring as floats and
    compares each one against ``inspect.signature``'s actual default — so a docstring edited to
    say e.g. 2.0/1.0 while the signature stays 1.0/0.5 fails here, not silently in production.
    """
    sig = inspect.signature(emb.extract_per_window_embeddings)
    window_default = float(sig.parameters["window_s"].default)
    hop_default = float(sig.parameters["hop_s"].default)

    doc = ast.get_docstring(ast.parse(_SOURCE)) or ""
    match = re.search(r"default\s+([\d.]+)\s*s\s+with\s+([\d.]+)\s*s\s+hop", doc)
    assert match is not None, "module docstring no longer states a 'default X s with Y s hop' claim"
    doc_window_s, doc_hop_s = float(match.group(1)), float(match.group(2))

    assert doc_window_s == window_default, (
        f"docstring claims window_s default {doc_window_s}, signature default is {window_default}"
    )
    assert doc_hop_s == hop_default, f"docstring claims hop_s default {doc_hop_s}, signature default is {hop_default}"
