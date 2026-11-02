"""`wer`: word error rate of a hypothesis `Transcript` against a reference, as the Open ASR Leaderboard scores it.

Each side is its segment texts joined by a space, normalised by the leaderboard's English normaliser (or left
as written), split on whitespace, and aligned by word edit distance in which a run of words that concatenates to one
word of the other side (`white paper`, `whitepaper`) costs nothing. The value is errors over reference words.
"""

from collections.abc import Callable, Mapping
from typing import Any, Literal

from senselab_core.kernel.declare import TaskParams, backend, task
from senselab_core.kernel.results import Metric, Transcript
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault

from senselab.metrics import _worker
from senselab.metrics.open_asr.normalizer import EnglishTextNormalizer

WORKER_MODULE = "senselab.metrics.wer"
load = _worker.load
_NORMALISER: Callable[[str], str] = EnglishTextNormalizer()  # type: ignore[no-untyped-call]


class WerParams(TaskParams):
    """Params of `wer`: `normalise` is `open_asr` (the leaderboard's English normaliser) or `none` (text as written)."""

    normalise: Literal["open_asr", "none"] = "open_asr"


@task(name="wer", version="1", kind="transform")
class Wer:
    """Score a hypothesis transcript against a reference: the word error rate, over the reference's words."""

    inputs = {"reference": Transcript, "hypothesis": Transcript}
    outputs = {"metric": Metric}
    Params = WerParams


@backend(task="wer", name="numpy-wer", worker=WORKER_MODULE, **_worker.HOST_BACKEND)
class NumpyWer:
    """Word edit distance with free compound merging, in pure Python."""


def _words(transcript: Transcript, normalise: str) -> list[str]:
    text = " ".join(segment.text for segment in transcript.segments)
    return (_NORMALISER(text) if normalise == "open_asr" else text).split()


def _merges(words: list[str], at: int, target: str) -> list[int]:
    """Return the lengths `n >= 2` for which `words[at:at + n]` concatenates to `target`."""
    joined, found = words[at], []
    for n in range(2, len(words) - at + 1):
        joined += words[at + n - 1]
        if len(joined) > len(target):
            break
        if joined == target:
            found.append(n)
    return found


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Return the fewest insertions, deletions and substitutions turning `ref` into `hyp`, compounds merged free."""
    big = len(ref) + len(hyp) + 1
    cost = [[big] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    cost[0][0] = 0
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            here = cost[i][j]
            if i < len(ref):
                cost[i + 1][j] = min(cost[i + 1][j], here + 1)
            if j < len(hyp):
                cost[i][j + 1] = min(cost[i][j + 1], here + 1)
            if i < len(ref) and j < len(hyp):
                cost[i + 1][j + 1] = min(cost[i + 1][j + 1], here + (ref[i] != hyp[j]))
            if i < len(ref) and j < len(hyp):
                for n in _merges(ref, i, hyp[j]):
                    cost[i + n][j + 1] = min(cost[i + n][j + 1], here)
                for n in _merges(hyp, j, ref[i]):
                    cost[i + 1][j + n] = min(cost[i + 1][j + n], here)
    return cost[len(ref)][len(hyp)]


def score(inputs: Mapping[str, Any], params: WerParams) -> Metric:
    """Return the `wer` of `inputs["hypothesis"]` against `inputs["reference"]`.

    Raises:
        ContentFault: if the reference has no words.
    """
    ref = _words(inputs["reference"], params.normalise)
    hyp = _words(inputs["hypothesis"], params.normalise)
    if not ref:
        raise ContentFault("the reference has no words to score against")
    return Metric(name="wer", value=edit_distance(ref, hyp) / len(ref), unit="ratio", items=len(ref))


def handle(state: FakeFaults, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `metric@1.0` wire form per item, or its typed fault."""
    return _worker.serve(state, items, params, WerParams, score)
