"""Query native-resolution signals at the samples a consumer wants, and cache the answers (D-25).

**Producers do not resample.** A producer that reduces onto a target grid has made an L2 decision —
which grid, and which reduction onto it — and destroyed the alternative before anyone could ask for it.
That is the defect D-18 found in the artifacts: ``native_window_s: 0.0619, resolution_s: 0.0169``
recorded on a row spanning ``0.0 → 0.1``, provenance describing a measurement the file did not contain.

So L1 emits at its own resolution (:mod:`.shapes` already does) and the *consumer* asks: this signal,
over this interval, reduced this way. The sampler answers, and remembers.

**The cache key is the derivative key.** D-21 names every projection ``(Target, Operator, Source)``, so
a query is one of those plus an interval — nothing new has to be invented to identify it. Three things
follow:

- D-22's *"materialisation is a caching and inspectability decision, not a semantic one"* becomes
  literal. A derivative is materialised iff something persisted it; the inline and stored forms are the
  same key with the same value.
- :class:`~.shapes.GridRelation` becomes the **dispatch**. ``RESAMPLE`` is arithmetic over finer frames,
  ``PROJECT`` assigns a window's value to the buckets it spans, ``REDUCE`` computes a per-bucket
  quantity the object does not have. Those are exactly the three ways a query can be answered.
- **Over-sampling stops being expressible by accident.** A consumer asking for 100 ms non-overlapping
  buckets gets them whatever the native hop is; a 0.1 s window at a 0.02 s hop cannot arise, because no
  producer chooses the output spacing.

This is not a storage layer. It reads signals and writes nothing; a materialised derivative is still
written by ``derive`` under ``StageIO``.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from senselab.audio.workflows.audio_analysis.keys import DerivativeKey, SignalKey
from senselab.audio.workflows.audio_analysis.shapes import (
    Categorical,
    GridRelation,
    Matrix,
    Measurement,
    Series,
    Spans,
    Tree,
)

__all__ = ["Sampler", "UnknownOperator"]


class UnknownOperator(Exception):
    """A query named a reduction the sampler does not implement.

    Raised rather than returning ``None``, because ``None`` already means *measured nothing here*. A
    typo borrowing that meaning is the absent-vs-zero confusion with a new disguise.
    """


class Sampler:
    """Answers ``(derivative key, interval)`` queries against native-resolution signals.

    Args:
        signals: ``{SignalKey → Measurement}`` at native resolution, as L1 emitted them.
        label_sets: ``{variant -> labels}`` for ``project_labels``. The selection is the operator's
            variant, so which labels count as speech is named on the key and replaceable without
            re-running a model.
    """

    def __init__(
        self,
        signals: Mapping[SignalKey, Measurement],
        *,
        label_sets: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        """Hold the native signals and an empty cache; nothing is computed until asked."""
        self._signals = dict(signals)
        self._label_sets = {k: tuple(v) for k, v in (label_sets or {}).items()}
        self._cache: dict[tuple[DerivativeKey, float, float], Any] = {}
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict[str, int]:
        """``{"hits", "misses"}`` — how much recomputation the cache avoided."""
        return {"hits": self._hits, "misses": self._misses}

    def at(self, key: DerivativeKey, start: float, end: float) -> Any:  # noqa: ANN401 — shape-dependent
        """The value of ``key`` over ``[start, end)``, or ``None`` where nothing was measured.

        Raises:
            ValueError: When ``key`` has more than one source. A fold is not a sample; the sampler
                answers per-signal projections and a cross-signal fold belongs to ``derive``.
            KeyError: When the source signal is absent — a signal that never ran is not a signal that
                measured nothing.
            UnknownOperator: For a reduction this does not implement.
        """
        if len(key.sources) != 1:
            raise ValueError(f"{key.target}/{key.operator.segment} has {len(key.sources)} sources; a sample takes one")
        cache_key = (key, round(start, 9), round(end, 9))
        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        self._misses += 1
        value = self._compute(key, start, end)
        self._cache[cache_key] = value
        return value

    def on_grid(
        self,
        key: DerivativeKey,
        *,
        duration_s: float,
        win_length: float,
        hop_length: float,
    ) -> list[dict[str, Any]]:
        """``[{"start", "end", "value"}, …]`` over a uniform grid the **consumer** chose.

        The grid is an argument here and nowhere in the producer, which is the whole of D-25. With
        ``win_length == hop_length`` the buckets do not overlap, so no two rows share a frame.
        """
        rows: list[dict[str, Any]] = []
        start = 0.0
        while start + win_length <= duration_s + 1e-9:
            end = start + win_length
            rows.append({"start": start, "end": end, "value": self.at(key, start, end)})
            start += hop_length
        return rows

    # ── dispatch on what a grid does to the shape ──────────────────────

    def _compute(self, key: DerivativeKey, start: float, end: float) -> Any:  # noqa: ANN401
        source = key.sources[0]
        if not isinstance(source, SignalKey):
            raise ValueError("a sample's source must be a signal; a derivative of a derivative is a fold")
        try:
            shape = self._signals[source]
        except KeyError:
            raise KeyError(f"no signal {source.target}/{source.producer} in this sampler") from None

        name, variant = key.operator.name, key.operator.variant
        if shape.grid_relation is GridRelation.RESAMPLE and name == "resample":
            return self._resample(shape, start, end, how=variant or "mean")
        if isinstance(shape, Categorical) and name == "project_labels":
            return self._project_labels(shape, start, end, variant=variant)
        if isinstance(shape, Spans) and name == "cover":
            return self._cover(shape, start, end)
        if isinstance(shape, Tree) and name == "word_coverage":
            return self._word_coverage(shape, start, end)
        raise UnknownOperator(f"{name!r} is not implemented for {type(shape).__name__}")

    @staticmethod
    def _frames_in(shape: Series | Matrix, start: float, end: float) -> range:
        """Frame indices whose start falls in ``[start, end)``."""
        n = len(shape.values) if isinstance(shape, Series) else len(shape.rows)
        lo = max(0, int((start - shape.start_s) / shape.hop_s + 1e-9))
        hi = min(n, int((end - shape.start_s) / shape.hop_s + 1e-9))
        return range(lo, hi)

    def _resample(self, shape: Measurement, start: float, end: float, *, how: str) -> Any:  # noqa: ANN401
        """Arithmetic over the native frames covering the interval — never a stored reduction."""
        if how not in ("mean", "max", "min"):
            raise UnknownOperator(f"resample variant {how!r} is not implemented")
        if isinstance(shape, Series):
            vals = [shape.values[i] for i in self._frames_in(shape, start, end)]
            return _reduce([v for v in vals if v is not None], how)
        if isinstance(shape, Matrix):
            frames = list(self._frames_in(shape, start, end))
            if not frames:
                return None
            out: dict[str, Optional[float]] = {}
            for index, channel in enumerate(shape.channels):
                col = [shape.rows[i][index] for i in frames]
                out[channel] = _reduce([v for v in col if v is not None], how)
            return out if any(v is not None for v in out.values()) else None
        raise UnknownOperator(f"resample is not implemented for {type(shape).__name__}")

    def _project_labels(self, shape: Categorical, start: float, end: float, *, variant: Optional[str]) -> Any:  # noqa: ANN401
        """Mass over a named label set, from the window covering the interval.

        A window's value is *assigned* to the buckets it spans, not divided among them: a 0.96 s
        window is one observation, and treating it as ten would assert ten measurements the model
        never made.
        """
        if variant is None or variant not in self._label_sets:
            raise UnknownOperator(f"project_labels needs a known label set; {variant!r} was not supplied")
        wanted = set(self._label_sets[variant])
        covering = [w for w in shape.windows if w.start < end and w.end > start]
        if not covering:
            return None
        masses = [sum(s.score for s in w.scores if s.label in wanted) for w in covering]
        return sum(masses) / len(masses)

    @staticmethod
    def _cover(shape: Spans, start: float, end: float) -> Optional[float]:
        """Fraction of the interval any span covers, counting overlap once."""
        width = end - start
        if width <= 0:
            return None
        clipped = sorted((max(s.start, start), min(s.end, end)) for s in shape.spans if s.end > start and s.start < end)
        if not clipped:
            return None
        total, reached = 0.0, float("-inf")
        for lo, hi in clipped:
            if hi <= reached:
                continue
            total += hi - max(lo, reached)
            reached = max(reached, hi)
        return total / width

    @staticmethod
    def _word_coverage(shape: Tree, start: float, end: float) -> Optional[float]:
        """Seconds of word overlap with the interval — the transcript's per-bucket reduction."""
        chunks = _chunks_of(shape.script_line)
        total = 0.0
        for chunk in chunks:
            lo, hi = chunk.get("start"), chunk.get("end")
            if lo is None or hi is None:
                continue
            overlap = min(float(hi), end) - max(float(lo), start)
            if overlap > 0:
                total += overlap
        return total if total > 0 else None


def _reduce(values: Sequence[float], how: str) -> Optional[float]:
    """``mean`` / ``max`` / ``min`` over the measured values, or ``None`` when there are none."""
    if not values:
        return None
    if how == "mean":
        return sum(values) / len(values)
    return max(values) if how == "max" else min(values)


def _chunks_of(script_line: Any) -> Iterable[Mapping[str, Any]]:  # noqa: ANN401
    """Word chunks from a ``ScriptLine`` or its dict form, flat."""
    if isinstance(script_line, Mapping):
        chunks = script_line.get("chunks") or []
        return [c for c in chunks if isinstance(c, Mapping)]
    chunks = getattr(script_line, "chunks", None) or []
    return [
        {"start": getattr(c, "start", None), "end": getattr(c, "end", None), "text": getattr(c, "text", None)}
        for c in chunks
    ]
