"""Capability-passing I/O: a stage can only name what it is allowed to name (D-18).

Four generations of guard were defeated, each by a mechanism its author had not enumerated — a name
list that omitted the fourth axis, a regex an alias slipped past, a glob that saw the workflow package
but not ``adaptive/``, a ``**`` with ``key=None`` permitting anything. That is not carelessness; it is
that **inspecting an undecidable property after the fact cannot terminate.** ``_PathResolver`` walks
the AST trying to evaluate path *expressions* to a declared pattern, and it cannot see a path handed
to a helper as a parameter — so an access it cannot prove conformant is not a permitted one, and the
worklist of deviations only grows.

**The reframing: there is nothing to resolve.** Paths are *derived from keys* (:mod:`.keys`), so a
stage holding a :class:`StageIO` cannot construct a path at all. It presents a key and is told yes or
no. The capability is therefore over **key kinds and rounds** — a finite predicate over a handful of
dataclasses — rather than over path strings, and two properties follow that the previous guards could
only approximate:

- **A stage cannot write outside its own directory**, because no method accepts a path. The guarantee
  is an *absence of capability*, not a check something could route around.
- **The DAG is acyclic by construction.** Every read is either a signal (upstream of everything), or
  an artifact from a strictly earlier round, or an earlier stage of the same round under
  :data:`STAGE_ORDER`. That is exhaustively checkable, unlike a graph built from pattern overlap —
  where ``pass_dir(run_dir, stream) / "asr"`` resolved to ``L1/*/asr``, whose ``*`` intersected the
  ``signals`` in ``L1/signals/**`` and silently permitted every ``adaptive/`` read of a
  per-perturbation directory.

The stage set is small and each member's directory is its identity:

==============  =========================================  ===========================
stage           writes                                     reads
==============  =========================================  ===========================
``L1``          ``L1/signals/``                            nothing inside the run
``DERIVE(n)``   ``L2/round/n/derivatives/``                signals, everything ``< n``
``ESTIMATE(n)`` ``L2/round/n/estimates/``                  the above ∪ derivatives ``n``
``REPORT(n)``   ``L2/round/n/report/``                     derivatives and estimates ``n``
``FINAL``       ``final/``                                 the last round
==============  =========================================  ===========================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final, Optional, Sequence, Union

from senselab.audio.workflows.audio_analysis.keys import (
    DerivativeKey,
    EstimateKey,
    Key,
    SignalKey,
)

__all__ = [
    "STAGE_ORDER",
    "Artifact",
    "ReportKey",
    "Stage",
    "StageIO",
    "UnauthorizedArtifact",
]


class UnauthorizedArtifact(Exception):
    """A stage named an artifact outside its declared inputs or outputs.

    Raised at the moment the key is presented, before any bytes exist — which is strictly better than
    the artifact guard it replaces, whose findings arrived only after a complete run.
    """


@dataclass(frozen=True, slots=True)
class ReportKey:
    """A round's human-facing output, or a deliverable in ``final/``.

    Not every artifact is a measurement. A timeline PNG and a round summary are *renderings*, so they
    have no target and no producer — but they still need a stage and a location, or they become the
    ``final/timeline.png`` that two producers overwrote in turn.

    Attributes:
        name: The filename, including its suffix.
        round: Which round rendered it, or ``None`` for a ``final/`` deliverable.
    """

    name: str
    round: Optional[int] = None

    def relative_path(self, suffix: str = "") -> str:
        """Where this rendering is stored, relative to the run directory.

        Reports live under ``report/`` rather than at the round's root so that a stage's write root is
        one whole directory. A root that meant "the files directly in ``L2/round/n/`` but not its
        subdirectories" would be the special case every other rule here avoids.
        """
        if self.round is None:
            return f"final/{self.name}{suffix}"
        return f"L2/round/{self.round}/report/{self.name}{suffix}"


Artifact = Union[Key, ReportKey]
"""Anything a stage can name: a keyed measurement, a derivative, an axis, or a rendering."""


class Stage(Enum):
    """A node in the pipeline DAG. Its directory is its identity.

    ``DERIVE`` and ``ESTIMATE`` are separate nodes because they write different directories and are
    ordered with respect to each other. As one ``L2_ROUND`` node the round read and wrote the same
    directory and was trivially its own predecessor, which is what made the ordering uncheckable at
    all.
    """

    L1 = "L1"
    DERIVE = "derive"
    ESTIMATE = "estimate"
    REPORT = "report"
    FINAL = "final"

    @property
    def is_round_scoped(self) -> bool:
        """Does this stage run once per L2 round?"""
        return self in (Stage.DERIVE, Stage.ESTIMATE, Stage.REPORT)


STAGE_ORDER: Final[tuple[Stage, ...]] = (
    Stage.L1,
    Stage.DERIVE,
    Stage.ESTIMATE,
    Stage.REPORT,
    Stage.FINAL,
)
"""Execution order within a round, and across the run. Declared **once**.

Two orderings of ``derive``/``estimate`` would let a reader and a writer disagree about the DAG, and
the disagreement would be invisible: both would run, and the second would read whatever the first
happened to have written.
"""


@dataclass(frozen=True, slots=True)
class StageIO:
    """One stage's capability: the artifacts it may read, and the one directory it may write.

    Obtained from :meth:`for_stage` and passed *in*. Nothing constructs a path from a string, and
    there is deliberately no method that accepts one — every previous guard was defeated by a path
    handed to a helper, and the fix is that the helper cannot be given one.

    Attributes:
        stage: Which node this is.
        run_dir: The run's root. The only absolute path in play, supplied once at the boundary rather
            than read from ambient state, which is what made ``run_dir`` reachable from everywhere.
        round: Which L2 round, for a round-scoped stage.
        last_round: Which round ``FINAL`` extracts from.
    """

    stage: Stage
    run_dir: Path
    round: Optional[int] = None
    last_round: Optional[int] = None

    @classmethod
    def for_stage(
        cls,
        stage: Stage,
        *,
        run_dir: Path,
        round: Optional[int] = None,  # noqa: A002 — the pipeline's own vocabulary
        last_round: Optional[int] = None,
    ) -> StageIO:
        """The capability for ``stage``.

        Raises:
            ValueError: When a round-scoped stage is given no round — defaulting to 0 would let a
                stage silently write round 0's directory from anywhere in the loop. Or when an
                unscoped stage is given one, which would imply a per-round L1 and is the re-entry
                confusion the perturbation register exists to resolve.
        """
        if stage.is_round_scoped and round is None:
            raise ValueError(f"{stage.value} is round-scoped and needs an explicit round")
        if not stage.is_round_scoped and round is not None:
            raise ValueError(f"{stage.value} has no round, but round={round!r} was given")
        return cls(stage=stage, run_dir=run_dir, round=round, last_round=last_round)

    # ── writing ────────────────────────────────────────────────────────

    def may_write(self, artifact: Artifact) -> bool:
        """Is this artifact one of this stage's declared outputs?"""
        if self.stage is Stage.L1:
            return isinstance(artifact, SignalKey)
        if self.stage is Stage.DERIVE:
            return isinstance(artifact, DerivativeKey) and artifact.round == self.round
        if self.stage is Stage.ESTIMATE:
            return isinstance(artifact, EstimateKey) and artifact.round == self.round
        if self.stage is Stage.REPORT:
            return isinstance(artifact, ReportKey) and artifact.round == self.round
        return isinstance(artifact, ReportKey) and artifact.round is None

    def path_for(self, artifact: Artifact, suffix: str = ".parquet") -> Path:
        """The absolute path to write ``artifact`` at.

        The only way a stage obtains a path. The suffix is the caller's because it follows from the
        *shape* — a ``Tree`` is JSON where a ``Series`` is parquet — and the shape is not part of the
        key.

        Raises:
            UnauthorizedArtifact: When this stage may not write it, naming the stage and what it was
                handed rather than only that something was refused.
        """
        if not self.may_write(artifact):
            raise UnauthorizedArtifact(
                f"stage {self.stage.value}"
                + (f" at round {self.round}" if self.round is not None else "")
                + f" may not write {_describe(artifact)}"
            )
        return self.run_dir / artifact.relative_path(suffix)

    def required_columns(self, artifact: Artifact) -> tuple[str, ...]:
        """Key dimensions the path does not supply, which must therefore appear as columns (D-17).

        A fold's source list is unbounded and cannot go in a path, so the path names the operator only
        and the members are materialised in the rows. Without them a stability computed over one route
        is indistinguishable from one computed over five — absent-vs-empty, at the schema level.
        """
        return artifact.required_columns if isinstance(artifact, DerivativeKey) else ()

    # ── reading ────────────────────────────────────────────────────────

    def may_read(self, artifact: Artifact) -> bool:
        """Is this artifact one of this stage's declared inputs?

        The read set is **generated from the round**, not enumerated: the pool is monotone, so a
        stage sees L1 plus every strictly earlier round rather than only ``n-1`` (D-22). Acyclicity
        comes from the index strictly decreasing, which is a sound argument and a different one from
        the adjacency restriction it replaces.
        """
        if isinstance(artifact, ReportKey):
            return False  # renderings are terminal; a report something reads is an intermediate
        if self.stage is Stage.L1:
            return False  # it measures the audio; a signal derived from a signal is a derivative
        if self.stage is Stage.FINAL:
            return self._round_of(artifact) == self.last_round
        earlier = self._round_of(artifact)
        if isinstance(artifact, SignalKey):
            return True  # every L2 stage may read L1 directly, at full resolution
        assert self.round is not None  # guaranteed by for_stage
        if earlier is None:
            return False
        if earlier < self.round:
            return True
        # Same round: only an earlier stage's output, which is the single intra-round edge.
        return (earlier == self.round and self.stage is Stage.ESTIMATE and isinstance(artifact, DerivativeKey)) or (
            earlier == self.round and self.stage is Stage.REPORT
        )

    def locate(self, artifact: Artifact, suffixes: Sequence[str]) -> Optional[Path]:
        """The first of ``suffixes`` that exists for ``artifact``, or ``None``.

        Here rather than in the caller because the caller must never hold a path. A helper doing its
        own ``path.exists()`` has a path outside the capability, which is the shape of every defeat
        the previous guards suffered — and the static guard correctly flagged exactly that when this
        lived in ``measurements.read_measurement``.

        Returning ``None`` for "nothing stored" keeps that distinguishable from an empty artifact,
        which the caller reports as the different thing it is.

        Raises:
            UnauthorizedArtifact: When this stage may not read it.
        """
        if not self.may_read(artifact):
            raise UnauthorizedArtifact(
                f"stage {self.stage.value}"
                + (f" at round {self.round}" if self.round is not None else "")
                + f" may not read {_describe(artifact)}"
            )
        for suffix in suffixes:
            candidate = self.run_dir / artifact.relative_path(suffix)
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _round_of(artifact: Artifact) -> Optional[int]:
        """The round an artifact belongs to, or ``None`` for a signal."""
        if isinstance(artifact, (DerivativeKey, EstimateKey)):
            return artifact.round
        return None


def _describe(artifact: Artifact) -> str:
    """A message naming what kind of thing was refused, so the error is actionable."""
    if isinstance(artifact, SignalKey):
        return f"signal {artifact.target}/{artifact.producer}"
    if isinstance(artifact, DerivativeKey):
        return f"derivative {artifact.target}/{artifact.operator.segment} at round {artifact.round}"
    if isinstance(artifact, EstimateKey):
        return f"estimate {artifact.axis} at round {artifact.round}"
    return f"report {artifact.name} at round {artifact.round}"
