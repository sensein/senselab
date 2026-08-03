"""What a measurement is, where it came from, and where it lands (D-20 – D-23).

Signals and derivatives share **one key space**, because a derivative is not a layer an axis must be
funnelled through — it is another signal (D-22). A signal key is the degenerate derivative key:

.. code-block:: text

    Key = (Target, Producer, Source)
      signal:      Producer = a model,     Source = a route (the recording under a transform)
      derivative:  Producer = an operator,  Source = the key(s) it consumed

``Source`` is therefore a **provenance tree** whose leaves are routes, and the recording is its root.
That is what makes :meth:`DerivativeKey.shares_evidence_with` a set intersection rather than string
matching.

**The target comes first, and it is what the tool measured — not how.** Every voter on one target
shares a first element, so cross-tool disagreement is computable without a ``"::"``-joined selector,
and a bundle cannot form: ``snr``, ``c50``, ``rolloff`` and ``clipping`` are four targets and cannot
share a file. Mechanism (``frame``, ``window``, ``span``, ``tree``) describes the *shape* of the
output and lives in :mod:`.shapes` beside units, not in the key.

**An axis is not a target.** One SNR measurement serves the speech-presence axis as a quality gate,
the background-mask axis as a scene descriptor, and the reliability weighting. The axis↔target
mapping is many-to-many L2 policy; what the key guarantees is only that tools sharing a target are
gatherable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final, Union

__all__ = [
    "Arity",
    "DEFAULT_PATHWAY",
    "DEFAULT_PERTURBATION",
    "DerivativeKey",
    "EstimateKey",
    "Key",
    "Operator",
    "Route",
    "SignalKey",
    "Source",
    "slug",
]

DEFAULT_PATHWAY: Final[str] = "direct"
"""The pathway on which the recording's own foreground is the primary information target."""

DEFAULT_PERTURBATION: Final[str] = "unmodified"
"""The transform that does nothing.

**Not** ``identity``. That is the mathematically obvious name and it collides: ``identity`` was the
per-speaker axis's name before the rename to ``speaker``, and it survives in CLAUDE.md's
``uncertainty/{presence,identity,utterance}.parquet`` and in the spec directory name — so it is still
what a reader of this repo expects the word to mean. A path ending ``direct/identity.parquet`` under
``L1/signals/`` then reads as an identity *estimate*, which is how the first reader read it, and
nothing under ``L1/signals/`` is an estimate at all.

``unmodified`` is what ``perturbations.IDENTITY_TRANSFORM`` already used, so this is the name the
codebase had before a colliding one was invented for it.
"""


def slug(identifier: str) -> str:
    """A path-safe segment for a model or operator id, injectively.

    ``/`` cannot appear in a path segment, so it becomes ``__``. The mapping has to be **injective**:
    if a producer whose id already contains ``__`` could collide with one whose ``/`` slugged to it,
    two models' measurements land in one file and read as one model measured twice.

    Injectivity is enforced by *refusal* rather than by an escape encoding, which is this codebase's
    habit — reject at construction instead of at read-back — and keeps the segment readable
    (``MIT__ast-finetuned-audioset``) where an escape would not.

    Raises:
        ValueError: For an id already containing ``__``, which no real model or operator id does.
    """
    if "__" in identifier:
        raise ValueError(f"id {identifier!r} contains the path separator '__'; it cannot be slugged injectively")
    return identifier.replace("/", "__")


@dataclass(frozen=True, slots=True)
class Route:
    """Where the audio came from: which component is primary, and what was done to it (D-23).

    Two independent dimensions, separated by the criterion that a perturbation *in ideal
    circumstances does not remove the primary information targets*:

    Attributes:
        pathway: Which signal component is **primary**. Changing it changes what "primary
            information target" means, which is why foreground suppression is a pathway switch
            rather than a perturbation — it removes the direct pathway's primary target by design.
        perturbation: A transform preserving the primary targets **of that pathway**. Speech
            enhancement qualifies: ideally it removes only non-target content. The background
            pathway has its own enhancement, for the same reason.
    """

    pathway: str = DEFAULT_PATHWAY
    perturbation: str = DEFAULT_PERTURBATION

    def __post_init__(self) -> None:
        """Reject a name that cannot be slugged injectively, at construction."""
        slug(self.pathway)
        slug(self.perturbation)

    @property
    def path(self) -> str:
        """The route's two path levels, always spelling both dimensions.

        **Two levels, not one joined segment.** Joining them with ``__`` — the same string ``slug``
        uses for ``/`` — would make ``a_b`` + ``c`` and ``a`` + ``b_c`` collide, and the filesystem
        already separates dimensions for free. It also makes each dimension globbable on its own:
        ``*/identity`` is every pathway unperturbed, ``direct/*`` is one pathway's perturbations.

        The identity route gets no shorthand. ``raw`` as a distinct name is how ``raw_vs_enhanced``
        came to masquerade as a third pass — a comparison indistinguishable from a member.
        """
        return f"{slug(self.pathway)}/{slug(self.perturbation)}"

    @property
    def is_unmodified(self) -> bool:
        """Is this the untransformed recording on its own pathway?"""
        return self.pathway == DEFAULT_PATHWAY and self.perturbation == DEFAULT_PERTURBATION

    def same_pathway_as(self, other: Route) -> bool:
        """Do these two routes hold the same component primary?

        The distinction D-23 turns on: only a within-pathway pair is a stability sample. Across
        pathways the difference is what the pathway was applied to produce, and reading it as
        instability down-weights the signal that noticed something.
        """
        return self.pathway == other.pathway


@dataclass(frozen=True, slots=True)
class Operator:
    """A named derivation and the choice inside it — a derivative's "producer".

    A signal's uncertainty comes from which model you asked; a derivative's from which choice you
    made. So the variant is not decoration: it **is** the policy, and a derivative whose choice is
    not in its key is the ``settled_below=0.35`` default argument all over again.

    Attributes:
        name: The derivation — ``project_labels``, ``resample``, ``cover``, ``censored_posterior``.
        variant: The policy — a label-set version, a threshold, an estimator. ``None`` only when the
            derivation genuinely has no choice in it.
    """

    name: str
    variant: str | None = None

    def __post_init__(self) -> None:
        """Reject a name or variant that cannot be slugged injectively, at construction."""
        slug(self.name)
        if self.variant is not None:
            slug(self.variant)

    @property
    def segment(self) -> str:
        """The path segment: ``name`` or ``name__variant``."""
        return slug(self.name) if self.variant is None else f"{slug(self.name)}__{slug(self.variant)}"


@dataclass(frozen=True, slots=True)
class SignalKey:
    """One tool's measurement of one target, on one route — the L1 emission.

    Attributes:
        target: What was measured, in the tool's terms — ``speech``, ``snr``, ``speaker_spans``,
            ``transcript``, ``scene_labels``. Not a domain and not a resolution.
        producer: The model id that measured it.
        route: Which pathway and perturbation the audio had.
    """

    target: str
    producer: str
    route: Route = field(default_factory=Route)

    def __post_init__(self) -> None:
        """Reject a producer id that cannot be slugged injectively, at construction.

        Here rather than at :meth:`relative_path`, so the ambiguity is refused where the key is made
        rather than at the moment a file is about to be written under a colliding name.
        """
        slug(self.producer)

    def relative_path(self, suffix: str) -> str:
        """Where this signal is stored, relative to the run directory.

        The suffix comes from the shape, not the key: a ``Tree`` is JSON and a ``Series`` is parquet.
        """
        return f"L1/signals/{self.target}/{slug(self.producer)}/{self.route.path}{suffix}"


@dataclass(frozen=True, slots=True)
class EstimateKey:
    """One axis at one round — the L2 product.

    Deliberately has no producer and no route: an axis aggregates across both, so neither can index
    its output (D-16). A pass is an input dimension to the fold, never an index on it.
    """

    axis: str
    round: int

    def relative_path(self, suffix: str = ".parquet") -> str:
        """Where this axis is stored, relative to the run directory."""
        return f"L2/round/{self.round}/estimates/{self.axis}{suffix}"


class Arity(Enum):
    """How many sources a derivative has, and what that licenses (D-21).

    The distinction exists because these license different things, and conflating them is the
    ``units: "mixed"`` defect one level up:

    - ``PROJECT`` — one source. Tool and route survive into the key, so the result is still *that
      tool's* measurement and remains comparable against another tool's.
    - ``FOLD`` — several sources **sharing a target**. They answer the same question, so a spread
      across them is meaningful. A fold has to justify why it is not an axis, and the answer is
      always that it produces a *value* where an axis produces *doubt about a family of values*.
    - ``COMPOSE`` — sources with **different targets**. A joint function, so a spread across them
      measures nothing.
    """

    PROJECT = "project"
    FOLD = "fold"
    COMPOSE = "compose"


@dataclass(frozen=True, slots=True)
class DerivativeKey:
    """A value derived from other keys, at one round.

    Attributes:
        target: What this is a value *of*. Shares the signal vocabulary, extended where a derivation
            changes the quantity — ``occupancy``, ``speaker_count``, ``target_free``, ``stability``.
        operator: The derivation and its policy.
        sources: The keys consumed. One for a projection, several for a fold or compose. May include
            an :class:`EstimateKey` from a **strictly earlier** round — the coupling channel, in the
            key so the feedback edge is visible in the artifact tree rather than inferable from a
            function name.
        round: Which round produced it.
    """

    target: str
    operator: Operator
    sources: tuple[Source, ...]
    round: int = 0

    def __post_init__(self) -> None:
        """Refuse a derivative of nothing, and an estimate from this round or later.

        Raises:
            ValueError: With no sources — a derivative of nothing is a measurement pretending to be
                a derivation. Or when an :class:`EstimateKey` source is not strictly earlier: that
                strictness is what keeps the round DAG acyclic, and as one node a round reading its
                own estimates is indistinguishable from a legitimate back-edge.
        """
        if not self.sources:
            raise ValueError(f"derivative {self.target}/{self.operator.segment} has no source")
        for source in self.sources:
            if isinstance(source, EstimateKey) and source.round >= self.round:
                raise ValueError(
                    f"derivative at round {self.round} may not read estimate {source.axis!r} from "
                    f"round {source.round}; only a strictly earlier round is acyclic"
                )

    @property
    def arity(self) -> Arity:
        """Derived from the sources, never declared — see :class:`Arity`."""
        if len(self.sources) == 1:
            return Arity.PROJECT
        targets = {self._target_of(source) for source in self.sources}
        return Arity.FOLD if len(targets) == 1 else Arity.COMPOSE

    @property
    def spread_is_meaningful(self) -> bool:
        """May this derivative report a spread across its inputs — D-21 rule 3?

        Only a fold's inputs answer the same question. A compose's spread is a number computed over
        different quantities, which is what ``units: "mixed"`` was.
        """
        return self.arity is Arity.FOLD

    @property
    def folds_within_one_pathway(self) -> bool:
        """Is this fold a stability sample, per D-23?

        ``True`` when every source is on one pathway, so |Δ| is the same question answered twice and
        may set a fusion weight. ``False`` across pathways: the sources hold different components
        primary, and their difference is complementary rather than corroborative.
        """
        pathways = {source.route.pathway for source in self.sources if isinstance(source, SignalKey)}
        pathways |= {
            pathway for source in self.sources if isinstance(source, DerivativeKey) for pathway in source._pathways()
        }
        return len(pathways) <= 1

    def source_closure(self) -> frozenset[SignalKey]:
        """Every signal this transitively rests on.

        Closed on :class:`SignalKey` rather than on routes, which is what "above the recording" means
        in D-21 rule 6: two keys sharing only the recording is every pair in a run, so a route in the
        closure would make the test vacuous.

        An :class:`EstimateKey` contributes nothing. An axis is a fold over everything, so counting
        it would make the closure universal — that edge is governed by the round index instead, and
        the limit is recorded rather than papered over.
        """
        signals: set[SignalKey] = set()
        for source in self.sources:
            if isinstance(source, SignalKey):
                signals.add(source)
            elif isinstance(source, DerivativeKey):
                signals |= source.source_closure()
        return frozenset(signals)

    def shares_evidence_with(self, other: Key) -> bool:
        """Are these two the same evidence twice, per D-21 rule 6?

        The hazard the merged input pool creates: a projection and the signal it came from are both
        in the pool and both eligible to vote, and an axis fusing them counts one tool twice while
        reporting two contributors. A consensus transcript has every transcript in its closure, so it
        is not an independent voter against them.
        """
        return bool(self.source_closure() & _closure_of(other))

    def shares_producer_with(self, other: Key) -> bool:
        """Do these rest on a common *model*, whatever it measured?

        A different overlap from :meth:`shares_evidence_with`, and neither subsumes the other.
        Brouhaha's SNR and C50 come from one forward pass: correlated through a shared trunk, but not
        the same measurement — so a fold should know, while discounting one as a copy of the other
        would be wrong. Collapsing the two notions either double-discounts or misses the correlation.

        The aligner case is the third kind and is caught by neither: two transcripts timed by one
        forced aligner have correlated word boundaries, and the aligner is ``timestamp_source``
        provenance rather than a source (D-20).
        """
        mine = {signal.producer for signal in self.source_closure()}
        theirs = {signal.producer for signal in _closure_of(other)}
        return bool(mine & theirs)

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Key dimensions the path cannot supply, which therefore must appear as columns (D-17).

        A fold or compose has an unbounded source list, so the path names the operator only and the
        members are materialised in the rows. Without them a stability computed over one route is
        indistinguishable from one computed over five.
        """
        if self.arity is Arity.PROJECT:
            return ()
        return ("contributing_producers", "contributing_routes")

    def relative_path(self, suffix: str) -> str:
        """Where this derivative is stored, relative to the run directory.

        A projection keeps its source's producer and route in the path, because the result is still
        that tool's measurement on that route. A fold or compose collapses to the operator, with
        :attr:`required_columns` carrying what the path dropped.
        """
        stem = f"L2/round/{self.round}/derivatives/{self.target}/{self.operator.segment}"
        if self.arity is Arity.PROJECT:
            source = self.sources[0]
            if isinstance(source, SignalKey):
                return f"{stem}/{slug(source.producer)}/{source.route.path}{suffix}"
        return f"{stem}{suffix}"

    def _pathways(self) -> frozenset[str]:
        """Pathways reachable through this derivative's closure."""
        return frozenset(signal.route.pathway for signal in self.source_closure())

    @staticmethod
    def _target_of(source: Source) -> str:
        """The target of any source kind, for the arity computation."""
        return source.axis if isinstance(source, EstimateKey) else source.target


Source = Union[SignalKey, "DerivativeKey", EstimateKey]
"""What a derivative may consume: a signal, another derivative, or an earlier round's estimate."""

Key = Union[SignalKey, DerivativeKey, EstimateKey]
"""Anything the input pool holds. One key space, several locations — the path says who made it."""


def _closure_of(key: Key) -> frozenset[SignalKey]:
    """Signal closure of any key kind, so overlap tests take signals and derivatives alike."""
    if isinstance(key, SignalKey):
        return frozenset({key})
    if isinstance(key, DerivativeKey):
        return key.source_closure()
    return frozenset()
