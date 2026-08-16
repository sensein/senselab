"""A shrinkage-aware statistical estimate.

A statistical review of `analyze_audio`
(`specs/20260815-215106-analyze-audio-audit/statistical-review.md`, finding N10) measured three
defects that share one cause: a number published without the count of things that produced it.

- Adding a single low-reliability signal moved published confidence from 0.800 to 0.420 — nothing
  in the published number said how many sources, or how reliable, produced either value.
- A bucket backed by 4 unanimous sources and one backed by 20 both published `P = 1.000` — the
  raw statistic saturates at unanimity regardless of sample size, so the published number could not
  distinguish "4 agreeing sources" from "20 agreeing sources" (statistical review N3).
- A crashed diarizer produced a confidence indistinguishable from one that ran and agreed — with no
  `n_evidence` field, "zero contributing sources" and "several agreeing sources" render the same.

`Estimate` makes all three unrepresentable: `value` is derived from `raw`, `n_evidence`, and a
named `prior`, never supplied directly, so a caller cannot publish a number the evidence does not
support. It has no consumers yet; wiring into `analyze_audio` outputs is Phases 2 and 3 of the
triage graph.
"""

import math
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self


class Estimate(BaseModel):
    """A statistic shrunk toward a named prior by the amount of evidence behind it.

    `value` collapses to `prior` when `n_evidence == 0` and moves toward `raw` as `n_evidence`
    grows, using `prior_weight` as the pseudo-count of the prior. `value` and `shrinkage` are
    computed on read (plain `@property`, not pydantic `computed_field`) so that a caller cannot
    construct an `Estimate` whose published value disagrees with the evidence behind it — see
    `test_value_is_not_settable`, which pins down that `extra="forbid"` rejects a `value=` kwarg
    even though `value` is not a declared field.

    `model_copy(update=...)` is overridden to re-validate (see below) because pydantic's own
    implementation writes straight into `__dict__` and skips every validator, which is exactly the
    gap that let `good.model_copy(update={"n_evidence": 0})` produce an `Estimate` with `raw` still
    set and `n_evidence == 0` — the raw/n_evidence invariant silently broken with no error.
    `model_construct` is *not* closed the same way: it is pydantic's documented, explicit
    validation-skipping constructor, not an API a caller reaches for by accident, so closing it
    would fight the library rather than the defect. A reader who calls `Estimate.model_construct`
    directly is opting out of the invariant on purpose; `model_copy(update=...)` is the one call
    that looks safe and isn't.

    Attributes:
        raw: The unshrunk sample statistic. `None` iff `n_evidence == 0` — there is no sample
            statistic when nothing was observed.
        n_evidence: Count of independent contributing sources. `>= 0`; `0` is legal and means
            "nothing observed," not "unknown."
        prior: What `value` collapses to as `n_evidence -> 0`.
        prior_key: Config key naming the prior, so its derivation is findable rather than a bare
            literal (see the "thresholds belong in `data/`" rule this codebase enforces elsewhere).
        prior_weight: Pseudo-count `k` given to the prior in the shrinkage blend. Must be `> 0`;
            a zero or negative weight would make the prior either inert or sign-flipping.
        population: The population this was validated on, e.g. `"adult-read-speech"`. Required and
            non-blank because an unstated population is how a threshold fitted on one population
            (e.g. adults) silently reaches a recording from a different one (e.g. children).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: Optional[float]
    n_evidence: int = Field(ge=0)
    prior: float
    prior_key: str
    prior_weight: float = Field(gt=0)
    population: str

    @field_validator("population")
    @classmethod
    def _population_not_blank(cls, v: str) -> str:
        """Reject a blank or whitespace-only population.

        An unstated population is how an adult-derived threshold reaches a child recording without
        anyone noticing which population actually validated the number.
        """
        if not v.strip():
            raise ValueError("population must not be blank")
        return v

    @field_validator("raw", "prior")
    @classmethod
    def _finite(cls, v: Optional[float], info: ValidationInfo) -> Optional[float]:
        """Reject NaN/Inf.

        Reviewer minor: a non-finite `raw` or `prior` propagates silently into `value` (e.g.
        `nan` poisons the blend, `inf` swamps it) with no validation error to say where it came
        from — the same "published number with no traceable evidence" defect this type exists to
        rule out, just arriving through a different door than `n_evidence`.
        """
        if v is not None and not math.isfinite(v):
            raise ValueError(f"{info.field_name} must be finite, got {v!r}")
        return v

    @model_validator(mode="after")
    def _raw_matches_evidence(self) -> "Estimate":
        """Enforce `raw is None` iff `n_evidence == 0`.

        A fabricated `raw` with no evidence, or evidence with no sample statistic to shrink, are
        both the "number published without the count that produced it" defect this type exists to
        make unrepresentable (statistical review N10, F-156).
        """
        if self.n_evidence == 0 and self.raw is not None:
            raise ValueError("raw must be None when n_evidence == 0 (no evidence, nothing to report as raw)")
        if self.n_evidence > 0 and self.raw is None:
            raise ValueError("raw must be set when n_evidence > 0 (evidence exists but no raw statistic was given)")
        return self

    @property
    def value(self) -> float:
        """The shrinkage-blended estimate.

        `prior` when `n_evidence == 0`; otherwise the weighted average of `raw` and `prior` with
        `n_evidence` and `prior_weight` as weights, so more evidence moves `value` toward `raw` and
        unanimity with few sources reads as less certain than unanimity with many.
        """
        if self.n_evidence == 0:
            return self.prior
        if self.raw is None:  # pragma: no cover — guaranteed by _raw_matches_evidence
            # A plain `assert` here would vanish under `python -O`, silently returning `None` where
            # a `float` is promised. Raise instead so the invariant holds even with optimizations on.
            raise RuntimeError("raw is None with n_evidence > 0; _raw_matches_evidence should have rejected this")
        return (self.n_evidence * self.raw + self.prior_weight * self.prior) / (self.n_evidence + self.prior_weight)

    @property
    def shrinkage(self) -> float:
        """Fraction of `value` attributable to the prior rather than the evidence.

        `1.0` at `n_evidence == 0` (all prior) and falls toward `0.0` as evidence accumulates.
        """
        return self.prior_weight / (self.n_evidence + self.prior_weight)

    @classmethod
    def no_evidence(cls, *, prior: float, prior_key: str, prior_weight: float, population: str) -> "Estimate":
        """Build an `Estimate` for the no-evidence case.

        This is the case callers build most often (nothing observed yet) and the one most easily
        got wrong by hand (forgetting to pair `raw=None` with `n_evidence=0`), so it gets a named
        constructor rather than leaving every call site to spell out both fields.
        """
        return cls(
            raw=None,
            n_evidence=0,
            prior=prior,
            prior_key=prior_key,
            prior_weight=prior_weight,
            population=population,
        )

    def model_copy(self, *, update: Optional[Mapping[str, Any]] = None, deep: bool = False) -> Self:
        """Copy, re-validating when `update` is given.

        `BaseModel.model_copy(update=...)` writes straight into `__dict__` and never re-runs a
        validator — the pydantic docs say so explicitly ("the data is not validated before
        creating the new model"). For most models that is a reasonable default; for this one it
        defeats the entire point, since `good.model_copy(update={"n_evidence": 0})` would return an
        `Estimate` with `raw` still set and `n_evidence == 0`, silently breaking the raw/n_evidence
        invariant with no error at the one call site most likely to be reached for by a caller who
        wants "the same estimate, but with this field changed." Routing `update` back through the
        constructor costs one extra validation pass and closes that gap. The no-update path is
        unaffected and delegates to `super()`.
        """
        if update is None:
            return super().model_copy(deep=deep)
        return type(self)(**{**self.model_dump(), **update})
