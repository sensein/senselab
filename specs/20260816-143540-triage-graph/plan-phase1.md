# Triage graph, Phase 1: the `Estimate` type and the config-discipline guard

> **For agentic workers:** implement task-by-task. Steps use `- [ ]` for tracking.

**Goal:** ship the two foundations the rest of the triage graph depends on, neither of which
requires a package-boundary decision.

**Architecture:** one new leaf data structure under `utils/data_structures/`, and one deletion +
guard test under `audio_analysis/`. The two tasks touch disjoint files.

**Tech stack:** pydantic v2, pytest. `uv run` for everything.

## Global Constraints

- **Every Python command goes through `uv run`.** Never bare `python`/`pip`.
- **Never construct an unmocked `HFModel`** in a test — it downloads a full snapshot.
- **Never run `pytest -n auto`.** Run the directory you changed.
- **Never `git add -A` or `git add .`** — add named paths only.
- Google-style docstrings, line length 120, type hints required (mypy + pydantic plugin).
- Tests live in `src/tests/` mirroring the package, named `*_test.py`.
- **Explain *why*, not *what*.** A docstring that restates a readable signature earns nothing.
  Record the measurement or failure behind a non-obvious choice.
- **Pre-alpha: rename and replace outright.** No parallel fields, no aliases, no deprecation shims.
- Run `uv run pre-commit run --all-files` before declaring done — `ruff` + `mypy` alone is *not*
  the CI gate; it also runs codespell and JSON formatting.

---

### Task 1: the `Estimate` type

**Files:**
- Create: `src/senselab/utils/data_structures/estimate.py`
- Modify: `src/senselab/utils/data_structures/__init__.py`
- Test: `src/tests/utils/data_structures/estimate_test.py`

**Interfaces produced:** `Estimate` (pydantic `BaseModel`), importable as
`from senselab.utils.data_structures import Estimate`.

**Why this exists.** A statistical review of `analyze_audio`
(`specs/20260815-215106-analyze-audio-audit/statistical-review.md`, finding N10) measured three
defects that share one cause — a number published without the count of things that produced it:

- adding a *low-reliability* signal moved published confidence from 0.800 to 0.420;
- a bucket backed by 4 unanimous sources and one backed by 20 both published `P = 1.000`;
- a crashed diarizer produced a confidence indistinguishable from an agreeing one.

This type makes all three unrepresentable. It has no consumers yet; wiring is Phases 2 and 3.

**The model.** `value` is a **computed property, never a field** — so a caller cannot publish a
value inconsistent with the evidence behind it.

```python
class Estimate(BaseModel):
    raw: Optional[float]      # the unshrunk sample statistic; None iff n_evidence == 0
    n_evidence: int           # independent contributing sources; >= 0, 0 is legal and meaningful
    prior: float              # what `value` collapses to as n_evidence -> 0
    prior_key: str            # config key naming the prior, so its derivation is findable
    prior_weight: float       # pseudo-count k; > 0
    population: str           # the population this was validated on, e.g. "adult-read-speech"
```

with

```
value = prior                                             if n_evidence == 0
value = (n_evidence*raw + prior_weight*prior) / (n_evidence + prior_weight)   otherwise
```

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the Estimate type."""

import pytest
from pydantic import ValidationError

from senselab.utils.data_structures import Estimate


def _est(**kw: object) -> Estimate:
    base = dict(raw=1.0, n_evidence=1, prior=0.5, prior_key="k", prior_weight=1.0, population="p")
    base.update(kw)
    return Estimate(**base)  # type: ignore[arg-type]


def test_no_evidence_collapses_to_the_prior() -> None:
    e = Estimate(raw=None, n_evidence=0, prior=0.42, prior_key="k", prior_weight=2.0, population="p")
    assert e.value == 0.42


def test_a_raw_value_without_evidence_is_rejected() -> None:
    """F-156: a fabricated number and a measured one must not be the same object."""
    with pytest.raises(ValidationError):
        _est(raw=0.5, n_evidence=0)


def test_evidence_without_a_raw_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _est(raw=None, n_evidence=3)


def test_negative_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _est(n_evidence=-1)


def test_a_non_positive_prior_weight_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _est(prior_weight=0.0)


def test_an_empty_population_is_rejected() -> None:
    """An unstated population is how an adult-derived threshold reaches a child recording."""
    with pytest.raises(ValidationError):
        _est(population="  ")


def test_four_and_twenty_unanimous_sources_differ() -> None:
    """Statistical review N3: both published P = 1.000 before this type existed."""
    four = _est(raw=1.0, n_evidence=4, prior=0.5, prior_weight=1.0)
    twenty = _est(raw=1.0, n_evidence=20, prior=0.5, prior_weight=1.0)
    assert four.value != twenty.value
    assert four.value < twenty.value < 1.0


def test_more_evidence_moves_the_value_toward_raw() -> None:
    values = [_est(raw=1.0, n_evidence=n, prior=0.0, prior_weight=1.0).value for n in (1, 2, 8, 64)]
    assert values == sorted(values)
    assert values[-1] < 1.0


def test_shrinkage_reports_how_much_of_the_value_is_prior() -> None:
    assert _est(n_evidence=1, prior_weight=1.0).shrinkage == pytest.approx(0.5)
    assert _est(n_evidence=9, prior_weight=1.0).shrinkage == pytest.approx(0.1)
    assert Estimate(
        raw=None, n_evidence=0, prior=0.1, prior_key="k", prior_weight=3.0, population="p"
    ).shrinkage == 1.0


def test_value_is_not_settable() -> None:
    """A published value that disagrees with its own evidence is the defect this type prevents."""
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        Estimate(  # type: ignore[call-arg]
            raw=1.0, n_evidence=1, prior=0.0, prior_key="k",
            prior_weight=1.0, population="p", value=0.999,
        )
```

- [ ] **Step 2: Run the tests, confirm they fail**

`uv run pytest src/tests/utils/data_structures/estimate_test.py -v`
Expected: collection error / ImportError — `Estimate` does not exist.

- [ ] **Step 3: Implement `estimate.py`**

Write the module. Requirements, all load-bearing:

- `model_config = ConfigDict(frozen=True, extra="forbid")` — `extra="forbid"` is what makes
  `test_value_is_not_settable` pass, and `frozen=True` keeps a published estimate immutable.
- `n_evidence: int = Field(ge=0)`, `prior_weight: float = Field(gt=0)`.
- A `model_validator(mode="after")` enforcing `raw is None` **iff** `n_evidence == 0`, with an
  error message naming both fields.
- A `field_validator` on `population` rejecting blank/whitespace-only strings.
- `value` and `shrinkage` as `@property` — **not** pydantic computed fields, so they cannot be
  supplied by a caller.
- `shrinkage` = `prior_weight / (n_evidence + prior_weight)`.
- Module docstring records *why* (the three measured defects above, cited to the review), not
  *what* the fields are.

Also add a convenience constructor, since "no evidence" is the case callers will most often build
and the one most easily got wrong:

```python
@classmethod
def no_evidence(cls, *, prior: float, prior_key: str, prior_weight: float, population: str) -> "Estimate":
```

- [ ] **Step 4: Export it**

Add to `src/senselab/utils/data_structures/__init__.py`, matching the existing
`from .module import Name  # noqa: F401` style, in alphabetical position.

- [ ] **Step 5: Tests pass, gate is green**

```bash
uv run pytest src/tests/utils/data_structures/estimate_test.py -v
uv run pre-commit run --all-files
```

- [ ] **Step 6: Commit** (named paths only)

---

### Task 2: delete the dead `RunConfig` policy fields, and guard against new ones

**Files:**
- Modify: `src/senselab/audio/workflows/audio_analysis/run_config.py`
- Test: `src/tests/audio/workflows/audio_analysis/run_config_liveness_test.py` (create)

**Why.** `remediation-config.md` found 14 dead config keys, 9 of which share one root cause:
`RunConfig` fields that `_build()` assigns and nothing ever reads. A key that advertises control it
does not have is worse than a bare literal — an operator sets `quality.floor_percentile: 25.0`,
the run reports the config hash as changed, and the value that actually governs is
`acoustic.py:116`'s `FLOOR_PERCENTILE = 10.0`.

**Candidates** (from `remediation-config.md` D5–D14): `rounds_policy`, `quality_policy`,
`labelstudio_policy`, `support_policy` — assigned at `run_config.py:480-484`.

**A discrepancy you must resolve rather than inherit.** That document's summary counts **four**
dead fields plus "two orphaned keys inside `speaker_policy`", but its own D3 and D4 rows state that
`RunConfig.speaker_policy` "is built and never read anywhere else in the tree". Those cannot both
be right. **Verify each candidate yourself before touching it**, and report what you actually
found — including whether `speaker_policy` is a fifth dead field or has a live reader the inventory
missed. Do not delete a field you have not personally confirmed is unread.

- [ ] **Step 1: Verify liveness for each candidate**

For each of `rounds_policy`, `quality_policy`, `labelstudio_policy`, `support_policy`,
`speaker_policy`, search the whole tree — `src/`, `scripts/`, `src/tests/` — for reads outside
`run_config.py`'s own assignment. Check **both** attribute access (`.rounds_policy`) and dynamic
access (`getattr(`, `asdict(`, `.raw`, `dataclasses.fields`, `**`-unpacking, serialization). A
field read only by a test still counts as read — say so and leave it.

Record the evidence per field; it goes in the report and the commit message.

- [ ] **Step 2: Write the guard test**

```python
"""Every RunConfig field must have a reader.

remediation-config.md found four policy fields that `_build()` assigned and nothing consumed.
An operator setting such a key gets a changed config hash and no behaviour change, which is
worse than a bare literal because it looks like control. This test fails when a new one appears.
"""
```

The test enumerates `dataclasses.fields(RunConfig)` and asserts each name is read somewhere under
`src/senselab/` or `scripts/` outside `run_config.py`. Implement the search by AST or by scanning
source text for the field name — your choice, but it must not match the assignment inside
`_build()` itself, and it must not match a name appearing only in a comment or docstring.

`KNOWN_UNREAD` starts **empty**. If a field survives Step 1 as genuinely unread but you judge it
should not be deleted, put it in `KNOWN_UNREAD` with a comment naming the reason and the register
id — do not weaken the assertion.

- [ ] **Step 3: Run it, confirm it fails** naming exactly the fields Step 1 found unread.

- [ ] **Step 4: Delete the confirmed-dead fields**

Remove each from the `RunConfig` dataclass declaration **and** from `_build()`'s constructor call.
Pre-alpha rule: delete outright, no alias, no shim.

Leave the corresponding YAML sections in `default.yaml` **in place** — `rounds:` holds the live
`max_rounds`, and the other sections' keys are the subject of separate register findings whose fix
is to thread them to a real call site, not to delete them. Deleting the YAML would destroy the
record of what was supposed to be configurable. Say this in the commit message.

- [ ] **Step 5: Full local gate**

```bash
uv run pytest src/tests/audio/workflows/audio_analysis/ -x -q
uv run pre-commit run --all-files
```

The workflow suite is the blast radius for a `RunConfig` change; run all of it, not just the new
test.

- [ ] **Step 6: Commit** (named paths only)

---

## Out of scope for Phase 1

- The layer-1 extraction lift. Deferred to Phase 2 — measured 2026-08-16, its stated justification
  did not survive: the package's lazy `__getattr__` already prevents the import cost cited, and the
  real closure is 14 modules / 5,523 lines reaching `axes.py`, not 2 files.
- Wiring `Estimate` into any existing output. Phases 2 and 3.
- Threading the *live-but-dead* keys (D1–D13) to real call sites. Each is its own register finding.
