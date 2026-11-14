"""The order of license policies: `laxer_than` agrees with what each policy accepts."""

import itertools
from collections.abc import Iterator

import pytest
from senselab_core.runtime.licenses import (
    LaxerLicensePolicy,
    LicensePolicy,
    laxer_than,
    license_record,
    policy_digest,
    policy_document,
    policy_of,
)

pytestmark = pytest.mark.logic

IDS = ("MIT", "Apache-2.0", "GPL-3.0-only")
OUTSIDE = "BSD-3-Clause"
VERIFIED = {"date": "2026-11-11", "how": "pep639"}


def _subsets() -> Iterator[frozenset[str]]:
    for size in range(len(IDS) + 1):
        for chosen in itertools.combinations(IDS, size):
            yield frozenset(i.lower() for i in chosen)


def _policies() -> list[LicensePolicy | None]:
    found: list[LicensePolicy | None] = [None]
    for allow in (None, *_subsets()):
        for refuse in _subsets():
            found += [LicensePolicy(allow, refuse, hits) for hits in (False, True)]
    return found


def _accepts(policy: LicensePolicy | None, spdx: str) -> bool:
    """Whether a call under `policy` is allowed a backend licensed `spdx`: the absent policy refuses none."""
    record = license_record(spdx, source="lock", verified=VERIFIED, gated=False)
    return policy is None or policy.refusal("x", record) is None


def _serves_unlicensed(policy: LicensePolicy | None) -> bool:
    return policy is not None and policy.allow_unlicensed_hits


def test_laxer_than_agrees_with_what_each_policy_accepts() -> None:
    """For every pair of small policies, `given` is laxer exactly when it accepts a license the stored one refuses."""
    policies = _policies()
    for given, stored in itertools.product(policies, policies):
        found = laxer_than(given, stored)
        loosens = any(_accepts(given, i) and not _accepts(stored, i) for i in (*IDS, OUTSIDE)) or (
            _serves_unlicensed(given) and not _serves_unlicensed(stored)
        )
        assert (found is not None) == loosens, (given, stored, found)


@pytest.mark.parametrize(
    ("given", "field", "ids"),
    [
        (None, "allow", ()),
        (LicensePolicy(frozenset({"mit", "bsd-3-clause"}), frozenset()), "refuse", ("bsd-3-clause",)),
        (
            LicensePolicy(frozenset({"mit", "bsd-3-clause", "apache-2.0"}), frozenset({"bsd-3-clause"})),
            "allow",
            ("apache-2.0",),
        ),
        (LicensePolicy(None, frozenset({"bsd-3-clause"})), "allow", ()),
        (
            LicensePolicy(frozenset({"mit", "bsd-3-clause"}), frozenset({"bsd-3-clause"}), True),
            "allow_unlicensed_hits",
            (),
        ),
    ],
    ids=["absent", "refuse-removed", "allow-added", "allow-unbounded", "unlicensed-hits"],
)
def test_a_laxer_policy_names_the_field_and_the_ids(
    given: LicensePolicy | None, field: str, ids: tuple[str, ...]
) -> None:
    """Each way of loosening the stored policy is refused with the field and the ids it loosens."""
    stored = LicensePolicy(frozenset({"mit", "bsd-3-clause"}), frozenset({"bsd-3-clause"}))
    found = laxer_than(given, stored)
    assert isinstance(found, LaxerLicensePolicy) and (found.field, found.ids) == (field, ids)


def test_an_unlicensed_policy_given_to_a_run_without_one_may_not_serve_unlicensed_hits() -> None:
    """The absent policy serves no unlicensed hit, so a policy that does is laxer than none."""
    assert laxer_than(LicensePolicy(allow_unlicensed_hits=True), None) is not None
    assert laxer_than(LicensePolicy(), None) is None and laxer_than(None, LicensePolicy()) is None


def test_the_document_round_trips_in_a_fixed_form() -> None:
    """Ids are lower-cased and sorted; the absent policy is null; the digest follows the document."""
    policy = LicensePolicy(frozenset({"MIT", "Apache-2.0"}), frozenset({"GPL-3.0-only"}), True)
    assert policy_document(policy) == {
        "format": "run-licenses@1",
        "policy": {"allow": ["apache-2.0", "mit"], "refuse": ["gpl-3.0-only"], "allow_unlicensed_hits": True},
    }
    assert policy_of(policy_document(policy)) == LicensePolicy(
        frozenset({"mit", "apache-2.0"}), frozenset({"gpl-3.0-only"}), True
    )
    assert policy_of(policy_document(None)) is None and policy_digest(None) != policy_digest(LicensePolicy())
    with pytest.raises(ValueError, match="run-licenses@1"):
        policy_of({"policy": None})
