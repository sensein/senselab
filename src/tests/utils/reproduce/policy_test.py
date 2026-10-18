"""Tests for the profile's ``policy`` block: each project rule is read from the site profile, not the library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from senselab.utils.reproduce import slurm
from senselab.utils.reproduce.cache import canonical_json, step_key, step_key_payload
from senselab.utils.reproduce.metrics import validate_metrics
from senselab.utils.reproduce.provenance import STORE_FILE, Artifact, record_step
from senselab.utils.reproduce.spec import MetricsSchema, Policy, Profile, StepSpec, check_step_profile
from tests.utils.reproduce.cache_test import _kwargs as _key_kwargs
from tests.utils.reproduce.cache_test import _profile as _key_profile
from tests.utils.reproduce.cache_test import _step_data
from tests.utils.reproduce.provenance_test import _kwargs as _prov_kwargs
from tests.utils.reproduce.slurm_test import DATE, EXPORTS, _spec
from tests.utils.reproduce.slurm_test import _profile as _slurm_profile

TODAY = {
    "min_cell": 5,
    "headline_rule": "balanced_accuracy_only",
    "require_deny_patterns": True,
    "require_canary": True,
    "confidential_provenance": True,
    "umask": "007",
}


def _with_policy(profile: Profile, **change: Any) -> Profile:  # noqa: ANN401
    return Profile.model_validate({**profile.model_dump(mode="json"), "policy": {**TODAY, **change}})


def test_min_cell_comes_from_the_policy() -> None:
    """A count of 4 is kept under ``min_cell: 3`` and written as ``"<5"`` under ``min_cell: 5``."""
    schema = MetricsSchema.model_validate({"kind": "count", "headline": ["count"]})
    obj = {"headline": [{"name": "count", "value": 4}]}
    lenient, strict = Policy.model_validate({**TODAY, "min_cell": 3}), Policy.model_validate(TODAY)
    clean, codes = validate_metrics(obj, schema, min_cell=lenient.min_cell)
    assert codes == [] and clean == {"headline": [{"name": "count", "value": 4}]}
    clean, codes = validate_metrics(obj, schema, min_cell=strict.min_cell)
    assert codes == [] and clean == {"headline": [{"name": "count", "value": "<5"}]}
    clean, _ = validate_metrics({"headline": [{"name": "count", "value": 2}]}, schema, min_cell=3)
    assert clean == {"headline": [{"name": "count", "value": "<3"}]}
    with pytest.raises(ValidationError):
        Policy.model_validate({**TODAY, "min_cell": 0})


def test_kind_vocabulary_allows_a_non_ba_classification_headline(tmp_path: Path) -> None:
    """``kind_vocabulary`` accepts a macro-F1 headline (floor and lift still required); the default rule refuses it."""
    data = {"kind": "classification", "headline": ["macro_f1"], "per_class": ["pos", "neg"]}
    with pytest.raises(ValidationError, match="E_SPEC_HEADLINE_NOT_BA"):
        MetricsSchema.model_validate(data)
    schema = MetricsSchema.model_validate(data, context={"headline_rule": "kind_vocabulary"})
    with pytest.raises(ValidationError, match="do not belong to kind"):
        MetricsSchema.model_validate({**data, "headline": ["f_stat"]}, context={"headline_rule": "kind_vocabulary"})
    obj: dict[str, Any] = {
        "headline": [{"name": "macro_f1", "value": 0.7, "floor": 0.0, "lift": 0.7}],
        "per_class": {
            "pos": {"recall": 0.8, "precision": 0.6, "prevalence": 0.4},
            "neg": {"recall": 0.6, "precision": 0.8, "prevalence": 0.6},
        },
        "split_report": {"protocol": "unseen_subject", "unit": "subject", "invariant_ok": True},
        "threshold_tuning": "validation",
    }
    assert validate_metrics(obj, schema, min_cell=5)[1] == []
    del obj["headline"][0]["floor"], obj["headline"][0]["lift"]
    assert validate_metrics(obj, schema, min_cell=5)[1] == ["E_METRICS_NO_FLOOR", "E_METRICS_NO_LIFT"]
    step_data = _step_data(metrics_schema=data, expect=[])
    step = StepSpec.model_validate(step_data, context={"headline_rule": "kind_vocabulary"})
    profile = _key_profile(tmp_path)
    check_step_profile(step, _with_policy(profile, headline_rule="kind_vocabulary"))
    with pytest.raises(ValueError, match="E_SPEC_HEADLINE_NOT_BA"):
        check_step_profile(step, _with_policy(profile))


def test_required_deny_patterns_and_canary_are_enforced(tmp_path: Path) -> None:
    """The policy is required; a profile without deny patterns or a canary is ``E_PROFILE_POLICY`` when required."""
    data = _slurm_profile(tmp_path).model_dump(mode="json")
    without = {k: v for k, v in data.items() if k != "policy"}
    with pytest.raises(ValidationError, match="policy"):
        Profile.model_validate(without)
    for field, flag in (("deny_patterns_file", "require_deny_patterns"), ("canary_host", "require_canary")):
        missing = {k: v for k, v in data.items() if k != field}
        with pytest.raises(ValidationError, match="E_PROFILE_POLICY"):
            Profile.model_validate(missing)
        relaxed = Profile.model_validate({**missing, "policy": {**TODAY, flag: False}})
        assert getattr(relaxed, field) is None
    assert Profile.model_validate(data).policy == Policy.model_validate(TODAY)


def test_non_confidential_provenance_records_paths_and_argv(tmp_path: Path) -> None:
    """Given paths and argv, the record carries both; confidential records carry neither."""
    argv = ["/venv/bin/python", "/src/step.py", "/data/in/a.txt"]
    paths = ("/data/in/*.txt",)
    inputs = [Artifact(digest="3" * 64, count=1, root="data", paths=paths)]
    record_step(tmp_path / "open", **_prov_kwargs(tmp_path, inputs=inputs, argv=argv, argv_sha256="2" * 64))
    store = (tmp_path / "open" / STORE_FILE).read_text()
    assert "/data/in/a.txt" in store and "/data/in/*.txt" in store
    records = [json.loads(line) for line in store.splitlines()]
    assert any(r.get("parameters", {}).get("argv") == argv for r in records)
    closed = [Artifact(digest="3" * 64, count=1, root="data")]
    record_step(tmp_path / "closed", **_prov_kwargs(tmp_path, inputs=closed, argv_sha256="2" * 64))
    assert "/data/" not in (tmp_path / "closed" / STORE_FILE).read_text()
    with pytest.raises(ValueError, match="argv"):
        record_step(tmp_path / "bad", **_prov_kwargs(tmp_path, argv=["ok", 3]))


def test_the_policy_is_keyed(tmp_path: Path) -> None:
    """``policy_sha256`` is the canonical-JSON sha of the policy; any policy change re-keys the step."""
    step = StepSpec.model_validate(_step_data())
    profile = _key_profile(tmp_path)
    payload = step_key_payload(step, profile, **_key_kwargs())
    expected = hashlib.sha256(canonical_json(profile.policy.model_dump(mode="json"))).hexdigest()
    assert payload["policy_sha256"] == expected
    base = step_key(step, profile, **_key_kwargs())
    assert step_key(step, _with_policy(profile), **_key_kwargs()) == base
    for change in ({"min_cell": 6}, {"confidential_provenance": False}, {"umask": "022"}):
        assert step_key(step, _with_policy(profile, **change), **_key_kwargs()) != base


def test_umask_reaches_the_batch_script(tmp_path: Path) -> None:
    """``umask: "022"`` gives ``umask 022`` in the script, a ``2755`` log dir and a 022 submit umask."""
    profile = _with_policy(_slurm_profile(tmp_path), umask="022")
    kwargs: dict[str, Any] = {"partition": "main", "command": ["run"], "date": DATE, "exports": EXPORTS}
    lines = slurm.render_step_job(_spec("prep"), profile, **kwargs).splitlines()
    assert "umask 022" in lines and "umask 007" not in lines
    finalize = slurm.render_finalize_job(profile, command=["run"], date=DATE, exports=EXPORTS, dependencies=[])
    assert "umask 022" in finalize.splitlines()
    logs = slurm.prepare_log_dir(profile, DATE)
    assert logs.stat().st_mode & 0o7777 == 0o2755 == slurm.log_dir_mode(profile)
    assert slurm.submit_umask(profile) == 0o022
    assert slurm.submit_umask(_with_policy(profile)) == 0o007 and slurm.log_dir_mode(_with_policy(profile)) == 0o2770
