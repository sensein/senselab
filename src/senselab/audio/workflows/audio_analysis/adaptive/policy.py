"""Policy loading, budget ledger, and the deterministic planner (contracts/policy-engine.md)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

_AXIS_PRIORITY = {"utterance": 0, "identity": 1, "presence": 2}
_COST_WEIGHT = {"light": 1.0, "medium": 4.0, "heavy": 16.0}

_DEFAULT_POLICY_PATH = Path(__file__).parent / "policy" / "default.yaml"


def load_policy(path: Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load the default policy, deep-merge an optional override file, attach ``policy_hash``.

    Precedence is packaged default < ``path`` file < ``overrides`` — the CLI wins
    over a policy file, per contracts/cli.md ("Overrides below win over the file").

    ``policy_hash`` is computed *after* all merging, so it identifies the policy
    that actually ran rather than the file on disk. That matters for
    reproducibility: two runs with the same ``--policy`` but different
    ``--budget-heavy`` must not claim the same hash.

    Args:
        path: Optional policy YAML to deep-merge over the packaged default.
        overrides: Optional in-memory overrides (e.g. built from CLI flags),
            deep-merged last. ``None`` values are dropped so an unset flag does
            not clobber a file's value.

    Returns:
        The merged policy dict with ``policy_hash`` attached.
    """
    import yaml  # type: ignore[import-untyped]

    with open(_DEFAULT_POLICY_PATH, encoding="utf-8") as f:
        policy = yaml.safe_load(f)
    if path is not None:
        with open(path, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        policy = _deep_merge(policy, override)
    if overrides:
        policy = _deep_merge(policy, _drop_none(overrides))
    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    policy["policy_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return policy


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip ``None`` values so an unset CLI flag overrides nothing."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = _drop_none(v)
            if nested:
                out[k] = nested
        elif v is not None:
            out[k] = v
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def model_family(model_id: str, policy: dict[str, Any]) -> str:
    """Family name for ``model_id`` per the policy prefix map; fallback = model id itself."""
    for family, prefixes in (policy.get("families") or {}).items():
        for prefix in prefixes:
            if model_id.startswith(prefix):
                return family
    return model_id


def family_weights(model_ids: list[str], policy: dict[str, Any]) -> dict[str, float]:
    """Per-model weight ``1 / |family members present|`` (FR-008)."""
    fams: dict[str, list[str]] = {}
    for m in model_ids:
        fams.setdefault(model_family(m, policy), []).append(m)
    return {m: 1.0 / len(members) for members in fams.values() for m in members}


class BudgetLedger:
    """Per-run intervention budget by cost class (FR-018). Light is uncapped."""

    def __init__(self, policy: dict[str, Any]) -> None:
        """Initialize caps from ``policy["budget"]``."""
        b = policy.get("budget") or {}
        self.caps = {"light": None, "medium": int(b.get("medium_per_run", 24)), "heavy": int(b.get("heavy_per_run", 4))}
        self.spent: dict[str, int] = {"light": 0, "medium": 0, "heavy": 0}
        self.by_rule: dict[str, int] = {}

    def can_admit(self, cost_class: str) -> bool:
        """True when the class cap has headroom."""
        cap = self.caps.get(cost_class)
        return cap is None or self.spent[cost_class] < cap

    def admit(self, cost_class: str, rule_id: str) -> None:
        """Record one spent unit (failed interventions still count — D11)."""
        self.spent[cost_class] = self.spent.get(cost_class, 0) + 1
        self.by_rule[rule_id] = self.by_rule.get(rule_id, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        """Serializable ledger state for round summaries / convergence.json."""
        return {"caps": self.caps, "spent": dict(self.spent), "by_rule": dict(self.by_rule)}


def plan_round(
    *,
    rules: list[dict[str, Any]],
    regions: list[dict[str, Any]],
    ctx: dict[str, Any],
    ledger: BudgetLedger,
    policy: dict[str, Any],
    round_idx: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match rules against regions, rank deterministically, admit within budget.

    Returns ``(admitted, not_admitted)`` where every element carries the full
    decision record (rule, region, trigger values, priority) so both ends of
    the decision surface land in iterations.json (FR-020). Pure given its
    inputs: stable total order (priority desc → axis priority → region start →
    rule id), floats rounded before comparison (FR-025).
    """
    candidates: list[dict[str, Any]] = []
    for rule in rules:
        enabled = ((policy.get("rules") or {}).get(rule["id"]) or {}).get("enabled", True)
        rule_regions: list[dict[str, Any] | None] = (
            [r for r in regions if r["axis"] in rule["axes"] and r.get("status") == "open"]
            if rule["axes"]
            else [None]  # stream-global rules (adjudication) run once per round
        )
        for region in rule_regions:
            fired, trigger = rule["trigger"](region, ctx)
            if not fired:
                continue
            guard_reason = rule["guard"](region, ctx) if rule.get("guard") else None
            gain = float(rule["gain"](region, ctx, trigger))
            priority = round(gain / _COST_WEIGHT[rule["cost"]], 9)
            cand = {
                "rule": rule["id"],
                "cost_class": rule["cost"],
                "region_id": region["region_id"] if region else None,
                "region": region,
                "axis": region["axis"] if region else rule.get("meta_axis", "presence"),
                "start": region["core_start"] if region else 0.0,
                "trigger": trigger,
                "priority": priority,
                "enabled": bool(enabled),
                "guard_reason": guard_reason,
            }
            candidates.append(cand)

    candidates.sort(key=lambda c: (-c["priority"], _AXIS_PRIORITY.get(c["axis"], 3), c["start"], c["rule"]))

    admitted: list[dict[str, Any]] = []
    not_admitted: list[dict[str, Any]] = []
    for cand in candidates:
        if not cand["enabled"]:
            cand["status"] = "blocked_guard"
            cand["error"] = "rule_disabled"
            not_admitted.append(cand)
        elif cand["guard_reason"]:
            cand["status"] = "blocked_guard"
            cand["error"] = cand["guard_reason"]
            not_admitted.append(cand)
        elif not ledger.can_admit(cand["cost_class"]):
            cand["status"] = "deferred_budget"
            not_admitted.append(cand)
        else:
            ledger.admit(cand["cost_class"], cand["rule"])
            cand["status"] = "admitted"
            cand["intervention_id"] = f"{round_idx}_{cand['rule']}_{cand['region_id'] or 'global'}"
            admitted.append(cand)
    return admitted, not_admitted
