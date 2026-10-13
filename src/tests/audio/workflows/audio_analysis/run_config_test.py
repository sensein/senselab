"""The run config carries the decisions, and the pipeline reads them from there.

A threshold reachable only by editing Python is a decision with no record in the artifact that
depended on it — which is the whole reason this repo keeps thresholds in ``data/`` with their
derivation. These guard the half of that rule that is easy to get wrong: the value being *declared*
in the config is worth nothing if nothing reads it.

``speech_presence_link.policy_from_params`` has always been able to read
``params["speech_presence_policy"]``. Until 2026-08-07 nothing ever wrote that key, so every linking
threshold came from a dataclass default and no config could move one — the read side existed, the
write side did not, and no test compared them.
"""

from __future__ import annotations

import pytest

# ── the linking thresholds reach the policy ──────────────────────────────────


def test_the_linking_block_carries_every_policy_field() -> None:
    """A threshold the config cannot move is a decision with no record in the artifact.

    ``policy_from_params`` has always been able to read ``params["speech_presence_policy"]``, and
    nothing ever wrote that key — so every one of these came from a ``SpeechPresencePolicy`` dataclass
    default, and no config could move one. This asserts the block exists and covers the policy.
    """
    from dataclasses import fields

    from senselab.audio.workflows.audio_analysis.run_config import load_run_config
    from senselab.audio.workflows.audio_analysis.speech_presence_link import SpeechPresencePolicy

    cfg = load_run_config(None)
    declared = {f.name for f in fields(SpeechPresencePolicy)}
    missing = sorted(declared - set(cfg.linking))
    assert not missing, f"the config cannot move {missing}"


def test_the_packaged_values_match_the_dataclass_defaults() -> None:
    """The move is about *where* a value is declared, not what it is.

    Pinned so that relocating a threshold cannot silently change a run: every packaged value must
    equal the default it replaced. A deliberate change to one of these should fail here and be made
    on purpose.
    """
    from dataclasses import fields

    from senselab.audio.workflows.audio_analysis.run_config import load_run_config
    from senselab.audio.workflows.audio_analysis.speech_presence_link import SpeechPresencePolicy

    cfg = load_run_config(None)
    for f in fields(SpeechPresencePolicy):
        assert cfg.linking[f.name] == pytest.approx(f.default), (
            f"{f.name}: config says {cfg.linking[f.name]}, the default it replaced was {f.default}"
        )


def test_a_config_override_reaches_the_policy_the_pipeline_applies() -> None:
    """End to end through the reader the pipeline uses, not through the dataclass.

    The failure this guards is silent by construction: ``policy_from_params`` drops keys the policy
    does not declare, so a misspelt threshold looks applied while doing nothing.
    """
    from senselab.audio.workflows.audio_analysis.run_config import load_run_config
    from senselab.audio.workflows.audio_analysis.speech_presence_link import policy_from_params

    cfg = load_run_config(None, overrides={"linking": {"speech_excess_db": 3.5}})
    policy = policy_from_params({"speech_presence_policy": dict(cfg.linking)})
    assert policy.speech_excess_db == pytest.approx(3.5)
    assert policy.lufs_speech == pytest.approx(-30.0), "an override must not disturb its neighbours"


# ── D2: the sections that replaced module-level constants ─────────────────────

#: ``config section -> key -> the module constant it was moved out of``. Written as the pairing rather
#: than as expected numbers, so the test compares the config against the code it replaced instead of
#: against a second copy of the same literal — which would pass even if both had drifted together.
MOVED_CONSTANTS = {
    "rounds_policy": {
        "epistemic_tolerance": ("rounds", "EPISTEMIC_TOLERANCE"),
        "cycle_window": ("rounds", "DEFAULT_CYCLE_WINDOW"),
    },
    "speaker_policy": {
        "centroid_min_similarity": ("harmonize", "MIN_CENTROID_SIMILARITY"),
    },
    "quality_policy": {
        "analysis_win_length": ("quality", "QUALITY_ANALYSIS_WIN_S"),
        "analysis_hop_length": ("quality", "QUALITY_ANALYSIS_HOP_S"),
        "floor_percentile": ("acoustic", "FLOOR_PERCENTILE"),
    },
    "labelstudio_policy": {
        "low_threshold": ("labelstudio", "LOW_THRESHOLD"),
        "high_threshold": ("labelstudio", "HIGH_THRESHOLD"),
    },
    "support_policy": {
        "min_evidence_spread": ("support", "MIN_EVIDENCE_SPREAD"),
        "evidence_low_threshold": ("support", "EVIDENCE_LOW_THRESHOLD"),
        "min_low_fraction": ("support", "MIN_LOW_FRACTION"),
    },
}


def test_each_moved_value_equals_the_constant_it_replaced() -> None:
    """Relocating a decision must not change it.

    The point of the move is *where* a value is declared, not what it is, so a run before and after
    must differ only in the config hash. Compared against the live module attribute rather than a
    literal repeated here — a test holding its own copy of the number passes just as happily when both
    copies drift.
    """
    import importlib

    from senselab.audio.workflows.audio_analysis.run_config import load_run_config

    cfg = load_run_config(None)
    for section, moved in MOVED_CONSTANTS.items():
        block = getattr(cfg, section)
        for key, (module_name, const) in moved.items():
            module = importlib.import_module(f"senselab.audio.workflows.audio_analysis.{module_name}")
            assert block[key] == pytest.approx(getattr(module, const)), (
                f"{section}.{key} is {block[key]}, but {module_name}.{const} is {getattr(module, const)}"
            )


PACKAGED_CONFIG_HASH = "e0e66114efc7ac08c04c90b354de94e98e477a5e0e275a0696f1bad1fcbe4136"
"""``config_hash`` of the packaged ``data/run_config/default.yaml`` as of schema ``version: 3``."""


def test_the_packaged_configs_identity_is_pinned_to_a_literal() -> None:
    """Run identity must move only when a value moves.

    ``load_run_config`` hashes the *merged mapping*, so every key in the packaged YAML is part of
    the run's identity — including ``derivation``, which reads like documentation and is a string
    value. A word changed inside it restamps the run summary, every L1 signal parquet, every L2
    estimate, the disagreements index and the LS bundle, and two behaviourally identical runs then
    report different configs. Only a literal catches that; comparing the hash to a recomputation of
    the same file cannot. See F-189 in ``specs/20260815-215106-analyze-audio-audit/register.md``.

    Update the literal deliberately, in the same commit as the value change that earned it.
    """
    from senselab.audio.workflows.audio_analysis.run_config import load_run_config

    identity = load_run_config(None).identity
    assert identity.version == "3", "bumping the schema version is a deliberate identity change"
    assert identity.config_hash == PACKAGED_CONFIG_HASH, (
        "the packaged config's identity moved. If a decision value changed, update "
        "PACKAGED_CONFIG_HASH here in the same commit. If only prose changed, it was edited inside "
        "a hashed value — put the correction in a `#` comment instead, which is not parsed."
    )


def test_an_override_reaches_each_section() -> None:
    """A section nothing can override is a section that only looks configurable."""
    from senselab.audio.workflows.audio_analysis.run_config import load_run_config

    probes = {
        "rounds": ("epistemic_tolerance", 0.5, "rounds_policy"),
        "speaker": ("centroid_min_similarity", 0.9, "speaker_policy"),
        "quality": ("floor_percentile", 25.0, "quality_policy"),
        "labelstudio": ("high_threshold", 0.8, "labelstudio_policy"),
        "support": ("min_low_fraction", 0.5, "support_policy"),
    }
    for section, (key, value, attr) in probes.items():
        cfg = load_run_config(None, overrides={section: {key: value}})
        assert getattr(cfg, attr)[key] == pytest.approx(value), f"{section}.{key} did not take"
