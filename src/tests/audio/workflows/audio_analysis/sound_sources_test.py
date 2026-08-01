"""Tests for the sound-source categorizer (feature 20260722-175022, US2).

Covers SC-003 (complete, non-overlapping category coverage of the classifier
vocabularies), mass normalization + dominant selection, a background-machine
scenario, and the null-safe path when no classifier ran (FR-023).
"""

from __future__ import annotations

import json
from importlib import resources
from types import SimpleNamespace

import pytest
import torch

from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.sound_sources import (
    SOURCE_CATEGORIES,
    harvest_source_categories,
    load_source_category_map,
)


def _classification_block(windows: list[dict]) -> dict:
    """Wrap per-window dicts as an AST/YAMNet pass_summary block."""
    return {"status": "ok", "result": [windows], "cache_key": "cls_k"}


def _window(start: float, end: float, labels: list[str], scores: list[float]) -> dict:
    """Build one classification window dict."""
    return {
        "start": start,
        "end": end,
        "labels": labels,
        "scores": scores,
        "win_length": end - start,
        "hop_length": 0.5,
    }


def test_category_map_covers_all_classifier_classes() -> None:
    """SC-003: every AST (527) and YAMNet (521) class maps to exactly one of the 4 categories."""
    doc = load_source_category_map()
    mapping = doc["map"]

    # Every value is a valid category (non-overlapping: dict → exactly one each).
    assert set(mapping.values()) <= set(SOURCE_CATEGORIES)
    for name, cat in mapping.items():
        assert cat in SOURCE_CATEGORIES, f"{name!r} → invalid category {cat!r}"

    # AST coverage: the map was authored over the full 527-class AST vocabulary.
    assert len(mapping) == 527

    # YAMNet coverage: the vendored 521-class list must all be present as keys.
    yam_res = resources.files("senselab.audio.workflows.audio_analysis").joinpath("data", "yamnet_class_names.json")
    yam = json.loads(yam_res.read_text(encoding="utf-8"))["names"]
    assert len(yam) == 521
    missing = [n for n in yam if n not in mapping]
    assert not missing, f"YAMNet classes missing from map: {missing[:10]}"


def test_masses_sum_to_one_and_dominant_is_argmax() -> None:
    """Per-bucket masses normalize to ~1 and src_dominant is their argmax."""
    windows = [_window(0.0, 0.5, ["Speech", "Vehicle", "Wind"], [0.7, 0.2, 0.1])]
    rows = harvest_source_categories(
        pass_summary={"duration_s": 0.5, "ast": _classification_block(windows)},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert rows
    r = rows[0]
    total = r["src_speech"] + r["src_people"] + r["src_machine"] + r["src_environment"]
    assert abs(total - 1.0) < 1e-6
    assert r["src_dominant"] == "speech"
    assert r["src_speech"] > r["src_machine"] > r["src_environment"]


def test_background_machine_window_dominant_machine() -> None:
    """A window dominated by vehicle/engine classes → src_machine dominant."""
    windows = [_window(0.0, 0.5, ["Engine", "Vehicle", "Speech"], [0.6, 0.3, 0.1])]
    rows = harvest_source_categories(
        pass_summary={"duration_s": 0.5, "yamnet": _classification_block(windows)},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert rows[0]["src_dominant"] == "machine"


def test_ast_and_yamnet_averaged() -> None:
    """When both classifiers run, masses are the mean of the two distributions."""
    ast_win = [_window(0.0, 0.5, ["Speech"], [1.0])]
    yam_win = [_window(0.0, 0.5, ["Vehicle"], [1.0])]
    rows = harvest_source_categories(
        pass_summary={
            "duration_s": 0.5,
            "ast": _classification_block(ast_win),
            "yamnet": _classification_block(yam_win),
        },
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    r = rows[0]
    assert abs(r["src_speech"] - 0.5) < 1e-6
    assert abs(r["src_machine"] - 0.5) < 1e-6


def test_null_when_no_classifier() -> None:
    """FR-023 / T017: no AST/YAMNet → all src_* columns null."""
    rows = harvest_source_categories(
        pass_summary={"duration_s": 1.0},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
    )
    assert rows
    for r in rows:
        assert r["src_speech"] is None
        assert r["src_dominant"] is None


# ── score comparability across classifiers (T014, FR-017c) ─────────────
#
# The two scene classifiers produce scores with different *competition structure*:
# one is a softmax across all 527 AudioSet classes, the other an independent
# per-class sigmoid. Per-window normalization cancels the scale difference but not
# the structure: softmax suppresses secondary classes multiplicatively, so a
# background source at the same underlying evidence gets a systematically smaller
# share than it would under sigmoid. Averaging the two therefore under-weights
# background whenever a dominant source is present — exactly the sources this
# feature exists to surface.


def _softmax(logits: list[float]) -> list[float]:
    import math

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _sigmoid(logits: list[float]) -> list[float]:
    import math

    return [1.0 / (1.0 + math.exp(-x)) for x in logits]


def test_softmax_suppresses_a_secondary_category_relative_to_sigmoid() -> None:
    """Quantifies the defect, so the chosen convention is a recorded decision.

    Same underlying evidence (a dominant speech logit plus a clear vehicle logit);
    only the output transform differs. The secondary category's normalized mass is
    several times smaller under competition.
    """
    from senselab.audio.workflows.audio_analysis.sound_sources import _window_category_masses, load_source_category_map

    doc = load_source_category_map()
    mapping, default = doc["map"], doc["default"]
    labels = ["Speech", "Vehicle"]
    logits = [3.0, 1.0]
    # 525 remaining classes sit far below and are outside a top-k window, but they
    # still take softmax mass — which is the whole mechanism.
    competitive = _softmax(logits + [-5.0] * 525)[:2]
    independent = _sigmoid(logits)

    comp = _window_category_masses({"labels": labels, "scores": competitive}, mapping, default)
    indep = _window_category_masses({"labels": labels, "scores": independent}, mapping, default)
    assert comp is not None and indep is not None
    assert indep["machine"] > 3.0 * comp["machine"], (
        f"expected competition to suppress the secondary category; got {comp['machine']:.3f} vs {indep['machine']:.3f}"
    )


def test_audioset_score_function_is_independent_per_class() -> None:
    """The convention this feature standardizes on (FR-017c).

    AudioSet is a multi-label task, so an independent per-class score is the correct
    reading of a 527-class head. Ranking is unaffected — both transforms are monotone
    in the logit — so only the mass proportions change, not which labels are selected.
    """
    from senselab.audio.workflows.audio_analysis.sound_sources import AUDIOSET_SCORE_FUNCTION

    assert AUDIOSET_SCORE_FUNCTION == "sigmoid"


def test_stage_scene_requests_the_independent_score_function(monkeypatch: "pytest.MonkeyPatch") -> None:
    """The wiring, not just the constant: AST must actually be called with it."""
    import senselab.audio.workflows.audio_analysis.stages as stages_mod
    from senselab.audio.workflows.audio_analysis.sound_sources import AUDIOSET_SCORE_FUNCTION
    from senselab.audio.workflows.audio_analysis.stage_context import StageContext

    seen: list[dict[str, object]] = []

    def _spy(*_args: object, **kwargs: object) -> list[list[dict[str, object]]]:
        seen.append(dict(kwargs))
        return [[{"start": 0.0, "end": 1.0, "labels": ["Speech"], "scores": [0.9]}]]

    monkeypatch.setattr(stages_mod, "classify_audios", _spy)

    audio = SimpleNamespace(waveform=torch.zeros((1, 16000)), sampling_rate=16000)
    ctx = StageContext(pass_label="raw_16k", audio_signature="s" * 64)
    stages_mod.stage_scene(
        # A waveform/sampling_rate stand-in: the stage reads nothing else off it.
        audio,  # type: ignore[arg-type]
        ctx,
        ast_model="MIT/ast-finetuned-audioset-10-10-0.4593",
        yamnet_model=None,
        ast_win_length=10.24,
        ast_hop_length=10.24,
        yamnet_win_length=0.96,
        yamnet_hop_length=0.48,
        top_k=50,
    )
    assert seen, "classify_audios was never called"
    assert seen[0].get("function_to_apply") == AUDIOSET_SCORE_FUNCTION


def test_label_mass_counts_a_second_ranked_speech_label() -> None:
    """The reason top-1 had to go: a runner-up speech label carries real evidence.

    ``top-1 in speech_labels`` on this window votes a confident *no speech*, because ``Music``
    edges out ``Speech`` by 0.02 — discarding 0.38 of speech mass out of 1.0.
    """
    from senselab.audio.workflows.audio_analysis.sound_sources import window_label_mass

    labels = ["Music", "Speech", "Guitar"]
    window = {
        "start": 0.0,
        "end": 0.96,
        "labels": labels,
        "scores": [0.40, 0.38, 0.22],
    }
    mass = window_label_mass(window, {"Speech"})
    assert mass is not None
    assert mass == pytest.approx(0.38)
    # And the old rule's verdict, stated so the regression is unambiguous.
    assert labels[0] not in {"Speech"}


def test_label_mass_sums_across_several_speech_labels() -> None:
    """Mass over the whole subset, since speech has more than one AudioSet class."""
    from senselab.audio.workflows.audio_analysis.sound_sources import window_label_mass

    window = {"labels": ["Speech", "Narration, monologue", "Rain"], "scores": [0.3, 0.3, 0.4]}
    assert window_label_mass(window, {"Speech", "Narration, monologue"}) == pytest.approx(0.6)


def test_label_mass_is_none_without_scores() -> None:
    """No scores is not zero mass — it is an absent measurement."""
    from senselab.audio.workflows.audio_analysis.sound_sources import window_label_mass

    assert window_label_mass({"labels": [], "scores": []}, {"Speech"}) is None
    assert window_label_mass(None, {"Speech"}) is None
