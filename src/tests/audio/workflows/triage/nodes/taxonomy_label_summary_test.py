"""TAXONOMY's whole-file label-score summary.

Its own file rather than a class in ``taxonomy_test.py``: every test here seeds a score sidecar and
reads one measurement back, which is a different setup from the kind-folding tests next door.
"""

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import window


def _floors(tmp_path: Path) -> TriageConfig:
    """The packaged config with the TAXONOMY floors supplied, so the fold itself does not blank.

    Args:
        tmp_path: Where the override YAML is written.

    Returns:
        The resolved configuration.
    """
    path = tmp_path / "summary-floors.yaml"
    path.write_text(
        "taxonomy:\n"
        "  presence_floor:\n"
        "    speech: {acoustic: 1, lexical: 1}\n"
        "    airway: {health_acoustic: 1, acoustic: 1}\n"
        "  voice_min_duration_s: 1.0\n"
        "  voice_uncertain_duration_s: 0.3\n"
        "  speech_labels: [Speech]\n"
    )
    return load_triage_config(path)


def _sidecar(tmp_path: Path, classifier: str, windows: list[dict[str, Any]]) -> None:
    """Write one classifier's verbatim windows where the store's path attribute points.

    Args:
        tmp_path: The run directory.
        classifier: ``"yamnet"``, ``"ast"`` or ``"hear"``.
        windows: The windows, in the shape ``label_scores`` reads.
    """
    path = tmp_path / "derivatives" / f"{classifier}_scores.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(windows))


def test_it_summarises_peak_and_median_per_label(
    store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
) -> None:
    """Peak and median come from every window the label appears in, not from a firing subset."""
    seed_preprocess_store(store, yamnet_labels=[["Speech"], ["Speech"], ["Speech"]])
    _sidecar(
        tmp_path,
        "yamnet",
        [
            window(0.0, 0.96, {"Speech": 0.10, "Cough": 0.90}),
            window(0.48, 1.44, {"Speech": 0.50, "Cough": 0.10}),
            window(0.96, 1.92, {"Speech": 0.90}),
        ],
    )
    taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
    summary = find_measurement(store, "yamnet_label_summary")
    assert summary is not None
    labels = summary.attributes["labels"]
    assert labels["Speech"]["peak"] == pytest.approx(0.9)
    assert labels["Speech"]["median"] == pytest.approx(0.5)
    assert labels["Speech"]["n_windows"] == 3
    assert labels["Cough"]["peak"] == pytest.approx(0.9)
    assert labels["Cough"]["median"] == pytest.approx(0.5)
    assert labels["Cough"]["n_windows"] == 2


def test_a_label_firing_weakly_in_many_windows_is_distinguishable(
    store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
) -> None:
    """A window count alone cannot separate these two, which is why the summary carries scores."""
    seed_preprocess_store(store, yamnet_labels=[["Speech"]] * 4)
    _sidecar(
        tmp_path,
        "yamnet",
        [
            window(0.0, 0.96, {"Weak": 0.05, "Strong": 0.95}),
            window(0.48, 1.44, {"Weak": 0.05}),
            window(0.96, 1.92, {"Weak": 0.05}),
            window(1.44, 2.40, {"Weak": 0.05}),
        ],
    )
    taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
    summary = find_measurement(store, "yamnet_label_summary")
    assert summary is not None
    labels = summary.attributes["labels"]
    assert labels["Weak"]["n_windows"] == 4
    assert labels["Weak"]["peak"] == pytest.approx(0.05)
    assert labels["Strong"]["n_windows"] == 1
    assert labels["Strong"]["peak"] == pytest.approx(0.95)
    assert list(labels)[0] == "Strong", "ordered by peak, so a panel's top rows are the loudest labels"


def test_it_is_written_under_the_packaged_config_whose_thresholds_are_null(
    store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
) -> None:
    """The point of the summary: it survives the nulls that leave every window fold absent."""
    seed_preprocess_store(store, yamnet_labels=[["Speech"]], scores_only=("yamnet",))
    taxonomy(store, "plain", config, run_dir=tmp_path)
    assert find_measurement(store, "yamnet_windows") is None
    summary = find_measurement(store, "yamnet_label_summary")
    assert summary is not None
    assert summary.attributes["labels"]


def test_it_reads_no_threshold(store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path) -> None:
    """A score below every plausible threshold still reaches the summary."""
    seed_preprocess_store(store, yamnet_labels=[["Speech"]])
    _sidecar(tmp_path, "yamnet", [window(0.0, 0.96, {"Barely": 0.001})])
    taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
    summary = find_measurement(store, "yamnet_label_summary")
    assert summary is not None
    assert summary.attributes["labels"]["Barely"]["peak"] == pytest.approx(0.001)


def test_a_classifier_that_never_ran_gets_no_summary(
    store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
) -> None:
    """Absent and all-zero must stay distinguishable, so nothing is written for a missing run."""
    seed_preprocess_store(store, yamnet_labels=[["Speech"]])
    taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
    assert find_measurement(store, "ast_label_summary") is None
    assert find_measurement(store, "hear_label_summary") is None


def test_it_records_the_grid_it_summarised(
    store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
) -> None:
    """A distribution over 0.96 s windows is not one over 10.24 s windows."""
    seed_preprocess_store(store, yamnet_labels=[["Speech"], ["Speech"]])
    taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
    summary = find_measurement(store, "yamnet_label_summary")
    assert summary is not None
    assert (summary.attributes["win_length_s"], summary.attributes["hop_s"]) == (0.96, 0.48)
    assert summary.attributes["n_windows"] == 2
    assert summary.extent is None, "the summary describes the whole file, so it carries no extent"


def test_it_writes_nothing_when_the_sidecar_is_missing(
    store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
) -> None:
    """A store whose path attribute points at a file nobody wrote yields no summary, not a crash."""
    seed_preprocess_store(store, yamnet_labels=[["Speech"]])
    (tmp_path / "derivatives" / "yamnet_scores.json").unlink()
    taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
    assert find_measurement(store, "yamnet_label_summary") is None
