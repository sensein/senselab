"""Module for testing audio evaluation functions."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Union

import pandas as pd
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control.evaluate import (
    evaluate_audio,
    evaluate_batch,
    evaluate_dataset,
)


def test_evaluate_audio(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests that evaluate_audio correctly processes audio and handles existing results."""
    # Save test audio file
    audio_path = str(tmp_path / "test.wav")
    resampled_mono_audio_sample.save_to_file(audio_path)

    # Create test evaluation functions with proper names
    def test_float(x: Audio) -> float:
        return 0.5

    def test_bool(x: Audio) -> bool:
        return True

    def test_str(x: Audio) -> str:
        return "test"

    evaluations: List[Callable[[Audio], Union[float, bool, str]]] = [
        test_float,
        test_bool,
        test_str,
    ]

    # Test basic evaluation
    results = evaluate_audio(audio_path, "test_activity", evaluations)
    assert results["id"] == Path(audio_path).stem
    assert results["path"] == audio_path
    assert results["activity"] == "test_activity"
    assert results["evaluations"]["test_float"] == 0.5
    assert results["evaluations"]["test_bool"] is True
    assert results["evaluations"]["test_str"] == "test"

    # Test with existing results - caching should work now
    cached_results = {
        "id": Path(audio_path).stem,
        "path": audio_path,
        "activity": "test_activity",
        "evaluations": {
            "test_float": 1.0,  # Should be preserved
            "test_str": "old",  # Should be preserved
        },
        "time_windows": None,
        "windowed_evaluations": None,
    }

    # Create results directory and save cached file
    results_dir = tmp_path / "audio_results"
    results_dir.mkdir(exist_ok=True, parents=True)
    cache_file = results_dir / f"{Path(audio_path).stem}.json"
    with open(cache_file, "w") as f:
        json.dump(cached_results, f)

    # Test loading from cache - cached values should be preserved
    results = evaluate_audio(audio_path, "test_activity", evaluations, output_dir=tmp_path)
    assert results["evaluations"]["test_float"] == 1.0, "Existing float result was not preserved"
    assert results["evaluations"]["test_str"] == "old", "Existing string result was not preserved"
    assert results["evaluations"]["test_bool"] is True, "New evaluation was not computed"


def test_evaluate_batch(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests that evaluate_batch correctly processes multiple audio files and handles caching."""
    # Create two test files
    audio_path1 = str(tmp_path / "test1.wav")
    audio_path2 = str(tmp_path / "test2.wav")
    resampled_mono_audio_sample.save_to_file(audio_path1)
    resampled_mono_audio_sample.save_to_file(audio_path2)

    # Setup test evaluation function
    def test_metric(_: Audio) -> Union[float, bool, str]:
        return 0.5

    # Setup test data
    batch_audio_paths = [audio_path1, audio_path2]
    audio_path_to_activity = {audio_path1: "test_activity", audio_path2: "test_activity"}
    activity_to_evaluations: Dict[
        str,
        List[Callable[[Audio], Union[float, bool, str]]],
    ] = {"test_activity": [test_metric]}

    # Run evaluate_batch
    results = evaluate_batch(
        batch_audio_paths=batch_audio_paths,
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
    )

    # Verify results
    assert len(results) == 2, "Expected results for both audio files"
    for result in results:
        assert "id" in result, "Expected 'id' in result"
        assert "path" in result, "Expected 'path' in result"
        assert "activity" in result, "Expected 'activity' in result"
        assert "evaluations" in result, "Expected 'evaluations' in result"
        assert "test_metric" in result["evaluations"], "Expected metric result"
        assert result["evaluations"]["test_metric"] == 0.5, "Expected metric value of 0.5"

    # Verify caching - files should exist
    results_dir = tmp_path / "audio_results"
    assert results_dir.exists(), "Results directory should be created"
    assert (results_dir / "test1.json").exists(), "Cache file for test1 should exist"
    assert (results_dir / "test2.json").exists(), "Cache file for test2 should exist"

    # Test with existing results
    cached_results = evaluate_batch(
        batch_audio_paths=batch_audio_paths,
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
    )

    # Verify cached results match original results
    # Note: time_windows may be tuples vs lists due to JSON serialization
    assert len(cached_results) == len(results), "Cached results length should match"
    for cached, original in zip(cached_results, results):
        assert cached["id"] == original["id"], "IDs should match"
        assert cached["path"] == original["path"], "Paths should match"
        assert cached["activity"] == original["activity"], "Activities should match"
        assert cached["evaluations"] == original["evaluations"], "Evaluations should match"
        assert cached["windowed_evaluations"] == original["windowed_evaluations"], "Windowed evaluations should match"
        # Time windows may be different format (tuples vs lists) due to JSON serialization


def test_evaluate_dataset(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests that evaluate_dataset correctly processes batches in parallel."""
    # Create multiple test files to test batching
    audio_paths = []
    for i in range(3):  # Create 3 files to test batching
        path = str(tmp_path / f"test{i}.wav")
        resampled_mono_audio_sample.save_to_file(path)
        audio_paths.append(path)

    # Setup test evaluation function
    def test_metric(_: Audio) -> Union[float, bool, str]:
        return 0.5

    # Setup test data
    audio_path_to_activity = {path: "test_activity" for path in audio_paths}
    activity_to_evaluations: Dict[
        str,
        List[Callable[[Audio], Union[float, bool, str]]],
    ] = {"test_activity": [test_metric]}

    # Run evaluations with different configurations
    # Test serial execution
    results_serial = evaluate_dataset(
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
        batch_size=2,  # Should create 2 batches
        n_cores=1,  # Force serial execution
        plugin="serial",
    )

    # Test parallel execution
    results_parallel = evaluate_dataset(
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
        batch_size=2,  # Should create 2 batches
        n_cores=2,  # Use parallel execution
        plugin="cf",
    )

    # Verify results
    assert isinstance(results_serial, pd.DataFrame), "Expected DataFrame output"
    assert isinstance(results_parallel, pd.DataFrame), "Expected DataFrame output"
    assert len(results_serial) == 3, "Expected results for all files"
    assert len(results_parallel) == 3, "Expected results for all files"

    # Verify both methods give same results
    pd.testing.assert_frame_equal(
        results_serial.sort_values("id").reset_index(drop=True),
        results_parallel.sort_values("id").reset_index(drop=True),
    )

    # Verify results content
    for df in [results_serial, results_parallel]:
        for _, row in df.iterrows():
            assert "id" in row, "Expected 'id' in result"
            assert "path" in row, "Expected 'path' in result"
            assert "activity" in row, "Expected 'activity' in result"
            assert "evaluations" in row, "Expected 'evaluations' in result"
            # Note: DataFrame stores nested dict as string, so we need to parse it
            evaluations = row["evaluations"]
            if isinstance(evaluations, str):
                evaluations = json.loads(evaluations)
            assert "test_metric" in evaluations, "Expected metric result"
            assert evaluations["test_metric"] == 0.5, "Expected metric value of 0.5"

    # Verify cache files exist
    results_dir = tmp_path / "audio_results"
    assert results_dir.exists(), "Results directory should be created"
    for i in range(3):
        assert (results_dir / f"test{i}.json").exists(), f"Cache file for test{i} should exist"
