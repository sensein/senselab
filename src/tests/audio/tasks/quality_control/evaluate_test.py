"""Module for testing audio evaluation functions."""

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Union

import pandas as pd

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
    assert results["evaluations"]["test_float"] == 1.0, "Cached float preserved"
    assert results["evaluations"]["test_str"] == "old", "Cached string preserved"
    assert results["evaluations"]["test_bool"] is True, "New evaluation computed"


def test_evaluate_audio_windowing(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests that windowing works correctly in evaluate_audio."""
    # Save test audio file
    audio_path = str(tmp_path / "test.wav")
    resampled_mono_audio_sample.save_to_file(audio_path)

    # Create test evaluation function
    def test_metric(audio: Audio) -> Union[float, bool, str]:
        # Return duration in seconds
        return len(audio.waveform[0]) / audio.sampling_rate

    evaluations = [test_metric]

    # Test with windowing enabled (default)
    results = evaluate_audio(
        audio_path, "test_activity", evaluations, window_size_sec=1.0, step_size_sec=0.5, skip_windowing=False
    )

    # Verify windowed results exist
    assert "windowed_evaluations" in results
    assert results["windowed_evaluations"] is not None
    assert "test_metric" in results["windowed_evaluations"]

    # Verify time_windows exist and are correct format
    assert "time_windows" in results
    assert results["time_windows"] is not None
    assert isinstance(results["time_windows"], list)
    assert len(results["time_windows"]) > 0

    # Each time window should be a tuple of (start, end)
    for window in results["time_windows"]:
        assert isinstance(window, tuple)
        assert len(window) == 2
        assert isinstance(window[0], (int, float))
        assert isinstance(window[1], (int, float))
        assert window[1] > window[0]  # end > start

    # Verify windowed evaluations match time windows
    windowed_values = results["windowed_evaluations"]["test_metric"]
    assert len(windowed_values) == len(results["time_windows"])

    # Test with windowing disabled
    results_no_windowing = evaluate_audio(audio_path, "test_activity", evaluations, skip_windowing=True)

    # Verify no windowed results when disabled
    assert results_no_windowing["windowed_evaluations"] is None
    assert results_no_windowing["time_windows"] is None


def test_evaluate_audio_windowing_cache(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests that windowed evaluation caching works correctly."""
    # Save test audio file
    audio_path = str(tmp_path / "test.wav")
    resampled_mono_audio_sample.save_to_file(audio_path)

    # Create test evaluation function
    def test_metric(audio: Audio) -> Union[float, bool, str]:
        return 0.5

    evaluations = [test_metric]

    # First run to create cached results
    results1 = evaluate_audio(
        audio_path, "test_activity", evaluations, output_dir=tmp_path, window_size_sec=1.0, step_size_sec=0.5
    )

    # Verify windowed results were created
    assert "windowed_evaluations" in results1
    assert "test_metric" in results1["windowed_evaluations"]

    # Modify cached results to test caching
    results_dir = tmp_path / "audio_results"
    cache_file = results_dir / f"{Path(audio_path).stem}.json"

    with open(cache_file, "r") as f:
        cached_data = json.load(f)

    # Modify windowed evaluations in cache with correct count
    # Use the actual values from the error: 10 windows for this audio
    modified_values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    cached_data["windowed_evaluations"]["test_metric"] = modified_values

    with open(cache_file, "w") as f:
        json.dump(cached_data, f)

    # Second run should load from cache
    results2 = evaluate_audio(
        audio_path, "test_activity", evaluations, output_dir=tmp_path, window_size_sec=1.0, step_size_sec=0.5
    )

    # Verify cached windowed values were loaded
    assert results2["windowed_evaluations"]["test_metric"] == modified_values
    # Time windows should also be preserved from cache
    assert results2["time_windows"] == results1["time_windows"]


def test_evaluate_audio_different_window_params(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests windowing with different window sizes and step sizes."""
    # Save test audio file
    audio_path = str(tmp_path / "test.wav")
    resampled_mono_audio_sample.save_to_file(audio_path)

    def test_metric(audio: Audio) -> Union[float, bool, str]:
        return 1.0

    evaluations = [test_metric]

    # Test with larger window and step size
    results_large = evaluate_audio(audio_path, "test_activity", evaluations, window_size_sec=2.0, step_size_sec=1.0)

    # Test with smaller window and step size
    results_small = evaluate_audio(audio_path, "test_activity", evaluations, window_size_sec=0.5, step_size_sec=0.25)

    # Large windows should have fewer windows than small windows
    large_window_count = len(results_large["time_windows"])
    small_window_count = len(results_small["time_windows"])
    assert small_window_count > large_window_count

    # Verify window sizes are correct
    for start, end in results_large["time_windows"]:
        # Window size should be 2.0
        assert abs((end - start) - 2.0) < 0.001

    for start, end in results_small["time_windows"]:
        # Window size should be 0.5
        assert abs((end - start) - 0.5) < 0.001


def test_evaluate_batch(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests that evaluate_batch correctly processes multiple audio files and handles caching."""
    # Create two test files
    audio_path1 = str(tmp_path / "test1.wav")
    audio_path2 = str(tmp_path / "test2.wav")
    resampled_mono_audio_sample.save_to_file(audio_path1)
    resampled_mono_audio_sample.save_to_file(audio_path2)

    # Setup test evaluation function
    def test_metric(audio: Audio) -> Union[float, bool, str]:
        return 0.5

    # Setup test data
    batch_audio_paths = [audio_path1, audio_path2]
    audio_path_to_activity = {audio_path1: "test_activity", audio_path2: "test_activity"}
    activity_to_evaluations: Dict[
        str,
        Sequence[Callable[[Audio], Union[float, bool, str]]],
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
        expected_value = 0.5
        actual_value = result["evaluations"]["test_metric"]
        msg = f"Expected {expected_value}, got {actual_value}"
        assert actual_value == expected_value, msg

    # Verify caching - files should exist
    results_dir = tmp_path / "audio_results"
    assert results_dir.exists(), "Results directory should be created"
    assert (results_dir / "test1.json").exists(), "Cache file for test1 exists"
    assert (results_dir / "test2.json").exists(), "Cache file for test2 exists"

    # Test with existing results
    cached_results = evaluate_batch(
        batch_audio_paths=batch_audio_paths,
        audio_path_to_activity=audio_path_to_activity,
        activity_to_evaluations=activity_to_evaluations,
        output_dir=tmp_path,
    )

    # Verify cached results match original results
    # Note: time_windows may be tuples vs lists due to JSON serialization
    assert len(cached_results) == len(results), "Cached results length matches"
    for cached, original in zip(cached_results, results):
        assert cached["id"] == original["id"], "IDs should match"
        assert cached["path"] == original["path"], "Paths should match"
        assert cached["activity"] == original["activity"], "Activities match"
        assert cached["evaluations"] == original["evaluations"], "Evaluations match"
        windowed_match = cached["windowed_evaluations"] == original["windowed_evaluations"]
        assert windowed_match, "Windowed evaluations should match"
        # Time windows may be different format due to JSON serialization


def test_evaluate_dataset(tmp_path: Path, resampled_mono_audio_sample: Audio) -> None:
    """Tests that evaluate_dataset correctly processes batches in parallel."""
    # Create multiple test files to test batching
    audio_paths = []
    for i in range(3):  # Create 3 files to test batching
        path = str(tmp_path / f"test{i}.wav")
        resampled_mono_audio_sample.save_to_file(path)
        audio_paths.append(path)

    # Setup test evaluation function
    def test_metric(audio: Audio) -> Union[float, bool, str]:
        return 0.5

    # Setup test data
    audio_path_to_activity = {path: "test_activity" for path in audio_paths}
    activity_to_evaluations: Dict[
        str,
        Sequence[Callable[[Audio], Union[float, bool, str]]],
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
    assert isinstance(results_serial, pd.DataFrame), "Expected DataFrame"
    assert isinstance(results_parallel, pd.DataFrame), "Expected DataFrame"
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
            # In flattened structure, evaluations become individual columns
            assert "test_metric" in row, "Expected 'test_metric' column in result"
            assert row["test_metric"] == 0.5, "Expected metric value"

    # Verify cache files exist
    results_dir = tmp_path / "audio_results"
    assert results_dir.exists(), "Results directory should be created"
    for i in range(3):
        cache_file = results_dir / f"test{i}.json"
        assert cache_file.exists(), f"Cache file for test{i} should exist"
