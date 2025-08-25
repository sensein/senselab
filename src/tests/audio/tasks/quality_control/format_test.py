"""Tests for quality control evaluation formatting utilities."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from senselab.audio.tasks.quality_control.format import (
    extract_windowed_data,
    flatten_results_to_dataframe,
    save_formatted_results,
)


@pytest.fixture
def sample_results() -> List[Dict[str, Any]]:
    """Sample evaluation results for testing."""
    return [
        {
            "id": "audio1",
            "path": "/path/to/audio1.wav",
            "activity": "speech",
            "evaluations": {
                "zero_crossing_rate": 0.123,
                "spectral_centroid": 1500.0,
                "is_clipped": False,
            },
            "time_windows": [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)],
            "windowed_evaluations": {
                "zero_crossing_rate": [0.1, 0.15, 0.12],
                "spectral_centroid": [1400.0, 1600.0, 1550.0],
                "is_clipped": [False, False, True],
            },
        },
        {
            "id": "audio2",
            "path": "/path/to/audio2.wav",
            "activity": "music",
            "evaluations": {
                "zero_crossing_rate": 0.089,
                "spectral_centroid": 2200.0,
                "is_clipped": True,
            },
            "time_windows": [(0.0, 1.0), (0.5, 1.5)],
            "windowed_evaluations": {
                "zero_crossing_rate": [0.08, 0.09],
                "spectral_centroid": [2100.0, 2300.0],
                "is_clipped": [True, True],
            },
        },
    ]


@pytest.fixture
def sample_results_no_windowing() -> List[Dict[str, Any]]:
    """Sample results without windowing for testing."""
    return [
        {
            "id": "audio1",
            "path": "/path/to/audio1.wav",
            "activity": "speech",
            "evaluations": {
                "zero_crossing_rate": 0.123,
                "spectral_centroid": 1500.0,
            },
            "time_windows": None,
            "windowed_evaluations": None,
        }
    ]


def test_flatten_results_to_dataframe(sample_results: List[Dict[str, Any]]) -> None:
    """Test flattening of evaluation results to DataFrame."""
    df = flatten_results_to_dataframe(sample_results)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2

    # Check required columns exist
    expected_cols = ["id", "path", "activity", "zero_crossing_rate", "spectral_centroid", "is_clipped"]
    for col in expected_cols:
        assert col in df.columns

    # Check values
    assert df.loc[0, "id"] == "audio1"
    assert df.loc[0, "activity"] == "speech"
    assert df.loc[0, "zero_crossing_rate"] == 0.123
    assert df.loc[0, "spectral_centroid"] == 1500.0
    assert not df.loc[0, "is_clipped"]

    assert df.loc[1, "id"] == "audio2"
    assert df.loc[1, "activity"] == "music"
    assert df.loc[1, "is_clipped"]


def test_flatten_results_empty_evaluations() -> None:
    """Test flattening with empty evaluations."""
    results = [
        {
            "id": "audio1",
            "path": "/path/to/audio1.wav",
            "activity": "speech",
            "evaluations": {},
        }
    ]

    df = flatten_results_to_dataframe(results)
    assert len(df) == 1
    assert list(df.columns) == ["id", "path", "activity"]


def test_flatten_results_missing_evaluations() -> None:
    """Test flattening with missing evaluations key."""
    results = [
        {
            "id": "audio1",
            "path": "/path/to/audio1.wav",
            "activity": "speech",
        }
    ]

    df = flatten_results_to_dataframe(results)
    assert len(df) == 1
    assert list(df.columns) == ["id", "path", "activity"]


def test_extract_windowed_data(sample_results: List[Dict[str, Any]]) -> None:
    """Test extraction of windowed evaluation data."""
    df = extract_windowed_data(sample_results)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    expected_cols = ["id", "evaluation_name", "window_start", "window_end", "value"]
    assert list(df.columns) == expected_cols

    # Check total number of rows (3 windows + 2 windows) * 3 evaluations each
    assert len(df) == (3 + 2) * 3  # 15 rows total

    # Check some specific values
    audio1_zcr = df[(df["id"] == "audio1") & (df["evaluation_name"] == "zero_crossing_rate")]
    assert len(audio1_zcr) == 3
    assert audio1_zcr.iloc[0]["value"] == 0.1
    assert audio1_zcr.iloc[0]["window_start"] == 0.0
    assert audio1_zcr.iloc[0]["window_end"] == 1.0

    # Check boolean values are preserved
    audio1_clipped = df[(df["id"] == "audio1") & (df["evaluation_name"] == "is_clipped")]
    assert audio1_clipped.iloc[2]["value"]


def test_extract_windowed_data_no_windowing(sample_results_no_windowing: List[Dict[str, Any]]) -> None:
    """Test extraction with no windowed data."""
    df = extract_windowed_data(sample_results_no_windowing)

    # Should return empty DataFrame
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_extract_windowed_data_missing_time_windows() -> None:
    """Test extraction with missing time_windows."""
    results = [
        {
            "id": "audio1",
            "windowed_evaluations": {
                "zero_crossing_rate": [0.1, 0.2],
            },
            # Missing time_windows
        }
    ]

    df = extract_windowed_data(results)
    assert len(df) == 0


def test_save_formatted_results(sample_results: List[Dict[str, Any]]) -> None:
    """Test saving results in multiple formats."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Test with windowing
        result_dfs = save_formatted_results(sample_results, output_dir)

        # Check returned DataFrames
        assert "summary" in result_dfs
        assert "windowed" in result_dfs
        assert isinstance(result_dfs["summary"], pd.DataFrame)
        assert isinstance(result_dfs["windowed"], pd.DataFrame)

        # Check files were created
        assert (output_dir / "results_summary.csv").exists()
        assert (output_dir / "results_windowed.csv").exists()
        assert (output_dir / "results_full.json").exists()

        # Check summary CSV content
        summary_df = pd.read_csv(output_dir / "results_summary.csv")
        assert len(summary_df) == 2
        assert "zero_crossing_rate" in summary_df.columns

        # Check windowed CSV content
        windowed_df = pd.read_csv(output_dir / "results_windowed.csv")
        assert len(windowed_df) == 15  # (3+2) windows * 3 evaluations
        assert "evaluation_name" in windowed_df.columns

        # Check JSON content
        with open(output_dir / "results_full.json") as f:
            json_data = json.load(f)
        assert len(json_data) == 2
        assert json_data[0]["id"] == "audio1"


def test_save_formatted_results_skip_windowing(sample_results: List[Dict[str, Any]]) -> None:
    """Test saving results with windowing skipped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        result_dfs = save_formatted_results(sample_results, output_dir, skip_windowing=True)

        # Check returned DataFrames
        assert "summary" in result_dfs
        assert "windowed" not in result_dfs

        # Check files
        assert (output_dir / "results_summary.csv").exists()
        assert not (output_dir / "results_windowed.csv").exists()
        assert (output_dir / "results_full.json").exists()


def test_save_formatted_results_no_windowed_data(sample_results_no_windowing: List[Dict[str, Any]]) -> None:
    """Test saving results with no windowed data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        result_dfs = save_formatted_results(sample_results_no_windowing, output_dir)

        # Should not create windowed file if no windowed data
        assert "summary" in result_dfs
        assert "windowed" not in result_dfs
        assert (output_dir / "results_summary.csv").exists()
        assert not (output_dir / "results_windowed.csv").exists()


def test_save_formatted_results_creates_directory() -> None:
    """Test that output directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "subdir" / "results"

        # Directory doesn't exist yet
        assert not output_dir.exists()

        save_formatted_results([], output_dir)

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()
