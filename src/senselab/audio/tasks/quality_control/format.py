"""Formatting utilities for quality control evaluation results."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from senselab.utils.data_structures.logging import logger


def flatten_non_windowed_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts nested non-windowed evaluation results to a flattened DataFrame.

    Args:
        results: List of result dictionaries from evaluate_audio

    Returns:
        pd.DataFrame: Flattened DataFrame with non-windowed evaluations as columns
    """
    flattened_records = []

    for record in results:
        # Start with basic info
        flat_record = {"id": record["id"], "path": record["path"], "activity": record["activity"]}

        # Flatten evaluations dict into individual columns
        if "evaluations" in record and record["evaluations"]:
            flat_record.update(record["evaluations"])

        flattened_records.append(flat_record)

    return pd.DataFrame(flattened_records)


def flatten_windowed_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flattens windowed evaluation data into normalized DataFrame format.

    Args:
        results: List of result dictionaries from evaluate_audio

    Returns:
        pd.DataFrame: Normalized windowed data with columns:
                     [id, evaluation_name, window_start, window_end, value]
    """
    windowed_records = []

    for record in results:
        audio_id = record["id"]
        time_windows = record.get("time_windows")
        windowed_evals = record.get("windowed_evaluations")

        # Skip if no windowed data
        if not time_windows or not windowed_evals:
            continue

        # Extract each evaluation's windowed values
        for evaluation_name, values in windowed_evals.items():
            if not isinstance(values, list):
                continue

            for i, value in enumerate(values):
                if i < len(time_windows):
                    start_time, end_time = time_windows[i]
                    windowed_records.append(
                        {
                            "id": audio_id,
                            "evaluation_name": evaluation_name,
                            "window_start": start_time,
                            "window_end": end_time,
                            "value": value,
                        }
                    )

    return pd.DataFrame(windowed_records)


def save_formatted_results(
    results: List[Dict[str, Any]], output_dir: Path, skip_windowing: bool = False
) -> Dict[str, pd.DataFrame]:
    """Saves results in multiple readable formats.

    Args:
        results: List of result dictionaries from evaluate_audio
        output_dir: Directory to save formatted results
        skip_windowing: If True, skip windowed data processing

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with DataFrames for
                                each output type
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create flattened results DataFrame
    flattened_df = flatten_non_windowed_results_to_dataframe(results)

    # Save main results CSV
    main_results_path = output_dir / "quality_control_results_non_windowed.csv"
    flattened_df.to_csv(main_results_path, index=False)
    logger.info(f"Saved flattened results to: {main_results_path}")

    output_dfs = {"summary": flattened_df}

    # Save windowed data if available
    if not skip_windowing:
        windowed_df = flatten_windowed_results_to_dataframe(results)
        if not windowed_df.empty:
            windowed_results_path = output_dir / "quality_control_results_windowed.csv"
            windowed_df.to_csv(windowed_results_path, index=False)
            logger.info(f"Saved windowed results to: {windowed_results_path}")
            output_dfs["windowed"] = windowed_df

    # Save full results as JSON for complete fidelity
    full_results_path = output_dir / "quality_control_results_all.json"
    with open(full_results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved full results to: {full_results_path}")

    return output_dfs
