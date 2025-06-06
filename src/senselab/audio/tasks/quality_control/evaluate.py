"""Evaluation utilities for bioacoustic quality control."""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import pydra
from pydra import Submitter

from senselab.audio.data_structures import Audio


def get_evaluation(
    audio_or_path: Union[Audio, str],
    metric_function: Callable[[Audio], float],
    df: Optional[pd.DataFrame] = None,
) -> float:
    """Return a metric, using a cached DataFrame when possible.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        metric_function: The metric function, e.g. ``zero_crossing_rate_metric``.
        df: Optional DataFrame that already contains pre-computed metrics.
            The DataFrame must have:
              * a column ``'audio_path_or_id'`` holding file names, and
              * a column named exactly ``metric_function.__name__``.

    Returns:
        The metric value for this ``audio`` item. If ``df`` is provided and
        contains the value, that cached value is returned; otherwise the metric
        is computed directly and optionally added to ``df``.
    """
    metric_name = metric_function.__name__

    filepath = None
    if isinstance(audio_or_path, str):
        filepath = audio_or_path
    else:
        filepath = audio_or_path.filepath()

    metric = None
    if df is not None and filepath:
        audio_file_name = os.path.basename(filepath)
        row = df[df["audio_path_or_id"] == audio_file_name]
        if not row.empty and metric_name in row.columns:
            metric = row[metric_name].iloc[0]

    if metric is None and isinstance(audio_or_path, Audio):
        metric = metric_function(audio_or_path)
        if df is not None and filepath:
            audio_file_name = os.path.basename(filepath)
            if metric_name not in df.columns:
                df[metric_name] = pd.NA
            df.loc[df["audio_path_or_id"] == audio_file_name, metric_name] = metric

    if metric is None:
        raise ValueError("Expected metric to be non-None.")

    return metric


def evaluate_audio(
    audio_path: str,
    activity: str,
    evaluations: List[Callable[[Audio], Union[float, bool, str]]],
    existing_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluates a single audio file using the given set of functions.

    Args:
        audio_path: Path to the audio file
        activity: Activity label associated with the audio file
        evaluations: List of evaluation functions to apply
        existing_results: Optional dictionary of existing evaluation results

    Returns:
        Dict[str, Any]: Dictionary containing evaluation results
    """
    audio_id = Path(audio_path).stem
    record: Dict[str, Any] = {
        "id": audio_id,
        "path": str(audio_path),
        "activity": activity,
    }

    # Start with existing results if available
    if existing_results:
        record.update(existing_results)

    # Determine which evaluations need to be computed
    missing_evaluations = [fn for fn in evaluations if not existing_results or fn.__name__ not in existing_results]

    if missing_evaluations:
        try:
            # Load audio only if we have evaluations to compute
            audio = Audio(filepath=audio_path)

            # Apply each missing evaluation function
            for fn in missing_evaluations:
                try:
                    result = fn(audio)
                    record[fn.__name__] = result
                except Exception as e:
                    print(f"Warning: Failed to compute {fn.__name__} for {audio_id}: {e}")
                    # Use empty string for failed string metrics, None otherwise
                    record[fn.__name__] = "" if fn.__annotations__.get("return") == str else None

        except Exception as e:
            print(f"Error processing {audio_id}: {e}")

    return record


def evaluate_batch(
    batch_audio_paths: List[str],
    audio_path_to_activity: Dict[str, str],
    activity_to_evaluations: Dict[str, List[Callable[[Audio], Union[float, bool, str]]]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Process a batch of audio files, saving individual results and avoiding recomputation.

    Args:
        batch_audio_paths: List of audio file paths to process
        audio_path_to_activity: Mapping of audio paths to their activities
        activity_to_evaluations: Mapping of activities to their evaluation functions
        output_dir: Directory to save individual results

    Returns:
        List[Dict[str, Any]]: List of processed records with evaluation results
    """
    results_dir = output_dir / "audio_results"
    results_dir.mkdir(exist_ok=True, parents=True)

    records = []
    for audio_path in batch_audio_paths:
        audio_id = Path(audio_path).stem
        result_path = results_dir / f"{audio_id}.parquet"

        # Load existing results if available
        existing_results = None
        if result_path.exists():
            try:
                existing_df = pd.read_parquet(result_path)
                if not existing_df.empty:
                    existing_results = existing_df.iloc[0].to_dict()
            except Exception as e:
                print(f"Warning: Could not read existing results for {audio_id}: {e}")

        # Get evaluations for this activity
        activity = audio_path_to_activity[str(audio_path)]
        evaluations = activity_to_evaluations[activity]

        # Evaluate audio and save results
        record = evaluate_audio(str(audio_path), activity, evaluations, existing_results)

        # Only save if we computed new results
        if not existing_results or record != existing_results:
            pd.DataFrame([record]).to_parquet(result_path)

        records.append(record)

    return records


def evaluate_dataset(
    audio_path_to_activity: Dict[str, str],
    activity_to_evaluations: Dict[str, List[Callable]],
    output_dir: Path,
    batch_size: int = 8,
    n_cores: int = 4,
    plugin: str = "cf",
) -> pd.DataFrame:
    """Runs quality evaluations on audio files in parallel batches using Pydra.

    Args:
        audio_path_to_activity: Maps audio paths to activity labels
        activity_to_evaluations: Maps activity labels to evaluation functions
        output_dir: Directory to save results
        batch_size: Number of files to process in a batch
        n_cores: Number of parallel processes to use
        plugin: Pydra execution plugin ("cf" for concurrent.futures)

    Returns:
        pd.DataFrame: Combined results from all processed batches
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if n_cores > 1:
        plugin_args = {"n_procs": n_cores} if plugin == "cf" else {}
    else:
        plugin = "serial"
        plugin_args = {}

    audio_paths = list(audio_path_to_activity.keys())
    batches = [audio_paths[i : i + batch_size] for i in range(0, len(audio_paths), batch_size)]

    def evaluate_batch_closure() -> Callable:
        """Creates a Pydra task for batch processing.

        Returns:
            Callable: Wrapped task function for batch processing
        """

        @pydra.mark.task
        def evaluate_batch_task(batch_audio_paths: List[str]) -> List[Dict[str, Any]]:
            return evaluate_batch(batch_audio_paths, audio_path_to_activity, activity_to_evaluations, output_dir)

        return evaluate_batch_task

    task = evaluate_batch_closure()()
    task.split("batch_audio_paths", batch_audio_paths=batches)

    with Submitter(plugin=plugin, **plugin_args) as sub:
        sub(task)

    # Concatenate batch results
    results = [record for r in task.result() for record in r.output.out]
    return pd.DataFrame(results)
