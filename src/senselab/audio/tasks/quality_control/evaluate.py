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
    evaluation_function: Callable[[Audio], Union[float, bool, str]],
    existing_results: Optional[Dict[str, Any]] = None,
) -> Union[float, bool, str, None]:
    """Return an evaluation result, using cached results when possible.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        evaluation_function: The evaluation function to run. Can be either:
            * A metric function returning a float (e.g. ``zero_crossing_rate_metric``)
            * A check function returning a bool (e.g. ``very_low_headroom_check``)
            * A string-based evaluation (e.g. ``quality_category``)
        existing_results: Optional dictionary of pre-computed results, mapping
            function names to their results.

    Returns:
        The evaluation result for this ``audio`` item. If the result exists in
        existing_results, that cached value is returned; otherwise the result
        is computed directly.

    Raises:
        ValueError: If evaluation fails and a proper fallback value cannot be determined.
    """
    function_name = evaluation_function.__name__

    # Check if result exists in cache
    if existing_results and function_name in existing_results:
        return existing_results[function_name]

    # Compute new result
    if isinstance(audio_or_path, str):
        audio = Audio(filepath=audio_or_path)
    else:
        audio = audio_or_path

    result: Union[float, bool, str, None] = None
    try:
        result = evaluation_function(audio)
    except Exception as e:
        print(f"Warning: Failed to compute {function_name}: {e}")
    return result


def evaluate_audio(
    audio_path: str,
    activity: str,
    evaluations: List[Callable[[Audio], Union[float, bool, str]]],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluates a single audio file using the given set of functions.

    Args:
        audio_path: Path to the audio file
        activity: Activity label associated with the audio file
        evaluations: List of evaluation functions to apply
        output_dir: Optional directory to load/save results from/to

    Returns:
        Dict[str, Any]: Dictionary containing evaluation results
    """
    audio_id = Path(audio_path).stem
    record: Dict[str, Any] = {
        "id": audio_id,
        "path": str(audio_path),
        "activity": activity,
    }

    # Try to load existing results from file if output_dir is provided
    existing_results = None
    if output_dir is not None:
        results_dir = output_dir / "audio_results"
        result_path = results_dir / f"{audio_id}.parquet"
        if result_path.exists():
            try:
                existing_df = pd.read_parquet(result_path)
                if not existing_df.empty:
                    existing_results = existing_df.iloc[0].to_dict()
            except Exception as e:
                print(f"Warning: Could not read existing results for {audio_id}: {e}")

    # Start with existing results if available
    if existing_results:
        record.update(existing_results)

    try:
        # Load audio only if we have evaluations to compute
        audio = Audio(filepath=audio_path)

        # Apply each missing evaluation function
        for fn in evaluations:
            result = get_evaluation(audio, fn, existing_results)
            record[fn.__name__] = result

        # Save results if output directory is provided
        if output_dir is not None:
            results_dir = output_dir / "audio_results"
            results_dir.mkdir(exist_ok=True, parents=True)
            result_path = results_dir / f"{audio_id}.parquet"

            # Only save if we computed new results
            if not existing_results or record != existing_results:
                pd.DataFrame([record]).to_parquet(result_path)

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
    records = []
    for audio_path in batch_audio_paths:
        # Get evaluations for this activity
        activity = audio_path_to_activity[str(audio_path)]
        evaluations = activity_to_evaluations[activity]

        # Evaluate audio with result loading/saving handled internally
        record = evaluate_audio(
            audio_path=str(audio_path), activity=activity, evaluations=evaluations, output_dir=output_dir
        )
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
