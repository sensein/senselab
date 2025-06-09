"""Evaluation utilities for bioacoustic quality control."""

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import pydra
from pydra import Submitter

from senselab.audio.data_structures import Audio

# Type aliases for improved readability
EvalResult = Union[float, bool, str]
EvalFunc = Callable[[Audio], EvalResult]
WindowedResult = Dict[str, List[Union[EvalResult, float]]]


def get_evaluation(
    audio: Audio,
    evaluation_function: EvalFunc,
    existing_results: Optional[Dict[str, Any]] = None,
    is_window: bool = False,
    window_idx: Optional[int] = None,
) -> Optional[EvalResult]:
    """Return evaluation result for an Audio object.

    Args:
        audio: An Audio instance to evaluate
        evaluation_function: The evaluation function to run. Can be either:
            * A metric function returning a float (e.g. zero_crossing_rate)
            * A check function returning a bool (e.g. very_low_headroom)
            * A string-based evaluation (e.g. quality_category)
        existing_results: Optional dictionary of pre-computed results
        is_window: If True, look for cached results in windowed_metrics
        window_idx: Index of the window being evaluated, used to find cached
            results at specific timestamps. Only used if is_window is True.

    Returns:
        The evaluation result for the audio, or None if evaluation fails
    """
    function_name = evaluation_function.__name__

    # Check if result exists in cache
    if existing_results:
        if is_window and window_idx is not None:
            # For windows, look in windowed_metrics at specific index
            has_windowed = (
                "windowed_metrics" in existing_results and function_name in existing_results["windowed_metrics"]
            )
            if has_windowed:
                windowed_result = existing_results["windowed_metrics"][function_name]
                has_valid_window = (
                    "values" in windowed_result
                    and "timestamps" in windowed_result
                    and len(windowed_result["values"]) > window_idx
                )
                if has_valid_window:
                    return windowed_result["values"][window_idx]
        else:
            # For non-windows, look in metrics
            has_metrics = "metrics" in existing_results and function_name in existing_results["metrics"]
            if has_metrics:
                return existing_results["metrics"][function_name]

    # Compute result
    try:
        result = evaluation_function(audio)
        return result
    except Exception as e:
        print(f"Warning: Failed to compute {function_name}: {e}")
        return None


def get_windowed_evaluation(
    audio: Audio,
    evaluation_function: EvalFunc,
    window_size_sec: float,
    step_size_sec: float,
    existing_results: Optional[Dict[str, Any]] = None,
) -> Optional[List[EvalResult]]:
    """Compute windowed evaluation results.

    Applies get_evaluation to each window of the audio file.

    Args:
        audio: An Audio instance to evaluate
        evaluation_function: The evaluation function to run on each window
        window_size_sec: Window size in seconds
        step_size_sec: Step size in seconds between windows
        existing_results: Optional dictionary of pre-computed results

    Returns:
        List of evaluation results for each window, or None if evaluation fails
    """
    window_size = int(window_size_sec * audio.sampling_rate)
    step_size = int(step_size_sec * audio.sampling_rate)

    values: List[EvalResult] = []

    try:
        # Generate windows and compute evaluations
        windows = audio.window_generator(window_size, step_size)
        for window_idx, window in enumerate(windows):
            result = get_evaluation(
                window, evaluation_function, existing_results, is_window=True, window_idx=window_idx
            )
            if result is None:
                return None
            values.append(result)

        return values
    except Exception as e:
        function_name = evaluation_function.__name__
        msg = f"Warning: Failed to compute windowed evaluation for '{function_name}': {e}"
        print(msg)
        return None


def evaluate_audio(
    audio_path: str,
    activity: str,
    evaluations: List[Callable[[Audio], Union[float, bool, str]]],
    output_dir: Optional[Path] = None,
    window_size_sec: float = 1.0,
    step_size_sec: float = 0.5,
    skip_windowing: bool = False,
) -> Dict[str, Any]:
    """Evaluates a single audio file using the given set of functions.

    Args:
        audio_path: Path to the audio file
        activity: Activity label associated with the audio file
        evaluations: List of evaluation functions to apply
        output_dir: Optional directory to load/save results from/to
        window_size_sec: Window size in seconds for windowed calculation (default: 1.0)
        step_size_sec: Step size in seconds between windows (default: 0.5)
        skip_windowing: If True, only compute scalar metrics without windowing

    Returns:
        Dict[str, Any]: Dictionary containing evaluation results
    """
    audio_id = Path(audio_path).stem
    record: Dict[str, Any] = {
        "id": audio_id,
        "path": str(audio_path),
        "activity": activity,
        "metrics": {},
        "windowed_metrics": {} if not skip_windowing else None,
        "window_timestamps": None,
    }

    # Try to load existing results from file if output_dir is provided
    existing_results = None
    if output_dir is not None:
        results_dir = output_dir / "audio_results"
        result_path = results_dir / f"{audio_id}.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    existing_results = json.load(f)
                    if existing_results:
                        record.update(existing_results)
            except Exception as e:
                msg = f"Warning: Could not read existing results for {audio_id}: {e}"
                print(msg)

    try:
        # Load audio only if we have evaluations to compute
        audio = Audio(filepath=audio_path)

        # Apply each missing evaluation function
        for fn in evaluations:
            # Get scalar result
            scalar_result = get_evaluation(audio, fn, existing_results)
            record["metrics"][fn.__name__] = scalar_result

            # Get windowed results unless explicitly skipped
            if not skip_windowing:
                windowed_result = get_windowed_evaluation(audio, fn, window_size_sec, step_size_sec, existing_results)
                if windowed_result is not None:
                    record["windowed_metrics"][fn.__name__] = windowed_result

                # Calculate timestamps if not already done
                if record["window_timestamps"] is None:
                    num_windows = len(windowed_result)
                    record["window_timestamps"] = [i * step_size_sec for i in range(num_windows)]

        # Save results if output directory is provided
        if output_dir is not None:
            results_dir = output_dir / "audio_results"
            results_dir.mkdir(exist_ok=True, parents=True)
            result_path = results_dir / f"{audio_id}.json"

            # Only save if we computed new results
            if not existing_results or record != existing_results:
                with open(result_path, "w") as f:
                    json.dump(record, f, indent=2)

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
