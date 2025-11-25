"""Evaluation utilities for quality control."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
from joblib import Parallel, delayed

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control.format import save_formatted_results
from senselab.utils.data_structures.logging import logger

# Type aliases for improved readability
EvalResult = Union[float, bool, str]
EvalFunc = Callable[[Audio], EvalResult]
EvalFuncSequence = Sequence[EvalFunc]
WindowedResult = Dict[str, List[Union[EvalResult, float]]]
ActivityEvalMap = Dict[str, EvalFuncSequence]


def get_evaluation(
    audio: Union[Audio, str],
    evaluation_function: EvalFunc,
    existing_results: Optional[Dict[str, Any]] = None,
    is_window: bool = False,
    window_idx: Optional[int] = None,
) -> Optional[EvalResult]:
    """Return evaluation result for an Audio object.

    Args:
        audio: An Audio instance or filepath to the audio file to evaluate
        evaluation_function: The evaluation function to run. Can be either:
            * A metric function returning a float (e.g. zero_crossing_rate)
            * A check function returning a bool (e.g. very_low_headroom)
            * A string-based evaluation (e.g. quality_category)
        existing_results: Optional dictionary of pre-computed results
        is_window: If True, look for existing results in windowed_evaluations
        window_idx: Index of the window being evaluated, used to find existing
            results at specific timestamps. Only used if is_window is True.

    Returns:
        The evaluation result for the audio, or None if:
            * Loading the audio file fails (when audio is a string filepath)
            * The evaluation function fails to compute a result
    """
    # Convert string path to Audio object if needed
    if isinstance(audio, str):
        try:
            audio = Audio(filepath=audio)
        except Exception as e:
            logger.warning(f"Failed to load audio from {audio}: {e}")
            return None

    function_name = evaluation_function.__name__

    # Check if result is in existing_results
    if existing_results:
        if is_window and window_idx is not None:
            # For windows, look in windowed_evaluations at specific index
            has_windowed = (
                "windowed_evaluations" in existing_results and function_name in existing_results["windowed_evaluations"]
            )
            if has_windowed:
                windowed_result = existing_results["windowed_evaluations"][function_name]
                if isinstance(windowed_result, list) and len(windowed_result) > window_idx:
                    return windowed_result[window_idx]
        else:
            # For non-windows, look in evaluations
            has_evaluations = "evaluations" in existing_results and function_name in existing_results["evaluations"]
            if has_evaluations:
                return existing_results["evaluations"][function_name]

    # Compute result
    try:
        result = evaluation_function(audio)
        return result
    except Exception as e:
        logger.warning(f"Failed to compute {function_name}: {e}")
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
        logger.warning(f"Failed to compute windowed evaluation for '{function_name}': {e}")
        return None


def evaluate_audio(
    audio_path: str,
    activity: str,
    evaluations: EvalFuncSequence,
    output_dir: Optional[Path] = None,
    window_size_sec: float = 0.02,
    step_size_sec: float = 0.01,
    skip_windowing: bool = False,
) -> Dict[str, Any]:
    """Evaluates a single audio file using the given set of functions.

    Args:
        audio_path: Path to the audio file
        activity: Activity label associated with the audio file
        evaluations: List of evaluation functions to apply
        output_dir: Optional directory to load/save results from/to
        window_size_sec: Window size in seconds for windowed calculation
                        (default: 0.025)
        step_size_sec: Step size in seconds between windows (default: 0.0125)
        skip_windowing: If True, only compute scalar metrics without windowing

    Returns:
        Dict[str, Any]: Dictionary containing evaluation results with
                       time_windows as list of (start_time, end_time) tuples
                       in seconds
    """
    audio_id = Path(audio_path).stem
    record: Dict[str, Any] = {
        "id": audio_id,
        "path": str(audio_path),
        "activity": activity,
        "evaluations": {},
        "time_windows": None,
        "windowed_evaluations": {} if not skip_windowing else None,
    }

    # Load existing results
    existing_results = None
    if output_dir:
        result_path = output_dir / "audio_results" / f"{audio_id}.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    existing_results = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read existing results for {audio_id}: {e}")

    try:
        audio = Audio(filepath=audio_path)

        for fn in evaluations:
            # Scalar and windowed evaluation
            record["evaluations"][fn.__name__] = get_evaluation(audio, fn, existing_results)

            if not skip_windowing:
                windowed_result = get_windowed_evaluation(audio, fn, window_size_sec, step_size_sec, existing_results)
                if windowed_result:
                    if record["windowed_evaluations"] is not None:
                        record["windowed_evaluations"][fn.__name__] = windowed_result
                    # Calculate time windows once
                    if not record["time_windows"]:
                        record["time_windows"] = [
                            (i * step_size_sec, i * step_size_sec + window_size_sec)
                            for i in range(len(windowed_result))
                        ]

        # Save results
        if output_dir and (not existing_results or record != existing_results):
            result_path.parent.mkdir(exist_ok=True, parents=True)
            with open(result_path, "w") as f:
                json.dump(record, f, indent=2)

    except Exception as e:
        logger.error(f"Error processing {audio_id}: {e}")

    return record


def evaluate_batch(
    batch_audio_paths: List[str],
    audio_path_to_activity: Dict[str, str],
    activity_to_evaluations: ActivityEvalMap,
    output_dir: Path,
    window_size_sec: float = 0.025,
    step_size_sec: float = 0.0125,
    skip_windowing: bool = False,
) -> List[Dict[str, Any]]:
    """Process a batch of audio files, saving individual results and avoiding recomputation.

    Args:
        batch_audio_paths: List of audio file paths to process
        audio_path_to_activity: Mapping of audio paths to their activities
        activity_to_evaluations: Mapping of activities to their evaluation functions
        output_dir: Directory to save individual results
        window_size_sec: Window size in seconds for windowed calculation
                        (default: 0.025)
        step_size_sec: Step size in seconds between windows (default: 0.0125)
        skip_windowing: If True, only compute scalar metrics without windowing

    Returns:
        List[Dict[str, Any]]: List of processed records with evaluation results
    """
    records = []
    for audio_path in batch_audio_paths:
        logger.debug(f"Processing {audio_path}")
        # Get evaluations for this activity
        activity = audio_path_to_activity[str(audio_path)]
        evaluations = activity_to_evaluations[activity]

        # Evaluate audio with result loading/saving handled internally
        record = evaluate_audio(
            audio_path=str(audio_path),
            activity=activity,
            evaluations=evaluations,
            output_dir=output_dir,
            window_size_sec=window_size_sec,
            step_size_sec=step_size_sec,
            skip_windowing=skip_windowing,
        )
        records.append(record)

    return records


def evaluate_dataset(
    audio_path_to_activity: Dict[str, str],
    activity_to_evaluations: ActivityEvalMap,
    output_dir: Path,
    batch_size: int = 8,
    n_cores: int = 4,
    backend: str = "loky",
    window_size_sec: float = 0.025,
    step_size_sec: float = 0.0125,
    skip_windowing: bool = False,
    verbose: int = 0,
) -> pd.DataFrame:
    """Runs quality evaluations on audio files in parallel batches using joblib.

    Args:
        audio_path_to_activity: Maps audio paths to activity labels
        activity_to_evaluations: Maps activity labels to evaluation functions
        output_dir: Directory to save results
        batch_size: Number of files to process in a batch
        n_cores: Number of parallel processes to use (-1 for all cores)
        backend: Joblib backend ("loky", "multiprocessing", "threading")
        window_size_sec: Window size in seconds for windowed calculation
                        (default: 0.025)
        step_size_sec: Step size in seconds between windows (default: 0.0125)
        skip_windowing: If True, only compute scalar metrics without windowing
        verbose: Verbosity level for joblib (0=silent, higher=more verbose)

    Returns:
        pd.DataFrame: Flattened DataFrame with evaluations as columns
                     (human-readable)
    """
    audio_paths = list(audio_path_to_activity.keys())

    # Create batches for parallel processing
    batches = [audio_paths[i : i + batch_size] for i in range(0, len(audio_paths), batch_size)]

    logger.info(f"Processing {len(audio_paths)} files in {len(batches)} batches using {n_cores} cores...")

    # Use joblib for parallel processing
    try:
        if n_cores == 1:
            # Serial processing
            results = []
            for batch in batches:
                batch_results = evaluate_batch(
                    batch,
                    audio_path_to_activity,
                    activity_to_evaluations,
                    output_dir,
                    window_size_sec=window_size_sec,
                    step_size_sec=step_size_sec,
                    skip_windowing=skip_windowing,
                )
                results.extend(batch_results)
        else:
            # Parallel processing with joblib
            batch_results_list = Parallel(
                n_jobs=n_cores,
                backend=backend,
                verbose=verbose,
                batch_size=1,  # Process one batch at a time
            )(
                delayed(evaluate_batch)(
                    batch,
                    audio_path_to_activity,
                    activity_to_evaluations,
                    output_dir,
                    window_size_sec=window_size_sec,
                    step_size_sec=step_size_sec,
                    skip_windowing=skip_windowing,
                )
                for batch in batches
            )

            # Flatten results from all batches
            results = [rec for batch_results in batch_results_list for rec in batch_results]

        # Save formatted results
        output_dfs = save_formatted_results(results, output_dir, skip_windowing)
        return output_dfs["summary"]

    except Exception as e:
        logger.error(f"Error in joblib parallel execution: {e}")
        logger.info("Falling back to serial processing...")

        # Fallback to serial processing
        results = []
        for batch in batches:
            batch_results = evaluate_batch(
                batch,
                audio_path_to_activity,
                activity_to_evaluations,
                output_dir,
                window_size_sec=window_size_sec,
                step_size_sec=step_size_sec,
                skip_windowing=skip_windowing,
            )
            results.extend(batch_results)

        output_dfs = save_formatted_results(results, output_dir, skip_windowing)
        return output_dfs["summary"]
