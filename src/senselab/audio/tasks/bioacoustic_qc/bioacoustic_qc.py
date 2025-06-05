"""Runs bioacoustic activity recording quality control on a set of Audio objects."""

import multiprocessing as mp
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import pydra
from pydra import Submitter

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_ACTIVITY_TAXONOMY


def activity_to_taxonomy_tree_path(activity: str) -> List[str]:
    """Gets the taxonomy tree path for a given activity.

    Args:
        activity (str): The activity name to find in the taxonomy tree.

    Returns:
        List[str]: A list representing the path from the root to the given activity.

    Raises:
        ValueError: If the activity is not found in the taxonomy tree.
    """

    def find_activity_path(tree: Dict, path: List[str]) -> Optional[List[str]]:
        """Recursively searches for the activity in the taxonomy tree."""
        for key, value in tree.items():
            new_path = path + [key]  # Extend path
            if key == activity:
                return new_path  # Activity  found, return path
            if isinstance(value, dict) and "subclass" in value and isinstance(value["subclass"], dict):
                result = find_activity_path(value["subclass"], new_path)
                if result:
                    return result
        return None

    path = find_activity_path(BIOACOUSTIC_ACTIVITY_TAXONOMY, [])
    if path is None:
        raise ValueError(f"Activity '{activity}' not found in taxonomy tree.")

    return path


def subtree_to_evaluations(subtree: Dict) -> List[Callable[[Audio], float | bool]]:
    """Recursively extracts all evaluation functions (metrics and checks) from a taxonomy subtree.

    Args:
        subtree (Dict): A subtree of the full taxonomy, either pruned or complete.

    Returns:
        List[Callable[[Audio], float | bool]]: An ordered list of evaluation functions to run.
    """
    evaluations = []

    def collect_evaluations(node: Dict) -> None:
        if not isinstance(node, dict):
            return None
        for key in node:
            for function in node[key].get("metrics", []) + node[key].get("checks", []):
                if function not in evaluations:
                    evaluations.append(function)
            children = node[key].get("subclass")
            if isinstance(children, dict):
                collect_evaluations(children)

    collect_evaluations(subtree)
    return evaluations


def activity_to_dataset_taxonomy_subtree(activity_name: str, activity_tree: Dict) -> Dict:
    """Constructs a pruned taxonomy tree containing only the specified activity.

    Args:
        activity_name (str): The name of the activity to isolate.
        activity_tree (Dict): The full taxonomy tree.

    Returns:
        Dict: A pruned version of the taxonomy containing only the target activity and its path.

    Raises:
        ValueError: If the activity is not found in the taxonomy.
    """
    activity_path = activity_to_taxonomy_tree_path(activity_name)
    valid_nodes: Set[str] = set(activity_path)

    pruned_tree: Dict = deepcopy(activity_tree)

    def prune_tree(subtree: Dict) -> bool:
        keys_to_delete = []
        for key in list(subtree.keys()):
            value = subtree[key]
            if isinstance(value, dict) and "subclass" in value and isinstance(value["subclass"], dict):
                keep_branch = prune_tree(value["subclass"])
                if not value["subclass"]:
                    value["subclass"] = None
                if not keep_branch and key not in valid_nodes:
                    keys_to_delete.append(key)
            elif key not in valid_nodes:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del subtree[key]
        return bool(subtree)

    subclass_tree = pruned_tree["bioacoustic"].get("subclass", None)
    if not prune_tree(subclass_tree):
        pruned_tree["bioacoustic"]["subclass"] = None

    return pruned_tree


def create_activity_to_evaluations(
    audio_path_to_activity: Dict[str, str], activity_tree: Dict
) -> Dict[str, List[Callable[[Audio], float | bool]]]:
    """Maps each activity label to its associated evaluation functions.

    Args:
        audio_path_to_activity (Dict[str, str]): Maps audio file paths to activity labels.
        activity_tree (Dict): The full taxonomy tree of activities.

    Returns:
        Dict[str, List[Callable[[Audio], float | bool]]]: Maps each activity label to a list of evaluation functions.
    """
    unique_activities = set(audio_path_to_activity.values())
    activity_to_evaluations = {}
    for activity in unique_activities:
        subtree = activity_to_dataset_taxonomy_subtree(activity, activity_tree)
        evaluations = subtree_to_evaluations(subtree)
        activity_to_evaluations[activity] = evaluations
    return activity_to_evaluations


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


def run_evaluations(
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


def check_quality(
    audio_paths: List[Union[str, os.PathLike]],
    audio_path_to_activity: Dict = {},
    activity_tree: Dict = BIOACOUSTIC_ACTIVITY_TAXONOMY,
    output_dir: Optional[Union[str, os.PathLike]] = None,
    batch_size: int = 8,
    n_cores: int = 4,
) -> pd.DataFrame:
    """Runs audio quality control evaluations across multiple audio files.

    Args:
        audio_paths: List of paths to audio files
        audio_path_to_activity: Maps audio paths to activity labels
        activity_tree: The full taxonomy tree
        output_dir: Directory to save results
        batch_size: Number of files to process in parallel batches
        n_cores: Number of CPU cores to use for parallel processing

    Returns:
        pd.DataFrame: Combined results from all processed audio files
    """
    # Convert output_dir to Path with default
    output_directory = Path(output_dir or "qc_results")
    output_directory.mkdir(exist_ok=True, parents=True)

    # Setup activity mappings
    audio_path_to_activity = {str(path): audio_path_to_activity.get(str(path), "bioacoustic") for path in audio_paths}

    # Create activity to evaluations mapping
    activity_to_evaluations = create_activity_to_evaluations(
        audio_path_to_activity=audio_path_to_activity, activity_tree=activity_tree
    )

    # Run evaluations with result caching
    evaluations_df = run_evaluations(
        audio_path_to_activity,
        activity_to_evaluations,
        output_dir=output_directory,
        batch_size=batch_size,
        n_cores=n_cores,
    )

    final_results_path = output_directory / "combined_results.csv"
    evaluations_df.to_csv(final_results_path, index=False)

    return evaluations_df

    # label include, exclude, review

    # create final metadata files
