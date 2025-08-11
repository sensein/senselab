"""Runs bioacoustic activity recording quality control on audio files."""

import os
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Union

import pandas as pd

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control.constants import BIOACOUSTIC_ACTIVITY_TAXONOMY
from senselab.audio.tasks.quality_control.evaluate import evaluate_dataset


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


def subtree_to_evaluations(subtree: Dict) -> Sequence[Callable[[Audio], Union[float, bool, str]]]:
    """Recursively gets all evaluation functions (metrics and checks) from a taxonomy subtree.

    Args:
        subtree (Dict): A subtree of the full taxonomy, either pruned or complete.

    Returns:
        Sequence[Callable[[Audio], Union[float, bool, str]]]: An ordered list of evaluation functions to run.
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
    """Constructs a pruned taxonomy tree for the specified activity.

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


def activity_to_evaluations(
    audio_path_to_activity: Dict[str, str], activity_tree: Dict
) -> Dict[str, Sequence[Callable[[Audio], Union[float, bool, str]]]]:
    """Maps each activity label to its associated evaluation functions.

    Args:
        audio_path_to_activity (Dict[str, str]): Maps audio file paths to activity labels.
        activity_tree (Dict): The full taxonomy tree of activities.

    Returns:
        Dict[str, Sequence[Callable[[Audio], Union[float, bool, str]]]]: Maps activity label to evaluations.
    """
    unique_activities = set(audio_path_to_activity.values())
    activity_to_evaluations = {}
    for activity in unique_activities:
        subtree = activity_to_dataset_taxonomy_subtree(activity, activity_tree)
        evaluations = subtree_to_evaluations(subtree)
        activity_to_evaluations[activity] = evaluations
    return activity_to_evaluations


def check_quality(
    audio_paths: List[Union[str, os.PathLike]],
    audio_path_to_activity: Dict = {},
    activity_tree: Dict = BIOACOUSTIC_ACTIVITY_TAXONOMY,
    output_dir: Optional[Union[str, os.PathLike]] = None,
    batch_size: int = 8,
    n_cores: int = 4,
    window_size_sec: float = 0.025,
    step_size_sec: float = 0.0125,
    skip_windowing: bool = False,
) -> pd.DataFrame:
    """Runs audio quality control evaluations across multiple audio files.

    The results are automatically saved in multiple formats:
    - results_summary.csv: Flattened CSV with evaluations as columns (main output)
    - results_windowed.csv: Normalized windowed data for temporal analysis
    - results_full.json: Complete nested data with full fidelity

    Args:
        audio_paths: List of paths to audio files
        audio_path_to_activity: Maps audio paths to activity labels
        activity_tree: The full taxonomy tree
        output_dir: Directory to save results
        batch_size: Number of files to process in parallel batches
        n_cores: Number of CPU cores to use for parallel processing
        window_size_sec: Window size in seconds for windowed calculation
                        (default: 0.025)
        step_size_sec: Step size in seconds between windows (default: 0.0125)
        skip_windowing: If True, only compute scalar metrics without windowing

    Returns:
        pd.DataFrame: Flattened DataFrame with evaluations as columns (human-readable)
    """
    # Convert output_dir to Path with default
    output_directory = Path(output_dir or "qc_results")
    output_directory.mkdir(exist_ok=True, parents=True)

    # Setup activity mappings
    audio_path_to_activity = {str(path): audio_path_to_activity.get(str(path), "bioacoustic") for path in audio_paths}

    # Create activity to evaluations mapping
    activity_evaluations_dict = activity_to_evaluations(
        audio_path_to_activity=audio_path_to_activity, activity_tree=activity_tree
    )

    # Run evaluations with result caching and formatted output
    evaluations_df = evaluate_dataset(
        audio_path_to_activity,
        activity_evaluations_dict,
        output_dir=output_directory,
        batch_size=batch_size,
        n_cores=n_cores,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        skip_windowing=skip_windowing,
    )

    return evaluations_df
