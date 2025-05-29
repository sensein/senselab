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


def get_evaluation(
    audio: Audio,
    evaluation_function: Callable[[Audio], float | bool],
    id: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Applies a single evaluation function to an Audio instance and caches the result in a DataFrame.

    Args:
        audio (Audio): The Audio object to evaluate.
        evaluation_function (Callable): A function that computes a metric or check from Audio.
        id (str): The unique ID for the current row in the DataFrame.
        df (pd.DataFrame): A DataFrame used to accumulate evaluation results.

    Returns:
        pd.DataFrame: The updated DataFrame with the new evaluation result.
    """
    evaluation_name = evaluation_function.__name__

    if evaluation_name in df.columns and not pd.isna(df.loc[df["id"] == id, evaluation_name]).all():
        return df  # Already computed

    if evaluation_name not in df.columns:
        df[evaluation_name] = None

    df.loc[df["id"] == id, evaluation_name] = evaluation_function(audio)
    return df


def evaluate_audio(
    audio_path: Union[str, Path],
    save_path: Union[str, Path],
    activity: str,
    evaluations: List[Callable[[Audio], float | bool]],
) -> None:
    """Evaluates a single audio file using the given set of functions and saves results as CSV.

    Args:
        audio_path (Union[str, Path]): Path to the audio file.
        save_path (Union[str, Path]): Directory to write the output CSV.
        activity (str): Activity label associated with the audio file.
        evaluations (List[Callable[[Audio], float | bool]]): List of evaluation functions to apply.

    Returns:
        None
    """
    id = Path(audio_path).stem
    output_file = save_path / Path(f"{id}.csv")
    if output_file.exists():
        return

    audio = Audio(filepath=audio_path)
    df = pd.DataFrame([{"id": id, "path": audio_path, "activity": activity}])
    for evaluation in evaluations:
        df = get_evaluation(audio=audio, evaluation_function=evaluation, df=df, id=id)
    df.to_csv(output_file, index=False)


def run_evaluations(
    audio_path_to_activity: Dict[str, str],
    activity_to_evaluations: Dict[str, List[Callable[[Audio], float | bool]]],
    batch_size: int,
    n_cores: int,
    plugin: str = "cf",
) -> pd.DataFrame:
    """Runs quality evaluations on a set of audio files in parallel batches using Pydra.

    Args:
        audio_path_to_activity (Dict[str, str]): Maps audio paths to activity labels.
        activity_to_evaluations (Dict[str, List[Callable]]): Maps activity labels to evaluation functions.
        batch_size (int): Number of files to process in a batch.
        n_cores (int): Number of parallel processes to use.
        plugin (str, optional): Pydra execution plugin ("cf" for concurrent.futures). Defaults to "cf".

    Returns:
        pd.DataFrame: A DataFrame with evaluation results for all audio files.
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

    def process_batch_task_closure(
        activity_to_evaluations: Dict[str, List[Callable[[Audio], float | bool]]],
        audio_path_to_activity: Dict[str, str],
    ) -> Callable[[], Any]:
        @pydra.mark.task
        def process_batch_task(batch_audio_paths: List[str]) -> List[Dict[str, Any]]:
            records = []
            for audio_path in batch_audio_paths:
                activity = audio_path_to_activity[audio_path]
                evaluations = activity_to_evaluations[activity]
                audio = Audio(filepath=audio_path)
                row: Dict[str, Any] = {"id": Path(audio_path).stem, "path": audio_path, "activity": activity}
                for fn in evaluations:
                    row[fn.__name__] = fn(audio)
                records.append(row)
            return records

        return process_batch_task

    task = process_batch_task_closure(activity_to_evaluations, audio_path_to_activity)()
    task.split("batch_audio_paths", batch_audio_paths=batches)

    with Submitter(plugin=plugin, **plugin_args) as sub:
        sub(task)

    # Concatenate batch DataFrames into one
    results = [record for r in task.result() for record in r.output.out]
    return pd.DataFrame(results)


def check_quality(
    audio_paths: List[Union[str, os.PathLike]],
    audio_path_to_activity: Dict = {},
    activity_tree: Dict = BIOACOUSTIC_ACTIVITY_TAXONOMY,
    save_dir: Union[str, os.PathLike, None] = None,
    batch_size: int = 8,
    n_cores: int = 4,
) -> pd.DataFrame:
    """Runs audio quality control evaluations across multiple audio files.

    Args:
        audio_paths (List[Union[str, os.PathLike]]): List of paths to audio files.
        audio_path_to_activity (Dict, optional): Maps audio paths to activity labels. Defaults to {}.
        activity_tree (Dict, optional): The full taxonomy tree. Defaults to BIOACOUSTIC_ACTIVITY_TAXONOMY.
        save_dir (Union[str, os.PathLike, None], optional): Directory containing results.
        batch_size (int, optional): Number of files per batch. Defaults to 8.
        n_cores (int, optional): Number of CPU cores to use. Defaults to 4.

    Returns:
        pd.DataFrame: A DataFrame containing all evaluation results from saved CSVs.
    """
    # get the paths to activity dict
    audio_path_to_activity = {path: audio_path_to_activity.get(path, "bioacoustic") for path in audio_paths}

    # create activity to evaluations dict
    activity_to_evaluations = create_activity_to_evaluations(
        audio_path_to_activity=audio_path_to_activity, activity_tree=activity_tree
    )

    # run_evaluations
    run_evaluations(audio_path_to_activity, activity_to_evaluations, batch_size=batch_size, n_cores=n_cores)

    # construct evaluations csv
    if save_dir is None:
        raise ValueError("save_dir must be provided to collect evaluation CSVs.")
    csv_paths = Path(save_dir).glob("*.csv")

    evaluations_df = pd.concat((pd.read_csv(p) for p in csv_paths), axis=0, ignore_index=True, sort=False)

    return evaluations_df

    # label include, exclude, review

    # create final metadata files
