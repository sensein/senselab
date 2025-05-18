"""Runs bioacoustic activity recording quality control on a set of Audio objects."""

import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from joblib import Parallel, delayed
from pydra.engine.core import Workflow  # Assuming you're using Pydra workflows

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_ACTIVITY_TAXONOMY


def apply_audio_quality_function(
    df: pd.DataFrame, activity_audios: List[Audio], function: Callable[[Audio], bool]
) -> pd.DataFrame:
    """Applies a function to each audio and stores results in a new column with the function name.

    Args:
        df (pd.DataFrame): DataFrame containing audio metadata with an 'audio_path_or_id' column.
        activity_audios (List[Audio]): List of Audio objects to check.
        function (Callable[[Audio], bool]): Function that evaluates an Audio object and returns a bool.

    Returns:
        pd.DataFrame: Updated DataFrame with a new column for the check results.
    """
    column_name = function.__name__
    audio_dict = {audio.orig_path_or_id: function(audio) for audio in activity_audios}
    df[column_name] = df["audio_path_or_id"].map(audio_dict)
    return df


def activity_to_taxonomy_tree_path(activity: str) -> List[str]:
    """Gets the taxonomy tree path for a given activity .

    Args:
        activity (str): The activity name to find in the taxonomy tree.

    Returns:
        List[str]: A list representing the path from the root to the given activity .

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


def subtree_to_evaluations(subtree: Dict) -> List[Callable[[Audio], bool]]:
    """Recursively extracts all evaluation functions (metrics and checks) from a taxonomy subtree.

    Args:
        subtree (Dict): A pruned or full subtree of the taxonomy.

    Returns:
        List[Callable[[Audio], bool]]: Ordered list of unique evaluation functions to run.
    """
    evaluations = []

    def collect_evaluations(node: Dict):
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
        activity_name (str): Activity name to retain.
        activity_tree (Dict): Full taxonomy tree defining the activity hierarchy.

    Returns:
        Dict: Pruned taxonomy tree with only relevant branches for the given activity.

    Raises:
        ValueError: If the activity does not exist in the taxonomy.
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


# def evaluate_node(
#     audios: List[Audio], activity_audios: List[Audio], tree: Dict[str, Any], results_df: pd.DataFrame
# ) -> pd.DataFrame:
#     """Runs quality checks on a given taxonomy tree node and updates the tree with results.

#     Args:
#         audios (List[Audio]): The full list of audio files, which may be modified if files are excluded.
#         activity_audios (List[Audio]): The subset of `audios` relevant to the current taxonomy node.
#         tree (Dict[str, Any]): The taxonomy tree node containing a "checks" key with check functions.
#         results_df (pd.DataFrame): DataFrame to store results of the quality checks.

#     Returns:
#         pd.DataFrame: Updated results DataFrame.
#     """
#     # Calculate metrics
#     metrics = tree.get("metrics")
#     if isinstance(metrics, list):
#         for metric in metrics:
#             results_df = apply_audio_quality_function(results_df, activity_audios, function=metric)

#     # Evaluate checks
#     checks = tree.get("checks")
#     if isinstance(checks, list):
#         for check in checks:
#             results_df = apply_audio_quality_function(results_df, activity_audios, function=check)

#     return results_df


# def evaluate_audio():
#     """Runs evaluations for one audio. Saves to file. Don't run if the audio files features file already exists.

#     Run efficiently without duplicating metrics or checks.

#     in:
#         activity2evaluations dict
#         audio_path
#         save_path
#     """


def run_evaluations(audio_path_to_activity: Dict, activity_to_evaluations: Dict, save_path, n_batches: int):
    """Constructs a Pydra workflow for running evaluations. Batches audio files. Splits over n_batches."""

    # use activity2evaluations dict
    # split across audio paths
    # run evaluate_audio
    # collect all the outputs

    # in check_quality, batch audio files
    # run separate workflow for each batch
    # at the end of each batch, update CSV with all evaluations

    # make crucial checks that automatically exclude audio if not passed
    # Run these first. Don't run anything else after if these fail.
    # Make them customizable.
    # Then run review checks. If any of these don't pass, run all other checks, but label review at the end.

    pass


def create_activity_to_evaluations(audio_path_to_activity: Dict[str, str], activity_tree: Dict) -> Dict[str, List[str]]:
    """Generates a mapping from each activity to the list of evaluation functions (metrics and checks)
    that should be applied, based on the activity's corresponding subtree in the taxonomy.

    Args:
        audio_path_to_activity (Dict[str, str]): Mapping of audio file paths to activity labels.
        activity_tree (Dict): Full taxonomy tree defining activity hierarchies and associated evaluations.

    Returns:
        Dict[str, List[str]]: Mapping from activity names to ordered lists of evaluation functions.
    """
    unique_activities = set(audio_path_to_activity.values())
    activity_to_evaluations = {}
    for activity in unique_activities:
        subtree = activity_to_dataset_taxonomy_subtree(activity, activity_tree)
        print(subtree)
        evaluations = subtree_to_evaluations(subtree)
        print(evaluations)
        activity_to_evaluations[activity] = evaluations
    return activity_to_evaluations


# def create_audio_path_to_activity(
#     audio_paths: List[str], audio_path_to_activity: Optional[Dict[str, str]] = None
# ) -> Dict[str, str]:
#     """Maps each audio path to an activity, defaulting to 'bioacoustic' if not provided.

#     Args:
#         audio_paths (List[str]): List of audio file paths.
#         path_to_activity (Optional[Dict[str, str]]): Optional mapping of paths to activity names.

#     Returns:
#         Dict[str, str]: Dictionary mapping each audio path to its activity.
#     """
#     return {}


def check_quality(
    audio_paths: Union[str, os.PathLike],
    audio_path_to_activity: Dict = {},
    activity_tree: Dict = BIOACOUSTIC_ACTIVITY_TAXONOMY,
    save_path: Union[str, os.PathLike, None] = None,
    n_batches: int = 1,
) -> pd.DataFrame:
    """Runs quality checks on audio files in n_batches and updates the taxonomy tree."""
    # get the paths to activity dict
    audio_path_to_activity = {path: audio_path_to_activity.get(path, "bioacoustic") for path in audio_paths}

    # create activity to evaluations dict
    activity_to_evaluations = create_activity_to_evaluations(
        audio_path_to_activity=audio_path_to_activity, activity_tree=activity_tree
    )
    print(activity_to_evaluations)

    # run_evaluations
    run_evaluations(audio_path_to_activity, activity_to_evaluations, save_path, n_batches)

    # construct evaluations csv

    # label include, exclude, review

    # run workflow
    # create final metadata files

    # audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
    # audio_paths = [
    #     os.path.join(root, fname)
    #     for root, _, files in os.walk(str(audio_dir))
    #     for fname in files
    #     if fname.lower().endswith(audio_extensions)
    # ]

    # print("Audio paths loaded.")

    # total_n_batches = (len(audio_paths) + batch_size - 1) // batch_size
    # print(f"{total_n_batches} n_batches of size {batch_size}")

    # def process_batch(batch_paths: List[str], batch_idx: int) -> pd.DataFrame:
    #     batch_audios = [Audio.from_filepath(p) for p in batch_paths]
    #     activity_dict = audios_to_activity_dict(batch_audios)
    #     dataset_tree = activity_dict_to_dataset_taxonomy_subtree(activity_dict, activity_tree=activity_tree)

    #     results_df = pd.DataFrame([a.orig_path_or_id for a in batch_audios], columns=["audio_path_or_id"])
    #     results_df = run_taxonomy_subtree_checks_recursively(
    #         audios=batch_audios,
    #         dataset_tree=dataset_tree,
    #         activity_dict=activity_dict,
    #         results_df=results_df,
    #     )
    #     del batch_audios
    #     return results_df

    # n_batches = [audio_paths[i : i + batch_size] for i in range(0, len(audio_paths), batch_size)]
    # batch_aqm_dataframes = Parallel(n_jobs=n_jobs, verbose=verbosity)(
    #     delayed(process_batch)(batch, idx) for idx, batch in enumerate(n_batches)
    # )

    # all_aqms = pd.concat(batch_aqm_dataframes, ignore_index=True)
    # all_aqms["audio_path_or_id"] = all_aqms["audio_path_or_id"].apply(os.path.basename)
    # if save_path:
    #     all_aqms.to_csv(save_path)
    #     print(f"Results saved to: {save_path}")
    # return all_aqms
