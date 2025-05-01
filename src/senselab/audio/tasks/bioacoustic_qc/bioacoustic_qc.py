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


def activity_dict_to_dataset_taxonomy_subtree(
    audio_paths_to_activities: Dict[str, str],
    activity_tree: Dict
) -> Dict:
    """Constructs a pruned taxonomy tree containing only relevant activities.

    Args:
        audio_paths_to_activities (Dict[str, str]): Maps audio file paths to activity names.
        activity_tree (Dict): Full taxonomy tree defining the activity hierarchy.

    Returns:
        Dict: Pruned taxonomy tree with only relevant activities.

    Raises:
        ValueError: If none of the provided activities exist in the taxonomy.
    """
    activity_keys = list(set(audio_paths_to_activities.values()))
    activity_paths = [activity_to_taxonomy_tree_path(activity) for activity in activity_keys]
    valid_nodes: Set[str] = set(node for path in activity_paths for node in path)

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

def evaluate_node(
    audios: List[Audio], activity_audios: List[Audio], tree: Dict[str, Any], results_df: pd.DataFrame
) -> pd.DataFrame:
    """Runs quality checks on a given taxonomy tree node and updates the tree with results.

    Args:
        audios (List[Audio]): The full list of audio files, which may be modified if files are excluded.
        activity_audios (List[Audio]): The subset of `audios` relevant to the current taxonomy node.
        tree (Dict[str, Any]): The taxonomy tree node containing a "checks" key with check functions.
        results_df (pd.DataFrame): DataFrame to store results of the quality checks.

    Returns:
        pd.DataFrame: Updated results DataFrame.
    """
    # Calculate metrics
    metrics = tree.get("metrics")
    if isinstance(metrics, list):
        for metric in metrics:
            results_df = apply_audio_quality_function(results_df, activity_audios, function=metric)

    # Evaluate checks
    checks = tree.get("checks")
    if isinstance(checks, list):
        for check in checks:
            results_df = apply_audio_quality_function(results_df, activity_audios, function=check)

    return results_df


def evaluate_audio():
    """Runs evaluations for one audio. Saves to file. Don't run if the audio files features file already exists.

    Run efficiently without duplicating metrics or checks.

    in:
        activity2evaluations dict
        audio_path
        save_path
    """


def taxonomy_subtree_to_pydra_workflow(subtree: Dict) -> Workflow:
    """Constructs a Pydra workflow that corresponds to a dataset taxonomy subtree."""

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


def create_activity_to_evaluations(audio_to_activity: Dict[str, str], activity_tree: Dict) -> Dict[str, List[str]]:
    """ """
    # get unique activities
    # for each activity:
    #   get subtree
    #   create evaluations list in order
    #   append
    # activity_dict_to_dataset_taxonomy_subtree
    # return activity_to_evaluations


def create_audio_path_to_activity(
    audio_paths: List[str], audio_path_to_activity: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Maps each audio path to an activity, defaulting to 'bioacoustic' if not provided.

    Args:
        audio_paths (List[str]): List of audio file paths.
        path_to_activity (Optional[Dict[str, str]]): Optional mapping of paths to activity names.

    Returns:
        Dict[str, str]: Dictionary mapping each audio path to its activity.
    """
    return {
        path: audio_path_to_activity.get(path, "bioacoustic") if audio_path_to_activity else "bioacoustic"
        for path in audio_paths
    }


def check_quality(
    audio_paths: Union[str, os.PathLike],
    audio_path_to_activity: Dict = None,
    activity_tree: Dict = BIOACOUSTIC_ACTIVITY_TAXONOMY,
    save_path: Union[str, os.PathLike, None] = None,
) -> pd.DataFrame:
    """Runs quality checks on audio files in batches and updates the taxonomy tree."""
    # get the paths to activity dict
    audio_to_activity = audio_path_to_activity(audio_paths, audio_path_to_activity)

    # create activity to evaluations dict

    # create workflow for all audio files
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

    # total_batches = (len(audio_paths) + batch_size - 1) // batch_size
    # print(f"{total_batches} batches of size {batch_size}")

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

    # batches = [audio_paths[i : i + batch_size] for i in range(0, len(audio_paths), batch_size)]
    # batch_aqm_dataframes = Parallel(n_jobs=n_jobs, verbose=verbosity)(
    #     delayed(process_batch)(batch, idx) for idx, batch in enumerate(batches)
    # )

    # all_aqms = pd.concat(batch_aqm_dataframes, ignore_index=True)
    # all_aqms["audio_path_or_id"] = all_aqms["audio_path_or_id"].apply(os.path.basename)
    # if save_path:
    #     all_aqms.to_csv(save_path)
    #     print(f"Results saved to: {save_path}")
    # return all_aqms
