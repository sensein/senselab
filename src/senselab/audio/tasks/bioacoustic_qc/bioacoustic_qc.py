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


def audios_to_activity_dict(audios: List[Audio]) -> Dict[str, List[Audio]]:
 
    """Creates a dictionary mapping activities to their corresponding Audio objects.

    Each Audio object is assigned to a activity category based on the `"activity "` field in its metadata.
    If an Audio object does not contain a `"activity "` field, it is categorized under `"bioacoustic"`.

    Args:
        audios (List[Audio]): A list of Audio objects.

    Returns:
        Dict[str, List[Audio]]: A dictionary where keys are activity names and values are lists of
        Audio objects belonging to that activity .

    Example:
        >>> audio1 = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, metadata={"activity ": "breathing"})
        >>> audio2 = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, metadata={"activity ": "cough"})
        >>> audio3 = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, metadata={})  # No activity
        >>> audios_to_activity_dict([audio1, audio2, audio3])
        {'breathing': [audio1], 'cough': [audio2], 'bioacoustic': [audio3]}
    """
    activity_dict: Dict[str, List[Audio]] = {}

    for audio in audios:
        activity = audio.metadata.get("activity", "bioacoustic")  # Default to "bioacoustic" if no activity
        if activity not in activity_dict:
            activity_dict[activity] = []
        activity_dict[activity].append(audio)

    return activity_dict


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


def activity_dict_to_dataset_taxonomy_subtree(activity_dict: Dict[str, List[Audio]], activity_tree: Dict) -> Dict:
    """Constructs a pruned taxonomy tree containing only relevant activities.

    This function takes a mapping of activities to audio files and removes irrelevant branches from a given taxonomy
    tree, keeping only activities that exist in `activity_dict`.

    Args:
        activity_dict (Dict[str, List[Audio]]): A dictionary mapping activity names to lists of Audio objects.
        activity_tree (Dict): The full taxonomy tree defining the activity hierarchy.

    Returns:
        Dict: A pruned version of `activity_tree` that retains only activities present in `activity_dict`.

    Raises:
        ValueError: If none of the provided activities exist in the taxonomy.
    """
    activity_keys = list(activity_dict.keys())
    activity_paths = [activity_to_taxonomy_tree_path(activity) for activity in activity_keys]
    valid_nodes: Set[str] = set(node for path in activity_paths for node in path)

    pruned_tree: Dict = deepcopy(activity_tree)

    def prune_tree(subtree: Dict) -> bool:
        """Recursively prunes the taxonomy tree, keeping only relevant branches.

        Args:
            subtree (Dict): The current subtree being processed.

        Returns:
            bool: True if the subtree contains relevant activities, False otherwise.
        """
        keys_to_delete = []

        # Determine keys to delete
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

        # Remove unwanted nodes
        for key in keys_to_delete:
            del subtree[key]

        return bool(subtree)  # Return True if subtree contains relevant data

    subclass_tree = pruned_tree["bioacoustic"].get("subclass", None)

    if not prune_tree(subclass_tree):
        pruned_tree["bioacoustic"]["subclass"] = None  # Ensure "bioacoustic" key remains

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
    
    pass


def run_taxonomy_subtree_checks_recursively(
    audios: List[Audio], dataset_tree: Dict, activity_dict: Dict, results_df: pd.DataFrame
) -> pd.DataFrame:
    """Runs quality checks recursively on a taxonomy subtree and updates the results DataFrame.

    This function iterates over a hierarchical taxonomy structure and applies quality control
    checks at each relevant node. It determines the relevant audios for each taxonomy node,
    applies the appropriate checks, and updates the `results_df` with check results.

    Args:
        audios (List[Audio]): The full list of audio files to be checked.
        dataset_tree (Dict): The taxonomy tree representing the dataset structure, which will be modified in-place.
        activity_dict (Dict[str, List[Audio]]): A dictionary mapping activity names to lists of `Audio` objects.
        results_df (pd.DataFrame): DataFrame to store quality check results.

    Returns:
        pd.DataFrame: Updated DataFrame with quality check results for each audio file.
    """
    activity_to_tree_path_dict = {activity: activity_to_taxonomy_tree_path(activity) for activity in activity_dict}

    def check_subtree_nodes(subtree: Dict, results_df: pd.DataFrame) -> pd.DataFrame:
        """Recursively processes each node in the subtree, applying checks where applicable.

        Args:
            subtree (Dict): The current subtree being processed.
            results_df (pd.DataFrame): DataFrame to store quality check results.
        """
        for key, node in subtree.items():
            # Construct activity-specific audio list for the current node
            activity_audios = [
                audio
                for activity in activity_dict
                if key in activity_to_tree_path_dict[activity]
                for audio in activity_dict[activity]
            ]

            results_df = evaluate_node(audios=audios, activity_audios=activity_audios, tree=node, results_df=results_df)

            # Recursively process subclasses if they exist
            if isinstance(node.get("subclass"), dict):  # Ensure subclass exists
                check_subtree_nodes(node["subclass"], results_df=results_df)  # Recurse on actual subtree
        return results_df

    check_subtree_nodes(dataset_tree, results_df=results_df)  # Start recursion from root
    return results_df


def check_quality(
    audio_dir: Union[str, os.PathLike],
    activity_tree: Dict = BIOACOUSTIC_ACTIVITY_TAXONOMY,
    complexity: str = "low",
    batch_size: int = 8,
    save_path: Union[str, os.PathLike, None] = None,
    n_jobs: int = -1,
    verbosity: int = 20,
) -> pd.DataFrame:
    """Runs quality checks on audio files in batches and updates the taxonomy tree."""
    audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
    audio_paths = [
        os.path.join(root, fname)
        for root, _, files in os.walk(str(audio_dir))
        for fname in files
        if fname.lower().endswith(audio_extensions)
    ]

    print("Audio paths loaded.")

    total_batches = (len(audio_paths) + batch_size - 1) // batch_size
    print(f"{total_batches} batches of size {batch_size}")

    def process_batch(batch_paths: List[str], batch_idx: int) -> pd.DataFrame:
        batch_audios = [Audio.from_filepath(p) for p in batch_paths]
        activity_dict = audios_to_activity_dict(batch_audios)
        dataset_tree = activity_dict_to_dataset_taxonomy_subtree(activity_dict, activity_tree=activity_tree)

        results_df = pd.DataFrame([a.orig_path_or_id for a in batch_audios], columns=["audio_path_or_id"])
        results_df = run_taxonomy_subtree_checks_recursively(
            audios=batch_audios,
            dataset_tree=dataset_tree,
            activity_dict=activity_dict,
            results_df=results_df,
        )
        del batch_audios
        return results_df

    batches = [audio_paths[i : i + batch_size] for i in range(0, len(audio_paths), batch_size)]
    batch_aqm_dataframes = Parallel(n_jobs=n_jobs, verbose=verbosity)(
        delayed(process_batch)(batch, idx) for idx, batch in enumerate(batches)
    )

    all_aqms = pd.concat(batch_aqm_dataframes, ignore_index=True)
    all_aqms["audio_path_or_id"] = all_aqms["audio_path_or_id"].apply(os.path.basename)
    if save_path:
        all_aqms.to_csv(save_path)
        print(f"Results saved to: {save_path}")
    return all_aqms
