"""Runs bioacoustic activity recording quality control on a set of Audio objects."""

import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from pydra.engine.core import Workflow  # Assuming you're using Pydra workflows

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_ACTIVITY_TAXONOMY


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
        evaluations = subtree_to_evaluations(subtree)
        activity_to_evaluations[activity] = evaluations
    return activity_to_evaluations


def get_evaluation(
    audio: Audio,
    evaluation_function: Callable[[Audio], float | bool],
    id: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes and caches an evaluation (metric or check) in the DataFrame.

    Args:
        audio: An `Audio` instance.
        evaluation_function: Function that takes `Audio` and returns a value.
        id: Identifier corresponding to the row in the DataFrame.
        df: DataFrame containing an 'id' column.

    Returns:
        Updated DataFrame with the evaluation result for the given id.
    """
    evaluation_name = evaluation_function.__name__

    if evaluation_name in df.columns and not pd.isna(df.loc[df["id"] == id, evaluation_name]).all():
        return df  # Already computed

    if evaluation_name not in df.columns:
        df[evaluation_name] = None

    df.loc[df["id"] == id, evaluation_name] = evaluation_function(audio)
    return df


def evaluate_audio(audio_path, save_path, activity, evaluations):
    """Runs evaluations iteratively. Skips audio if output file already exists."""
    id = Path(audio_path).stem
    output_file = save_path / f"{id}.csv"
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
    save_path: Union[str, Path],
    batch_size: int,
):
    """
    Runs evaluation functions over audio files, optionally in parallel.

    Args:
        audio_path_to_activity (Dict): Maps audio file paths to activity labels.
        activity_to_evaluations (Dict): Maps activity labels to a list of evaluation functions.
        save_path (str | Path): Directory where evaluation CSVs should be saved.
        batch_size (int): Number of audio files to process in each parallel batch.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    audio_paths = sorted(audio_path_to_activity.keys())
    batches = [audio_paths[i:i + batch_size] for i in range(0, len(audio_paths), batch_size)]

    def process_batch(batch_audio_paths):
        for audio_path in batch_audio_paths:
            activity = audio_path_to_activity[audio_path]
            evaluations = activity_to_evaluations[activity]
            evaluate_audio(audio_path, save_path, activity, evaluations)

    Parallel(n_jobs=len(batches))(delayed(process_batch)(batch) for batch in batches)


def check_quality(
    audio_paths: Union[str, os.PathLike],
    audio_path_to_activity: Dict = {},
    activity_tree: Dict = BIOACOUSTIC_ACTIVITY_TAXONOMY,
    save_dir: Union[str, os.PathLike, None] = None,
    batch_size: int = 8
) -> pd.DataFrame:
    """Runs quality checks on audio files in n_batches and updates the taxonomy tree."""
    # get the paths to activity dict
    audio_path_to_activity = {path: audio_path_to_activity.get(path, "bioacoustic") for path in audio_paths}

    # create activity to evaluations dict
    activity_to_evaluations = create_activity_to_evaluations(
        audio_path_to_activity=audio_path_to_activity, activity_tree=activity_tree
    )

    # run_evaluations
    run_evaluations(audio_path_to_activity, activity_to_evaluations, save_dir, batch_size=batch_size)

    # construct evaluations csv
    csv_paths = Path(save_dir).glob("*.csv")
    evaluations_df = pd.concat((pd.read_csv(p) for p in csv_paths), axis=0, ignore_index=True, sort=False)

    # label include, exclude, review
    return evaluations_df



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
