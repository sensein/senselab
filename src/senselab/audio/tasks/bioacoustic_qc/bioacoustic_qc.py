"""Runs bioacoustic task recording quality control on a set of Audio objects."""

from copy import deepcopy
from typing import Dict, List, Optional, Set

from pydra.engine.core import Workflow  # Assuming you're using Pydra workflows

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.bioacoustic_qc.constants import BIOACOUSTIC_TASK_TREE


def audios_to_task_dict(audios: List[Audio]) -> Dict[str, List[Audio]]:
    """Creates a dictionary mapping tasks to their corresponding Audio objects.

    Each Audio object is assigned to a task category based on the `"task"` field in its metadata.
    If an Audio object does not contain a `"task"` field, it is categorized under `"bioacoustic"`.

    Args:
        audios (List[Audio]): A list of Audio objects.

    Returns:
        Dict[str, List[Audio]]: A dictionary where keys are task names and values are lists of
        Audio objects belonging to that task.

    Example:
        >>> audio1 = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, metadata={"task": "breathing"})
        >>> audio2 = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, metadata={"task": "cough"})
        >>> audio3 = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000, metadata={})  # No task
        >>> audios_to_task_dict([audio1, audio2, audio3])
        {'breathing': [audio1], 'cough': [audio2], 'bioacoustic': [audio3]}
    """
    task_dict: Dict[str, List[Audio]] = {}

    for audio in audios:
        task = audio.metadata.get("task", "bioacoustic")  # Default to "bioacoustic" if no task
        if task not in task_dict:
            task_dict[task] = []
        task_dict[task].append(audio)

    return task_dict


def task_to_taxonomy_tree_path(task: str) -> List[str]:
    """Gets the taxonomy tree path for a given task.

    Args:
        task (str): The task name to find in the taxonomy tree.

    Returns:
        List[str]: A list representing the path from the root to the given task.

    Raises:
        ValueError: If the task is not found in the taxonomy tree.
    """

    def find_task_path(tree: Dict, path: List[str]) -> Optional[List[str]]:
        """Recursively searches for the task in the taxonomy tree."""
        for key, value in tree.items():
            new_path = path + [key]  # Extend path
            if key == task:
                return new_path  # Task found, return path
            if isinstance(value, dict) and "subclass" in value and isinstance(value["subclass"], dict):
                result = find_task_path(value["subclass"], new_path)
                if result:
                    return result
        return None

    path = find_task_path(BIOACOUSTIC_TASK_TREE, [])
    if path is None:
        raise ValueError(f"Task '{task}' not found in taxonomy tree.")

    return path


def task_dict_to_dataset_taxonomy_subtree(task_dict: Dict[str, List[Audio]]) -> Dict:
    """Constructs a sub-tree of the taxonomy based on tasks in the dataset.

    Args:
        task_dict (Dict[str, List[Audio]]): A dictionary mapping task names to lists of Audio objects.

    Returns:
        Dict: A pruned version of the taxonomy tree that only contains the relevant tasks.

    Raises:
        ValueError: If none of the provided tasks exist in the taxonomy.
    """
    task_keys = list(task_dict.keys())  # Ensure keys are a list
    task_paths = [task_to_taxonomy_tree_path(task) for task in task_keys]

    # Get a set of all valid nodes in the taxonomy tree paths
    unique_nodes: Set[str] = set(node for path in task_paths for node in path)

    # Deep copy the taxonomy tree to avoid modifying the original
    pruned_tree: Dict = deepcopy(BIOACOUSTIC_TASK_TREE)

    # Ensure 'subclass' exists and is a dictionary before pruning
    subclass_tree = pruned_tree.get("bioacoustic", {}).get("subclass", None)
    if not isinstance(subclass_tree, dict):
        raise TypeError("Expected 'subclass' to be a dictionary in the taxonomy tree.")

    def prune_tree(subtree: Dict[str, Dict]) -> bool:
        """Recursively prunes the taxonomy tree, keeping only relevant branches.

        Args:
            subtree (Dict[str, Dict]): The current subtree being processed.

        Returns:
            bool: True if the subtree contains relevant tasks, False otherwise.
        """
        keys_to_delete = []

        for key in list(subtree.keys()):  # Copy keys to avoid modifying while iterating
            value = subtree[key]
            if isinstance(value, dict) and "subclass" in value and isinstance(value["subclass"], dict):
                keep_branch = prune_tree(value["subclass"])
                if not keep_branch:
                    keys_to_delete.append(key)
            elif key not in unique_nodes:
                keys_to_delete.append(key)

        # Remove non-matching branches
        for key in keys_to_delete:
            del subtree[key]

        return bool(subtree)

    # Start pruning at the correct level of the tree
    if not prune_tree(subclass_tree):
        raise ValueError("None of the provided tasks exist in the taxonomy tree.")

    return pruned_tree


def taxonomy_subtree_to_pydra_workflow(subtree: Dict) -> Workflow:
    """Constructs a Pydra workflow that corresponds to a dataset taxonomy subtree."""
    pass


def check_quality(audios: List[Audio], complexity: str = "low") -> None:
    """Runs quality checks on audio data."""
    # audios_to_task_dict
    # for each key in task_dict:
    # replace key with tasks_to_taxonomy_tree_path
    # task_dict_to_dataset_taxonomy_subtree
    # taxonomy_subtree_to_pydra_workflow
    # run workflow
    # save results
    pass
