"""Runs bioacoustic task recording quality control on a set of Audio objects."""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple

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


def task_dict_to_dataset_taxonomy_subtree(task_dict: Dict[str, List[Audio]], task_tree: Dict) -> Dict:
    """Constructs a pruned taxonomy tree containing only relevant tasks.

    This function takes a mapping of tasks to audio files and removes irrelevant branches from a given taxonomy tree,
    keeping only tasks that exist in `task_dict`.

    Args:
        task_dict (Dict[str, List[Audio]]): A dictionary mapping task names to lists of Audio objects.
        task_tree (Dict): The full taxonomy tree defining the task hierarchy.

    Returns:
        Dict: A pruned version of `task_tree` that retains only tasks present in `task_dict`.

    Raises:
        ValueError: If none of the provided tasks exist in the taxonomy.
    """
    task_keys = list(task_dict.keys())
    task_paths = [task_to_taxonomy_tree_path(task) for task in task_keys]
    valid_nodes: Set[str] = set(node for path in task_paths for node in path)

    pruned_tree: Dict = deepcopy(task_tree)

    def prune_tree(subtree: Dict) -> bool:
        """Recursively prunes the taxonomy tree, keeping only relevant branches.

        Args:
            subtree (Dict): The current subtree being processed.

        Returns:
            bool: True if the subtree contains relevant tasks, False otherwise.
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


def check_node(audios: List[Audio], task_audios: List[Audio], tree: Dict[str, Any]) -> None:
    """Runs quality checks on a given taxonomy tree node and updates the tree with results.

    This function applies all checks defined in the node to the provided `task_audios` list.
    It modifies `audios` by removing excluded files and updates `tree` with check results.

    Args:
        audios (List[Audio]): The full list of audio files, which may be modified if files are excluded.
        task_audios (List[Audio]): The subset of `audios` relevant to the current taxonomy node.
        tree (Dict[str, Any]): The taxonomy tree node containing a "checks" key with check functions.
    """
    # Ensure "checks_results" is always a dictionary
    tree.setdefault("checks_results", {})

    # Only iterate over checks if it's a list
    checks = tree.get("checks")
    if not isinstance(checks, list):
        return  # Exit early if there are no checks

    for check in checks:
        if callable(check):
            tree["checks_results"][check.__name__] = check(audios=audios, task_audios=task_audios)


def taxonomy_subtree_to_pydra_workflow(subtree: Dict) -> Workflow:
    """Constructs a Pydra workflow that corresponds to a dataset taxonomy subtree."""
    pass


def run_taxonomy_subtree_checks_recursively(audios: List[Audio], dataset_tree: Dict, task_dict: Dict) -> Dict:
    """Runs quality checks recursively on a taxonomy subtree and updates the tree with results.

    This function iterates over a hierarchical taxonomy structure and applies quality control
    checks at each relevant node. It determines the relevant audios for each taxonomy node,
    applies the appropriate checks, and updates the `dataset_tree` with check results.

    Args:
        audios (List[Audio]): The full list of audio files to be checked.
        dataset_tree (Dict): The taxonomy tree representing the dataset structure, which will be modified in-place.
        task_dict (Dict[str, List[Audio]]): A dictionary mapping task names to lists of associated `Audio` objects.

    Returns:
        Dict: The updated taxonomy tree with quality check results stored in the `checks_results` field
        at relevant levels.
    """
    task_to_tree_path_dict = {task: task_to_taxonomy_tree_path(task) for task in task_dict}

    def check_subtree_nodes(subtree: Dict) -> None:
        """Recursively processes each node in the subtree, applying checks where applicable.

        Args:
            subtree (Dict): The current subtree being processed.
        """
        for key, node in subtree.items():
            # Construct task-specific audio list for the current node
            task_audios = [
                audio for task in task_dict if key in task_to_tree_path_dict[task] for audio in task_dict[task]
            ]
            check_node(audios=audios, task_audios=task_audios, tree=node)

            # Recursively process subclasses if they exist
            if isinstance(node.get("subclass"), dict):  # Ensure subclass exists
                check_subtree_nodes(node["subclass"])  # Recurse on actual subtree

    check_subtree_nodes(dataset_tree)  # Start recursion from root
    return dataset_tree


def check_quality(
    audios: List[Audio], task_tree: Dict = BIOACOUSTIC_TASK_TREE, complexity: str = "low"
) -> Tuple[Dict, List[Audio]]:
    """Runs quality checks on audio files and updates the taxonomy tree.

    Maps `Audio` objects to tasks, prunes the taxonomy tree, and applies quality checks recursively. Returns the
    updated taxonomy tree and the modified list of audios.

    Args:
        audios (List[Audio]): Audio files to analyze.
        task_tree (Dict, optional): Taxonomy tree defining task hierarchy. Defaults to `BIOACOUSTIC_TASK_TREE`.
        complexity (str, optional): Processing complexity level (unused, reserved for future use). Defaults to `"low"`.

    Returns:
        Tuple[Dict, List[Audio]]: The updated taxonomy tree with `checks_results` and the list of audios not excluded.
    """
    task_dict = audios_to_task_dict(audios)
    dataset_tree = task_dict_to_dataset_taxonomy_subtree(task_dict, task_tree=task_tree)
    run_taxonomy_subtree_checks_recursively(audios, dataset_tree=dataset_tree, task_dict=task_dict)
    return dataset_tree, audios
