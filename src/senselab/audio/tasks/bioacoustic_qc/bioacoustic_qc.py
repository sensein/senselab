"""Runs bioacoustic task recording quality control on a set of Audio objects."""

from typing import Dict, List

from pydra.engine.core import Workflow  # Assuming you're using Pydra workflows

from senselab.audio.data_structures import Audio


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


def tasks_to_taxonomy_tree_path(task: str) -> List[str]:
    """Gets the taxonomy tree path for a task."""
    pass


def task_dict_to_dataset_taxonomy_subtree(task_dict: Dict[str, List[Audio]]) -> Dict:
    """Constructs a sub-tree of the taxonomy based on tasks in the dataset."""
    # get keys in task_dict
    # traverse the taxonomy tree, deleting paths that don't correspond to tasks
    pass


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
