"""Runs bioacoustic activity recording quality control on audio files."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import pandas as pd

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.quality_control.trees import (
    BIOACOUSTIC_ACTIVITY_TAXONOMY,
)
from senselab.audio.tasks.quality_control.evaluate import evaluate_dataset


def activity_to_evaluations(
    audio_path_to_activity: Dict[str, str], activity_tree
) -> Dict[str, Sequence[Callable[[Audio], Union[float, bool, str]]]]:
    """Maps each activity label to its associated evaluation functions.

    Args:
        audio_path_to_activity (Dict[str, str]): Maps audio file paths
        to activity labels.
        activity_tree: The full taxonomy tree of activities (TaxonomyNode).

    Returns:
        Dict[str, Sequence[Callable[[Audio], Union[float, bool, str]]]]:
        Maps activity label to evaluations.
    """
    unique_activities = set(audio_path_to_activity.values())
    activity_to_evaluations = {}
    for activity in unique_activities:
        # Directly call TaxonomyNode methods instead of wrapper functions
        subtree = activity_tree.prune_to_activity(activity)
        if subtree is None:
            raise ValueError(f"Activity '{activity}' not found in taxonomy.")
        evaluations = subtree.get_all_evaluations()
        activity_to_evaluations[activity] = evaluations
    return activity_to_evaluations


def check_quality(
    audio_paths: List[Union[str, os.PathLike]],
    audio_path_to_activity: Dict = {},
    activity_tree=BIOACOUSTIC_ACTIVITY_TAXONOMY,
    output_dir: Optional[Union[str, os.PathLike]] = None,
    batch_size: int = 8,
    n_cores: int = 4,
    window_size_sec: float = 0.025,
    step_size_sec: float = 0.0125,
    skip_windowing: bool = False,
) -> pd.DataFrame:
    """Runs audio quality control evaluations across multiple audio files.

    The results are automatically saved in multiple formats:
    - results_summary.csv: Flattened CSV with evaluations as columns
      (main output)
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
        step_size_sec: Step size in seconds between windows
                      (default: 0.0125)
        skip_windowing: If True, only compute scalar metrics without
                       windowing

    Returns:
        pd.DataFrame: Flattened DataFrame with evaluations as columns
        (human-readable)
    """
    # Convert output_dir to Path with default
    output_directory = Path(output_dir or "qc_results")
    output_directory.mkdir(exist_ok=True, parents=True)

    # Setup activity mappings
    audio_path_to_activity = {str(path): audio_path_to_activity.get(str(path), "bioacoustic") for path in audio_paths}

    # Create activity to evaluations mapping
    activity_evaluations_dict = activity_to_evaluations(
        audio_path_to_activity=audio_path_to_activity,
        activity_tree=activity_tree,
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
