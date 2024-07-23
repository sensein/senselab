"""This module implements some IOTask utilities.

This modules is deprecated and will be removed soon.
"""

import os
from typing import Any, Dict, List, Union

import pydra
from datasets import Audio, Dataset, Image, load_dataset

from senselab.utils.data_structures.file import File, from_strings_to_files
from senselab.utils.data_structures.model import check_hf_repo_exists


def read_files_from_disk(files: Union[str, List[str]]) -> Dict[str, Any]:
    """Read files from disk and create a Hugging Face `Dataset` object."""
    if isinstance(files, str):
        files = [files]
    formatted_files = from_strings_to_files(files)

    def _from_files_to_dataset(files: List[File]) -> Dataset:
        """Reading files from disk and create a HuggingFace `Dataset` object."""
        # Checking if all files are of the same type
        if not all(file.type == files[0].type for file in files):
            raise ValueError("All files must be of the same type.")

        # Loading file paths
        file_data = [str(file.filepath) for file in files]

        # Creating the Dataset object
        dict_obj = {files[0].type: file_data}
        # Using the type of the first file as the key

        return _from_dict_to_hf_dataset(dict_obj)

    dataset = _from_files_to_dataset(formatted_files)
    return _from_hf_dataset_to_dict(dataset)


def read_dataset_from_disk(input_path: str, split: str, streaming: bool = False) -> Dict[str, Any]:
    """Loads a Hugging Face `Dataset` object from disk.

    It determines the format based on the file extension or directory.
    """
    # Determine the input format.
    if os.path.isdir(input_path):
        input_format = "arrow"
    else:
        input_format = os.path.splitext(input_path)[1].strip(".")

    # Load the dataset
    try:
        dataset = load_dataset(
            input_format,
            split=split,
            data_files=input_path,
            streaming=streaming,
        )
        return _from_hf_dataset_to_dict(dataset)
    except Exception as e:
        # Generic error handling, e.g., network issues, data loading issues
        raise RuntimeError(f"An error occurred while loading the dataset: {str(e)}")


def read_dataset_from_hub(
    remote_repository: str,
    revision: str = "main",
    split: str = "all",
) -> Dict[str, Any]:
    """Loads a Hugging Face `Dataset` object from the Hugging Face Hub.

    It includes support for private repositories.
    """
    if not check_hf_repo_exists(remote_repository, "main", "dataset"):
        raise RuntimeError(
            f"The repository {remote_repository} - {revision} - {split}" " does not exist or could not be accessed."
        )

    # Load the dataset
    try:
        dataset = load_dataset(path=remote_repository, revision=revision, split=split)
    except Exception as e:
        # Generic error handling, e.g., network issues, data loading issues
        raise RuntimeError(f"An error occurred while loading the dataset: {str(e)}")

    return _from_hf_dataset_to_dict(dataset)


def push_dataset_to_hub(
    dataset: Dict[str, Any], remote_repository: str, revision: str = "main", split: str = "all"
) -> None:
    """Uploads a Hugging Face `Dataset` object to the Hugging Face Hub."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)

    try:
        # Upload the dataset to the Hugging Face Hub
        hf_dataset.push_to_hub(repo_id=remote_repository, revision=revision, split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to push dataset to the hub: {str(e)}")
    return


def save_dataset_to_disk(
    dataset: Dict[str, Any],
    output_directory: str,
    output_name: str,
    output_format: str = "parquet",
) -> None:
    """Saves a Hugging Face `Dataset` object to disk."""
    # TODO: optimize saving process (playing with batch size and num of cpus)
    hf_dataset = _from_dict_to_hf_dataset(dataset)

    # Prepare the output path differently if the format is Arrow
    if output_format == "arrow":
        output_path = os.path.join(output_directory, output_name)
        # No extension for Arrow, it's a directory
    else:
        output_path = os.path.join(output_directory, f"{output_name}.{output_format}")

    # Create the output directory, ignore error if it already exists
    os.makedirs(output_directory, exist_ok=True)

    if output_format == "parquet":

        def _save_hf_dataset_as_parquet(dataset: Dataset, output_path: str) -> None:
            """Saves a Hugging Face `Dataset` object to parquet format."""
            dataset.to_parquet(output_path)

        _save_hf_dataset_as_parquet(hf_dataset, output_path)
    elif output_format == "json":

        def _save_hf_dataset_as_json(dataset: Dataset, output_path: str) -> None:
            """Saves a Hugging Face `Dataset` object to json format."""
            dataset.to_json(output_path)

        _save_hf_dataset_as_json(hf_dataset, output_path)
    elif output_format == "csv":

        def _save_hf_dataset_as_csv(dataset: Dataset, output_path: str) -> None:
            """Saves a Hugging Face `Dataset` object to csv format."""
            dataset.to_csv(output_path)

        _save_hf_dataset_as_csv(hf_dataset, output_path)
    elif output_format == "sql":

        def _save_hf_dataset_as_sql(dataset: Dataset, output_path: str) -> None:
            """Saves a Hugging Face `Dataset` object to sql format."""
            dataset.to_sql(output_path)

        _save_hf_dataset_as_sql(hf_dataset, output_path)
    elif output_format == "arrow":

        def _save_hf_dataset_as_arrow(dataset: Dataset, output_path: str) -> None:
            """Saves a Hugging Face `Dataset` object in Apache Arrow format."""
            dataset.save_to_disk(output_path)

        _save_hf_dataset_as_arrow(hf_dataset, output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return


def _from_hf_dataset_to_dict(dataset: Dataset) -> Dict[str, Any]:
    """Converts a Hugging Face `Dataset` object to a dictionary."""
    return dataset.to_dict()


def _from_dict_to_hf_dataset(
    data: Dict[str, Any],
    image_columns: list[str] = ["image"],
    audio_columns: list[str] = ["audio"],
) -> Dataset:
    """Converts a dictionary to a Hugging Face `Dataset` object.

    It casts image and audio columns.
    """
    dataset = Dataset.from_dict(data)

    # Cast image columns
    for column in image_columns:
        if column in dataset.column_names:
            dataset = dataset.cast_column(column, Image())

    # Cast audio columns
    for column in audio_columns:
        if column in dataset.column_names:
            dataset = dataset.cast_column(column, Audio(mono=False))

    return dataset


read_files_from_disk_pt = pydra.mark.task(read_files_from_disk)
read_dataset_from_disk_pt = pydra.mark.task(read_dataset_from_disk)
read_dataset_from_hub_pt = pydra.mark.task(read_dataset_from_hub)
push_dataset_to_hub_pt = pydra.mark.task(push_dataset_to_hub)
save_dataset_to_disk_pt = pydra.mark.task(save_dataset_to_disk)
