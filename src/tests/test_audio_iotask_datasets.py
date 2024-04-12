"""This module tests the audio's IOTask class."""


import os
from typing import Any, Dict, List

import pytest
from datasets import Dataset

from pipepal.audio.tasks import IOTask as AudioIOTask


def test_read_audios_from_disk_input_errors() -> None:
    """Test the read_audios_from_disk method.
    
    This test checks if the read_audios_from_disk method raises appropriate errors for invalid inputs.
    This tests:
    1. Missing 'service' or 'data' keys in the input dictionary.
    2. Invalid file paths in the 'files' list (non-existent files).
    """
    with pytest.raises(ValueError):
        # Missing 'service' key
        AudioIOTask().read_audios_from_disk({
            "data": {
                "files": ["/path/to/audio/file1.wav"]
            }
        })

    with pytest.raises(ValueError):
        # Missing 'data' key
        AudioIOTask().read_audios_from_disk({
            "service": {
                "service_name": "Datasets"
            }
        })

    with pytest.raises(FileNotFoundError):
        # Non-existent file path
        AudioIOTask().read_audios_from_disk({
            "data": {
                "files": ["/non/existent/path/file1.wav"]
            },
            "service": {
                "service_name": "Datasets"
            }
        })

def test_read_audios_from_disk_output_type() -> None:
    """Test the read_audios_from_disk method to check if the output is of type HF datasets."""
    test_input = {        
        "data": {
            "files": [f"{os.path.dirname(__file__)}/data_for_testing/audio_48khz_mono_16bits.wav", f"{os.path.dirname(__file__)}/data_for_testing/audio_48khz_stereo_16bits.wav"]
        },
        "service": {
            "service_name": "Datasets"
        }
    }
    response = AudioIOTask().read_audios_from_disk(test_input)
    assert isinstance(response["output"], Dataset), "The output should be an instance of HF_datasets."

def test_read_audios_from_disk_output_dimensions() -> None:
    """Test the read_audios_from_disk method.
     
    This test checks if the dimensions of the output HF datasets object match the input list of audio files.
    Uses mocker to patch the DatasetsService to avoid actual file I/O and simulate reading files.
    """
    test_input: Dict[str, Any] = {
        "data": {
            "files": [f"{os.path.dirname(__file__)}/data_for_testing/audio_48khz_mono_16bits.wav", f"{os.path.dirname(__file__)}/data_for_testing/audio_48khz_stereo_16bits.wav"]
        },
        "service": {
            "service_name": "Datasets"
        }
    }
    response: Dict[str, List[str]]  = AudioIOTask().read_audios_from_disk(test_input)
    assert len(response["output"]) == len(test_input["data"]["files"]), "The number of items in the output should match the number of input files."


def test_save_HF_dataset_to_disk() -> None:
    """Test the `save_HF_dataset_to_disk` method for successful execution with valid inputs.
    
    This test ensures that when given a correctly formatted input dictionary that includes
    a valid 'service' and 'data' key, the `save_HF_dataset_to_disk` method completes
    without throwing any errors. The test assumes that the input 'service' corresponds
    to a service that exists and can process the 'data' provided. It is designed to
    validate the method's ability to handle expected inputs correctly.

    Assumptions:
    - 'service' is a placeholder for an actual service name expected by the application.
    - 'valid_dataset_identifier_or_path' is a placeholder for an actual dataset identifier
      or path that the method can process.
    
    The test will fail if the method throws any exceptions, indicating issues with the
    method's error handling or functionality with assumed valid inputs.
    """
    # Set up the input dictionary as expected by the method
    input_data = {
        "service": {
            "service_name": "Datasets"
        },
        "data": {
            "dataset": Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"], "type": ["grass", "water"]}),
            "output_path": f"{os.path.dirname(__file__)}/data_for_testing/output_dataset"
        }
    }

    # Test if the function runs without errors with valid inputs
    try:
        AudioIOTask().save_HF_dataset_to_disk(input_data)
    except Exception as e:
        pytest.fail(f"Function raised an exception with valid input: {e}")
    
    # shutil.rmtree(f"{os.path.dirname(__file__)}/data_for_testing/output_dataset")


def test_upload_HF_dataset_to_HF_hub() -> None:
    """Test the `upload_HF_dataset_to_HF_hub` method for successful execution with valid inputs.
    
    This test checks that the `upload_HF_dataset_to_HF_hub` method does not produce any
    errors when executed with an input dictionary containing correct 'service' and 'data' keys.
    This verifies the method's capacity to operate as expected under normal conditions. The
    'service' should be a real service name within the application's context, capable of
    processing the provided 'data' (dataset identifier or path).

    Assumptions:
    - 'valid_service' should be replaced with a real service name known to the system.
    - 'valid_dataset_identifier_or_path' should be an actual path or identifier that the
      service can handle.

    If the method throws exceptions with these inputs, the test will fail, highlighting
    potential problems in the method's implementation or issues with handling inputs
    that are presumed to be correct.

    Todo:
        - We may want to set up a lab HF account
    """
    # Set up the input dictionary as expected by the method
    input_data = {
        "service": {
            "service_name": "Datasets"
        },
        "data": {
            "dataset": Dataset.from_dict({"pokemon": ["bulbasaur", "squirtle"], "type": ["grass", "water"]}),
            "output_uri": "fabiocat/test"
        }
    }

    # Test if the function runs without errors with valid inputs
    try:
        AudioIOTask().upload_HF_dataset_to_HF_hub(input_data)
    except Exception as e:
        pytest.fail(f"Function raised an exception with valid input: {e}")


def test_read_local_HF_dataset() -> None:
    """Test the `read_local_HF_dataset` method for successful loading with valid local path input.

    This test ensures that the `read_local_HF_dataset` method can correctly load a dataset from
    a specified local path without throwing any exceptions when provided with a valid path.
    If the method throws exceptions with these inputs, the test will fail, which would indicate issues
    with the method's implementation or the provided path.
    """
    # Set up the input dictionary as expected by the method
    input_data = {
        "service": {
            "service_name": "Datasets"
        },
        "data": {
            "path": f'{os.path.dirname(__file__)}/data_for_testing/output_dataset'
        }
    }

    # Test if the function runs without errors with valid inputs
    try:
        response = AudioIOTask().read_local_HF_dataset(input_data)
        assert "output" in response, "The key 'output' was not found in the result dictionary."
        assert isinstance(response["output"], Dataset), "The result is not a Dataset object as expected."
    except Exception as e:
        pytest.fail(f"Function raised an exception with valid input: {e}")

def test_read_HF_dataset_from_HF_hub() -> None:
    """Test the `read_HF_dataset_from_HF_hub` method for successful loading with valid URI input.

    This test checks that the `read_HF_dataset_from_HF_hub` method can correctly load a dataset from
    the Hugging Face Hub using a provided URI without errors, assuming the URI points to an accessible dataset.
    If the method throws exceptions with these inputs, the test will fail, which would indicate problems
    either in the method's implementation or in the accessibility of the dataset at the specified URI.
    """
    # Set up the input dictionary as expected by the method
    input_data = {
        "service": {
            "service_name": "Datasets"
        },
        "data": {
            "uri": "fabiocat/test" # Assuming a valid URI that is accessible
        }
    }

    # Test if the function runs without errors with valid inputs
    try:
        response = AudioIOTask().read_HF_dataset_from_HF_hub(input_data)
        assert "output" in response, "The key 'output' was not found in the result dictionary."
        assert isinstance(response["output"]["train"], Dataset), "The result is not a Dataset object as expected."
    except Exception as e:
        pytest.fail(f"Function raised an exception with valid input: {e}")