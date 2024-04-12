"""Tests for the run method of the ExampleService in the audio's ExampleTask."""

import pytest

from pipepal.audio.tasks import ExampleTask

def test_exampletask_run():
    """Test the run method of ExampleTask.
    
    This test verifies:
    1. The output type is a dictionary.
    2. The output's 'output' key correctly returns the expected string.
    """
    # Instantiate your class
    exampleTask = ExampleTask()
    
    # Call the method you wish to test
    output = exampleTask.run({
        "data": {
            "hello": "world"
        },
        "service": {
            "service_name": "ExampleService", 
            "model_checkpoint": "model.ckpt",
            "model_version": "1.0",
        }
    })
    
    # Assert conditions about the output
    assert type(output) == dict, "Output should be a dictionary"
    expected_output_output = "ExampleService output"
    assert output['output'] == expected_output_output, "The output of the run method does not match the expected output"

def test_exampletask_run_missing_service_fields():
    """Test the run method of ExampleTask.
    
    This test checks for its handling of missing required service fields.
    This test iteratively removes each required service field (service_name, model_checkpoint, 
    model_version) from the input and checks if a ValueError is raised with the appropriate 
    error message indicating the missing field.
    
    The test ensures that:
    1. A ValueError is thrown for missing service fields.
    2. The exception message contains the name of the missing field.
    """
    # Instantiate your class
    example_task = ExampleTask()
    
    # Define a list of required service fields
    required_service_fields = ["service_name", "model_checkpoint", "model_version"]

    # Iterate over each required field and test by removing each one by one
    for missing_field in required_service_fields:
        # Create a payload with all necessary fields
        payload_with_all_fields = {
            "data": {
                "hello": "world"
            },
            "service": {
                "service_name": "my_service",
                "model_checkpoint": "model.ckpt",
                "model_version": "1.0",
            }
        }

        # Remove the field being tested
        del payload_with_all_fields["service"][missing_field]

        # Use pytest.raises to assert that a ValueError is raised due to the missing field
        with pytest.raises(ValueError) as excinfo:
            example_task.run(payload_with_all_fields)

        # Assert that the error message contains the name of the missing field
        assert missing_field in str(excinfo.value), f"Expected ValueError due to missing '{missing_field}' field"
