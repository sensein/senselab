"""Tests for the run method of the ExampleService in ExampleTask."""

import pytest

from pipepal.audio.tasks import ExampleTask


def test_exampletask_run():
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
    # 1
    assert type(output) == dict, "Output should be a dictionary"
    
    # 2
    expected_output_output = "ExampleService output"
    assert output['output'] == expected_output_output, "The output of the run method does not match the expected output"


def test_exampletask_run_missing_service_field():
    # Instantiate your class
    exampleTask = ExampleTask()
    
    # Define a payload with a missing field in the 'service' dictionary
    payload_with_missing_field = {
        "data": {
            "hello": "world"
        },
        "service": {
            # 'service_name' is required but missing
            "model_checkpoint": "model.ckpt",
            "model_version": "1.0",
        }
    }
    
    # Use pytest.raises to assert that a ValueError is raised due to the missing field
    with pytest.raises(ValueError) as excinfo:
        exampleTask.run(payload_with_missing_field)

    assert "service_name" in str(excinfo.value), "Expected ValueError due to missing 'service_name' field"
