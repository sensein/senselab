"""This module defines an API for the task."""

import os
from typing import Any, Dict

from pipepal.utils.abstract_component import AbstractComponent

from .services import ExampleService


class Interface(AbstractComponent):
    """A factory class for creating and managing service instances.

    It ensures a single instance per service type based on a unique key.
    """

    _instances: Dict[str, Any] = {}  # Class attribute for shared instance cache

    def __init__(self) -> None:
        """Initialize the Interface class with the path to the base directory."""
        super().__init__(os.path.dirname(__file__))

    @classmethod
    def get_service(cls, service_data: Dict[str, Any]) -> Any:
        """Retrieves or creates a service instance based on the provided service data.

        Parameters:
            service_data (Dict[str, Any]): Data required to identify or create the service instance.

        Returns:
            Any: An instance of the requested service.

        Raises:
            ValueError: If the service name is unsupported.
        """
        # Use a composite key to uniquely identify instances
        key: str = cls.get_data_uuid(service_data)

        if key not in cls._instances:
            if service_data["service_name"] == ExampleService.NAME:
                cls._instances[key] = ExampleService(service_data)
            else:
                raise ValueError(f"Unsupported service: {service_data['service_name']}")
        return cls._instances[key]

    @AbstractComponent.get_response_time
    @AbstractComponent.schema_validator
    def run(self, input: Dict[str, Any]) -> Any:
        """Processes input through a workflow: preprocessing, processing, and postprocessing.

        Parameters:
            input (Dict[str, Any]): Input data containing service information and data to process.

        Returns:
            Any: The postprocessed output from the service.
        """
        service = self.get_service(input["service"])
        preprocessing_output = service.preprocess(input["data"])
        processing_output = service.process(preprocessing_output)
        postprocessing_output = service.postprocess(processing_output)
        return postprocessing_output