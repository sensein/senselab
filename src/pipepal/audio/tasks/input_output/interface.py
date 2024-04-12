"""This module defines an API for the task."""

import os
from typing import Any, Dict

from pipepal.utils.abstract_component import AbstractComponent

from .services import DatasetsService


class Interface(AbstractComponent):
    """A factory class for creating and managing service instances.

    It ensures a single instance per service type based on a unique key.
    """

    _instances: Dict[str, Any] = {}  # Class attribute for shared instance cache

    def __init__(self) -> None:
        """Initialize the Interface class with the path to the base directory."""
        super().__init__(os.path.dirname(__file__))

    @classmethod
    def get_service(cls, service_data: Dict[str, Any]) -> Any:  # noqa: ANN401
        """Retrieves or creates a service instance based on the provided service data.

        Parameters:
            service_data (Dict[str, Any]): Data required to identify or create the service instance.

        Returns:
            Any: An instance of the requested service.

        Raises:
            ValueError: If the service name is unsupported.
        """
        # Use a composite key to uniquely identify instances
        key: str = str(cls.get_data_uuid(service_data))

        if key not in cls._instances:
            if service_data["service_name"] == DatasetsService.NAME:
                cls._instances[key] = DatasetsService(service_data)
            else:
                raise ValueError(f"Unsupported service: {service_data['service_name']}")
        return cls._instances[key]

    @AbstractComponent.get_response_time
    @AbstractComponent.schema_validator
    def read_audios_from_disk(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Processes input through a workflow.

        Parameters:
            input (Dict[str, Any]): Input data containing service information and data to process.

        Returns:
            Any: The postprocessed output from the service.
        """
        service = self.get_service(input["service"])
        output = service.read_audios_from_disk(input["data"])
        return output
    
    @AbstractComponent.get_response_time
    @AbstractComponent.schema_validator
    def save_HF_dataset_to_disk(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Saves HF dataset to disk.

        Parameters:
            input (Dict[str, Any]): Input data containing service information and data to process.

        Returns:
            Any: The postprocessed output from the service.

        Todo:
            - This method is not audio specific and may be moved out of this class.
        """
        service = self.get_service(input["service"])
        output = service.save_HF_dataset_to_disk(input["data"])
        return output
    
    @AbstractComponent.get_response_time
    @AbstractComponent.schema_validator
    def upload_HF_dataset_to_HF_hub(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Uploads HF dataset to HF hub.

        Parameters:
            input (Dict[str, Any]): Input data containing service information and data to process.

        Returns:
            Any: The postprocessed output from the service.

        Todo:
            - This method is not audio specific and may be moved out of this class.
        """
        service = self.get_service(input["service"])
        output = service.upload_HF_dataset_to_HF_hub(input["data"])
        return output

    @AbstractComponent.get_response_time
    @AbstractComponent.schema_validator
    def read_local_HF_dataset(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Reads HF dataset from disk.

        Parameters:
            input (Dict[str, Any]): Input data containing service information and data to process.

        Returns:
            Any: The postprocessed output from the service.

        Todo:
            - This method is not audio specific and may be moved out of this class.
        """
        service = self.get_service(input["service"])
        output = service.read_local_HF_dataset(input["data"])
        return output
    
    @AbstractComponent.get_response_time
    @AbstractComponent.schema_validator
    def read_HF_dataset_from_HF_hub(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Reads HF dataset from HF hub.

        Parameters:
            input (Dict[str, Any]): Input data containing service information and data to process.

        Returns:
            Any: The postprocessed output from the service.

        Todo:
            - This method is not audio specific and may be moved out of this class.
        """
        service = self.get_service(input["service"])
        output = service.read_HF_dataset_from_HF_hub(input["data"])
        return output