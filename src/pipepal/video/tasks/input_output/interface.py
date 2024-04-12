"""This module defines an API for the task."""

import os
from typing import Any, Dict

from pipepal.utils.abstract_component import AbstractComponent

from .services import FfmpegService


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
            if service_data["service_name"] == FfmpegService.NAME:
                cls._instances[key] = FfmpegService(service_data)
            else:
                raise ValueError(f"Unsupported service: {service_data['service_name']}")
        return cls._instances[key]

    @AbstractComponent.get_response_time
    @AbstractComponent.schema_validator
    def extract_audios_from_videos(self, input_obj: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts audios from videos using the specified service and input data.

        Args:
            input_obj (Dict[str, Any]): The input object containing the service and data.
        
        Returns:
            Dict[str, Any]: The output from the service after extracting audios from videos.
        """
        service = self.get_service(input_obj["service"])
        output = service.extract_audios_from_videos(input_obj["data"])
        return output