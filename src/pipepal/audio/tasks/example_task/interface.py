"""This module defines an API for managing services related to the example task."""

import os
from typing import Any, Dict

from pipepal.utils.abstract_component import AbstractComponent

from .services import ExampleService


class Interface(AbstractComponent):
    """A factory class for creating and managing instances of services related to the example task.

    This class facilitates the retrieval and singleton management of service instances based on
    a unique identifier derived from the service data. The main functionality is provided through
    the `run` method which processes input through a comprehensive workflow including
    preprocessing, processing, and postprocessing phases, using the specified service instance.

    Attributes:
        _instances (Dict[str, Any]): A class-level dictionary that caches service instances
                                     to ensure they are singleton per type, indexed by a unique key.

    Examples:
        >>> exampleTask = Interface()
        >>> example_response = exampleTask.run({
        ...     "service": {
        ...         "service_name": "ExampleService",
        ...         "model_checkpoint": "model.ckpt",
        ...         "model_version": "1.0",
        ...     },
        ...     "data": {
        ...         "hello": "world"
        ...     }
        ... })
        >>> print(example_response)
        The output from the ExampleService after processing the input data.
    """

    _instances: Dict[str, Any] = {}  # Cache to store unique service instances

    def __init__(self) -> None:
        """Initialize the Interface class with the path to the directory where this file is located."""  # noqa: E501
        super().__init__(os.path.dirname(__file__))

    @classmethod
    def get_service(cls, service_data: Dict[str, Any]) -> Any: # noqa: ANN401
        """Retrieves or creates a service instance based on the provided service data.

        This method ensures that each service type, identified by a composite key
        (including the service name, model checkpoint, and version), has only one instance.

        Parameters:
            service_data (Dict[str, Any]): A dictionary containing the service configuration,
                which must include 'service_name', and may include 'model_checkpoint' and 'model_version'
                for specific service setups.

        Returns:
            Any: An instance of the requested service.

        Raises:
            ValueError: If the 'service_name' in service_data is unsupported or not recognized.

        Examples:
            >>> service_data = {
            ...     "service_name": "ExampleService",
            ...     "model_checkpoint": "model.ckpt",
            ...     "model_version": "1.0"
            ... }
            >>> service = Interface.get_service(service_data)
            >>> print(service)
            Instance of ExampleService configured with model checkpoint 'model.ckpt' and version 1.0
        """
        key: str = f"{service_data.get('service_name')}|{service_data.get('model_checkpoint')}|{service_data.get('model_version')}"  # noqa: E501

        if key not in cls._instances:
            if service_data["service_name"] == ExampleService.NAME:
                cls._instances[key] = ExampleService(service_data)
            else:
                raise ValueError(f"Unsupported service: {service_data['service_name']}")
        return cls._instances[key]

    @AbstractComponent.get_response_time
    @AbstractComponent.schema_validator
    def run(self, input_obj: Dict[str, Any]) -> Any:  # noqa: ANN401
        """Processes input through a workflow of preprocessing, processing, and postprocessing.

        This method uses a service instance, which is fetched based on the service details provided
        in 'input', to run the given data through the service's workflow. This includes preprocessing
        the data, processing it according to the service's logic, and then postprocessing results.

        Parameters:
            input_obj (Dict[str, Any]): A dictionary containing:
                                    - 'service': A dictionary with the service configuration.
                                    - 'data': The data to be processed by the service.

        Returns:
            Any: The output of the service after the workflow has been applied to the input.

        Examples:
            >>> input_data = {
            ...     "service": {
            ...         "service_name": "ExampleService",
            ...         "model_checkpoint": "model.ckpt",
            ...         "model_version": "1.0"
            ...     },
            ...     "data": {"hello": "world"}
            ... }
            >>> exampleTask = Interface()
            >>> output = exampleTask.run(input_data)
            >>> print(output)
            The processed data.
        """
        service = self.get_service(input_obj["service"])
        preprocessing_output = service.preprocess(input_obj["data"])
        processing_output = service.process(preprocessing_output)
        postprocessing_output = service.postprocess(processing_output)
        return postprocessing_output