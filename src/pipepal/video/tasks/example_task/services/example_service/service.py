"""This module implements an example service class which extends the AbstractService.

The service is designed to demonstrate a typical implementation setup where
preprocessing, processing, and postprocessing steps are defined to handle data
in a manner specific to the service's requirements.

Example:
    Demonstrate how to use the ExampleService:

    >>> from path.to.this.module import Service
    >>> input_obj = {
        "data": {
            "hello": "world"
        },
        "service": {
            "service_name": "ExampleService", 
            "model_checkpoint": "model.ckpt",
            "model_version": "1.0",
        }
    }
    >>> service = Service()
    >>> preprocessing_output = service.preprocess(input_obj["data"])
    >>> processing_output = service.process(preprocessing_output)
    >>> postprocessing_output = service.postprocess(processing_output)
    >>> print(postprocessing_output)
    {'output': 'ExampleService output'}

Attributes:
    NAME (str): A class attribute which gives the service a name, used internally.
"""

from typing import Any, Dict

from ...abstract_service import AbstractService


class Service(AbstractService):
    """Example service that extends AbstractService to demonstrate custom processing steps.

    This service class exemplifies a basic structure of a service in a system designed
    for processing data through computational steps: preprocess, process, and postprocess.

    Attributes:
        NAME (str): The public name of the service, intended for identification in registries.
    """

    NAME: str = "ExampleService"

    def __init__(self, configs: Dict[str, Any]) -> None:
        """Initialize the service class with the path to the base directory.

        The initialization involves setting up the base directory and potentially other
        configurations necessary for the service's operations.

        Args:
            configs (Dict[str, Any]): The configs dictionary for the service.
        """
        super().__init__()
            
    def preprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the input data to fit the requirements of this service's processing step.

        Args:
            data (Dict[str, Any]): The input data to preprocess, expected to be in dictionary format.

        Returns:
            Dict[str, Any]: The preprocessed data, adjusted according to the service's needs. This
                 implementation simply passes the data through without modification.

        Example:
            >>> service.preprocess({"hello": "world"})
            {'hello': 'world'}
        """
        return data

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input data to produce a service-specific output.

        This method is the core of the service where the main data manipulation happens. The current
        implementation outputs a placeholder dictionary for demonstration purposes.

        Args:
            data (Dict[str, Any]): The preprocessed data ready for processing.

        Returns:
            Dict[str, Any]: A dictionary containing 'output' key with a string value representing
                            the result of data processing.

        Example:
            >>> service.process({"hello": "preprocessed world"})
            {'output': 'ExampleService output'}
        """
        return {"output": "ExampleService output"}

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess the processed data to format the output as required by downstream services or storage solutions.

        Args:
            data (Dict[str, Any]): The data to postprocess after it has been processed. Typically involves
                        final adjustments before sending the data to the next step or storing it.

        Returns:
            Dict[str, Any]: The postprocessed data, which in this case is the same as the input data.

        Example:
            >>> service.postprocess({'output': 'ExampleService output'})
            {'output': 'ExampleService output'}
        """
        return data
