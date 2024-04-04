"""This module implements an example service for the task."""

from typing import Any, Dict

from ...abstract_service import AbstractService


class Service(AbstractService):
    """Example service that extends AbstractService."""

    NAME: str = "ExampleService"

    def __init__(self, configs: Dict[str, Any]) -> None: # noqa: ANN401
        """Initialize the service with given configurations.

        Args:
            configs: A dictionary of configurations for the service.
        """
        super().__init__()

    def preprocess(self, data: Any) -> Any: # noqa: ANN401
        """Preprocess input data. Implementation can be customized.

        Args:
            data: The input data to preprocess.

        Returns:
            The preprocessed data.
        """
        return super().preprocess(data)

    def process(self, data: Any) -> Dict[str, Any]: # noqa: ANN401
        """Process input data. Custom implementation for ExampleService.

        Args:
            data: The input data to process.

        Returns:
            A dictionary containing 'output' key with a sample output.
        """
        return {"output": "ExampleService output"}

    def postprocess(self, data: Any) -> Any: # noqa: ANN401
        """Postprocess processed data. Implementation can be customized.

        Args:
            data: The data to postprocess.

        Returns:
            The postprocessed data.
        """
        return super().postprocess(data)