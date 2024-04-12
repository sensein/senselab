"""This module defines an abstract base component for schema validation.

It enforces input and output validation against specified JSON schemas for deriving classes.
Utilities for schema validation, UUID generation based on data, and execution time measurement
for methods are included. The purpose is to provide a structured approach to implementing components
with validated inputs and outputs according to predefined schemas.
"""

import hashlib
import json
import os
import time
import uuid
from abc import ABC
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

from jsonschema import ValidationError, validate


class AbstractComponent(ABC):
    """Abstract base class for components with JSON schema validation.

    This class provides mechanisms for validating inputs and outputs against specified JSON schemas.
    It includes methods for reading schemas, validating data, generating UUIDs from data, and
    measuring method execution times.
    """

    def __init__(
        self,
        base_dir: str,
        input_schema_file: str = "schemas/__FUNCTION_NAME_PLACEHOLDER__/input.json",
        output_schema_file: str = "schemas/__FUNCTION_NAME_PLACEHOLDER__/output.json",
    ) -> None:
        """Initializes the component with paths to input and output JSON schemas."""
        self.base_dir = base_dir
        self.base_input_schema = input_schema_file
        self.base_output_schema = output_schema_file

    @staticmethod
    def schema_validator(func: Callable) -> Callable:
        """Decorator to validate input and output against schemas."""
        @wraps(func)
        def wrapper(self: "AbstractComponent", *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            input_schema = self.read_json_schema(os.path.join(self.base_dir, self.base_input_schema.replace("__FUNCTION_NAME_PLACEHOLDER__", func.__name__)))
            output_schema = self.read_json_schema(os.path.join(self.base_dir, self.base_output_schema.replace("__FUNCTION_NAME_PLACEHOLDER__", func.__name__)))

            # Validate input
            input_data = kwargs.get("input_data") or (args[0] if args else {})
            if input_schema is not None:
                try:
                    validate(instance=input_data, schema=input_schema)
                except ValidationError as e:
                    raise ValueError(f"Input validation error: {e}") from e

            # Execute the function
            result = func(self, *args, **kwargs)

            # Validate output
            if output_schema is not None:
                try:
                    validate(instance=result, schema=output_schema)
                except ValidationError as e:
                    raise ValueError(f"Output validation error: {e}") from e

            result["input"] = input_data
            return result

        return wrapper

    @staticmethod
    def read_json_schema(file_path: str) -> Optional[Dict[str, Any]]:
        """Reads and returns a JSON schema from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file: {file_path}")
        except Exception as e:  # pylint: disable=broad-except
            # Catch-all for any other unexpected exceptions, to be used sparingly
            print(f"An unexpected error occurred while reading {file_path}: {e}")
        return None

    @staticmethod
    def get_data_uuid(data: Dict[str, Any]) -> uuid.UUID:
        """Generates a UUID for a given data based on its SHA-256 hash."""
        data_str = json.dumps(data, sort_keys=True)
        hash_obj = hashlib.sha256(data_str.encode())
        hash_bytes = hash_obj.digest()[:16]
        return uuid.UUID(bytes=hash_bytes)

    @staticmethod
    def get_response_time(func: Callable) -> Callable:
        """Decorator to measure and append response time information to a method's result."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time

            start_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S.%f")
            end_str = datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S.%f")

            if isinstance(result, dict):  # Ensure result is a dict before adding time data.
                result["time"] = {"start": start_str, "end": end_str, "duration": f"{duration:.6f}s"}

            return result

        return wrapper
