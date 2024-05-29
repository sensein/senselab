"""This module is for decorator functions."""

import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable


def get_response_time(func: Callable) -> Callable:
    """Decorator to measure and print response time information."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        print(f"Hello from {func.__name__}!")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        start_str = datetime.fromtimestamp(start_time).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        end_str = datetime.fromtimestamp(end_time).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )

        print(
            "start_str: ",
            start_str,
            "end_str: ",
            end_str,
            "duration: ",
            duration,
        )

        return result

    return wrapper
