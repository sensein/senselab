"""This module provides utilities for logging."""

import logging

# Configure the logger
logger = logging.getLogger("senselab")
logger.setLevel(logging.DEBUG)

# Create a console handler and set the level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add the formatter to the handler
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)
