"""This module contains the definition of the LLMResponse object."""

from collections import namedtuple

LLMResponse = namedtuple("LLMResponse", ["content", "latency", "in_tokens", "out_tokens"])
