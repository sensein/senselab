"""This module provides a wrapper for invoking various Large Language Models (LLMs).

Classes:
    LLM: A unified interface for interacting with various LLMs.
"""

import os
from typing import Dict, List, Optional

import torch
from openai import OpenAI


class LLM:
    """Wrapper for invoking various LLMs.

    This class provides a unified interface for interacting with LLMs,
    running on a vllm server at localhost:8000.

    Parameters:
    -----------
    model_name : str
        The name of the model to use. This is a required argument. Options:
        - "llama3-8b"
        - "llama3-70b"

    Methods:
    --------
    call(messages: List[Dict], system_instruction: Optional[str] = "",
          max_tokens: Optional[int] = 100, temperature: Optional[float] = 0.3) -> str:
        Invokes the model with the given message and system instruction.
    start_server(num_gpus: int, base_url: str) -> None:
        Starts the VLLM server with the specified number of GPUs.
    """

    def __init__(self, model_name: str) -> None:
        """Initializes the LLM instance with a model name and OpenAI client.

        Args:
            model_name (str): The name of the model to use.
        """
        self._model_name = self._get_model(model_name)

    def start_server(self, num_gpus: int, base_url: str = "http://localhost:8000/v1") -> None:
        """Starts the VLLM server with the specified number of GPUs.

        Args:
            num_gpus (int): The number of GPUs to use for tensor parallelism in the VLLM server.
            base_url (str): The base URL of the VLLM server, from which the host and port are extracted.
        """
        if torch.cuda.is_available():
            host, port = base_url.split("//")[1].split(":")
            port = port.split("/")[0]
            os.system(
                f"vllm serve {self._model_name} --host {host} --port {port} " f"--tensor-parallel-size {num_gpus}"
            )
            self._client = OpenAI(base_url=base_url, api_key="EMPTY")

        else:
            print("Please migrate to a compute node with GPU resources.")

    def call(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = "",
        max_tokens: Optional[int] = 100,
        temperature: Optional[float] = 0.3,
    ) -> str:
        """Invokes the model with a given message and system instruction.

        Args:
            messages (List[Dict[str, str]]): The conversation history.
            system_instruction (Optional[str]): The system instruction for the model.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature ranging between 0 and 2.

        Returns:
            str: The content of the model's response.
        """
        if system_instruction:
            system_message = {"role": "system", "content": system_instruction}
            messages.insert(0, system_message)

        completion = self._client.chat.completions.create(
            model=self._model_name, messages=messages, max_tokens=max_tokens, temperature=temperature
        )

        return completion.choices[0].message.content

    def _get_model(self, model: str) -> str:
        """Maps a model name to the corresponding model identifier.

        Args:
            model (str): The name of the model.

        Returns:
            str: The model identifier.

        Raises:
            ValueError: If the model name is unsupported.
        """
        model_mapping = {
            "llama3-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "llama3-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        }
        if model in model_mapping:
            return model_mapping[model]
        available_options = ",\n\t".join(model_mapping.keys())
        raise ValueError(f"Unsupported model. Available options: \n\t{available_options}")
