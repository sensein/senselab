"""This module provides a wrapper for invoking various Large Language Models (LLMs).

Classes:
    LLM: A unified interface for interacting with various LLMs.
"""

import time
from subprocess import PIPE, Popen, check_output
from typing import List, Optional, Tuple

import requests
import torch
from openai import OpenAI
from transformers import AutoTokenizer  # type: ignore

from senselab.utils.data_structures.llm_response import LLMResponse
from senselab.utils.data_structures.script_line import ScriptLine


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
        - "gpt-4o"

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
        self._model_name, self._serving_url = self._get_model(model_name)

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        self._client = OpenAI(base_url=self._serving_url)

    def start_server(self, num_gpus: int, timeout: int = 300) -> Optional[Popen]:
        """Starts the VLLM server with the specified number of GPUs and logs the output.

        Args:
            num_gpus (int): The number of GPUs to use for tensor parallelism in the VLLM server.
            base_url (str): The base URL of the VLLM server, from which the host and port are extracted.
            timeout (int): Time, in seconds, to wait for the server to start before termination.

        Returns:
            Popen instance from subprocess module
        """
        if torch.cuda.is_available():
            host = check_output("hostname -I | awk '{print $1}'", shell=True, text=True).strip()
            port = 8000
            command = f"vllm serve {self._model_name} --host {host} --port {port} --tensor-parallel-size {num_gpus}"
            self._serving_url = f"http://{host}:{port}/v1"

            # Run the server in the background
            process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)

            # Wait for the server to start
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(self._serving_url, timeout=5)
                    if response.status_code == 200:
                        print("Server is up and running with a 200 response!")
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(5)
            else:
                print(f"Server did not respond with a 200 status code within {timeout} seconds.")
                process.terminate()
                return None

            self._client = OpenAI(base_url=self._serving_url, api_key="EMPTY")
            print(f"Serving on Host: {host}\tPort: {port}")
            return process
        else:
            print("Please migrate to a compute node with GPU resources.")
            return None

    def call(
        self,
        messages: List[ScriptLine],
        system_instruction: Optional[str] = "",
        max_tokens: Optional[int] = 100,
        temperature: Optional[float] = 0.3,
        measure: Optional[bool] = False,
    ) -> LLMResponse:
        """Invokes the model with a given message and system instruction.

        Args:
            messages (List[ScriptLine]): Conversation history.
            system_instruction (Optional[str]): The system instruction for the model.
            max_tokens (Optional[int]): Maximum number of tokens to generate.
            temperature (Optional[float]): Sampling temperature ranging between 0 and 2.
            measure (Optional[bool]): Whether to measure token counts and latency.

        Returns:
            LLMResponse: Named tuple with model's response, token counts, and latency (if measured).
        """
        openai_messages = [{"role": msg.speaker, "content": msg.text} for msg in messages]

        if system_instruction:
            system_message = {"role": "system", "content": system_instruction}  # type: ignore
            openai_messages.insert(0, system_message)  # type: ignore

        in_tokens = out_tokens = latency = None

        # initialize latency measurements
        if measure:
            in_tokens = sum(len(self._tokenizer.encode(message["content"])) for message in openai_messages)
            start_time = time.time()

        completion = self._client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = completion.choices[0].message.content

        if measure:
            latency = time.time() - start_time
            out_tokens = len(self._tokenizer.encode(content))

        return LLMResponse(content=content, latency=latency, in_tokens=in_tokens, out_tokens=out_tokens)

    def _get_model(self, model: str) -> Tuple[str, str]:
        """Maps a model name to the corresponding model identifier and url.

        Args:
            model (str): The name of the model.

        Returns:
            Tuple[str,str]: 1) model identifier 2) URL

        Raises:
            ValueError: If the model name is unsupported.
        """
        model_mapping = {
            "llama3-70b": ("meta-llama/Meta-Llama-3.1-70B-Instruct", "http://localhost:8000/v1"),
            "llama3-8b": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "http://localhost:8000/v1"),
            "gpt-4o": ("gpt-4o", "https://api.openai.com/v1"),
        }
        if model in model_mapping:
            return model_mapping[model]
        available_options = ",\n\t".join(model_mapping.keys())
        raise ValueError(f"Unsupported model. Available options: \n\t{available_options}")
