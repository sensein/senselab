# LLMs


## Overview
This module provides the API for making LLM calls in senselab.

This project focuses on ingesting and processing data, utilizing language models, and handling transcript data. It provides utilities for parsing unstructured text and generating meaningful insights using a combination of custom functions and pre-trained models.

## Structure
The project contains the following main components:

transcript_manager.py: Handles data ingestion and preprocessing tasks.

llm.py: Integrates language model-related functionality.

process_transcript_example.py: Demonstrates how to process transcript data, using methods provided in this package.


## transcript_manager.py

The `transcript_manager` module provides a data manager for handling interactions with a large language model (LLM). It allows the loading of transcripts, converting JSON data into scriptline objects, and extracting conversation data in a format that can be used to query potential AI responses.

### Class: `Transcript`

The `Transcript` class manages message data for interactions with a LLM. It provides methods to load transcripts, convert JSON transcript data into a usable format, and extract conversation segments for AI response opportunities. You will use it by initializing it on a valid transcript path. That transcript data is loaded in and stored as a list of scriptlines. These can then be printed in a readable format, you can see the number of tokens in the transcript, and the data is ready to be called by the LLM class in llm.py.

### Attributes:
 **`scriptlines (List[ScriptLine])`**: A list of `ScriptLine` objects representing the conversation. See documentionation in senselab/utils/data_structures/script_line.py.

### Methods

#### 1. `__init__(self, transcript_path: Path) -> None`

Initializes the `MessagesManager` with a path to the JSON transcript file. Loads the transcript and converts it into scriptline objects.

**Parameters:**
- `transcript_path (Path)`: The path to the JSON transcript file.


#### 2. `print_human_readable(self) -> None`

Prints the scriptlines attribute in a human-readable format, where each message is displayed with the speaker and content.


#### 3. `extract_response_opportunities(self) -> List[List[Dict[str, str]]]`

Extracts consecutive sublists from the message list, ending after every 'user' response. These sublists can be used to compare AI responses to human responses over the course of a conversation.

**Returns:**
- `List[List[Dict[str, str]]]`: A list of consecutive sublists of messages, each ending with a 'user' message.

Example:
```python
response_opportunities = manager.extract_response_opportunities()
```


#### 4. `convert_json_to_scriptlines(self, json_obj: Dict) -> List[ScriptLine]`

Converts transcript segments from a JSON object into a list of `ScriptLine` objects, where each scriptline contains the text and speaker. This method also maps "teacher" to "assistant" and "kid" to "user".

**Parameters:**
- `json_obj (Dict)`: The JSON object containing the conversation segments.

    The input JSON object should have the following structure:
    ```
                {
                    "segments": [
                        {
                            "start": <float>,
                            "end": <float>,
                            "text": <string>,
                            "words": [
                                {
                                    "word": <string>,
                                    "start": <float>,
                                    "end": <float>,
                                    "score": <float>,
                                    "speaker": <string> [kid|teacher]
                                },
                                ...
                            ],
                            "speaker": <string> [kid|teacher]
                        },
                        ...
                    ]
                }
    ```

**Returns:**
- `List[ScriptLine]`: A list of `ScriptLine` objects representing the conversation.

**Raises:**
- `ValueError`: If the input JSON structure is invalid or contains an unknown speaker role.


#### 5. `get_num_tokens(self) -> int`

Returns the total number of tokens in the stored scriptlines. Uses OpenAI GPT-4o tokenizer.

**Returns:**
- `int`: Number of tokens in the transcript.
---

## Example Usage

```python
from pathlib import Path
from transcript_manager import Transcript

# Initialize the manager with the path to a transcript
transcript = Transcript(Path("transcript.json"))

transcript.print_human_readable(messages)

# Extract response opportunities from the conversation
response_opportunities = transcript.extract_response_opportunities()

# Get the number of tokens used in the conversation
num_tokens = transcript.get_num_tokens()

print(f"Total tokens: {num_tokens}")
```
---



### Class: `LLM`

The `LLM` class abstracts the interaction with different large language models (LLMs) such as `llama3-8b`, `llama3-70b`, and `gpt-4o`. The `LLM` class is designed to start a server for model interaction, handle inputs, and produce outputs based on the model selected.

Note that some models (like `gpt-4o`) are called through external endpoints, while others (like `llama3-8b`) are hosted locally and need to be initialized first. Depending on the model, the `call` function sends requests either to an external server or a locally hosted server.

#### Attributes:
 **`_model_name (str)`**: The name of the model being used (e.g., `"llama3-70b"`).
 **`_base_url (str)`**: The URL where the server is hosted.
 **`_tokenizer (AutoTokenizer)`**: Tokenizer for the selected model.

---

#### Methods

##### 1. `__init__(self, model_name: str) -> None`

Initializes the `LLM` instance with the specified model name, setting up the necessary client and tokenizer.

**Parameters:**
- `model_name (str)`: The name of the model to initialize.
---

##### 2. `start_server(self, num_gpus: int, base_url: str = "http://localhost:8000/v1") -> Popen`

Starts a VLLM server with the specified number of GPUs, serving the specified local model. The server enables tensor parallelism to manage large models efficiently.

**Parameters:**
- `num_gpus (int)`: The number of GPUs to initialize the model with.
- `base_url (Optional[str])`: The URL where the server is to be hosted. Default is `"http://localhost:8000/v1"`.

**Returns:**
- `Popen`: A `Popen` object representing the running server process.
---

##### 3. `call(self, messages: List[Dict], system_instruction: Optional[str] = "", max_tokens: Optional[int] = 100, temperature: Optional[float] = 0.3, measure: Optional[bool] = False) -> LLMResponse`

Sends a series of messages to the model server and returns the model’s output. The `system_instruction` parameter provides additional context for the model, while the `measure` flag allows for token and latency measurements.

**Parameters:**
- `messages (List[Dict])`: List of messages in the conversation. Each message is a dictionary with `role` and `content` keys.
- `system_instruction (Optional[str])`: Instruction for the system. Default is an empty string.
- `max_tokens (Optional[int])`: Maximum number of tokens for the output.
- `temperature (Optional[float])`: Sampling temperature, controlling randomness. Default is `0.3`.
- `measure (Optional[bool])`: If `True`, measures latency and token usage. Default is `False`.

**Returns:**
- `LLMResponse`: An object containing the response content, latency, and token information (if measure flag set to True). See documentation at senselab/utils/data_structures/llm_response.py.

### Example Usage

```
llm = LLM("llama3-70b")

llm.start_server(num_gpus=4)

messages = [{"role": "user", "content": "Tell me a joke."}]
response = llm.call(messages, system_instruction="You are a friendly assistant")
print(response.content)
```
---
