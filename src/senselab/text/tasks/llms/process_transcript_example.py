"""Example usage of llms directory to process AI responses from transcript."""

import os
import pickle
import time
from pathlib import Path
from typing import Generator, List

from tqdm import tqdm

from senselab.text.tasks.llms.llm import LLM
from senselab.utils.data_structures.script_line import ScriptLine
from senselab.utils.data_structures.transcript_input import TranscriptInput
from senselab.utils.data_structures.transcript_output import TranscriptOutput


def generate_ai_conversation(
    transcript_path: Path, prompt_path: Path, temp: float, model_name: str, cache_path: Path, llm: LLM
) -> TranscriptOutput:
    """Generates an AI conversation based on transcript and prompt data.

    Args:
        transcript_path (Path): Path to the transcript file.
        prompt_path (Path): Path to the prompt file.
        temp (float): Temperature parameter for the LLM.
        model_name (str): Name of the model to use.
        cache_path (Path): Path to store the cached responses.
        llm (LLM): instantiated model being used.

    Returns:
        TranscriptOutput: The resulting transcript and data as a `TranscriptOutput` object.
    """
    manager = TranscriptInput(transcript_path)

    with open(prompt_path, "r") as f:
        system_instruction = f.read()

    all_messages = manager.extract_response_opportunities()

    # Check if cached responses already exist
    if cache_path.exists():  # type: ignore
        with open(cache_path, "rb") as f:  # type: ignore
            responses = pickle.load(f)  # type: ignore
        print(f"Loaded cached responses for {transcript_path.name}")
    else:
        responses = [
            llm.call(messages=messages, system_instruction=system_instruction, max_tokens=200, temperature=temp)
            for messages in tqdm(all_messages, desc=f"Processing: {transcript_path.name}")
        ]

        with open(cache_path, "wb") as f:  # type: ignore
            pickle.dump(responses, f)  # type: ignore

    def response_gen() -> Generator[ScriptLine, None, None]:
        """Generates responses from the cached or newly generated data."""
        yield from responses

    gen = response_gen()

    conversation = []

    for i, message in enumerate(manager.scriptlines):
        content = message.text
        if message.speaker == "assistant":
            conversation.append({"speaker": "Tutor", "text": content})
            if i > 0:
                response_content = next(gen)
                conversation.append(response_content.to_dict())
        else:
            conversation.append({"speaker": "Student", "text": content})

    return TranscriptOutput(
        temp=temp, model=model_name, prompt=prompt_path.name, transcript=transcript_path.name, data=conversation
    )


def generate_all_transcripts(
    transcript_dir: Path, prompt_path: Path, temp: float, model_name: str, cache_dir: Path, llm: LLM
) -> List[TranscriptOutput]:
    """Generates AI conversations for all transcripts in a directory.

    Args:
        transcript_dir (Path): Directory containing transcript files.
        prompt_path (Path): Path to the prompt file.
        temp (float): Temperature parameter for the LLM.
        model_name (str): Name of the model to use.
        cache_dir (Path): Directory to store cached responses.
        llm (LLM): instantiated model being used.

    Returns:
        List[TranscriptOutput]: A list of `TranscriptOutput` objects.
    """
    outputs = []
    for transcript_path in transcript_dir.iterdir():
        cache_path = cache_dir / f"{transcript_path.stem}_cache.pkl"
        outputs.append(generate_ai_conversation(transcript_path, prompt_path, temp, model_name, cache_path, llm))
    return outputs


if __name__ == "__main__":
    transcript_dir = Path("/home/goshdam/to_do")
    prompt_path = Path("/home/goshdam/prompts/V2_1038.txt")
    temp = 0.5
    model_name = "llama3-70b"
    llm = LLM(model_name)

    timeout = 700  # in seconds
    poll_interval = 5  # interval to check in seconds
    start_time = time.time()

    while os.getenv("VLLM_STATUS") != "Running":
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(f"Timed out after {timeout} seconds waiting for VLLM_STATUS to be 'Running'.")
        time.sleep(poll_interval)

    output_path = Path("/home/goshdam/outputs/ai_outputs")
    cache_dir = Path("/home/goshdam/outputs/cache")

    cache_dir.mkdir(parents=True, exist_ok=True)

    outputs = generate_all_transcripts(transcript_dir, prompt_path, temp, model_name, cache_dir, llm)

    for output in outputs:
        output.save_to_json(output_path / f"{output.transcript}.json")

    print(f"Successfully saved all {len(outputs)} outputs to {output_path}")
