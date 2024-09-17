"""Example usage of llms directory to process AI responses from transcript."""

import pickle
import sys
from pathlib import Path
from typing import Generator, List

import pandas as pd
from tqdm import tqdm

from senselab.text.tasks.llms.llm import LLM
from senselab.text.tasks.llms.transcript_manager import Transcript
from senselab.utils.data_structures.llm_response import LLMResponse
from senselab.utils.data_structures.transcript_output import TranscriptOutput


def generate_ai_conversation(
    transcript_path: Path, prompt_path: Path, temp: float, model_name: str, measure: bool, cache_path: Path, llm: LLM
) -> TranscriptOutput:
    """Generates an AI conversation based on transcript and prompt data.

    Args:
        transcript_path (Path): Path to the transcript file.
        prompt_path (Path): Path to the prompt file.
        temp (float): Temperature parameter for the LLM.
        model_name (str): Name of the model to use.
        measure (bool): Whether to measure performance (e.g., tokens, latency).
        cache_path (Path): Path to store the cached responses.
        llm (LLM): instantiated model being used.

    Returns:
        TranscriptOutput: The resulting transcript and data as a `TranscriptOutput` object.
    """
    manager = Transcript(transcript_path)

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
            llm.call(
                messages=messages,
                system_instruction=system_instruction,
                max_tokens=200,
                temperature=temp,
                measure=measure,
            )
            for messages in tqdm(all_messages, desc=f"Processing: {transcript_path.name}")
        ]

        with open(cache_path, "wb") as f:  # type: ignore
            pickle.dump(responses, f)  # type: ignore

    def response_gen() -> Generator[LLMResponse, None, None]:
        """Generates responses from the cached or newly generated data."""
        yield from responses

    gen = response_gen()

    df = pd.DataFrame(columns=["student", "teacher", "AI", "in_tokens", "out_tokens", "latency"])
    j = 0  # student-response pair number
    for i, message in enumerate(manager.scriptlines):
        content = message.text
        if message.speaker == "assistant":
            if i > 0:
                response_content = next(gen)
                df.at[j, "teacher"] = content
                df.at[j, "AI"] = response_content.content
                if measure:
                    df.at[j, "in_tokens"] = response_content.in_tokens
                    df.at[j, "out_tokens"] = response_content.out_tokens
                    df.at[j, "latency"] = response_content.latency
            else:
                df.at[j, "teacher"] = content
            j += 1
        else:
            df.at[j, "student"] = content

    df.fillna("", inplace=True)

    return TranscriptOutput(
        temp=temp, model=model_name, prompt=prompt_path.name, transcript=transcript_path.name, data=df
    )


def generate_all_transcripts(
    transcript_dir: Path, prompt_path: Path, temp: float, model_name: str, measure: bool, cache_dir: Path, llm: LLM
) -> List[TranscriptOutput]:
    """Generates AI conversations for all transcripts in a directory.

    Args:
        transcript_dir (Path): Directory containing transcript files.
        prompt_path (Path): Path to the prompt file.
        temp (float): Temperature parameter for the LLM.
        model_name (str): Name of the model to use.
        measure (bool): Whether to measure performance (e.g., tokens, latency).
        cache_dir (Path): Directory to store cached responses.
        llm (LLM): instantiated model being used.

    Returns:
        List[TranscriptOutput]: A list of `TranscriptOutput` objects.
    """
    outputs = []
    for transcript_path in transcript_dir.iterdir():
        cache_path = cache_dir / f"{transcript_path.stem}_cache.pkl"
        outputs.append(
            generate_ai_conversation(transcript_path, prompt_path, temp, model_name, measure, cache_path, llm)
        )
    return outputs


if __name__ == "__main__":
    transcript_dir = Path("/home/goshdam/transcripts")
    prompt_path = Path("/home/goshdam/prompts/V2_1076.txt")
    temp = 0.5
    model_name = "llama3-70b"

    llm = LLM(model_name)

    if sys.argv[1] == "run":
        output_path = Path("/home/goshdam/outputs/outputs_llama.pkl")
        cache_dir = Path("/home/goshdam/outputs/cache")
        measure = True

        cache_dir.mkdir(parents=True, exist_ok=True)

        outputs = generate_all_transcripts(transcript_dir, prompt_path, temp, model_name, measure, cache_dir, llm)

        with open(output_path, "wb") as f:
            pickle.dump(outputs, f)

        print(f"Successfully saved all {len(outputs)} outputs to {output_path}")

    elif sys.argv[1] == "server":
        llm.start_server(num_gpus=4)
