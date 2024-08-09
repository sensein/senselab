"""Example usage of llms directory to process AI responses from transcript."""

import sys
from pathlib import Path
from typing import Generator

from tqdm import tqdm

from senselab.text.tasks.llms.data_ingest import MessagesManager
from senselab.text.tasks.llms.llm import LLM

if __name__ == "__main__":
    manager = MessagesManager(Path("/home/goshdam/sample_transcript.json"))
    llm = LLM("llama3_70b")

    # manager.print_human_readable(manager.messages)

    SYSTEM_INSTRUCTION = (
        "You are a friendly, supportive tutoring assistant for a child, "
        "helping them to learn vocabulary, "
        "interspersed with friendly human interaction."
    )

    all_messages = manager.extract_response_opportunities()

    responses = [
        llm.call(messages=messages, system_instruction=SYSTEM_INSTRUCTION, max_tokens=200, temperature=0.4)
        for messages in tqdm(all_messages, file=sys.stderr)
    ]

    def response_gen() -> Generator[str, None, None]:
        """Generator function that yields responses from the responses list.

        Yields:
            str: Each response in the responses list.
        """
        yield from responses

    gen = response_gen()

    for i, message in enumerate(manager.messages):
        content = message["content"]

        if message["role"] == "assistant":
            if i > 0:
                response_content = next(gen)
                print(f"Teacher:\t{content}\n\nAI:\t{response_content}\n\n")
            else:
                print(f"Teacher:\t{content}\n\n")
        else:
            print(f"Student:\t{content}\n\n")
