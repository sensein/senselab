"""This module provides functionality for chunking text into overlapping segments."""

from typing import List

from transformers import AutoTokenizer


def chunk_text(text: str, tokenizer: AutoTokenizer, max_length: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text (str): The input text to chunk.
        tokenizer (AutoTokenizer): The tokenizer to use.
        max_length (int): Maximum length of each chunk.
        overlap (int): Overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i : i + max_length - 2]  # -2 for [CLS] and [SEP]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks
