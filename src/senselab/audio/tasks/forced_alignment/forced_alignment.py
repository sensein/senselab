"""Align function based on WhisperX implementation."""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.forced_alignment.constants import (
    DEFAULT_ALIGN_MODELS_HF,
    LANGUAGES_WITHOUT_SPACES,
    MINIMUM_SEGMENT_SIZE,
    PUNKT_ABBREVIATIONS,
    SAMPLE_RATE,
)
from senselab.audio.tasks.forced_alignment.data_structures import (
    Point,
    Segment,
    SingleSegment,
)
from senselab.audio.tasks.preprocessing import extract_segments, pad_audios
from senselab.utils.data_structures import DeviceType, HFModel, Language, ScriptLine, _select_device_and_dtype

try:
    from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

    NLTK_AVAILABLE = True
except ModuleNotFoundError:
    NLTK_AVAILABLE = False


def _preprocess_segments(
    transcript: List[SingleSegment],
    model_dictionary: Dict[str, int],
    model_lang: Language,
    print_progress: bool,
    combined_progress: bool,
) -> List[SingleSegment]:
    """Preprocess segments by cleaning characters and handling spaces.

    Args:
        transcript (List[SingleSegment]): The transcription segments.
        model_dictionary (Dict[str, int]): The model's character dictionary.
        model_lang (Language): The language configuration of the model.
        print_progress (bool): If True, print progress updates.
        combined_progress (bool): If True, combine progress into a single percentage.

    Returns:
        List[SingleSegment]: The preprocessed transcription segments.
    """
    if not NLTK_AVAILABLE:
        raise ModuleNotFoundError(
            "`nltk` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    total_segments = len(transcript)

    for sdx, segment in enumerate(transcript):
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        if model_lang.alpha_2 not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = [text]

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.upper()
            if model_lang.alpha_2 not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            if cdx < num_leading or cdx > len(text) - num_trailing - 1:
                continue
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = []
        for wdx, wrd in enumerate(per_word):
            if any(c in model_dictionary.keys() for c in wrd):
                clean_wdx.append(wdx)

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["sentence_spans"] = sentence_spans

    return transcript


def _can_align_segment(
    segment: SingleSegment, model_dictionary: Dict[str, int], t1: float, max_duration: float
) -> bool:
    """Check if a segment can be aligned based on content and timing.

    Args:
        segment (SingleSegment): The segment to check.
        model_dictionary (Dict[str, int]): Model character dictionary.
        t1 (float): Segment start time.
        max_duration (float): Maximum allowed audio duration.

    Returns:
        bool: True if alignable, False otherwise.
    """
    # Check if segment has clean characters
    if segment["clean_char"] is None or len(segment["clean_char"]) == 0:
        return False

    # Check if segment is within duration bounds
    if t1 >= max_duration:
        return False

    # Check if all clean chars are in model dictionary
    for char in segment["clean_char"]:
        if char not in model_dictionary:
            return False

    return True


def _get_prediction_matrix(
    model: torch.nn.Module,
    waveform_segment: torch.Tensor,
    lengths: Optional[torch.Tensor],
    model_type: str,
    device: DeviceType,
) -> torch.Tensor:
    """Obtain the prediction (emission) matrix from the model.

    Args:
        model (torch.nn.Module): The alignment model.
        waveform_segment (torch.Tensor): Audio segment tensor.
        lengths (Optional[torch.Tensor]): Lengths of audio segments.
        model_type (str): 'torchaudio' or 'huggingface'.
        device (DeviceType): Device for computation.

    Returns:
        torch.Tensor: The log-softmax emissions.
    """
    with torch.inference_mode():
        if model_type == "torchaudio":
            emissions, _ = model(waveform_segment.to(device.value), lengths=lengths)
        elif model_type == "huggingface":
            emissions = model(waveform_segment.to(device.value)).logits
        else:
            raise NotImplementedError(f"Align model of type {model_type} not supported.")

        emissions = torch.log_softmax(emissions, dim=-1)

    return emissions


def _assign_timestamps(
    segment: SingleSegment, char_segments: List[Segment], ratio: float, t1: float, model_lang: Language
) -> Dict[str, Any]:
    """Assign timestamps to each character, grouping them into words and sentences.

    Args:
        segment (SingleSegment): The segment containing text and timing information.
        char_segments (List[Segment]): A list of character-level alignment segments.
        ratio (float): The ratio used to convert frame indices to timestamps.
        t1 (float): The starting time of the segment in seconds.
        model_lang (Language): The language configuration, used to determine word boundaries.

    Returns:
        Dict[str, Any]: A nested structure with aligned sentence, word, and character chunks.
    """
    text = segment["text"]
    start = segment["start"]
    end = segment["end"]

    clean_cdx = segment["clean_cdx"] or []

    aligned_segment_dict: Dict[str, Any] = {"text": text, "timestamps": [start, end], "chunks": []}
    current_word_dict: Dict[str, Any] = {"text": "", "timestamps": [], "chunks": []}
    current_subsegment_dict: Dict[str, Any] = {"text": "", "timestamps": [], "chunks": []}

    for cdx, char in enumerate(text):
        if cdx in clean_cdx:
            char_seg_index = clean_cdx.index(cdx)
            char_seg = char_segments[char_seg_index]
            char_dict = {"text": char, "timestamps": [round(x * ratio + t1, 3) for x in [char_seg.start, char_seg.end]]}
            current_word_dict["chunks"].append(char_dict)
            current_word_dict["text"] += char

        is_end_of_text = cdx == len(text) - 1
        next_is_space = (not is_end_of_text) and (text[cdx + 1] == " ")
        if (model_lang.alpha_2 in LANGUAGES_WITHOUT_SPACES) or is_end_of_text or next_is_space:
            if current_word_dict["chunks"]:
                merged_timestamps = [t for c in current_word_dict["chunks"] for t in c["timestamps"]]
                current_word_dict["timestamps"] = [min(merged_timestamps), max(merged_timestamps)]
            current_subsegment_dict["text"] += current_word_dict["text"]
            current_subsegment_dict["chunks"].append(current_word_dict)
            current_word_dict = {"text": "", "timestamps": [], "chunks": []}

        if char == "." or is_end_of_text:
            if current_subsegment_dict["chunks"]:
                merged_timestamps = [t for c in current_subsegment_dict["chunks"] for t in c["timestamps"]]
                current_subsegment_dict["timestamps"] = [min(merged_timestamps), max(merged_timestamps)]
            aligned_segment_dict["chunks"].append(current_subsegment_dict)
            current_subsegment_dict = {"text": "", "timestamps": [], "chunks": []}

    for subsegment in aligned_segment_dict["chunks"]:
        subsegment["text"] = subsegment["text"].strip() + "."
        for word in subsegment["chunks"]:
            word["text"] = word["text"].strip()

    aligned_segment_dict["timestamps"][0] = aligned_segment_dict["chunks"][0]["timestamps"][0]
    aligned_segment_dict["timestamps"][1] = aligned_segment_dict["chunks"][-1]["timestamps"][1]
    return aligned_segment_dict


def _align_single_segment(
    segment: SingleSegment,
    model: torch.nn.Module,
    model_dictionary: Dict[str, int],
    model_lang: Language,
    model_type: str,
    audio: Audio,
    device: DeviceType,
    t1: float,
    t2: float,
) -> Optional[dict]:
    """Align a single transcription segment by extracting audio, running inference, and decoding alignments.

    Args:
        segment (SingleSegment): The segment with text and timing.
        model (torch.nn.Module): The trained alignment model.
        model_dictionary (Dict[str, int]): Mapping of characters to tokens.
        model_lang (Language): The language configuration.
        model_type (str): 'huggingface' or 'torchaudio'.
        audio (Audio): The audio data.
        device (DeviceType): The device to run inference on.
        t1 (float): Segment start time.
        t2 (float): Segment end time.

    Returns:
        dict or None: A dictionary with aligned segment info, or None if alignment fails.
    """
    text_clean = "".join(segment["clean_char"] or [])
    tokens = [model_dictionary[c] for c in text_clean]
    extracted_segment = extract_segments([(audio, [(t1, t2)])])[0][0]
    lengths = torch.tensor([extracted_segment.waveform.shape[-1]])
    waveform_segment = pad_audios([extracted_segment], MINIMUM_SEGMENT_SIZE)[0].waveform
    emissions = _get_prediction_matrix(
        model=model, waveform_segment=waveform_segment, lengths=lengths, model_type=model_type, device=device
    )
    emission = emissions[0].cpu().detach()

    blank_id = 0
    for char, code in model_dictionary.items():
        if char in ["[pad]", "<pad>"]:
            blank_id = code

    trellis = _get_trellis(emission, tokens, blank_id)
    path = _backtrack(trellis, emission, tokens, blank_id)

    if path is None:
        print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
        return None

    char_segments = _merge_repeats(path, text_clean)
    duration = t2 - t1
    ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)
    aligned_segment = _assign_timestamps(segment, char_segments, ratio, t1, model_lang)
    return aligned_segment


def _get_trellis(emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> torch.Tensor:
    """Compute the trellis matrix for alignment.

    Args:
        emission (torch.Tensor): Emission log probabilities.
        tokens (List[int]): Target token sequence.
        blank_id (int): Blank token ID.

    Returns:
        torch.Tensor: The computed trellis matrix.
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis


def _backtrack(
    trellis: torch.Tensor, emission: torch.Tensor, tokens: List[int], blank_id: int = 0
) -> Optional[List[Point]]:
    """Backtrack through the trellis to find the best path.

    Args:
        trellis (torch.Tensor): The trellis matrix.
        emission (torch.Tensor): The emission matrix.
        tokens (List[int]): Target tokens.
        blank_id (int): Blank token ID.

    Returns:
        Optional[List[Point]]: The best path of Points or None if not found.
    """
    j = trellis.size(1) - 1
    t_start = int(torch.argmax(trellis[:, j]).item())

    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        path.append(Point(j - 1, t - 1, prob))

        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return path[::-1]


def _merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    """Merge repeated tokens in the alignment path.

    Args:
        path (List[Point]): The alignment path points.
        transcript (str): Original transcript string.

    Returns:
        List[Segment]: Segments after merging repeated tokens.
    """
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


def _align_segments(
    transcript: List[SingleSegment],
    model: torch.nn.Module,
    model_dictionary: Dict[str, int],
    model_lang: Language,
    model_type: str,
    audio: Audio,
    device: DeviceType,
    max_duration: float,
) -> List[ScriptLine | None]:
    """Align each transcription segment with the audio and return aligned segments.

    Args:
        transcript (List[SingleSegment]): The segments to align.
        model (torch.nn.Module): The alignment model.
        model_dictionary (Dict[str, int]): Model char-to-token dictionary.
        model_lang (Language): Language configuration.
        model_type (str): 'huggingface' or 'torchaudio'.
        audio (Audio): The audio data.
        device (DeviceType): Device for inference.
        max_duration (float): Max allowed audio duration.

    Returns:
        List[ScriptLine | None]: A list of aligned segments.
    """
    aligned_segments: List[ScriptLine | None] = []

    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        if _can_align_segment(segment, model_dictionary, t1, max_duration):
            aligned_segment = _align_single_segment(
                segment, model, model_dictionary, model_lang, model_type, audio, device, t1, t2
            )
        else:
            print(f'Failed to align segment ("{segment["text"]}"), skipping...')
            aligned_segment = None

        aligned_segments.append(ScriptLine.from_dict(aligned_segment) if aligned_segment else None)
    return aligned_segments


def _align_transcription(
    transcript: List[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: Dict[str, Any],
    audio: Audio,
    device: DeviceType,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> List[ScriptLine | None]:
    """Align the given transcription segments against the audio using the model.

    Args:
        transcript (List[SingleSegment]): Segments to align.
        model (torch.nn.Module): The alignment model.
        align_model_metadata (Dict[str, Any]): Model dictionary, language, and type info.
        audio (Audio): Audio data.
        device (DeviceType): Device for inference.
        print_progress (bool): If True, print progress updates.
        combined_progress (bool): If True, combine progress percentages.

    Returns:
        List[ScriptLine | None]: Aligned transcription segments.
    """
    max_duration = audio.waveform.shape[1] / audio.sampling_rate
    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    transcript = _preprocess_segments(transcript, model_dictionary, model_lang, print_progress, combined_progress)
    aligned_segments = _align_segments(
        transcript=transcript,
        model=model,
        model_dictionary=model_dictionary,
        model_lang=model_lang,
        model_type=model_type,
        audio=audio,
        device=device,
        max_duration=max_duration,
    )
    return aligned_segments


def remove_chunks_by_level(
    scriptline: ScriptLine, level: int, keep_lower: bool = True
) -> Union[List[ScriptLine], ScriptLine, None]:
    """Recursively removes chunks from a ScriptLine object at a given depth level.

    ScriptLine objects can contain nested chunks representing different levels of speech
    (e.g. utterance -> sentence -> word -> character). This function traverses the chunk hierarchy
    and removes chunks at the specified level, optionally preserving or removing lower levels.

    For example, with level=1 and keep_lower=True, it will remove word-level chunks but keep
    character-level chunks. With keep_lower=False, it removes both word and character chunks.

    Args:
        scriptline (Optional[ScriptLine]): The root ScriptLine object to modify.
        level (int): The depth level at which to remove chunks (0=utterance, 1=sentence, 2=word, 3=char).
        keep_lower (bool): Whether to keep chunks at levels below the target level.

    Returns:
        Union[List[ScriptLine], ScriptLine, None]:
            - If level == 0, returns the list of chunks from the root ScriptLine.
            - If level > 0, modifies scriptline in place and returns the modified ScriptLine.
            - Returns None if scriptline is None or all chunks are removed.

    Raises:
        ValueError: If scriptline argument is None.
    """
    if scriptline is None:
        raise ValueError("scriptline argument cannot be None")

    if level == 0:
        if not keep_lower:
            return None
        return scriptline.chunks

    if scriptline.chunks:
        updated_chunks: List[ScriptLine] = []
        for chunk in scriptline.chunks:
            updated = remove_chunks_by_level(chunk, level - 1, keep_lower)
            if isinstance(updated, list):
                updated_chunks.extend(updated)
            elif updated is not None:
                updated_chunks.append(updated)

        scriptline.chunks = updated_chunks

    return scriptline


def flatten_script_lines(nested: list[ScriptLine | None]) -> List[ScriptLine | None]:
    """Flattens a list by unpacking any inner lists of `ScriptLine` while keeping `None` values.

    Args:
        nested (list[ScriptLine | None]):
            A list that may contain `ScriptLine | None` elements, or lists of `ScriptLine`.

    Returns:
        List[ScriptLine | None]: A flattened list where all `ScriptLine` objects are at the top level,
        with `None` values preserved.
    """
    flattened: List[ScriptLine | None] = []

    for item in nested:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    return flattened


def filter_aligned_script_lines(
    aligned_script_lines: List[List[ScriptLine | None]], levels_to_keep: Dict[str, bool]
) -> list[list[ScriptLine | None]]:
    """Filters aligned script lines by removing specific chunk levels based on `levels_to_keep`.

    Args:
        aligned_script_lines (List[List[ScriptLine | None]]):
            A list of lists containing `ScriptLine | None` elements.
        levels_to_keep (Dict[str, bool]):
            A dictionary specifying which levels to retain:
            - "utterance" (bool): Whether to keep utterance-level chunks.
            - "word" (bool): Whether to keep word-level chunks.
            - "char" (bool): Whether to keep character-level chunks.

    Returns:
        list[list[ScriptLine | None]]: The filtered aligned script lines.
    """
    for i, scriptline_list in enumerate(aligned_script_lines):
        updated_scriptline_list: list[ScriptLine | None] = []
        for j, scriptline in enumerate(scriptline_list):
            if scriptline is None or isinstance(scriptline, list):
                continue

            updated_scriptline: Union[list[ScriptLine], ScriptLine, None] = scriptline
            if not levels_to_keep["word"] and not levels_to_keep["char"]:
                updated_scriptline = remove_chunks_by_level(scriptline, level=2, keep_lower=False)
            elif not levels_to_keep["word"] and levels_to_keep["char"]:
                updated_scriptline = remove_chunks_by_level(scriptline, level=2)
            elif levels_to_keep["word"] and not levels_to_keep["char"]:
                updated_scriptline = remove_chunks_by_level(scriptline, level=3)

            if not levels_to_keep["utterance"]:
                updated_scriptline = remove_chunks_by_level(scriptline, level=0)

            if isinstance(updated_scriptline, list):
                updated_scriptline_list.extend(updated_scriptline)
            elif updated_scriptline is not None:
                updated_scriptline_list.append(updated_scriptline)

        aligned_script_lines[i] = updated_scriptline_list
    return [flatten_script_lines(scriptline_list) for scriptline_list in aligned_script_lines]


def align_transcriptions(
    audios_and_transcriptions_and_language: List[Tuple[Audio, ScriptLine, Language]],
    levels_to_keep: Dict = {"utterance": False, "word": False, "char": False},
) -> List[List[ScriptLine | None]]:
    """Align multiple transcriptions with their respective audios using a wav2vec2.0 model.

    Args:
        audios_and_transcriptions_and_language (List[Tuple[Audio, ScriptLine, Language]]):
            Each tuple contains an Audio object, a ScriptLine with transcription,
            and an optional Language (default is English).
        levels_to_keep (Dict): Levels of transcription to keep in output.

    Returns:
        List[List[ScriptLine | None]]: A list of aligned results for each audio.
    """
    aligned_script_lines: list[list[ScriptLine | None]] = []
    loaded_processors_and_models = {}
    device = _select_device_and_dtype()[0]

    for recording in audios_and_transcriptions_and_language:
        audio, transcription, language = (*recording, None)[:3]
        if language is None:
            language = Language(language_code="en")

        model_dict = DEFAULT_ALIGN_MODELS_HF.get(language.language_code, DEFAULT_ALIGN_MODELS_HF["en"])
        model_variant: HFModel = HFModel(path_or_uri=model_dict["path_or_uri"], revision=model_dict["revision"])
        if model_variant.path_or_uri not in loaded_processors_and_models:
            processor = Wav2Vec2Processor.from_pretrained(model_variant.path_or_uri)
            model = Wav2Vec2ForCTC.from_pretrained(model_variant.path_or_uri).to(device.value)
            loaded_processors_and_models[model_variant.path_or_uri] = (processor, model)

        processor, model = loaded_processors_and_models[model_variant.path_or_uri]

        if audio.sampling_rate != SAMPLE_RATE:
            raise ValueError(f"{audio.sampling_rate} rate is not equal to {SAMPLE_RATE}.")

        start = transcription.start if transcription.start is not None else 0.0
        end = transcription.end if transcription.end is not None else audio.waveform.shape[1] / audio.sampling_rate
        text = transcription.text if transcription.text is not None else ""

        segments = [
            SingleSegment(
                start=start, end=end, text=text, clean_char=None, clean_cdx=None, clean_wdx=None, sentence_spans=None
            )
        ]

        with threadpool_limits(limits=1, user_api="blas"):
            alignment = _align_transcription(
                transcript=segments,
                model=model,
                align_model_metadata={
                    "dictionary": processor.tokenizer.get_vocab(),
                    "language": language,
                    "type": "huggingface" if isinstance(model_variant, HFModel) else "torchaudio",
                },
                audio=audio,
                device=device,
            )

            aligned_script_lines.append([item if isinstance(item, ScriptLine) else None for item in alignment])
    aligned_script_lines = filter_aligned_script_lines(aligned_script_lines, levels_to_keep)
    return aligned_script_lines


# Note: this code is derived from: https://github.com/m-bain/whisperX/tree/main

# Copyright (c) 2022, Max Bain
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement:
#    This product includes software developed by Max Bain.
# 4. Neither the name of Max Bain nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
