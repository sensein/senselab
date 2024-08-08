"""Align function based on WhisperX implementation."""

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from threadpoolctl import threadpool_limits
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.forced_alignment.constants import (
    DEFAULT_ALIGN_MODELS_HF,
    LANGUAGES_WITHOUT_SPACES,
    MINIMUM_SEGMENT_SIZE,
    PUNKT_ABBREVIATIONS,
    SAMPLE_RATE,
)
from senselab.audio.tasks.forced_alignment.data_structures import (
    AlignedTranscriptionResult,
    Point,
    Segment,
    SingleAlignedSegment,
    SingleSegment,
    SingleWordSegment,
)
from senselab.audio.tasks.preprocessing.preprocessing import extract_segments, pad_audios
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel
from senselab.utils.data_structures.script_line import ScriptLine


def _preprocess_segments(
    transcript: List[SingleSegment],
    model_dictionary: Dict[str, int],
    model_lang: Language,
    print_progress: bool,
    combined_progress: bool,
) -> List[SingleSegment]:
    """Preprocess transcription segments by filtering characters, handling spaces, and preparing text.

    Args:
        transcript (List[SingleSegment]): The list of transcription segments.
        model_dictionary (Dict[str, int]): Dictionary for the alignment model.
        model_lang (Language): Language of the model.
        print_progress (bool): Whether to print progress.
        combined_progress (bool): Whether to combine progress percentage.

    Returns:
        List[SingleSegment]: The preprocessed transcription segments.
    """
    total_segments = len(transcript)

    for sdx, segment in enumerate(transcript):
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # Split into words
        if model_lang.alpha_2 not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = [text]

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
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
    """Checks if a segment can be aligned.

    Args:
        segment (SingleSegment): The segment to check.
        model_dictionary (Dict[str, int]): Dictionary for character indices.
        t1 (float): Start time of the segment.
        max_duration (float): Maximum duration of the audio.

    Returns:
        bool: True if the segment can be aligned, False otherwise.
    """
    if segment["clean_char"] is None or len(segment["clean_char"]) == 0:
        return False
    if t1 >= max_duration:
        return False
    return True


def _get_prediction_matrix(
    model: torch.nn.Module,
    waveform_segment: torch.Tensor,
    lengths: Optional[torch.Tensor],
    model_type: str,
    device: DeviceType,
) -> torch.Tensor:
    """Generate prediction matrix from the alignment model.

    Args:
        model (torch.nn.Module): The alignment model.
        waveform_segment (torch.Tensor): The audio segment to be processed.
        lengths (Optional[torch.Tensor]): Lengths of the audio segments.
        model_type (str): The type of the model ('torchaudio' or 'huggingface').
        device (DeviceType): The device to run the model on.

    Returns:
        torch.Tensor: The prediction matrix.
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


def _assign_timestamps_to_characters(
    text: str, segment: SingleSegment, char_segments: list, ratio: float, t1: float, model_lang: Language
) -> pd.DataFrame:
    """Assigns timestamps to aligned characters and organizes them into a DataFrame.

    Args:
        text (str): The text to align with the segment.
        segment (SingleSegment): The segment containing character indices.
        char_segments (list): List of character segments with alignment information.
        ratio (float): The ratio of duration to waveform segment size.
        t1 (float): Start time of the segment.
        model_lang (Language): Language of the model.

    Returns:
        pd.DataFrame: DataFrame containing character alignments with timestamps and word indices.
    """
    char_segments_arr = []
    word_idx = 0
    for cdx, char in enumerate(text):
        start, end, score = None, None, None
        if segment["clean_cdx"] is not None and cdx in segment["clean_cdx"]:
            char_seg = char_segments[segment["clean_cdx"].index(cdx)]
            start = round(char_seg.start * ratio + t1, 3)
            end = round(char_seg.end * ratio + t1, 3)
            score = round(char_seg.score, 3)

        char_segments_arr.append(
            {
                "char": char,
                "start": start,
                "end": end,
                "score": score,
                "word-idx": word_idx,
            }
        )

        if model_lang.alpha_2 in LANGUAGES_WITHOUT_SPACES:
            word_idx += 1
        elif cdx == len(text) - 1 or text[cdx + 1] == " ":
            word_idx += 1

    return pd.DataFrame(char_segments_arr)


def _align_subsegments(
    segment: SingleSegment,
    char_segments_df: pd.DataFrame,
    text: str,
    word_segments: List[SingleWordSegment],
    aligned_subsegments: List[SingleAlignedSegment],
    return_char_alignments: bool,
) -> None:
    """Aligns sentence spans to create subsegments and update word segments.

    Args:
        segment (SingleSegment): The segment containing sentence spans.
        char_segments_df (pd.DataFrame): DataFrame with character alignments.
        text (str): The text to align with the segment.
        word_segments (List[SingleWordSegment]): List to store word segments.
        aligned_subsegments (List[SingleAlignedSegment]): List to store aligned subsegments.
        return_char_alignments (bool): Flag to return character alignments.

    Returns:
        None: The function modifies the word_segments and aligned_subsegments lists in place.
    """
    for sdx, (sstart, send) in enumerate(segment["sentence_spans"] or []):
        curr_chars = char_segments_df.loc[(char_segments_df.index >= sstart) & (char_segments_df.index <= send)]
        char_segments_df.loc[(char_segments_df.index >= sstart) & (char_segments_df.index <= send), "sentence-idx"] = (
            sdx
        )

        sentence_text = text[sstart:send]
        sentence_start = curr_chars["start"].min()
        end_chars = curr_chars[curr_chars["char"] != " "]
        sentence_end = end_chars["end"].max()
        sentence_words = []

        for word_idx in curr_chars["word-idx"].unique():
            word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
            word_text = "".join(word_chars["char"].tolist()).strip()
            if len(word_text) == 0:
                continue

            word_chars = word_chars[word_chars["char"] != " "]

            word_start = word_chars["start"].min()
            word_end = word_chars["end"].max()
            word_score = round(word_chars["score"].mean(), 3)

            word_segment = SingleWordSegment(word=word_text, start=word_start, end=word_end, score=word_score)

            sentence_words.append(word_segment)
            word_segments.append(word_segment)

        aligned_subsegment = SingleAlignedSegment(
            text=sentence_text, start=sentence_start, end=sentence_end, words=sentence_words, chars=word_chars
        )
        aligned_subsegments.append(aligned_subsegment)

        if return_char_alignments:
            curr_chars = curr_chars[["char", "start", "end", "score"]]
            curr_chars.fillna(-1, inplace=True)
            curr_chars = curr_chars.to_dict("records")
            curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
            aligned_subsegments[-1]["chars"] = curr_chars


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
    return_char_alignments: bool,
    interpolate_method: str,
    aligned_segment: SingleAlignedSegment,
    aligned_segments: List[SingleAlignedSegment],
    word_segments: List[SingleWordSegment],
) -> None:
    """Processes and aligns a single segment.

    Args:
        segment (SingleSegment): The segment to align.
        model (torch.nn.Module): The alignment model.
        model_dictionary (Dict[str, int]): Dictionary for character indices.
        model_lang (Language): Language of the model.
        model_type (str): Either 'huggingface' or 'torchaudio'.
        audio (Audio): The audio data.
        device (DeviceType): The device to run the model on.
        t1 (float): Start time of the segment.
        t2 (float): End time of the segment.
        return_char_alignments (bool): Flag to return character alignments.
        interpolate_method (str): Method for interpolating NaNs.
        aligned_segment (SingleAlignedSegment ): The aligned segment data.
        aligned_segments (List[SingleAlignedSegment]): List to store aligned segments.
        word_segments (List[SingleWordSegment]): List to store word segments.

    Returns:
        None: The function modifies the aligned_segments and word_segments lists in place.
    """
    text_clean = "".join(segment["clean_char"] or [])
    tokens = [model_dictionary[c] for c in text_clean]

    extracted_segment = extract_segments([(audio, [(t1, t2)])])[0][0]
    lengths = extracted_segment.waveform.shape[-1]
    waveform_segment = pad_audios([extracted_segment], MINIMUM_SEGMENT_SIZE)[0].waveform

    emissions = _get_prediction_matrix(model, waveform_segment, lengths, model_type, device)
    emission = emissions[0].cpu().detach()

    blank_id = 0
    for char, code in model_dictionary.items():
        if char == "[pad]" or char == "<pad>":
            blank_id = code

    trellis = _get_trellis(emission, tokens, blank_id)
    path = _backtrack(trellis, emission, tokens, blank_id)

    if path is None:
        print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
        aligned_segments.append(aligned_segment)
        return

    char_segments = _merge_repeats(path, text_clean)

    duration = t2 - t1
    ratio = duration * waveform_segment.size(0) / (trellis.size(0) - 1)

    char_segments_df = _assign_timestamps_to_characters(segment["text"], segment, char_segments, ratio, t1, model_lang)

    aligned_subsegments: List[SingleAlignedSegment] = []
    if isinstance(char_segments_df, pd.DataFrame):
        char_segments_df["sentence-idx"] = None
    else:
        raise TypeError("char_segments_df must be a pandas DataFrame.")

    if segment["sentence_spans"] is not None:
        _align_subsegments(
            segment=segment,
            char_segments_df=char_segments_df,
            text=segment["text"],
            word_segments=word_segments,
            aligned_subsegments=aligned_subsegments,
            return_char_alignments=True,
        )

        if aligned_subsegments:
            aligned_subsegments_df = pd.DataFrame(aligned_subsegments)

            aligned_subsegments_df["start"] = _interpolate_nans(
                aligned_subsegments_df["start"], method=interpolate_method
            )
            aligned_subsegments_df["end"] = _interpolate_nans(aligned_subsegments_df["end"], method=interpolate_method)
            agg_dict = {"text": " ".join, "words": "sum"}
            if model_lang.alpha_2 in LANGUAGES_WITHOUT_SPACES:
                agg_dict["text"] = "".join
            if return_char_alignments:
                agg_dict["chars"] = "sum"
            aligned_subsegments_df.groupby(["start", "end"], as_index=False).agg(agg_dict)
            aligned_subsegments = aligned_subsegments_df.to_dict("records")

    aligned_segments.extend(aligned_subsegments)


def _get_trellis(emission: torch.Tensor, tokens: List[int], blank_id: int = 0) -> torch.Tensor:
    """Gets the trellis for token alignment.

    Args:
        emission (torch.Tensor): The emission matrix from the model.
        tokens (List[int]): The token IDs.
        blank_id (int): The ID for the blank token.

    Returns:
        torch.Tensor: The trellis matrix.
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
    """Backtracks to find the best path through the trellis.

    Args:
        trellis (torch.Tensor): The trellis matrix.
        emission (torch.Tensor): The emission matrix from the model.
        tokens (List[int]): The token IDs.
        blank_id (int): The ID for the blank token.

    Returns:
        Optional[List[Point]]: The best path as a list of Points.
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
    """Merges repeated tokens in the alignment path.

    Args:
        path (List[Point]): The alignment path.
        transcript (str): The transcript text.

    Returns:
        List[Segment]: The merged segments.
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


def _interpolate_nans(x: pd.Series, method: str = "nearest") -> pd.Series:
    """Interpolates NaN values in a pandas Series.

    Args:
        x (pd.Series): The pandas Series.
        method (str): The interpolation method (default: "nearest").

    Returns:
        pd.Series: The Series with interpolated NaNs.
    """
    if x.notnull().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()


def _align_segments(
    transcript: List[SingleSegment],
    model: torch.nn.Module,
    model_dictionary: Dict[str, int],
    model_lang: Language,
    model_type: str,
    audio: Audio,
    device: DeviceType,
    max_duration: float,
    return_char_alignments: bool,
    interpolate_method: str,
) -> Tuple[List[SingleAlignedSegment], List[SingleWordSegment]]:
    """Align segments based on the predictions.

    Args:
        transcript (List[SingleSegment]): The list of transcription segments.
        model (torch.nn.Module): The alignment model.
        model_dictionary (Dict[str, int]): Dictionary for character indices.
        model_lang (str): Language of the model.
        model_type (str): The type of the model ('torchaudio' or 'huggingface').
        audio (Audio): The audio data.
        device (DeviceType): The device to run the model on.
        max_duration (float): Maximum duration of the audio.
        return_char_alignments (bool): Flag to return character alignments.
        interpolate_method (str): Method for interpolating NaNs.

    Returns:
        Tuple[List[SingleAlignedSegment], List[SingleWordSegment]]: The aligned segments and word segments.
    """
    aligned_segments: List[SingleAlignedSegment] = []
    word_segments: List[SingleWordSegment] = []

    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]

        aligned_segment: SingleAlignedSegment = {"start": t1, "end": t2, "text": text, "words": [], "chars": None}

        if return_char_alignments:
            aligned_segment["chars"] = []

        if _can_align_segment(segment, model_dictionary, t1, max_duration):
            _align_single_segment(
                segment,
                model,
                model_dictionary,
                model_lang,
                model_type,
                audio,
                device,
                t1,
                t2,
                return_char_alignments,
                interpolate_method,
                aligned_segment,
                aligned_segments,
                word_segments,
            )
        else:
            print(f'Failed to align segment ("{segment["text"]}"), skipping...')
            aligned_segments.append(aligned_segment)

    return (aligned_segments, word_segments)


def _convert_to_scriptline(data: AlignedTranscriptionResult) -> List[ScriptLine]:
    """Convert a dictionary of segments and word segments to a list of ScriptLine objects.

    Args:
        data (AlignedTranscriptionResult): The input dictionary with segments and word segments.

    Returns:
        List[ScriptLine]: The list of ScriptLine objects.
    """
    segments = data["segments"]
    script_lines = []

    for segment in segments:
        words = segment["words"]
        word_chunks = [ScriptLine(text=word["word"]) for word in words]

        # Handle 'nan' end values by setting them to None
        start = segment["start"]
        end: Optional[float] = segment["end"]
        if end is not None and (isinstance(end, float) and math.isnan(end)):
            end = None

        script_line = ScriptLine(text=segment["text"], start=start, end=end, chunks=word_chunks)
        script_lines.append(script_line)

    return script_lines


def _align_transcription(
    transcript: List[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: Dict[str, Any],
    audio: Audio,
    device: DeviceType,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
) -> AlignedTranscriptionResult:
    """Aligns phoneme recognition predictions to known transcription.

    Args:
        transcript (List[SingleSegment]): The list of transcription segments.
        model (torch.nn.Module): The alignment model.
        align_model_metadata (Dict[str, Any]): Metadata for the alignment model.
        audio (Audio): The audio data.
        device (DeviceType): The device to run the model on.
        interpolate_method (str): The method for interpolating NaNs (default: "nearest").
        return_char_alignments (bool): Whether to return character alignments (default: False).
        print_progress (bool): Whether to print progress (default: False).
        combined_progress (bool): Whether to combine progress (default: False).

    Returns:
        AlignedTranscriptionResult: The aligned transcription result.
    """
    max_duration = audio.waveform.shape[1] / audio.sampling_rate

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    transcript = _preprocess_segments(
        transcript,
        align_model_metadata["dictionary"],
        align_model_metadata["language"],
        print_progress,
        combined_progress,
    )

    aligned_segments, word_segments = _align_segments(
        transcript=transcript,
        model=model,
        model_dictionary=model_dictionary,
        model_lang=model_lang,
        model_type=model_type,
        audio=audio,
        device=device,
        max_duration=max_duration,
        return_char_alignments=return_char_alignments,
        interpolate_method=interpolate_method,
    )
    return {"segments": aligned_segments, "word_segments": word_segments}


def align_transcriptions(
    audios_and_transcriptions_and_language: List[Tuple[Audio, ScriptLine, Language]],
) -> List[List[ScriptLine]]:
    """Aligns transcriptions with the given audio using a wav2vec2.0 model.

    Args:
        audios_and_transcriptions_and_language (List[tuple[Audio, ScriptLine, Language]):
            A list of tuples audios, corresponding transcriptions, and optionally a language.
            The default language is English.

    Returns:
        List[List[ScriptLine]]: A list of lists, where each inner list contains the aligned script lines for each audio.
    """
    aligned_script_lines = []
    loaded_processors_and_models = {}

    for item in audios_and_transcriptions_and_language:
        audio, transcription, language = (*item, None)[:3]

        # Set default language to English if not provided
        if language is None:
            language = Language(language_code="en")

        # Define the language code and load model
        device = _select_device_and_dtype()[0]  # DeviceType object
        model_variant = DEFAULT_ALIGN_MODELS_HF.get(language.language_code, DEFAULT_ALIGN_MODELS_HF["en"])

        if model_variant.path_or_uri not in loaded_processors_and_models:  # Load model
            processor = Wav2Vec2Processor.from_pretrained(model_variant.path_or_uri)
            model = Wav2Vec2ForCTC.from_pretrained(model_variant.path_or_uri).to(device.value)
            loaded_processors_and_models[model_variant.path_or_uri] = (processor, model)

        processor, model = loaded_processors_and_models[model_variant.path_or_uri]

        if audio.sampling_rate != SAMPLE_RATE:
            raise ValueError(f"{audio.sampling_rate} rate is not equal to the training rate of {SAMPLE_RATE}.")

        # Ensure start and end are not None
        start = transcription.start if transcription.start is not None else 0.0
        end = transcription.end if transcription.end is not None else audio.waveform.shape[1] / audio.sampling_rate

        # Ensure text is not None
        text = transcription.text if transcription.text is not None else ""

        # Align each segment of the transcription
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
                return_char_alignments=True,
            )
            aligned_script_lines.append(_convert_to_scriptline(alignment))

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
