"""This module contains functions that extract features from provided transcripts.
"""

from typing import Any, Dict, List
import numpy as np

from senselab.utils.data_structures import ScriptLine


def number_of_turns(segments: List[ScriptLine], speaker: str) -> int:
    """Count the number of turns taken by the given speaker in the conversation.

    A *turn* is counted every time the speaker label changes from one segment
    to the next.  The first segment is considered the beginning of the first
    turn.
    """
    if not segments:
        raise ValueError("No segments provided")
    
    turns = 0
    if segments[0].speaker == speaker:
        turns += 1
    for seg1, seg2 in zip(segments, segments[1:]):
        if seg2.speaker == speaker and seg1.speaker != speaker:
            turns += 1
    return turns


def percent_talk_time(segments: List[ScriptLine], speaker: str) -> float:
    """Return the percentage of total speaking time attributable to the given speaker."""
    total_time = 0.0
    speaker_time = 0.0
    for seg in segments:
        dur = max(0.0, seg.end - seg.start)
        total_time += dur
        if seg.speaker == speaker:
            speaker_time += dur
    if total_time == 0:
        return 0.0
    return speaker_time / total_time


def mean_length_utterance(segments: List[ScriptLine], speaker: str) -> float:
    """Return the mean length of utterance for the given speaker.
    
    TODO: This currently counts words, but canonical MLU is calculated with morphemes.
    """
    utterance_lengths = []
    for seg in segments:
        if seg.speaker == speaker:
            utterance_lengths.append(len(seg.text.split()))

    if not utterance_lengths:
        return 0.0
    return np.mean(utterance_lengths)


def words_per_minute(segments: List[ScriptLine], speaker: str) -> float:
    """Return the mean words per minute of the given speaker.
    
    TODO: Would standard deviation / variance be useful?"""
    total_time = 0.0
    num_words = 0
    for seg in segments:
        if seg.speaker != speaker:
            continue
        dur = seg.end - seg.start
        num_words += len(seg.text.split())
        total_time += dur
    if total_time == 0:
        return 0.0
    return num_words / (total_time / 60.0)


def mean_length_of_speech(segments: List[ScriptLine], speaker: str) -> float:
    """Return the mean length of speech for the given speaker.
    
    TODO: It would probably be better to calculate this straight from audio/VAD."""
    total_time = 0.0
    num_turns = 0
    for seg in segments:
        if seg.speaker == speaker:
            total_time += seg.end - seg.start
            num_turns += 1
    if num_turns == 0:
        return 0.0
    return total_time / num_turns


def response_latency(segments: List[ScriptLine], speaker: str) -> float:
    """Return the mean length of silence before the given speaker's turns.
    
    TODO: It would probably be better to calculate this straight from audio/diarization.
    """
    silences = 0.0
    num_turns = 0
    for prev, curr in zip(segments, segments[1:]):
        if curr.speaker == prev.speaker:
            continue
        if curr.speaker != speaker:
            continue
        gap = curr.start - prev.end
        
        silences += gap
        num_turns += 1
    if num_turns == 0:
        return 0.0
    return silences / num_turns


def silent_pause_statistics(segments: List[ScriptLine], speaker: str, pause_threshold: float = 0.2) -> Dict[str, float]:
    """
    Args:
        segments (List[ScriptLine]): A list of ScriptLine objects.
        speaker (str): The speaker to calculate statistics for.
        pause_threshold (float): The threshold of silence to be considered a pause.

    Returns:
        Dict[str, float]: A dictionary containing the following statistics for the given speaker:
            - amount_of_pauses_per_minute: The average number of silent pauses per minute.
            - duration_of_pauses_per_minute: The average duration of silent pauses per minute.
            - percentage_of_speech_spent_in_pauses: The percentage of speech that is spent in pauses.
            - mean_pause_duration: The mean duration of silent pauses.
            - std_dev_pause_duration: The standard deviation of the duration of silent pauses.

    NOTE: Assumes the transcripts are lists of ScriptLine objects, where each
    ScriptLine object is a single utterance by a single speaker, and is further
    chunked into words.

    TODO: It is likely better to calculate this straight from audio/VAD.
    """
    pauses = []
    num_pauses = 0
    total_pause_duration = 0.0
    total_speech_duration = 0.0
    for seg in segments:
        if seg.speaker != speaker:
            continue
        words = seg.chunks  # List[ScriptLine] of words in the segment
        if not words:
            continue
        for w1, w2 in zip(words, words[1:]):
            gap = w2.start - w1.end
            if gap >= pause_threshold:
                pauses.append(gap)
                num_pauses += 1
                total_pause_duration += gap
        total_speech_duration += seg.end - seg.start

    if not pauses or total_speech_duration == 0:
        return {
            "amount_of_pauses_per_minute": 0.0,
            "duration_of_pauses_per_minute": 0.0,
            "percentage_of_speech_spent_in_pauses": 0.0,
            "mean_pause_duration": 0.0,
            "std_dev_pause_duration": 0.0
        }

    return {
        "amount_of_pauses_per_minute": num_pauses / (total_speech_duration / 60.0),
        "duration_of_pauses_per_minute": total_pause_duration / (total_speech_duration / 60.0),
        "percentage_of_speech_spent_in_pauses": total_pause_duration / total_speech_duration,
        "mean_pause_duration": np.mean(pauses),
        "std_dev_pause_duration": np.std(pauses)
    }


def extract_transcript_features(
    transcripts: List[ScriptLine],
) -> List[Dict[str, Dict[str, Any]]]:
    """Extract features from a list of ScriptLine objects and return a JSON-like dictionary.

    NOTE: Assumes the transcripts are lists of ScriptLine objects, where each
    ScriptLine object is a single utterance by a single speaker, and is further
    chunked into words.

    Args:
        transcripts (List[ScriptLine]): A list of ScriptLine objects.

    Returns:
        List[Dict[str, Dict[str, Any]]]: A list of dictionary of dictionaries, 
        each containing extracted features for every speaker in each given transcript.

        Currently, these features are:
            - average + variance length in seconds of given person's speech
            - average + variance of words per minute of given person's speech
            - mean length of utterance of given person's speech
            - average + variance length in seconds of silence between turns / response latency
            - mean length + variance of pauses mid-utterance
    """
    all_features = []
    for transcript in transcripts:
        features = {}
        for speaker in set([seg.speaker for seg in transcripts]):
            features[speaker] = {
                "number_of_turns": number_of_turns(transcripts, speaker),
                "percent_talk_time": percent_talk_time(transcripts, speaker),
                "mean_length_utterance": mean_length_utterance(transcripts, speaker),
                "words_per_minute": words_per_minute(transcripts, speaker),
                "mean_length_of_speech": mean_length_of_speech(transcripts, speaker),
                "response_latency": response_latency(transcripts, speaker),
            }
            features[speaker].update(silent_pause_statistics(transcripts, speaker))

        all_features.append(features)

    return all_features