"""Functionality for interfacing with TalkBank data."""

from typing import Any, Dict, List

from senselab.utils.data_structures.script_line import ScriptLine

try:
    import pylangacq

    PYLANGACQ_AVAILABLE = True
except ModuleNotFoundError:
    PYLANGACQ_AVAILABLE = False


def chats_to_script_lines(
    path: str,
    **kwargs: Any,  # noqa: ANN401
) -> Dict[str, List[ScriptLine]]:
    """Connvert .cha files to script lines.

    Converts .cha files to script lines using pylangacq's built-in read_chats functionality.

    Args:
      path (str): The path to a .cha file or a directory to recursively scan and use
      **kwargs (Any): key-word arguments to be used with pylangacq's read_chats. For further info,
        please refer to https://pylangacq.org/api.html#pylangacq.read_chat

    Returns:
      Dictionary mapping filepaths (e.g. individual .cha files) to a list of ScriptLines
    """
    if not PYLANGACQ_AVAILABLE:
        raise ModuleNotFoundError(
            "`pylangacq` is not installed. "
            "Please install senselab text dependencies using `pip install 'senselab[text]'`."
        )

    reader = pylangacq.read_chat(path, **kwargs)
    script_lines_by_file: Dict[str, List[ScriptLine]] = {}

    # Support both pylangacq <0.20 (methods) and >=0.20 (properties/rustling)
    file_paths = reader.file_paths() if callable(reader.file_paths) else reader.file_paths

    try:
        utterances_by_file = reader.utterances(by_files=True)
        words_by_file = reader.words(by_files=True, by_utterances=True)
    except TypeError:
        # pylangacq >=0.20 (rustling): iterate per-chat instead
        utterances_by_file = None
        words_by_file = None

    if utterances_by_file is not None:
        # pylangacq <0.20 API
        for i, fp in enumerate(file_paths):
            script_lines_by_file[fp] = []
            for utt_idx, utterance in enumerate(utterances_by_file[i]):
                if utterance.time_marks:
                    start = utterance.time_marks[0] / 1000
                    end = utterance.time_marks[1] / 1000
                else:
                    start = None
                    end = None
                words_in_utterance = words_by_file[i][utt_idx]
                if len(words_in_utterance) > 0:
                    utterance_transcript = " ".join(words_in_utterance[:-1]) + words_in_utterance[-1]
                else:
                    utterance_transcript = ""
                script_lines_by_file[fp].append(
                    ScriptLine(text=utterance_transcript, speaker=utterance.participant, start=start, end=end)
                )
        return script_lines_by_file

    # pylangacq >=0.20 API (rustling backend)
    for chat in reader:
        file_path = chat.file_paths[0] if isinstance(chat.file_paths, list) else chat.file_paths()[0]
        script_lines_by_file[file_path] = []
        for utterance in chat.utterances():
            if utterance.time_marks:
                start = utterance.time_marks[0] / 1000
                end = utterance.time_marks[1] / 1000
            else:
                start = None
                end = None

            words_in_utterance = [token.word for token in utterance.tokens if token.word]
            if len(words_in_utterance) > 0:
                utterance_transcript = " ".join(words_in_utterance[:-1]) + words_in_utterance[-1]
            else:
                utterance_transcript = ""
            script_lines_by_file[file_path].append(
                ScriptLine(text=utterance_transcript, speaker=utterance.participant, start=start, end=end)
            )
    return script_lines_by_file
