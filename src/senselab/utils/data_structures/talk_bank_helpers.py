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
    for chat in reader:
        file_path = chat.file_paths[0]
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
