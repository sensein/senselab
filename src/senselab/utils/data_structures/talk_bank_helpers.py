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
            "Please install senselab text dependencies using `pip install senselab['text']`."
        )

    chats = pylangacq.read_chat(path, **kwargs)
    script_lines_by_file: Dict[str, List[ScriptLine]] = {}
    paths = chats.file_paths()
    utterances = chats.utterances(by_files=True)
    words = chats.words(by_files=True, by_utterances=True)
    assert len(paths) == len(utterances) and len(utterances) == len(words)
    for i in range(len(paths)):
        path = paths[i]
        script_lines_by_file[path] = []
        utterances_in_file = utterances[i]
        words_in_file = words[i]
        assert len(words_in_file) == len(utterances_in_file)
        for utt_idx, utterance in enumerate(utterances_in_file):
            words_in_utterance = words_in_file[utt_idx]
            if utterance.time_marks:
                start = utterance.time_marks[0] / 1000
                end = utterance.time_marks[1] / 1000
            else:
                start = None
                end = None

            if len(words_in_utterance) > 0:
                utterance_transcript = " ".join(words_in_utterance[:-1]) + words_in_utterance[-1]
            else:
                utterance_transcript = ""
            script_lines_by_file[path].append(
                ScriptLine(text=utterance_transcript, speaker=utterance.participant, start=start, end=end)
            )
    return script_lines_by_file
