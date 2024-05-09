






"""
Pyannote https://huggingface.co/pyannote/speaker-diarization-3.1
function accepting as input a HFDataset and returning the output of pyannote speaker-diarization model for each file. pyannote-audio's pipeline accept as an input an AudioFile object, that can be a map of "waveform" and "sample_rate" (https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/core/io.py#L43), hence it should be an easy adaptation of the code they have. you can look at the speech_to_text.py script to get inspired. you can do mapping and batching to speed up the processio.py
AudioFile = Union[Text, Path, IOBase, Mapping]
"""
