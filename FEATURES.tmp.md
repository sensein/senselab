# Functionalities
This file is here just as a support for development.

AUDIO

[TODO]:
- speech to text
    1. to transcribe speech into text
        - INPUT:
            1. a datasets object with the audio recordings in the "audio" column
            2. the audio column (default = "audio")
            3. the speech to text service to use (including the name, the version, the revision, and - for some services only and sometimes it's optional - the language of the transcription model we want to use)
        - PREPROCESSING:
            1. adapt the language to the service format
            2. organize the dataset into batches
        - PROCESSING:
            1. transcribe the dataset
        - POSTPROCESSING:
            1. formatting the transcripts to follow a standard organization
        - OUTPUT:
            1. a new dataset including only the transcripts of the audios in a standardized json format (plus an index?)
        - TESTS:
            1. test input errors (a field is missing, the audio column exists and contains audio objects, params missing)
            2. test the transcript of a test file is ok
            3. test the language is supported (and the tool handles errors)

    2. to compute word error rate
        - INPUT:
            1. a dataset object with the "transcript" and the "groundtruth" columns
            2. a service with a name (default is jitter)
        - PROCESSING:
            1. computing the per-row WER between the 2 columns
        - OUTPUT:
            1. a dataset with the "WER" column
        - TESTS:
            1. test input errors (a field is missing, fields missing, the 2 columns don't contain strings)
            2. test output is ok
